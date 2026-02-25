"""
Dashboard State — Pipeline control, live monitoring, and data management.

Architecture: Frontend-driven polling via rx.call_script + setTimeout.
- start_analysis launches subprocess with subprocess.Popen (non-blocking)
- JavaScript setTimeout calls tick_timer every 1 second
- tick_timer is a SIMPLE SYNCHRONOUS event — checks files, updates state
- No background tasks. No async with self. No StateProxy.
- Completely immune to Reflex hot-reload killing background tasks.
"""

import reflex as rx
import json
import base64
import subprocess
import os
import sys
import time
import re
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHARED_OUTPUTS = PROJECT_ROOT / "shared_outputs"

# Module-level dict to track subprocess Popen objects by PID.
# This survives across event handler calls (same Python process).
_PROCS: dict[int, subprocess.Popen] = {}

AGENT_REGISTRY = [
    {"key": "technical_fardeen", "name": "Fardeen", "type": "Technical", "weight": "25%"},
    {"key": "technical_tamer", "name": "Tamer", "type": "Technical", "weight": "25%"},
    {"key": "fundamental_daria", "name": "Daria", "type": "Fundamental", "weight": "12.5%"},
    {"key": "fundamental_shakzod", "name": "Shakzod", "type": "Fundamental", "weight": "12.5%"},
    {"key": "fundamental_lary", "name": "Lary", "type": "Fundamental", "weight": "12.5%"},
    {"key": "fundamental_mohamed", "name": "Mohamed", "type": "Fundamental", "weight": "12.5%"},
]

# # LARY_FACTS = [
#     "Lary has made a minimal contribution towards this project",
#     "Lary prioritised running over this project",
#     "Lary was chilling in Oxford while everyone were actively working on this project",
#     "Lary has made his agent in a few hours using ChatGPT",
#     "Lary had no contribution towards the report",
#     "Lary had no contribution towards the presentation slides",
#     "Lary had no contribution towards building the master agent",
#     "Lary had no contribution towards building this beautiful UI",
#     "Lary skipped 90% of the meetings in 2026",
#     "Lary's overall contribution towards this project was calculated mathematically, and is equal to 2%",
#     "Lary acquised the team members of going dark on him a day before the deadline",
#     "The team really wanted to exclude Lary from this project",
#     "Lary did not care aboout the report, he was sleeping a day before the deadline",
#     "Lary ignored team concerns regarding the project",
# ]


# ══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def _agent_filename(agent_key: str, ticker: str) -> str:
    if agent_key == "technical_tamer":
        return f"technical_tamer_{ticker.lower()}.json"
    return f"{agent_key}_{ticker.upper()}.json"


def _sig_color(signal: str) -> str:
    s = signal.upper()
    if "BUY" in s:
        return "#10b981"
    if "SELL" in s:
        return "#ef4444"
    return "#f59e0b"


def _hex_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}"


def _fp(val) -> str:
    try:
        return f"${float(val):,.2f}"
    except (ValueError, TypeError):
        return "N/A"


def _fpct(val, sign=False) -> str:
    try:
        v = float(val)
        return f"{v:+.1f}%" if sign else f"{v:.1f}%"
    except (ValueError, TypeError):
        return "N/A"


def _normalize_signal(val) -> str:
    if not val:
        return ""
    s = str(val).upper().strip()
    if "STRONG BUY" in s or "STRONG_BUY" in s:
        return "STRONG BUY"
    if "STRONG SELL" in s or "STRONG_SELL" in s:
        return "STRONG SELL"
    if "BUY" in s:
        return "BUY"
    if "SELL" in s:
        return "SELL"
    if "HOLD" in s or "NEUTRAL" in s:
        return "HOLD"
    return s


def _extract_signal(data: dict) -> str:
    signal_keys = ["signal", "recommendation", "action", "overall_signal",
                   "trade_action", "direction", "rating"]
    for key in signal_keys:
        val = data.get(key)
        if val and isinstance(val, str):
            s = _normalize_signal(val)
            if s:
                return s
        if isinstance(val, dict):
            for sub in ["rating", "action", "signal", "recommendation",
                        "recommendation_display"]:
                if sub in val:
                    s = _normalize_signal(val[sub])
                    if s:
                        return s
    nested_keys = ["investment_memo", "trade_note", "analysis", "summary",
                   "result", "output"]
    for nk in nested_keys:
        nested = data.get(nk)
        if not isinstance(nested, dict):
            continue
        for key in signal_keys:
            val = nested.get(key)
            if val and isinstance(val, str):
                s = _normalize_signal(val)
                if s:
                    return s
            if isinstance(val, dict):
                for sub in ["rating", "action", "signal", "recommendation",
                            "recommendation_display"]:
                    if sub in val:
                        s = _normalize_signal(val[sub])
                        if s:
                            return s
    return "N/A"


def _extract_target(data: dict) -> str:
    target_keys = ["target_price", "price_target", "target", "tp",
                   "exit_price", "blended_target", "dcf_target",
                   "fair_value", "intrinsic_value", "target_1",
                   "take_profit", "multiples_target"]

    def _try_price(val) -> str:
        if val is None:
            return ""
        try:
            fv = float(val)
            if fv > 0:
                return _fp(fv)
        except (ValueError, TypeError):
            pass
        return ""

    for key in target_keys:
        r = _try_price(data.get(key))
        if r:
            return r
    nested_keys = ["recommendation", "trade_specifications", "trade_setup",
                   "valuation_summary", "investment_memo", "trade_note",
                   "analysis", "result", "output"]
    for nk in nested_keys:
        nested = data.get(nk)
        if not isinstance(nested, dict):
            continue
        for key in target_keys:
            r = _try_price(nested.get(key))
            if r:
                return r
        for sub_nk in ["recommendation", "valuation", "summary"]:
            sub = nested.get(sub_nk)
            if isinstance(sub, dict):
                for key in target_keys:
                    r = _try_price(sub.get(key))
                    if r:
                        return r
    return "N/A"


def _strip_ansi(text: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


# ══════════════════════════════════════════════════════════════
# STATE CLASS
# ══════════════════════════════════════════════════════════════

class DashboardState(rx.State):

    # ── View Mode ──
    view_mode: str = "launch"
    transition_phase: str = ""  # "" | "out" | "in" — controls screen transition overlay

    # ── Launch Screen ──
    ticker_input: str = "AAPL"
    has_existing_data: bool = False
    existing_ticker: str = ""
    existing_date: str = ""

    # ── Pipeline Execution ──
    is_running: bool = False
    pipeline_step: str = ""
    pipeline_progress: int = 0
    pipeline_log: list[str] = []
    run_error: str = ""
    elapsed_str: str = "0:00"
    agents_completed_count: str = "0"
    agent_statuses: list[dict[str, str]] = []
    fun_fact: str = ""
    fun_fact_index: int = 0

    # ── Pipeline internal tracking (persisted in state) ──
    pipeline_start_ts: float = 0.0
    pipeline_pid: int = 0
    pipeline_ticker: str = ""
    completed_agent_keys: list[str] = []
    final_detected: bool = False
    charts_detected: bool = False
    tick_count: int = 0

    # ── Results ──
    is_loaded: bool = False
    error_msg: str = ""
    active_tab: str = "overview"

    ticker: str = ""
    company_name: str = ""
    analysis_date: str = ""

    recommendation: str = "N/A"
    confidence: str = "0"
    conviction_rationale: str = ""
    investment_horizon: str = ""
    rec_color: str = "#f59e0b"

    current_price_fmt: str = "$0.00"
    target_price_fmt: str = "$0.00"
    stop_loss_fmt: str = "$0.00"
    expected_return_fmt: str = "0.0%"
    max_drawdown_fmt: str = "0.0%"
    risk_reward_fmt: str = "0.0x"

    pos_recommended: str = "0%"
    pos_max: str = "0%"
    pos_rationale: str = ""

    consensus_tech: str = "N/A"
    consensus_fund: str = "N/A"
    consensus_tech_color: str = "#f59e0b"
    consensus_fund_color: str = "#f59e0b"
    agreement_level: str = ""
    divergence_notes: str = ""

    executive_summary: str = ""
    technical_synthesis: str = ""
    fundamental_synthesis: str = ""
    investment_thesis: str = ""
    risk_narrative: str = ""

    sharpe: str = "N/A"
    sortino: str = "N/A"
    calmar: str = "N/A"
    vol_pct: str = "N/A"
    max_dd: str = "N/A"
    win_rate: str = "N/A"
    profit_factor_v: str = "N/A"
    total_trades_v: str = "N/A"
    var95: str = "N/A"
    cagr_v: str = "N/A"

    support_str: str = ""
    resistance_str: str = ""
    entry_zone_str: str = ""
    profit_targets_str: str = ""

    targets_chart_src: str = ""
    price_chart_src: str = ""

    agents_list: list[dict[str, str]] = []
    scenario_list: list[dict[str, str]] = []
    risks_list: list[dict[str, str]] = []
    catalysts_list: list[dict[str, str]] = []
    monitoring_list: list[dict[str, str]] = []
    conflicts_list: list[dict[str, str]] = []

    signal_weights_data: list[dict] = []
    signal_buy_weight: int = 0
    signal_hold_weight: int = 0
    signal_sell_weight: int = 0
    agent_targets_data: list[dict] = []

    # Enhanced chart data
    current_price_num: int = 0
    target_price_num: int = 0
    stop_loss_num: int = 0
    scenario_chart_data: list[dict] = []
    performance_radar_data: list[dict] = []
    catalyst_chart_data: list[dict] = []
    risk_impact_data: list[dict] = []
    price_levels_data: list[dict] = []
    agent_comparison_data: list[dict] = []

    tamer_momentum: list[dict[str, str]] = []
    tamer_trend: list[dict[str, str]] = []
    tamer_volatility: list[dict[str, str]] = []
    tamer_volume: list[dict[str, str]] = []
    fardeen_rsi: str = ""
    fardeen_macd_hist: str = ""
    fardeen_adx: str = ""
    fardeen_atr: str = ""
    fardeen_bb: str = ""
    fardeen_regime: str = ""
    fardeen_hurst: str = ""
    fardeen_vol_regime: str = ""

    fund_pe: str = "N/A"
    fund_pb: str = "N/A"
    fund_ev_ebitda: str = "N/A"
    fund_ps: str = "N/A"
    fund_roe: str = "N/A"
    fund_roa: str = "N/A"
    fund_net_margin: str = "N/A"
    fund_gross_margin: str = "N/A"
    fund_revenue_growth: str = "N/A"
    fund_net_income_growth: str = "N/A"
    fund_current_ratio: str = "N/A"
    fund_debt_equity: str = "N/A"
    fund_altman_z: str = "N/A"

    # Extra detail data
    recommendation_source: str = ""
    vol_pct_raw: str = ""
    expected_return_raw: str = ""
    lary_image_path: str = "/image.png"

    # ── New chart data (visual overhaul) ──
    # Issue 5: Price zone chart for overview
    price_zone_data: list[dict] = []
    sma_50_num: int = 0
    sma_200_num: int = 0
    bb_upper_num: int = 0
    bb_lower_num: int = 0

    # Issue 6: Agent signal distribution
    agent_signal_distribution: list[dict] = []

    # Issue 7: Analysis enrichment
    regime_data: list[dict] = []
    regime_performance_data: list[dict] = []
    regime_radar_data: list[dict] = []

    # Issue 8: Risk summary
    risk_count_str: str = "0"
    avg_risk_impact_str: str = "N/A"
    risk_radar_data: list[dict] = []

    # Issue 9: Monitoring summary
    monitoring_rsi_current: str = "N/A"
    monitoring_pe_current: str = "N/A"
    monitoring_sma50_current: str = "N/A"
    next_catalyst_str: str = "N/A"
    next_catalyst_date: str = "N/A"

    # Issue 10: Lary tab
    contribution_chart_data: list[dict] = []
    lary_milestone_data: list[dict] = []
    lary_signals_data: list[dict] = []

    # ══════════════════════════════════════════════════════════
    # SIMPLE HANDLERS
    # ══════════════════════════════════════════════════════════

    def set_tab(self, tab: str):
        self.active_tab = tab

    def set_ticker(self, value: str):
        self.ticker_input = value.strip().upper()

    @rx.event
    def check_existing_data(self):
        final_path = SHARED_OUTPUTS / "FINAL_RECOMMENDATION.json"
        if final_path.exists():
            try:
                with open(final_path) as f:
                    data = json.load(f)
                self.has_existing_data = True
                self.existing_ticker = data.get("ticker", "Unknown")
                self.existing_date = data.get("analysis_date", "Unknown")
            except Exception:
                self.has_existing_data = False
        else:
            self.has_existing_data = False

        # Resilience: if the page reloaded while a pipeline was running
        # (e.g. hot-reload), restart the timer chain automatically.
        if self.is_running and self.pipeline_pid > 0:
            # Check the process is still alive
            proc = _PROCS.get(self.pipeline_pid)
            still_alive = False
            if proc:
                still_alive = proc.poll() is None
            else:
                try:
                    os.kill(self.pipeline_pid, 0)
                    still_alive = True
                except (OSError, ProcessLookupError):
                    pass

            if still_alive:
                print(f"[POLL] Resuming timer for PID {self.pipeline_pid}", flush=True)
                return rx.call_script(
                    "new Promise(r => setTimeout(() => r('t'), 1000))",
                    callback=DashboardState.tick_timer,
                )
            else:
                # Process died while page was reloading — finalize
                self._finalize_pipeline(None)
                if self.pipeline_pid in _PROCS:
                    del _PROCS[self.pipeline_pid]

    @rx.event
    def view_existing_results(self):
        self._load_all_data()
        if self.is_loaded:
            self.view_mode = "results"

    @rx.event
    def commit_results_view(self, _result=None):
        """Called after warp-out animation finishes — switch to results."""
        self.view_mode = "results"
        self.transition_phase = "warp-in"
        return rx.call_script(
            "new Promise(r => setTimeout(r, 700))",
            callback=DashboardState.end_transition,
        )

    @rx.event
    def end_transition(self, _result=None):
        self.transition_phase = ""

    @rx.event
    def go_to_launch(self):
        self.view_mode = "launch"
        self.is_running = False
        self.pipeline_progress = 0
        self.run_error = ""
        self.check_existing_data()

    @rx.event
    def view_results_now(self):
        if self.is_loaded:
            self.view_mode = "results"

    # ══════════════════════════════════════════════════════════
    # PIPELINE EXECUTION — Frontend-driven polling
    # subprocess.Popen launches the process (non-blocking).
    # rx.call_script(setTimeout) triggers tick_timer every second.
    # tick_timer is a plain synchronous event — no background task.
    # ══════════════════════════════════════════════════════════

    @rx.event
    def initiate_launch(self):
        """Trigger warp-out transition, then start_analysis after delay."""
        if not self.ticker_input or self.is_running:
            return
        self.transition_phase = "warp-out"
        # After 800ms, call start_analysis
        return rx.call_script(
            "new Promise(r => setTimeout(r, 800))",
            callback=DashboardState.start_analysis,
        )

    @rx.event
    def start_analysis(self, _result=None):
        if not self.ticker_input:
            return
        if self.is_running and self.view_mode == "running":
            return
        ticker = self.ticker_input
        self.view_mode = "running"
        self.is_running = True
        self.pipeline_progress = 0
        self.pipeline_step = "Initializing pipeline..."
        now = datetime.now().strftime('%H:%M:%S')
        self.pipeline_log = [f"[{now}] Starting analysis for {ticker}..."]
        self.run_error = ""
        self.elapsed_str = "0:00"
        self.agents_completed_count = "0"
        self.fun_fact = ""
        self.fun_fact_index = 0
        self.pipeline_ticker = ticker
        self.pipeline_start_ts = time.time()
        self.completed_agent_keys = []
        self.final_detected = False
        self.charts_detected = False
        self.tick_count = 0
        self._build_initial_statuses()

        # Mark all agents as running
        self.agent_statuses = [
            {**a, "status": "running", "status_color": "#f59e0b", "status_text": "Running..."}
            for a in self.agent_statuses
        ]

        # Clean stale output files from any prior run
        SHARED_OUTPUTS.mkdir(exist_ok=True)
        for _f in list(SHARED_OUTPUTS.glob("*.json")) + \
                  list(SHARED_OUTPUTS.glob("*.png")) + \
                  list(SHARED_OUTPUTS.glob("*.pdf")):
            try:
                _f.unlink()
            except OSError:
                pass

        # Launch master_agent directly (skip run_demo.py wrapper overhead)
        log_path = SHARED_OUTPUTS / "_pipeline_output.log"
        try:
            log_file = open(log_path, "w")
            proc = subprocess.Popen(
                [sys.executable, "-u", str(PROJECT_ROOT / "src" / "master_agent.py"), ticker],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )
            log_file.close()
            self.pipeline_pid = proc.pid
            _PROCS[proc.pid] = proc
            print(f"[POLL] Pipeline launched PID={proc.pid} for {ticker}", flush=True)
        except Exception as e:
            self.run_error = f"Failed to start pipeline: {e}"
            self._add_log(f"ERROR: {e}")
            self.pipeline_step = "Failed to start"
            self.is_running = False
            return

        self._add_log(f"Pipeline launched (PID {self.pipeline_pid})")
        self.pipeline_progress = 5
        self.pipeline_step = "Launching 6 AI agents in parallel..."

        # Start the frontend timer loop
        return rx.call_script(
            "new Promise(r => setTimeout(() => r('t'), 1000))",
            callback=DashboardState.tick_timer,
        )

    def _build_initial_statuses(self):
        statuses = []
        for reg in AGENT_REGISTRY:
            type_clr = "#3b82f6" if reg["type"] == "Technical" else "#8b5cf6"
            type_rgb = _hex_rgb(type_clr)
            statuses.append({
                "key": reg["key"],
                "name": reg["name"],
                "type": reg["type"],
                "weight": reg["weight"],
                "status": "pending",
                "duration": "—",
                "signal": "—",
                "target": "—",
                "status_color": "#64748b",
                "status_text": "Pending",
                "type_color": type_clr,
                "type_badge_bg": f"rgba({type_rgb}, 0.12)",
                "type_badge_border": f"1px solid rgba({type_rgb}, 0.25)",
                "card_accent": f"linear-gradient(90deg, {type_clr}, transparent)",
            })
        self.agent_statuses = statuses

    def _add_log(self, msg: str):
        ts = datetime.now().strftime('%H:%M:%S')
        self.pipeline_log = self.pipeline_log + [f"[{ts}] {msg}"]

    @rx.event
    def tick_timer(self, result: str = ""):
        """
        Called every 1 second by JavaScript setTimeout callback.
        This is a SIMPLE SYNCHRONOUS event handler — no background task,
        no async, no StateProxy. Just checks files and updates state.
        """
        if self.transition_phase:
            self.transition_phase = ""
        if not self.is_running:
            return

        self.tick_count = self.tick_count + 1
        elapsed = time.time() - self.pipeline_start_ts
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        self.elapsed_str = f"{mins}:{secs:02d}"


        # Check if process is still alive
        proc = _PROCS.get(self.pipeline_pid)
        proc_done = False
        proc_returncode = None
        if proc:
            proc_returncode = proc.poll()
            if proc_returncode is not None:
                proc_done = True
        else:
            # Proc not in dict (maybe after hot-reload) — check PID
            try:
                os.kill(self.pipeline_pid, 0)
            except (OSError, ProcessLookupError):
                proc_done = True

        # Check agent output files every 3 seconds
        ticker = self.pipeline_ticker
        if self.tick_count % 3 == 0:
            for reg in AGENT_REGISTRY:
                if reg["key"] in self.completed_agent_keys:
                    continue
                fname = _agent_filename(reg["key"], ticker)
                path = SHARED_OUTPUTS / fname
                try:
                    if path.exists() and path.stat().st_mtime > self.pipeline_start_ts:
                        dur = time.time() - self.pipeline_start_ts
                        signal = "—"
                        target = "—"
                        try:
                            with open(path) as f:
                                adata = json.load(f)
                            signal = _extract_signal(adata)
                            target = _extract_target(adata)
                        except Exception:
                            pass
                        self.completed_agent_keys = self.completed_agent_keys + [reg["key"]]
                        self.agent_statuses = [
                            {**a, "status": "completed", "duration": f"{dur:.0f}s",
                             "signal": signal, "target": target,
                             "status_color": "#10b981", "status_text": "Done"}
                            if a["key"] == reg["key"] else a
                            for a in self.agent_statuses
                        ]
                        self._add_log(f"{reg['name']} completed ({dur:.0f}s) — {signal}")
                except OSError:
                    pass

            # Check final recommendation and charts
            final_path = SHARED_OUTPUTS / "FINAL_RECOMMENDATION.json"
            chart1 = SHARED_OUTPUTS / "analyst_targets_chart.png"
            if not self.final_detected:
                try:
                    if final_path.exists() and final_path.stat().st_mtime > self.pipeline_start_ts:
                        self.final_detected = True
                        self._add_log("Final recommendation generated!")
                except OSError:
                    pass
            if not self.charts_detected:
                try:
                    if chart1.exists() and chart1.stat().st_mtime > self.pipeline_start_ts:
                        self.charts_detected = True
                        self._add_log("Charts generated")
                except OSError:
                    pass

        # Update progress
        n = len(self.completed_agent_keys)
        self.agents_completed_count = str(n)
        if not self.final_detected:
            if n == 0:
                self.pipeline_progress = 5
                self.pipeline_step = "Waiting for agents..."
            else:
                self.pipeline_progress = 5 + int(n / 6 * 55)
                self.pipeline_step = f"{n}/6 agents completed..."
        elif not self.charts_detected:
            self.pipeline_progress = 80
            self.pipeline_step = "Master synthesizing results..."
        else:
            self.pipeline_progress = 95
            self.pipeline_step = "Finalizing outputs..."

        if self.tick_count % 5 == 0:
            print(f"[POLL] tick={self.tick_count} elapsed={elapsed:.0f}s agents={n}/6 done={proc_done}", flush=True)

        # If process finished, finalize
        if proc_done:
            self._finalize_pipeline(proc_returncode)
            # Clean up
            if self.pipeline_pid in _PROCS:
                del _PROCS[self.pipeline_pid]
            return  # Don't schedule next tick

        # Timeout check (10 minutes)
        if elapsed > 600:
            self.run_error = "Pipeline timed out (10 minutes)"
            self._add_log("Pipeline timed out")
            self.pipeline_step = "Timed out"
            self.is_running = False
            if proc:
                try:
                    proc.kill()
                except Exception:
                    pass
            if self.pipeline_pid in _PROCS:
                del _PROCS[self.pipeline_pid]
            return

        # Schedule next tick (1 second from now)
        return rx.call_script(
            "new Promise(r => setTimeout(() => r('t'), 1000))",
            callback=DashboardState.tick_timer,
        )

    def _finalize_pipeline(self, returncode):
        """Called when the subprocess exits. Loads results."""
        elapsed = time.time() - self.pipeline_start_ts
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        self.elapsed_str = f"{mins}:{secs:02d}"
        ticker = self.pipeline_ticker
        print(f"[POLL] Pipeline finished returncode={returncode} elapsed={elapsed:.0f}s", flush=True)

        # Check if successful (returncode 0 or final file exists)
        final_path = SHARED_OUTPUTS / "FINAL_RECOMMENDATION.json"
        success = (returncode == 0) or (
            final_path.exists() and final_path.stat().st_mtime > self.pipeline_start_ts
        )

        if success:
            self._add_log("Pipeline completed successfully!")
            self.pipeline_step = "Loading results..."
            self.pipeline_progress = 99

            # Resolve any agents not detected during polling
            final_signals = {}
            try:
                if final_path.exists():
                    with open(final_path) as f:
                        fdata = json.load(f)
                    attr = fdata.get("agent_attribution", {})
                    for aname, ainfo in attr.items():
                        final_signals[aname] = {
                            "signal": ainfo.get("signal", "N/A"),
                            "target": _fp(ainfo.get("target")) if ainfo.get("target") else "N/A",
                        }
            except Exception:
                pass

            final_statuses = []
            for a in self.agent_statuses:
                if a["status"] != "completed":
                    key = a["key"]
                    fname = _agent_filename(key, ticker)
                    path = SHARED_OUTPUTS / fname
                    sig = "N/A"
                    tgt = "N/A"
                    if path.exists() and path.stat().st_mtime > self.pipeline_start_ts:
                        try:
                            with open(path) as f:
                                adata = json.load(f)
                            sig = _extract_signal(adata)
                            tgt = _extract_target(adata)
                        except Exception:
                            pass
                    if sig == "N/A" and key in final_signals:
                        sig = final_signals[key].get("signal", "N/A")
                        tgt = final_signals[key].get("target", "N/A")
                    if not path.exists():
                        final_statuses.append({
                            **a, "status": "completed", "status_color": "#64748b",
                            "status_text": "No Output", "signal": "N/A",
                            "target": "N/A", "duration": "—",
                        })
                        self._add_log(f"{a['name']} — no output file generated")
                    else:
                        final_statuses.append({
                            **a, "status": "completed", "status_color": "#10b981",
                            "status_text": "Done", "signal": sig,
                            "target": tgt, "duration": f"{elapsed:.0f}s",
                        })
                        self._add_log(f"{a['name']} completed — {sig}")
                else:
                    final_statuses.append(a)
            self.agent_statuses = final_statuses
            self.agents_completed_count = "6"

            try:
                self._load_all_data()
            except Exception as e:
                self._add_log(f"Warning: Error loading some data: {e}")

            self.pipeline_progress = 100
            self.pipeline_step = "Analysis complete!"
            self._add_log(f"Total time: {mins}m {secs}s")
            self._add_log("Ready — Click 'View Results' to explore the analysis")
        else:
            err_text = ""
            log_path = SHARED_OUTPUTS / "_pipeline_output.log"
            try:
                if log_path.exists():
                    with open(log_path) as f:
                        err_text = _strip_ansi(f.read())[-800:]
            except Exception:
                pass
            self.run_error = f"Pipeline exited with code {returncode}"
            if err_text:
                self._add_log(f"Output: {err_text}")
            self._add_log(f"Pipeline failed (exit code {returncode})")
            self.pipeline_step = "Pipeline failed"

        self.is_running = False

    # ══════════════════════════════════════════════════════════
    # DATA LOADING
    # ══════════════════════════════════════════════════════════

    def _load_all_data(self):
        try:
            final_path = SHARED_OUTPUTS / "FINAL_RECOMMENDATION.json"
            if not final_path.exists():
                self.error_msg = "No analysis data found."
                return

            with open(final_path) as f:
                data = json.load(f)

            self.ticker = data.get("ticker", "N/A")
            self.company_name = data.get("company_name", "N/A")
            self.analysis_date = data.get("analysis_date", "N/A")

            rec = data.get("recommendation", "N/A")
            self.recommendation = rec
            self.rec_color = _sig_color(rec)
            self.confidence = str(int(data.get("confidence", 0)))
            self.conviction_rationale = data.get("conviction_rationale", "")
            self.investment_horizon = data.get("investment_horizon", "N/A")

            self.current_price_fmt = _fp(data.get("current_price", 0))
            self.target_price_fmt = _fp(data.get("target_price", 0))
            self.stop_loss_fmt = _fp(data.get("stop_loss", 0))
            self.expected_return_fmt = _fpct(data.get("expected_return_pct", 0), sign=True)
            self.max_drawdown_fmt = _fpct(data.get("max_drawdown_pct", 0))
            rr = data.get("risk_reward_ratio", 0)
            self.risk_reward_fmt = f"{rr:.2f}x" if rr else "N/A"

            ps = data.get("position_sizing", {})
            self.pos_recommended = f"{ps.get('recommended_pct', 0)}%"
            self.pos_max = f"{ps.get('max_pct', 0)}%"
            self.pos_rationale = ps.get("sizing_rationale", "")

            con = data.get("consensus", {})
            ct = con.get("technical", "N/A")
            cf = con.get("fundamental", "N/A")
            self.consensus_tech = ct
            self.consensus_fund = cf
            self.consensus_tech_color = _sig_color(ct)
            self.consensus_fund_color = _sig_color(cf)
            self.agreement_level = con.get("agreement_level", "")
            self.divergence_notes = con.get("divergence_notes", "")

            self.executive_summary = data.get("executive_summary", "No data available.")
            self.technical_synthesis = data.get("technical_analysis_synthesis", "No data available.")
            self.fundamental_synthesis = data.get("fundamental_analysis_synthesis", "No data available.")
            self.investment_thesis = data.get("investment_thesis", "No data available.")
            self.recommendation_source = data.get("recommendation_source", "").replace("_", " ").title()
            self.vol_pct_raw = f"{data.get('performance_metrics', {}).get('volatility_pct', 0):.1f}%"
            self.expected_return_raw = f"{data.get('expected_return_pct', 0):+.1f}%"

            rd = data.get("risk_assessment", {})
            if isinstance(rd, dict):
                self.risk_narrative = rd.get("risk_narrative", "")
                pri = rd.get("primary_risks", [])
            else:
                self.risk_narrative = str(rd)
                pri = []
            self.risks_list = []
            impacts_for_avg = []
            for r in pri:
                if isinstance(r, dict):
                    ws = r.get("warning_signs", [])
                    self.risks_list.append({
                        "risk": r.get("risk", ""),
                        "category": r.get("category", "").upper(),
                        "probability": r.get("probability", "").title(),
                        "severity": r.get("severity", "").title(),
                        "impact": f"{r.get('potential_impact_pct', 0)}%",
                        "mitigation": r.get("mitigation", ""),
                        "warning_signs": " | ".join(ws) if ws else "",
                    })
                    impacts_for_avg.append(abs(r.get("potential_impact_pct", 0)))
            self.risk_count_str = str(len(impacts_for_avg))
            self.avg_risk_impact_str = f"-{sum(impacts_for_avg)/len(impacts_for_avg):.1f}%" if impacts_for_avg else "N/A"

            perf = data.get("performance_metrics", {})
            self.sharpe = str(perf.get("sharpe_ratio", "N/A"))
            self.sortino = str(perf.get("sortino_ratio", "N/A"))
            self.calmar = str(perf.get("calmar_ratio", "N/A"))
            self.vol_pct = f"{perf.get('volatility_pct', 'N/A')}%"
            self.max_dd = f"{perf.get('max_drawdown_pct', 'N/A')}%"
            self.win_rate = f"{perf.get('win_rate_pct', 'N/A')}%"
            self.profit_factor_v = str(perf.get("profit_factor", "N/A"))
            self.total_trades_v = str(perf.get("total_trades", "N/A"))
            self.var95 = f"{perf.get('var_95_pct', 'N/A')}%"
            self.cagr_v = f"{perf.get('cagr_pct', 'N/A')}%"

            lv = data.get("key_levels", {})
            self.support_str = "  |  ".join(_fp(s) for s in lv.get("support", []))
            self.resistance_str = "  |  ".join(_fp(r) for r in lv.get("resistance", []))
            self.entry_zone_str = " — ".join(_fp(e) for e in lv.get("entry_zone", []))
            self.profit_targets_str = "  |  ".join(_fp(p) for p in lv.get("profit_targets", []))

            wts = data.get("weighted_signal_breakdown", {})
            buy_w = float(wts.get("BUY", 0))
            hold_w = float(wts.get("HOLD", 0))
            sell_w = float(wts.get("SELL", 0))
            self.signal_weights_data = [
                {"name": "BUY", "value": buy_w, "fill": "#10b981"},
                {"name": "HOLD", "value": hold_w, "fill": "#f59e0b"},
                {"name": "SELL", "value": sell_w, "fill": "#ef4444"},
            ]
            self.signal_buy_weight = round(buy_w)
            self.signal_hold_weight = round(hold_w)
            self.signal_sell_weight = round(sell_w)

            attr = data.get("agent_attribution", {})
            agents = []
            for name, info in attr.items():
                sig = info.get("signal", "HOLD")
                atype = "Technical" if "technical" in name else "Fundamental"
                disp = name.replace("technical_", "").replace("fundamental_", "").title()
                tgt = info.get("target")
                sig_clr = _sig_color(sig)
                type_clr = "#3b82f6" if atype == "Technical" else "#8b5cf6"
                type_rgb = _hex_rgb(type_clr)
                agents.append({
                    "name": disp, "full_name": name, "type": atype,
                    "signal": sig, "target": _fp(tgt) if tgt else "N/A",
                    "weight": f"{info.get('weight', 0) * 100:.0f}%",
                    "insight": info.get("key_insight", ""),
                    "color": sig_clr, "type_color": type_clr,
                    "type_badge_bg": f"rgba({type_rgb}, 0.12)",
                    "type_badge_border": f"1px solid rgba({type_rgb}, 0.25)",
                    "card_accent": f"linear-gradient(90deg, {type_clr}, transparent)",
                })
            self.agents_list = agents

            self.agent_targets_data = []
            self.agent_comparison_data = []
            cur_p = float(data.get("current_price", 0))
            tgt_p = float(data.get("target_price", 0))
            sl_p = float(data.get("stop_loss", 0))
            self.current_price_num = round(cur_p)
            self.target_price_num = round(tgt_p)
            self.stop_loss_num = round(sl_p)
            for a in agents:
                if a["target"] != "N/A":
                    try:
                        tv = float(a["target"].replace("$", "").replace(",", ""))
                        sig_clr_v = a["color"]
                        self.agent_targets_data.append({
                            "name": a["name"], "target": tv,
                            "fill": a["type_color"],
                            "signal": a["signal"],
                            "weight": a["weight"],
                            "type": a["type"],
                        })
                        self.agent_comparison_data.append({
                            "name": a["name"], "target": tv,
                            "current": cur_p,
                            "upside": round(tv - cur_p, 2),
                            "fill": sig_clr_v,
                            "type": a["type"],
                        })
                    except ValueError:
                        pass

            # Agent signal distribution
            buy_c = sum(1 for a in agents if "BUY" in a["signal"].upper())
            hold_c = sum(1 for a in agents if a["signal"].upper() in ("HOLD", "NEUTRAL"))
            sell_c = sum(1 for a in agents if "SELL" in a["signal"].upper())
            self.agent_signal_distribution = [
                {"name": "BUY", "count": buy_c, "fill": "#10b981"},
                {"name": "HOLD", "count": hold_c, "fill": "#f59e0b"},
                {"name": "SELL", "count": sell_c, "fill": "#ef4444"},
            ]

            # Scenario chart data
            scen_raw = data.get("scenario_analysis", {})
            self.scenario_chart_data = []
            for key, label, clr in [
                ("bull_case", "Bull", "#10b981"),
                ("base_case", "Base", "#3b82f6"),
                ("bear_case", "Bear", "#ef4444"),
            ]:
                sc = scen_raw.get(key, {})
                if sc:
                    self.scenario_chart_data.append({
                        "name": label,
                        "probability": round(sc.get("probability", 0) * 100),
                        "target": round(sc.get("target", 0), 2),
                        "return_pct": round(sc.get("return_pct", 0), 1),
                        "fill": clr,
                    })

            # Performance radar data (normalize to 0-100 scale)
            self.performance_radar_data = [
                {"metric": "Sharpe", "value": min(round(perf.get("sharpe_ratio", 0) / 3 * 100), 100), "fullMark": 100},
                {"metric": "Sortino", "value": min(round(perf.get("sortino_ratio", 0) / 3 * 100), 100), "fullMark": 100},
                {"metric": "Calmar", "value": min(round(perf.get("calmar_ratio", 0) / 3 * 100), 100), "fullMark": 100},
                {"metric": "Win Rate", "value": round(perf.get("win_rate_pct", 0)), "fullMark": 100},
                {"metric": "CAGR", "value": min(round(perf.get("cagr_pct", 0) / 30 * 100), 100), "fullMark": 100},
                {"metric": "Profit Factor", "value": min(round(perf.get("profit_factor", 0) / 5 * 100), 100), "fullMark": 100},
            ]

            # Catalyst chart data
            cats_raw = data.get("catalysts_timeline", [])
            self.catalyst_chart_data = []
            for ct_item in cats_raw:
                if isinstance(ct_item, dict):
                    move = ct_item.get("expected_move_pct", 0)
                    d = ct_item.get("impact_direction", "uncertain")
                    clr = "#10b981" if d == "bullish" else "#ef4444" if d == "bearish" else "#f59e0b"
                    cat_name = ct_item.get("catalyst", "")
                    # Shorten name for chart labels
                    short = cat_name[:15] + "..." if len(cat_name) > 17 else cat_name
                    self.catalyst_chart_data.append({
                        "name": short, "impact": move,
                        "fill": clr, "fullName": cat_name,
                    })

            # Risk impact data
            pri = data.get("risk_assessment", {}).get("primary_risks", [])
            self.risk_impact_data = []
            for r in pri:
                if isinstance(r, dict):
                    imp = abs(r.get("potential_impact_pct", 0))
                    prob = r.get("probability", "low")
                    risk_text = r.get("risk", "")
                    short_risk = risk_text[:18] + "..." if len(risk_text) > 20 else risk_text
                    self.risk_impact_data.append({
                        "name": short_risk, "impact": imp,
                        "probability": prob.title(),
                        "fill": "#ef4444" if prob == "high" else "#f59e0b" if prob == "medium" else "#64748b",
                    })

            # Risk radar data — each risk is a spoke, impact is the value
            self.risk_radar_data = []
            for rd in self.risk_impact_data:
                self.risk_radar_data.append({
                    "risk": rd["name"],
                    "impact": rd["impact"],
                    "fullMark": max(30, max((x["impact"] for x in self.risk_impact_data), default=30)),
                })

            # Price levels chart data
            levels = data.get("key_levels", {})
            self.price_levels_data = []
            for s in levels.get("support", []):
                self.price_levels_data.append({"name": f"S ${s:.0f}", "value": float(s), "fill": "#10b981", "type": "Support"})
            for r_val in levels.get("resistance", []):
                self.price_levels_data.append({"name": f"R ${r_val:.0f}", "value": float(r_val), "fill": "#ef4444", "type": "Resistance"})
            for e in levels.get("entry_zone", []):
                self.price_levels_data.append({"name": f"E ${e:.0f}", "value": float(e), "fill": "#3b82f6", "type": "Entry"})
            for p in levels.get("profit_targets", []):
                self.price_levels_data.append({"name": f"T ${p:.0f}", "value": float(p), "fill": "#8b5cf6", "type": "Target"})
            self.price_levels_data.sort(key=lambda x: x["value"])

            scen = data.get("scenario_analysis", {})
            self.scenario_list = []
            for key, label, icon, clr in [
                ("bull_case", "Bull Case", "trending-up", "#10b981"),
                ("base_case", "Base Case", "target", "#3b82f6"),
                ("bear_case", "Bear Case", "trending-down", "#ef4444"),
            ]:
                c = scen.get(key, {})
                if c:
                    clr_rgb = _hex_rgb(clr)
                    self.scenario_list.append({
                        "label": label,
                        "probability": f"{c.get('probability', 0) * 100:.0f}%",
                        "target": _fp(c.get("target", 0)),
                        "return_pct": _fpct(c.get("return_pct", 0), sign=True),
                        "timeline": c.get("timeline", "N/A"),
                        "narrative": c.get("narrative", ""),
                        "assumptions": " | ".join(c.get("key_assumptions", [])[:3]),
                        "color": clr, "icon": icon,
                        "border_css": f"1px solid rgba({clr_rgb}, 0.15)",
                        "hover_border_css": f"1px solid rgba({clr_rgb}, 0.35)",
                    })

            cats = data.get("catalysts_timeline", [])
            self.catalysts_list = []
            for ct_item in cats:
                if isinstance(ct_item, dict):
                    d = ct_item.get("impact_direction", "uncertain")
                    clr = "#10b981" if d == "bullish" else "#ef4444" if d == "bearish" else "#f59e0b"
                    clr_rgb = _hex_rgb(clr)
                    self.catalysts_list.append({
                        "catalyst": ct_item.get("catalyst", ""),
                        "category": ct_item.get("category", "").upper(),
                        "date": ct_item.get("expected_date", ""),
                        "direction": d.title(),
                        "move": f"{ct_item.get('expected_move_pct', 0):+.0f}%",
                        "reasoning": ct_item.get("reasoning", ""),
                        "color": clr, "icon_bg": f"rgba({clr_rgb}, 0.1)",
                    })

            mon = data.get("monitoring_checklist", [])
            self.monitoring_list = []
            for m in mon:
                if isinstance(m, dict):
                    self.monitoring_list.append({
                        "metric": m.get("metric", ""),
                        "current": m.get("current_value", ""),
                        "bullish": m.get("bullish_threshold", ""),
                        "bearish": m.get("bearish_threshold", ""),
                        "action_bull": m.get("action_if_bullish", ""),
                        "action_bear": m.get("action_if_bearish", ""),
                    })

            # Monitoring summary stats
            if mon:
                for m in mon:
                    name = m.get("metric", "").lower()
                    val = m.get("current_value", "N/A")
                    if "rsi" in name:
                        self.monitoring_rsi_current = val
                    elif "p/e" in name or "pe ratio" in name:
                        self.monitoring_pe_current = val
                    elif "sma" in name:
                        self.monitoring_sma50_current = val
            cats_raw2 = data.get("catalysts_timeline", [])
            if cats_raw2:
                c0 = cats_raw2[0] if isinstance(cats_raw2[0], dict) else {}
                nc = c0.get("catalyst", "N/A")
                self.next_catalyst_str = nc[:28] + "..." if len(nc) > 30 else nc
                self.next_catalyst_date = c0.get("expected_date", "N/A")

            # Price zone data for native chart (overview)
            levels = data.get("key_levels", {})
            self.price_zone_data = []
            if sl_p > 0:
                self.price_zone_data.append({"name": "Stop Loss", "value": round(sl_p), "fill": "#ef4444"})
            for s in levels.get("support", []):
                self.price_zone_data.append({"name": f"Support ${s:.0f}", "value": round(s), "fill": "#10b981"})
            for e in levels.get("entry_zone", []):
                self.price_zone_data.append({"name": f"Entry ${e:.0f}", "value": round(e), "fill": "#3b82f6"})
            if cur_p > 0:
                self.price_zone_data.append({"name": "Current", "value": round(cur_p), "fill": "#06b6d4"})
            for r_val in levels.get("resistance", []):
                self.price_zone_data.append({"name": f"Resist ${r_val:.0f}", "value": round(r_val), "fill": "#f43f5e"})
            if tgt_p > 0:
                self.price_zone_data.append({"name": "Target", "value": round(tgt_p), "fill": "#10b981"})
            for p in levels.get("profit_targets", []):
                if round(p) != round(tgt_p):
                    self.price_zone_data.append({"name": f"PT ${p:.0f}", "value": round(p), "fill": "#8b5cf6"})
            self.price_zone_data.sort(key=lambda x: x["value"])

            # Contribution data for Lary tab
            self.contribution_chart_data = [
                {"name": "Tamer", "value": 30, "fill": "#3b82f6"},
                {"name": "Fardeen", "value": 20, "fill": "#06b6d4"},
                {"name": "Daria", "value": 15, "fill": "#8b5cf6"},
                {"name": "Shakzod", "value": 15, "fill": "#f59e0b"},
                {"name": "Mohamed", "value": 10, "fill": "#10b981"},
                {"name": "Lary", "value": 2, "fill": "#ef4444"},
            ]
            self.lary_milestone_data = [
                {"name": "Wk 1", "expected": 15, "actual": 0},
                {"name": "Wk 4", "expected": 35, "actual": 0},
                {"name": "Wk 8", "expected": 60, "actual": 1},
                {"name": "Wk 12", "expected": 85, "actual": 2},
                {"name": "Final", "expected": 100, "actual": 2},
            ]

            conf = data.get("conflicts_resolved", [])
            self.conflicts_list = []
            for c in conf:
                if isinstance(c, dict):
                    self.conflicts_list.append({
                        "conflict": c.get("conflict", ""),
                        "resolution": c.get("resolution", ""),
                        "favored": c.get("favored_view", "").replace("technical_", "").replace("fundamental_", "").title(),
                        "reasoning": c.get("reasoning", ""),
                    })

            for fname, attr_name in [
                ("analyst_targets_chart.png", "targets_chart_src"),
                ("price_action_chart.png", "price_chart_src"),
            ]:
                cp = SHARED_OUTPUTS / fname
                if cp.exists():
                    with open(cp, "rb") as fb:
                        b64 = base64.b64encode(fb.read()).decode()
                        setattr(self, attr_name, f"data:image/png;base64,{b64}")

            ticker_val = self.ticker or "AAPL"
            self._load_technical_detail(ticker_val)
            self._load_fundamental_detail(ticker_val)
            self.is_loaded = True

        except Exception as e:
            self.error_msg = f"Error loading data: {str(e)}"

    def _load_technical_detail(self, ticker: str):
        tp = SHARED_OUTPUTS / f"technical_tamer_{ticker.lower()}.json"
        if tp.exists():
            with open(tp) as f:
                td = json.load(f)

            def _make_ind(i: dict) -> dict[str, str]:
                sig = i.get("signal", "")
                return {
                    "name": i.get("name", ""),
                    "signal": sig,
                    "value": str(round(i.get("value", 0), 2)),
                    "zone": i.get("zone", ""),
                    "confidence": f"{i.get('confidence', 0) * 100:.0f}%",
                    "color": _sig_color(sig),
                }

            self.tamer_momentum = [_make_ind(i) for i in td.get("momentum_indicators", [])]
            self.tamer_trend = [_make_ind(i) for i in td.get("trend_indicators", [])]
            self.tamer_volatility = [_make_ind(i) for i in td.get("volatility_indicators", [])]
            self.tamer_volume = [_make_ind(i) for i in td.get("volume_indicators", [])]

            # Regime data from tamer
            self.regime_data = [
                {"name": "Bull", "probability": round(td.get("regime_prob_bull", 0)), "fill": "#10b981"},
                {"name": "Sideways", "probability": round(td.get("regime_prob_sideways", 0)), "fill": "#f59e0b"},
                {"name": "Bear", "probability": round(td.get("regime_prob_bear", 0)), "fill": "#ef4444"},
            ]
            self.regime_performance_data = [
                {"name": "Bull", "value": round(td.get("perf_bull_return", 0), 1), "sharpe": str(round(td.get("perf_bull_sharpe", 0), 2)), "fill": "#10b981"},
                {"name": "Sideways", "value": round(td.get("perf_sideways_return", 0), 1), "sharpe": str(round(td.get("perf_sideways_sharpe", 0), 2)), "fill": "#f59e0b"},
                {"name": "Bear", "value": round(td.get("perf_bear_return", 0), 1), "sharpe": str(round(td.get("perf_bear_sharpe", 0), 2)), "fill": "#ef4444"},
            ]
            # Build radar data: each metric is an axis, each regime is a series
            bull_prob = round(td.get("regime_prob_bull", 0))
            side_prob = round(td.get("regime_prob_sideways", 0))
            bear_prob = round(td.get("regime_prob_bear", 0))
            bull_ret = abs(round(td.get("perf_bull_return", 0), 1))
            side_ret = abs(round(td.get("perf_sideways_return", 0), 1))
            bear_ret = abs(round(td.get("perf_bear_return", 0), 1))
            bull_sh = abs(round(td.get("perf_bull_sharpe", 0) * 33, 0))
            side_sh = abs(round(td.get("perf_sideways_sharpe", 0) * 33, 0))
            bear_sh = abs(round(td.get("perf_bear_sharpe", 0) * 33, 0))
            self.regime_radar_data = [
                {"metric": "Probability", "bull": min(bull_prob, 100), "sideways": min(side_prob, 100), "bear": min(bear_prob, 100), "fullMark": 100},
                {"metric": "Exp. Return", "bull": min(int(bull_ret * 3), 100), "sideways": min(int(side_ret * 3), 100), "bear": min(int(bear_ret * 3), 100), "fullMark": 100},
                {"metric": "Sharpe", "bull": min(int(bull_sh), 100), "sideways": min(int(side_sh), 100), "bear": min(int(bear_sh), 100), "fullMark": 100},
            ]

        fp = SHARED_OUTPUTS / f"technical_fardeen_{ticker.upper()}.json"
        if fp.exists():
            with open(fp) as f:
                fd = json.load(f)
            ta = fd.get("technical_analysis", {})
            mom = ta.get("momentum", {})
            trend = ta.get("trend", {})
            vol = ta.get("volatility", {})
            regime = fd.get("regime_analysis", {})
            self.fardeen_rsi = str(round(mom.get("rsi", 0), 1))
            self.fardeen_macd_hist = str(round(mom.get("macd_histogram", 0), 4))
            self.fardeen_adx = str(round(trend.get("adx", 0), 1))
            self.fardeen_atr = str(round(vol.get("atr", 0), 2))
            self.fardeen_bb = str(round(vol.get("bb_percent_b", 0), 3))
            self.fardeen_regime = regime.get("market_regime", "N/A")
            self.fardeen_hurst = str(round(regime.get("hurst_exponent", 0), 3))
            self.fardeen_vol_regime = regime.get("volatility_regime", "N/A")

            # SMA and BB values for price zone chart
            self.sma_50_num = round(trend.get("sma_50", 0))
            self.sma_200_num = round(trend.get("sma_200", 0))
            vol_levels = vol.get("levels", {})
            self.bb_upper_num = round(vol_levels.get("bb_upper", 0))
            self.bb_lower_num = round(vol_levels.get("bb_lower", 0))

    def _load_fundamental_detail(self, ticker: str):
        lp = SHARED_OUTPUTS / f"fundamental_lary_{ticker.upper()}.json"
        if lp.exists():
            with open(lp) as f:
                ld = json.load(f)
            ratios = ld.get("ratios", {})
            prof = ratios.get("profitability", {})
            liq = ratios.get("liquidity", {})
            lev = ratios.get("leverage", {})
            growth = ratios.get("growth", {})
            risk = ratios.get("risk", {})
            val = ratios.get("valuation", {})
            self.fund_pe = str(round(val.get("pe_ratio", 0), 2))
            self.fund_pb = str(round(val.get("pb_ratio", 0), 2))
            self.fund_ev_ebitda = str(round(val.get("ev_ebitda", 0), 2))
            self.fund_ps = str(round(val.get("ps_ratio", 0), 2))
            self.fund_roe = f"{round(prof.get('roe', 0), 2)}%"
            self.fund_roa = f"{round(prof.get('roa', 0), 2)}%"
            self.fund_net_margin = f"{round(prof.get('net_margin', 0), 2)}%"
            self.fund_gross_margin = f"{round(prof.get('gross_margin', 0), 2)}%"
            self.fund_revenue_growth = f"{round(growth.get('revenue_growth', 0), 2)}%"
            self.fund_net_income_growth = f"{round(growth.get('net_income_growth', 0), 2)}%"
            self.fund_current_ratio = str(round(liq.get("current_ratio", 0), 3))
            self.fund_debt_equity = str(round(lev.get("debt_to_equity", 0), 3))
            self.fund_altman_z = str(round(risk.get("altman_z_score", 0), 2))

            # Lary signals for Lary tab
            signals = ld.get("signals", {})
            self.lary_signals_data = []
            for sname, sinfo in signals.items():
                if isinstance(sinfo, dict):
                    sig = sinfo.get("signal", "N/A")
                    self.lary_signals_data.append({
                        "name": sname, "signal": sig,
                        "value": str(sinfo.get("value", "N/A")),
                        "color": _sig_color(sig),
                    })
