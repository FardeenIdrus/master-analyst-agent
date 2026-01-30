"""
Master Agent v2 - Orchestrates parallel execution of analyst agents and synthesizes their outputs.

Pipeline:
1. Run all registered analyst agents concurrently as subprocesses
2. Collect and process JSON outputs from shared_outputs/ directory
3. Pre-analyze: consensus, conflicts, weighted targets, key levels
4. Send structured prompt to LLM for synthesis
5. Generate charts and professional PDF report
"""

import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
load_dotenv()

# Ticker from command line or default
TICKER = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"

# =============================================================================
# CONFIGURATION
# =============================================================================

REPO_ROOT = Path(__file__).parent.parent
SHARED_OUTPUTS = REPO_ROOT / "shared_outputs"

# Agent weights (sum to 1.0)
AGENT_WEIGHTS = {
    "technical_fardeen": 0.20,
    "technical_tamer": 0.25,
    "fundamental_daria": 0.15,
    "fundamental_shakzod": 0.15,
    "fundamental_lary": 0.10,
    "fundamental_mohamed": 0.15,
}

# Agent type classification
TECHNICAL_AGENTS = ["technical_fardeen", "technical_tamer"]
FUNDAMENTAL_AGENTS = ["fundamental_daria", "fundamental_shakzod", "fundamental_lary", "fundamental_mohamed"]

# Agent registry: (display_name, entry_script_path)
AGENTS = [
    ("technical_fardeen", REPO_ROOT / "technical_agents" / "fardeen_technical_agent" / "src" / "llm_agent.py"),
    ("fundamental_daria", REPO_ROOT / "fundamental_agents" / "daria_fundamental_agent" / "run_demo.py"),
    ("fundamental_shakzod", REPO_ROOT / "fundamental_agents" / "shakzod_fundamental_agent" / "run_demo.py"),
    ("fundamental_lary", REPO_ROOT / "fundamental_agents" / "lary_fundamental_agent" / "agents.py"),
    ("fundamental_mohamed", REPO_ROOT / "fundamental_agents" / "mohamed_fundamental_agent" / "src" / "ai_agent.py"),
    ("technical_tamer", REPO_ROOT / "technical_agents" / "tamer_technical_agent" / "run_demo.py"),
]

# Truncation settings
# Only these explicit field names are removed (raw data arrays)
RAW_DATA_FIELDS = {"ohlcv", "prices", "price_history", "raw_data", "close_prices", "daily_prices"}

# Thesis text truncation limit (words) - adjust as needed
MAX_THESIS_WORDS = 1000


# =============================================================================
# AGENT EXECUTION
# =============================================================================

async def run_agent(name: str, script_path: Path) -> dict:
    """Execute a single agent as an async subprocess."""
    print(f"[{name}] Starting...")
    start = datetime.now()

    try:
        cmd = ["python", str(script_path), TICKER]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=script_path.parent
        )

        stdout, stderr = await process.communicate()
        duration = (datetime.now() - start).total_seconds()

        if process.returncode == 0:
            print(f"[{name}] Completed in {duration:.1f}s")
            return {"agent": name, "status": "success", "duration": duration}
        else:
            print(f"[{name}] Failed: {stderr.decode()[:200]}")
            return {"agent": name, "status": "failed", "error": stderr.decode()[:500]}

    except Exception as e:
        print(f"[{name}] Error: {e}")
        return {"agent": name, "status": "error", "error": str(e)}


async def run_all_agents() -> list[dict]:
    """Run all registered agents concurrently."""
    print("=" * 60)
    print("MASTER AGENT v2 - Parallel Execution")
    print("=" * 60)

    SHARED_OUTPUTS.mkdir(exist_ok=True)

    tasks = [run_agent(name, path) for name, path in AGENTS if path.exists()]
    results = await asyncio.gather(*tasks)

    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    for r in results:
        status = "OK" if r["status"] == "success" else "FAIL"
        print(f"  [{status}] {r['agent']}: {r['status']}")

    return results


# =============================================================================
# OUTPUT PROCESSING
# =============================================================================

def classify_agent(filename: str) -> Optional[str]:
    """Map filename to agent name."""
    filename_lower = filename.lower()

    if "fardeen" in filename_lower or "technical_fardeen" in filename_lower:
        return "technical_fardeen"
    elif "tamer" in filename_lower or "technical_tamer" in filename_lower:
        return "technical_tamer"
    elif "daria" in filename_lower or "fundamental_daria" in filename_lower:
        return "fundamental_daria"
    elif "shakzod" in filename_lower or "fundamental_shakzod" in filename_lower:
        return "fundamental_shakzod"
    elif "lary" in filename_lower or "lary" in filename_lower or "fundamental_lary" in filename_lower:
        return "fundamental_lary"
    elif "mohamed" in filename_lower or "fundamental_mohamed" in filename_lower:
        return "fundamental_mohamed"

    return None


def truncate_text(text: str, max_words: int) -> str:
    """Truncate text to max words, preserving complete sentences where possible."""
    if not text or not isinstance(text, str):
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    # Try to end at a sentence boundary
    last_period = truncated.rfind('.')
    if last_period > len(truncated) * 0.7:  # Only if we keep at least 70%
        truncated = truncated[:last_period + 1]
    return truncated + " [truncated]"


def smart_truncate(data: Any, depth: int = 0) -> Any:
    """
    Conservative truncation - only removes:
    1. Fields starting with '_' (internal)
    2. Explicit raw data fields (ohlcv, prices, etc.)
    3. Long numeric arrays (50+ numbers)

    Keeps ALL qualitative reasoning, indicator values, and analysis.
    """
    if depth > 15:  # Prevent infinite recursion
        return data

    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            key_lower = key.lower()

            # Skip internal fields
            if key.startswith("_"):
                continue

            # Skip explicit raw data fields
            if key_lower in RAW_DATA_FIELDS:
                continue

            # Truncate very long thesis text (but keep most of it)
            if key_lower in {"investment_thesis", "thesis", "trade_rationale", "analysis_summary"}:
                if isinstance(value, str) and len(value.split()) > MAX_THESIS_WORDS:
                    result[key] = truncate_text(value, MAX_THESIS_WORDS)
                else:
                    result[key] = value
            else:
                result[key] = smart_truncate(value, depth + 1)
        return result

    elif isinstance(data, list):
        # Only remove if it's a long array of pure numbers (likely price/indicator history)
        if len(data) > 50:
            # Check if it's all numbers
            sample = data[:10]
            if all(isinstance(x, (int, float)) for x in sample):
                return f"[array of {len(data)} numeric values]"
        # Keep lists but limit extremely long ones
        return [smart_truncate(item, depth + 1) for item in data[:200]]

    return data


def normalize_signal(raw: Any) -> str:
    """Normalize recommendation to: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL"""
    if raw is None:
        return "HOLD"

    raw_str = str(raw).upper().replace(" ", "_").replace("-", "_")

    if "STRONG" in raw_str and "BUY" in raw_str:
        return "STRONG_BUY"
    if "STRONG" in raw_str and "SELL" in raw_str:
        return "STRONG_SELL"
    if "BUY" in raw_str or "BULLISH" in raw_str or "LONG" in raw_str:
        return "BUY"
    if "SELL" in raw_str or "BEARISH" in raw_str or "SHORT" in raw_str:
        return "SELL"

    return "HOLD"


def extract_signal(output: dict) -> dict:
    """Extract recommendation and target from any agent output format."""
    signal = None
    target = None

    # Try common field names for signal
    signal_keys = ["recommendation", "signal", "action", "overall_signal", "trade_action", "direction", "rating"]
    for key in signal_keys:
        if key in output and output[key]:
            val = output[key]
            # Handle nested recommendation dict (e.g., Shakzod's {"recommendation": {"rating": "BUY"}})
            if isinstance(val, dict):
                for sub_key in ["rating", "action", "signal", "recommendation"]:
                    if sub_key in val:
                        signal = normalize_signal(val[sub_key])
                        break
            else:
                signal = normalize_signal(val)
            if signal:
                break

    # Check nested structures
    if not signal:
        for nested_key in ["trade_note", "analysis", "summary", "result", "recommendation", "investment_memo"]:
            if nested_key in output and isinstance(output[nested_key], dict):
                nested_obj = output[nested_key]
                for key in signal_keys:
                    if key in nested_obj and nested_obj[key]:
                        signal = normalize_signal(nested_obj[key])
                        break
                if signal:
                    break
                # Check one level deeper (e.g., investment_memo.recommendation.rating)
                for sub_key in ["recommendation", "summary"]:
                    if sub_key in nested_obj and isinstance(nested_obj[sub_key], dict):
                        for sig_key in signal_keys:
                            if sig_key in nested_obj[sub_key]:
                                signal = normalize_signal(nested_obj[sub_key][sig_key])
                                break
                        if signal:
                            break
                if signal:
                    break

    # Try common field names for target price (expanded list)
    target_keys = [
        "target_price", "price_target", "target", "tp", "exit_price",
        "blended_target", "dcf_target", "multiples_target",  # Mohamed's agent
        "target_1", "fair_value", "intrinsic_value", "intrinsic_price",  # Fardeen, valuation
    ]
    for key in target_keys:
        if key in output:
            try:
                val = output[key]
                if isinstance(val, (int, float)) and val > 0:
                    target = float(val)
                    break
                elif isinstance(val, str):
                    nums = re.findall(r'[\d.]+', val)
                    if nums:
                        target = float(nums[0])
                        break
            except (ValueError, TypeError):
                pass

    # Check nested structures for target (including valuation_summary, dcf, etc.)
    if not target:
        nested_keys = ["trade_note", "analysis", "summary", "result", "trade_setup",
                       "valuation_summary", "dcf", "valuation", "recommendation",
                       "trade_specifications", "investment_memo"]
        for nested_key in nested_keys:
            if nested_key in output and isinstance(output[nested_key], dict):
                nested_obj = output[nested_key]
                for key in target_keys:
                    if key in nested_obj:
                        try:
                            val = nested_obj[key]
                            if isinstance(val, (int, float)) and val > 0:
                                target = float(val)
                                break
                        except (ValueError, TypeError):
                            pass
                if target:
                    break
                # Check one more level deep (e.g., investment_memo.recommendation.target_price)
                for sub_nested_key in ["recommendation", "valuation", "summary"]:
                    if sub_nested_key in nested_obj and isinstance(nested_obj[sub_nested_key], dict):
                        sub_obj = nested_obj[sub_nested_key]
                        for key in target_keys:
                            if key in sub_obj:
                                try:
                                    val = sub_obj[key]
                                    if isinstance(val, (int, float)) and val > 0:
                                        target = float(val)
                                        break
                                except (ValueError, TypeError):
                                    pass
                        if target:
                            break
                if target:
                    break

    # Last resort: check for trade_setup.targets list (Fardeen's format)
    if not target:
        for container_key in ["trade_setup", "trade_specifications"]:
            if container_key in output and isinstance(output[container_key], dict):
                ts = output[container_key]
                for key in ["target_1", "target_2", "target", "take_profit"]:
                    if key in ts:
                        try:
                            val = ts[key]
                            if isinstance(val, (int, float)) and val > 0:
                                target = float(val)
                                break
                        except (ValueError, TypeError):
                            pass
                if target:
                    break

    return {"signal": signal or "HOLD", "target": target}


def load_all_outputs() -> tuple[list[dict], dict]:
    """
    Load all JSON files from shared_outputs/.
    Returns (raw_outputs, agent_data) where agent_data maps agent_name -> {signal, target, output}
    """
    raw_outputs = []
    agent_data = {}

    for json_file in SHARED_OUTPUTS.glob("*.json"):
        if json_file.name == "FINAL_RECOMMENDATION.json":
            continue

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            agent_name = classify_agent(json_file.name)
            if not agent_name:
                print(f"  Warning: Could not classify {json_file.name}")
                continue

            # Extract signal and target
            extracted = extract_signal(data)

            # Smart truncate for LLM (conservative - keeps all reasoning)
            truncated = smart_truncate(data)
            truncated["_agent"] = agent_name
            truncated["_source_file"] = json_file.name

            raw_outputs.append(truncated)
            agent_data[agent_name] = {
                "signal": extracted["signal"],
                "target": extracted["target"],
                "output": truncated,
            }

            target_str = f"${extracted['target']:.2f}" if extracted['target'] else "N/A"
            print(f"  Loaded: {json_file.name} -> {agent_name} ({extracted['signal']}, {target_str})")

        except Exception as e:
            print(f"  Failed to load {json_file.name}: {e}")

    return raw_outputs, agent_data


# =============================================================================
# PRE-SYNTHESIS ANALYSIS
# =============================================================================

def get_active_weights(active_agents: list) -> dict:
    """Normalize weights for only the agents that produced output."""
    active = {k: v for k, v in AGENT_WEIGHTS.items() if k in active_agents}
    total = sum(active.values())
    if total == 0:
        return {k: 1.0 / len(active) for k in active}
    return {k: v / total for k, v in active.items()}


def calculate_weighted_target(agent_data: dict) -> tuple[float, float, float]:
    """Calculate weighted average target price. Returns (weighted_avg, min, max)."""
    targets = []
    weighted_sum = 0
    weight_sum = 0

    weights = get_active_weights(list(agent_data.keys()))

    for agent, data in agent_data.items():
        target = data.get("target")
        if target and target > 0:
            targets.append(target)
            weight = weights.get(agent, 0)
            weighted_sum += target * weight
            weight_sum += weight

    if not targets:
        return 0, 0, 0

    weighted_avg = weighted_sum / weight_sum if weight_sum > 0 else sum(targets) / len(targets)
    return round(weighted_avg, 2), min(targets), max(targets)


def calculate_consensus(agent_data: dict) -> dict:
    """Calculate technical and fundamental consensus."""
    tech_signals = []
    fund_signals = []

    for agent, data in agent_data.items():
        signal = data.get("signal", "HOLD")
        if agent in TECHNICAL_AGENTS:
            tech_signals.append(signal)
        elif agent in FUNDAMENTAL_AGENTS:
            fund_signals.append(signal)

    def get_consensus(signals: list) -> str:
        if not signals:
            return "N/A"
        buys = sum(1 for s in signals if s in ["STRONG_BUY", "BUY"])
        sells = sum(1 for s in signals if s in ["STRONG_SELL", "SELL"])

        if buys > sells:
            return "BUY"
        elif sells > buys:
            return "SELL"
        return "HOLD"

    return {
        "technical": get_consensus(tech_signals),
        "fundamental": get_consensus(fund_signals),
        "tech_signals": tech_signals,
        "fund_signals": fund_signals,
    }


def calculate_confidence(agent_data: dict) -> int:
    """
    Confidence based on agreement level, not agent-reported confidence.
    Returns 40-85 based on how many agents agree.
    """
    signals = [data.get("signal", "HOLD") for data in agent_data.values()]
    total = len(signals)

    if total == 0:
        return 50

    buys = sum(1 for s in signals if s in ["STRONG_BUY", "BUY"])
    sells = sum(1 for s in signals if s in ["STRONG_SELL", "SELL"])
    holds = sum(1 for s in signals if s == "HOLD")

    max_agreement = max(buys, sells, holds)
    ratio = max_agreement / total

    if ratio >= 0.83:   # 5/6 or 6/6
        return 85
    elif ratio >= 0.67:  # 4/6
        return 70
    elif ratio >= 0.5:   # 3/6
        return 55
    else:
        return 40


def detect_conflicts(agent_data: dict) -> list[str]:
    """Find direction conflicts between agents."""
    conflicts = []
    agents = list(agent_data.items())

    for i, (agent1, data1) in enumerate(agents):
        sig1 = data1.get("signal", "HOLD")
        for agent2, data2 in agents[i+1:]:
            sig2 = data2.get("signal", "HOLD")

            is_bullish1 = sig1 in ["BUY", "STRONG_BUY"]
            is_bearish1 = sig1 in ["SELL", "STRONG_SELL"]
            is_bullish2 = sig2 in ["BUY", "STRONG_BUY"]
            is_bearish2 = sig2 in ["SELL", "STRONG_SELL"]

            if (is_bullish1 and is_bearish2) or (is_bearish1 and is_bullish2):
                conflicts.append(f"{agent1} ({sig1}) vs {agent2} ({sig2})")

    return conflicts


def aggregate_key_levels(agent_data: dict) -> dict:
    """Aggregate support/resistance levels from all agents."""
    supports = []
    resistances = []

    for agent, data in agent_data.items():
        output = data.get("output", {})

        # Try various field names for support
        for key in ["support", "support_levels", "supports", "key_support"]:
            if key in output:
                val = output[key]
                if isinstance(val, list):
                    supports.extend([float(v) for v in val if isinstance(v, (int, float))])
                elif isinstance(val, (int, float)):
                    supports.append(float(val))

        # Try various field names for resistance
        for key in ["resistance", "resistance_levels", "resistances", "key_resistance"]:
            if key in output:
                val = output[key]
                if isinstance(val, list):
                    resistances.extend([float(v) for v in val if isinstance(v, (int, float))])
                elif isinstance(val, (int, float)):
                    resistances.append(float(val))

        # Check nested key_levels
        if "key_levels" in output and isinstance(output["key_levels"], dict):
            kl = output["key_levels"]
            for sup_key in ["support", "supports"]:
                if sup_key in kl:
                    val = kl[sup_key]
                    if isinstance(val, list):
                        supports.extend([float(v) for v in val if isinstance(v, (int, float))])
                    elif isinstance(val, (int, float)):
                        supports.append(float(val))
            for res_key in ["resistance", "resistances"]:
                if res_key in kl:
                    val = kl[res_key]
                    if isinstance(val, list):
                        resistances.extend([float(v) for v in val if isinstance(v, (int, float))])
                    elif isinstance(val, (int, float)):
                        resistances.append(float(val))

    # Sort and deduplicate (within 1% considered same level)
    def dedupe(levels: list, tolerance: float = 0.01) -> list:
        if not levels:
            return []
        levels = sorted(set(levels))
        result = [levels[0]]
        for lvl in levels[1:]:
            if (lvl - result[-1]) / result[-1] > tolerance:
                result.append(lvl)
        return [round(x, 2) for x in result[:5]]

    return {
        "support": dedupe(supports),
        "resistance": dedupe(resistances),
    }


def get_current_price(agent_data: dict) -> Optional[float]:
    """Try to extract current price from any agent output."""
    for agent, data in agent_data.items():
        output = data.get("output", {})
        for key in ["current_price", "price", "last_price", "close", "current"]:
            if key in output:
                try:
                    val = output[key]
                    if isinstance(val, (int, float)) and val > 0:
                        return float(val)
                except (ValueError, TypeError):
                    pass
        # Check nested
        for nested in ["trade_note", "analysis", "summary", "market_data"]:
            if nested in output and isinstance(output[nested], dict):
                for key in ["current_price", "price", "last_price", "close"]:
                    if key in output[nested]:
                        try:
                            val = output[nested][key]
                            if isinstance(val, (int, float)) and val > 0:
                                return float(val)
                        except (ValueError, TypeError):
                            pass
    return None


# =============================================================================
# LLM SYNTHESIS
# =============================================================================

# System prompt for the synthesis LLM
SYSTEM_PROMPT = """You are a Senior Portfolio Manager at a top-tier institutional asset management firm. You lead the Investment Committee's final recommendation process, synthesizing research from your team of specialized analysts.

YOUR ROLE:
- Synthesize multiple analyst reports into a single, actionable investment recommendation
- Weigh technical and fundamental perspectives according to the firm's methodology
- Identify and resolve conflicts between analysts with clear reasoning
- Produce institutional-quality investment memos suitable for portfolio allocation decisions

YOUR METHODOLOGY:
1. SIGNAL AGGREGATION: Weight analyst signals according to the firm's allocation (Technical: 40%, Fundamental: 60%)
2. CONFLICT RESOLUTION: When analysts disagree, examine their underlying reasoning and data quality to determine which view is more credible
3. RISK-ADJUSTED RETURNS: Always frame recommendations in terms of expected return vs. downside risk
4. POSITION SIZING: Recommend appropriate position sizes based on conviction level and risk profile

OUTPUT STANDARDS:
- Be specific with numbers (prices, percentages, ratios)
- Cite which analyst's reasoning supports each conclusion
- Address every conflict explicitly - never ignore disagreements
- Provide actionable entry/exit levels, not vague directional calls"""


def truncate_json_smart(data: dict, max_chars: int = 15000) -> str:
    """
    Truncate JSON intelligently - preserves complete fields, prioritizes key analysis fields.
    Never cuts mid-value, always removes complete key-value pairs if needed.
    """
    # Priority fields to always keep (reasoning, signals, recommendations)
    PRIORITY_FIELDS = {
        "recommendation", "signal", "action", "direction", "overall_signal",
        "investment_thesis", "thesis", "trade_rationale", "analysis_summary",
        "executive_summary", "rationale", "reasoning", "conclusion",
        "target_price", "price_target", "stop_loss", "entry_price",
        "confidence", "conviction", "risk_reward",
        "key_risks", "risks", "catalysts", "key_catalysts",
        "support", "resistance", "key_levels",
        "valuation", "fair_value", "intrinsic_value",
        "earnings", "revenue", "growth", "margins",
    }

    # Fields that can be dropped if needed (verbose/secondary)
    DROPPABLE_FIELDS = {
        "metadata", "debug", "raw", "intermediate", "cache",
        "timestamps", "version", "source", "methodology_notes",
    }

    def get_field_priority(key: str) -> int:
        key_lower = key.lower()
        if any(p in key_lower for p in PRIORITY_FIELDS):
            return 0  # Highest priority
        if any(d in key_lower for d in DROPPABLE_FIELDS):
            return 2  # Lowest priority
        return 1  # Normal priority

    # First attempt: full JSON
    full_json = json.dumps(data, indent=2, default=str)
    if len(full_json) <= max_chars:
        return full_json

    # Second attempt: remove droppable fields
    filtered = {k: v for k, v in data.items() if get_field_priority(k) < 2}
    filtered_json = json.dumps(filtered, indent=2, default=str)
    if len(filtered_json) <= max_chars:
        return filtered_json

    # Third attempt: compact formatting for non-priority fields
    result = {}
    for key, value in data.items():
        priority = get_field_priority(key)
        if priority == 0:
            result[key] = value  # Keep priority fields as-is
        elif priority == 1:
            # Truncate long strings in normal-priority fields
            if isinstance(value, str) and len(value) > 500:
                result[key] = value[:500] + "..."
            elif isinstance(value, dict):
                # Keep dict but truncate nested strings
                result[key] = {
                    k: (v[:300] + "..." if isinstance(v, str) and len(v) > 300 else v)
                    for k, v in value.items()
                }
            else:
                result[key] = value

    final_json = json.dumps(result, indent=2, default=str)

    # Last resort: hard truncate at field boundary
    if len(final_json) > max_chars:
        # Find last complete field before limit
        truncated = final_json[:max_chars]
        last_newline = truncated.rfind('\n  "')
        if last_newline > max_chars * 0.7:
            truncated = truncated[:last_newline]
        return truncated + '\n  "_truncated": true\n}'

    return final_json


def build_synthesis_prompt(
    ticker: str,
    agent_data: dict,
    raw_outputs: list,
    consensus: dict,
    conflicts: list,
    weighted_target: float,
    target_range: tuple,
    confidence: int,
    key_levels: dict,
    current_price: Optional[float],
) -> str:
    """Build the structured LLM prompt - industrial grade."""

    weights = get_active_weights(list(agent_data.keys()))
    analysis_date = datetime.now().strftime('%Y-%m-%d')

    # Calculate additional metrics for context
    total_agents = len(agent_data)
    tech_count = sum(1 for a in agent_data if a in TECHNICAL_AGENTS)
    fund_count = sum(1 for a in agent_data if a in FUNDAMENTAL_AGENTS)

    buys = sum(1 for d in agent_data.values() if d.get("signal") in ["STRONG_BUY", "BUY"])
    sells = sum(1 for d in agent_data.values() if d.get("signal") in ["STRONG_SELL", "SELL"])
    holds = total_agents - buys - sells

    # Upside/downside calculation
    if current_price and weighted_target:
        upside_pct = ((weighted_target - current_price) / current_price) * 100
    else:
        upside_pct = None

    # Build analyst summary table
    analyst_table = "| Analyst | Type | Weight | Signal | Target | Conviction |\n"
    analyst_table += "|---------|------|--------|--------|--------|------------|\n"

    for agent in TECHNICAL_AGENTS + FUNDAMENTAL_AGENTS:
        if agent in agent_data:
            d = agent_data[agent]
            agent_type = "Technical" if agent in TECHNICAL_AGENTS else "Fundamental"
            w = weights.get(agent, 0) * 100
            target_str = f"${d['target']:.2f}" if d['target'] else "N/A"
            analyst_table += f"| {agent} | {agent_type} | {w:.0f}% | {d['signal']} | {target_str} | - |\n"

    # Build conflicts section with more detail
    if conflicts:
        conflicts_detail = "**CRITICAL: The following analyst conflicts require explicit resolution:**\n\n"
        for i, c in enumerate(conflicts, 1):
            conflicts_detail += f"{i}. {c}\n"
        conflicts_detail += "\nYou MUST address each conflict in your synthesis. Explain:\n"
        conflicts_detail += "- Which view you favor and why\n"
        conflicts_detail += "- What data/reasoning supports your chosen position\n"
        conflicts_detail += "- Under what conditions the opposing view would be correct\n"
    else:
        conflicts_detail = "**No major conflicts detected.** Analysts show broad agreement on direction."

    # Build full agent reports section with smart truncation
    agent_reports = ""
    for output in raw_outputs:
        agent_name = output.get("_agent", "unknown")
        agent_type = "TECHNICAL" if agent_name in TECHNICAL_AGENTS else "FUNDAMENTAL"
        weight = weights.get(agent_name, 0) * 100

        agent_reports += f"\n{'='*60}\n"
        agent_reports += f"ANALYST: {agent_name.upper()} ({agent_type}, {weight:.0f}% weight)\n"
        agent_reports += f"{'='*60}\n"

        # Remove internal fields and smart truncate
        clean_output = {k: v for k, v in output.items() if not k.startswith("_")}
        agent_reports += truncate_json_smart(clean_output, max_chars=15000)
        agent_reports += "\n"

    # Build the prompt
    price_str = f"${current_price:.2f}" if current_price else "N/A"
    target_range_str = f"${target_range[0]:.2f} - ${target_range[1]:.2f}" if target_range[0] else "N/A"
    upside_str = f"{upside_pct:+.1f}%" if upside_pct else "N/A"

    support_str = ", ".join(f"${s:.2f}" for s in key_levels.get("support", [])) or "N/A"
    resistance_str = ", ".join(f"${r:.2f}" for r in key_levels.get("resistance", [])) or "N/A"

    prompt = f"""
================================================================================
                        INVESTMENT COMMITTEE BRIEFING
                              {ticker} ANALYSIS
                              {analysis_date}
================================================================================

SECTION 1: EXECUTIVE DASHBOARD
--------------------------------------------------------------------------------

**Current Price:** {price_str}
**Weighted Target Price:** ${weighted_target:.2f}
**Expected Upside/Downside:** {upside_str}
**Target Range (Min-Max):** {target_range_str}

**Analyst Consensus:**
- Technical Consensus ({tech_count} analysts, 40% weight): {consensus['technical']}
- Fundamental Consensus ({fund_count} analysts, 60% weight): {consensus['fundamental']}

**Signal Distribution:** {buys} BUY | {holds} HOLD | {sells} SELL
**Agreement-Based Confidence:** {confidence}%

**Key Technical Levels:**
- Support: {support_str}
- Resistance: {resistance_str}

--------------------------------------------------------------------------------
SECTION 2: ANALYST SIGNALS SUMMARY
--------------------------------------------------------------------------------

{analyst_table}

--------------------------------------------------------------------------------
SECTION 3: CONFLICT ANALYSIS
--------------------------------------------------------------------------------

{conflicts_detail}

--------------------------------------------------------------------------------
SECTION 4: FULL ANALYST REPORTS
--------------------------------------------------------------------------------

The following contains the complete analysis from each analyst. Use this detailed
reasoning to inform your synthesis - do not rely solely on the signal summaries above.

{agent_reports}

================================================================================
SECTION 5: SYNTHESIS REQUIREMENTS
================================================================================

Produce a comprehensive investment recommendation in JSON format. Your output must:

1. **AGGREGATE SIGNALS** using the weighting methodology (Tech 40%, Fund 60%)
2. **RESOLVE ALL CONFLICTS** explicitly - cite which analyst's reasoning you favor
3. **SYNTHESIZE REASONING** - reference specific insights from analyst reports
4. **PROVIDE ACTIONABLE LEVELS** - specific entry, target, and stop-loss prices
5. **QUANTIFY RISK/REWARD** - expected return, probability-weighted scenarios

OUTPUT JSON SCHEMA:

{{
    "ticker": "{ticker}",
    "company_name": "<full company name>",
    "analysis_date": "{analysis_date}",

    "recommendation": "<STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL>",
    "confidence": <40-85 based on agreement level>,
    "conviction_rationale": "<1-2 sentences explaining confidence level>",
    "investment_horizon": "<1-3 months|3-6 months|6-12 months>",

    "current_price": <float>,
    "target_price": <float>,
    "stop_loss": <float>,
    "expected_return_pct": <float>,
    "max_drawdown_pct": <float>,
    "risk_reward_ratio": <float>,

    "position_sizing": {{
        "recommended_pct": <1-5>,
        "max_pct": <5-10>,
        "sizing_rationale": "<why this size>"
    }},

    "consensus": {{
        "technical": "{consensus['technical']}",
        "fundamental": "{consensus['fundamental']}",
        "agreement_level": "{buys + holds if consensus['technical'] != 'SELL' else sells + holds}/{total_agents} agents aligned",
        "divergence_notes": "<any notable disagreements>"
    }},

    "key_levels": {{
        "support": [<price1>, <price2>],
        "resistance": [<price1>, <price2>],
        "entry_zone": [<low>, <high>],
        "profit_targets": [<target1>, <target2>]
    }},

    "agent_attribution": {{
        "<agent_name>": {{
            "signal": "<signal>",
            "target": <price or null>,
            "weight": <0.0-1.0>,
            "key_insight": "<most important point from this analyst>"
        }}
    }},

    "conflicts_resolved": [
        {{
            "conflict": "<description of conflict>",
            "resolution": "<how you resolved it>",
            "favored_view": "<which analyst's view you sided with>",
            "reasoning": "<why>"
        }}
    ],

    "executive_summary": "<DETAILED - 150-250 words: Start with the recommendation and conviction level. Then explain the key drivers from both technical and fundamental perspectives. Include specific price targets, expected return, and investment timeline. Reference which analysts support this view and why their analysis is compelling. End with the risk-reward proposition.>",

    "technical_analysis_synthesis": "<DETAILED - 150-200 words synthesizing ALL technical analysts' findings. Structure: (1) Current trend direction and strength with specific indicator values (RSI, MACD, ADX). (2) Key support/resistance levels with price points. (3) Volume analysis and what it indicates. (4) Market regime assessment (trending/ranging/volatile). Cite which analyst (Fardeen, Tamer) provided each insight by name.>",

    "fundamental_analysis_synthesis": "<DETAILED - 150-200 words synthesizing ALL fundamental analysts' findings. Structure: (1) Valuation assessment with specific multiples (P/E, EV/EBITDA, P/FCF). (2) Growth prospects with revenue/earnings growth rates. (3) Profitability metrics (margins, ROE, ROIC). (4) Financial health (cash position, debt levels). (5) Competitive positioning. Cite which analyst (Shakzod, Lary, Mohamed, Daria) provided each insight by name.>",

    "investment_thesis": "<DETAILED - 200-300 words covering: (1) Core bull case - why this stock should appreciate. (2) Key catalysts with specific timing (earnings dates, product launches, regulatory milestones). (3) Valuation support - why the target price is justified. (4) Competitive moats and sustainable advantages. (5) What needs to go right for this trade to work. Be specific with numbers, dates, and analyst citations.>",

    "risk_assessment": {{
        "primary_risks": [
            {{
                "risk": "<specific risk description - be detailed, 1-2 sentences explaining the risk and its mechanism>",
                "category": "<market|company|sector|macro|execution>",
                "probability": "<low|medium|high>",
                "severity": "<minor|moderate|severe|critical>",
                "potential_impact_pct": <estimated % impact on position if risk materializes>,
                "warning_signs": ["<specific early indicator 1>", "<specific early indicator 2>"],
                "mitigation": "<specific action: e.g., stop-loss at $X, reduce position by 50% if Y happens>"
            }}
        ],
        "risk_narrative": "<DETAILED - 100-150 words explaining: (1) How these risks collectively inform the stop-loss placement. (2) Why the position size is appropriate given the risk profile. (3) How the investment horizon accounts for risk timing. Include specific numbers.>"
    }},

    "scenario_analysis": {{
        "bull_case": {{
            "probability": <0.0-1.0>,
            "target": <price>,
            "return_pct": <float>,
            "timeline": "<specific timeframe e.g., '3-6 months'>",
            "key_assumptions": ["<assumption 1>", "<assumption 2>", "<assumption 3>"],
            "catalysts": ["<specific catalyst with date/timeframe>", "<specific catalyst with date/timeframe>"],
            "narrative": "<50-75 words: What specific events need to occur? What metrics need to improve? Reference analyst insights.>"
        }},
        "base_case": {{
            "probability": <0.0-1.0>,
            "target": <price>,
            "return_pct": <float>,
            "timeline": "<specific timeframe>",
            "key_assumptions": ["<assumption 1>", "<assumption 2>", "<assumption 3>"],
            "catalysts": ["<specific catalyst with date/timeframe>", "<specific catalyst with date/timeframe>"],
            "narrative": "<50-75 words: The most probable path. What earnings/metrics need to materialize? How does current technical setup support this?>"
        }},
        "bear_case": {{
            "probability": <0.0-1.0>,
            "target": <price>,
            "return_pct": <float>,
            "timeline": "<specific timeframe>",
            "key_assumptions": ["<what goes wrong 1>", "<what goes wrong 2>", "<what goes wrong 3>"],
            "catalysts": ["<specific negative catalyst with timing>", "<specific negative catalyst with timing>"],
            "narrative": "<50-75 words: What breaks the thesis? What macro/company events cause the decline? At what price levels does the stop-loss trigger?>"
        }}
    }},

    "catalysts_timeline": [
        {{
            "catalyst": "<specific event e.g., 'Q2 2026 Earnings Release'>",
            "category": "<earnings|product|regulatory|macro|technical>",
            "expected_date": "<specific date or range e.g., 'April 25-28, 2026'>",
            "impact_direction": "<bullish|bearish|uncertain>",
            "expected_move_pct": <estimated % price move if catalyst plays out>,
            "reasoning": "<1-2 sentences: why this catalyst matters for the investment thesis and what to watch for>"
        }}
    ],

    "monitoring_checklist": [
        {{
            "metric": "<specific metric to track e.g., 'RSI (14-day)', 'Revenue Growth QoQ'>",
            "current_value": "<current value with units e.g., '55', '8.2%'>",
            "bullish_threshold": "<specific level that confirms thesis e.g., 'Above 60', 'Above 10%'>",
            "bearish_threshold": "<specific level that invalidates thesis e.g., 'Below 40', 'Below 5%'>",
            "action_if_bullish": "<specific action e.g., 'Add 1% to position', 'Raise stop-loss to $260'>",
            "action_if_bearish": "<specific action e.g., 'Exit 50% of position', 'Close position at market'>"
        }}
    ]
}}

CRITICAL CONTENT REQUIREMENTS:
- All narrative fields MUST meet the minimum word counts specified
- Reference specific analyst names (Fardeen, Tamer, Shakzod, Lary, Mohamed, Daria) when citing insights
- Include specific numbers (prices, percentages, ratios) throughout
- Include at least 3-4 items in catalysts_timeline and monitoring_checklist arrays
- Include at least 2-3 items in primary_risks array

IMPORTANT: Output ONLY valid JSON. No markdown, no explanatory text outside the JSON structure.
"""

    return prompt


def synthesize_with_llm(
    raw_outputs: list,
    agent_data: dict,
    consensus: dict,
    conflicts: list,
    weighted_target: float,
    target_range: tuple,
    confidence: int,
    key_levels: dict,
    current_price: Optional[float],
) -> dict:
    """Send structured prompt to LLM for synthesis. Uses Anthropic (Claude) as primary, OpenAI as fallback."""

    prompt = build_synthesis_prompt(
        TICKER, agent_data, raw_outputs, consensus, conflicts,
        weighted_target, target_range, confidence, key_levels, current_price
    )

    print(f"  Prompt length: {len(prompt):,} chars")

    # Try Anthropic first
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    text = None

    if anthropic_key:
        try:
            from anthropic import Anthropic
            print("  Using Anthropic Claude...")
            client = Anthropic(api_key=anthropic_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text
        except Exception as e:
            print(f"  Anthropic failed: {e}")
            text = None

    # Fallback to OpenAI
    if text is None and openai_key:
        try:
            from openai import OpenAI
            print("  Falling back to OpenAI GPT-4...")
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=8000,
                temperature=0.3
            )
            text = response.choices[0].message.content
        except Exception as e:
            print(f"  OpenAI failed: {e}")
            text = None

    if text is None:
        raise RuntimeError("Both Anthropic and OpenAI API calls failed. Check your API keys.")

    # Extract JSON from response (handle potential markdown wrapping)
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Try to parse JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to find JSON object in response
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        print("  Warning: Failed to parse LLM JSON response")
        return {"raw_response": text, "parse_error": True}


# =============================================================================
# CHART GENERATION
# =============================================================================

def generate_targets_chart(agent_data: dict, current_price: float, weighted_target: float) -> Optional[Path]:
    """Generate analyst target comparison dot plot chart."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  Warning: matplotlib not installed, skipping charts")
        return None

    # Prepare data - separate technical and fundamental
    tech_agents = []
    tech_targets = []
    tech_signals = []
    fund_agents = []
    fund_targets = []
    fund_signals = []

    for agent, data in agent_data.items():
        if data.get("target"):
            display_name = agent.replace("technical_", "").replace("fundamental_", "").title()
            if agent in TECHNICAL_AGENTS:
                tech_agents.append(display_name)
                tech_targets.append(data["target"])
                tech_signals.append(data.get("signal", "HOLD"))
            else:
                fund_agents.append(display_name)
                fund_targets.append(data["target"])
                fund_signals.append(data.get("signal", "HOLD"))

    all_agents = tech_agents + fund_agents
    all_targets = tech_targets + fund_targets
    all_signals = tech_signals + fund_signals

    if not all_targets:
        return None

    # Color mapping based on signal
    def get_color(sig):
        if sig in ["STRONG_BUY", "BUY"]:
            return "#10b981"  # Green
        elif sig in ["STRONG_SELL", "SELL"]:
            return "#ef4444"  # Red
        return "#6b7280"  # Gray

    colors = [get_color(s) for s in all_signals]

    # Calculate spread metrics
    min_target = min(all_targets)
    max_target = max(all_targets)
    spread = max_target - min_target
    spread_pct = (spread / current_price) * 100

    fig, ax = plt.subplots(figsize=(12, 7))

    # Y positions
    y_positions = list(range(len(all_agents)))

    # Draw horizontal lines from current price to each target (lollipop style)
    for i, (target, color) in enumerate(zip(all_targets, colors)):
        ax.hlines(y=i, xmin=current_price, xmax=target, color=color, alpha=0.4, linewidth=2)

    # Plot dots for each analyst target
    ax.scatter(all_targets, y_positions, c=colors, s=200, zorder=5, edgecolors='white', linewidths=2)

    # Add target price labels next to dots
    for i, (target, sig) in enumerate(zip(all_targets, all_signals)):
        offset = 5 if target >= current_price else -5
        ha = 'left' if target >= current_price else 'right'
        ax.annotate(f'${target:.0f}', (target, i), textcoords="offset points",
                    xytext=(offset, 0), ha=ha, va='center', fontsize=10, fontweight='bold')

    # Draw vertical reference lines
    ax.axvline(x=current_price, color='#3b82f6', linestyle='-', linewidth=3,
               label=f'Current: ${current_price:.2f}', zorder=3)
    ax.axvline(x=weighted_target, color='#8b5cf6', linestyle='--', linewidth=2,
               label=f'Consensus Target: ${weighted_target:.2f}', zorder=3)

    # Add shaded region showing target spread
    ax.axvspan(min_target, max_target, alpha=0.08, color='gray')

    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(all_agents, fontsize=11)
    ax.set_xlabel('Price ($)', fontsize=12)
    ax.set_title(f'{TICKER} - Analyst Target Price Comparison', fontsize=14, fontweight='bold', pad=15)

    # Add section divider if we have both types
    if tech_agents and fund_agents:
        ax.axhline(y=len(tech_agents) - 0.5, color='gray', linestyle=':', alpha=0.5)
        mid_x = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
        ax.text(mid_x, len(tech_agents) - 0.5, '  Tech ↑ | Fund ↓  ',
                fontsize=9, color='gray', va='center', ha='center', backgroundcolor='white')

    # Legend
    buy_patch = mpatches.Patch(color='#10b981', label='Bullish (BUY)')
    sell_patch = mpatches.Patch(color='#ef4444', label='Bearish (SELL)')
    hold_patch = mpatches.Patch(color='#6b7280', label='Neutral (HOLD)')
    current_line = plt.Line2D([0], [0], color='#3b82f6', linewidth=3, label=f'Current: ${current_price:.2f}')
    target_line = plt.Line2D([0], [0], color='#8b5cf6', linestyle='--', linewidth=2, label=f'Consensus: ${weighted_target:.2f}')

    ax.legend(handles=[buy_patch, hold_patch, sell_patch, current_line, target_line],
              loc='upper right', fontsize=9, framealpha=0.95)

    # Add upside/downside annotation
    upside_pct = ((weighted_target - current_price) / current_price) * 100
    direction = "Upside" if upside_pct > 0 else "Downside"
    ax.text(0.02, 0.98, f'Expected {direction}: {abs(upside_pct):.1f}%\nSpread: ${spread:.0f} ({spread_pct:.1f}%)',
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            color='#10b981' if upside_pct > 0 else '#ef4444',
            va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    # Grid
    ax.grid(True, axis='x', alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)

    # Adjust x-axis padding
    x_min = min(min_target, current_price) * 0.95
    x_max = max(max_target, current_price) * 1.05
    ax.set_xlim(x_min, x_max)

    plt.tight_layout()

    chart_path = SHARED_OUTPUTS / "analyst_targets_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return chart_path


def generate_price_chart(synthesis: dict) -> Optional[Path]:
    """Generate price action chart with key levels."""
    try:
        import matplotlib.pyplot as plt
        import yfinance as yf
    except ImportError:
        print("  Warning: matplotlib/yfinance not installed, skipping price chart")
        return None

    try:
        ticker = yf.Ticker(TICKER)
        hist = ticker.history(period="6mo")

        if hist.empty:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(hist.index, hist['Close'], color='#1e40af', linewidth=1.5, label='Price')

        if len(hist) >= 50:
            ma50 = hist['Close'].rolling(50).mean()
            ax.plot(hist.index, ma50, color='#f59e0b', linewidth=1, linestyle='--', label='50 MA')

        target = synthesis.get('target_price')
        stop_loss = synthesis.get('stop_loss')

        if target:
            ax.axhline(y=target, color='#10b981', linestyle='-', linewidth=2, label=f'Target: ${target:.2f}')
        if stop_loss:
            ax.axhline(y=stop_loss, color='#ef4444', linestyle='-', linewidth=2, label=f'Stop: ${stop_loss:.2f}')

        key_levels = synthesis.get('key_levels', {})
        for sup in key_levels.get('support', [])[:2]:
            ax.axhline(y=sup, color='#22c55e', linestyle=':', alpha=0.5)
        for res in key_levels.get('resistance', [])[:2]:
            ax.axhline(y=res, color='#f87171', linestyle=':', alpha=0.5)

        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.set_title(f'{TICKER} - 6 Month Price Action', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        chart_path = SHARED_OUTPUTS / "price_action_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()

        return chart_path

    except Exception as e:
        print(f"  Warning: Failed to generate price chart: {e}")
        return None


# =============================================================================
# PDF REPORT GENERATION
# =============================================================================

def generate_pdf_report(synthesis: dict, agent_data: dict, charts: dict) -> Optional[Path]:
    """Generate professional PDF report with compact layout."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak, Image, CondPageBreak
        )
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    except ImportError:
        print("  Warning: reportlab not installed, skipping PDF")
        return None

    pdf_path = SHARED_OUTPUTS / "FINAL_RECOMMENDATION.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=letter,
        rightMargin=0.6*inch, leftMargin=0.6*inch,
        topMargin=0.6*inch, bottomMargin=0.6*inch
    )

    # Colors
    NAVY = HexColor('#0f172a')
    BLUE = HexColor('#1e40af')
    GREEN = HexColor('#059669')
    RED = HexColor('#dc2626')
    GRAY = HexColor('#64748b')
    LIGHT = HexColor('#f1f5f9')

    # Styles - reduced spacing for tighter layout
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=22,
                                  textColor=NAVY, spaceAfter=6, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle('Subtitle', fontSize=11, textColor=GRAY,
                                     alignment=TA_CENTER, spaceAfter=12)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=13,
                                    textColor=BLUE, spaceBefore=10, spaceAfter=5)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=9,
                                 leading=12, alignment=TA_JUSTIFY, spaceAfter=5)
    note_style = ParagraphStyle('Note', parent=styles['Normal'], fontSize=8,
                                 textColor=GRAY, leading=10, spaceAfter=4, leftIndent=10)

    rec = synthesis.get('recommendation', 'HOLD').upper()
    if 'BUY' in rec:
        rec_color = GREEN
    elif 'SELL' in rec:
        rec_color = RED
    else:
        rec_color = GRAY

    rec_style = ParagraphStyle('Rec', fontSize=20, textColor=rec_color,
                                alignment=TA_CENTER, spaceAfter=15, fontName='Helvetica-Bold')

    story = []

    # === PAGE 1: Cover + Executive Summary ===
    ticker = synthesis.get('ticker', TICKER)
    company = synthesis.get('company_name', ticker)

    story.append(Paragraph(f"{ticker} Investment Memo", title_style))
    story.append(Paragraph(f"{company}", subtitle_style))
    story.append(Paragraph(f"Recommendation: {rec}", rec_style))

    # Key Metrics Table
    target = synthesis.get('target_price', 0)
    current = synthesis.get('current_price', 0)
    confidence = synthesis.get('confidence', 0)
    expected_return = synthesis.get('expected_return_pct', 0)
    rr_ratio = synthesis.get('risk_reward_ratio', 0)
    horizon = synthesis.get('investment_horizon', 'N/A')
    stop_loss = synthesis.get('stop_loss', 0)

    metrics_data = [
        ['Current Price', f'${current:.2f}' if current else 'N/A', 'Target Price', f'${target:.2f}' if target else 'N/A'],
        ['Expected Return', f'{expected_return:.1f}%' if expected_return else 'N/A', 'Stop Loss', f'${stop_loss:.2f}' if stop_loss else 'N/A'],
        ['Risk/Reward', f'{rr_ratio:.1f}x' if rr_ratio else 'N/A', 'Confidence', f'{confidence:.0f}%' if confidence else 'N/A'],
        ['Horizon', horizon, 'Date', synthesis.get('analysis_date', 'N/A')],
    ]

    metrics_table = Table(metrics_data, colWidths=[1.6*inch, 1.6*inch, 1.6*inch, 1.6*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), LIGHT),
        ('BACKGROUND', (2, 0), (2, -1), LIGHT),
        ('TEXTCOLOR', (0, 0), (-1, -1), NAVY),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cbd5e1')),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 10))

    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    exec_summary = synthesis.get('executive_summary', 'N/A')
    for para in exec_summary.split('\n\n'):
        if para.strip():
            story.append(Paragraph(para.strip(), body_style))

    # === Charts Section (after exec summary, same page if fits) ===
    story.append(CondPageBreak(4*inch))  # Only break if less than 4 inches left

    if charts.get('price') and charts['price'].exists():
        story.append(Paragraph("Price Action (6 Months)", heading_style))
        story.append(Image(str(charts['price']), width=6.2*inch, height=3.2*inch))
        story.append(Spacer(1, 8))

    if charts.get('targets') and charts['targets'].exists():
        story.append(CondPageBreak(3.5*inch))
        story.append(Paragraph("Analyst Target Prices", heading_style))
        story.append(Image(str(charts['targets']), width=6.2*inch, height=3.2*inch))

    # === Analysis Section ===
    story.append(CondPageBreak(3*inch))

    story.append(Paragraph("Technical Analysis", heading_style))
    tech_analysis = synthesis.get('technical_analysis_synthesis', synthesis.get('technical_analysis', 'N/A'))
    if isinstance(tech_analysis, str):
        for para in tech_analysis.split('\n\n'):
            if para.strip():
                story.append(Paragraph(para.strip(), body_style))

    story.append(CondPageBreak(2*inch))
    story.append(Paragraph("Fundamental Analysis", heading_style))
    fund_analysis = synthesis.get('fundamental_analysis_synthesis', synthesis.get('fundamental_analysis', 'N/A'))
    if isinstance(fund_analysis, str):
        for para in fund_analysis.split('\n\n'):
            if para.strip():
                story.append(Paragraph(para.strip(), body_style))

    # === Investment Thesis ===
    story.append(CondPageBreak(2.5*inch))

    story.append(Paragraph("Investment Thesis", heading_style))
    thesis = synthesis.get('investment_thesis', 'N/A')
    for para in thesis.split('\n\n'):
        if para.strip():
            story.append(Paragraph(para.strip(), body_style))

    scenarios = synthesis.get('scenario_analysis', {})
    if scenarios:
        story.append(CondPageBreak(2*inch))
        story.append(Paragraph("Scenario Analysis", heading_style))

        scenario_data = [['Scenario', 'Probability', 'Target', 'Return', 'Timeline']]
        for case in ['bull_case', 'base_case', 'bear_case']:
            if case in scenarios:
                s = scenarios[case]
                scenario_data.append([
                    case.replace('_', ' ').title(),
                    f"{s.get('probability', 0)*100:.0f}%",
                    f"${s.get('target', 0):.2f}",
                    f"{s.get('return_pct', 0):.1f}%",
                    s.get('timeline', 'N/A')
                ])

        scenario_table = Table(scenario_data, colWidths=[1.3*inch, 1.1*inch, 1.1*inch, 1.0*inch, 1.3*inch])
        scenario_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), NAVY),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
            ('TEXTCOLOR', (0, 1), (-1, -1), NAVY),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cbd5e1')),
        ]))
        story.append(scenario_table)

        # Add scenario narratives
        for case in ['bull_case', 'base_case', 'bear_case']:
            if case in scenarios and scenarios[case].get('narrative'):
                story.append(Paragraph(f"<b>{case.replace('_', ' ').title()}:</b> {scenarios[case]['narrative']}", note_style))

    # === Risk Assessment Section ===
    story.append(CondPageBreak(2.5*inch))

    story.append(Paragraph("Risk Assessment", heading_style))
    risk_assessment = synthesis.get('risk_assessment', {})

    # Handle new nested structure or legacy string format
    if isinstance(risk_assessment, dict):
        # Risk narrative first
        risk_narrative = risk_assessment.get('risk_narrative', '')
        if isinstance(risk_narrative, str):
            for para in risk_narrative.split('\n\n'):
                if para.strip():
                    story.append(Paragraph(para.strip(), body_style))

        # Key risks with full detail
        primary_risks = risk_assessment.get('primary_risks', [])
        if primary_risks:
            story.append(Spacer(1, 4))
            for risk_obj in primary_risks[:4]:
                if isinstance(risk_obj, dict):
                    risk_text = risk_obj.get('risk', '')
                    category = risk_obj.get('category', '')
                    prob = risk_obj.get('probability', '')
                    severity = risk_obj.get('severity', '')
                    impact_pct = risk_obj.get('potential_impact_pct', '')
                    mitigation = risk_obj.get('mitigation', '')
                    warnings = risk_obj.get('warning_signs', [])

                    if risk_text:
                        # Risk header with category/probability/severity
                        header = f"<b>{risk_text}</b>"
                        if category or prob or severity:
                            details = []
                            if category:
                                details.append(category.upper())
                            if prob:
                                details.append(f"Prob: {prob}")
                            if severity:
                                details.append(f"Severity: {severity}")
                            if impact_pct:
                                details.append(f"Impact: {impact_pct}%")
                            header += f" <font color='#64748b'>[{' | '.join(details)}]</font>"
                        story.append(Paragraph(f"• {header}", body_style))

                        # Warning signs and mitigation
                        if warnings:
                            warn_text = "Warning signs: " + ", ".join(warnings[:3])
                            story.append(Paragraph(warn_text, note_style))
                        if mitigation:
                            story.append(Paragraph(f"Mitigation: {mitigation}", note_style))
                elif isinstance(risk_obj, str):
                    story.append(Paragraph(f"• {risk_obj}", body_style))
    elif isinstance(risk_assessment, str):
        for para in risk_assessment.split('\n\n'):
            if para.strip():
                story.append(Paragraph(para.strip(), body_style))

    # === Catalysts Section ===
    catalysts = synthesis.get('catalysts_timeline', synthesis.get('key_catalysts', []))
    if catalysts:
        story.append(CondPageBreak(1.5*inch))
        story.append(Paragraph("Key Catalysts", heading_style))
        for cat in catalysts[:5]:
            if isinstance(cat, dict):
                cat_text = cat.get('catalyst', '')
                date = cat.get('expected_date', '')
                direction = cat.get('impact_direction', '')
                move_pct = cat.get('expected_move_pct', '')
                reasoning = cat.get('reasoning', '')

                if cat_text:
                    # Catalyst with direction indicator
                    direction_icon = "↑" if direction == "bullish" else "↓" if direction == "bearish" else "↔"
                    cat_line = f"<b>{direction_icon} {cat_text}</b>"
                    if date:
                        cat_line += f" ({date})"
                    if move_pct:
                        cat_line += f" - Expected move: {move_pct}%"
                    story.append(Paragraph(f"• {cat_line}", body_style))
                    if reasoning:
                        story.append(Paragraph(reasoning, note_style))
            elif isinstance(cat, str):
                story.append(Paragraph(f"• {cat}", body_style))

    # === Conflicts Section ===
    conflicts = synthesis.get('conflicts_resolved', synthesis.get('conflicts_addressed', []))
    if conflicts:
        story.append(CondPageBreak(1.5*inch))
        story.append(Paragraph("Conflicts Addressed", heading_style))
        for c in conflicts:
            if isinstance(c, dict):
                conflict_desc = c.get('conflict', '')
                resolution = c.get('resolution', '')
                reasoning = c.get('reasoning', '')
                if conflict_desc:
                    story.append(Paragraph(f"<b>Conflict:</b> {conflict_desc}", body_style))
                if resolution:
                    story.append(Paragraph(f"<b>Resolution:</b> {resolution}", body_style))
                if reasoning:
                    story.append(Paragraph(f"<i>{reasoning}</i>", note_style))
                story.append(Spacer(1, 4))
            elif isinstance(c, str):
                story.append(Paragraph(f"• {c}", body_style))

    # === Monitoring Checklist ===
    monitoring = synthesis.get('monitoring_checklist', [])
    if monitoring:
        story.append(CondPageBreak(1.5*inch))
        story.append(Paragraph("Monitoring Checklist", heading_style))
        # Add explanation note
        story.append(Paragraph(
            "<i>Post-entry metrics to track. These indicators help assess whether the investment thesis remains valid "
            "and guide position management decisions. Monitor these weekly or at key events.</i>",
            note_style
        ))
        story.append(Spacer(1, 4))

        for item in monitoring[:5]:
            if isinstance(item, dict):
                metric = item.get('metric', '')
                current = item.get('current_value', '')
                bullish = item.get('bullish_threshold', '')
                bearish = item.get('bearish_threshold', '')
                action_bull = item.get('action_if_bullish', '')
                action_bear = item.get('action_if_bearish', '')

                if metric:
                    line = f"<b>{metric}</b>"
                    if current:
                        line += f" (Current: {current})"
                    story.append(Paragraph(f"• {line}", body_style))

                    # Thresholds and actions
                    details = []
                    if bullish:
                        details.append(f"Bullish if {bullish}")
                        if action_bull:
                            details.append(f"→ {action_bull}")
                    if bearish:
                        details.append(f"Bearish if {bearish}")
                        if action_bear:
                            details.append(f"→ {action_bear}")
                    if details:
                        story.append(Paragraph(" | ".join(details), note_style))
            elif isinstance(item, str):
                story.append(Paragraph(f"• {item}", body_style))

    # Disclaimer
    story.append(Spacer(1, 15))
    disclaimer_style = ParagraphStyle('Disclaimer', fontSize=7, textColor=GRAY, alignment=TA_CENTER)
    story.append(Paragraph(
        "This report is for informational purposes only and does not constitute investment advice. "
        "Past performance is not indicative of future results. Generated by Master Agent v2.",
        disclaimer_style
    ))

    doc.build(story)
    return pdf_path


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_final_report(synthesis: dict, agent_data: dict, charts: dict) -> Path:
    """Save synthesized report to JSON and PDF."""
    json_path = SHARED_OUTPUTS / "FINAL_RECOMMENDATION.json"
    with open(json_path, 'w') as f:
        json.dump(synthesis, f, indent=2, default=str)
    print(f"\nFinal JSON saved: {json_path}")

    pdf_path = generate_pdf_report(synthesis, agent_data, charts)
    if pdf_path:
        print(f"Final PDF saved: {pdf_path}")

    return json_path


# =============================================================================
# MAIN PIPELINE
# =============================================================================

async def main():
    """Main pipeline: run agents -> analyze -> synthesize -> report."""

    print(f"\nAnalyzing: {TICKER}\n")

    # Step 1: Execute all agents in parallel
    await run_all_agents()

    # Step 2: Load and process outputs
    print("\n" + "=" * 60)
    print("LOADING & PROCESSING AGENT OUTPUTS")
    print("=" * 60)
    raw_outputs, agent_data = load_all_outputs()

    if not agent_data:
        print("No outputs found! Check if agents ran correctly.")
        return

    # Step 3: Pre-synthesis analysis
    print("\n" + "=" * 60)
    print("PRE-SYNTHESIS ANALYSIS")
    print("=" * 60)

    consensus = calculate_consensus(agent_data)
    print(f"  Technical Consensus: {consensus['technical']}")
    print(f"  Fundamental Consensus: {consensus['fundamental']}")

    conflicts = detect_conflicts(agent_data)
    print(f"  Conflicts: {len(conflicts)} detected")
    for c in conflicts:
        print(f"    - {c}")

    weighted_target, min_target, max_target = calculate_weighted_target(agent_data)
    print(f"  Weighted Target: ${weighted_target:.2f} (range: ${min_target:.2f} - ${max_target:.2f})")

    confidence = calculate_confidence(agent_data)
    print(f"  Agreement-Based Confidence: {confidence}%")

    key_levels = aggregate_key_levels(agent_data)
    print(f"  Support Levels: {key_levels.get('support', [])}")
    print(f"  Resistance Levels: {key_levels.get('resistance', [])}")

    current_price = get_current_price(agent_data)
    print(f"  Current Price: ${current_price:.2f}" if current_price else "  Current Price: N/A")

    # Step 4: Synthesize with LLM
    print("\n" + "=" * 60)
    print("SYNTHESIZING WITH LLM")
    print("=" * 60)

    synthesis = synthesize_with_llm(
        raw_outputs, agent_data, consensus, conflicts,
        weighted_target, (min_target, max_target), confidence,
        key_levels, current_price
    )

    # Step 5: Generate charts
    print("\n" + "=" * 60)
    print("GENERATING CHARTS")
    print("=" * 60)

    charts = {}

    if current_price:
        charts['targets'] = generate_targets_chart(agent_data, current_price, weighted_target)
        if charts['targets']:
            print(f"  Targets chart: {charts['targets']}")

    charts['price'] = generate_price_chart(synthesis)
    if charts['price']:
        print(f"  Price chart: {charts['price']}")

    # Step 6: Save final report
    print("\n" + "=" * 60)
    print("SAVING FINAL REPORT")
    print("=" * 60)

    save_final_report(synthesis, agent_data, charts)

    # Step 7: Display summary
    print("\n" + "=" * 60)
    print("FINAL RECOMMENDATION")
    print("=" * 60)
    print(f"  Ticker: {synthesis.get('ticker', TICKER)}")
    print(f"  Recommendation: {synthesis.get('recommendation', 'N/A')}")
    print(f"  Target Price: ${synthesis.get('target_price', 0):.2f}")
    print(f"  Stop Loss: ${synthesis.get('stop_loss', 0):.2f}")
    print(f"  Confidence: {synthesis.get('confidence', 0):.0f}%")
    print(f"  Risk/Reward: {synthesis.get('risk_reward_ratio', 0):.1f}x")
    print(f"\nExecutive Summary:")
    exec_sum = synthesis.get('executive_summary', 'N/A')
    print(f"  {exec_sum[:500]}..." if len(exec_sum) > 500 else f"  {exec_sum}")


if __name__ == "__main__":
    asyncio.run(main())
