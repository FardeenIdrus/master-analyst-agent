#!/usr/bin/env python3
"""
Agent Module - Fundamental Analyst AI Agent Orchestration
===============================================================================

MSc Coursework: IFTE0001 - AI Agents in Asset Management
Track A: Fundamental Analyst Agent - Step 11

This module provides the main FundamentalAnalystAgent class that orchestrates
the complete 10-step financial analysis pipeline from data collection through
memo generation and validation.

Pipeline Architecture:
    Step 1:  Data Collection (DataCollector)
    Step 2:  Data Processing (DataProcessor)
    Step 3:  Profitability Analysis (ProfitabilityAnalyzer)
    Step 4:  Cash Flow Analysis (CashFlowAnalyzer)
    Step 5:  Earnings Quality Analysis (EarningsQualityAnalyzer)
    Step 6:  Working Capital Analysis (WorkingCapitalAnalyzer)
    Step 7:  Financial Ratios (RatioCalculator)
    Step 8:  Valuation Analysis (ValuationAnalyzer)
    Step 9:  Memo Generation (MemoGenerator with Claude AI)
    Step 10: Validation (ValidationEngine)

Export Formats:
    - JSON: Complete structured data with all metrics
    - Markdown: Comprehensive investment memorandum
    - HTML: Professional interactive web report
    - PDF: Institutional-quality printable document

Author: MSc AI Agents in Asset Management
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class AnalysisStatus(Enum):
    """Overall analysis status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class StepStatus(Enum):
    """Individual step execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class StepResult:
    """Result from a single analysis step."""
    step_name: str
    step_number: int
    status: StepStatus
    result: Any
    execution_time_ms: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class AnalysisResults:
    """Complete analysis results container."""
    ticker: str
    company_name: str
    analysis_date: datetime
    overall_status: AnalysisStatus
    steps: Dict[str, StepResult] = field(default_factory=dict)
    memo: Any = None
    validation_report: Any = None
    execution_time_total_ms: float = 0.0
    
    def all_steps_successful(self) -> bool:
        return all(s.status in [StepStatus.SUCCESS, StepStatus.WARNING, StepStatus.SKIPPED] for s in self.steps.values())
    
    def get_errors(self) -> List[str]:
        return [e for s in self.steps.values() for e in s.errors]
    
    def get_warnings(self) -> List[str]:
        return [w for s in self.steps.values() for w in s.warnings]


# =============================================================================
# FUNDAMENTAL ANALYST AGENT
# =============================================================================

class FundamentalAnalystAgent:
    """
    Fundamental Analyst AI Agent - Main Orchestration Class.
    
    Coordinates the complete financial analysis pipeline for any stock ticker,
    generating comprehensive investment memoranda using Claude AI integration.
    """
    
    def __init__(self, ticker: str, api_key: Optional[str] = None):
        """Initialize agent with ticker and optional API key."""
        self._ticker = ticker.upper().strip()
        self._api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self._results: Optional[AnalysisResults] = None
        self._collected_data = None
        self._processed_data = None
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.info(f"FundamentalAnalystAgent initialized for {self._ticker}")
    
    @property
    def ticker(self) -> str:
        return self._ticker
    
    @property
    def results(self) -> Optional[AnalysisResults]:
        return self._results
    
    def run_analysis(self) -> AnalysisResults:
        """Execute the complete 10-step analysis pipeline."""
        logger.info(f"Starting analysis for {self._ticker}")
        start_time = time.time()
        
        self._results = AnalysisResults(
            ticker=self._ticker,
            company_name="",
            analysis_date=datetime.now(),
            overall_status=AnalysisStatus.IN_PROGRESS
        )
        
        try:
            self._execute_step_1()
            self._execute_step_2()
            self._execute_step_3()
            self._execute_step_4()
            self._execute_step_5()
            self._execute_step_6()
            self._execute_step_7()
            self._execute_step_8()
            self._execute_step_9()
            self._execute_step_10()
            
            self._results.overall_status = AnalysisStatus.COMPLETED if self._results.all_steps_successful() else AnalysisStatus.PARTIAL
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            self._results.overall_status = AnalysisStatus.FAILED
            raise
        finally:
            self._results.execution_time_total_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Analysis complete in {self._results.execution_time_total_ms:.0f}ms")
        return self._results
    
    def _execute_step(self, step_num: int, step_name: str, step_key: str, func) -> None:
        """Generic step execution wrapper."""
        logger.info(f"Step {step_num}: {step_name}")
        start = time.time()
        errors, warnings = [], []
        result = None
        status = StepStatus.SUCCESS
        
        try:
            result, warnings = func()
            if warnings:
                status = StepStatus.WARNING
        except Exception as e:
            logger.error(f"Step {step_num} failed: {e}")
            errors.append(str(e))
            status = StepStatus.FAILED
        
        self._results.steps[step_key] = StepResult(
            step_name=step_name, step_number=step_num, status=status,
            result=result, execution_time_ms=(time.time() - start) * 1000,
            errors=errors, warnings=warnings
        )
    
    def _execute_step_1(self):
        """Step 1: Data Collection."""
        def func():
            from data_collector import DataCollector
            collector = DataCollector()
            self._collected_data = collector.collect(self._ticker)
            if self._collected_data.company_info:
                self._results.company_name = self._collected_data.company_info.name
            warnings = self._collected_data.validation.warnings if self._collected_data.validation else []
            return self._collected_data, warnings
        self._execute_step(1, "Data Collection", "data_collection", func)
    
    def _execute_step_2(self):
        """Step 2: Data Processing."""
        def func():
            from data_processor import DataProcessor
            processor = DataProcessor()
            self._processed_data = processor.process(self._collected_data)
            warnings = self._processed_data.quality_metrics.anomalies if self._processed_data.quality_metrics else []
            return self._processed_data, warnings
        self._execute_step(2, "Data Processing", "data_processing", func)
    
    def _execute_step_3(self):
        """Step 3: Profitability Analysis."""
        def func():
            from profitability_analyzer import ProfitabilityAnalyzer
            analyzer = ProfitabilityAnalyzer(self._processed_data)
            result = analyzer.analyze()
            warnings = list(result.warnings) if result.warnings else []
            if hasattr(result, 'bridge') and result.bridge and not result.bridge.is_reconciled:
                warnings.append("EBIT bridge did not reconcile exactly")
            return result, warnings
        self._execute_step(3, "Profitability Analysis", "profitability", func)
    
    def _execute_step_4(self):
        """Step 4: Cash Flow Analysis."""
        def func():
            from cash_flow_analyzer import CashFlowAnalyzer
            analyzer = CashFlowAnalyzer(self._processed_data)
            result = analyzer.analyze()
            warnings = list(result.warnings) if result.warnings else []
            if result.red_flags:
                warnings.extend([f"RED FLAG: {rf}" for rf in result.red_flags])
            return result, warnings
        self._execute_step(4, "Cash Flow Analysis", "cash_flow", func)
    
    def _execute_step_5(self):
        """Step 5: Earnings Quality Analysis."""
        def func():
            from earnings_quality_analyzer import EarningsQualityAnalyzer
            analyzer = EarningsQualityAnalyzer(self._processed_data)
            result = analyzer.analyze()
            warnings = list(result.warnings) if result.warnings else []
            if result.red_flags:
                warnings.extend([f"RED FLAG: {rf.description}" for rf in result.red_flags])
            return result, warnings
        self._execute_step(5, "Earnings Quality Analysis", "earnings_quality", func)
    
    def _execute_step_6(self):
        """Step 6: Working Capital Analysis."""
        def func():
            from working_capital_analyzer import WorkingCapitalAnalyzer
            analyzer = WorkingCapitalAnalyzer(self._processed_data)
            result = analyzer.analyze()
            warnings = list(result.warnings) if result.warnings else []
            if result.alerts:
                warnings.extend([f"ALERT: {a.message}" for a in result.alerts])
            return result, warnings
        self._execute_step(6, "Working Capital Analysis", "working_capital", func)
    
    def _execute_step_7(self):
        """Step 7: Financial Ratios."""
        def func():
            from ratio_calculator import RatioCalculator
            calc = RatioCalculator(self._processed_data)
            result = calc.calculate_all()
            warnings = list(result.warnings) if result.warnings else []
            return result, warnings
        self._execute_step(7, "Financial Ratios", "ratios", func)
    
    def _execute_step_8(self):
        """Step 8: Valuation Analysis."""
        def func():
            from valuation import ValuationAnalyzer
            analyzer = ValuationAnalyzer(self._processed_data)
            result = analyzer.analyze()
            warnings = list(result.warnings) if result.warnings else []
            return result, warnings
        self._execute_step(8, "Valuation Analysis", "valuation", func)
    
    def _execute_step_9(self):
        """Step 9: Memo Generation."""
        def func():
            summary = self._build_analysis_summary()
            from memo_generator import MemoGenerator
            gen = MemoGenerator(api_key=self._api_key)
            self._results.memo = gen.generate(summary)
            return self._results.memo, []
        self._execute_step(9, "Memo Generation", "memo", func)
    
    def _execute_step_10(self):
        """Step 10: Validation."""
        def func():
            try:
                from validation import ValidationEngine, ValidationSeverity
                inp = {
                    'ticker': self._ticker,
                    'profitability': self._results.steps.get('profitability', StepResult("",0,StepStatus.SKIPPED,None,0)).result,
                    'cash_flow': self._results.steps.get('cash_flow', StepResult("",0,StepStatus.SKIPPED,None,0)).result,
                    'earnings_quality': self._results.steps.get('earnings_quality', StepResult("",0,StepStatus.SKIPPED,None,0)).result,
                    'working_capital': self._results.steps.get('working_capital', StepResult("",0,StepStatus.SKIPPED,None,0)).result,
                    'ratios': self._results.steps.get('ratios', StepResult("",0,StepStatus.SKIPPED,None,0)).result,
                    'valuation': self._results.steps.get('valuation', StepResult("",0,StepStatus.SKIPPED,None,0)).result,
                    'memo': self._results.memo
                }
                validator = ValidationEngine(inp)
                result = validator.run_full_validation()
                self._results.validation_report = result
                warnings = []
                for f in result.findings:
                    if f.severity == ValidationSeverity.WARNING:
                        warnings.append(f.message)
                return result, warnings
            except ImportError:
                return None, ["Validation module not available"]
        self._execute_step(10, "Validation", "validation", func)
    
    def _build_analysis_summary(self):
        """Build comprehensive AnalysisSummary from all step results."""
        from memo_generator import AnalysisSummary
        
        # Helper functions
        def safe_get(obj, *attrs, default=0.0):
            for attr in attrs:
                if obj is None:
                    return default
                obj = getattr(obj, attr, None)
            return obj if obj is not None else default
        
        def get_step_result(key):
            step = self._results.steps.get(key)
            return step.result if step else None
        
        prof = get_step_result('profitability')
        cf = get_step_result('cash_flow')
        eq = get_step_result('earnings_quality')
        wc = get_step_result('working_capital')
        ratios = get_step_result('ratios')
        val = get_step_result('valuation')
        
        # Company info
        info = self._collected_data.company_info if self._collected_data else None
        
        # Extract profitability
        bridge = safe_get(prof, 'bridge')
        metrics = safe_get(prof, 'metrics')
        
        ebit_current = safe_get(bridge, 'current_ebit', default=0)
        ebit_prior = safe_get(bridge, 'prior_ebit', default=0)
        ebit_change = safe_get(bridge, 'ebit_change', default=0)
        ebit_change_pct = ebit_change / ebit_prior if ebit_prior else 0
        
        volume_effect = 0.0
        margin_effect = 0.0
        primary_driver = 'Volume'
        comps = safe_get(bridge, 'components', default=[]) or []
        for c in comps:
            name = str(getattr(c, 'name', '')).lower()
            amt = getattr(c, 'amount', 0) or 0
            if 'volume' in name:
                volume_effect = amt
            elif 'margin' in name or 'gross' in name or 'rate' in name:
                margin_effect += amt
        if abs(margin_effect) > abs(volume_effect):
            primary_driver = 'Margin'
        
        # Margins - FIXED: Calculate changes directly to avoid unit conversion issues
        margins = safe_get(metrics, 'margins', default={}) or {}
        gm, gm_pr = 0, 0
        om, om_pr = 0, 0
        for k, v in margins.items():
            kn = str(k).lower()
            if 'gross' in kn:
                gm = getattr(v, 'current_value', 0) or 0
                gm_pr = getattr(v, 'prior_value', 0) or 0
            elif 'operating' in kn:
                om = getattr(v, 'current_value', 0) or 0
                om_pr = getattr(v, 'prior_value', 0) or 0
        
        # Normalize margins to decimals if they're percentages
        if gm > 1: gm = gm / 100
        if gm_pr > 1: gm_pr = gm_pr / 100
        if om > 1: om = om / 100
        if om_pr > 1: om_pr = om_pr / 100
        
        # Calculate changes directly (avoids source data unit issues)
        gm_ch = gm - gm_pr
        om_ch = om - om_pr
        
        # Cash flow
        cf_bridge = safe_get(cf, 'bridge')
        cf_metrics = safe_get(cf, 'conversion_metrics')
        cf_capex = safe_get(cf, 'capex_analysis')
        
        ni = safe_get(cf_bridge, 'net_income', default=0)
        ocf = safe_get(cf_bridge, 'operating_cash_flow', default=0)
        fcf = safe_get(cf_bridge, 'free_cash_flow', default=0)
        capex = abs(safe_get(cf_bridge, 'capital_expenditure', default=0))
        
        ccr = safe_get(cf_metrics, 'cash_conversion_rate', default=0)
        fcf_conv = safe_get(cf_metrics, 'fcf_conversion_rate', default=0)
        fcf_margin = safe_get(cf_metrics, 'fcf_margin', default=0)
        cf_quality = safe_get(cf_metrics, 'quality_rating', default='Adequate')
        if hasattr(cf_quality, 'value'):
            cf_quality = cf_quality.value
        
        capex_rev = safe_get(cf_capex, 'capex_to_revenue', default=0)
        capex_da = safe_get(cf_capex, 'capex_to_depreciation', default=1.0)
        
        # Earnings quality
        eq_score = safe_get(eq, 'overall_score', default=70)
        eq_rating = safe_get(eq, 'overall_rating', default='Good')
        if hasattr(eq_rating, 'value'):
            eq_rating = eq_rating.value
        accruals = safe_get(eq, 'accruals_analysis', 'accruals_ratio', default=0)
        red_flags = []
        rf_list = safe_get(eq, 'red_flags', default=[]) or []
        for rf in rf_list:
            if hasattr(rf, 'description'):
                red_flags.append(rf.description)
        
        # Working capital
        dso = safe_get(wc, 'dso', 'current_value', default=0)
        dso_ch = safe_get(wc, 'dso', 'change', default=0)
        dio = safe_get(wc, 'dio', 'current_value', default=0)
        dio_ch = safe_get(wc, 'dio', 'change', default=0)
        dpo = safe_get(wc, 'dpo', 'current_value', default=0)
        dpo_ch = safe_get(wc, 'dpo', 'change', default=0)
        ccc = safe_get(wc, 'cash_conversion_cycle', 'current_value', default=0)
        ccc_ch = safe_get(wc, 'cash_conversion_cycle', 'change', default=0)
        nwc = safe_get(wc, 'position', 'current_nwc', default=0)
        cr = safe_get(wc, 'position', 'current_ratio', default=1.0)
        qr = safe_get(wc, 'position', 'quick_ratio', default=1.0)
        wc_trend = safe_get(wc, 'cash_conversion_cycle', 'trend_direction', default='Stable')
        if hasattr(wc_trend, 'value'):
            wc_trend = wc_trend.value
        
        # Ratios - FIXED: Better extraction with fallbacks
        key_ratios = safe_get(ratios, 'key_ratios_summary', default={}) or {}
        roe = key_ratios.get('ROE', 0) or 0
        roa = key_ratios.get('ROA', 0) or 0
        roic = key_ratios.get('ROIC', 0) or 0
        de = key_ratios.get('D/E', 0) or 0
        icr = key_ratios.get('ICR', 0) or key_ratios.get('Interest Coverage', 0) or 0
        
        # Try to get asset turnover from various sources
        asset_turn = key_ratios.get('AT', 0) or key_ratios.get('Asset Turnover', 0) or 0
        if asset_turn == 0:
            efficiency = safe_get(ratios, 'efficiency_ratios', default={}) or {}
            # efficiency_ratios uses lowercase abbreviations as keys
            asset_turn = efficiency.get('at', 0) or efficiency.get('asset_turnover', 0) or efficiency.get('total_asset_turnover', 0) or 0
        
        # If still 0, try efficiency_ratios_dict property
        if asset_turn == 0 and hasattr(ratios, 'efficiency_ratios_dict'):
            eff_dict = ratios.efficiency_ratios_dict
            asset_turn = eff_dict.get('at', 0) or 0
        
        growth_ratios = safe_get(ratios, 'growth_ratios', default={}) or {}
        rev_growth = growth_ratios.get('revenue_growth', 0) or 0
        ni_growth = growth_ratios.get('ni_growth', 0) or growth_ratios.get('net_income_growth', 0) or 0
        
        # Valuation
        multiples = safe_get(val, 'key_multiples', default={}) or {}
        pe = multiples.get('P/E', 15) or 15
        ev_ebitda = multiples.get('EV/EBITDA', 10) or 10
        p_fcf = multiples.get('P/FCF', 20) or 20
        p_book = multiples.get('P/B', 3) or 3
        p_sales = multiples.get('P/S', 3) or 3
        
        if pe > 500:
            pe = 30
        if ev_ebitda > 500:
            ev_ebitda = 15
        
        val_assess = safe_get(val, 'valuation_assessment', default='Fairly Valued')
        if hasattr(val_assess, 'value'):
            val_assess = val_assess.value
        impl_growth = safe_get(val, 'implied_growth', 'implied_growth_rate', default=0.05)
        
        # Revenue - FIXED: Calculate growth directly if not available from ratios
        revenue_current = safe_get(metrics, 'revenue_current', default=0)
        revenue_prior = safe_get(metrics, 'revenue_prior', default=0)
        net_margin = ni / revenue_current if revenue_current else 0
        
        # Calculate revenue growth if not available from ratios
        if rev_growth == 0 and revenue_prior > 0:
            rev_growth = (revenue_current - revenue_prior) / revenue_prior
        # Normalize to decimal if it's a percentage
        if rev_growth > 1:
            rev_growth = rev_growth / 100
        
        # Calculate interest coverage if not available
        if icr == 0 and ebit_current > 0:
            # Try to get interest expense
            interest_exp = safe_get(cf_bridge, 'interest_expense', default=0)
            if interest_exp > 0:
                icr = ebit_current / interest_exp
        
        current_period = str(safe_get(bridge, 'current_period', default=datetime.now().strftime('%Y')))
        prior_period = str(safe_get(bridge, 'prior_period', default=str(int(datetime.now().strftime('%Y')) - 1)))
        
        return AnalysisSummary(
            ticker=self._ticker,
            company_name=self._results.company_name,
            sector=getattr(info, 'sector', 'Technology') if info else 'Technology',
            industry=getattr(info, 'industry', 'Unknown') if info else 'Unknown',
            current_price=getattr(info, 'current_price', 0) if info else 0,
            market_cap=getattr(info, 'market_cap', 0) if info else 0,
            revenue_current=revenue_current,
            revenue_prior=revenue_prior,
            revenue_growth=rev_growth,
            ebit_current=ebit_current,
            ebit_prior=ebit_prior,
            ebit_change=ebit_change,
            ebit_change_pct=ebit_change_pct,
            volume_effect=volume_effect,
            margin_effect=margin_effect,
            gross_margin_current=gm,
            gross_margin_prior=gm_pr,
            gross_margin_change=gm_ch,
            operating_margin_current=om,
            operating_margin_prior=om_pr,
            operating_margin_change=om_ch,
            net_margin_current=net_margin,
            primary_profit_driver=primary_driver,
            net_income=ni,
            operating_cash_flow=ocf,
            free_cash_flow=fcf,
            capex=capex,
            cash_conversion_rate=ccr if ccr <= 5 else ccr / 100,
            fcf_conversion_rate=fcf_conv if fcf_conv <= 5 else fcf_conv / 100,
            fcf_margin=fcf_margin if fcf_margin <= 1 else fcf_margin / 100,
            capex_to_revenue=capex_rev if capex_rev <= 1 else capex_rev / 100,
            capex_to_da=capex_da,
            cash_flow_quality=str(cf_quality),
            accruals_ratio=accruals if abs(accruals) <= 1 else accruals / 100,
            earnings_quality_score=eq_score,
            earnings_quality_rating=str(eq_rating),
            red_flags=red_flags,
            dso=dso,
            dso_change=dso_ch,
            dio=dio,
            dio_change=dio_ch,
            dpo=dpo,
            dpo_change=dpo_ch,
            ccc=ccc,
            ccc_change=ccc_ch,
            working_capital_trend=str(wc_trend),
            net_working_capital=nwc,
            current_ratio=cr,
            quick_ratio=qr,
            roe=roe if roe <= 5 else roe / 100,
            roa=roa if roa <= 1 else roa / 100,
            roic=roic if roic <= 5 else roic / 100,
            debt_to_equity=de,
            interest_coverage=icr,
            ni_growth=ni_growth if abs(ni_growth) <= 1 else ni_growth / 100,
            asset_turnover=asset_turn,
            pe_ratio=pe,
            ev_ebitda=ev_ebitda,
            price_to_fcf=p_fcf,
            price_to_book=p_book,
            price_to_sales=p_sales,
            valuation_assessment=str(val_assess),
            implied_growth=impl_growth if impl_growth <= 1 else impl_growth / 100,
            current_period=current_period,
            prior_period=prior_period
        )
    
    def export_results(self, output_dir: str) -> Dict[str, str]:
        """Export results to all professional formats."""
        if not self._results:
            raise ValueError("No results. Call run_analysis() first.")
        
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        files = {}
        files['json'] = str(self._export_json(path, ts))
        # files['markdown'] = str(self._export_markdown(path, ts))
        # files['html'] = str(self._export_html(path, ts))
        files['pdf'] = str(self._export_pdf(path, ts))
        files['shared'] = str(self._export_to_shared_outputs())  # Add this line
        
        logger.info(f"Results exported to {path}")
        return files
    
    def _export_json(self, path: Path, ts: str) -> Path:
        """Export comprehensive JSON with all metrics."""
        file = path / f"{self._ticker}_analysis_{ts}.json"
        
        # Get memo dict for complete data
        memo_dict = self._results.memo.to_dict() if self._results.memo and hasattr(self._results.memo, 'to_dict') else {}
        
        data = {
            "metadata": {
                "ticker": self._ticker,
                "company_name": self._results.company_name,
                "analysis_date": self._results.analysis_date.isoformat(),
                "analysis_date_formatted": self._results.analysis_date.strftime("%B %d, %Y"),
                "status": self._results.overall_status.value,
                "execution_time_ms": self._results.execution_time_total_ms
            },
            "recommendation": memo_dict.get("recommendation", {}),
            "score_breakdown": memo_dict.get("score_breakdown", {}),
            "company_profile": memo_dict.get("company_profile", {}),
            "financial_metrics": memo_dict.get("financial_metrics", {}),
            "analysis_sections": memo_dict.get("analysis_sections", []),
            "pipeline_results": {},
            "validation": {}
        }
        
        # Pipeline step details
        for name, step in self._results.steps.items():
            data["pipeline_results"][name] = {
                "step_number": step.step_number,
                "step_name": step.step_name,
                "status": step.status.value,
                "execution_time_ms": step.execution_time_ms,
                "errors": step.errors,
                "warnings": step.warnings[:20]
            }
        
        # Validation
        if self._results.validation_report:
            vr = self._results.validation_report
            data["validation"] = {
                "is_valid": vr.is_valid,
                "validation_score": vr.validation_score,
                "passed_checks": vr.passed_checks,
                "total_checks": vr.total_checks
            }
        
        with open(file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return file
    
    def _export_to_shared_outputs(self) -> Path:
        """Export JSON to shared_outputs for master agent."""
        # Go up 3 levels: agent.py → src → shakzod_fundamental_agent → fundamental_agents → master-analyst-agent
        master_repo_root = Path(__file__).parent.parent.parent.parent
        shared_dir = master_repo_root / "shared_outputs"
        shared_dir.mkdir(parents=True, exist_ok=True)
        
        file = shared_dir / f"fundamental_shakzod_{self._ticker}.json"
        
        memo_dict = self._results.memo.to_dict() if self._results.memo and hasattr(self._results.memo, 'to_dict') else {}
        
        data = {
            "metadata": {
                "ticker": self._ticker,
                "company_name": self._results.company_name,
                "analysis_date": self._results.analysis_date.isoformat(),
                "status": self._results.overall_status.value,
            },
            "recommendation": memo_dict.get("recommendation", {}),
            "score_breakdown": memo_dict.get("score_breakdown", {}),
            "financial_metrics": memo_dict.get("financial_metrics", {}),
        }
        
        with open(file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Shared output saved to: {file}")
        return file
    
    def _export_markdown(self, path: Path, ts: str) -> Path:
        """Export comprehensive Markdown memo."""
        file = path / f"{self._ticker}_investment_memo_{ts}.md"
        
        if self._results.memo and hasattr(self._results.memo, 'to_markdown'):
            content = self._results.memo.to_markdown()
        else:
            content = f"# Investment Memorandum: {self._ticker}\n\nAnalysis pending."
        
        with open(file, 'w') as f:
            f.write(content)
        
        return file
    
    def _export_html(self, path: Path, ts: str) -> Path:
        """Export professional HTML report."""
        file = path / f"{self._ticker}_investment_memo_{ts}.html"
        
        if self._results.memo and hasattr(self._results.memo, 'to_html'):
            content = self._results.memo.to_html()
        else:
            content = f"<html><body><h1>{self._ticker}</h1></body></html>"
        
        with open(file, 'w') as f:
            f.write(content)
        
        return file
    
    def _export_pdf(self, path: Path, ts: str) -> Path:
        """Export comprehensive institutional-quality PDF."""
        file = path / f"{self._ticker}_report_{ts}.pdf"
        
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch, cm
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
            from reportlab.graphics.shapes import Drawing, Rect
            from reportlab.graphics.charts.barcharts import VerticalBarChart
            
            doc = SimpleDocTemplate(str(file), pagesize=A4, rightMargin=0.6*inch, leftMargin=0.6*inch, topMargin=0.6*inch, bottomMargin=0.6*inch)
            
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=28, spaceAfter=20, textColor=colors.HexColor('#0f172a'), alignment=TA_CENTER)
            h1_style = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=16, spaceBefore=25, spaceAfter=12, textColor=colors.HexColor('#0f172a'))
            h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=13, spaceBefore=18, spaceAfter=8, textColor=colors.HexColor('#1e40af'))
            body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, alignment=TA_JUSTIFY, leading=14)
            small_style = ParagraphStyle('Small', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor('#64748b'))
            
            story = []
            memo = self._results.memo
            s = memo.summary if memo else None
            
            # Title page
            story.append(Spacer(1, 1*inch))
            story.append(Paragraph("INVESTMENT MEMORANDUM", title_style))
            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph(f"<b>{self._results.company_name}</b>", ParagraphStyle('Company', parent=styles['Heading1'], fontSize=24, alignment=TA_CENTER, textColor=colors.HexColor('#1e40af'))))
            story.append(Paragraph(f"({self._ticker})", ParagraphStyle('Ticker', parent=styles['Normal'], fontSize=16, alignment=TA_CENTER, textColor=colors.HexColor('#64748b'))))
            story.append(Spacer(1, 0.5*inch))
            
            # Recommendation box
            if memo:
                rec_color = {'STRONG BUY': '#059669', 'BUY': '#10b981', 'HOLD': '#f59e0b', 'SELL': '#ef4444', 'STRONG SELL': '#dc2626'}.get(memo.recommendation.value, '#6b7280')
                rec_data = [[f"RECOMMENDATION: {memo.recommendation.value}", f"CONFIDENCE: {memo.confidence_score:.0f}%"]]
                if memo.target_price:
                    rec_data[0].append(f"TARGET: ${memo.target_price:.2f}")
                rec_table = Table(rec_data, colWidths=[2.5*inch] * len(rec_data[0]))
                rec_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(rec_color)),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 12),
                    ('TOPPADDING', (0, 0), (-1, -1), 15),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
                    ('GRID', (0, 0), (-1, -1), 2, colors.white)
                ]))
                story.append(rec_table)
            
            story.append(Spacer(1, 0.4*inch))
            
            # Report info
            info_data = [
                ["Report Date", self._results.analysis_date.strftime("%B %d, %Y")],
                ["Sector", s.sector if s else "N/A"],
                ["Market Cap", f"${s.market_cap/1e9:.1f}B" if s and s.market_cap else "N/A"],
                ["Current Price", f"${s.current_price:.2f}" if s and s.current_price else "N/A"],
                ["Analysis Mode", memo.generation_mode.value.upper() if memo else "N/A"]
            ]
            info_table = Table(info_data, colWidths=[2*inch, 3*inch])
            info_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#64748b')),
            ]))
            story.append(info_table)
            
            story.append(PageBreak())
            
            # Score Breakdown
            if memo and memo.score_breakdown:
                story.append(Paragraph("Recommendation Score Analysis", h1_style))
                scores = memo.score_breakdown.get('scores', {})
                weights = memo.score_breakdown.get('weights', {})
                total = memo.score_breakdown.get('total_score', 0)
                
                score_data = [['Factor', 'Score', 'Weight', 'Contribution']]
                for factor in ['cash_flow', 'earnings_quality', 'profitability', 'valuation', 'growth']:
                    sc = scores.get(factor, 0)
                    wt = weights.get(factor, 0)
                    contrib = sc * wt
                    score_data.append([factor.replace('_', ' ').title(), f"{sc:.0f}/100", f"{wt:.0%}", f"{contrib:.1f}"])
                score_data.append(['TOTAL', '', '', f"{total:.1f}"])
                
                score_table = Table(score_data, colWidths=[2.2*inch, 1.2*inch, 1.2*inch, 1.4*inch])
                score_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
                    ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#f1f5f9')),
                    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ]))
                story.append(score_table)
                story.append(Spacer(1, 8))
                story.append(Paragraph("Score Thresholds: Strong Buy (78+), Buy (65-77), Hold (50-64), Sell (38-49), Strong Sell (<38)", small_style))
                story.append(Spacer(1, 20))
            
            # Financial Metrics Dashboard
            if s:
                story.append(Paragraph("Financial Metrics Dashboard", h1_style))
                
                # Profitability metrics
                story.append(Paragraph("Profitability", h2_style))
                prof_data = [
                    ['Metric', 'Current', 'Prior', 'Change'],
                    ['Revenue', f"${s.revenue_current:,.0f}M", f"${s.revenue_prior:,.0f}M", f"{s.revenue_growth*100:+.1f}%"],
                    ['EBIT', f"${s.ebit_current:,.0f}M", f"${s.ebit_prior:,.0f}M", f"{s.ebit_change_pct*100:+.1f}%"],
                    ['Gross Margin', f"{s.gross_margin_current*100:.1f}%", f"{s.gross_margin_prior*100:.1f}%", f"{s.gross_margin_change*100:+.1f}pp"],
                    ['Operating Margin', f"{s.operating_margin_current*100:.1f}%", f"{s.operating_margin_prior*100:.1f}%", f"{s.operating_margin_change*100:+.1f}pp"],
                    ['Net Margin', f"{s.net_margin_current*100:.1f}%", '-', '-'],
                ]
                self._add_metric_table(story, prof_data)
                
                # Cash flow metrics
                story.append(Paragraph("Cash Flow Quality", h2_style))
                cf_data = [
                    ['Metric', 'Value', 'Metric', 'Value'],
                    ['Net Income', f"${s.net_income:,.0f}M", 'Cash Conversion', f"{s.cash_conversion_rate*100:.0f}%"],
                    ['Operating CF', f"${s.operating_cash_flow:,.0f}M", 'FCF Margin', f"{s.fcf_margin*100:.1f}%"],
                    ['Free Cash Flow', f"${s.free_cash_flow:,.0f}M", 'Quality', s.cash_flow_quality],
                    ['CapEx', f"${s.capex:,.0f}M", 'CapEx/Rev', f"{s.capex_to_revenue*100:.1f}%"],
                ]
                self._add_metric_table(story, cf_data, cols=4)
                
                # Return metrics
                story.append(Paragraph("Return Metrics", h2_style))
                ret_data = [
                    ['ROE', f"{s.roe*100:.1f}%", 'ROIC', f"{s.roic*100:.1f}%"],
                    ['ROA', f"{s.roa*100:.1f}%", 'Asset Turnover', f"{s.asset_turnover:.2f}x"],
                ]
                self._add_metric_table(story, ret_data, cols=4, header=False)
                
                # Valuation
                story.append(Paragraph("Valuation Multiples", h2_style))
                val_data = [
                    ['P/E', f"{s.pe_ratio:.1f}x", 'P/FCF', f"{s.price_to_fcf:.1f}x"],
                    ['EV/EBITDA', f"{s.ev_ebitda:.1f}x", 'P/Book', f"{s.price_to_book:.2f}x"],
                    ['P/Sales', f"{s.price_to_sales:.2f}x", 'Assessment', s.valuation_assessment],
                ]
                self._add_metric_table(story, val_data, cols=4, header=False)
                
                # Working capital
                story.append(Paragraph("Working Capital Efficiency", h2_style))
                wc_data = [
                    ['Metric', 'Days', 'YoY Change'],
                    ['DSO', f"{s.dso:.0f}", f"{s.dso_change:+.0f}"],
                    ['DIO', f"{s.dio:.0f}", f"{s.dio_change:+.0f}"],
                    ['DPO', f"{s.dpo:.0f}", f"{s.dpo_change:+.0f}"],
                    ['CCC', f"{s.ccc:.0f}", f"{s.ccc_change:+.0f}"],
                ]
                self._add_metric_table(story, wc_data)
                
                # Earnings quality
                story.append(Paragraph("Earnings Quality", h2_style))
                eq_data = [
                    ['Quality Score', f"{s.earnings_quality_score:.0f}/100", 'Accruals Ratio', f"{s.accruals_ratio*100:.1f}%"],
                    ['Rating', s.earnings_quality_rating, 'Red Flags', str(len(s.red_flags) if s.red_flags else 0)],
                ]
                self._add_metric_table(story, eq_data, cols=4, header=False)
                
                # Financial health
                story.append(Paragraph("Financial Health", h2_style))
                icr_display = f"{s.interest_coverage:.1f}x" if s.interest_coverage > 0 else "N/A"
                fh_data = [
                    ['Current Ratio', f"{s.current_ratio:.2f}x", 'Interest Coverage', icr_display],
                    ['Quick Ratio', f"{s.quick_ratio:.2f}x", 'Debt/Equity', f"{s.debt_to_equity:.2f}x"],
                ]
                self._add_metric_table(story, fh_data, cols=4, header=False)
                
                story.append(PageBreak())
            
            # Analysis sections
            if memo and memo.sections:
                story.append(Paragraph("Detailed Analysis", h1_style))
                
                for section in memo.sections:
                    story.append(Paragraph(section.title, h2_style))
                    
                    # Clean content - remove markdown formatting for PDF
                    content = self._clean_markdown_for_pdf(section.content)
                    content = content.replace('\n\n', '<br/><br/>').replace('\n', ' ')
                    story.append(Paragraph(content, body_style))
                    
                    if section.key_points:
                        story.append(Spacer(1, 8))
                        story.append(Paragraph("<b>Key Points:</b>", body_style))
                        for pt in section.key_points[:5]:
                            clean_pt = self._clean_markdown_for_pdf(pt)
                            story.append(Paragraph(f"  - {clean_pt}", body_style))
                    
                    story.append(Spacer(1, 15))
            
            # Validation Results
            if self._results.validation_report:
                story.append(PageBreak())
                story.append(Paragraph("Validation Results", h1_style))
                vr = self._results.validation_report
                val_data = [
                    ['Status', 'PASSED' if vr.is_valid else 'REVIEW REQUIRED'],
                    ['Validation Score', f"{vr.validation_score:.0f}/100"],
                    ['Checks Passed', f"{vr.passed_checks}/{vr.total_checks}"],
                ]
                val_table = Table(val_data, colWidths=[2*inch, 2*inch])
                val_table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 11),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                    ('BACKGROUND', (1, 0), (1, 0), colors.HexColor('#d1fae5') if vr.is_valid else colors.HexColor('#fee2e2')),
                ]))
                story.append(val_table)
            
            # Pipeline execution
            story.append(PageBreak())
            story.append(Paragraph("Analysis Pipeline Execution", h1_style))
            pipe_data = [['Step', 'Name', 'Status', 'Time']]
            for step in self._results.steps.values():
                status_text = step.status.value.upper()
                pipe_data.append([str(step.step_number), step.step_name, status_text, f"{step.execution_time_ms:.0f}ms"])
            
            pipe_table = Table(pipe_data, colWidths=[0.5*inch, 2.5*inch, 1*inch, 1*inch])
            pipe_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(pipe_table)
            
            # Footer
            story.append(Spacer(1, 0.5*inch))
            story.append(Paragraph("Generated by Fundamental Analyst AI Agent | MSc Coursework: AI Agents in Asset Management", small_style))
            story.append(Paragraph("Powered by Claude AI (Anthropic) | This analysis is for educational purposes only.", small_style))
            
            doc.build(story)
            
        except ImportError:
            logger.warning("reportlab not available")
            with open(file, 'w') as f:
                f.write(f"PDF for {self._ticker} - Install reportlab for full PDF.\n")
        
        return file
    
    def _add_metric_table(self, story, data, cols=3, header=True):
        """Helper to add formatted metric table."""
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import Table, TableStyle, Spacer
        
        col_width = 5.5 * inch / cols
        table = Table(data, colWidths=[col_width] * cols)
        
        style = [
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ]
        
        if header:
            style.extend([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ])
        
        table.setStyle(TableStyle(style))
        story.append(table)
        story.append(Spacer(1, 12))
    
    def _clean_markdown_for_pdf(self, text: str) -> str:
        """
        Clean markdown formatting from text for PDF rendering.
        
        Removes asterisks used for bold/italic, hash symbols for headers,
        and other markdown syntax that would display incorrectly in PDF.
        
        Args:
            text: Raw text potentially containing markdown formatting
            
        Returns:
            Clean text suitable for PDF rendering
        """
        import re
        
        if not text:
            return ""
        
        # Remove bold+italic (triple asterisks or underscores) - must be first
        text = re.sub(r'\*\*\*([^*]+)\*\*\*', r'\1', text)
        text = re.sub(r'___([^_]+)___', r'\1', text)
        
        # Remove bold formatting (double asterisks or double underscores)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        
        # Remove italic formatting (single asterisks or single underscores)
        text = re.sub(r'\*([^*\n]+)\*', r'\1', text)
        text = re.sub(r'(?<!\w)_([^_\n]+)_(?!\w)', r'\1', text)
        
        # Remove any remaining standalone asterisks used for emphasis
        text = re.sub(r'(?<![*\w])\*(?![*\s])', '', text)
        text = re.sub(r'(?<![*\s])\*(?![*\w])', '', text)
        
        # Remove markdown headers (# symbols at start of lines)
        text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
        
        # Remove markdown bullet points (replace with dash)
        text = re.sub(r'^\s*[\*\-\+]\s+', '- ', text, flags=re.MULTILINE)
        
        # Remove numbered list formatting but keep numbers
        text = re.sub(r'^(\d+)\.\s+', r'\1. ', text, flags=re.MULTILINE)
        
        # Remove markdown links but keep the text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove inline code backticks
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove code block markers
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
        
        # Remove HTML-style tags that might have been left
        text = re.sub(r'</?(?:b|i|strong|em|u)>', '', text, flags=re.IGNORECASE)
        
        # Clean up multiple spaces
        text = re.sub(r'  +', ' ', text)
        
        # Clean up multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        # CRITICAL: Escape ampersands for reportlab's XML parser
        # First decode any existing HTML entities to plain text
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&apos;', "'")
        
        # Then re-encode ampersands for reportlab (which uses XML-style parsing)
        # This must be done LAST to prevent reportlab from misinterpreting & as entity start
        text = text.replace('&', '&amp;')
        
        return text
    
    def print_summary(self):
        """Print analysis summary to console."""
        if not self._results:
            print("No results available.")
            return
        
        print("\n" + "=" * 80)
        print(f"ANALYSIS SUMMARY: {self._results.company_name} ({self._ticker})")
        print("=" * 80)
        print(f"Status: {self._results.overall_status.value.upper()}")
        print(f"Execution Time: {self._results.execution_time_total_ms:.0f}ms")
        
        print("\nPipeline Steps:")
        for step in self._results.steps.values():
            status = {'success': '[OK]', 'warning': '[!!]', 'failed': '[XX]', 'skipped': '[--]'}.get(step.status.value, '[??]')
            print(f"  {status} Step {step.step_number}: {step.step_name} ({step.execution_time_ms:.0f}ms)")
        
        if self._results.memo:
            print(f"\nRecommendation: {self._results.memo.recommendation.value}")
            print(f"Confidence: {self._results.memo.confidence_score:.0f}%")
        
        if self._results.validation_report:
            vr = self._results.validation_report
            print(f"\nValidation: {'PASSED' if vr.is_valid else 'REVIEW'} ({vr.validation_score:.0f}/100)")
        
        print("=" * 80)


def run_fundamental_analysis(ticker: str, output_dir: Optional[str] = None) -> AnalysisResults:
    """Convenience function to run analysis."""
    agent = FundamentalAnalystAgent(ticker)
    results = agent.run_analysis()
    if output_dir:
        agent.export_results(output_dir)
    return results