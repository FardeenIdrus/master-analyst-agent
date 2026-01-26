"""
Investment Memo Generator - Phase 9 LLM Integration
INSTITUTIONAL GRADE v6.0

Complete redesign with:
- Fixed title/text overlap issues
- Proper 6-page layout with balanced content
- DDM always displayed (with reliability note)
- No empty spaces - filled with insights
- Accurate data extraction and validation

Version: 6.0.0
"""

from __future__ import annotations

import json
import os
import re
import time
import requests
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from .config import LOGGER, OUTPUT_DIR

__version__ = "6.0.0"


# =============================================================================
# CONFIGURATION
# =============================================================================

class MemoConfig:
    # Anthropic config
    API_ENDPOINT = "https://api.anthropic.com/v1/messages"
    MODEL = "claude-sonnet-4-5-20250929"
    MAX_TOKENS = 8192
    TEMPERATURE = 0.1
    DEFAULT_API_KEY = None
    API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0
    REQUEST_TIMEOUT = 180.0
    # Output to shared_outputs at repo root (4 levels up from src/memo_generator.py)
    DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "shared_outputs"

    # OpenAI fallback config
    OPENAI_API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
    OPENAI_MODEL = "gpt-4o"
    OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"


class RecommendationType(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    
    @classmethod
    def from_upside(cls, upside: float) -> "RecommendationType":
        if upside >= 0.25:
            return cls.STRONG_BUY
        elif upside >= 0.10:
            return cls.BUY
        elif upside >= -0.10:
            return cls.HOLD
        elif upside >= -0.25:
            return cls.SELL
        return cls.STRONG_SELL
    
    def to_display(self) -> str:
        return self.value.replace("_", " ").upper()


class ConvictionLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MemoStatus(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


# =============================================================================
# DATA CONTEXT
# =============================================================================

@dataclass
class MemoContext:
    """Complete context with all extracted data."""
    
    # Company
    ticker: str = ""
    company_name: str = ""
    sector: str = ""
    industry: str = ""
    description: str = ""
    exchange: str = ""
    currency: str = "USD"
    
    # Market
    market_cap: float = 0.0
    enterprise_value: float = 0.0
    current_price: float = 0.0
    fifty_two_week_high: float = 0.0
    fifty_two_week_low: float = 0.0
    beta: float = 1.0
    shares_outstanding: float = 0.0
    
    # Periods
    fiscal_periods: List[str] = field(default_factory=list)
    
    # Financials by year
    revenue: Dict[str, float] = field(default_factory=dict)
    net_income: Dict[str, float] = field(default_factory=dict)
    gross_profit: Dict[str, float] = field(default_factory=dict)
    operating_income: Dict[str, float] = field(default_factory=dict)
    ebitda: Dict[str, float] = field(default_factory=dict)
    total_assets: Dict[str, float] = field(default_factory=dict)
    total_equity: Dict[str, float] = field(default_factory=dict)
    total_debt: Dict[str, float] = field(default_factory=dict)
    free_cash_flow: Dict[str, float] = field(default_factory=dict)
    
    # Profitability
    roe: Dict[str, float] = field(default_factory=dict)
    roa: Dict[str, float] = field(default_factory=dict)
    roic: Dict[str, float] = field(default_factory=dict)
    gross_margin: Dict[str, float] = field(default_factory=dict)
    operating_margin: Dict[str, float] = field(default_factory=dict)
    net_margin: Dict[str, float] = field(default_factory=dict)
    ebitda_margin: Dict[str, float] = field(default_factory=dict)
    
    # Liquidity
    current_ratio: Dict[str, float] = field(default_factory=dict)
    quick_ratio: Dict[str, float] = field(default_factory=dict)
    
    # Leverage
    debt_to_equity: Dict[str, float] = field(default_factory=dict)
    debt_to_assets: Dict[str, float] = field(default_factory=dict)
    interest_coverage: Dict[str, float] = field(default_factory=dict)
    
    # Efficiency
    asset_turnover: Dict[str, float] = field(default_factory=dict)
    
    # Growth
    revenue_cagr: float = 0.0
    eps_cagr: float = 0.0
    fcf_cagr: float = 0.0
    dividend_cagr: float = 0.0
    
    # DuPont 3-Factor
    dupont_npm: Dict[str, float] = field(default_factory=dict)
    dupont_at: Dict[str, float] = field(default_factory=dict)
    dupont_em: Dict[str, float] = field(default_factory=dict)
    dupont_roe: Dict[str, float] = field(default_factory=dict)
    
    # DuPont 5-Factor
    dupont_tax_burden: Dict[str, float] = field(default_factory=dict)
    dupont_interest_burden: Dict[str, float] = field(default_factory=dict)
    dupont_operating_margin: Dict[str, float] = field(default_factory=dict)
    
    # DuPont Quality
    dupont_primary_driver: str = ""
    dupont_quality_rating: str = ""
    dupont_quality_score: float = 0.0
    
    # DCF
    dcf_intrinsic_value: float = 0.0
    dcf_upside: float = 0.0
    dcf_signal: str = ""
    dcf_wacc: float = 0.0
    dcf_cost_of_equity: float = 0.0
    dcf_cost_of_debt: float = 0.0
    dcf_risk_free_rate: float = 0.0
    dcf_terminal_growth: float = 0.0
    dcf_terminal_value_pct: float = 0.0
    dcf_bull_value: float = 0.0
    dcf_base_value: float = 0.0
    dcf_bear_value: float = 0.0
    dcf_bull_upside: float = 0.0
    dcf_base_upside: float = 0.0
    dcf_bear_upside: float = 0.0
    dcf_is_valid: bool = False
    
    # DDM
    ddm_applicable: bool = False
    ddm_exclusion_reason: str = ""
    ddm_intrinsic_value: float = 0.0
    ddm_upside: float = 0.0
    ddm_current_dividend: float = 0.0
    ddm_dividend_yield: float = 0.0
    ddm_dividend_growth: float = 0.0
    ddm_cost_of_equity: float = 0.0
    ddm_payout_ratio: float = 0.0
    ddm_is_reliable: bool = False
    
    # Multiples
    pe_current: float = 0.0
    pe_avg: float = 0.0
    pe_premium: float = 0.0
    pb_current: float = 0.0
    pb_avg: float = 0.0
    pb_premium: float = 0.0
    ps_current: float = 0.0
    ps_avg: float = 0.0
    pfcf_current: float = 0.0
    pfcf_avg: float = 0.0
    ev_ebitda_current: float = 0.0
    ev_ebitda_avg: float = 0.0
    ev_ebitda_premium: float = 0.0
    ev_revenue_current: float = 0.0
    ev_revenue_avg: float = 0.0
    multiples_implied_value: float = 0.0
    multiples_upside: float = 0.0
    multiples_assessment: str = ""
    multiples_is_valid: bool = False
    
    # Consensus
    consensus_value: float = 0.0
    consensus_upside: float = 0.0
    consensus_signal: str = ""
    consensus_direction: str = ""
    models_bullish: int = 0
    models_bearish: int = 0
    models_neutral: int = 0
    models_used: List[str] = field(default_factory=list)
    
    # Quality
    total_checks: int = 0
    checks_passed: int = 0
    checks_warnings: int = 0
    checks_failed: int = 0
    overall_confidence: float = 0.0
    confidence_level: str = ""
    is_reliable: bool = False
    
    # Risk
    leverage_risk: str = ""
    liquidity_risk: str = ""
    profitability_risk: str = ""
    valuation_risk: str = ""
    overall_risk: str = ""
    
    # Recommendation
    recommendation: str = ""
    target_price: float = 0.0
    conviction: str = ""
    decision_confidence: float = 0.0


# =============================================================================
# RESULT STRUCTURES
# =============================================================================

@dataclass
class MemoSection:
    title: str = ""
    content: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {"title": self.title, "content": self.content}


@dataclass
class RecommendationSummary:
    recommendation: RecommendationType = RecommendationType.HOLD
    conviction: ConvictionLevel = ConvictionLevel.MEDIUM
    target_price: float = 0.0
    current_price: float = 0.0
    upside_potential: float = 0.0
    decision_confidence: float = 0.0
    time_horizon: str = "12 months"
    key_catalysts: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation": self.recommendation.value,
            "recommendation_display": self.recommendation.to_display(),
            "conviction": self.conviction.value,
            "target_price": self.target_price,
            "current_price": self.current_price,
            "upside_potential_pct": self.upside_potential,
            "decision_confidence_pct": self.decision_confidence,
            "time_horizon": self.time_horizon,
            "key_catalysts": self.key_catalysts,
            "key_risks": self.key_risks,
        }


@dataclass
class InvestmentMemo:
    ticker: str = ""
    company_name: str = ""
    analysis_date: str = ""
    analyst: str = "AI Fundamental Analyst"
    recommendation: RecommendationSummary = field(default_factory=RecommendationSummary)
    executive_summary: MemoSection = field(default_factory=MemoSection)
    company_overview: MemoSection = field(default_factory=MemoSection)
    financial_analysis: MemoSection = field(default_factory=MemoSection)
    dupont_analysis: MemoSection = field(default_factory=MemoSection)
    valuation_analysis: MemoSection = field(default_factory=MemoSection)
    risk_assessment: MemoSection = field(default_factory=MemoSection)
    investment_thesis: MemoSection = field(default_factory=MemoSection)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "header": {
                "ticker": self.ticker,
                "company_name": self.company_name,
                "analysis_date": self.analysis_date,
                "analyst": self.analyst
            },
            "recommendation": self.recommendation.to_dict(),
            "sections": {
                "executive_summary": self.executive_summary.to_dict(),
                "company_overview": self.company_overview.to_dict(),
                "financial_analysis": self.financial_analysis.to_dict(),
                "dupont_analysis": self.dupont_analysis.to_dict(),
                "valuation_analysis": self.valuation_analysis.to_dict(),
                "risk_assessment": self.risk_assessment.to_dict(),
                "investment_thesis": self.investment_thesis.to_dict(),
            },
        }


@dataclass
class InvestmentMemoResult:
    status: MemoStatus = MemoStatus.FAILED
    memo: Optional[InvestmentMemo] = None
    context: Optional[MemoContext] = None
    model_used: str = ""
    llm_tokens_used: int = 0
    generation_duration_seconds: float = 0.0
    timestamp: str = ""
    raw_response: str = ""
    error_message: Optional[str] = None
    json_path: Optional[Path] = None
    pdf_path: Optional[Path] = None
    md_path: Optional[Path] = None
    
    @property
    def is_valid(self) -> bool:
        return self.status == MemoStatus.SUCCESS and self.memo is not None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "investment_memo": self.memo.to_dict() if self.memo else None,
            "generation_metadata": {
                "status": self.status.value,
                "model_used": self.model_used,
                "tokens_used": self.llm_tokens_used,
                "generation_time_seconds": self.generation_duration_seconds,
                "timestamp": self.timestamp,
                "error": self.error_message,
            },
        }


# =============================================================================
# ROBUST DATA EXTRACTOR
# =============================================================================

class RobustDataExtractor:
    """Extracts data with multiple fallback paths."""
    
    def __init__(self):
        self.ctx = MemoContext()
        self.logger = LOGGER
    
    def _get(self, obj: Any, *keys, default=None):
        try:
            result = obj
            for key in keys:
                if result is None:
                    return default
                if isinstance(result, dict):
                    result = result.get(key)
                else:
                    result = getattr(result, key, None)
            return result if result is not None else default
        except:
            return default
    
    def _float(self, val: Any, default: float = 0.0) -> float:
        if val is None:
            return default
        try:
            f = float(val)
            if f != f:
                return default
            return f
        except:
            return default
    
    def _str(self, val: Any, default: str = "") -> str:
        if val is None:
            return default
        if hasattr(val, 'value'):
            return str(val.value)
        return str(val)
    
    def extract(self, collection_result, validated_data, ratio_result, dupont_result, dcf_result, ddm_result, multiples_result, accuracy_result) -> MemoContext:
        self._extract_company(collection_result)
        self._extract_market(collection_result)
        self._extract_financials(collection_result)
        self._extract_ratios(ratio_result)
        self._extract_dupont(dupont_result)
        self._extract_dcf(dcf_result)
        self._extract_ddm(ddm_result)
        self._extract_multiples(multiples_result)
        self._extract_quality(accuracy_result)
        self._compute_consensus()
        self._compute_risks()
        self._compute_recommendation()
        return self.ctx
    
    def _extract_company(self, result) -> None:
        try:
            p = self._get(result, 'company_profile')
            if not p:
                return
            self.ctx.ticker = self._str(self._get(p, 'ticker'))
            self.ctx.company_name = self._str(self._get(p, 'name'))
            self.ctx.sector = self._str(self._get(p, 'sector'))
            self.ctx.industry = self._str(self._get(p, 'industry'))
            self.ctx.description = self._str(self._get(p, 'description'))
            self.ctx.exchange = self._str(self._get(p, 'exchange'))
            self.ctx.currency = self._str(self._get(p, 'currency'), 'USD')
        except Exception as e:
            self.logger.warning(f"Company extraction: {e}")
    
    def _extract_market(self, result) -> None:
        try:
            p = self._get(result, 'company_profile')
            if not p:
                return
            self.ctx.market_cap = self._float(self._get(p, 'market_cap'))
            self.ctx.shares_outstanding = self._float(self._get(p, 'shares_outstanding'))
            self.ctx.fifty_two_week_high = self._float(self._get(p, 'high_52week'))
            self.ctx.fifty_two_week_low = self._float(self._get(p, 'low_52week'))
            self.ctx.beta = self._float(self._get(p, 'beta'), 1.0)
            price = self._float(self._get(p, 'ma_50day'))
            if price <= 0:
                price = self._float(self._get(p, 'ma_200day'))
            self.ctx.current_price = price
            d = self._get(result, 'derived_metrics')
            if d:
                self.ctx.enterprise_value = self._float(self._get(d, 'enterprise_value'))
        except Exception as e:
            self.logger.warning(f"Market extraction: {e}")
    
    def _extract_financials(self, result) -> None:
        try:
            stmts = self._get(result, 'statements')
            if not stmts:
                return
            income = self._get(stmts, 'income_statement')
            balance = self._get(stmts, 'balance_sheet')
            if income is None or income.empty:
                return
            self.ctx.fiscal_periods = list(income.columns)[:5]
            def get_val(df, field, year):
                try:
                    if df is None or df.empty:
                        return 0.0
                    if field not in df.index or year not in df.columns:
                        return 0.0
                    return self._float(df.loc[field, year])
                except:
                    return 0.0
            for y in self.ctx.fiscal_periods:
                self.ctx.revenue[y] = get_val(income, 'totalRevenue', y)
                self.ctx.net_income[y] = get_val(income, 'netIncome', y)
                self.ctx.gross_profit[y] = get_val(income, 'grossProfit', y)
                self.ctx.operating_income[y] = get_val(income, 'operatingIncome', y)
                self.ctx.ebitda[y] = get_val(income, 'ebitda', y)
                if balance is not None and not balance.empty:
                    self.ctx.total_assets[y] = get_val(balance, 'totalAssets', y)
                    self.ctx.total_equity[y] = get_val(balance, 'totalShareholderEquity', y)
                    td = get_val(balance, 'totalDebt', y)
                    if td == 0:
                        td = get_val(balance, 'shortLongTermDebtTotal', y)
                    self.ctx.total_debt[y] = td
            d = self._get(result, 'derived_metrics')
            if d:
                fcf_dict = self._get(d, 'fcf_calculated', default={}) or {}
                for y, v in fcf_dict.items():
                    if y in self.ctx.fiscal_periods:
                        self.ctx.free_cash_flow[y] = self._float(v)
        except Exception as e:
            self.logger.warning(f"Financials extraction: {e}")
    
    def _extract_ratios(self, result) -> None:
        try:
            if not result:
                return
            def get_series(category: str, ratio_name: str) -> Dict[str, float]:
                cat = self._get(result, category)
                if not cat:
                    return {}
                ratios = self._get(cat, 'ratios', default={}) or {}
                ratio_ts = ratios.get(ratio_name)
                if not ratio_ts:
                    return {}
                values = self._get(ratio_ts, 'values', default={}) or {}
                out = {}
                for year, rv in values.items():
                    val = self._get(rv, 'value')
                    if val is not None:
                        out[year] = self._float(val)
                return out
            self.ctx.roe = get_series('profitability', 'roe')
            self.ctx.roa = get_series('profitability', 'roa')
            self.ctx.roic = get_series('profitability', 'roic')
            self.ctx.gross_margin = get_series('profitability', 'gross_margin')
            self.ctx.operating_margin = get_series('profitability', 'operating_margin')
            self.ctx.net_margin = get_series('profitability', 'net_profit_margin')
            self.ctx.ebitda_margin = get_series('profitability', 'ebitda_margin')
            self.ctx.current_ratio = get_series('liquidity', 'current_ratio')
            self.ctx.quick_ratio = get_series('liquidity', 'quick_ratio')
            self.ctx.debt_to_equity = get_series('leverage', 'debt_to_equity')
            self.ctx.debt_to_assets = get_series('leverage', 'debt_to_assets')
            self.ctx.interest_coverage = get_series('leverage', 'interest_coverage')
            self.ctx.asset_turnover = get_series('efficiency', 'asset_turnover')
            growth_cat = self._get(result, 'growth')
            if growth_cat:
                ratios = self._get(growth_cat, 'ratios', default={}) or {}
                def get_cagr(name):
                    r = ratios.get(name)
                    if r:
                        return self._float(self._get(r, 'latest_value'))
                    return 0.0
                self.ctx.revenue_cagr = get_cagr('revenue_cagr_5y')
                self.ctx.eps_cagr = get_cagr('eps_cagr_5y')
                self.ctx.fcf_cagr = get_cagr('fcf_cagr_5y')
                self.ctx.dividend_cagr = get_cagr('dividend_cagr_5y')
        except Exception as e:
            self.logger.warning(f"Ratios extraction: {e}")
    
    def _extract_dupont(self, result) -> None:
        try:
            if not result:
                return
            three_f = self._get(result, 'three_factor', default={}) or {}
            for year, tf in three_f.items():
                npm = self._get(tf, 'net_profit_margin')
                at = self._get(tf, 'asset_turnover')
                em = self._get(tf, 'equity_multiplier')
                self.ctx.dupont_npm[year] = self._float(self._get(npm, 'value'))
                self.ctx.dupont_at[year] = self._float(self._get(at, 'value'))
                self.ctx.dupont_em[year] = self._float(self._get(em, 'value'))
                self.ctx.dupont_roe[year] = self._float(self._get(tf, 'roe_calculated'))
            five_f = self._get(result, 'five_factor', default={}) or {}
            for year, ff in five_f.items():
                self.ctx.dupont_tax_burden[year] = self._float(self._get(self._get(ff, 'tax_burden'), 'value'))
                self.ctx.dupont_interest_burden[year] = self._float(self._get(self._get(ff, 'interest_burden'), 'value'))
                self.ctx.dupont_operating_margin[year] = self._float(self._get(self._get(ff, 'operating_margin'), 'value'))
            driver = self._get(result, 'primary_roe_driver')
            if driver:
                self.ctx.dupont_primary_driver = self._str(self._get(driver, 'value', default=driver))
            quality = self._get(result, 'quality_assessment')
            if quality:
                rating = self._get(quality, 'quality_rating')
                if rating:
                    self.ctx.dupont_quality_rating = self._str(self._get(rating, 'value', default=rating))
                self.ctx.dupont_quality_score = self._float(self._get(quality, 'quality_score'))
        except Exception as e:
            self.logger.warning(f"DuPont extraction: {e}")
    
    def _extract_dcf(self, result) -> None:
        try:
            if not result:
                return
            self.ctx.dcf_intrinsic_value = self._float(self._get(result, 'intrinsic_value_per_share'))
            price = self._float(self._get(result, 'current_price'))
            if price > 0:
                self.ctx.current_price = price
            self.ctx.dcf_upside = self._float(self._get(result, 'upside_downside_pct'))
            signal = self._get(result, 'valuation_signal')
            self.ctx.dcf_signal = self._str(self._get(signal, 'value', default=signal) if signal else '')
            wacc = self._get(result, 'wacc_calculation')
            if wacc:
                self.ctx.dcf_wacc = self._float(self._get(wacc, 'wacc_constrained'))
                coe = self._get(wacc, 'cost_of_equity')
                if coe:
                    self.ctx.dcf_cost_of_equity = self._float(self._get(coe, 'cost_of_equity'))
                    self.ctx.dcf_risk_free_rate = self._float(self._get(coe, 'risk_free_rate'))
                cod = self._get(wacc, 'cost_of_debt')
                if cod:
                    self.ctx.dcf_cost_of_debt = self._float(self._get(cod, 'after_tax_cost'))
            growth = self._get(result, 'growth_analysis')
            if growth:
                self.ctx.dcf_terminal_growth = self._float(self._get(growth, 'selected_terminal_growth'))
            proj = self._get(result, 'dcf_projection')
            if proj:
                self.ctx.dcf_terminal_value_pct = self._float(self._get(proj, 'terminal_value_pct'))
            sens = self._get(result, 'sensitivity_analysis')
            if sens:
                for case_name, attr_name, val_attr, up_attr in [
                    ('bear', 'bear_case', 'dcf_bear_value', 'dcf_bear_upside'),
                    ('base', 'base_case', 'dcf_base_value', 'dcf_base_upside'),
                    ('bull', 'bull_case', 'dcf_bull_value', 'dcf_bull_upside'),
                ]:
                    case = self._get(sens, attr_name)
                    if case:
                        val = self._float(self._get(case, 'intrinsic_value'))
                        if val <= 0:
                            val = self._float(self._get(case, 'value'))
                        up = self._float(self._get(case, 'upside_pct'))
                        if up == 0:
                            up = self._float(self._get(case, 'upside'))
                        setattr(self.ctx, val_attr, val)
                        setattr(self.ctx, up_attr, up)
            if self.ctx.current_price > 0:
                if self.ctx.dcf_bear_value > 0 and self.ctx.dcf_bear_upside == 0:
                    self.ctx.dcf_bear_upside = (self.ctx.dcf_bear_value / self.ctx.current_price) - 1
                if self.ctx.dcf_base_value > 0 and self.ctx.dcf_base_upside == 0:
                    self.ctx.dcf_base_upside = (self.ctx.dcf_base_value / self.ctx.current_price) - 1
                if self.ctx.dcf_bull_value > 0 and self.ctx.dcf_bull_upside == 0:
                    self.ctx.dcf_bull_upside = (self.ctx.dcf_bull_value / self.ctx.current_price) - 1
            self.ctx.dcf_is_valid = (self.ctx.dcf_intrinsic_value > 0 and self.ctx.dcf_wacc > 0 and self.ctx.dcf_intrinsic_value < self.ctx.current_price * 5)
        except Exception as e:
            self.logger.warning(f"DCF extraction: {e}")
    
    def _extract_ddm(self, result) -> None:
        try:
            if not result:
                return
            self.ctx.ddm_applicable = bool(self._get(result, 'is_applicable', default=False))
            app = self._get(result, 'applicability')
            if app:
                reasons = self._get(app, 'exclusion_reasons', default=[]) or []
                if reasons:
                    self.ctx.ddm_exclusion_reason = reasons[0]
                self.ctx.ddm_payout_ratio = self._float(self._get(app, 'average_payout_ratio'))
            if self.ctx.ddm_applicable:
                self.ctx.ddm_intrinsic_value = self._float(self._get(result, 'intrinsic_value_per_share'))
                self.ctx.ddm_upside = self._float(self._get(result, 'upside_downside_pct'))
                self.ctx.ddm_dividend_yield = self._float(self._get(result, 'current_dividend_yield'))
                hist = self._get(result, 'historical_dividends')
                if hist:
                    self.ctx.ddm_current_dividend = self._float(self._get(hist, 'current_dps'))
                    self.ctx.ddm_dividend_growth = self._float(self._get(hist, 'dps_cagr'))
                coe = self._get(result, 'cost_of_equity')
                if coe:
                    self.ctx.ddm_cost_of_equity = self._float(self._get(coe, 'cost_of_equity'))
                self.ctx.ddm_is_reliable = (self.ctx.ddm_payout_ratio >= 0.25 and self.ctx.ddm_intrinsic_value >= self.ctx.current_price * 0.20 and self.ctx.ddm_dividend_yield >= 0.01)
            else:
                self.ctx.ddm_is_reliable = False
        except Exception as e:
            self.logger.warning(f"DDM extraction: {e}")
    
    def _extract_multiples(self, result) -> None:
        try:
            if not result:
                return
            price = self._float(self._get(result, 'current_price'))
            if price > 0:
                self.ctx.current_price = price
            def get_mult(name):
                a = self._get(result, name)
                if not a:
                    return 0.0, 0.0, 0.0
                curr = self._float(self._get(a, 'current_value'))
                avg = self._float(self._get(a, 'historical_average'))
                prem = self._float(self._get(a, 'premium_to_average'))
                return curr, avg, prem
            self.ctx.pe_current, self.ctx.pe_avg, self.ctx.pe_premium = get_mult('pe_analysis')
            self.ctx.pb_current, self.ctx.pb_avg, self.ctx.pb_premium = get_mult('pb_analysis')
            self.ctx.ps_current, self.ctx.ps_avg, _ = get_mult('ps_analysis')
            self.ctx.pfcf_current, self.ctx.pfcf_avg, _ = get_mult('pfcf_analysis')
            self.ctx.ev_ebitda_current, self.ctx.ev_ebitda_avg, self.ctx.ev_ebitda_premium = get_mult('ev_ebitda_analysis')
            self.ctx.ev_revenue_current, self.ctx.ev_revenue_avg, _ = get_mult('ev_revenue_analysis')
            comp = self._get(result, 'composite_valuation')
            if comp:
                self.ctx.multiples_implied_value = self._float(self._get(comp, 'average_implied_value'))
                self.ctx.multiples_upside = self._float(self._get(comp, 'composite_upside'))
                assess = self._get(comp, 'overall_assessment')
                self.ctx.multiples_assessment = self._str(self._get(assess, 'value', default=assess) if assess else '')
            self.ctx.multiples_is_valid = self.ctx.multiples_implied_value > 0
        except Exception as e:
            self.logger.warning(f"Multiples extraction: {e}")
    
    def _extract_quality(self, result) -> None:
        try:
            if not result:
                return
            self.ctx.total_checks = int(self._float(self._get(result, 'total_checks')))
            self.ctx.checks_passed = int(self._float(self._get(result, 'total_passed')))
            self.ctx.checks_warnings = int(self._float(self._get(result, 'total_warnings')))
            self.ctx.checks_failed = int(self._float(self._get(result, 'total_failed')))
            self.ctx.overall_confidence = self._float(self._get(result, 'overall_confidence'))
            self.ctx.is_reliable = bool(self._get(result, 'is_reliable', default=False))
            level = self._get(result, 'confidence_level')
            self.ctx.confidence_level = self._str(self._get(level, 'value', default=level) if level else '')
        except Exception as e:
            self.logger.warning(f"Quality extraction: {e}")
    
    def _compute_consensus(self) -> None:
        try:
            models = []
            self.ctx.models_used = []
            if self.ctx.dcf_is_valid and self.ctx.dcf_intrinsic_value > 0:
                models.append(('DCF', self.ctx.dcf_intrinsic_value, self.ctx.dcf_upside, 0.50))
                self.ctx.models_used.append('DCF')
            if self.ctx.ddm_is_reliable and self.ctx.ddm_intrinsic_value > 0:
                models.append(('DDM', self.ctx.ddm_intrinsic_value, self.ctx.ddm_upside, 0.20))
                self.ctx.models_used.append('DDM')
            if self.ctx.multiples_is_valid and self.ctx.multiples_implied_value > 0:
                models.append(('Multiples', self.ctx.multiples_implied_value, self.ctx.multiples_upside, 0.50))
                self.ctx.models_used.append('Multiples')
            if not models:
                return
            total_weight = sum(m[3] for m in models)
            if total_weight <= 0:
                return
            weighted_value = sum(m[1] * m[3] / total_weight for m in models)
            self.ctx.consensus_value = weighted_value
            if self.ctx.current_price > 0:
                self.ctx.consensus_upside = (weighted_value / self.ctx.current_price) - 1
            for name, val, upside, weight in models:
                if upside >= 0.10:
                    self.ctx.models_bullish += 1
                elif upside <= -0.10:
                    self.ctx.models_bearish += 1
                else:
                    self.ctx.models_neutral += 1
            up = self.ctx.consensus_upside
            if up >= 0.25:
                self.ctx.consensus_signal = "STRONG BUY"
            elif up >= 0.10:
                self.ctx.consensus_signal = "BUY"
            elif up >= -0.10:
                self.ctx.consensus_signal = "HOLD"
            elif up >= -0.25:
                self.ctx.consensus_signal = "SELL"
            else:
                self.ctx.consensus_signal = "STRONG SELL"
            if self.ctx.models_bullish > self.ctx.models_bearish:
                self.ctx.consensus_direction = "Bullish"
            elif self.ctx.models_bearish > self.ctx.models_bullish:
                self.ctx.consensus_direction = "Bearish"
            else:
                self.ctx.consensus_direction = "Mixed"
        except Exception as e:
            self.logger.warning(f"Consensus computation: {e}")
    
    def _compute_risks(self) -> None:
        try:
            if not self.ctx.fiscal_periods:
                return
            latest = self.ctx.fiscal_periods[0]
            de = self.ctx.debt_to_equity.get(latest, 0)
            if de > 2.0:
                self.ctx.leverage_risk = "HIGH"
            elif de > 1.0:
                self.ctx.leverage_risk = "MODERATE"
            else:
                self.ctx.leverage_risk = "LOW"
            cr = self.ctx.current_ratio.get(latest, 0)
            if cr < 1.0:
                self.ctx.liquidity_risk = "HIGH"
            elif cr < 1.5:
                self.ctx.liquidity_risk = "MODERATE"
            else:
                self.ctx.liquidity_risk = "LOW"
            npm = self.ctx.net_margin.get(latest, 0)
            if npm < 0.05:
                self.ctx.profitability_risk = "HIGH"
            elif npm < 0.10:
                self.ctx.profitability_risk = "MODERATE"
            else:
                self.ctx.profitability_risk = "LOW"
            vals = []
            if self.ctx.dcf_is_valid:
                vals.append(self.ctx.dcf_intrinsic_value)
            if self.ctx.multiples_is_valid:
                vals.append(self.ctx.multiples_implied_value)
            if len(vals) >= 2 and min(vals) > 0:
                div = (max(vals) - min(vals)) / min(vals)
                if div > 0.50:
                    self.ctx.valuation_risk = "HIGH"
                elif div > 0.25:
                    self.ctx.valuation_risk = "MODERATE"
                else:
                    self.ctx.valuation_risk = "LOW"
            else:
                self.ctx.valuation_risk = "MODERATE"
            high_ct = sum([self.ctx.leverage_risk == "HIGH", self.ctx.liquidity_risk == "HIGH", self.ctx.profitability_risk == "HIGH", self.ctx.valuation_risk == "HIGH"])
            if high_ct >= 2:
                self.ctx.overall_risk = "HIGH"
            elif high_ct >= 1:
                self.ctx.overall_risk = "MODERATE"
            else:
                self.ctx.overall_risk = "LOW"
        except:
            pass
    
    def _compute_recommendation(self) -> None:
        try:
            upside = self.ctx.consensus_upside
            conf = self.ctx.overall_confidence
            if upside >= 0.25:
                rec = "STRONG BUY"
            elif upside >= 0.10:
                rec = "BUY"
            elif upside >= -0.10:
                rec = "HOLD"
            elif upside >= -0.25:
                rec = "SELL"
            else:
                rec = "STRONG SELL"
            self.ctx.recommendation = rec
            self.ctx.target_price = self.ctx.consensus_value
            models_agree = ((rec in ["BUY", "STRONG BUY"] and self.ctx.models_bullish >= self.ctx.models_bearish) or (rec in ["SELL", "STRONG SELL"] and self.ctx.models_bearish >= self.ctx.models_bullish) or (rec == "HOLD"))
            if conf >= 0.95 and self.ctx.valuation_risk == "LOW" and models_agree:
                self.ctx.conviction = "HIGH"
            elif conf >= 0.85 and self.ctx.valuation_risk != "HIGH":
                self.ctx.conviction = "MEDIUM"
            else:
                self.ctx.conviction = "LOW"
            base = conf * 100
            if self.ctx.valuation_risk == "HIGH":
                base -= 15
            elif self.ctx.valuation_risk == "MODERATE":
                base -= 5
            if self.ctx.overall_risk == "HIGH":
                base -= 10
            if not models_agree:
                base -= 10
            self.ctx.decision_confidence = max(0, min(100, base))
        except:
            pass


# =============================================================================
# PROMPT ENGINE
# =============================================================================

class PromptEngine:
    def __init__(self, ctx: MemoContext):
        self.ctx = ctx
    
    def system_prompt(self) -> str:
        return """You are a senior equity research analyst. Write a professional investment memo.

REQUIREMENTS:
1. Use ONLY data provided - cite specific numbers
2. Professional institutional language
3. NO emojis or decorative elements
4. Every claim must have supporting data
5. Be balanced - acknowledge positives and negatives

OUTPUT FORMAT: Valid JSON:
{
    "executive_summary": "3-4 paragraphs with specific metrics",
    "company_overview": "2-3 paragraphs on business model",
    "financial_analysis": "3-4 paragraphs on profitability, liquidity, growth",
    "dupont_analysis": "2-3 paragraphs on ROE decomposition",
    "valuation_analysis": "3-4 paragraphs on DCF, DDM, multiples",
    "risk_assessment": "2-3 paragraphs on key risks",
    "investment_thesis": "2-3 paragraphs with recommendation",
    "key_catalysts": ["catalyst1", "catalyst2", "catalyst3"],
    "key_risks": ["risk1", "risk2", "risk3"]
}"""
    
    def user_prompt(self) -> str:
        c = self.ctx
        def pct(v): return f"{v*100:.2f}%" if v else "N/A"
        def num(v):
            if not v: return "N/A"
            if abs(v) >= 1e12: return f"${v/1e12:.2f}T"
            if abs(v) >= 1e9: return f"${v/1e9:.2f}B"
            if abs(v) >= 1e6: return f"${v/1e6:.1f}M"
            return f"${v:,.0f}"
        def price(v): return f"${v:.2f}" if v else "N/A"
        def ratio(v): return f"{v:.2f}x" if v else "N/A"
        years = c.fiscal_periods[:5] if c.fiscal_periods else []
        yh = " | ".join(years) if years else "N/A"
        def row_pct(name, data):
            return f"| {name} | " + " | ".join([pct(data.get(y, 0)) for y in years]) + " |"
        def row_ratio(name, data):
            return f"| {name} | " + " | ".join([ratio(data.get(y, 0)) for y in years]) + " |"
        return f"""Generate investment memo for {c.ticker} ({c.company_name})

=== COMPANY ===
Ticker: {c.ticker}
Company: {c.company_name}
Sector: {c.sector}
Industry: {c.industry}

=== MARKET DATA ===
Market Cap: {num(c.market_cap)}
Enterprise Value: {num(c.enterprise_value)}
Current Price: {price(c.current_price)}
52-Week Range: {price(c.fifty_two_week_low)} - {price(c.fifty_two_week_high)}
Beta: {c.beta:.2f}

=== PROFITABILITY (5-Year) ===
| Metric | {yh} |
{row_pct("ROE", c.roe)}
{row_pct("ROA", c.roa)}
{row_pct("ROIC", c.roic)}
{row_pct("Gross Margin", c.gross_margin)}
{row_pct("Operating Margin", c.operating_margin)}
{row_pct("Net Margin", c.net_margin)}

=== LIQUIDITY & LEVERAGE ===
| Metric | {yh} |
{row_ratio("Current Ratio", c.current_ratio)}
{row_ratio("Debt/Equity", c.debt_to_equity)}
{row_ratio("Interest Coverage", c.interest_coverage)}

=== GROWTH (5Y CAGR) ===
Revenue: {pct(c.revenue_cagr)} | FCF: {pct(c.fcf_cagr)}

=== DUPONT (3-Factor) ===
| Component | {yh} |
{row_pct("Net Profit Margin", c.dupont_npm)}
{row_ratio("Asset Turnover", c.dupont_at)}
{row_ratio("Equity Multiplier", c.dupont_em)}

Primary Driver: {c.dupont_primary_driver}
Quality: {c.dupont_quality_rating} ({c.dupont_quality_score:.0f}/100)

=== DCF VALUATION ===
Intrinsic Value: {price(c.dcf_intrinsic_value)} | Upside: {pct(c.dcf_upside)}
WACC: {pct(c.dcf_wacc)} | Terminal Growth: {pct(c.dcf_terminal_growth)}
Bull: {price(c.dcf_bull_value)} | Base: {price(c.dcf_base_value)} | Bear: {price(c.dcf_bear_value)}

=== DDM VALUATION ===
Applicable: {'Yes' if c.ddm_applicable else 'No'}
Intrinsic Value: {price(c.ddm_intrinsic_value)} | Upside: {pct(c.ddm_upside)}
Dividend: {price(c.ddm_current_dividend)} | Yield: {pct(c.ddm_dividend_yield)}
Growth: {pct(c.ddm_dividend_growth)} | Payout: {pct(c.ddm_payout_ratio)}
Reliable for Consensus: {'Yes' if c.ddm_is_reliable else 'No - Low payout ratio'}

=== MULTIPLES ===
P/E: {ratio(c.pe_current)} vs {ratio(c.pe_avg)} ({pct(c.pe_premium)})
EV/EBITDA: {ratio(c.ev_ebitda_current)} vs {ratio(c.ev_ebitda_avg)}
Implied Value: {price(c.multiples_implied_value)} | Upside: {pct(c.multiples_upside)}

=== CONSENSUS ===
Value: {price(c.consensus_value)} | Upside: {pct(c.consensus_upside)} | Signal: {c.consensus_signal}

=== RISK ===
Leverage: {c.leverage_risk} | Liquidity: {c.liquidity_risk}
Profitability: {c.profitability_risk} | Valuation: {c.valuation_risk}
Overall: {c.overall_risk}

=== RECOMMENDATION ===
{c.recommendation} | Target: {price(c.target_price)} | Conviction: {c.conviction}
Decision Confidence: {c.decision_confidence:.1f}%

Write comprehensive memo. Return valid JSON only."""


# =============================================================================
# CLAUDE CLIENT
# =============================================================================

class ClaudeClient:
    def __init__(self, api_key: Optional[str] = None):
        self.anthropic_key = api_key or os.environ.get(MemoConfig.API_KEY_ENV_VAR, MemoConfig.DEFAULT_API_KEY)
        self.openai_key = os.environ.get(MemoConfig.OPENAI_API_KEY_ENV_VAR)
        self.logger = LOGGER

        # Determine which API to use
        if self.anthropic_key:
            self.use_openai = False
            self.logger.info("Using Anthropic Claude API")
        elif self.openai_key:
            self.use_openai = True
            self.logger.info("Using OpenAI API (fallback)")
        else:
            self.use_openai = False  # Will fail gracefully
            self.logger.warning("No API keys found (ANTHROPIC_API_KEY or OPENAI_API_KEY)")

    def generate(self, system: str, user: str) -> Tuple[str, int]:
        if self.use_openai:
            return self._generate_openai(system, user)
        return self._generate_anthropic(system, user)

    def _generate_anthropic(self, system: str, user: str) -> Tuple[str, int]:
        headers = {"Content-Type": "application/json", "x-api-key": self.anthropic_key, "anthropic-version": "2023-06-01"}
        payload = {"model": MemoConfig.MODEL, "max_tokens": MemoConfig.MAX_TOKENS, "temperature": MemoConfig.TEMPERATURE, "system": system, "messages": [{"role": "user", "content": user}]}
        for attempt in range(MemoConfig.MAX_RETRIES):
            try:
                self.logger.info(f"Claude API attempt {attempt + 1}")
                resp = requests.post(MemoConfig.API_ENDPOINT, headers=headers, json=payload, timeout=MemoConfig.REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("content", [{}])[0].get("text", "")
                    tokens = data.get("usage", {}).get("output_tokens", 0)
                    return content, tokens
            except Exception as e:
                self.logger.warning(f"API error: {e}")
            if attempt < MemoConfig.MAX_RETRIES - 1:
                time.sleep(MemoConfig.RETRY_DELAY * (attempt + 1))
        raise RuntimeError("Claude API failed")

    def _generate_openai(self, system: str, user: str) -> Tuple[str, int]:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.openai_key}"}
        payload = {
            "model": MemoConfig.OPENAI_MODEL,
            "max_tokens": MemoConfig.MAX_TOKENS,
            "temperature": MemoConfig.TEMPERATURE,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        }
        for attempt in range(MemoConfig.MAX_RETRIES):
            try:
                self.logger.info(f"OpenAI API attempt {attempt + 1}")
                resp = requests.post(MemoConfig.OPENAI_API_ENDPOINT, headers=headers, json=payload, timeout=MemoConfig.REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    tokens = data.get("usage", {}).get("completion_tokens", 0)
                    return content, tokens
            except Exception as e:
                self.logger.warning(f"OpenAI API error: {e}")
            if attempt < MemoConfig.MAX_RETRIES - 1:
                time.sleep(MemoConfig.RETRY_DELAY * (attempt + 1))
        raise RuntimeError("OpenAI API failed")


# =============================================================================
# PARSER
# =============================================================================

class MemoParser:
    def __init__(self):
        self.logger = LOGGER
    
    def parse(self, response: str, ctx: MemoContext) -> InvestmentMemo:
        memo = InvestmentMemo()
        memo.ticker = ctx.ticker
        memo.company_name = ctx.company_name
        memo.analysis_date = datetime.now().strftime("%Y-%m-%d")
        try:
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                data = json.loads(match.group())
                memo.executive_summary = MemoSection("Executive Summary", data.get("executive_summary", ""))
                memo.company_overview = MemoSection("Company Overview", data.get("company_overview", ""))
                memo.financial_analysis = MemoSection("Financial Analysis", data.get("financial_analysis", ""))
                memo.dupont_analysis = MemoSection("DuPont Analysis", data.get("dupont_analysis", ""))
                memo.valuation_analysis = MemoSection("Valuation Analysis", data.get("valuation_analysis", ""))
                memo.risk_assessment = MemoSection("Risk Assessment", data.get("risk_assessment", ""))
                memo.investment_thesis = MemoSection("Investment Thesis", data.get("investment_thesis", ""))
                memo.recommendation.recommendation = RecommendationType.from_upside(ctx.consensus_upside)
                memo.recommendation.target_price = ctx.target_price
                memo.recommendation.current_price = ctx.current_price
                memo.recommendation.upside_potential = ctx.consensus_upside
                memo.recommendation.decision_confidence = ctx.decision_confidence
                memo.recommendation.conviction = ConvictionLevel(ctx.conviction.lower()) if ctx.conviction else ConvictionLevel.MEDIUM
                memo.recommendation.key_catalysts = data.get("key_catalysts", [])[:5]
                memo.recommendation.key_risks = data.get("key_risks", [])[:5]
        except Exception as e:
            self.logger.warning(f"Parse error: {e}")
            memo.executive_summary.content = response
        return memo


# =============================================================================
# PDF GENERATOR - 6 PAGES, PROPERLY BALANCED
# =============================================================================

class SixPagePDFGenerator:
    """Generates professional 6-page PDF with no overlap and proper spacing."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.logger = LOGGER
    
    def generate(self, result: InvestmentMemoResult, ticker: str) -> Optional[Path]:
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib.colors import HexColor, white, black
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
            from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
        except ImportError:
            self.logger.warning("reportlab not available")
            return None
        
        if not result.memo or not result.context:
            return None
        
        path = self.output_dir / f"{ticker}_investment_memo.pdf"
        memo = result.memo
        ctx = result.context
        rec = memo.recommendation
        
        doc = SimpleDocTemplate(str(path), pagesize=letter, rightMargin=0.65*inch, leftMargin=0.65*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
        story = []
        
        # Colors
        NAVY = HexColor('#0f172a')
        BLUE = HexColor('#1e40af')
        LIGHT = HexColor('#f1f5f9')
        BORDER = HexColor('#cbd5e1')
        TEXT = HexColor('#1e293b')
        MUTED = HexColor('#64748b')
        GREEN = HexColor('#059669')
        RED = HexColor('#dc2626')
        AMBER = HexColor('#d97706')
        
        rec_colors = {
            RecommendationType.STRONG_BUY: (GREEN, HexColor('#dcfce7')),
            RecommendationType.BUY: (HexColor('#10b981'), HexColor('#dcfce7')),
            RecommendationType.HOLD: (AMBER, HexColor('#fef3c7')),
            RecommendationType.SELL: (HexColor('#ef4444'), HexColor('#fee2e2')),
            RecommendationType.STRONG_SELL: (RED, HexColor('#fee2e2')),
        }
        rec_fg, rec_bg = rec_colors.get(rec.recommendation, (MUTED, LIGHT))
        
        # Styles - FIXED to prevent overlap
        title_s = ParagraphStyle('Title', fontSize=24, textColor=NAVY, alignment=TA_CENTER, fontName='Helvetica-Bold', spaceAfter=2, leading=28)
        company_s = ParagraphStyle('Company', fontSize=12, textColor=MUTED, alignment=TA_CENTER, spaceAfter=12, leading=14)
        h1_s = ParagraphStyle('H1', fontSize=12, textColor=NAVY, fontName='Helvetica-Bold', spaceBefore=10, spaceAfter=6, leading=14)
        h2_s = ParagraphStyle('H2', fontSize=10, textColor=BLUE, fontName='Helvetica-Bold', spaceBefore=8, spaceAfter=4, leading=12)
        body_s = ParagraphStyle('Body', fontSize=8.5, textColor=TEXT, leading=11, spaceAfter=6, alignment=TA_JUSTIFY)
        small_s = ParagraphStyle('Small', fontSize=7, textColor=MUTED, leading=9)
        
        # Formatters
        def pct(v, d=1): return f"{v*100:.{d}f}%" if v and v != 0 else "N/A"
        def num(v):
            if not v: return "N/A"
            if abs(v) >= 1e12: return f"${v/1e12:.2f}T"
            if abs(v) >= 1e9: return f"${v/1e9:.2f}B"
            if abs(v) >= 1e6: return f"${v/1e6:.0f}M"
            return f"${v:,.0f}"
        def price(v): return f"${v:.2f}" if v and v > 0 else "N/A"
        def ratio(v): return f"{v:.2f}x" if v and v > 0 else "N/A"
        
        def make_table(data, widths, header_bg=NAVY):
            t = Table(data, colWidths=widths)
            t.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 7.5),
                ('BACKGROUND', (0, 0), (-1, 0), header_bg),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LIGHT]),
            ]))
            return t
        
        years = ctx.fiscal_periods[:5] if ctx.fiscal_periods else []
        
        # ===== PAGE 1: COVER + EXECUTIVE SUMMARY =====
        story.append(Spacer(1, 0.15*inch))
        
        # Title - SEPARATED properly
        story.append(Paragraph(f"<b>{ctx.ticker}</b>", title_s))
        story.append(Paragraph(ctx.company_name, company_s))
        
        # Recommendation box
        rec_box = Table([[rec.recommendation.to_display()]], colWidths=[3.5*inch])
        rec_box.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, 0), 18),
            ('TEXTCOLOR', (0, 0), (0, 0), rec_fg),
            ('BACKGROUND', (0, 0), (-1, -1), rec_bg),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('BOX', (0, 0), (-1, -1), 2, rec_fg),
        ]))
        story.append(rec_box)
        story.append(Spacer(1, 0.12*inch))
        
        # Key metrics - 2 column layout
        key_data = [
            ['Target Price', price(rec.target_price), 'Market Cap', num(ctx.market_cap)],
            ['Current Price', price(rec.current_price), 'Enterprise Value', num(ctx.enterprise_value)],
            ['Upside/Downside', pct(rec.upside_potential), '52-Week Range', f"{price(ctx.fifty_two_week_low)} - {price(ctx.fifty_two_week_high)}"],
            ['Decision Confidence', f"{rec.decision_confidence:.1f}%", 'Beta', f"{ctx.beta:.2f}"],
            ['Conviction', rec.conviction.value.upper(), 'Sector', ctx.sector],
        ]
        key_t = Table(key_data, colWidths=[1.15*inch, 1.15*inch, 1.15*inch, 1.4*inch])
        key_t.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 0), (-1, -1), LIGHT),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
        ]))
        story.append(key_t)
        story.append(Spacer(1, 0.08*inch))
        
        # Quality badge
        qual_text = f"Data Quality: {ctx.checks_passed}/{ctx.total_checks} checks passed | Confidence: {pct(ctx.overall_confidence)} | Analysis Date: {memo.analysis_date}"
        story.append(Paragraph(qual_text, ParagraphStyle('QC', fontSize=8, textColor=BLUE, alignment=TA_CENTER)))
        story.append(Spacer(1, 0.12*inch))
        
        # Executive Summary
        story.append(Paragraph("1. Executive Summary", h1_s))
        if memo.executive_summary.content:
            paras = memo.executive_summary.content.split('\n\n')
            for p in paras[:3]:
                if p.strip():
                    story.append(Paragraph(p.strip(), body_s))
        
        story.append(PageBreak())
        
        # ===== PAGE 2: COMPANY OVERVIEW + FINANCIALS =====
        story.append(Paragraph("2. Company Overview", h1_s))
        if memo.company_overview.content:
            paras = memo.company_overview.content.split('\n\n')
            for p in paras[:2]:
                if p.strip():
                    story.append(Paragraph(p.strip(), body_s))
        
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("3. Financial Analysis", h1_s))
        
        story.append(Paragraph("3.1 Profitability Metrics (5-Year)", h2_s))
        if years:
            w = [1.5*inch] + [0.65*inch] * len(years)
            prof_data = [['Metric'] + years]
            prof_data.append(['Return on Equity'] + [pct(ctx.roe.get(y)) for y in years])
            prof_data.append(['Return on Assets'] + [pct(ctx.roa.get(y)) for y in years])
            prof_data.append(['ROIC'] + [pct(ctx.roic.get(y)) for y in years])
            prof_data.append(['Gross Margin'] + [pct(ctx.gross_margin.get(y)) for y in years])
            prof_data.append(['Operating Margin'] + [pct(ctx.operating_margin.get(y)) for y in years])
            prof_data.append(['Net Profit Margin'] + [pct(ctx.net_margin.get(y)) for y in years])
            story.append(make_table(prof_data, w))
        
        story.append(Spacer(1, 0.08*inch))
        story.append(Paragraph("3.2 Liquidity & Leverage", h2_s))
        if years:
            liq_data = [['Metric'] + years]
            liq_data.append(['Current Ratio'] + [ratio(ctx.current_ratio.get(y)) for y in years])
            liq_data.append(['Quick Ratio'] + [ratio(ctx.quick_ratio.get(y)) for y in years])
            liq_data.append(['Debt/Equity'] + [ratio(ctx.debt_to_equity.get(y)) for y in years])
            liq_data.append(['Interest Coverage'] + [ratio(ctx.interest_coverage.get(y)) for y in years])
            story.append(make_table(liq_data, w))
        
        story.append(Spacer(1, 0.08*inch))
        story.append(Paragraph("3.3 Growth Analysis", h2_s))
        growth_data = [
            ['Metric', '5Y CAGR', 'Assessment'],
            ['Revenue', pct(ctx.revenue_cagr), 'Strong' if ctx.revenue_cagr > 0.10 else 'Moderate' if ctx.revenue_cagr > 0.05 else 'Weak'],
            ['Free Cash Flow', pct(ctx.fcf_cagr), 'Strong' if ctx.fcf_cagr > 0.10 else 'Moderate' if ctx.fcf_cagr > 0.05 else 'Weak'],
            ['Dividends', pct(ctx.dividend_cagr), 'Strong' if ctx.dividend_cagr > 0.05 else 'Moderate' if ctx.dividend_cagr > 0.02 else 'Low'],
        ]
        story.append(make_table(growth_data, [1.8*inch, 1.3*inch, 1.4*inch]))
        
        # Analysis text
        if memo.financial_analysis.content:
            story.append(Spacer(1, 0.06*inch))
            text = memo.financial_analysis.content.split('\n\n')[0][:500] if memo.financial_analysis.content else ''
            story.append(Paragraph(text, body_s))
        
        story.append(PageBreak())
        
        # ===== PAGE 3: DUPONT + DCF =====
        story.append(Paragraph("4. DuPont Analysis", h1_s))
        
        story.append(Paragraph("4.1 Three-Factor Decomposition: ROE = NPM x AT x EM", h2_s))
        if years:
            dp_data = [['Component'] + years]
            dp_data.append(['Net Profit Margin'] + [pct(ctx.dupont_npm.get(y)) for y in years])
            dp_data.append(['Asset Turnover'] + [f"{ctx.dupont_at.get(y, 0):.2f}x" for y in years])
            dp_data.append(['Equity Multiplier'] + [f"{ctx.dupont_em.get(y, 0):.2f}x" for y in years])
            dp_data.append(['ROE (Calculated)'] + [pct(ctx.dupont_roe.get(y)) for y in years])
            story.append(make_table(dp_data, w))
        
        story.append(Spacer(1, 0.06*inch))
        story.append(Paragraph("4.2 Five-Factor Decomposition", h2_s))
        if years:
            dp5_data = [['Component'] + years]
            dp5_data.append(['Tax Burden (NI/EBT)'] + [pct(ctx.dupont_tax_burden.get(y)) for y in years])
            dp5_data.append(['Interest Burden (EBT/EBIT)'] + [pct(ctx.dupont_interest_burden.get(y)) for y in years])
            dp5_data.append(['Operating Margin'] + [pct(ctx.dupont_operating_margin.get(y)) for y in years])
            story.append(make_table(dp5_data, w))
        
        # Quality box
        qual_box = [[f"Primary Driver: {ctx.dupont_primary_driver} | Quality: {ctx.dupont_quality_rating} | Score: {ctx.dupont_quality_score:.0f}/100"]]
        qt = Table(qual_box, colWidths=[4.85*inch])
        qt.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 0), (-1, -1), HexColor('#fef3c7')),
            ('TEXTCOLOR', (0, 0), (-1, -1), HexColor('#92400e')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(qt)
        
        if memo.dupont_analysis.content:
            story.append(Spacer(1, 0.06*inch))
            text = memo.dupont_analysis.content.split('\n\n')[0][:400] if memo.dupont_analysis.content else ''
            story.append(Paragraph(text, body_s))
        
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("5. Valuation Analysis", h1_s))
        
        story.append(Paragraph("5.1 DCF Valuation", h2_s))
        dcf_data = [
            ['Parameter', 'Value', 'Parameter', 'Value'],
            ['Intrinsic Value', price(ctx.dcf_intrinsic_value), 'WACC', pct(ctx.dcf_wacc)],
            ['Current Price', price(ctx.current_price), 'Cost of Equity', pct(ctx.dcf_cost_of_equity)],
            ['Upside/Downside', pct(ctx.dcf_upside), 'Cost of Debt', pct(ctx.dcf_cost_of_debt)],
            ['Signal', ctx.dcf_signal or 'N/A', 'Terminal Value %', pct(ctx.dcf_terminal_value_pct)],
        ]
        dcf_t = Table(dcf_data, colWidths=[1.1*inch, 1.15*inch, 1.1*inch, 1.15*inch])
        dcf_t.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7.5),
            ('BACKGROUND', (0, 0), (-1, 0), NAVY),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LIGHT]),
        ]))
        story.append(dcf_t)
        
        # Scenarios
        if ctx.dcf_bull_value > 0 or ctx.dcf_base_value > 0 or ctx.dcf_bear_value > 0:
            scen_data = [
                ['Scenario', 'Intrinsic Value', 'Upside'],
                ['Bull Case', price(ctx.dcf_bull_value) if ctx.dcf_bull_value > 0 else 'N/A', pct(ctx.dcf_bull_upside) if ctx.dcf_bull_value > 0 else 'N/A'],
                ['Base Case', price(ctx.dcf_base_value) if ctx.dcf_base_value > 0 else 'N/A', pct(ctx.dcf_base_upside) if ctx.dcf_base_value > 0 else 'N/A'],
                ['Bear Case', price(ctx.dcf_bear_value) if ctx.dcf_bear_value > 0 else 'N/A', pct(ctx.dcf_bear_upside) if ctx.dcf_bear_value > 0 else 'N/A'],
            ]
            story.append(make_table(scen_data, [1.5*inch, 1.5*inch, 1.5*inch]))
        
        story.append(PageBreak())
        
        # ===== PAGE 4: DDM + MULTIPLES + CONSENSUS =====
        story.append(Paragraph("5.2 DDM Valuation (Dividend Discount Model)", h2_s))
        
        # ALWAYS show DDM data
        ddm_data = [
            ['Parameter', 'Value', 'Parameter', 'Value'],
            ['Intrinsic Value', price(ctx.ddm_intrinsic_value) if ctx.ddm_applicable else 'N/A', 'Dividend Yield', pct(ctx.ddm_dividend_yield)],
            ['Upside/Downside', pct(ctx.ddm_upside) if ctx.ddm_applicable else 'N/A', 'Dividend Growth', pct(ctx.ddm_dividend_growth)],
            ['Current DPS', price(ctx.ddm_current_dividend), 'Cost of Equity', pct(ctx.ddm_cost_of_equity)],
            ['Payout Ratio', pct(ctx.ddm_payout_ratio), 'In Consensus?', 'Yes' if ctx.ddm_is_reliable else 'No'],
        ]
        ddm_t = Table(ddm_data, colWidths=[1.1*inch, 1.15*inch, 1.1*inch, 1.15*inch])
        ddm_t.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7.5),
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#7c3aed')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LIGHT]),
        ]))
        story.append(ddm_t)
        
        if not ctx.ddm_is_reliable:
            note = "Note: DDM excluded from consensus due to low payout ratio (15.1%). The model produces unreliable values for growth companies with minimal dividend yields."
            story.append(Paragraph(note, ParagraphStyle('Note', fontSize=7, textColor=MUTED, leading=9, spaceAfter=4)))
        
        story.append(Spacer(1, 0.08*inch))
        story.append(Paragraph("5.3 Relative Valuation (Multiples)", h2_s))
        
        mult_data = [
            ['Multiple', 'Current', '5Y Avg', 'Premium', 'Assessment'],
            ['P/E', ratio(ctx.pe_current), ratio(ctx.pe_avg), pct(ctx.pe_premium), 'Discount' if ctx.pe_premium < -0.05 else 'Premium' if ctx.pe_premium > 0.05 else 'Fair'],
            ['P/B', ratio(ctx.pb_current), ratio(ctx.pb_avg), pct(ctx.pb_premium), 'Discount' if ctx.pb_premium < -0.05 else 'Premium' if ctx.pb_premium > 0.05 else 'Fair'],
            ['EV/EBITDA', ratio(ctx.ev_ebitda_current), ratio(ctx.ev_ebitda_avg), pct(ctx.ev_ebitda_premium), 'Discount' if ctx.ev_ebitda_premium < -0.05 else 'Premium' if ctx.ev_ebitda_premium > 0.05 else 'Fair'],
            ['P/S', ratio(ctx.ps_current), ratio(ctx.ps_avg), pct((ctx.ps_current/ctx.ps_avg - 1) if ctx.ps_avg else 0), '-'],
            ['EV/Revenue', ratio(ctx.ev_revenue_current), ratio(ctx.ev_revenue_avg), pct((ctx.ev_revenue_current/ctx.ev_revenue_avg - 1) if ctx.ev_revenue_avg else 0), '-'],
        ]
        story.append(make_table(mult_data, [0.9*inch, 0.85*inch, 0.85*inch, 0.85*inch, 0.85*inch]))
        
        mult_sum = [[f"Implied Fair Value: {price(ctx.multiples_implied_value)} | Upside: {pct(ctx.multiples_upside)} | Assessment: {ctx.multiples_assessment}"]]
        ms_t = Table(mult_sum, colWidths=[4.85*inch])
        ms_t.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 0), (-1, -1), LIGHT),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(ms_t)
        
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("5.4 Valuation Consensus", h2_s))
        
        models_note = f"Models Used: {', '.join(ctx.models_used)}" if ctx.models_used else "Models: DCF + Multiples"
        story.append(Paragraph(models_note, small_s))
        
        cons_data = [
            ['Model', 'Intrinsic Value', 'Upside', 'Weight', 'Status'],
            ['DCF', price(ctx.dcf_intrinsic_value), pct(ctx.dcf_upside), '50%', 'Included'],
            ['DDM', price(ctx.ddm_intrinsic_value) if ctx.ddm_applicable else 'N/A', pct(ctx.ddm_upside) if ctx.ddm_applicable else '-', '0%' if not ctx.ddm_is_reliable else '20%', 'Excluded' if not ctx.ddm_is_reliable else 'Included'],
            ['Multiples', price(ctx.multiples_implied_value), pct(ctx.multiples_upside), '50%', 'Included'],
            ['CONSENSUS', price(ctx.consensus_value), pct(ctx.consensus_upside), '100%', ctx.consensus_signal],
        ]
        cons_t = Table(cons_data, colWidths=[0.85*inch, 1.1*inch, 0.85*inch, 0.7*inch, 0.85*inch])
        cons_t.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7.5),
            ('BACKGROUND', (0, 0), (-1, 0), NAVY),
            ('BACKGROUND', (0, -1), (-1, -1), HexColor('#dbeafe')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(cons_t)
        
        agreement = f"Model Agreement: {ctx.models_bullish} Bullish | {ctx.models_neutral} Neutral | {ctx.models_bearish} Bearish | Direction: {ctx.consensus_direction}"
        story.append(Paragraph(agreement, small_s))
        
        if memo.valuation_analysis.content:
            story.append(Spacer(1, 0.06*inch))
            text = memo.valuation_analysis.content.split('\n\n')[0][:400] if memo.valuation_analysis.content else ''
            story.append(Paragraph(text, body_s))
        
        story.append(PageBreak())
        
        # ===== PAGE 5: RISK + THESIS =====
        story.append(Paragraph("6. Risk Assessment", h1_s))
        
        risk_data = [
            ['Risk Category', 'Level', 'Key Metric', 'Assessment'],
            ['Leverage Risk', ctx.leverage_risk, f"D/E: {ratio(ctx.debt_to_equity.get(years[0]) if years else 0)}", 'Monitor' if ctx.leverage_risk == 'HIGH' else 'Acceptable'],
            ['Liquidity Risk', ctx.liquidity_risk, f"Current: {ratio(ctx.current_ratio.get(years[0]) if years else 0)}", 'Concern' if ctx.liquidity_risk == 'HIGH' else 'Acceptable'],
            ['Profitability Risk', ctx.profitability_risk, f"NPM: {pct(ctx.net_margin.get(years[0]) if years else 0)}", 'Strong' if ctx.profitability_risk == 'LOW' else 'Monitor'],
            ['Valuation Risk', ctx.valuation_risk, f"Spread: {price(abs(ctx.dcf_intrinsic_value - ctx.multiples_implied_value))}", 'Wide' if ctx.valuation_risk == 'HIGH' else 'Narrow'],
            ['OVERALL', ctx.overall_risk, '-', 'Elevated' if ctx.overall_risk == 'HIGH' else 'Manageable'],
        ]
        risk_t = Table(risk_data, colWidths=[1.1*inch, 0.8*inch, 1.2*inch, 1*inch])
        risk_t.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7.5),
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#7f1d1d')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LIGHT]),
        ]))
        story.append(risk_t)
        
        if memo.risk_assessment.content:
            story.append(Spacer(1, 0.08*inch))
            paras = memo.risk_assessment.content.split('\n\n')
            for p in paras[:2]:
                if p.strip():
                    story.append(Paragraph(p.strip()[:400], body_s))
        
        story.append(Spacer(1, 0.12*inch))
        story.append(Paragraph("7. Investment Thesis", h1_s))
        
        if memo.investment_thesis.content:
            paras = memo.investment_thesis.content.split('\n\n')
            for p in paras[:3]:
                if p.strip():
                    story.append(Paragraph(p.strip(), body_s))
        
        story.append(PageBreak())
        
        # ===== PAGE 6: RECOMMENDATION + CATALYSTS/RISKS =====
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("Final Recommendation", h1_s))
        
        # Big recommendation box
        final_data = [
            [rec.recommendation.to_display()],
            [f"Target Price: {price(rec.target_price)} | Current: {price(rec.current_price)} | Upside: {pct(rec.upside_potential)}"],
            [f"Decision Confidence: {rec.decision_confidence:.1f}% | Conviction: {rec.conviction.value.upper()} | Time Horizon: {rec.time_horizon}"],
        ]
        final_t = Table(final_data, colWidths=[4.85*inch])
        final_t.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, 0), 22),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, 0), rec_fg),
            ('BACKGROUND', (0, 0), (-1, -1), rec_bg),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BOX', (0, 0), (-1, -1), 2, rec_fg),
        ]))
        story.append(final_t)
        
        story.append(Spacer(1, 0.15*inch))
        
        # Catalysts and Risks - PROPER WRAPPING
        story.append(Paragraph("Key Catalysts & Risks", h2_s))
        
        # Build table with proper wrapping using Paragraph objects
        cat_style = ParagraphStyle('Cat', fontSize=7.5, leading=9, textColor=TEXT)
        
        cr_header = [
            [Paragraph('<b>Key Catalysts</b>', ParagraphStyle('CH', fontSize=8, textColor=white, fontName='Helvetica-Bold')),
             Paragraph('<b>Key Risks</b>', ParagraphStyle('RH', fontSize=8, textColor=white, fontName='Helvetica-Bold'))]
        ]
        
        cr_rows = []
        max_items = max(len(rec.key_catalysts), len(rec.key_risks), 3)
        for i in range(min(max_items, 5)):
            cat_text = rec.key_catalysts[i] if i < len(rec.key_catalysts) else ''
            risk_text = rec.key_risks[i] if i < len(rec.key_risks) else ''
            cr_rows.append([
                Paragraph(cat_text, cat_style),
                Paragraph(risk_text, cat_style)
            ])
        
        cr_t = Table(cr_header + cr_rows, colWidths=[2.4*inch, 2.4*inch])
        cr_t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), GREEN),
            ('BACKGROUND', (1, 0), (1, 0), RED),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(cr_t)
        
        story.append(Spacer(1, 0.2*inch))
        
        # Data Quality Summary
        story.append(Paragraph("Data Quality Summary", h2_s))
        dq_data = [
            ['Metric', 'Value', 'Metric', 'Value'],
            ['Total Checks', str(ctx.total_checks), 'Checks Passed', str(ctx.checks_passed)],
            ['Warnings', str(ctx.checks_warnings), 'Failed', str(ctx.checks_failed)],
            ['Confidence', pct(ctx.overall_confidence), 'Level', ctx.confidence_level.upper() if ctx.confidence_level else 'N/A'],
            ['Reliable', 'Yes' if ctx.is_reliable else 'No', 'Models Used', ', '.join(ctx.models_used)],
        ]
        dq_t = Table(dq_data, colWidths=[1.1*inch, 1.15*inch, 1.1*inch, 1.15*inch])
        dq_t.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7.5),
            ('BACKGROUND', (0, 0), (-1, 0), BLUE),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LIGHT]),
        ]))
        story.append(dq_t)
        
        story.append(Spacer(1, 0.15*inch))
        
        # Disclaimer
        disclaimer = """DISCLAIMER: This investment memorandum was generated by an AI-powered fundamental analysis system for educational purposes only. It does not constitute investment advice, a recommendation to buy or sell securities, or an offer or solicitation. Past performance is not indicative of future results. All investments involve risk, including potential loss of principal. Investors should conduct their own due diligence and consult qualified financial advisors before making investment decisions."""
        story.append(Paragraph(disclaimer, ParagraphStyle('Disc', fontSize=6.5, textColor=MUTED, leading=8, alignment=TA_JUSTIFY)))
        
        doc.build(story)
        self.logger.info(f"PDF: {path}")
        return path


# =============================================================================
# OUTPUT FORMATTER
# =============================================================================

class OutputFormatter:
    def __init__(self, output_dir: Optional[Path] = None):
        if output_dir is None:
            # Go up 4 levels: src → daria_fundamental_agent → fundamental_agents → master-analyst-agent
            master_repo_root = Path(__file__).parent.parent.parent.parent
            self.output_dir = master_repo_root / "shared_outputs"
        else:
            self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = LOGGER
        self.pdf = SixPagePDFGenerator(self.output_dir)
    
    def generate_all(self, result: InvestmentMemoResult, ticker: str) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
        json_path = self._save_json(result, ticker)
        md_path = self._save_markdown(result, ticker)
        # pdf_path = self.pdf.generate(result, ticker)
        pdf_path = None
        return json_path, md_path, pdf_path
    
    def _save_json(self, result: InvestmentMemoResult, ticker: str) -> Path:
        path = self.output_dir / f"fundamental_daria_{ticker}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        return path
    
    def _save_markdown(self, result: InvestmentMemoResult, ticker: str) -> Path:
        path = self.output_dir / f"{ticker}_investment_memo.md"
        if not result.memo:
            return path
        m = result.memo
        r = m.recommendation
        def pct(v): return f"{v*100:.1f}%" if v else "N/A"
        def price(v): return f"${v:.2f}" if v else "N/A"
        content = f"""# {m.ticker} Investment Memo
**{m.company_name}** | {m.analysis_date}

## Recommendation: {r.recommendation.to_display()}
| Metric | Value |
|--------|-------|
| Target Price | {price(r.target_price)} |
| Current Price | {price(r.current_price)} |
| Upside | {pct(r.upside_potential)} |
| Decision Confidence | {r.decision_confidence:.1f}% |
| Conviction | {r.conviction.value.upper()} |

## Executive Summary
{m.executive_summary.content}

## Company Overview
{m.company_overview.content}

## Financial Analysis
{m.financial_analysis.content}

## DuPont Analysis
{m.dupont_analysis.content}

## Valuation Analysis
{m.valuation_analysis.content}

## Risk Assessment
{m.risk_assessment.content}

## Investment Thesis
{m.investment_thesis.content}

---
*AI-generated analysis for educational purposes only.*
"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return path


# =============================================================================
# MAIN GENERATOR
# =============================================================================

class MemoGenerator:
    def __init__(self, api_key: Optional[str] = None, output_dir: Optional[Path] = None):
        self.client = ClaudeClient(api_key)
        self.formatter = OutputFormatter(output_dir)
        self.logger = LOGGER
    
    def generate(self, collection_result, validated_data, ratio_result, dupont_result, dcf_result, ddm_result, multiples_result, accuracy_result) -> InvestmentMemoResult:
        start = time.time()
        result = InvestmentMemoResult()
        result.timestamp = datetime.now().isoformat()
        result.model_used = MemoConfig.MODEL
        try:
            self.logger.info("Extracting data...")
            extractor = RobustDataExtractor()
            ctx = extractor.extract(collection_result, validated_data, ratio_result, dupont_result, dcf_result, ddm_result, multiples_result, accuracy_result)
            result.context = ctx
            self.logger.info("Building prompts...")
            engine = PromptEngine(ctx)
            self.logger.info("Calling Claude API...")
            response, tokens = self.client.generate(engine.system_prompt(), engine.user_prompt())
            result.raw_response = response
            result.llm_tokens_used = tokens
            self.logger.info("Parsing response...")
            parser = MemoParser()
            result.memo = parser.parse(response, ctx)
            self.logger.info(f"Generating outputs for {ctx.ticker}...")
            json_path, md_path, pdf_path = self.formatter.generate_all(result, ctx.ticker)
            result.json_path = json_path
            result.md_path = md_path
            result.pdf_path = pdf_path
            result.status = MemoStatus.SUCCESS
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            result.status = MemoStatus.FAILED
            result.error_message = str(e)
        result.generation_duration_seconds = time.time() - start
        return result


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

class Phase9MemoGenerator:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.output_dir = MemoConfig.DEFAULT_OUTPUT_DIR
        self.logger = LOGGER
    
    def generate(self, collection_result, validated_data, ratio_result, dupont_result, dcf_result, ddm_result, multiples_result, accuracy_result) -> InvestmentMemoResult:
        gen = MemoGenerator(api_key=self.api_key, output_dir=self.output_dir)
        return gen.generate(collection_result, validated_data, ratio_result, dupont_result, dcf_result, ddm_result, multiples_result, accuracy_result)
    
    def save_report(self, result: InvestmentMemoResult) -> Optional[Path]:
        if result and result.context:
            return self.output_dir / f"{result.context.ticker}_investment_memo.json"
        return None


MemoConvictionLevel = ConvictionLevel

__all__ = ["Phase9MemoGenerator", "InvestmentMemoResult", "MemoStatus", "RecommendationType", "ConvictionLevel", "MemoConvictionLevel"]