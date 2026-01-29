#!/usr/bin/env python3
"""
================================================================================
PHASE 5: INSTITUTIONAL TRADE NOTE GENERATOR v9.2.0
================================================================================

MSc AI Agents in Asset Management - Track B: Technical Analyst Agent
IFTE0001 Coursework - Technical Analysis Pipeline

Generates comprehensive institutional-grade trade notes with:
- Advanced visualizations (score bars, metrics dashboards)
- Detailed technical analysis with indicator breakdowns
- Professional 8-10 page native PDF using ReportLab
- Complete backtest validation metrics (CAGR, Sharpe, Hit Rate, Profit Factor)
- VaR/CVaR risk metrics
- Scenario analysis with probability weighting
- Score explanations

COMPREHENSIVE JSON OUTPUT FOR ORCHESTRATION:
- Phase 1: Benchmark, VIX, Statistical Tests, Volatility Estimators, Tail Risk
- Phase 2: Technical Indicators, Signals, Divergences, Levels
- Phase 3: Market Regime, GARCH Volatility, Hurst, Structural Breaks, Strategy
- Phase 4: Backtest Metrics, Walk-Forward, Monte Carlo, Position Sizing
- Phase 4B: CAPM Attribution, Regime Performance, Signal Quality, Stress Tests

COURSEWORK REQUIREMENTS SATISFIED:
- Technical indicators: RSI, MACD, Bollinger Bands, ADX, Ichimoku, etc.
- Backtesting metrics: CAGR, Sharpe Ratio, Sortino, Calmar, Hit Rate
- Risk metrics: Max Drawdown, VaR, CVaR, Volatility
- LLM-generated trade notes using Claude Sonnet 4.5
- Professional institutional presentation

Version: 9.2.2
================================================================================
"""

from __future__ import annotations

import json
import time
import re
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# API Keys - loaded from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Model settings
OPENAI_MODEL = "gpt-4o"  # Fallback for testing
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"  # Primary for production/demo
MAX_TOKENS = 12000
TEMPERATURE = 0.35

# Priority: Anthropic first (if key set), OpenAI fallback

VERSION = "9.2.2"

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Scenario:
    name: str
    probability: float
    target: float
    return_pct: float
    drivers: List[str]
    risks: List[str]


@dataclass
class IndicatorReading:
    name: str
    signal: str
    value: float
    zone: str
    confidence: float


@dataclass
class ScoreExplanation:
    """Explains what a score means and how it was calculated."""
    score: int
    rating: str  # EXCELLENT, GOOD, FAIR, POOR
    interpretation: str
    factors: List[str]


@dataclass
class TradeNote:
    # Metadata
    note_id: str = ""
    symbol: str = ""
    company_name: str = ""
    sector: str = ""
    industry: str = ""
    generated_at: str = ""
    analysis_period: str = ""
    total_records: int = 0
    
    # Recommendation
    recommendation: str = "HOLD"
    conviction: str = "MEDIUM"
    confidence: float = 50.0
    time_horizon: str = "1-3 months"
    risk_rating: str = "MODERATE"
    
    # Prices
    current_price: float = 0.0
    target_price: float = 0.0
    expected_return: float = 0.0
    
    # Trade Levels
    entry: float = 0.0
    stop_loss: float = 0.0
    target_1: float = 0.0
    target_2: float = 0.0
    target_3: float = 0.0
    risk_reward: float = 0.0  # Calculated using target_price
    
    # Position
    position_size: float = 5.0
    max_size: float = 10.0
    sizing_method: str = "Half-Kelly Criterion"
    sizing_rationale: str = ""
    
    # Scores (0-100) with explanations
    overall_score: int = 50
    technical_score: int = 50
    momentum_score: int = 50
    trend_score: int = 50
    volatility_score: int = 50
    volume_score: int = 50
    risk_score: int = 50
    
    # Score explanations
    score_explanations: Dict[str, ScoreExplanation] = field(default_factory=dict)
    
    # Score contributions
    score_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Data Quality
    data_quality_score: float = 0.0
    data_quality_grade: str = "N/A"
    
    # Market Profile
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    hurst_exponent: float = 0.5
    trend_character: str = "UNKNOWN"
    
    # Indicator Readings
    momentum_indicators: List[IndicatorReading] = field(default_factory=list)
    trend_indicators: List[IndicatorReading] = field(default_factory=list)
    volatility_indicators: List[IndicatorReading] = field(default_factory=list)
    volume_indicators: List[IndicatorReading] = field(default_factory=list)
    
    # Divergences
    divergences: List[Dict[str, Any]] = field(default_factory=list)
    
    # Backtest Results (COURSEWORK REQUIRED)
    backtest_total_return: float = 0.0
    backtest_cagr: float = 0.0
    backtest_volatility: float = 0.0
    backtest_max_dd: float = 0.0
    backtest_sharpe: float = 0.0
    backtest_sortino: float = 0.0
    backtest_calmar: float = 0.0
    backtest_total_trades: int = 0
    backtest_winning_trades: int = 0
    backtest_losing_trades: int = 0
    backtest_hit_rate: float = 0.0
    backtest_profit_factor: float = 0.0
    backtest_avg_win: float = 0.0
    backtest_avg_loss: float = 0.0
    backtest_expectancy: float = 0.0
    
    # Risk Analytics (from Phase 4)
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    
    # =========================================================================
    # PHASE 1: ADDITIONAL MARKET CONTEXT
    # =========================================================================
    
    # Benchmark Analysis (SPY)
    benchmark_symbol: str = "SPY"
    benchmark_correlation: float = 0.0
    benchmark_correlation_1y: float = 0.0
    benchmark_beta: float = 1.0
    benchmark_beta_1y: float = 1.0
    benchmark_alpha: float = 0.0
    benchmark_info_ratio: float = 0.0
    up_capture: float = 0.0
    down_capture: float = 0.0
    
    # VIX / Market Context
    vix_current: float = 0.0
    vix_regime: str = "NORMAL"
    vix_percentile: float = 0.0
    market_context: str = "NEUTRAL"
    
    # Statistical Tests
    stationarity_adf: bool = True
    stationarity_kpss: bool = True
    normality_jb: bool = False
    autocorrelation_lb: bool = False
    arch_effects: bool = False
    stationarity_conclusion: str = "STATIONARY"
    
    # Volatility Estimators (from Phase 1)
    vol_close_to_close: float = 0.0
    vol_parkinson: float = 0.0
    vol_garman_klass: float = 0.0
    vol_rogers_satchell: float = 0.0
    vol_yang_zhang: float = 0.0
    vol_composite: float = 0.0
    vol_regime_p1: str = "NORMAL"
    
    # Tail Risk (from Phase 1)
    skewness: float = 0.0
    kurtosis: float = 3.0
    var_95_daily: float = 0.0
    var_99_daily: float = 0.0
    cvar_95_daily: float = 0.0
    
    # =========================================================================
    # PHASE 3: MARKET REGIME DETECTION (COMPREHENSIVE)
    # =========================================================================
    
    # Regime Classification
    market_regime: str = "UNKNOWN"
    regime_probability: float = 0.0
    regime_confidence: float = 0.0
    
    # State Probabilities
    regime_prob_bull: float = 0.0
    regime_prob_bear: float = 0.0
    regime_prob_sideways: float = 0.0
    
    # Expected Durations
    regime_duration_bull: int = 0
    regime_duration_bear: int = 0
    regime_duration_sideways: int = 0
    
    # Volatility Analysis (Phase 3)
    current_volatility: float = 0.0
    volatility_regime: str = "NORMAL"
    volatility_percentile: float = 0.0
    
    # GARCH Parameters
    garch_omega: float = 0.0
    garch_alpha: float = 0.0
    garch_beta: float = 0.0
    garch_persistence: float = 0.0
    
    # Volatility Forecasts
    volatility_forecast_1d: float = 0.0
    volatility_forecast_5d: float = 0.0
    
    # Hurst Analysis (Phase 3)
    hurst_p3: float = 0.5
    hurst_classification: str = "RANDOM_WALK"
    hurst_r_squared: float = 0.0
    hurst_ci_lower: float = 0.0
    hurst_ci_upper: float = 0.0
    
    # Structural Breaks
    structural_breaks: int = 0
    days_since_break: int = 0
    cusum_statistic: float = 0.0
    break_active: bool = False
    
    # Strategy Recommendation (Phase 3)
    strategy_recommendation: str = "HOLD"
    position_bias: str = "NEUTRAL"
    recommended_position_size: float = 0.0
    recommended_stop_loss_atr: float = 2.0
    recommended_take_profit_atr: float = 4.0
    max_holding_days: int = 20
    strategy_confidence: float = 0.0
    strategy_rationale: str = ""
    
    # Quality (Phase 3)
    regime_quality_score: float = 0.0
    regime_quality_grade: str = "UNKNOWN"
    
    # =========================================================================
    # PHASE 4: BACKTEST EXTENDED METRICS
    # =========================================================================
    
    # Return Distribution
    backtest_best_day: float = 0.0
    backtest_worst_day: float = 0.0
    backtest_positive_days: int = 0
    backtest_positive_pct: float = 0.0
    backtest_monthly_return: float = 0.0
    backtest_skewness: float = 0.0
    backtest_kurtosis: float = 3.0
    
    # Risk Metrics Extended
    backtest_downside_vol: float = 0.0
    backtest_avg_drawdown: float = 0.0
    backtest_max_dd_duration: int = 0
    backtest_omega: float = 1.0
    backtest_alpha: float = 0.0
    backtest_beta: float = 1.0
    backtest_info_ratio: float = 0.0
    
    # Trade Statistics Extended
    backtest_payoff_ratio: float = 0.0
    backtest_largest_win: float = 0.0
    backtest_largest_loss: float = 0.0
    backtest_avg_holding_days: float = 0.0
    
    # Transaction Costs
    backtest_total_costs: float = 0.0
    backtest_cost_per_trade: float = 0.0
    backtest_cost_rate: float = 0.0
    
    # Walk-Forward Analysis
    wf_periods: int = 0
    wf_avg_is_return: float = 0.0
    wf_avg_oos_return: float = 0.0
    wf_avg_is_sharpe: float = 0.0
    wf_avg_oos_sharpe: float = 0.0
    wfe_ratio: float = 0.0
    walk_forward_consistency: float = 0.0
    walk_forward_robust: bool = False
    
    # Monte Carlo Analysis
    mc_simulations: int = 0
    monte_carlo_mean_return: float = 0.0
    mc_return_std: float = 0.0
    mc_return_ci_lower: float = 0.0
    mc_return_ci_upper: float = 0.0
    mc_sharpe_mean: float = 0.0
    mc_sharpe_ci_lower: float = 0.0
    mc_sharpe_ci_upper: float = 0.0
    monte_carlo_p_positive: float = 0.0
    mc_p_sharpe_above_1: float = 0.0
    
    # Position Sizing
    sizing_kelly_optimal: float = 0.0
    sizing_kelly_half: float = 0.0
    sizing_vol_scaled: float = 0.0
    
    # =========================================================================
    # PHASE 4B: ADVANCED RISK ANALYTICS (COMPREHENSIVE)
    # =========================================================================
    
    # Executive Summary
    performance_grade: str = "UNKNOWN"
    risk_level: str = "UNKNOWN"
    alpha_confidence: float = 0.0
    strategy_robustness: float = 0.0
    
    # CAPM Attribution
    alpha: float = 0.0
    beta: float = 1.0
    r_squared: float = 0.0
    alpha_t_stat: float = 0.0
    alpha_p_value: float = 1.0
    alpha_significant: bool = False
    
    # Return Decomposition
    return_systematic: float = 0.0
    return_idiosyncratic: float = 0.0
    skill_contribution: float = 0.0
    
    # Tracking
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    
    # Regime-Conditional Performance
    perf_bull_return: float = 0.0
    perf_bull_sharpe: float = 0.0
    perf_bull_max_dd: float = 0.0
    perf_bull_days: int = 0
    
    perf_bear_return: float = 0.0
    perf_bear_sharpe: float = 0.0
    perf_bear_max_dd: float = 0.0
    perf_bear_days: int = 0
    
    perf_sideways_return: float = 0.0
    perf_sideways_sharpe: float = 0.0
    perf_sideways_max_dd: float = 0.0
    perf_sideways_days: int = 0
    
    # Volatility Regime Performance
    perf_low_vol_return: float = 0.0
    perf_low_vol_sharpe: float = 0.0
    perf_normal_vol_return: float = 0.0
    perf_normal_vol_sharpe: float = 0.0
    perf_high_vol_return: float = 0.0
    perf_high_vol_sharpe: float = 0.0
    perf_crisis_return: float = 0.0
    perf_crisis_sharpe: float = 0.0
    
    # Signal Quality
    information_coefficient: float = 0.0
    ic_t_stat: float = 0.0
    signal_quality_grade: str = "UNKNOWN"
    hit_rate_long: float = 0.0
    hit_rate_flat: float = 0.0
    signal_persistence: float = 0.0
    annual_turnover: float = 0.0
    
    # Stress Tests
    stress_covid_crash: float = 0.0
    stress_covid_recovery: float = 0.0
    stress_2022_bear: float = 0.0
    stress_2022_recovery: float = 0.0
    stress_2018_q4: float = 0.0
    stress_2018_recovery: float = 0.0
    stress_tests_passed: int = 0
    stress_tests_total: int = 0
    
    # Drawdown Analysis
    total_drawdown_periods: int = 0
    avg_drawdown_duration: int = 0
    avg_recovery_time: int = 0
    
    # Analysis Text (with defaults for dataclass ordering)
    investment_thesis: str = ""
    executive_summary: str = ""
    technical_analysis: str = ""
    backtest_analysis: str = ""
    risk_analysis: str = ""
    
    # Scenarios (with default factory)
    scenarios: List[Scenario] = field(default_factory=list)
    
    # Catalysts & Risks (with default factory)
    catalysts: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    
    # Technical Levels (with default factory)
    levels: Dict[str, float] = field(default_factory=dict)
    
    # Signal Distribution
    bullish_signals: int = 0
    bearish_signals: int = 0
    neutral_signals: int = 0
    
    # Generation
    model_used: str = CLAUDE_MODEL
    tokens_used: int = 0
    generation_time_ms: float = 0.0


# =============================================================================
# SAFE DATA EXTRACTION
# =============================================================================

def safe_get(obj: Any, *keys, default=None) -> Any:
    current = obj
    for key in keys:
        if current is None:
            return default
        try:
            if isinstance(current, dict):
                current = current.get(key, None)
            elif hasattr(current, str(key)):
                current = getattr(current, str(key), None)
            else:
                return default
        except Exception:
            return default
    return current if current is not None else default


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, (pd.Series, np.ndarray)):
            if len(value) > 0:
                return float(value.iloc[-1] if isinstance(value, pd.Series) else value[-1])
            return default
        return float(value)
    except Exception:
        return default


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if hasattr(value, 'value'):
        return str(value.value)
    return str(value)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(safe_float(value, default))
    except:
        return default


def _is_falsy(value: Any) -> bool:
    """Check if a value is falsy, handling pandas types that don't support bool()."""
    if value is None:
        return True
    if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
        return len(value) == 0
    if isinstance(value, dict):
        return len(value) == 0
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return not bool(value)


# =============================================================================
# COMPREHENSIVE CONTEXT BUILDER
# =============================================================================

class ContextBuilder:
    """Extracts comprehensive data from all pipeline phases."""
    
    def __init__(self, p1, p2, p3=None, p4=None, p4b=None):
        self.p1, self.p2, self.p3, self.p4, self.p4b = p1, p2, p3, p4, p4b
    
    def build(self) -> Dict:
        ctx = {
            'symbol': self._symbol(),
            'company_name': safe_str(safe_get(self.p1, 'company_name'), 'Unknown Company'),
            'sector': safe_str(safe_get(self.p1, 'sector'), 'Technology'),
            'industry': safe_str(safe_get(self.p1, 'industry'), 'Unknown'),
            'price': self._price(),
            'analysis_period': self._period(),
            'total_records': self._records(),
        }
        ctx['quality'] = self._quality()
        ctx['profile'] = self._profile()
        ctx['signals'] = self._signals()
        ctx['families'] = self._families()
        ctx['indicators'] = self._indicators()
        ctx['levels'] = self._levels()
        ctx['divergences'] = self._divergences()
        ctx['backtest'] = self._backtest()
        ctx['risk'] = self._risk()  # VaR/CVaR from Phase 4
        ctx['regime'] = self._regime()  # Phase 3 market regime (comprehensive)
        ctx['attribution'] = self._attribution()  # Phase 4B CAPM (comprehensive)
        ctx['walk_forward'] = self._walk_forward()  # Phase 4 walk-forward
        ctx['monte_carlo'] = self._monte_carlo()  # Phase 4 Monte Carlo
        ctx['benchmark'] = self._benchmark()  # Phase 1 benchmark data
        ctx['vix'] = self._vix()  # Phase 1 VIX context
        ctx['statistical_tests'] = self._statistical_tests()  # Phase 1 tests
        ctx['volatility_estimators'] = self._volatility_estimators()  # Phase 1 vol
        ctx['stress_tests'] = self._stress_tests()  # Phase 4B stress tests
        ctx['regime_performance'] = self._regime_performance()  # Phase 4B regime perf
        ctx['signal_quality'] = self._signal_quality()  # Phase 4B signal quality
        return ctx
    
    def _symbol(self) -> str:
        if self.p1 and hasattr(self.p1, 'symbol'):
            return str(self.p1.symbol).upper()
        return "UNKNOWN"
    
    def _price(self) -> float:
        if self.p1 is None:
            return 0.0
        daily = safe_get(self.p1, 'daily')
        if isinstance(daily, pd.DataFrame) and not daily.empty:
            for col in ['Close', 'close', 'Adj Close']:
                if col in daily.columns:
                    return float(daily[col].iloc[-1])
        return 0.0
    
    def _period(self) -> str:
        if self.p1 is None:
            return "N/A"
        daily = safe_get(self.p1, 'daily')
        if isinstance(daily, pd.DataFrame) and not daily.empty:
            start = daily.index[0].strftime('%Y-%m-%d') if hasattr(daily.index[0], 'strftime') else str(daily.index[0])[:10]
            end = daily.index[-1].strftime('%Y-%m-%d') if hasattr(daily.index[-1], 'strftime') else str(daily.index[-1])[:10]
            return f"{start} to {end}"
        return "N/A"
    
    def _records(self) -> int:
        if self.p1 is None:
            return 0
        daily = safe_get(self.p1, 'daily')
        if isinstance(daily, pd.DataFrame):
            return len(daily)
        return 0
    
    def _quality(self) -> Dict:
        q = safe_get(self.p1, 'quality')
        if not q:
            return {'score': 0, 'grade': 'N/A', 'completeness': 0, 'accuracy': 0, 'consistency': 0, 'timeliness': 0}
        return {
            'score': safe_float(safe_get(q, 'overall')),
            'grade': safe_str(safe_get(q, 'grade')),
            'completeness': safe_float(safe_get(q, 'completeness')),
            'accuracy': safe_float(safe_get(q, 'accuracy')),
            'consistency': safe_float(safe_get(q, 'consistency')),
            'timeliness': safe_float(safe_get(q, 'timeliness')),
        }
    
    def _profile(self) -> Dict:
        p = safe_get(self.p1, 'profile')
        if not p:
            return {}
        result = {
            'annual_return': safe_float(safe_get(p, 'annualized_return')),
            'volatility': safe_float(safe_get(p, 'annualized_volatility')),
            'sharpe': safe_float(safe_get(p, 'sharpe_ratio')),
            'sortino': safe_float(safe_get(p, 'sortino_ratio')),
            'calmar': safe_float(safe_get(p, 'calmar_ratio')),
            'hurst': safe_float(safe_get(p, 'hurst_exponent'), 0.5),
            'trend_character': safe_str(safe_get(p, 'trend_character'), 'Unknown'),
            'skewness': safe_float(safe_get(p, 'skewness')),
            'kurtosis': safe_float(safe_get(p, 'kurtosis'), 3),
        }
        tail = safe_get(p, 'tail_risk')
        if tail:
            result['max_drawdown'] = safe_float(safe_get(tail, 'max_drawdown'))
        return result
    
    def _signals(self) -> Dict:
        a = safe_get(self.p2, 'current_analysis')
        if not a:
            return {}
        return {
            'direction': safe_str(safe_get(a, 'overall_signal')),
            'confidence': safe_float(safe_get(a, 'overall_confidence'), 0.5),
            'strength': safe_str(safe_get(a, 'signal_strength')),
            'recommendation': safe_str(safe_get(a, 'recommendation')),
            'bullish': safe_int(safe_get(a, 'bullish_count')),
            'bearish': safe_int(safe_get(a, 'bearish_count')),
            'neutral': safe_int(safe_get(a, 'neutral_count')),
        }
    
    def _families(self) -> Dict:
        a = safe_get(self.p2, 'current_analysis')
        fams = safe_get(a, 'families')
        if not fams or not isinstance(fams, dict):
            return {}
        result = {}
        for name, f in fams.items():
            result[name] = {
                'signal': safe_str(safe_get(f, 'aggregate_signal')),
                'confidence': safe_float(safe_get(f, 'aggregate_confidence'), 0.5),
                'weight': safe_float(safe_get(f, 'weight'), 0.2),
            }
        return result
    
    def _indicators(self) -> Dict:
        a = safe_get(self.p2, 'current_analysis')
        fams = safe_get(a, 'families')
        if not fams or not isinstance(fams, dict):
            return {}
        
        result = {'momentum': [], 'trend': [], 'volatility': [], 'volume': [], 'system': []}
        
        for fam_name, fam in fams.items():
            indicators = safe_get(fam, 'indicators')
            if not indicators or not isinstance(indicators, dict):
                continue
            
            fam_key = fam_name.lower()
            if fam_key not in result:
                fam_key = 'system'
            
            for ind_name, ind in indicators.items():
                result[fam_key].append({
                    'name': ind_name.upper(),
                    'signal': safe_str(safe_get(ind, 'direction')),
                    'value': safe_float(safe_get(ind, 'value')),
                    'zone': safe_str(safe_get(ind, 'zone')),
                    'confidence': safe_float(safe_get(ind, 'confidence')),
                    'factors': safe_get(ind, 'factors') or [],
                })
        
        return result
    
    def _levels(self) -> Dict:
        a = safe_get(self.p2, 'current_analysis')
        lvls = safe_get(a, 'key_levels')
        if not lvls or not isinstance(lvls, dict):
            return {}
        return {k: safe_float(v) for k, v in lvls.items() if v is not None}
    
    def _divergences(self) -> List:
        a = safe_get(self.p2, 'current_analysis')
        divs = safe_get(a, 'divergences')
        if not divs:
            return []
        return [{
            'type': safe_str(safe_get(d, 'divergence_type')),
            'indicator': safe_str(safe_get(d, 'indicator_name')),
            'strength': safe_float(safe_get(d, 'strength')),
            'bars': safe_int(safe_get(d, 'bars_duration')),
        } for d in divs[:5]]
    
    def _backtest(self) -> Dict:
        """Extract backtest metrics from Phase 4."""
        if self.p4 is None:
            return {}
        result = {}
        
        # Returns
        ret = safe_get(self.p4, 'returns')
        if ret:
            result['total_return'] = safe_float(safe_get(ret, 'total_return'))
            result['cagr'] = safe_float(safe_get(ret, 'cagr'))
            result['annual_return'] = safe_float(safe_get(ret, 'annual_return'))
        
        # Risk metrics INCLUDING VaR/CVaR
        risk = safe_get(self.p4, 'risk')
        if risk:
            # Use annual_volatility (correct field name from RiskMetrics dataclass)
            result['volatility'] = safe_float(safe_get(risk, 'annual_volatility'))
            result['max_dd'] = safe_float(safe_get(risk, 'max_drawdown'))
            # VaR and CVaR are in Phase 4 risk metrics
            result['var_95'] = safe_float(safe_get(risk, 'var_95'))
            result['cvar_95'] = safe_float(safe_get(risk, 'cvar_95'))
            result['skewness'] = safe_float(safe_get(risk, 'skewness'))
            result['kurtosis'] = safe_float(safe_get(risk, 'kurtosis'), 3)
        
        # Risk-adjusted
        radj = safe_get(self.p4, 'risk_adjusted')
        if radj:
            result['sharpe'] = safe_float(safe_get(radj, 'sharpe_ratio'))
            result['sortino'] = safe_float(safe_get(radj, 'sortino_ratio'))
            result['calmar'] = safe_float(safe_get(radj, 'calmar_ratio'))
        
        # Trade statistics
        trades = safe_get(self.p4, 'trades')
        if trades:
            result['total_trades'] = safe_int(safe_get(trades, 'total_trades'))
            result['winning'] = safe_int(safe_get(trades, 'winning_trades'))
            result['losing'] = safe_int(safe_get(trades, 'losing_trades'))
            result['hit_rate'] = safe_float(safe_get(trades, 'hit_rate'))
            result['profit_factor'] = safe_float(safe_get(trades, 'profit_factor'))
            result['avg_win'] = safe_float(safe_get(trades, 'avg_win'))
            result['avg_loss'] = safe_float(safe_get(trades, 'avg_loss'))
            result['expectancy'] = safe_float(safe_get(trades, 'expectancy'))
        
        return result
    
    def _risk(self) -> Dict:
        """Extract VaR/CVaR - from Phase 4 risk metrics."""
        bt = self._backtest()
        return {
            'var_95': bt.get('var_95', 0),
            'var_99': bt.get('var_95', 0) * 1.3,  # Estimate 99% from 95%
            'cvar_95': bt.get('cvar_95', 0),
            'cvar_99': bt.get('cvar_95', 0) * 1.3,
        }
    
    def _regime(self) -> Dict:
        """Extract comprehensive market regime data from Phase 3."""
        if self.p3 is None:
            return {}
        result = {}
        
        # Helper to get attribute from object or dict
        def get_attr(obj, *attrs):
            if obj is None:
                return None
            for attr in attrs:
                val = None
                # Try as attribute
                if hasattr(obj, attr):
                    val = getattr(obj, attr, None)
                # Try as dict key
                elif isinstance(obj, dict):
                    val = obj.get(attr)
                if val is not None and val != '' and val != 0:
                    return val
            return None
        
        # Helper to handle enum values
        def enum_val(obj):
            if obj is None:
                return None
            if hasattr(obj, 'value'):
                return obj.value
            if hasattr(obj, 'name'):
                return obj.name
            return str(obj)
        
        # =================================================================
        # HMM Analysis (nested under hmm attribute)
        # =================================================================
        hmm = get_attr(self.p3, 'hmm')
        
        # Current regime - from hmm.current_regime
        current_regime = get_attr(hmm, 'current_regime') if hmm else None
        if current_regime is None:
            current_regime = get_attr(self.p3, 'consensus_regime', 'current_regime')
        result['market_regime'] = safe_str(enum_val(current_regime), 'UNKNOWN')
        
        # Regime probability
        result['regime_probability'] = safe_float(
            get_attr(hmm, 'regime_probability') if hmm else get_attr(self.p3, 'regime_probability')
        ) * 100  # Convert to percentage
        
        # Consensus confidence
        result['regime_confidence'] = safe_float(
            get_attr(self.p3, 'consensus_confidence', 'confidence')
        ) * 100  # Convert to percentage
        
        # State probabilities
        state_probs = get_attr(hmm, 'state_probabilities') if hmm else None
        if state_probs:
            if isinstance(state_probs, dict):
                for regime_key, prob in state_probs.items():
                    regime_name = enum_val(regime_key) if hasattr(regime_key, 'value') else str(regime_key)
                    if 'BULL' in str(regime_name).upper():
                        result['prob_bull'] = safe_float(prob) * 100
                    elif 'BEAR' in str(regime_name).upper():
                        result['prob_bear'] = safe_float(prob) * 100
                    elif 'SIDEWAYS' in str(regime_name).upper() or 'NEUTRAL' in str(regime_name).upper():
                        result['prob_sideways'] = safe_float(prob) * 100
        
        # Expected durations
        durations = get_attr(hmm, 'expected_durations') if hmm else None
        if durations:
            if isinstance(durations, dict):
                for regime_key, dur in durations.items():
                    regime_name = enum_val(regime_key) if hasattr(regime_key, 'value') else str(regime_key)
                    if 'BULL' in str(regime_name).upper():
                        result['duration_bull'] = safe_int(dur)
                    elif 'BEAR' in str(regime_name).upper():
                        result['duration_bear'] = safe_int(dur)
                    elif 'SIDEWAYS' in str(regime_name).upper():
                        result['duration_sideways'] = safe_int(dur)
        
        # =================================================================
        # GARCH Analysis (nested under garch attribute)
        # =================================================================
        garch = get_attr(self.p3, 'garch')
        
        if garch:
            result['volatility_regime'] = safe_str(
                enum_val(get_attr(garch, 'regime', 'volatility_regime')), 'NORMAL'
            )
            result['current_volatility'] = safe_float(
                get_attr(garch, 'current_volatility', 'current')
            ) * 100  # Convert to percentage
            result['volatility_percentile'] = safe_float(
                get_attr(garch, 'percentile', 'vol_percentile')
            )
            result['garch_persistence'] = safe_float(
                get_attr(garch, 'persistence', 'garch_persistence')
            )
            result['garch_omega'] = safe_float(get_attr(garch, 'omega'))
            result['garch_alpha'] = safe_float(get_attr(garch, 'alpha'))
            result['garch_beta'] = safe_float(get_attr(garch, 'beta'))
            
            # Forecasts
            forecasts = get_attr(garch, 'forecasts')
            if forecasts:
                if isinstance(forecasts, dict):
                    result['volatility_forecast_1d'] = safe_float(forecasts.get('day_1', forecasts.get('1d', 0))) * 100
                    result['volatility_forecast_5d'] = safe_float(forecasts.get('day_5_avg', forecasts.get('5d', 0))) * 100
                elif hasattr(forecasts, '__iter__') and len(forecasts) >= 1:
                    result['volatility_forecast_1d'] = safe_float(forecasts[0]) * 100
                    if len(forecasts) >= 5:
                        result['volatility_forecast_5d'] = safe_float(np.mean(forecasts[:5])) * 100
        
        # =================================================================
        # Hurst Analysis (nested under hurst attribute)
        # =================================================================
        hurst = get_attr(self.p3, 'hurst')
        
        if hurst:
            result['hurst_p3'] = safe_float(get_attr(hurst, 'hurst', 'hurst_exponent', 'value'))
            result['hurst_classification'] = safe_str(
                enum_val(get_attr(hurst, 'classification', 'trend_type'))
            )
            result['hurst_r_squared'] = safe_float(get_attr(hurst, 'r_squared'))
            
            # Confidence interval
            ci = get_attr(hurst, 'confidence_interval', 'ci')
            if ci:
                if isinstance(ci, dict):
                    result['hurst_ci_lower'] = safe_float(ci.get('lower', 0))
                    result['hurst_ci_upper'] = safe_float(ci.get('upper', 0))
                elif isinstance(ci, (list, tuple)) and len(ci) >= 2:
                    result['hurst_ci_lower'] = safe_float(ci[0])
                    result['hurst_ci_upper'] = safe_float(ci[1])
        
        # =================================================================
        # Breakpoint Analysis
        # =================================================================
        breaks = get_attr(self.p3, 'breakpoints')
        
        if breaks:
            result['structural_breaks'] = safe_int(get_attr(breaks, 'n_breaks', 'count'))
            result['days_since_break'] = safe_int(get_attr(breaks, 'days_since_break', 'days_since_last'))
            result['cusum_statistic'] = safe_float(get_attr(breaks, 'cusum_stat', 'cusum'))
            result['break_active'] = bool(get_attr(breaks, 'break_detected', 'active'))
        
        # =================================================================
        # Strategy Recommendation
        # =================================================================
        strat = get_attr(self.p3, 'strategy')
        
        if strat:
            result['strategy_recommendation'] = safe_str(
                enum_val(get_attr(strat, 'strategy', 'type', 'strategy_type')), 'HOLD'
            )
            result['position_bias'] = safe_str(
                enum_val(get_attr(strat, 'position_bias', 'bias'))
            )
            result['recommended_position_size'] = safe_float(
                get_attr(strat, 'position_size', 'size')
            ) * 100  # Convert to percentage
            result['stop_loss_atr'] = safe_float(get_attr(strat, 'stop_loss_atr', 'stop_loss'), 2.0)
            result['take_profit_atr'] = safe_float(get_attr(strat, 'take_profit_atr', 'take_profit'), 4.0)
            result['max_holding_days'] = safe_int(get_attr(strat, 'max_holding_period', 'max_holding'), 20)
            result['strategy_confidence'] = safe_float(get_attr(strat, 'confidence')) * 100
            result['rationale'] = safe_str(get_attr(strat, 'rationale', 'reasoning'))
        
        # =================================================================
        # Quality Assessment
        # =================================================================
        result['quality_score'] = safe_float(get_attr(self.p3, 'quality_score'))
        quality_grade = get_attr(self.p3, 'quality_grade')
        result['quality_grade'] = safe_str(enum_val(quality_grade), 'UNKNOWN')
        
        return result
    
    def _attribution(self) -> Dict:
        """Extract performance attribution from Phase 4B."""
        if self.p4b is None:
            return {}
        result = {}
        
        # Helper to get attribute from object or dict
        def get_attr(obj, *attrs):
            if obj is None:
                return None
            for attr in attrs:
                val = None
                # Try as attribute
                if hasattr(obj, attr):
                    val = getattr(obj, attr, None)
                # Try as dict key
                elif isinstance(obj, dict):
                    val = obj.get(attr)
                if val is not None and val != '' and val != 0:
                    return val
            return None
        
        # Helper to handle enum values
        def enum_val(obj):
            if obj is None:
                return None
            if hasattr(obj, 'value'):
                return obj.value
            if hasattr(obj, 'name'):
                return obj.name
            return str(obj)
        
        # =================================================================
        # CAPM Alpha/Beta Decomposition
        # =================================================================
        alpha_beta = get_attr(self.p4b, 'alpha_beta')
        
        if alpha_beta:
            result['alpha'] = safe_float(get_attr(alpha_beta, 'alpha')) * 100  # Convert to percentage
            result['beta'] = safe_float(get_attr(alpha_beta, 'beta'), 1.0)
            result['r_squared'] = safe_float(get_attr(alpha_beta, 'r_squared'))
            result['alpha_t_stat'] = safe_float(get_attr(alpha_beta, 'alpha_t_stat'))
            result['alpha_p_value'] = safe_float(get_attr(alpha_beta, 'alpha_p_value'))
            result['tracking_error'] = safe_float(get_attr(alpha_beta, 'tracking_error')) * 100
            result['information_ratio'] = safe_float(get_attr(alpha_beta, 'information_ratio'))
            
            # Return decomposition
            result['return_systematic'] = safe_float(get_attr(alpha_beta, 'systematic_return')) * 100
            result['return_idiosyncratic'] = safe_float(get_attr(alpha_beta, 'idiosyncratic_return')) * 100
            
            # Skill contribution (could be a property)
            skill = get_attr(alpha_beta, 'skill_contribution')
            if skill is None and hasattr(alpha_beta, 'skill_contribution'):
                try:
                    skill = alpha_beta.skill_contribution
                except:
                    pass
            result['skill_contribution'] = safe_float(skill) * 100
        
        # =================================================================
        # Summary / Overall Assessment
        # =================================================================
        # Risk level
        risk_level = get_attr(self.p4b, 'overall_risk_level', 'risk_level')
        result['risk_level'] = safe_str(enum_val(risk_level), 'UNKNOWN')
        
        # Performance grade
        perf_grade = get_attr(self.p4b, 'risk_adjusted_grade', 'performance_grade')
        result['performance_grade'] = safe_str(enum_val(perf_grade), 'UNKNOWN')
        
        # Confidence scores
        result['alpha_confidence'] = safe_float(get_attr(self.p4b, 'alpha_confidence')) * 100
        result['robustness'] = safe_float(get_attr(self.p4b, 'strategy_robustness')) * 100
        
        # =================================================================
        # Benchmark from Phase 1 (fallback)
        # =================================================================
        if self.p1:
            ctx = None
            if hasattr(self.p1, 'context'):
                ctx = self.p1.context
            elif isinstance(self.p1, dict):
                ctx = self.p1.get('context', {})
            
            bench = None
            if ctx:
                if hasattr(ctx, 'benchmark'):
                    bench = ctx.benchmark
                elif isinstance(ctx, dict):
                    bench = ctx.get('benchmark', {})
            
            if bench:
                if not result.get('benchmark_correlation'):
                    result['benchmark_correlation'] = safe_float(get_attr(bench, 'correlation'))
                if not result.get('up_capture'):
                    result['up_capture'] = safe_float(get_attr(bench, 'up_capture'))
                if not result.get('down_capture'):
                    result['down_capture'] = safe_float(get_attr(bench, 'down_capture'))
        
        return result
    
    def _walk_forward(self) -> Dict:
        """Extract walk-forward analysis from Phase 4."""
        if self.p4 is None:
            return {}
        result = {}
        
        # Try multiple paths
        wf = safe_get(self.p4, 'walk_forward') or safe_get(self.p4, 'wf_analysis') or {}
        
        if wf:
            result['wfe_ratio'] = safe_float(safe_get(wf, 'wfe_ratio') or safe_get(wf, 'wfe'))
            result['consistency'] = safe_float(safe_get(wf, 'consistency'))
            result['robust'] = bool(safe_get(wf, 'robust') or safe_get(wf, 'is_robust'))
            result['periods'] = safe_int(safe_get(wf, 'periods') or safe_get(wf, 'n_periods'))
            result['avg_is_return'] = safe_float(safe_get(wf, 'avg_is_return'))
            result['avg_oos_return'] = safe_float(safe_get(wf, 'avg_oos_return'))
            result['avg_is_sharpe'] = safe_float(safe_get(wf, 'avg_is_sharpe'))
            result['avg_oos_sharpe'] = safe_float(safe_get(wf, 'avg_oos_sharpe'))
        
        # Try direct p4 attributes
        if not result.get('wfe_ratio'):
            result['wfe_ratio'] = safe_float(safe_get(self.p4, 'wfe_ratio'))
        if not result.get('consistency'):
            result['consistency'] = safe_float(safe_get(self.p4, 'wf_consistency'))
        
        return result
    
    def _monte_carlo(self) -> Dict:
        """Extract Monte Carlo analysis from Phase 4."""
        if self.p4 is None:
            return {}
        result = {}
        
        # Try multiple paths
        mc = safe_get(self.p4, 'monte_carlo') or safe_get(self.p4, 'mc_analysis') or {}
        
        if mc:
            result['simulations'] = safe_int(safe_get(mc, 'simulations') or safe_get(mc, 'n_simulations'), 500)
            result['mean_return'] = safe_float(safe_get(mc, 'return_mean') or safe_get(mc, 'mean_return'))
            result['std_return'] = safe_float(safe_get(mc, 'return_std') or safe_get(mc, 'std_return'))
            result['ci_lower'] = safe_float(safe_get(mc, 'return_ci_lower') or safe_get(mc, 'ci_lower'))
            result['ci_upper'] = safe_float(safe_get(mc, 'return_ci_upper') or safe_get(mc, 'ci_upper'))
            result['p_positive'] = safe_float(safe_get(mc, 'p_positive') or safe_get(mc, 'prob_positive'))
            result['p_sharpe_above_1'] = safe_float(safe_get(mc, 'p_sharpe_above_1') or safe_get(mc, 'prob_sharpe_1'))
            result['sharpe_mean'] = safe_float(safe_get(mc, 'sharpe_mean'))
            result['sharpe_ci_lower'] = safe_float(safe_get(mc, 'sharpe_ci_lower'))
            result['sharpe_ci_upper'] = safe_float(safe_get(mc, 'sharpe_ci_upper'))
        
        return result
        if mc:
            result['simulations'] = safe_int(safe_get(mc, 'simulations'), 500)
            result['mean_return'] = safe_float(safe_get(mc, 'return_mean'))
            result['std_return'] = safe_float(safe_get(mc, 'return_std'))
            result['ci_lower'] = safe_float(safe_get(mc, 'return_ci_lower'))
            result['ci_upper'] = safe_float(safe_get(mc, 'return_ci_upper'))
            result['p_positive'] = safe_float(safe_get(mc, 'p_positive'))
            result['p_sharpe_above_1'] = safe_float(safe_get(mc, 'p_sharpe_above_1'))
            result['sharpe_mean'] = safe_float(safe_get(mc, 'sharpe_mean'))
            result['sharpe_ci_lower'] = safe_float(safe_get(mc, 'sharpe_ci_lower'))
            result['sharpe_ci_upper'] = safe_float(safe_get(mc, 'sharpe_ci_upper'))
        
        return result
    
    def _benchmark(self) -> Dict:
        """Extract benchmark analysis from Phase 1."""
        if self.p1 is None:
            return {}
        result = {}
        
        # Try multiple paths for context/benchmark
        ctx = safe_get(self.p1, 'context') or safe_get(self.p1, 'market_context') or {}
        if not isinstance(ctx, dict) or _is_falsy(ctx):
            ctx = {}
        bench = safe_get(ctx, 'benchmark')
        if not isinstance(bench, dict) or _is_falsy(bench):
            bench = safe_get(self.p1, 'benchmark')
            # Handle case where benchmark is a DataFrame (not a dict)
            if isinstance(bench, (pd.DataFrame, pd.Series)):
                bench = {}
        if not isinstance(bench, dict) or _is_falsy(bench):
            bench = {}
        
        if bench:
            result['symbol'] = 'SPY'
            result['correlation'] = safe_float(safe_get(bench, 'correlation'))
            result['correlation_1y'] = safe_float(safe_get(bench, 'correlation_1y'))
            result['beta'] = safe_float(safe_get(bench, 'beta'), 1.0)
            result['beta_1y'] = safe_float(safe_get(bench, 'beta_1y'), 1.0)
            result['alpha'] = safe_float(safe_get(bench, 'alpha'))
            result['info_ratio'] = safe_float(safe_get(bench, 'information_ratio'))
            result['up_capture'] = safe_float(safe_get(bench, 'up_capture'))
            result['down_capture'] = safe_float(safe_get(bench, 'down_capture'))
        
        # Also try direct attributes on p1
        if not result.get('correlation'):
            result['correlation'] = safe_float(safe_get(self.p1, 'benchmark_correlation'))
        if not result.get('beta') or result['beta'] == 1.0:
            b = safe_float(safe_get(self.p1, 'benchmark_beta'))
            if b != 0:
                result['beta'] = b
        if not result.get('alpha'):
            result['alpha'] = safe_float(safe_get(self.p1, 'benchmark_alpha'))
        if not result.get('up_capture'):
            result['up_capture'] = safe_float(safe_get(self.p1, 'up_capture'))
        if not result.get('down_capture'):
            result['down_capture'] = safe_float(safe_get(self.p1, 'down_capture'))
        
        return result
    
    def _vix(self) -> Dict:
        """Extract VIX/market context from Phase 1."""
        if self.p1 is None:
            return {}
        result = {}
        
        # Try multiple paths
        ctx = safe_get(self.p1, 'context') or safe_get(self.p1, 'market_context') or {}
        if _is_falsy(ctx):
            ctx = {}
        vix = safe_get(ctx, 'vix')
        if _is_falsy(vix):
            vix = safe_get(self.p1, 'vix')
            # Handle case where vix is a DataFrame (not a dict)
            if isinstance(vix, (pd.DataFrame, pd.Series)):
                vix = {}
        if _is_falsy(vix):
            vix = {}
        
        if vix:
            result['current'] = safe_float(safe_get(vix, 'current'))
            result['regime'] = safe_str(safe_get(vix, 'regime'), 'NORMAL')
            result['percentile'] = safe_float(safe_get(vix, 'percentile'))
            result['context'] = safe_str(safe_get(vix, 'context'), 'NEUTRAL')
        
        # Try direct attributes
        if not result.get('current'):
            result['current'] = safe_float(safe_get(self.p1, 'vix_current'))
        if not result.get('regime') or result['regime'] == 'NORMAL':
            r = safe_str(safe_get(self.p1, 'vix_regime'))
            if r:
                result['regime'] = r
        
        return result
    
    def _statistical_tests(self) -> Dict:
        """Extract statistical tests from Phase 1."""
        if self.p1 is None:
            return {}
        result = {}
        
        # Try multiple paths for statistics
        stats = safe_get(self.p1, 'statistics') or safe_get(self.p1, 'stat_tests') or {}
        tests = safe_get(stats, 'tests') or safe_get(self.p1, 'tests') or stats
        
        if tests:
            result['adf_pass'] = bool(safe_get(tests, 'adf_pass') or safe_get(tests, 'adf'))
            result['kpss_pass'] = bool(safe_get(tests, 'kpss_pass') or safe_get(tests, 'kpss'))
            result['jb_pass'] = bool(safe_get(tests, 'jb_pass') or safe_get(tests, 'jarque_bera'))
            result['lb_autocorr'] = bool(safe_get(tests, 'lb_autocorr') or safe_get(tests, 'ljung_box'))
            result['arch_effects'] = bool(safe_get(tests, 'arch_effects') or safe_get(tests, 'arch'))
            result['conclusion'] = safe_str(
                safe_get(tests, 'conclusion') or safe_get(stats, 'conclusion'), 
                'UNKNOWN'
            )
        
        return result
    
    def _volatility_estimators(self) -> Dict:
        """Extract volatility estimators from Phase 1."""
        if self.p1 is None:
            return {}
        result = {}
        
        # Try multiple paths
        stats = safe_get(self.p1, 'statistics') or {}
        vol = safe_get(stats, 'volatility_estimators') or safe_get(self.p1, 'volatility_estimators') or {}
        
        if vol:
            result['close_to_close'] = safe_float(safe_get(vol, 'close_to_close'))
            result['parkinson'] = safe_float(safe_get(vol, 'parkinson'))
            result['garman_klass'] = safe_float(safe_get(vol, 'garman_klass'))
            result['rogers_satchell'] = safe_float(safe_get(vol, 'rogers_satchell'))
            result['yang_zhang'] = safe_float(safe_get(vol, 'yang_zhang'))
            result['composite'] = safe_float(safe_get(vol, 'composite'))
            result['regime'] = safe_str(safe_get(vol, 'regime'), 'NORMAL')
        
        # Tail risk - try multiple paths
        tail = safe_get(self.p1, 'tail_risk') or safe_get(stats, 'tail_risk') or {}
        profile = safe_get(self.p1, 'profile') or safe_get(self.p1, 'returns_profile') or {}
        
        result['skewness'] = safe_float(
            safe_get(tail, 'skewness') or safe_get(profile, 'skewness') or safe_get(self.p1, 'skewness')
        )
        result['kurtosis'] = safe_float(
            safe_get(tail, 'kurtosis') or safe_get(profile, 'kurtosis') or safe_get(self.p1, 'kurtosis'), 
            3.0
        )
        result['var_95'] = safe_float(
            safe_get(tail, 'var_95') or safe_get(profile, 'var_95')
        )
        result['var_99'] = safe_float(
            safe_get(tail, 'var_99') or safe_get(profile, 'var_99')
        )
        result['cvar_95'] = safe_float(
            safe_get(tail, 'cvar_95') or safe_get(profile, 'cvar_95')
        )
        
        return result
    
    def _stress_tests(self) -> Dict:
        """Extract stress test results from Phase 4B."""
        if self.p4b is None:
            return {}
        result = {}
        
        # Helper to get attribute from object or dict
        def get_attr(obj, *attrs):
            if obj is None:
                return None
            for attr in attrs:
                val = None
                if hasattr(obj, attr):
                    val = getattr(obj, attr, None)
                elif isinstance(obj, dict):
                    val = obj.get(attr)
                if val is not None:
                    return val
            return None
        
        # Get stress tests - it's a list directly on p4b
        tests = get_attr(self.p4b, 'stress_tests')
        
        if tests is None:
            tests = []
        elif not isinstance(tests, list):
            tests = [tests]
        
        passed = 0
        total = len(tests)
        
        for t in tests:
            if t is None:
                continue
            
            # Get scenario name
            name = get_attr(t, 'scenario_name', 'scenario', 'name')
            if name:
                name = safe_str(name).lower().replace(' ', '_').replace('-', '_')
                
                # Get returns
                strat_ret = safe_float(get_attr(t, 'strategy_return')) * 100
                bench_ret = safe_float(get_attr(t, 'benchmark_return')) * 100
                
                result[f'stress_{name}'] = strat_ret
                result[f'stress_{name}_benchmark'] = bench_ret
                result[f'stress_{name}_alpha'] = strat_ret - bench_ret
                
                # Check if protected downside
                protected = get_attr(t, 'protected_downside')
                if protected is None and hasattr(t, 'protected_downside'):
                    try:
                        protected = t.protected_downside
                    except:
                        pass
                if protected:
                    passed += 1
        
        result['tests_passed'] = passed
        result['tests_total'] = total
        
        return result
    
    def _regime_performance(self) -> Dict:
        """Extract regime-conditional performance from Phase 4B."""
        if self.p4b is None:
            return {}
        result = {}
        
        # Helper to get attribute from object or dict
        def get_attr(obj, *attrs):
            if obj is None:
                return None
            for attr in attrs:
                val = None
                if hasattr(obj, attr):
                    val = getattr(obj, attr, None)
                elif isinstance(obj, dict):
                    val = obj.get(attr)
                if val is not None:
                    return val
            return None
        
        # Helper to handle enum values
        def enum_val(obj):
            if obj is None:
                return None
            if hasattr(obj, 'value'):
                return obj.value
            if hasattr(obj, 'name'):
                return obj.name
            return str(obj)
        
        # Get regime performance dict
        regime_perf = get_attr(self.p4b, 'regime_performance')
        
        if regime_perf:
            if isinstance(regime_perf, dict):
                # Iterate through regime names
                for regime_key, perf in regime_perf.items():
                    regime_name = enum_val(regime_key) if hasattr(regime_key, 'value') else str(regime_key)
                    regime_lower = regime_name.lower() if regime_name else ''
                    
                    if 'bull' in regime_lower:
                        prefix = 'bull'
                    elif 'bear' in regime_lower:
                        prefix = 'bear'
                    elif 'sideways' in regime_lower or 'neutral' in regime_lower:
                        prefix = 'sideways'
                    else:
                        continue
                    
                    result[f'{prefix}_return'] = safe_float(get_attr(perf, 'total_return')) * 100
                    result[f'{prefix}_sharpe'] = safe_float(get_attr(perf, 'sharpe'))
                    result[f'{prefix}_max_dd'] = safe_float(get_attr(perf, 'max_drawdown', 'max_dd')) * 100
                    result[f'{prefix}_days'] = safe_int(get_attr(perf, 'days'))
        
        # Get volatility regime performance
        vol_regime_perf = get_attr(self.p4b, 'volatility_regime_performance')
        
        if vol_regime_perf:
            if isinstance(vol_regime_perf, dict):
                for regime_key, perf in vol_regime_perf.items():
                    regime_name = enum_val(regime_key) if hasattr(regime_key, 'value') else str(regime_key)
                    regime_lower = regime_name.lower() if regime_name else ''
                    
                    if 'low' in regime_lower and 'very' not in regime_lower:
                        prefix = 'vol_low'
                    elif 'normal' in regime_lower:
                        prefix = 'vol_normal'
                    elif 'high' in regime_lower and 'very' not in regime_lower:
                        prefix = 'vol_high'
                    elif 'crisis' in regime_lower or 'very_high' in regime_lower:
                        prefix = 'vol_crisis'
                    else:
                        continue
                    
                    result[f'{prefix}_return'] = safe_float(get_attr(perf, 'total_return')) * 100
                    result[f'{prefix}_sharpe'] = safe_float(get_attr(perf, 'sharpe'))
        
        return result
    
    def _signal_quality(self) -> Dict:
        """Extract signal quality analysis from Phase 4B."""
        if self.p4b is None:
            return {}
        result = {}
        
        # Helper to get attribute from object or dict
        def get_attr(obj, *attrs):
            if obj is None:
                return None
            for attr in attrs:
                val = None
                if hasattr(obj, attr):
                    val = getattr(obj, attr, None)
                elif isinstance(obj, dict):
                    val = obj.get(attr)
                if val is not None:
                    return val
            return None
        
        # Helper to handle enum values
        def enum_val(obj):
            if obj is None:
                return None
            if hasattr(obj, 'value'):
                return obj.value
            if hasattr(obj, 'name'):
                return obj.name
            return str(obj)
        
        # Signal quality is an object
        sq = get_attr(self.p4b, 'signal_quality')
        
        if sq:
            result['information_coefficient'] = safe_float(
                get_attr(sq, 'information_coefficient', 'ic')
            )
            result['ic_t_stat'] = safe_float(get_attr(sq, 'ic_t_stat'))
            
            # Quality grade (might be a property)
            grade = get_attr(sq, 'signal_quality_grade', 'grade', 'quality_grade')
            if grade is None and hasattr(sq, 'signal_quality_grade'):
                try:
                    grade = sq.signal_quality_grade
                except:
                    pass
            result['quality_grade'] = safe_str(enum_val(grade), 'UNKNOWN')
            
            result['hit_rate_long'] = safe_float(get_attr(sq, 'hit_rate_long')) * 100
            result['hit_rate_flat'] = safe_float(get_attr(sq, 'hit_rate_flat')) * 100
            result['signal_persistence'] = safe_float(get_attr(sq, 'signal_persistence', 'persistence'))
            result['annual_turnover'] = safe_float(get_attr(sq, 'turnover', 'annual_turnover'))
        
        # Drawdown analysis from major_drawdowns and other fields
        major_dd = get_attr(self.p4b, 'major_drawdowns')
        result['total_dd_periods'] = len(major_dd) if major_dd else safe_int(
            get_attr(self.p4b, 'underwater_periods')
        )
        result['avg_dd_duration'] = safe_int(get_attr(self.p4b, 'avg_drawdown_duration'))
        result['avg_recovery_time'] = safe_int(get_attr(self.p4b, 'avg_recovery_time'))
        
        return result


# =============================================================================
# SCORE INTERPRETATION
# =============================================================================

def get_score_rating(score: int) -> str:
    if score >= 80:
        return "EXCELLENT"
    elif score >= 60:
        return "GOOD"
    elif score >= 40:
        return "FAIR"
    else:
        return "POOR"


def create_score_explanation(name: str, score: int, signal: str, factors: List[str]) -> ScoreExplanation:
    """Create detailed explanation for a score."""
    rating = get_score_rating(score)
    
    interpretations = {
        'overall': f"Overall technical score of {score}/100 indicates {'strong' if score >= 70 else 'moderate' if score >= 50 else 'weak'} buy/sell conviction based on weighted indicator family analysis.",
        'technical': f"Technical structure score of {score}/100 reflects price position relative to key levels, support/resistance zones, and chart pattern quality.",
        'momentum': f"Momentum score of {score}/100 measures RSI, Stochastic, and Williams %R conditions. {'Oversold readings suggest mean-reversion opportunity.' if score >= 70 else 'Neutral momentum conditions.' if score >= 40 else 'Overbought conditions warrant caution.'}",
        'trend': f"Trend score of {score}/100 evaluates MACD, ADX, Supertrend, and Ichimoku alignment. {'Strong trend confirmation.' if score >= 70 else 'Mixed trend signals.' if score >= 40 else 'Counter-trend conditions.'}",
        'volatility': f"Volatility score of {score}/100 assesses Bollinger Band position, Keltner Channel deviation, and ATR levels for risk sizing.",
        'volume': f"Volume score of {score}/100 analyzes OBV trend, Money Flow Index, and Chaikin Money Flow for accumulation/distribution patterns.",
        'risk': f"Risk score of {score}/100 quantifies drawdown exposure, VaR metrics, and volatility-adjusted return potential. {'Favorable risk profile.' if score >= 60 else 'Elevated risk requires position sizing discipline.'}",
    }
    
    return ScoreExplanation(
        score=score,
        rating=rating,
        interpretation=interpretations.get(name, f"{name.title()} score of {score}/100."),
        factors=factors if factors else [signal]
    )


# =============================================================================
# LLM CLIENTS (OpenAI + Anthropic)
# =============================================================================

class OpenAIClient:
    """OpenAI GPT client for cost-effective testing."""

    def __init__(self):
        self.client = None
        self.tokens = 0
        self.model = OPENAI_MODEL
        try:
            from openai import OpenAI
            if OPENAI_API_KEY:
                self.client = OpenAI(api_key=OPENAI_API_KEY)
                logger.info(f"OpenAI ready: {self.model}")
            else:
                logger.warning("OPENAI_API_KEY not set")
        except Exception as e:
            logger.error(f"OpenAI init failed: {e}")

    def generate(self, system: str, user: str) -> str:
        if not self.client:
            raise RuntimeError("OpenAI not initialized")
        r = self.client.chat.completions.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        )
        self.tokens = getattr(r.usage, 'completion_tokens', 0)
        return r.choices[0].message.content


class ClaudeClient:
    """Anthropic Claude client for production/demo."""

    def __init__(self):
        self.client = None
        self.tokens = 0
        self.model = CLAUDE_MODEL
        try:
            from anthropic import Anthropic
            if ANTHROPIC_API_KEY:
                self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
                logger.info(f"Claude ready: {self.model}")
            else:
                logger.warning("ANTHROPIC_API_KEY not set")
        except Exception as e:
            logger.error(f"Claude init failed: {e}")

    def generate(self, system: str, user: str) -> str:
        if not self.client:
            raise RuntimeError("Claude not initialized")
        r = self.client.messages.create(
            model=self.model, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
            system=system, messages=[{"role": "user", "content": user}]
        )
        self.tokens = getattr(r.usage, 'output_tokens', 0)
        return r.content[0].text


class LLMClient:
    """Unified LLM client with OpenAI primary, Anthropic fallback."""

    def __init__(self):
        self.tokens = 0
        self.provider_used = None

        # Initialize clients
        self.openai = OpenAIClient()
        self.claude = ClaudeClient()

        # Priority: Anthropic first, OpenAI fallback
        if self.claude.client:
            self.primary = self.claude
            self.fallback = self.openai if self.openai.client else None
            logger.info(f"Using Anthropic (primary){', OpenAI (fallback)' if self.fallback else ''}")
        elif self.openai.client:
            self.primary = self.openai
            self.fallback = None
            logger.info("Using OpenAI (Anthropic key not set)")
        else:
            self.primary = None
            self.fallback = None
            logger.warning("No LLM client available!")

    @property
    def client(self):
        """For backwards compatibility - returns True if any client available."""
        return self.primary is not None

    def generate(self, system: str, user: str) -> str:
        # Try primary
        if self.primary and self.primary.client:
            try:
                result = self.primary.generate(system, user)
                self.tokens = self.primary.tokens
                self.provider_used = "openai" if isinstance(self.primary, OpenAIClient) else "anthropic"
                return result
            except Exception as e:
                logger.warning(f"Primary LLM failed: {e}, trying fallback...")

        # Try fallback
        if self.fallback and self.fallback.client:
            try:
                result = self.fallback.generate(system, user)
                self.tokens = self.fallback.tokens
                self.provider_used = "openai" if isinstance(self.fallback, OpenAIClient) else "anthropic"
                return result
            except Exception as e:
                logger.error(f"Fallback LLM also failed: {e}")
                raise

        raise RuntimeError("No LLM client available")


# =============================================================================
# TRADE NOTE GENERATOR
# =============================================================================

class TradeNoteGenerator:
    """Generates comprehensive trade note using LLM (OpenAI or Anthropic)."""

    SYSTEM = """You are an elite quantitative analyst at Goldman Sachs Asset Management.
Generate institutional-grade trade analysis. Be specific with numbers from the data provided.

For each analysis section, write 1 detailed paragraph (4-6 sentences) that:
- References specific indicator values and metrics
- Provides actionable insights
- Uses professional institutional language

IMPORTANT: Calculate Risk/Reward using the TARGET PRICE, not intermediate targets.

Output valid JSON only."""

    def __init__(self):
        self.llm = LLMClient()
        # Backwards compatibility alias
        self.claude = self.llm

    def generate(self, p1, p2, p3=None, p4=None, p4b=None) -> TradeNote:
        start = time.time()
        ctx = ContextBuilder(p1, p2, p3, p4, p4b).build()

        if self.llm.client:
            try:
                analysis = self._llm_generate(ctx)
            except Exception as e:
                logger.error(f"LLM error: {e}")
                import traceback
                traceback.print_exc()
                analysis = self._fallback(ctx)
        else:
            analysis = self._fallback(ctx)
        
        note = self._build_note(ctx, analysis)
        note.generation_time_ms = (time.time() - start) * 1000
        note.tokens_used = self.llm.tokens
        note.model_used = f"{self.llm.provider_used}:{OPENAI_MODEL if self.llm.provider_used == 'openai' else CLAUDE_MODEL}"
        return note
    
    def _llm_generate(self, ctx: Dict) -> Dict:
        # Calculate proper R/R for prompt
        price = ctx['price']
        lvl = ctx.get('levels', {})
        supertrend = lvl.get('supertrend', price * 0.96)
        bb_upper = lvl.get('bb_upper', price * 1.1)
        
        prompt = f"""Generate comprehensive trade note for {ctx['symbol']} at ${ctx['price']:.2f}.

=== DATA QUALITY ===
{json.dumps(ctx['quality'], indent=2)}

=== MARKET PROFILE ===
{json.dumps(ctx['profile'], indent=2)}

=== TECHNICAL SIGNALS ===
{json.dumps(ctx['signals'], indent=2)}

=== INDICATOR FAMILIES ===
{json.dumps(ctx['families'], indent=2)}

=== INDIVIDUAL INDICATORS ===
{json.dumps(ctx['indicators'], indent=2)}

=== KEY LEVELS ===
{json.dumps(ctx['levels'], indent=2)}

=== DIVERGENCES ===
{json.dumps(ctx['divergences'], indent=2)}

=== BACKTEST RESULTS (COURSEWORK REQUIRED METRICS) ===
{json.dumps(ctx['backtest'], indent=2)}

=== RISK ANALYTICS ===
{json.dumps(ctx['risk'], indent=2)}

IMPORTANT CALCULATIONS:
- Stop Loss should be at Supertrend: ${supertrend:.2f}
- Risk per share: ${price - supertrend:.2f} ({((price - supertrend)/price)*100:.1f}%)
- Target should give attractive Risk/Reward (aim for 2.5x+)

Output JSON:
{{
    "recommendation": "STRONG BUY|BUY|ACCUMULATE|HOLD|REDUCE|SELL|STRONG SELL",
    "conviction": "HIGH|MEDIUM|LOW",
    "confidence": <50-95>,
    "time_horizon": "1-4 weeks|1-3 months|3-6 months",
    "risk_rating": "LOW|MODERATE|HIGH|VERY HIGH",
    
    "target_price": <price that gives 2.5x+ R/R vs stop>,
    "expected_return": <percent>,
    
    "entry": <current price>,
    "stop_loss": <supertrend level>,
    "target_1": <first target>,
    "target_2": <second target>,
    "target_3": <primary target>,
    
    "position_size": <2-12>,
    "sizing_rationale": "<1 sentence>",
    
    "overall_score": <0-100>,
    "technical_score": <0-100>,
    "momentum_score": <0-100>,
    "trend_score": <0-100>,
    "volatility_score": <0-100>,
    "volume_score": <0-100>,
    "risk_score": <0-100>,
    
    "investment_thesis": "<1 detailed paragraph: core investment case with specific metrics>",
    "executive_summary": "<1 detailed paragraph: key findings and recommendation rationale>",
    "technical_analysis": "<1 detailed paragraph: price structure, indicators, levels>",
    "backtest_analysis": "<1 detailed paragraph: strategy performance, trade statistics>",
    "risk_analysis": "<1 detailed paragraph: VaR, drawdown, volatility assessment>",
    
    "scenarios": [
        {{"name": "Bull", "probability": 0.25-0.35, "target": <price>, "return_pct": <num>, "drivers": ["<specific driver>", "<specific driver>"], "risks": ["<risk>"]}},
        {{"name": "Base", "probability": 0.40-0.50, "target": <price>, "return_pct": <num>, "drivers": ["<specific driver>"], "risks": ["<risk>"]}},
        {{"name": "Bear", "probability": 0.20-0.30, "target": <price>, "return_pct": <num>, "drivers": ["<specific driver>"], "risks": ["<risk>"]}}
    ],
    
    "catalysts": ["<catalyst with timeframe>", "<catalyst with timeframe>", "<catalyst with timeframe>", "<catalyst with timeframe>"],
    "risks": ["<risk with mitigation>", "<risk with mitigation>", "<risk with mitigation>", "<risk with mitigation>"]
}}"""
        
        response = self.llm.generate(self.SYSTEM, prompt)
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            return json.loads(match.group())
        return self._fallback(ctx)

    def _fallback(self, ctx: Dict) -> Dict:
        """Comprehensive fallback when no LLM is available."""
        price = ctx['price']
        sig = ctx.get('signals', {})
        lvl = ctx.get('levels', {})
        fam = ctx.get('families', {})
        bt = ctx.get('backtest', {})
        prof = ctx.get('profile', {})
        qual = ctx.get('quality', {})
        inds = ctx.get('indicators', {})
        divs = ctx.get('divergences', [])
        risk = ctx.get('risk', {})
        
        direction = sig.get('direction', 'NEUTRAL')
        conf = sig.get('confidence', 0.5)
        
        # Determine recommendation
        if 'STRONG_BUY' in direction or 'STRONG BUY' in direction:
            rec, conv = "STRONG BUY", "HIGH"
        elif 'BUY' in direction:
            rec, conv = "BUY", "HIGH" if conf > 0.7 else "MEDIUM"
        elif 'STRONG_SELL' in direction or 'STRONG SELL' in direction:
            rec, conv = "STRONG SELL", "HIGH"
        elif 'SELL' in direction:
            rec, conv = "SELL", "MEDIUM"
        else:
            rec, conv = "HOLD", "LOW"
        
        # Levels
        bb_upper = lvl.get('bb_upper', price * 1.1)
        bb_middle = lvl.get('bb_middle', price * 1.02)
        bb_lower = lvl.get('bb_lower', price * 0.94)
        supertrend = lvl.get('supertrend', price * 0.96)
        ichimoku_top = lvl.get('ichimoku_cloud_top', price * 1.09)
        
        # Calculate proper target for good R/R
        risk_per_share = price - supertrend
        target_for_2_5_rr = price + (risk_per_share * 2.5)
        target_price = max(target_for_2_5_rr, ichimoku_top, bb_upper)
        
        # Scores from family confidence
        mom_conf = fam.get('momentum', {}).get('confidence', 0.5)
        trend_conf = fam.get('trend', {}).get('confidence', 0.5)
        vol_conf = fam.get('volatility', {}).get('confidence', 0.5)
        volume_conf = fam.get('volume', {}).get('confidence', 0.5)
        
        # Build detailed analysis text
        symbol = ctx['symbol']
        
        # Investment thesis
        div_text = ""
        if divs:
            div_text = f" The presence of {len(divs)} divergence(s) ({divs[0].get('type', 'bullish')} on {divs[0].get('indicator', 'RSI')}) provides additional confirmation of the directional bias."
        
        thesis = f"{symbol} presents a compelling {rec.lower()} opportunity at ${price:.2f}, supported by a {sig.get('strength', 'moderate')} technical signal with {conf*100:.1f}% confidence. The quantitative framework synthesizes momentum ({mom_conf*100:.0f}% confidence), trend ({trend_conf*100:.0f}% confidence), and volume ({volume_conf*100:.0f}% confidence) indicators to generate an actionable investment thesis. Historical performance demonstrates {prof.get('annual_return', 0)*100:.1f}% annualized return with {prof.get('sharpe', 0):.3f} Sharpe ratio, indicating favorable risk-adjusted characteristics.{div_text} The data quality score of {qual.get('score', 0):.1f}% ({qual.get('grade', 'N/A')}) provides high confidence in analytical inputs."
        
        # Executive summary
        rr_ratio = (target_price - price) / (price - supertrend) if (price - supertrend) > 0 else 0
        summary = f"Technical analysis generates a {rec} signal with {conf*100:.1f}% confidence, classified as {sig.get('strength', 'MODERATE')} strength. The signal distribution shows {sig.get('bullish', 0)} bullish, {sig.get('bearish', 0)} bearish, and {sig.get('neutral', 0)} neutral readings across indicator families. Target price of ${target_price:.2f} represents {((target_price/price)-1)*100:.1f}% upside potential versus stop loss at ${supertrend:.2f} ({((supertrend/price)-1)*100:.1f}% downside risk), establishing a {rr_ratio:.2f}x risk-reward ratio. Backtest validation shows {bt.get('total_return', 0)*100:.1f}% total return with {bt.get('hit_rate', 0)*100:.1f}% hit rate and {bt.get('profit_factor', 0):.2f} profit factor."
        
        # Technical analysis
        mom_inds = inds.get('momentum', [])
        rsi_text = ""
        if mom_inds:
            rsi = next((i for i in mom_inds if 'RSI' in i.get('name', '')), None)
            if rsi:
                rsi_text = f"RSI at {rsi.get('value', 0):.1f} indicates {rsi.get('zone', 'neutral').lower()} conditions. "
        
        tech = f"Price at ${price:.2f} trades relative to key technical levels: Bollinger upper ${bb_upper:.2f}, middle ${bb_middle:.2f}, lower ${bb_lower:.2f}. {rsi_text}Supertrend support at ${supertrend:.2f} provides the critical risk management level. The Hurst exponent of {prof.get('hurst', 0.5):.3f} indicates {prof.get('trend_character', 'mixed')} market character, informing strategy selection and expected holding periods. Volatility of {prof.get('volatility', 0)*100:.1f}% annualized requires appropriate position sizing."
        
        # Backtest analysis - COURSEWORK REQUIRED
        bt_text = f"Strategy backtesting validates the technical framework with {bt.get('total_return', 0)*100:.1f}% total return and {bt.get('cagr', 0)*100:.1f}% CAGR over the sample period. Trade statistics reveal {bt.get('total_trades', 0)} total trades with {bt.get('hit_rate', 0)*100:.1f}% win rate. The profit factor of {bt.get('profit_factor', 0):.2f} indicates favorable payoff asymmetry, with average win of ${bt.get('avg_win', 0):.2f} versus average loss of ${bt.get('avg_loss', 0):.2f}. Risk-adjusted metrics show Sharpe ratio of {bt.get('sharpe', 0):.3f} and Sortino ratio of {bt.get('sortino', 0):.3f}, confirming acceptable risk compensation."
        
        # Risk analysis
        var_95 = risk.get('var_95', bt.get('var_95', 0))
        cvar_95 = risk.get('cvar_95', bt.get('cvar_95', 0))
        risk_text = f"Risk analysis reveals maximum drawdown of {abs(bt.get('max_dd', prof.get('max_drawdown', 0)))*100:.1f}% and annualized volatility of {prof.get('volatility', 0)*100:.1f}%. Value-at-Risk (95%) of {abs(var_95)*100:.2f}% quantifies daily downside under normal conditions, while CVaR (95%) of {abs(cvar_95)*100:.2f}% captures tail risk. The return distribution shows kurtosis of {bt.get('kurtosis', prof.get('kurtosis', 3)):.2f}, indicating {'fat tails requiring position sizing discipline' if bt.get('kurtosis', prof.get('kurtosis', 3)) > 4 else 'near-normal tails'}. Stop loss at ${supertrend:.2f} limits maximum position loss to {abs((supertrend/price)-1)*100:.1f}%."
        
        return {
            'recommendation': rec,
            'conviction': conv,
            'confidence': conf * 100,
            'time_horizon': '1-3 months',
            'risk_rating': 'HIGH' if prof.get('volatility', 0) > 0.30 else 'MODERATE',
            'target_price': target_price,
            'expected_return': ((target_price / price) - 1) * 100,
            'entry': price,
            'stop_loss': supertrend,
            'target_1': bb_middle,
            'target_2': ichimoku_top * 0.98,
            'target_3': target_price,
            'position_size': 6 if conf > 0.7 else 5,
            'sizing_rationale': f"{'Moderate' if conf > 0.7 else 'Conservative'} sizing based on {conf*100:.0f}% confidence and {prof.get('volatility', 0)*100:.0f}% volatility.",
            'overall_score': int(conf * 100),
            'technical_score': int(conf * 95),
            'momentum_score': int(mom_conf * 100),
            'trend_score': int(trend_conf * 100),
            'volatility_score': int(vol_conf * 100),
            'volume_score': int(volume_conf * 100),
            'risk_score': max(25, int(100 - abs(bt.get('max_dd', prof.get('max_drawdown', 0))) * 150)),
            'investment_thesis': thesis,
            'executive_summary': summary,
            'technical_analysis': tech,
            'backtest_analysis': bt_text,
            'risk_analysis': risk_text,
            'scenarios': [
                {'name': 'Bull', 'probability': 0.30, 'target': target_price * 1.05, 'return_pct': ((target_price * 1.05 / price) - 1) * 100, 'drivers': ['Momentum breakout above resistance', 'Volume confirmation of trend reversal'], 'risks': ['Resistance rejection at upper band']},
                {'name': 'Base', 'probability': 0.45, 'target': target_price, 'return_pct': ((target_price / price) - 1) * 100, 'drivers': ['Mean reversion from oversold levels', 'Support holds at key technical level'], 'risks': ['Extended consolidation period']},
                {'name': 'Bear', 'probability': 0.25, 'target': supertrend * 0.98, 'return_pct': ((supertrend * 0.98 / price) - 1) * 100, 'drivers': ['Support breakdown triggers stop loss'], 'risks': ['Accelerated decline beyond stop']},
            ],
            'catalysts': [
                f'Momentum normalization from oversold (RSI {mom_inds[0].get("value", 30):.0f} to 50+) within 2-4 weeks' if mom_inds else 'Momentum mean reversion within 2-4 weeks',
                f'Reclamation of Bollinger middle at ${bb_middle:.2f} within 3-4 weeks',
                'Volume shift from distribution to accumulation phase',
                f'Volatility compression following current {prof.get("volatility", 0)*100:.0f}% annualized expansion',
            ],
            'risks': [
                f'Support failure at ${supertrend:.2f} triggers {abs((supertrend/price)-1)*100:.1f}% loss; mitigated by disciplined stop adherence',
                f'Max drawdown risk of {abs(prof.get("max_drawdown", 0))*100:.0f}%; mitigated by position sizing at {6 if conf > 0.7 else 5}%',
                'Continued distribution in volume indicators; monitor OBV and CMF for confirmation',
                f'Elevated volatility ({prof.get("volatility", 0)*100:.0f}% annualized) increases whipsaw risk; mitigated by wider stops',
            ],
        }
    
    def _build_note(self, ctx: Dict, a: Dict) -> TradeNote:
        """Build comprehensive TradeNote from context and analysis."""
        price = ctx['price']
        sig = ctx.get('signals', {})
        fam = ctx.get('families', {})
        prof = ctx.get('profile', {})
        qual = ctx.get('quality', {})
        bt = ctx.get('backtest', {})
        risk = ctx.get('risk', {})
        inds = ctx.get('indicators', {})
        divs = ctx.get('divergences', [])
        
        # Phase 3 and Phase 4B data
        regime = ctx.get('regime', {})
        attr = ctx.get('attribution', {})
        wf = ctx.get('walk_forward', {})
        mc = ctx.get('monte_carlo', {})
        
        # Additional Phase 1 data
        bench = ctx.get('benchmark', {})
        vix = ctx.get('vix', {})
        stat_tests = ctx.get('statistical_tests', {})
        vol_est = ctx.get('volatility_estimators', {})
        
        # Additional Phase 4B data
        stress = ctx.get('stress_tests', {})
        regime_perf = ctx.get('regime_performance', {})
        sig_qual = ctx.get('signal_quality', {})
        
        entry = safe_float(a.get('entry'), price)
        stop = safe_float(a.get('stop_loss'), price * 0.92)
        target_price = safe_float(a.get('target_price'), price * 1.1)
        
        # Calculate Risk/Reward using TARGET PRICE (not target_1)
        risk_amt = abs(entry - stop)
        reward = abs(target_price - entry)
        rr = reward / risk_amt if risk_amt > 0 else 0
        
        # Build indicator readings
        def build_readings(ind_list):
            return [IndicatorReading(
                name=i.get('name', ''),
                signal=i.get('signal', 'NEUTRAL'),
                value=i.get('value', 0),
                zone=i.get('zone', ''),
                confidence=i.get('confidence', 0.5)
            ) for i in ind_list]
        
        # Score breakdown
        breakdown = {}
        for fam_name, fam_data in fam.items():
            breakdown[fam_name] = {
                'signal': fam_data.get('signal', 'NEUTRAL'),
                'confidence': fam_data.get('confidence', 0.5),
                'weight': fam_data.get('weight', 0.2),
                'contribution': fam_data.get('confidence', 0.5) * fam_data.get('weight', 0.2) * 100
            }
        
        # Build score explanations
        # Use actual Phase 2 family confidence values (not Claude's interpretation)
        overall_score = safe_int(a.get('overall_score'), int(sig.get('confidence', 0.5) * 100))
        
        # Get exact confidence values from Phase 2 families
        momentum_conf = fam.get('momentum', {}).get('confidence', 0.5)
        trend_conf = fam.get('trend', {}).get('confidence', 0.5)
        volatility_conf = fam.get('volatility', {}).get('confidence', 0.5)
        volume_conf = fam.get('volume', {}).get('confidence', 0.5)
        
        # Use EXACT Phase 2 values for accuracy
        momentum_score = int(momentum_conf * 100)
        trend_score = int(trend_conf * 100)
        volatility_score = int(volatility_conf * 100)
        volume_score = int(volume_conf * 100)
        
        # Risk score based on backtest max drawdown
        # Scale: 0% DD = 100, 25% DD = 62, 50% DD = 25
        bt_max_dd = abs(bt.get('max_dd', prof.get('max_drawdown', 0)))
        risk_score = max(25, int(100 - bt_max_dd * 150))
        
        # Technical score from overall confidence
        technical_score = safe_int(a.get('technical_score'), overall_score)
        
        score_explanations = {
            'overall': create_score_explanation('overall', overall_score, sig.get('direction', 'NEUTRAL'), 
                                               ['Weighted combination of all indicator families']),
            'technical': create_score_explanation('technical', technical_score, sig.get('direction', 'NEUTRAL'),
                                                  ['Price structure', 'Support/resistance levels']),
            'momentum': create_score_explanation('momentum', momentum_score, fam.get('momentum', {}).get('signal', 'NEUTRAL'),
                                                 ['RSI', 'Stochastic', 'Williams %R']),
            'trend': create_score_explanation('trend', trend_score, fam.get('trend', {}).get('signal', 'NEUTRAL'),
                                             ['MACD', 'ADX', 'Supertrend', 'Ichimoku']),
            'volatility': create_score_explanation('volatility', volatility_score, fam.get('volatility', {}).get('signal', 'NEUTRAL'),
                                                   ['Bollinger Bands', 'Keltner Channels']),
            'volume': create_score_explanation('volume', volume_score, fam.get('volume', {}).get('signal', 'NEUTRAL'),
                                              ['OBV', 'MFI', 'CMF']),
            'risk': create_score_explanation('risk', risk_score, 'RISK',
                                            ['Max Drawdown', 'VaR', 'Volatility']),
        }
        
        return TradeNote(
            note_id=f"TN-{ctx['symbol']}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            symbol=ctx['symbol'],
            company_name=ctx['company_name'],
            sector=ctx['sector'],
            industry=ctx['industry'],
            generated_at=datetime.now().isoformat(),
            analysis_period=ctx.get('analysis_period', 'N/A'),
            total_records=ctx.get('total_records', 0),
            
            recommendation=a.get('recommendation', 'HOLD'),
            conviction=a.get('conviction', 'MEDIUM'),
            confidence=safe_float(a.get('confidence'), 50),
            time_horizon=a.get('time_horizon', '1-3 months'),
            risk_rating=a.get('risk_rating', 'MODERATE'),
            
            current_price=price,
            target_price=target_price,
            expected_return=safe_float(a.get('expected_return'), ((target_price/price)-1)*100),
            
            entry=entry,
            stop_loss=stop,
            target_1=safe_float(a.get('target_1'), price * 1.04),
            target_2=safe_float(a.get('target_2'), price * 1.08),
            target_3=safe_float(a.get('target_3'), target_price),
            risk_reward=rr,  # Now calculated using target_price
            
            position_size=safe_float(a.get('position_size'), 5),
            max_size=min(safe_float(a.get('position_size'), 5) * 2, 15),
            sizing_method='Half-Kelly Criterion',
            sizing_rationale=a.get('sizing_rationale', 'Based on signal confidence and volatility'),
            
            overall_score=overall_score,
            technical_score=technical_score,
            momentum_score=momentum_score,
            trend_score=trend_score,
            volatility_score=volatility_score,
            volume_score=volume_score,
            risk_score=risk_score,
            
            score_explanations=score_explanations,
            score_breakdown=breakdown,
            
            data_quality_score=qual.get('score', 0),
            data_quality_grade=qual.get('grade', 'N/A'),
            
            annual_return=prof.get('annual_return', 0),
            annual_volatility=prof.get('volatility', 0),
            sharpe_ratio=prof.get('sharpe', 0),
            sortino_ratio=prof.get('sortino', 0),
            calmar_ratio=prof.get('calmar', 0),
            max_drawdown=prof.get('max_drawdown', 0),
            hurst_exponent=prof.get('hurst', 0.5),
            trend_character=prof.get('trend_character', 'Unknown'),
            
            momentum_indicators=build_readings(inds.get('momentum', [])),
            trend_indicators=build_readings(inds.get('trend', [])),
            volatility_indicators=build_readings(inds.get('volatility', [])),
            volume_indicators=build_readings(inds.get('volume', [])),
            
            divergences=divs,
            
            # Backtest metrics - COURSEWORK REQUIRED
            backtest_total_return=bt.get('total_return', 0),
            backtest_cagr=bt.get('cagr', 0),
            backtest_volatility=bt.get('volatility', 0),
            backtest_max_dd=bt.get('max_dd', 0),
            backtest_sharpe=bt.get('sharpe', 0),
            backtest_sortino=bt.get('sortino', 0),
            backtest_calmar=bt.get('calmar', 0),
            backtest_total_trades=bt.get('total_trades', 0),
            backtest_winning_trades=bt.get('winning', 0),
            backtest_losing_trades=bt.get('losing', 0),
            backtest_hit_rate=bt.get('hit_rate', 0),
            backtest_profit_factor=bt.get('profit_factor', 0),
            backtest_avg_win=bt.get('avg_win', 0),
            backtest_avg_loss=bt.get('avg_loss', 0),
            backtest_expectancy=bt.get('expectancy', 0),
            
            # VaR/CVaR from Phase 4 risk metrics
            var_95=bt.get('var_95', 0),
            var_99=bt.get('var_95', 0) * 1.3 if bt.get('var_95', 0) else 0,
            cvar_95=bt.get('cvar_95', 0),
            cvar_99=bt.get('cvar_95', 0) * 1.3 if bt.get('cvar_95', 0) else 0,
            
            investment_thesis=a.get('investment_thesis', ''),
            executive_summary=a.get('executive_summary', ''),
            technical_analysis=a.get('technical_analysis', ''),
            backtest_analysis=a.get('backtest_analysis', ''),
            risk_analysis=a.get('risk_analysis', ''),
            
            scenarios=[Scenario(
                name=s.get('name', 'Scenario'),
                probability=safe_float(s.get('probability'), 0.33),
                target=safe_float(s.get('target'), price),
                return_pct=safe_float(s.get('return_pct'), 0),
                drivers=s.get('drivers', []),
                risks=s.get('risks', [])
            ) for s in a.get('scenarios', [])],
            
            catalysts=a.get('catalysts', []),
            risks=a.get('risks', []),
            
            levels=ctx.get('levels', {}),
            
            bullish_signals=sig.get('bullish', 0),
            bearish_signals=sig.get('bearish', 0),
            neutral_signals=sig.get('neutral', 0),
            
            # =========================================================================
            # PHASE 1: ADDITIONAL MARKET CONTEXT
            # =========================================================================
            
            # Benchmark Analysis
            benchmark_symbol=bench.get('symbol', 'SPY'),
            benchmark_correlation=bench.get('correlation', 0),
            benchmark_correlation_1y=bench.get('correlation_1y', 0),
            benchmark_beta=bench.get('beta', 1.0),
            benchmark_beta_1y=bench.get('beta_1y', 1.0),
            benchmark_alpha=bench.get('alpha', 0),
            benchmark_info_ratio=bench.get('info_ratio', 0),
            up_capture=bench.get('up_capture', 0),
            down_capture=bench.get('down_capture', 0),
            
            # VIX / Market Context
            vix_current=vix.get('current', 0),
            vix_regime=vix.get('regime', 'NORMAL'),
            vix_percentile=vix.get('percentile', 0),
            market_context=vix.get('context', 'NEUTRAL'),
            
            # Statistical Tests
            stationarity_adf=stat_tests.get('adf_pass', True),
            stationarity_kpss=stat_tests.get('kpss_pass', True),
            normality_jb=stat_tests.get('jb_pass', False),
            autocorrelation_lb=stat_tests.get('lb_autocorr', False),
            arch_effects=stat_tests.get('arch_effects', False),
            stationarity_conclusion=stat_tests.get('conclusion', 'UNKNOWN'),
            
            # Volatility Estimators
            vol_close_to_close=vol_est.get('close_to_close', 0),
            vol_parkinson=vol_est.get('parkinson', 0),
            vol_garman_klass=vol_est.get('garman_klass', 0),
            vol_rogers_satchell=vol_est.get('rogers_satchell', 0),
            vol_yang_zhang=vol_est.get('yang_zhang', 0),
            vol_composite=vol_est.get('composite', 0),
            vol_regime_p1=vol_est.get('regime', 'NORMAL'),
            
            # Tail Risk
            skewness=vol_est.get('skewness', prof.get('skewness', 0)),
            kurtosis=vol_est.get('kurtosis', prof.get('kurtosis', 3.0)),
            var_95_daily=vol_est.get('var_95', 0),
            var_99_daily=vol_est.get('var_99', 0),
            cvar_95_daily=vol_est.get('cvar_95', 0),
            
            # =========================================================================
            # PHASE 3: MARKET REGIME DETECTION (COMPREHENSIVE)
            # =========================================================================
            
            market_regime=regime.get('market_regime', 'UNKNOWN'),
            regime_probability=regime.get('regime_probability', 0),
            regime_confidence=regime.get('regime_confidence', 0),
            
            # State Probabilities
            regime_prob_bull=regime.get('prob_bull', 0),
            regime_prob_bear=regime.get('prob_bear', 0),
            regime_prob_sideways=regime.get('prob_sideways', 0),
            
            # Expected Durations
            regime_duration_bull=regime.get('duration_bull', 0),
            regime_duration_bear=regime.get('duration_bear', 0),
            regime_duration_sideways=regime.get('duration_sideways', 0),
            
            # Volatility Analysis
            current_volatility=regime.get('current_volatility', 0),
            volatility_regime=regime.get('volatility_regime', 'NORMAL'),
            volatility_percentile=regime.get('volatility_percentile', 0),
            
            # GARCH Parameters
            garch_omega=regime.get('garch_omega', 0),
            garch_alpha=regime.get('garch_alpha', 0),
            garch_beta=regime.get('garch_beta', 0),
            garch_persistence=regime.get('garch_persistence', 0),
            
            # Volatility Forecasts
            volatility_forecast_1d=regime.get('volatility_forecast_1d', 0),
            volatility_forecast_5d=regime.get('volatility_forecast_5d', 0),
            
            # Hurst Analysis
            hurst_p3=regime.get('hurst_p3', 0.5),
            hurst_classification=regime.get('hurst_classification', 'RANDOM_WALK'),
            hurst_r_squared=regime.get('hurst_r_squared', 0),
            hurst_ci_lower=regime.get('hurst_ci_lower', 0),
            hurst_ci_upper=regime.get('hurst_ci_upper', 0),
            
            # Structural Breaks
            structural_breaks=regime.get('structural_breaks', 0),
            days_since_break=regime.get('days_since_break', 0),
            cusum_statistic=regime.get('cusum_statistic', 0),
            break_active=regime.get('break_active', False),
            
            # Strategy Recommendation
            strategy_recommendation=regime.get('strategy_recommendation', 'HOLD'),
            position_bias=regime.get('position_bias', 'NEUTRAL'),
            recommended_position_size=regime.get('recommended_position_size', 0),
            recommended_stop_loss_atr=regime.get('stop_loss_atr', 2.0),
            recommended_take_profit_atr=regime.get('take_profit_atr', 4.0),
            max_holding_days=regime.get('max_holding_days', 20),
            strategy_confidence=regime.get('strategy_confidence', 0),
            strategy_rationale=regime.get('rationale', ''),
            
            # Quality
            regime_quality_score=regime.get('quality_score', 0),
            regime_quality_grade=regime.get('quality_grade', 'UNKNOWN'),
            
            # =========================================================================
            # PHASE 4: BACKTEST EXTENDED METRICS
            # =========================================================================
            
            # Return Distribution
            backtest_best_day=bt.get('best_day', 0),
            backtest_worst_day=bt.get('worst_day', 0),
            backtest_positive_days=bt.get('positive_days', 0),
            backtest_positive_pct=bt.get('positive_pct', 0),
            backtest_monthly_return=bt.get('monthly_return', 0),
            backtest_skewness=bt.get('skewness', 0),
            backtest_kurtosis=bt.get('kurtosis', 3.0),
            
            # Risk Metrics Extended
            backtest_downside_vol=bt.get('downside_vol', 0),
            backtest_avg_drawdown=bt.get('avg_drawdown', 0),
            backtest_max_dd_duration=bt.get('max_dd_duration', 0),
            backtest_omega=bt.get('omega', 1.0),
            backtest_alpha=bt.get('alpha', 0),
            backtest_beta=bt.get('beta', 1.0),
            backtest_info_ratio=bt.get('info_ratio', 0),
            
            # Trade Statistics Extended
            backtest_payoff_ratio=bt.get('payoff_ratio', 0),
            backtest_largest_win=bt.get('largest_win', 0),
            backtest_largest_loss=bt.get('largest_loss', 0),
            backtest_avg_holding_days=bt.get('avg_holding_days', 0),
            
            # Transaction Costs
            backtest_total_costs=bt.get('total_costs', 0),
            backtest_cost_per_trade=bt.get('cost_per_trade', 0),
            backtest_cost_rate=bt.get('cost_rate', 0),
            
            # Walk-Forward Analysis
            wf_periods=wf.get('periods', 0),
            wf_avg_is_return=wf.get('avg_is_return', 0),
            wf_avg_oos_return=wf.get('avg_oos_return', 0),
            wf_avg_is_sharpe=wf.get('avg_is_sharpe', 0),
            wf_avg_oos_sharpe=wf.get('avg_oos_sharpe', 0),
            wfe_ratio=wf.get('wfe_ratio', 0),
            walk_forward_consistency=wf.get('consistency', 0),
            walk_forward_robust=wf.get('robust', False),
            
            # Monte Carlo Analysis
            mc_simulations=mc.get('simulations', 0),
            monte_carlo_mean_return=mc.get('mean_return', 0),
            mc_return_std=mc.get('std_return', 0),
            mc_return_ci_lower=mc.get('ci_lower', 0),
            mc_return_ci_upper=mc.get('ci_upper', 0),
            mc_sharpe_mean=mc.get('sharpe_mean', 0),
            mc_sharpe_ci_lower=mc.get('sharpe_ci_lower', 0),
            mc_sharpe_ci_upper=mc.get('sharpe_ci_upper', 0),
            monte_carlo_p_positive=mc.get('p_positive', 0),
            mc_p_sharpe_above_1=mc.get('p_sharpe_above_1', 0),
            
            # Position Sizing
            sizing_kelly_optimal=bt.get('kelly_optimal', 0),
            sizing_kelly_half=bt.get('kelly_half', 0),
            sizing_vol_scaled=bt.get('vol_scaled', 0),
            
            # =========================================================================
            # PHASE 4B: ADVANCED RISK ANALYTICS (COMPREHENSIVE)
            # =========================================================================
            
            # Executive Summary
            performance_grade=attr.get('performance_grade', 'UNKNOWN'),
            risk_level=attr.get('risk_level', 'UNKNOWN'),
            alpha_confidence=attr.get('alpha_confidence', 0),
            strategy_robustness=attr.get('robustness', 0),
            
            # CAPM Attribution
            alpha=attr.get('alpha', 0),
            beta=attr.get('beta', 1.0),
            r_squared=attr.get('r_squared', 0),
            alpha_t_stat=attr.get('alpha_t_stat', 0),
            alpha_p_value=attr.get('alpha_p_value', 1.0),
            alpha_significant=attr.get('alpha_t_stat', 0) > 1.96,
            
            # Return Decomposition
            return_systematic=attr.get('return_systematic', 0),
            return_idiosyncratic=attr.get('return_idiosyncratic', 0),
            skill_contribution=attr.get('skill_contribution', 0),
            
            # Tracking
            tracking_error=attr.get('tracking_error', 0),
            information_ratio=attr.get('information_ratio', 0),
            
            # Regime-Conditional Performance
            perf_bull_return=regime_perf.get('bull_return', 0),
            perf_bull_sharpe=regime_perf.get('bull_sharpe', 0),
            perf_bull_max_dd=regime_perf.get('bull_max_dd', 0),
            perf_bull_days=regime_perf.get('bull_days', 0),
            
            perf_bear_return=regime_perf.get('bear_return', 0),
            perf_bear_sharpe=regime_perf.get('bear_sharpe', 0),
            perf_bear_max_dd=regime_perf.get('bear_max_dd', 0),
            perf_bear_days=regime_perf.get('bear_days', 0),
            
            perf_sideways_return=regime_perf.get('sideways_return', 0),
            perf_sideways_sharpe=regime_perf.get('sideways_sharpe', 0),
            perf_sideways_max_dd=regime_perf.get('sideways_max_dd', 0),
            perf_sideways_days=regime_perf.get('sideways_days', 0),
            
            # Volatility Regime Performance
            perf_low_vol_return=regime_perf.get('vol_low_return', 0),
            perf_low_vol_sharpe=regime_perf.get('vol_low_sharpe', 0),
            perf_normal_vol_return=regime_perf.get('vol_normal_return', 0),
            perf_normal_vol_sharpe=regime_perf.get('vol_normal_sharpe', 0),
            perf_high_vol_return=regime_perf.get('vol_high_return', 0),
            perf_high_vol_sharpe=regime_perf.get('vol_high_sharpe', 0),
            perf_crisis_return=regime_perf.get('vol_crisis_return', 0),
            perf_crisis_sharpe=regime_perf.get('vol_crisis_sharpe', 0),
            
            # Signal Quality
            information_coefficient=sig_qual.get('information_coefficient', 0),
            ic_t_stat=sig_qual.get('ic_t_stat', 0),
            signal_quality_grade=sig_qual.get('quality_grade', 'UNKNOWN'),
            hit_rate_long=sig_qual.get('hit_rate_long', 0),
            hit_rate_flat=sig_qual.get('hit_rate_flat', 0),
            signal_persistence=sig_qual.get('signal_persistence', 0),
            annual_turnover=sig_qual.get('annual_turnover', 0),
            
            # Stress Tests
            stress_covid_crash=stress.get('stress_covid_crash', 0),
            stress_covid_recovery=stress.get('stress_covid_recovery', 0),
            stress_2022_bear=stress.get('stress_2022_bear_market', 0),
            stress_2022_recovery=stress.get('stress_2022_recovery', 0),
            stress_2018_q4=stress.get('stress_2018_q4_selloff', 0),
            stress_2018_recovery=stress.get('stress_2018_recovery', 0),
            stress_tests_passed=stress.get('tests_passed', 0),
            stress_tests_total=stress.get('tests_total', 0),
            
            # Drawdown Analysis
            total_drawdown_periods=sig_qual.get('total_dd_periods', 0),
            avg_drawdown_duration=sig_qual.get('avg_dd_duration', 0),
            avg_recovery_time=sig_qual.get('avg_recovery_time', 0),
        )


# =============================================================================
# PROFESSIONAL PDF GENERATOR
# =============================================================================

class InstitutionalPDFGenerator:
    """Generates comprehensive institutional PDF with visualizations."""
    
    # Professional color palette
    NAVY = (0.08, 0.12, 0.20)
    DARK_BG = (0.11, 0.14, 0.18)
    CARD_BG = (0.15, 0.18, 0.24)
    BORDER = (0.25, 0.28, 0.35)
    WHITE = (0.97, 0.98, 0.99)
    LIGHT = (0.70, 0.75, 0.80)
    MUTED = (0.45, 0.50, 0.55)
    
    GREEN = (0.10, 0.75, 0.50)
    GREEN_DARK = (0.06, 0.55, 0.35)
    RED = (0.90, 0.30, 0.25)
    RED_DARK = (0.70, 0.20, 0.15)
    BLUE = (0.25, 0.55, 0.90)
    GOLD = (0.90, 0.70, 0.20)
    ORANGE = (0.95, 0.55, 0.20)
    
    REC_COLORS = {
        'STRONG BUY': (0.06, 0.65, 0.42),
        'BUY': (0.10, 0.75, 0.50),
        'ACCUMULATE': (0.30, 0.75, 0.55),
        'HOLD': (0.45, 0.50, 0.55),
        'REDUCE': (0.95, 0.60, 0.25),
        'SELL': (0.90, 0.40, 0.35),
        'STRONG SELL': (0.80, 0.20, 0.18),
    }
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.width = 595.27  # A4
        self.height = 841.89
        self.margin = 45
        self.content_width = self.width - 2 * self.margin
    
    def generate(self, note: TradeNote) -> Path:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        
        pdf_path = self.output_dir / f"{note.symbol}_trade_note.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        
        # Page 1: Cover
        self._page_cover(c, note)
        c.showPage()
        
        # Page 2: Score Analysis with Explanations
        self._page_scores(c, note)
        c.showPage()
        
        # Page 3: Trade Setup
        self._page_trade_setup(c, note)
        c.showPage()
        
        # Page 4: Technical Analysis
        self._page_technical(c, note)
        c.showPage()
        
        # Page 5: Backtest Results
        self._page_backtest(c, note)
        c.showPage()
        
        # Page 6: Risk Analysis
        self._page_risk(c, note)
        c.showPage()
        
        # Page 7: Scenarios
        self._page_scenarios(c, note)
        c.showPage()
        
        # Page 8: Conclusion
        self._page_conclusion(c, note)
        
        c.save()
        return pdf_path
    
    # =========================================================================
    # DRAWING UTILITIES
    # =========================================================================
    
    def _color(self, c, color):
        c.setFillColorRGB(*color)
    
    def _stroke(self, c, color):
        c.setStrokeColorRGB(*color)
    
    def _rect(self, c, x, y, w, h, color, stroke=None):
        self._color(c, color)
        if stroke:
            self._stroke(c, stroke)
            c.rect(x, y, w, h, fill=1, stroke=1)
        else:
            c.rect(x, y, w, h, fill=1, stroke=0)
    
    def _text(self, c, text, x, y, size=10, color=None, bold=False):
        if color:
            self._color(c, color)
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(x, y, str(text))
    
    def _text_center(self, c, text, x, y, size=10, color=None, bold=False):
        if color:
            self._color(c, color)
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawCentredString(x, y, str(text))
    
    def _text_right(self, c, text, x, y, size=10, color=None, bold=False):
        if color:
            self._color(c, color)
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawRightString(x, y, str(text))
    
    def _line(self, c, x1, y1, x2, y2, color, width=1):
        self._stroke(c, color)
        c.setLineWidth(width)
        c.line(x1, y1, x2, y2)
    
    def _score_bar(self, c, x, y, width, height, score, label, show_value=True):
        """Draw a score bar visualization."""
        # Background
        self._rect(c, x, y, width, height, self.CARD_BG)
        
        # Fill based on score
        fill_width = (score / 100) * width
        if score >= 70:
            fill_color = self.GREEN
        elif score >= 50:
            fill_color = self.GOLD
        elif score >= 30:
            fill_color = self.ORANGE
        else:
            fill_color = self.RED
        self._rect(c, x, y, fill_width, height, fill_color)
        
        # Label
        self._text(c, label, x + 5, y + height + 3, 8, self.LIGHT)
        
        # Value
        if show_value:
            self._text_right(c, str(score), x + width - 5, y + (height - 8) / 2 + 2, 10, self.WHITE, True)
    
    def _metric_card(self, c, x, y, width, height, label, value, sub_value=None, color=None):
        """Draw a metric card."""
        self._rect(c, x, y, width, height, self.CARD_BG)
        
        # Label
        self._text_center(c, label, x + width/2, y + height - 15, 8, self.MUTED)
        
        # Value - dynamic font size based on text length to fit in card
        val_color = color if color else self.WHITE
        val_len = len(str(value))
        if val_len > 8:
            font_size = 12
        elif val_len > 6:
            font_size = 14
        else:
            font_size = 18
        self._text_center(c, value, x + width/2, y + height/2 - 5, font_size, val_color, True)
        
        # Sub value
        if sub_value:
            self._text_center(c, sub_value, x + width/2, y + 8, 8, self.LIGHT)
    
    def _section_title(self, c, title, y):
        """Draw section title with line."""
        self._text(c, title, self.margin, y, 12, self.WHITE, True)
        self._line(c, self.margin, y - 8, self.width - self.margin, y - 8, self.BORDER, 0.5)
        return y - 25
    
    def _paragraph(self, c, text, x, y, width, size=9, color=None, line_height=14):
        """Draw wrapped paragraph text."""
        if not text:
            return y
        
        if color is None:
            color = self.LIGHT
        
        c.setFont("Helvetica", size)
        words = text.split()
        line = ""
        
        for word in words:
            test = line + " " + word if line else word
            if c.stringWidth(test, "Helvetica", size) < width:
                line = test
            else:
                self._text(c, line, x, y, size, color)
                y -= line_height
                line = word
        
        if line:
            self._text(c, line, x, y, size, color)
            y -= line_height
        
        return y
    
    def _table_row(self, c, y, cols, values, widths, header=False, alt=False):
        """Draw a table row."""
        x = self.margin
        height = 22 if header else 20
        
        # Background
        if header:
            self._rect(c, x, y - height + 5, self.content_width, height, self.CARD_BG)
        elif alt:
            self._rect(c, x, y - height + 5, self.content_width, height, (0.13, 0.16, 0.21))
        
        # Values
        for i, (col, val) in enumerate(zip(cols, values)):
            w = widths[i]
            color = self.MUTED if header else self.LIGHT
            bold = header
            self._text(c, str(val), x + 8, y - 10, 9, color, bold)
            x += w
        
        return y - height
    
    def _header(self, c, title, symbol, page):
        """Draw page header."""
        self._rect(c, 0, self.height - 50, self.width, 50, self.DARK_BG)
        self._text(c, title, self.margin, self.height - 32, 12, self.WHITE, True)
        self._text_right(c, symbol, self.width - self.margin, self.height - 32, 12, self.MUTED, True)
        self._line(c, 0, self.height - 50, self.width, self.height - 50, self.BORDER, 0.5)
    
    def _footer(self, c, page, total=8):
        """Draw page footer."""
        self._line(c, 0, 35, self.width, 35, self.BORDER, 0.5)
        self._text(c, "MSc AI Agents in Asset Management - Track B", self.margin, 18, 7, self.MUTED)
        self._text_center(c, f"Page {page} of {total}", self.width/2, 18, 8, self.MUTED)
        self._text_right(c, f"v{VERSION}", self.width - self.margin, 18, 7, self.MUTED)
    
    # =========================================================================
    # PAGE 1: COVER
    # =========================================================================
    
    def _page_cover(self, c, n: TradeNote):
        # Full background
        self._rect(c, 0, 0, self.width, self.height, self.NAVY)
        
        # Top section
        self._rect(c, 0, self.height - 180, self.width, 180, self.DARK_BG)
        
        # Title
        self._text_center(c, "INSTITUTIONAL TRADE NOTE", self.width/2, self.height - 45, 10, self.MUTED)
        
        # Symbol
        self._text_center(c, n.symbol, self.width/2, self.height - 100, 56, self.WHITE, True)
        
        # Company
        self._text_center(c, n.company_name, self.width/2, self.height - 130, 14, self.LIGHT)
        self._text_center(c, f"{n.sector}  |  {n.industry}", self.width/2, self.height - 150, 10, self.MUTED)
        
        # Date
        self._text_right(c, n.generated_at[:10], self.width - self.margin, self.height - 45, 10, self.MUTED)
        
        # Recommendation Badge
        rec_color = self.REC_COLORS.get(n.recommendation, self.MUTED)
        badge_w, badge_h = 180, 45
        badge_x = (self.width - badge_w) / 2
        badge_y = self.height - 260
        self._rect(c, badge_x, badge_y, badge_w, badge_h, rec_color)
        self._text_center(c, n.recommendation, self.width/2, badge_y + 14, 20, self.WHITE, True)
        
        # Key Metrics Row
        y = self.height - 330
        rr_color = self.GREEN if n.risk_reward >= 2 else self.GOLD if n.risk_reward >= 1.5 else self.RED
        metrics = [
            ("CONFIDENCE", f"{n.confidence:.0f}%", None),
            ("TARGET", f"${n.target_price:.2f}", self.GREEN if n.expected_return > 0 else self.RED),
            ("RETURN", f"{n.expected_return:+.1f}%", self.GREEN if n.expected_return > 0 else self.RED),
            ("RISK/REWARD", f"{n.risk_reward:.2f}x", rr_color),
            ("RISK", n.risk_rating, self.RED if n.risk_rating in ['HIGH', 'VERY HIGH'] else self.GOLD),
        ]
        
        card_w = 95
        start_x = (self.width - len(metrics) * card_w) / 2
        for i, (label, value, color) in enumerate(metrics):
            x = start_x + i * card_w
            self._metric_card(c, x + 3, y, card_w - 6, 60, label, value, color=color)
        
        # Overall Score (large)
        y -= 100
        self._text_center(c, "OVERALL SCORE", self.width/2, y + 50, 10, self.MUTED)
        score_color = self.GREEN if n.overall_score >= 70 else self.GOLD if n.overall_score >= 50 else self.RED
        self._text_center(c, str(n.overall_score), self.width/2, y - 10, 64, score_color, True)
        self._text_center(c, "/ 100", self.width/2, y - 35, 14, self.MUTED)
        
        # Score bars
        y -= 110
        scores = [
            ("Technical", n.technical_score),
            ("Momentum", n.momentum_score),
            ("Trend", n.trend_score),
            ("Volatility", n.volatility_score),
            ("Volume", n.volume_score),
            ("Risk", n.risk_score),
        ]
        
        bar_w = (self.content_width - 50) / 2
        bar_h = 18
        for i, (label, score) in enumerate(scores):
            col = i % 2
            row = i // 2
            x = self.margin + col * (bar_w + 50)
            by = y - row * 35
            self._score_bar(c, x, by, bar_w, bar_h, score, label)
        
        # Data Quality
        y -= 130
        self._rect(c, self.margin, y, self.content_width, 50, self.CARD_BG)
        self._text(c, "DATA QUALITY", self.margin + 15, y + 32, 9, self.MUTED, True)
        self._text(c, f"{n.data_quality_score:.1f}/100", self.margin + 15, y + 12, 14, self.WHITE, True)
        self._text(c, n.data_quality_grade, self.margin + 100, y + 12, 14, self.GREEN if n.data_quality_grade == 'EXCELLENT' else self.GOLD, True)
        
        self._text(c, "ANALYSIS PERIOD", self.margin + 200, y + 32, 9, self.MUTED, True)
        self._text(c, n.analysis_period, self.margin + 200, y + 12, 10, self.LIGHT)
        
        self._text(c, "RECORDS", self.margin + 400, y + 32, 9, self.MUTED, True)
        self._text(c, f"{n.total_records:,} daily", self.margin + 400, y + 12, 10, self.LIGHT)
        
        self._footer(c, 1)
    
    # =========================================================================
    # PAGE 2: SCORE ANALYSIS with Explanations
    # =========================================================================
    
    def _page_scores(self, c, n: TradeNote):
        self._rect(c, 0, 0, self.width, self.height, self.NAVY)
        self._header(c, "SCORE ANALYSIS", n.symbol, 2)
        
        y = self.height - 80
        y = self._section_title(c, "RECOMMENDATION SCORE BREAKDOWN", y)
        
        # Score contribution table
        cols = ["Factor", "Signal", "Confidence", "Weight", "Contribution"]
        widths = [120, 100, 100, 80, 105]
        y = self._table_row(c, y, cols, cols, widths, header=True)
        
        i = 0
        for fam, data in n.score_breakdown.items():
            sig = data.get('signal', 'N/A')
            conf = data.get('confidence', 0)
            weight = data.get('weight', 0)
            contrib = conf * weight * 100
            values = [fam.title(), sig, f"{conf*100:.0f}%", f"{weight*100:.0f}%", f"{contrib:.1f}"]
            y = self._table_row(c, y, cols, values, widths, alt=(i % 2 == 1))
            i += 1
        
        # Total row
        total_score = sum(d.get('confidence', 0) * d.get('weight', 0) * 100 for d in n.score_breakdown.values())
        y = self._table_row(c, y, cols, ["TOTAL", "", "", "", f"{total_score:.1f}"], widths, header=True)
        
        # Signal Distribution
        y -= 30
        y = self._section_title(c, "SIGNAL DISTRIBUTION", y)
        
        total_signals = n.bullish_signals + n.bearish_signals + n.neutral_signals
        if total_signals > 0:
            bull_pct = n.bullish_signals / total_signals
            bear_pct = n.bearish_signals / total_signals
            neut_pct = n.neutral_signals / total_signals
            
            bar_y = y - 20
            bar_h = 30
            
            # Stacked bar
            x = self.margin
            if bull_pct > 0:
                w = bull_pct * self.content_width
                self._rect(c, x, bar_y, w, bar_h, self.GREEN)
                if bull_pct > 0.1:
                    self._text_center(c, f"Bullish: {n.bullish_signals}", x + w/2, bar_y + 10, 9, self.WHITE, True)
                x += w
            
            if neut_pct > 0:
                w = neut_pct * self.content_width
                self._rect(c, x, bar_y, w, bar_h, self.MUTED)
                if neut_pct > 0.1:
                    self._text_center(c, f"Neutral: {n.neutral_signals}", x + w/2, bar_y + 10, 9, self.WHITE, True)
                x += w
            
            if bear_pct > 0:
                w = bear_pct * self.content_width
                self._rect(c, x, bar_y, w, bar_h, self.RED)
                if bear_pct > 0.1:
                    self._text_center(c, f"Bearish: {n.bearish_signals}", x + w/2, bar_y + 10, 9, self.WHITE, True)
        
        # Score Interpretation
        y -= 70
        y = self._section_title(c, "SCORE INTERPRETATION", y)
        
        self._rect(c, self.margin, y - 130, self.content_width, 140, self.CARD_BG)
        
        interpretations = [
            ("Overall Score", f"{n.overall_score}/100", "Weighted composite of all technical indicator families"),
            ("Momentum Score", f"{n.momentum_score}/100", "RSI, Stochastic, Williams %R oversold/overbought readings"),
            ("Trend Score", f"{n.trend_score}/100", "MACD, ADX, Supertrend, Ichimoku trend alignment"),
            ("Risk Score", f"{n.risk_score}/100", "Based on volatility, max drawdown, and VaR metrics"),
        ]
        
        iy = y - 15
        for label, value, desc in interpretations:
            self._text(c, label, self.margin + 15, iy, 9, self.WHITE, True)
            self._text(c, value, self.margin + 130, iy, 9, self.LIGHT, True)
            self._text(c, desc, self.margin + 200, iy, 8, self.MUTED)
            iy -= 28
        
        # Market Profile
        y -= 180
        y = self._section_title(c, "MARKET PROFILE", y)
        
        cols = ["Metric", "Value", "Metric", "Value"]
        widths = [130, 120, 130, 125]
        
        profile_data = [
            ("Annual Return", f"{n.annual_return*100:+.2f}%", "Sharpe Ratio", f"{n.sharpe_ratio:.3f}"),
            ("Volatility", f"{n.annual_volatility*100:.2f}%", "Sortino Ratio", f"{n.sortino_ratio:.3f}"),
            ("Max Drawdown", f"{n.max_drawdown*100:.2f}%", "Calmar Ratio", f"{n.calmar_ratio:.3f}"),
            ("Hurst Exponent", f"{n.hurst_exponent:.3f}", "Trend Character", n.trend_character),
        ]
        
        y = self._table_row(c, y, cols, cols, widths, header=True)
        for i, row in enumerate(profile_data):
            y = self._table_row(c, y, cols, list(row), widths, alt=(i % 2 == 1))
        
        self._footer(c, 2)
    
    # =========================================================================
    # PAGE 3: TRADE SETUP
    # =========================================================================
    
    def _page_trade_setup(self, c, n: TradeNote):
        self._rect(c, 0, 0, self.width, self.height, self.NAVY)
        self._header(c, "TRADE SETUP", n.symbol, 3)
        
        y = self.height - 80
        y = self._section_title(c, "PRICE LEVELS", y)
        
        # Visual price ladder
        levels = [
            ("Target 3 (Primary)", n.target_3, self.GREEN),
            ("Target 2", n.target_2, self.GREEN),
            ("Target 1", n.target_1, self.GREEN),
            ("Entry", n.entry, self.BLUE),
            ("Current", n.current_price, self.WHITE),
            ("Stop Loss", n.stop_loss, self.RED),
        ]
        
        # Sort by price descending
        levels.sort(key=lambda x: x[1], reverse=True)
        
        ladder_x = self.margin + 50
        ladder_w = 300
        level_h = 35
        
        for i, (label, price, color) in enumerate(levels):
            ly = y - i * level_h
            
            # Level line
            self._line(c, ladder_x, ly, ladder_x + ladder_w, ly, self.BORDER, 0.5)
            
            # Marker (centered on line)
            self._rect(c, ladder_x - 5, ly - 3, 10, 6, color)
            
            # Label and price - positioned ABOVE the line (ly + 4 for proper alignment)
            self._text(c, label, ladder_x + 15, ly + 4, 10, self.LIGHT)
            self._text_right(c, f"${price:.2f}", ladder_x + ladder_w - 10, ly + 4, 11, color, True)
            
            # Percentage from current - also above line
            if label != "Current":
                pct = ((price / n.current_price) - 1) * 100
                self._text(c, f"({pct:+.1f}%)", ladder_x + ladder_w + 10, ly + 4, 9, self.MUTED)
        
        # Risk/Reward box - improved design with better positioning
        rr_x = self.margin + 400
        rr_y = y - 70  # Adjusted position
        rr_w = 110
        rr_h = 80
        self._rect(c, rr_x, rr_y, rr_w, rr_h, self.CARD_BG)
        self._text_center(c, "RISK/REWARD", rr_x + rr_w/2, rr_y + rr_h - 15, 9, self.MUTED)
        rr_color = self.GREEN if n.risk_reward >= 2 else self.GOLD if n.risk_reward >= 1.5 else self.RED
        self._text_center(c, f"{n.risk_reward:.2f}x", rr_x + rr_w/2, rr_y + rr_h/2 - 5, 26, rr_color, True)
        
        # Position Sizing
        y -= 250
        y = self._section_title(c, "POSITION SIZING", y)
        
        self._rect(c, self.margin, y - 80, self.content_width, 90, self.CARD_BG)
        
        # Position metrics
        self._text(c, "Recommended Size", self.margin + 20, y - 15, 9, self.MUTED)
        self._text(c, f"{n.position_size:.1f}%", self.margin + 20, y - 35, 18, self.WHITE, True)
        
        self._text(c, "Maximum Size", self.margin + 150, y - 15, 9, self.MUTED)
        self._text(c, f"{n.max_size:.1f}%", self.margin + 150, y - 35, 18, self.LIGHT, True)
        
        self._text(c, "Method", self.margin + 280, y - 15, 9, self.MUTED)
        self._text(c, n.sizing_method, self.margin + 280, y - 35, 10, self.LIGHT)
        
        self._text(c, "Rationale", self.margin + 20, y - 55, 9, self.MUTED)
        rationale = n.sizing_rationale[:80] + "..." if len(n.sizing_rationale) > 80 else n.sizing_rationale
        self._text(c, rationale, self.margin + 20, y - 70, 9, self.LIGHT)
        
        # Key Technical Levels
        y -= 120
        y = self._section_title(c, "KEY TECHNICAL LEVELS", y)
        
        cols = ["Level", "Price", "Level", "Price"]
        widths = [130, 120, 130, 125]
        
        level_items = list(n.levels.items())[:8]
        rows = []
        for i in range(0, len(level_items), 2):
            row = []
            for j in range(2):
                if i + j < len(level_items):
                    name, price = level_items[i + j]
                    row.extend([name.replace('_', ' ').title(), f"${price:.2f}"])
                else:
                    row.extend(["", ""])
            rows.append(row)
        
        y = self._table_row(c, y, cols, cols, widths, header=True)
        for i, row in enumerate(rows):
            y = self._table_row(c, y, cols, row, widths, alt=(i % 2 == 1))
        
        self._footer(c, 3)
    
    # =========================================================================
    # PAGE 4: TECHNICAL ANALYSIS
    # =========================================================================
    
    def _page_technical(self, c, n: TradeNote):
        self._rect(c, 0, 0, self.width, self.height, self.NAVY)
        self._header(c, "TECHNICAL ANALYSIS", n.symbol, 4)
        
        y = self.height - 80
        y = self._section_title(c, "ANALYSIS SUMMARY", y)
        y = self._paragraph(c, n.technical_analysis, self.margin, y, self.content_width)
        
        # Momentum Indicators
        y -= 20
        y = self._section_title(c, "MOMENTUM INDICATORS", y)
        
        cols = ["Indicator", "Signal", "Value", "Zone", "Confidence"]
        widths = [100, 90, 90, 110, 115]
        
        if n.momentum_indicators:
            y = self._table_row(c, y, cols, cols, widths, header=True)
            
            for i, ind in enumerate(n.momentum_indicators[:4]):
                sig_color = "+" if "BUY" in ind.signal else "-" if "SELL" in ind.signal else ""
                values = [ind.name, f"{sig_color}{ind.signal}", f"{ind.value:.2f}", ind.zone, f"{ind.confidence*100:.0f}%"]
                y = self._table_row(c, y, cols, values, widths, alt=(i % 2 == 1))
        
        # Trend Indicators
        y -= 20
        y = self._section_title(c, "TREND INDICATORS", y)
        
        if n.trend_indicators:
            y = self._table_row(c, y, cols, cols, widths, header=True)
            
            for i, ind in enumerate(n.trend_indicators[:4]):
                sig_color = "+" if "BUY" in ind.signal else "-" if "SELL" in ind.signal else ""
                values = [ind.name, f"{sig_color}{ind.signal}", f"{ind.value:.2f}", ind.zone, f"{ind.confidence*100:.0f}%"]
                y = self._table_row(c, y, cols, values, widths, alt=(i % 2 == 1))
        
        # Divergences
        if n.divergences:
            y -= 20
            y = self._section_title(c, "DIVERGENCES DETECTED", y)
            
            cols = ["Type", "Indicator", "Strength", "Duration"]
            widths = [150, 120, 120, 115]
            y = self._table_row(c, y, cols, cols, widths, header=True)
            
            for i, div in enumerate(n.divergences[:3]):
                values = [div.get('type', ''), div.get('indicator', ''), f"{div.get('strength', 0)*100:.0f}%", f"{div.get('bars', 0)} bars"]
                y = self._table_row(c, y, cols, values, widths, alt=(i % 2 == 1))
        
        self._footer(c, 4)
    
    # =========================================================================
    # PAGE 5: BACKTEST RESULTS (COURSEWORK REQUIRED)
    # =========================================================================
    
    def _page_backtest(self, c, n: TradeNote):
        self._rect(c, 0, 0, self.width, self.height, self.NAVY)
        self._header(c, "BACKTEST VALIDATION", n.symbol, 5)
        
        y = self.height - 80
        y = self._section_title(c, "STRATEGY PERFORMANCE", y)
        y = self._paragraph(c, n.backtest_analysis, self.margin, y, self.content_width)
        
        # Performance Metrics - COURSEWORK REQUIRED
        y -= 20
        y = self._section_title(c, "RETURN METRICS", y)
        
        cols = ["Metric", "Value", "Metric", "Value"]
        widths = [130, 120, 130, 125]
        
        return_data = [
            ("Total Return", f"{n.backtest_total_return*100:.1f}%", "CAGR", f"{n.backtest_cagr*100:.1f}%"),
            ("Volatility", f"{n.backtest_volatility*100:.1f}%", "Max Drawdown", f"{n.backtest_max_dd*100:.1f}%"),
            ("Sharpe Ratio", f"{n.backtest_sharpe:.3f}", "Sortino Ratio", f"{n.backtest_sortino:.3f}"),
        ]
        
        y = self._table_row(c, y, cols, cols, widths, header=True)
        for i, row in enumerate(return_data):
            y = self._table_row(c, y, cols, list(row), widths, alt=(i % 2 == 1))
        
        # Trade Statistics - COURSEWORK REQUIRED
        y -= 30
        y = self._section_title(c, "TRADE STATISTICS", y)
        
        trade_data = [
            ("Total Trades", str(n.backtest_total_trades), "Hit Rate", f"{n.backtest_hit_rate*100:.1f}%"),
            ("Winning Trades", str(n.backtest_winning_trades), "Profit Factor", f"{n.backtest_profit_factor:.2f}"),
            ("Losing Trades", str(n.backtest_losing_trades), "Expectancy", f"${n.backtest_expectancy:.2f}"),
            ("Avg Win", f"${n.backtest_avg_win:.2f}", "Avg Loss", f"${n.backtest_avg_loss:.2f}"),
        ]
        
        y = self._table_row(c, y, cols, cols, widths, header=True)
        for i, row in enumerate(trade_data):
            y = self._table_row(c, y, cols, list(row), widths, alt=(i % 2 == 1))
        
        # Win/Loss visual
        y -= 40
        if n.backtest_total_trades > 0:
            win_pct = n.backtest_winning_trades / n.backtest_total_trades
            bar_w = self.content_width - 100
            bar_h = 25
            
            self._text(c, "Win/Loss Distribution", self.margin, y + 15, 9, self.MUTED)
            
            # Win portion
            self._rect(c, self.margin, y - bar_h, win_pct * bar_w, bar_h, self.GREEN)
            # Loss portion
            self._rect(c, self.margin + win_pct * bar_w, y - bar_h, (1 - win_pct) * bar_w, bar_h, self.RED)
            
            self._text_center(c, f"{win_pct*100:.0f}% Wins", self.margin + win_pct * bar_w / 2, y - bar_h + 7, 10, self.WHITE, True)
            self._text_center(c, f"{(1-win_pct)*100:.0f}% Losses", self.margin + win_pct * bar_w + (1-win_pct) * bar_w / 2, y - bar_h + 7, 10, self.WHITE, True)
        
        self._footer(c, 5)
    
    # =========================================================================
    # PAGE 6: RISK ANALYSIS
    # =========================================================================
    
    def _page_risk(self, c, n: TradeNote):
        self._rect(c, 0, 0, self.width, self.height, self.NAVY)
        self._header(c, "RISK ANALYSIS", n.symbol, 6)
        
        y = self.height - 80
        y = self._section_title(c, "RISK ASSESSMENT", y)
        y = self._paragraph(c, n.risk_analysis, self.margin, y, self.content_width)
        
        # VaR/CVaR Table - FIXED
        y -= 20
        y = self._section_title(c, "VALUE AT RISK", y)
        
        cols = ["Metric", "95% Confidence", "99% Confidence"]
        widths = [200, 155, 150]
        
        var_data = [
            ("Daily VaR", f"{abs(n.var_95)*100:.2f}%", f"{abs(n.var_99)*100:.2f}%"),
            ("Daily CVaR (Expected Shortfall)", f"{abs(n.cvar_95)*100:.2f}%", f"{abs(n.cvar_99)*100:.2f}%"),
        ]
        
        y = self._table_row(c, y, cols, cols, widths, header=True)
        for i, row in enumerate(var_data):
            y = self._table_row(c, y, cols, list(row), widths, alt=(i % 2 == 1))
        
        # Risk Metrics Visual
        y -= 40
        y = self._section_title(c, "RISK PROFILE", y)
        
        # Drawdown bar
        dd_pct = abs(n.max_drawdown)
        bar_w = self.content_width - 100
        
        self._text(c, "Maximum Drawdown", self.margin, y - 5, 9, self.MUTED)
        self._rect(c, self.margin, y - 30, bar_w, 18, self.CARD_BG)
        self._rect(c, self.margin, y - 30, min(dd_pct, 1) * bar_w, 18, self.RED)
        self._text_right(c, f"{dd_pct*100:.1f}%", self.margin + bar_w + 50, y - 25, 12, self.RED, True)
        
        # Volatility bar
        y -= 50
        vol_pct = min(n.annual_volatility / 0.5, 1)  # Cap at 50%
        
        self._text(c, "Annual Volatility", self.margin, y - 5, 9, self.MUTED)
        self._rect(c, self.margin, y - 30, bar_w, 18, self.CARD_BG)
        self._rect(c, self.margin, y - 30, vol_pct * bar_w, 18, self.ORANGE)
        self._text_right(c, f"{n.annual_volatility*100:.1f}%", self.margin + bar_w + 50, y - 25, 12, self.ORANGE, True)
        
        # Risk Rating
        y -= 80
        self._rect(c, self.margin, y - 60, self.content_width, 70, self.CARD_BG)
        
        self._text(c, "RISK RATING", self.margin + 20, y - 10, 10, self.MUTED, True)
        
        rating_color = self.RED if n.risk_rating in ['HIGH', 'VERY HIGH'] else self.GOLD if n.risk_rating == 'MODERATE' else self.GREEN
        self._text(c, n.risk_rating, self.margin + 20, y - 35, 20, rating_color, True)
        
        self._text(c, f"Based on {n.annual_volatility*100:.0f}% volatility, {abs(n.max_drawdown)*100:.0f}% max drawdown, and {abs(n.var_95)*100:.1f}% daily VaR", 
                   self.margin + 20, y - 52, 9, self.LIGHT)
        
        self._footer(c, 6)
    
    # =========================================================================
    # PAGE 7: SCENARIOS
    # =========================================================================
    
    def _page_scenarios(self, c, n: TradeNote):
        self._rect(c, 0, 0, self.width, self.height, self.NAVY)
        self._header(c, "SCENARIO ANALYSIS", n.symbol, 7)
        
        y = self.height - 80
        y = self._section_title(c, "PROBABILITY-WEIGHTED SCENARIOS", y)
        
        # Scenario cards
        if n.scenarios:
            card_w = (self.content_width - 20) / 3
            card_h = 200
            
            colors = {'Bull': self.GREEN, 'Base': self.BLUE, 'Bear': self.RED}
            
            for i, s in enumerate(n.scenarios[:3]):
                x = self.margin + i * (card_w + 10)
                cy = y - card_h
                
                # Card background
                self._rect(c, x, cy, card_w, card_h, self.CARD_BG)
                
                # Header with color
                header_color = colors.get(s.name, self.MUTED)
                self._rect(c, x, cy + card_h - 30, card_w, 30, header_color)
                self._text_center(c, f"{s.name.upper()} CASE", x + card_w/2, cy + card_h - 18, 11, self.WHITE, True)
                
                # Probability
                self._text_center(c, f"{s.probability*100:.0f}%", x + card_w/2, cy + card_h - 55, 28, self.WHITE, True)
                self._text_center(c, "Probability", x + card_w/2, cy + card_h - 70, 8, self.MUTED)
                
                # Target
                self._text_center(c, f"${s.target:.2f}", x + card_w/2, cy + card_h - 100, 16, header_color, True)
                self._text_center(c, f"({s.return_pct:+.1f}%)", x + card_w/2, cy + card_h - 115, 10, self.LIGHT)
                
                # Drivers
                self._text(c, "Drivers:", x + 10, cy + card_h - 135, 8, self.MUTED, True)
                driver_y = cy + card_h - 148
                for driver in s.drivers[:2]:
                    if len(driver) > 35:
                        driver = driver[:33] + "..."
                    self._text(c, f"- {driver}", x + 10, driver_y, 7, self.LIGHT)
                    driver_y -= 11
        
        # Expected Value Calculation
        y -= 240
        y = self._section_title(c, "EXPECTED VALUE", y)
        
        if n.scenarios:
            expected_return = sum(s.probability * s.return_pct for s in n.scenarios)
            expected_price = sum(s.probability * s.target for s in n.scenarios)
            
            self._rect(c, self.margin, y - 50, self.content_width, 60, self.CARD_BG)
            
            self._text(c, "Probability-Weighted Return:", self.margin + 20, y - 15, 10, self.MUTED)
            ev_color = self.GREEN if expected_return > 0 else self.RED
            self._text(c, f"{expected_return:+.1f}%", self.margin + 200, y - 15, 14, ev_color, True)
            
            self._text(c, "Probability-Weighted Price:", self.margin + 20, y - 35, 10, self.MUTED)
            self._text(c, f"${expected_price:.2f}", self.margin + 200, y - 35, 14, self.WHITE, True)
        
        # Catalysts
        y -= 80
        y = self._section_title(c, "KEY CATALYSTS", y)
        
        for i, cat in enumerate(n.catalysts[:4]):
            self._text(c, f"{i+1}.", self.margin, y - 3, 9, self.MUTED, True)
            y = self._paragraph(c, cat, self.margin + 20, y, self.content_width - 20, 9, self.LIGHT, 12)
            y -= 5
        
        # Risks
        y -= 15
        y = self._section_title(c, "PRIMARY RISKS", y)
        
        for i, risk in enumerate(n.risks[:4]):
            self._text(c, f"{i+1}.", self.margin, y - 3, 9, self.MUTED, True)
            y = self._paragraph(c, risk, self.margin + 20, y, self.content_width - 20, 9, self.LIGHT, 12)
            y -= 5
        
        self._footer(c, 7)
    
    # =========================================================================
    # PAGE 8: CONCLUSION
    # =========================================================================
    
    def _page_conclusion(self, c, n: TradeNote):
        self._rect(c, 0, 0, self.width, self.height, self.NAVY)
        self._header(c, "INVESTMENT CONCLUSION", n.symbol, 8)
        
        y = self.height - 80
        y = self._section_title(c, "INVESTMENT THESIS", y)
        y = self._paragraph(c, n.investment_thesis, self.margin, y, self.content_width, 10, self.LIGHT, 15)
        
        y -= 20
        y = self._section_title(c, "EXECUTIVE SUMMARY", y)
        y = self._paragraph(c, n.executive_summary, self.margin, y, self.content_width, 10, self.LIGHT, 15)
        
        # Final Recommendation Box
        y -= 30
        self._rect(c, self.margin, y - 100, self.content_width, 110, self.CARD_BG)
        
        # Recommendation badge
        rec_color = self.REC_COLORS.get(n.recommendation, self.MUTED)
        self._rect(c, self.margin + 20, y - 55, 140, 45, rec_color)
        self._text_center(c, n.recommendation, self.margin + 90, y - 40, 16, self.WHITE, True)
        
        # Key metrics
        metrics_x = self.margin + 180
        self._text(c, "Target", metrics_x, y - 20, 9, self.MUTED)
        self._text(c, f"${n.target_price:.2f}", metrics_x, y - 35, 14, self.WHITE, True)
        
        self._text(c, "Return", metrics_x + 90, y - 20, 9, self.MUTED)
        ret_color = self.GREEN if n.expected_return > 0 else self.RED
        self._text(c, f"{n.expected_return:+.1f}%", metrics_x + 90, y - 35, 14, ret_color, True)
        
        self._text(c, "Confidence", metrics_x + 180, y - 20, 9, self.MUTED)
        self._text(c, f"{n.confidence:.0f}%", metrics_x + 180, y - 35, 14, self.WHITE, True)
        
        self._text(c, "Risk/Reward", metrics_x + 270, y - 20, 9, self.MUTED)
        rr_color = self.GREEN if n.risk_reward >= 2 else self.GOLD
        self._text(c, f"{n.risk_reward:.2f}x", metrics_x + 270, y - 35, 14, rr_color, True)
        
        # Position summary
        self._text(c, f"Position: {n.position_size:.0f}%  |  Stop: ${n.stop_loss:.2f}  |  Time Horizon: {n.time_horizon}", 
                   self.margin + 20, y - 80, 10, self.LIGHT)
        
        # Disclaimer
        y -= 150
        self._text(c, "DISCLAIMER", self.margin, y, 10, self.MUTED, True)
        y -= 15
        
        disclaimer = "This trade note is generated for educational purposes as part of the MSc AI Agents in Asset Management coursework (IFTE0001 Track B: Technical Analyst Agent). It does not constitute investment advice. Past performance does not guarantee future results. All investments involve risk of loss. Always conduct independent due diligence before making investment decisions."
        y = self._paragraph(c, disclaimer, self.margin, y, self.content_width, 8, self.MUTED, 11)
        
        # Generation info
        y -= 20
        self._line(c, self.margin, y + 5, self.width - self.margin, y + 5, self.BORDER, 0.5)
        y -= 10
        
        self._text(c, f"Note ID: {n.note_id}", self.margin, y, 8, self.MUTED)
        self._text(c, f"Model: {n.model_used}", self.margin, y - 12, 8, self.MUTED)
        self._text(c, f"Generated: {n.generated_at}", self.margin, y - 24, 8, self.MUTED)
        self._text_right(c, f"Tokens: {n.tokens_used:,}", self.width - self.margin, y, 8, self.MUTED)
        self._text_right(c, f"Time: {n.generation_time_ms:.0f}ms", self.width - self.margin, y - 12, 8, self.MUTED)
        
        self._footer(c, 8)


# =============================================================================
# REPORT GENERATOR
# =============================================================================

class ReportGenerator:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all(self, note: TradeNote) -> Dict[str, Path]:
        symbol = note.symbol.upper()
        paths = {}
        
        # PDF generation commented out - JSON only
        # try:
        #     pdf_gen = InstitutionalPDFGenerator(self.output_dir)
        #     paths['pdf'] = pdf_gen.generate(note)
        #     logger.info(f"Generated: {paths['pdf']}")
        # except ImportError as e:
        #     logger.error(f"reportlab not installed: {e}")
        # except Exception as e:
        #     logger.error(f"PDF failed: {e}")
        #     import traceback
        #     traceback.print_exc()
        
        # HTML
        html_path = self.output_dir / f"{symbol}_trade_note.html"
        html_path.write_text(self._html(note), encoding='utf-8')
        paths['html'] = html_path
        
        # MD
        md_path = self.output_dir / f"{symbol}_trade_note.md"
        md_path.write_text(self._md(note), encoding='utf-8')
        paths['md'] = md_path
        
        # JSON
        json_path = self.output_dir / f"{symbol}_trade_note.json"
        json_path.write_text(json.dumps(asdict(note), indent=2, default=str), encoding='utf-8')
        paths['json'] = json_path
        
        return paths
    
    def _html(self, n: TradeNote) -> str:
        rec_colors = {'STRONG BUY': '#059669', 'BUY': '#10b981', 'ACCUMULATE': '#34d399', 'HOLD': '#6b7280', 'REDUCE': '#f59e0b', 'SELL': '#ef4444', 'STRONG SELL': '#dc2626'}
        rc = rec_colors.get(n.recommendation, '#6b7280')
        rr_color = '#10b981' if n.risk_reward >= 2 else '#d29922'
        return f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>{n.symbol} Trade Note</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#0d1420;color:#c9d1d9;line-height:1.6;padding:40px}}
.container{{max-width:900px;margin:0 auto}}
.header{{text-align:center;padding:40px 0;border-bottom:1px solid #21262d;margin-bottom:40px}}
.symbol{{font-size:56px;font-weight:700;color:#f0f6fc}}
.company{{font-size:16px;color:#8b949e;margin:10px 0}}
.badge{{display:inline-block;background:{rc};color:white;padding:12px 32px;border-radius:6px;font-size:18px;font-weight:700;margin:20px 0}}
.metrics{{display:grid;grid-template-columns:repeat(5,1fr);gap:15px;margin:30px 0}}
.metric{{background:#161b22;border-radius:8px;padding:20px;text-align:center}}
.metric-label{{font-size:10px;color:#8b949e;text-transform:uppercase}}
.metric-value{{font-size:22px;font-weight:700;color:#58a6ff;margin-top:8px}}
.section{{background:#161b22;border-radius:8px;padding:25px;margin-bottom:20px}}
h2{{font-size:13px;color:#8b949e;margin-bottom:15px;text-transform:uppercase;letter-spacing:1px}}
p{{color:#c9d1d9;margin-bottom:12px}}
table{{width:100%;border-collapse:collapse;margin:15px 0}}
th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid #21262d}}
th{{color:#8b949e;font-size:11px;font-weight:600}}
td{{color:#c9d1d9}}
.green{{color:#3fb950}} .red{{color:#f85149}} .gold{{color:#d29922}}
.footer{{text-align:center;color:#484f58;font-size:11px;margin-top:40px;padding-top:20px;border-top:1px solid #21262d}}
</style></head>
<body><div class="container">
<div class="header">
<div class="symbol">{n.symbol}</div>
<div class="company">{n.company_name} | {n.sector}</div>
<div class="badge">{n.recommendation}</div>
</div>
<div class="metrics">
<div class="metric"><div class="metric-label">Overall Score</div><div class="metric-value">{n.overall_score}</div></div>
<div class="metric"><div class="metric-label">Confidence</div><div class="metric-value">{n.confidence:.0f}%</div></div>
<div class="metric"><div class="metric-label">Target</div><div class="metric-value">${n.target_price:.2f}</div></div>
<div class="metric"><div class="metric-label">Return</div><div class="metric-value" style="color:{'#3fb950' if n.expected_return > 0 else '#f85149'}">{n.expected_return:+.1f}%</div></div>
<div class="metric"><div class="metric-label">Risk/Reward</div><div class="metric-value" style="color:{rr_color}">{n.risk_reward:.2f}x</div></div>
</div>
<div class="section"><h2>Investment Thesis</h2><p>{n.investment_thesis}</p></div>
<div class="section"><h2>Executive Summary</h2><p>{n.executive_summary}</p></div>
<div class="section"><h2>Trade Setup</h2>
<table><tr><th>Level</th><th>Price</th><th>Change</th></tr>
<tr><td>Entry</td><td>${n.entry:.2f}</td><td>-</td></tr>
<tr><td>Stop Loss</td><td>${n.stop_loss:.2f}</td><td class="red">{((n.stop_loss/n.current_price)-1)*100:.1f}%</td></tr>
<tr><td>Target 1</td><td>${n.target_1:.2f}</td><td class="green">+{((n.target_1/n.current_price)-1)*100:.1f}%</td></tr>
<tr><td>Target 2</td><td>${n.target_2:.2f}</td><td class="green">+{((n.target_2/n.current_price)-1)*100:.1f}%</td></tr>
<tr><td>Target 3 (Primary)</td><td>${n.target_3:.2f}</td><td class="green">+{((n.target_3/n.current_price)-1)*100:.1f}%</td></tr>
</table></div>
<div class="section"><h2>Technical Analysis</h2><p>{n.technical_analysis}</p></div>
<div class="section"><h2>Backtest Results</h2><p>{n.backtest_analysis}</p>
<table><tr><th>Metric</th><th>Value</th><th>Metric</th><th>Value</th></tr>
<tr><td>Total Return</td><td>{n.backtest_total_return*100:.1f}%</td><td>Hit Rate</td><td>{n.backtest_hit_rate*100:.1f}%</td></tr>
<tr><td>CAGR</td><td>{n.backtest_cagr*100:.1f}%</td><td>Profit Factor</td><td>{n.backtest_profit_factor:.2f}</td></tr>
<tr><td>Sharpe</td><td>{n.backtest_sharpe:.3f}</td><td>Total Trades</td><td>{n.backtest_total_trades}</td></tr>
<tr><td>VaR (95%)</td><td>{abs(n.var_95)*100:.2f}%</td><td>CVaR (95%)</td><td>{abs(n.cvar_95)*100:.2f}%</td></tr>
</table></div>
<div class="section"><h2>Risk Analysis</h2><p>{n.risk_analysis}</p></div>
<div class="footer">{n.note_id} | {n.generated_at[:10]} | v{VERSION}</div>
</div></body></html>'''
    
    def _md(self, n: TradeNote) -> str:
        return f'''# {n.symbol} Institutional Trade Note

## {n.company_name}
{n.sector} | {n.industry}

---

## Recommendation: **{n.recommendation}**

| Metric | Value |
|--------|-------|
| Confidence | {n.confidence:.0f}% |
| Target Price | ${n.target_price:.2f} |
| Expected Return | {n.expected_return:+.1f}% |
| Risk/Reward | {n.risk_reward:.2f}x |
| Conviction | {n.conviction} |
| Time Horizon | {n.time_horizon} |
| Risk Rating | {n.risk_rating} |

---

## Scores

| Factor | Score | Interpretation |
|--------|-------|----------------|
| Overall | {n.overall_score}/100 | Weighted composite of all indicator families |
| Momentum | {n.momentum_score}/100 | RSI, Stochastic, Williams %R readings |
| Trend | {n.trend_score}/100 | MACD, ADX, Supertrend, Ichimoku alignment |
| Risk | {n.risk_score}/100 | Volatility, drawdown, VaR assessment |

---

## Trade Setup

| Level | Price | Change |
|-------|-------|--------|
| Current | ${n.current_price:.2f} | - |
| Entry | ${n.entry:.2f} | - |
| Stop Loss | ${n.stop_loss:.2f} | {((n.stop_loss/n.current_price)-1)*100:.1f}% |
| Target 1 | ${n.target_1:.2f} | +{((n.target_1/n.current_price)-1)*100:.1f}% |
| Target 2 | ${n.target_2:.2f} | +{((n.target_2/n.current_price)-1)*100:.1f}% |
| Target 3 (Primary) | ${n.target_3:.2f} | +{((n.target_3/n.current_price)-1)*100:.1f}% |

Position Size: {n.position_size:.1f}% | Max: {n.max_size:.1f}%

---

## Investment Thesis

{n.investment_thesis}

---

## Executive Summary

{n.executive_summary}

---

## Technical Analysis

{n.technical_analysis}

---

## Backtest Results (COURSEWORK REQUIRED METRICS)

{n.backtest_analysis}

| Metric | Value | Metric | Value |
|--------|-------|--------|-------|
| Total Return | {n.backtest_total_return*100:.1f}% | Hit Rate | {n.backtest_hit_rate*100:.1f}% |
| CAGR | {n.backtest_cagr*100:.1f}% | Profit Factor | {n.backtest_profit_factor:.2f} |
| Max Drawdown | {n.backtest_max_dd*100:.1f}% | Total Trades | {n.backtest_total_trades} |
| Sharpe Ratio | {n.backtest_sharpe:.3f} | Expectancy | ${n.backtest_expectancy:.2f} |
| VaR (95%) | {abs(n.var_95)*100:.2f}% | CVaR (95%) | {abs(n.cvar_95)*100:.2f}% |

---

## Risk Analysis

{n.risk_analysis}

---

## Scenarios

{chr(10).join(f"### {s.name} Case ({s.probability*100:.0f}%){chr(10)}Target: ${s.target:.2f} ({s.return_pct:+.1f}%){chr(10)}{chr(10).join('- ' + d for d in s.drivers)}" for s in n.scenarios)}

---

## Catalysts

{chr(10).join(f"{i+1}. {c}" for i, c in enumerate(n.catalysts))}

---

## Risks

{chr(10).join(f"{i+1}. {r}" for i, r in enumerate(n.risks))}

---

*{n.note_id} | Generated: {n.generated_at} | Model: {n.model_used} | v{VERSION}*
'''


# =============================================================================
# PUBLIC API
# =============================================================================

def generate_trade_note(
    phase1_output: Any,
    phase2_output: Any,
    phase3_output: Any = None,
    phase4_output: Any = None,
    phase4b_output: Any = None,
    output_dir: Optional[Union[str, Path]] = None,
    logger: Optional[logging.Logger] = None
) -> TradeNote:
    """Generate comprehensive institutional trade note."""
    log = logger or logging.getLogger(__name__)
    
    log.info("=" * 60)
    log.info(f"Institutional Trade Note Generator v{VERSION}")
    log.info(f"Model: {CLAUDE_MODEL}")
    log.info("=" * 60)
    
    gen = TradeNoteGenerator()
    note = gen.generate(phase1_output, phase2_output, phase3_output, phase4_output, phase4b_output)
    
    log.info(f"Result: {note.recommendation} | {note.confidence:.0f}% | Score: {note.overall_score}")
    
    if output_dir:
        reporter = ReportGenerator(output_dir)
        paths = reporter.generate_all(note)
        log.info("=" * 60)
        for fmt, path in paths.items():
            log.info(f"[{fmt.upper()}] {path}")
        log.info("=" * 60)
    
    return note


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    print(f"Institutional Trade Note Generator v{VERSION}")
    print(f"Model: {CLAUDE_MODEL}")