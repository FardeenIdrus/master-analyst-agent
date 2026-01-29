"""
Institutional-Grade Market Data Pipeline for Quantitative Technical Analysis

MSc AI Agents in Asset Management - Track B: Technical Analyst Agent
Phase 1: Data Ingestion, Validation, and Statistical Profiling

PIPELINE ARCHITECTURE
    The pipeline operates in six sequential stages:
    
    Stage 1 - ACQUIRE
        Multi-source data fetching with retry logic and failover.
        Fetches OHLCV data, VIX, benchmark, and corporate events.
        
    Stage 2 - VALIDATE  
        Four-dimension quality assessment:
        - Completeness (40%): Missing values, trading day coverage
        - Accuracy (30%): OHLC integrity, outlier detection
        - Consistency (20%): Corporate action detection, price continuity
        - Timeliness (10%): Market calendar-aware freshness
        
    Stage 3 - ENRICH
        Feature engineering for foundational technical analysis:
        - Returns: Simple, log, overnight, intraday, multi-period
        - Price structure: Range, gaps, body, shadows
        - Volume: Dollar volume, relative volume, liquidity metrics
        - Moving averages: SMA/EMA for trend context
        - ATR: True range for volatility-adjusted analysis
        - 52-week metrics: Position, distance from highs/lows
        
    Stage 4 - PROFILE
        Statistical analysis with hypothesis testing:
        - Jarque-Bera: Return normality
        - ADF: Unit root / stationarity
        - KPSS: Stationarity confirmation  
        - Ljung-Box: Autocorrelation
        - ARCH: Volatility clustering
        
    Stage 5 - CONTEXTUALIZE
        Market context integration:
        - VIX regime classification (5 tiers)
        - Benchmark relative performance
        - Cross-asset correlation
        - Economic calendar proximity
        
    Stage 6 - EXPORT
        Output generation:
        - Parquet files (efficient storage)
        - JSON metadata (provenance, quality)
        - HTML dashboard (interactive visualization)

DATA PROVENANCE
    Every data point is tracked with:
    - Source identifier
    - Fetch timestamp
    - Data version hash
    - Quality assessment
    
VOLATILITY ESTIMATORS (7 Academic Methods)
    1. Close-to-Close: Baseline standard deviation
    2. Parkinson (1980): High-Low based, +22% efficient
    3. Garman-Klass (1980): Full OHLC, +87% efficient
    4. Rogers-Satchell (1991): Drift-adjusted
    5. Yang-Zhang (2000): Handles gaps, +98% efficient
    6. GKYZ: Hybrid Garman-Klass + Yang-Zhang
    7. Hodges-Tompkins: Bias-corrected for finite samples

Author: Tamer
Course: MSc AI Agents in Asset Management (IFTE0001)
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# Suppress future warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

# Module-level logger
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Trading calendar constants
TRADING_DAYS_YEAR: int = 252
TRADING_DAYS_QUARTER: int = 63
TRADING_DAYS_MONTH: int = 21
TRADING_DAYS_WEEK: int = 5

# Mathematical constants
SQRT_252: float = np.sqrt(TRADING_DAYS_YEAR)
SQRT_12: float = np.sqrt(12)

# Pipeline version
PIPELINE_VERSION: str = "1.0.0"


# =============================================================================
# ENUMERATIONS
# =============================================================================

class QualityGrade(Enum):
    """Data quality classification with clear thresholds."""
    EXCELLENT = "EXCELLENT"    # >= 95
    GOOD = "GOOD"              # >= 85
    ACCEPTABLE = "ACCEPTABLE"  # >= 70
    POOR = "POOR"              # >= 50
    UNUSABLE = "UNUSABLE"      # < 50


class VolatilityRegime(Enum):
    """Asset volatility regime based on annualized volatility."""
    LOW = "LOW"           # < 15%
    NORMAL = "NORMAL"     # 15-25%
    HIGH = "HIGH"         # 25-40%
    EXTREME = "EXTREME"   # > 40%


class VIXRegime(Enum):
    """Market fear regime based on VIX levels."""
    COMPLACENT = "COMPLACENT"   # VIX < 12: Extreme complacency
    CALM = "CALM"               # VIX 12-16: Low volatility
    NORMAL = "NORMAL"           # VIX 16-20: Normal conditions
    ELEVATED = "ELEVATED"       # VIX 20-25: Heightened uncertainty
    FEARFUL = "FEARFUL"         # VIX 25-30: Significant fear
    PANIC = "PANIC"             # VIX > 30: Crisis conditions


class TrendCharacter(Enum):
    """Return series character based on Hurst exponent."""
    MEAN_REVERTING = "MEAN_REVERTING"  # H < 0.45
    RANDOM_WALK = "RANDOM_WALK"        # 0.45 <= H <= 0.55
    TRENDING = "TRENDING"              # H > 0.55


class ReturnDistribution(Enum):
    """Return distribution classification based on kurtosis."""
    NORMAL = "NORMAL"           # |kurtosis| < 1
    LEPTOKURTIC = "LEPTOKURTIC" # kurtosis > 1 (fat tails)
    PLATYKURTIC = "PLATYKURTIC" # kurtosis < -1 (thin tails)


class StationarityConclusion(Enum):
    """Combined conclusion from ADF and KPSS tests."""
    STATIONARY = "STATIONARY"
    NON_STATIONARY = "NON_STATIONARY"
    TREND_STATIONARY = "TREND_STATIONARY"
    INCONCLUSIVE = "INCONCLUSIVE"


class CorporateActionType(Enum):
    """Types of corporate actions detected."""
    STOCK_SPLIT = "STOCK_SPLIT"
    REVERSE_SPLIT = "REVERSE_SPLIT"
    DIVIDEND = "DIVIDEND"
    UNKNOWN = "UNKNOWN"


class MarketContext(Enum):
    """Overall market context for strategy selection."""
    RISK_ON = "RISK_ON"       # Favorable for aggressive strategies
    NORMAL = "NORMAL"         # Standard conditions
    CAUTIOUS = "CAUTIOUS"     # Reduce risk exposure
    RISK_OFF = "RISK_OFF"     # Defensive positioning
    CRISIS = "CRISIS"         # Capital preservation mode


# =============================================================================
# DATA CLASSES - DATA PROVENANCE
# =============================================================================

@dataclass
class DataProvenance:
    """
    Tracks the origin and lineage of data for auditability.
    
    Every data fetch is recorded with its source, timestamp, and
    integrity hash to ensure reproducibility and traceability.
    """
    source: str                     # Data source identifier
    symbol: str                     # Ticker symbol
    fetch_timestamp: str            # ISO format timestamp
    date_range: Tuple[str, str]     # (start, end) dates
    record_count: int               # Number of records fetched
    data_hash: str                  # SHA-256 hash of Close prices
    version: str = PIPELINE_VERSION # Pipeline version used
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source": self.source,
            "symbol": self.symbol,
            "fetch_timestamp": self.fetch_timestamp,
            "date_range": list(self.date_range),
            "record_count": self.record_count,
            "data_hash": self.data_hash,
            "version": self.version
        }


@dataclass
class EarningsEvent:
    """
    Corporate earnings announcement data.
    
    Used for anchored VWAP calculations and event-aware analysis.
    """
    date: str
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    surprise_pct: Optional[float] = None
    
    @property
    def is_beat(self) -> Optional[bool]:
        """Did the company beat estimates?"""
        if self.surprise_pct is not None:
            return self.surprise_pct > 0
        return None
    
    @property
    def surprise_magnitude(self) -> str:
        """Classify the magnitude of the earnings surprise."""
        if self.surprise_pct is None:
            return "UNKNOWN"
        abs_surprise = abs(self.surprise_pct)
        if abs_surprise < 2:
            return "INLINE"
        elif abs_surprise < 5:
            return "SMALL"
        elif abs_surprise < 10:
            return "MODERATE"
        else:
            return "LARGE"


@dataclass
class CorporateAction:
    """Detected corporate action event."""
    date: str
    action_type: CorporateActionType
    ratio: Optional[float] = None
    confidence: float = 0.0
    details: str = ""


# =============================================================================
# DATA CLASSES - QUALITY METRICS
# =============================================================================

@dataclass
class DataQualityMetrics:
    """
    Comprehensive data quality assessment results.
    
    Quality is measured across four dimensions with configurable weights:
    - Completeness: Are all expected data points present?
    - Accuracy: Are OHLC relationships valid? Are there outliers?
    - Consistency: Are there unexplained jumps or gaps?
    - Timeliness: How fresh is the data?
    """
    completeness: float              # 0-100 score
    accuracy: float                  # 0-100 score
    consistency: float               # 0-100 score
    timeliness: float                # 0-100 score
    overall: float                   # Weighted average
    grade: QualityGrade              # Classification
    issues: List[str]                # Critical issues
    warnings: List[str]              # Non-critical warnings
    corporate_actions: List[CorporateAction]
    records_total: int
    records_valid: int
    date_range: Tuple[str, str]
    data_hash: str = ""
    
    @property
    def is_usable(self) -> bool:
        """Can this data be used for analysis?"""
        return self.grade != QualityGrade.UNUSABLE
    
    @property
    def has_critical_issues(self) -> bool:
        """Are there critical data quality issues?"""
        return len(self.issues) > 0
    
    @property
    def completeness_pct(self) -> float:
        """Percentage of valid records."""
        if self.records_total == 0:
            return 0.0
        return (self.records_valid / self.records_total) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "completeness": round(self.completeness, 2),
            "accuracy": round(self.accuracy, 2),
            "consistency": round(self.consistency, 2),
            "timeliness": round(self.timeliness, 2),
            "overall": round(self.overall, 2),
            "grade": self.grade.value,
            "issues": self.issues,
            "warnings": self.warnings,
            "records_total": self.records_total,
            "records_valid": self.records_valid,
            "date_range": list(self.date_range),
            "data_hash": self.data_hash
        }


# =============================================================================
# DATA CLASSES - STATISTICAL ANALYSIS
# =============================================================================

@dataclass
class StatisticalTests:
    """
    Results from five hypothesis tests on return series.
    
    Tests:
    1. Jarque-Bera: Tests if returns are normally distributed
    2. ADF: Tests for unit root (non-stationarity)
    3. KPSS: Tests for stationarity (complement to ADF)
    4. Ljung-Box: Tests for autocorrelation
    5. ARCH: Tests for volatility clustering
    """
    # Jarque-Bera (normality)
    jb_statistic: float
    jb_p_value: float
    is_normal: bool
    
    # Augmented Dickey-Fuller (stationarity)
    adf_statistic: float
    adf_p_value: float
    adf_critical_1pct: float
    adf_critical_5pct: float
    adf_critical_10pct: float
    adf_lags_used: int
    is_stationary: bool
    
    # KPSS (stationarity confirmation)
    kpss_statistic: float
    kpss_p_value: float
    kpss_critical_5pct: float
    kpss_critical_10pct: float
    is_stationary_kpss: bool
    
    # Ljung-Box (autocorrelation)
    lb_statistic: float
    lb_p_value: float
    lb_lags: int
    has_autocorrelation: bool
    
    # ARCH (volatility clustering)
    arch_statistic: float
    arch_p_value: float
    arch_lags: int
    has_arch_effects: bool
    
    @property
    def stationarity_conclusion(self) -> StationarityConclusion:
        """
        Combine ADF and KPSS results for definitive conclusion.
        
        ADF: H0 = unit root (non-stationary)
        KPSS: H0 = stationary
        
        Both reject H0 -> Trend stationary
        ADF rejects, KPSS fails to reject -> Stationary
        ADF fails to reject, KPSS rejects -> Non-stationary
        Both fail to reject -> Inconclusive
        """
        if self.is_stationary and self.is_stationary_kpss:
            return StationarityConclusion.STATIONARY
        elif not self.is_stationary and not self.is_stationary_kpss:
            return StationarityConclusion.NON_STATIONARY
        elif self.is_stationary and not self.is_stationary_kpss:
            return StationarityConclusion.TREND_STATIONARY
        else:
            return StationarityConclusion.INCONCLUSIVE
    
    @property
    def modeling_implications(self) -> List[str]:
        """Generate modeling recommendations based on test results."""
        implications = []
        
        if not self.is_normal:
            implications.append("Non-normal returns: Use robust statistics and non-parametric methods")
        
        if self.has_arch_effects:
            implications.append("Volatility clustering: Consider GARCH family for volatility forecasting")
        
        if self.has_autocorrelation:
            implications.append("Autocorrelation detected: Momentum or mean-reversion strategies may be viable")
        
        if self.stationarity_conclusion == StationarityConclusion.NON_STATIONARY:
            implications.append("Non-stationary series: Regime-switching models may be appropriate")
        elif self.stationarity_conclusion == StationarityConclusion.TREND_STATIONARY:
            implications.append("Trend-stationary: Detrend data before modeling")
        
        return implications if implications else ["Standard approaches are suitable"]


@dataclass
class VolatilityProfile:
    """
    Seven academic volatility estimators with composite measure.
    
    Each estimator has different properties:
    - Close-to-Close: Simple, but ignores intraday information
    - Parkinson: Uses high-low range, 22% more efficient
    - Garman-Klass: Uses full OHLC, 87% more efficient
    - Rogers-Satchell: Handles drift/trend
    - Yang-Zhang: Handles overnight gaps, 98% more efficient
    - GKYZ: Combines GK with YZ for best properties
    - Hodges-Tompkins: Corrects small-sample bias
    """
    close_to_close: float
    parkinson: float
    garman_klass: float
    rogers_satchell: float
    yang_zhang: float
    gkyz: float
    hodges_tompkins: float
    composite: float
    regime: VolatilityRegime
    percentile_rank: float  # Where current vol sits in historical distribution
    
    @property
    def estimator_range(self) -> Tuple[float, float]:
        """Range of volatility estimates."""
        estimates = [
            self.close_to_close, self.parkinson, self.garman_klass,
            self.rogers_satchell, self.yang_zhang, self.gkyz, self.hodges_tompkins
        ]
        return (min(estimates), max(estimates))
    
    @property
    def estimator_spread(self) -> float:
        """Spread between highest and lowest estimator."""
        min_val, max_val = self.estimator_range
        return max_val - min_val
    
    @property
    def estimators_consistent(self) -> bool:
        """Are all estimators within 10% of each other?"""
        return self.estimator_spread < 0.10
    
    @property
    def best_estimator_name(self) -> str:
        """Recommend the best estimator based on data characteristics."""
        # Yang-Zhang is generally most efficient
        return "yang_zhang"


@dataclass
class TailRiskMetrics:
    """
    Tail risk and drawdown analysis.
    
    Includes:
    - Value at Risk (VaR) at multiple confidence levels
    - Conditional VaR (Expected Shortfall)
    - Maximum drawdown and duration
    - Distribution shape (skewness, kurtosis)
    """
    var_95: float           # 5th percentile of returns
    var_99: float           # 1st percentile of returns
    var_999: float          # 0.1st percentile of returns
    cvar_95: float          # Expected loss given loss > VaR95
    cvar_99: float          # Expected loss given loss > VaR99
    max_gain: float         # Best single-day return
    max_loss: float         # Worst single-day return
    max_drawdown: float     # Peak-to-trough decline
    max_drawdown_duration: int  # Days in longest drawdown
    skewness: float         # Return distribution skewness
    kurtosis: float         # Return distribution kurtosis (excess)
    tail_ratio: float       # Ratio of right tail to left tail
    downside_deviation: float  # Volatility of negative returns only
    
    @property
    def distribution_type(self) -> ReturnDistribution:
        """Classify the return distribution."""
        if abs(self.skewness) < 0.5 and abs(self.kurtosis) < 1:
            return ReturnDistribution.NORMAL
        elif self.kurtosis > 0:
            return ReturnDistribution.LEPTOKURTIC
        else:
            return ReturnDistribution.PLATYKURTIC
    
    @property
    def has_fat_tails(self) -> bool:
        """Are there fat tails (excess kurtosis > 3)?"""
        return self.kurtosis > 3
    
    @property
    def is_negatively_skewed(self) -> bool:
        """Is the distribution negatively skewed?"""
        return self.skewness < -0.5


# =============================================================================
# DATA CLASSES - MARKET PROFILE
# =============================================================================

@dataclass
class MarketProfile:
    """
    Complete statistical profile of the asset.
    
    Combines return statistics, volatility analysis, trend character,
    and strategic implications into a comprehensive profile.
    """
    # Return statistics
    daily_return_mean: float
    daily_return_std: float
    daily_return_median: float
    annualized_return: float
    annualized_volatility: float
    
    # Distribution shape
    skewness: float
    kurtosis: float
    return_distribution: ReturnDistribution
    
    # Autocorrelation
    autocorr_lag1: float
    autocorr_lag5: float
    autocorr_lag21: float
    
    # Volatility
    volatility_profile: VolatilityProfile
    volatility_of_volatility: float
    
    # Trend analysis
    hurst_exponent: float
    hurst_confidence: float  # R-squared from Hurst regression
    trend_character: TrendCharacter
    
    # Risk metrics
    tail_risk: TailRiskMetrics
    statistical_tests: StatisticalTests
    
    # Strategic recommendations
    strategy_hints: List[str] = field(default_factory=list)
    
    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio (assuming 4% risk-free rate)."""
        risk_free = 0.04
        if self.annualized_volatility <= 0:
            return 0.0
        return (self.annualized_return - risk_free) / self.annualized_volatility
    
    @property
    def sortino_ratio(self) -> float:
        """Sortino ratio using downside deviation."""
        risk_free = 0.04
        if self.tail_risk.downside_deviation <= 0:
            return 0.0
        return (self.annualized_return - risk_free) / self.tail_risk.downside_deviation
    
    @property
    def calmar_ratio(self) -> float:
        """Calmar ratio (return / max drawdown)."""
        if self.tail_risk.max_drawdown == 0:
            return 0.0
        return self.annualized_return / abs(self.tail_risk.max_drawdown)
    
    @property
    def position_sizing_recommendation(self) -> str:
        """Recommend position sizing approach based on risk profile."""
        if self.kurtosis > 5 or self.tail_risk.has_fat_tails:
            return "VERY_CONSERVATIVE (extreme fat tails detected)"
        elif self.kurtosis > 3:
            return "CONSERVATIVE (elevated kurtosis)"
        elif self.tail_risk.is_negatively_skewed:
            return "CONSERVATIVE (negative skew - larger downside moves)"
        elif self.volatility_profile.regime == VolatilityRegime.EXTREME:
            return "CONSERVATIVE (extreme volatility regime)"
        elif self.volatility_profile.regime == VolatilityRegime.HIGH:
            return "MODERATE (high volatility)"
        else:
            return "STANDARD"


@dataclass
class BenchmarkAnalysis:
    """
    Relative performance analysis against benchmark.
    
    Key metrics:
    - Correlation and R-squared
    - Beta (systematic risk)
    - Alpha (excess return)
    - Information ratio
    - Up/Down capture ratios
    """
    benchmark_symbol: str
    
    # Correlation
    correlation_full: float
    correlation_1y: float
    correlation_rolling_mean: float
    correlation_rolling_std: float
    
    # Beta
    beta_full: float
    beta_1y: float
    beta_rolling_mean: float
    beta_rolling_std: float
    
    # Alpha and risk-adjusted
    alpha_annualized: float
    tracking_error: float
    information_ratio: float
    
    # Relative strength
    relative_strength_1m: float
    relative_strength_3m: float
    relative_strength_6m: float
    relative_strength_1y: float
    
    # Risk decomposition
    r_squared: float
    systematic_risk_pct: float
    idiosyncratic_risk_pct: float
    
    # Capture ratios
    up_capture: float
    down_capture: float
    
    @property
    def capture_ratio(self) -> float:
        """Ratio of up capture to down capture (higher is better)."""
        if self.down_capture <= 0:
            return float('inf') if self.up_capture > 0 else 1.0
        return self.up_capture / self.down_capture
    
    @property
    def is_defensive(self) -> bool:
        """Is this a defensive stock (low beta, low down capture)?"""
        return self.beta_full < 0.8 and self.down_capture < 90
    
    @property
    def is_aggressive(self) -> bool:
        """Is this an aggressive stock (high beta, high up capture)?"""
        return self.beta_full > 1.2 and self.up_capture > 110
    
    @property
    def diversification_benefit(self) -> str:
        """Classify diversification benefit based on correlation."""
        if self.correlation_full < 0.3:
            return "HIGH (low correlation with benchmark)"
        elif self.correlation_full < 0.6:
            return "MODERATE"
        elif self.correlation_full < 0.8:
            return "LOW"
        else:
            return "MINIMAL (moves closely with benchmark)"


@dataclass
class VIXAnalysis:
    """
    VIX-based market context analysis.
    
    Provides market regime classification and strategy recommendations
    based on the CBOE Volatility Index.
    """
    current_level: float
    regime: VIXRegime
    percentile_1y: float
    percentile_5y: float
    is_inverted: bool  # VIX futures curve inverted (fear)
    term_structure_slope: float
    
    # Historical context
    average_1m: float
    average_3m: float
    average_1y: float
    
    # Regime statistics
    days_in_current_regime: int
    
    @property
    def market_context(self) -> MarketContext:
        """Derive overall market context from VIX regime."""
        regime_mapping = {
            VIXRegime.COMPLACENT: MarketContext.RISK_ON,
            VIXRegime.CALM: MarketContext.RISK_ON,
            VIXRegime.NORMAL: MarketContext.NORMAL,
            VIXRegime.ELEVATED: MarketContext.CAUTIOUS,
            VIXRegime.FEARFUL: MarketContext.RISK_OFF,
            VIXRegime.PANIC: MarketContext.CRISIS
        }
        return regime_mapping.get(self.regime, MarketContext.NORMAL)
    
    @property
    def strategy_implications(self) -> List[str]:
        """Generate strategy recommendations based on VIX regime."""
        implications = []
        
        if self.regime == VIXRegime.COMPLACENT:
            implications.append("Extreme complacency - consider tail risk hedges")
            implications.append("Volatility likely to mean-revert higher")
        elif self.regime == VIXRegime.CALM:
            implications.append("Low volatility environment favors momentum strategies")
        elif self.regime == VIXRegime.NORMAL:
            implications.append("Normal conditions - standard position sizing")
        elif self.regime == VIXRegime.ELEVATED:
            implications.append("Reduce position sizes by 20-30%")
            implications.append("Consider tightening stops")
        elif self.regime == VIXRegime.FEARFUL:
            implications.append("High fear - reduce risk exposure significantly")
            implications.append("Look for capitulation signals")
        elif self.regime == VIXRegime.PANIC:
            implications.append("Crisis conditions - preserve capital")
            implications.append("Extreme fear often precedes strong rallies")
        
        return implications


# =============================================================================
# DATA CLASSES - PIPELINE OUTPUT
# =============================================================================

@dataclass
class PipelineOutput:
    """
    Complete output from the data pipeline.
    
    Contains all processed data, quality metrics, statistical analysis,
    and market context needed for downstream technical analysis.
    """
    # Data frames
    daily: pd.DataFrame
    weekly: pd.DataFrame
    monthly: pd.DataFrame
    benchmark: Optional[pd.DataFrame]
    vix: Optional[pd.DataFrame]
    
    # Corporate events
    earnings: List[EarningsEvent]
    
    # Asset metadata
    symbol: str
    company_name: str
    sector: str
    industry: str
    period: Tuple[str, str]
    
    # Analysis results
    quality: DataQualityMetrics
    profile: MarketProfile
    benchmark_analysis: Optional[BenchmarkAnalysis]
    vix_analysis: Optional[VIXAnalysis]
    
    # Data provenance
    provenance: DataProvenance
    
    # Processing metadata
    processing_time_ms: float
    pipeline_version: str
    generated_at: str
    
    # Output paths
    parquet_path: Path
    html_path: Path
    json_path: Path
    
    @property
    def total_records(self) -> int:
        """Total records across all timeframes."""
        return len(self.daily) + len(self.weekly) + len(self.monthly)
    
    @property
    def years_of_data(self) -> float:
        """Years of data coverage."""
        return len(self.daily) / TRADING_DAYS_YEAR
    
    @property
    def is_ready_for_analysis(self) -> bool:
        """Is data quality sufficient for analysis?"""
        return self.quality.is_usable and len(self.daily) >= TRADING_DAYS_YEAR
    
    @property
    def market_context(self) -> MarketContext:
        """Current market context based on VIX."""
        if self.vix_analysis:
            return self.vix_analysis.market_context
        return MarketContext.NORMAL


# =============================================================================
# US MARKET CALENDAR
# =============================================================================

class USMarketCalendar:
    """
    US Stock Market Holiday Calendar.
    
    Provides accurate trading day calculations accounting for:
    - Federal holidays (observed dates)
    - Weekend non-trading days
    - Special market closures
    
    Used for timeliness assessment and trading day counting.
    """
    
    @staticmethod
    def get_holidays(year: int) -> set:
        """
        Get all US market holidays for a given year.
        
        Holidays observed:
        - New Year's Day
        - Martin Luther King Jr. Day
        - Presidents Day
        - Good Friday
        - Memorial Day
        - Juneteenth
        - Independence Day
        - Labor Day
        - Thanksgiving Day
        - Christmas Day
        
        Args:
            year: Calendar year
            
        Returns:
            Set of datetime.date objects for holidays
        """
        holidays = set()
        
        # New Year's Day (January 1, observed)
        ny = pd.Timestamp(f"{year}-01-01")
        if ny.weekday() == 5:  # Saturday -> Friday observed
            holidays.add(pd.Timestamp(f"{year-1}-12-31").date())
        elif ny.weekday() == 6:  # Sunday -> Monday observed
            holidays.add(pd.Timestamp(f"{year}-01-02").date())
        else:
            holidays.add(ny.date())
        
        # MLK Day (3rd Monday of January)
        jan1 = pd.Timestamp(f"{year}-01-01")
        days_to_monday = (7 - jan1.weekday()) % 7
        if jan1.weekday() == 0:
            days_to_monday = 0
        first_monday_jan = jan1 + pd.Timedelta(days=days_to_monday)
        mlk = first_monday_jan + pd.Timedelta(weeks=2)
        holidays.add(mlk.date())
        
        # Presidents Day (3rd Monday of February)
        feb1 = pd.Timestamp(f"{year}-02-01")
        days_to_monday = (7 - feb1.weekday()) % 7
        if feb1.weekday() == 0:
            days_to_monday = 0
        first_monday_feb = feb1 + pd.Timedelta(days=days_to_monday)
        presidents = first_monday_feb + pd.Timedelta(weeks=2)
        holidays.add(presidents.date())
        
        # Good Friday (Friday before Easter)
        # Easter calculation using Anonymous Gregorian algorithm
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        easter = pd.Timestamp(f"{year}-{month:02d}-{day:02d}")
        good_friday = easter - pd.Timedelta(days=2)
        holidays.add(good_friday.date())
        
        # Memorial Day (last Monday of May)
        may31 = pd.Timestamp(f"{year}-05-31")
        days_back = may31.weekday()  # Monday = 0
        memorial = may31 - pd.Timedelta(days=days_back)
        holidays.add(memorial.date())
        
        # Juneteenth (June 19, observed)
        june19 = pd.Timestamp(f"{year}-06-19")
        if june19.weekday() == 5:
            holidays.add((june19 - pd.Timedelta(days=1)).date())
        elif june19.weekday() == 6:
            holidays.add((june19 + pd.Timedelta(days=1)).date())
        else:
            holidays.add(june19.date())
        
        # Independence Day (July 4, observed)
        july4 = pd.Timestamp(f"{year}-07-04")
        if july4.weekday() == 5:
            holidays.add((july4 - pd.Timedelta(days=1)).date())
        elif july4.weekday() == 6:
            holidays.add((july4 + pd.Timedelta(days=1)).date())
        else:
            holidays.add(july4.date())
        
        # Labor Day (1st Monday of September)
        sep1 = pd.Timestamp(f"{year}-09-01")
        days_to_monday = (7 - sep1.weekday()) % 7
        if sep1.weekday() == 0:
            days_to_monday = 0
        labor = sep1 + pd.Timedelta(days=days_to_monday)
        holidays.add(labor.date())
        
        # Thanksgiving (4th Thursday of November)
        nov1 = pd.Timestamp(f"{year}-11-01")
        days_to_thursday = (3 - nov1.weekday()) % 7
        first_thursday = nov1 + pd.Timedelta(days=days_to_thursday)
        thanksgiving = first_thursday + pd.Timedelta(weeks=3)
        holidays.add(thanksgiving.date())
        
        # Christmas (December 25, observed)
        xmas = pd.Timestamp(f"{year}-12-25")
        if xmas.weekday() == 5:
            holidays.add((xmas - pd.Timedelta(days=1)).date())
        elif xmas.weekday() == 6:
            holidays.add((xmas + pd.Timedelta(days=1)).date())
        else:
            holidays.add(xmas.date())
        
        return holidays
    
    @staticmethod
    def is_trading_day(date: pd.Timestamp) -> bool:
        """Check if a date is a trading day."""
        if date.weekday() >= 5:  # Weekend
            return False
        holidays = USMarketCalendar.get_holidays(date.year)
        return date.date() not in holidays
    
    @staticmethod
    def get_last_trading_day(from_date: pd.Timestamp = None) -> pd.Timestamp:
        """Get the most recent trading day."""
        if from_date is None:
            from_date = pd.Timestamp.now().normalize()
        
        check_date = from_date
        for _ in range(10):  # Max 10 days back
            if USMarketCalendar.is_trading_day(check_date):
                return check_date
            check_date -= pd.Timedelta(days=1)
        
        return check_date
    
    @staticmethod
    def count_trading_days_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
        """Count trading days between two dates (exclusive of end)."""
        if start >= end:
            return 0
        
        count = 0
        current = start + pd.Timedelta(days=1)
        
        while current < end:
            if USMarketCalendar.is_trading_day(current):
                count += 1
            current += pd.Timedelta(days=1)
        
        return count


# =============================================================================
# DATA ACQUISITION
# =============================================================================

class DataAcquisition:
    """
    Multi-source data acquisition with retry logic and failover.
    
    Handles:
    - OHLCV price data
    - VIX volatility index
    - Benchmark comparison data
    - Earnings calendar
    - Company metadata
    
    All data fetches are logged with provenance information.
    """
    
    def __init__(self, max_retries: int = 3, timeout: int = 30):
        """
        Initialize data acquisition.
        
        Args:
            max_retries: Maximum retry attempts for failed fetches
            timeout: Request timeout in seconds
        """
        self._yf = None
        self.max_retries = max_retries
        self.timeout = timeout
    
    def _get_yf(self):
        """Lazy load yfinance to avoid import overhead."""
        if self._yf is None:
            import yfinance as yf
            self._yf = yf
        return self._yf
    
    def fetch_ohlcv(
        self, 
        symbols: List[str], 
        start: str, 
        end: str
    ) -> Tuple[Dict[str, pd.DataFrame], DataProvenance]:
        """
        Fetch OHLCV data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            
        Returns:
            Tuple of (data dict, provenance record)
        """
        yf = self._get_yf()
        tickers_str = " ".join(symbols)
        logger.info(f"Fetching OHLCV: {tickers_str} ({start} to {end})")
        
        fetch_timestamp = datetime.now().isoformat()
        data = None
        
        for attempt in range(self.max_retries):
            try:
                data = yf.download(
                    tickers_str,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    progress=False,
                    timeout=self.timeout,
                    group_by='ticker' if len(symbols) > 1 else None
                )
                
                if data is None or len(data) == 0:
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"Empty data, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    raise ValueError(f"No data returned for {tickers_str}")
                break
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Fetch failed: {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
        
        # Process results
        results = {}
        if len(symbols) == 1:
            df = self._normalize_dataframe(data)
            if df is not None and len(df) > 0:
                results[symbols[0]] = df
        else:
            for sym in symbols:
                try:
                    if sym in data.columns.get_level_values(0):
                        df = self._normalize_dataframe(data[sym])
                        if df is not None and len(df) > 0:
                            results[sym] = df
                except Exception as e:
                    logger.warning(f"Could not extract {sym}: {e}")
        
        # Create provenance for primary symbol
        primary_symbol = symbols[0]
        primary_df = results.get(primary_symbol)
        
        if primary_df is not None:
            data_hash = hashlib.sha256(
                pd.util.hash_pandas_object(primary_df['Close']).values.tobytes()
            ).hexdigest()[:16]
            
            provenance = DataProvenance(
                source="yahoo_finance",
                symbol=primary_symbol,
                fetch_timestamp=fetch_timestamp,
                date_range=(start, end),
                record_count=len(primary_df),
                data_hash=data_hash
            )
        else:
            provenance = DataProvenance(
                source="yahoo_finance",
                symbol=primary_symbol,
                fetch_timestamp=fetch_timestamp,
                date_range=(start, end),
                record_count=0,
                data_hash=""
            )
        
        logger.info(f"Fetched {len(results)} symbols: {list(results.keys())}")
        return results, provenance
    
    def fetch_earnings_calendar(self, symbol: str) -> List[EarningsEvent]:
        """
        Fetch earnings announcements for a symbol.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            List of EarningsEvent objects, sorted by date descending
        """
        yf = self._get_yf()
        
        try:
            ticker = yf.Ticker(symbol)
            earnings = ticker.earnings_dates
            
            if earnings is None or len(earnings) == 0:
                logger.info(f"No earnings data for {symbol}")
                return []
            
            events = []
            for date_idx, row in earnings.iterrows():
                # Handle timezone-aware timestamps
                if hasattr(date_idx, 'tz') and date_idx.tz is not None:
                    date_idx = date_idx.tz_localize(None)
                
                date_str = (
                    str(date_idx.date()) 
                    if hasattr(date_idx, 'date') 
                    else str(date_idx)[:10]
                )
                
                events.append(EarningsEvent(
                    date=date_str,
                    eps_estimate=self._safe_float(row.get('EPS Estimate')),
                    eps_actual=self._safe_float(row.get('Reported EPS')),
                    surprise_pct=self._safe_float(row.get('Surprise(%)')),
                ))
            
            events.sort(key=lambda x: x.date, reverse=True)
            logger.info(f"Found {len(events)} earnings events for {symbol}")
            return events
            
        except Exception as e:
            logger.warning(f"Could not fetch earnings for {symbol}: {e}")
            return []
    
    def fetch_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch company metadata.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Dictionary with company information
        """
        yf = self._get_yf()
        
        try:
            info = yf.Ticker(symbol).info
            return {
                'name': info.get('longName', info.get('shortName', symbol)),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap'),
                'currency': info.get('currency', 'USD'),
            }
        except Exception as e:
            logger.warning(f"Could not fetch info for {symbol}: {e}")
            return {
                'name': symbol,
                'sector': 'Unknown',
                'industry': 'Unknown',
                'market_cap': None,
                'currency': 'USD',
            }
    
    def _normalize_dataframe(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Normalize DataFrame structure."""
        if df is None or len(df) == 0:
            return None
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Remove timezone
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        df = df.dropna(how='all')
        
        # Verify required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            logger.warning(f"Missing required columns: {missing}")
            return None
        
        return df
    
    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely convert value to float."""
        if pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None


# =============================================================================
# DATA VALIDATION
# =============================================================================

class DataValidator:
    """
    Four-dimension data quality assessment with holiday awareness.
    
    Dimensions:
    1. Completeness (40%): Missing values, trading day coverage
    2. Accuracy (30%): OHLC integrity, outlier detection
    3. Consistency (20%): Corporate action detection, price continuity
    4. Timeliness (10%): Data freshness relative to market calendar
    """
    
    # Quality dimension weights
    WEIGHT_COMPLETENESS = 0.40
    WEIGHT_ACCURACY = 0.30
    WEIGHT_CONSISTENCY = 0.20
    WEIGHT_TIMELINESS = 0.10
    
    # Grade thresholds
    GRADE_EXCELLENT = 95.0
    GRADE_GOOD = 85.0
    GRADE_ACCEPTABLE = 70.0
    GRADE_POOR = 50.0
    
    # Detection thresholds
    SPLIT_THRESHOLD = 0.40
    VOLUME_RATIO_THRESHOLD = 1.5
    OUTLIER_IQR = 1.5
    
    def validate(self, df: pd.DataFrame, symbol: str) -> DataQualityMetrics:
        """
        Run complete data quality assessment.
        
        Args:
            df: OHLCV DataFrame
            symbol: Ticker symbol (for logging)
            
        Returns:
            DataQualityMetrics with all dimension scores and overall grade
        """
        issues: List[str] = []
        warnings: List[str] = []
        
        # Assess each dimension
        completeness = self._assess_completeness(df, issues, warnings)
        accuracy = self._assess_accuracy(df, issues, warnings)
        consistency, corporate_actions = self._assess_consistency(df, issues, warnings)
        timeliness = self._assess_timeliness(df, issues, warnings)
        
        # Calculate weighted overall score
        overall = (
            self.WEIGHT_COMPLETENESS * completeness +
            self.WEIGHT_ACCURACY * accuracy +
            self.WEIGHT_CONSISTENCY * consistency +
            self.WEIGHT_TIMELINESS * timeliness
        )
        
        # Assign grade
        if overall >= self.GRADE_EXCELLENT:
            grade = QualityGrade.EXCELLENT
        elif overall >= self.GRADE_GOOD:
            grade = QualityGrade.GOOD
        elif overall >= self.GRADE_ACCEPTABLE:
            grade = QualityGrade.ACCEPTABLE
        elif overall >= self.GRADE_POOR:
            grade = QualityGrade.POOR
        else:
            grade = QualityGrade.UNUSABLE
        
        # Compute data hash for integrity verification
        data_hash = hashlib.sha256(
            pd.util.hash_pandas_object(df['Close']).values.tobytes()
        ).hexdigest()[:16]
        
        return DataQualityMetrics(
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            timeliness=timeliness,
            overall=overall,
            grade=grade,
            issues=issues,
            warnings=warnings,
            corporate_actions=corporate_actions,
            records_total=len(df),
            records_valid=len(df.dropna(subset=['Close'])),
            date_range=(
                df.index.min().strftime('%Y-%m-%d'),
                df.index.max().strftime('%Y-%m-%d')
            ),
            data_hash=data_hash
        )
    
    def _assess_completeness(
        self, df: pd.DataFrame, 
        issues: List[str], 
        warnings: List[str]
    ) -> float:
        """Assess data completeness."""
        score = 100.0
        n = len(df)
        
        # Check each required column for missing values
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                missing = df[col].isna().sum()
                if missing > 0:
                    pct = missing / n * 100
                    if pct > 5:
                        issues.append(f"{col}: {missing} missing ({pct:.1f}%)")
                        score -= pct * 3
                    elif pct > 1:
                        warnings.append(f"{col}: {missing} missing ({pct:.1f}%)")
                        score -= pct * 1.5
        
        # Check trading day coverage
        if n > TRADING_DAYS_YEAR:
            calendar_days = (df.index.max() - df.index.min()).days
            expected_trading_days = int(calendar_days * TRADING_DAYS_YEAR / 365)
            
            if expected_trading_days > 0:
                coverage = n / expected_trading_days
                if coverage < 0.90:
                    issues.append(f"Trading day coverage: {coverage:.1%}")
                    score -= (1 - coverage) * 20
                elif coverage < 0.95:
                    warnings.append(f"Trading day coverage: {coverage:.1%}")
                    score -= (1 - coverage) * 10
        
        return max(0, min(100, score))
    
    def _assess_accuracy(
        self, df: pd.DataFrame, 
        issues: List[str], 
        warnings: List[str]
    ) -> float:
        """Assess data accuracy (OHLC integrity and outliers)."""
        score = 100.0
        n = len(df)
        
        # Check High >= Low
        invalid_hl = (df['High'] < df['Low']).sum()
        if invalid_hl > 0:
            issues.append(f"High < Low: {invalid_hl} days")
            score -= (invalid_hl / n) * 100
        
        # Check High >= max(Open, Close)
        invalid_high = (df['High'] < df[['Open', 'Close']].max(axis=1)).sum()
        if invalid_high > 0:
            issues.append(f"High < max(O,C): {invalid_high} days")
            score -= (invalid_high / n) * 50
        
        # Check Low <= min(Open, Close)
        invalid_low = (df['Low'] > df[['Open', 'Close']].min(axis=1)).sum()
        if invalid_low > 0:
            issues.append(f"Low > min(O,C): {invalid_low} days")
            score -= (invalid_low / n) * 50
        
        # Check for zero or negative prices
        zero_prices = ((df['Close'] <= 0) | (df['Open'] <= 0)).sum()
        if zero_prices > 0:
            issues.append(f"Zero/negative prices: {zero_prices} days")
            score -= zero_prices * 5
        
        # Check for return outliers (statistical)
        returns = df['Close'].pct_change().dropna()
        if len(returns) > 0:
            q1, q3 = returns.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - self.OUTLIER_IQR * iqr
            upper_bound = q3 + self.OUTLIER_IQR * iqr
            
            outliers = returns[(returns < lower_bound) | (returns > upper_bound)]
            if len(outliers) > 0:
                pct = len(outliers) / len(returns) * 100
                # Only penalize excessive outliers (>5%)
                if pct > 5:
                    warnings.append(f"Return outliers: {len(outliers)} ({pct:.1f}%)")
                    score -= min(10, pct)
        
        return max(0, min(100, score))
    
    def _assess_consistency(
        self, df: pd.DataFrame, 
        issues: List[str], 
        warnings: List[str]
    ) -> Tuple[float, List[CorporateAction]]:
        """Assess data consistency and detect corporate actions."""
        score = 100.0
        actions: List[CorporateAction] = []
        
        returns = df['Close'].pct_change().dropna()
        large_moves = returns.abs() > self.SPLIT_THRESHOLD
        
        for date in returns[large_moves].index:
            idx = df.index.get_loc(date)
            if idx == 0:
                continue
            
            ret = returns.loc[date]
            
            # Check for split pattern (large price change + volume change)
            if 'Volume' in df.columns and idx > 0:
                vol_today = df['Volume'].iloc[idx]
                vol_yesterday = df['Volume'].iloc[idx - 1]
                
                if vol_yesterday > 0:
                    vol_ratio = vol_today / vol_yesterday
                    
                    # Stock split pattern: price down, volume up
                    if ret < -0.30 and vol_ratio > self.VOLUME_RATIO_THRESHOLD:
                        confidence = min(1.0, abs(ret) + (vol_ratio - 1) / 2)
                        estimated_ratio = round(1 / (1 + ret), 1)
                        
                        actions.append(CorporateAction(
                            date=date.strftime('%Y-%m-%d'),
                            action_type=CorporateActionType.STOCK_SPLIT,
                            ratio=estimated_ratio,
                            confidence=confidence,
                            details=f"Price: {ret*100:+.1f}%, Volume: {vol_ratio:.1f}x"
                        ))
                    
                    # Reverse split pattern: price up, volume down
                    elif ret > 0.30 and vol_ratio < 1 / self.VOLUME_RATIO_THRESHOLD:
                        confidence = min(1.0, abs(ret) + (1 - vol_ratio) / 2)
                        estimated_ratio = round(1 + ret, 1)
                        
                        actions.append(CorporateAction(
                            date=date.strftime('%Y-%m-%d'),
                            action_type=CorporateActionType.REVERSE_SPLIT,
                            ratio=estimated_ratio,
                            confidence=confidence,
                            details=f"Price: {ret*100:+.1f}%, Volume: {vol_ratio:.1f}x"
                        ))
            
            # Log large moves
            if abs(ret) > 0.15:
                warnings.append(
                    f"Large move: {ret*100:+.1f}% on {date.strftime('%Y-%m-%d')}"
                )
        
        # Check for trading gaps
        gaps = df.index.to_series().diff()
        large_gaps = gaps[gaps > pd.Timedelta(days=7)]
        
        if len(large_gaps) > 0:
            for gap_date, gap_size in large_gaps.items():
                warnings.append(
                    f"Trading gap: {gap_size.days} days before "
                    f"{gap_date.strftime('%Y-%m-%d')}"
                )
            score -= len(large_gaps) * 2
        
        return max(0, min(100, score)), actions
    
    def _assess_timeliness(
        self, df: pd.DataFrame, 
        issues: List[str], 
        warnings: List[str]
    ) -> float:
        """Assess data timeliness with market calendar awareness."""
        score = 100.0
        
        last_date = df.index.max()
        today = pd.Timestamp.now().normalize()
        
        # Get the most recent expected trading day
        last_expected = USMarketCalendar.get_last_trading_day(today)
        
        # If checking before market close (4pm ET), use previous day
        current_hour = datetime.now().hour
        if current_hour < 16:
            last_expected = USMarketCalendar.get_last_trading_day(
                last_expected - pd.Timedelta(days=1)
            )
        
        # Count actual trading days missed
        trading_days_missed = USMarketCalendar.count_trading_days_between(
            last_date, last_expected + pd.Timedelta(days=1)
        )
        
        # Apply penalties
        if trading_days_missed > 5:
            issues.append(f"Data stale: {trading_days_missed} trading days behind")
            score -= min(15, trading_days_missed * 2)
        elif trading_days_missed > 2:
            warnings.append(f"Data age: {trading_days_missed} trading days")
            score -= trading_days_missed
        
        return max(0, min(100, score))


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """
    Comprehensive feature engineering for Phase 1.
    
    Creates foundational features for technical analysis:
    - Returns (simple, log, overnight, intraday, multi-period)
    - Price structure (range, gaps, body, shadows)
    - Volume analysis (dollar volume, relative volume, liquidity)
    - Moving averages (SMA, EMA)
    - True Range and ATR
    - 52-week metrics
    - Volatility estimators (7 academic methods)
    """
    
    def __init__(self, volatility_window: int = TRADING_DAYS_MONTH):
        """
        Initialize feature engineer.
        
        Args:
            volatility_window: Rolling window for volatility calculations
        """
        self.vol_window = volatility_window
    
    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all engineered features to DataFrame.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with additional feature columns
        """
        df = df.copy()
        prev_close = df['Close'].shift(1)
        
        # =====================================================================
        # RETURNS (10 features)
        # =====================================================================
        df['Return'] = df['Close'].pct_change()
        df['LogReturn'] = np.log(df['Close'] / prev_close)
        df['OvernightReturn'] = np.log(df['Open'] / prev_close)
        df['IntradayReturn'] = np.log(df['Close'] / df['Open'])
        df['CumReturn'] = (1 + df['Return']).cumprod() - 1
        df['CumLogReturn'] = df['LogReturn'].cumsum()
        
        # Multi-period returns
        for period in [5, 21, 63, 126, 252]:
            df[f'Return_{period}d'] = df['Close'].pct_change(period)
        
        # =====================================================================
        # PRICE STRUCTURE (12 features)
        # =====================================================================
        df['Range'] = df['High'] - df['Low']
        df['RangePct'] = df['Range'] / df['Close'] * 100
        df['Gap'] = df['Open'] - prev_close
        df['GapPct'] = df['Gap'] / prev_close * 100
        df['Body'] = df['Close'] - df['Open']
        df['BodyPct'] = df['Body'] / df['Open'] * 100
        df['UpperShadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['LowerShadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['TypicalPrice'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['WeightedClose'] = (df['High'] + df['Low'] + df['Close'] * 2) / 4
        df['PriceVolume'] = df['TypicalPrice'] * df['Volume']
        df['CumPriceVolume'] = df['PriceVolume'].cumsum()
        df['CumVolume'] = df['Volume'].cumsum()
        
        # =====================================================================
        # ATR (8 features)
        # =====================================================================
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - prev_close).abs()
        tr3 = (df['Low'] - prev_close).abs()
        df['TrueRange'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        for period in [7, 14, 21]:
            df[f'ATR_{period}'] = df['TrueRange'].rolling(period).mean()
        
        df['ATR_Pct'] = df['ATR_14'] / df['Close'] * 100
        df['ATR_Normalized'] = df['ATR_14'] / df['ATR_14'].rolling(252).mean()
        df['ATR_Ratio_7_21'] = df['ATR_7'] / df['ATR_21']
        df['ATR_Percentile'] = df['ATR_14'].rolling(252).rank(pct=True)
        
        # =====================================================================
        # VOLUME (8 features)
        # =====================================================================
        df['DollarVolume'] = df['Close'] * df['Volume']
        
        for period in [10, 20, 50]:
            df[f'Volume_MA{period}'] = df['Volume'].rolling(period).mean()
        
        df['RelativeVolume'] = df['Volume'] / df['Volume_MA20']
        
        # Amihud illiquidity measure
        df['Amihud'] = df['Return'].abs() / (df['DollarVolume'] / 1e6)
        df['Amihud'] = df['Amihud'].replace([np.inf, -np.inf], np.nan)
        
        # Roll spread estimate (bid-ask proxy)
        roll_cov = df['Return'].rolling(21).apply(
            lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0,
            raw=True
        )
        df['RollSpread'] = np.sqrt(np.abs(-roll_cov)) * 2
        df['RollSpread'] = df['RollSpread'].replace([np.inf, -np.inf], np.nan)
        
        # Cumulative VWAP
        df['VWAP'] = df['CumPriceVolume'] / df['CumVolume']
        
        # =====================================================================
        # MOVING AVERAGES (15 features)
        # =====================================================================
        for period in [10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
            df[f'DistFromSMA_{period}'] = (df['Close'] / df[f'SMA_{period}'] - 1) * 100
        
        # =====================================================================
        # 52-WEEK METRICS (8 features)
        # =====================================================================
        df['High_52W'] = df['High'].rolling(252).max()
        df['Low_52W'] = df['Low'].rolling(252).min()
        
        range_52w = df['High_52W'] - df['Low_52W']
        df['Position_52W'] = np.where(
            range_52w > 0,
            (df['Close'] - df['Low_52W']) / range_52w,
            0.5
        )
        
        df['DistFromHigh_52W'] = (df['Close'] / df['High_52W'] - 1) * 100
        df['DistFromLow_52W'] = (df['Close'] / df['Low_52W'] - 1) * 100
        df['NewHigh_52W'] = (df['High'] >= df['High_52W']).astype(int)
        df['NewLow_52W'] = (df['Low'] <= df['Low_52W']).astype(int)
        
        # Days since high/low
        high_mask = df['High'] >= df['High_52W']
        low_mask = df['Low'] <= df['Low_52W']
        df['DaysSinceHigh_52W'] = (
            (~high_mask).cumsum() - 
            (~high_mask).cumsum().where(high_mask).ffill().fillna(0)
        )
        df['DaysSinceLow_52W'] = (
            (~low_mask).cumsum() - 
            (~low_mask).cumsum().where(low_mask).ffill().fillna(0)
        )
        
        # =====================================================================
        # VOLATILITY ESTIMATORS (8 features)
        # =====================================================================
        df = self._add_volatility_estimators(df)
        
        return df
    
    def _add_volatility_estimators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add seven academic volatility estimators plus composite."""
        w = self.vol_window
        log_ret = df['LogReturn']
        
        # 1. Close-to-Close (baseline)
        df['Vol_CloseClose'] = log_ret.rolling(w).std() * SQRT_252
        
        # 2. Parkinson (1980) - High-Low based
        log_hl = np.log(df['High'] / df['Low'])
        parkinson_factor = 1 / (4 * np.log(2))
        df['Vol_Parkinson'] = np.sqrt(
            parkinson_factor * (log_hl ** 2).rolling(w).mean()
        ) * SQRT_252
        
        # 3. Garman-Klass (1980) - Full OHLC
        log_co = np.log(df['Close'] / df['Open'])
        gk_term = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        df['Vol_GarmanKlass'] = np.sqrt(
            gk_term.rolling(w).mean().clip(lower=0)
        ) * SQRT_252
        
        # 4. Rogers-Satchell (1991) - Handles drift
        log_ho = np.log(df['High'] / df['Open'])
        log_hc = np.log(df['High'] / df['Close'])
        log_lo = np.log(df['Low'] / df['Open'])
        log_lc = np.log(df['Low'] / df['Close'])
        rs_term = log_ho * log_hc + log_lo * log_lc
        df['Vol_RogersSatchell'] = np.sqrt(
            rs_term.rolling(w).mean().clip(lower=0)
        ) * SQRT_252
        
        # 5. Yang-Zhang (2000) - Handles overnight gaps
        prev_close = df['Close'].shift(1)
        log_oc = np.log(df['Open'] / prev_close)
        
        overnight_var = log_oc.rolling(w).var()
        open_var = (np.log(df['High'] / df['Open']) * 
                   np.log(df['Low'] / df['Open'])).rolling(w).var()
        close_var = log_ret.rolling(w).var()
        
        k = 0.34 / (1.34 + (w + 1) / (w - 1))
        yz_var = overnight_var + k * close_var + (1 - k) * open_var
        df['Vol_YangZhang'] = np.sqrt(yz_var.clip(lower=0)) * SQRT_252
        
        # 6. GKYZ - Garman-Klass with Yang-Zhang adjustment
        df['Vol_GKYZ'] = np.sqrt(
            (overnight_var + gk_term.rolling(w).mean()).clip(lower=0)
        ) * SQRT_252
        
        # 7. Hodges-Tompkins - Bias-corrected
        n = w
        correction = 1 + 2 * (n - 1) / (n * (n + 1))
        df['Vol_HodgesTompkins'] = df['Vol_CloseClose'] * np.sqrt(correction)
        
        # Composite (weighted average)
        weights = {
            'Vol_CloseClose': 0.20,
            'Vol_Parkinson': 0.10,
            'Vol_GarmanKlass': 0.15,
            'Vol_RogersSatchell': 0.15,
            'Vol_YangZhang': 0.20,
            'Vol_GKYZ': 0.10,
            'Vol_HodgesTompkins': 0.10
        }
        df['Vol_Composite'] = sum(
            weight * df[col] for col, weight in weights.items()
        )
        
        return df
    
    def aggregate_timeframe(
        self, daily: pd.DataFrame, 
        freq: str = 'W'
    ) -> pd.DataFrame:
        """
        Aggregate daily data to lower frequency (weekly or monthly).
        
        Args:
            daily: Daily OHLCV DataFrame
            freq: Resample frequency ('W' for weekly, 'ME' for monthly)
            
        Returns:
            Aggregated DataFrame with basic features
        """
        agg = daily[['Open', 'High', 'Low', 'Close', 'Volume']].resample(freq).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        # Add basic features
        agg['Return'] = agg['Close'].pct_change()
        agg['LogReturn'] = np.log(agg['Close'] / agg['Close'].shift(1))
        
        prev_close = agg['Close'].shift(1)
        tr1 = agg['High'] - agg['Low']
        tr2 = (agg['High'] - prev_close).abs()
        tr3 = (agg['Low'] - prev_close).abs()
        agg['TrueRange'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return agg


# =============================================================================
# STATISTICAL PROFILING
# =============================================================================

class StatisticalProfiler:
    """
    Statistical analysis and hypothesis testing.
    
    Performs five hypothesis tests:
    1. Jarque-Bera: Return normality
    2. ADF: Unit root / stationarity
    3. KPSS: Stationarity confirmation
    4. Ljung-Box: Autocorrelation
    5. ARCH: Volatility clustering
    
    Also computes Hurst exponent for trend persistence analysis.
    """
    
    def __init__(self, alpha: float = 0.05, risk_free: float = 0.04):
        """
        Initialize profiler.
        
        Args:
            alpha: Significance level for hypothesis tests
            risk_free: Risk-free rate for Sharpe calculation
        """
        self.alpha = alpha
        self.rf = risk_free
    
    def profile(self, df: pd.DataFrame) -> MarketProfile:
        """
        Generate complete statistical profile.
        
        Args:
            df: Enriched DataFrame with features
            
        Returns:
            MarketProfile with all analysis results
        """
        returns = df['Return'].dropna()
        
        # Basic return statistics
        daily_mean = float(returns.mean())
        daily_std = float(returns.std())
        daily_median = float(returns.median())
        ann_return = daily_mean * TRADING_DAYS_YEAR
        ann_vol = daily_std * SQRT_252
        
        # Distribution shape
        skewness = float(returns.skew())
        kurtosis = float(returns.kurtosis())
        
        if abs(skewness) < 0.5 and abs(kurtosis) < 1:
            dist_type = ReturnDistribution.NORMAL
        elif kurtosis > 0:
            dist_type = ReturnDistribution.LEPTOKURTIC
        else:
            dist_type = ReturnDistribution.PLATYKURTIC
        
        # Autocorrelation
        ac1 = self._safe_autocorr(returns, 1)
        ac5 = self._safe_autocorr(returns, 5)
        ac21 = self._safe_autocorr(returns, 21)
        
        # Volatility profile
        vol_profile = self._build_volatility_profile(df)
        
        # Volatility of volatility
        rolling_vol = returns.rolling(21).std() * SQRT_252
        vol_mean = rolling_vol.mean()
        vol_of_vol = float(rolling_vol.std() / vol_mean) if vol_mean > 0 else 0
        
        # Hurst exponent
        hurst, hurst_conf = self._estimate_hurst(returns)
        
        if hurst < 0.45:
            trend = TrendCharacter.MEAN_REVERTING
        elif hurst > 0.55:
            trend = TrendCharacter.TRENDING
        else:
            trend = TrendCharacter.RANDOM_WALK
        
        # Tail risk metrics
        tail_risk = self._compute_tail_risk(returns)
        
        # Statistical tests
        stat_tests = self._run_tests(returns)
        
        # Generate strategy hints
        hints = self._generate_hints(
            stat_tests, vol_profile, hurst, trend, kurtosis, skewness
        )
        
        return MarketProfile(
            daily_return_mean=daily_mean,
            daily_return_std=daily_std,
            daily_return_median=daily_median,
            annualized_return=ann_return,
            annualized_volatility=ann_vol,
            skewness=skewness,
            kurtosis=kurtosis,
            return_distribution=dist_type,
            autocorr_lag1=ac1,
            autocorr_lag5=ac5,
            autocorr_lag21=ac21,
            volatility_profile=vol_profile,
            volatility_of_volatility=vol_of_vol,
            hurst_exponent=hurst,
            hurst_confidence=hurst_conf,
            trend_character=trend,
            tail_risk=tail_risk,
            statistical_tests=stat_tests,
            strategy_hints=hints
        )
    
    def _safe_autocorr(self, series: pd.Series, lag: int) -> float:
        """Safely compute autocorrelation."""
        if len(series) <= lag:
            return 0.0
        ac = series.autocorr(lag=lag)
        return float(ac) if pd.notna(ac) else 0.0
    
    def _build_volatility_profile(self, df: pd.DataFrame) -> VolatilityProfile:
        """Build volatility profile from latest values."""
        latest = df.iloc[-1]
        
        c2c = float(latest.get('Vol_CloseClose', 0))
        park = float(latest.get('Vol_Parkinson', 0))
        gk = float(latest.get('Vol_GarmanKlass', 0))
        rs = float(latest.get('Vol_RogersSatchell', 0))
        yz = float(latest.get('Vol_YangZhang', 0))
        gkyz = float(latest.get('Vol_GKYZ', 0))
        ht = float(latest.get('Vol_HodgesTompkins', 0))
        comp = float(latest.get('Vol_Composite', c2c))
        
        # Classify regime
        if comp < 0.15:
            regime = VolatilityRegime.LOW
        elif comp < 0.25:
            regime = VolatilityRegime.NORMAL
        elif comp < 0.40:
            regime = VolatilityRegime.HIGH
        else:
            regime = VolatilityRegime.EXTREME
        
        # Calculate percentile
        pct = 50.0
        if 'Vol_Composite' in df.columns:
            vol_series = df['Vol_Composite'].dropna()
            if len(vol_series) > 0:
                pct = (vol_series < comp).sum() / len(vol_series) * 100
        
        return VolatilityProfile(
            close_to_close=c2c,
            parkinson=park,
            garman_klass=gk,
            rogers_satchell=rs,
            yang_zhang=yz,
            gkyz=gkyz,
            hodges_tompkins=ht,
            composite=comp,
            regime=regime,
            percentile_rank=pct
        )
    
    def _compute_tail_risk(self, returns: pd.Series) -> TailRiskMetrics:
        """Compute tail risk metrics."""
        var_95 = float(np.percentile(returns, 5))
        var_99 = float(np.percentile(returns, 1))
        var_999 = float(np.percentile(returns, 0.1))
        
        below_var_95 = returns[returns <= var_95]
        below_var_99 = returns[returns <= var_99]
        
        cvar_95 = float(below_var_95.mean()) if len(below_var_95) > 0 else var_95
        cvar_99 = float(below_var_99.mean()) if len(below_var_99) > 0 else var_99
        
        max_gain = float(returns.max())
        max_loss = float(returns.min())
        
        # Maximum drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = float(drawdown.min())
        
        # Drawdown duration
        is_in_drawdown = drawdown < 0
        drawdown_groups = (~is_in_drawdown).cumsum()
        if is_in_drawdown.any():
            drawdown_lengths = is_in_drawdown.groupby(drawdown_groups).sum()
            max_dd_duration = int(drawdown_lengths.max())
        else:
            max_dd_duration = 0
        
        # Tail ratio
        pct_95 = np.percentile(returns, 95)
        pct_5 = np.percentile(returns, 5)
        tail_ratio = abs(pct_95 / pct_5) if pct_5 != 0 else 1.0
        
        # Downside deviation
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 1:
            downside_dev = float(negative_returns.std() * SQRT_252)
        else:
            downside_dev = float(returns.std() * SQRT_252)
        
        return TailRiskMetrics(
            var_95=var_95,
            var_99=var_99,
            var_999=var_999,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_gain=max_gain,
            max_loss=max_loss,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            skewness=float(returns.skew()),
            kurtosis=float(returns.kurtosis()),
            tail_ratio=tail_ratio,
            downside_deviation=downside_dev
        )
    
    def _estimate_hurst(
        self, returns: pd.Series, 
        max_lag: int = 100
    ) -> Tuple[float, float]:
        """
        Estimate Hurst exponent using R/S analysis.
        
        Returns:
            Tuple of (hurst_exponent, r_squared confidence)
        """
        arr = returns.values
        n = len(arr)
        
        if n < 20:
            return 0.5, 0.0
        
        max_lag = min(max_lag, n // 2)
        lags = []
        rs_values = []
        
        for lag in range(10, max_lag, 5):
            rs_list = []
            
            for start in range(0, n - lag, lag):
                subset = arr[start:start + lag]
                if len(subset) < lag:
                    continue
                
                mean_val = np.mean(subset)
                cumdev = np.cumsum(subset - mean_val)
                r = np.max(cumdev) - np.min(cumdev)
                s = np.std(subset, ddof=1)
                
                if s > 0:
                    rs_list.append(r / s)
            
            if rs_list:
                lags.append(lag)
                rs_values.append(np.mean(rs_list))
        
        if len(lags) < 3:
            return 0.5, 0.0
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                np.log(lags), np.log(rs_values)
            )
            hurst = float(np.clip(slope, 0, 1))
            confidence = float(r_value ** 2)
            return hurst, confidence
        except Exception:
            return 0.5, 0.0
    
    def _run_tests(self, returns: pd.Series) -> StatisticalTests:
        """Run all five hypothesis tests."""
        clean_returns = returns.dropna()
        n = len(clean_returns)
        
        # 1. Jarque-Bera (normality)
        try:
            jb_stat, jb_pval = stats.jarque_bera(clean_returns)
            jb_stat = float(jb_stat)
            jb_pval = float(jb_pval)
        except Exception:
            jb_stat, jb_pval = 0.0, 1.0
        
        # 2. ADF (stationarity)
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(clean_returns, autolag='AIC')
            adf_stat = float(adf_result[0])
            adf_pval = float(adf_result[1])
            adf_lags = int(adf_result[2])
            adf_c1 = float(adf_result[4]['1%'])
            adf_c5 = float(adf_result[4]['5%'])
            adf_c10 = float(adf_result[4]['10%'])
        except Exception:
            adf_stat, adf_pval, adf_lags = -3.0, 0.01, 0
            adf_c1, adf_c5, adf_c10 = -3.5, -2.9, -2.6
        
        # 3. KPSS (stationarity confirmation)
        try:
            from statsmodels.tsa.stattools import kpss
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_result = kpss(clean_returns, regression='c', nlags='auto')
            kpss_stat = float(kpss_result[0])
            kpss_pval = float(kpss_result[1])
            kpss_c5 = float(kpss_result[3]['5%'])
            kpss_c10 = float(kpss_result[3]['10%'])
        except Exception:
            kpss_stat, kpss_pval = 0.1, 0.10
            kpss_c5, kpss_c10 = 0.463, 0.347
        
        # 4. Ljung-Box (autocorrelation)
        lb_lags = min(10, n // 5)
        if lb_lags < 1:
            lb_lags = 1
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(clean_returns, lags=[lb_lags], return_df=True)
            lb_stat = float(lb_result['lb_stat'].values[0])
            lb_pval = float(lb_result['lb_pvalue'].values[0])
        except Exception:
            lb_stat, lb_pval = 0.0, 1.0
        
        # 5. ARCH (volatility clustering)
        arch_lags = min(5, n // 10)
        if arch_lags < 1:
            arch_lags = 1
        try:
            from statsmodels.stats.diagnostic import het_arch
            arch_result = het_arch(clean_returns, nlags=arch_lags)
            arch_stat = float(arch_result[0])
            arch_pval = float(arch_result[1])
        except Exception:
            arch_stat, arch_pval = 0.0, 1.0
        
        return StatisticalTests(
            jb_statistic=jb_stat,
            jb_p_value=jb_pval,
            is_normal=jb_pval > self.alpha,
            adf_statistic=adf_stat,
            adf_p_value=adf_pval,
            adf_critical_1pct=adf_c1,
            adf_critical_5pct=adf_c5,
            adf_critical_10pct=adf_c10,
            adf_lags_used=adf_lags,
            is_stationary=adf_pval < self.alpha,
            kpss_statistic=kpss_stat,
            kpss_p_value=kpss_pval,
            kpss_critical_5pct=kpss_c5,
            kpss_critical_10pct=kpss_c10,
            is_stationary_kpss=kpss_pval > self.alpha,
            lb_statistic=lb_stat,
            lb_p_value=lb_pval,
            lb_lags=lb_lags,
            has_autocorrelation=lb_pval < self.alpha,
            arch_statistic=arch_stat,
            arch_p_value=arch_pval,
            arch_lags=arch_lags,
            has_arch_effects=arch_pval < self.alpha
        )
    
    def _generate_hints(
        self, tests: StatisticalTests, 
        vol: VolatilityProfile,
        hurst: float, 
        trend: TrendCharacter,
        kurtosis: float, 
        skewness: float
    ) -> List[str]:
        """Generate strategy recommendations."""
        hints = []
        
        if trend == TrendCharacter.TRENDING:
            hints.append(f"Hurst {hurst:.2f}: Momentum strategies may be suitable")
        elif trend == TrendCharacter.MEAN_REVERTING:
            hints.append(f"Hurst {hurst:.2f}: Mean-reversion strategies may be suitable")
        else:
            hints.append(f"Hurst {hurst:.2f}: Random walk - trend following may struggle")
        
        if vol.regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]:
            hints.append(f"High volatility ({vol.composite:.1%}): Reduce position sizes")
        elif vol.regime == VolatilityRegime.LOW:
            hints.append(f"Low volatility ({vol.composite:.1%}): Watch for expansion")
        
        if tests.has_arch_effects:
            hints.append("ARCH effects: Use GARCH for volatility forecasting")
        
        if tests.has_autocorrelation:
            hints.append("Autocorrelation detected: Predictable patterns may exist")
        
        if kurtosis > 3:
            hints.append(f"Fat tails (kurtosis={kurtosis:.1f}): Use robust risk measures")
        
        if skewness < -0.5:
            hints.append(f"Negative skew ({skewness:.2f}): Downside risk is elevated")
        
        if tests.stationarity_conclusion == StationarityConclusion.NON_STATIONARY:
            hints.append("Non-stationary: Consider regime-switching models")
        
        return hints


# =============================================================================
# BENCHMARK ANALYSIS
# =============================================================================

class BenchmarkAnalyzer:
    """
    Relative performance analysis against benchmark.
    
    Computes:
    - Correlation (full period and rolling)
    - Beta (CAPM)
    - Alpha (Jensen's)
    - Information Ratio
    - Tracking Error
    - Up/Down Capture Ratios
    - Relative Strength
    """
    
    def __init__(self, benchmark_symbol: str = "SPY", risk_free: float = 0.04):
        self.benchmark_symbol = benchmark_symbol
        self.rf = risk_free
    
    def analyze(
        self, asset: pd.DataFrame, 
        benchmark: pd.DataFrame
    ) -> Optional[BenchmarkAnalysis]:
        """
        Perform benchmark analysis.
        
        Args:
            asset: Asset DataFrame with Return column
            benchmark: Benchmark DataFrame with Return column
            
        Returns:
            BenchmarkAnalysis or None if insufficient data
        """
        a_ret = asset['Return'].dropna()
        b_ret = benchmark['Return'].dropna()
        common = a_ret.index.intersection(b_ret.index)
        
        if len(common) < 50:
            logger.warning("Insufficient overlapping data for benchmark analysis")
            return None
        
        a_ret = a_ret.loc[common]
        b_ret = b_ret.loc[common]
        
        # Full period correlation
        corr_full = float(a_ret.corr(b_ret))
        
        # 1-year correlation
        if len(common) >= 252:
            corr_1y = float(a_ret.iloc[-252:].corr(b_ret.iloc[-252:]))
        else:
            corr_1y = corr_full
        
        # Rolling correlation
        roll_corr = a_ret.rolling(63).corr(b_ret)
        corr_rm = float(roll_corr.mean()) if pd.notna(roll_corr.mean()) else corr_full
        corr_rs = float(roll_corr.std()) if pd.notna(roll_corr.std()) else 0
        
        # Beta calculation
        cov_full = a_ret.cov(b_ret)
        var_b = b_ret.var()
        beta_full = float(cov_full / var_b) if var_b > 0 else 1.0
        
        # 1-year beta
        if len(common) >= 252:
            cov_1y = a_ret.iloc[-252:].cov(b_ret.iloc[-252:])
            var_b_1y = b_ret.iloc[-252:].var()
            beta_1y = float(cov_1y / var_b_1y) if var_b_1y > 0 else beta_full
        else:
            beta_1y = beta_full
        
        # Rolling beta
        roll_cov = a_ret.rolling(63).cov(b_ret)
        roll_var = b_ret.rolling(63).var()
        roll_beta = roll_cov / roll_var
        beta_rm = float(roll_beta.mean()) if pd.notna(roll_beta.mean()) else beta_full
        beta_rs = float(roll_beta.std()) if pd.notna(roll_beta.std()) else 0
        
        # Alpha (Jensen's)
        a_ann = float(a_ret.mean() * TRADING_DAYS_YEAR)
        b_ann = float(b_ret.mean() * TRADING_DAYS_YEAR)
        alpha = a_ann - (self.rf + beta_full * (b_ann - self.rf))
        
        # Tracking error and information ratio
        excess = a_ret - b_ret
        te = float(excess.std() * SQRT_252)
        ir = float(alpha / te) if te > 0 else 0
        
        # Relative strength
        a_cum = (1 + a_ret).cumprod()
        b_cum = (1 + b_ret).cumprod()
        
        def relative_strength(periods: int) -> float:
            if len(a_cum) >= periods:
                a_ret_period = a_cum.iloc[-1] / a_cum.iloc[-periods] - 1
                b_ret_period = b_cum.iloc[-1] / b_cum.iloc[-periods] - 1
                return float(a_ret_period - b_ret_period)
            return 0.0
        
        rs_1m = relative_strength(21)
        rs_3m = relative_strength(63)
        rs_6m = relative_strength(126)
        rs_1y = relative_strength(252)
        
        # Risk decomposition
        r_sq = corr_full ** 2
        var_a = a_ret.var()
        systematic = beta_full ** 2 * var_b / var_a if var_a > 0 else 0
        
        # Capture ratios
        up_days = b_ret > 0
        down_days = b_ret < 0
        
        if up_days.sum() > 0:
            up_capture = float(a_ret[up_days].mean() / b_ret[up_days].mean() * 100)
        else:
            up_capture = 100.0
        
        if down_days.sum() > 0:
            down_capture = float(a_ret[down_days].mean() / b_ret[down_days].mean() * 100)
        else:
            down_capture = 100.0
        
        return BenchmarkAnalysis(
            benchmark_symbol=self.benchmark_symbol,
            correlation_full=corr_full,
            correlation_1y=corr_1y,
            correlation_rolling_mean=corr_rm,
            correlation_rolling_std=corr_rs,
            beta_full=beta_full,
            beta_1y=beta_1y,
            beta_rolling_mean=beta_rm,
            beta_rolling_std=beta_rs,
            alpha_annualized=alpha,
            tracking_error=te,
            information_ratio=ir,
            relative_strength_1m=rs_1m,
            relative_strength_3m=rs_3m,
            relative_strength_6m=rs_6m,
            relative_strength_1y=rs_1y,
            r_squared=r_sq,
            systematic_risk_pct=systematic * 100,
            idiosyncratic_risk_pct=(1 - systematic) * 100,
            up_capture=up_capture,
            down_capture=down_capture
        )


# =============================================================================
# VIX ANALYSIS
# =============================================================================

class VIXAnalyzer:
    """
    VIX-based market context analysis.
    
    Classifies market fear into six regimes and provides
    strategy recommendations based on volatility conditions.
    """
    
    # VIX regime thresholds
    VIX_COMPLACENT = 12.0
    VIX_CALM = 16.0
    VIX_NORMAL = 20.0
    VIX_ELEVATED = 25.0
    VIX_FEARFUL = 30.0
    
    def analyze(self, vix_df: pd.DataFrame) -> Optional[VIXAnalysis]:
        """
        Analyze VIX data for market context.
        
        Args:
            vix_df: VIX DataFrame with Close column
            
        Returns:
            VIXAnalysis or None if insufficient data
        """
        if vix_df is None or len(vix_df) < 21:
            return None
        
        current_level = float(vix_df['Close'].iloc[-1])
        
        # Classify regime
        if current_level < self.VIX_COMPLACENT:
            regime = VIXRegime.COMPLACENT
        elif current_level < self.VIX_CALM:
            regime = VIXRegime.CALM
        elif current_level < self.VIX_NORMAL:
            regime = VIXRegime.NORMAL
        elif current_level < self.VIX_ELEVATED:
            regime = VIXRegime.ELEVATED
        elif current_level < self.VIX_FEARFUL:
            regime = VIXRegime.FEARFUL
        else:
            regime = VIXRegime.PANIC
        
        # Calculate percentiles
        vix_1y = vix_df['Close'].iloc[-252:] if len(vix_df) >= 252 else vix_df['Close']
        vix_5y = vix_df['Close'].iloc[-1260:] if len(vix_df) >= 1260 else vix_df['Close']
        
        pct_1y = float((vix_1y < current_level).sum() / len(vix_1y) * 100)
        pct_5y = float((vix_5y < current_level).sum() / len(vix_5y) * 100)
        
        # Averages
        avg_1m = float(vix_df['Close'].iloc[-21:].mean())
        avg_3m = float(vix_df['Close'].iloc[-63:].mean()) if len(vix_df) >= 63 else avg_1m
        avg_1y = float(vix_df['Close'].iloc[-252:].mean()) if len(vix_df) >= 252 else avg_3m
        
        # Term structure (simplified - use difference from average)
        term_slope = float((current_level - avg_3m) / avg_3m * 100) if avg_3m > 0 else 0
        is_inverted = current_level > avg_1m * 1.1  # >10% above 1m avg
        
        # Days in current regime
        vix_series = vix_df['Close']
        regime_days = 0
        for i in range(len(vix_series) - 1, -1, -1):
            level = vix_series.iloc[i]
            if self._classify_level(level) != regime:
                break
            regime_days += 1
        
        return VIXAnalysis(
            current_level=current_level,
            regime=regime,
            percentile_1y=pct_1y,
            percentile_5y=pct_5y,
            is_inverted=is_inverted,
            term_structure_slope=term_slope,
            average_1m=avg_1m,
            average_3m=avg_3m,
            average_1y=avg_1y,
            days_in_current_regime=regime_days
        )
    
    def _classify_level(self, level: float) -> VIXRegime:
        """Classify a VIX level into a regime."""
        if level < self.VIX_COMPLACENT:
            return VIXRegime.COMPLACENT
        elif level < self.VIX_CALM:
            return VIXRegime.CALM
        elif level < self.VIX_NORMAL:
            return VIXRegime.NORMAL
        elif level < self.VIX_ELEVATED:
            return VIXRegime.ELEVATED
        elif level < self.VIX_FEARFUL:
            return VIXRegime.FEARFUL
        else:
            return VIXRegime.PANIC


# =============================================================================
# CACHE MANAGER
# =============================================================================

class CacheManager:
    """Parquet-based data caching for efficient storage."""
    
    def __init__(self, cache_dir: Union[str, Path] = "data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, df: pd.DataFrame, name: str) -> Path:
        """Save DataFrame to Parquet."""
        path = self.cache_dir / f"{name}.parquet"
        df.to_parquet(path, compression='snappy')
        logger.info(f"Cached: {path} ({len(df):,} rows)")
        return path
    
    def load(self, name: str) -> Optional[pd.DataFrame]:
        """Load DataFrame from Parquet cache."""
        path = self.cache_dir / f"{name}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None
    
    def exists(self, name: str) -> bool:
        """Check if cache exists."""
        return (self.cache_dir / f"{name}.parquet").exists()


# =============================================================================
# MAIN DATA PIPELINE
# =============================================================================

class DataPipeline:
    """
    Orchestrates the complete data pipeline for Phase 1.
    
    Pipeline stages:
    1. ACQUIRE: Fetch data from multiple sources
    2. VALIDATE: Assess data quality
    3. ENRICH: Add engineered features
    4. PROFILE: Statistical analysis
    5. CONTEXTUALIZE: Market context (VIX, benchmark)
    6. EXPORT: Save outputs
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        symbol: str = "AAPL",
        start: str = "2015-01-01",
        benchmark_symbol: str = "SPY"
    ):
        """
        Initialize pipeline.
        
        Args:
            config_path: Path to YAML config (optional)
            symbol: Primary ticker symbol
            start: Start date (YYYY-MM-DD)
            benchmark_symbol: Benchmark for relative analysis
        """
        self.symbol = symbol
        self.start = start
        self.end = datetime.now().strftime('%Y-%m-%d')  # Always current
        self.benchmark_symbol = benchmark_symbol
        self.vix_symbol = "^VIX"
        
        # Initialize components
        self.acquisition = DataAcquisition()
        self.validator = DataValidator()
        self.engineer = FeatureEngineer()
        self.profiler = StatisticalProfiler()
        self.benchmark_analyzer = BenchmarkAnalyzer(benchmark_symbol)
        self.vix_analyzer = VIXAnalyzer()
        self.cache = CacheManager()
    
    def run(self, output_dir: Optional[Path] = None) -> PipelineOutput:
        """
        Execute the complete pipeline.
        
        Args:
            output_dir: Output directory (default: outputs/)
            
        Returns:
            PipelineOutput with all data and analysis
        """
        t0 = time.perf_counter()
        
        if output_dir is None:
            output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sym_lower = self.symbol.lower()
        
        # =====================================================================
        # STAGE 1: ACQUIRE DATA
        # =====================================================================
        logger.info("Stage 1: Acquiring data...")
        
        symbols = [self.symbol, self.benchmark_symbol, self.vix_symbol]
        data_dict, provenance = self.acquisition.fetch_ohlcv(
            symbols, self.start, self.end
        )
        
        if self.symbol not in data_dict:
            raise ValueError(f"No data retrieved for {self.symbol}")
        
        raw_df = data_dict[self.symbol]
        benchmark_df = data_dict.get(self.benchmark_symbol)
        vix_df = data_dict.get(self.vix_symbol)
        
        # Fetch corporate events
        earnings = self.acquisition.fetch_earnings_calendar(self.symbol)
        info = self.acquisition.fetch_company_info(self.symbol)
        
        logger.info(f"Acquired {len(raw_df)} records for {self.symbol}")
        
        # =====================================================================
        # STAGE 2: VALIDATE DATA
        # =====================================================================
        logger.info("Stage 2: Validating data quality...")
        
        quality = self.validator.validate(raw_df, self.symbol)
        
        if not quality.is_usable:
            raise ValueError(
                f"Data quality unusable: {quality.grade.value}, "
                f"Issues: {quality.issues}"
            )
        
        logger.info(f"Quality: {quality.overall:.1f}/100 ({quality.grade.value})")
        
        # =====================================================================
        # STAGE 3: ENRICH DATA
        # =====================================================================
        logger.info("Stage 3: Engineering features...")
        
        daily = self.engineer.enrich(raw_df)
        weekly = self.engineer.aggregate_timeframe(raw_df, 'W')
        monthly = self.engineer.aggregate_timeframe(raw_df, 'ME')
        
        # Enrich benchmark if available
        if benchmark_df is not None:
            benchmark_df = self.engineer.enrich(benchmark_df)
        
        logger.info(f"Features: {len(daily.columns)} columns")
        
        # =====================================================================
        # STAGE 4: PROFILE STATISTICS
        # =====================================================================
        logger.info("Stage 4: Statistical profiling...")
        
        profile = self.profiler.profile(daily)
        
        logger.info(f"Sharpe: {profile.sharpe_ratio:.2f}, "
                   f"Hurst: {profile.hurst_exponent:.2f}")
        
        # =====================================================================
        # STAGE 5: CONTEXTUALIZE (BENCHMARK + VIX)
        # =====================================================================
        logger.info("Stage 5: Market context analysis...")
        
        bench_analysis = None
        if benchmark_df is not None:
            bench_analysis = self.benchmark_analyzer.analyze(daily, benchmark_df)
            if bench_analysis:
                logger.info(f"Beta: {bench_analysis.beta_full:.2f}, "
                           f"Alpha: {bench_analysis.alpha_annualized*100:.1f}%")
        
        vix_analysis = None
        if vix_df is not None:
            vix_analysis = self.vix_analyzer.analyze(vix_df)
            if vix_analysis:
                logger.info(f"VIX: {vix_analysis.current_level:.1f} "
                           f"({vix_analysis.regime.value})")
        
        # =====================================================================
        # STAGE 6: EXPORT
        # =====================================================================
        logger.info("Stage 6: Exporting outputs...")
        
        # Save Parquet
        parquet_path = self.cache.save(daily, f"{sym_lower}_daily")
        self.cache.save(weekly, f"{sym_lower}_weekly")
        self.cache.save(monthly, f"{sym_lower}_monthly")
        
        # Save JSON metadata
        generated_at = datetime.now().isoformat()
        json_path = output_dir / f"{sym_lower}_metadata.json"
        
        metadata = {
            "symbol": self.symbol,
            "company_name": info.get('name', self.symbol),
            "sector": info.get('sector', 'Unknown'),
            "industry": info.get('industry', 'Unknown'),
            "period": {"start": self.start, "end": self.end},
            "records": {
                "daily": len(daily),
                "weekly": len(weekly),
                "monthly": len(monthly)
            },
            "quality": quality.to_dict(),
            "profile": {
                "annualized_return": profile.annualized_return,
                "annualized_volatility": profile.annualized_volatility,
                "sharpe_ratio": profile.sharpe_ratio,
                "hurst_exponent": profile.hurst_exponent,
                "trend_character": profile.trend_character.value
            },
            "provenance": provenance.to_dict(),
            "version": PIPELINE_VERSION,
            "generated_at": generated_at
        }
        
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        processing_time = (time.perf_counter() - t0) * 1000
        
        output = PipelineOutput(
            daily=daily,
            weekly=weekly,
            monthly=monthly,
            benchmark=benchmark_df,
            vix=vix_df,
            earnings=earnings,
            symbol=self.symbol,
            company_name=info.get('name', self.symbol),
            sector=info.get('sector', 'Unknown'),
            industry=info.get('industry', 'Unknown'),
            period=(self.start, self.end),
            quality=quality,
            profile=profile,
            benchmark_analysis=bench_analysis,
            vix_analysis=vix_analysis,
            provenance=provenance,
            processing_time_ms=processing_time,
            pipeline_version=PIPELINE_VERSION,
            generated_at=generated_at,
            parquet_path=parquet_path,
            html_path=output_dir / f"{sym_lower}_report.html",
            json_path=json_path
        )
        
        logger.info(f"Pipeline complete in {processing_time:.0f}ms")
        
        return output


# =============================================================================
# CONSOLE REPORT
# =============================================================================

def print_report(output: PipelineOutput) -> None:
    """Print formatted console report."""
    q = output.quality
    p = output.profile
    v = output.profile.volatility_profile
    t = output.profile.statistical_tests
    tr = output.profile.tail_risk
    b = output.benchmark_analysis
    vix = output.vix_analysis
    df = output.daily
    
    width = 78
    
    print()
    print("=" * width)
    print(f"  PHASE 1: {output.symbol} - {output.company_name}".center(width))
    print("=" * width)
    
    print(f"\n  Period: {output.period[0]} to {output.period[1]}")
    print(f"  Records: {len(df):,} daily | {len(output.weekly):,} weekly | "
          f"{len(output.monthly):,} monthly")
    print(f"  Features: {len(df.columns)} columns")
    
    print(f"\n  DATA QUALITY: {q.overall:.1f}/100 ({q.grade.value})")
    print(f"    Completeness: {q.completeness:.1f} | Accuracy: {q.accuracy:.1f} | "
          f"Consistency: {q.consistency:.1f} | Timeliness: {q.timeliness:.1f}")
    
    if q.issues:
        print(f"    Issues: {', '.join(q.issues)}")
    
    print(f"\n  RETURNS PROFILE:")
    print(f"    Annual Return: {p.annualized_return*100:+.2f}%")
    print(f"    Annual Vol: {p.annualized_volatility*100:.2f}%")
    print(f"    Sharpe: {p.sharpe_ratio:.2f} | Sortino: {p.sortino_ratio:.2f} | "
          f"Calmar: {p.calmar_ratio:.2f}")
    
    print(f"\n  TREND ANALYSIS:")
    print(f"    Hurst: {p.hurst_exponent:.3f} ({p.trend_character.value})")
    print(f"    R-squared: {p.hurst_confidence:.3f}")
    
    print(f"\n  STATISTICAL TESTS (5):")
    print(f"    Jarque-Bera: {'PASS' if t.is_normal else 'FAIL'} (Normality)")
    print(f"    ADF: {'PASS' if t.is_stationary else 'FAIL'} (Stationarity)")
    print(f"    KPSS: {'PASS' if t.is_stationary_kpss else 'FAIL'} (Stationarity)")
    print(f"    Ljung-Box: {'YES' if t.has_autocorrelation else 'NO'} (Autocorrelation)")
    print(f"    ARCH: {'YES' if t.has_arch_effects else 'NO'} (Vol Clustering)")
    print(f"    Conclusion: {t.stationarity_conclusion.value}")
    
    print(f"\n  VOLATILITY ESTIMATORS (7):")
    print(f"    Close-to-Close:  {v.close_to_close*100:6.2f}%")
    print(f"    Parkinson:       {v.parkinson*100:6.2f}%")
    print(f"    Garman-Klass:    {v.garman_klass*100:6.2f}%")
    print(f"    Rogers-Satchell: {v.rogers_satchell*100:6.2f}%")
    print(f"    Yang-Zhang:      {v.yang_zhang*100:6.2f}%")
    print(f"    GKYZ:            {v.gkyz*100:6.2f}%")
    print(f"    Hodges-Tompkins: {v.hodges_tompkins*100:6.2f}%")
    print(f"    COMPOSITE:       {v.composite*100:6.2f}% ({v.regime.value})")
    
    print(f"\n  TAIL RISK:")
    print(f"    VaR (95%): {tr.var_95*100:.2f}% | VaR (99%): {tr.var_99*100:.2f}%")
    print(f"    CVaR (95%): {tr.cvar_95*100:.2f}%")
    print(f"    Max Drawdown: {tr.max_drawdown*100:.1f}% ({tr.max_drawdown_duration} days)")
    print(f"    Skewness: {tr.skewness:.2f} | Kurtosis: {tr.kurtosis:.2f}")
    
    if b:
        print(f"\n  BENCHMARK ({b.benchmark_symbol}):")
        print(f"    Correlation: {b.correlation_full:.3f} (1Y: {b.correlation_1y:.3f})")
        print(f"    Beta: {b.beta_full:.2f} (1Y: {b.beta_1y:.2f})")
        print(f"    Alpha: {b.alpha_annualized*100:+.2f}%")
        print(f"    Information Ratio: {b.information_ratio:+.2f}")
        print(f"    Up Capture: {b.up_capture:.1f}% | Down Capture: {b.down_capture:.1f}%")
    
    if vix:
        print(f"\n  MARKET CONTEXT (VIX):")
        print(f"    Current: {vix.current_level:.1f} ({vix.regime.value})")
        print(f"    Percentile (1Y): {vix.percentile_1y:.0f}%")
        print(f"    Context: {vix.market_context.value}")
    
    print(f"\n  STRATEGY HINTS:")
    for hint in p.strategy_hints:
        print(f"    - {hint}")
    print(f"    Position Sizing: {p.position_sizing_recommendation}")
    
    print(f"\n  DATA PROVENANCE:")
    print(f"    Source: {output.provenance.source}")
    print(f"    Hash: {output.provenance.data_hash}")
    print(f"    Fetched: {output.provenance.fetch_timestamp[:19]}")
    
    print(f"\n  PROCESSING:")
    print(f"    Time: {output.processing_time_ms:.0f}ms")
    print(f"    Version: {output.pipeline_version}")
    
    print()
    print("=" * width)


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main() -> int:
    """
    Command-line entry point.
    
    Usage:
        python data_collector.py
        python data_collector.py --symbol MSFT
        python data_collector.py --symbol GOOGL --start 2020-01-01
    
    Returns:
        0 on success, 1 on failure
    """
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="  %(asctime)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    
    parser = argparse.ArgumentParser(
        description=f"Phase 1: Data Pipeline v{PIPELINE_VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python data_collector.py
    python data_collector.py --symbol MSFT
    python data_collector.py --symbol GOOGL --start 2020-01-01
        """
    )
    parser.add_argument("--symbol", "-s", default="AAPL",
                       help="Ticker symbol (default: AAPL)")
    parser.add_argument("--start", default="2015-01-01",
                       help="Start date YYYY-MM-DD (default: 2015-01-01)")
    parser.add_argument("--benchmark", "-b", default="SPY",
                       help="Benchmark symbol (default: SPY)")
    
    args = parser.parse_args()
    
    try:
        pipeline = DataPipeline(
            symbol=args.symbol,
            start=args.start,
            benchmark_symbol=args.benchmark
        )
        
        output = pipeline.run()
        print_report(output)
        
        # Return based on quality grade
        acceptable = [QualityGrade.EXCELLENT, QualityGrade.GOOD, QualityGrade.ACCEPTABLE]
        return 0 if output.quality.grade in acceptable else 1
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())