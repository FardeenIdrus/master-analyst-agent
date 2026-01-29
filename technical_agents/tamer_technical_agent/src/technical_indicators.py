"""
Institutional-Grade Technical Indicator Engine for Quantitative Analysis

MSc AI Agents in Asset Management - Track B: Technical Analyst Agent
Phase 2: Technical Indicator Computation, Signal Generation, and Confluence Analysis

INDICATOR ARCHITECTURE
    This module implements a comprehensive technical analysis framework organized
    into five indicator families, each providing distinct analytical perspectives:
    
    Family 1 - MOMENTUM OSCILLATORS
        Measure the velocity and magnitude of price movements to identify
        overbought/oversold conditions and potential reversal points.
        
        - RSI (Relative Strength Index): Wilder's momentum oscillator [0-100]
        - Stochastic Oscillator: Lane's %K/%D momentum system
        - Williams %R: Momentum oscillator [-100 to 0]
        - ROC (Rate of Change): Price momentum as percentage
        
    Family 2 - TREND INDICATORS
        Identify the direction and strength of prevailing trends to align
        trading positions with the dominant market direction.
        
        - MACD: Moving Average Convergence Divergence with histogram
        - ADX/DMI: Average Directional Index with +DI/-DI
        - Aroon: Trend strength and direction [0-100]
        - Supertrend: Volatility-based trend following
        
    Family 3 - VOLATILITY SYSTEMS
        Measure price dispersion and construct dynamic support/resistance
        bands that adapt to changing market conditions.
        
        - Bollinger Bands: Mean reversion with 2 standard deviation bands
        - Keltner Channels: ATR-based volatility bands
        - Donchian Channels: Breakout system with N-period highs/lows
        
    Family 4 - VOLUME ANALYSIS
        Confirm price movements through volume patterns and identify
        accumulation/distribution dynamics.
        
        - OBV (On-Balance Volume): Cumulative volume flow
        - CMF (Chaikin Money Flow): Volume-weighted price position
        - MFI (Money Flow Index): Volume-weighted RSI
        - VWMA (Volume Weighted Moving Average): Price-volume integration
        
    Family 5 - COMPLETE TRADING SYSTEMS
        Self-contained trading frameworks that provide multiple signals
        from a unified analytical approach.
        
        - Ichimoku Kinko Hyo: Japanese equilibrium system (5 components)

SIGNAL GENERATION FRAMEWORK
    Each indicator produces standardized signals with:
    - Direction: STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
    - Confidence: 0.0 to 1.0 based on signal strength
    - Regime adjustment: Thresholds adapt to volatility conditions
    
REGIME-AWARE ANALYSIS
    Integration with Phase 1 market profile enables dynamic threshold
    adjustment based on volatility regime:
    - LOW volatility: Tighter bands, stricter overbought/oversold
    - NORMAL volatility: Standard parameters
    - HIGH volatility: Wider bands, extended extremes
    - EXTREME volatility: Maximum tolerance, trend-following priority

DIVERGENCE DETECTION
    Automated identification of price-indicator divergences:
    - Regular Bullish: Price lower low, indicator higher low
    - Regular Bearish: Price higher high, indicator lower high
    - Hidden Bullish: Price higher low, indicator lower low
    - Hidden Bearish: Price lower high, indicator higher high

MULTI-TIMEFRAME CONFLUENCE
    Signal aggregation across daily, weekly, and monthly timeframes
    with weighted scoring to identify high-probability setups.

Author: Tamer
Course: MSc AI Agents in Asset Management (IFTE0001)
Version: 2.0.0
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import argrelextrema

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
pd.set_option('future.no_silent_downcasting', True)

# Module-level logger
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Trading calendar
TRADING_DAYS_YEAR: int = 252
TRADING_DAYS_QUARTER: int = 63
TRADING_DAYS_MONTH: int = 21
TRADING_DAYS_WEEK: int = 5

# RSI parameters
RSI_PERIOD: int = 14
RSI_OVERBOUGHT: float = 70.0
RSI_OVERSOLD: float = 30.0
RSI_EXTREME_OB: float = 80.0
RSI_EXTREME_OS: float = 20.0

# Stochastic parameters
STOCH_K_PERIOD: int = 14
STOCH_D_PERIOD: int = 3
STOCH_SMOOTH: int = 3
STOCH_OVERBOUGHT: float = 80.0
STOCH_OVERSOLD: float = 20.0

# Williams %R parameters
WILLIAMS_PERIOD: int = 14
WILLIAMS_OVERBOUGHT: float = -20.0
WILLIAMS_OVERSOLD: float = -80.0

# MACD parameters
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9

# ADX parameters
ADX_PERIOD: int = 14
ADX_STRONG_TREND: float = 25.0
ADX_VERY_STRONG: float = 50.0

# Bollinger Bands parameters
BB_PERIOD: int = 20
BB_STD_DEV: float = 2.0

# Keltner Channel parameters
KC_PERIOD: int = 20
KC_ATR_MULT: float = 2.0

# Ichimoku parameters (traditional settings)
ICHIMOKU_TENKAN: int = 9
ICHIMOKU_KIJUN: int = 26
ICHIMOKU_SENKOU_B: int = 52
ICHIMOKU_DISPLACEMENT: int = 26

# Aroon parameters
AROON_PERIOD: int = 25

# Supertrend parameters
SUPERTREND_PERIOD: int = 10
SUPERTREND_MULTIPLIER: float = 3.0

# OBV/CMF/MFI parameters
MFI_PERIOD: int = 14
CMF_PERIOD: int = 20

# Divergence detection
DIVERGENCE_LOOKBACK: int = 14
DIVERGENCE_MIN_BARS: int = 5

# Signal weights for confluence
# Ichimoku is a complete trading system and deserves significant weight
# Momentum is critical for timing entries/exits
# Trend confirms direction
# Volatility and Volume provide confirmation
MOMENTUM_WEIGHT: float = 0.25
TREND_WEIGHT: float = 0.25
VOLATILITY_WEIGHT: float = 0.15
VOLUME_WEIGHT: float = 0.15
SYSTEM_WEIGHT: float = 0.20  # Ichimoku is a complete system

# Module version
INDICATOR_VERSION: str = "2.0.0"


# =============================================================================
# ENUMERATIONS
# =============================================================================

class SignalDirection(Enum):
    """Trading signal direction with numeric mapping for aggregation."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"
    
    @property
    def numeric(self) -> float:
        """Convert to numeric value for mathematical operations."""
        mapping = {
            SignalDirection.STRONG_BUY: 1.0,
            SignalDirection.BUY: 0.5,
            SignalDirection.NEUTRAL: 0.0,
            SignalDirection.SELL: -0.5,
            SignalDirection.STRONG_SELL: -1.0
        }
        return mapping[self]
    
    @classmethod
    def from_numeric(cls, value: float) -> 'SignalDirection':
        """
        Convert numeric value back to signal direction.
        
        Thresholds are calibrated so that:
        - Strong signals require clear consensus (>= 0.55 or <= -0.55)
        - Regular signals require moderate lean (>= 0.18 or <= -0.18)
        - Neutral is reserved for truly mixed signals
        """
        if value >= 0.55:
            return cls.STRONG_BUY
        elif value >= 0.18:
            return cls.BUY
        elif value > -0.18:
            return cls.NEUTRAL
        elif value > -0.55:
            return cls.SELL
        else:
            return cls.STRONG_SELL


class SignalStrength(Enum):
    """Signal quality classification based on confluence and confirmation."""
    VERY_STRONG = "VERY_STRONG"   # Multiple confirmations, high confidence
    STRONG = "STRONG"             # Good confirmation
    MODERATE = "MODERATE"         # Some confirmation
    WEAK = "WEAK"                 # Limited confirmation
    VERY_WEAK = "VERY_WEAK"       # Single indicator, no confirmation
    
    @property
    def min_confidence(self) -> float:
        """Minimum confidence threshold for this strength level."""
        return {
            SignalStrength.VERY_STRONG: 0.85,
            SignalStrength.STRONG: 0.70,
            SignalStrength.MODERATE: 0.55,
            SignalStrength.WEAK: 0.40,
            SignalStrength.VERY_WEAK: 0.0
        }[self]


class TrendState(Enum):
    """Market trend classification."""
    STRONG_UPTREND = "STRONG_UPTREND"
    UPTREND = "UPTREND"
    SIDEWAYS = "SIDEWAYS"
    DOWNTREND = "DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"


class VolatilityRegime(Enum):
    """Volatility regime for threshold adjustment."""
    LOW = "LOW"           # Annualized vol < 15%
    NORMAL = "NORMAL"     # 15-25%
    HIGH = "HIGH"         # 25-40%
    EXTREME = "EXTREME"   # > 40%


class MomentumZone(Enum):
    """Momentum oscillator zone classification."""
    EXTREME_OVERBOUGHT = "EXTREME_OVERBOUGHT"
    OVERBOUGHT = "OVERBOUGHT"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    OVERSOLD = "OVERSOLD"
    EXTREME_OVERSOLD = "EXTREME_OVERSOLD"


class DivergenceType(Enum):
    """Price-indicator divergence classification."""
    REGULAR_BULLISH = "REGULAR_BULLISH"     # Price LL, indicator HL - reversal
    REGULAR_BEARISH = "REGULAR_BEARISH"     # Price HH, indicator LH - reversal
    HIDDEN_BULLISH = "HIDDEN_BULLISH"       # Price HL, indicator LL - continuation
    HIDDEN_BEARISH = "HIDDEN_BEARISH"       # Price LH, indicator HH - continuation
    NONE = "NONE"


class BandPosition(Enum):
    """Price position relative to volatility bands."""
    ABOVE_UPPER = "ABOVE_UPPER"       # Overbought / breakout
    UPPER_ZONE = "UPPER_ZONE"         # Near upper band
    MIDDLE_UPPER = "MIDDLE_UPPER"     # Between middle and upper
    AT_MIDDLE = "AT_MIDDLE"           # Near middle band
    MIDDLE_LOWER = "MIDDLE_LOWER"     # Between middle and lower
    LOWER_ZONE = "LOWER_ZONE"         # Near lower band
    BELOW_LOWER = "BELOW_LOWER"       # Oversold / breakdown


class IchimokuSignal(Enum):
    """Ichimoku trading signal classification."""
    STRONG_BULLISH = "STRONG_BULLISH"   # All 5 conditions bullish
    BULLISH = "BULLISH"                 # Most conditions bullish
    NEUTRAL = "NEUTRAL"                 # Mixed signals
    BEARISH = "BEARISH"                 # Most conditions bearish
    STRONG_BEARISH = "STRONG_BEARISH"   # All 5 conditions bearish


class CrossoverType(Enum):
    """Line crossover classification."""
    BULLISH_CROSS = "BULLISH_CROSS"     # Fast crosses above slow
    BEARISH_CROSS = "BEARISH_CROSS"     # Fast crosses below slow
    BULLISH_ABOVE = "BULLISH_ABOVE"     # Fast above slow, no cross
    BEARISH_BELOW = "BEARISH_BELOW"     # Fast below slow, no cross
    CONVERGING = "CONVERGING"           # Lines approaching
    NONE = "NONE"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class IndicatorSignal:
    """
    Standardized signal output from any indicator.
    
    Provides a unified interface for all indicator signals, enabling
    consistent aggregation and comparison across different indicator types.
    """
    indicator_name: str
    direction: SignalDirection
    confidence: float                        # 0.0 to 1.0
    value: float                             # Raw indicator value
    zone: str                                # Current zone/state
    factors: List[str] = field(default_factory=list)  # Contributing factors
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate and set timestamp."""
        self.confidence = max(0.0, min(1.0, self.confidence))
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class DivergenceSignal:
    """
    Detected divergence between price and indicator.
    
    Divergences are powerful reversal/continuation signals that occur when
    price action and indicator readings move in opposite directions.
    """
    divergence_type: DivergenceType
    indicator_name: str
    start_date: datetime
    end_date: datetime
    price_start: float
    price_end: float
    indicator_start: float
    indicator_end: float
    strength: float                          # 0.0 to 1.0
    bars_duration: int


@dataclass
class CrossoverSignal:
    """
    Line crossover detection result.
    
    Crossovers between fast and slow moving averages or indicator lines
    are classic trading signals.
    """
    crossover_type: CrossoverType
    fast_value: float
    slow_value: float
    bars_since_cross: int
    cross_price: Optional[float] = None


@dataclass
class BandAnalysis:
    """
    Analysis of price position within volatility bands.
    
    Tracks price location relative to upper, middle, and lower bands
    for mean reversion and breakout strategies.
    """
    position: BandPosition
    upper_band: float
    middle_band: float
    lower_band: float
    bandwidth: float                         # (upper - lower) / middle
    percent_b: float                         # (price - lower) / (upper - lower)
    squeeze: bool                            # Bandwidth below threshold


@dataclass
class TrendAnalysis:
    """
    Comprehensive trend assessment from multiple indicators.
    
    Aggregates trend signals from ADX, moving averages, and other
    trend-following indicators.
    """
    state: TrendState
    strength: float                          # 0.0 to 1.0
    adx_value: float
    plus_di: float
    minus_di: float
    trend_duration_bars: int
    ma_alignment: str                        # "bullish", "bearish", "mixed"


@dataclass
class IchimokuAnalysis:
    """
    Complete Ichimoku Kinko Hyo analysis.
    
    The Ichimoku system provides trend direction, support/resistance levels,
    and momentum in a single glance.
    """
    signal: IchimokuSignal
    tenkan_sen: float                        # Conversion line
    kijun_sen: float                         # Base line
    senkou_span_a: float                     # Leading span A
    senkou_span_b: float                     # Leading span B
    chikou_span: float                       # Lagging span
    cloud_top: float
    cloud_bottom: float
    cloud_color: str                         # "bullish" or "bearish"
    price_vs_cloud: str                      # "above", "inside", "below"
    tk_cross: CrossoverType
    price_vs_kijun: str                      # "above", "below"
    chikou_vs_price: str                     # "above", "below"
    bullish_signals: int                     # Count of bullish conditions
    bearish_signals: int                     # Count of bearish conditions


@dataclass
class IndicatorFamily:
    """
    Collection of related indicators with aggregated signal.
    
    Groups indicators by analytical purpose (momentum, trend, etc.)
    for organized signal generation.
    """
    family_name: str
    indicators: Dict[str, IndicatorSignal]
    aggregate_signal: SignalDirection
    aggregate_confidence: float
    weight: float


@dataclass
class ConfluenceAnalysis:
    """
    Multi-indicator confluence assessment.
    
    Aggregates signals across all indicator families to provide
    a unified trading recommendation.
    """
    overall_signal: SignalDirection
    overall_confidence: float
    signal_strength: SignalStrength
    families: Dict[str, IndicatorFamily]
    bullish_count: int
    bearish_count: int
    neutral_count: int
    divergences: List[DivergenceSignal]
    key_levels: Dict[str, float]
    recommendation: str
    risk_factors: List[str]


@dataclass
class TimeframeSignals:
    """
    Signals aggregated across multiple timeframes.
    
    Higher timeframe alignment increases signal reliability.
    """
    daily: ConfluenceAnalysis
    weekly: Optional[ConfluenceAnalysis]
    monthly: Optional[ConfluenceAnalysis]
    alignment: str                           # "aligned", "mixed", "conflicting"
    alignment_score: float                   # 0.0 to 1.0


@dataclass
class IndicatorOutput:
    """
    Complete output from the indicator engine.
    
    Contains all computed indicators, signals, and analysis
    for use in backtesting and trade generation.
    """
    # Raw indicator DataFrames
    indicators_df: pd.DataFrame              # All indicator values
    signals_df: pd.DataFrame                 # All signals over time
    
    # Current state analysis
    current_analysis: ConfluenceAnalysis
    
    # Multi-timeframe (if available)
    timeframe_analysis: Optional[TimeframeSignals]
    
    # Metadata
    symbol: str
    period: Tuple[str, str]
    volatility_regime: VolatilityRegime
    regime_adjusted: bool
    generated_at: str
    version: str


# =============================================================================
# REGIME-AWARE THRESHOLD ADJUSTMENT
# =============================================================================

class RegimeAdjuster:
    """
    Adjusts indicator thresholds based on volatility regime.
    
    In high volatility environments, traditional overbought/oversold levels
    are less meaningful. This class provides regime-appropriate thresholds
    that adapt to market conditions.
    
    Research basis:
    - Chande & Kroll (1994): Dynamic thresholds for momentum indicators
    - Kaufman (2013): Adaptive trading systems
    """
    
    # Regime multipliers for threshold adjustment
    REGIME_MULTIPLIERS = {
        VolatilityRegime.LOW: 0.85,      # Tighter thresholds
        VolatilityRegime.NORMAL: 1.00,   # Standard thresholds
        VolatilityRegime.HIGH: 1.15,     # Wider thresholds
        VolatilityRegime.EXTREME: 1.30   # Maximum tolerance
    }
    
    def __init__(self, regime: VolatilityRegime = VolatilityRegime.NORMAL):
        """
        Initialize with volatility regime.
        
        Parameters
        ----------
        regime : VolatilityRegime
            Current market volatility regime from Phase 1 analysis
        """
        self.regime = regime
        self.multiplier = self.REGIME_MULTIPLIERS[regime]
    
    def adjust_rsi_thresholds(self) -> Tuple[float, float, float, float]:
        """
        Adjust RSI overbought/oversold thresholds.
        
        Returns
        -------
        Tuple[float, float, float, float]
            (oversold, overbought, extreme_oversold, extreme_overbought)
        """
        # In high vol, RSI can stay extreme longer - widen thresholds
        spread = (RSI_OVERBOUGHT - RSI_OVERSOLD) / 2
        adjusted_spread = spread * self.multiplier
        
        center = 50.0
        oversold = max(10.0, center - adjusted_spread)
        overbought = min(90.0, center + adjusted_spread)
        
        extreme_spread = (RSI_EXTREME_OB - RSI_EXTREME_OS) / 2
        adjusted_extreme = extreme_spread * self.multiplier
        extreme_os = max(5.0, center - adjusted_extreme)
        extreme_ob = min(95.0, center + adjusted_extreme)
        
        return oversold, overbought, extreme_os, extreme_ob
    
    def adjust_stochastic_thresholds(self) -> Tuple[float, float]:
        """
        Adjust Stochastic overbought/oversold thresholds.
        
        Returns
        -------
        Tuple[float, float]
            (oversold, overbought)
        """
        spread = (STOCH_OVERBOUGHT - STOCH_OVERSOLD) / 2
        adjusted_spread = spread * self.multiplier
        
        center = 50.0
        oversold = max(10.0, center - adjusted_spread)
        overbought = min(90.0, center + adjusted_spread)
        
        return oversold, overbought
    
    def adjust_williams_thresholds(self) -> Tuple[float, float]:
        """
        Adjust Williams %R thresholds.
        
        Returns
        -------
        Tuple[float, float]
            (oversold, overbought) - note Williams %R is inverted
        """
        # Williams %R: -100 to 0, inverted scale
        spread = abs(WILLIAMS_OVERBOUGHT - WILLIAMS_OVERSOLD) / 2
        adjusted_spread = spread * self.multiplier
        
        center = -50.0
        overbought = min(-5.0, center + adjusted_spread)
        oversold = max(-95.0, center - adjusted_spread)
        
        return oversold, overbought
    
    def adjust_adx_threshold(self) -> float:
        """
        Adjust ADX strong trend threshold.
        
        Returns
        -------
        float
            Adjusted threshold for strong trend identification
        """
        # In low vol, trends are weaker - lower threshold
        # In high vol, need stronger ADX to confirm trend
        return ADX_STRONG_TREND * self.multiplier
    
    def adjust_bollinger_std(self) -> float:
        """
        Adjust Bollinger Band standard deviation multiplier.
        
        Returns
        -------
        float
            Adjusted standard deviation multiplier
        """
        # In high vol, use wider bands to avoid false signals
        return BB_STD_DEV * self.multiplier
    
    def get_regime_description(self) -> str:
        """
        Get human-readable regime description.
        
        Returns
        -------
        str
            Description of current regime and its implications
        """
        descriptions = {
            VolatilityRegime.LOW: (
                "Low volatility environment. Indicators more sensitive, "
                "tighter thresholds applied. Mean reversion strategies favored."
            ),
            VolatilityRegime.NORMAL: (
                "Normal volatility environment. Standard thresholds applied. "
                "Both trend-following and mean reversion viable."
            ),
            VolatilityRegime.HIGH: (
                "High volatility environment. Wider thresholds to avoid "
                "whipsaws. Trend-following strategies favored."
            ),
            VolatilityRegime.EXTREME: (
                "Extreme volatility environment. Maximum threshold tolerance. "
                "Focus on trend direction, reduce position sizes."
            )
        }
        return descriptions[self.regime]


# =============================================================================
# MOMENTUM OSCILLATORS
# =============================================================================

class MomentumIndicators:
    """
    Momentum oscillator calculations and signal generation.
    
    Momentum indicators measure the rate of price change to identify
    overbought/oversold conditions and potential reversals.
    
    Indicators implemented:
    - RSI (Relative Strength Index): Wilder, 1978
    - Stochastic Oscillator: Lane, 1950s
    - Williams %R: Williams, 1966
    - ROC (Rate of Change): Simple momentum
    """
    
    def __init__(self, adjuster: RegimeAdjuster):
        """
        Initialize with regime adjuster for dynamic thresholds.
        
        Parameters
        ----------
        adjuster : RegimeAdjuster
            Threshold adjuster based on volatility regime
        """
        self.adjuster = adjuster
        
        # Get regime-adjusted thresholds
        self.rsi_os, self.rsi_ob, self.rsi_extreme_os, self.rsi_extreme_ob = \
            adjuster.adjust_rsi_thresholds()
        self.stoch_os, self.stoch_ob = adjuster.adjust_stochastic_thresholds()
        self.williams_os, self.williams_ob = adjuster.adjust_williams_thresholds()
    
    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
        """
        Calculate Relative Strength Index using Wilder's smoothing.
        
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        
        Parameters
        ----------
        close : pd.Series
            Closing prices
        period : int
            Lookback period (default: 14)
            
        Returns
        -------
        pd.Series
            RSI values [0, 100]
        """
        delta = close.diff()
        
        gains = delta.where(delta > 0, 0.0)
        losses = (-delta).where(delta < 0, 0.0)
        
        # Wilder's smoothing (exponential with alpha = 1/period)
        alpha = 1.0 / period
        avg_gain = gains.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        avg_loss = losses.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi.fillna(50.0)
    
    @staticmethod
    def calculate_stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = STOCH_K_PERIOD,
        d_period: int = STOCH_D_PERIOD,
        smooth: int = STOCH_SMOOTH
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator (%K and %D).
        
        %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
        %D = SMA(%K, d_period)
        
        Parameters
        ----------
        high, low, close : pd.Series
            OHLC data
        k_period : int
            Lookback for highest high / lowest low
        d_period : int
            Smoothing period for %D
        smooth : int
            Initial smoothing for %K (slow stochastic)
            
        Returns
        -------
        Tuple[pd.Series, pd.Series]
            (%K, %D) both in range [0, 100]
        """
        lowest_low = low.rolling(window=k_period, min_periods=1).min()
        highest_high = high.rolling(window=k_period, min_periods=1).max()
        
        range_hl = highest_high - lowest_low
        range_hl = range_hl.replace(0, np.nan)
        
        # Fast %K
        fast_k = 100.0 * (close - lowest_low) / range_hl
        
        # Slow %K (smoothed)
        slow_k = fast_k.rolling(window=smooth, min_periods=1).mean()
        
        # %D (signal line)
        slow_d = slow_k.rolling(window=d_period, min_periods=1).mean()
        
        return slow_k.fillna(50.0), slow_d.fillna(50.0)
    
    @staticmethod
    def calculate_williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = WILLIAMS_PERIOD
    ) -> pd.Series:
        """
        Calculate Williams %R.
        
        %R = -100 * (Highest High - Close) / (Highest High - Lowest Low)
        
        Parameters
        ----------
        high, low, close : pd.Series
            OHLC data
        period : int
            Lookback period
            
        Returns
        -------
        pd.Series
            Williams %R values [-100, 0]
        """
        highest_high = high.rolling(window=period, min_periods=1).max()
        lowest_low = low.rolling(window=period, min_periods=1).min()
        
        range_hl = highest_high - lowest_low
        range_hl = range_hl.replace(0, np.nan)
        
        williams_r = -100.0 * (highest_high - close) / range_hl
        
        return williams_r.fillna(-50.0)
    
    @staticmethod
    def calculate_roc(close: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate Rate of Change (momentum).
        
        ROC = 100 * (Close - Close[n]) / Close[n]
        
        Parameters
        ----------
        close : pd.Series
            Closing prices
        period : int
            Lookback period
            
        Returns
        -------
        pd.Series
            ROC as percentage
        """
        roc = 100.0 * (close - close.shift(period)) / close.shift(period)
        return roc.fillna(0.0)
    
    def classify_rsi_zone(self, rsi: float) -> MomentumZone:
        """
        Classify RSI value into momentum zone.
        
        Parameters
        ----------
        rsi : float
            Current RSI value
            
        Returns
        -------
        MomentumZone
            Zone classification
        """
        if rsi >= self.rsi_extreme_ob:
            return MomentumZone.EXTREME_OVERBOUGHT
        elif rsi >= self.rsi_ob:
            return MomentumZone.OVERBOUGHT
        elif rsi > 50:
            return MomentumZone.BULLISH
        elif rsi > self.rsi_os:
            if rsi >= 45:
                return MomentumZone.NEUTRAL
            else:
                return MomentumZone.BEARISH
        elif rsi > self.rsi_extreme_os:
            return MomentumZone.OVERSOLD
        else:
            return MomentumZone.EXTREME_OVERSOLD
    
    def generate_rsi_signal(
        self,
        rsi: pd.Series,
        close: pd.Series
    ) -> IndicatorSignal:
        """
        Generate trading signal from RSI.
        
        Parameters
        ----------
        rsi : pd.Series
            RSI values
        close : pd.Series
            Close prices for divergence check
            
        Returns
        -------
        IndicatorSignal
            RSI-based trading signal
        """
        current_rsi = rsi.iloc[-1]
        zone = self.classify_rsi_zone(current_rsi)
        
        factors = []
        
        # Determine direction and confidence
        # Extreme readings warrant higher confidence as they rarely fail
        if zone == MomentumZone.EXTREME_OVERSOLD:
            direction = SignalDirection.STRONG_BUY
            confidence = 0.90  # Extreme readings are highly reliable
            factors.append(f"RSI extremely oversold at {current_rsi:.1f} - high probability reversal zone")
        elif zone == MomentumZone.OVERSOLD:
            direction = SignalDirection.BUY
            # Boost confidence for deeper oversold readings
            # RSI below 26 is significantly oversold in any regime
            if current_rsi <= 26:
                confidence = 0.85
                factors.append(f"RSI deeply oversold at {current_rsi:.1f}")
            else:
                confidence = 0.75
                factors.append(f"RSI oversold at {current_rsi:.1f}")
        elif zone == MomentumZone.EXTREME_OVERBOUGHT:
            direction = SignalDirection.STRONG_SELL
            confidence = 0.90
            factors.append(f"RSI extremely overbought at {current_rsi:.1f} - high probability reversal zone")
        elif zone == MomentumZone.OVERBOUGHT:
            direction = SignalDirection.SELL
            if current_rsi > 75:
                confidence = 0.85
                factors.append(f"RSI deeply overbought at {current_rsi:.1f}")
            else:
                confidence = 0.75
                factors.append(f"RSI overbought at {current_rsi:.1f}")
        else:
            direction = SignalDirection.NEUTRAL
            confidence = 0.50
            factors.append(f"RSI neutral at {current_rsi:.1f}")
        
        # Check for RSI direction (momentum)
        if len(rsi) >= 5:
            rsi_change = rsi.iloc[-1] - rsi.iloc[-5]
            if abs(rsi_change) > 10:
                if rsi_change > 0:
                    factors.append("RSI momentum rising")
                    if direction == SignalDirection.BUY:
                        confidence += 0.05
                else:
                    factors.append("RSI momentum falling")
                    if direction == SignalDirection.SELL:
                        confidence += 0.05
        
        return IndicatorSignal(
            indicator_name="RSI",
            direction=direction,
            confidence=min(1.0, confidence),
            value=current_rsi,
            zone=zone.value,
            factors=factors
        )
    
    def generate_stochastic_signal(
        self,
        stoch_k: pd.Series,
        stoch_d: pd.Series
    ) -> IndicatorSignal:
        """
        Generate trading signal from Stochastic Oscillator.
        
        Parameters
        ----------
        stoch_k : pd.Series
            %K values
        stoch_d : pd.Series
            %D values
            
        Returns
        -------
        IndicatorSignal
            Stochastic-based trading signal
        """
        k = stoch_k.iloc[-1]
        d = stoch_d.iloc[-1]
        
        factors = []
        
        # Check crossover
        if len(stoch_k) >= 2:
            prev_k = stoch_k.iloc[-2]
            prev_d = stoch_d.iloc[-2]
            
            # Bullish crossover
            if prev_k <= prev_d and k > d:
                factors.append("%K crossed above %D (bullish)")
                if k < self.stoch_os:
                    direction = SignalDirection.STRONG_BUY
                    confidence = 0.85
                    factors.append("Crossover in oversold zone")
                else:
                    direction = SignalDirection.BUY
                    confidence = 0.70
            # Bearish crossover
            elif prev_k >= prev_d and k < d:
                factors.append("%K crossed below %D (bearish)")
                if k > self.stoch_ob:
                    direction = SignalDirection.STRONG_SELL
                    confidence = 0.85
                    factors.append("Crossover in overbought zone")
                else:
                    direction = SignalDirection.SELL
                    confidence = 0.70
            # No crossover - zone-based signal
            else:
                if k <= self.stoch_os:
                    # Boost for extreme readings
                    if k < 15:
                        direction = SignalDirection.STRONG_BUY
                        confidence = 0.85
                        factors.append(f"Stochastic extremely oversold at {k:.1f}")
                    else:
                        direction = SignalDirection.BUY
                        confidence = 0.70
                        factors.append(f"Stochastic oversold at {k:.1f}")
                elif k >= self.stoch_ob:
                    if k > 85:
                        direction = SignalDirection.STRONG_SELL
                        confidence = 0.85
                        factors.append(f"Stochastic extremely overbought at {k:.1f}")
                    else:
                        direction = SignalDirection.SELL
                        confidence = 0.70
                        factors.append(f"Stochastic overbought at {k:.1f}")
                else:
                    direction = SignalDirection.NEUTRAL
                    confidence = 0.50
                    factors.append(f"Stochastic neutral at {k:.1f}")
        else:
            direction = SignalDirection.NEUTRAL
            confidence = 0.40
            factors.append("Insufficient data for crossover")
        
        # Determine zone
        if k >= self.stoch_ob:
            zone = "OVERBOUGHT"
        elif k <= self.stoch_os:
            zone = "OVERSOLD"
        else:
            zone = "NEUTRAL"
        
        return IndicatorSignal(
            indicator_name="Stochastic",
            direction=direction,
            confidence=confidence,
            value=k,
            zone=zone,
            factors=factors
        )
    
    def generate_williams_signal(self, williams_r: pd.Series) -> IndicatorSignal:
        """
        Generate trading signal from Williams %R.
        
        Parameters
        ----------
        williams_r : pd.Series
            Williams %R values
            
        Returns
        -------
        IndicatorSignal
            Williams %R-based trading signal
        """
        current = williams_r.iloc[-1]
        factors = []
        
        # Williams %R is inverted: -100 is oversold, 0 is overbought
        if current <= self.williams_os:
            zone = "OVERSOLD"
            # Extreme readings below -95 are highly reliable
            if current < -95:
                direction = SignalDirection.STRONG_BUY
                confidence = 0.85
                factors.append(f"Williams %R extremely oversold at {current:.1f}")
            else:
                direction = SignalDirection.BUY
                confidence = 0.75
                factors.append(f"Williams %R oversold at {current:.1f}")
        elif current >= self.williams_ob:
            zone = "OVERBOUGHT"
            if current > -5:
                direction = SignalDirection.STRONG_SELL
                confidence = 0.85
                factors.append(f"Williams %R extremely overbought at {current:.1f}")
            else:
                direction = SignalDirection.SELL
                confidence = 0.75
                factors.append(f"Williams %R overbought at {current:.1f}")
        else:
            direction = SignalDirection.NEUTRAL
            confidence = 0.50
            zone = "NEUTRAL"
            factors.append(f"Williams %R neutral at {current:.1f}")
        
        # Check for reversal from extreme
        if len(williams_r) >= 3:
            prev = williams_r.iloc[-3:-1].mean()
            if prev <= self.williams_os and current > self.williams_os:
                factors.append("Exiting oversold zone (bullish)")
                if direction == SignalDirection.NEUTRAL:
                    direction = SignalDirection.BUY
                    confidence = 0.60
            elif prev >= self.williams_ob and current < self.williams_ob:
                factors.append("Exiting overbought zone (bearish)")
                if direction == SignalDirection.NEUTRAL:
                    direction = SignalDirection.SELL
                    confidence = 0.60
        
        return IndicatorSignal(
            indicator_name="Williams_R",
            direction=direction,
            confidence=confidence,
            value=current,
            zone=zone,
            factors=factors
        )
    
    def compute_all(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, IndicatorSignal]]:
        """
        Compute all momentum indicators and generate signals.
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with 'Open', 'High', 'Low', 'Close', 'Volume'
            
        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, IndicatorSignal]]
            DataFrame with indicator columns and dictionary of signals
        """
        result = pd.DataFrame(index=df.index)
        
        # RSI
        result['rsi'] = self.calculate_rsi(df['Close'])
        
        # Stochastic
        result['stoch_k'], result['stoch_d'] = self.calculate_stochastic(
            df['High'], df['Low'], df['Close']
        )
        
        # Williams %R
        result['williams_r'] = self.calculate_williams_r(
            df['High'], df['Low'], df['Close']
        )
        
        # ROC (multiple periods)
        result['roc_10'] = self.calculate_roc(df['Close'], 10)
        result['roc_21'] = self.calculate_roc(df['Close'], 21)
        
        # Generate signals
        signals = {
            'rsi': self.generate_rsi_signal(result['rsi'], df['Close']),
            'stochastic': self.generate_stochastic_signal(
                result['stoch_k'], result['stoch_d']
            ),
            'williams_r': self.generate_williams_signal(result['williams_r'])
        }
        
        return result, signals


# =============================================================================
# TREND INDICATORS
# =============================================================================

class TrendIndicators:
    """
    Trend-following indicator calculations and signal generation.
    
    Trend indicators identify the direction and strength of the prevailing
    market trend to align positions with dominant momentum.
    
    Indicators implemented:
    - MACD: Appel, 1979
    - ADX/DMI: Wilder, 1978
    - Aroon: Chande, 1995
    - Supertrend: Olivier Seban
    """
    
    def __init__(self, adjuster: RegimeAdjuster):
        """
        Initialize with regime adjuster.
        
        Parameters
        ----------
        adjuster : RegimeAdjuster
            Threshold adjuster based on volatility regime
        """
        self.adjuster = adjuster
        self.adx_threshold = adjuster.adjust_adx_threshold()
    
    @staticmethod
    def calculate_macd(
        close: pd.Series,
        fast: int = MACD_FAST,
        slow: int = MACD_SLOW,
        signal: int = MACD_SIGNAL
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD, Signal line, and Histogram.
        
        MACD = EMA(fast) - EMA(slow)
        Signal = EMA(MACD, signal_period)
        Histogram = MACD - Signal
        
        Parameters
        ----------
        close : pd.Series
            Closing prices
        fast, slow, signal : int
            Period parameters
            
        Returns
        -------
        Tuple[pd.Series, pd.Series, pd.Series]
            (MACD line, Signal line, Histogram)
        """
        ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_adx_dmi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = ADX_PERIOD
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate ADX and Directional Movement indicators.
        
        ADX measures trend strength (0-100), regardless of direction.
        +DI measures upward movement strength.
        -DI measures downward movement strength.
        
        Parameters
        ----------
        high, low, close : pd.Series
            OHLC data
        period : int
            Smoothing period
            
        Returns
        -------
        Tuple[pd.Series, pd.Series, pd.Series]
            (ADX, +DI, -DI)
        """
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)
        
        # Wilder's smoothing
        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        
        # Directional Indicators
        plus_di = 100.0 * plus_dm_smooth / atr.replace(0, np.nan)
        minus_di = 100.0 * minus_dm_smooth / atr.replace(0, np.nan)
        
        # Directional Index
        dx = 100.0 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        
        # ADX (smoothed DX)
        adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        
        return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)
    
    @staticmethod
    def calculate_aroon(
        high: pd.Series,
        low: pd.Series,
        period: int = AROON_PERIOD
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Aroon Up, Aroon Down, and Aroon Oscillator.
        
        Aroon Up = 100 * (period - bars since highest high) / period
        Aroon Down = 100 * (period - bars since lowest low) / period
        Oscillator = Aroon Up - Aroon Down
        
        Parameters
        ----------
        high, low : pd.Series
            High and Low prices
        period : int
            Lookback period
            
        Returns
        -------
        Tuple[pd.Series, pd.Series, pd.Series]
            (Aroon Up, Aroon Down, Oscillator)
        """
        def bars_since_high(x):
            return period - x.argmax()
        
        def bars_since_low(x):
            return period - x.argmin()
        
        aroon_up = high.rolling(window=period + 1, min_periods=period + 1).apply(
            bars_since_high, raw=True
        ) * (100.0 / period)
        
        aroon_down = low.rolling(window=period + 1, min_periods=period + 1).apply(
            bars_since_low, raw=True
        ) * (100.0 / period)
        
        oscillator = aroon_up - aroon_down
        
        return aroon_up.fillna(50), aroon_down.fillna(50), oscillator.fillna(0)
    
    @staticmethod
    def calculate_supertrend(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = SUPERTREND_PERIOD,
        multiplier: float = SUPERTREND_MULTIPLIER
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Supertrend indicator.
        
        Upper Band = (High + Low) / 2 + multiplier * ATR
        Lower Band = (High + Low) / 2 - multiplier * ATR
        
        Parameters
        ----------
        high, low, close : pd.Series
            OHLC data
        period : int
            ATR period
        multiplier : float
            ATR multiplier
            
        Returns
        -------
        Tuple[pd.Series, pd.Series]
            (Supertrend value, Trend direction: 1=up, -1=down)
        """
        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        
        # Calculate bands
        hl2 = (high + low) / 2
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr
        
        # Initialize
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=float)
        
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = -1
        
        for i in range(1, len(close)):
            # Adjust bands based on previous values
            if lower_band.iloc[i] > lower_band.iloc[i-1] or close.iloc[i-1] < lower_band.iloc[i-1]:
                final_lower = lower_band.iloc[i]
            else:
                final_lower = lower_band.iloc[i-1]
            
            if upper_band.iloc[i] < upper_band.iloc[i-1] or close.iloc[i-1] > upper_band.iloc[i-1]:
                final_upper = upper_band.iloc[i]
            else:
                final_upper = upper_band.iloc[i-1]
            
            # Determine trend
            if supertrend.iloc[i-1] == upper_band.iloc[i-1]:
                if close.iloc[i] > final_upper:
                    supertrend.iloc[i] = final_lower
                    direction.iloc[i] = 1
                else:
                    supertrend.iloc[i] = final_upper
                    direction.iloc[i] = -1
            else:
                if close.iloc[i] < final_lower:
                    supertrend.iloc[i] = final_upper
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = final_lower
                    direction.iloc[i] = 1
        
        return supertrend, direction
    
    def generate_macd_signal(
        self,
        macd: pd.Series,
        signal: pd.Series,
        histogram: pd.Series
    ) -> IndicatorSignal:
        """
        Generate trading signal from MACD.
        
        Parameters
        ----------
        macd, signal, histogram : pd.Series
            MACD components
            
        Returns
        -------
        IndicatorSignal
            MACD-based trading signal
        """
        current_macd = macd.iloc[-1]
        current_signal = signal.iloc[-1]
        current_hist = histogram.iloc[-1]
        
        factors = []
        
        # Check for crossover
        crossover = CrossoverType.NONE
        if len(macd) >= 2:
            prev_macd = macd.iloc[-2]
            prev_signal = signal.iloc[-2]
            
            if prev_macd <= prev_signal and current_macd > current_signal:
                crossover = CrossoverType.BULLISH_CROSS
                factors.append("MACD bullish crossover")
            elif prev_macd >= prev_signal and current_macd < current_signal:
                crossover = CrossoverType.BEARISH_CROSS
                factors.append("MACD bearish crossover")
        
        # Histogram analysis
        if len(histogram) >= 3:
            hist_direction = histogram.iloc[-1] - histogram.iloc[-3]
            if current_hist > 0 and hist_direction > 0:
                factors.append("Histogram expanding bullish")
            elif current_hist < 0 and hist_direction < 0:
                factors.append("Histogram expanding bearish")
            elif current_hist > 0 and hist_direction < 0:
                factors.append("Histogram contracting (bullish weakening)")
            elif current_hist < 0 and hist_direction > 0:
                factors.append("Histogram contracting (bearish weakening)")
        
        # Zero line analysis
        if current_macd > 0:
            factors.append("MACD above zero line")
        else:
            factors.append("MACD below zero line")
        
        # Determine signal
        if crossover == CrossoverType.BULLISH_CROSS:
            if current_macd > 0:
                direction = SignalDirection.STRONG_BUY
                confidence = 0.85
            else:
                direction = SignalDirection.BUY
                confidence = 0.75
        elif crossover == CrossoverType.BEARISH_CROSS:
            if current_macd < 0:
                direction = SignalDirection.STRONG_SELL
                confidence = 0.85
            else:
                direction = SignalDirection.SELL
                confidence = 0.75
        elif current_hist > 0:
            direction = SignalDirection.BUY if current_macd > current_signal else SignalDirection.NEUTRAL
            confidence = 0.60
        elif current_hist < 0:
            direction = SignalDirection.SELL if current_macd < current_signal else SignalDirection.NEUTRAL
            confidence = 0.60
        else:
            direction = SignalDirection.NEUTRAL
            confidence = 0.50
        
        zone = "BULLISH" if current_hist > 0 else "BEARISH" if current_hist < 0 else "NEUTRAL"
        
        return IndicatorSignal(
            indicator_name="MACD",
            direction=direction,
            confidence=confidence,
            value=current_macd,
            zone=zone,
            factors=factors
        )
    
    def generate_adx_signal(
        self,
        adx: pd.Series,
        plus_di: pd.Series,
        minus_di: pd.Series
    ) -> IndicatorSignal:
        """
        Generate trading signal from ADX/DMI.
        
        Parameters
        ----------
        adx, plus_di, minus_di : pd.Series
            ADX and DI values
            
        Returns
        -------
        IndicatorSignal
            ADX-based trading signal
        """
        current_adx = adx.iloc[-1]
        current_plus = plus_di.iloc[-1]
        current_minus = minus_di.iloc[-1]
        
        factors = []
        
        # Trend strength
        if current_adx >= ADX_VERY_STRONG:
            factors.append(f"Very strong trend (ADX={current_adx:.1f})")
            trend_strength = "VERY_STRONG"
        elif current_adx >= self.adx_threshold:
            factors.append(f"Strong trend (ADX={current_adx:.1f})")
            trend_strength = "STRONG"
        else:
            factors.append(f"Weak/no trend (ADX={current_adx:.1f})")
            trend_strength = "WEAK"
        
        # Directional analysis
        if current_plus > current_minus:
            factors.append(f"+DI above -DI (bullish directional)")
            di_direction = "BULLISH"
        else:
            factors.append(f"-DI above +DI (bearish directional)")
            di_direction = "BEARISH"
        
        # DI crossover
        if len(plus_di) >= 2:
            prev_plus = plus_di.iloc[-2]
            prev_minus = minus_di.iloc[-2]
            
            if prev_plus <= prev_minus and current_plus > current_minus:
                factors.append("+DI crossed above -DI (bullish)")
            elif prev_plus >= prev_minus and current_plus < current_minus:
                factors.append("-DI crossed above +DI (bearish)")
        
        # Generate signal
        if trend_strength == "WEAK":
            direction = SignalDirection.NEUTRAL
            confidence = 0.40
        elif di_direction == "BULLISH":
            if trend_strength == "VERY_STRONG":
                direction = SignalDirection.STRONG_BUY
                confidence = 0.85
            else:
                direction = SignalDirection.BUY
                confidence = 0.70
        else:
            if trend_strength == "VERY_STRONG":
                direction = SignalDirection.STRONG_SELL
                confidence = 0.85
            else:
                direction = SignalDirection.SELL
                confidence = 0.70
        
        return IndicatorSignal(
            indicator_name="ADX",
            direction=direction,
            confidence=confidence,
            value=current_adx,
            zone=f"{trend_strength}_{di_direction}",
            factors=factors
        )
    
    def generate_supertrend_signal(
        self,
        supertrend: pd.Series,
        direction: pd.Series,
        close: pd.Series
    ) -> IndicatorSignal:
        """
        Generate trading signal from Supertrend.
        
        Parameters
        ----------
        supertrend : pd.Series
            Supertrend values
        direction : pd.Series
            Trend direction (1=up, -1=down)
        close : pd.Series
            Close prices
            
        Returns
        -------
        IndicatorSignal
            Supertrend-based trading signal
        """
        current_dir = direction.iloc[-1]
        current_st = supertrend.iloc[-1]
        current_close = close.iloc[-1]
        
        factors = []
        
        # Check for trend change
        if len(direction) >= 2:
            prev_dir = direction.iloc[-2]
            if prev_dir != current_dir:
                if current_dir == 1:
                    factors.append("Supertrend flipped bullish")
                else:
                    factors.append("Supertrend flipped bearish")
        
        # Distance from supertrend
        distance_pct = abs(current_close - current_st) / current_close * 100
        factors.append(f"Price {distance_pct:.1f}% from Supertrend")
        
        # Generate signal
        if current_dir == 1:
            if distance_pct < 1:
                direction_signal = SignalDirection.BUY
                confidence = 0.65
                factors.append("Close to support")
            else:
                direction_signal = SignalDirection.BUY
                confidence = 0.75
            zone = "UPTREND"
        else:
            if distance_pct < 1:
                direction_signal = SignalDirection.SELL
                confidence = 0.65
                factors.append("Close to resistance")
            else:
                direction_signal = SignalDirection.SELL
                confidence = 0.75
            zone = "DOWNTREND"
        
        return IndicatorSignal(
            indicator_name="Supertrend",
            direction=direction_signal,
            confidence=confidence,
            value=current_st,
            zone=zone,
            factors=factors
        )
    
    def compute_all(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, IndicatorSignal]]:
        """
        Compute all trend indicators and generate signals.
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data
            
        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, IndicatorSignal]]
            DataFrame with indicator columns and dictionary of signals
        """
        result = pd.DataFrame(index=df.index)
        
        # MACD
        result['macd'], result['macd_signal'], result['macd_histogram'] = \
            self.calculate_macd(df['Close'])
        
        # ADX/DMI
        result['adx'], result['plus_di'], result['minus_di'] = \
            self.calculate_adx_dmi(df['High'], df['Low'], df['Close'])
        
        # Aroon
        result['aroon_up'], result['aroon_down'], result['aroon_osc'] = \
            self.calculate_aroon(df['High'], df['Low'])
        
        # Supertrend
        result['supertrend'], result['supertrend_dir'] = \
            self.calculate_supertrend(df['High'], df['Low'], df['Close'])
        
        # Generate signals
        signals = {
            'macd': self.generate_macd_signal(
                result['macd'], result['macd_signal'], result['macd_histogram']
            ),
            'adx': self.generate_adx_signal(
                result['adx'], result['plus_di'], result['minus_di']
            ),
            'supertrend': self.generate_supertrend_signal(
                result['supertrend'], result['supertrend_dir'], df['Close']
            )
        }
        
        return result, signals


# =============================================================================
# VOLATILITY INDICATORS
# =============================================================================

class VolatilityIndicators:
    """
    Volatility band calculations and signal generation.
    
    Volatility indicators create dynamic support/resistance bands that
    expand and contract with market volatility, useful for mean reversion
    and breakout strategies.
    
    Indicators implemented:
    - Bollinger Bands: Bollinger, 1983
    - Keltner Channels: Keltner, 1960 (modified by Linda Raschke)
    - Donchian Channels: Donchian, 1960s
    """
    
    def __init__(self, adjuster: RegimeAdjuster):
        """
        Initialize with regime adjuster.
        
        Parameters
        ----------
        adjuster : RegimeAdjuster
            Threshold adjuster based on volatility regime
        """
        self.adjuster = adjuster
        self.bb_std = adjuster.adjust_bollinger_std()
    
    @staticmethod
    def calculate_bollinger_bands(
        close: pd.Series,
        period: int = BB_PERIOD,
        std_dev: float = BB_STD_DEV
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Middle = SMA(close, period)
        Upper = Middle + std_dev * StdDev(close, period)
        Lower = Middle - std_dev * StdDev(close, period)
        
        Parameters
        ----------
        close : pd.Series
            Closing prices
        period : int
            Moving average period
        std_dev : float
            Standard deviation multiplier
            
        Returns
        -------
        Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]
            (Upper, Middle, Lower, Bandwidth, %B)
        """
        middle = close.rolling(window=period, min_periods=1).mean()
        std = close.rolling(window=period, min_periods=1).std()
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        # Bandwidth: (Upper - Lower) / Middle * 100
        bandwidth = (upper - lower) / middle * 100
        
        # %B: (Price - Lower) / (Upper - Lower)
        percent_b = (close - lower) / (upper - lower)
        
        return upper, middle, lower, bandwidth, percent_b
    
    @staticmethod
    def calculate_keltner_channels(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = KC_PERIOD,
        atr_mult: float = KC_ATR_MULT
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Keltner Channels.
        
        Middle = EMA(close, period)
        Upper = Middle + atr_mult * ATR
        Lower = Middle - atr_mult * ATR
        
        Parameters
        ----------
        high, low, close : pd.Series
            OHLC data
        period : int
            EMA/ATR period
        atr_mult : float
            ATR multiplier
            
        Returns
        -------
        Tuple[pd.Series, pd.Series, pd.Series]
            (Upper, Middle, Lower)
        """
        # EMA for middle
        middle = close.ewm(span=period, adjust=False, min_periods=period).mean()
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False, min_periods=period).mean()
        
        upper = middle + atr_mult * atr
        lower = middle - atr_mult * atr
        
        return upper, middle, lower
    
    @staticmethod
    def calculate_donchian_channels(
        high: pd.Series,
        low: pd.Series,
        period: int = 20
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Donchian Channels.
        
        Upper = Highest High over period
        Lower = Lowest Low over period
        Middle = (Upper + Lower) / 2
        
        Parameters
        ----------
        high, low : pd.Series
            High and Low prices
        period : int
            Lookback period
            
        Returns
        -------
        Tuple[pd.Series, pd.Series, pd.Series]
            (Upper, Middle, Lower)
        """
        upper = high.rolling(window=period, min_periods=1).max()
        lower = low.rolling(window=period, min_periods=1).min()
        middle = (upper + lower) / 2
        
        return upper, middle, lower
    
    def detect_bollinger_squeeze(
        self,
        bandwidth: pd.Series,
        threshold_percentile: float = 20
    ) -> pd.Series:
        """
        Detect Bollinger Band squeeze (low volatility, potential breakout).
        
        Parameters
        ----------
        bandwidth : pd.Series
            Bollinger Bandwidth values
        threshold_percentile : float
            Percentile below which squeeze is detected
            
        Returns
        -------
        pd.Series
            Boolean series indicating squeeze
        """
        threshold = bandwidth.rolling(window=126, min_periods=20).quantile(
            threshold_percentile / 100
        )
        return bandwidth < threshold
    
    def classify_band_position(
        self,
        close: float,
        upper: float,
        middle: float,
        lower: float
    ) -> BandPosition:
        """
        Classify price position relative to bands.
        
        Parameters
        ----------
        close : float
            Current close price
        upper, middle, lower : float
            Band values
            
        Returns
        -------
        BandPosition
            Position classification
        """
        band_width = upper - lower
        
        if close > upper:
            return BandPosition.ABOVE_UPPER
        elif close > upper - 0.1 * band_width:
            return BandPosition.UPPER_ZONE
        elif close > middle:
            return BandPosition.MIDDLE_UPPER
        elif abs(close - middle) < 0.05 * band_width:
            return BandPosition.AT_MIDDLE
        elif close > lower + 0.1 * band_width:
            return BandPosition.MIDDLE_LOWER
        elif close > lower:
            return BandPosition.LOWER_ZONE
        else:
            return BandPosition.BELOW_LOWER
    
    def generate_bollinger_signal(
        self,
        close: pd.Series,
        upper: pd.Series,
        middle: pd.Series,
        lower: pd.Series,
        bandwidth: pd.Series,
        percent_b: pd.Series
    ) -> IndicatorSignal:
        """
        Generate trading signal from Bollinger Bands.
        
        Parameters
        ----------
        close, upper, middle, lower, bandwidth, percent_b : pd.Series
            Bollinger Band components
            
        Returns
        -------
        IndicatorSignal
            Bollinger-based trading signal
        """
        current_close = close.iloc[-1]
        current_upper = upper.iloc[-1]
        current_middle = middle.iloc[-1]
        current_lower = lower.iloc[-1]
        current_bw = bandwidth.iloc[-1]
        current_pb = percent_b.iloc[-1]
        
        factors = []
        position = self.classify_band_position(
            current_close, current_upper, current_middle, current_lower
        )
        
        # Squeeze detection
        squeeze = self.detect_bollinger_squeeze(bandwidth)
        if squeeze.iloc[-1]:
            factors.append("Bollinger squeeze detected (low volatility)")
        
        # Band touch analysis
        if position == BandPosition.ABOVE_UPPER:
            factors.append("Price above upper band (overbought/breakout)")
            direction = SignalDirection.SELL
            confidence = 0.65
        elif position == BandPosition.UPPER_ZONE:
            factors.append("Price near upper band")
            direction = SignalDirection.SELL
            confidence = 0.55
        elif position == BandPosition.BELOW_LOWER:
            factors.append("Price below lower band (oversold/breakdown)")
            direction = SignalDirection.BUY
            confidence = 0.65
        elif position == BandPosition.LOWER_ZONE:
            factors.append("Price near lower band")
            direction = SignalDirection.BUY
            confidence = 0.55
        else:
            factors.append(f"Price in middle zone (%B={current_pb:.2f})")
            direction = SignalDirection.NEUTRAL
            confidence = 0.50
        
        # Bandwidth expansion/contraction
        if len(bandwidth) >= 5:
            bw_change = current_bw - bandwidth.iloc[-5]
            if bw_change > 0:
                factors.append("Volatility expanding")
            else:
                factors.append("Volatility contracting")
        
        return IndicatorSignal(
            indicator_name="Bollinger_Bands",
            direction=direction,
            confidence=confidence,
            value=current_pb,
            zone=position.value,
            factors=factors
        )
    
    def generate_keltner_signal(
        self,
        close: pd.Series,
        upper: pd.Series,
        middle: pd.Series,
        lower: pd.Series
    ) -> IndicatorSignal:
        """
        Generate trading signal from Keltner Channels.
        
        Parameters
        ----------
        close, upper, middle, lower : pd.Series
            Keltner Channel components
            
        Returns
        -------
        IndicatorSignal
            Keltner-based trading signal
        """
        current_close = close.iloc[-1]
        current_upper = upper.iloc[-1]
        current_middle = middle.iloc[-1]
        current_lower = lower.iloc[-1]
        
        factors = []
        position = self.classify_band_position(
            current_close, current_upper, current_middle, current_lower
        )
        
        # Price vs middle (trend)
        if current_close > current_middle:
            factors.append("Price above Keltner middle (bullish)")
        else:
            factors.append("Price below Keltner middle (bearish)")
        
        # Band position signal
        if position == BandPosition.ABOVE_UPPER:
            factors.append("Keltner breakout to upside")
            direction = SignalDirection.BUY  # Breakout strategy
            confidence = 0.70
        elif position == BandPosition.BELOW_LOWER:
            factors.append("Keltner breakdown to downside")
            direction = SignalDirection.SELL  # Breakdown strategy
            confidence = 0.70
        elif position in [BandPosition.UPPER_ZONE, BandPosition.MIDDLE_UPPER]:
            direction = SignalDirection.BUY
            confidence = 0.55
        elif position in [BandPosition.LOWER_ZONE, BandPosition.MIDDLE_LOWER]:
            direction = SignalDirection.SELL
            confidence = 0.55
        else:
            direction = SignalDirection.NEUTRAL
            confidence = 0.50
        
        # Calculate percent position
        range_width = current_upper - current_lower
        if range_width > 0:
            percent_position = (current_close - current_lower) / range_width
        else:
            percent_position = 0.5
        
        return IndicatorSignal(
            indicator_name="Keltner_Channels",
            direction=direction,
            confidence=confidence,
            value=percent_position,
            zone=position.value,
            factors=factors
        )
    
    def compute_all(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, IndicatorSignal]]:
        """
        Compute all volatility indicators and generate signals.
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data
            
        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, IndicatorSignal]]
            DataFrame with indicator columns and dictionary of signals
        """
        result = pd.DataFrame(index=df.index)
        
        # Bollinger Bands (with regime-adjusted std dev)
        result['bb_upper'], result['bb_middle'], result['bb_lower'], \
        result['bb_bandwidth'], result['bb_percent_b'] = \
            self.calculate_bollinger_bands(df['Close'], BB_PERIOD, self.bb_std)
        
        # Keltner Channels
        result['kc_upper'], result['kc_middle'], result['kc_lower'] = \
            self.calculate_keltner_channels(
                df['High'], df['Low'], df['Close']
            )
        
        # Donchian Channels
        result['dc_upper'], result['dc_middle'], result['dc_lower'] = \
            self.calculate_donchian_channels(df['High'], df['Low'])
        
        # Squeeze indicator (BB inside KC)
        result['squeeze'] = (result['bb_upper'] < result['kc_upper']) & \
                           (result['bb_lower'] > result['kc_lower'])
        
        # Generate signals
        signals = {
            'bollinger': self.generate_bollinger_signal(
                df['Close'], result['bb_upper'], result['bb_middle'],
                result['bb_lower'], result['bb_bandwidth'], result['bb_percent_b']
            ),
            'keltner': self.generate_keltner_signal(
                df['Close'], result['kc_upper'], result['kc_middle'],
                result['kc_lower']
            )
        }
        
        return result, signals


# =============================================================================
# VOLUME INDICATORS
# =============================================================================

class VolumeIndicators:
    """
    Volume-based indicator calculations and signal generation.
    
    Volume indicators confirm price movements and identify accumulation
    or distribution phases that precede price moves.
    
    Indicators implemented:
    - OBV (On-Balance Volume): Granville, 1963
    - CMF (Chaikin Money Flow): Chaikin
    - MFI (Money Flow Index): Quong & Soudack
    - VWMA (Volume Weighted Moving Average)
    """
    
    def __init__(self, adjuster: RegimeAdjuster):
        """
        Initialize with regime adjuster.
        
        Parameters
        ----------
        adjuster : RegimeAdjuster
            Threshold adjuster based on volatility regime
        """
        self.adjuster = adjuster
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume.
        
        OBV adds volume on up days and subtracts on down days,
        creating a cumulative indicator of buying/selling pressure.
        
        Parameters
        ----------
        close : pd.Series
            Closing prices
        volume : pd.Series
            Volume data
            
        Returns
        -------
        pd.Series
            OBV values
        """
        direction = np.sign(close.diff())
        direction.iloc[0] = 0
        
        obv = (direction * volume).cumsum()
        
        return obv
    
    @staticmethod
    def calculate_cmf(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = CMF_PERIOD
    ) -> pd.Series:
        """
        Calculate Chaikin Money Flow.
        
        CMF = Sum(Money Flow Volume, period) / Sum(Volume, period)
        where Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        
        Parameters
        ----------
        high, low, close : pd.Series
            OHLC data
        volume : pd.Series
            Volume data
        period : int
            Lookback period
            
        Returns
        -------
        pd.Series
            CMF values [-1, 1]
        """
        # Money Flow Multiplier
        range_hl = high - low
        range_hl = range_hl.replace(0, np.nan)
        mf_mult = ((close - low) - (high - close)) / range_hl
        
        # Money Flow Volume
        mf_volume = mf_mult * volume
        
        # CMF
        cmf = mf_volume.rolling(window=period, min_periods=1).sum() / \
              volume.rolling(window=period, min_periods=1).sum()
        
        return cmf.fillna(0)
    
    @staticmethod
    def calculate_mfi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = MFI_PERIOD
    ) -> pd.Series:
        """
        Calculate Money Flow Index.
        
        MFI is a volume-weighted RSI, measuring buying and selling pressure.
        
        Parameters
        ----------
        high, low, close : pd.Series
            OHLC data
        volume : pd.Series
            Volume data
        period : int
            Lookback period
            
        Returns
        -------
        pd.Series
            MFI values [0, 100]
        """
        # Typical Price
        typical_price = (high + low + close) / 3
        
        # Raw Money Flow
        raw_mf = typical_price * volume
        
        # Positive and Negative Money Flow
        tp_diff = typical_price.diff()
        positive_mf = raw_mf.where(tp_diff > 0, 0)
        negative_mf = raw_mf.where(tp_diff < 0, 0)
        
        # Money Flow Ratio
        positive_sum = positive_mf.rolling(window=period, min_periods=1).sum()
        negative_sum = negative_mf.rolling(window=period, min_periods=1).sum()
        
        mf_ratio = positive_sum / negative_sum.replace(0, np.nan)
        
        # MFI
        mfi = 100 - (100 / (1 + mf_ratio))
        
        return mfi.fillna(50)
    
    @staticmethod
    def calculate_vwma(
        close: pd.Series,
        volume: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """
        Calculate Volume Weighted Moving Average.
        
        VWMA = Sum(Price * Volume, period) / Sum(Volume, period)
        
        Parameters
        ----------
        close : pd.Series
            Closing prices
        volume : pd.Series
            Volume data
        period : int
            Lookback period
            
        Returns
        -------
        pd.Series
            VWMA values
        """
        pv = close * volume
        vwma = pv.rolling(window=period, min_periods=1).sum() / \
               volume.rolling(window=period, min_periods=1).sum()
        
        return vwma
    
    def generate_obv_signal(
        self,
        obv: pd.Series,
        close: pd.Series
    ) -> IndicatorSignal:
        """
        Generate trading signal from OBV.
        
        Parameters
        ----------
        obv : pd.Series
            OBV values
        close : pd.Series
            Close prices
            
        Returns
        -------
        IndicatorSignal
            OBV-based trading signal
        """
        factors = []
        
        # OBV trend (21-day)
        obv_ma = obv.rolling(window=21, min_periods=5).mean()
        current_obv = obv.iloc[-1]
        current_obv_ma = obv_ma.iloc[-1]
        
        if current_obv > current_obv_ma:
            factors.append("OBV above its 21-day MA (accumulation)")
            obv_trend = "ACCUMULATION"
        else:
            factors.append("OBV below its 21-day MA (distribution)")
            obv_trend = "DISTRIBUTION"
        
        # Price vs OBV divergence check
        if len(obv) >= 20:
            price_change = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]
            obv_change = (obv.iloc[-1] - obv.iloc[-20]) / abs(obv.iloc[-20]) if obv.iloc[-20] != 0 else 0
            
            if price_change > 0.05 and obv_change < -0.05:
                factors.append("Bearish divergence: Price up, OBV down")
            elif price_change < -0.05 and obv_change > 0.05:
                factors.append("Bullish divergence: Price down, OBV up")
        
        # Generate signal
        if obv_trend == "ACCUMULATION":
            direction = SignalDirection.BUY
            confidence = 0.65
        else:
            direction = SignalDirection.SELL
            confidence = 0.65
        
        return IndicatorSignal(
            indicator_name="OBV",
            direction=direction,
            confidence=confidence,
            value=current_obv,
            zone=obv_trend,
            factors=factors
        )
    
    def generate_mfi_signal(self, mfi: pd.Series) -> IndicatorSignal:
        """
        Generate trading signal from MFI.
        
        Parameters
        ----------
        mfi : pd.Series
            MFI values
            
        Returns
        -------
        IndicatorSignal
            MFI-based trading signal
        """
        current_mfi = mfi.iloc[-1]
        factors = []
        
        # Zone classification (similar to RSI)
        if current_mfi >= 80:
            zone = "OVERBOUGHT"
            direction = SignalDirection.SELL
            confidence = 0.70
            factors.append(f"MFI overbought at {current_mfi:.1f}")
        elif current_mfi <= 20:
            zone = "OVERSOLD"
            direction = SignalDirection.BUY
            confidence = 0.70
            factors.append(f"MFI oversold at {current_mfi:.1f}")
        elif current_mfi > 50:
            zone = "BULLISH"
            direction = SignalDirection.BUY
            confidence = 0.55
            factors.append(f"MFI bullish at {current_mfi:.1f}")
        elif current_mfi < 50:
            zone = "BEARISH"
            direction = SignalDirection.SELL
            confidence = 0.55
            factors.append(f"MFI bearish at {current_mfi:.1f}")
        else:
            zone = "NEUTRAL"
            direction = SignalDirection.NEUTRAL
            confidence = 0.50
            factors.append(f"MFI neutral at {current_mfi:.1f}")
        
        return IndicatorSignal(
            indicator_name="MFI",
            direction=direction,
            confidence=confidence,
            value=current_mfi,
            zone=zone,
            factors=factors
        )
    
    def generate_cmf_signal(self, cmf: pd.Series) -> IndicatorSignal:
        """
        Generate trading signal from CMF.
        
        Parameters
        ----------
        cmf : pd.Series
            CMF values
            
        Returns
        -------
        IndicatorSignal
            CMF-based trading signal
        """
        current_cmf = cmf.iloc[-1]
        factors = []
        
        # CMF interpretation
        if current_cmf > 0.25:
            zone = "STRONG_ACCUMULATION"
            direction = SignalDirection.STRONG_BUY
            confidence = 0.75
            factors.append(f"Strong buying pressure (CMF={current_cmf:.3f})")
        elif current_cmf > 0.05:
            zone = "ACCUMULATION"
            direction = SignalDirection.BUY
            confidence = 0.65
            factors.append(f"Buying pressure (CMF={current_cmf:.3f})")
        elif current_cmf < -0.25:
            zone = "STRONG_DISTRIBUTION"
            direction = SignalDirection.STRONG_SELL
            confidence = 0.75
            factors.append(f"Strong selling pressure (CMF={current_cmf:.3f})")
        elif current_cmf < -0.05:
            zone = "DISTRIBUTION"
            direction = SignalDirection.SELL
            confidence = 0.65
            factors.append(f"Selling pressure (CMF={current_cmf:.3f})")
        else:
            zone = "NEUTRAL"
            direction = SignalDirection.NEUTRAL
            confidence = 0.50
            factors.append(f"Neutral money flow (CMF={current_cmf:.3f})")
        
        return IndicatorSignal(
            indicator_name="CMF",
            direction=direction,
            confidence=confidence,
            value=current_cmf,
            zone=zone,
            factors=factors
        )
    
    def compute_all(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, IndicatorSignal]]:
        """
        Compute all volume indicators and generate signals.
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data
            
        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, IndicatorSignal]]
            DataFrame with indicator columns and dictionary of signals
        """
        result = pd.DataFrame(index=df.index)
        
        # OBV
        result['obv'] = self.calculate_obv(df['Close'], df['Volume'])
        
        # CMF
        result['cmf'] = self.calculate_cmf(
            df['High'], df['Low'], df['Close'], df['Volume']
        )
        
        # MFI
        result['mfi'] = self.calculate_mfi(
            df['High'], df['Low'], df['Close'], df['Volume']
        )
        
        # VWMA (multiple periods)
        result['vwma_10'] = self.calculate_vwma(df['Close'], df['Volume'], 10)
        result['vwma_20'] = self.calculate_vwma(df['Close'], df['Volume'], 20)
        
        # Generate signals
        signals = {
            'obv': self.generate_obv_signal(result['obv'], df['Close']),
            'mfi': self.generate_mfi_signal(result['mfi']),
            'cmf': self.generate_cmf_signal(result['cmf'])
        }
        
        return result, signals


# =============================================================================
# ICHIMOKU KINKO HYO
# =============================================================================

class IchimokuIndicator:
    """
    Complete Ichimoku Kinko Hyo implementation.
    
    Ichimoku ("one glance equilibrium chart") is a comprehensive trading
    system developed by Goichi Hosoda in the 1930s. It provides:
    - Trend direction and strength
    - Support and resistance levels
    - Momentum signals
    - Future projected support/resistance (the "cloud")
    
    Five components:
    1. Tenkan-sen (Conversion Line): 9-period midpoint
    2. Kijun-sen (Base Line): 26-period midpoint
    3. Senkou Span A (Leading Span A): Midpoint of Tenkan/Kijun, displaced forward
    4. Senkou Span B (Leading Span B): 52-period midpoint, displaced forward
    5. Chikou Span (Lagging Span): Close displaced backward
    """
    
    def __init__(
        self,
        tenkan_period: int = ICHIMOKU_TENKAN,
        kijun_period: int = ICHIMOKU_KIJUN,
        senkou_b_period: int = ICHIMOKU_SENKOU_B,
        displacement: int = ICHIMOKU_DISPLACEMENT
    ):
        """
        Initialize Ichimoku with parameters.
        
        Parameters
        ----------
        tenkan_period : int
            Tenkan-sen period (default: 9)
        kijun_period : int
            Kijun-sen period (default: 26)
        senkou_b_period : int
            Senkou Span B period (default: 52)
        displacement : int
            Cloud displacement (default: 26)
        """
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement
    
    @staticmethod
    def _midpoint(high: pd.Series, low: pd.Series, period: int) -> pd.Series:
        """Calculate period midpoint (highest high + lowest low) / 2."""
        highest = high.rolling(window=period, min_periods=1).max()
        lowest = low.rolling(window=period, min_periods=1).min()
        return (highest + lowest) / 2
    
    def calculate(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> Dict[str, pd.Series]:
        """
        Calculate all Ichimoku components.
        
        Parameters
        ----------
        high, low, close : pd.Series
            OHLC data
            
        Returns
        -------
        Dict[str, pd.Series]
            Dictionary of Ichimoku components
        """
        # Tenkan-sen (Conversion Line)
        tenkan = self._midpoint(high, low, self.tenkan_period)
        
        # Kijun-sen (Base Line)
        kijun = self._midpoint(high, low, self.kijun_period)
        
        # Senkou Span A (Leading Span A) - displaced forward
        senkou_a = ((tenkan + kijun) / 2).shift(self.displacement)
        
        # Senkou Span B (Leading Span B) - displaced forward
        senkou_b = self._midpoint(high, low, self.senkou_b_period).shift(self.displacement)
        
        # Chikou Span (Lagging Span) - close displaced backward
        chikou = close.shift(-self.displacement)
        
        return {
            'tenkan_sen': tenkan,
            'kijun_sen': kijun,
            'senkou_span_a': senkou_a,
            'senkou_span_b': senkou_b,
            'chikou_span': chikou
        }
    
    def analyze(
        self,
        df: pd.DataFrame,
        components: Dict[str, pd.Series]
    ) -> IchimokuAnalysis:
        """
        Generate comprehensive Ichimoku analysis.
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data
        components : Dict[str, pd.Series]
            Ichimoku components from calculate()
            
        Returns
        -------
        IchimokuAnalysis
            Complete analysis including signal
        """
        close = df['Close']
        current_close = close.iloc[-1]
        
        tenkan = components['tenkan_sen'].iloc[-1]
        kijun = components['kijun_sen'].iloc[-1]
        senkou_a = components['senkou_span_a'].iloc[-1]
        senkou_b = components['senkou_span_b'].iloc[-1]
        chikou = components['chikou_span'].iloc[-self.displacement - 1] \
                 if len(close) > self.displacement else current_close
        
        # Cloud boundaries
        cloud_top = max(senkou_a, senkou_b) if not pd.isna(senkou_a) else senkou_b
        cloud_bottom = min(senkou_a, senkou_b) if not pd.isna(senkou_a) else senkou_b
        cloud_color = "bullish" if senkou_a > senkou_b else "bearish"
        
        # Price vs Cloud
        if current_close > cloud_top:
            price_vs_cloud = "above"
        elif current_close < cloud_bottom:
            price_vs_cloud = "below"
        else:
            price_vs_cloud = "inside"
        
        # Tenkan-Kijun cross
        if len(components['tenkan_sen']) >= 2:
            prev_tenkan = components['tenkan_sen'].iloc[-2]
            prev_kijun = components['kijun_sen'].iloc[-2]
            
            if prev_tenkan <= prev_kijun and tenkan > kijun:
                tk_cross = CrossoverType.BULLISH_CROSS
            elif prev_tenkan >= prev_kijun and tenkan < kijun:
                tk_cross = CrossoverType.BEARISH_CROSS
            elif tenkan > kijun:
                tk_cross = CrossoverType.BULLISH_ABOVE
            else:
                tk_cross = CrossoverType.BEARISH_BELOW
        else:
            tk_cross = CrossoverType.NONE
        
        # Price vs Kijun
        price_vs_kijun = "above" if current_close > kijun else "below"
        
        # Chikou vs historical price
        if len(close) > self.displacement:
            chikou_price = close.iloc[-self.displacement - 1]
            chikou_vs_price = "above" if chikou > chikou_price else "below"
        else:
            chikou_vs_price = "neutral"
        
        # Count bullish/bearish signals
        bullish_signals = 0
        bearish_signals = 0
        
        # 1. Price above/below cloud
        if price_vs_cloud == "above":
            bullish_signals += 1
        elif price_vs_cloud == "below":
            bearish_signals += 1
        
        # 2. Cloud color
        if cloud_color == "bullish":
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # 3. Tenkan vs Kijun
        if tenkan > kijun:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # 4. Price vs Kijun
        if price_vs_kijun == "above":
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # 5. Chikou vs Price
        if chikou_vs_price == "above":
            bullish_signals += 1
        elif chikou_vs_price == "below":
            bearish_signals += 1
        
        # Determine overall signal
        if bullish_signals >= 4:
            signal = IchimokuSignal.STRONG_BULLISH
        elif bullish_signals >= 3:
            signal = IchimokuSignal.BULLISH
        elif bearish_signals >= 4:
            signal = IchimokuSignal.STRONG_BEARISH
        elif bearish_signals >= 3:
            signal = IchimokuSignal.BEARISH
        else:
            signal = IchimokuSignal.NEUTRAL
        
        return IchimokuAnalysis(
            signal=signal,
            tenkan_sen=tenkan,
            kijun_sen=kijun,
            senkou_span_a=senkou_a if not pd.isna(senkou_a) else 0,
            senkou_span_b=senkou_b if not pd.isna(senkou_b) else 0,
            chikou_span=chikou,
            cloud_top=cloud_top if not pd.isna(cloud_top) else 0,
            cloud_bottom=cloud_bottom if not pd.isna(cloud_bottom) else 0,
            cloud_color=cloud_color,
            price_vs_cloud=price_vs_cloud,
            tk_cross=tk_cross,
            price_vs_kijun=price_vs_kijun,
            chikou_vs_price=chikou_vs_price,
            bullish_signals=bullish_signals,
            bearish_signals=bearish_signals
        )
    
    def generate_signal(self, analysis: IchimokuAnalysis) -> IndicatorSignal:
        """
        Generate trading signal from Ichimoku analysis.
        
        Parameters
        ----------
        analysis : IchimokuAnalysis
            Ichimoku analysis result
            
        Returns
        -------
        IndicatorSignal
            Ichimoku-based trading signal
        """
        factors = []
        
        # Build factors list
        factors.append(f"Price {analysis.price_vs_cloud} cloud")
        factors.append(f"Cloud is {analysis.cloud_color}")
        
        if analysis.tk_cross in [CrossoverType.BULLISH_CROSS, CrossoverType.BEARISH_CROSS]:
            factors.append(f"TK cross: {analysis.tk_cross.value}")
        
        factors.append(f"Price {analysis.price_vs_kijun} Kijun-sen")
        factors.append(f"Chikou {analysis.chikou_vs_price} price")
        factors.append(f"Score: {analysis.bullish_signals} bullish, {analysis.bearish_signals} bearish")
        
        # Map signal to direction
        signal_map = {
            IchimokuSignal.STRONG_BULLISH: (SignalDirection.STRONG_BUY, 0.90),
            IchimokuSignal.BULLISH: (SignalDirection.BUY, 0.70),
            IchimokuSignal.NEUTRAL: (SignalDirection.NEUTRAL, 0.50),
            IchimokuSignal.BEARISH: (SignalDirection.SELL, 0.70),
            IchimokuSignal.STRONG_BEARISH: (SignalDirection.STRONG_SELL, 0.90)
        }
        
        direction, confidence = signal_map[analysis.signal]
        
        return IndicatorSignal(
            indicator_name="Ichimoku",
            direction=direction,
            confidence=confidence,
            value=analysis.bullish_signals - analysis.bearish_signals,
            zone=analysis.signal.value,
            factors=factors
        )
    
    def compute(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, IndicatorSignal, IchimokuAnalysis]:
        """
        Compute Ichimoku and generate signal.
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data
            
        Returns
        -------
        Tuple[pd.DataFrame, IndicatorSignal, IchimokuAnalysis]
            DataFrame with components, signal, and full analysis
        """
        components = self.calculate(df['High'], df['Low'], df['Close'])
        
        result = pd.DataFrame(index=df.index)
        for name, series in components.items():
            result[f'ichimoku_{name}'] = series
        
        analysis = self.analyze(df, components)
        signal = self.generate_signal(analysis)
        
        return result, signal, analysis


# =============================================================================
# DIVERGENCE DETECTION
# =============================================================================

class DivergenceDetector:
    """
    Automated divergence detection between price and indicators.
    
    Divergences occur when price and an indicator move in opposite directions,
    often preceding reversals or continuations.
    
    Types:
    - Regular Bullish: Price makes lower low, indicator makes higher low (reversal up)
    - Regular Bearish: Price makes higher high, indicator makes lower high (reversal down)
    - Hidden Bullish: Price makes higher low, indicator makes lower low (continuation up)
    - Hidden Bearish: Price makes lower high, indicator makes higher high (continuation down)
    """
    
    def __init__(
        self,
        lookback: int = DIVERGENCE_LOOKBACK,
        min_bars: int = DIVERGENCE_MIN_BARS
    ):
        """
        Initialize divergence detector.
        
        Parameters
        ----------
        lookback : int
            Number of bars to look back for swing points
        min_bars : int
            Minimum bars between swing points
        """
        self.lookback = lookback
        self.min_bars = min_bars
    
    def find_swing_points(
        self,
        series: pd.Series,
        order: int = 5
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Find local maxima and minima in a series.
        
        Parameters
        ----------
        series : pd.Series
            Data series to analyze
        order : int
            How many points on each side to use for comparison
            
        Returns
        -------
        Tuple[pd.Series, pd.Series]
            (highs, lows) - Series with NaN except at swing points
        """
        # Find indices of local maxima and minima
        high_idx = argrelextrema(series.values, np.greater, order=order)[0]
        low_idx = argrelextrema(series.values, np.less, order=order)[0]
        
        highs = pd.Series(index=series.index, dtype=float)
        lows = pd.Series(index=series.index, dtype=float)
        
        highs.iloc[high_idx] = series.iloc[high_idx]
        lows.iloc[low_idx] = series.iloc[low_idx]
        
        return highs, lows
    
    def detect(
        self,
        price: pd.Series,
        indicator: pd.Series,
        indicator_name: str
    ) -> List[DivergenceSignal]:
        """
        Detect divergences between price and indicator.
        
        Parameters
        ----------
        price : pd.Series
            Price series (typically Close)
        indicator : pd.Series
            Indicator values
        indicator_name : str
            Name of the indicator
            
        Returns
        -------
        List[DivergenceSignal]
            List of detected divergences
        """
        divergences = []
        
        # Find swing points
        price_highs, price_lows = self.find_swing_points(price)
        ind_highs, ind_lows = self.find_swing_points(indicator)
        
        # Get recent swing lows for bullish divergence
        recent_price_lows = price_lows.dropna().tail(self.lookback)
        recent_ind_lows = ind_lows.dropna().tail(self.lookback)
        
        # Get recent swing highs for bearish divergence
        recent_price_highs = price_highs.dropna().tail(self.lookback)
        recent_ind_highs = ind_highs.dropna().tail(self.lookback)
        
        # Check for regular bullish divergence (price LL, indicator HL)
        if len(recent_price_lows) >= 2 and len(recent_ind_lows) >= 2:
            p1, p2 = recent_price_lows.iloc[-2], recent_price_lows.iloc[-1]
            i1, i2 = recent_ind_lows.iloc[-2], recent_ind_lows.iloc[-1]
            
            if p2 < p1 and i2 > i1:  # Regular bullish
                strength = abs(i2 - i1) / abs(i1) if i1 != 0 else 0
                divergences.append(DivergenceSignal(
                    divergence_type=DivergenceType.REGULAR_BULLISH,
                    indicator_name=indicator_name,
                    start_date=recent_price_lows.index[-2],
                    end_date=recent_price_lows.index[-1],
                    price_start=p1,
                    price_end=p2,
                    indicator_start=i1,
                    indicator_end=i2,
                    strength=min(1.0, strength),
                    bars_duration=(recent_price_lows.index[-1] - recent_price_lows.index[-2]).days
                ))
            elif p2 > p1 and i2 < i1:  # Hidden bullish
                strength = abs(i2 - i1) / abs(i1) if i1 != 0 else 0
                divergences.append(DivergenceSignal(
                    divergence_type=DivergenceType.HIDDEN_BULLISH,
                    indicator_name=indicator_name,
                    start_date=recent_price_lows.index[-2],
                    end_date=recent_price_lows.index[-1],
                    price_start=p1,
                    price_end=p2,
                    indicator_start=i1,
                    indicator_end=i2,
                    strength=min(1.0, strength),
                    bars_duration=(recent_price_lows.index[-1] - recent_price_lows.index[-2]).days
                ))
        
        # Check for bearish divergence (price HH, indicator LH)
        if len(recent_price_highs) >= 2 and len(recent_ind_highs) >= 2:
            p1, p2 = recent_price_highs.iloc[-2], recent_price_highs.iloc[-1]
            i1, i2 = recent_ind_highs.iloc[-2], recent_ind_highs.iloc[-1]
            
            if p2 > p1 and i2 < i1:  # Regular bearish
                strength = abs(i2 - i1) / abs(i1) if i1 != 0 else 0
                divergences.append(DivergenceSignal(
                    divergence_type=DivergenceType.REGULAR_BEARISH,
                    indicator_name=indicator_name,
                    start_date=recent_price_highs.index[-2],
                    end_date=recent_price_highs.index[-1],
                    price_start=p1,
                    price_end=p2,
                    indicator_start=i1,
                    indicator_end=i2,
                    strength=min(1.0, strength),
                    bars_duration=(recent_price_highs.index[-1] - recent_price_highs.index[-2]).days
                ))
            elif p2 < p1 and i2 > i1:  # Hidden bearish
                strength = abs(i2 - i1) / abs(i1) if i1 != 0 else 0
                divergences.append(DivergenceSignal(
                    divergence_type=DivergenceType.HIDDEN_BEARISH,
                    indicator_name=indicator_name,
                    start_date=recent_price_highs.index[-2],
                    end_date=recent_price_highs.index[-1],
                    price_start=p1,
                    price_end=p2,
                    indicator_start=i1,
                    indicator_end=i2,
                    strength=min(1.0, strength),
                    bars_duration=(recent_price_highs.index[-1] - recent_price_highs.index[-2]).days
                ))
        
        return divergences


# =============================================================================
# CONFLUENCE ANALYZER
# =============================================================================

class ConfluenceAnalyzer:
    """
    Multi-indicator confluence analysis and signal aggregation.
    
    Aggregates signals from all indicator families with weighted scoring
    to produce unified trading recommendations.
    """
    
    def __init__(self):
        """Initialize confluence analyzer with default weights."""
        self.weights = {
            'momentum': MOMENTUM_WEIGHT,
            'trend': TREND_WEIGHT,
            'volatility': VOLATILITY_WEIGHT,
            'volume': VOLUME_WEIGHT,
            'system': SYSTEM_WEIGHT
        }
    
    def aggregate_family(
        self,
        family_name: str,
        signals: Dict[str, IndicatorSignal]
    ) -> IndicatorFamily:
        """
        Aggregate signals within an indicator family.
        
        Parameters
        ----------
        family_name : str
            Name of the indicator family
        signals : Dict[str, IndicatorSignal]
            Individual indicator signals
            
        Returns
        -------
        IndicatorFamily
            Aggregated family signal
        """
        if not signals:
            return IndicatorFamily(
                family_name=family_name,
                indicators={},
                aggregate_signal=SignalDirection.NEUTRAL,
                aggregate_confidence=0.0,
                weight=self.weights.get(family_name, 0.1)
            )
        
        # Calculate weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        weighted_confidence = 0.0
        
        for name, signal in signals.items():
            weighted_sum += signal.direction.numeric * signal.confidence
            weighted_confidence += signal.confidence
            total_weight += 1
        
        if total_weight > 0:
            avg_numeric = weighted_sum / total_weight
            avg_confidence = weighted_confidence / total_weight
        else:
            avg_numeric = 0.0
            avg_confidence = 0.0
        
        return IndicatorFamily(
            family_name=family_name,
            indicators=signals,
            aggregate_signal=SignalDirection.from_numeric(avg_numeric),
            aggregate_confidence=avg_confidence,
            weight=self.weights.get(family_name, 0.1)
        )
    
    def analyze(
        self,
        families: Dict[str, IndicatorFamily],
        divergences: List[DivergenceSignal],
        key_levels: Dict[str, float]
    ) -> ConfluenceAnalysis:
        """
        Perform complete confluence analysis with divergence and extreme reading adjustments.
        
        This method aggregates signals across all indicator families, factors in
        divergences as confirmation/warning signals, and adjusts confidence based
        on extreme readings that often precede reversals.
        
        Parameters
        ----------
        families : Dict[str, IndicatorFamily]
            Aggregated family signals
        divergences : List[DivergenceSignal]
            Detected divergences
        key_levels : Dict[str, float]
            Key price levels
            
        Returns
        -------
        ConfluenceAnalysis
            Complete confluence analysis
        """
        # Count signal directions
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        weighted_sum = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        
        # Track extreme readings for conviction boost
        extreme_oversold = False
        extreme_overbought = False
        
        for family_name, family in families.items():
            if family.aggregate_signal in [SignalDirection.STRONG_BUY, SignalDirection.BUY]:
                bullish_count += 1
            elif family.aggregate_signal in [SignalDirection.STRONG_SELL, SignalDirection.SELL]:
                bearish_count += 1
            else:
                neutral_count += 1
            
            weighted_sum += family.aggregate_signal.numeric * family.weight * family.aggregate_confidence
            weighted_confidence += family.aggregate_confidence * family.weight
            total_weight += family.weight
            
            # Check for extreme readings in momentum family
            if family_name == 'momentum':
                for ind_name, signal in family.indicators.items():
                    if 'OVERSOLD' in signal.zone.upper() or 'EXTREME' in signal.zone.upper():
                        if signal.direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
                            extreme_oversold = True
                    if 'OVERBOUGHT' in signal.zone.upper() or 'EXTREME' in signal.zone.upper():
                        if signal.direction in [SignalDirection.SELL, SignalDirection.STRONG_SELL]:
                            extreme_overbought = True
        
        if total_weight > 0:
            overall_numeric = weighted_sum / total_weight
            overall_confidence = weighted_confidence / total_weight
        else:
            overall_numeric = 0.0
            overall_confidence = 0.0
        
        # Classify divergences
        bullish_divergences = [d for d in divergences if d.divergence_type in 
                              [DivergenceType.REGULAR_BULLISH, DivergenceType.HIDDEN_BULLISH]]
        bearish_divergences = [d for d in divergences if d.divergence_type in 
                              [DivergenceType.REGULAR_BEARISH, DivergenceType.HIDDEN_BEARISH]]
        
        # Calculate divergence impact
        # Divergences are powerful reversal signals - they significantly shift conviction
        # Research shows divergences have ~65% predictive accuracy for reversals
        divergence_boost = 0.0
        divergence_confidence_boost = 0.0
        
        for div in bullish_divergences:
            # Bullish divergence adds to bullish signal
            # Regular divergence is stronger than hidden (reversal vs continuation)
            if div.divergence_type == DivergenceType.REGULAR_BULLISH:
                divergence_boost += 0.25 * div.strength
            else:  # Hidden bullish
                divergence_boost += 0.20 * div.strength
            divergence_confidence_boost += 0.08 * div.strength
        
        for div in bearish_divergences:
            # Bearish divergence adds to bearish signal
            if div.divergence_type == DivergenceType.REGULAR_BEARISH:
                divergence_boost -= 0.25 * div.strength
            else:  # Hidden bearish
                divergence_boost -= 0.20 * div.strength
            divergence_confidence_boost += 0.08 * div.strength
        
        # Apply divergence adjustment
        overall_numeric += divergence_boost
        overall_numeric = max(-1.0, min(1.0, overall_numeric))  # Clamp to [-1, 1]
        
        # Extreme reading boost
        # When momentum is at extremes AND divergences confirm, this is a high-conviction setup
        # Academic research on mean reversion supports this (Lo & MacKinlay, Poterba & Summers)
        extreme_boost = 0.0
        if extreme_oversold and bullish_divergences:
            # This is a textbook reversal setup - very high conviction
            extreme_boost = 0.18
            overall_numeric += 0.25  # Strong push toward BUY
        elif extreme_overbought and bearish_divergences:
            extreme_boost = 0.18
            overall_numeric -= 0.25  # Strong push toward SELL
        elif extreme_oversold:
            # Extreme oversold without divergence - still meaningful but less certain
            extreme_boost = 0.10
            overall_numeric += 0.15
        elif extreme_overbought:
            extreme_boost = 0.10
            overall_numeric -= 0.15
        
        # Clamp again after extreme boost
        overall_numeric = max(-1.0, min(1.0, overall_numeric))
        
        # Adjust confidence with divergence and extreme boosts
        # BUT also penalize for conflicting signals across families
        overall_confidence += divergence_confidence_boost + extreme_boost
        
        # CRITICAL: Penalize confidence when signals conflict
        # If we have both bullish AND bearish families, reduce confidence
        if bullish_count > 0 and bearish_count > 0:
            # Conflict penalty proportional to the minority side
            minority_count = min(bullish_count, bearish_count)
            total_directional = bullish_count + bearish_count
            conflict_severity = minority_count / total_directional  # 0.33 to 0.50
            conflict_penalty = 0.15 * conflict_severity * (bullish_count + bearish_count)
            overall_confidence -= conflict_penalty
        
        # Also penalize if neutral families dominate (lack of conviction)
        if neutral_count >= 2:
            indecision_penalty = 0.03 * neutral_count
            overall_confidence -= indecision_penalty
        
        overall_confidence = max(0.40, min(0.92, overall_confidence))  # Cap at 92%, floor at 40%
        
        # Determine overall signal from adjusted numeric
        overall_signal = SignalDirection.from_numeric(overall_numeric)
        
        # Determine signal strength with calibrated thresholds
        # These thresholds are based on both confidence and direction magnitude
        if overall_confidence >= 0.78 and abs(overall_numeric) >= 0.45:
            signal_strength = SignalStrength.VERY_STRONG
        elif overall_confidence >= 0.68 and abs(overall_numeric) >= 0.30:
            signal_strength = SignalStrength.STRONG
        elif overall_confidence >= 0.58 and abs(overall_numeric) >= 0.18:
            signal_strength = SignalStrength.MODERATE
        elif overall_confidence >= 0.48:
            signal_strength = SignalStrength.WEAK
        else:
            signal_strength = SignalStrength.VERY_WEAK
        
        # Build risk factors
        risk_factors = []
        
        if bullish_count > 0 and bearish_count > 0:
            risk_factors.append("Mixed signals across indicator families")
        
        # Divergence warnings (when divergence conflicts with signal)
        if bullish_divergences and overall_signal in [SignalDirection.SELL, SignalDirection.STRONG_SELL]:
            risk_factors.append(f"Bullish divergence ({len(bullish_divergences)}) conflicts with bearish signal - potential reversal")
        if bearish_divergences and overall_signal in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
            risk_factors.append(f"Bearish divergence ({len(bearish_divergences)}) conflicts with bullish signal - potential reversal")
        
        # Divergence confirmations (when divergence supports signal)
        confirmation_notes = []
        if bullish_divergences and overall_signal in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
            confirmation_notes.append(f"{len(bullish_divergences)} bullish divergence(s) confirm buy signal")
        if bearish_divergences and overall_signal in [SignalDirection.SELL, SignalDirection.STRONG_SELL]:
            confirmation_notes.append(f"{len(bearish_divergences)} bearish divergence(s) confirm sell signal")
        
        # Generate recommendation text
        if signal_strength == SignalStrength.VERY_STRONG:
            if overall_signal in [SignalDirection.STRONG_BUY, SignalDirection.BUY]:
                if extreme_oversold and bullish_divergences:
                    recommendation = "STRONG BUY: Extreme oversold with bullish divergence confirmation. High-conviction reversal setup. Consider full position."
                elif extreme_oversold:
                    recommendation = "STRONG BUY: Extreme oversold conditions. Consider aggressive entry with standard stops."
                else:
                    recommendation = "STRONG BUY: Strong bullish confluence across indicators. Consider full position with momentum."
            else:
                if extreme_overbought and bearish_divergences:
                    recommendation = "STRONG SELL: Extreme overbought with bearish divergence confirmation. High-conviction reversal setup. Exit longs or initiate shorts."
                elif extreme_overbought:
                    recommendation = "STRONG SELL: Extreme overbought conditions. Consider taking profits or hedging."
                else:
                    recommendation = "STRONG SELL: Strong bearish confluence. Exit long positions or consider short."
        elif signal_strength == SignalStrength.STRONG:
            if overall_signal in [SignalDirection.STRONG_BUY, SignalDirection.BUY]:
                if bullish_divergences:
                    recommendation = f"BUY: Bullish setup with divergence support. Scale into position on weakness."
                else:
                    recommendation = "BUY: Bullish bias with good confirmation. Consider scaled entry."
            else:
                if bearish_divergences:
                    recommendation = f"SELL: Bearish setup with divergence support. Reduce exposure or hedge."
                else:
                    recommendation = "SELL: Bearish bias confirmed. Reduce long exposure."
        elif signal_strength == SignalStrength.MODERATE:
            if overall_signal in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
                recommendation = "LEAN BULLISH: Moderate buy signal. Consider partial position or wait for pullback entry."
            elif overall_signal in [SignalDirection.SELL, SignalDirection.STRONG_SELL]:
                recommendation = "LEAN BEARISH: Moderate sell signal. Consider reducing exposure or tightening stops."
            else:
                recommendation = "NEUTRAL: Mixed signals. Wait for clearer setup before committing capital."
        else:
            recommendation = "NO TRADE: Weak or conflicting signals. Preserve capital and wait for better opportunity."
        
        # Add confirmation notes to recommendation if present
        if confirmation_notes:
            recommendation += f" ({', '.join(confirmation_notes)})"
        
        return ConfluenceAnalysis(
            overall_signal=overall_signal,
            overall_confidence=overall_confidence,
            signal_strength=signal_strength,
            families=families,
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            divergences=divergences,
            key_levels=key_levels,
            recommendation=recommendation,
            risk_factors=risk_factors
        )


# =============================================================================
# MAIN INDICATOR ENGINE
# =============================================================================

class TechnicalIndicatorEngine:
    """
    Main orchestrator for technical indicator computation.
    
    Integrates all indicator families, divergence detection, and
    confluence analysis into a unified pipeline.
    
    Usage
    -----
    >>> engine = TechnicalIndicatorEngine(volatility_regime=VolatilityRegime.NORMAL)
    >>> output = engine.process(df)
    >>> print(output.current_analysis.recommendation)
    """
    
    def __init__(
        self,
        volatility_regime: VolatilityRegime = VolatilityRegime.NORMAL,
        enable_divergence: bool = True
    ):
        """
        Initialize the indicator engine.
        
        Parameters
        ----------
        volatility_regime : VolatilityRegime
            Current market volatility regime for threshold adjustment
        enable_divergence : bool
            Whether to detect divergences (can be slow)
        """
        self.volatility_regime = volatility_regime
        self.enable_divergence = enable_divergence
        
        # Initialize regime adjuster
        self.adjuster = RegimeAdjuster(volatility_regime)
        
        # Initialize indicator calculators
        self.momentum = MomentumIndicators(self.adjuster)
        self.trend = TrendIndicators(self.adjuster)
        self.volatility = VolatilityIndicators(self.adjuster)
        self.volume = VolumeIndicators(self.adjuster)
        self.ichimoku = IchimokuIndicator()
        
        # Initialize analyzers
        self.divergence_detector = DivergenceDetector()
        self.confluence = ConfluenceAnalyzer()
        
        logger.info(f"TechnicalIndicatorEngine initialized with {volatility_regime.value} regime")
    
    def process(self, df: pd.DataFrame) -> IndicatorOutput:
        """
        Process OHLCV data through the complete indicator pipeline.
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with columns: Open, High, Low, Close, Volume
            Index should be DatetimeIndex
            
        Returns
        -------
        IndicatorOutput
            Complete indicator analysis including signals and recommendations
        """
        logger.info(f"Processing {len(df)} bars of data")
        
        # Validate input
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Initialize results DataFrame
        indicators_df = pd.DataFrame(index=df.index)
        signals_df = pd.DataFrame(index=df.index)
        all_signals = {}
        
        # 1. Compute Momentum Indicators
        momentum_df, momentum_signals = self.momentum.compute_all(df)
        indicators_df = pd.concat([indicators_df, momentum_df], axis=1)
        all_signals['momentum'] = momentum_signals
        
        # 2. Compute Trend Indicators
        trend_df, trend_signals = self.trend.compute_all(df)
        indicators_df = pd.concat([indicators_df, trend_df], axis=1)
        all_signals['trend'] = trend_signals
        
        # 3. Compute Volatility Indicators
        volatility_df, volatility_signals = self.volatility.compute_all(df)
        indicators_df = pd.concat([indicators_df, volatility_df], axis=1)
        all_signals['volatility'] = volatility_signals
        
        # 4. Compute Volume Indicators
        volume_df, volume_signals = self.volume.compute_all(df)
        indicators_df = pd.concat([indicators_df, volume_df], axis=1)
        all_signals['volume'] = volume_signals
        
        # 5. Compute Ichimoku
        ichimoku_df, ichimoku_signal, ichimoku_analysis = self.ichimoku.compute(df)
        indicators_df = pd.concat([indicators_df, ichimoku_df], axis=1)
        all_signals['system'] = {'ichimoku': ichimoku_signal}
        
        # 6. Detect Divergences
        divergences = []
        if self.enable_divergence:
            # RSI divergence
            divergences.extend(
                self.divergence_detector.detect(
                    df['Close'], indicators_df['rsi'], 'RSI'
                )
            )
            # MACD divergence
            divergences.extend(
                self.divergence_detector.detect(
                    df['Close'], indicators_df['macd_histogram'], 'MACD'
                )
            )
        
        # 7. Aggregate by Family
        families = {}
        for family_name, signals in all_signals.items():
            families[family_name] = self.confluence.aggregate_family(family_name, signals)
        
        # 8. Extract Key Levels
        key_levels = {
            'bb_upper': indicators_df['bb_upper'].iloc[-1],
            'bb_middle': indicators_df['bb_middle'].iloc[-1],
            'bb_lower': indicators_df['bb_lower'].iloc[-1],
            'kc_upper': indicators_df['kc_upper'].iloc[-1],
            'kc_lower': indicators_df['kc_lower'].iloc[-1],
            'ichimoku_cloud_top': ichimoku_analysis.cloud_top,
            'ichimoku_cloud_bottom': ichimoku_analysis.cloud_bottom,
            'supertrend': indicators_df['supertrend'].iloc[-1]
        }
        
        # 9. Confluence Analysis
        current_analysis = self.confluence.analyze(families, divergences, key_levels)
        
        # 10. Generate signal time series
        for col in ['rsi', 'stoch_k', 'macd', 'adx', 'bb_percent_b', 'mfi']:
            if col in indicators_df.columns:
                signals_df[f'{col}_signal'] = self._generate_signal_series(
                    indicators_df[col], col
                )
        
        # Determine period
        period = (
            df.index[0].strftime('%Y-%m-%d'),
            df.index[-1].strftime('%Y-%m-%d')
        )
        
        return IndicatorOutput(
            indicators_df=indicators_df,
            signals_df=signals_df,
            current_analysis=current_analysis,
            timeframe_analysis=None,  # Would need weekly/monthly data
            symbol=df.attrs.get('symbol', 'UNKNOWN'),
            period=period,
            volatility_regime=self.volatility_regime,
            regime_adjusted=self.volatility_regime != VolatilityRegime.NORMAL,
            generated_at=datetime.now().isoformat(),
            version=INDICATOR_VERSION
        )
    
    def _generate_signal_series(
        self,
        indicator: pd.Series,
        indicator_name: str
    ) -> pd.Series:
        """
        Generate signal series for an indicator over time.
        
        Parameters
        ----------
        indicator : pd.Series
            Indicator values
        indicator_name : str
            Name of the indicator
            
        Returns
        -------
        pd.Series
            Signal values (-1, 0, 1) over time
        """
        signals = pd.Series(index=indicator.index, dtype=float)
        
        if indicator_name == 'rsi':
            signals[indicator < self.momentum.rsi_os] = 1
            signals[indicator > self.momentum.rsi_ob] = -1
            signals[(indicator >= self.momentum.rsi_os) & 
                   (indicator <= self.momentum.rsi_ob)] = 0
        elif indicator_name == 'stoch_k':
            signals[indicator < self.momentum.stoch_os] = 1
            signals[indicator > self.momentum.stoch_ob] = -1
            signals[(indicator >= self.momentum.stoch_os) & 
                   (indicator <= self.momentum.stoch_ob)] = 0
        elif indicator_name == 'mfi':
            signals[indicator < 20] = 1
            signals[indicator > 80] = -1
            signals[(indicator >= 20) & (indicator <= 80)] = 0
        elif indicator_name == 'bb_percent_b':
            signals[indicator < 0] = 1
            signals[indicator > 1] = -1
            signals[(indicator >= 0) & (indicator <= 1)] = 0
        else:
            # Default: use zero crossover
            signals[indicator > 0] = 1
            signals[indicator < 0] = -1
            signals[indicator == 0] = 0
        
        return signals.fillna(0)


# =============================================================================
# REPORT GENERATION
# =============================================================================

def print_indicator_report(output: IndicatorOutput) -> None:
    """
    Print a comprehensive indicator report to console.
    
    Parameters
    ----------
    output : IndicatorOutput
        Output from TechnicalIndicatorEngine.process()
    """
    analysis = output.current_analysis
    
    print("\n" + "=" * 70)
    print("TECHNICAL INDICATOR ANALYSIS REPORT")
    print("=" * 70)
    print(f"Symbol: {output.symbol}")
    print(f"Period: {output.period[0]} to {output.period[1]}")
    print(f"Volatility Regime: {output.volatility_regime.value}")
    print(f"Generated: {output.generated_at}")
    print(f"Version: {output.version}")
    
    # Overall Signal
    print("\n" + "-" * 70)
    print("OVERALL SIGNAL")
    print("-" * 70)
    print(f"Direction: {analysis.overall_signal.value}")
    print(f"Confidence: {analysis.overall_confidence:.1%}")
    print(f"Strength: {analysis.signal_strength.value}")
    print(f"\nRecommendation: {analysis.recommendation}")
    
    # Signal Count
    print(f"\nSignal Distribution:")
    print(f"  Bullish families: {analysis.bullish_count}")
    print(f"  Bearish families: {analysis.bearish_count}")
    print(f"  Neutral families: {analysis.neutral_count}")
    
    # Family Breakdown
    print("\n" + "-" * 70)
    print("INDICATOR FAMILY BREAKDOWN")
    print("-" * 70)
    
    for family_name, family in analysis.families.items():
        print(f"\n{family_name.upper()} (weight: {family.weight:.0%})")
        print(f"  Aggregate: {family.aggregate_signal.value} ({family.aggregate_confidence:.1%})")
        for ind_name, signal in family.indicators.items():
            print(f"    {ind_name}: {signal.direction.value} ({signal.confidence:.1%}) - {signal.zone}")
            for factor in signal.factors[:2]:  # Limit factors shown
                print(f"      -> {factor}")
    
    # Divergences
    if analysis.divergences:
        print("\n" + "-" * 70)
        print("DETECTED DIVERGENCES")
        print("-" * 70)
        for div in analysis.divergences:
            print(f"  {div.divergence_type.value} on {div.indicator_name}")
            print(f"    Period: {div.bars_duration} bars, Strength: {div.strength:.1%}")
    
    # Key Levels
    print("\n" + "-" * 70)
    print("KEY TECHNICAL LEVELS")
    print("-" * 70)
    for name, level in analysis.key_levels.items():
        if not pd.isna(level) and level != 0:
            print(f"  {name}: {level:.2f}")
    
    # Risk Factors
    if analysis.risk_factors:
        print("\n" + "-" * 70)
        print("RISK FACTORS")
        print("-" * 70)
        for risk in analysis.risk_factors:
            print(f"  ! {risk}")
    
    print("\n" + "=" * 70)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main() -> int:
    """
    Main entry point for command-line execution.
    
    Returns
    -------
    int
        Exit code (0 for success, 1 for failure)
    """
    import argparse
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description="Technical Indicator Engine - Phase 2"
    )
    parser.add_argument("--input", "-i", required=True,
                       help="Input Parquet file from Phase 1")
    parser.add_argument("--regime", "-r", default="NORMAL",
                       choices=["LOW", "NORMAL", "HIGH", "EXTREME"],
                       help="Volatility regime (default: NORMAL)")
    parser.add_argument("--output", "-o", default=None,
                       help="Output Parquet file for indicators")
    
    args = parser.parse_args()
    
    try:
        # Load data
        logger.info(f"Loading data from {args.input}")
        df = pd.read_parquet(args.input)
        
        # Determine regime
        regime = VolatilityRegime[args.regime]
        
        # Process
        engine = TechnicalIndicatorEngine(volatility_regime=regime)
        output = engine.process(df)
        
        # Print report
        print_indicator_report(output)
        
        # Save if requested
        if args.output:
            output.indicators_df.to_parquet(args.output)
            logger.info(f"Saved indicators to {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())