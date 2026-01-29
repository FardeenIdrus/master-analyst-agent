#!/usr/bin/env python3
"""
Phase 3: Quantitative Market Regime Detection System
=====================================================

MSc AI Agents in Asset Management (IFTE0001) - Track B: Technical Analyst Agent

This module implements institutional-grade market regime detection using
probabilistic models and statistical analysis. The detected regime informs
position sizing, strategy selection, and risk management in Phase 4 backtesting.

COURSEWORK REQUIREMENT ADDRESSED
--------------------------------
"Backtest with transaction costs and position sizing" requires understanding
current market regime to:
    1. Adapt strategy parameters to market conditions
    2. Scale position sizes based on volatility regime
    3. Set appropriate stop-losses for the environment
    4. Generate regime-aware trading signals

ACADEMIC FOUNDATIONS
--------------------
Hidden Markov Models (HMM):
    Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of
    Nonstationary Time Series and the Business Cycle." Econometrica, 57(2).
    
    Financial returns exhibit regime-switching behavior where the 
    data-generating process alternates between distinct states.

GARCH Volatility:
    Bollerslev, T. (1986). "Generalized Autoregressive Conditional
    Heteroskedasticity." Journal of Econometrics, 31(3).
    
    Volatility clusters - large changes tend to follow large changes.

Hurst Exponent:
    Hurst, H.E. (1951). "Long-term storage capacity of reservoirs."
    
    H > 0.5: Persistent (trending)
    H = 0.5: Random walk
    H < 0.5: Mean-reverting

ARCHITECTURE
------------
    Layer 1: Core Regime Detectors
        - GaussianHMM: 3-state regime classification
        - GARCH: Volatility regime and forecasting
        - HurstAnalyzer: Trend persistence measurement
        - BreakpointDetector: Structural change detection

    Layer 2: Ensemble Integration  
        - RegimeEnsemble: Combine multiple signals
        - ConfidenceCalibrator: Score reliability
        - StrategyRecommender: Regime-adaptive positioning

    Layer 3: Output Generation
        - RegimeReport: Complete analysis output
        - Integration with Phase 4 backtesting

Author: Tamer
Version: 2.0.0
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

# Suppress numerical warnings for clean output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

VERSION = "2.0.0"


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

class Config:
    """
    Centralized configuration for regime detection parameters.
    
    All parameters are based on academic research and institutional practice.
    Values can be adjusted for different asset classes or market conditions.
    """
    
    # -------------------------------------------------------------------------
    # Trading Calendar
    # -------------------------------------------------------------------------
    TRADING_DAYS_YEAR: int = 252      # Standard US equity trading days
    TRADING_DAYS_QUARTER: int = 63    # ~3 months
    TRADING_DAYS_MONTH: int = 21      # ~1 month
    TRADING_DAYS_WEEK: int = 5        # ~1 week
    
    # -------------------------------------------------------------------------
    # Hidden Markov Model Parameters
    # -------------------------------------------------------------------------
    HMM_N_STATES: int = 3             # Bull, Bear, Sideways
    HMM_N_ITER: int = 100             # EM iterations
    HMM_N_INIT: int = 10              # Random initializations
    HMM_COVARIANCE_TYPE: str = 'full' # Full covariance matrix
    HMM_MIN_SAMPLES: int = 252        # Minimum 1 year for training
    HMM_CONVERGENCE_TOL: float = 1e-4 # Convergence threshold
    
    # -------------------------------------------------------------------------
    # GARCH Parameters
    # -------------------------------------------------------------------------
    GARCH_P: int = 1                  # ARCH order
    GARCH_Q: int = 1                  # GARCH order
    GARCH_DIST: str = 'normal'        # Error distribution
    GARCH_FORECAST_HORIZON: int = 5   # Days to forecast
    
    # -------------------------------------------------------------------------
    # Hurst Exponent Parameters
    # -------------------------------------------------------------------------
    HURST_MIN_WINDOW: int = 20        # Minimum R/S window
    HURST_MAX_WINDOW: int = 200       # Maximum R/S window
    HURST_N_WINDOWS: int = 20         # Number of window sizes
    
    # -------------------------------------------------------------------------
    # Volatility Regime Thresholds (annualized)
    # -------------------------------------------------------------------------
    VOL_LOW_THRESHOLD: float = 0.12   # Below 12% = low vol
    VOL_NORMAL_LOW: float = 0.15      # 15% = normal low
    VOL_NORMAL_HIGH: float = 0.25     # 25% = normal high
    VOL_HIGH_THRESHOLD: float = 0.35  # Above 35% = high vol
    VOL_CRISIS_THRESHOLD: float = 0.50 # Above 50% = crisis
    
    # -------------------------------------------------------------------------
    # Regime Classification Thresholds
    # -------------------------------------------------------------------------
    BULL_RETURN_THRESHOLD: float = 0.0003   # Daily return for bull
    BEAR_RETURN_THRESHOLD: float = -0.0003  # Daily return for bear
    
    # -------------------------------------------------------------------------
    # Strategy Recommendation
    # -------------------------------------------------------------------------
    TREND_POSITION_SIZE: float = 1.0       # Full position in trends
    SIDEWAYS_POSITION_SIZE: float = 0.5    # Half position sideways
    BEAR_POSITION_SIZE: float = 0.25       # Quarter position in bear
    
    TREND_STOP_ATR: float = 2.0            # Stop loss in ATR units
    SIDEWAYS_STOP_ATR: float = 1.5
    BEAR_STOP_ATR: float = 1.0
    
    # -------------------------------------------------------------------------
    # Confidence Scoring
    # -------------------------------------------------------------------------
    MIN_CONFIDENCE: float = 0.50
    MAX_CONFIDENCE: float = 0.95
    
    CONFIDENCE_WEIGHTS: Dict[str, float] = {
        'hmm_probability': 0.35,
        'vol_agreement': 0.25,
        'hurst_clarity': 0.20,
        'trend_strength': 0.20,
    }


# =============================================================================
# SECTION 2: ENUMERATIONS
# =============================================================================

class MarketRegime(Enum):
    """
    Primary market regime classifications.
    
    Derived from HMM state inference where states are mapped to
    regimes based on return and volatility characteristics.
    """
    BULL = "BULL"           # Positive returns, often lower volatility
    BEAR = "BEAR"           # Negative returns, higher volatility
    SIDEWAYS = "SIDEWAYS"   # Near-zero returns, variable volatility


class VolatilityRegime(Enum):
    """
    Volatility regime classification based on annualized volatility.
    
    Thresholds calibrated to historical S&P 500 data:
        - VIX < 12: Extremely low (unusual calm)
        - VIX 12-20: Normal/low
        - VIX 20-30: Elevated
        - VIX 30-50: High stress
        - VIX > 50: Crisis
    """
    VERY_LOW = "VERY_LOW"   # < 12% annualized
    LOW = "LOW"             # 12-15%
    NORMAL = "NORMAL"       # 15-25%
    HIGH = "HIGH"           # 25-35%
    VERY_HIGH = "VERY_HIGH" # 35-50%
    CRISIS = "CRISIS"       # > 50%


class TrendPersistence(Enum):
    """
    Trend persistence based on Hurst exponent.
    
    H = 0.5: Random walk (Brownian motion)
    H > 0.5: Persistent (trending) - past direction likely to continue
    H < 0.5: Anti-persistent (mean-reverting) - reversals likely
    """
    STRONG_TRENDING = "STRONG_TRENDING"         # H > 0.65
    WEAK_TRENDING = "WEAK_TRENDING"             # 0.55 < H <= 0.65
    RANDOM_WALK = "RANDOM_WALK"                 # 0.45 <= H <= 0.55
    WEAK_MEAN_REVERTING = "WEAK_MEAN_REVERTING" # 0.35 <= H < 0.45
    STRONG_MEAN_REVERTING = "STRONG_MEAN_REVERTING"  # H < 0.35


class StrategyType(Enum):
    """
    Recommended strategy based on regime analysis.
    
    Each regime favors different trading approaches:
        - Trending: Momentum/trend-following
        - Mean-reverting: Range trading/mean reversion
        - High volatility: Defensive/reduced exposure
    """
    TREND_FOLLOWING = "TREND_FOLLOWING"
    MOMENTUM = "MOMENTUM"
    MEAN_REVERSION = "MEAN_REVERSION"
    DEFENSIVE = "DEFENSIVE"
    NEUTRAL = "NEUTRAL"


class PositionBias(Enum):
    """Position direction recommendation."""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class QualityGrade(Enum):
    """Analysis quality assessment grades."""
    EXCELLENT = "EXCELLENT"   # 90-100
    GOOD = "GOOD"             # 75-89
    ACCEPTABLE = "ACCEPTABLE" # 60-74
    POOR = "POOR"             # 40-59
    UNUSABLE = "UNUSABLE"     # < 40


# =============================================================================
# SECTION 3: DATA STRUCTURES
# =============================================================================

@dataclass
class HMMStateParameters:
    """
    Parameters for a single HMM state.
    
    These describe the emission distribution for each hidden state,
    allowing interpretation of what each state represents.
    """
    state_id: int
    regime: MarketRegime
    mean_return: float          # Daily mean return
    volatility: float           # Daily volatility
    annualized_return: float    # Annualized mean
    annualized_volatility: float  # Annualized vol
    stationary_probability: float  # Long-run probability


@dataclass
class HMMAnalysis:
    """
    Complete HMM regime analysis output.
    
    Contains current regime classification, probabilities, and
    state parameters for all detected regimes.
    """
    # Current state
    current_regime: MarketRegime
    regime_probability: float   # Probability of current regime
    
    # State probabilities
    state_probabilities: Dict[MarketRegime, float]
    
    # State parameters
    state_parameters: Dict[MarketRegime, HMMStateParameters]
    
    # Regime history
    regime_sequence: np.ndarray  # Historical regime classifications
    
    # Transition probabilities
    transition_matrix: np.ndarray
    
    # Model quality
    log_likelihood: float
    aic: float
    bic: float
    converged: bool
    
    # Expected durations (days)
    expected_durations: Dict[MarketRegime, float]


@dataclass
class GARCHAnalysis:
    """
    GARCH volatility analysis output.
    
    Provides current volatility estimate and forecasts for
    risk management and position sizing.
    """
    # Current volatility
    current_volatility: float       # Annualized
    daily_volatility: float         # Daily
    
    # Volatility regime
    vol_regime: VolatilityRegime
    
    # Forecasts
    forecast_1d: float              # 1-day ahead
    forecast_5d: float              # 5-day ahead (average)
    forecast_horizon: np.ndarray    # Full forecast path
    
    # Model parameters
    omega: float      # Constant
    alpha: float      # ARCH coefficient
    beta: float       # GARCH coefficient
    persistence: float  # alpha + beta
    
    # Model quality
    log_likelihood: float
    aic: float
    converged: bool
    
    # Volatility statistics
    long_run_volatility: float    # Unconditional vol
    vol_percentile: float         # Current vs historical


@dataclass
class HurstAnalysis:
    """
    Hurst exponent analysis output.
    
    Measures trend persistence/mean reversion tendency
    using Rescaled Range (R/S) analysis.
    """
    # Main result
    hurst_exponent: float
    persistence: TrendPersistence
    
    # Statistical significance
    standard_error: float
    confidence_interval: Tuple[float, float]
    r_squared: float              # Regression fit quality
    
    # Interpretation
    interpretation: str
    
    # Raw data
    log_n: np.ndarray             # Log of window sizes
    log_rs: np.ndarray            # Log of R/S values


@dataclass
class BreakpointAnalysis:
    """
    Structural breakpoint detection output.
    
    Identifies potential regime changes based on
    statistical tests for mean/variance shifts.
    """
    # Breakpoints detected
    breakpoints: List[datetime]
    n_breaks: int
    
    # Most recent break
    last_break_date: Optional[datetime]
    days_since_break: int
    
    # Current segment statistics
    segment_mean: float
    segment_volatility: float
    segment_length: int
    
    # Test statistics
    cusum_stat: float
    cusum_critical: float
    break_detected: bool


@dataclass  
class StrategyRecommendation:
    """
    Trading strategy recommendation based on regime.
    
    Combines regime analysis to provide actionable guidance
    for position sizing and risk management.
    """
    # Strategy type
    strategy: StrategyType
    position_bias: PositionBias
    
    # Position sizing
    position_size: float          # 0.0 to 1.0
    leverage_factor: float        # Multiplier (typically 1.0)
    
    # Risk management
    stop_loss_atr: float          # ATR multiple for stop
    take_profit_atr: float        # ATR multiple for target
    max_holding_days: int         # Maximum hold period
    
    # Confidence
    confidence: float             # 0.5 to 0.95
    
    # Rationale
    rationale: List[str]
    warnings: List[str]


@dataclass
class RegimeReport:
    """
    Complete regime detection report.
    
    Master output structure containing all analysis results
    for use by Phase 4 backtesting and Phase 5 LLM generation.
    """
    # Metadata
    symbol: str
    analysis_date: datetime
    period_start: datetime
    period_end: datetime
    n_observations: int
    
    # Core analysis
    hmm: HMMAnalysis
    garch: GARCHAnalysis
    hurst: HurstAnalysis
    breakpoints: BreakpointAnalysis
    
    # Ensemble result
    consensus_regime: MarketRegime
    consensus_confidence: float
    
    # Strategy recommendation
    strategy: StrategyRecommendation
    
    # Quality assessment
    quality_score: float
    quality_grade: QualityGrade
    
    # Processing info
    processing_time_ms: int
    version: str = VERSION


# =============================================================================
# SECTION 4: UTILITY FUNCTIONS
# =============================================================================

def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Compute log returns from price series.
    
    Log returns are preferred for:
        1. Time additivity (sum of log returns = total return)
        2. Closer to normal distribution
        3. Better statistical properties
    
    Args:
        prices: Price series (Close prices)
        
    Returns:
        Log return series (length n-1)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        log_prices = np.log(prices)
        returns = np.diff(log_prices)
    # Remove NaN/Inf values
    returns = returns[np.isfinite(returns)]
    return returns


def annualize_return(daily_return: float) -> float:
    """Annualize daily mean return assuming 252 trading days."""
    return daily_return * Config.TRADING_DAYS_YEAR


def annualize_volatility(daily_vol: float) -> float:
    """Annualize daily volatility using square-root of time rule."""
    return daily_vol * np.sqrt(Config.TRADING_DAYS_YEAR)


def classify_volatility_regime(annualized_vol: float) -> VolatilityRegime:
    """
    Classify volatility into regime based on thresholds.
    
    Thresholds calibrated to historical market data.
    """
    if annualized_vol < Config.VOL_LOW_THRESHOLD:
        return VolatilityRegime.VERY_LOW
    elif annualized_vol < Config.VOL_NORMAL_LOW:
        return VolatilityRegime.LOW
    elif annualized_vol < Config.VOL_NORMAL_HIGH:
        return VolatilityRegime.NORMAL
    elif annualized_vol < Config.VOL_HIGH_THRESHOLD:
        return VolatilityRegime.HIGH
    elif annualized_vol < Config.VOL_CRISIS_THRESHOLD:
        return VolatilityRegime.VERY_HIGH
    else:
        return VolatilityRegime.CRISIS


def classify_hurst(h: float) -> TrendPersistence:
    """
    Classify Hurst exponent into persistence category.
    
    Args:
        h: Hurst exponent (0 to 1)
        
    Returns:
        TrendPersistence classification
    """
    if h > 0.65:
        return TrendPersistence.STRONG_TRENDING
    elif h > 0.55:
        return TrendPersistence.WEAK_TRENDING
    elif h >= 0.45:
        return TrendPersistence.RANDOM_WALK
    elif h >= 0.35:
        return TrendPersistence.WEAK_MEAN_REVERTING
    else:
        return TrendPersistence.STRONG_MEAN_REVERTING


def score_to_grade(score: float) -> QualityGrade:
    """Convert numerical quality score to grade."""
    if score >= 90:
        return QualityGrade.EXCELLENT
    elif score >= 75:
        return QualityGrade.GOOD
    elif score >= 60:
        return QualityGrade.ACCEPTABLE
    elif score >= 40:
        return QualityGrade.POOR
    else:
        return QualityGrade.UNUSABLE


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division handling zero and invalid values."""
    try:
        if b == 0 or not np.isfinite(b):
            return default
        result = a / b
        return default if not np.isfinite(result) else result
    except (ZeroDivisionError, TypeError, ValueError):
        return default


# =============================================================================
# SECTION 5: HIDDEN MARKOV MODEL REGIME DETECTOR
# =============================================================================

class GaussianHMM:
    """
    Gaussian Hidden Markov Model for regime detection.
    
    Implements a 3-state Gaussian HMM where each hidden state corresponds
    to a market regime (Bull, Bear, Sideways). The model is trained using
    the Baum-Welch (EM) algorithm and inference is done via Forward-Backward.
    
    Mathematical Framework:
    -----------------------
    Let X_t be the hidden state at time t, and Y_t be the observed return.
    
    Transition Model: P(X_t | X_{t-1}) = A_{X_{t-1}, X_t}
    Emission Model: P(Y_t | X_t) = N(μ_{X_t}, σ²_{X_t})
    
    The model parameters θ = (π, A, μ, σ²) are estimated via EM:
        E-step: Compute posterior P(X_t | Y_{1:T}, θ)
        M-step: Update θ to maximize expected complete log-likelihood
    
    State Mapping:
    -------------
    After training, states are mapped to regimes based on mean return:
        - Highest mean → BULL
        - Lowest mean → BEAR
        - Middle mean → SIDEWAYS
    
    References:
        Hamilton (1989), Rabiner (1989)
    """
    
    def __init__(
        self,
        n_states: int = Config.HMM_N_STATES,
        n_iter: int = Config.HMM_N_ITER,
        n_init: int = Config.HMM_N_INIT,
        random_state: int = 42
    ):
        """
        Initialize Gaussian HMM.
        
        Args:
            n_states: Number of hidden states (default: 3)
            n_iter: Maximum EM iterations per initialization
            n_init: Number of random initializations (best selected)
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.n_init = n_init
        self.random_state = random_state
        
        # Model parameters (to be learned)
        self.pi: np.ndarray = None      # Initial state distribution
        self.A: np.ndarray = None       # Transition matrix
        self.means: np.ndarray = None   # State means
        self.variances: np.ndarray = None  # State variances
        
        # State mapping
        self._state_to_regime: Dict[int, MarketRegime] = {}
        
        # Fit diagnostics
        self._fitted = False
        self._log_likelihood = -np.inf
        self._n_iter_final = 0
        self._converged = False
        
    def _initialize_parameters(self, X: np.ndarray) -> None:
        """
        Initialize model parameters using data-driven approach.
        
        Uses K-means-like initialization for means, empirical variance,
        and uniform initial/transition probabilities.
        """
        n = len(X)
        rng = np.random.default_rng(self.random_state)
        
        # Initialize means using percentiles (robust to outliers)
        percentiles = [20, 50, 80]
        self.means = np.percentile(X, percentiles)
        
        # Initialize variances as fraction of total variance
        total_var = np.var(X)
        self.variances = np.array([total_var * 1.5, total_var, total_var * 0.8])
        
        # Uniform initial distribution
        self.pi = np.ones(self.n_states) / self.n_states
        
        # Initialize transition matrix with high self-transition
        # (regimes tend to persist)
        self.A = np.full((self.n_states, self.n_states), 0.05)
        np.fill_diagonal(self.A, 0.90)
        self.A = self.A / self.A.sum(axis=1, keepdims=True)
        
    def _gaussian_pdf(self, x: float, mean: float, var: float) -> float:
        """Compute Gaussian PDF value."""
        if var <= 0:
            var = 1e-10
        return np.exp(-0.5 * (x - mean)**2 / var) / np.sqrt(2 * np.pi * var)
    
    def _compute_emission_probs(self, X: np.ndarray) -> np.ndarray:
        """
        Compute emission probabilities P(Y_t | X_t = k) for all t, k.
        
        Returns:
            B: Array of shape (T, n_states) with emission probabilities
        """
        T = len(X)
        B = np.zeros((T, self.n_states))
        
        for k in range(self.n_states):
            for t in range(T):
                B[t, k] = self._gaussian_pdf(X[t], self.means[k], self.variances[k])
                
        # Prevent underflow
        B = np.clip(B, 1e-300, None)
        return B
    
    def _forward(self, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward algorithm to compute P(Y_{1:t}, X_t = k).
        
        Args:
            B: Emission probabilities (T x n_states)
            
        Returns:
            alpha: Forward probabilities (T x n_states)
            scale: Scaling factors for numerical stability
        """
        T = B.shape[0]
        alpha = np.zeros((T, self.n_states))
        scale = np.zeros(T)
        
        # Initialize
        alpha[0] = self.pi * B[0]
        scale[0] = alpha[0].sum()
        if scale[0] > 0:
            alpha[0] /= scale[0]
        
        # Forward recursion
        for t in range(1, T):
            for k in range(self.n_states):
                alpha[t, k] = B[t, k] * np.sum(alpha[t-1] * self.A[:, k])
            scale[t] = alpha[t].sum()
            if scale[t] > 0:
                alpha[t] /= scale[t]
                
        return alpha, scale
    
    def _backward(self, B: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """
        Backward algorithm to compute P(Y_{t+1:T} | X_t = k).
        
        Args:
            B: Emission probabilities
            scale: Scaling factors from forward pass
            
        Returns:
            beta: Backward probabilities (T x n_states)
        """
        T = B.shape[0]
        beta = np.zeros((T, self.n_states))
        
        # Initialize
        beta[-1] = 1.0
        
        # Backward recursion
        for t in range(T - 2, -1, -1):
            for k in range(self.n_states):
                beta[t, k] = np.sum(self.A[k] * B[t+1] * beta[t+1])
            if scale[t+1] > 0:
                beta[t] /= scale[t+1]
                
        return beta
    
    def _e_step(self, X: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        E-step: Compute posterior probabilities.
        
        Returns:
            gamma: P(X_t = k | Y_{1:T}) for state marginals
            xi: P(X_t = j, X_{t+1} = k | Y_{1:T}) for transitions
            log_likelihood: Log P(Y_{1:T})
        """
        alpha, scale = self._forward(B)
        beta = self._backward(B, scale)
        
        T = len(X)
        
        # Compute gamma (state marginals)
        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum = np.where(gamma_sum == 0, 1, gamma_sum)
        gamma = gamma / gamma_sum
        
        # Compute xi (transition marginals)
        xi = np.zeros((T - 1, self.n_states, self.n_states))
        for t in range(T - 1):
            denom = 0.0
            for j in range(self.n_states):
                for k in range(self.n_states):
                    xi[t, j, k] = alpha[t, j] * self.A[j, k] * B[t+1, k] * beta[t+1, k]
                    denom += xi[t, j, k]
            if denom > 0:
                xi[t] /= denom
                
        # Log-likelihood from scaling factors
        log_likelihood = np.sum(np.log(scale + 1e-300))
        
        return gamma, xi, log_likelihood
    
    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray) -> None:
        """
        M-step: Update model parameters using posteriors.
        """
        T = len(X)
        
        # Update initial distribution
        self.pi = gamma[0] / gamma[0].sum()
        
        # Update transition matrix
        for j in range(self.n_states):
            denom = gamma[:-1, j].sum()
            if denom > 0:
                for k in range(self.n_states):
                    self.A[j, k] = xi[:, j, k].sum() / denom
        # Normalize rows
        self.A = self.A / self.A.sum(axis=1, keepdims=True)
        
        # Update means and variances
        for k in range(self.n_states):
            gamma_sum = gamma[:, k].sum()
            if gamma_sum > 0:
                self.means[k] = np.sum(gamma[:, k] * X) / gamma_sum
                self.variances[k] = np.sum(gamma[:, k] * (X - self.means[k])**2) / gamma_sum
                # Ensure minimum variance
                self.variances[k] = max(self.variances[k], 1e-10)
    
    def fit(self, X: np.ndarray) -> 'GaussianHMM':
        """
        Fit HMM to observation sequence using EM algorithm.
        
        Uses multiple random initializations and selects the model
        with highest log-likelihood.
        
        Args:
            X: Observation sequence (returns)
            
        Returns:
            self (fitted model)
        """
        if len(X) < Config.HMM_MIN_SAMPLES:
            raise ValueError(f"Need at least {Config.HMM_MIN_SAMPLES} samples")
        
        X = np.asarray(X).flatten()
        best_ll = -np.inf
        best_params = None
        
        for init in range(self.n_init):
            # Initialize with random perturbation
            rng = np.random.default_rng(self.random_state + init)
            self._initialize_parameters(X)
            
            # Add random perturbation to means
            self.means += rng.normal(0, np.std(X) * 0.1, self.n_states)
            
            prev_ll = -np.inf
            for iteration in range(self.n_iter):
                B = self._compute_emission_probs(X)
                gamma, xi, ll = self._e_step(X, B)
                self._m_step(X, gamma, xi)
                
                # Check convergence
                if abs(ll - prev_ll) < Config.HMM_CONVERGENCE_TOL:
                    break
                prev_ll = ll
            
            # Keep best model
            if ll > best_ll:
                best_ll = ll
                best_params = (
                    self.pi.copy(),
                    self.A.copy(),
                    self.means.copy(),
                    self.variances.copy(),
                    iteration + 1
                )
        
        # Restore best parameters
        if best_params is not None:
            self.pi, self.A, self.means, self.variances, self._n_iter_final = best_params
            self._log_likelihood = best_ll
            self._converged = self._n_iter_final < self.n_iter
        
        # Map states to regimes based on mean return
        self._map_states_to_regimes()
        self._fitted = True
        
        return self
    
    def _map_states_to_regimes(self) -> None:
        """
        Map HMM states to market regimes based on mean returns.
        
        Highest mean → BULL, Lowest mean → BEAR, Middle → SIDEWAYS
        """
        sorted_indices = np.argsort(self.means)
        self._state_to_regime = {
            int(sorted_indices[0]): MarketRegime.BEAR,
            int(sorted_indices[1]): MarketRegime.SIDEWAYS,
            int(sorted_indices[2]): MarketRegime.BULL
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict most likely state sequence (Viterbi algorithm).
        
        Args:
            X: Observation sequence
            
        Returns:
            Most likely hidden state sequence
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")
            
        X = np.asarray(X).flatten()
        T = len(X)
        B = self._compute_emission_probs(X)
        
        # Viterbi algorithm
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # Initialize
        delta[0] = np.log(self.pi + 1e-300) + np.log(B[0] + 1e-300)
        
        # Forward pass
        for t in range(1, T):
            for k in range(self.n_states):
                trans_probs = delta[t-1] + np.log(self.A[:, k] + 1e-300)
                psi[t, k] = np.argmax(trans_probs)
                delta[t, k] = np.max(trans_probs) + np.log(B[t, k] + 1e-300)
        
        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
            
        return states
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute posterior state probabilities.
        
        Args:
            X: Observation sequence
            
        Returns:
            Posterior probabilities (T x n_states)
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")
            
        X = np.asarray(X).flatten()
        B = self._compute_emission_probs(X)
        alpha, scale = self._forward(B)
        beta = self._backward(B, scale)
        
        gamma = alpha * beta
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        
        return gamma
    
    def get_current_regime(self, X: np.ndarray) -> Tuple[MarketRegime, float]:
        """
        Get current regime and its probability.
        
        Args:
            X: Observation sequence
            
        Returns:
            (current_regime, probability)
        """
        proba = self.predict_proba(X)[-1]
        state = np.argmax(proba)
        regime = self._state_to_regime[state]
        probability = proba[state]
        
        return regime, probability
    
    def get_state_probabilities(self, X: np.ndarray) -> Dict[MarketRegime, float]:
        """
        Get probabilities for all regimes at current time.
        
        Returns:
            Dict mapping regime to probability
        """
        proba = self.predict_proba(X)[-1]
        return {
            self._state_to_regime[k]: proba[k]
            for k in range(self.n_states)
        }
    
    def get_transition_matrix(self) -> Dict[Tuple[MarketRegime, MarketRegime], float]:
        """
        Get transition matrix with regime labels.
        
        Returns:
            Dict mapping (from_regime, to_regime) to probability
        """
        result = {}
        for i in range(self.n_states):
            for j in range(self.n_states):
                from_regime = self._state_to_regime[i]
                to_regime = self._state_to_regime[j]
                result[(from_regime, to_regime)] = self.A[i, j]
        return result
    
    def get_expected_durations(self) -> Dict[MarketRegime, float]:
        """
        Compute expected duration in each regime.
        
        Expected duration in state k = 1 / (1 - A[k,k])
        
        Returns:
            Dict mapping regime to expected duration in days
        """
        durations = {}
        for k in range(self.n_states):
            regime = self._state_to_regime[k]
            self_trans = self.A[k, k]
            duration = 1.0 / (1.0 - self_trans + 1e-10)
            durations[regime] = min(duration, 1000)  # Cap at reasonable max
        return durations
    
    def get_stationary_distribution(self) -> Dict[MarketRegime, float]:
        """
        Compute stationary (long-run) distribution.
        
        This is the left eigenvector of A corresponding to eigenvalue 1.
        
        Returns:
            Dict mapping regime to stationary probability
        """
        # Solve π = π A (find left eigenvector)
        eigenvalues, eigenvectors = np.linalg.eig(self.A.T)
        # Find eigenvector for eigenvalue ≈ 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        
        return {
            self._state_to_regime[k]: max(0, stationary[k])
            for k in range(self.n_states)
        }
    
    def compute_aic_bic(self, X: np.ndarray) -> Tuple[float, float]:
        """
        Compute AIC and BIC for model selection.
        
        AIC = -2 * log(L) + 2 * k
        BIC = -2 * log(L) + k * log(n)
        
        where k = number of parameters, n = number of observations
        """
        n = len(X)
        # Parameters: (n_states - 1) initial + n_states*(n_states-1) transition
        # + n_states means + n_states variances
        k = (self.n_states - 1) + self.n_states * (self.n_states - 1) + 2 * self.n_states
        
        aic = -2 * self._log_likelihood + 2 * k
        bic = -2 * self._log_likelihood + k * np.log(n)
        
        return aic, bic


# =============================================================================
# SECTION 6: GARCH VOLATILITY ANALYZER
# =============================================================================

class GARCHAnalyzer:
    """
    GARCH(1,1) volatility model for regime-aware risk management.
    
    Implements the standard GARCH(1,1) model:
        σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
    
    where:
        - σ²_t: Conditional variance at time t
        - ω: Constant (baseline variance)
        - α: ARCH coefficient (shock sensitivity)
        - β: GARCH coefficient (variance persistence)
        - ε_t: Return shock (r_t - μ)
    
    Key Properties:
        - Persistence = α + β (should be < 1 for stationarity)
        - Long-run variance = ω / (1 - α - β)
        - Half-life of shocks = log(0.5) / log(α + β)
    
    Reference:
        Bollerslev, T. (1986). "Generalized Autoregressive Conditional
        Heteroskedasticity." Journal of Econometrics.
    """
    
    def __init__(
        self,
        p: int = Config.GARCH_P,
        q: int = Config.GARCH_Q
    ):
        """
        Initialize GARCH model.
        
        Args:
            p: ARCH order (lag of squared returns)
            q: GARCH order (lag of conditional variance)
        """
        self.p = p
        self.q = q
        
        # Parameters
        self.omega: float = None
        self.alpha: float = None
        self.beta: float = None
        self.mu: float = None  # Mean return
        
        # Fit diagnostics
        self._fitted = False
        self._log_likelihood = -np.inf
        self._converged = False
        self._n_obs = 0
        
    def _garch_likelihood(
        self,
        params: np.ndarray,
        returns: np.ndarray
    ) -> float:
        """
        Compute negative log-likelihood for GARCH(1,1).
        
        Used as objective function for optimization.
        """
        omega, alpha, beta, mu = params
        
        # Parameter constraints
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        
        n = len(returns)
        eps = returns - mu  # Residuals
        
        # Initialize variance with unconditional variance
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(eps)
        
        # GARCH recursion
        for t in range(1, n):
            sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
            sigma2[t] = max(sigma2[t], 1e-10)  # Ensure positive
        
        # Log-likelihood (assuming normal errors)
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + eps**2 / sigma2)
        
        return -ll  # Return negative for minimization
    
    def fit(self, returns: np.ndarray) -> 'GARCHAnalyzer':
        """
        Fit GARCH(1,1) to return series using maximum likelihood.
        
        Args:
            returns: Return series
            
        Returns:
            self (fitted model)
        """
        returns = np.asarray(returns).flatten()
        returns = returns[np.isfinite(returns)]
        
        if len(returns) < 100:
            raise ValueError("Need at least 100 observations for GARCH")
        
        self._n_obs = len(returns)
        
        # Initial parameter guesses
        sample_var = np.var(returns)
        mu_init = np.mean(returns)
        omega_init = sample_var * 0.05  # Small baseline
        alpha_init = 0.10
        beta_init = 0.85
        
        initial_params = np.array([omega_init, alpha_init, beta_init, mu_init])
        
        # Bounds: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1
        bounds = [
            (1e-10, sample_var),      # omega
            (0.001, 0.5),             # alpha
            (0.001, 0.999),           # beta
            (-0.01, 0.01)             # mu
        ]
        
        # Optimize
        result = minimize(
            self._garch_likelihood,
            initial_params,
            args=(returns,),
            method='L-BFGS-B',
            bounds=bounds
        )
        
        self.omega, self.alpha, self.beta, self.mu = result.x
        self._log_likelihood = -result.fun
        self._converged = result.success
        self._fitted = True
        
        return self
    
    def get_conditional_variance(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute conditional variance series.
        
        Args:
            returns: Return series
            
        Returns:
            Conditional variance series (same length as returns)
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")
            
        returns = np.asarray(returns).flatten()
        n = len(returns)
        eps = returns - self.mu
        
        sigma2 = np.zeros(n)
        sigma2[0] = self.omega / (1 - self.alpha - self.beta)  # Unconditional
        
        for t in range(1, n):
            sigma2[t] = self.omega + self.alpha * eps[t-1]**2 + self.beta * sigma2[t-1]
            sigma2[t] = max(sigma2[t], 1e-10)
            
        return sigma2
    
    def forecast(
        self,
        returns: np.ndarray,
        horizon: int = Config.GARCH_FORECAST_HORIZON
    ) -> np.ndarray:
        """
        Forecast conditional variance for multiple periods ahead.
        
        For GARCH(1,1), the h-step ahead forecast is:
            σ²_{t+h|t} = ω * Σ_{i=0}^{h-1} (α+β)^i + (α+β)^h * σ²_t
        
        Args:
            returns: Historical returns
            horizon: Forecast horizon
            
        Returns:
            Forecasted variance path (length = horizon)
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")
        
        # Get current variance
        sigma2 = self.get_conditional_variance(returns)
        current_var = sigma2[-1]
        
        persistence = self.alpha + self.beta
        long_run_var = self.omega / (1 - persistence) if persistence < 1 else current_var
        
        # Forecast
        forecasts = np.zeros(horizon)
        for h in range(horizon):
            forecasts[h] = long_run_var + (persistence ** (h + 1)) * (current_var - long_run_var)
            
        return forecasts
    
    @property
    def persistence(self) -> float:
        """GARCH persistence (α + β)."""
        if not self._fitted:
            return 0.0
        return self.alpha + self.beta
    
    @property
    def long_run_variance(self) -> float:
        """Unconditional (long-run) variance."""
        if not self._fitted or self.persistence >= 1:
            return 0.0
        return self.omega / (1 - self.persistence)
    
    @property
    def half_life(self) -> float:
        """Half-life of volatility shocks (in periods)."""
        if not self._fitted or self.persistence <= 0 or self.persistence >= 1:
            return np.inf
        return np.log(0.5) / np.log(self.persistence)
    
    def compute_aic_bic(self) -> Tuple[float, float]:
        """Compute AIC and BIC."""
        k = 4  # omega, alpha, beta, mu
        aic = -2 * self._log_likelihood + 2 * k
        bic = -2 * self._log_likelihood + k * np.log(self._n_obs)
        return aic, bic


# =============================================================================
# SECTION 7: HURST EXPONENT CALCULATOR
# =============================================================================

class HurstCalculator:
    """
    Hurst exponent calculation using Rescaled Range (R/S) analysis.
    
    The Hurst exponent H measures the long-range dependence of a time series:
    
        H > 0.5: Persistent/trending (past trend likely to continue)
        H = 0.5: Random walk (no memory)
        H < 0.5: Anti-persistent/mean-reverting (reversals likely)
    
    Method: Rescaled Range (R/S) Analysis
    ------------------------------------
    For a window of size n:
        1. Compute mean: m = (1/n) Σ x_i
        2. Compute cumulative deviations: Y_i = Σ_{j=1}^{i} (x_j - m)
        3. Compute range: R = max(Y) - min(Y)
        4. Compute standard deviation: S = sqrt((1/n) Σ (x_i - m)²)
        5. Compute rescaled range: R/S
    
    The Hurst exponent is estimated from:
        E[R/S] ~ c * n^H
        
    Taking logs: log(R/S) = log(c) + H * log(n)
    
    H is the slope of the regression of log(R/S) on log(n).
    
    Reference:
        Hurst, H.E. (1951). "Long-term storage capacity of reservoirs."
    """
    
    def __init__(
        self,
        min_window: int = Config.HURST_MIN_WINDOW,
        max_window: int = Config.HURST_MAX_WINDOW,
        n_windows: int = Config.HURST_N_WINDOWS
    ):
        """
        Initialize Hurst calculator.
        
        Args:
            min_window: Minimum window size for R/S calculation
            max_window: Maximum window size
            n_windows: Number of window sizes to use
        """
        self.min_window = min_window
        self.max_window = max_window
        self.n_windows = n_windows
        
    def _compute_rs(self, X: np.ndarray, window: int) -> float:
        """
        Compute R/S statistic for a given window size.
        
        Divides series into non-overlapping windows and averages R/S.
        """
        n = len(X)
        n_windows = n // window
        
        if n_windows < 1:
            return np.nan
            
        rs_values = []
        
        for i in range(n_windows):
            window_data = X[i * window:(i + 1) * window]
            
            # Mean-adjusted cumulative sum
            mean = np.mean(window_data)
            cumsum = np.cumsum(window_data - mean)
            
            # Range
            R = np.max(cumsum) - np.min(cumsum)
            
            # Standard deviation
            S = np.std(window_data, ddof=1)
            
            if S > 0:
                rs_values.append(R / S)
                
        return np.mean(rs_values) if rs_values else np.nan
    
    def calculate(self, returns: np.ndarray) -> HurstAnalysis:
        """
        Calculate Hurst exponent via R/S analysis.
        
        Args:
            returns: Return series
            
        Returns:
            HurstAnalysis with results
        """
        returns = np.asarray(returns).flatten()
        returns = returns[np.isfinite(returns)]
        
        n = len(returns)
        
        # Adjust max_window if necessary
        max_win = min(self.max_window, n // 4)
        min_win = max(self.min_window, 10)
        
        if max_win < min_win:
            # Not enough data - return default
            return HurstAnalysis(
                hurst_exponent=0.5,
                persistence=TrendPersistence.RANDOM_WALK,
                standard_error=0.1,
                confidence_interval=(0.4, 0.6),
                r_squared=0.0,
                interpretation="Insufficient data for Hurst calculation",
                log_n=np.array([]),
                log_rs=np.array([])
            )
        
        # Generate window sizes (log-spaced)
        windows = np.unique(np.logspace(
            np.log10(min_win),
            np.log10(max_win),
            self.n_windows
        ).astype(int))
        
        # Compute R/S for each window
        log_n = []
        log_rs = []
        
        for w in windows:
            rs = self._compute_rs(returns, w)
            if np.isfinite(rs) and rs > 0:
                log_n.append(np.log(w))
                log_rs.append(np.log(rs))
        
        log_n = np.array(log_n)
        log_rs = np.array(log_rs)
        
        if len(log_n) < 3:
            # Not enough valid points
            return HurstAnalysis(
                hurst_exponent=0.5,
                persistence=TrendPersistence.RANDOM_WALK,
                standard_error=0.1,
                confidence_interval=(0.4, 0.6),
                r_squared=0.0,
                interpretation="Insufficient valid R/S values",
                log_n=log_n,
                log_rs=log_rs
            )
        
        # Linear regression: log(R/S) = c + H * log(n)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_rs)
        
        # Hurst exponent is the slope
        H = np.clip(slope, 0.0, 1.0)
        
        # Confidence interval
        ci_low = H - 1.96 * std_err
        ci_high = H + 1.96 * std_err
        
        # Classify persistence
        persistence = classify_hurst(H)
        
        # Generate interpretation
        if H > 0.55:
            interpretation = f"Trending market (H={H:.3f}). Past price direction likely to persist."
        elif H < 0.45:
            interpretation = f"Mean-reverting market (H={H:.3f}). Price reversals more likely."
        else:
            interpretation = f"Random walk behavior (H={H:.3f}). No predictable pattern from past movements."
        
        return HurstAnalysis(
            hurst_exponent=H,
            persistence=persistence,
            standard_error=std_err,
            confidence_interval=(ci_low, ci_high),
            r_squared=r_value ** 2,
            interpretation=interpretation,
            log_n=log_n,
            log_rs=log_rs
        )


# =============================================================================
# SECTION 8: STRUCTURAL BREAKPOINT DETECTOR
# =============================================================================

class BreakpointDetector:
    """
    Structural breakpoint detection using CUSUM test.
    
    Detects significant changes in the mean or variance of a time series
    that may indicate regime transitions.
    
    Method: CUSUM (Cumulative Sum) Test
    -----------------------------------
    The CUSUM statistic tracks cumulative deviations from the mean:
        S_t = Σ_{i=1}^{t} (x_i - x̄)
    
    A structural break is indicated when |S_t| exceeds a critical value.
    
    The test is particularly useful for:
        1. Detecting mean shifts (level changes)
        2. Identifying regime change points
        3. Validating regime model assumptions
    
    Reference:
        Brown, R.L., Durbin, J., Evans, J.M. (1975). "Techniques for Testing
        the Constancy of Regression Relationships over Time."
    """
    
    def __init__(
        self,
        min_segment: int = 50,
        significance: float = 0.05
    ):
        """
        Initialize breakpoint detector.
        
        Args:
            min_segment: Minimum observations between breakpoints
            significance: Significance level for break detection
        """
        self.min_segment = min_segment
        self.significance = significance
    
    def _cusum_test(self, X: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Perform CUSUM test on series.
        
        Returns:
            (cusum_stat, critical_value, break_detected)
        """
        n = len(X)
        mean = np.mean(X)
        std = np.std(X) if np.std(X) > 0 else 1.0
        
        # Standardized CUSUM
        cumsum = np.cumsum(X - mean) / (std * np.sqrt(n))
        
        # Maximum deviation
        max_stat = np.max(np.abs(cumsum))
        
        # Critical value (approximate for 5% level)
        # Based on Brownian bridge asymptotic distribution
        critical = 1.36 if self.significance == 0.05 else 1.63
        
        break_detected = max_stat > critical
        
        return cumsum, max_stat, break_detected
    
    def _find_breakpoint(self, X: np.ndarray) -> Optional[int]:
        """
        Find location of most significant breakpoint.
        
        Returns index of breakpoint or None if no significant break.
        """
        cumsum, stat, detected = self._cusum_test(X)
        
        if not detected:
            return None
            
        # Breakpoint is at maximum CUSUM deviation
        break_idx = np.argmax(np.abs(cumsum))
        
        # Ensure minimum segment size
        if break_idx < self.min_segment or len(X) - break_idx < self.min_segment:
            return None
            
        return break_idx
    
    def detect(
        self,
        returns: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> BreakpointAnalysis:
        """
        Detect structural breakpoints in return series.
        
        Args:
            returns: Return series
            dates: Corresponding dates (optional)
            
        Returns:
            BreakpointAnalysis with results
        """
        returns = np.asarray(returns).flatten()
        returns = returns[np.isfinite(returns)]
        n = len(returns)
        
        if dates is None:
            # Create dummy dates
            dates = pd.date_range(end=datetime.now(), periods=n, freq='D')
        
        # Find breakpoints iteratively
        breakpoints = []
        remaining = returns.copy()
        offset = 0
        
        while len(remaining) > 2 * self.min_segment:
            bp = self._find_breakpoint(remaining)
            if bp is None:
                break
            
            # Convert to original index
            original_idx = offset + bp
            breakpoints.append(original_idx)
            
            # Continue with larger segment
            if bp > len(remaining) - bp:
                remaining = remaining[:bp]
            else:
                remaining = remaining[bp:]
                offset = original_idx
        
        # Convert to dates
        break_dates = [dates[i] for i in breakpoints if i < len(dates)]
        
        # Current segment analysis
        if breakpoints:
            last_break = max(breakpoints)
            segment = returns[last_break:]
            last_break_date = dates[last_break] if last_break < len(dates) else None
            days_since = n - last_break
        else:
            segment = returns
            last_break_date = None
            days_since = n
        
        # CUSUM on full series
        cumsum, stat, detected = self._cusum_test(returns)
        
        return BreakpointAnalysis(
            breakpoints=break_dates,
            n_breaks=len(breakpoints),
            last_break_date=last_break_date,
            days_since_break=days_since,
            segment_mean=np.mean(segment) if len(segment) > 0 else 0,
            segment_volatility=np.std(segment) if len(segment) > 0 else 0,
            segment_length=len(segment),
            cusum_stat=stat,
            cusum_critical=1.36,  # 5% level
            break_detected=detected
        )


# =============================================================================
# SECTION 9: STRATEGY RECOMMENDER
# =============================================================================

class StrategyRecommender:
    """
    Generate trading strategy recommendations based on regime analysis.
    
    Combines regime detection outputs to provide actionable guidance:
        - Strategy type (trend-following, mean-reversion, defensive)
        - Position bias (long, short, neutral)
        - Position sizing
        - Risk management parameters
    
    The recommendations are regime-adaptive:
        - BULL + Trending: Momentum/trend-following with full position
        - BEAR + High Vol: Defensive with reduced position
        - SIDEWAYS + Mean-reverting: Range trading with half position
    """
    
    def recommend(
        self,
        market_regime: MarketRegime,
        vol_regime: VolatilityRegime,
        hurst: HurstAnalysis,
        confidence: float
    ) -> StrategyRecommendation:
        """
        Generate strategy recommendation.
        
        Args:
            market_regime: Current market regime
            vol_regime: Current volatility regime
            hurst: Hurst analysis results
            confidence: Overall confidence in regime detection
            
        Returns:
            StrategyRecommendation
        """
        rationale = []
        warnings = []
        
        # Determine strategy type based on regime and Hurst
        if market_regime == MarketRegime.BULL:
            if hurst.persistence in [TrendPersistence.STRONG_TRENDING, TrendPersistence.WEAK_TRENDING]:
                strategy = StrategyType.TREND_FOLLOWING
                position_bias = PositionBias.LONG
                rationale.append("Bull market with trending behavior favors trend-following")
            else:
                strategy = StrategyType.MOMENTUM
                position_bias = PositionBias.LONG
                rationale.append("Bull market suggests momentum strategies")
                
        elif market_regime == MarketRegime.BEAR:
            if vol_regime in [VolatilityRegime.HIGH, VolatilityRegime.VERY_HIGH, VolatilityRegime.CRISIS]:
                strategy = StrategyType.DEFENSIVE
                position_bias = PositionBias.NEUTRAL
                rationale.append("Bear market with high volatility - defensive stance")
                warnings.append("High volatility environment - reduce exposure")
            else:
                strategy = StrategyType.DEFENSIVE
                position_bias = PositionBias.SHORT
                rationale.append("Bear market with manageable volatility")
                
        else:  # SIDEWAYS
            if hurst.persistence in [TrendPersistence.WEAK_MEAN_REVERTING, TrendPersistence.STRONG_MEAN_REVERTING]:
                strategy = StrategyType.MEAN_REVERSION
                position_bias = PositionBias.NEUTRAL
                rationale.append("Sideways market with mean-reverting behavior")
            else:
                strategy = StrategyType.NEUTRAL
                position_bias = PositionBias.NEUTRAL
                rationale.append("Sideways market - wait for clearer regime")
                warnings.append("Unclear market direction")
        
        # Position sizing based on regime and volatility
        base_size = {
            MarketRegime.BULL: Config.TREND_POSITION_SIZE,
            MarketRegime.BEAR: Config.BEAR_POSITION_SIZE,
            MarketRegime.SIDEWAYS: Config.SIDEWAYS_POSITION_SIZE
        }[market_regime]
        
        # Adjust for volatility
        vol_adjustment = {
            VolatilityRegime.VERY_LOW: 1.2,
            VolatilityRegime.LOW: 1.1,
            VolatilityRegime.NORMAL: 1.0,
            VolatilityRegime.HIGH: 0.7,
            VolatilityRegime.VERY_HIGH: 0.5,
            VolatilityRegime.CRISIS: 0.25
        }[vol_regime]
        
        position_size = min(1.0, base_size * vol_adjustment * confidence)
        
        # Risk management
        stop_atr = {
            MarketRegime.BULL: Config.TREND_STOP_ATR,
            MarketRegime.BEAR: Config.BEAR_STOP_ATR,
            MarketRegime.SIDEWAYS: Config.SIDEWAYS_STOP_ATR
        }[market_regime]
        
        # Holding period based on regime
        holding_days = {
            MarketRegime.BULL: 20,
            MarketRegime.BEAR: 5,
            MarketRegime.SIDEWAYS: 10
        }[market_regime]
        
        # Add volatility context to rationale
        rationale.append(f"Volatility regime: {vol_regime.value}")
        rationale.append(f"Hurst exponent: {hurst.hurst_exponent:.3f} ({hurst.persistence.value})")
        
        # Add warnings for extreme conditions
        if vol_regime == VolatilityRegime.CRISIS:
            warnings.append("CRISIS volatility - significant capital at risk")
        if confidence < 0.6:
            warnings.append("Low confidence in regime detection")
        
        return StrategyRecommendation(
            strategy=strategy,
            position_bias=position_bias,
            position_size=position_size,
            leverage_factor=1.0,  # No leverage by default
            stop_loss_atr=stop_atr,
            take_profit_atr=stop_atr * 2,  # 2:1 reward/risk
            max_holding_days=holding_days,
            confidence=confidence,
            rationale=rationale,
            warnings=warnings
        )


# =============================================================================
# SECTION 10: REGIME DETECTION PIPELINE
# =============================================================================

class RegimeDetectionPipeline:
    """
    Complete regime detection pipeline combining all analyzers.
    
    This is the main entry point for Phase 3 regime detection.
    It coordinates:
        1. HMM regime classification
        2. GARCH volatility analysis
        3. Hurst exponent calculation
        4. Breakpoint detection
        5. Strategy recommendation
    
    The output RegimeReport integrates seamlessly with:
        - Phase 4: Backtesting (position sizing, risk management)
        - Phase 5: LLM trade note generation
    
    Usage:
        pipeline = RegimeDetectionPipeline()
        report = pipeline.analyze(df, symbol='AAPL')
    """
    
    def __init__(self):
        """Initialize pipeline components."""
        self.hmm = GaussianHMM()
        self.garch = GARCHAnalyzer()
        self.hurst = HurstCalculator()
        self.breakpoints = BreakpointDetector()
        self.recommender = StrategyRecommender()
        
    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN"
    ) -> RegimeReport:
        """
        Run complete regime detection analysis.
        
        Args:
            df: DataFrame with OHLCV data (must have 'Close' column)
            symbol: Security symbol for reporting
            
        Returns:
            RegimeReport with all analysis results
        """
        import time
        start_time = time.time()
        
        # Validate input
        if 'Close' not in df.columns and 'close' not in df.columns:
            # Try to find close column
            close_cols = [c for c in df.columns if 'close' in c.lower()]
            if close_cols:
                df = df.rename(columns={close_cols[0]: 'Close'})
            else:
                raise ValueError("DataFrame must have 'Close' column")
        
        if 'close' in df.columns:
            df = df.rename(columns={'close': 'Close'})
            
        # Get price series
        prices = df['Close'].values
        returns = compute_log_returns(prices)
        
        # Get dates
        if isinstance(df.index, pd.DatetimeIndex):
            dates = df.index
        else:
            dates = pd.to_datetime(df.index) if df.index.dtype == 'datetime64[ns]' else None
        
        n_obs = len(returns)
        
        # =========================================================================
        # 1. HMM REGIME DETECTION
        # =========================================================================
        try:
            self.hmm.fit(returns)
            
            current_regime, regime_prob = self.hmm.get_current_regime(returns)
            state_probs = self.hmm.get_state_probabilities(returns)
            regime_sequence = self.hmm.predict(returns)
            
            # Map states to regimes in sequence
            regime_sequence_mapped = np.array([
                self.hmm._state_to_regime[s].value for s in regime_sequence
            ])
            
            # Build state parameters
            state_params = {}
            stationary = self.hmm.get_stationary_distribution()
            
            for k in range(self.hmm.n_states):
                regime = self.hmm._state_to_regime[k]
                state_params[regime] = HMMStateParameters(
                    state_id=k,
                    regime=regime,
                    mean_return=float(self.hmm.means[k]),
                    volatility=float(np.sqrt(self.hmm.variances[k])),
                    annualized_return=annualize_return(self.hmm.means[k]),
                    annualized_volatility=annualize_volatility(np.sqrt(self.hmm.variances[k])),
                    stationary_probability=stationary.get(regime, 0.33)
                )
            
            aic, bic = self.hmm.compute_aic_bic(returns)
            
            hmm_analysis = HMMAnalysis(
                current_regime=current_regime,
                regime_probability=regime_prob,
                state_probabilities=state_probs,
                state_parameters=state_params,
                regime_sequence=regime_sequence_mapped,
                transition_matrix=self.hmm.A,
                log_likelihood=self.hmm._log_likelihood,
                aic=aic,
                bic=bic,
                converged=self.hmm._converged,
                expected_durations=self.hmm.get_expected_durations()
            )
            
        except Exception as e:
            logger.warning(f"HMM fitting failed: {e}")
            # Fallback to simple regime detection
            recent_return = np.mean(returns[-21:]) if len(returns) >= 21 else np.mean(returns)
            if recent_return > Config.BULL_RETURN_THRESHOLD:
                current_regime = MarketRegime.BULL
            elif recent_return < Config.BEAR_RETURN_THRESHOLD:
                current_regime = MarketRegime.BEAR
            else:
                current_regime = MarketRegime.SIDEWAYS
            
            hmm_analysis = HMMAnalysis(
                current_regime=current_regime,
                regime_probability=0.5,
                state_probabilities={r: 0.33 for r in MarketRegime},
                state_parameters={},
                regime_sequence=np.array([]),
                transition_matrix=np.eye(3) * 0.9 + 0.033,
                log_likelihood=0,
                aic=0,
                bic=0,
                converged=False,
                expected_durations={r: 30.0 for r in MarketRegime}
            )
        
        # =========================================================================
        # 2. GARCH VOLATILITY ANALYSIS
        # =========================================================================
        try:
            self.garch.fit(returns)
            
            sigma2 = self.garch.get_conditional_variance(returns)
            current_vol_daily = np.sqrt(sigma2[-1])
            current_vol_annual = annualize_volatility(current_vol_daily)
            
            forecasts = self.garch.forecast(returns, horizon=5)
            
            # Volatility percentile
            historical_vol = np.sqrt(sigma2)
            vol_percentile = stats.percentileofscore(historical_vol, current_vol_daily)
            
            aic, bic = self.garch.compute_aic_bic()
            
            garch_analysis = GARCHAnalysis(
                current_volatility=current_vol_annual,
                daily_volatility=current_vol_daily,
                vol_regime=classify_volatility_regime(current_vol_annual),
                forecast_1d=np.sqrt(forecasts[0]) * np.sqrt(252),
                forecast_5d=np.sqrt(np.mean(forecasts)) * np.sqrt(252),
                forecast_horizon=np.sqrt(forecasts) * np.sqrt(252),
                omega=self.garch.omega,
                alpha=self.garch.alpha,
                beta=self.garch.beta,
                persistence=self.garch.persistence,
                log_likelihood=self.garch._log_likelihood,
                aic=aic,
                converged=self.garch._converged,
                long_run_volatility=annualize_volatility(np.sqrt(self.garch.long_run_variance)),
                vol_percentile=vol_percentile
            )
            
        except Exception as e:
            logger.warning(f"GARCH fitting failed: {e}")
            # Fallback to simple volatility
            simple_vol = np.std(returns[-63:]) if len(returns) >= 63 else np.std(returns)
            annual_vol = annualize_volatility(simple_vol)
            
            garch_analysis = GARCHAnalysis(
                current_volatility=annual_vol,
                daily_volatility=simple_vol,
                vol_regime=classify_volatility_regime(annual_vol),
                forecast_1d=annual_vol,
                forecast_5d=annual_vol,
                forecast_horizon=np.array([simple_vol] * 5) * np.sqrt(252),
                omega=0.0,
                alpha=0.1,
                beta=0.85,
                persistence=0.95,
                log_likelihood=0,
                aic=0,
                converged=False,
                long_run_volatility=annual_vol,
                vol_percentile=50.0
            )
        
        # =========================================================================
        # 3. HURST EXPONENT ANALYSIS
        # =========================================================================
        hurst_analysis = self.hurst.calculate(returns)
        
        # =========================================================================
        # 4. BREAKPOINT DETECTION
        # =========================================================================
        break_analysis = self.breakpoints.detect(
            returns,
            dates[1:] if dates is not None and len(dates) > 1 else None
        )
        
        # =========================================================================
        # 5. ENSEMBLE CONFIDENCE CALCULATION
        # =========================================================================
        # Combine signals for consensus confidence
        confidence_components = {
            'hmm_probability': hmm_analysis.regime_probability,
            'vol_agreement': 1.0 if garch_analysis.vol_regime in [
                VolatilityRegime.NORMAL, VolatilityRegime.LOW
            ] else 0.7,
            'hurst_clarity': abs(hurst_analysis.hurst_exponent - 0.5) * 2,  # 0-1 scale
            'trend_strength': hurst_analysis.r_squared
        }
        
        consensus_confidence = sum(
            Config.CONFIDENCE_WEIGHTS[k] * v
            for k, v in confidence_components.items()
        )
        consensus_confidence = np.clip(consensus_confidence, Config.MIN_CONFIDENCE, Config.MAX_CONFIDENCE)
        
        # =========================================================================
        # 6. STRATEGY RECOMMENDATION
        # =========================================================================
        strategy = self.recommender.recommend(
            market_regime=hmm_analysis.current_regime,
            vol_regime=garch_analysis.vol_regime,
            hurst=hurst_analysis,
            confidence=consensus_confidence
        )
        
        # =========================================================================
        # 7. QUALITY ASSESSMENT
        # =========================================================================
        quality_components = [
            hmm_analysis.converged * 30,  # HMM convergence
            garch_analysis.converged * 25,  # GARCH convergence
            hurst_analysis.r_squared * 20,  # Hurst fit quality
            min(n_obs / 1000, 1.0) * 15,  # Data sufficiency
            (1 - break_analysis.break_detected * 0.5) * 10  # Stability
        ]
        quality_score = sum(quality_components)
        quality_grade = score_to_grade(quality_score)
        
        # =========================================================================
        # BUILD FINAL REPORT
        # =========================================================================
        processing_time = int((time.time() - start_time) * 1000)
        
        return RegimeReport(
            symbol=symbol,
            analysis_date=datetime.now(),
            period_start=dates[0] if dates is not None else datetime.now(),
            period_end=dates[-1] if dates is not None else datetime.now(),
            n_observations=n_obs,
            hmm=hmm_analysis,
            garch=garch_analysis,
            hurst=hurst_analysis,
            breakpoints=break_analysis,
            consensus_regime=hmm_analysis.current_regime,
            consensus_confidence=consensus_confidence,
            strategy=strategy,
            quality_score=quality_score,
            quality_grade=quality_grade,
            processing_time_ms=processing_time,
            version=VERSION
        )


# =============================================================================
# SECTION 11: CONVENIENCE FUNCTIONS
# =============================================================================

def detect_regime(
    df: pd.DataFrame,
    symbol: str = "UNKNOWN"
) -> RegimeReport:
    """
    Convenience function for regime detection.
    
    Creates pipeline and runs analysis in one call.
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Security symbol
        
    Returns:
        RegimeReport
    
    Example:
        >>> import pandas as pd
        >>> df = pd.read_parquet('data/aapl_daily.parquet')
        >>> report = detect_regime(df, symbol='AAPL')
        >>> print(f"Regime: {report.hmm.current_regime.value}")
        >>> print(f"Strategy: {report.strategy.strategy.value}")
    """
    pipeline = RegimeDetectionPipeline()
    return pipeline.analyze(df, symbol)


def format_regime_report(report: RegimeReport) -> str:
    """
    Format regime report as human-readable text.
    
    Args:
        report: RegimeReport from pipeline
        
    Returns:
        Formatted string
    """
    lines = [
        "=" * 70,
        "MARKET REGIME DETECTION REPORT",
        "=" * 70,
        f"Symbol: {report.symbol}",
        f"Analysis Date: {report.analysis_date.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Period: {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}",
        f"Observations: {report.n_observations:,}",
        "",
        "-" * 70,
        "REGIME CLASSIFICATION",
        "-" * 70,
        f"Current Regime: {report.hmm.current_regime.value}",
        f"Probability: {report.hmm.regime_probability:.1%}",
        f"Consensus Confidence: {report.consensus_confidence:.1%}",
        "",
        "State Probabilities:",
    ]
    
    for regime, prob in report.hmm.state_probabilities.items():
        lines.append(f"  {regime.value}: {prob:.1%}")
    
    lines.extend([
        "",
        "Expected Durations (days):",
    ])
    
    for regime, days in report.hmm.expected_durations.items():
        lines.append(f"  {regime.value}: {days:.0f}")
    
    lines.extend([
        "",
        "-" * 70,
        "VOLATILITY ANALYSIS",
        "-" * 70,
        f"Current Volatility: {report.garch.current_volatility:.1%} (annualized)",
        f"Volatility Regime: {report.garch.vol_regime.value}",
        f"Percentile: {report.garch.vol_percentile:.0f}th",
        "",
        f"GARCH Parameters:",
        f"  Omega: {report.garch.omega:.6f}",
        f"  Alpha: {report.garch.alpha:.4f}",
        f"  Beta: {report.garch.beta:.4f}",
        f"  Persistence: {report.garch.persistence:.4f}",
        "",
        f"Forecasts:",
        f"  1-day: {report.garch.forecast_1d:.1%}",
        f"  5-day avg: {report.garch.forecast_5d:.1%}",
        "",
        "-" * 70,
        "TREND PERSISTENCE (HURST EXPONENT)",
        "-" * 70,
        f"Hurst Exponent: {report.hurst.hurst_exponent:.3f}",
        f"Classification: {report.hurst.persistence.value}",
        f"R-squared: {report.hurst.r_squared:.3f}",
        f"95% CI: [{report.hurst.confidence_interval[0]:.3f}, {report.hurst.confidence_interval[1]:.3f}]",
        f"Interpretation: {report.hurst.interpretation}",
        "",
        "-" * 70,
        "STRUCTURAL BREAKS",
        "-" * 70,
        f"Breaks Detected: {report.breakpoints.n_breaks}",
        f"Days Since Last Break: {report.breakpoints.days_since_break}",
        f"CUSUM Statistic: {report.breakpoints.cusum_stat:.3f}",
        f"Break Active: {'Yes' if report.breakpoints.break_detected else 'No'}",
        "",
        "-" * 70,
        "STRATEGY RECOMMENDATION",
        "-" * 70,
        f"Strategy: {report.strategy.strategy.value}",
        f"Position Bias: {report.strategy.position_bias.value}",
        f"Position Size: {report.strategy.position_size:.0%}",
        f"Stop Loss: {report.strategy.stop_loss_atr:.1f}x ATR",
        f"Take Profit: {report.strategy.take_profit_atr:.1f}x ATR",
        f"Max Holding: {report.strategy.max_holding_days} days",
        f"Confidence: {report.strategy.confidence:.1%}",
        "",
        "Rationale:",
    ])
    
    for r in report.strategy.rationale:
        lines.append(f"  • {r}")
    
    if report.strategy.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in report.strategy.warnings:
            lines.append(f"  ⚠ {w}")
    
    lines.extend([
        "",
        "-" * 70,
        "QUALITY ASSESSMENT",
        "-" * 70,
        f"Quality Score: {report.quality_score:.1f}/100",
        f"Quality Grade: {report.quality_grade.value}",
        "",
        "=" * 70,
        f"Processing Time: {report.processing_time_ms}ms | Version: {report.version}",
        "=" * 70,
    ])
    
    return "\n".join(lines)


# =============================================================================
# SECTION 12: MODULE EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    'Config',
    'VERSION',
    
    # Enumerations
    'MarketRegime',
    'VolatilityRegime',
    'TrendPersistence',
    'StrategyType',
    'PositionBias',
    'QualityGrade',
    
    # Data structures
    'HMMStateParameters',
    'HMMAnalysis',
    'GARCHAnalysis',
    'HurstAnalysis',
    'BreakpointAnalysis',
    'StrategyRecommendation',
    'RegimeReport',
    
    # Analyzers
    'GaussianHMM',
    'GARCHAnalyzer',
    'HurstCalculator',
    'BreakpointDetector',
    'StrategyRecommender',
    
    # Pipeline
    'RegimeDetectionPipeline',
    
    # Convenience functions
    'detect_regime',
    'format_regime_report',
    
    # Utilities
    'compute_log_returns',
    'annualize_return',
    'annualize_volatility',
    'classify_volatility_regime',
    'classify_hurst',
    'score_to_grade',
]