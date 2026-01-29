#!/usr/bin/env python3
"""
Phase 4: Institutional-Grade Backtesting Engine
================================================

MSc AI Agents in Asset Management (IFTE0001) - Track B: Technical Analyst Agent

This module implements a comprehensive backtesting framework that satisfies
the coursework requirement:
    "Backtest with transaction costs and position sizing"
    "LLM-generated 1–2 page trade note with metrics (CAGR, Sharpe, drawdown, hit rate)"

COURSEWORK METRICS DELIVERED
----------------------------
All metrics required for the LLM trade note (Phase 5):
    1. CAGR (Compound Annual Growth Rate)
    2. Sharpe Ratio (risk-adjusted return)
    3. Maximum Drawdown (peak-to-trough decline)
    4. Hit Rate / Win Rate (percentage of winning trades)
    + Additional professional metrics: Sortino, Calmar, Profit Factor, etc.

ACADEMIC FOUNDATIONS
--------------------
Transaction Costs:
    Kissell, R. (2013). "The Science of Algorithmic Trading and Portfolio Management."
    
    Real-world costs include:
        - Commission: Fixed percentage per trade
        - Slippage: Market impact from order execution
        - Spread: Bid-ask spread cost

Position Sizing:
    Kelly, J.L. (1956). "A New Interpretation of Information Rate."
    Bell System Technical Journal.
    
    Optimal fraction: f* = (bp - q) / b
    where b = odds, p = win probability, q = 1 - p

Risk Metrics:
    Sharpe, W.F. (1994). "The Sharpe Ratio." Journal of Portfolio Management.
    Sortino, F.A. & van der Meer, R. (1991). "Downside Risk."
    Young, T.W. (1991). "Calmar Ratio: A Smoother Tool."

Walk-Forward Analysis:
    Pardo, R. (2008). "The Evaluation and Optimization of Trading Strategies."
    
    Tests strategy robustness by training on historical data and
    validating on out-of-sample periods.

ARCHITECTURE
------------
    Layer 1: Core Components
        - TransactionCostModel: Realistic cost simulation
        - PositionSizer: Kelly, fixed fractional, volatility targeting
        - TradeManager: Entry/exit tracking and logging

    Layer 2: Metrics Calculators
        - ReturnCalculator: CAGR, total return, annualization
        - RiskCalculator: Volatility, VaR, CVaR, drawdown
        - RiskAdjustedCalculator: Sharpe, Sortino, Calmar, Omega
        - TradeAnalyzer: Hit rate, profit factor, expectancy

    Layer 3: Validation
        - WalkForwardValidator: Out-of-sample testing
        - MonteCarloSimulator: Bootstrap confidence intervals

    Layer 4: Output
        - BacktestResult: Complete results container
        - Integration with Phase 5 LLM trade note generation

Author: Tamer
Version: 2.0.0
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

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
    Centralized configuration for backtesting parameters.
    
    All values are based on institutional trading practices and
    academic research on realistic cost modeling.
    """
    
    # -------------------------------------------------------------------------
    # Trading Calendar
    # -------------------------------------------------------------------------
    TRADING_DAYS_YEAR: int = 252      # Standard US equity trading days
    TRADING_DAYS_MONTH: int = 21
    
    # -------------------------------------------------------------------------
    # Transaction Costs (Institutional Retail)
    # -------------------------------------------------------------------------
    COMMISSION_RATE: float = 0.001    # 10 basis points per trade
    SLIPPAGE_RATE: float = 0.0005     # 5 basis points slippage
    SPREAD_RATE: float = 0.0001       # 1 basis point spread
    MIN_COMMISSION: float = 1.0       # Minimum $1 per trade
    
    # -------------------------------------------------------------------------
    # Position Sizing
    # -------------------------------------------------------------------------
    KELLY_FRACTION: float = 0.25      # Quarter-Kelly (conservative)
    MAX_POSITION_SIZE: float = 1.0    # Maximum 100% of capital
    MIN_POSITION_SIZE: float = 0.0    # Minimum 0% (no shorting default)
    TARGET_VOLATILITY: float = 0.15   # 15% target annual volatility
    DEFAULT_POSITION_SIZE: float = 1.0  # Default 100% if no sizing
    
    # -------------------------------------------------------------------------
    # Risk Parameters
    # -------------------------------------------------------------------------
    RISK_FREE_RATE: float = 0.05      # 5% annual risk-free rate
    VAR_CONFIDENCE: float = 0.95      # 95% VaR
    CVAR_CONFIDENCE: float = 0.95     # 95% CVaR
    
    # -------------------------------------------------------------------------
    # Walk-Forward Analysis
    # -------------------------------------------------------------------------
    WF_TRAIN_RATIO: float = 0.75      # 75% training, 25% testing
    WF_MIN_TRAIN_DAYS: int = 252      # Minimum 1 year training
    WF_MIN_TEST_DAYS: int = 63        # Minimum 3 months testing
    WF_N_SPLITS: int = 5              # Number of walk-forward periods
    
    # -------------------------------------------------------------------------
    # Monte Carlo Simulation
    # -------------------------------------------------------------------------
    MC_N_SIMULATIONS: int = 1000      # Number of bootstrap simulations
    MC_BLOCK_SIZE: int = 21           # Block size for stationary bootstrap
    MC_CONFIDENCE_LEVELS: List[float] = [0.05, 0.25, 0.50, 0.75, 0.95]
    
    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    MIN_TRADES_FOR_VALIDITY: int = 30  # Minimum trades for statistical validity


# =============================================================================
# SECTION 2: ENUMERATIONS
# =============================================================================

class SignalType(Enum):
    """Trading signal types from Phase 2 indicators."""
    LONG = 1       # Buy signal
    SHORT = -1     # Sell/short signal
    FLAT = 0       # No position


class PositionSizingMethod(Enum):
    """
    Position sizing methodologies.
    
    FIXED_FRACTIONAL: Fixed percentage of capital per trade
    KELLY_CRITERION: Optimal growth rate sizing
    VOLATILITY_TARGET: Size based on target portfolio volatility
    EQUAL_WEIGHT: Equal position sizes
    """
    FIXED_FRACTIONAL = "FIXED_FRACTIONAL"
    KELLY_CRITERION = "KELLY_CRITERION"
    VOLATILITY_TARGET = "VOLATILITY_TARGET"
    EQUAL_WEIGHT = "EQUAL_WEIGHT"


class BacktestStatus(Enum):
    """Backtest execution status."""
    SUCCESS = "SUCCESS"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NO_TRADES = "NO_TRADES"
    NO_SIGNALS = "NO_SIGNALS"
    ERROR = "ERROR"


class TradeDirection(Enum):
    """Trade direction."""
    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(Enum):
    """Trade outcome status."""
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"


# =============================================================================
# SECTION 3: DATA STRUCTURES
# =============================================================================

@dataclass
class TransactionCosts:
    """
    Transaction cost model for realistic backtesting.
    
    Components modeled:
        - Commission: Broker fees (typically 5-10 bps for retail)
        - Slippage: Price impact from execution delay
        - Spread: Bid-ask spread cost
    
    Reference:
        Kissell (2013) "The Science of Algorithmic Trading"
    """
    commission_rate: float = Config.COMMISSION_RATE
    slippage_rate: float = Config.SLIPPAGE_RATE
    spread_rate: float = Config.SPREAD_RATE
    min_commission: float = Config.MIN_COMMISSION
    
    @property
    def total_rate(self) -> float:
        """Total cost as percentage per trade."""
        return self.commission_rate + self.slippage_rate + self.spread_rate
    
    def calculate_cost(
        self,
        trade_value: float,
        volatility: float = 0.02
    ) -> float:
        """
        Calculate total transaction cost for a trade.
        
        Slippage scales with volatility - higher vol = more slippage.
        
        Args:
            trade_value: Absolute dollar value of trade
            volatility: Current volatility for slippage scaling
            
        Returns:
            Total cost in dollars
        """
        # Commission with minimum
        commission = max(trade_value * self.commission_rate, self.min_commission)
        
        # Slippage scales with volatility
        vol_multiplier = max(1.0, volatility / 0.02)
        slippage = trade_value * self.slippage_rate * vol_multiplier
        
        # Fixed spread cost
        spread = trade_value * self.spread_rate
        
        return commission + slippage + spread


@dataclass
class Trade:
    """
    Individual trade record with full attribution.
    
    Tracks entry, exit, P&L, and duration for trade analysis.
    """
    trade_id: int
    direction: TradeDirection
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    shares: float = 0.0
    position_value: float = 0.0
    
    # Costs
    entry_cost: float = 0.0
    exit_cost: float = 0.0
    total_cost: float = 0.0
    
    # P&L
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    return_pct: float = 0.0
    
    # Duration
    holding_days: int = 0
    
    # Status
    status: TradeStatus = TradeStatus.BREAKEVEN
    is_open: bool = True
    
    def close(
        self,
        exit_date: datetime,
        exit_price: float,
        exit_cost: float
    ) -> None:
        """Close the trade and calculate P&L."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_cost = exit_cost
        self.total_cost = self.entry_cost + self.exit_cost
        self.is_open = False
        
        # Calculate P&L
        if self.direction == TradeDirection.LONG:
            self.gross_pnl = (exit_price - self.entry_price) * self.shares
        else:
            self.gross_pnl = (self.entry_price - exit_price) * self.shares
            
        self.net_pnl = self.gross_pnl - self.total_cost
        self.return_pct = self.net_pnl / self.position_value if self.position_value > 0 else 0
        
        # Duration
        self.holding_days = (exit_date - self.entry_date).days
        
        # Status
        if self.net_pnl > 0:
            self.status = TradeStatus.WIN
        elif self.net_pnl < 0:
            self.status = TradeStatus.LOSS
        else:
            self.status = TradeStatus.BREAKEVEN


@dataclass
class ReturnMetrics:
    """
    Return performance metrics.
    
    CAGR is the key metric required by coursework.
    """
    total_return: float           # Total percentage return
    cagr: float                   # Compound Annual Growth Rate (REQUIRED)
    annual_return: float          # Annualized return
    monthly_return: float         # Average monthly return
    daily_return: float           # Average daily return
    
    best_day: float               # Best single day return
    worst_day: float              # Worst single day return
    best_month: float             # Best month return
    worst_month: float            # Worst month return
    
    positive_days: int            # Number of positive days
    negative_days: int            # Number of negative days
    positive_ratio: float         # Ratio of positive days


@dataclass
class RiskMetrics:
    """
    Risk metrics including drawdown analysis.
    
    Maximum Drawdown is required by coursework.
    """
    # Volatility
    daily_volatility: float
    annual_volatility: float
    downside_volatility: float
    
    # Drawdown (REQUIRED)
    max_drawdown: float           # Maximum peak-to-trough decline
    avg_drawdown: float           # Average drawdown
    max_drawdown_duration: int    # Days in max drawdown
    current_drawdown: float       # Current drawdown level
    
    # Tail Risk
    var_95: float                 # 95% Value at Risk
    cvar_95: float                # 95% Conditional VaR (Expected Shortfall)
    
    # Distribution
    skewness: float
    kurtosis: float


@dataclass
class RiskAdjustedMetrics:
    """
    Risk-adjusted performance metrics.
    
    Sharpe Ratio is required by coursework.
    """
    # Primary (REQUIRED)
    sharpe_ratio: float           # Sharpe Ratio = (Return - Rf) / Volatility
    
    # Secondary
    sortino_ratio: float          # Uses downside volatility
    calmar_ratio: float           # CAGR / Max Drawdown
    
    # Additional
    information_ratio: float      # Alpha / Tracking Error
    omega_ratio: float            # Probability-weighted gains/losses
    
    # Benchmark comparison
    alpha: float                  # Excess return vs benchmark
    beta: float                   # Sensitivity to benchmark
    treynor_ratio: float          # Excess return per unit of systematic risk


@dataclass
class TradeStatistics:
    """
    Trade analysis statistics.
    
    Hit Rate (Win Rate) is required by coursework.
    """
    # Counts
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    
    # Hit Rate (REQUIRED)
    hit_rate: float               # Win Rate = Winning / Total
    
    # P&L Analysis
    avg_win: float                # Average winning trade
    avg_loss: float               # Average losing trade
    largest_win: float            # Largest single win
    largest_loss: float           # Largest single loss
    
    # Ratios
    profit_factor: float          # Gross Profit / Gross Loss
    payoff_ratio: float           # Avg Win / Avg Loss
    expectancy: float             # Expected value per trade
    
    # Duration
    avg_holding_days: float       # Average trade duration
    avg_winning_days: float       # Average duration of winners
    avg_losing_days: float        # Average duration of losers
    
    # Costs
    total_costs: float            # Total transaction costs
    cost_per_trade: float         # Average cost per trade


@dataclass
class DrawdownPeriod:
    """A single drawdown period."""
    start_date: datetime
    end_date: Optional[datetime]
    trough_date: datetime
    peak_value: float
    trough_value: float
    drawdown: float
    duration_days: int
    recovery_days: Optional[int]
    is_recovered: bool


@dataclass
class WalkForwardResult:
    """Walk-forward analysis result for a single period."""
    period: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # In-sample performance
    train_return: float
    train_sharpe: float
    train_trades: int
    
    # Out-of-sample performance
    test_return: float
    test_sharpe: float
    test_trades: int
    
    # Efficiency
    efficiency_ratio: float       # OOS Sharpe / IS Sharpe


@dataclass
class WalkForwardAnalysis:
    """Complete walk-forward analysis results."""
    periods: List[WalkForwardResult]
    n_periods: int
    
    # Aggregate metrics
    avg_is_return: float
    avg_oos_return: float
    avg_is_sharpe: float
    avg_oos_sharpe: float
    
    # Walk-Forward Efficiency
    wfe_ratio: float              # Avg OOS / Avg IS performance
    consistency: float            # % of periods where OOS > 0
    
    # Robustness
    is_robust: bool               # Strategy passes validation


@dataclass
class MonteCarloAnalysis:
    """Monte Carlo simulation results."""
    n_simulations: int
    
    # Return distribution
    return_mean: float
    return_std: float
    return_percentiles: Dict[float, float]
    
    # Drawdown distribution
    drawdown_mean: float
    drawdown_std: float
    drawdown_percentiles: Dict[float, float]
    
    # Sharpe distribution
    sharpe_mean: float
    sharpe_std: float
    sharpe_percentiles: Dict[float, float]
    
    # Confidence intervals
    return_95_ci: Tuple[float, float]
    sharpe_95_ci: Tuple[float, float]
    
    # Probability analysis
    prob_positive_return: float
    prob_sharpe_above_1: float


@dataclass
class PositionSizeRecommendation:
    """Position sizing recommendation."""
    method: PositionSizingMethod
    recommended_size: float       # As fraction of capital
    kelly_optimal: float          # Full Kelly size
    kelly_half: float             # Half Kelly
    volatility_scaled: float      # Based on target vol
    
    rationale: str


@dataclass
class BacktestResult:
    """
    Complete backtest result container.
    
    Contains all metrics required for Phase 5 LLM trade note:
        - CAGR (returns.cagr)
        - Sharpe Ratio (risk_adjusted.sharpe_ratio)
        - Maximum Drawdown (risk.max_drawdown)
        - Hit Rate (trades.hit_rate)
    """
    # Status
    status: BacktestStatus
    
    # Identification
    symbol: str
    strategy_name: str
    
    # Period
    start_date: datetime
    end_date: datetime
    trading_days: int
    
    # Capital
    initial_capital: float
    final_capital: float
    
    # Core Metrics (COURSEWORK REQUIRED)
    returns: ReturnMetrics
    risk: RiskMetrics
    risk_adjusted: RiskAdjustedMetrics
    trades: TradeStatistics
    
    # Advanced Analysis
    walk_forward: Optional[WalkForwardAnalysis] = None
    monte_carlo: Optional[MonteCarloAnalysis] = None
    position_sizing: Optional[PositionSizeRecommendation] = None
    
    # Trade Log
    trade_log: List[Trade] = field(default_factory=list)
    
    # Time Series
    equity_curve: Optional[pd.Series] = None
    daily_returns: Optional[pd.Series] = None
    drawdown_series: Optional[pd.Series] = None
    signals: Optional[pd.Series] = None  # Trading signals (1=long, 0=flat)
    
    # Benchmark Comparison
    benchmark_return: Optional[float] = None
    excess_return: Optional[float] = None
    
    # Metadata
    transaction_costs: Optional[TransactionCosts] = None
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    version: str = VERSION


# =============================================================================
# SECTION 4: RETURN CALCULATOR
# =============================================================================

class ReturnCalculator:
    """
    Calculate return metrics including CAGR.
    
    CAGR Formula:
        CAGR = (Final Value / Initial Value)^(1/Years) - 1
    
    This is the primary return metric required by the coursework.
    """
    
    @staticmethod
    def calculate(
        equity_curve: pd.Series,
        initial_capital: float
    ) -> ReturnMetrics:
        """
        Calculate all return metrics.
        
        Args:
            equity_curve: Portfolio value over time
            initial_capital: Starting capital
            
        Returns:
            ReturnMetrics with CAGR and other metrics
        """
        if len(equity_curve) < 2:
            return ReturnMetrics(
                total_return=0, cagr=0, annual_return=0, monthly_return=0,
                daily_return=0, best_day=0, worst_day=0, best_month=0,
                worst_month=0, positive_days=0, negative_days=0, positive_ratio=0
            )
        
        # Total return
        final_value = equity_curve.iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # CAGR calculation
        n_days = len(equity_curve)
        years = n_days / Config.TRADING_DAYS_YEAR
        
        if years > 0 and final_value > 0 and initial_capital > 0:
            cagr = (final_value / initial_capital) ** (1 / years) - 1
        else:
            cagr = 0.0
        
        # Daily returns
        daily_returns = equity_curve.pct_change().dropna()
        
        if len(daily_returns) == 0:
            return ReturnMetrics(
                total_return=total_return, cagr=cagr, annual_return=cagr,
                monthly_return=cagr/12 if cagr else 0, daily_return=0,
                best_day=0, worst_day=0, best_month=0, worst_month=0,
                positive_days=0, negative_days=0, positive_ratio=0
            )
        
        # Daily statistics
        daily_mean = daily_returns.mean()
        best_day = daily_returns.max()
        worst_day = daily_returns.min()
        
        positive_days = (daily_returns > 0).sum()
        negative_days = (daily_returns < 0).sum()
        total_days = positive_days + negative_days
        positive_ratio = positive_days / total_days if total_days > 0 else 0
        
        # Monthly returns
        monthly_returns = daily_returns.resample('ME').apply(
            lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0
        )
        best_month = monthly_returns.max() if len(monthly_returns) > 0 else 0
        worst_month = monthly_returns.min() if len(monthly_returns) > 0 else 0
        monthly_mean = monthly_returns.mean() if len(monthly_returns) > 0 else 0
        
        # Annualized return
        annual_return = (1 + daily_mean) ** Config.TRADING_DAYS_YEAR - 1
        
        return ReturnMetrics(
            total_return=total_return,
            cagr=cagr,
            annual_return=annual_return,
            monthly_return=monthly_mean,
            daily_return=daily_mean,
            best_day=best_day,
            worst_day=worst_day,
            best_month=best_month,
            worst_month=worst_month,
            positive_days=int(positive_days),
            negative_days=int(negative_days),
            positive_ratio=positive_ratio
        )


# =============================================================================
# SECTION 5: RISK CALCULATOR
# =============================================================================

class RiskCalculator:
    """
    Calculate risk metrics including maximum drawdown.
    
    Maximum Drawdown Formula:
        DD = (Trough Value - Peak Value) / Peak Value
        Max DD = min(all drawdowns)
    
    This is a key risk metric required by the coursework.
    """
    
    @staticmethod
    def calculate_drawdown_series(equity_curve: pd.Series) -> pd.Series:
        """
        Calculate drawdown series.
        
        Drawdown at each point = (current - running max) / running max
        """
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return drawdown
    
    @staticmethod
    def calculate(
        daily_returns: pd.Series,
        equity_curve: pd.Series
    ) -> RiskMetrics:
        """
        Calculate all risk metrics.
        
        Args:
            daily_returns: Daily return series
            equity_curve: Portfolio value over time
            
        Returns:
            RiskMetrics with max drawdown and other metrics
        """
        if len(daily_returns) < 2:
            return RiskMetrics(
                daily_volatility=0, annual_volatility=0, downside_volatility=0,
                max_drawdown=0, avg_drawdown=0, max_drawdown_duration=0,
                current_drawdown=0, var_95=0, cvar_95=0, skewness=0, kurtosis=0
            )
        
        # Volatility
        daily_vol = daily_returns.std()
        annual_vol = daily_vol * np.sqrt(Config.TRADING_DAYS_YEAR)
        
        # Downside volatility (only negative returns)
        negative_returns = daily_returns[daily_returns < 0]
        downside_vol = negative_returns.std() * np.sqrt(Config.TRADING_DAYS_YEAR) if len(negative_returns) > 0 else 0
        
        # Drawdown analysis
        drawdown_series = RiskCalculator.calculate_drawdown_series(equity_curve)
        max_drawdown = abs(drawdown_series.min())
        avg_drawdown = abs(drawdown_series.mean())
        current_drawdown = abs(drawdown_series.iloc[-1])
        
        # Max drawdown duration
        is_in_drawdown = drawdown_series < 0
        drawdown_periods = []
        current_period = 0
        
        for in_dd in is_in_drawdown:
            if in_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        max_dd_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # VaR and CVaR
        var_95 = np.percentile(daily_returns, 5)  # 5th percentile = 95% VaR
        cvar_95 = daily_returns[daily_returns <= var_95].mean() if len(daily_returns[daily_returns <= var_95]) > 0 else var_95
        
        # Distribution moments
        skewness = stats.skew(daily_returns)
        kurtosis = stats.kurtosis(daily_returns)
        
        return RiskMetrics(
            daily_volatility=daily_vol,
            annual_volatility=annual_vol,
            downside_volatility=downside_vol,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            max_drawdown_duration=max_dd_duration,
            current_drawdown=current_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            skewness=skewness,
            kurtosis=kurtosis
        )


# =============================================================================
# SECTION 6: RISK-ADJUSTED CALCULATOR
# =============================================================================

class RiskAdjustedCalculator:
    """
    Calculate risk-adjusted performance metrics.
    
    Sharpe Ratio Formula:
        SR = (Return - Risk Free Rate) / Volatility
    
    This is the primary risk-adjusted metric required by the coursework.
    
    Reference:
        Sharpe, W.F. (1994). "The Sharpe Ratio."
    """
    
    @staticmethod
    def calculate(
        daily_returns: pd.Series,
        cagr: float,
        max_drawdown: float,
        annual_volatility: float,
        downside_volatility: float,
        benchmark_returns: Optional[pd.Series] = None
    ) -> RiskAdjustedMetrics:
        """
        Calculate risk-adjusted metrics.
        
        Args:
            daily_returns: Daily return series
            cagr: Compound annual growth rate
            max_drawdown: Maximum drawdown
            annual_volatility: Annualized volatility
            downside_volatility: Annualized downside volatility
            benchmark_returns: Optional benchmark return series
            
        Returns:
            RiskAdjustedMetrics with Sharpe ratio and others
        """
        rf = Config.RISK_FREE_RATE
        
        # Sharpe Ratio (REQUIRED)
        if annual_volatility > 0:
            sharpe_ratio = (cagr - rf) / annual_volatility
        else:
            sharpe_ratio = 0.0
        
        # Sortino Ratio (uses downside vol)
        if downside_volatility > 0:
            sortino_ratio = (cagr - rf) / downside_volatility
        else:
            sortino_ratio = sharpe_ratio  # Fallback
        
        # Calmar Ratio (return / drawdown)
        if max_drawdown > 0:
            calmar_ratio = cagr / max_drawdown
        else:
            calmar_ratio = float('inf') if cagr > 0 else 0
        
        # Omega Ratio
        # Ω = Σ(returns above threshold) / |Σ(returns below threshold)|
        threshold = rf / Config.TRADING_DAYS_YEAR  # Daily rf
        excess_returns = daily_returns - threshold
        gains = excess_returns[excess_returns > 0].sum()
        losses = abs(excess_returns[excess_returns < 0].sum())
        omega_ratio = gains / losses if losses > 0 else float('inf') if gains > 0 else 0
        
        # Benchmark comparison
        alpha = 0.0
        beta = 1.0
        information_ratio = 0.0
        treynor_ratio = 0.0
        
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            # Align series
            aligned = pd.concat([daily_returns, benchmark_returns], axis=1).dropna()
            if len(aligned) > 10:
                aligned.columns = ['strategy', 'benchmark']
                
                # Beta
                cov = np.cov(aligned['strategy'], aligned['benchmark'])
                beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.0
                
                # Alpha
                benchmark_annual = (1 + aligned['benchmark'].mean()) ** Config.TRADING_DAYS_YEAR - 1
                alpha = cagr - (rf + beta * (benchmark_annual - rf))
                
                # Tracking error
                tracking_diff = aligned['strategy'] - aligned['benchmark']
                tracking_error = tracking_diff.std() * np.sqrt(Config.TRADING_DAYS_YEAR)
                
                # Information Ratio
                information_ratio = alpha / tracking_error if tracking_error > 0 else 0
                
                # Treynor Ratio
                treynor_ratio = (cagr - rf) / beta if beta != 0 else 0
        
        return RiskAdjustedMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            omega_ratio=omega_ratio,
            alpha=alpha,
            beta=beta,
            treynor_ratio=treynor_ratio
        )


# =============================================================================
# SECTION 7: TRADE ANALYZER
# =============================================================================

class TradeAnalyzer:
    """
    Analyze trade statistics including hit rate.
    
    Hit Rate (Win Rate) Formula:
        Hit Rate = Winning Trades / Total Trades
    
    This is a key metric required by the coursework.
    """
    
    @staticmethod
    def analyze(trades: List[Trade]) -> TradeStatistics:
        """
        Calculate trade statistics.
        
        Args:
            trades: List of closed trades
            
        Returns:
            TradeStatistics with hit rate and other metrics
        """
        if not trades:
            return TradeStatistics(
                total_trades=0, winning_trades=0, losing_trades=0, breakeven_trades=0,
                hit_rate=0, avg_win=0, avg_loss=0, largest_win=0, largest_loss=0,
                profit_factor=0, payoff_ratio=0, expectancy=0,
                avg_holding_days=0, avg_winning_days=0, avg_losing_days=0,
                total_costs=0, cost_per_trade=0
            )
        
        # Filter closed trades only
        closed_trades = [t for t in trades if not t.is_open]
        
        if not closed_trades:
            return TradeStatistics(
                total_trades=0, winning_trades=0, losing_trades=0, breakeven_trades=0,
                hit_rate=0, avg_win=0, avg_loss=0, largest_win=0, largest_loss=0,
                profit_factor=0, payoff_ratio=0, expectancy=0,
                avg_holding_days=0, avg_winning_days=0, avg_losing_days=0,
                total_costs=0, cost_per_trade=0
            )
        
        # Categorize trades
        winners = [t for t in closed_trades if t.status == TradeStatus.WIN]
        losers = [t for t in closed_trades if t.status == TradeStatus.LOSS]
        breakevens = [t for t in closed_trades if t.status == TradeStatus.BREAKEVEN]
        
        total = len(closed_trades)
        n_winners = len(winners)
        n_losers = len(losers)
        n_breakeven = len(breakevens)
        
        # Hit Rate (REQUIRED)
        hit_rate = n_winners / total if total > 0 else 0
        
        # Win/Loss analysis
        winning_pnls = [t.net_pnl for t in winners] if winners else [0]
        losing_pnls = [t.net_pnl for t in losers] if losers else [0]
        
        avg_win = np.mean(winning_pnls) if winners else 0
        avg_loss = abs(np.mean(losing_pnls)) if losers else 0
        largest_win = max(winning_pnls) if winners else 0
        largest_loss = abs(min(losing_pnls)) if losers else 0
        
        # Profit Factor
        gross_profit = sum(winning_pnls)
        gross_loss = abs(sum(losing_pnls))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # Payoff Ratio
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf') if avg_win > 0 else 0
        
        # Expectancy
        # E = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)
        expectancy = (hit_rate * avg_win) - ((1 - hit_rate) * avg_loss)
        
        # Duration analysis
        all_days = [t.holding_days for t in closed_trades]
        winning_days = [t.holding_days for t in winners] if winners else [0]
        losing_days = [t.holding_days for t in losers] if losers else [0]
        
        avg_holding = np.mean(all_days) if all_days else 0
        avg_winning_hold = np.mean(winning_days) if winning_days else 0
        avg_losing_hold = np.mean(losing_days) if losing_days else 0
        
        # Costs
        total_costs = sum(t.total_cost for t in closed_trades)
        cost_per_trade = total_costs / total if total > 0 else 0
        
        return TradeStatistics(
            total_trades=total,
            winning_trades=n_winners,
            losing_trades=n_losers,
            breakeven_trades=n_breakeven,
            hit_rate=hit_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            payoff_ratio=payoff_ratio,
            expectancy=expectancy,
            avg_holding_days=avg_holding,
            avg_winning_days=avg_winning_hold,
            avg_losing_days=avg_losing_hold,
            total_costs=total_costs,
            cost_per_trade=cost_per_trade
        )


# =============================================================================
# SECTION 8: POSITION SIZER
# =============================================================================

class PositionSizer:
    """
    Position sizing calculator.
    
    Implements multiple sizing methodologies:
        - Kelly Criterion: Optimal growth rate
        - Fixed Fractional: Fixed percentage per trade
        - Volatility Targeting: Size based on target volatility
    
    Reference:
        Kelly, J.L. (1956). "A New Interpretation of Information Rate."
    """
    
    @staticmethod
    def kelly_criterion(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = Config.KELLY_FRACTION
    ) -> float:
        """
        Calculate Kelly-optimal position size.
        
        Kelly Formula:
            f* = (b*p - q) / b
        where:
            b = avg_win / avg_loss (odds)
            p = win_rate
            q = 1 - p
        
        Args:
            win_rate: Historical win rate
            avg_win: Average winning trade
            avg_loss: Average losing trade
            fraction: Kelly fraction (0.25 = quarter Kelly)
            
        Returns:
            Optimal position size as fraction of capital
        """
        if avg_loss <= 0 or win_rate <= 0:
            return 0.0
        
        # Odds ratio
        b = avg_win / avg_loss
        
        # Win/loss probabilities
        p = win_rate
        q = 1 - p
        
        # Kelly formula
        kelly_full = (b * p - q) / b if b > 0 else 0
        
        # Apply fraction (quarter-Kelly is conservative)
        kelly_adjusted = kelly_full * fraction
        
        # Constrain to valid range
        return np.clip(kelly_adjusted, Config.MIN_POSITION_SIZE, Config.MAX_POSITION_SIZE)
    
    @staticmethod
    def volatility_target(
        current_volatility: float,
        target_volatility: float = Config.TARGET_VOLATILITY
    ) -> float:
        """
        Size position to achieve target portfolio volatility.
        
        Formula:
            Position Size = Target Vol / Current Vol
        
        Args:
            current_volatility: Current annualized volatility
            target_volatility: Target annualized volatility
            
        Returns:
            Position size as fraction of capital
        """
        if current_volatility <= 0:
            return Config.DEFAULT_POSITION_SIZE
        
        size = target_volatility / current_volatility
        return np.clip(size, Config.MIN_POSITION_SIZE, Config.MAX_POSITION_SIZE)
    
    @staticmethod
    def calculate(
        trade_stats: TradeStatistics,
        annual_volatility: float,
        method: PositionSizingMethod = PositionSizingMethod.KELLY_CRITERION
    ) -> PositionSizeRecommendation:
        """
        Calculate position size recommendation.
        
        Args:
            trade_stats: Historical trade statistics
            annual_volatility: Current annualized volatility
            method: Sizing methodology
            
        Returns:
            Position size recommendation with rationale
        """
        # Kelly calculations
        kelly_full = PositionSizer.kelly_criterion(
            trade_stats.hit_rate,
            trade_stats.avg_win,
            trade_stats.avg_loss,
            fraction=1.0  # Full Kelly
        )
        kelly_half = kelly_full * 0.5
        kelly_quarter = kelly_full * 0.25
        
        # Volatility-based
        vol_scaled = PositionSizer.volatility_target(annual_volatility)
        
        # Select based on method
        if method == PositionSizingMethod.KELLY_CRITERION:
            recommended = kelly_quarter  # Conservative
            rationale = f"Quarter-Kelly sizing based on {trade_stats.hit_rate:.1%} win rate"
        elif method == PositionSizingMethod.VOLATILITY_TARGET:
            recommended = vol_scaled
            rationale = f"Volatility-targeted sizing for {Config.TARGET_VOLATILITY:.0%} annual vol"
        elif method == PositionSizingMethod.FIXED_FRACTIONAL:
            recommended = 0.02  # 2% risk per trade
            rationale = "Fixed 2% risk per trade"
        else:
            recommended = 1.0
            rationale = "Equal weight sizing"
        
        return PositionSizeRecommendation(
            method=method,
            recommended_size=recommended,
            kelly_optimal=kelly_full,
            kelly_half=kelly_half,
            volatility_scaled=vol_scaled,
            rationale=rationale
        )


# =============================================================================
# SECTION 9: WALK-FORWARD VALIDATOR
# =============================================================================

class WalkForwardValidator:
    """
    Walk-forward analysis for strategy validation.
    
    Tests strategy robustness by:
        1. Training on historical data (in-sample)
        2. Testing on subsequent data (out-of-sample)
        3. Rolling forward through the entire dataset
    
    Reference:
        Pardo, R. (2008). "The Evaluation and Optimization of Trading Strategies."
    """
    
    def __init__(
        self,
        n_splits: int = Config.WF_N_SPLITS,
        train_ratio: float = Config.WF_TRAIN_RATIO
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            n_splits: Number of walk-forward periods
            train_ratio: Proportion for training (rest is testing)
        """
        self.n_splits = n_splits
        self.train_ratio = train_ratio
    
    def analyze(
        self,
        daily_returns: pd.Series
    ) -> WalkForwardAnalysis:
        """
        Perform walk-forward analysis.
        
        Args:
            daily_returns: Daily return series
            
        Returns:
            WalkForwardAnalysis with period results
        """
        n = len(daily_returns)
        
        if n < Config.WF_MIN_TRAIN_DAYS + Config.WF_MIN_TEST_DAYS:
            return WalkForwardAnalysis(
                periods=[], n_periods=0, avg_is_return=0, avg_oos_return=0,
                avg_is_sharpe=0, avg_oos_sharpe=0, wfe_ratio=0,
                consistency=0, is_robust=False
            )
        
        # Calculate period sizes
        period_size = n // self.n_splits
        train_size = int(period_size * self.train_ratio)
        test_size = period_size - train_size
        
        # Minimum sizes
        train_size = max(train_size, Config.WF_MIN_TRAIN_DAYS)
        test_size = max(test_size, Config.WF_MIN_TEST_DAYS)
        
        results = []
        
        for i in range(self.n_splits):
            # Calculate period boundaries
            start_idx = i * period_size
            train_end = start_idx + train_size
            test_end = min(train_end + test_size, n)
            
            if train_end >= n or test_end > n:
                break
            
            # Split data
            train_returns = daily_returns.iloc[start_idx:train_end]
            test_returns = daily_returns.iloc[train_end:test_end]
            
            # Calculate metrics
            train_total = (1 + train_returns).prod() - 1
            test_total = (1 + test_returns).prod() - 1
            
            train_sharpe = self._calculate_sharpe(train_returns)
            test_sharpe = self._calculate_sharpe(test_returns)
            
            # Efficiency ratio
            efficiency = test_sharpe / train_sharpe if train_sharpe != 0 else 0
            
            results.append(WalkForwardResult(
                period=i + 1,
                train_start=daily_returns.index[start_idx],
                train_end=daily_returns.index[train_end - 1],
                test_start=daily_returns.index[train_end],
                test_end=daily_returns.index[test_end - 1],
                train_return=train_total,
                train_sharpe=train_sharpe,
                train_trades=len(train_returns),
                test_return=test_total,
                test_sharpe=test_sharpe,
                test_trades=len(test_returns),
                efficiency_ratio=efficiency
            ))
        
        if not results:
            return WalkForwardAnalysis(
                periods=[], n_periods=0, avg_is_return=0, avg_oos_return=0,
                avg_is_sharpe=0, avg_oos_sharpe=0, wfe_ratio=0,
                consistency=0, is_robust=False
            )
        
        # Aggregate metrics
        avg_is_return = np.mean([r.train_return for r in results])
        avg_oos_return = np.mean([r.test_return for r in results])
        avg_is_sharpe = np.mean([r.train_sharpe for r in results])
        avg_oos_sharpe = np.mean([r.test_sharpe for r in results])
        
        # Walk-forward efficiency
        wfe = avg_oos_sharpe / avg_is_sharpe if avg_is_sharpe != 0 else 0
        
        # Consistency (% of periods with positive OOS return)
        positive_oos = sum(1 for r in results if r.test_return > 0)
        consistency = positive_oos / len(results) if results else 0
        
        # Robustness check
        is_robust = (
            wfe > 0.5 and  # OOS performance at least 50% of IS
            consistency >= 0.6 and  # At least 60% positive OOS periods
            avg_oos_return > 0  # Overall positive OOS
        )
        
        return WalkForwardAnalysis(
            periods=results,
            n_periods=len(results),
            avg_is_return=avg_is_return,
            avg_oos_return=avg_oos_return,
            avg_is_sharpe=avg_is_sharpe,
            avg_oos_sharpe=avg_oos_sharpe,
            wfe_ratio=wfe,
            consistency=consistency,
            is_robust=is_robust
        )
    
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio for a return series."""
        if len(returns) < 2:
            return 0.0
        
        mean_return = returns.mean() * Config.TRADING_DAYS_YEAR
        volatility = returns.std() * np.sqrt(Config.TRADING_DAYS_YEAR)
        
        if volatility == 0:
            return 0.0
        
        return (mean_return - Config.RISK_FREE_RATE) / volatility


# =============================================================================
# SECTION 10: MONTE CARLO SIMULATOR
# =============================================================================

class MonteCarloSimulator:
    """
    Monte Carlo simulation for strategy robustness testing.
    
    Uses stationary block bootstrap to preserve autocorrelation
    structure in the return series while generating alternative
    return paths.
    
    Reference:
        Politis, D.N. & Romano, J.P. (1994). "The Stationary Bootstrap."
    """
    
    def __init__(
        self,
        n_simulations: int = Config.MC_N_SIMULATIONS,
        block_size: int = Config.MC_BLOCK_SIZE
    ):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            n_simulations: Number of bootstrap simulations
            block_size: Block size for stationary bootstrap
        """
        self.n_simulations = n_simulations
        self.block_size = block_size
    
    def simulate(
        self,
        daily_returns: pd.Series,
        initial_capital: float = 100000
    ) -> MonteCarloAnalysis:
        """
        Run Monte Carlo simulation.
        
        Args:
            daily_returns: Historical daily returns
            initial_capital: Starting capital for equity curve
            
        Returns:
            MonteCarloAnalysis with distribution statistics
        """
        returns_array = daily_returns.values
        n = len(returns_array)
        
        if n < self.block_size * 2:
            return self._empty_result()
        
        # Storage for simulation results
        sim_returns = []
        sim_drawdowns = []
        sim_sharpes = []
        
        for _ in range(self.n_simulations):
            # Generate bootstrapped return series
            bootstrapped = self._stationary_bootstrap(returns_array)
            
            # Calculate metrics for this simulation
            total_return = (1 + bootstrapped).prod() - 1
            sim_returns.append(total_return)
            
            # Equity curve and drawdown
            equity = initial_capital * np.cumprod(1 + bootstrapped)
            running_max = np.maximum.accumulate(equity)
            drawdown = (running_max - equity) / running_max
            max_dd = drawdown.max()
            sim_drawdowns.append(max_dd)
            
            # Sharpe
            annual_return = (1 + bootstrapped.mean()) ** Config.TRADING_DAYS_YEAR - 1
            annual_vol = bootstrapped.std() * np.sqrt(Config.TRADING_DAYS_YEAR)
            sharpe = (annual_return - Config.RISK_FREE_RATE) / annual_vol if annual_vol > 0 else 0
            sim_sharpes.append(sharpe)
        
        # Calculate statistics
        return MonteCarloAnalysis(
            n_simulations=self.n_simulations,
            
            return_mean=np.mean(sim_returns),
            return_std=np.std(sim_returns),
            return_percentiles={p: np.percentile(sim_returns, p*100) for p in Config.MC_CONFIDENCE_LEVELS},
            
            drawdown_mean=np.mean(sim_drawdowns),
            drawdown_std=np.std(sim_drawdowns),
            drawdown_percentiles={p: np.percentile(sim_drawdowns, p*100) for p in Config.MC_CONFIDENCE_LEVELS},
            
            sharpe_mean=np.mean(sim_sharpes),
            sharpe_std=np.std(sim_sharpes),
            sharpe_percentiles={p: np.percentile(sim_sharpes, p*100) for p in Config.MC_CONFIDENCE_LEVELS},
            
            return_95_ci=(np.percentile(sim_returns, 2.5), np.percentile(sim_returns, 97.5)),
            sharpe_95_ci=(np.percentile(sim_sharpes, 2.5), np.percentile(sim_sharpes, 97.5)),
            
            prob_positive_return=np.mean([r > 0 for r in sim_returns]),
            prob_sharpe_above_1=np.mean([s > 1 for s in sim_sharpes])
        )
    
    def _stationary_bootstrap(self, returns: np.ndarray) -> np.ndarray:
        """
        Generate a bootstrapped return series using stationary bootstrap.
        
        Each block has a geometric probability of ending, which creates
        variable-length blocks that preserve time series properties.
        """
        n = len(returns)
        bootstrapped = []
        
        # Probability of ending a block
        p = 1 / self.block_size
        
        while len(bootstrapped) < n:
            # Random starting point
            start = np.random.randint(0, n)
            
            # Random block length (geometric distribution)
            block_len = np.random.geometric(p)
            
            # Extract block (with wraparound)
            for i in range(block_len):
                if len(bootstrapped) >= n:
                    break
                idx = (start + i) % n
                bootstrapped.append(returns[idx])
        
        return np.array(bootstrapped[:n])
    
    def _empty_result(self) -> MonteCarloAnalysis:
        """Return empty result when simulation cannot run."""
        return MonteCarloAnalysis(
            n_simulations=0,
            return_mean=0, return_std=0, return_percentiles={},
            drawdown_mean=0, drawdown_std=0, drawdown_percentiles={},
            sharpe_mean=0, sharpe_std=0, sharpe_percentiles={},
            return_95_ci=(0, 0), sharpe_95_ci=(0, 0),
            prob_positive_return=0, prob_sharpe_above_1=0
        )


# =============================================================================
# SECTION 11: BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    """
    Core backtesting engine with transaction costs and position sizing.
    
    This is the main class that runs the backtest simulation and
    calculates all metrics required by the coursework:
        - CAGR
        - Sharpe Ratio
        - Maximum Drawdown
        - Hit Rate
    
    Features:
        - Realistic transaction cost modeling
        - Signal-based entry/exit execution
        - Comprehensive trade logging
        - Position sizing support
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_costs: Optional[TransactionCosts] = None,
        position_sizing_method: PositionSizingMethod = PositionSizingMethod.KELLY_CRITERION
    ):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital ($100,000 default)
            transaction_costs: Cost model (uses default if None)
            position_sizing_method: Position sizing methodology
        """
        self.initial_capital = initial_capital
        self.costs = transaction_costs or TransactionCosts()
        self.sizing_method = position_sizing_method
        
        # State
        self._trades: List[Trade] = []
        self._trade_counter = 0
        self._current_trade: Optional[Trade] = None
    
    def run(
        self,
        prices: pd.Series,
        signals: pd.Series,
        benchmark_prices: Optional[pd.Series] = None,
        symbol: str = "UNKNOWN",
        strategy_name: str = "Technical Strategy"
    ) -> BacktestResult:
        """
        Run backtest on price series with signals.
        
        Args:
            prices: Price series (typically Close prices)
            signals: Signal series (1=long, -1=short, 0=flat)
            benchmark_prices: Optional benchmark for comparison
            symbol: Asset symbol
            strategy_name: Name of strategy
            
        Returns:
            BacktestResult with all metrics
        """
        import time
        start_time = time.time()
        
        # Reset state
        self._trades = []
        self._trade_counter = 0
        self._current_trade = None
        
        # Validate inputs
        if len(prices) < 50:
            return self._error_result(
                BacktestStatus.INSUFFICIENT_DATA,
                symbol, strategy_name,
                "Insufficient data (need at least 50 observations)"
            )
        
        # Align signals with prices
        aligned = pd.concat([prices, signals], axis=1).dropna()
        if len(aligned) < 50:
            return self._error_result(
                BacktestStatus.INSUFFICIENT_DATA,
                symbol, strategy_name,
                "Insufficient aligned data"
            )
        
        aligned.columns = ['price', 'signal']
        
        # Convert signals to position (-1, 0, 1)
        positions = aligned['signal'].apply(lambda x: np.sign(x) if abs(x) > 0.5 else 0)
        
        # Check for any signals
        if (positions != 0).sum() == 0:
            return self._error_result(
                BacktestStatus.NO_SIGNALS,
                symbol, strategy_name,
                "No trading signals generated"
            )
        
        # Calculate returns
        price_returns = aligned['price'].pct_change().fillna(0)
        
        # Strategy returns (position from previous day * today's return)
        # This models execution at close with next-day return capture
        strategy_returns = positions.shift(1).fillna(0) * price_returns
        
        # Apply transaction costs on position changes
        position_changes = positions.diff().abs().fillna(0)
        cost_series = position_changes * self.costs.total_rate
        strategy_returns = strategy_returns - cost_series
        
        # Build equity curve
        equity_multiplier = (1 + strategy_returns).cumprod()
        equity_curve = self.initial_capital * equity_multiplier
        
        # Generate trade log
        self._generate_trades(aligned, positions, strategy_returns)
        
        if len(self._trades) == 0:
            return self._error_result(
                BacktestStatus.NO_TRADES,
                symbol, strategy_name,
                "No trades executed"
            )
        
        # Calculate all metrics
        final_capital = equity_curve.iloc[-1]
        
        # Return metrics (includes CAGR)
        return_metrics = ReturnCalculator.calculate(equity_curve, self.initial_capital)
        
        # Risk metrics (includes max drawdown)
        daily_returns = strategy_returns.dropna()
        risk_metrics = RiskCalculator.calculate(daily_returns, equity_curve)
        
        # Risk-adjusted metrics (includes Sharpe)
        benchmark_returns = benchmark_prices.pct_change().dropna() if benchmark_prices is not None else None
        risk_adjusted = RiskAdjustedCalculator.calculate(
            daily_returns,
            return_metrics.cagr,
            risk_metrics.max_drawdown,
            risk_metrics.annual_volatility,
            risk_metrics.downside_volatility,
            benchmark_returns
        )
        
        # Trade statistics (includes hit rate)
        trade_stats = TradeAnalyzer.analyze(self._trades)
        
        # Position sizing recommendation
        position_sizing = PositionSizer.calculate(
            trade_stats,
            risk_metrics.annual_volatility,
            self.sizing_method
        )
        
        # Benchmark comparison
        benchmark_return = None
        excess_return = None
        if benchmark_prices is not None:
            benchmark_total = (benchmark_prices.iloc[-1] / benchmark_prices.iloc[0]) - 1
            benchmark_return = benchmark_total
            excess_return = return_metrics.total_return - benchmark_total
        
        execution_time = (time.time() - start_time) * 1000
        
        return BacktestResult(
            status=BacktestStatus.SUCCESS,
            symbol=symbol,
            strategy_name=strategy_name,
            start_date=aligned.index[0],
            end_date=aligned.index[-1],
            trading_days=len(aligned),
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            returns=return_metrics,
            risk=risk_metrics,
            risk_adjusted=risk_adjusted,
            trades=trade_stats,
            trade_log=self._trades.copy(),
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            drawdown_series=RiskCalculator.calculate_drawdown_series(equity_curve),
            signals=positions,  # Store the actual trading signals
            benchmark_return=benchmark_return,
            excess_return=excess_return,
            transaction_costs=self.costs,
            position_sizing=position_sizing,
            execution_time_ms=execution_time
        )
    
    def _generate_trades(
        self,
        aligned: pd.DataFrame,
        positions: pd.Series,
        returns: pd.Series
    ) -> None:
        """
        Generate trade log from position changes.
        
        Tracks entry/exit points and calculates P&L for each trade.
        """
        current_position = 0
        entry_price = 0.0
        entry_date = None
        
        for i, (date, row) in enumerate(aligned.iterrows()):
            price = row['price']
            new_position = positions.iloc[i]
            
            # Position change detected
            if new_position != current_position:
                # Close existing trade if any
                if current_position != 0 and self._current_trade is not None:
                    exit_cost = self.costs.calculate_cost(
                        self._current_trade.position_value
                    )
                    self._current_trade.close(date, price, exit_cost)
                    self._trades.append(self._current_trade)
                    self._current_trade = None
                
                # Open new trade if entering position
                if new_position != 0:
                    self._trade_counter += 1
                    direction = TradeDirection.LONG if new_position > 0 else TradeDirection.SHORT
                    position_value = self.initial_capital  # Full position for now
                    shares = position_value / price
                    entry_cost = self.costs.calculate_cost(position_value)
                    
                    self._current_trade = Trade(
                        trade_id=self._trade_counter,
                        direction=direction,
                        entry_date=date,
                        entry_price=price,
                        shares=shares,
                        position_value=position_value,
                        entry_cost=entry_cost
                    )
                
                current_position = new_position
        
        # Close any remaining open trade
        if self._current_trade is not None:
            final_price = aligned['price'].iloc[-1]
            final_date = aligned.index[-1]
            exit_cost = self.costs.calculate_cost(self._current_trade.position_value)
            self._current_trade.close(final_date, final_price, exit_cost)
            self._trades.append(self._current_trade)
    
    def _error_result(
        self,
        status: BacktestStatus,
        symbol: str,
        strategy_name: str,
        message: str
    ) -> BacktestResult:
        """Create error result with empty metrics."""
        logger.warning(f"Backtest failed: {message}")
        
        return BacktestResult(
            status=status,
            symbol=symbol,
            strategy_name=strategy_name,
            start_date=datetime.now(),
            end_date=datetime.now(),
            trading_days=0,
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            returns=ReturnMetrics(0,0,0,0,0,0,0,0,0,0,0,0),
            risk=RiskMetrics(0,0,0,0,0,0,0,0,0,0,0),
            risk_adjusted=RiskAdjustedMetrics(0,0,0,0,0,0,0,0),
            trades=TradeStatistics(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
        )


# =============================================================================
# SECTION 12: BACKTEST PIPELINE
# =============================================================================

class BacktestPipeline:
    """
    Complete backtesting pipeline orchestrator.
    
    Integrates:
        - Core backtesting engine
        - Walk-forward validation
        - Monte Carlo simulation
        - Position sizing
    
    This is the main entry point for Phase 4 backtesting.
    
    Usage:
        pipeline = BacktestPipeline()
        result = pipeline.run(prices, signals, symbol='AAPL')
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_costs: Optional[TransactionCosts] = None,
        run_walk_forward: bool = True,
        run_monte_carlo: bool = True,
        wf_splits: int = Config.WF_N_SPLITS,
        mc_simulations: int = Config.MC_N_SIMULATIONS
    ):
        """
        Initialize backtest pipeline.
        
        Args:
            initial_capital: Starting capital
            transaction_costs: Transaction cost model
            run_walk_forward: Whether to run walk-forward analysis
            run_monte_carlo: Whether to run Monte Carlo simulation
            wf_splits: Number of walk-forward periods
            mc_simulations: Number of MC simulations
        """
        self.initial_capital = initial_capital
        self.costs = transaction_costs or TransactionCosts()
        self.run_wf = run_walk_forward
        self.run_mc = run_monte_carlo
        
        # Components
        self.engine = BacktestEngine(initial_capital, self.costs)
        self.wf_validator = WalkForwardValidator(wf_splits)
        self.mc_simulator = MonteCarloSimulator(mc_simulations)
    
    def run(
        self,
        prices: pd.Series,
        signals: pd.Series,
        benchmark_prices: Optional[pd.Series] = None,
        symbol: str = "UNKNOWN",
        strategy_name: str = "Technical Strategy"
    ) -> BacktestResult:
        """
        Run complete backtest pipeline.
        
        Args:
            prices: Price series
            signals: Signal series (from Phase 2)
            benchmark_prices: Optional benchmark
            symbol: Asset symbol
            strategy_name: Strategy name
            
        Returns:
            BacktestResult with all analyses
        """
        logger.info(f"Running backtest pipeline for {symbol}...")
        
        # Run core backtest
        result = self.engine.run(
            prices, signals, benchmark_prices,
            symbol, strategy_name
        )
        
        if result.status != BacktestStatus.SUCCESS:
            return result
        
        # Run walk-forward analysis
        if self.run_wf and result.daily_returns is not None:
            logger.info("Running walk-forward analysis...")
            result.walk_forward = self.wf_validator.analyze(result.daily_returns)
        
        # Run Monte Carlo simulation
        if self.run_mc and result.daily_returns is not None:
            logger.info(f"Running Monte Carlo simulation ({self.mc_simulator.n_simulations} iterations)...")
            result.monte_carlo = self.mc_simulator.simulate(
                result.daily_returns,
                self.initial_capital
            )
        
        return result


# =============================================================================
# SECTION 13: OUTPUT FORMATTING
# =============================================================================

def format_backtest_report(result: BacktestResult) -> str:
    """
    Format backtest result as human-readable text report.
    
    Highlights the four key metrics required by coursework:
        1. CAGR
        2. Sharpe Ratio
        3. Maximum Drawdown
        4. Hit Rate
    
    Args:
        result: BacktestResult from pipeline
        
    Returns:
        Formatted string report
    """
    lines = [
        "=" * 70,
        "BACKTEST PERFORMANCE REPORT",
        "=" * 70,
        f"Symbol: {result.symbol}",
        f"Strategy: {result.strategy_name}",
        f"Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}",
        f"Trading Days: {result.trading_days:,}",
        f"Status: {result.status.value}",
        "",
        "-" * 70,
        "CAPITAL",
        "-" * 70,
        f"Initial Capital: ${result.initial_capital:,.2f}",
        f"Final Capital:   ${result.final_capital:,.2f}",
        f"Total Return:    {result.returns.total_return:+.2%}",
        "",
        "-" * 70,
        "KEY METRICS (COURSEWORK REQUIRED)",
        "-" * 70,
        f"  CAGR:              {result.returns.cagr:+.2%}",
        f"  Sharpe Ratio:      {result.risk_adjusted.sharpe_ratio:.3f}",
        f"  Maximum Drawdown:  {result.risk.max_drawdown:.2%}",
        f"  Hit Rate:          {result.trades.hit_rate:.1%}",
        "",
        "-" * 70,
        "RETURN ANALYSIS",
        "-" * 70,
        f"Annualized Return:   {result.returns.annual_return:+.2%}",
        f"Monthly Return:      {result.returns.monthly_return:+.2%}",
        f"Best Day:            {result.returns.best_day:+.2%}",
        f"Worst Day:           {result.returns.worst_day:+.2%}",
        f"Positive Days:       {result.returns.positive_days} ({result.returns.positive_ratio:.1%})",
        "",
        "-" * 70,
        "RISK ANALYSIS",
        "-" * 70,
        f"Annual Volatility:   {result.risk.annual_volatility:.2%}",
        f"Downside Volatility: {result.risk.downside_volatility:.2%}",
        f"Average Drawdown:    {result.risk.avg_drawdown:.2%}",
        f"Max DD Duration:     {result.risk.max_drawdown_duration} days",
        f"VaR (95%):           {result.risk.var_95:.2%}",
        f"CVaR (95%):          {result.risk.cvar_95:.2%}",
        f"Skewness:            {result.risk.skewness:+.3f}",
        f"Kurtosis:            {result.risk.kurtosis:.3f}",
        "",
        "-" * 70,
        "RISK-ADJUSTED METRICS",
        "-" * 70,
        f"Sharpe Ratio:        {result.risk_adjusted.sharpe_ratio:.3f}",
        f"Sortino Ratio:       {result.risk_adjusted.sortino_ratio:.3f}",
        f"Calmar Ratio:        {result.risk_adjusted.calmar_ratio:.3f}",
        f"Omega Ratio:         {result.risk_adjusted.omega_ratio:.3f}",
    ]
    
    if result.benchmark_return is not None:
        lines.extend([
            f"Alpha:               {result.risk_adjusted.alpha:+.2%}",
            f"Beta:                {result.risk_adjusted.beta:.3f}",
            f"Information Ratio:   {result.risk_adjusted.information_ratio:.3f}",
        ])
    
    lines.extend([
        "",
        "-" * 70,
        "TRADE STATISTICS",
        "-" * 70,
        f"Total Trades:        {result.trades.total_trades}",
        f"Winning Trades:      {result.trades.winning_trades}",
        f"Losing Trades:       {result.trades.losing_trades}",
        f"Hit Rate:            {result.trades.hit_rate:.1%}",
        f"Profit Factor:       {result.trades.profit_factor:.2f}",
        f"Payoff Ratio:        {result.trades.payoff_ratio:.2f}",
        f"Expectancy:          ${result.trades.expectancy:,.2f}",
        f"Avg Win:             ${result.trades.avg_win:,.2f}",
        f"Avg Loss:            ${result.trades.avg_loss:,.2f}",
        f"Largest Win:         ${result.trades.largest_win:,.2f}",
        f"Largest Loss:        ${result.trades.largest_loss:,.2f}",
        f"Avg Holding (days):  {result.trades.avg_holding_days:.1f}",
        "",
        "-" * 70,
        "TRANSACTION COSTS",
        "-" * 70,
        f"Total Costs:         ${result.trades.total_costs:,.2f}",
        f"Cost per Trade:      ${result.trades.cost_per_trade:,.2f}",
        f"Cost Rate:           {result.transaction_costs.total_rate:.2%}" if result.transaction_costs else "N/A",
    ])
    
    # Walk-forward results
    if result.walk_forward is not None:
        wf = result.walk_forward
        lines.extend([
            "",
            "-" * 70,
            "WALK-FORWARD ANALYSIS",
            "-" * 70,
            f"Periods:             {wf.n_periods}",
            f"Avg IS Return:       {wf.avg_is_return:+.2%}",
            f"Avg OOS Return:      {wf.avg_oos_return:+.2%}",
            f"Avg IS Sharpe:       {wf.avg_is_sharpe:.3f}",
            f"Avg OOS Sharpe:      {wf.avg_oos_sharpe:.3f}",
            f"WFE Ratio:           {wf.wfe_ratio:.2f}",
            f"Consistency:         {wf.consistency:.1%}",
            f"Robust:              {'Yes' if wf.is_robust else 'No'}",
        ])
    
    # Monte Carlo results
    if result.monte_carlo is not None:
        mc = result.monte_carlo
        lines.extend([
            "",
            "-" * 70,
            "MONTE CARLO ANALYSIS",
            "-" * 70,
            f"Simulations:         {mc.n_simulations:,}",
            f"Return (mean):       {mc.return_mean:+.2%}",
            f"Return (std):        {mc.return_std:.2%}",
            f"Return 95% CI:       [{mc.return_95_ci[0]:+.2%}, {mc.return_95_ci[1]:+.2%}]",
            f"Sharpe (mean):       {mc.sharpe_mean:.3f}",
            f"Sharpe 95% CI:       [{mc.sharpe_95_ci[0]:.3f}, {mc.sharpe_95_ci[1]:.3f}]",
            f"P(Return > 0):       {mc.prob_positive_return:.1%}",
            f"P(Sharpe > 1):       {mc.prob_sharpe_above_1:.1%}",
        ])
    
    # Position sizing
    if result.position_sizing is not None:
        ps = result.position_sizing
        lines.extend([
            "",
            "-" * 70,
            "POSITION SIZING",
            "-" * 70,
            f"Method:              {ps.method.value}",
            f"Recommended Size:    {ps.recommended_size:.1%}",
            f"Kelly Optimal:       {ps.kelly_optimal:.1%}",
            f"Kelly Half:          {ps.kelly_half:.1%}",
            f"Vol Scaled:          {ps.volatility_scaled:.1%}",
            f"Rationale:           {ps.rationale}",
        ])
    
    lines.extend([
        "",
        "=" * 70,
        f"Processing Time: {result.execution_time_ms:.0f}ms | Version: {result.version}",
        "=" * 70,
    ])
    
    return "\n".join(lines)


# =============================================================================
# SECTION 14: CONVENIENCE FUNCTIONS
# =============================================================================

def run_backtest(
    prices: pd.Series,
    signals: pd.Series,
    symbol: str = "UNKNOWN",
    initial_capital: float = 100000,
    include_walk_forward: bool = True,
    include_monte_carlo: bool = True
) -> BacktestResult:
    """
    Convenience function for running a complete backtest.
    
    Args:
        prices: Price series
        signals: Signal series from Phase 2
        symbol: Asset symbol
        initial_capital: Starting capital
        include_walk_forward: Run walk-forward analysis
        include_monte_carlo: Run Monte Carlo simulation
        
    Returns:
        BacktestResult with all metrics
        
    Example:
        >>> from technical_indicators import TechnicalIndicatorEngine
        >>> signals = engine.compute_all(df).current_analysis.overall_signal
        >>> result = run_backtest(df['Close'], signal_series, symbol='AAPL')
        >>> print(f"CAGR: {result.returns.cagr:.2%}")
        >>> print(f"Sharpe: {result.risk_adjusted.sharpe_ratio:.3f}")
    """
    pipeline = BacktestPipeline(
        initial_capital=initial_capital,
        run_walk_forward=include_walk_forward,
        run_monte_carlo=include_monte_carlo
    )
    return pipeline.run(prices, signals, symbol=symbol)


# =============================================================================
# SECTION 15: MODULE EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    'Config',
    'VERSION',
    
    # Enumerations
    'SignalType',
    'PositionSizingMethod',
    'BacktestStatus',
    'TradeDirection',
    'TradeStatus',
    
    # Data structures
    'TransactionCosts',
    'Trade',
    'ReturnMetrics',
    'RiskMetrics',
    'RiskAdjustedMetrics',
    'TradeStatistics',
    'DrawdownPeriod',
    'WalkForwardResult',
    'WalkForwardAnalysis',
    'MonteCarloAnalysis',
    'PositionSizeRecommendation',
    'BacktestResult',
    
    # Calculators
    'ReturnCalculator',
    'RiskCalculator',
    'RiskAdjustedCalculator',
    'TradeAnalyzer',
    'PositionSizer',
    
    # Validators
    'WalkForwardValidator',
    'MonteCarloSimulator',
    
    # Engine
    'BacktestEngine',
    'BacktestPipeline',
    
    # Convenience
    'run_backtest',
    'format_backtest_report',
]