"""
================================================================================
PHASE 4B: ADVANCED RISK ANALYTICS & PERFORMANCE ATTRIBUTION
================================================================================

MSc AI Agents in Asset Management - Track B: Technical Analyst Agent

This module provides institutional-grade risk analytics and performance attribution
that bridges the gap between academic theory and professional practice.

Components:
-----------
1. PERFORMANCE ATTRIBUTION
   - Alpha/Beta decomposition (CAPM)
   - Factor attribution analysis
   - Information Ratio & Tracking Error

2. REGIME-CONDITIONAL ANALYSIS
   - Performance by market regime (Bull/Bear/Sideways)
   - Performance by volatility regime (Low/Normal/High)
   - Rolling metrics analysis

3. SIGNAL QUALITY ANALYSIS  
   - Information Coefficient (IC)
   - Signal predictive power
   - Hit rate by regime

4. STRESS TESTING
   - Historical scenario analysis (COVID, 2022 Bear, etc.)
   - Monte Carlo stress testing
   - Tail risk assessment

5. DRAWDOWN ANALYSIS
   - Underwater equity analysis
   - Drawdown attribution by period
   - Recovery time analysis

6. CORRELATION & DEPENDENCY
   - Rolling correlation with benchmark
   - Correlation breakdown in stress
   - Beta stability analysis

Academic References:
-------------------
- Grinold & Kahn (2000): "Active Portfolio Management"
- Litterman (2003): "Modern Investment Management"
- Ang (2014): "Asset Management: A Systematic Approach"
- Bailey & Lopez de Prado (2012): "The Sharpe Ratio Efficient Frontier"

Version: 1.0.0
Author: TAMER Technical Analysis Agent
================================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# ENUMERATIONS
# =============================================================================

class PerformanceGrade(Enum):
    """Performance quality grade"""
    EXCEPTIONAL = "EXCEPTIONAL"      # Top decile
    EXCELLENT = "EXCELLENT"          # Top quartile
    GOOD = "GOOD"                    # Above median
    AVERAGE = "AVERAGE"              # Median
    BELOW_AVERAGE = "BELOW_AVERAGE"  # Below median
    POOR = "POOR"                    # Bottom quartile


class RiskLevel(Enum):
    """Risk assessment level"""
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"
    EXTREME = "EXTREME"


class MarketRegime(Enum):
    """Market regime classification"""
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"


class VolatilityRegime(Enum):
    """Volatility regime classification"""
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRISIS = "CRISIS"


# =============================================================================
# DATA CLASSES FOR RESULTS
# =============================================================================

@dataclass
class AlphaBetaDecomposition:
    """
    CAPM Alpha/Beta decomposition results.
    
    Academic Reference: Sharpe (1964) - Capital Asset Pricing Model
    
    Attributes
    ----------
    alpha : float
        Jensen's alpha (annualized excess return above CAPM prediction)
    beta : float
        Systematic risk exposure to the benchmark
    r_squared : float
        Proportion of variance explained by benchmark
    alpha_t_stat : float
        T-statistic for alpha significance
    alpha_p_value : float
        P-value for alpha significance test
    systematic_return : float
        Return attributable to market exposure
    idiosyncratic_return : float
        Return attributable to skill/alpha
    tracking_error : float
        Standard deviation of active returns
    information_ratio : float
        Alpha divided by tracking error
    """
    alpha: float
    beta: float
    r_squared: float
    alpha_t_stat: float
    alpha_p_value: float
    systematic_return: float
    idiosyncratic_return: float
    tracking_error: float
    information_ratio: float
    
    @property
    def alpha_significant(self) -> bool:
        """Is alpha statistically significant at 5% level?"""
        return self.alpha_p_value < 0.05
    
    @property
    def skill_contribution(self) -> float:
        """Percentage of return from skill vs market"""
        total = abs(self.systematic_return) + abs(self.idiosyncratic_return)
        if total == 0:
            return 0
        return abs(self.idiosyncratic_return) / total


@dataclass
class RegimePerformance:
    """
    Performance metrics for a specific market regime.
    
    Attributes
    ----------
    regime : str
        Regime name (BULL/BEAR/SIDEWAYS)
    days : int
        Number of trading days in regime
    pct_time : float
        Percentage of total time spent in regime
    total_return : float
        Cumulative return during regime
    cagr : float
        Annualized return during regime
    volatility : float
        Annualized volatility during regime
    sharpe : float
        Risk-adjusted return during regime
    max_drawdown : float
        Maximum drawdown during regime
    hit_rate : float
        Win rate during regime
    avg_daily_return : float
        Average daily return
    """
    regime: str
    days: int
    pct_time: float
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    avg_daily_return: float
    contribution: float = 0.0  # Contribution to total return


@dataclass
class SignalQuality:
    """
    Signal quality and predictive power metrics.
    
    Academic Reference: Grinold (1989) - Information Coefficient
    
    Attributes
    ----------
    information_coefficient : float
        Correlation between signals and forward returns
    ic_t_stat : float
        T-statistic for IC significance
    hit_rate_long : float
        Win rate when signal is long
    hit_rate_flat : float
        Win rate when signal is flat (avoided losses)
    signal_persistence : float
        Average number of days signal stays constant
    turnover : float
        Annualized portfolio turnover
    """
    information_coefficient: float
    ic_t_stat: float
    hit_rate_long: float
    hit_rate_flat: float
    signal_persistence: float
    turnover: float
    
    @property
    def ic_significant(self) -> bool:
        """Is IC statistically significant?"""
        return abs(self.ic_t_stat) > 1.96
    
    @property
    def signal_quality_grade(self) -> str:
        """Grade the signal quality"""
        ic = self.information_coefficient
        if ic > 0.10:
            return "EXCEPTIONAL"
        elif ic > 0.05:
            return "EXCELLENT"
        elif ic > 0.03:
            return "GOOD"
        elif ic > 0.01:
            return "AVERAGE"
        else:
            return "POOR"


@dataclass
class StressTestResult:
    """
    Results from a single stress test scenario.
    
    Attributes
    ----------
    scenario_name : str
        Name of the stress scenario
    period_start : str
        Start date of scenario
    period_end : str
        End date of scenario
    benchmark_return : float
        Benchmark return during scenario
    strategy_return : float
        Strategy return during scenario
    outperformance : float
        Strategy return minus benchmark return
    max_drawdown : float
        Maximum drawdown during scenario
    days_to_recovery : int
        Days to recover from max drawdown (if recovered)
    beta_during_stress : float
        Beta during the stress period
    """
    scenario_name: str
    period_start: str
    period_end: str
    benchmark_return: float
    strategy_return: float
    outperformance: float
    max_drawdown: float
    days_to_recovery: Optional[int]
    beta_during_stress: float
    
    @property
    def protected_downside(self) -> bool:
        """Did strategy protect against downside?"""
        if self.benchmark_return < 0:
            return self.strategy_return > self.benchmark_return
        return False


@dataclass
class DrawdownPeriod:
    """
    Information about a specific drawdown period.
    
    Attributes
    ----------
    start_date : datetime
        Start of drawdown
    trough_date : datetime
        Date of maximum drawdown
    end_date : datetime or None
        End of drawdown (recovery date) or None if ongoing
    depth : float
        Maximum drawdown depth (negative)
    duration_days : int
        Total days from peak to recovery (or current)
    recovery_days : int or None
        Days from trough to recovery
    peak_value : float
        Portfolio value at peak
    trough_value : float
        Portfolio value at trough
    """
    start_date: datetime
    trough_date: datetime
    end_date: Optional[datetime]
    depth: float
    duration_days: int
    recovery_days: Optional[int]
    peak_value: float
    trough_value: float
    
    @property
    def recovered(self) -> bool:
        """Has the drawdown been recovered?"""
        return self.end_date is not None


@dataclass
class RiskAnalyticsReport:
    """
    Complete risk analytics report combining all analyses.
    
    This is the master output of Phase 4B containing:
    - Performance attribution
    - Regime-conditional analysis
    - Signal quality metrics
    - Stress test results
    - Drawdown analysis
    - Overall risk assessment
    """
    # Metadata
    symbol: str
    analysis_date: datetime
    period_start: datetime
    period_end: datetime
    trading_days: int
    
    # Core metrics
    total_return: float
    cagr: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Attribution
    alpha_beta: AlphaBetaDecomposition
    
    # Regime analysis
    regime_performance: Dict[str, RegimePerformance]
    volatility_regime_performance: Dict[str, RegimePerformance]
    
    # Signal quality
    signal_quality: SignalQuality
    
    # Stress testing
    stress_tests: List[StressTestResult]
    
    # Drawdown analysis
    major_drawdowns: List[DrawdownPeriod]
    underwater_periods: int
    avg_drawdown_duration: float
    avg_recovery_time: float
    
    # Rolling analysis
    rolling_sharpe: pd.Series
    rolling_beta: pd.Series
    rolling_correlation: pd.Series
    
    # Risk assessment
    overall_risk_level: RiskLevel
    risk_adjusted_grade: PerformanceGrade
    
    # Confidence scores
    alpha_confidence: float  # Confidence in alpha being real
    strategy_robustness: float  # Overall robustness score


# =============================================================================
# RISK ANALYTICS ENGINE
# =============================================================================

class RiskAnalyticsEngine:
    """
    Advanced Risk Analytics & Performance Attribution Engine.
    
    This engine provides institutional-grade risk analysis including:
    - CAPM alpha/beta decomposition
    - Regime-conditional performance analysis
    - Signal quality assessment
    - Historical stress testing
    - Comprehensive drawdown analysis
    
    Parameters
    ----------
    risk_free_rate : float
        Annual risk-free rate (default: 5%)
    trading_days_year : int
        Trading days per year (default: 252)
    
    Example
    -------
    >>> engine = RiskAnalyticsEngine()
    >>> report = engine.analyze(
    ...     strategy_returns=strategy_returns,
    ...     benchmark_returns=benchmark_returns,
    ...     signals=signals,
    ...     regimes=regimes
    ... )
    >>> print(report.alpha_beta.alpha)
    0.0523  # 5.23% annualized alpha
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.05,
        trading_days_year: int = 252
    ):
        """Initialize the risk analytics engine."""
        self.rf = risk_free_rate
        self.trading_days = trading_days_year
        self.daily_rf = risk_free_rate / trading_days_year
    
    # =========================================================================
    # MAIN ANALYSIS METHOD
    # =========================================================================
    
    def analyze(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        signals: pd.Series,
        prices: pd.Series,
        regimes: Optional[pd.Series] = None,
        symbol: str = "UNKNOWN"
    ) -> RiskAnalyticsReport:
        """
        Perform comprehensive risk analytics.
        
        Parameters
        ----------
        strategy_returns : pd.Series
            Daily strategy returns (after costs)
        benchmark_returns : pd.Series
            Daily benchmark returns
        signals : pd.Series
            Trading signals (1=long, 0=flat)
        prices : pd.Series
            Price series for the asset
        regimes : pd.Series, optional
            Market regime classification
        symbol : str
            Asset symbol
            
        Returns
        -------
        RiskAnalyticsReport
            Complete risk analytics report
        """
        # Align all series
        common_idx = strategy_returns.index.intersection(benchmark_returns.index)
        strategy_returns = strategy_returns.loc[common_idx]
        benchmark_returns = benchmark_returns.loc[common_idx]
        signals = signals.loc[common_idx] if signals is not None else None
        prices = prices.loc[common_idx]
        
        # Build equity curve
        equity = (1 + strategy_returns).cumprod()
        benchmark_equity = (1 + benchmark_returns).cumprod()
        
        # Core metrics
        total_return = equity.iloc[-1] - 1
        years = len(strategy_returns) / self.trading_days
        cagr = (1 + total_return) ** (1/years) - 1
        vol = strategy_returns.std() * np.sqrt(self.trading_days)
        sharpe = (cagr - self.rf) / vol if vol > 0 else 0
        
        # Max drawdown
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_dd = abs(drawdown.min())
        
        # 1. Performance Attribution
        alpha_beta = self._compute_alpha_beta(strategy_returns, benchmark_returns)
        
        # 2. Regime Analysis
        regime_perf = self._analyze_regime_performance(
            strategy_returns, prices, regimes
        )
        vol_regime_perf = self._analyze_volatility_regime_performance(
            strategy_returns, benchmark_returns
        )
        
        # 3. Signal Quality
        signal_quality = self._analyze_signal_quality(
            signals, strategy_returns, benchmark_returns
        )
        
        # 4. Stress Testing
        stress_tests = self._run_stress_tests(
            strategy_returns, benchmark_returns, prices
        )
        
        # 5. Drawdown Analysis
        major_drawdowns, underwater_stats = self._analyze_drawdowns(equity)
        
        # 6. Rolling Analysis
        rolling_sharpe = self._compute_rolling_sharpe(strategy_returns)
        rolling_beta = self._compute_rolling_beta(strategy_returns, benchmark_returns)
        rolling_corr = self._compute_rolling_correlation(strategy_returns, benchmark_returns)
        
        # 7. Risk Assessment
        risk_level = self._assess_risk_level(
            vol, max_dd, alpha_beta.beta, stress_tests
        )
        grade = self._grade_performance(
            cagr, sharpe, alpha_beta, signal_quality
        )
        
        # 8. Confidence Scores
        alpha_conf = self._compute_alpha_confidence(alpha_beta, years)
        robustness = self._compute_robustness_score(
            signal_quality, stress_tests, regime_perf
        )
        
        return RiskAnalyticsReport(
            symbol=symbol,
            analysis_date=datetime.now(),
            period_start=strategy_returns.index[0].to_pydatetime() if hasattr(strategy_returns.index[0], 'to_pydatetime') else strategy_returns.index[0],
            period_end=strategy_returns.index[-1].to_pydatetime() if hasattr(strategy_returns.index[-1], 'to_pydatetime') else strategy_returns.index[-1],
            trading_days=len(strategy_returns),
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            alpha_beta=alpha_beta,
            regime_performance=regime_perf,
            volatility_regime_performance=vol_regime_perf,
            signal_quality=signal_quality,
            stress_tests=stress_tests,
            major_drawdowns=major_drawdowns,
            underwater_periods=underwater_stats['periods'],
            avg_drawdown_duration=underwater_stats['avg_duration'],
            avg_recovery_time=underwater_stats['avg_recovery'],
            rolling_sharpe=rolling_sharpe,
            rolling_beta=rolling_beta,
            rolling_correlation=rolling_corr,
            overall_risk_level=risk_level,
            risk_adjusted_grade=grade,
            alpha_confidence=alpha_conf,
            strategy_robustness=robustness
        )
    
    # =========================================================================
    # PERFORMANCE ATTRIBUTION
    # =========================================================================
    
    def _compute_alpha_beta(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> AlphaBetaDecomposition:
        """
        Compute CAPM alpha/beta decomposition.
        
        Uses OLS regression: R_strategy - Rf = alpha + beta * (R_benchmark - Rf) + epsilon
        
        Academic Reference: Jensen (1968) - "The Performance of Mutual Funds"
        """
        # Excess returns
        strat_excess = strategy_returns - self.daily_rf
        bench_excess = benchmark_returns - self.daily_rf
        
        # Align and clean
        valid_idx = strat_excess.notna() & bench_excess.notna()
        strat_excess = strat_excess[valid_idx]
        bench_excess = bench_excess[valid_idx]
        
        n = len(strat_excess)
        if n < 30:
            # Insufficient data
            return AlphaBetaDecomposition(
                alpha=0, beta=1, r_squared=0,
                alpha_t_stat=0, alpha_p_value=1,
                systematic_return=0, idiosyncratic_return=0,
                tracking_error=0, information_ratio=0
            )
        
        # OLS regression
        X = bench_excess.values
        y = strat_excess.values
        
        X_mean = X.mean()
        y_mean = y.mean()
        
        # Beta = Cov(X,Y) / Var(X)
        beta = np.cov(X, y)[0, 1] / np.var(X) if np.var(X) > 0 else 1
        
        # Alpha (daily)
        alpha_daily = y_mean - beta * X_mean
        
        # Annualize alpha
        alpha_annual = alpha_daily * self.trading_days
        
        # Residuals
        y_pred = alpha_daily + beta * X
        residuals = y - y_pred
        
        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # Standard error of alpha
        mse = ss_res / (n - 2) if n > 2 else 0
        se_alpha = np.sqrt(mse / n) if mse > 0 else 0
        
        # T-statistic and p-value for alpha
        from scipy import stats
        t_stat = alpha_daily / se_alpha if se_alpha > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2)) if n > 2 else 1
        
        # Systematic vs idiosyncratic return
        bench_return = (1 + benchmark_returns).prod() - 1
        years = len(strategy_returns) / self.trading_days
        bench_cagr = (1 + bench_return) ** (1/years) - 1 if years > 0 else 0
        
        systematic_return = beta * (bench_cagr - self.rf)
        strat_cagr = ((1 + strategy_returns).prod()) ** (1/years) - 1 if years > 0 else 0
        idiosyncratic_return = (strat_cagr - self.rf) - systematic_return
        
        # Tracking error (annualized)
        active_returns = strategy_returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(self.trading_days)
        
        # Information ratio
        active_return = strat_cagr - bench_cagr
        information_ratio = active_return / tracking_error if tracking_error > 0 else 0
        
        return AlphaBetaDecomposition(
            alpha=alpha_annual,
            beta=beta,
            r_squared=r_squared,
            alpha_t_stat=t_stat,
            alpha_p_value=p_value,
            systematic_return=systematic_return,
            idiosyncratic_return=idiosyncratic_return,
            tracking_error=tracking_error,
            information_ratio=information_ratio
        )
    
    # =========================================================================
    # REGIME-CONDITIONAL ANALYSIS
    # =========================================================================
    
    def _analyze_regime_performance(
        self,
        strategy_returns: pd.Series,
        prices: pd.Series,
        regimes: Optional[pd.Series]
    ) -> Dict[str, RegimePerformance]:
        """
        Analyze strategy performance by market regime.
        
        If regimes not provided, uses price-based regime classification:
        - BULL: Price > SMA(200) and SMA(50) > SMA(200)
        - BEAR: Price < SMA(200) and SMA(50) < SMA(200)
        - SIDEWAYS: Everything else
        """
        if regimes is None:
            # Generate regimes from price data
            sma_50 = prices.rolling(50).mean()
            sma_200 = prices.rolling(200).mean()
            
            regimes = pd.Series('SIDEWAYS', index=prices.index)
            bull_mask = (prices > sma_200) & (sma_50 > sma_200)
            bear_mask = (prices < sma_200) & (sma_50 < sma_200)
            regimes[bull_mask] = 'BULL'
            regimes[bear_mask] = 'BEAR'
        
        # Align
        common_idx = strategy_returns.index.intersection(regimes.index)
        strategy_returns = strategy_returns.loc[common_idx]
        regimes = regimes.loc[common_idx]
        
        result = {}
        total_return = (1 + strategy_returns).prod() - 1
        
        for regime in ['BULL', 'BEAR', 'SIDEWAYS']:
            mask = regimes == regime
            if mask.sum() == 0:
                continue
            
            regime_returns = strategy_returns[mask]
            n_days = len(regime_returns)
            pct_time = n_days / len(strategy_returns)
            
            # Returns
            regime_total = (1 + regime_returns).prod() - 1
            years = n_days / self.trading_days if n_days > 0 else 1
            regime_cagr = (1 + regime_total) ** (1/max(years, 0.01)) - 1
            
            # Volatility & Sharpe
            regime_vol = regime_returns.std() * np.sqrt(self.trading_days) if n_days > 1 else 0
            regime_sharpe = (regime_cagr - self.rf) / regime_vol if regime_vol > 0 else 0
            
            # Drawdown
            regime_equity = (1 + regime_returns).cumprod()
            regime_max = regime_equity.expanding().max()
            regime_dd = ((regime_equity - regime_max) / regime_max).min()
            
            # Hit rate
            hit_rate = (regime_returns > 0).mean()
            
            # Contribution
            contribution = regime_total / total_return if total_return != 0 else 0
            
            result[regime] = RegimePerformance(
                regime=regime,
                days=n_days,
                pct_time=pct_time,
                total_return=regime_total,
                cagr=regime_cagr,
                volatility=regime_vol,
                sharpe=regime_sharpe,
                max_drawdown=abs(regime_dd) if pd.notna(regime_dd) else 0,
                hit_rate=hit_rate,
                avg_daily_return=regime_returns.mean(),
                contribution=contribution
            )
        
        return result
    
    def _analyze_volatility_regime_performance(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, RegimePerformance]:
        """
        Analyze performance by volatility regime.
        
        Volatility regimes defined by rolling 20-day benchmark volatility percentiles:
        - LOW: < 25th percentile
        - NORMAL: 25th-75th percentile
        - HIGH: 75th-90th percentile
        - CRISIS: > 90th percentile
        """
        # Rolling volatility (20-day)
        rolling_vol = benchmark_returns.rolling(20).std() * np.sqrt(self.trading_days)
        
        # Percentiles
        vol_pcts = rolling_vol.rank(pct=True)
        
        vol_regimes = pd.Series('NORMAL', index=rolling_vol.index)
        vol_regimes[vol_pcts < 0.25] = 'LOW'
        vol_regimes[(vol_pcts >= 0.75) & (vol_pcts < 0.90)] = 'HIGH'
        vol_regimes[vol_pcts >= 0.90] = 'CRISIS'
        
        # Align
        common_idx = strategy_returns.index.intersection(vol_regimes.index)
        strategy_returns = strategy_returns.loc[common_idx]
        vol_regimes = vol_regimes.loc[common_idx]
        
        result = {}
        total_return = (1 + strategy_returns).prod() - 1
        
        for regime in ['LOW', 'NORMAL', 'HIGH', 'CRISIS']:
            mask = vol_regimes == regime
            if mask.sum() == 0:
                continue
            
            regime_returns = strategy_returns[mask]
            n_days = len(regime_returns)
            pct_time = n_days / len(strategy_returns)
            
            regime_total = (1 + regime_returns).prod() - 1
            years = n_days / self.trading_days if n_days > 0 else 1
            regime_cagr = (1 + regime_total) ** (1/max(years, 0.01)) - 1
            
            regime_vol = regime_returns.std() * np.sqrt(self.trading_days) if n_days > 1 else 0
            regime_sharpe = (regime_cagr - self.rf) / regime_vol if regime_vol > 0 else 0
            
            regime_equity = (1 + regime_returns).cumprod()
            regime_max = regime_equity.expanding().max()
            regime_dd = ((regime_equity - regime_max) / regime_max).min()
            
            hit_rate = (regime_returns > 0).mean()
            contribution = regime_total / total_return if total_return != 0 else 0
            
            result[regime] = RegimePerformance(
                regime=regime,
                days=n_days,
                pct_time=pct_time,
                total_return=regime_total,
                cagr=regime_cagr,
                volatility=regime_vol,
                sharpe=regime_sharpe,
                max_drawdown=abs(regime_dd) if pd.notna(regime_dd) else 0,
                hit_rate=hit_rate,
                avg_daily_return=regime_returns.mean(),
                contribution=contribution
            )
        
        return result
    
    # =========================================================================
    # SIGNAL QUALITY ANALYSIS
    # =========================================================================
    
    def _analyze_signal_quality(
        self,
        signals: pd.Series,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> SignalQuality:
        """
        Analyze signal quality and predictive power.
        
        Key metric: Information Coefficient (IC) - correlation between
        signals and subsequent returns.
        
        For trend-following strategies, IC is typically low because they
        don't predict returns - they follow trends. More relevant metrics:
        - Hit rate when long (win rate during exposure)
        - Hit rate when flat (% of avoided losses)
        - Signal timing value
        
        Academic Reference: Grinold & Kahn (2000) - "Active Portfolio Management"
        """
        if signals is None:
            return SignalQuality(
                information_coefficient=0,
                ic_t_stat=0,
                hit_rate_long=0.5,
                hit_rate_flat=0.5,
                signal_persistence=1,
                turnover=0
            )
        
        # Align
        common_idx = signals.index.intersection(benchmark_returns.index)
        if len(common_idx) < 30:
            return SignalQuality(
                information_coefficient=0,
                ic_t_stat=0,
                hit_rate_long=0.5,
                hit_rate_flat=0.5,
                signal_persistence=1,
                turnover=0
            )
        
        signals = signals.loc[common_idx].copy()
        benchmark_returns = benchmark_returns.loc[common_idx].copy()
        strategy_returns = strategy_returns.loc[common_idx].copy()
        
        # Forward returns (next day)
        forward_returns = benchmark_returns.shift(-1)
        
        # Clean data
        valid_mask = signals.notna() & forward_returns.notna()
        sig_clean = signals[valid_mask]
        fwd_clean = forward_returns[valid_mask]
        
        # Information Coefficient
        # For binary signals (0/1), IC measures if signal=1 predicts positive returns
        if len(sig_clean) > 30:
            # Check if signals have variance
            if sig_clean.std() > 0.01:
                ic = sig_clean.corr(fwd_clean)
                # IC t-statistic
                n = len(sig_clean)
                if abs(ic) < 0.9999:
                    ic_t = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2)
                else:
                    ic_t = 0
            else:
                # All signals same value - calculate timing value instead
                # Compare avg return when long vs overall avg return
                long_mask = sig_clean > 0.5
                if long_mask.sum() > 0 and (~long_mask).sum() > 0:
                    avg_ret_long = fwd_clean[long_mask].mean()
                    avg_ret_flat = fwd_clean[~long_mask].mean()
                    # IC proxy: positive if we're long on better days
                    ic = (avg_ret_long - avg_ret_flat) / (fwd_clean.std() + 1e-10)
                    ic = max(-1, min(1, ic))  # Bound to [-1, 1]
                    ic_t = 0  # Not a true IC, so no t-stat
                else:
                    ic = 0
                    ic_t = 0
        else:
            ic = 0
            ic_t = 0
        
        # Hit rates by signal
        long_mask = signals > 0.5
        flat_mask = signals <= 0.5
        
        # Hit rate long: % of positive return days when holding
        if long_mask.sum() > 0:
            hit_rate_long = (strategy_returns[long_mask] > 0).mean()
        else:
            hit_rate_long = 0.5
            
        # Hit rate flat: % of benchmark down days that we avoided
        if flat_mask.sum() > 0:
            hit_rate_flat = (benchmark_returns[flat_mask] < 0).mean()
        else:
            hit_rate_flat = 0  # No flat periods to evaluate
        
        # Signal persistence (average consecutive days same signal)
        signal_diff = signals.diff().abs()
        signal_changes = (signal_diff > 0.1).sum()  # Use threshold for float comparison
        signal_persistence = len(signals) / max(signal_changes, 1)
        
        # Turnover (annualized)
        daily_turnover = signal_diff.mean() if signal_diff.notna().any() else 0
        turnover = daily_turnover * self.trading_days if pd.notna(daily_turnover) else 0
        
        return SignalQuality(
            information_coefficient=ic if pd.notna(ic) else 0,
            ic_t_stat=ic_t if pd.notna(ic_t) else 0,
            hit_rate_long=float(hit_rate_long) if pd.notna(hit_rate_long) else 0.5,
            hit_rate_flat=float(hit_rate_flat) if pd.notna(hit_rate_flat) else 0.5,
            signal_persistence=float(signal_persistence),
            turnover=float(turnover)
        )
    
    # =========================================================================
    # STRESS TESTING
    # =========================================================================
    
    def _run_stress_tests(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        prices: pd.Series
    ) -> List[StressTestResult]:
        """
        Run historical stress tests against known market events.
        
        Scenarios tested:
        1. COVID Crash (Feb-Mar 2020)
        2. 2022 Bear Market (Jan-Oct 2022)
        3. 2018 Q4 Selloff (Oct-Dec 2018)
        4. China Devaluation (Aug 2015)
        5. Flash Crash Recovery (Aug 2015 - Nov 2015)
        """
        scenarios = [
            ("COVID Crash", "2020-02-19", "2020-03-23"),
            ("COVID Recovery", "2020-03-23", "2020-08-31"),
            ("2022 Bear Market", "2022-01-03", "2022-10-12"),
            ("2022 Recovery", "2022-10-12", "2023-07-31"),
            ("2018 Q4 Selloff", "2018-10-01", "2018-12-24"),
            ("2018 Recovery", "2018-12-24", "2019-04-30"),
            ("Aug 2015 Flash Crash", "2015-08-17", "2015-08-25"),
            ("Post-Flash Recovery", "2015-08-25", "2015-11-30"),
        ]
        
        results = []
        
        for name, start, end in scenarios:
            try:
                start_dt = pd.Timestamp(start)
                end_dt = pd.Timestamp(end)
                
                # Check if period exists in data
                if start_dt < strategy_returns.index[0] or end_dt > strategy_returns.index[-1]:
                    continue
                
                # Get returns for period
                mask = (strategy_returns.index >= start_dt) & (strategy_returns.index <= end_dt)
                if mask.sum() < 5:
                    continue
                
                period_strat = strategy_returns[mask]
                period_bench = benchmark_returns[mask]
                
                # Returns
                strat_return = (1 + period_strat).prod() - 1
                bench_return = (1 + period_bench).prod() - 1
                outperformance = strat_return - bench_return
                
                # Max drawdown during period
                period_equity = (1 + period_strat).cumprod()
                period_max = period_equity.expanding().max()
                period_dd = ((period_equity - period_max) / period_max).min()
                
                # Beta during stress
                if len(period_strat) > 10:
                    cov = np.cov(period_strat, period_bench)[0, 1]
                    var = np.var(period_bench)
                    stress_beta = cov / var if var > 0 else 1
                else:
                    stress_beta = 1
                
                # Recovery time (if applicable)
                recovery_days = None
                if strat_return < 0:
                    # Look for recovery after end date
                    post_mask = strategy_returns.index > end_dt
                    if post_mask.sum() > 0:
                        post_equity = (1 + strategy_returns[post_mask]).cumprod()
                        target = 1 / (1 + strat_return)  # Level needed to recover
                        recovered = post_equity >= target
                        if recovered.any():
                            recovery_days = recovered.idxmax() - end_dt
                            recovery_days = recovery_days.days
                
                results.append(StressTestResult(
                    scenario_name=name,
                    period_start=start,
                    period_end=end,
                    benchmark_return=bench_return,
                    strategy_return=strat_return,
                    outperformance=outperformance,
                    max_drawdown=abs(period_dd) if pd.notna(period_dd) else 0,
                    days_to_recovery=recovery_days,
                    beta_during_stress=stress_beta
                ))
                
            except Exception:
                continue
        
        return results
    
    # =========================================================================
    # DRAWDOWN ANALYSIS
    # =========================================================================
    
    def _analyze_drawdowns(
        self,
        equity: pd.Series,
        top_n: int = 5
    ) -> Tuple[List[DrawdownPeriod], Dict]:
        """
        Comprehensive drawdown analysis.
        
        Returns:
        - List of top N drawdown periods
        - Statistics on underwater periods
        """
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        
        # Identify separate drawdown periods
        dd_starts = in_drawdown & ~in_drawdown.shift(1).fillna(False)
        dd_ends = ~in_drawdown & in_drawdown.shift(1).fillna(False)
        
        periods = []
        current_start = None
        
        for i, (date, is_start) in enumerate(dd_starts.items()):
            if is_start:
                current_start = date
            
            if dd_ends.iloc[i] and current_start is not None:
                # End of drawdown - get period details
                period_mask = (drawdown.index >= current_start) & (drawdown.index <= date)
                period_dd = drawdown[period_mask]
                
                if len(period_dd) > 0:
                    trough_idx = period_dd.idxmin()
                    trough_depth = period_dd.min()
                    
                    # Duration
                    duration = (date - current_start).days
                    recovery = (date - trough_idx).days
                    
                    periods.append(DrawdownPeriod(
                        start_date=current_start,
                        trough_date=trough_idx,
                        end_date=date,
                        depth=trough_depth,
                        duration_days=duration,
                        recovery_days=recovery,
                        peak_value=running_max.loc[current_start],
                        trough_value=equity.loc[trough_idx]
                    ))
                
                current_start = None
        
        # Check if currently in drawdown
        if in_drawdown.iloc[-1] and current_start is not None:
            period_mask = drawdown.index >= current_start
            period_dd = drawdown[period_mask]
            trough_idx = period_dd.idxmin()
            
            periods.append(DrawdownPeriod(
                start_date=current_start,
                trough_date=trough_idx,
                end_date=None,  # Ongoing
                depth=period_dd.min(),
                duration_days=(equity.index[-1] - current_start).days,
                recovery_days=None,
                peak_value=running_max.loc[current_start],
                trough_value=equity.loc[trough_idx]
            ))
        
        # Sort by depth and take top N
        periods.sort(key=lambda x: x.depth)
        major_drawdowns = periods[:top_n]
        
        # Underwater statistics
        underwater_days = in_drawdown.sum()
        n_periods = len(periods)
        avg_duration = np.mean([p.duration_days for p in periods]) if periods else 0
        recovered = [p for p in periods if p.recovered]
        avg_recovery = np.mean([p.recovery_days for p in recovered]) if recovered else 0
        
        stats = {
            'periods': n_periods,
            'underwater_days': underwater_days,
            'pct_underwater': underwater_days / len(equity),
            'avg_duration': avg_duration,
            'avg_recovery': avg_recovery
        }
        
        return major_drawdowns, stats
    
    # =========================================================================
    # ROLLING ANALYSIS
    # =========================================================================
    
    def _compute_rolling_sharpe(
        self,
        returns: pd.Series,
        window: int = 252
    ) -> pd.Series:
        """Compute rolling Sharpe ratio."""
        rolling_mean = returns.rolling(window).mean() * self.trading_days
        rolling_std = returns.rolling(window).std() * np.sqrt(self.trading_days)
        rolling_sharpe = (rolling_mean - self.rf) / rolling_std
        return rolling_sharpe.dropna()
    
    def _compute_rolling_beta(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 60
    ) -> pd.Series:
        """Compute rolling beta."""
        cov = strategy_returns.rolling(window).cov(benchmark_returns)
        var = benchmark_returns.rolling(window).var()
        rolling_beta = cov / var
        return rolling_beta.dropna()
    
    def _compute_rolling_correlation(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 60
    ) -> pd.Series:
        """Compute rolling correlation."""
        return strategy_returns.rolling(window).corr(benchmark_returns).dropna()
    
    # =========================================================================
    # RISK ASSESSMENT
    # =========================================================================
    
    def _assess_risk_level(
        self,
        volatility: float,
        max_drawdown: float,
        beta: float,
        stress_tests: List[StressTestResult]
    ) -> RiskLevel:
        """
        Assess overall risk level.
        
        Uses multiple factors:
        - Volatility level
        - Maximum drawdown
        - Beta exposure
        - Stress test performance
        """
        risk_score = 0
        
        # Volatility contribution
        if volatility < 0.10:
            risk_score += 1
        elif volatility < 0.20:
            risk_score += 2
        elif volatility < 0.30:
            risk_score += 3
        elif volatility < 0.40:
            risk_score += 4
        else:
            risk_score += 5
        
        # Drawdown contribution
        if max_drawdown < 0.10:
            risk_score += 1
        elif max_drawdown < 0.20:
            risk_score += 2
        elif max_drawdown < 0.30:
            risk_score += 3
        elif max_drawdown < 0.40:
            risk_score += 4
        else:
            risk_score += 5
        
        # Beta contribution
        if beta < 0.5:
            risk_score += 1
        elif beta < 0.8:
            risk_score += 2
        elif beta < 1.2:
            risk_score += 3
        elif beta < 1.5:
            risk_score += 4
        else:
            risk_score += 5
        
        # Stress test contribution
        if stress_tests:
            stress_losses = [st.strategy_return for st in stress_tests if st.strategy_return < 0]
            if stress_losses:
                worst_stress = min(stress_losses)
                if worst_stress > -0.10:
                    risk_score += 1
                elif worst_stress > -0.20:
                    risk_score += 2
                elif worst_stress > -0.30:
                    risk_score += 3
                else:
                    risk_score += 4
        
        # Map to risk level
        if risk_score <= 5:
            return RiskLevel.VERY_LOW
        elif risk_score <= 8:
            return RiskLevel.LOW
        elif risk_score <= 11:
            return RiskLevel.MODERATE
        elif risk_score <= 14:
            return RiskLevel.HIGH
        elif risk_score <= 17:
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.EXTREME
    
    def _grade_performance(
        self,
        cagr: float,
        sharpe: float,
        alpha_beta: AlphaBetaDecomposition,
        signal_quality: SignalQuality
    ) -> PerformanceGrade:
        """
        Grade risk-adjusted performance.
        
        Uses:
        - Sharpe ratio
        - Alpha significance
        - Information ratio
        - Signal quality
        """
        score = 0
        
        # Sharpe contribution (0-30 points)
        if sharpe > 1.5:
            score += 30
        elif sharpe > 1.0:
            score += 25
        elif sharpe > 0.7:
            score += 20
        elif sharpe > 0.5:
            score += 15
        elif sharpe > 0.3:
            score += 10
        else:
            score += 5
        
        # Alpha contribution (0-30 points)
        if alpha_beta.alpha_significant and alpha_beta.alpha > 0.10:
            score += 30
        elif alpha_beta.alpha > 0.05:
            score += 20
        elif alpha_beta.alpha > 0.02:
            score += 15
        elif alpha_beta.alpha > 0:
            score += 10
        else:
            score += 0
        
        # IR contribution (0-20 points)
        ir = alpha_beta.information_ratio
        if ir > 1.0:
            score += 20
        elif ir > 0.5:
            score += 15
        elif ir > 0.25:
            score += 10
        elif ir > 0:
            score += 5
        
        # Signal quality contribution (0-20 points)
        ic = signal_quality.information_coefficient
        if ic > 0.10:
            score += 20
        elif ic > 0.05:
            score += 15
        elif ic > 0.02:
            score += 10
        elif ic > 0:
            score += 5
        
        # Map to grade
        if score >= 85:
            return PerformanceGrade.EXCEPTIONAL
        elif score >= 70:
            return PerformanceGrade.EXCELLENT
        elif score >= 55:
            return PerformanceGrade.GOOD
        elif score >= 40:
            return PerformanceGrade.AVERAGE
        elif score >= 25:
            return PerformanceGrade.BELOW_AVERAGE
        else:
            return PerformanceGrade.POOR
    
    def _compute_alpha_confidence(
        self,
        alpha_beta: AlphaBetaDecomposition,
        years: float
    ) -> float:
        """
        Compute confidence that alpha is real (not luck).
        
        Based on:
        - Statistical significance of alpha
        - Length of track record
        - R-squared (how well model fits)
        """
        confidence = 0.0
        
        # P-value contribution (0-40%)
        if alpha_beta.alpha_p_value < 0.01:
            confidence += 0.40
        elif alpha_beta.alpha_p_value < 0.05:
            confidence += 0.30
        elif alpha_beta.alpha_p_value < 0.10:
            confidence += 0.20
        elif alpha_beta.alpha_p_value < 0.20:
            confidence += 0.10
        
        # Track record length (0-30%)
        if years >= 10:
            confidence += 0.30
        elif years >= 5:
            confidence += 0.20
        elif years >= 3:
            confidence += 0.15
        elif years >= 1:
            confidence += 0.10
        
        # R-squared quality (0-15%)
        # Low R-squared means more idiosyncratic return, which is good if alpha is positive
        if alpha_beta.r_squared < 0.5 and alpha_beta.alpha > 0:
            confidence += 0.15
        elif alpha_beta.r_squared < 0.7:
            confidence += 0.10
        else:
            confidence += 0.05
        
        # Alpha sign consistency (0-15%)
        if alpha_beta.alpha > 0 and alpha_beta.information_ratio > 0:
            confidence += 0.15
        elif alpha_beta.alpha > 0:
            confidence += 0.10
        
        return min(confidence, 1.0)
    
    def _compute_robustness_score(
        self,
        signal_quality: SignalQuality,
        stress_tests: List[StressTestResult],
        regime_performance: Dict[str, RegimePerformance]
    ) -> float:
        """
        Compute overall strategy robustness score.
        
        Considers:
        - Signal quality consistency
        - Performance across stress scenarios
        - Performance across regimes
        """
        score = 0.0
        
        # Signal quality (0-25%)
        if signal_quality.information_coefficient > 0.05:
            score += 0.25
        elif signal_quality.information_coefficient > 0.02:
            score += 0.15
        elif signal_quality.information_coefficient > 0:
            score += 0.10
        
        # Stress test performance (0-35%)
        if stress_tests:
            # Count how many stress tests strategy outperformed
            outperformed = sum(1 for st in stress_tests if st.outperformance > 0)
            pct_outperformed = outperformed / len(stress_tests)
            score += 0.35 * pct_outperformed
        
        # Regime consistency (0-40%)
        if regime_performance:
            # Check if profitable in each regime
            regimes_profitable = sum(
                1 for rp in regime_performance.values()
                if rp.total_return > 0
            )
            pct_profitable = regimes_profitable / len(regime_performance)
            score += 0.40 * pct_profitable
        
        return min(score, 1.0)


# =============================================================================
# REPORT FORMATTING
# =============================================================================

def format_risk_analytics_report(report: RiskAnalyticsReport) -> str:
    """
    Format the risk analytics report as a professional text report.
    
    Parameters
    ----------
    report : RiskAnalyticsReport
        Complete risk analytics report
        
    Returns
    -------
    str
        Formatted report string
    """
    lines = []
    
    # Header
    lines.append("=" * 70)
    lines.append("ADVANCED RISK ANALYTICS & PERFORMANCE ATTRIBUTION REPORT")
    lines.append("=" * 70)
    lines.append(f"Symbol: {report.symbol}")
    lines.append(f"Analysis Date: {report.analysis_date.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Period: {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}")
    lines.append(f"Trading Days: {report.trading_days:,}")
    lines.append("")
    
    # Executive Summary
    lines.append("-" * 70)
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Performance Grade:      {report.risk_adjusted_grade.value}")
    lines.append(f"  Overall Risk Level:     {report.overall_risk_level.value}")
    lines.append(f"  Alpha Confidence:       {report.alpha_confidence:.1%}")
    lines.append(f"  Strategy Robustness:    {report.strategy_robustness:.1%}")
    lines.append("")
    
    # Core Metrics
    lines.append("-" * 70)
    lines.append("CORE PERFORMANCE METRICS")
    lines.append("-" * 70)
    lines.append(f"  Total Return:           {report.total_return:+.2%}")
    lines.append(f"  CAGR:                   {report.cagr:+.2%}")
    lines.append(f"  Sharpe Ratio:           {report.sharpe_ratio:.3f}")
    lines.append(f"  Maximum Drawdown:       {report.max_drawdown:.2%}")
    lines.append("")
    
    # Alpha/Beta Decomposition
    ab = report.alpha_beta
    lines.append("-" * 70)
    lines.append("PERFORMANCE ATTRIBUTION (CAPM)")
    lines.append("-" * 70)
    lines.append(f"  Jensen's Alpha:         {ab.alpha:+.2%} {'*' if ab.alpha_significant else ''}")
    lines.append(f"  Beta:                   {ab.beta:.3f}")
    lines.append(f"  R-Squared:              {ab.r_squared:.3f}")
    lines.append(f"  Alpha T-Stat:           {ab.alpha_t_stat:.2f}")
    lines.append(f"  Alpha P-Value:          {ab.alpha_p_value:.4f}")
    lines.append("")
    lines.append("  Return Decomposition:")
    lines.append(f"    Systematic (Beta):    {ab.systematic_return:+.2%}")
    lines.append(f"    Idiosyncratic:        {ab.idiosyncratic_return:+.2%}")
    lines.append(f"    Skill Contribution:   {ab.skill_contribution:.1%}")
    lines.append("")
    lines.append(f"  Tracking Error:         {ab.tracking_error:.2%}")
    lines.append(f"  Information Ratio:      {ab.information_ratio:.3f}")
    if ab.alpha_significant:
        lines.append("  ✓ Alpha is statistically significant at 5% level")
    else:
        lines.append("  ⚠ Alpha is NOT statistically significant")
    lines.append("")
    
    # Regime-Conditional Performance
    lines.append("-" * 70)
    lines.append("REGIME-CONDITIONAL PERFORMANCE")
    lines.append("-" * 70)
    lines.append("  Market Regime Analysis:")
    lines.append("  " + "-" * 66)
    lines.append(f"  {'Regime':<12} {'Days':>8} {'% Time':>8} {'Return':>10} {'Sharpe':>8} {'Max DD':>8}")
    lines.append("  " + "-" * 66)
    for regime, perf in report.regime_performance.items():
        lines.append(
            f"  {regime:<12} {perf.days:>8,} {perf.pct_time:>7.1%} "
            f"{perf.total_return:>+9.2%} {perf.sharpe:>8.2f} {perf.max_drawdown:>7.2%}"
        )
    lines.append("")
    
    lines.append("  Volatility Regime Analysis:")
    lines.append("  " + "-" * 66)
    lines.append(f"  {'Vol Regime':<12} {'Days':>8} {'% Time':>8} {'Return':>10} {'Sharpe':>8} {'Max DD':>8}")
    lines.append("  " + "-" * 66)
    for regime, perf in report.volatility_regime_performance.items():
        lines.append(
            f"  {regime:<12} {perf.days:>8,} {perf.pct_time:>7.1%} "
            f"{perf.total_return:>+9.2%} {perf.sharpe:>8.2f} {perf.max_drawdown:>7.2%}"
        )
    lines.append("")
    
    # Signal Quality
    sq = report.signal_quality
    lines.append("-" * 70)
    lines.append("SIGNAL QUALITY ANALYSIS")
    lines.append("-" * 70)
    lines.append(f"  Information Coefficient: {sq.information_coefficient:.4f}")
    lines.append(f"  IC T-Statistic:          {sq.ic_t_stat:.2f}")
    lines.append(f"  Signal Quality Grade:    {sq.signal_quality_grade}")
    lines.append("")
    lines.append(f"  Hit Rate (Long):         {sq.hit_rate_long:.1%}")
    lines.append(f"  Hit Rate (Flat):         {sq.hit_rate_flat:.1%}")
    lines.append(f"  Signal Persistence:      {sq.signal_persistence:.1f} days")
    lines.append(f"  Annual Turnover:         {sq.turnover:.1%}")
    if sq.ic_significant:
        lines.append("  ✓ IC is statistically significant")
    lines.append("")
    
    # Stress Testing
    if report.stress_tests:
        lines.append("-" * 70)
        lines.append("HISTORICAL STRESS TEST RESULTS")
        lines.append("-" * 70)
        lines.append(f"  {'Scenario':<25} {'Benchmark':>10} {'Strategy':>10} {'Alpha':>10} {'Protected'}")
        lines.append("  " + "-" * 66)
        for st in report.stress_tests:
            protected = "✓" if st.protected_downside else ""
            lines.append(
                f"  {st.scenario_name:<25} {st.benchmark_return:>+9.2%} "
                f"{st.strategy_return:>+9.2%} {st.outperformance:>+9.2%} {protected:>8}"
            )
        lines.append("")
    
    # Drawdown Analysis
    if report.major_drawdowns:
        lines.append("-" * 70)
        lines.append("MAJOR DRAWDOWN ANALYSIS")
        lines.append("-" * 70)
        lines.append(f"  Total Drawdown Periods:   {report.underwater_periods}")
        lines.append(f"  Avg Drawdown Duration:    {report.avg_drawdown_duration:.0f} days")
        lines.append(f"  Avg Recovery Time:        {report.avg_recovery_time:.0f} days")
        lines.append("")
        lines.append("  Top 5 Drawdowns:")
        lines.append("  " + "-" * 66)
        for i, dd in enumerate(report.major_drawdowns[:5], 1):
            status = "Recovered" if dd.recovered else "ONGOING"
            lines.append(
                f"  {i}. {dd.depth:+.2%} | {dd.start_date.strftime('%Y-%m-%d')} to "
                f"{dd.trough_date.strftime('%Y-%m-%d')} | {dd.duration_days} days | {status}"
            )
        lines.append("")
    
    # Footer
    lines.append("=" * 70)
    lines.append("End of Risk Analytics Report")
    lines.append("=" * 70)
    
    return "\n".join(lines)


# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================

if __name__ == "__main__":
    print("Risk Analytics Module loaded successfully.")
    print("Use RiskAnalyticsEngine.analyze() to generate reports.")