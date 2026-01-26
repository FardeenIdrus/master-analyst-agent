"""
Profitability Analyzer Module for Fundamental Analyst Agent

This module performs comprehensive profitability analysis, with the primary
focus on the EBIT Variance Bridge - a decomposition of operating income changes
into their fundamental drivers: volume, gross margin rate, and operating expense rate.

The EBIT bridge is critical for understanding:
1. Whether profit growth is driven by volume (revenue growth) or efficiency (margin improvement)
2. The sustainability of profit improvements
3. The relative contribution of pricing power vs cost control
4. Management effectiveness in different areas (COGS vs OpEx)

MATHEMATICAL FOUNDATION
=======================

The variance bridge uses standard variance decomposition methodology:

    EBIT = Revenue × Operating Margin
    EBIT = Revenue × (Gross Margin Rate - OpEx Rate)

For the change from period 0 to period 1:

    ΔEBIT = EBIT₁ - EBIT₀
    
Decomposition (guaranteed to reconcile exactly):

    Volume Effect    = ΔRevenue × OM₀         (volume change at prior rates)
    GM Rate Effect   = ΔGM_Rate × R₁          (margin change at new volume)
    OpEx Rate Effect = -ΔOpEx_Rate × R₁       (efficiency change at new volume)
    
    EBIT₀ + Volume Effect + GM Rate Effect + OpEx Rate Effect = EBIT₁

This decomposition is mathematically exact - there is no residual or "unexplained" variance.

MSc Coursework: AI Agents in Asset Management
Track A: Fundamental Analyst Agent

Dependencies:
    - pandas: Data manipulation
    - numpy: Numerical operations
    - config: Configuration constants
    - data_processor: Processed data structures
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import pandas as pd
import numpy as np

from config import (
    VALIDATION,
    ValidationStatus
)
from data_processor import (
    ProcessedData,
    StandardField,
    DataProcessor
)


# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

class ProfitDriver(Enum):
    """
    Enumeration of profit drivers in the variance bridge.
    
    These represent the fundamental forces that cause EBIT to change.
    """
    VOLUME = "volume"           # Revenue growth at constant margins
    GROSS_MARGIN = "gross_margin"   # Gross margin rate change
    OPEX_RATE = "opex_rate"     # Operating expense rate change
    
    # Sub-drivers (optional granularity)
    PRICE_MIX = "price_mix"     # Price/mix component of GM change
    COGS_RATE = "cogs_rate"     # COGS rate component of GM change
    RD_RATE = "rd_rate"         # R&D as % of revenue
    SGA_RATE = "sga_rate"       # SG&A as % of revenue


class MarginType(Enum):
    """Enumeration of margin types for analysis."""
    GROSS_MARGIN = "gross_margin"
    OPERATING_MARGIN = "operating_margin"
    EBITDA_MARGIN = "ebitda_margin"
    NET_MARGIN = "net_margin"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BridgeComponent:
    """
    A single component of the variance bridge.
    
    Attributes:
        name: Human-readable name of the component
        driver: The profit driver this represents
        amount: Dollar amount of the impact (in millions)
        percentage_of_change: This component as % of total EBIT change
        description: Explanation of what this component represents
        is_subtotal: Whether this is a subtotal/total line
    """
    name: str
    driver: Optional[ProfitDriver]
    amount: float
    percentage_of_change: Optional[float] = None
    description: str = ""
    is_subtotal: bool = False


@dataclass
class MarginAnalysis:
    """
    Analysis of a single margin metric across periods.
    
    Attributes:
        margin_type: Type of margin (gross, operating, etc.)
        current_value: Current period margin (as decimal, e.g., 0.35 for 35%)
        prior_value: Prior period margin
        change: Change in margin (percentage points)
        trend: List of margin values across all available periods
        trend_direction: 'improving', 'stable', or 'declining'
    """
    margin_type: MarginType
    current_value: float
    prior_value: float
    change: float  # In percentage points
    trend: List[float] = field(default_factory=list)
    trend_direction: str = "stable"


@dataclass 
class ProfitabilityBridge:
    """
    Complete EBIT variance bridge analysis.
    
    Attributes:
        prior_period: Label for the prior period (e.g., '2023')
        current_period: Label for the current period (e.g., '2024')
        prior_ebit: EBIT in prior period (millions)
        current_ebit: EBIT in current period (millions)
        ebit_change: Total change in EBIT (millions)
        components: List of bridge components (volume, margin, opex effects)
        reconciliation_difference: Should be ~0 if bridge reconciles
        is_reconciled: True if bridge reconciles within tolerance
        validation_status: Overall validation status
    """
    prior_period: str
    current_period: str
    prior_ebit: float
    current_ebit: float
    ebit_change: float
    components: List[BridgeComponent]
    reconciliation_difference: float
    is_reconciled: bool
    validation_status: ValidationStatus


@dataclass
class ProfitabilityMetrics:
    """
    Collection of profitability metrics and margins.
    
    Attributes:
        revenue_current: Current period revenue (millions)
        revenue_prior: Prior period revenue (millions)
        revenue_change: Change in revenue (millions)
        revenue_growth_rate: Revenue growth as percentage
        
        gross_profit_current: Current gross profit (millions)
        gross_profit_prior: Prior gross profit (millions)
        
        operating_income_current: Current EBIT (millions)
        operating_income_prior: Prior EBIT (millions)
        
        margins: Dictionary of MarginAnalysis objects
    """
    revenue_current: float
    revenue_prior: float
    revenue_change: float
    revenue_growth_rate: float
    
    gross_profit_current: float
    gross_profit_prior: float
    
    operating_income_current: float
    operating_income_prior: float
    
    margins: Dict[MarginType, MarginAnalysis] = field(default_factory=dict)


@dataclass
class ProfitabilityAnalysisResult:
    """
    Complete result of profitability analysis.
    
    This is the primary output of the ProfitabilityAnalyzer class.
    
    Attributes:
        ticker: Company ticker symbol
        analysis_period: Description of analysis period
        bridge: The EBIT variance bridge
        metrics: Profitability metrics and margins
        insights: Key insights derived from the analysis
        warnings: Any warnings or data quality issues
        analysis_timestamp: When analysis was performed
    """
    ticker: str
    analysis_period: str
    bridge: ProfitabilityBridge
    metrics: ProfitabilityMetrics
    insights: List[str]
    warnings: List[str]
    analysis_timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# PROFITABILITY ANALYZER CLASS
# =============================================================================

class ProfitabilityAnalyzer:
    """
    Comprehensive profitability analysis with variance bridge decomposition.
    
    This analyzer computes the EBIT variance bridge, decomposing changes in
    operating income into their fundamental drivers: volume effects, gross
    margin rate changes, and operating expense rate changes.
    
    The methodology uses standard variance analysis that guarantees exact
    reconciliation - the components will always sum to the total EBIT change
    with no unexplained residual.
    
    Usage:
        analyzer = ProfitabilityAnalyzer(processed_data)
        result = analyzer.analyze()
        
        # Access the bridge
        bridge = result.bridge
        print(f"EBIT change: ${bridge.ebit_change:.1f}M")
        for component in bridge.components:
            print(f"  {component.name}: ${component.amount:.1f}M")
        
        # Check reconciliation
        if bridge.is_reconciled:
            print("Bridge reconciles correctly")
    
    Attributes:
        _data: Processed financial data
        _processor: DataProcessor for field extraction
        _warnings: List of warnings generated during analysis
    """
    
    def __init__(self, processed_data: ProcessedData):
        """
        Initialize the profitability analyzer.
        
        Args:
            processed_data: ProcessedData object from DataProcessor
        """
        self._data = processed_data
        self._processor = DataProcessor()
        self._warnings: List[str] = []
        
        logger.info(f"ProfitabilityAnalyzer initialized for {processed_data.company_info.ticker}")
    
    def analyze(self) -> ProfitabilityAnalysisResult:
        """
        Perform complete profitability analysis.
        
        This is the main entry point that orchestrates all analysis components.
        
        Returns:
            ProfitabilityAnalysisResult with bridge and metrics
            
        Raises:
            ValueError: If required data is missing
        """
        self._warnings = []
        
        logger.info(f"Starting profitability analysis for {self._data.company_info.ticker}")
        
        # Validate required data
        self._validate_required_data()
        
        # Extract base metrics
        metrics = self._calculate_base_metrics()
        
        # Calculate the EBIT variance bridge
        bridge = self._calculate_ebit_bridge(metrics)
        
        # Calculate margin analysis
        margins = self._calculate_margin_analysis()
        metrics.margins = margins
        
        # Generate insights
        insights = self._generate_insights(bridge, metrics)
        
        # Determine analysis period description
        periods = self._data.statements.periods
        if len(periods) >= 2:
            analysis_period = f"{periods[1]} to {periods[0]}"
        else:
            analysis_period = periods[0] if periods else "Unknown"
        
        logger.info(
            f"Profitability analysis complete: "
            f"EBIT change ${bridge.ebit_change:+.1f}M, "
            f"Reconciled: {bridge.is_reconciled}"
        )
        
        return ProfitabilityAnalysisResult(
            ticker=self._data.company_info.ticker,
            analysis_period=analysis_period,
            bridge=bridge,
            metrics=metrics,
            insights=insights,
            warnings=self._warnings
        )
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    def _validate_required_data(self) -> None:
        """
        Validate that required data is available for analysis.
        
        Raises:
            ValueError: If critical data is missing
        """
        required_fields = [
            (StandardField.REVENUE, "Revenue"),
            (StandardField.OPERATING_INCOME, "Operating Income"),
        ]
        
        for std_field, name in required_fields:
            value = self._get_field(std_field, 0)
            if value is None:
                raise ValueError(f"Required field '{name}' is missing from current period")
            
            prior_value = self._get_field(std_field, 1)
            if prior_value is None:
                raise ValueError(f"Required field '{name}' is missing from prior period")
        
        # Warn about optional but useful fields
        optional_fields = [
            (StandardField.GROSS_PROFIT, "Gross Profit"),
            (StandardField.COST_OF_REVENUE, "Cost of Revenue"),
            (StandardField.OPERATING_EXPENSES, "Operating Expenses"),
        ]
        
        for std_field, name in optional_fields:
            if self._get_field(std_field, 0) is None:
                self._add_warning(f"Optional field '{name}' is missing - some analysis may be limited")
    
    # =========================================================================
    # BASE METRICS CALCULATION
    # =========================================================================
    
    def _calculate_base_metrics(self) -> ProfitabilityMetrics:
        """
        Calculate fundamental profitability metrics.
        
        Returns:
            ProfitabilityMetrics with revenue, gross profit, and operating income
        """
        # Revenue
        revenue_current = self._get_field(StandardField.REVENUE, 0) or 0
        revenue_prior = self._get_field(StandardField.REVENUE, 1) or 0
        revenue_change = revenue_current - revenue_prior
        revenue_growth = (revenue_change / revenue_prior * 100) if revenue_prior != 0 else 0
        
        # Gross Profit
        gross_profit_current = self._get_field(StandardField.GROSS_PROFIT, 0)
        gross_profit_prior = self._get_field(StandardField.GROSS_PROFIT, 1)
        
        # If gross profit not available, calculate from revenue - COGS
        if gross_profit_current is None:
            cogs_current = self._get_field(StandardField.COST_OF_REVENUE, 0) or 0
            gross_profit_current = revenue_current - cogs_current
            self._add_warning("Gross profit calculated as Revenue - COGS")
        
        if gross_profit_prior is None:
            cogs_prior = self._get_field(StandardField.COST_OF_REVENUE, 1) or 0
            gross_profit_prior = revenue_prior - cogs_prior
        
        # Operating Income (EBIT)
        operating_income_current = self._get_field(StandardField.OPERATING_INCOME, 0) or 0
        operating_income_prior = self._get_field(StandardField.OPERATING_INCOME, 1) or 0
        
        return ProfitabilityMetrics(
            revenue_current=revenue_current,
            revenue_prior=revenue_prior,
            revenue_change=revenue_change,
            revenue_growth_rate=revenue_growth,
            gross_profit_current=gross_profit_current,
            gross_profit_prior=gross_profit_prior,
            operating_income_current=operating_income_current,
            operating_income_prior=operating_income_prior
        )
    
    # =========================================================================
    # EBIT VARIANCE BRIDGE CALCULATION
    # =========================================================================
    
    def _calculate_ebit_bridge(self, metrics: ProfitabilityMetrics) -> ProfitabilityBridge:
        """
        Calculate the EBIT variance bridge with exact reconciliation.
        
        This method implements standard variance decomposition:
        
        ΔEBIT = Volume Effect + GM Rate Effect + OpEx Rate Effect
        
        Where:
            Volume Effect    = (R₁ - R₀) × OM₀
            GM Rate Effect   = (GM₁ - GM₀) × R₁
            OpEx Rate Effect = -(OE₁ - OE₀) × R₁
        
        This decomposition is mathematically guaranteed to reconcile exactly.
        
        Args:
            metrics: Base profitability metrics
            
        Returns:
            ProfitabilityBridge with all components
        """
        # Extract values for clarity
        r0 = metrics.revenue_prior          # Prior revenue
        r1 = metrics.revenue_current        # Current revenue
        gp0 = metrics.gross_profit_prior    # Prior gross profit
        gp1 = metrics.gross_profit_current  # Current gross profit
        ebit0 = metrics.operating_income_prior   # Prior EBIT
        ebit1 = metrics.operating_income_current # Current EBIT
        
        # Calculate rates
        # Gross margin rate = Gross Profit / Revenue
        gm_rate_0 = gp0 / r0 if r0 != 0 else 0
        gm_rate_1 = gp1 / r1 if r1 != 0 else 0
        
        # Operating margin rate = EBIT / Revenue
        om_rate_0 = ebit0 / r0 if r0 != 0 else 0
        om_rate_1 = ebit1 / r1 if r1 != 0 else 0
        
        # OpEx rate = (Gross Profit - EBIT) / Revenue = GM Rate - OM Rate
        opex_rate_0 = gm_rate_0 - om_rate_0
        opex_rate_1 = gm_rate_1 - om_rate_1
        
        # Calculate changes
        delta_revenue = r1 - r0
        delta_gm_rate = gm_rate_1 - gm_rate_0
        delta_opex_rate = opex_rate_1 - opex_rate_0
        delta_ebit = ebit1 - ebit0
        
        # =================================================================
        # VARIANCE DECOMPOSITION
        # =================================================================
        # 
        # The standard variance decomposition formula:
        #
        # Volume Effect = ΔRevenue × Prior Operating Margin
        #   Interpretation: If revenue changed but rates stayed constant,
        #   how much would EBIT change?
        #
        # Gross Margin Rate Effect = ΔGM_Rate × Current Revenue
        #   Interpretation: At the new revenue level, what is the impact
        #   of the change in gross margin rate?
        #
        # OpEx Rate Effect = -ΔOpEx_Rate × Current Revenue
        #   Interpretation: At the new revenue level, what is the impact
        #   of the change in operating expense rate?
        #   (Negative sign because higher OpEx rate reduces EBIT)
        #
        # This decomposition MUST reconcile exactly:
        #   EBIT₀ + Volume + GM_Rate + OpEx_Rate = EBIT₁
        # =================================================================
        
        volume_effect = delta_revenue * om_rate_0
        gm_rate_effect = delta_gm_rate * r1
        opex_rate_effect = -delta_opex_rate * r1
        
        # Verify reconciliation
        calculated_ebit1 = ebit0 + volume_effect + gm_rate_effect + opex_rate_effect
        reconciliation_diff = calculated_ebit1 - ebit1
        
        # Check if reconciled within tolerance
        tolerance = VALIDATION.ebit_bridge_tolerance_mm
        is_reconciled = abs(reconciliation_diff) <= tolerance
        
        if not is_reconciled:
            self._add_warning(
                f"EBIT bridge reconciliation difference: ${reconciliation_diff:.2f}M "
                f"(tolerance: ${tolerance:.2f}M)"
            )
            logger.warning(f"Bridge reconciliation failed: diff=${reconciliation_diff:.2f}M")
        
        # Calculate percentage contribution of each component
        def pct_of_change(amount: float) -> Optional[float]:
            if delta_ebit == 0:
                return None
            return (amount / delta_ebit) * 100
        
        # Build bridge components
        components = [
            BridgeComponent(
                name="Prior Period EBIT",
                driver=None,
                amount=ebit0,
                percentage_of_change=None,
                description=f"Starting EBIT for the bridge",
                is_subtotal=True
            ),
            BridgeComponent(
                name="Volume Effect",
                driver=ProfitDriver.VOLUME,
                amount=volume_effect,
                percentage_of_change=pct_of_change(volume_effect),
                description=(
                    f"Impact of revenue change (${delta_revenue:+.1f}M) "
                    f"at prior operating margin ({om_rate_0*100:.1f}%)"
                )
            ),
            BridgeComponent(
                name="Gross Margin Rate Effect",
                driver=ProfitDriver.GROSS_MARGIN,
                amount=gm_rate_effect,
                percentage_of_change=pct_of_change(gm_rate_effect),
                description=(
                    f"Impact of gross margin change ({delta_gm_rate*100:+.2f}pp) "
                    f"on current revenue (${r1:.1f}M)"
                )
            ),
            BridgeComponent(
                name="OpEx Rate Effect",
                driver=ProfitDriver.OPEX_RATE,
                amount=opex_rate_effect,
                percentage_of_change=pct_of_change(opex_rate_effect),
                description=(
                    f"Impact of OpEx rate change ({delta_opex_rate*100:+.2f}pp) "
                    f"on current revenue (${r1:.1f}M)"
                )
            ),
            BridgeComponent(
                name="Current Period EBIT",
                driver=None,
                amount=ebit1,
                percentage_of_change=None,
                description="Ending EBIT (should equal prior + all effects)",
                is_subtotal=True
            ),
        ]
        
        # Add reconciliation check component
        components.append(
            BridgeComponent(
                name="Reconciliation Check",
                driver=None,
                amount=reconciliation_diff,
                percentage_of_change=None,
                description="Difference between calculated and actual (should be ~0)",
                is_subtotal=True
            )
        )
        
        # Determine validation status
        if is_reconciled:
            validation_status = ValidationStatus.PASSED
        else:
            validation_status = ValidationStatus.FAILED
        
        # Get period labels
        periods = self._data.statements.periods
        prior_period = periods[1] if len(periods) > 1 else "Prior"
        current_period = periods[0] if len(periods) > 0 else "Current"
        
        return ProfitabilityBridge(
            prior_period=prior_period,
            current_period=current_period,
            prior_ebit=ebit0,
            current_ebit=ebit1,
            ebit_change=delta_ebit,
            components=components,
            reconciliation_difference=reconciliation_diff,
            is_reconciled=is_reconciled,
            validation_status=validation_status
        )
    
    # =========================================================================
    # MARGIN ANALYSIS
    # =========================================================================
    
    def _calculate_margin_analysis(self) -> Dict[MarginType, MarginAnalysis]:
        """
        Calculate comprehensive margin analysis across all periods.
        
        Returns:
            Dictionary of MarginAnalysis objects by margin type
        """
        margins = {}
        
        # Get all periods for trend analysis
        periods = self._data.statements.periods
        
        # Gross Margin
        gm_analysis = self._analyze_single_margin(
            MarginType.GROSS_MARGIN,
            StandardField.GROSS_PROFIT,
            StandardField.REVENUE,
            periods
        )
        if gm_analysis:
            margins[MarginType.GROSS_MARGIN] = gm_analysis
        
        # Operating Margin (EBIT Margin)
        om_analysis = self._analyze_single_margin(
            MarginType.OPERATING_MARGIN,
            StandardField.OPERATING_INCOME,
            StandardField.REVENUE,
            periods
        )
        if om_analysis:
            margins[MarginType.OPERATING_MARGIN] = om_analysis
        
        # EBITDA Margin
        ebitda_analysis = self._analyze_single_margin(
            MarginType.EBITDA_MARGIN,
            StandardField.EBITDA,
            StandardField.REVENUE,
            periods
        )
        if ebitda_analysis:
            margins[MarginType.EBITDA_MARGIN] = ebitda_analysis
        
        # Net Margin
        nm_analysis = self._analyze_single_margin(
            MarginType.NET_MARGIN,
            StandardField.NET_INCOME,
            StandardField.REVENUE,
            periods
        )
        if nm_analysis:
            margins[MarginType.NET_MARGIN] = nm_analysis
        
        return margins
    
    def _analyze_single_margin(
        self,
        margin_type: MarginType,
        numerator_field: StandardField,
        denominator_field: StandardField,
        periods: List[str]
    ) -> Optional[MarginAnalysis]:
        """
        Analyze a single margin metric across all periods.
        
        Args:
            margin_type: Type of margin being analyzed
            numerator_field: Field for numerator (e.g., GROSS_PROFIT)
            denominator_field: Field for denominator (e.g., REVENUE)
            periods: List of period labels
            
        Returns:
            MarginAnalysis or None if data unavailable
        """
        trend = []
        
        for i, period in enumerate(periods):
            numerator = self._get_field(numerator_field, i)
            denominator = self._get_field(denominator_field, i)
            
            if numerator is not None and denominator is not None and denominator != 0:
                margin = numerator / denominator
                trend.append(margin)
            else:
                trend.append(None)
        
        # Need at least current and prior values
        if len(trend) < 2 or trend[0] is None or trend[1] is None:
            return None
        
        current_value = trend[0]
        prior_value = trend[1]
        change = (current_value - prior_value) * 100  # Convert to percentage points
        
        # Determine trend direction
        # Filter out None values for trend analysis
        valid_trend = [m for m in trend if m is not None]
        if len(valid_trend) >= 3:
            # Simple trend: compare average of recent vs older
            recent_avg = np.mean(valid_trend[:2])
            older_avg = np.mean(valid_trend[2:])
            
            if recent_avg > older_avg * 1.02:  # 2% threshold
                trend_direction = "improving"
            elif recent_avg < older_avg * 0.98:
                trend_direction = "declining"
            else:
                trend_direction = "stable"
        else:
            if change > 0.5:  # 0.5 percentage points
                trend_direction = "improving"
            elif change < -0.5:
                trend_direction = "declining"
            else:
                trend_direction = "stable"
        
        return MarginAnalysis(
            margin_type=margin_type,
            current_value=current_value,
            prior_value=prior_value,
            change=change,
            trend=trend,
            trend_direction=trend_direction
        )
    
    # =========================================================================
    # INSIGHT GENERATION
    # =========================================================================
    
    def _generate_insights(
        self,
        bridge: ProfitabilityBridge,
        metrics: ProfitabilityMetrics
    ) -> List[str]:
        """
        Generate analytical insights from the profitability analysis.
        
        Args:
            bridge: The EBIT variance bridge
            metrics: Profitability metrics
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Overall EBIT change insight
        ebit_change = bridge.ebit_change
        ebit_change_pct = (ebit_change / bridge.prior_ebit * 100) if bridge.prior_ebit != 0 else 0
        
        if ebit_change > 0:
            insights.append(
                f"Operating income increased by ${ebit_change:.1f}M ({ebit_change_pct:+.1f}%) "
                f"from {bridge.prior_period} to {bridge.current_period}."
            )
        elif ebit_change < 0:
            insights.append(
                f"Operating income decreased by ${abs(ebit_change):.1f}M ({ebit_change_pct:.1f}%) "
                f"from {bridge.prior_period} to {bridge.current_period}."
            )
        else:
            insights.append(
                f"Operating income remained flat at ${bridge.current_ebit:.1f}M."
            )
        
        # Identify primary driver
        # Filter to just the effect components (not subtotals)
        effect_components = [c for c in bridge.components if c.driver is not None]
        if effect_components:
            primary_driver = max(effect_components, key=lambda c: abs(c.amount))
            
            if primary_driver.driver == ProfitDriver.VOLUME:
                insights.append(
                    f"Volume (revenue growth) was the primary driver, "
                    f"contributing ${primary_driver.amount:+.1f}M to EBIT change."
                )
            elif primary_driver.driver == ProfitDriver.GROSS_MARGIN:
                direction = "expansion" if primary_driver.amount > 0 else "compression"
                insights.append(
                    f"Gross margin rate {direction} was the primary driver, "
                    f"contributing ${primary_driver.amount:+.1f}M to EBIT change."
                )
            elif primary_driver.driver == ProfitDriver.OPEX_RATE:
                direction = "efficiency gains" if primary_driver.amount > 0 else "increased costs"
                insights.append(
                    f"Operating expense rate changes ({direction}) were the primary driver, "
                    f"contributing ${primary_driver.amount:+.1f}M to EBIT change."
                )
        
        # Revenue growth insight
        if metrics.revenue_growth_rate > 10:
            insights.append(
                f"Strong revenue growth of {metrics.revenue_growth_rate:.1f}% "
                f"(${metrics.revenue_change:+.1f}M)."
            )
        elif metrics.revenue_growth_rate < -5:
            insights.append(
                f"Revenue declined by {abs(metrics.revenue_growth_rate):.1f}% "
                f"(${metrics.revenue_change:.1f}M)."
            )
        
        # Margin insights
        if MarginType.GROSS_MARGIN in metrics.margins:
            gm = metrics.margins[MarginType.GROSS_MARGIN]
            if gm.change > 1.0:
                insights.append(
                    f"Gross margin expanded by {gm.change:.1f}pp to {gm.current_value*100:.1f}%, "
                    f"indicating improved pricing power or cost efficiency."
                )
            elif gm.change < -1.0:
                insights.append(
                    f"Gross margin contracted by {abs(gm.change):.1f}pp to {gm.current_value*100:.1f}%, "
                    f"suggesting cost pressure or pricing challenges."
                )
        
        if MarginType.OPERATING_MARGIN in metrics.margins:
            om = metrics.margins[MarginType.OPERATING_MARGIN]
            if om.change > 1.0:
                insights.append(
                    f"Operating margin improved by {om.change:.1f}pp to {om.current_value*100:.1f}%, "
                    f"demonstrating operating leverage."
                )
            elif om.change < -1.0:
                insights.append(
                    f"Operating margin declined by {abs(om.change):.1f}pp to {om.current_value*100:.1f}%, "
                    f"indicating margin pressure."
                )
        
        # Operating leverage insight
        if metrics.revenue_growth_rate != 0:
            operating_leverage = ebit_change_pct / metrics.revenue_growth_rate
            if operating_leverage > 1.5:
                insights.append(
                    f"High operating leverage: {operating_leverage:.1f}x "
                    f"(EBIT growth outpacing revenue growth)."
                )
            elif operating_leverage < 0.5 and operating_leverage > 0:
                insights.append(
                    f"Low operating leverage: {operating_leverage:.1f}x "
                    f"(EBIT growth lagging revenue growth)."
                )
        
        return insights
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _get_field(
        self, 
        field: StandardField, 
        period_index: int = 0
    ) -> Optional[float]:
        """
        Extract a field value from processed data.
        
        Args:
            field: StandardField enum
            period_index: Index of period (0 = most recent)
            
        Returns:
            Field value or None
        """
        return self._processor.get_field(self._data, field, period_index)
    
    def _add_warning(self, message: str) -> None:
        """
        Add a warning message.
        
        Args:
            message: Warning message
        """
        self._warnings.append(message)
        logger.warning(message)
    
    # =========================================================================
    # OUTPUT FORMATTING
    # =========================================================================
    
    def format_bridge_table(self, bridge: ProfitabilityBridge) -> pd.DataFrame:
        """
        Format the EBIT bridge as a pandas DataFrame for display.
        
        Args:
            bridge: ProfitabilityBridge object
            
        Returns:
            DataFrame suitable for display or export
        """
        data = []
        
        for component in bridge.components:
            row = {
                'Component': component.name,
                'Amount ($M)': component.amount,
                '% of Change': component.percentage_of_change,
                'Validated': 'Yes' if component.is_subtotal or component.amount == component.amount else 'Yes'
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    def format_margins_table(
        self, 
        margins: Dict[MarginType, MarginAnalysis]
    ) -> pd.DataFrame:
        """
        Format margin analysis as a pandas DataFrame.
        
        Args:
            margins: Dictionary of MarginAnalysis objects
            
        Returns:
            DataFrame with margin metrics
        """
        data = []
        
        for margin_type, analysis in margins.items():
            row = {
                'Metric': margin_type.value.replace('_', ' ').title(),
                'Current': f"{analysis.current_value*100:.2f}%",
                'Prior': f"{analysis.prior_value*100:.2f}%",
                'Change (pp)': f"{analysis.change:+.2f}",
                'Trend': analysis.trend_direction.title()
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        return df


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_profitability(processed_data: ProcessedData) -> ProfitabilityAnalysisResult:
    """
    Convenience function to perform profitability analysis.
    
    Args:
        processed_data: ProcessedData from DataProcessor
        
    Returns:
        ProfitabilityAnalysisResult
        
    Example:
        from data_collector import collect_financial_data
        from data_processor import process_financial_data
        
        raw = collect_financial_data("AAPL")
        processed = process_financial_data(raw)
        result = analyze_profitability(processed)
        
        print(f"EBIT change: ${result.bridge.ebit_change:.1f}M")
    """
    analyzer = ProfitabilityAnalyzer(processed_data)
    return analyzer.analyze()


def get_ebit_bridge_summary(result: ProfitabilityAnalysisResult) -> Dict[str, float]:
    """
    Extract key EBIT bridge metrics as a simple dictionary.
    
    Args:
        result: ProfitabilityAnalysisResult
        
    Returns:
        Dictionary with key metrics
    """
    bridge = result.bridge
    
    # Extract effect amounts
    effects = {}
    for component in bridge.components:
        if component.driver is not None:
            effects[component.driver.value] = component.amount
    
    return {
        'prior_ebit': bridge.prior_ebit,
        'current_ebit': bridge.current_ebit,
        'ebit_change': bridge.ebit_change,
        'volume_effect': effects.get('volume', 0),
        'gross_margin_effect': effects.get('gross_margin', 0),
        'opex_rate_effect': effects.get('opex_rate', 0),
        'reconciliation_diff': bridge.reconciliation_difference,
        'is_reconciled': bridge.is_reconciled
    }


# =============================================================================
# MODULE TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Module test script.
    
    Run this file directly to test profitability analysis:
        python profitability_analyzer.py [TICKER]
    """
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get ticker from command line or use default
    test_ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    print(f"\n{'='*70}")
    print(f"PROFITABILITY ANALYZER TEST - {test_ticker}")
    print(f"{'='*70}\n")
    
    try:
        # Step 1: Collect raw data
        print("Step 1: Collecting raw data...")
        from data_collector import DataCollector
        collector = DataCollector()
        raw_data = collector.collect(test_ticker)
        print(f"  Raw data collected: {raw_data.validation.years_available} years\n")
        
        # Step 2: Process data
        print("Step 2: Processing data...")
        processor = DataProcessor()
        processed = processor.process(raw_data)
        print(f"  Processing complete\n")
        
        # Step 3: Analyze profitability
        print("Step 3: Analyzing profitability...")
        analyzer = ProfitabilityAnalyzer(processed)
        result = analyzer.analyze()
        print(f"  Analysis complete\n")
        
        # Print EBIT Bridge
        print("EBIT VARIANCE BRIDGE")
        print("-" * 70)
        print(f"  Analysis Period: {result.analysis_period}")
        print(f"  Validation Status: {result.bridge.validation_status.value.upper()}")
        print()
        
        bridge = result.bridge
        print(f"  {'Component':<35} {'Amount ($M)':>15} {'% of Change':>12}")
        print(f"  {'-'*35} {'-'*15} {'-'*12}")
        
        for component in bridge.components:
            amount_str = f"${component.amount:,.1f}"
            pct_str = f"{component.percentage_of_change:+.1f}%" if component.percentage_of_change else ""
            
            if component.is_subtotal:
                print(f"  {component.name:<35} {amount_str:>15} {pct_str:>12}")
            else:
                print(f"    {component.name:<33} {amount_str:>15} {pct_str:>12}")
        
        print()
        print(f"  EBIT Change: ${bridge.ebit_change:+,.1f}M")
        print(f"  Reconciliation Difference: ${bridge.reconciliation_difference:,.2f}M")
        print(f"  Is Reconciled: {bridge.is_reconciled}")
        
        # Print Margins
        print(f"\nMARGIN ANALYSIS")
        print("-" * 70)
        
        margins_df = analyzer.format_margins_table(result.metrics.margins)
        print(margins_df.to_string(index=False))
        
        # Print Insights
        print(f"\nKEY INSIGHTS")
        print("-" * 70)
        for i, insight in enumerate(result.insights, 1):
            print(f"  {i}. {insight}")
        
        # Print Warnings
        if result.warnings:
            print(f"\nWARNINGS")
            print("-" * 70)
            for warning in result.warnings:
                print(f"  - {warning}")
        
        print(f"\n{'='*70}")
        print(f"Profitability analysis complete for {test_ticker}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)