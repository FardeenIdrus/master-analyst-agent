"""
Working Capital Analyzer Module for Fundamental Analyst Agent

This module performs comprehensive working capital analysis, measuring the
efficiency of a company's management of its operating assets and liabilities.
Working capital management directly impacts cash flow, profitability, and
liquidity risk.

ANALYTICAL FRAMEWORK
====================

1. WORKING CAPITAL EFFICIENCY METRICS
   - Days Sales Outstanding (DSO): How quickly AR converts to cash
   - Days Inventory Outstanding (DIO): How long inventory sits before sale
   - Days Payable Outstanding (DPO): How long the company takes to pay suppliers

2. CASH CONVERSION CYCLE (CCC)
   - CCC = DSO + DIO - DPO
   - Measures days of cash tied up in operating cycle
   - Lower (or negative) CCC indicates more efficient working capital

3. WORKING CAPITAL INVESTMENT
   - Net Working Capital = Current Assets - Current Liabilities
   - Operating Working Capital = AR + Inventory - AP
   - Working Capital / Revenue ratio (capital intensity)

4. TREND ANALYSIS
   - Year-over-year changes in efficiency metrics
   - Working capital as driver of cash flow
   - Seasonal patterns and anomalies

FORMULAS
========

DSO = (Accounts Receivable / Revenue) * 365
    Interpretation: Days to collect payment from customers

DIO = (Inventory / Cost of Revenue) * 365
    Interpretation: Days inventory sits before being sold

DPO = (Accounts Payable / Cost of Revenue) * 365
    Interpretation: Days to pay suppliers

CCC = DSO + DIO - DPO
    Interpretation: Net days of cash tied up in operations
    - Positive: Cash is tied up (need to finance working capital)
    - Negative: Company is financed by suppliers (favorable)

BENCHMARKS
==========
Efficiency varies by industry. General guidelines:
- DSO: 30-45 days is typical; >90 days concerning
- DIO: Varies widely; compare to industry peers
- DPO: 30-60 days typical; stretching beyond may signal stress
- CCC: Lower is better; negative is excellent

MSc Coursework: AI Agents in Asset Management
Track A: Fundamental Analyst Agent

Author: MSc AI Agents in Asset Management
Version: 1.0.1

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
from typing import Dict, List, Optional, Any
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


# =============================================================================
# MODULE CONFIGURATION
# =============================================================================

logger = logging.getLogger(__name__)

# Default thresholds for efficiency metrics
DEFAULT_DSO_EXCELLENT = 30      # Days - excellent collection
DEFAULT_DSO_GOOD = 45           # Days - good collection
DEFAULT_DSO_AVERAGE = 60        # Days - average collection
DEFAULT_DSO_CONCERNING = 90     # Days - concerning collection

DEFAULT_DIO_EXCELLENT = 30      # Days - excellent turnover
DEFAULT_DIO_GOOD = 60           # Days - good turnover
DEFAULT_DIO_AVERAGE = 90        # Days - average turnover

DEFAULT_DPO_LOW = 20            # Days - paying too quickly
DEFAULT_DPO_OPTIMAL_MIN = 30    # Days - optimal range start
DEFAULT_DPO_OPTIMAL_MAX = 60    # Days - optimal range end
DEFAULT_DPO_STRETCHED = 90      # Days - may indicate stress

# Trend analysis thresholds
DEFAULT_TREND_THRESHOLD = 0.05          # 5% change for trend detection
DEFAULT_DSO_CHANGE_THRESHOLD = 15       # Days - significant DSO change
DEFAULT_DIO_CHANGE_THRESHOLD = 20       # Days - significant DIO change
DEFAULT_DPO_CHANGE_THRESHOLD = 15       # Days - significant DPO change
DEFAULT_CCC_CHANGE_THRESHOLD = 15       # Days - significant CCC change

# Working capital thresholds
DEFAULT_WC_GROWTH_THRESHOLD = 0.20      # 20% WC growth is significant


# =============================================================================
# ENUMERATIONS
# =============================================================================

class EfficiencyRating(Enum):
    """
    Rating for working capital efficiency.
    
    Used to classify the effectiveness of working capital management
    across different metrics (DSO, DIO, DPO).
    """
    EXCELLENT = "Excellent"          # Best-in-class efficiency
    GOOD = "Good"                    # Above average performance
    AVERAGE = "Average"              # Industry typical
    BELOW_AVERAGE = "Below Average"  # Needs improvement
    POOR = "Poor"                    # Significant concern


class TrendDirection(Enum):
    """
    Direction of metric trend over time.
    
    Used to indicate whether working capital metrics are getting
    better or worse over the analysis period.
    """
    IMPROVING = "Improving"
    STABLE = "Stable"
    DETERIORATING = "Deteriorating"


class AlertType(Enum):
    """
    Types of working capital alerts.
    
    Categorizes the different types of working capital concerns
    that can be identified during analysis.
    """
    DSO_HIGH = "High DSO"
    DSO_RISING = "Rising DSO"
    DIO_HIGH = "High DIO"
    DIO_RISING = "Rising DIO"
    DPO_STRETCHED = "Stretched DPO"
    DPO_TOO_LOW = "DPO Too Low"
    CCC_DETERIORATING = "Deteriorating CCC"
    CCC_EXTENDED = "Extended CCC"
    WC_CASH_DRAIN = "Working Capital Cash Drain"
    NEGATIVE_WC = "Negative Working Capital"
    LIQUIDITY_CONCERN = "Liquidity Concern"


class AlertSeverity(Enum):
    """
    Severity levels for working capital alerts.
    
    LOW: Minor concern, worth monitoring
    MEDIUM: Moderate concern, warrants investigation
    HIGH: Significant concern, requires attention
    CRITICAL: Severe concern, immediate action needed
    """
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EfficiencyMetrics:
    """
    A single working capital efficiency metric with full analysis.
    
    Contains the current value, historical trend, rating, and interpretation
    for efficiency metrics like DSO, DIO, and DPO.
    
    Attributes:
        name: Full name of the metric (e.g., 'Days Sales Outstanding')
        abbreviation: Short form (e.g., 'DSO')
        current_value: Current period value in days
        prior_value: Prior period value in days (None if unavailable)
        change: Change from prior period in days (None if unavailable)
        trend: Historical values list (most recent first)
        trend_direction: Whether metric is improving, stable, or deteriorating
        rating: Efficiency rating based on current value
        interpretation: Human-readable explanation of the metric
    """
    name: str
    abbreviation: str
    current_value: float
    prior_value: Optional[float]
    change: Optional[float]
    trend: List[float]
    trend_direction: TrendDirection
    rating: EfficiencyRating
    interpretation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "abbreviation": self.abbreviation,
            "current_value": self.current_value,
            "prior_value": self.prior_value,
            "change": self.change,
            "trend": self.trend,
            "trend_direction": self.trend_direction.value,
            "rating": self.rating.value,
            "interpretation": self.interpretation
        }


@dataclass
class CashConversionCycle:
    """
    Cash Conversion Cycle analysis.
    
    The CCC measures the number of days cash is tied up in the operating
    cycle. It is calculated as: CCC = DSO + DIO - DPO
    
    A negative CCC means the company is financed by suppliers (excellent).
    A positive CCC means cash is tied up in operations (needs financing).
    
    Attributes:
        current_value: Current CCC in days
        prior_value: Prior period CCC (None if unavailable)
        change: Change from prior period (None if unavailable)
        dso_contribution: DSO component value
        dio_contribution: DIO component value
        dpo_contribution: DPO component value (subtracted in formula)
        trend: Historical CCC values (most recent first)
        trend_direction: Whether CCC is improving, stable, or deteriorating
        interpretation: Human-readable explanation of the CCC
    """
    current_value: float
    prior_value: Optional[float]
    change: Optional[float]
    dso_contribution: float
    dio_contribution: float
    dpo_contribution: float
    trend: List[float]
    trend_direction: TrendDirection
    interpretation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "current_value": self.current_value,
            "prior_value": self.prior_value,
            "change": self.change,
            "dso_contribution": self.dso_contribution,
            "dio_contribution": self.dio_contribution,
            "dpo_contribution": self.dpo_contribution,
            "trend": self.trend,
            "trend_direction": self.trend_direction.value,
            "interpretation": self.interpretation
        }


@dataclass
class WorkingCapitalPosition:
    """
    Working capital balance position and liquidity metrics.
    
    Contains the key balance sheet ratios and metrics that describe
    the company's working capital position and liquidity.
    
    Attributes:
        net_working_capital: Current Assets - Current Liabilities (millions)
        operating_working_capital: AR + Inventory - AP (millions)
        wc_to_revenue: Working Capital / Revenue ratio
        current_ratio: Current Assets / Current Liabilities
        quick_ratio: (Current Assets - Inventory) / Current Liabilities
        cash_tied_up: Estimated cash tied up in working capital (millions)
        yoy_change: Year-over-year change in operating WC (millions)
        cash_impact: Description of cash flow impact from WC changes
    """
    net_working_capital: float
    operating_working_capital: float
    wc_to_revenue: float
    current_ratio: float
    quick_ratio: float
    cash_tied_up: float
    yoy_change: float
    cash_impact: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "net_working_capital": self.net_working_capital,
            "operating_working_capital": self.operating_working_capital,
            "wc_to_revenue": self.wc_to_revenue,
            "current_ratio": self.current_ratio,
            "quick_ratio": self.quick_ratio,
            "cash_tied_up": self.cash_tied_up,
            "yoy_change": self.yoy_change,
            "cash_impact": self.cash_impact
        }


@dataclass
class WorkingCapitalComponent:
    """
    Detailed analysis of a single working capital component.
    
    Provides in-depth analysis of AR, Inventory, or AP including
    balance, changes, efficiency metric, and interpretation.
    
    Attributes:
        name: Component name (e.g., 'Accounts Receivable')
        current_balance: Current period balance in millions
        prior_balance: Prior period balance in millions (None if unavailable)
        change: Change in balance in millions (None if unavailable)
        change_percent: Percentage change (None if unavailable)
        days_metric: Related efficiency metric value (DSO, DIO, or DPO)
        as_percent_of_revenue: Balance as percentage of annual revenue
        interpretation: Human-readable analysis of the component
    """
    name: str
    current_balance: float
    prior_balance: Optional[float]
    change: Optional[float]
    change_percent: Optional[float]
    days_metric: float
    as_percent_of_revenue: float
    interpretation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "current_balance": self.current_balance,
            "prior_balance": self.prior_balance,
            "change": self.change,
            "change_percent": self.change_percent,
            "days_metric": self.days_metric,
            "as_percent_of_revenue": self.as_percent_of_revenue,
            "interpretation": self.interpretation
        }


@dataclass
class WorkingCapitalAlert:
    """
    An alert for working capital concerns.
    
    Represents a specific concern identified during working capital
    analysis, including severity, description, and recommendation.
    
    Attributes:
        alert_type: Type of alert from AlertType enum
        severity: Severity level from AlertSeverity enum
        metric: The metric that triggered the alert
        current_value: Current value of the metric
        threshold: The threshold that was exceeded (None if not applicable)
        description: Detailed explanation of the concern
        recommendation: Suggested action to address the concern
    """
    alert_type: AlertType
    severity: AlertSeverity
    metric: str
    current_value: float
    threshold: Optional[float]
    description: str
    recommendation: str
    
    @property
    def message(self) -> str:
        """
        Alias for description - provides backward compatibility.
        
        Returns:
            The alert description string.
        """
        return self.description
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "metric": self.metric,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "description": self.description,
            "message": self.description,  # Include for compatibility
            "recommendation": self.recommendation
        }


@dataclass
class WorkingCapitalAnalysisResult:
    """
    Complete result of working capital analysis.
    
    This is the primary output of the WorkingCapitalAnalyzer class,
    containing all efficiency metrics, cash conversion cycle analysis,
    working capital position, alerts, and insights.
    
    Attributes:
        ticker: Company ticker symbol
        analysis_period: Description of analysis period (e.g., 'FY2024')
        dso: Days Sales Outstanding analysis
        dio: Days Inventory Outstanding analysis
        dpo: Days Payable Outstanding analysis
        cash_conversion_cycle: Complete CCC analysis
        position: Working capital position metrics
        components: Detailed analysis of AR, Inventory, AP
        alerts: List of working capital alerts
        insights: Key insights from analysis
        warnings: Data quality warnings
        analysis_timestamp: When analysis was performed
    """
    ticker: str
    analysis_period: str
    dso: EfficiencyMetrics
    dio: EfficiencyMetrics
    dpo: EfficiencyMetrics
    cash_conversion_cycle: CashConversionCycle
    position: WorkingCapitalPosition
    components: List[WorkingCapitalComponent]
    alerts: List[WorkingCapitalAlert]
    insights: List[str]
    warnings: List[str]
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ticker": self.ticker,
            "analysis_period": self.analysis_period,
            "dso": self.dso.to_dict(),
            "dio": self.dio.to_dict(),
            "dpo": self.dpo.to_dict(),
            "cash_conversion_cycle": self.cash_conversion_cycle.to_dict(),
            "position": self.position.to_dict(),
            "components": [c.to_dict() for c in self.components],
            "alerts": [a.to_dict() for a in self.alerts],
            "insights": self.insights,
            "warnings": self.warnings,
            "analysis_timestamp": self.analysis_timestamp.isoformat()
        }


# =============================================================================
# WORKING CAPITAL ANALYZER CLASS
# =============================================================================

class WorkingCapitalAnalyzer:
    """
    Comprehensive working capital efficiency analyzer.
    
    This analyzer calculates efficiency metrics (DSO, DIO, DPO), the cash
    conversion cycle, and assesses working capital management effectiveness.
    
    The analysis helps identify:
    - Cash tied up in operations
    - Collection efficiency issues
    - Inventory management problems
    - Supplier payment patterns
    - Liquidity risks
    
    Usage:
        analyzer = WorkingCapitalAnalyzer(processed_data)
        result = analyzer.analyze()
        
        # Access key metrics
        print(f"DSO: {result.dso.current_value:.1f} days")
        print(f"CCC: {result.cash_conversion_cycle.current_value:.1f} days")
        
        # Check alerts
        for alert in result.alerts:
            print(f"[{alert.severity.value}] {alert.description}")
    
    Attributes:
        _data: ProcessedData object containing standardized financial data
        _processor: DataProcessor instance for field extraction
        _warnings: List of warning messages generated during analysis
        _alerts: List of working capital alerts identified
    """
    
    def __init__(self, processed_data: ProcessedData) -> None:
        """
        Initialize the WorkingCapitalAnalyzer.
        
        Args:
            processed_data: ProcessedData object from DataProcessor containing
                           standardized financial statements for analysis.
        
        Raises:
            ValueError: If processed_data is None or invalid.
        """
        if processed_data is None:
            raise ValueError("processed_data cannot be None")
        
        self._data = processed_data
        self._processor = DataProcessor()
        self._warnings: List[str] = []
        self._alerts: List[WorkingCapitalAlert] = []
        
        logger.info(f"WorkingCapitalAnalyzer initialized for {processed_data.ticker}")
    
    def analyze(self) -> WorkingCapitalAnalysisResult:
        """
        Perform complete working capital analysis.
        
        Executes all analysis components in sequence:
        1. Calculate efficiency metrics (DSO, DIO, DPO)
        2. Calculate cash conversion cycle
        3. Calculate working capital position
        4. Analyze individual components
        5. Generate alerts
        6. Generate insights
        
        Returns:
            WorkingCapitalAnalysisResult containing complete analysis.
        """
        logger.info(f"Starting working capital analysis for {self._data.ticker}")
        
        # Reset state for fresh analysis
        self._warnings = []
        self._alerts = []
        
        # Get analysis period label
        periods = self._data.statements.periods
        analysis_period = self._format_analysis_period(periods)
        
        # Calculate efficiency metrics
        dso = self._calculate_dso()
        dio = self._calculate_dio()
        dpo = self._calculate_dpo()
        
        # Calculate cash conversion cycle
        ccc = self._calculate_cash_conversion_cycle(dso, dio, dpo)
        
        # Calculate working capital position
        position = self._calculate_wc_position()
        
        # Analyze individual components
        components = self._analyze_components()
        
        # Generate alerts based on analysis
        self._generate_alerts(dso, dio, dpo, ccc, position)
        
        # Generate insights
        insights = self._generate_insights(dso, dio, dpo, ccc, position)
        
        # Build result object
        result = WorkingCapitalAnalysisResult(
            ticker=self._data.ticker,
            analysis_period=analysis_period,
            dso=dso,
            dio=dio,
            dpo=dpo,
            cash_conversion_cycle=ccc,
            position=position,
            components=components,
            alerts=self._alerts.copy(),
            insights=insights,
            warnings=self._warnings.copy()
        )
        
        logger.info(f"Working capital analysis complete for {self._data.ticker}")
        return result
    
    # =========================================================================
    # PERIOD FORMATTING
    # =========================================================================
    
    def _format_analysis_period(self, periods: List[str]) -> str:
        """
        Format the analysis period string for display.
        
        Args:
            periods: List of fiscal period labels from processed data.
                    Periods are strings in format like "2024" or "2024-09-30".
        
        Returns:
            Formatted period string (e.g., "FY2024").
        """
        if not periods:
            return "Current"
        
        # Get the most recent period (first in list)
        period = periods[0]
        
        # Handle different period string formats
        if isinstance(period, str):
            # Period is already a string
            if period.startswith("FY"):
                return period
            elif "-" in period:
                # Date format like "2024-09-30", extract year
                year = period.split("-")[0]
                return f"FY{year}"
            else:
                # Just a year like "2024"
                return f"FY{period}"
        else:
            # Fallback for unexpected types - convert to string
            return f"FY{str(period)}"
    
    # =========================================================================
    # EFFICIENCY METRIC CALCULATIONS
    # =========================================================================
    
    def _calculate_dso(self) -> EfficiencyMetrics:
        """
        Calculate Days Sales Outstanding using trade receivables.
        
        DSO = (Trade Accounts Receivable / Revenue) * 365
        
        For data sources that combine trade and non-trade receivables,
        applies adjustment to estimate trade-only portion.
        """
        trend: List[float] = []
        periods = self._data.statements.periods
        
        for i in range(len(periods)):
            ar = self._get_field(StandardField.ACCOUNTS_RECEIVABLE, i)
            revenue = self._get_field(StandardField.REVENUE, i)
            
            if ar is not None and revenue is not None and revenue > 0:
                raw_dso = (ar / revenue) * 365.0
                
                # If DSO > 50 days, likely includes non-trade receivables
                # Apply adjustment to estimate trade-only (typically ~55% of total)
                if raw_dso > 50:
                    trade_ar = ar * 0.55
                    dso = (trade_ar / revenue) * 365.0
                else:
                    dso = raw_dso
                
                trend.append(dso)
            else:
                break
        
        if not trend:
            return self._create_empty_efficiency_metric(
                "Days Sales Outstanding", "DSO",
                "Insufficient data to calculate DSO"
            )
        
        current_value = trend[0]
        prior_value = trend[1] if len(trend) > 1 else None
        change = current_value - prior_value if prior_value is not None else None
        
        trend_direction = self._assess_trend_direction(trend, lower_is_better=True)
        rating = self._rate_dso(current_value)
        interpretation = self._interpret_dso(current_value, change, rating)
        
        return EfficiencyMetrics(
            name="Days Sales Outstanding",
            abbreviation="DSO",
            current_value=current_value,
            prior_value=prior_value,
            change=change,
            trend=trend,
            trend_direction=trend_direction,
            rating=rating,
            interpretation=interpretation
        )
    
    def _calculate_dio(self) -> EfficiencyMetrics:
        """
        Calculate Days Inventory Outstanding.
        
        DIO = (Inventory / Cost of Revenue) * 365
        
        Measures the average number of days inventory is held before sale.
        Lower DIO indicates faster inventory turnover.
        
        Note: Some companies (services, software) don't carry inventory.
        
        Returns:
            EfficiencyMetrics object containing DIO analysis.
        """
        trend: List[float] = []
        periods = self._data.statements.periods
        
        for i in range(len(periods)):
            inventory = self._get_field(StandardField.INVENTORY, i)
            cogs = self._get_field(StandardField.COST_OF_REVENUE, i)
            
            # Service-based companies may not have inventory
            if inventory is None:
                if i == 0:
                    self._add_warning(
                        "No inventory data available - company may be service-based "
                        "or use an asset-light business model"
                    )
                break
            
            if cogs is not None and cogs > 0:
                dio = (inventory / cogs) * 365.0
                trend.append(dio)
            else:
                break
        
        # Handle companies with no inventory
        if not trend:
            return EfficiencyMetrics(
                name="Days Inventory Outstanding",
                abbreviation="DIO",
                current_value=0.0,
                prior_value=None,
                change=None,
                trend=[],
                trend_direction=TrendDirection.STABLE,
                rating=EfficiencyRating.AVERAGE,
                interpretation=(
                    "Company does not carry significant inventory - "
                    "service-based or asset-light business model"
                )
            )
        
        # Extract current and prior values
        current_value = trend[0]
        prior_value = trend[1] if len(trend) > 1 else None
        change = current_value - prior_value if prior_value is not None else None
        
        # Determine trend direction (lower DIO is better)
        trend_direction = self._assess_trend_direction(trend, lower_is_better=True)
        
        # Rate the metric
        rating = self._rate_dio(current_value)
        
        # Generate interpretation
        interpretation = self._interpret_dio(current_value, change, rating)
        
        return EfficiencyMetrics(
            name="Days Inventory Outstanding",
            abbreviation="DIO",
            current_value=current_value,
            prior_value=prior_value,
            change=change,
            trend=trend,
            trend_direction=trend_direction,
            rating=rating,
            interpretation=interpretation
        )
    
    def _calculate_dpo(self) -> EfficiencyMetrics:
        """
        Calculate Days Payable Outstanding.
        
        DPO = (Accounts Payable / Cost of Revenue) * 365
        
        Measures the average number of days to pay suppliers.
        Higher DPO means the company holds onto cash longer (generally favorable),
        but excessively high DPO may indicate payment difficulties or strained
        supplier relationships.
        
        Returns:
            EfficiencyMetrics object containing DPO analysis.
        """
        trend: List[float] = []
        periods = self._data.statements.periods
        
        for i in range(len(periods)):
            ap = self._get_field(StandardField.ACCOUNTS_PAYABLE, i)
            cogs = self._get_field(StandardField.COST_OF_REVENUE, i)
            
            if ap is not None and cogs is not None and cogs > 0:
                dpo = (ap / cogs) * 365.0
                trend.append(dpo)
            else:
                break
        
        # Handle missing data
        if not trend:
            self._add_warning("Unable to calculate DPO - missing AP or COGS data")
            return self._create_empty_efficiency_metric(
                "Days Payable Outstanding", "DPO",
                "Insufficient data to calculate DPO"
            )
        
        # Extract current and prior values
        current_value = trend[0]
        prior_value = trend[1] if len(trend) > 1 else None
        change = current_value - prior_value if prior_value is not None else None
        
        # For DPO, higher is generally better (within reason)
        trend_direction = self._assess_trend_direction(trend, lower_is_better=False)
        
        # Rate the metric
        rating = self._rate_dpo(current_value)
        
        # Generate interpretation
        interpretation = self._interpret_dpo(current_value, change, rating)
        
        return EfficiencyMetrics(
            name="Days Payable Outstanding",
            abbreviation="DPO",
            current_value=current_value,
            prior_value=prior_value,
            change=change,
            trend=trend,
            trend_direction=trend_direction,
            rating=rating,
            interpretation=interpretation
        )
    
    def _create_empty_efficiency_metric(
        self,
        name: str,
        abbreviation: str,
        interpretation: str
    ) -> EfficiencyMetrics:
        """
        Create an empty efficiency metric when data is unavailable.
        
        Args:
            name: Full name of the metric.
            abbreviation: Short form of the metric.
            interpretation: Explanation of why data is unavailable.
        
        Returns:
            EfficiencyMetrics object with default values.
        """
        return EfficiencyMetrics(
            name=name,
            abbreviation=abbreviation,
            current_value=0.0,
            prior_value=None,
            change=None,
            trend=[],
            trend_direction=TrendDirection.STABLE,
            rating=EfficiencyRating.AVERAGE,
            interpretation=interpretation
        )
    
    # =========================================================================
    # CASH CONVERSION CYCLE
    # =========================================================================
    
    def _calculate_cash_conversion_cycle(
        self,
        dso: EfficiencyMetrics,
        dio: EfficiencyMetrics,
        dpo: EfficiencyMetrics
    ) -> CashConversionCycle:
        """
        Calculate the Cash Conversion Cycle.
        
        CCC = DSO + DIO - DPO
        
        The CCC measures the number of days cash is tied up in the
        operating cycle. A lower (or negative) CCC indicates more
        efficient working capital management.
        
        Negative CCC: Company is financed by suppliers (excellent)
        Positive CCC: Cash tied up in operations (needs financing)
        
        Args:
            dso: Days Sales Outstanding analysis.
            dio: Days Inventory Outstanding analysis.
            dpo: Days Payable Outstanding analysis.
        
        Returns:
            CashConversionCycle object with complete analysis.
        """
        # Calculate current CCC
        current_ccc = dso.current_value + dio.current_value - dpo.current_value
        
        # Calculate prior CCC if data available
        prior_ccc: Optional[float] = None
        change: Optional[float] = None
        
        if dso.prior_value is not None and dpo.prior_value is not None:
            dio_prior = dio.prior_value if dio.prior_value is not None else 0.0
            prior_ccc = dso.prior_value + dio_prior - dpo.prior_value
            change = current_ccc - prior_ccc
        
        # Build trend from component trends
        trend: List[float] = []
        
        # Determine minimum trend length across components
        dso_len = len(dso.trend)
        dio_len = len(dio.trend) if dio.trend else 0
        dpo_len = len(dpo.trend)
        
        # Use minimum of non-zero lengths
        if dio_len == 0:
            # No inventory - CCC = DSO - DPO
            min_len = min(dso_len, dpo_len)
        else:
            min_len = min(dso_len, dio_len, dpo_len)
        
        for i in range(min_len):
            dio_val = dio.trend[i] if i < len(dio.trend) else 0.0
            ccc_val = dso.trend[i] + dio_val - dpo.trend[i]
            trend.append(ccc_val)
        
        # Assess trend direction (lower CCC is better)
        trend_direction = self._assess_trend_direction(trend, lower_is_better=True)
        
        # Generate interpretation
        interpretation = self._interpret_ccc(current_ccc, change, trend_direction)
        
        return CashConversionCycle(
            current_value=current_ccc,
            prior_value=prior_ccc,
            change=change,
            dso_contribution=dso.current_value,
            dio_contribution=dio.current_value,
            dpo_contribution=dpo.current_value,
            trend=trend,
            trend_direction=trend_direction,
            interpretation=interpretation
        )
    
    # =========================================================================
    # WORKING CAPITAL POSITION
    # =========================================================================
    
    def _calculate_wc_position(self) -> WorkingCapitalPosition:
        """
        Calculate working capital position and liquidity ratios.
        
        Computes key balance sheet metrics including:
        - Net Working Capital (Current Assets - Current Liabilities)
        - Operating Working Capital (AR + Inventory - AP)
        - Current Ratio and Quick Ratio
        - Year-over-year changes and cash impact
        
        Returns:
            WorkingCapitalPosition object with all metrics.
        """
        # Get current period values (default to 0 if unavailable)
        current_assets = self._get_field(StandardField.CURRENT_ASSETS, 0) or 0.0
        current_liabilities = self._get_field(StandardField.CURRENT_LIABILITIES, 0) or 0.0
        ar = self._get_field(StandardField.ACCOUNTS_RECEIVABLE, 0) or 0.0
        inventory = self._get_field(StandardField.INVENTORY, 0) or 0.0
        ap = self._get_field(StandardField.ACCOUNTS_PAYABLE, 0) or 0.0
        revenue = self._get_field(StandardField.REVENUE, 0) or 0.0
        
        # Get prior period values for comparison
        ar_prior = self._get_field(StandardField.ACCOUNTS_RECEIVABLE, 1) or 0.0
        inventory_prior = self._get_field(StandardField.INVENTORY, 1) or 0.0
        ap_prior = self._get_field(StandardField.ACCOUNTS_PAYABLE, 1) or 0.0
        
        # Calculate Net Working Capital
        net_wc = current_assets - current_liabilities
        
        # Calculate Operating Working Capital (AR + Inventory - AP)
        operating_wc = ar + inventory - ap
        operating_wc_prior = ar_prior + inventory_prior - ap_prior
        
        # Working capital to revenue ratio
        wc_to_revenue = operating_wc / revenue if revenue > 0 else 0.0
        
        # Current ratio
        current_ratio = (
            current_assets / current_liabilities
            if current_liabilities > 0 else 0.0
        )
        
        # Quick ratio (excludes inventory)
        quick_assets = current_assets - inventory
        quick_ratio = (
            quick_assets / current_liabilities
            if current_liabilities > 0 else 0.0
        )
        
        # Year-over-year change in operating WC
        yoy_change = operating_wc - operating_wc_prior
        
        # Determine cash impact description
        if abs(yoy_change) < 1.0:
            cash_impact = "Working capital essentially unchanged, neutral cash impact"
        elif yoy_change > 0:
            cash_impact = (
                f"Working capital increased ${yoy_change:.1f}M, "
                f"consuming cash from operations"
            )
        else:
            cash_impact = (
                f"Working capital decreased ${abs(yoy_change):.1f}M, "
                f"releasing cash to operations"
            )
        
        return WorkingCapitalPosition(
            net_working_capital=net_wc,
            operating_working_capital=operating_wc,
            wc_to_revenue=wc_to_revenue,
            current_ratio=current_ratio,
            quick_ratio=quick_ratio,
            cash_tied_up=operating_wc,
            yoy_change=yoy_change,
            cash_impact=cash_impact
        )
    
    # =========================================================================
    # COMPONENT ANALYSIS
    # =========================================================================
    
    def _analyze_components(self) -> List[WorkingCapitalComponent]:
        """
        Analyze individual working capital components in detail.
        
        Examines Accounts Receivable, Inventory, and Accounts Payable
        individually with balance changes, efficiency metrics, and
        interpretations.
        
        Returns:
            List of WorkingCapitalComponent objects.
        """
        components: List[WorkingCapitalComponent] = []
        
        # Get revenue and COGS for calculations (avoid division by zero)
        revenue = self._get_field(StandardField.REVENUE, 0)
        if revenue is None or revenue <= 0:
            revenue = 1.0  # Fallback to avoid division by zero
        
        cogs = self._get_field(StandardField.COST_OF_REVENUE, 0)
        if cogs is None or cogs <= 0:
            cogs = 1.0  # Fallback to avoid division by zero
        
        # Analyze Accounts Receivable
        ar_component = self._analyze_ar_component(revenue)
        if ar_component is not None:
            components.append(ar_component)
        
        # Analyze Inventory
        inv_component = self._analyze_inventory_component(revenue, cogs)
        if inv_component is not None:
            components.append(inv_component)
        
        # Analyze Accounts Payable
        ap_component = self._analyze_ap_component(revenue, cogs)
        if ap_component is not None:
            components.append(ap_component)
        
        return components
    
    def _analyze_ar_component(self, revenue: float) -> Optional[WorkingCapitalComponent]:
        """
        Analyze the Accounts Receivable component.
        
        Args:
            revenue: Annual revenue for ratio calculations.
        
        Returns:
            WorkingCapitalComponent for AR, or None if data unavailable.
        """
        ar = self._get_field(StandardField.ACCOUNTS_RECEIVABLE, 0)
        ar_prior = self._get_field(StandardField.ACCOUNTS_RECEIVABLE, 1)
        
        if ar is None:
            return None
        
        # Calculate changes
        change = ar - ar_prior if ar_prior is not None else None
        change_pct = (
            change / ar_prior
            if ar_prior is not None and ar_prior != 0 else None
        )
        
        # Calculate DSO
        dso = (ar / revenue) * 365.0
        
        # Generate interpretation
        interpretation = self._interpret_ar_component(ar, change_pct, dso)
        
        return WorkingCapitalComponent(
            name="Accounts Receivable",
            current_balance=ar,
            prior_balance=ar_prior,
            change=change,
            change_percent=change_pct,
            days_metric=dso,
            as_percent_of_revenue=(ar / revenue) * 100.0,
            interpretation=interpretation
        )
    
    def _analyze_inventory_component(
        self,
        revenue: float,
        cogs: float
    ) -> Optional[WorkingCapitalComponent]:
        """
        Analyze the Inventory component.
        
        Args:
            revenue: Annual revenue for ratio calculations.
            cogs: Cost of goods sold for DIO calculation.
        
        Returns:
            WorkingCapitalComponent for Inventory, or None if data unavailable.
        """
        inventory = self._get_field(StandardField.INVENTORY, 0)
        inventory_prior = self._get_field(StandardField.INVENTORY, 1)
        
        if inventory is None:
            return None
        
        # Calculate changes
        change = inventory - inventory_prior if inventory_prior is not None else None
        change_pct = (
            change / inventory_prior
            if inventory_prior is not None and inventory_prior != 0 else None
        )
        
        # Calculate DIO
        dio = (inventory / cogs) * 365.0
        
        # Generate interpretation
        interpretation = self._interpret_inv_component(inventory, change_pct, dio)
        
        return WorkingCapitalComponent(
            name="Inventory",
            current_balance=inventory,
            prior_balance=inventory_prior,
            change=change,
            change_percent=change_pct,
            days_metric=dio,
            as_percent_of_revenue=(inventory / revenue) * 100.0,
            interpretation=interpretation
        )
    
    def _analyze_ap_component(
        self,
        revenue: float,
        cogs: float
    ) -> Optional[WorkingCapitalComponent]:
        """
        Analyze the Accounts Payable component.
        
        Args:
            revenue: Annual revenue for ratio calculations.
            cogs: Cost of goods sold for DPO calculation.
        
        Returns:
            WorkingCapitalComponent for AP, or None if data unavailable.
        """
        ap = self._get_field(StandardField.ACCOUNTS_PAYABLE, 0)
        ap_prior = self._get_field(StandardField.ACCOUNTS_PAYABLE, 1)
        
        if ap is None:
            return None
        
        # Calculate changes
        change = ap - ap_prior if ap_prior is not None else None
        change_pct = (
            change / ap_prior
            if ap_prior is not None and ap_prior != 0 else None
        )
        
        # Calculate DPO
        dpo = (ap / cogs) * 365.0
        
        # Generate interpretation
        interpretation = self._interpret_ap_component(ap, change_pct, dpo)
        
        return WorkingCapitalComponent(
            name="Accounts Payable",
            current_balance=ap,
            prior_balance=ap_prior,
            change=change,
            change_percent=change_pct,
            days_metric=dpo,
            as_percent_of_revenue=(ap / revenue) * 100.0,
            interpretation=interpretation
        )
    
    # =========================================================================
    # RATING AND INTERPRETATION METHODS
    # =========================================================================
    
    def _assess_trend_direction(
        self,
        trend: List[float],
        lower_is_better: bool = True
    ) -> TrendDirection:
        """
        Assess the direction of a metric trend over time.
        
        Compares recent values to older values to determine if the
        metric is improving, stable, or deteriorating.
        
        Args:
            trend: List of values (most recent first).
            lower_is_better: Whether decreasing values are favorable.
        
        Returns:
            TrendDirection enum value.
        """
        if len(trend) < 2:
            return TrendDirection.STABLE
        
        # Compare recent average to older average
        if len(trend) >= 4:
            recent = float(np.mean(trend[:2]))
            older = float(np.mean(trend[2:4]))
        else:
            recent = trend[0]
            older = trend[-1]
        
        # Calculate percentage change
        if older == 0:
            change_pct = 0.0
        else:
            change_pct = (recent - older) / abs(older)
        
        # Apply threshold for meaningful change
        threshold = DEFAULT_TREND_THRESHOLD
        
        if lower_is_better:
            if change_pct < -threshold:
                return TrendDirection.IMPROVING
            elif change_pct > threshold:
                return TrendDirection.DETERIORATING
        else:
            if change_pct > threshold:
                return TrendDirection.IMPROVING
            elif change_pct < -threshold:
                return TrendDirection.DETERIORATING
        
        return TrendDirection.STABLE
    
    def _rate_dso(self, dso: float) -> EfficiencyRating:
        """
        Rate DSO efficiency.
        
        Args:
            dso: Days Sales Outstanding value.
        
        Returns:
            EfficiencyRating based on DSO value.
        """
        # Use validation thresholds from config
        max_dso = getattr(VALIDATION, 'max_reasonable_dso', DEFAULT_DSO_CONCERNING)
        
        if dso <= DEFAULT_DSO_EXCELLENT:
            return EfficiencyRating.EXCELLENT
        elif dso <= DEFAULT_DSO_GOOD:
            return EfficiencyRating.GOOD
        elif dso <= DEFAULT_DSO_AVERAGE:
            return EfficiencyRating.AVERAGE
        elif dso <= max_dso:
            return EfficiencyRating.BELOW_AVERAGE
        else:
            return EfficiencyRating.POOR
    
    def _rate_dio(self, dio: float) -> EfficiencyRating:
        """
        Rate DIO efficiency.
        
        Args:
            dio: Days Inventory Outstanding value.
        
        Returns:
            EfficiencyRating based on DIO value.
        """
        max_dio = getattr(VALIDATION, 'max_reasonable_dio', 365)
        
        if dio <= DEFAULT_DIO_EXCELLENT:
            return EfficiencyRating.EXCELLENT
        elif dio <= DEFAULT_DIO_GOOD:
            return EfficiencyRating.GOOD
        elif dio <= DEFAULT_DIO_AVERAGE:
            return EfficiencyRating.AVERAGE
        elif dio <= max_dio:
            return EfficiencyRating.BELOW_AVERAGE
        else:
            return EfficiencyRating.POOR
    
    def _rate_dpo(self, dpo: float) -> EfficiencyRating:
        """
        Rate DPO.
        
        For DPO, moderate values are optimal. Too low means not utilizing
        supplier credit; too high may indicate payment stress.
        
        Args:
            dpo: Days Payable Outstanding value.
        
        Returns:
            EfficiencyRating based on DPO value.
        """
        max_dpo = getattr(VALIDATION, 'max_reasonable_dpo', 180)
        
        if DEFAULT_DPO_OPTIMAL_MIN <= dpo <= DEFAULT_DPO_OPTIMAL_MAX:
            return EfficiencyRating.EXCELLENT
        elif DEFAULT_DPO_LOW <= dpo <= DEFAULT_DPO_STRETCHED:
            return EfficiencyRating.GOOD
        elif dpo < DEFAULT_DPO_LOW:
            # Paying too quickly - not using credit terms
            return EfficiencyRating.BELOW_AVERAGE
        elif dpo <= max_dpo:
            return EfficiencyRating.AVERAGE
        else:
            # Stretched payments - may indicate stress
            return EfficiencyRating.POOR
    
    def _interpret_dso(
        self,
        dso: float,
        change: Optional[float],
        rating: EfficiencyRating
    ) -> str:
        """
        Generate human-readable interpretation for DSO.
        
        Args:
            dso: Current DSO value.
            change: Change from prior period.
            rating: Efficiency rating.
        
        Returns:
            Interpretation string.
        """
        base = f"Customers take {dso:.0f} days on average to pay"
        
        # Add quality assessment
        if rating == EfficiencyRating.EXCELLENT:
            quality = "indicating efficient receivables collection"
        elif rating == EfficiencyRating.GOOD:
            quality = "within healthy range for most industries"
        elif rating == EfficiencyRating.AVERAGE:
            quality = "typical but room for improvement"
        elif rating == EfficiencyRating.BELOW_AVERAGE:
            quality = "suggesting collection inefficiencies"
        else:
            quality = "indicating potential collection problems"
        
        # Add trend information
        trend_part = ""
        if change is not None:
            if change > 5:
                trend_part = f"; DSO increased {change:.0f} days YoY (deteriorating)"
            elif change < -5:
                trend_part = f"; DSO improved {abs(change):.0f} days YoY"
        
        return f"{base}, {quality}{trend_part}"
    
    def _interpret_dio(
        self,
        dio: float,
        change: Optional[float],
        rating: EfficiencyRating
    ) -> str:
        """
        Generate human-readable interpretation for DIO.
        
        Args:
            dio: Current DIO value.
            change: Change from prior period.
            rating: Efficiency rating.
        
        Returns:
            Interpretation string.
        """
        base = f"Inventory sits {dio:.0f} days on average before sale"
        
        # Add quality assessment
        if rating == EfficiencyRating.EXCELLENT:
            quality = "indicating excellent inventory turnover"
        elif rating == EfficiencyRating.GOOD:
            quality = "showing healthy inventory management"
        elif rating == EfficiencyRating.AVERAGE:
            quality = "within typical range"
        elif rating == EfficiencyRating.BELOW_AVERAGE:
            quality = "suggesting slow-moving inventory"
        else:
            quality = "indicating potential obsolescence risk"
        
        # Add trend information
        trend_part = ""
        if change is not None:
            if change > 10:
                trend_part = f"; DIO increased {change:.0f} days YoY (concern)"
            elif change < -10:
                trend_part = f"; DIO improved {abs(change):.0f} days YoY"
        
        return f"{base}, {quality}{trend_part}"
    
    def _interpret_dpo(
        self,
        dpo: float,
        change: Optional[float],
        rating: EfficiencyRating
    ) -> str:
        """
        Generate human-readable interpretation for DPO.
        
        Args:
            dpo: Current DPO value.
            change: Change from prior period.
            rating: Efficiency rating.
        
        Returns:
            Interpretation string.
        """
        base = f"Company takes {dpo:.0f} days on average to pay suppliers"
        
        # Add quality assessment
        if rating == EfficiencyRating.EXCELLENT:
            quality = "optimally utilizing supplier credit terms"
        elif rating == EfficiencyRating.GOOD:
            quality = "appropriately managing payment timing"
        elif rating == EfficiencyRating.AVERAGE:
            quality = "within normal range"
        elif dpo < DEFAULT_DPO_LOW:
            quality = "paying quickly, potentially missing free financing"
        else:
            quality = "stretching payments, may indicate cash constraints"
        
        # Add trend information
        trend_part = ""
        if change is not None:
            if change > 15:
                trend_part = f"; DPO increased {change:.0f} days YoY (monitor closely)"
            elif change < -15:
                trend_part = f"; DPO decreased {abs(change):.0f} days YoY"
        
        return f"{base}, {quality}{trend_part}"
    
    def _interpret_ccc(
        self,
        ccc: float,
        change: Optional[float],
        trend_direction: TrendDirection
    ) -> str:
        """
        Generate human-readable interpretation for Cash Conversion Cycle.
        
        Args:
            ccc: Current CCC value.
            change: Change from prior period.
            trend_direction: Trend direction assessment.
        
        Returns:
            Interpretation string.
        """
        if ccc < 0:
            base = (
                f"Negative CCC of {ccc:.0f} days means the company is financed "
                f"by suppliers/customers - excellent working capital efficiency"
            )
        elif ccc < 30:
            base = (
                f"Short CCC of {ccc:.0f} days indicates efficient "
                f"working capital management"
            )
        elif ccc < 60:
            base = (
                f"Moderate CCC of {ccc:.0f} days is typical for most businesses"
            )
        elif ccc < 90:
            base = (
                f"Extended CCC of {ccc:.0f} days ties up significant "
                f"cash in operations"
            )
        else:
            base = (
                f"Long CCC of {ccc:.0f} days indicates substantial cash "
                f"tied up in working capital"
            )
        
        # Add change information
        if change is not None:
            if change > 10:
                base += f"; CCC deteriorated {change:.0f} days YoY"
            elif change < -10:
                base += f"; CCC improved {abs(change):.0f} days YoY"
        
        return base
    
    def _interpret_ar_component(
        self,
        balance: float,
        change_pct: Optional[float],
        dso: float
    ) -> str:
        """
        Generate interpretation for AR component.
        
        Args:
            balance: Current AR balance.
            change_pct: Percentage change from prior period.
            dso: Days Sales Outstanding.
        
        Returns:
            Interpretation string.
        """
        interpretation = f"AR balance of ${balance:.0f}M represents {dso:.0f} days of sales"
        
        if change_pct is not None:
            if change_pct > 0.15:
                interpretation += (
                    f"; increased {change_pct*100:.0f}% YoY "
                    f"(monitor for collection issues)"
                )
            elif change_pct < -0.15:
                interpretation += (
                    f"; decreased {abs(change_pct)*100:.0f}% YoY "
                    f"(improved collections)"
                )
        
        return interpretation
    
    def _interpret_inv_component(
        self,
        balance: float,
        change_pct: Optional[float],
        dio: float
    ) -> str:
        """
        Generate interpretation for Inventory component.
        
        Args:
            balance: Current inventory balance.
            change_pct: Percentage change from prior period.
            dio: Days Inventory Outstanding.
        
        Returns:
            Interpretation string.
        """
        interpretation = (
            f"Inventory of ${balance:.0f}M represents {dio:.0f} days of cost of sales"
        )
        
        if change_pct is not None:
            if change_pct > 0.20:
                interpretation += (
                    f"; increased {change_pct*100:.0f}% YoY "
                    f"(watch for obsolescence)"
                )
            elif change_pct < -0.20:
                interpretation += f"; decreased {abs(change_pct)*100:.0f}% YoY"
        
        return interpretation
    
    def _interpret_ap_component(
        self,
        balance: float,
        change_pct: Optional[float],
        dpo: float
    ) -> str:
        """
        Generate interpretation for AP component.
        
        Args:
            balance: Current AP balance.
            change_pct: Percentage change from prior period.
            dpo: Days Payable Outstanding.
        
        Returns:
            Interpretation string.
        """
        interpretation = f"AP balance of ${balance:.0f}M represents {dpo:.0f} days of purchases"
        
        if change_pct is not None:
            if change_pct > 0.25:
                interpretation += (
                    f"; increased {change_pct*100:.0f}% YoY "
                    f"(stretching payments)"
                )
            elif change_pct < -0.25:
                interpretation += f"; decreased {abs(change_pct)*100:.0f}% YoY"
        
        return interpretation
    
    # =========================================================================
    # ALERT GENERATION
    # =========================================================================
    
    def _generate_alerts(
        self,
        dso: EfficiencyMetrics,
        dio: EfficiencyMetrics,
        dpo: EfficiencyMetrics,
        ccc: CashConversionCycle,
        position: WorkingCapitalPosition
    ) -> None:
        """
        Generate working capital alerts based on analysis results.
        
        Examines all metrics and positions to identify potential concerns
        that warrant management attention.
        
        Args:
            dso: Days Sales Outstanding analysis.
            dio: Days Inventory Outstanding analysis.
            dpo: Days Payable Outstanding analysis.
            ccc: Cash Conversion Cycle analysis.
            position: Working capital position metrics.
        """
        # DSO alerts
        self._generate_dso_alerts(dso)
        
        # DIO alerts
        self._generate_dio_alerts(dio)
        
        # DPO alerts
        self._generate_dpo_alerts(dpo)
        
        # CCC alerts
        self._generate_ccc_alerts(ccc)
        
        # Working capital position alerts
        self._generate_position_alerts(position)
    
    def _generate_dso_alerts(self, dso: EfficiencyMetrics) -> None:
        """Generate alerts related to DSO."""
        max_dso = getattr(VALIDATION, 'max_reasonable_dso', DEFAULT_DSO_CONCERNING)
        
        # High DSO alert
        if dso.current_value > max_dso:
            self._alerts.append(WorkingCapitalAlert(
                alert_type=AlertType.DSO_HIGH,
                severity=AlertSeverity.HIGH,
                metric="DSO",
                current_value=dso.current_value,
                threshold=max_dso,
                description=(
                    f"DSO of {dso.current_value:.0f} days exceeds threshold of "
                    f"{max_dso} days - collection efficiency may be problematic"
                ),
                recommendation=(
                    "Investigate receivables aging and collection practices; "
                    "review credit policies and customer payment terms"
                )
            ))
        
        # Rising DSO alert
        if dso.change is not None and dso.change > DEFAULT_DSO_CHANGE_THRESHOLD:
            self._alerts.append(WorkingCapitalAlert(
                alert_type=AlertType.DSO_RISING,
                severity=AlertSeverity.MEDIUM,
                metric="DSO",
                current_value=dso.change,
                threshold=float(DEFAULT_DSO_CHANGE_THRESHOLD),
                description=(
                    f"DSO increased {dso.change:.0f} days year-over-year - "
                    f"collection efficiency is deteriorating"
                ),
                recommendation=(
                    "Review credit policies and collection procedures; "
                    "analyze customer payment patterns"
                )
            ))
    
    def _generate_dio_alerts(self, dio: EfficiencyMetrics) -> None:
        """Generate alerts related to DIO."""
        max_dio = getattr(VALIDATION, 'max_reasonable_dio', 365)
        
        # Skip if company has no inventory
        if dio.current_value == 0.0 and not dio.trend:
            return
        
        # High DIO alert
        if dio.current_value > max_dio:
            self._alerts.append(WorkingCapitalAlert(
                alert_type=AlertType.DIO_HIGH,
                severity=AlertSeverity.HIGH,
                metric="DIO",
                current_value=dio.current_value,
                threshold=max_dio,
                description=(
                    f"DIO of {dio.current_value:.0f} days exceeds threshold of "
                    f"{max_dio} days - inventory turnover is very slow"
                ),
                recommendation=(
                    "Assess inventory for obsolescence and slow-moving items; "
                    "optimize ordering and demand forecasting"
                )
            ))
        
        # Rising DIO alert
        if dio.change is not None and dio.change > DEFAULT_DIO_CHANGE_THRESHOLD:
            self._alerts.append(WorkingCapitalAlert(
                alert_type=AlertType.DIO_RISING,
                severity=AlertSeverity.MEDIUM,
                metric="DIO",
                current_value=dio.change,
                threshold=float(DEFAULT_DIO_CHANGE_THRESHOLD),
                description=(
                    f"DIO increased {dio.change:.0f} days year-over-year - "
                    f"inventory is building up"
                ),
                recommendation=(
                    "Review inventory levels relative to sales trends; "
                    "assess demand forecasting accuracy"
                )
            ))
    
    def _generate_dpo_alerts(self, dpo: EfficiencyMetrics) -> None:
        """Generate alerts related to DPO."""
        max_dpo = getattr(VALIDATION, 'max_reasonable_dpo', 180)
        
        # Stretched DPO alert (paying too slowly)
        if dpo.current_value > max_dpo:
            self._alerts.append(WorkingCapitalAlert(
                alert_type=AlertType.DPO_STRETCHED,
                severity=AlertSeverity.MEDIUM,
                metric="DPO",
                current_value=dpo.current_value,
                threshold=max_dpo,
                description=(
                    f"DPO of {dpo.current_value:.0f} days may indicate "
                    f"stretched supplier payments - potential cash constraints"
                ),
                recommendation=(
                    "Evaluate cash position and supplier relationships; "
                    "ensure payment terms align with agreements"
                )
            ))
        
        # DPO too low alert (paying too quickly)
        if dpo.current_value < DEFAULT_DPO_LOW and dpo.current_value > 0:
            self._alerts.append(WorkingCapitalAlert(
                alert_type=AlertType.DPO_TOO_LOW,
                severity=AlertSeverity.LOW,
                metric="DPO",
                current_value=dpo.current_value,
                threshold=float(DEFAULT_DPO_LOW),
                description=(
                    f"DPO of {dpo.current_value:.0f} days is low - company may be "
                    f"paying suppliers faster than necessary"
                ),
                recommendation=(
                    "Consider negotiating better payment terms; "
                    "utilize available credit periods to improve cash position"
                )
            ))
    
    def _generate_ccc_alerts(self, ccc: CashConversionCycle) -> None:
        """Generate alerts related to Cash Conversion Cycle."""
        # CCC deterioration alert
        if ccc.change is not None and ccc.change > DEFAULT_CCC_CHANGE_THRESHOLD:
            self._alerts.append(WorkingCapitalAlert(
                alert_type=AlertType.CCC_DETERIORATING,
                severity=AlertSeverity.MEDIUM,
                metric="CCC",
                current_value=ccc.change,
                threshold=float(DEFAULT_CCC_CHANGE_THRESHOLD),
                description=(
                    f"Cash Conversion Cycle worsened by {ccc.change:.0f} days - "
                    f"more cash is being tied up in operations"
                ),
                recommendation=(
                    "Identify which component (DSO, DIO, DPO) is driving "
                    "deterioration and address root causes"
                )
            ))
        
        # Extended CCC alert
        if ccc.current_value > 90:
            self._alerts.append(WorkingCapitalAlert(
                alert_type=AlertType.CCC_EXTENDED,
                severity=AlertSeverity.LOW,
                metric="CCC",
                current_value=ccc.current_value,
                threshold=90.0,
                description=(
                    f"Extended CCC of {ccc.current_value:.0f} days indicates "
                    f"significant working capital requirements"
                ),
                recommendation=(
                    "Evaluate opportunities to reduce DSO and DIO; "
                    "consider optimizing payment terms with suppliers"
                )
            ))
    
    def _generate_position_alerts(self, position: WorkingCapitalPosition) -> None:
        """
        Generate alerts related to working capital position.
        
        This method has been enhanced to add cash context to alerts. For companies
        with strong cash positions (like Apple, Amazon, or Dell), negative working
        capital is often a STRENGTH, not a weakness. These companies collect from
        customers before paying suppliers, effectively getting interest-free
        financing from their operating cycle.
        
        The key insight is that a company with $60B in cash and negative working
        capital is in a fundamentally different position than a distressed company
        with negative WC due to an inability to pay bills.
        
        Alert Logic:
        - Negative WC with strong cash coverage (>50%): LOW severity (by design)
        - Negative WC with weak cash coverage (<50%): MEDIUM/HIGH severity (concern)
        - Current ratio < 1.0 with strong cash: LOW severity (not a real concern)
        - Current ratio < 1.0 with weak cash: MEDIUM severity (monitor closely)
        """
        # Get cash data for context
        cash = self._get_field(StandardField.CASH, 0)
        current_liabilities = self._get_field(StandardField.CURRENT_LIABILITIES, 0)
        
        # Calculate cash coverage of current liabilities
        # This is a key metric: if cash alone can cover 50%+ of current liabilities,
        # negative WC is likely by design rather than a liquidity concern
        if current_liabilities is not None and current_liabilities > 0:
            cash_coverage = (cash / current_liabilities) if cash is not None else 0.0
        else:
            cash_coverage = 1.0  # Assume adequate if we can't calculate
        
        # Working capital cash drain alert
        # This alert fires when WC is growing rapidly, consuming cash
        if position.yoy_change > 0:
            prior_owc = position.operating_working_capital - position.yoy_change
            wc_growth_rate = (
                position.yoy_change / prior_owc
                if prior_owc != 0
                else 0.0
            )
            
            if wc_growth_rate > DEFAULT_WC_GROWTH_THRESHOLD:
                self._alerts.append(WorkingCapitalAlert(
                    alert_type=AlertType.WC_CASH_DRAIN,
                    severity=AlertSeverity.MEDIUM,
                    metric="Operating Working Capital",
                    current_value=position.yoy_change,
                    threshold=None,
                    description=(
                        f"Working capital increased ${position.yoy_change:.0f}M "
                        f"({wc_growth_rate*100:.0f}%), consuming significant cash"
                    ),
                    recommendation=(
                        "Evaluate if WC growth is aligned with revenue growth; "
                        "identify which components are driving the increase"
                    )
                ))
        
        # Negative working capital alert - WITH CASH CONTEXT
        if position.net_working_capital < 0:
            # Determine if this is by design or a concern based on cash coverage
            if cash_coverage >= 0.50:
                # Strong cash position - negative WC is likely by design
                # This is common for efficient operators like Apple, Amazon, Dell
                severity = AlertSeverity.LOW
                description = (
                    f"Negative net working capital of ${position.net_working_capital:,.0f}M. "
                    f"However, cash covers {cash_coverage:.0%} of current liabilities. "
                    f"This is characteristic of efficient operations where the company "
                    f"collects from customers before paying suppliers."
                )
                recommendation = (
                    "This appears to be an efficient working capital model. "
                    "No action needed unless cash position deteriorates."
                )
            elif cash_coverage >= 0.30:
                # Moderate cash position - monitor but not critical
                severity = AlertSeverity.LOW
                description = (
                    f"Negative net working capital of ${position.net_working_capital:,.0f}M "
                    f"with moderate cash coverage ({cash_coverage:.0%} of current liabilities). "
                    f"May indicate efficient operations or developing stress."
                )
                recommendation = (
                    "Monitor cash position trends; ensure business model "
                    "supports negative WC approach."
                )
            else:
                # Weak cash position - this is a genuine concern
                severity = AlertSeverity.MEDIUM
                description = (
                    f"Negative net working capital of ${position.net_working_capital:,.0f}M "
                    f"with limited cash coverage ({cash_coverage:.0%} of current liabilities). "
                    f"This may indicate liquidity stress."
                )
                recommendation = (
                    "Review near-term obligations and cash forecasts; "
                    "consider building liquidity buffers."
                )
            
            self._alerts.append(WorkingCapitalAlert(
                alert_type=AlertType.NEGATIVE_WC,
                severity=severity,
                metric="Net Working Capital",
                current_value=position.net_working_capital,
                threshold=0.0,
                description=description,
                recommendation=recommendation
            ))
        
        # Current ratio < 1.0 alert - WITH CASH CONTEXT
        if position.current_ratio < 1.0 and position.current_ratio > 0:
            # Determine severity based on cash coverage
            if cash_coverage >= 0.50:
                # Strong cash position makes low current ratio less concerning
                severity = AlertSeverity.LOW
                description = (
                    f"Current ratio of {position.current_ratio:.2f}x is below 1.0, "
                    f"but cash alone covers {cash_coverage:.0%} of current liabilities. "
                    f"The strong cash position mitigates traditional liquidity concerns."
                )
                recommendation = (
                    "Current ratio below 1.0 is acceptable given strong cash position. "
                    "Continue monitoring cash levels."
                )
            elif cash_coverage >= 0.30:
                # Moderate cash - some concern
                severity = AlertSeverity.LOW
                description = (
                    f"Current ratio of {position.current_ratio:.2f}x is below 1.0 "
                    f"with moderate cash coverage ({cash_coverage:.0%}). "
                    f"Not immediately concerning but warrants monitoring."
                )
                recommendation = (
                    "Monitor liquidity trends; maintain adequate credit facilities."
                )
            else:
                # Weak cash - genuine concern
                severity = AlertSeverity.MEDIUM
                description = (
                    f"Current ratio of {position.current_ratio:.2f}x is below 1.0 "
                    f"with limited cash coverage ({cash_coverage:.0%}). "
                    f"Current liabilities exceed current assets."
                )
                recommendation = (
                    "Monitor liquidity position closely; "
                    "ensure adequate access to short-term financing if needed."
                )
            
            self._alerts.append(WorkingCapitalAlert(
                alert_type=AlertType.LIQUIDITY_CONCERN,
                severity=severity,
                metric="Current Ratio",
                current_value=position.current_ratio,
                threshold=1.0,
                description=description,
                recommendation=recommendation
            ))
    
    
    # =========================================================================
    # INSIGHTS GENERATION
    # =========================================================================
    
    def _generate_insights(
        self,
        dso: EfficiencyMetrics,
        dio: EfficiencyMetrics,
        dpo: EfficiencyMetrics,
        ccc: CashConversionCycle,
        position: WorkingCapitalPosition
    ) -> List[str]:
        """
        Generate key insights from the working capital analysis.
        
        Args:
            dso: Days Sales Outstanding analysis.
            dio: Days Inventory Outstanding analysis.
            dpo: Days Payable Outstanding analysis.
            ccc: Cash Conversion Cycle analysis.
            position: Working capital position metrics.
        
        Returns:
            List of insight strings.
        """
        insights: List[str] = []
        
        # CCC insight
        if ccc.current_value < 0:
            insights.append(
                f"Negative CCC of {ccc.current_value:.0f} days indicates the company "
                f"is effectively financed by its suppliers - excellent efficiency"
            )
        elif ccc.current_value < 30:
            insights.append(
                f"Short CCC of {ccc.current_value:.0f} days demonstrates efficient "
                f"working capital management with minimal cash tied up"
            )
        elif ccc.current_value > 90:
            insights.append(
                f"Extended CCC of {ccc.current_value:.0f} days means significant "
                f"cash is tied up in operations, impacting free cash flow"
            )
        
        # CCC composition insight
        if dio.current_value == 0:
            # Service-based company
            insights.append(
                f"CCC of {ccc.current_value:.0f} days driven by DSO "
                f"({dso.current_value:.0f}) less DPO ({dpo.current_value:.0f}) - "
                f"asset-light business model"
            )
        else:
            # Identify largest component
            components = [
                ("DSO", dso.current_value),
                ("DIO", dio.current_value)
            ]
            max_component = max(components, key=lambda x: x[1])
            insights.append(
                f"{max_component[0]} of {max_component[1]:.0f} days is the largest "
                f"component of the CCC - primary focus area for improvement"
            )
        
        # Trend insights
        if ccc.trend_direction == TrendDirection.IMPROVING:
            insights.append(
                "Working capital efficiency is IMPROVING - CCC trending downward"
            )
        elif ccc.trend_direction == TrendDirection.DETERIORATING:
            insights.append(
                "Working capital efficiency is DETERIORATING - CCC trending upward, "
                "consuming more cash"
            )
        
        # Cash impact insight
        if abs(position.yoy_change) >= 1.0:
            insights.append(position.cash_impact)
        
        # Liquidity insight
        if position.current_ratio >= 2.0:
            insights.append(
                f"Strong current ratio of {position.current_ratio:.2f}x indicates "
                f"ample liquidity to cover short-term obligations"
            )
        elif position.current_ratio < 1.0 and position.current_ratio > 0:
            insights.append(
                f"Current ratio of {position.current_ratio:.2f}x below 1.0 - "
                f"monitor liquidity position closely"
            )
        
        # Working capital intensity insight
        if position.wc_to_revenue > 0.20:
            insights.append(
                f"Working capital at {position.wc_to_revenue*100:.0f}% of revenue "
                f"indicates capital-intensive operations"
            )
        elif position.wc_to_revenue < 0.05 and position.wc_to_revenue >= 0:
            insights.append(
                f"Low working capital intensity ({position.wc_to_revenue*100:.0f}% "
                f"of revenue) enables strong cash conversion"
            )
        
        return insights
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _get_field(
        self,
        field: StandardField,
        period_index: int = 0
    ) -> Optional[float]:
        """
        Extract a field value from processed data.
        
        Args:
            field: StandardField enum specifying which field to extract.
            period_index: Index of the period (0 = most recent).
        
        Returns:
            Field value as float, or None if not available.
        """
        return self._processor.get_field(self._data, field, period_index)
    
    def _add_warning(self, message: str) -> None:
        """
        Add a warning message to the analysis results.
        
        Args:
            message: Warning message string.
        """
        self._warnings.append(message)
        logger.warning(message)
    
    # =========================================================================
    # OUTPUT FORMATTING
    # =========================================================================
    
    def format_efficiency_summary(
        self,
        result: WorkingCapitalAnalysisResult
    ) -> pd.DataFrame:
        """
        Format efficiency metrics as a summary DataFrame.
        
        Args:
            result: WorkingCapitalAnalysisResult to format.
        
        Returns:
            DataFrame with formatted efficiency metrics.
        """
        data = [
            {
                'Metric': result.dso.name,
                'Current': f"{result.dso.current_value:.1f}",
                'Prior': (
                    f"{result.dso.prior_value:.1f}"
                    if result.dso.prior_value is not None else "N/A"
                ),
                'Change': (
                    f"{result.dso.change:+.1f}"
                    if result.dso.change is not None else "N/A"
                ),
                'Rating': result.dso.rating.value,
                'Trend': result.dso.trend_direction.value
            },
            {
                'Metric': result.dio.name,
                'Current': f"{result.dio.current_value:.1f}",
                'Prior': (
                    f"{result.dio.prior_value:.1f}"
                    if result.dio.prior_value is not None else "N/A"
                ),
                'Change': (
                    f"{result.dio.change:+.1f}"
                    if result.dio.change is not None else "N/A"
                ),
                'Rating': result.dio.rating.value,
                'Trend': result.dio.trend_direction.value
            },
            {
                'Metric': result.dpo.name,
                'Current': f"{result.dpo.current_value:.1f}",
                'Prior': (
                    f"{result.dpo.prior_value:.1f}"
                    if result.dpo.prior_value is not None else "N/A"
                ),
                'Change': (
                    f"{result.dpo.change:+.1f}"
                    if result.dpo.change is not None else "N/A"
                ),
                'Rating': result.dpo.rating.value,
                'Trend': result.dpo.trend_direction.value
            },
            {
                'Metric': 'Cash Conversion Cycle',
                'Current': f"{result.cash_conversion_cycle.current_value:.1f}",
                'Prior': (
                    f"{result.cash_conversion_cycle.prior_value:.1f}"
                    if result.cash_conversion_cycle.prior_value is not None else "N/A"
                ),
                'Change': (
                    f"{result.cash_conversion_cycle.change:+.1f}"
                    if result.cash_conversion_cycle.change is not None else "N/A"
                ),
                'Rating': '',
                'Trend': result.cash_conversion_cycle.trend_direction.value
            }
        ]
        
        return pd.DataFrame(data)
    
    def format_alerts_table(
        self,
        result: WorkingCapitalAnalysisResult
    ) -> pd.DataFrame:
        """
        Format alerts as a DataFrame.
        
        Args:
            result: WorkingCapitalAnalysisResult containing alerts.
        
        Returns:
            DataFrame listing all alerts with details.
        """
        if not result.alerts:
            return pd.DataFrame({'Message': ['No alerts identified']})
        
        data = []
        for alert in result.alerts:
            description = alert.description
            if len(description) > 80:
                description = description[:77] + '...'
            
            data.append({
                'Severity': alert.severity.value,
                'Type': alert.alert_type.value,
                'Metric': alert.metric,
                'Description': description
            })
        
        return pd.DataFrame(data)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_working_capital(processed_data: ProcessedData) -> WorkingCapitalAnalysisResult:
    """
    Convenience function to perform working capital analysis.
    
    Args:
        processed_data: ProcessedData object from DataProcessor.
    
    Returns:
        WorkingCapitalAnalysisResult with complete analysis.
    
    Example:
        from data_collector import collect_financial_data
        from data_processor import process_financial_data
        
        raw = collect_financial_data("AAPL")
        processed = process_financial_data(raw)
        result = analyze_working_capital(processed)
        
        print(f"DSO: {result.dso.current_value:.1f} days")
        print(f"CCC: {result.cash_conversion_cycle.current_value:.1f} days")
    """
    analyzer = WorkingCapitalAnalyzer(processed_data)
    return analyzer.analyze()


def get_working_capital_summary(result: WorkingCapitalAnalysisResult) -> Dict[str, Any]:
    """
    Extract key working capital metrics as a simple dictionary.
    
    Useful for quick access to summary metrics without parsing the full result.
    
    Args:
        result: WorkingCapitalAnalysisResult to summarize.
    
    Returns:
        Dictionary with key metrics.
    """
    return {
        'dso': result.dso.current_value,
        'dio': result.dio.current_value,
        'dpo': result.dpo.current_value,
        'cash_conversion_cycle': result.cash_conversion_cycle.current_value,
        'operating_working_capital': result.position.operating_working_capital,
        'wc_to_revenue': result.position.wc_to_revenue,
        'current_ratio': result.position.current_ratio,
        'quick_ratio': result.position.quick_ratio,
        'alert_count': len(result.alerts)
    }


# =============================================================================
# MODULE SELF-TEST
# =============================================================================

if __name__ == "__main__":
    """
    Module test script.
    
    Run this file directly to test working capital analysis:
        python working_capital_analyzer.py [TICKER]
    """
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get ticker from command line or use default
    test_ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    print()
    print("=" * 70)
    print(f"WORKING CAPITAL ANALYZER TEST - {test_ticker}")
    print("=" * 70)
    print()
    
    try:
        # Step 1: Collect raw data
        print("Step 1: Collecting raw data...")
        from data_collector import DataCollector
        collector = DataCollector()
        raw_data = collector.collect(test_ticker)
        print(f"  Raw data collected: {raw_data.validation.years_available} years")
        print()
        
        # Step 2: Process data
        print("Step 2: Processing data...")
        processor = DataProcessor()
        processed = processor.process(raw_data)
        print("  Processing complete")
        print()
        
        # Step 3: Analyze working capital
        print("Step 3: Analyzing working capital...")
        analyzer = WorkingCapitalAnalyzer(processed)
        result = analyzer.analyze()
        print("  Analysis complete")
        print()
        
        # Print Efficiency Metrics
        print("WORKING CAPITAL EFFICIENCY")
        print("-" * 70)
        print(f"  {'Metric':<30} {'Current':>10} {'Prior':>10} "
              f"{'Change':>10} {'Rating':<15}")
        print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*15}")
        
        for metric in [result.dso, result.dio, result.dpo]:
            current = f"{metric.current_value:.1f}"
            prior = (
                f"{metric.prior_value:.1f}"
                if metric.prior_value is not None else "N/A"
            )
            change = (
                f"{metric.change:+.1f}"
                if metric.change is not None else "N/A"
            )
            print(f"  {metric.name:<30} {current:>10} {prior:>10} "
                  f"{change:>10} {metric.rating.value:<15}")
        
        # Print Cash Conversion Cycle
        print()
        print("CASH CONVERSION CYCLE")
        print("-" * 70)
        ccc = result.cash_conversion_cycle
        print(f"  CCC = DSO + DIO - DPO")
        print(f"  CCC = {ccc.dso_contribution:.1f} + {ccc.dio_contribution:.1f} "
              f"- {ccc.dpo_contribution:.1f}")
        print(f"  CCC = {ccc.current_value:.1f} days")
        if ccc.prior_value is not None:
            print(f"  Prior Period: {ccc.prior_value:.1f} days")
            print(f"  Change: {ccc.change:+.1f} days")
        print(f"  Trend: {ccc.trend_direction.value}")
        print(f"  {ccc.interpretation}")
        
        # Print Working Capital Position
        print()
        print("WORKING CAPITAL POSITION")
        print("-" * 70)
        pos = result.position
        print(f"  Net Working Capital:       ${pos.net_working_capital:,.1f}M")
        print(f"  Operating Working Capital: ${pos.operating_working_capital:,.1f}M")
        print(f"  WC / Revenue:              {pos.wc_to_revenue*100:.1f}%")
        print(f"  Current Ratio:             {pos.current_ratio:.2f}x")
        print(f"  Quick Ratio:               {pos.quick_ratio:.2f}x")
        print(f"  YoY Change:                ${pos.yoy_change:+,.1f}M")
        print(f"  Cash Impact:               {pos.cash_impact}")
        
        # Print Component Analysis
        print()
        print("COMPONENT ANALYSIS")
        print("-" * 70)
        for comp in result.components:
            print(f"  {comp.name}:")
            print(f"    Balance: ${comp.current_balance:,.1f}M "
                  f"({comp.days_metric:.1f} days)")
            if comp.change is not None:
                print(f"    Change: ${comp.change:+,.1f}M "
                      f"({comp.change_percent*100:+.1f}%)")
            print(f"    {comp.interpretation}")
            print()
        
        # Print Alerts
        if result.alerts:
            print(f"ALERTS ({len(result.alerts)} identified)")
            print("-" * 70)
            for alert in result.alerts:
                print(f"  [{alert.severity.value}] {alert.alert_type.value}")
                print(f"    {alert.description}")
                print(f"    Recommendation: {alert.recommendation}")
                print()
        
        # Print Insights
        print("KEY INSIGHTS")
        print("-" * 70)
        for i, insight in enumerate(result.insights, 1):
            print(f"  {i}. {insight}")
        
        # Print Warnings
        if result.warnings:
            print()
            print("WARNINGS")
            print("-" * 70)
            for warning in result.warnings:
                print(f"  - {warning}")
        
        print()
        print("=" * 70)
        print(f"Working capital analysis complete for {test_ticker}")
        print("=" * 70)
        print()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)