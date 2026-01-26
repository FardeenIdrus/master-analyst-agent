"""
Earnings Quality Analyzer Module for Fundamental Analyst Agent

This module assesses the quality and sustainability of reported earnings through
multiple analytical lenses. High-quality earnings are characterized by being
backed by cash, free from aggressive accounting, and sustainable over time.

The analysis is critical for identifying:
    1. Potential earnings manipulation or aggressive accounting
    2. Sustainability of current profitability
    3. Divergence between reported profits and economic reality
    4. Red flags that may precede earnings restatements or declines

Analytical Framework:
    1. Accruals Analysis
       - Accruals Ratio = (Net Income - OCF) / Average Total Assets
       - High accruals indicate earnings not backed by cash
       - Based on Sloan (1996) research showing high accruals predict earnings declines

    2. Cash Conversion Analysis
       - Cash Conversion = OCF / Net Income
       - Measures how much of reported earnings converts to cash
       - Persistent low conversion suggests earnings quality issues

    3. Growth Divergence Analysis
       - Compare AR growth vs Revenue growth (channel stuffing detection)
       - Compare Inventory growth vs COGS growth (obsolescence detection)
       - Compare Payables growth vs COGS growth (cash flow stress detection)
       - Large divergences are red flags for manipulation

    4. Revenue Recognition Analysis
       - Days Sales Outstanding (DSO) trends
       - Deferred revenue changes
       - Revenue vs cash receipts alignment

    5. Expense Manipulation Detection
       - Capitalization vs expensing patterns
       - R&D and CapEx to revenue ratios
       - Depreciation policy analysis

Quality Ratings:
    - HIGH: Accruals <5%, strong cash conversion, no red flags
    - MODERATE: Accruals 5-10%, acceptable conversion, minor flags
    - LOW: Accruals 10-15%, weak conversion, some red flags
    - CONCERN: Accruals >15%, poor conversion, multiple red flags

MSc Coursework: AI Agents in Asset Management
Track A: Fundamental Analyst Agent

Author: MSc AI Agents in Asset Management
Version: 1.0.1
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
    EARNINGS_QUALITY,
    EarningsQualityRating,
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

# Default thresholds for accruals quality assessment
# These are used when configuration values are not available
DEFAULT_ACCRUALS_HIGH_QUALITY = 0.05       # 5% - earnings well-backed by cash
DEFAULT_ACCRUALS_MODERATE_QUALITY = 0.10   # 10% - acceptable but monitor
DEFAULT_ACCRUALS_LOW_QUALITY = 0.15        # 15% - significant quality concerns

# Default thresholds for cash conversion assessment
DEFAULT_CASH_CONVERSION_EXCELLENT = 1.10   # 110% - OCF exceeds net income
DEFAULT_CASH_CONVERSION_GOOD = 0.90        # 90% - strong conversion
DEFAULT_CASH_CONVERSION_ACCEPTABLE = 0.70  # 70% - minimum acceptable

# Default thresholds for growth divergence detection
DEFAULT_AR_VS_REVENUE_THRESHOLD = 0.10     # 10pp divergence triggers flag
DEFAULT_INVENTORY_VS_COGS_THRESHOLD = 0.10 # 10pp divergence triggers flag
DEFAULT_PAYABLES_VS_COGS_THRESHOLD = 0.15  # 15pp divergence triggers flag

# Gross margin volatility threshold
DEFAULT_GROSS_MARGIN_VOLATILITY_THRESHOLD = 0.05  # 5pp standard deviation

# Accruals trend deterioration threshold
DEFAULT_ACCRUALS_DETERIORATION_THRESHOLD = 0.03   # 3pp increase is concerning


# =============================================================================
# ENUMERATIONS
# =============================================================================

class RedFlagCategory(Enum):
    """
    Categories of earnings quality red flags.
    
    Used to classify the type of potential earnings quality issue detected.
    """
    ACCRUALS = "Accruals"
    REVENUE_RECOGNITION = "Revenue Recognition"
    EXPENSE_MANIPULATION = "Expense Manipulation"
    WORKING_CAPITAL = "Working Capital"
    CASH_FLOW = "Cash Flow"
    ACCOUNTING_POLICY = "Accounting Policy"


class RedFlagSeverity(Enum):
    """
    Severity levels for red flags.
    
    LOW: Minor concern, worth monitoring
    MEDIUM: Moderate concern, warrants investigation
    HIGH: Significant concern, requires detailed analysis
    CRITICAL: Severe concern, immediate attention required
    """
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AccrualsAnalysis:
    """
    Analysis of accounting accruals.
    
    Accruals represent the difference between reported earnings (net income)
    and actual cash generated from operations. The formula is:
    
        Accruals = Net Income - Operating Cash Flow
        Accruals Ratio = Accruals / Average Total Assets
    
    High accruals relative to assets suggest earnings that are not backed
    by cash and may be less sustainable. Research by Sloan (1996) demonstrated
    that high accruals predict future earnings declines.
    
    Attributes:
        total_accruals: Absolute accruals amount in millions
        average_total_assets: Average total assets used as denominator
        accruals_ratio: Accruals as percentage of average assets
        accruals_rating: Quality rating based on accruals ratio
        interpretation: Human-readable explanation of the finding
        historical_trend: List of accruals ratios across available periods
    """
    total_accruals: float
    average_total_assets: float
    accruals_ratio: float
    accruals_rating: EarningsQualityRating
    interpretation: str
    historical_trend: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_accruals": self.total_accruals,
            "average_total_assets": self.average_total_assets,
            "accruals_ratio": self.accruals_ratio,
            "accruals_rating": self.accruals_rating.value,
            "interpretation": self.interpretation,
            "historical_trend": self.historical_trend
        }


@dataclass
class GrowthDivergence:
    """
    Analysis of growth rate divergences between related metrics.
    
    When related financial metrics grow at significantly different rates,
    it may indicate earnings manipulation or operational issues. For example:
    - AR growing faster than revenue may indicate channel stuffing
    - Inventory growing faster than COGS may indicate obsolescence
    
    Attributes:
        metric_name: Descriptive name of the comparison
        growth_rate_1: Growth rate of the first metric
        growth_rate_2: Growth rate of the second (benchmark) metric
        divergence: Difference between growth rates (rate_1 - rate_2)
        is_flagged: Whether divergence exceeds the threshold
        threshold: The threshold used for flagging
        interpretation: Human-readable explanation of the finding
    """
    metric_name: str
    growth_rate_1: float
    growth_rate_2: float
    divergence: float
    is_flagged: bool
    threshold: float
    interpretation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metric_name": self.metric_name,
            "growth_rate_1": self.growth_rate_1,
            "growth_rate_2": self.growth_rate_2,
            "divergence": self.divergence,
            "is_flagged": self.is_flagged,
            "threshold": self.threshold,
            "interpretation": self.interpretation
        }


@dataclass
class RedFlag:
    """
    A specific earnings quality red flag identified during analysis.
    
    Red flags are potential warning signs that warrant further investigation.
    Each flag includes the category, severity, description, and recommendation
    for follow-up action.
    
    Attributes:
        category: Type of red flag (accruals, revenue recognition, etc.)
        severity: How serious the concern is (LOW to CRITICAL)
        title: Brief title for the red flag
        description: Detailed explanation of the issue
        metric_value: The actual value that triggered the flag
        threshold_value: The threshold that was exceeded
        recommendation: Suggested follow-up action
    """
    category: RedFlagCategory
    severity: RedFlagSeverity
    title: str
    description: str
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None
    recommendation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "metric_value": self.metric_value,
            "threshold_value": self.threshold_value,
            "recommendation": self.recommendation
        }


@dataclass
class QualityScore:
    """
    Quantitative earnings quality score for a specific component.
    
    The overall earnings quality score is a weighted average of component
    scores. Each component is scored from 0-100 with specific weights.
    
    Attributes:
        component: Name of the scoring component
        score: Raw score from 0-100
        weight: Weight applied to this component (0.0-1.0)
        weighted_score: Score multiplied by weight
        explanation: Human-readable explanation of the score
    """
    component: str
    score: float
    weight: float
    weighted_score: float
    explanation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "component": self.component,
            "score": self.score,
            "weight": self.weight,
            "weighted_score": self.weighted_score,
            "explanation": self.explanation
        }


@dataclass
class EarningsQualityResult:
    """
    Complete result of earnings quality analysis.
    
    This is the primary output of the EarningsQualityAnalyzer class,
    containing all analysis results, scores, red flags, and insights.
    
    Attributes:
        ticker: Stock ticker symbol analyzed
        analysis_period: Fiscal period label (e.g., "FY2024")
        overall_rating: Final quality rating (HIGH, MODERATE, LOW, CONCERN)
        overall_score: Composite quality score from 0-100
        accruals_analysis: Detailed accruals analysis results
        cash_conversion_rate: OCF / Net Income ratio
        growth_divergences: List of growth divergence analyses
        quality_scores: Component scores with weights
        red_flags: List of identified red flags
        positive_indicators: List of positive quality signals
        insights: Key analytical insights
        warnings: Any warnings generated during analysis
        analysis_timestamp: When the analysis was performed
    """
    ticker: str
    analysis_period: str
    overall_rating: EarningsQualityRating
    overall_score: float
    accruals_analysis: AccrualsAnalysis
    cash_conversion_rate: float
    growth_divergences: List[GrowthDivergence]
    quality_scores: List[QualityScore]
    red_flags: List[RedFlag]
    positive_indicators: List[str]
    insights: List[str]
    warnings: List[str]
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ticker": self.ticker,
            "analysis_period": self.analysis_period,
            "overall_rating": self.overall_rating.value,
            "overall_score": self.overall_score,
            "accruals_analysis": self.accruals_analysis.to_dict(),
            "cash_conversion_rate": self.cash_conversion_rate,
            "growth_divergences": [d.to_dict() for d in self.growth_divergences],
            "quality_scores": [s.to_dict() for s in self.quality_scores],
            "red_flags": [f.to_dict() for f in self.red_flags],
            "positive_indicators": self.positive_indicators,
            "insights": self.insights,
            "warnings": self.warnings,
            "analysis_timestamp": self.analysis_timestamp.isoformat()
        }


# =============================================================================
# EARNINGS QUALITY ANALYZER CLASS
# =============================================================================

class EarningsQualityAnalyzer:
    """
    Comprehensive earnings quality assessment analyzer.
    
    This analyzer evaluates the quality and sustainability of reported
    earnings through multiple analytical lenses: accruals analysis,
    cash conversion, growth divergences, and red flag detection.
    
    The methodology draws on academic research (Sloan 1996, Beneish M-Score)
    and practitioner frameworks for detecting earnings manipulation.
    
    Usage:
        analyzer = EarningsQualityAnalyzer(processed_data)
        result = analyzer.analyze()
        print(f"Rating: {result.overall_rating.value}")
        print(f"Score: {result.overall_score:.0f}/100")
    
    Attributes:
        _data: ProcessedData object containing financial statements
        _processor: DataProcessor instance for field extraction
        _warnings: List of warnings generated during analysis
        _red_flags: List of red flags identified
        _positive_indicators: List of positive quality signals found
    """
    
    def __init__(self, processed_data: ProcessedData) -> None:
        """
        Initialize the EarningsQualityAnalyzer.
        
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
        self._red_flags: List[RedFlag] = []
        self._positive_indicators: List[str] = []
        
        logger.info(f"EarningsQualityAnalyzer initialized for {processed_data.ticker}")
    
    def analyze(self) -> EarningsQualityResult:
        """
        Perform complete earnings quality analysis.
        
        Executes all analysis components in sequence:
        1. Accruals analysis
        2. Cash conversion calculation
        3. Growth divergence analysis
        4. Red flag detection
        5. Positive indicator identification
        6. Quality scoring
        7. Insight generation
        
        Returns:
            EarningsQualityResult containing complete analysis results.
        """
        logger.info(f"Starting earnings quality analysis for {self._data.ticker}")
        
        # Reset state for fresh analysis
        self._warnings = []
        self._red_flags = []
        self._positive_indicators = []
        
        # Get analysis period label
        periods = self._data.statements.periods
        analysis_period = self._format_analysis_period(periods)
        
        # Core analyses
        accruals_analysis = self._analyze_accruals()
        cash_conversion = self._calculate_cash_conversion()
        growth_divergences = self._analyze_growth_divergences()
        
        # Detection and identification
        self._detect_red_flags(accruals_analysis, cash_conversion, growth_divergences)
        self._identify_positive_indicators(accruals_analysis, cash_conversion)
        
        # Scoring
        quality_scores = self._calculate_quality_scores(
            accruals_analysis, cash_conversion, growth_divergences
        )
        overall_score = self._calculate_overall_score(quality_scores)
        overall_rating = self._determine_overall_rating(
            overall_score, accruals_analysis, len(self._red_flags)
        )
        
        # Generate insights
        insights = self._generate_insights(
            accruals_analysis, cash_conversion, growth_divergences, overall_rating
        )
        
        # Build result object
        result = EarningsQualityResult(
            ticker=self._data.ticker,
            analysis_period=analysis_period,
            overall_rating=overall_rating,
            overall_score=overall_score,
            accruals_analysis=accruals_analysis,
            cash_conversion_rate=cash_conversion,
            growth_divergences=growth_divergences,
            quality_scores=quality_scores,
            red_flags=self._red_flags.copy(),
            positive_indicators=self._positive_indicators.copy(),
            insights=insights,
            warnings=self._warnings.copy()
        )
        
        logger.info(
            f"Earnings quality analysis complete for {self._data.ticker}: "
            f"{overall_rating.value} ({overall_score:.0f}/100)"
        )
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
            # Extract year from period string
            # Format could be "2024", "2024-09-30", or "FY2024"
            if period.startswith("FY"):
                return period
            elif "-" in period:
                # Date format like "2024-09-30", extract year
                year = period.split("-")[0]
                return f"FY{year}"
            else:
                return f"FY{period}"
        else:
            # Fallback for unexpected types
            return f"FY{period}"
    
    # =========================================================================
    # ACCRUALS ANALYSIS
    # =========================================================================
    
    def _analyze_accruals(self) -> AccrualsAnalysis:
        """
        Analyze accounting accruals to assess earnings quality.
        
        Calculates the accruals ratio using the formula:
            Accruals = Net Income - Operating Cash Flow
            Accruals Ratio = Accruals / Average Total Assets
        
        A high positive accruals ratio indicates earnings not backed by cash,
        which is a negative quality signal. Negative accruals (OCF > NI)
        indicate high-quality earnings.
        
        Returns:
            AccrualsAnalysis object with complete accruals assessment.
        """
        logger.debug("Analyzing accruals")
        
        # Get current period values
        net_income = self._get_field(StandardField.NET_INCOME, 0)
        ocf = self._get_field(StandardField.OPERATING_CASH_FLOW, 0)
        
        # Handle missing data
        if net_income is None:
            net_income = 0.0
            self._add_warning("Net income unavailable for current period")
        if ocf is None:
            ocf = 0.0
            self._add_warning("Operating cash flow unavailable for current period")
        
        # Calculate average total assets
        total_assets_current = self._get_field(StandardField.TOTAL_ASSETS, 0)
        total_assets_prior = self._get_field(StandardField.TOTAL_ASSETS, 1)
        
        if total_assets_current is None or total_assets_current <= 0:
            total_assets_current = 0.0
            self._add_warning("Current period total assets unavailable or invalid")
        
        if total_assets_prior is not None and total_assets_prior > 0:
            avg_assets = (total_assets_current + total_assets_prior) / 2.0
        else:
            avg_assets = total_assets_current
            if total_assets_current > 0:
                self._add_warning(
                    "Prior period assets unavailable; using current period only "
                    "for average assets calculation"
                )
        
        # Calculate accruals and ratio
        total_accruals = net_income - ocf
        
        if avg_assets > 0:
            accruals_ratio = total_accruals / avg_assets
        else:
            accruals_ratio = 0.0
            self._add_warning(
                "Average assets is zero or negative; accruals ratio set to 0"
            )
        
        # Calculate historical trend
        historical_trend = self._calculate_accruals_trend()
        
        # Rate the accruals
        accruals_rating, interpretation = self._rate_accruals(accruals_ratio)
        
        return AccrualsAnalysis(
            total_accruals=total_accruals,
            average_total_assets=avg_assets,
            accruals_ratio=accruals_ratio,
            accruals_rating=accruals_rating,
            interpretation=interpretation,
            historical_trend=historical_trend
        )
    
    def _calculate_accruals_trend(self) -> List[float]:
        """
        Calculate accruals ratios for all available periods.
        
        Returns:
            List of accruals ratios from most recent to oldest period.
        """
        trend: List[float] = []
        periods = self._data.statements.periods
        
        for i in range(len(periods)):
            ni = self._get_field(StandardField.NET_INCOME, i)
            ocf = self._get_field(StandardField.OPERATING_CASH_FLOW, i)
            assets_curr = self._get_field(StandardField.TOTAL_ASSETS, i)
            assets_prior = self._get_field(StandardField.TOTAL_ASSETS, i + 1)
            
            # Only calculate if we have the essential data
            if ni is not None and ocf is not None and assets_curr is not None:
                accruals = ni - ocf
                
                # Use average assets if prior period available
                if assets_prior is not None and assets_prior > 0:
                    avg_assets = (assets_curr + assets_prior) / 2.0
                else:
                    avg_assets = assets_curr
                
                if avg_assets > 0:
                    trend.append(accruals / avg_assets)
        
        return trend
    
    def _rate_accruals(
        self,
        accruals_ratio: float
    ) -> Tuple[EarningsQualityRating, str]:
        """
        Rate earnings quality based on accruals ratio.
        
        Args:
            accruals_ratio: Calculated accruals ratio (accruals / avg assets).
        
        Returns:
            Tuple of (rating, interpretation_string).
        """
        # Get thresholds from config, with fallbacks
        high_quality = getattr(
            EARNINGS_QUALITY, 'accruals_high_quality', DEFAULT_ACCRUALS_HIGH_QUALITY
        )
        moderate_quality = getattr(
            EARNINGS_QUALITY, 'accruals_moderate_quality', DEFAULT_ACCRUALS_MODERATE_QUALITY
        )
        low_quality = getattr(
            EARNINGS_QUALITY, 'accruals_low_quality', DEFAULT_ACCRUALS_LOW_QUALITY
        )
        
        # Negative accruals mean OCF > NI, which is positive
        if accruals_ratio < 0:
            return (
                EarningsQualityRating.HIGH,
                f"Negative accruals ({accruals_ratio*100:.1f}%) indicate operating "
                f"cash flow exceeds net income - earnings are well-backed by cash"
            )
        
        # Use absolute ratio for threshold comparison
        abs_ratio = abs(accruals_ratio)
        
        if abs_ratio <= high_quality:
            return (
                EarningsQualityRating.HIGH,
                f"Low accruals ratio ({accruals_ratio*100:.1f}%) indicates high "
                f"quality earnings that are well-supported by operating cash flow"
            )
        elif abs_ratio <= moderate_quality:
            return (
                EarningsQualityRating.MODERATE,
                f"Moderate accruals ratio ({accruals_ratio*100:.1f}%) suggests "
                f"acceptable earnings quality but warrants monitoring"
            )
        elif abs_ratio <= low_quality:
            return (
                EarningsQualityRating.LOW,
                f"Elevated accruals ratio ({accruals_ratio*100:.1f}%) indicates "
                f"a meaningful gap between reported earnings and cash generation"
            )
        else:
            return (
                EarningsQualityRating.CONCERN,
                f"High accruals ratio ({accruals_ratio*100:.1f}%) is a significant "
                f"concern - earnings are not well-supported by cash flow"
            )
    
    # =========================================================================
    # CASH CONVERSION ANALYSIS
    # =========================================================================
    
    def _calculate_cash_conversion(self) -> float:
        """
        Calculate cash conversion rate (OCF / Net Income).
        
        Cash conversion measures how effectively reported earnings convert
        to actual operating cash flow. A ratio above 100% indicates operating
        cash flow exceeds net income, which is a positive quality signal.
        
        Returns:
            Cash conversion rate as a decimal (1.0 = 100%).
        """
        net_income = self._get_field(StandardField.NET_INCOME, 0)
        ocf = self._get_field(StandardField.OPERATING_CASH_FLOW, 0)
        
        # Handle None values
        if net_income is None:
            net_income = 0.0
        if ocf is None:
            ocf = 0.0
        
        # Calculate conversion rate
        if net_income > 0:
            return ocf / net_income
        elif net_income < 0 and ocf > 0:
            # Unusual case: positive cash flow despite loss
            self._add_warning(
                "Positive operating cash flow despite net loss - "
                "cash conversion rate not meaningful in this context"
            )
            return 0.0
        else:
            # Net income is zero or negative with non-positive OCF
            return 0.0
    
    # =========================================================================
    # GROWTH DIVERGENCE ANALYSIS
    # =========================================================================
    
    def _analyze_growth_divergences(self) -> List[GrowthDivergence]:
        """
        Analyze divergences between related growth rates.
        
        Examines three key relationships:
        1. Accounts Receivable growth vs Revenue growth
        2. Inventory growth vs Cost of Revenue growth
        3. Accounts Payable growth vs Cost of Revenue growth
        
        Returns:
            List of GrowthDivergence objects for each analyzed relationship.
        """
        divergences: List[GrowthDivergence] = []
        
        # Analyze each divergence type
        ar_div = self._analyze_ar_revenue_divergence()
        if ar_div is not None:
            divergences.append(ar_div)
        
        inv_div = self._analyze_inventory_cogs_divergence()
        if inv_div is not None:
            divergences.append(inv_div)
        
        ap_div = self._analyze_payables_divergence()
        if ap_div is not None:
            divergences.append(ap_div)
        
        return divergences
    
    def _analyze_ar_revenue_divergence(self) -> Optional[GrowthDivergence]:
        """
        Analyze divergence between Accounts Receivable and Revenue growth.
        
        When AR grows significantly faster than revenue, it MAY indicate:
        - Channel stuffing (pushing sales to distributors)
        - Aggressive revenue recognition
        - Collection issues with existing receivables
        
        However, this analysis now incorporates DSO context to reduce false
        positives. A divergence with healthy DSO levels is likely seasonal
        or timing-related, not a quality concern. For example:
        - Holiday-heavy businesses see AR spike at fiscal year end
        - Q4 sales may not collect until Q1 of next year
        - Large enterprise deals often have longer payment terms
        
        The revised logic only flags divergences when BOTH conditions are met:
        1. AR growth exceeds revenue growth by more than 15 percentage points
        2. DSO is elevated (> 45 days), indicating actual collection concerns
        
        Returns:
            GrowthDivergence object if sufficient data, None otherwise.
        """
        # Get required data points for AR and revenue
        ar_current = self._get_field(StandardField.ACCOUNTS_RECEIVABLE, 0)
        ar_prior = self._get_field(StandardField.ACCOUNTS_RECEIVABLE, 1)
        rev_current = self._get_field(StandardField.REVENUE, 0)
        rev_prior = self._get_field(StandardField.REVENUE, 1)
        
        # Validate data availability
        if None in [ar_current, ar_prior, rev_current, rev_prior]:
            return None
        
        # Validate positive denominators
        if ar_prior <= 0 or rev_prior <= 0:
            return None
        
        # Calculate growth rates
        ar_growth = (ar_current - ar_prior) / ar_prior
        rev_growth = (rev_current - rev_prior) / rev_prior
        divergence = ar_growth - rev_growth
        
        # Calculate current DSO for context
        # DSO = (AR / Revenue) * 365
        dso_current = (ar_current / rev_current) * 365.0 if rev_current > 0 else 0.0
        
        # Get configured threshold (default increased from 10% to 15%)
        base_threshold = getattr(
            EARNINGS_QUALITY, 'ar_vs_revenue_threshold', DEFAULT_AR_VS_REVENUE_THRESHOLD
        )
        # Use a more lenient threshold
        threshold = max(base_threshold, 0.15)
        
        # Determine if this should be flagged based on BOTH divergence AND DSO
        # Only flag if divergence exceeds threshold AND DSO is elevated
        divergence_exceeds_threshold = divergence > threshold
        dso_is_elevated = dso_current > 45.0
        
        # Flag only when both conditions indicate a real concern
        is_flagged = divergence_exceeds_threshold and dso_is_elevated
        
        # Generate context-aware interpretation
        if divergence > threshold:
            if dso_current > 60:
                # High divergence with high DSO - genuine concern
                interpretation = (
                    f"AR growth ({ar_growth*100:.1f}%) exceeds revenue growth "
                    f"({rev_growth*100:.1f}%) by {divergence*100:.1f}pp with elevated "
                    f"DSO of {dso_current:.0f} days. This combination suggests "
                    f"potential collection issues or aggressive revenue recognition "
                    f"that warrants investigation."
                )
            elif dso_current > 45:
                # High divergence with moderate DSO - monitor
                interpretation = (
                    f"AR growth ({ar_growth*100:.1f}%) exceeds revenue growth "
                    f"({rev_growth*100:.1f}%) by {divergence*100:.1f}pp with moderate "
                    f"DSO of {dso_current:.0f} days. Monitor for trends but may be "
                    f"attributable to sales timing or customer mix changes."
                )
            else:
                # High divergence but healthy DSO - likely timing
                interpretation = (
                    f"AR growth ({ar_growth*100:.1f}%) exceeds revenue growth "
                    f"({rev_growth*100:.1f}%) by {divergence*100:.1f}pp, but DSO "
                    f"remains healthy at {dso_current:.0f} days. Likely attributable "
                    f"to seasonal timing or fiscal year-end sales patterns rather "
                    f"than underlying quality concerns."
                )
        elif divergence > 0:
            interpretation = (
                f"AR growth ({ar_growth*100:.1f}%) modestly exceeds revenue growth "
                f"({rev_growth*100:.1f}%) by {divergence*100:.1f}pp with DSO of "
                f"{dso_current:.0f} days - within acceptable range."
            )
        else:
            interpretation = (
                f"AR growth ({ar_growth*100:.1f}%) at or below revenue growth "
                f"({rev_growth*100:.1f}%) with DSO of {dso_current:.0f} days - "
                f"positive quality indicator showing good collection efficiency."
            )
        
        return GrowthDivergence(
            metric_name="AR Growth vs Revenue Growth",
            growth_rate_1=ar_growth,
            growth_rate_2=rev_growth,
            divergence=divergence,
            is_flagged=is_flagged,
            threshold=threshold,
            interpretation=interpretation
        )
    
    def _analyze_inventory_cogs_divergence(self) -> Optional[GrowthDivergence]:
        """
        Analyze divergence between Inventory and Cost of Revenue growth.
        
        When inventory grows significantly faster than COGS, it may indicate:
        - Inventory obsolescence risk
        - Demand softening
        - Supply chain issues
        
        Returns:
            GrowthDivergence object if sufficient data, None otherwise.
        """
        # Get required data points
        inv_current = self._get_field(StandardField.INVENTORY, 0)
        inv_prior = self._get_field(StandardField.INVENTORY, 1)
        cogs_current = self._get_field(StandardField.COST_OF_REVENUE, 0)
        cogs_prior = self._get_field(StandardField.COST_OF_REVENUE, 1)
        
        # Validate data availability
        if inv_current is None or inv_prior is None:
            return None
        if cogs_current is None or cogs_prior is None:
            return None
        
        # Validate positive denominators
        if inv_prior <= 0 or cogs_prior <= 0:
            return None
        
        # Calculate growth rates
        inv_growth = (inv_current - inv_prior) / inv_prior
        cogs_growth = (cogs_current - cogs_prior) / cogs_prior
        divergence = inv_growth - cogs_growth
        
        # Get threshold
        threshold = getattr(
            EARNINGS_QUALITY, 'inventory_vs_cogs_threshold',
            DEFAULT_INVENTORY_VS_COGS_THRESHOLD
        )
        is_flagged = divergence > threshold
        
        # Generate interpretation
        if is_flagged:
            interpretation = (
                f"Inventory growth ({inv_growth*100:.1f}%) significantly exceeds "
                f"COGS growth ({cogs_growth*100:.1f}%) by {divergence*100:.1f}pp - "
                f"may indicate inventory buildup or potential obsolescence risk"
            )
        elif divergence > 0:
            interpretation = (
                f"Inventory growth ({inv_growth*100:.1f}%) modestly exceeds COGS "
                f"growth ({cogs_growth*100:.1f}%) - within acceptable range"
            )
        else:
            interpretation = (
                f"Inventory growth ({inv_growth*100:.1f}%) at or below COGS growth "
                f"({cogs_growth*100:.1f}%) - efficient inventory management"
            )
        
        return GrowthDivergence(
            metric_name="Inventory Growth vs COGS Growth",
            growth_rate_1=inv_growth,
            growth_rate_2=cogs_growth,
            divergence=divergence,
            is_flagged=is_flagged,
            threshold=threshold,
            interpretation=interpretation
        )
    
    def _analyze_payables_divergence(self) -> Optional[GrowthDivergence]:
        """
        Analyze divergence between Accounts Payable and Cost of Revenue growth.
        
        When payables grow significantly faster than COGS, it may indicate:
        - Cash flow stress (stretching payments)
        - Aggressive working capital management
        - Potential liquidity concerns
        
        Returns:
            GrowthDivergence object if sufficient data, None otherwise.
        """
        # Get required data points
        ap_current = self._get_field(StandardField.ACCOUNTS_PAYABLE, 0)
        ap_prior = self._get_field(StandardField.ACCOUNTS_PAYABLE, 1)
        cogs_current = self._get_field(StandardField.COST_OF_REVENUE, 0)
        cogs_prior = self._get_field(StandardField.COST_OF_REVENUE, 1)
        
        # Validate data availability
        if None in [ap_current, ap_prior, cogs_current, cogs_prior]:
            return None
        
        # Validate positive denominators
        if ap_prior <= 0 or cogs_prior <= 0:
            return None
        
        # Calculate growth rates
        ap_growth = (ap_current - ap_prior) / ap_prior
        cogs_growth = (cogs_current - cogs_prior) / cogs_prior
        divergence = ap_growth - cogs_growth
        
        # Use configurable threshold
        threshold = DEFAULT_PAYABLES_VS_COGS_THRESHOLD
        is_flagged = divergence > threshold
        
        # Generate interpretation
        if is_flagged:
            interpretation = (
                f"Payables growth ({ap_growth*100:.1f}%) significantly exceeds COGS "
                f"growth ({cogs_growth*100:.1f}%) by {divergence*100:.1f}pp - may "
                f"indicate cash flow stress or aggressive working capital management"
            )
        elif divergence > 0.05:
            interpretation = (
                f"Payables growth ({ap_growth*100:.1f}%) modestly exceeds COGS growth "
                f"({cogs_growth*100:.1f}%) - monitor for trends"
            )
        else:
            interpretation = (
                f"Payables growth ({ap_growth*100:.1f}%) aligned with COGS growth "
                f"({cogs_growth*100:.1f}%) - normal payment patterns"
            )
        
        return GrowthDivergence(
            metric_name="Payables Growth vs COGS Growth",
            growth_rate_1=ap_growth,
            growth_rate_2=cogs_growth,
            divergence=divergence,
            is_flagged=is_flagged,
            threshold=threshold,
            interpretation=interpretation
        )
    
    # =========================================================================
    # RED FLAG DETECTION
    # =========================================================================
    
    def _detect_red_flags(
        self,
        accruals: AccrualsAnalysis,
        cash_conversion: float,
        divergences: List[GrowthDivergence]
    ) -> None:
        """
        Detect earnings quality red flags across all analysis dimensions.
        
        Args:
            accruals: Results from accruals analysis.
            cash_conversion: Calculated cash conversion rate.
            divergences: List of growth divergence analyses.
        """
        self._detect_accruals_flags(accruals)
        self._detect_cash_conversion_flags(cash_conversion)
        self._detect_divergence_flags(divergences)
        self._detect_pattern_flags()
    
    def _detect_accruals_flags(self, accruals: AccrualsAnalysis) -> None:
        """
        Detect red flags related to accruals.
        
        Flags are raised for:
        - High accruals ratio (above low quality threshold)
        - Deteriorating accruals trend over time
        
        Args:
            accruals: AccrualsAnalysis object with calculated metrics.
        """
        ratio = accruals.accruals_ratio
        low_quality = getattr(
            EARNINGS_QUALITY, 'accruals_low_quality', DEFAULT_ACCRUALS_LOW_QUALITY
        )
        
        # Flag high accruals ratio
        if ratio > low_quality:
            severity = RedFlagSeverity.HIGH if ratio > 0.20 else RedFlagSeverity.MEDIUM
            self._red_flags.append(RedFlag(
                category=RedFlagCategory.ACCRUALS,
                severity=severity,
                title="High Accruals Ratio",
                description=(
                    f"Accruals ratio of {ratio*100:.1f}% indicates significant gap "
                    f"between reported earnings and cash generation"
                ),
                metric_value=ratio,
                threshold_value=low_quality,
                recommendation="Investigate components of non-cash earnings"
            ))
        
        # Check for deteriorating trend
        if len(accruals.historical_trend) >= 3:
            recent_avg = float(np.mean(accruals.historical_trend[:2]))
            older_avg = float(np.mean(accruals.historical_trend[2:]))
            deterioration = recent_avg - older_avg
            
            if deterioration > DEFAULT_ACCRUALS_DETERIORATION_THRESHOLD:
                self._red_flags.append(RedFlag(
                    category=RedFlagCategory.ACCRUALS,
                    severity=RedFlagSeverity.MEDIUM,
                    title="Deteriorating Accruals Trend",
                    description=(
                        f"Accruals ratio has increased from {older_avg*100:.1f}% "
                        f"to {recent_avg*100:.1f}% over recent periods"
                    ),
                    metric_value=deterioration,
                    threshold_value=DEFAULT_ACCRUALS_DETERIORATION_THRESHOLD,
                    recommendation="Investigate causes of rising accruals"
                ))
    
    def _detect_cash_conversion_flags(self, cash_conversion: float) -> None:
        """
        Detect red flags related to cash conversion.
        
        Flags are raised for:
        - Low cash conversion (below acceptable threshold)
        - Negative OCF despite positive net income (critical)
        
        Args:
            cash_conversion: Calculated cash conversion rate.
        """
        acceptable = getattr(
            EARNINGS_QUALITY, 'cash_conversion_acceptable',
            DEFAULT_CASH_CONVERSION_ACCEPTABLE
        )
        
        # Flag low cash conversion
        if 0 < cash_conversion < acceptable:
            severity = RedFlagSeverity.HIGH if cash_conversion < 0.50 else RedFlagSeverity.MEDIUM
            self._red_flags.append(RedFlag(
                category=RedFlagCategory.CASH_FLOW,
                severity=severity,
                title="Low Cash Conversion",
                description=(
                    f"Cash conversion rate of {cash_conversion*100:.0f}% indicates "
                    f"earnings are not converting well to operating cash flow"
                ),
                metric_value=cash_conversion,
                threshold_value=acceptable,
                recommendation="Analyze working capital changes and non-cash items"
            ))
        
        # Check for critical case: positive NI but negative OCF
        ni = self._get_field(StandardField.NET_INCOME, 0)
        ocf = self._get_field(StandardField.OPERATING_CASH_FLOW, 0)
        
        if ni is not None and ocf is not None:
            if ni > 0 and ocf < 0:
                self._red_flags.append(RedFlag(
                    category=RedFlagCategory.CASH_FLOW,
                    severity=RedFlagSeverity.CRITICAL,
                    title="Negative OCF Despite Profit",
                    description=(
                        f"Company reports net income of ${ni:.0f}M but operating "
                        f"cash flow is negative ${abs(ocf):.0f}M - fundamental "
                        f"disconnect between earnings and cash generation"
                    ),
                    metric_value=ocf,
                    threshold_value=0.0,
                    recommendation="Immediate investigation required - review working "
                                   "capital changes, non-cash items, and accruals"
                ))
    
    def _detect_divergence_flags(self, divergences: List[GrowthDivergence]) -> None:
        """
        Detect red flags from growth divergences.
        
        Converts flagged divergences into red flag objects with appropriate
        severity levels based on the magnitude of divergence.
        
        Args:
            divergences: List of GrowthDivergence analysis results.
        """
        for div in divergences:
            if not div.is_flagged:
                continue
            
            # Determine severity based on divergence magnitude
            if div.divergence > 0.25:
                severity = RedFlagSeverity.HIGH
            elif div.divergence > 0.15:
                severity = RedFlagSeverity.MEDIUM
            else:
                severity = RedFlagSeverity.LOW
            
            # Determine category based on metric type
            if "AR" in div.metric_name:
                category = RedFlagCategory.REVENUE_RECOGNITION
            elif "Inventory" in div.metric_name:
                category = RedFlagCategory.WORKING_CAPITAL
            else:
                category = RedFlagCategory.WORKING_CAPITAL
            
            self._red_flags.append(RedFlag(
                category=category,
                severity=severity,
                title=f"Abnormal {div.metric_name}",
                description=div.interpretation,
                metric_value=div.divergence,
                threshold_value=div.threshold,
                recommendation=f"Investigate drivers of {div.metric_name.lower()}"
            ))
    
    def _detect_pattern_flags(self) -> None:
        """
        Detect additional pattern-based red flags.
        
        Looks for unusual patterns such as:
        - Strong revenue growth without asset growth
        - Volatile gross margins
        """
        # Check revenue vs asset growth pattern
        rev_current = self._get_field(StandardField.REVENUE, 0)
        rev_prior = self._get_field(StandardField.REVENUE, 1)
        assets_current = self._get_field(StandardField.TOTAL_ASSETS, 0)
        assets_prior = self._get_field(StandardField.TOTAL_ASSETS, 1)
        
        if all(v is not None and v > 0 for v in 
               [rev_current, rev_prior, assets_current, assets_prior]):
            rev_growth = (rev_current - rev_prior) / rev_prior
            asset_growth = (assets_current - assets_prior) / assets_prior
            
            # Flag if strong revenue growth but declining assets
            if rev_growth > 0.20 and asset_growth < 0:
                self._red_flags.append(RedFlag(
                    category=RedFlagCategory.ACCOUNTING_POLICY,
                    severity=RedFlagSeverity.MEDIUM,
                    title="Revenue Growth Without Asset Growth",
                    description=(
                        f"Revenue grew {rev_growth*100:.1f}% while assets declined "
                        f"{asset_growth*100:.1f}% - unusual pattern that may warrant "
                        f"investigation of revenue recognition policies"
                    ),
                    metric_value=rev_growth - asset_growth,
                    recommendation="Verify revenue recognition policies and asset disposals"
                ))
        
        # Check gross margin volatility
        gm_trend = self._calculate_margin_trend(
            StandardField.GROSS_PROFIT, StandardField.REVENUE
        )
        if len(gm_trend) >= 3:
            gm_std = float(np.std(gm_trend))
            if gm_std > DEFAULT_GROSS_MARGIN_VOLATILITY_THRESHOLD:
                self._red_flags.append(RedFlag(
                    category=RedFlagCategory.ACCOUNTING_POLICY,
                    severity=RedFlagSeverity.LOW,
                    title="Volatile Gross Margins",
                    description=(
                        f"Gross margin standard deviation of {gm_std*100:.1f}pp "
                        f"indicates potential revenue or cost recognition volatility"
                    ),
                    metric_value=gm_std,
                    threshold_value=DEFAULT_GROSS_MARGIN_VOLATILITY_THRESHOLD,
                    recommendation="Analyze drivers of margin volatility"
                ))
    
    def _calculate_margin_trend(
        self,
        numerator_field: StandardField,
        denominator_field: StandardField
    ) -> List[float]:
        """
        Calculate margin trend across available periods.
        
        Args:
            numerator_field: Field to use as numerator (e.g., gross profit).
            denominator_field: Field to use as denominator (e.g., revenue).
        
        Returns:
            List of margin ratios from most recent to oldest period.
        """
        margins: List[float] = []
        periods = self._data.statements.periods
        
        for i in range(len(periods)):
            num = self._get_field(numerator_field, i)
            den = self._get_field(denominator_field, i)
            
            if num is not None and den is not None and den > 0:
                margins.append(num / den)
        
        return margins
    
    # =========================================================================
    # POSITIVE INDICATORS
    # =========================================================================
    
    def _identify_positive_indicators(
        self,
        accruals: AccrualsAnalysis,
        cash_conversion: float
    ) -> None:
        """
        Identify positive earnings quality indicators.
        
        These are signals that support the quality and sustainability of
        reported earnings.
        
        Args:
            accruals: Results from accruals analysis.
            cash_conversion: Calculated cash conversion rate.
        """
        excellent = getattr(
            EARNINGS_QUALITY, 'cash_conversion_excellent',
            DEFAULT_CASH_CONVERSION_EXCELLENT
        )
        high_quality = getattr(
            EARNINGS_QUALITY, 'accruals_high_quality',
            DEFAULT_ACCRUALS_HIGH_QUALITY
        )
        
        # Negative accruals indicate OCF > NI
        if accruals.accruals_ratio < 0:
            self._positive_indicators.append(
                "Operating cash flow exceeds net income - earnings are cash-backed"
            )
        
        # Excellent cash conversion
        if cash_conversion >= excellent:
            self._positive_indicators.append(
                f"Excellent cash conversion of {cash_conversion*100:.0f}% indicates "
                f"high-quality earnings"
            )
        
        # Low accruals ratio
        if 0 <= accruals.accruals_ratio <= high_quality:
            self._positive_indicators.append(
                f"Low accruals ratio of {accruals.accruals_ratio*100:.1f}% indicates "
                f"earnings are well-supported by cash"
            )
        
        # Improving accruals trend
        if len(accruals.historical_trend) >= 3:
            recent = accruals.historical_trend[0]
            older = accruals.historical_trend[-1]
            if recent < older - 0.02:
                self._positive_indicators.append(
                    f"Accruals ratio improved from {older*100:.1f}% to {recent*100:.1f}%"
                )
        
        # Stable gross margins
        gm_trend = self._calculate_margin_trend(
            StandardField.GROSS_PROFIT, StandardField.REVENUE
        )
        if len(gm_trend) >= 3:
            gm_std = float(np.std(gm_trend))
            if gm_std < 0.02:
                self._positive_indicators.append(
                    f"Consistent gross margins (std dev {gm_std*100:.1f}pp) indicate "
                    f"stable revenue recognition"
                )
    
    # =========================================================================
    # QUALITY SCORING
    # =========================================================================
    
    def _calculate_quality_scores(
        self,
        accruals: AccrualsAnalysis,
        cash_conversion: float,
        divergences: List[GrowthDivergence]
    ) -> List[QualityScore]:
        """
        Calculate component quality scores.
        
        Scoring weights:
        - Accruals Quality: 35%
        - Cash Conversion: 35%
        - Growth Consistency: 20%
        - Red Flag Assessment: 10%
        
        Args:
            accruals: Results from accruals analysis.
            cash_conversion: Calculated cash conversion rate.
            divergences: List of growth divergence analyses.
        
        Returns:
            List of QualityScore objects for each component.
        """
        scores: List[QualityScore] = []
        
        # Accruals quality score (35% weight)
        accruals_score = self._score_accruals(accruals.accruals_ratio)
        scores.append(QualityScore(
            component="Accruals Quality",
            score=accruals_score,
            weight=0.35,
            weighted_score=accruals_score * 0.35,
            explanation=f"Based on accruals ratio of {accruals.accruals_ratio*100:.1f}%"
        ))
        
        # Cash conversion score (35% weight)
        conversion_score = self._score_cash_conversion(cash_conversion)
        scores.append(QualityScore(
            component="Cash Conversion",
            score=conversion_score,
            weight=0.35,
            weighted_score=conversion_score * 0.35,
            explanation=f"Based on cash conversion rate of {cash_conversion*100:.0f}%"
        ))
        
        # Growth consistency score (20% weight)
        divergence_score = self._score_divergences(divergences)
        flagged_count = len([d for d in divergences if d.is_flagged])
        scores.append(QualityScore(
            component="Growth Consistency",
            score=divergence_score,
            weight=0.20,
            weighted_score=divergence_score * 0.20,
            explanation=f"Based on {flagged_count} flagged divergences"
        ))
        
        # Red flag assessment score (10% weight)
        flag_score = self._score_red_flags()
        scores.append(QualityScore(
            component="Red Flag Assessment",
            score=flag_score,
            weight=0.10,
            weighted_score=flag_score * 0.10,
            explanation=f"Based on {len(self._red_flags)} identified red flags"
        ))
        
        return scores
    
    def _score_accruals(self, accruals_ratio: float) -> float:
        """
        Score accruals quality from 0-100.
        
        Scoring logic:
        - Negative ratio: 90-100 (excellent)
        - <5%: 80-90 (high quality)
        - 5-10%: 60-80 (moderate)
        - 10-15%: 40-60 (low quality)
        - >15%: 0-40 (concern)
        
        Args:
            accruals_ratio: Calculated accruals ratio.
        
        Returns:
            Score from 0-100.
        """
        high_quality = getattr(
            EARNINGS_QUALITY, 'accruals_high_quality', DEFAULT_ACCRUALS_HIGH_QUALITY
        )
        
        if accruals_ratio < 0:
            # Negative accruals = OCF > NI = excellent
            return min(100.0, 90.0 + abs(accruals_ratio) * 100.0)
        
        if accruals_ratio <= high_quality:
            # Scale from 90 at 0% to 80 at threshold
            return 90.0 - (accruals_ratio / high_quality) * 10.0
        elif accruals_ratio <= 0.10:
            # Scale from 80 at 5% to 60 at 10%
            return 80.0 - ((accruals_ratio - 0.05) / 0.05) * 20.0
        elif accruals_ratio <= 0.15:
            # Scale from 60 at 10% to 40 at 15%
            return 60.0 - ((accruals_ratio - 0.10) / 0.05) * 20.0
        else:
            # Scale from 40 at 15% to 0 at 25%
            return max(0.0, 40.0 - ((accruals_ratio - 0.15) / 0.10) * 40.0)
    
    def _score_cash_conversion(self, conversion_rate: float) -> float:
        """
        Score cash conversion from 0-100.
        
        Scoring logic:
        - >=120%: 100
        - 110-120%: 90-100
        - 90-110%: 75-90
        - 70-90%: 50-75
        - 50-70%: 25-50
        - <50%: 0-25
        
        Args:
            conversion_rate: Cash conversion rate (OCF / NI).
        
        Returns:
            Score from 0-100.
        """
        excellent = getattr(
            EARNINGS_QUALITY, 'cash_conversion_excellent',
            DEFAULT_CASH_CONVERSION_EXCELLENT
        )
        good = getattr(
            EARNINGS_QUALITY, 'cash_conversion_good',
            DEFAULT_CASH_CONVERSION_GOOD
        )
        acceptable = getattr(
            EARNINGS_QUALITY, 'cash_conversion_acceptable',
            DEFAULT_CASH_CONVERSION_ACCEPTABLE
        )
        
        if conversion_rate >= 1.20:
            return 100.0
        elif conversion_rate >= excellent:
            return 90.0 + (conversion_rate - 1.10) * 100.0
        elif conversion_rate >= good:
            return 75.0 + (conversion_rate - 0.90) * 75.0
        elif conversion_rate >= acceptable:
            return 50.0 + (conversion_rate - 0.70) * 125.0
        elif conversion_rate >= 0.50:
            return 25.0 + (conversion_rate - 0.50) * 125.0
        else:
            return max(0.0, conversion_rate * 50.0)
    
    def _score_divergences(self, divergences: List[GrowthDivergence]) -> float:
        """
        Score based on growth divergences from 0-100.
        
        Args:
            divergences: List of GrowthDivergence analyses.
        
        Returns:
            Score from 0-100.
        """
        if not divergences:
            # No divergence data available - neutral score
            return 75.0
        
        flagged_count = sum(1 for d in divergences if d.is_flagged)
        total_count = len(divergences)
        
        # Base score: reduce by proportion of flagged divergences
        score = 100.0 - (flagged_count / total_count) * 50.0
        
        # Additional penalty for severe divergences
        for div in divergences:
            if div.is_flagged and div.divergence > 0.25:
                score -= 15.0
        
        return max(0.0, score)
    
    def _score_red_flags(self) -> float:
        """
        Score based on red flag count and severity from 0-100.
        
        Penalty by severity:
        - CRITICAL: -30 points each
        - HIGH: -20 points each
        - MEDIUM: -10 points each
        - LOW: -5 points each
        
        Returns:
            Score from 0-100.
        """
        if not self._red_flags:
            return 100.0
        
        score = 100.0
        for flag in self._red_flags:
            if flag.severity == RedFlagSeverity.CRITICAL:
                score -= 30.0
            elif flag.severity == RedFlagSeverity.HIGH:
                score -= 20.0
            elif flag.severity == RedFlagSeverity.MEDIUM:
                score -= 10.0
            else:
                score -= 5.0
        
        return max(0.0, score)
    
    def _calculate_overall_score(self, scores: List[QualityScore]) -> float:
        """
        Calculate weighted overall score.
        
        Args:
            scores: List of component QualityScore objects.
        
        Returns:
            Overall weighted score from 0-100.
        """
        return sum(s.weighted_score for s in scores)
    
    def _determine_overall_rating(
        self,
        overall_score: float,
        accruals: AccrualsAnalysis,
        red_flag_count: int
    ) -> EarningsQualityRating:
        """
        Determine overall earnings quality rating.
        
        Rating logic:
        - Any CRITICAL flag: CONCERN
        - 2+ HIGH flags: LOW or CONCERN
        - Score >= 80: HIGH
        - Score 60-80: MODERATE
        - Score 40-60: LOW
        - Score < 40: CONCERN
        
        Args:
            overall_score: Calculated overall score.
            accruals: AccrualsAnalysis results.
            red_flag_count: Number of red flags identified.
        
        Returns:
            EarningsQualityRating enum value.
        """
        # Check for critical flags - automatically CONCERN
        critical_flags = [
            f for f in self._red_flags if f.severity == RedFlagSeverity.CRITICAL
        ]
        if critical_flags:
            return EarningsQualityRating.CONCERN
        
        # Check for multiple high-severity flags
        high_flags = [
            f for f in self._red_flags if f.severity == RedFlagSeverity.HIGH
        ]
        if len(high_flags) >= 2:
            if overall_score >= 60:
                return EarningsQualityRating.LOW
            else:
                return EarningsQualityRating.CONCERN
        
        # Score-based rating
        if overall_score >= 80:
            return EarningsQualityRating.HIGH
        elif overall_score >= 60:
            return EarningsQualityRating.MODERATE
        elif overall_score >= 40:
            return EarningsQualityRating.LOW
        else:
            return EarningsQualityRating.CONCERN
    
    # =========================================================================
    # INSIGHTS GENERATION
    # =========================================================================
    
    def _generate_insights(
        self,
        accruals: AccrualsAnalysis,
        cash_conversion: float,
        divergences: List[GrowthDivergence],
        rating: EarningsQualityRating
    ) -> List[str]:
        """
        Generate key insights from the analysis.
        
        Insights provide a narrative summary of the most important findings
        from the earnings quality analysis.
        
        Args:
            accruals: Results from accruals analysis.
            cash_conversion: Calculated cash conversion rate.
            divergences: List of growth divergence analyses.
            rating: Overall quality rating.
        
        Returns:
            List of insight strings.
        """
        insights: List[str] = []
        
        # Overall rating insight
        if rating == EarningsQualityRating.HIGH:
            insights.append(
                "Earnings quality is HIGH - profits are well-supported by cash flow "
                "with minimal concerning indicators"
            )
        elif rating == EarningsQualityRating.CONCERN:
            insights.append(
                "Earnings quality is a CONCERN - significant gap between reported "
                "profits and cash generation warrants investigation"
            )
        
        # Accruals insights
        if accruals.accruals_ratio < 0:
            insights.append(
                f"Negative accruals ratio ({accruals.accruals_ratio*100:.1f}%) indicates "
                f"operating cash flow exceeds net income - a positive quality signal"
            )
        elif accruals.accruals_ratio > 0.10:
            insights.append(
                f"Elevated accruals ratio ({accruals.accruals_ratio*100:.1f}%) suggests "
                f"a meaningful portion of earnings is not backed by cash"
            )
        
        # Cash conversion insights
        excellent = getattr(
            EARNINGS_QUALITY, 'cash_conversion_excellent',
            DEFAULT_CASH_CONVERSION_EXCELLENT
        )
        acceptable = getattr(
            EARNINGS_QUALITY, 'cash_conversion_acceptable',
            DEFAULT_CASH_CONVERSION_ACCEPTABLE
        )
        
        if cash_conversion >= excellent:
            insights.append(
                f"Strong cash conversion ({cash_conversion*100:.0f}%) demonstrates "
                f"earnings translate effectively into operating cash"
            )
        elif cash_conversion < acceptable:
            insights.append(
                f"Weak cash conversion ({cash_conversion*100:.0f}%) indicates "
                f"earnings are not converting well to cash - investigate causes"
            )
        
        # Divergence insights
        flagged_divergences = [d for d in divergences if d.is_flagged]
        if flagged_divergences:
            for div in flagged_divergences:
                insights.append(div.interpretation)
        elif divergences:
            insights.append(
                "No significant growth divergences detected - related metrics "
                "are growing in alignment"
            )
        
        # Red flag insights
        if self._red_flags:
            high_critical = [
                f for f in self._red_flags
                if f.severity in [RedFlagSeverity.HIGH, RedFlagSeverity.CRITICAL]
            ]
            if high_critical:
                insights.append(
                    f"{len(high_critical)} high-severity red flag(s) identified - "
                    f"recommend detailed investigation"
                )
        
        return insights
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _get_field(self, field: StandardField, period_index: int = 0) -> Optional[float]:
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
    
    def format_quality_summary(self, result: EarningsQualityResult) -> pd.DataFrame:
        """
        Format quality scores as a summary DataFrame.
        
        Args:
            result: EarningsQualityResult to format.
        
        Returns:
            DataFrame with component scores and overall summary.
        """
        data = []
        for score in result.quality_scores:
            data.append({
                'Component': score.component,
                'Score': f"{score.score:.0f}",
                'Weight': f"{score.weight*100:.0f}%",
                'Weighted': f"{score.weighted_score:.1f}",
                'Explanation': score.explanation
            })
        
        # Add overall row
        data.append({
            'Component': 'OVERALL',
            'Score': f"{result.overall_score:.0f}",
            'Weight': '100%',
            'Weighted': f"{result.overall_score:.1f}",
            'Explanation': result.overall_rating.value
        })
        
        return pd.DataFrame(data)
    
    def format_red_flags_table(self, result: EarningsQualityResult) -> pd.DataFrame:
        """
        Format red flags as a DataFrame.
        
        Args:
            result: EarningsQualityResult containing red flags.
        
        Returns:
            DataFrame listing all red flags with details.
        """
        if not result.red_flags:
            return pd.DataFrame({'Message': ['No red flags identified']})
        
        data = []
        for flag in result.red_flags:
            description = flag.description
            if len(description) > 80:
                description = description[:77] + '...'
            
            data.append({
                'Severity': flag.severity.value,
                'Category': flag.category.value,
                'Title': flag.title,
                'Description': description
            })
        
        return pd.DataFrame(data)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_earnings_quality(processed_data: ProcessedData) -> EarningsQualityResult:
    """
    Convenience function to perform earnings quality analysis.
    
    Args:
        processed_data: ProcessedData object from DataProcessor.
    
    Returns:
        EarningsQualityResult with complete analysis.
    """
    analyzer = EarningsQualityAnalyzer(processed_data)
    return analyzer.analyze()


def get_earnings_quality_summary(result: EarningsQualityResult) -> Dict[str, Any]:
    """
    Extract key earnings quality metrics as a simple dictionary.
    
    Useful for quick access to summary metrics without parsing the full result.
    
    Args:
        result: EarningsQualityResult to summarize.
    
    Returns:
        Dictionary with key metrics.
    """
    return {
        'overall_rating': result.overall_rating.value,
        'overall_score': result.overall_score,
        'accruals_ratio': result.accruals_analysis.accruals_ratio,
        'cash_conversion_rate': result.cash_conversion_rate,
        'red_flag_count': len(result.red_flags),
        'critical_flags': len([
            f for f in result.red_flags if f.severity == RedFlagSeverity.CRITICAL
        ]),
        'high_flags': len([
            f for f in result.red_flags if f.severity == RedFlagSeverity.HIGH
        ])
    }


# =============================================================================
# MODULE SELF-TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    print()
    print("=" * 70)
    print(f"EARNINGS QUALITY ANALYZER MODULE TEST - {test_ticker}")
    print("=" * 70)
    print()
    
    try:
        print("Step 1: Collecting raw data...")
        from data_collector import DataCollector
        collector = DataCollector()
        raw_data = collector.collect(test_ticker)
        print(f"  Raw data collected: {raw_data.validation.years_available} years")
        print()
        
        print("Step 2: Processing data...")
        processor = DataProcessor()
        processed = processor.process(raw_data)
        print("  Processing complete")
        print()
        
        print("Step 3: Analyzing earnings quality...")
        analyzer = EarningsQualityAnalyzer(processed)
        result = analyzer.analyze()
        print("  Analysis complete")
        print()
        
        print("OVERALL ASSESSMENT")
        print("-" * 70)
        print(f"  Quality Rating:   {result.overall_rating.value}")
        print(f"  Quality Score:    {result.overall_score:.0f}/100")
        print(f"  Analysis Period:  {result.analysis_period}")
        
        print()
        print("ACCRUALS ANALYSIS")
        print("-" * 70)
        accruals = result.accruals_analysis
        print(f"  Total Accruals:        ${accruals.total_accruals:,.1f}M")
        print(f"  Average Total Assets:  ${accruals.average_total_assets:,.1f}M")
        print(f"  Accruals Ratio:        {accruals.accruals_ratio*100:.2f}%")
        print(f"  Rating:                {accruals.accruals_rating.value}")
        
        print()
        print("CASH CONVERSION")
        print("-" * 70)
        print(f"  Cash Conversion Rate:  {result.cash_conversion_rate*100:.1f}%")
        
        print()
        print("QUALITY SCORES")
        print("-" * 70)
        print(f"  {'Component':<25} {'Score':>8} {'Weight':>8} {'Weighted':>10}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10}")
        for score in result.quality_scores:
            print(f"  {score.component:<25} {score.score:>8.0f} "
                  f"{score.weight*100:>7.0f}% {score.weighted_score:>10.1f}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10}")
        print(f"  {'OVERALL':<25} {result.overall_score:>8.0f} "
              f"{'100':>7}% {result.overall_score:>10.1f}")
        
        if result.red_flags:
            print()
            print(f"RED FLAGS ({len(result.red_flags)} identified)")
            print("-" * 70)
            for flag in result.red_flags:
                print(f"  [{flag.severity.value.upper()}] {flag.title}")
        
        if result.positive_indicators:
            print()
            print("POSITIVE INDICATORS")
            print("-" * 70)
            for indicator in result.positive_indicators:
                print(f"  + {indicator}")
        
        print()
        print("=" * 70)
        print(f"Earnings quality analysis complete for {test_ticker}")
        print("=" * 70)
        print()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)