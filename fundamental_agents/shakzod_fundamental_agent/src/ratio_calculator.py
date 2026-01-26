"""
Financial Ratio Calculator Module for Fundamental Analyst Agent

This module calculates comprehensive financial ratios across five categories:
profitability, liquidity, solvency, efficiency, and market/valuation ratios.
These ratios form the quantitative foundation for fundamental analysis.

RATIO CATEGORIES
================

1. PROFITABILITY RATIOS
   Measure the company's ability to generate profits relative to sales,
   assets, and equity.
   
   - Gross Margin = Gross Profit / Revenue
   - Operating Margin = Operating Income / Revenue
   - Net Margin = Net Income / Revenue
   - ROE = Net Income / Shareholders' Equity
   - ROA = Net Income / Total Assets
   - ROIC = NOPAT / Invested Capital

2. LIQUIDITY RATIOS
   Measure the company's ability to meet short-term obligations.
   
   - Current Ratio = Current Assets / Current Liabilities
   - Quick Ratio = (Current Assets - Inventory) / Current Liabilities
   - Cash Ratio = Cash & Equivalents / Current Liabilities

3. SOLVENCY RATIOS
   Measure the company's ability to meet long-term obligations and
   financial leverage.
   
   - Debt-to-Equity = Total Debt / Shareholders' Equity
   - Debt-to-Assets = Total Debt / Total Assets
   - Interest Coverage = EBIT / Interest Expense
   - Debt-to-EBITDA = Total Debt / EBITDA

4. EFFICIENCY RATIOS
   Measure how effectively the company uses its assets.
   
   - Asset Turnover = Revenue / Average Total Assets
   - Inventory Turnover = COGS / Average Inventory
   - Receivables Turnover = Revenue / Average AR
   - Payables Turnover = COGS / Average AP

5. MARKET/VALUATION RATIOS
   Measure market valuation relative to fundamentals.
   
   - P/E = Price / EPS
   - P/B = Price / Book Value per Share
   - P/S = Market Cap / Revenue
   - EV/EBITDA = Enterprise Value / EBITDA
   - EV/Revenue = Enterprise Value / Revenue

DUPONT DECOMPOSITION
====================
ROE = Net Margin × Asset Turnover × Financial Leverage
    = (NI/Revenue) × (Revenue/Assets) × (Assets/Equity)

This decomposition reveals whether ROE is driven by:
- Profitability (Net Margin)
- Efficiency (Asset Turnover)
- Leverage (Financial structure)

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
from typing import Dict, List, Optional, Tuple, Any, Union
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
# SPECIAL COLLECTIONS
# =============================================================================

class RatioList(list):
    """
    A list that also supports dict-like .get() access.
    
    This class allows accessing ratios either as a list or via .get(key)
    where the key is matched against the ratio abbreviation.
    
    Used for backward compatibility with code that expects dict-like access.
    """
    
    def get(self, key: str, default=None):
        """
        Get a ratio value by abbreviation (dict-like access).
        
        Args:
            key: Ratio abbreviation (case-insensitive).
            default: Value to return if not found.
        
        Returns:
            The ratio's current_value or default if not found.
        """
        key_lower = key.lower().replace('_', '/')
        for item in self:
            if hasattr(item, 'abbreviation'):
                if item.abbreviation.lower() == key_lower:
                    return item.current_value if item.current_value is not None else default
        return default
    
    def __getitem__(self, key):
        """Support both integer indexing and string key access."""
        if isinstance(key, str):
            return self.get(key)
        return super().__getitem__(key)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to a plain dict keyed by lowercase abbreviation."""
        result = {}
        for item in self:
            if hasattr(item, 'abbreviation') and hasattr(item, 'current_value'):
                abbr = item.abbreviation.lower().replace('/', '_')
                result[abbr] = item.current_value if item.current_value is not None else 0.0
        return result


# =============================================================================
# CONSTANTS AND ENUMERATIONS
# =============================================================================

class RatioCategory(Enum):
    """Categories of financial ratios."""
    PROFITABILITY = "Profitability"
    LIQUIDITY = "Liquidity"
    SOLVENCY = "Solvency"
    EFFICIENCY = "Efficiency"
    VALUATION = "Valuation"


class RatioInterpretation(Enum):
    """Interpretation of ratio values."""
    STRONG = "Strong"
    HEALTHY = "Healthy"
    ADEQUATE = "Adequate"
    WEAK = "Weak"
    CONCERNING = "Concerning"
    NOT_APPLICABLE = "N/A"


class TrendAssessment(Enum):
    """Assessment of ratio trends."""
    IMPROVING = "Improving"
    STABLE = "Stable"
    DECLINING = "Declining"
    VOLATILE = "Volatile"
    INSUFFICIENT_DATA = "Insufficient Data"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FinancialRatio:
    """
    A single financial ratio with metadata and trend.
    
    Attributes:
        name: Full name of the ratio
        abbreviation: Short form (e.g., 'ROE')
        category: Category of the ratio
        current_value: Current period value
        prior_value: Prior period value
        change: Change from prior period
        trend: Historical values (most recent first)
        trend_assessment: Assessment of the trend
        interpretation: Quality interpretation
        formula: Formula description
        description: What the ratio measures
        is_percentage: Whether to display as percentage
        higher_is_better: Whether higher values are favorable
    """
    name: str
    abbreviation: str
    category: RatioCategory
    current_value: Optional[float]
    prior_value: Optional[float]
    change: Optional[float]
    trend: List[float]
    trend_assessment: TrendAssessment
    interpretation: RatioInterpretation
    formula: str
    description: str
    is_percentage: bool = True
    higher_is_better: bool = True


@dataclass
class DuPontAnalysis:
    """
    DuPont decomposition of Return on Equity.
    
    ROE = Net Margin × Asset Turnover × Equity Multiplier
    
    Attributes:
        roe: Return on Equity (the product)
        net_margin: Net Income / Revenue (profitability)
        asset_turnover: Revenue / Assets (efficiency)
        equity_multiplier: Assets / Equity (leverage)
        primary_driver: Which component drives ROE most
        analysis: Interpretation of the decomposition
    """
    roe: float
    net_margin: float
    asset_turnover: float
    equity_multiplier: float
    primary_driver: str
    analysis: str


@dataclass
class RatioComparison:
    """
    Year-over-year comparison of a ratio.
    
    Attributes:
        ratio_name: Name of the ratio
        periods: List of period labels
        values: List of values corresponding to periods
        cagr: Compound annual growth rate (if applicable)
        volatility: Standard deviation of values
        trend_description: Description of the trend
    """
    ratio_name: str
    periods: List[str]
    values: List[float]
    cagr: Optional[float]
    volatility: float
    trend_description: str


@dataclass
class RatioCalculatorResult:
    """
    Complete result of ratio calculations.
    
    This is the primary output of the RatioCalculator class.
    
    Attributes:
        ticker: Company ticker symbol
        analysis_period: Description of analysis period
        profitability_ratios: List of profitability ratios
        liquidity_ratios: List of liquidity ratios
        solvency_ratios: List of solvency ratios
        efficiency_ratios: List of efficiency ratios
        valuation_ratios: List of valuation ratios
        dupont_analysis: DuPont decomposition
        key_ratios_summary: Dictionary of key ratios for quick access
        ratio_comparisons: Year-over-year comparisons
        insights: Key insights from ratio analysis
        warnings: Data quality warnings
        analysis_timestamp: When analysis was performed
    """
    ticker: str
    analysis_period: str
    profitability_ratios: List[FinancialRatio]
    liquidity_ratios: List[FinancialRatio]
    solvency_ratios: List[FinancialRatio]
    efficiency_ratios: List[FinancialRatio]
    valuation_ratios: List[FinancialRatio]
    dupont_analysis: DuPontAnalysis
    key_ratios_summary: Dict[str, Optional[float]]
    ratio_comparisons: List[RatioComparison]
    insights: List[str]
    warnings: List[str]
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    
    def _ratios_list_to_dict(self, ratios: List[FinancialRatio]) -> Dict[str, float]:
        """
        Convert a list of FinancialRatio objects to a dict keyed by abbreviation.
        
        Args:
            ratios: List of FinancialRatio objects.
        
        Returns:
            Dict mapping lowercase abbreviation to current_value.
        """
        result = {}
        for ratio in ratios:
            # Use lowercase abbreviation as key for compatibility
            key = ratio.abbreviation.lower().replace('/', '_')
            result[key] = ratio.current_value if ratio.current_value is not None else 0.0
        return result
    
    def get(self, key: str, default=None):
        """
        Dict-like get method for compatibility with agent.py.
        
        Args:
            key: Attribute name to get.
            default: Default value if attribute not found.
        
        Returns:
            Attribute value or default.
        """
        return getattr(self, key, default)
    
    @property
    def profitability_ratios_dict(self) -> Dict[str, float]:
        """Return profitability ratios as a dict keyed by abbreviation."""
        return self._ratios_list_to_dict(self.profitability_ratios)
    
    @property
    def liquidity_ratios_dict(self) -> Dict[str, float]:
        """Return liquidity ratios as a dict keyed by abbreviation."""
        return self._ratios_list_to_dict(self.liquidity_ratios)
    
    @property
    def solvency_ratios_dict(self) -> Dict[str, float]:
        """Return solvency ratios as a dict keyed by abbreviation."""
        return self._ratios_list_to_dict(self.solvency_ratios)
    
    @property
    def efficiency_ratios_dict(self) -> Dict[str, float]:
        """Return efficiency ratios as a dict keyed by abbreviation."""
        return self._ratios_list_to_dict(self.efficiency_ratios)
    
    @property
    def growth_ratios(self) -> Dict[str, float]:
        """
        Return growth-related ratios as a dict for compatibility.
        
        Growth ratios are extracted from key_ratios_summary.
        """
        return {
            'revenue_growth': self.key_ratios_summary.get('Revenue Growth', 0.0) or 0.0,
            'ni_growth': self.key_ratios_summary.get('NI Growth', 0.0) or 0.0,
            'eps_growth': self.key_ratios_summary.get('EPS Growth', 0.0) or 0.0,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ticker": self.ticker,
            "analysis_period": self.analysis_period,
            "profitability_ratios": [self._ratio_to_dict(r) for r in self.profitability_ratios],
            "liquidity_ratios": [self._ratio_to_dict(r) for r in self.liquidity_ratios],
            "solvency_ratios": [self._ratio_to_dict(r) for r in self.solvency_ratios],
            "efficiency_ratios": [self._ratio_to_dict(r) for r in self.efficiency_ratios],
            "valuation_ratios": [self._ratio_to_dict(r) for r in self.valuation_ratios],
            "key_ratios_summary": self.key_ratios_summary,
            "insights": self.insights,
            "warnings": self.warnings,
            "analysis_timestamp": self.analysis_timestamp.isoformat()
        }
    
    def _ratio_to_dict(self, ratio: FinancialRatio) -> Dict[str, Any]:
        """Convert a single FinancialRatio to dict."""
        return {
            "name": ratio.name,
            "abbreviation": ratio.abbreviation,
            "category": ratio.category.value,
            "current_value": ratio.current_value,
            "prior_value": ratio.prior_value,
            "change": ratio.change,
            "trend_assessment": ratio.trend_assessment.value,
            "interpretation": ratio.interpretation.value
        }


# =============================================================================
# RATIO CALCULATOR CLASS
# =============================================================================

class RatioCalculator:
    """
    Comprehensive financial ratio calculator.
    
    This calculator computes ratios across all major categories and provides
    trend analysis, DuPont decomposition, and interpretive insights.
    
    Usage:
        calculator = RatioCalculator(processed_data)
        result = calculator.calculate()
        
        # Access specific ratios
        for ratio in result.profitability_ratios:
            print(f"{ratio.abbreviation}: {ratio.current_value:.1%}")
        
        # DuPont analysis
        print(f"ROE driven primarily by: {result.dupont_analysis.primary_driver}")
    
    Attributes:
        _data: ProcessedData object containing standardized financial data
        _processor: DataProcessor instance for field extraction
        _warnings: List of warning messages
    """
    
    def __init__(self, processed_data: ProcessedData):
        """
        Initialize the RatioCalculator.
        
        Args:
            processed_data: ProcessedData object from DataProcessor
        """
        self._data = processed_data
        self._processor = DataProcessor()
        self._warnings: List[str] = []
        
        logger.info(f"RatioCalculator initialized for {processed_data.ticker}")
    
    def calculate(self) -> RatioCalculatorResult:
        """
        Calculate all financial ratios.
        
        Returns:
            RatioCalculatorResult with complete analysis
        """
        logger.info(f"Starting ratio calculations for {self._data.ticker}")
        
        # Reset warnings
        self._warnings = []
        
        # Get analysis period
        periods = self._data.statements.periods
        analysis_period = self._format_analysis_period(periods)
        
        # Calculate ratio categories
        profitability_ratios = self._calculate_profitability_ratios()
        liquidity_ratios = self._calculate_liquidity_ratios()
        solvency_ratios = self._calculate_solvency_ratios()
        efficiency_ratios = self._calculate_efficiency_ratios()
        valuation_ratios = self._calculate_valuation_ratios()
        
        # DuPont analysis
        dupont = self._perform_dupont_analysis()
        
        # Build key ratios summary
        key_ratios = self._build_key_ratios_summary(
            profitability_ratios, liquidity_ratios, solvency_ratios,
            efficiency_ratios, valuation_ratios
        )
        
        # Year-over-year comparisons for key ratios
        comparisons = self._build_ratio_comparisons()
        
        # Generate insights
        insights = self._generate_insights(
            profitability_ratios, liquidity_ratios, solvency_ratios,
            efficiency_ratios, dupont
        )
        
        result = RatioCalculatorResult(
            ticker=self._data.ticker,
            analysis_period=analysis_period,
            profitability_ratios=RatioList(profitability_ratios),
            liquidity_ratios=RatioList(liquidity_ratios),
            solvency_ratios=RatioList(solvency_ratios),
            efficiency_ratios=RatioList(efficiency_ratios),
            valuation_ratios=RatioList(valuation_ratios),
            dupont_analysis=dupont,
            key_ratios_summary=key_ratios,
            ratio_comparisons=comparisons,
            insights=insights,
            warnings=self._warnings.copy()
        )
        
        logger.info(f"Ratio calculations complete for {self._data.ticker}")
        return result
    
    def calculate_all(self) -> RatioCalculatorResult:
        """
        Alias for calculate() - provides backward compatibility.
        
        Returns:
            RatioCalculatorResult with complete analysis
        """
        return self.calculate()
    
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
            # Fallback for unexpected types
            return f"FY{str(period)}"
    
    def _extract_year_from_period(self, period) -> str:
        """
        Extract the year from a period string or return as-is.
        
        Args:
            period: Period value (string like "2024" or "2024-09-30")
        
        Returns:
            Year string (e.g., "2024")
        """
        if isinstance(period, str):
            if period.startswith("FY"):
                return period[2:]  # Remove "FY" prefix
            elif "-" in period:
                return period.split("-")[0]
            else:
                return period
        else:
            # Fallback - convert to string
            return str(period)
    
    # =========================================================================
    # PROFITABILITY RATIOS
    # =========================================================================
    
    def _calculate_profitability_ratios(self) -> List[FinancialRatio]:
        """
        Calculate profitability ratios.
        
        Returns:
            List of FinancialRatio objects for profitability metrics
        """
        ratios = []
        
        # Gross Margin
        ratios.append(self._calculate_single_ratio(
            name="Gross Margin",
            abbreviation="GM",
            category=RatioCategory.PROFITABILITY,
            numerator_field=StandardField.GROSS_PROFIT,
            denominator_field=StandardField.REVENUE,
            formula="Gross Profit / Revenue",
            description="Percentage of revenue retained after direct costs",
            thresholds=(0.40, 0.30, 0.20, 0.10)  # Strong, Healthy, Adequate, Weak
        ))
        
        # Operating Margin
        ratios.append(self._calculate_single_ratio(
            name="Operating Margin",
            abbreviation="OM",
            category=RatioCategory.PROFITABILITY,
            numerator_field=StandardField.OPERATING_INCOME,
            denominator_field=StandardField.REVENUE,
            formula="Operating Income / Revenue",
            description="Percentage of revenue retained after operating expenses",
            thresholds=(0.20, 0.15, 0.10, 0.05)
        ))
        
        # Net Profit Margin
        ratios.append(self._calculate_single_ratio(
            name="Net Profit Margin",
            abbreviation="NPM",
            category=RatioCategory.PROFITABILITY,
            numerator_field=StandardField.NET_INCOME,
            denominator_field=StandardField.REVENUE,
            formula="Net Income / Revenue",
            description="Percentage of revenue retained as net profit",
            thresholds=(0.15, 0.10, 0.05, 0.02)
        ))
        
        # Return on Equity (ROE)
        ratios.append(self._calculate_return_ratio(
            name="Return on Equity",
            abbreviation="ROE",
            numerator_field=StandardField.NET_INCOME,
            denominator_field=StandardField.TOTAL_EQUITY,
            formula="Net Income / Shareholders' Equity",
            description="Return generated on shareholders' investment",
            thresholds=(0.20, 0.15, 0.10, 0.05)
        ))
        
        # Return on Assets (ROA)
        ratios.append(self._calculate_return_ratio(
            name="Return on Assets",
            abbreviation="ROA",
            numerator_field=StandardField.NET_INCOME,
            denominator_field=StandardField.TOTAL_ASSETS,
            formula="Net Income / Total Assets",
            description="Return generated on total asset base",
            thresholds=(0.10, 0.07, 0.04, 0.02)
        ))
        
        # Return on Invested Capital (ROIC)
        ratios.append(self._calculate_roic())
        
        return [r for r in ratios if r is not None]
    
    def _calculate_roic(self) -> Optional[FinancialRatio]:
        """
        Calculate Return on Invested Capital.
        
        ROIC = NOPAT / Average Invested Capital
        NOPAT = Operating Income × (1 - Tax Rate)
        Invested Capital = Total Equity + Total Debt - Cash
        
        Uses AVERAGE invested capital for consistency with ROE methodology.
        This provides a more accurate measure of return on capital employed
        throughout the period.
        
        Returns:
            FinancialRatio for ROIC
        """
        trend = []
        periods = self._data.statements.periods
        
        for i in range(len(periods)):
            operating_income = self._get_field(StandardField.OPERATING_INCOME, i)
            net_income = self._get_field(StandardField.NET_INCOME, i)
            pretax_income = self._get_field(StandardField.PRETAX_INCOME, i)
            
            # Current period balance sheet values
            total_equity_curr = self._get_field(StandardField.TOTAL_EQUITY, i)
            total_debt_curr = self._get_field(StandardField.TOTAL_DEBT, i)
            cash_curr = self._get_field(StandardField.CASH, i)
            
            # Prior period balance sheet values for averaging
            total_equity_prior = self._get_field(StandardField.TOTAL_EQUITY, i + 1)
            total_debt_prior = self._get_field(StandardField.TOTAL_DEBT, i + 1)
            cash_prior = self._get_field(StandardField.CASH, i + 1)
            
            if operating_income is None or total_equity_curr is None:
                break
            
            # Estimate tax rate from actual tax paid
            if pretax_income and pretax_income > 0 and net_income is not None:
                tax_rate = 1 - (net_income / pretax_income)
                tax_rate = max(0, min(0.40, tax_rate))  # Cap between 0-40%
            else:
                tax_rate = 0.21  # Default corporate rate
            
            # Calculate NOPAT
            nopat = operating_income * (1 - tax_rate)
            
            # Calculate current period Invested Capital
            debt_curr = total_debt_curr if total_debt_curr else 0
            cash_val_curr = cash_curr if cash_curr else 0
            ic_current = total_equity_curr + debt_curr - cash_val_curr
            
            # Use AVERAGE invested capital if prior period available (preferred)
            if total_equity_prior is not None:
                debt_prior = total_debt_prior if total_debt_prior else debt_curr
                cash_val_prior = cash_prior if cash_prior else cash_val_curr
                ic_prior = total_equity_prior + debt_prior - cash_val_prior
                invested_capital = (ic_current + ic_prior) / 2  # Average IC
            else:
                invested_capital = ic_current  # Fallback to current if no prior
            
            if invested_capital > 0:
                roic = nopat / invested_capital
                trend.append(roic)
            else:
                break
        
        if not trend:
            self._add_warning("Unable to calculate ROIC - insufficient data")
            return None
        
        current_value = trend[0]
        prior_value = trend[1] if len(trend) > 1 else None
        change = current_value - prior_value if prior_value is not None else None
        
        return FinancialRatio(
            name="Return on Invested Capital",
            abbreviation="ROIC",
            category=RatioCategory.PROFITABILITY,
            current_value=current_value,
            prior_value=prior_value,
            change=change,
            trend=trend,
            trend_assessment=self._assess_trend(trend, higher_is_better=True),
            interpretation=self._interpret_ratio(current_value, (0.15, 0.12, 0.08, 0.05)),
            formula="NOPAT / Average Invested Capital",
            description="Return on capital invested (using average IC for consistency with ROE)",
            is_percentage=True,
            higher_is_better=True
        )
    
    # =========================================================================
    # LIQUIDITY RATIOS
    # =========================================================================
    
    def _calculate_liquidity_ratios(self) -> List[FinancialRatio]:
        """
        Calculate liquidity ratios.
        
        Returns:
            List of FinancialRatio objects for liquidity metrics
        """
        ratios = []
        
        # Current Ratio
        ratios.append(self._calculate_single_ratio(
            name="Current Ratio",
            abbreviation="CR",
            category=RatioCategory.LIQUIDITY,
            numerator_field=StandardField.CURRENT_ASSETS,
            denominator_field=StandardField.CURRENT_LIABILITIES,
            formula="Current Assets / Current Liabilities",
            description="Ability to cover short-term obligations with short-term assets",
            thresholds=(2.0, 1.5, 1.0, 0.8),
            is_percentage=False
        ))
        
        # Quick Ratio
        ratios.append(self._calculate_quick_ratio())
        
        # Cash Ratio
        ratios.append(self._calculate_single_ratio(
            name="Cash Ratio",
            abbreviation="CashR",
            category=RatioCategory.LIQUIDITY,
            numerator_field=StandardField.CASH,
            denominator_field=StandardField.CURRENT_LIABILITIES,
            formula="Cash & Equivalents / Current Liabilities",
            description="Ability to cover short-term obligations with cash only",
            thresholds=(0.5, 0.3, 0.2, 0.1),
            is_percentage=False
        ))
        
        return [r for r in ratios if r is not None]
    
    def _calculate_quick_ratio(self) -> Optional[FinancialRatio]:
        """
        Calculate Quick Ratio (Acid Test).
        
        Quick Ratio = (Current Assets - Inventory) / Current Liabilities
        
        Returns:
            FinancialRatio for Quick Ratio
        """
        trend = []
        periods = self._data.statements.periods
        
        for i in range(len(periods)):
            current_assets = self._get_field(StandardField.CURRENT_ASSETS, i)
            inventory = self._get_field(StandardField.INVENTORY, i) or 0
            current_liab = self._get_field(StandardField.CURRENT_LIABILITIES, i)
            
            if current_assets is not None and current_liab is not None and current_liab > 0:
                quick_ratio = (current_assets - inventory) / current_liab
                trend.append(quick_ratio)
            else:
                break
        
        if not trend:
            return None
        
        current_value = trend[0]
        prior_value = trend[1] if len(trend) > 1 else None
        change = current_value - prior_value if prior_value is not None else None
        
        return FinancialRatio(
            name="Quick Ratio",
            abbreviation="QR",
            category=RatioCategory.LIQUIDITY,
            current_value=current_value,
            prior_value=prior_value,
            change=change,
            trend=trend,
            trend_assessment=self._assess_trend(trend, higher_is_better=True),
            interpretation=self._interpret_ratio(current_value, (1.5, 1.0, 0.8, 0.5)),
            formula="(Current Assets - Inventory) / Current Liabilities",
            description="Ability to cover short-term obligations without selling inventory",
            is_percentage=False,
            higher_is_better=True
        )
    
    # =========================================================================
    # SOLVENCY RATIOS
    # =========================================================================
    
    def _calculate_solvency_ratios(self) -> List[FinancialRatio]:
        """
        Calculate solvency ratios.
        
        Returns:
            List of FinancialRatio objects for solvency metrics
        """
        ratios = []
        
        # Debt-to-Equity
        ratios.append(self._calculate_single_ratio(
            name="Debt-to-Equity",
            abbreviation="D/E",
            category=RatioCategory.SOLVENCY,
            numerator_field=StandardField.TOTAL_DEBT,
            denominator_field=StandardField.TOTAL_EQUITY,
            formula="Total Debt / Shareholders' Equity",
            description="Financial leverage - debt relative to equity",
            thresholds=(0.5, 1.0, 1.5, 2.0),  # Lower is better for D/E
            is_percentage=False,
            higher_is_better=False
        ))
        
        # Debt-to-Assets
        ratios.append(self._calculate_single_ratio(
            name="Debt-to-Assets",
            abbreviation="D/A",
            category=RatioCategory.SOLVENCY,
            numerator_field=StandardField.TOTAL_DEBT,
            denominator_field=StandardField.TOTAL_ASSETS,
            formula="Total Debt / Total Assets",
            description="Proportion of assets financed by debt",
            thresholds=(0.20, 0.35, 0.50, 0.65),
            is_percentage=True,
            higher_is_better=False
        ))
        
        # Interest Coverage
        ratios.append(self._calculate_interest_coverage())
        
        # Debt-to-EBITDA
        ratios.append(self._calculate_debt_to_ebitda())
        
        return [r for r in ratios if r is not None]
    
    def _calculate_interest_coverage(self) -> Optional[FinancialRatio]:
        """
        Calculate Interest Coverage Ratio.
        
        Interest Coverage = EBIT / Interest Expense
        
        Returns:
            FinancialRatio for Interest Coverage, or None if interest expense data unavailable
        """
        trend = []
        periods = self._data.statements.periods
        
        for i in range(len(periods)):
            ebit = self._get_field(StandardField.OPERATING_INCOME, i)
            interest = self._get_field(StandardField.INTEREST_EXPENSE, i)
            
            if ebit is not None and interest is not None and interest > 0:
                coverage = ebit / interest
                trend.append(coverage)
            else:
                # If interest expense data is unavailable, stop calculating
                # Don't default to 100.0x as this can be misleading
                break
        
        if not trend:
            return None
        
        current_value = trend[0]
        prior_value = trend[1] if len(trend) > 1 else None
        change = current_value - prior_value if prior_value is not None else None
        
        return FinancialRatio(
            name="Interest Coverage",
            abbreviation="ICR",
            category=RatioCategory.SOLVENCY,
            current_value=current_value,
            prior_value=prior_value,
            change=change,
            trend=trend,
            trend_assessment=self._assess_trend(trend, higher_is_better=True),
            interpretation=self._interpret_ratio(current_value, (8.0, 5.0, 3.0, 1.5)),
            formula="EBIT / Interest Expense",
            description="Ability to cover interest payments from operating profit",
            is_percentage=False,
            higher_is_better=True
        )
    
    def _calculate_debt_to_ebitda(self) -> Optional[FinancialRatio]:
        """
        Calculate Debt-to-EBITDA Ratio.
        
        Debt/EBITDA = Total Debt / EBITDA
        
        Returns:
            FinancialRatio for Debt/EBITDA
        """
        trend = []
        periods = self._data.statements.periods
        
        for i in range(len(periods)):
            total_debt = self._get_field(StandardField.TOTAL_DEBT, i)
            ebitda = self._get_field(StandardField.EBITDA, i)
            
            if total_debt is not None and ebitda is not None and ebitda > 0:
                ratio = total_debt / ebitda
                trend.append(ratio)
            elif total_debt is not None and total_debt == 0:
                trend.append(0.0)  # No debt
            else:
                break
        
        if not trend:
            return None
        
        current_value = trend[0]
        prior_value = trend[1] if len(trend) > 1 else None
        change = current_value - prior_value if prior_value is not None else None
        
        return FinancialRatio(
            name="Debt-to-EBITDA",
            abbreviation="D/EBITDA",
            category=RatioCategory.SOLVENCY,
            current_value=current_value,
            prior_value=prior_value,
            change=change,
            trend=trend,
            trend_assessment=self._assess_trend(trend, higher_is_better=False),
            interpretation=self._interpret_ratio(
                current_value, (1.0, 2.0, 3.0, 4.0), higher_is_better=False
            ),
            formula="Total Debt / EBITDA",
            description="Years of EBITDA needed to pay off all debt",
            is_percentage=False,
            higher_is_better=False
        )
    
    # =========================================================================
    # EFFICIENCY RATIOS
    # =========================================================================
    
    def _calculate_efficiency_ratios(self) -> List[FinancialRatio]:
        """
        Calculate efficiency ratios.
        
        Returns:
            List of FinancialRatio objects for efficiency metrics
        """
        ratios = []
        
        # Asset Turnover
        ratios.append(self._calculate_turnover_ratio(
            name="Asset Turnover",
            abbreviation="AT",
            numerator_field=StandardField.REVENUE,
            denominator_field=StandardField.TOTAL_ASSETS,
            formula="Revenue / Average Total Assets",
            description="Revenue generated per dollar of assets",
            thresholds=(1.5, 1.0, 0.7, 0.5)
        ))
        
        # Inventory Turnover
        ratios.append(self._calculate_turnover_ratio(
            name="Inventory Turnover",
            abbreviation="InvT",
            numerator_field=StandardField.COST_OF_REVENUE,
            denominator_field=StandardField.INVENTORY,
            formula="Cost of Revenue / Average Inventory",
            description="Times inventory is sold and replaced per year",
            thresholds=(10.0, 6.0, 4.0, 2.0)
        ))
        
        # Receivables Turnover
        ratios.append(self._calculate_turnover_ratio(
            name="Receivables Turnover",
            abbreviation="ART",
            numerator_field=StandardField.REVENUE,
            denominator_field=StandardField.ACCOUNTS_RECEIVABLE,
            formula="Revenue / Average Accounts Receivable",
            description="Times receivables are collected per year",
            thresholds=(12.0, 8.0, 6.0, 4.0)
        ))
        
        # Payables Turnover
        ratios.append(self._calculate_turnover_ratio(
            name="Payables Turnover",
            abbreviation="APT",
            numerator_field=StandardField.COST_OF_REVENUE,
            denominator_field=StandardField.ACCOUNTS_PAYABLE,
            formula="Cost of Revenue / Average Accounts Payable",
            description="Times payables are paid per year",
            thresholds=(12.0, 8.0, 6.0, 4.0),
            higher_is_better=False  # Lower means longer payment terms
        ))
        
        return [r for r in ratios if r is not None]
    
    def _calculate_turnover_ratio(
        self,
        name: str,
        abbreviation: str,
        numerator_field: StandardField,
        denominator_field: StandardField,
        formula: str,
        description: str,
        thresholds: Tuple[float, ...],
        higher_is_better: bool = True
    ) -> Optional[FinancialRatio]:
        """
        Calculate a turnover ratio using average denominator.
        
        Args:
            name: Ratio name
            abbreviation: Short form
            numerator_field: Field for numerator (flow metric)
            denominator_field: Field for denominator (stock metric, averaged)
            formula: Formula description
            description: What the ratio measures
            thresholds: Interpretation thresholds
            higher_is_better: Whether higher values are favorable
            
        Returns:
            FinancialRatio object
        """
        trend = []
        periods = self._data.statements.periods
        
        for i in range(len(periods)):
            numerator = self._get_field(numerator_field, i)
            denom_current = self._get_field(denominator_field, i)
            denom_prior = self._get_field(denominator_field, i + 1)
            
            if numerator is None or denom_current is None:
                break
            
            # Use average if prior available
            if denom_prior is not None:
                avg_denom = (denom_current + denom_prior) / 2
            else:
                avg_denom = denom_current
            
            if avg_denom > 0:
                ratio = numerator / avg_denom
                trend.append(ratio)
            else:
                break
        
        if not trend:
            return None
        
        current_value = trend[0]
        prior_value = trend[1] if len(trend) > 1 else None
        change = current_value - prior_value if prior_value is not None else None
        
        return FinancialRatio(
            name=name,
            abbreviation=abbreviation,
            category=RatioCategory.EFFICIENCY,
            current_value=current_value,
            prior_value=prior_value,
            change=change,
            trend=trend,
            trend_assessment=self._assess_trend(trend, higher_is_better=higher_is_better),
            interpretation=self._interpret_ratio(current_value, thresholds, higher_is_better),
            formula=formula,
            description=description,
            is_percentage=False,
            higher_is_better=higher_is_better
        )
    
    # =========================================================================
    # VALUATION RATIOS
    # =========================================================================
    
    def _calculate_valuation_ratios(self) -> List[FinancialRatio]:
        """
        Calculate valuation ratios.
        
        Note: Market-based ratios require market data which may not be
        available in ProcessedData. These are calculated if data exists.
        
        Returns:
            List of FinancialRatio objects for valuation metrics
        """
        ratios = []
        
        # Get market data from company info if available (company_info is a dataclass)
        market_cap = getattr(self._data.company_info, 'market_cap', None)
        share_price = getattr(self._data.company_info, 'current_price', None)
        shares_outstanding = getattr(self._data.company_info, 'shares_outstanding', None)
        enterprise_value = getattr(self._data.company_info, 'enterprise_value', None)
        
        # P/E Ratio
        pe = self._calculate_pe_ratio(market_cap, shares_outstanding)
        if pe:
            ratios.append(pe)
        
        # P/B Ratio
        pb = self._calculate_pb_ratio(market_cap)
        if pb:
            ratios.append(pb)
        
        # P/S Ratio
        ps = self._calculate_ps_ratio(market_cap)
        if ps:
            ratios.append(ps)
        
        # EV/EBITDA
        ev_ebitda = self._calculate_ev_ebitda(enterprise_value)
        if ev_ebitda:
            ratios.append(ev_ebitda)
        
        # EV/Revenue
        ev_rev = self._calculate_ev_revenue(enterprise_value)
        if ev_rev:
            ratios.append(ev_rev)
        
        return ratios
    
    def _calculate_pe_ratio(
        self, 
        market_cap: Optional[float],
        shares_outstanding: Optional[float]
    ) -> Optional[FinancialRatio]:
        """Calculate Price-to-Earnings ratio."""
        net_income = self._get_field(StandardField.NET_INCOME, 0)
        
        if market_cap is None or net_income is None or net_income <= 0:
            return None
        
        pe_ratio = market_cap / net_income
        
        # Historical P/E not readily available, use single value
        return FinancialRatio(
            name="Price-to-Earnings",
            abbreviation="P/E",
            category=RatioCategory.VALUATION,
            current_value=pe_ratio,
            prior_value=None,
            change=None,
            trend=[pe_ratio],
            trend_assessment=TrendAssessment.INSUFFICIENT_DATA,
            interpretation=self._interpret_ratio(
                pe_ratio, (15.0, 20.0, 30.0, 50.0), higher_is_better=False
            ),
            formula="Market Cap / Net Income",
            description="Price paid per dollar of earnings",
            is_percentage=False,
            higher_is_better=False
        )
    
    def _calculate_pb_ratio(self, market_cap: Optional[float]) -> Optional[FinancialRatio]:
        """Calculate Price-to-Book ratio."""
        book_value = self._get_field(StandardField.TOTAL_EQUITY, 0)
        
        if market_cap is None or book_value is None or book_value <= 0:
            return None
        
        pb_ratio = market_cap / book_value
        
        return FinancialRatio(
            name="Price-to-Book",
            abbreviation="P/B",
            category=RatioCategory.VALUATION,
            current_value=pb_ratio,
            prior_value=None,
            change=None,
            trend=[pb_ratio],
            trend_assessment=TrendAssessment.INSUFFICIENT_DATA,
            interpretation=self._interpret_ratio(
                pb_ratio, (1.0, 2.0, 3.0, 5.0), higher_is_better=False
            ),
            formula="Market Cap / Book Value",
            description="Price paid per dollar of book equity",
            is_percentage=False,
            higher_is_better=False
        )
    
    def _calculate_ps_ratio(self, market_cap: Optional[float]) -> Optional[FinancialRatio]:
        """Calculate Price-to-Sales ratio."""
        revenue = self._get_field(StandardField.REVENUE, 0)
        
        if market_cap is None or revenue is None or revenue <= 0:
            return None
        
        ps_ratio = market_cap / revenue
        
        return FinancialRatio(
            name="Price-to-Sales",
            abbreviation="P/S",
            category=RatioCategory.VALUATION,
            current_value=ps_ratio,
            prior_value=None,
            change=None,
            trend=[ps_ratio],
            trend_assessment=TrendAssessment.INSUFFICIENT_DATA,
            interpretation=self._interpret_ratio(
                ps_ratio, (1.0, 2.0, 4.0, 8.0), higher_is_better=False
            ),
            formula="Market Cap / Revenue",
            description="Price paid per dollar of revenue",
            is_percentage=False,
            higher_is_better=False
        )
    
    def _calculate_ev_ebitda(
        self, 
        enterprise_value: Optional[float]
    ) -> Optional[FinancialRatio]:
        """Calculate EV/EBITDA ratio."""
        ebitda = self._get_field(StandardField.EBITDA, 0)
        
        if enterprise_value is None or ebitda is None or ebitda <= 0:
            return None
        
        ev_ebitda = enterprise_value / ebitda
        
        return FinancialRatio(
            name="EV/EBITDA",
            abbreviation="EV/EBITDA",
            category=RatioCategory.VALUATION,
            current_value=ev_ebitda,
            prior_value=None,
            change=None,
            trend=[ev_ebitda],
            trend_assessment=TrendAssessment.INSUFFICIENT_DATA,
            interpretation=self._interpret_ratio(
                ev_ebitda, (8.0, 12.0, 16.0, 25.0), higher_is_better=False
            ),
            formula="Enterprise Value / EBITDA",
            description="Enterprise value relative to operating cash proxy",
            is_percentage=False,
            higher_is_better=False
        )
    
    def _calculate_ev_revenue(
        self, 
        enterprise_value: Optional[float]
    ) -> Optional[FinancialRatio]:
        """Calculate EV/Revenue ratio."""
        revenue = self._get_field(StandardField.REVENUE, 0)
        
        if enterprise_value is None or revenue is None or revenue <= 0:
            return None
        
        ev_rev = enterprise_value / revenue
        
        return FinancialRatio(
            name="EV/Revenue",
            abbreviation="EV/Rev",
            category=RatioCategory.VALUATION,
            current_value=ev_rev,
            prior_value=None,
            change=None,
            trend=[ev_rev],
            trend_assessment=TrendAssessment.INSUFFICIENT_DATA,
            interpretation=self._interpret_ratio(
                ev_rev, (1.0, 2.0, 4.0, 8.0), higher_is_better=False
            ),
            formula="Enterprise Value / Revenue",
            description="Enterprise value relative to sales",
            is_percentage=False,
            higher_is_better=False
        )
    
    # =========================================================================
    # DUPONT ANALYSIS
    # =========================================================================
    
    def _perform_dupont_analysis(self) -> DuPontAnalysis:
        """
        Perform DuPont decomposition of ROE.
        
        ROE = Net Margin × Asset Turnover × Equity Multiplier
            = (NI/Revenue) × (Revenue/Assets) × (Assets/Equity)
        
        Returns:
            DuPontAnalysis object
        """
        net_income = self._get_field(StandardField.NET_INCOME, 0) or 0
        revenue = self._get_field(StandardField.REVENUE, 0) or 1
        total_assets = self._get_field(StandardField.TOTAL_ASSETS, 0) or 1
        total_equity = self._get_field(StandardField.TOTAL_EQUITY, 0) or 1
        
        # Calculate components
        net_margin = net_income / revenue if revenue > 0 else 0
        asset_turnover = revenue / total_assets if total_assets > 0 else 0
        equity_multiplier = total_assets / total_equity if total_equity > 0 else 0
        
        # Calculate ROE
        roe = net_margin * asset_turnover * equity_multiplier
        
        # Determine primary driver
        # Compare relative contribution to ROE
        components = {
            'Profitability (Net Margin)': net_margin,
            'Efficiency (Asset Turnover)': asset_turnover,
            'Leverage (Equity Multiplier)': equity_multiplier / 3  # Normalize leverage
        }
        
        primary_driver = max(components, key=components.get)
        
        # Generate analysis
        analysis_parts = []
        
        analysis_parts.append(
            f"ROE of {roe*100:.1f}% decomposes into: "
            f"Net Margin ({net_margin*100:.1f}%) × "
            f"Asset Turnover ({asset_turnover:.2f}x) × "
            f"Equity Multiplier ({equity_multiplier:.2f}x)"
        )
        
        if equity_multiplier > 3.0:
            analysis_parts.append(
                "High leverage is boosting ROE - sustainability depends on debt servicing ability"
            )
        elif net_margin > 0.15:
            analysis_parts.append(
                "Strong profitability is the primary driver of ROE"
            )
        elif asset_turnover > 1.5:
            analysis_parts.append(
                "High asset efficiency is contributing significantly to ROE"
            )
        
        return DuPontAnalysis(
            roe=roe,
            net_margin=net_margin,
            asset_turnover=asset_turnover,
            equity_multiplier=equity_multiplier,
            primary_driver=primary_driver,
            analysis=". ".join(analysis_parts)
        )
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _calculate_single_ratio(
        self,
        name: str,
        abbreviation: str,
        category: RatioCategory,
        numerator_field: StandardField,
        denominator_field: StandardField,
        formula: str,
        description: str,
        thresholds: Tuple[float, ...],
        is_percentage: bool = True,
        higher_is_better: bool = True
    ) -> Optional[FinancialRatio]:
        """
        Calculate a single ratio from two fields.
        
        Args:
            name: Full name of the ratio
            abbreviation: Short form
            category: Ratio category
            numerator_field: StandardField for numerator
            denominator_field: StandardField for denominator
            formula: Formula description
            description: What the ratio measures
            thresholds: Tuple of thresholds for interpretation
            is_percentage: Whether to display as percentage
            higher_is_better: Whether higher values are favorable
            
        Returns:
            FinancialRatio object or None if data unavailable
        """
        trend = []
        periods = self._data.statements.periods
        
        for i in range(len(periods)):
            numerator = self._get_field(numerator_field, i)
            denominator = self._get_field(denominator_field, i)
            
            if numerator is not None and denominator is not None and denominator != 0:
                ratio = numerator / denominator
                trend.append(ratio)
            else:
                break
        
        if not trend:
            return None
        
        current_value = trend[0]
        prior_value = trend[1] if len(trend) > 1 else None
        change = current_value - prior_value if prior_value is not None else None
        
        return FinancialRatio(
            name=name,
            abbreviation=abbreviation,
            category=category,
            current_value=current_value,
            prior_value=prior_value,
            change=change,
            trend=trend,
            trend_assessment=self._assess_trend(trend, higher_is_better),
            interpretation=self._interpret_ratio(current_value, thresholds, higher_is_better),
            formula=formula,
            description=description,
            is_percentage=is_percentage,
            higher_is_better=higher_is_better
        )
    
    def _calculate_return_ratio(
        self,
        name: str,
        abbreviation: str,
        numerator_field: StandardField,
        denominator_field: StandardField,
        formula: str,
        description: str,
        thresholds: Tuple[float, ...]
    ) -> Optional[FinancialRatio]:
        """
        Calculate a return ratio using average denominator.
        
        For return ratios like ROE and ROA, we use average of beginning
        and ending balance for the denominator.
        """
        trend = []
        periods = self._data.statements.periods
        
        for i in range(len(periods)):
            numerator = self._get_field(numerator_field, i)
            denom_current = self._get_field(denominator_field, i)
            denom_prior = self._get_field(denominator_field, i + 1)
            
            if numerator is None or denom_current is None:
                break
            
            # Use average if prior available
            if denom_prior is not None:
                avg_denom = (denom_current + denom_prior) / 2
            else:
                avg_denom = denom_current
            
            if avg_denom > 0:
                ratio = numerator / avg_denom
                trend.append(ratio)
            else:
                break
        
        if not trend:
            return None
        
        current_value = trend[0]
        prior_value = trend[1] if len(trend) > 1 else None
        change = current_value - prior_value if prior_value is not None else None
        
        return FinancialRatio(
            name=name,
            abbreviation=abbreviation,
            category=RatioCategory.PROFITABILITY,
            current_value=current_value,
            prior_value=prior_value,
            change=change,
            trend=trend,
            trend_assessment=self._assess_trend(trend, higher_is_better=True),
            interpretation=self._interpret_ratio(current_value, thresholds),
            formula=formula,
            description=description,
            is_percentage=True,
            higher_is_better=True
        )
    
    def _assess_trend(
        self, 
        trend: List[float], 
        higher_is_better: bool = True
    ) -> TrendAssessment:
        """
        Assess the trend direction of a ratio.
        
        Args:
            trend: List of ratio values (most recent first)
            higher_is_better: Whether increasing values are favorable
            
        Returns:
            TrendAssessment enum
        """
        if len(trend) < 2:
            return TrendAssessment.INSUFFICIENT_DATA
        
        # Calculate trend direction
        if len(trend) >= 3:
            recent_avg = np.mean(trend[:2])
            older_avg = np.mean(trend[2:min(4, len(trend))])
            std_dev = np.std(trend)
            mean_val = np.mean(trend)
        else:
            recent_avg = trend[0]
            older_avg = trend[-1]
            std_dev = np.std(trend)
            mean_val = np.mean(trend)
        
        # Check for volatility
        if mean_val != 0 and std_dev / abs(mean_val) > 0.25:
            return TrendAssessment.VOLATILE
        
        # Calculate change
        if older_avg != 0:
            change_pct = (recent_avg - older_avg) / abs(older_avg)
        else:
            change_pct = 0
        
        # Threshold for meaningful change
        threshold = 0.05
        
        if higher_is_better:
            if change_pct > threshold:
                return TrendAssessment.IMPROVING
            elif change_pct < -threshold:
                return TrendAssessment.DECLINING
        else:
            if change_pct < -threshold:
                return TrendAssessment.IMPROVING
            elif change_pct > threshold:
                return TrendAssessment.DECLINING
        
        return TrendAssessment.STABLE
    
    def _interpret_ratio(
        self, 
        value: float, 
        thresholds: Tuple[float, ...],
        higher_is_better: bool = True
    ) -> RatioInterpretation:
        """
        Interpret a ratio value based on thresholds.
        
        Args:
            value: Ratio value
            thresholds: Tuple of (strong, healthy, adequate, weak) thresholds
            higher_is_better: Whether higher values are favorable
            
        Returns:
            RatioInterpretation enum
        """
        if value is None:
            return RatioInterpretation.NOT_APPLICABLE
        
        strong, healthy, adequate, weak = thresholds
        
        if higher_is_better:
            if value >= strong:
                return RatioInterpretation.STRONG
            elif value >= healthy:
                return RatioInterpretation.HEALTHY
            elif value >= adequate:
                return RatioInterpretation.ADEQUATE
            elif value >= weak:
                return RatioInterpretation.WEAK
            else:
                return RatioInterpretation.CONCERNING
        else:
            # Lower is better (e.g., debt ratios, P/E)
            if value <= strong:
                return RatioInterpretation.STRONG
            elif value <= healthy:
                return RatioInterpretation.HEALTHY
            elif value <= adequate:
                return RatioInterpretation.ADEQUATE
            elif value <= weak:
                return RatioInterpretation.WEAK
            else:
                return RatioInterpretation.CONCERNING
    
    def _build_key_ratios_summary(
        self,
        profitability: List[FinancialRatio],
        liquidity: List[FinancialRatio],
        solvency: List[FinancialRatio],
        efficiency: List[FinancialRatio],
        valuation: List[FinancialRatio]
    ) -> Dict[str, Optional[float]]:
        """Build dictionary of key ratios for quick access."""
        summary = {}
        
        # Collect all ratios
        all_ratios = profitability + liquidity + solvency + efficiency + valuation
        
        # Key ratios to include
        key_abbrevs = [
            'GM', 'OM', 'NPM', 'ROE', 'ROA', 'ROIC',  # Profitability
            'CR', 'QR',  # Liquidity
            'D/E', 'ICR', 'D/EBITDA',  # Solvency
            'AT', 'InvT',  # Efficiency
            'P/E', 'P/B', 'EV/EBITDA'  # Valuation
        ]
        
        for ratio in all_ratios:
            if ratio.abbreviation in key_abbrevs:
                summary[ratio.abbreviation] = ratio.current_value
        
        return summary
    
    def _build_ratio_comparisons(self) -> List[RatioComparison]:
        """Build year-over-year comparisons for key ratios."""
        comparisons = []
        periods = self._data.statements.periods
        period_labels = [self._extract_year_from_period(p) for p in periods]
        
        # Build comparison for ROE
        roe_values = []
        for i in range(len(periods)):
            ni = self._get_field(StandardField.NET_INCOME, i)
            eq_curr = self._get_field(StandardField.TOTAL_EQUITY, i)
            eq_prior = self._get_field(StandardField.TOTAL_EQUITY, i + 1)
            
            if ni is not None and eq_curr is not None:
                avg_eq = (eq_curr + eq_prior) / 2 if eq_prior else eq_curr
                if avg_eq > 0:
                    roe_values.append(ni / avg_eq)
        
        if len(roe_values) >= 2:
            comparisons.append(RatioComparison(
                ratio_name="Return on Equity",
                periods=period_labels[:len(roe_values)],
                values=roe_values,
                cagr=self._calculate_cagr(roe_values) if len(roe_values) >= 2 else None,
                volatility=np.std(roe_values),
                trend_description=self._describe_trend(roe_values)
            ))
        
        # Build comparison for Operating Margin
        om_values = []
        for i in range(len(periods)):
            oi = self._get_field(StandardField.OPERATING_INCOME, i)
            rev = self._get_field(StandardField.REVENUE, i)
            
            if oi is not None and rev is not None and rev > 0:
                om_values.append(oi / rev)
        
        if len(om_values) >= 2:
            comparisons.append(RatioComparison(
                ratio_name="Operating Margin",
                periods=period_labels[:len(om_values)],
                values=om_values,
                cagr=None,  # CAGR not meaningful for margins
                volatility=np.std(om_values),
                trend_description=self._describe_trend(om_values)
            ))
        
        return comparisons
    
    def _calculate_cagr(self, values: List[float]) -> Optional[float]:
        """Calculate compound annual growth rate."""
        if len(values) < 2 or values[-1] <= 0 or values[0] <= 0:
            return None
        
        n = len(values) - 1
        cagr = (values[0] / values[-1]) ** (1 / n) - 1
        return cagr
    
    def _describe_trend(self, values: List[float]) -> str:
        """Generate description of value trend."""
        if len(values) < 2:
            return "Insufficient data for trend analysis"
        
        start = values[-1]
        end = values[0]
        
        if start == 0:
            return "Starting value is zero - trend not calculable"
        
        change = (end - start) / abs(start)
        
        if change > 0.15:
            return f"Strong improvement of {change*100:.1f}% over the period"
        elif change > 0.05:
            return f"Moderate improvement of {change*100:.1f}% over the period"
        elif change > -0.05:
            return f"Relatively stable with {change*100:.1f}% change"
        elif change > -0.15:
            return f"Moderate decline of {abs(change)*100:.1f}% over the period"
        else:
            return f"Significant decline of {abs(change)*100:.1f}% over the period"
    
    def _generate_insights(
        self,
        profitability: List[FinancialRatio],
        liquidity: List[FinancialRatio],
        solvency: List[FinancialRatio],
        efficiency: List[FinancialRatio],
        dupont: DuPontAnalysis
    ) -> List[str]:
        """Generate key insights from ratio analysis."""
        insights = []
        
        # Profitability insights
        for ratio in profitability:
            if ratio.abbreviation == 'ROE':
                if ratio.current_value and ratio.current_value > 0.20:
                    insights.append(
                        f"Strong ROE of {ratio.current_value*100:.1f}% indicates "
                        f"efficient use of shareholder capital"
                    )
                elif ratio.current_value and ratio.current_value < 0.08:
                    insights.append(
                        f"Low ROE of {ratio.current_value*100:.1f}% suggests "
                        f"suboptimal return on equity capital"
                    )
            
            if ratio.abbreviation == 'OM':
                if ratio.current_value and ratio.current_value > 0.20:
                    insights.append(
                        f"Operating margin of {ratio.current_value*100:.1f}% "
                        f"demonstrates strong operational efficiency"
                    )
        
        # DuPont insight
        insights.append(dupont.analysis)
        
        # Liquidity insights
        for ratio in liquidity:
            if ratio.abbreviation == 'CR':
                if ratio.current_value and ratio.current_value < 1.0:
                    insights.append(
                        f"Current ratio of {ratio.current_value:.2f}x below 1.0 "
                        f"indicates potential short-term liquidity pressure"
                    )
                elif ratio.current_value and ratio.current_value > 2.5:
                    insights.append(
                        f"High current ratio of {ratio.current_value:.2f}x may "
                        f"indicate inefficient use of working capital"
                    )
        
        # Solvency insights
        for ratio in solvency:
            if ratio.abbreviation == 'D/E':
                if ratio.current_value and ratio.current_value > 1.5:
                    insights.append(
                        f"Debt-to-equity of {ratio.current_value:.2f}x indicates "
                        f"significant financial leverage"
                    )
            
            if ratio.abbreviation == 'ICR':
                if ratio.current_value and ratio.current_value < 3.0:
                    insights.append(
                        f"Interest coverage of {ratio.current_value:.1f}x is tight - "
                        f"debt servicing capacity should be monitored"
                    )
                elif ratio.current_value and ratio.current_value > 10.0:
                    insights.append(
                        f"Strong interest coverage of {ratio.current_value:.1f}x "
                        f"provides comfortable debt servicing margin"
                    )
        
        # Efficiency insights
        for ratio in efficiency:
            if ratio.abbreviation == 'AT' and ratio.current_value:
                if ratio.current_value > 1.5:
                    insights.append(
                        f"Asset turnover of {ratio.current_value:.2f}x indicates "
                        f"efficient asset utilization"
                    )
        
        return insights
    
    def _get_field(
        self, 
        field: StandardField, 
        period_index: int = 0
    ) -> Optional[float]:
        """Extract a field value from processed data."""
        return self._processor.get_field(self._data, field, period_index)
    
    def _add_warning(self, message: str) -> None:
        """Add a warning message."""
        self._warnings.append(message)
        logger.warning(message)
    
    # =========================================================================
    # OUTPUT FORMATTING
    # =========================================================================
    
    def format_ratios_table(
        self, 
        ratios: List[FinancialRatio]
    ) -> pd.DataFrame:
        """Format a list of ratios as a DataFrame."""
        data = []
        for ratio in ratios:
            if ratio.is_percentage and ratio.current_value is not None:
                value_str = f"{ratio.current_value*100:.1f}%"
                prior_str = f"{ratio.prior_value*100:.1f}%" if ratio.prior_value else "N/A"
            elif ratio.current_value is not None:
                value_str = f"{ratio.current_value:.2f}"
                prior_str = f"{ratio.prior_value:.2f}" if ratio.prior_value else "N/A"
            else:
                value_str = "N/A"
                prior_str = "N/A"
            
            data.append({
                'Ratio': ratio.name,
                'Current': value_str,
                'Prior': prior_str,
                'Interpretation': ratio.interpretation.value,
                'Trend': ratio.trend_assessment.value
            })
        
        return pd.DataFrame(data)
    
    def format_summary_table(self, result: RatioCalculatorResult) -> pd.DataFrame:
        """Format key ratios as a summary table."""
        data = []
        
        # Add key ratios from each category
        for ratio in result.profitability_ratios:
            if ratio.abbreviation in ['GM', 'OM', 'NPM', 'ROE', 'ROA']:
                data.append(self._format_ratio_row(ratio))
        
        for ratio in result.liquidity_ratios:
            if ratio.abbreviation in ['CR', 'QR']:
                data.append(self._format_ratio_row(ratio))
        
        for ratio in result.solvency_ratios:
            if ratio.abbreviation in ['D/E', 'ICR']:
                data.append(self._format_ratio_row(ratio))
        
        return pd.DataFrame(data)
    
    def _format_ratio_row(self, ratio: FinancialRatio) -> Dict[str, Any]:
        """Format a single ratio as a dictionary row."""
        if ratio.is_percentage and ratio.current_value is not None:
            value_str = f"{ratio.current_value*100:.1f}%"
        elif ratio.current_value is not None:
            value_str = f"{ratio.current_value:.2f}"
        else:
            value_str = "N/A"
        
        return {
            'Ratio': f"{ratio.name} ({ratio.abbreviation})",
            'Value': value_str,
            'Quality': ratio.interpretation.value,
            'Trend': ratio.trend_assessment.value
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_ratios(processed_data: ProcessedData) -> RatioCalculatorResult:
    """
    Convenience function to calculate financial ratios.
    
    Args:
        processed_data: ProcessedData from DataProcessor
        
    Returns:
        RatioCalculatorResult
        
    Example:
        from data_collector import collect_financial_data
        from data_processor import process_financial_data
        
        raw = collect_financial_data("AAPL")
        processed = process_financial_data(raw)
        result = calculate_ratios(processed)
        
        print(f"ROE: {result.key_ratios_summary.get('ROE', 0):.1%}")
    """
    calculator = RatioCalculator(processed_data)
    return calculator.calculate()


def get_key_ratios(result: RatioCalculatorResult) -> Dict[str, Optional[float]]:
    """
    Extract key ratios as a simple dictionary.
    
    Args:
        result: RatioCalculatorResult
        
    Returns:
        Dictionary with key ratios
    """
    return result.key_ratios_summary.copy()


# =============================================================================
# MODULE TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Module test script.
    
    Run this file directly to test ratio calculations:
        python ratio_calculator.py [TICKER]
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
    print(f"RATIO CALCULATOR TEST - {test_ticker}")
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
        
        # Step 3: Calculate ratios
        print("Step 3: Calculating ratios...")
        calculator = RatioCalculator(processed)
        result = calculator.calculate()
        print(f"  Calculations complete\n")
        
        # Print Profitability Ratios
        print("PROFITABILITY RATIOS")
        print("-" * 70)
        print(f"  {'Ratio':<25} {'Current':>12} {'Prior':>12} {'Interpretation':<15}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*15}")
        for ratio in result.profitability_ratios:
            curr = f"{ratio.current_value*100:.1f}%" if ratio.current_value else "N/A"
            prior = f"{ratio.prior_value*100:.1f}%" if ratio.prior_value else "N/A"
            print(f"  {ratio.name:<25} {curr:>12} {prior:>12} {ratio.interpretation.value:<15}")
        
        # Print Liquidity Ratios
        print(f"\nLIQUIDITY RATIOS")
        print("-" * 70)
        for ratio in result.liquidity_ratios:
            curr = f"{ratio.current_value:.2f}x" if ratio.current_value else "N/A"
            prior = f"{ratio.prior_value:.2f}x" if ratio.prior_value else "N/A"
            print(f"  {ratio.name:<25} {curr:>12} {prior:>12} {ratio.interpretation.value:<15}")
        
        # Print Solvency Ratios
        print(f"\nSOLVENCY RATIOS")
        print("-" * 70)
        for ratio in result.solvency_ratios:
            if ratio.is_percentage:
                curr = f"{ratio.current_value*100:.1f}%" if ratio.current_value else "N/A"
                prior = f"{ratio.prior_value*100:.1f}%" if ratio.prior_value else "N/A"
            else:
                curr = f"{ratio.current_value:.2f}x" if ratio.current_value else "N/A"
                prior = f"{ratio.prior_value:.2f}x" if ratio.prior_value else "N/A"
            print(f"  {ratio.name:<25} {curr:>12} {prior:>12} {ratio.interpretation.value:<15}")
        
        # Print Efficiency Ratios
        print(f"\nEFFICIENCY RATIOS")
        print("-" * 70)
        for ratio in result.efficiency_ratios:
            curr = f"{ratio.current_value:.2f}x" if ratio.current_value else "N/A"
            prior = f"{ratio.prior_value:.2f}x" if ratio.prior_value else "N/A"
            print(f"  {ratio.name:<25} {curr:>12} {prior:>12} {ratio.interpretation.value:<15}")
        
        # Print Valuation Ratios
        if result.valuation_ratios:
            print(f"\nVALUATION RATIOS")
            print("-" * 70)
            for ratio in result.valuation_ratios:
                curr = f"{ratio.current_value:.2f}x" if ratio.current_value else "N/A"
                print(f"  {ratio.name:<25} {curr:>12}")
        
        # Print DuPont Analysis
        print(f"\nDUPONT ANALYSIS")
        print("-" * 70)
        dupont = result.dupont_analysis
        print(f"  ROE = Net Margin × Asset Turnover × Equity Multiplier")
        print(f"  {dupont.roe*100:.1f}% = {dupont.net_margin*100:.1f}% × {dupont.asset_turnover:.2f}x × {dupont.equity_multiplier:.2f}x")
        print(f"  Primary Driver: {dupont.primary_driver}")
        print(f"  {dupont.analysis}")
        
        # Print Key Insights
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
        print(f"Ratio calculations complete for {test_ticker}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)