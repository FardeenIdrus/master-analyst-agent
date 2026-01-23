"""
Ratio Analysis Module - Phase 3 Financial Ratio Analysis
Fundamental Analyst Agent

Implements comprehensive financial ratio analysis including:
- 39 ratios across 5 categories (Profitability, Leverage, Liquidity, Efficiency, Growth)
- Time-series ratio calculation with multi-year trends
- Benchmark-based assessment and scoring
- Interest expense estimation for companies with consolidated reporting
- Category-level and overall financial health scoring

Inputs: CollectionResult from Phase 1 (or ValidatedData from Phase 2)
Outputs: RatioAnalysisResult container with all computed ratios

MSc Coursework: IFTE0001 AI Agents in Asset Management
Track A: Fundamental Analyst Agent
Phase 3: Financial Ratio Analysis

Version: 1.1.0
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union

from .config import (
    LOGGER,
    OUTPUT_DIR,
    RatioCategory,
    RatioAssessment,
    RatioTrend,
    RatioDefinition,
    PROFITABILITY_RATIOS,
    LEVERAGE_RATIOS,
    LIQUIDITY_RATIOS,
    EFFICIENCY_RATIOS,
    GROWTH_RATIOS,
    ALL_RATIO_DEFINITIONS,
    RATIO_ANALYSIS_CONFIG,
    get_ratio_definition,
    assess_ratio_value,
)


__version__ = "1.1.0"


# =============================================================================
# DATA CONTAINERS
# =============================================================================

@dataclass
class RatioValue:
    """
    Single ratio value with assessment and metadata.
    
    Stores computed ratio value along with benchmark assessment,
    contributing factors, and any calculation notes.
    """
    
    ratio_name: str
    fiscal_year: str
    value: Optional[float]
    assessment: RatioAssessment
    numerator: Optional[float] = None
    denominator: Optional[float] = None
    calculation_note: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ratio_name": self.ratio_name,
            "fiscal_year": self.fiscal_year,
            "value": self.value,
            "assessment": self.assessment.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "calculation_note": self.calculation_note,
        }


@dataclass
class RatioTimeSeries:
    """
    Time series of a single ratio across multiple fiscal years.
    
    Contains all computed values, trend analysis, and summary statistics
    for one financial ratio.
    """
    
    ratio_name: str
    definition: RatioDefinition
    values: Dict[str, RatioValue] = field(default_factory=dict)  # year -> RatioValue
    
    # Trend analysis
    trend: RatioTrend = RatioTrend.INSUFFICIENT_DATA
    cagr: Optional[float] = None
    average_value: Optional[float] = None
    latest_value: Optional[float] = None
    latest_assessment: RatioAssessment = RatioAssessment.NOT_APPLICABLE
    
    # Change metrics
    yoy_change: Optional[float] = None  # Most recent year-over-year change
    period_change: Optional[float] = None  # First to last year change
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ratio_name": self.ratio_name,
            "display_name": self.definition.name,
            "category": self.definition.category.value,
            "formula": self.definition.formula,
            "values": {k: v.to_dict() for k, v in self.values.items()},
            "trend": self.trend.value,
            "cagr": self.cagr,
            "average_value": self.average_value,
            "latest_value": self.latest_value,
            "latest_assessment": self.latest_assessment.value,
            "yoy_change": self.yoy_change,
            "period_change": self.period_change,
        }


@dataclass
class CategoryAnalysis:
    """
    Analysis results for a single ratio category.
    
    Aggregates all ratios within a category and provides
    category-level scoring and assessment.
    """
    
    category: RatioCategory
    ratios: Dict[str, RatioTimeSeries] = field(default_factory=dict)
    
    # Category scores
    category_score: float = 0.0  # 0.0 to 1.0
    assessment_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Summary
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "category_score": self.category_score,
            "assessment_distribution": self.assessment_distribution,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "ratios": {k: v.to_dict() for k, v in self.ratios.items()},
        }


@dataclass
class OverallAssessment:
    """
    Overall financial health assessment aggregating all categories.
    """
    
    overall_score: float = 0.0  # 0.0 to 1.0
    overall_assessment: RatioAssessment = RatioAssessment.NOT_APPLICABLE
    
    # Category scores
    profitability_score: float = 0.0
    leverage_score: float = 0.0
    liquidity_score: float = 0.0
    efficiency_score: float = 0.0
    
    # Key findings
    key_strengths: List[str] = field(default_factory=list)
    key_weaknesses: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "overall_assessment": self.overall_assessment.value,
            "profitability_score": self.profitability_score,
            "leverage_score": self.leverage_score,
            "liquidity_score": self.liquidity_score,
            "efficiency_score": self.efficiency_score,
            "key_strengths": self.key_strengths,
            "key_weaknesses": self.key_weaknesses,
            "key_risks": self.key_risks,
        }


@dataclass
class RatioAnalysisResult:
    """
    Complete output of Phase 3 ratio analysis.
    
    Contains all computed ratios organized by category,
    trend analysis, and overall financial health assessment.
    """
    
    ticker: str
    company_name: str
    fiscal_periods: List[str]
    
    # Category analyses
    profitability: CategoryAnalysis
    leverage: CategoryAnalysis
    liquidity: CategoryAnalysis
    efficiency: CategoryAnalysis
    growth: CategoryAnalysis
    
    # Overall assessment
    overall_assessment: OverallAssessment
    
    # Metadata
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    ratios_calculated: int = 0
    data_source: str = "Phase 1/2 Pipeline"
    
    @property
    def is_valid(self) -> bool:
        """Check if analysis produced valid results."""
        return self.ratios_calculated > 0
    
    def get_ratio(self, ratio_name: str) -> Optional[RatioTimeSeries]:
        """Get a specific ratio by name."""
        for category in [self.profitability, self.leverage, self.liquidity, self.efficiency, self.growth]:
            if ratio_name in category.ratios:
                return category.ratios[ratio_name]
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "fiscal_periods": self.fiscal_periods,
            "profitability": self.profitability.to_dict(),
            "leverage": self.leverage.to_dict(),
            "liquidity": self.liquidity.to_dict(),
            "efficiency": self.efficiency.to_dict(),
            "growth": self.growth.to_dict(),
            "overall_assessment": self.overall_assessment.to_dict(),
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "ratios_calculated": self.ratios_calculated,
            "data_source": self.data_source,
        }


# =============================================================================
# RATIO CALCULATOR BASE
# =============================================================================

class RatioCalculatorBase:
    """
    Base class for ratio calculations.
    
    Provides common functionality for safe division, value extraction,
    and average calculations used by all ratio calculators.
    """
    
    @staticmethod
    def safe_divide(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
        """
        Perform division with null and zero checks.
        
        Args:
            numerator: Dividend value
            denominator: Divisor value
            
        Returns:
            Result of division or None if invalid
        """
        if numerator is None or denominator is None:
            return None
        if denominator == 0:
            return None
        return numerator / denominator
    
    @staticmethod
    def get_value(df: pd.DataFrame, field: str, year: str) -> Optional[float]:
        """
        Safely extract value from DataFrame.
        
        Args:
            df: Financial statement DataFrame
            field: Field name (row index)
            year: Fiscal year (column)
            
        Returns:
            Float value or None if not available
        """
        if df is None or df.empty:
            return None
        if field not in df.index:
            return None
        if year not in df.columns:
            return None
        value = df.loc[field, year]
        if pd.isna(value):
            return None
        return float(value)
    
    @staticmethod
    def get_average(df: pd.DataFrame, field: str, year: str, years: List[str]) -> Optional[float]:
        """
        Calculate average of current and prior year values.
        
        Args:
            df: Financial statement DataFrame
            field: Field name
            year: Current fiscal year
            years: List of all fiscal years (descending order)
            
        Returns:
            Average of current and prior year, or current if prior unavailable
        """
        current = RatioCalculatorBase.get_value(df, field, year)
        if current is None:
            return None
        
        # Find prior year
        try:
            year_idx = years.index(year)
            if year_idx < len(years) - 1:
                prior_year = years[year_idx + 1]
                prior = RatioCalculatorBase.get_value(df, field, prior_year)
                if prior is not None:
                    return (current + prior) / 2
        except (ValueError, IndexError):
            pass
        
        return current
    
    def calculate_ratio(
        self,
        ratio_name: str,
        numerator: Optional[float],
        denominator: Optional[float],
        year: str,
    ) -> RatioValue:
        """
        Calculate a single ratio value with assessment.
        
        Args:
            ratio_name: Name of the ratio
            numerator: Numerator value
            denominator: Denominator value
            year: Fiscal year
            
        Returns:
            RatioValue with computed ratio and assessment
        """
        value = self.safe_divide(numerator, denominator)
        assessment = assess_ratio_value(ratio_name, value) if value is not None else RatioAssessment.NOT_APPLICABLE
        
        note = None
        if numerator is None or denominator is None:
            note = "Missing data"
        elif denominator == 0:
            note = "Division by zero"
        
        return RatioValue(
            ratio_name=ratio_name,
            fiscal_year=year,
            value=value,
            assessment=assessment,
            numerator=numerator,
            denominator=denominator,
            calculation_note=note,
        )


# =============================================================================
# PROFITABILITY RATIO CALCULATOR
# =============================================================================

class ProfitabilityCalculator(RatioCalculatorBase):
    """
    Calculates profitability ratios measuring earnings performance.
    
    Ratios:
    - Gross Margin: Gross Profit / Revenue
    - Operating Margin: Operating Income / Revenue
    - Net Profit Margin: Net Income / Revenue
    - EBITDA Margin: EBITDA / Revenue
    - ROE: Net Income / Average Equity
    - ROA: Net Income / Average Assets
    - ROIC: NOPAT / Invested Capital
    - ROCE: EBIT / Capital Employed
    """
    
    def calculate_all(
        self,
        income: pd.DataFrame,
        balance: pd.DataFrame,
        derived: Dict[str, Dict[str, Optional[float]]],
        years: List[str],
    ) -> Dict[str, RatioTimeSeries]:
        """
        Calculate all profitability ratios for all years.
        
        Args:
            income: Income statement DataFrame
            balance: Balance sheet DataFrame
            derived: Derived metrics from Phase 1
            years: List of fiscal years
            
        Returns:
            Dictionary of ratio name to RatioTimeSeries
        """
        results = {}
        
        for ratio_name, definition in PROFITABILITY_RATIOS.items():
            series = RatioTimeSeries(ratio_name=ratio_name, definition=definition)
            
            for year in years:
                ratio_value = self._calculate_single(ratio_name, income, balance, derived, year, years)
                series.values[year] = ratio_value
            
            self._compute_trend(series, years)
            results[ratio_name] = series
        
        return results
    
    def _calculate_single(
        self,
        ratio_name: str,
        income: pd.DataFrame,
        balance: pd.DataFrame,
        derived: Dict[str, Dict[str, Optional[float]]],
        year: str,
        years: List[str],
    ) -> RatioValue:
        """Calculate a single profitability ratio for one year."""
        
        revenue = self.get_value(income, "total_revenue", year)
        gross_profit = self.get_value(income, "gross_profit", year)
        operating_income = self.get_value(income, "operating_income", year)
        net_income = self.get_value(income, "net_income", year)
        ebit = self.get_value(income, "ebit", year)
        
        # Get derived metrics
        ebitda = derived.get("ebitda_calculated", {}).get(year)
        invested_capital = derived.get("invested_capital", {}).get(year)
        fcf = derived.get("fcf_calculated", {}).get(year)
        
        # Balance sheet items (use averages for return ratios)
        avg_equity = self.get_average(balance, "total_equity", year, years)
        avg_assets = self.get_average(balance, "total_assets", year, years)
        current_liabilities = self.get_value(balance, "current_liabilities", year)
        total_assets = self.get_value(balance, "total_assets", year)
        
        if ratio_name == "gross_margin":
            return self.calculate_ratio(ratio_name, gross_profit, revenue, year)
        
        elif ratio_name == "operating_margin":
            return self.calculate_ratio(ratio_name, operating_income, revenue, year)
        
        elif ratio_name == "net_profit_margin":
            return self.calculate_ratio(ratio_name, net_income, revenue, year)
        
        elif ratio_name == "ebitda_margin":
            return self.calculate_ratio(ratio_name, ebitda, revenue, year)
        
        elif ratio_name == "fcf_margin":
            return self.calculate_ratio(ratio_name, fcf, revenue, year)
        
        elif ratio_name == "roe":
            return self.calculate_ratio(ratio_name, net_income, avg_equity, year)
        
        elif ratio_name == "roa":
            return self.calculate_ratio(ratio_name, net_income, avg_assets, year)
        
        elif ratio_name == "roic":
            # ROIC = NOPAT / Average Invested Capital
            # NOPAT = Operating Income * (1 - Tax Rate), approximate as 75%
            nopat = operating_income * 0.75 if operating_income else None
            # Use average invested capital for better accuracy
            curr_ic = derived.get("invested_capital", {}).get(year)
            # Find prior year IC
            avg_ic = curr_ic
            try:
                year_idx = years.index(year)
                if year_idx < len(years) - 1 and curr_ic is not None:
                    prior_year = years[year_idx + 1]
                    prior_ic = derived.get("invested_capital", {}).get(prior_year)
                    if prior_ic is not None:
                        avg_ic = (curr_ic + prior_ic) / 2
            except (ValueError, IndexError):
                pass
            return self.calculate_ratio(ratio_name, nopat, avg_ic, year)
        
        elif ratio_name == "roce":
            # ROCE = EBIT / Capital Employed
            # Capital Employed = Total Assets - Current Liabilities
            capital_employed = None
            if total_assets is not None and current_liabilities is not None:
                capital_employed = total_assets - current_liabilities
            return self.calculate_ratio(ratio_name, ebit, capital_employed, year)
        
        else:
            return RatioValue(
                ratio_name=ratio_name,
                fiscal_year=year,
                value=None,
                assessment=RatioAssessment.NOT_APPLICABLE,
                calculation_note="Unknown ratio",
            )
    
    def _compute_trend(self, series: RatioTimeSeries, years: List[str]) -> None:
        """Compute trend analysis for a ratio series."""
        valid_values = [(y, series.values[y].value) for y in years if series.values.get(y) and series.values[y].value is not None]
        
        if not valid_values:
            return
        
        # Latest value and assessment
        series.latest_value = valid_values[0][1]
        series.latest_assessment = series.values[valid_values[0][0]].assessment
        
        # Average
        values = [v[1] for v in valid_values]
        series.average_value = np.mean(values)
        
        if len(valid_values) < 2:
            return
        
        # YoY change (most recent)
        series.yoy_change = valid_values[0][1] - valid_values[1][1]
        
        # Period change
        series.period_change = valid_values[0][1] - valid_values[-1][1]
        
        # Trend classification
        if len(valid_values) >= RATIO_ANALYSIS_CONFIG.min_years_for_trend:
            cv = np.std(values) / abs(np.mean(values)) if np.mean(values) != 0 else 0
            
            if cv > RATIO_ANALYSIS_CONFIG.volatility_cv_threshold:
                series.trend = RatioTrend.VOLATILE
            elif series.period_change > RATIO_ANALYSIS_CONFIG.trend_improving_threshold:
                series.trend = RatioTrend.IMPROVING
            elif series.period_change < RATIO_ANALYSIS_CONFIG.trend_deteriorating_threshold:
                series.trend = RatioTrend.DETERIORATING
            else:
                series.trend = RatioTrend.STABLE
            
            # CAGR for ratios (if positive values)
            if valid_values[-1][1] > 0 and valid_values[0][1] > 0:
                n_years = len(valid_values) - 1
                series.cagr = (valid_values[0][1] / valid_values[-1][1]) ** (1 / n_years) - 1


# =============================================================================
# LEVERAGE RATIO CALCULATOR
# =============================================================================

class LeverageCalculator(RatioCalculatorBase):
    """
    Calculates leverage/solvency ratios measuring financial risk.
    
    Ratios:
    - Debt-to-Equity: Total Debt / Total Equity
    - Debt-to-Assets: Total Debt / Total Assets
    - Debt-to-Capital: Total Debt / (Total Debt + Equity)
    - Interest Coverage: EBIT / Interest Expense
    - Cash Coverage: (EBIT + D&A) / Interest Expense
    - Debt-to-EBITDA: Total Debt / EBITDA
    - Equity Multiplier: Total Assets / Total Equity
    - Long-term Debt-to-Equity: Long-term Debt / Equity
    
    Note: Includes estimation logic for interest coverage when companies
    consolidate interest expense into "Other Income/(Expense)" (e.g., Apple FY2024+)
    """
    
    # Typical interest rates by credit rating for estimation
    ESTIMATED_INTEREST_RATE = 0.045  # 4.5% default for investment-grade
    MINIMUM_DEBT_FOR_ESTIMATION = 1e9  # $1B threshold
    
    def calculate_all(
        self,
        income: pd.DataFrame,
        balance: pd.DataFrame,
        cashflow: pd.DataFrame,
        derived: Dict[str, Dict[str, Optional[float]]],
        years: List[str],
    ) -> Dict[str, RatioTimeSeries]:
        """Calculate all leverage ratios for all years."""
        results = {}
        
        # Calculate historical interest rate from available data
        self._historical_rate = self._calculate_historical_interest_rate(income, balance, years)
        
        for ratio_name, definition in LEVERAGE_RATIOS.items():
            series = RatioTimeSeries(ratio_name=ratio_name, definition=definition)
            
            for year in years:
                ratio_value = self._calculate_single(ratio_name, income, balance, cashflow, derived, year)
                series.values[year] = ratio_value
            
            self._compute_trend(series, years)
            results[ratio_name] = series
        
        return results
    
    def _calculate_historical_interest_rate(
        self,
        income: pd.DataFrame,
        balance: pd.DataFrame,
        years: List[str],
    ) -> Optional[float]:
        """
        Calculate historical average interest rate from years with available data.
        
        This is used to estimate interest expense for years where it's unavailable
        (e.g., when companies consolidate interest into Other Income).
        
        Returns:
            Average interest rate as decimal, or None if insufficient data
        """
        rates = []
        
        for year in years:
            interest_expense = self.get_value(income, "interest_expense", year)
            total_debt = self.get_value(balance, "total_debt", year)
            
            if interest_expense and interest_expense > 0 and total_debt and total_debt > 0:
                rate = interest_expense / total_debt
                # Sanity check: reasonable rates between 1% and 15%
                if 0.01 <= rate <= 0.15:
                    rates.append(rate)
        
        if rates:
            return np.mean(rates)
        return None
    
    def _estimate_interest_expense(
        self,
        total_debt: Optional[float],
    ) -> Tuple[Optional[float], str]:
        """
        Estimate interest expense from total debt when unavailable.
        
        Args:
            total_debt: Total debt from balance sheet
            
        Returns:
            Tuple of (estimated_interest, note_string)
        """
        if total_debt is None or total_debt < self.MINIMUM_DEBT_FOR_ESTIMATION:
            return None, ""
        
        # Use historical rate if available, otherwise default
        rate = self._historical_rate or self.ESTIMATED_INTEREST_RATE
        estimated_interest = total_debt * rate
        
        rate_source = "historical" if self._historical_rate else "default 4.5%"
        note = f"Estimated from debt (${total_debt/1e9:.1f}B × {rate*100:.1f}% {rate_source})"
        
        return estimated_interest, note
    
    def _calculate_single(
        self,
        ratio_name: str,
        income: pd.DataFrame,
        balance: pd.DataFrame,
        cashflow: pd.DataFrame,
        derived: Dict[str, Dict[str, Optional[float]]],
        year: str,
    ) -> RatioValue:
        """Calculate a single leverage ratio for one year."""
        
        # Balance sheet items
        total_debt = self.get_value(balance, "total_debt", year)
        total_equity = self.get_value(balance, "total_equity", year)
        total_assets = self.get_value(balance, "total_assets", year)
        long_term_debt = self.get_value(balance, "long_term_debt", year)
        cash = self.get_value(balance, "cash_and_equivalents", year)
        
        # Income statement items
        ebit = self.get_value(income, "ebit", year)
        interest_expense = self.get_value(income, "interest_expense", year)
        da = self.get_value(income, "depreciation_amortization", year)
        
        # Derived metrics
        ebitda = derived.get("ebitda_calculated", {}).get(year)
        net_debt = derived.get("net_debt", {}).get(year)
        
        if ratio_name == "debt_to_equity":
            return self.calculate_ratio(ratio_name, total_debt, total_equity, year)
        
        elif ratio_name == "debt_to_assets":
            return self.calculate_ratio(ratio_name, total_debt, total_assets, year)
        
        elif ratio_name == "debt_to_capital":
            total_capital = None
            if total_debt is not None and total_equity is not None:
                total_capital = total_debt + total_equity
            return self.calculate_ratio(ratio_name, total_debt, total_capital, year)
        
        elif ratio_name == "interest_coverage":
            return self._calculate_interest_coverage(
                ebit, interest_expense, total_debt, year, use_ebitda=False, da=da
            )
        
        elif ratio_name == "cash_coverage":
            return self._calculate_interest_coverage(
                ebit, interest_expense, total_debt, year, use_ebitda=True, da=da
            )
        
        elif ratio_name == "debt_to_ebitda":
            return self.calculate_ratio(ratio_name, total_debt, ebitda, year)
        
        elif ratio_name == "net_debt_to_ebitda":
            return self.calculate_ratio(ratio_name, net_debt, ebitda, year)
        
        elif ratio_name == "equity_multiplier":
            return self.calculate_ratio(ratio_name, total_assets, total_equity, year)
        
        elif ratio_name == "long_term_debt_to_equity":
            return self.calculate_ratio(ratio_name, long_term_debt, total_equity, year)
        
        else:
            return RatioValue(
                ratio_name=ratio_name,
                fiscal_year=year,
                value=None,
                assessment=RatioAssessment.NOT_APPLICABLE,
                calculation_note="Unknown ratio",
            )
    
    def _calculate_interest_coverage(
        self,
        ebit: Optional[float],
        interest_expense: Optional[float],
        total_debt: Optional[float],
        year: str,
        use_ebitda: bool = False,
        da: Optional[float] = None,
    ) -> RatioValue:
        """
        Calculate interest coverage or cash coverage ratio with estimation fallback.
        
        When interest_expense is unavailable (None or 0) but company has significant
        debt, estimates interest expense using historical or typical rates.
        
        Args:
            ebit: Earnings before interest and taxes
            interest_expense: Interest expense from income statement
            total_debt: Total debt from balance sheet
            year: Fiscal year
            use_ebitda: If True, calculates Cash Coverage (EBIT + D&A)
            da: Depreciation and amortization
            
        Returns:
            RatioValue with calculated or estimated coverage ratio
        """
        ratio_name = "cash_coverage" if use_ebitda else "interest_coverage"
        
        # Calculate numerator
        if use_ebitda:
            numerator = (ebit + da) if (ebit is not None and da is not None) else None
        else:
            numerator = ebit
        
        # Case 1: Interest expense is available and valid
        if interest_expense is not None and interest_expense > 0:
            return self.calculate_ratio(ratio_name, numerator, interest_expense, year)
        
        # Case 2: Interest expense unavailable but company has significant debt
        # This handles Apple FY2024+ where interest is consolidated into Other Income
        if total_debt is not None and total_debt > self.MINIMUM_DEBT_FOR_ESTIMATION:
            estimated_interest, note = self._estimate_interest_expense(total_debt)
            
            if estimated_interest and numerator:
                coverage = numerator / estimated_interest
                assessment = assess_ratio_value(ratio_name, coverage)
                
                return RatioValue(
                    ratio_name=ratio_name,
                    fiscal_year=year,
                    value=coverage,
                    assessment=assessment,
                    numerator=numerator,
                    denominator=estimated_interest,
                    calculation_note=f"Estimated: {note}",
                )
        
        # Case 3: No interest expense and minimal/no debt (likely debt-free)
        if interest_expense == 0 and (total_debt is None or total_debt < self.MINIMUM_DEBT_FOR_ESTIMATION):
            return RatioValue(
                ratio_name=ratio_name,
                fiscal_year=year,
                value=None,
                assessment=RatioAssessment.NOT_APPLICABLE,
                calculation_note="No significant debt",
            )
        
        # Case 4: Data unavailable
        return RatioValue(
            ratio_name=ratio_name,
            fiscal_year=year,
            value=None,
            assessment=RatioAssessment.NOT_APPLICABLE,
            calculation_note="Interest expense data unavailable",
        )
    
    def _compute_trend(self, series: RatioTimeSeries, years: List[str]) -> None:
        """Compute trend analysis for a ratio series."""
        valid_values = [(y, series.values[y].value) for y in years if series.values.get(y) and series.values[y].value is not None]
        
        if not valid_values:
            return
        
        series.latest_value = valid_values[0][1]
        series.latest_assessment = series.values[valid_values[0][0]].assessment
        
        values = [v[1] for v in valid_values]
        series.average_value = np.mean(values)
        
        if len(valid_values) < 2:
            return
        
        series.yoy_change = valid_values[0][1] - valid_values[1][1]
        series.period_change = valid_values[0][1] - valid_values[-1][1]
        
        if len(valid_values) >= RATIO_ANALYSIS_CONFIG.min_years_for_trend:
            cv = np.std(values) / abs(np.mean(values)) if np.mean(values) != 0 else 0
            
            # For leverage ratios, lower is typically better, so improving = decreasing
            if series.definition.invert_assessment:
                if cv > RATIO_ANALYSIS_CONFIG.volatility_cv_threshold:
                    series.trend = RatioTrend.VOLATILE
                elif series.period_change < -RATIO_ANALYSIS_CONFIG.trend_improving_threshold:
                    series.trend = RatioTrend.IMPROVING
                elif series.period_change > RATIO_ANALYSIS_CONFIG.trend_deteriorating_threshold:
                    series.trend = RatioTrend.DETERIORATING
                else:
                    series.trend = RatioTrend.STABLE
            else:
                if cv > RATIO_ANALYSIS_CONFIG.volatility_cv_threshold:
                    series.trend = RatioTrend.VOLATILE
                elif series.period_change > RATIO_ANALYSIS_CONFIG.trend_improving_threshold:
                    series.trend = RatioTrend.IMPROVING
                elif series.period_change < RATIO_ANALYSIS_CONFIG.trend_deteriorating_threshold:
                    series.trend = RatioTrend.DETERIORATING
                else:
                    series.trend = RatioTrend.STABLE


# =============================================================================
# LIQUIDITY RATIO CALCULATOR
# =============================================================================

class LiquidityCalculator(RatioCalculatorBase):
    """
    Calculates liquidity ratios measuring short-term solvency.
    
    Ratios:
    - Current Ratio: Current Assets / Current Liabilities
    - Quick Ratio: (Current Assets - Inventory) / Current Liabilities
    - Cash Ratio: Cash / Current Liabilities
    - Operating Cash Flow Ratio: OCF / Current Liabilities
    - Working Capital to Assets: Working Capital / Total Assets
    """
    
    def calculate_all(
        self,
        balance: pd.DataFrame,
        cashflow: pd.DataFrame,
        derived: Dict[str, Dict[str, Optional[float]]],
        years: List[str],
    ) -> Dict[str, RatioTimeSeries]:
        """Calculate all liquidity ratios for all years."""
        results = {}
        
        for ratio_name, definition in LIQUIDITY_RATIOS.items():
            series = RatioTimeSeries(ratio_name=ratio_name, definition=definition)
            
            for year in years:
                ratio_value = self._calculate_single(ratio_name, balance, cashflow, derived, year)
                series.values[year] = ratio_value
            
            self._compute_trend(series, years)
            results[ratio_name] = series
        
        return results
    
    def _calculate_single(
        self,
        ratio_name: str,
        balance: pd.DataFrame,
        cashflow: pd.DataFrame,
        derived: Dict[str, Dict[str, Optional[float]]],
        year: str,
    ) -> RatioValue:
        """Calculate a single liquidity ratio for one year."""
        
        # Balance sheet items
        current_assets = self.get_value(balance, "current_assets", year)
        current_liabilities = self.get_value(balance, "current_liabilities", year)
        cash = self.get_value(balance, "cash_and_equivalents", year)
        inventory = self.get_value(balance, "inventory", year)
        total_assets = self.get_value(balance, "total_assets", year)
        
        # Cash flow items
        ocf = self.get_value(cashflow, "operating_cash_flow", year)
        
        # Derived metrics
        working_capital = derived.get("working_capital", {}).get(year)
        
        if ratio_name == "current_ratio":
            return self.calculate_ratio(ratio_name, current_assets, current_liabilities, year)
        
        elif ratio_name == "quick_ratio":
            quick_assets = None
            if current_assets is not None and inventory is not None:
                quick_assets = current_assets - inventory
            elif current_assets is not None:
                quick_assets = current_assets  # No inventory data
            return self.calculate_ratio(ratio_name, quick_assets, current_liabilities, year)
        
        elif ratio_name == "cash_ratio":
            return self.calculate_ratio(ratio_name, cash, current_liabilities, year)
        
        elif ratio_name == "operating_cash_flow_ratio":
            return self.calculate_ratio(ratio_name, ocf, current_liabilities, year)
        
        elif ratio_name == "working_capital_to_assets":
            return self.calculate_ratio(ratio_name, working_capital, total_assets, year)
        
        else:
            return RatioValue(
                ratio_name=ratio_name,
                fiscal_year=year,
                value=None,
                assessment=RatioAssessment.NOT_APPLICABLE,
                calculation_note="Unknown ratio",
            )
    
    def _compute_trend(self, series: RatioTimeSeries, years: List[str]) -> None:
        """Compute trend analysis for a ratio series."""
        valid_values = [(y, series.values[y].value) for y in years if series.values.get(y) and series.values[y].value is not None]
        
        if not valid_values:
            return
        
        series.latest_value = valid_values[0][1]
        series.latest_assessment = series.values[valid_values[0][0]].assessment
        
        values = [v[1] for v in valid_values]
        series.average_value = np.mean(values)
        
        if len(valid_values) < 2:
            return
        
        series.yoy_change = valid_values[0][1] - valid_values[1][1]
        series.period_change = valid_values[0][1] - valid_values[-1][1]
        
        if len(valid_values) >= RATIO_ANALYSIS_CONFIG.min_years_for_trend:
            cv = np.std(values) / abs(np.mean(values)) if np.mean(values) != 0 else 0
            
            if cv > RATIO_ANALYSIS_CONFIG.volatility_cv_threshold:
                series.trend = RatioTrend.VOLATILE
            elif series.period_change > RATIO_ANALYSIS_CONFIG.trend_improving_threshold:
                series.trend = RatioTrend.IMPROVING
            elif series.period_change < RATIO_ANALYSIS_CONFIG.trend_deteriorating_threshold:
                series.trend = RatioTrend.DETERIORATING
            else:
                series.trend = RatioTrend.STABLE


# =============================================================================
# EFFICIENCY RATIO CALCULATOR
# =============================================================================

class EfficiencyCalculator(RatioCalculatorBase):
    """
    Calculates efficiency/activity ratios measuring operational effectiveness.
    
    Ratios:
    - Asset Turnover: Revenue / Average Assets
    - Fixed Asset Turnover: Revenue / Average PPE
    - Inventory Turnover: COGS / Average Inventory
    - Receivables Turnover: Revenue / Average Receivables
    - Payables Turnover: COGS / Average Payables
    - Days Sales Outstanding (DSO): 365 / Receivables Turnover
    - Days Inventory Outstanding (DIO): 365 / Inventory Turnover
    - Days Payables Outstanding (DPO): 365 / Payables Turnover
    - Cash Conversion Cycle (CCC): DSO + DIO - DPO
    """
    
    def calculate_all(
        self,
        income: pd.DataFrame,
        balance: pd.DataFrame,
        years: List[str],
    ) -> Dict[str, RatioTimeSeries]:
        """Calculate all efficiency ratios for all years."""
        results = {}
        
        for ratio_name, definition in EFFICIENCY_RATIOS.items():
            series = RatioTimeSeries(ratio_name=ratio_name, definition=definition)
            
            for year in years:
                ratio_value = self._calculate_single(ratio_name, income, balance, year, years)
                series.values[year] = ratio_value
            
            self._compute_trend(series, years, ratio_name)
            results[ratio_name] = series
        
        return results
    
    def _calculate_single(
        self,
        ratio_name: str,
        income: pd.DataFrame,
        balance: pd.DataFrame,
        year: str,
        years: List[str],
    ) -> RatioValue:
        """Calculate a single efficiency ratio for one year."""
        
        # Income statement items (in raw dollars)
        revenue = self.get_value(income, "total_revenue", year)
        cogs = self.get_value(income, "cogs", year)
        if cogs is None:
            cogs = self.get_value(income, "cost_of_revenue", year)
        
        # Balance sheet items (use averages - already in raw dollars like income statement)
        avg_assets = self.get_average(balance, "total_assets", year, years)
        avg_ppe = self.get_average(balance, "ppe_gross", year, years)
        avg_inventory = self.get_average(balance, "inventory", year, years)
        avg_receivables = self.get_average(balance, "accounts_receivable", year, years)
        avg_payables = self.get_average(balance, "accounts_payable", year, years)
        
        if ratio_name == "asset_turnover":
            return self.calculate_ratio(ratio_name, revenue, avg_assets, year)
        
        elif ratio_name == "fixed_asset_turnover":
            return self.calculate_ratio(ratio_name, revenue, avg_ppe, year)
        
        elif ratio_name == "inventory_turnover":
            return self.calculate_ratio(ratio_name, cogs, avg_inventory, year)
        
        elif ratio_name == "receivables_turnover":
            return self.calculate_ratio(ratio_name, revenue, avg_receivables, year)
        
        elif ratio_name == "payables_turnover":
            return self.calculate_ratio(ratio_name, cogs, avg_payables, year)
        
        elif ratio_name == "days_sales_outstanding":
            recv_turnover = self.safe_divide(revenue, avg_receivables)
            dso = self.safe_divide(365.0, recv_turnover)
            assessment = assess_ratio_value(ratio_name, dso)
            return RatioValue(
                ratio_name=ratio_name,
                fiscal_year=year,
                value=dso,
                assessment=assessment,
                numerator=365.0,
                denominator=recv_turnover,
            )
        
        elif ratio_name == "days_inventory_outstanding":
            inv_turnover = self.safe_divide(cogs, avg_inventory)
            dio = self.safe_divide(365.0, inv_turnover)
            assessment = assess_ratio_value(ratio_name, dio)
            return RatioValue(
                ratio_name=ratio_name,
                fiscal_year=year,
                value=dio,
                assessment=assessment,
                numerator=365.0,
                denominator=inv_turnover,
            )
        
        elif ratio_name == "days_payables_outstanding":
            pay_turnover = self.safe_divide(cogs, avg_payables)
            dpo = self.safe_divide(365.0, pay_turnover)
            assessment = assess_ratio_value(ratio_name, dpo)
            return RatioValue(
                ratio_name=ratio_name,
                fiscal_year=year,
                value=dpo,
                assessment=assessment,
                numerator=365.0,
                denominator=pay_turnover,
            )
        
        elif ratio_name == "cash_conversion_cycle":
            # CCC = DSO + DIO - DPO
            recv_turnover = self.safe_divide(revenue, avg_receivables)
            inv_turnover = self.safe_divide(cogs, avg_inventory)
            pay_turnover = self.safe_divide(cogs, avg_payables)
            
            dso = self.safe_divide(365.0, recv_turnover)
            dio = self.safe_divide(365.0, inv_turnover)
            dpo = self.safe_divide(365.0, pay_turnover)
            
            ccc = None
            if all(v is not None for v in [dso, dio, dpo]):
                ccc = dso + dio - dpo
            
            assessment = assess_ratio_value(ratio_name, ccc)
            return RatioValue(
                ratio_name=ratio_name,
                fiscal_year=year,
                value=ccc,
                assessment=assessment,
                calculation_note=f"DSO={dso:.1f}, DIO={dio:.1f}, DPO={dpo:.1f}" if ccc else None,
            )
        
        else:
            return RatioValue(
                ratio_name=ratio_name,
                fiscal_year=year,
                value=None,
                assessment=RatioAssessment.NOT_APPLICABLE,
                calculation_note="Unknown ratio",
            )
    
    def _compute_trend(self, series: RatioTimeSeries, years: List[str], ratio_name: str) -> None:
        """Compute trend analysis for a ratio series."""
        valid_values = [(y, series.values[y].value) for y in years if series.values.get(y) and series.values[y].value is not None]
        
        if not valid_values:
            return
        
        series.latest_value = valid_values[0][1]
        series.latest_assessment = series.values[valid_values[0][0]].assessment
        
        values = [v[1] for v in valid_values]
        series.average_value = np.mean(values)
        
        if len(valid_values) < 2:
            return
        
        series.yoy_change = valid_values[0][1] - valid_values[1][1]
        series.period_change = valid_values[0][1] - valid_values[-1][1]
        
        if len(valid_values) >= RATIO_ANALYSIS_CONFIG.min_years_for_trend:
            cv = np.std(values) / abs(np.mean(values)) if np.mean(values) != 0 else 0
            
            # For inverted ratios, decreasing = improving
            if series.definition.invert_assessment:
                if cv > RATIO_ANALYSIS_CONFIG.volatility_cv_threshold:
                    series.trend = RatioTrend.VOLATILE
                elif series.period_change < -RATIO_ANALYSIS_CONFIG.trend_improving_threshold:
                    series.trend = RatioTrend.IMPROVING
                elif series.period_change > RATIO_ANALYSIS_CONFIG.trend_deteriorating_threshold:
                    series.trend = RatioTrend.DETERIORATING
                else:
                    series.trend = RatioTrend.STABLE
            else:
                if cv > RATIO_ANALYSIS_CONFIG.volatility_cv_threshold:
                    series.trend = RatioTrend.VOLATILE
                elif series.period_change > RATIO_ANALYSIS_CONFIG.trend_improving_threshold:
                    series.trend = RatioTrend.IMPROVING
                elif series.period_change < RATIO_ANALYSIS_CONFIG.trend_deteriorating_threshold:
                    series.trend = RatioTrend.DETERIORATING
                else:
                    series.trend = RatioTrend.STABLE


# =============================================================================
# GROWTH RATIO CALCULATOR
# =============================================================================

class GrowthRatioCalculator(RatioCalculatorBase):
    """
    Calculates growth and payout ratios.
    
    Ratios:
    - Revenue CAGR (5Y)
    - Net Income CAGR (5Y)
    - FCF CAGR (5Y)
    - Dividend CAGR (5Y)
    - Sustainable Growth Rate
    - Dividend Payout Ratio
    - Retention Ratio
    """
    
    def calculate_all(
        self,
        income: pd.DataFrame,
        balance: pd.DataFrame,
        cashflow: pd.DataFrame,
        derived: Dict[str, Dict[str, Optional[float]]],
        dividend_history: Any,
        years: List[str],
    ) -> Dict[str, RatioTimeSeries]:
        """Calculate all growth ratios."""
        # Store balance sheet for SGR calculation
        self._balance = balance
        
        results = {}
        
        for ratio_name, definition in GROWTH_RATIOS.items():
            series = RatioTimeSeries(ratio_name=ratio_name, definition=definition)
            
            # CAGR ratios only have one value (covering the full period)
            if ratio_name in ("revenue_cagr_5y", "net_income_cagr_5y", "fcf_cagr_5y", "dividend_cagr_5y"):
                ratio_value = self._calculate_cagr(ratio_name, income, derived, dividend_history, years)
                if ratio_value.value is not None:
                    series.values[years[0]] = ratio_value
                    series.latest_value = ratio_value.value
                    series.latest_assessment = ratio_value.assessment
                    series.average_value = ratio_value.value
                    
                    # Calculate volatility for more accurate trend assessment
                    volatility = self._calculate_metric_volatility(ratio_name, income, derived, years)
                    
                    # Set trend based on CAGR value AND volatility
                    if volatility is not None and volatility > 15:  # High volatility (>15% std dev of YoY)
                        if ratio_value.value > 0.05:  # > 5% CAGR with high vol = still improving
                            series.trend = RatioTrend.IMPROVING
                        elif ratio_value.value < -0.05:  # < -5% CAGR with high vol = deteriorating
                            series.trend = RatioTrend.DETERIORATING
                        else:
                            series.trend = RatioTrend.VOLATILE  # Use new VOLATILE trend
                    else:  # Normal volatility
                        if ratio_value.value > 0.02:  # > 2% growth
                            series.trend = RatioTrend.IMPROVING
                        elif ratio_value.value < -0.02:  # < -2% (decline)
                            series.trend = RatioTrend.DETERIORATING
                        else:
                            series.trend = RatioTrend.STABLE
            else:
                # Annual ratios
                for year in years:
                    ratio_value = self._calculate_annual(ratio_name, income, cashflow, derived, dividend_history, year, years)
                    series.values[year] = ratio_value
                
                self._compute_trend(series, years)
            
            results[ratio_name] = series
        
        return results
    
    def _calculate_cagr(
        self,
        ratio_name: str,
        income: pd.DataFrame,
        derived: Dict[str, Dict[str, Optional[float]]],
        dividend_history: Any,
        years: List[str],
    ) -> RatioValue:
        """Calculate CAGR for a metric over the full period."""
        
        if len(years) < 2:
            return RatioValue(
                ratio_name=ratio_name,
                fiscal_year=years[0] if years else "N/A",
                value=None,
                assessment=RatioAssessment.NOT_APPLICABLE,
                calculation_note="Insufficient years for CAGR",
            )
        
        # Get start and end values
        start_year = years[-1]  # Oldest
        end_year = years[0]     # Most recent
        n_years = len(years) - 1
        
        if ratio_name == "revenue_cagr_5y":
            start_val = self.get_value(income, "total_revenue", start_year)
            end_val = self.get_value(income, "total_revenue", end_year)
        elif ratio_name == "net_income_cagr_5y":
            start_val = self.get_value(income, "net_income", start_year)
            end_val = self.get_value(income, "net_income", end_year)
        elif ratio_name == "fcf_cagr_5y":
            start_val = derived.get("fcf_calculated", {}).get(start_year)
            end_val = derived.get("fcf_calculated", {}).get(end_year)
        elif ratio_name == "dividend_cagr_5y":
            if dividend_history and hasattr(dividend_history, 'dividend_cagr'):
                cagr = dividend_history.dividend_cagr
                assessment = assess_ratio_value(ratio_name, cagr) if cagr else RatioAssessment.NOT_APPLICABLE
                return RatioValue(
                    ratio_name=ratio_name,
                    fiscal_year=end_year,
                    value=cagr,
                    assessment=assessment,
                    calculation_note="From dividend history",
                )
            return RatioValue(
                ratio_name=ratio_name,
                fiscal_year=end_year,
                value=None,
                assessment=RatioAssessment.NOT_APPLICABLE,
                calculation_note="No dividend history",
            )
        else:
            return RatioValue(
                ratio_name=ratio_name,
                fiscal_year=end_year,
                value=None,
                assessment=RatioAssessment.NOT_APPLICABLE,
                calculation_note="Unknown CAGR ratio",
            )
        
        # Calculate CAGR
        cagr = None
        if start_val and end_val and start_val > 0 and end_val > 0:
            cagr = (end_val / start_val) ** (1 / n_years) - 1
        elif start_val and end_val and start_val < 0 and end_val < 0:
            # Handle negative values (e.g., negative earnings improving)
            cagr = -((abs(end_val) / abs(start_val)) ** (1 / n_years) - 1)
        
        assessment = assess_ratio_value(ratio_name, cagr) if cagr is not None else RatioAssessment.NOT_APPLICABLE
        
        return RatioValue(
            ratio_name=ratio_name,
            fiscal_year=end_year,
            value=cagr,
            assessment=assessment,
            numerator=end_val,
            denominator=start_val,
            calculation_note=f"{n_years}Y CAGR",
        )
    
    def _calculate_annual(
        self,
        ratio_name: str,
        income: pd.DataFrame,
        cashflow: pd.DataFrame,
        derived: Dict[str, Dict[str, Optional[float]]],
        dividend_history: Any,
        year: str,
        years: List[str],
    ) -> RatioValue:
        """Calculate annual growth/payout ratios."""
        
        net_income = self.get_value(income, "net_income", year)
        dividends = self.get_value(cashflow, "dividends_paid", year)
        
        if ratio_name == "dividend_payout_ratio":
            return self.calculate_ratio(ratio_name, dividends, net_income, year)
        
        elif ratio_name == "retention_ratio":
            payout = self.safe_divide(dividends, net_income)
            retention = 1 - payout if payout is not None else None
            assessment = assess_ratio_value(ratio_name, retention) if retention is not None else RatioAssessment.NOT_APPLICABLE
            return RatioValue(
                ratio_name=ratio_name,
                fiscal_year=year,
                value=retention,
                assessment=assessment,
                calculation_note="1 - Payout Ratio",
            )
        
        elif ratio_name == "sustainable_growth_rate":
            # SGR = ROE x Retention Ratio (capped at 30% for reasonableness)
            payout = self.safe_divide(dividends, net_income)
            retention = 1 - payout if payout is not None else None
            
            # Calculate ROE using stored balance sheet
            roe = None
            if net_income is not None and hasattr(self, '_balance') and self._balance is not None:
                avg_equity = self.get_average(self._balance, "total_equity", year, years)
                roe = self.safe_divide(net_income, avg_equity)
            
            sgr = None
            sgr_note = "Missing data"
            if roe is not None and retention is not None:
                raw_sgr = roe * retention
                # Cap SGR at 30% (0.30) - values above this are economically unrealistic
                # High SGR often results from artificially low equity (e.g., due to buybacks)
                MAX_SGR = 0.30
                if raw_sgr > MAX_SGR:
                    sgr = MAX_SGR
                    sgr_note = f"Capped at 30% (raw: {raw_sgr*100:.1f}% from ROE={roe*100:.1f}%)"
                else:
                    sgr = raw_sgr
                    sgr_note = f"ROE={roe*100:.1f}% x Ret={retention*100:.1f}%"
            
            assessment = assess_ratio_value(ratio_name, sgr) if sgr is not None else RatioAssessment.NOT_APPLICABLE
            
            return RatioValue(
                ratio_name=ratio_name,
                fiscal_year=year,
                value=sgr,
                assessment=assessment,
                numerator=roe,
                denominator=retention,
                calculation_note=sgr_note,
            )
        
        else:
            return RatioValue(
                ratio_name=ratio_name,
                fiscal_year=year,
                value=None,
                assessment=RatioAssessment.NOT_APPLICABLE,
                calculation_note="Unknown ratio",
            )
    
    def _calculate_metric_volatility(
        self,
        ratio_name: str,
        income: pd.DataFrame,
        derived: Dict[str, Dict[str, Optional[float]]],
        years: List[str],
    ) -> Optional[float]:
        """
        Calculate year-over-year volatility (std dev of YoY changes) for a metric.
        
        Returns volatility as a percentage (e.g., 16.8 means 16.8% std dev).
        """
        values = []
        
        for year in years:
            if ratio_name == "revenue_cagr_5y":
                val = self.get_value(income, "total_revenue", year)
            elif ratio_name == "net_income_cagr_5y":
                val = self.get_value(income, "net_income", year)
            elif ratio_name == "fcf_cagr_5y":
                val = derived.get("fcf_calculated", {}).get(year)
            elif ratio_name == "dividend_cagr_5y":
                val = derived.get("dividends_paid", {}).get(year)
            else:
                return None
            
            if val is not None and val > 0:
                values.append(val)
        
        if len(values) < 3:
            return None
        
        # Calculate year-over-year percentage changes
        yoy_changes = []
        for i in range(len(values) - 1):
            if values[i+1] > 0:  # Avoid division by zero
                yoy = (values[i] / values[i+1] - 1) * 100  # As percentage
                yoy_changes.append(yoy)
        
        if len(yoy_changes) < 2:
            return None
        
        # Return standard deviation of YoY changes
        import statistics
        try:
            return statistics.stdev(yoy_changes)
        except:
            return None
    
    def _compute_trend(self, series: RatioTimeSeries, years: List[str]) -> None:
        """Compute trend for annual ratios."""
        valid_values = [
            (y, series.values[y].value) 
            for y in years 
            if series.values.get(y) and series.values[y].value is not None
        ]
        
        if not valid_values:
            return
        
        series.latest_value = valid_values[0][1]
        series.latest_assessment = series.values[valid_values[0][0]].assessment
        
        values = [v[1] for v in valid_values]
        series.average_value = np.mean(values)
        
        if len(valid_values) >= 2:
            series.yoy_change = valid_values[0][1] - valid_values[1][1]
            series.period_change = valid_values[0][1] - valid_values[-1][1]
        
        # Trend classification
        if len(valid_values) >= RATIO_ANALYSIS_CONFIG.min_years_for_trend:
            cv = np.std(values) / abs(np.mean(values)) if np.mean(values) != 0 else 0
            
            if cv > RATIO_ANALYSIS_CONFIG.volatility_cv_threshold:
                series.trend = RatioTrend.VOLATILE
            elif series.period_change > RATIO_ANALYSIS_CONFIG.trend_improving_threshold:
                series.trend = RatioTrend.IMPROVING
            elif series.period_change < RATIO_ANALYSIS_CONFIG.trend_deteriorating_threshold:
                series.trend = RatioTrend.DETERIORATING
            else:
                series.trend = RatioTrend.STABLE


# =============================================================================
# CATEGORY ANALYZER
# =============================================================================

class CategoryAnalyzer:
    """
    Analyzes a category of ratios and produces summary scores.
    
    Computes category-level scores, identifies strengths and weaknesses,
    and produces assessment distributions.
    """
    
    def analyze(
        self,
        category: RatioCategory,
        ratios: Dict[str, RatioTimeSeries],
    ) -> CategoryAnalysis:
        """
        Analyze a category of ratios.
        
        Args:
            category: The ratio category
            ratios: Dictionary of computed ratio time series
            
        Returns:
            CategoryAnalysis with scores and findings
        """
        analysis = CategoryAnalysis(category=category, ratios=ratios)
        
        if not ratios:
            return analysis
        
        # Compute assessment distribution and score
        assessment_counts = {a.value: 0 for a in RatioAssessment}
        scores = []
        
        for ratio_name, series in ratios.items():
            if series.latest_assessment != RatioAssessment.NOT_APPLICABLE:
                assessment_counts[series.latest_assessment.value] += 1
                scores.append(self._assessment_to_score(series.latest_assessment))
                
                # Identify strengths and weaknesses
                if series.latest_assessment in (RatioAssessment.EXCELLENT, RatioAssessment.GOOD):
                    trend_str = f" ({series.trend.value})" if series.trend != RatioTrend.INSUFFICIENT_DATA else ""
                    analysis.strengths.append(f"{series.definition.name}: {series.latest_assessment.value}{trend_str}")
                elif series.latest_assessment in (RatioAssessment.WEAK, RatioAssessment.CRITICAL):
                    trend_str = f" ({series.trend.value})" if series.trend != RatioTrend.INSUFFICIENT_DATA else ""
                    analysis.weaknesses.append(f"{series.definition.name}: {series.latest_assessment.value}{trend_str}")
        
        analysis.assessment_distribution = {k: v for k, v in assessment_counts.items() if v > 0}
        
        if scores:
            analysis.category_score = np.mean(scores)
        
        return analysis
    
    def _assessment_to_score(self, assessment: RatioAssessment) -> float:
        """Convert assessment to numeric score."""
        score_map = {
            RatioAssessment.EXCELLENT: RATIO_ANALYSIS_CONFIG.score_excellent,
            RatioAssessment.GOOD: RATIO_ANALYSIS_CONFIG.score_good,
            RatioAssessment.ACCEPTABLE: RATIO_ANALYSIS_CONFIG.score_acceptable,
            RatioAssessment.WEAK: RATIO_ANALYSIS_CONFIG.score_weak,
            RatioAssessment.CRITICAL: RATIO_ANALYSIS_CONFIG.score_critical,
            RatioAssessment.NOT_APPLICABLE: 0.5,  # Neutral
        }
        return score_map.get(assessment, 0.5)


# =============================================================================
# MAIN PHASE 3 ANALYZER
# =============================================================================

class Phase3Analyzer:
    """
    Main orchestrator for Phase 3 financial ratio analysis.
    
    Coordinates all ratio calculations across categories and produces
    comprehensive financial analysis results.
    
    Usage:
        analyzer = Phase3Analyzer()
        result = analyzer.analyze(collection_result)
        
        # Access specific ratios
        roe = result.get_ratio("roe")
        print(f"ROE: {roe.latest_value:.2%}")
        
        # Access category scores
        print(f"Profitability: {result.profitability.category_score:.2%}")
    """
    
    def __init__(self):
        """Initialize ratio calculators."""
        self.profitability_calc = ProfitabilityCalculator()
        self.leverage_calc = LeverageCalculator()
        self.liquidity_calc = LiquidityCalculator()
        self.efficiency_calc = EfficiencyCalculator()
        self.growth_calc = GrowthRatioCalculator()
        self.category_analyzer = CategoryAnalyzer()
        
        LOGGER.info(f"Phase3Analyzer initialized (v{__version__})")
    
    def analyze(self, collection_result) -> RatioAnalysisResult:
        """
        Perform comprehensive ratio analysis on financial data.
        
        Args:
            collection_result: CollectionResult from Phase 1 or ValidatedData from Phase 2
            
        Returns:
            RatioAnalysisResult with all computed ratios
        """
        LOGGER.info(f"Phase 3: Starting ratio analysis for {collection_result.ticker}")
        
        # Extract data
        income = collection_result.statements.income_statement
        balance = collection_result.statements.balance_sheet
        cashflow = collection_result.statements.cash_flow
        years = collection_result.statements.fiscal_periods
        
        # Get derived metrics and dividend history
        derived = self._extract_derived_metrics(collection_result)
        dividend_history = getattr(collection_result, 'dividend_history', None)
        
        # Calculate all ratio categories
        LOGGER.info("  Calculating profitability ratios")
        profitability_ratios = self.profitability_calc.calculate_all(income, balance, derived, years)
        
        LOGGER.info("  Calculating leverage ratios")
        leverage_ratios = self.leverage_calc.calculate_all(income, balance, cashflow, derived, years)
        
        LOGGER.info("  Calculating liquidity ratios")
        liquidity_ratios = self.liquidity_calc.calculate_all(balance, cashflow, derived, years)
        
        LOGGER.info("  Calculating efficiency ratios")
        efficiency_ratios = self.efficiency_calc.calculate_all(income, balance, years)
        
        LOGGER.info("  Calculating growth ratios")
        growth_ratios = self.growth_calc.calculate_all(income, balance, cashflow, derived, dividend_history, years)
        
        # Analyze categories
        LOGGER.info("  Analyzing categories")
        profitability_analysis = self.category_analyzer.analyze(RatioCategory.PROFITABILITY, profitability_ratios)
        leverage_analysis = self.category_analyzer.analyze(RatioCategory.LEVERAGE, leverage_ratios)
        liquidity_analysis = self.category_analyzer.analyze(RatioCategory.LIQUIDITY, liquidity_ratios)
        efficiency_analysis = self.category_analyzer.analyze(RatioCategory.EFFICIENCY, efficiency_ratios)
        growth_analysis = self.category_analyzer.analyze(RatioCategory.EFFICIENCY, growth_ratios)  # Using EFFICIENCY as placeholder
        
        # Compute overall assessment
        overall = self._compute_overall_assessment(
            profitability_analysis,
            leverage_analysis,
            liquidity_analysis,
            efficiency_analysis,
            growth_analysis,
        )
        
        # Count ratios
        total_ratios = (
            len(profitability_ratios) + len(leverage_ratios) +
            len(liquidity_ratios) + len(efficiency_ratios) + len(growth_ratios)
        )
        
        LOGGER.info(
            f"Phase 3 complete for {collection_result.ticker}: "
            f"{total_ratios} ratios, overall={overall.overall_score:.2%}"
        )
        
        return RatioAnalysisResult(
            ticker=collection_result.ticker,
            company_name=collection_result.company_name,
            fiscal_periods=years,
            profitability=profitability_analysis,
            leverage=leverage_analysis,
            liquidity=liquidity_analysis,
            efficiency=efficiency_analysis,
            growth=growth_analysis,
            overall_assessment=overall,
            ratios_calculated=total_ratios,
        )
    
    def _extract_derived_metrics(self, collection_result) -> Dict[str, Dict[str, Optional[float]]]:
        """Extract derived metrics from collection result."""
        derived = {}
        
        if hasattr(collection_result, 'derived_metrics'):
            dm = collection_result.derived_metrics
            derived["fcf_calculated"] = getattr(dm, 'fcf_calculated', {})
            derived["ebitda_calculated"] = getattr(dm, 'ebitda_calculated', {})
            derived["working_capital"] = getattr(dm, 'working_capital', {})
            derived["net_debt"] = getattr(dm, 'net_debt', {})
            derived["invested_capital"] = getattr(dm, 'invested_capital', {})
        
        return derived
    
    def _compute_overall_assessment(
        self,
        profitability: CategoryAnalysis,
        leverage: CategoryAnalysis,
        liquidity: CategoryAnalysis,
        efficiency: CategoryAnalysis,
        growth: CategoryAnalysis,
    ) -> OverallAssessment:
        """Compute overall financial health assessment."""
        overall = OverallAssessment()
        
        # Store category scores
        overall.profitability_score = profitability.category_score
        overall.leverage_score = leverage.category_score
        overall.liquidity_score = liquidity.category_score
        overall.efficiency_score = efficiency.category_score
        
        # Weighted overall score (4 main categories)
        overall.overall_score = (
            profitability.category_score * RATIO_ANALYSIS_CONFIG.profitability_weight +
            leverage.category_score * RATIO_ANALYSIS_CONFIG.leverage_weight +
            liquidity.category_score * RATIO_ANALYSIS_CONFIG.liquidity_weight +
            efficiency.category_score * RATIO_ANALYSIS_CONFIG.efficiency_weight
        )
        
        # Determine overall assessment
        if overall.overall_score >= 0.85:
            overall.overall_assessment = RatioAssessment.EXCELLENT
        elif overall.overall_score >= 0.70:
            overall.overall_assessment = RatioAssessment.GOOD
        elif overall.overall_score >= 0.50:
            overall.overall_assessment = RatioAssessment.ACCEPTABLE
        elif overall.overall_score >= 0.30:
            overall.overall_assessment = RatioAssessment.WEAK
        else:
            overall.overall_assessment = RatioAssessment.CRITICAL
        
        # Collect key findings from all categories including growth
        for category in [profitability, leverage, liquidity, efficiency, growth]:
            overall.key_strengths.extend(category.strengths[:2])
            overall.key_weaknesses.extend(category.weaknesses[:2])
        
        # Identify key risks
        if leverage.category_score < 0.5:
            overall.key_risks.append("High financial leverage risk")
        if liquidity.category_score < 0.5:
            overall.key_risks.append("Liquidity concerns")
        if profitability.category_score < 0.5:
            overall.key_risks.append("Weak profitability")
        if growth.category_score < 0.4:
            overall.key_risks.append("Weak growth profile")
        
        return overall
    
    def save_report(self, result: RatioAnalysisResult, output_dir: Path = None) -> Path:
        """Save ratio analysis report to JSON file."""
        if output_dir is None:
            output_dir = OUTPUT_DIR
        
        ticker_dir = output_dir / result.ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = ticker_dir / f"{result.ticker}_ratio_analysis.json"
        
        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        LOGGER.info(f"Saved ratio analysis to {filepath}")
        return filepath


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def analyze_financial_ratios(collection_result) -> RatioAnalysisResult:
    """
    Convenience function to perform ratio analysis.
    
    Args:
        collection_result: CollectionResult from Phase 1
        
    Returns:
        RatioAnalysisResult with all computed ratios
    """
    analyzer = Phase3Analyzer()
    return analyzer.analyze(collection_result)