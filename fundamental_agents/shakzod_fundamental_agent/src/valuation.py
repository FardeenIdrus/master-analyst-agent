"""
Valuation Analysis Module for Fundamental Analyst Agent

This module performs comprehensive valuation analysis by calculating and
interpreting valuation multiples, comparing to historical averages, and
providing a framework for relative valuation assessment.

VALUATION FRAMEWORK
===================

1. EARNINGS-BASED MULTIPLES
   - P/E (Price-to-Earnings): Market cap relative to net income
   - Forward P/E: Based on analyst estimates (if available)
   - PEG Ratio: P/E relative to growth rate

2. ASSET-BASED MULTIPLES
   - P/B (Price-to-Book): Market cap relative to book equity
   - P/TBV (Price-to-Tangible Book): Excludes intangibles

3. REVENUE-BASED MULTIPLES
   - P/S (Price-to-Sales): Market cap relative to revenue
   - EV/Revenue: Enterprise value relative to revenue

4. CASH FLOW MULTIPLES
   - EV/EBITDA: Enterprise value relative to EBITDA
   - EV/EBIT: Enterprise value relative to operating income
   - P/FCF: Price relative to free cash flow

5. ENTERPRISE VALUE CALCULATION
   EV = Market Cap + Total Debt - Cash & Equivalents
   
   Enterprise Value represents the theoretical takeover price and is
   used for multiples that consider the entire capital structure.

VALUATION ASSESSMENT
====================
The module provides a qualitative assessment:
- SIGNIFICANTLY_UNDERVALUED: Multiple <50% of fair value estimate
- UNDERVALUED: Multiple 50-80% of fair value
- FAIRLY_VALUED: Multiple 80-120% of fair value
- OVERVALUED: Multiple 120-150% of fair value
- SIGNIFICANTLY_OVERVALUED: Multiple >150% of fair value

Assessment is based on:
1. Comparison to historical average multiples
2. Implied growth rates vs actual growth
3. Multiple relative to profitability (P/E vs ROE)

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
    VALUATION_PARAMS,
    ValuationAssessment,
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
# CONSTANTS AND ENUMERATIONS
# =============================================================================

class MultipleType(Enum):
    """Types of valuation multiples."""
    EARNINGS = "Earnings-Based"
    ASSET = "Asset-Based"
    REVENUE = "Revenue-Based"
    CASH_FLOW = "Cash Flow-Based"


class MultipleQuality(Enum):
    """Quality/reliability of a multiple for valuation."""
    HIGH = "High"       # Reliable for most companies
    MODERATE = "Moderate"  # Useful with caveats
    LOW = "Low"         # Limited applicability
    NOT_APPLICABLE = "N/A"  # Cannot be calculated


# =============================================================================
# DATA CLASSES
# =============================================================================

class MultipleList(list):
    """
    A list that also supports dict-like .get() access.
    
    This class allows accessing multiples either as a list or via .get(key)
    where the key is matched against the multiple abbreviation.
    
    Used for backward compatibility with code that expects dict-like access.
    """
    
    def get(self, key: str, default=None):
        """
        Get a multiple value by abbreviation (dict-like access).
        
        Args:
            key: Multiple abbreviation like 'pe' or 'P/E' (case-insensitive).
            default: Value to return if not found.
        
        Returns:
            The multiple's current_value or default if not found.
        """
        # Normalize key: pe -> p/e, ev_ebitda -> ev/ebitda
        key_lower = key.lower().replace('_', '/')
        
        for item in self:
            if hasattr(item, 'abbreviation'):
                abbr_lower = item.abbreviation.lower()
                if abbr_lower == key_lower:
                    return item.current_value if item.current_value is not None else default
        return default
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to a plain dict keyed by lowercase abbreviation."""
        result = {}
        for item in self:
            if hasattr(item, 'abbreviation') and hasattr(item, 'current_value'):
                key = item.abbreviation.lower().replace('/', '_')
                result[key] = item.current_value if item.current_value is not None else 0.0
        return result


@dataclass
class ValuationMultiple:
    """
    A single valuation multiple with context.
    
    Attributes:
        name: Full name of the multiple
        abbreviation: Short form (e.g., 'P/E')
        multiple_type: Category of multiple
        current_value: Current multiple value
        historical_average: Average over available history
        historical_median: Median over available history
        percentile_rank: Where current sits vs history (0-100)
        sector_average: Sector average (if available)
        quality: Reliability rating for this multiple
        interpretation: What the multiple indicates
        formula: Formula description
        numerator: What the numerator represents
        denominator: What the denominator represents
    """
    name: str
    abbreviation: str
    multiple_type: MultipleType
    current_value: Optional[float]
    historical_average: Optional[float]
    historical_median: Optional[float]
    percentile_rank: Optional[float]
    sector_average: Optional[float]
    quality: MultipleQuality
    interpretation: str
    formula: str
    numerator: str
    denominator: str


@dataclass
class EnterpriseValueBreakdown:
    """
    Breakdown of Enterprise Value calculation.
    
    EV = Market Cap + Total Debt - Cash
    
    Attributes:
        market_cap: Market capitalization
        total_debt: Total debt (short + long term)
        cash: Cash and cash equivalents
        enterprise_value: Calculated EV
        net_debt: Total Debt - Cash
        ev_to_market_cap: EV / Market Cap ratio
    """
    market_cap: float
    total_debt: float
    cash: float
    enterprise_value: float
    net_debt: float
    ev_to_market_cap: float


@dataclass
class ImpliedGrowthAnalysis:
    """
    Analysis of growth implied by current valuation.
    
    Attributes:
        current_pe: Current P/E ratio
        cost_of_equity: Estimated cost of equity
        implied_growth_rate: Growth rate implied by P/E
        historical_growth_rate: Actual historical growth
        growth_gap: Implied vs actual gap
        interpretation: What the gap indicates
    """
    current_pe: float
    cost_of_equity: float
    implied_growth_rate: float
    historical_growth_rate: float
    growth_gap: float
    interpretation: str


@dataclass
class RelativeValuationSummary:
    """
    Summary of relative valuation assessment.
    
    Attributes:
        primary_multiple: The most relevant multiple for this company
        primary_value: Value of the primary multiple
        vs_history: Assessment vs historical average
        overall_assessment: Overall valuation assessment
        confidence_level: Confidence in the assessment
        key_considerations: Important factors to consider
    """
    primary_multiple: str
    primary_value: float
    vs_history: str
    overall_assessment: ValuationAssessment
    confidence_level: str
    key_considerations: List[str]


@dataclass
class ValuationAnalysisResult:
    """
    Complete result of valuation analysis.
    
    This is the primary output of the ValuationAnalyzer class.
    
    Attributes:
        ticker: Company ticker symbol
        analysis_date: Date of analysis
        ev_breakdown: Enterprise value calculation
        multiples: List of calculated valuation multiples
        implied_growth: Implied growth analysis
        relative_summary: Relative valuation summary
        valuation_assessment: Overall valuation assessment
        key_multiples: Dictionary of key multiples for quick access
        insights: Key valuation insights
        warnings: Data quality warnings
        analysis_timestamp: When analysis was performed
    """
    ticker: str
    analysis_date: str
    ev_breakdown: EnterpriseValueBreakdown
    multiples: List[ValuationMultiple]
    implied_growth: Optional[ImpliedGrowthAnalysis]
    relative_summary: RelativeValuationSummary
    valuation_assessment: ValuationAssessment
    key_multiples: Dict[str, Optional[float]]
    insights: List[str]
    warnings: List[str]
    analysis_timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# VALUATION ANALYZER CLASS
# =============================================================================

class ValuationAnalyzer:
    """
    Comprehensive valuation analysis using multiples.
    
    This analyzer calculates valuation multiples, compares to historical
    averages, assesses implied growth, and provides an overall valuation
    assessment.
    
    Usage:
        analyzer = ValuationAnalyzer(processed_data)
        result = analyzer.analyze()
        
        # Access key multiples
        print(f"P/E: {result.key_multiples.get('P/E', 'N/A')}")
        print(f"EV/EBITDA: {result.key_multiples.get('EV/EBITDA', 'N/A')}")
        
        # Overall assessment
        print(f"Valuation: {result.valuation_assessment.value}")
    
    Attributes:
        _data: ProcessedData object containing standardized financial data
        _processor: DataProcessor instance for field extraction
        _warnings: List of warning messages
        _market_cap: Market capitalization from company info
        _enterprise_value: Calculated enterprise value
    """
    
    def __init__(self, processed_data: ProcessedData):
        """
        Initialize the ValuationAnalyzer.
        
        Args:
            processed_data: ProcessedData object from DataProcessor
        """
        self._data = processed_data
        self._processor = DataProcessor()
        self._warnings: List[str] = []
        
        # Extract market data (company_info is a dataclass, not a dict)
        # Scale market cap to millions to match financial statement data
        raw_market_cap = self._data.company_info.market_cap
        self._market_cap = raw_market_cap / 1e6 if raw_market_cap else None
        self._share_price = self._data.company_info.current_price
        self._shares_outstanding = self._data.company_info.shares_outstanding
        
        # Will be calculated
        self._enterprise_value: Optional[float] = None
        self._ev_breakdown: Optional[EnterpriseValueBreakdown] = None
        
        logger.info(f"ValuationAnalyzer initialized for {processed_data.ticker}")
    
    def analyze(self) -> ValuationAnalysisResult:
        """
        Perform complete valuation analysis.
        
        Returns:
            ValuationAnalysisResult with complete analysis
        """
        logger.info(f"Starting valuation analysis for {self._data.ticker}")
        
        # Reset warnings
        self._warnings = []
        
        # Get analysis date (periods are strings, not datetime objects)
        periods = self._data.statements.periods
        analysis_date = self._format_analysis_date(periods)
        
        # Calculate Enterprise Value
        ev_breakdown = self._calculate_enterprise_value()
        self._ev_breakdown = ev_breakdown
        self._enterprise_value = ev_breakdown.enterprise_value
        
        # Calculate all multiples
        multiples = self._calculate_all_multiples()
        
        # Build key multiples dictionary
        key_multiples = self._build_key_multiples_dict(multiples)
        
        # Analyze implied growth
        implied_growth = self._analyze_implied_growth(key_multiples)
        
        # Generate relative valuation summary
        relative_summary = self._generate_relative_summary(multiples, implied_growth)
        
        # Determine overall assessment
        valuation_assessment = self._determine_overall_assessment(
            multiples, implied_growth, relative_summary
        )
        
        # Generate insights
        insights = self._generate_insights(
            multiples, ev_breakdown, implied_growth, valuation_assessment
        )
        
        result = ValuationAnalysisResult(
            ticker=self._data.ticker,
            analysis_date=analysis_date,
            ev_breakdown=ev_breakdown,
            multiples=MultipleList(multiples),
            implied_growth=implied_growth,
            relative_summary=relative_summary,
            valuation_assessment=valuation_assessment,
            key_multiples=key_multiples,
            insights=insights,
            warnings=self._warnings.copy()
        )
        
        logger.info(
            f"Valuation analysis complete for {self._data.ticker}: "
            f"{valuation_assessment.value}"
        )
        return result
    
    def _format_analysis_date(self, periods: List[str]) -> str:
        """
        Format the analysis date string for display.
        
        Args:
            periods: List of fiscal period labels from processed data.
                    Periods are strings in format like "2024" or "2024-09-30".
        
        Returns:
            Formatted date string.
        """
        if not periods:
            return "Current"
        
        # Get the most recent period (first in list)
        period = periods[0]
        
        # Handle different period string formats
        if isinstance(period, str):
            if "-" in period:
                # Already in date format like "2024-09-30"
                return period
            elif period.startswith("FY"):
                # Extract year from "FY2024" format
                return period.replace("FY", "") + "-12-31"
            else:
                # Just a year like "2024", assume fiscal year end
                return f"{period}-12-31"
        else:
            # Fallback for unexpected types
            return str(period)
    
    # =========================================================================
    # ENTERPRISE VALUE CALCULATION
    # =========================================================================
    
    def _calculate_enterprise_value(self) -> EnterpriseValueBreakdown:
        """
        Calculate Enterprise Value and its components.
        
        EV = Market Cap + Total Debt - Cash
        
        Returns:
            EnterpriseValueBreakdown object
        """
        # Get market cap
        market_cap = self._market_cap
        if market_cap is None or market_cap <= 0:
            self._add_warning("Market cap unavailable - using estimated value")
            # Try to estimate from shares and price (scale to millions)
            if self._shares_outstanding and self._share_price:
                market_cap = (self._shares_outstanding * self._share_price) / 1e6
            else:
                market_cap = 0
        
        # Get debt and cash from financials
        total_debt = self._get_field(StandardField.TOTAL_DEBT, 0) or 0
        cash = self._get_field(StandardField.CASH, 0) or 0
        
        # Calculate EV
        enterprise_value = market_cap + total_debt - cash
        net_debt = total_debt - cash
        
        # EV to Market Cap ratio
        ev_to_mc = enterprise_value / market_cap if market_cap > 0 else 0
        
        return EnterpriseValueBreakdown(
            market_cap=market_cap,
            total_debt=total_debt,
            cash=cash,
            enterprise_value=enterprise_value,
            net_debt=net_debt,
            ev_to_market_cap=ev_to_mc
        )
    
    # =========================================================================
    # MULTIPLE CALCULATIONS
    # =========================================================================
    
    def _calculate_all_multiples(self) -> List[ValuationMultiple]:
        """
        Calculate all valuation multiples.
        
        Returns:
            List of ValuationMultiple objects
        """
        multiples = []
        
        # Earnings-based multiples
        multiples.extend(self._calculate_earnings_multiples())
        
        # Asset-based multiples
        multiples.extend(self._calculate_asset_multiples())
        
        # Revenue-based multiples
        multiples.extend(self._calculate_revenue_multiples())
        
        # Cash flow multiples
        multiples.extend(self._calculate_cashflow_multiples())
        
        return [m for m in multiples if m is not None]
    
    def _calculate_earnings_multiples(self) -> List[Optional[ValuationMultiple]]:
        """Calculate earnings-based valuation multiples."""
        multiples = []
        
        # P/E Ratio
        net_income = self._get_field(StandardField.NET_INCOME, 0)
        if self._market_cap and net_income and net_income > 0:
            pe = self._market_cap / net_income
            
            # Calculate historical P/E values for context
            historical_pe = self._calculate_historical_multiple(
                lambda mc, ni: mc / ni if ni and ni > 0 else None,
                StandardField.NET_INCOME
            )
            
            multiples.append(ValuationMultiple(
                name="Price-to-Earnings",
                abbreviation="P/E",
                multiple_type=MultipleType.EARNINGS,
                current_value=pe,
                historical_average=np.mean(historical_pe) if historical_pe else None,
                historical_median=np.median(historical_pe) if historical_pe else None,
                percentile_rank=self._calculate_percentile(pe, historical_pe),
                sector_average=getattr(VALUATION_PARAMS, "sector_pe_average", 18.0),
                quality=MultipleQuality.HIGH,
                interpretation=self._interpret_pe(pe),
                formula="Market Cap / Net Income",
                numerator="Market Capitalization",
                denominator="Net Income (TTM)"
            ))
        else:
            if net_income and net_income <= 0:
                multiples.append(ValuationMultiple(
                    name="Price-to-Earnings",
                    abbreviation="P/E",
                    multiple_type=MultipleType.EARNINGS,
                    current_value=None,
                    historical_average=None,
                    historical_median=None,
                    percentile_rank=None,
                    sector_average=getattr(VALUATION_PARAMS, "sector_pe_average", 18.0),
                    quality=MultipleQuality.NOT_APPLICABLE,
                    interpretation="P/E not meaningful - company has negative earnings",
                    formula="Market Cap / Net Income",
                    numerator="Market Capitalization",
                    denominator="Net Income (TTM)"
                ))
        
        # Earnings Yield (inverse of P/E)
        if self._market_cap and net_income and net_income > 0:
            earnings_yield = net_income / self._market_cap
            multiples.append(ValuationMultiple(
                name="Earnings Yield",
                abbreviation="E/P",
                multiple_type=MultipleType.EARNINGS,
                current_value=earnings_yield,
                historical_average=None,
                historical_median=None,
                percentile_rank=None,
                sector_average=None,
                quality=MultipleQuality.HIGH,
                interpretation=f"Earnings yield of {earnings_yield*100:.1f}% - compare to bond yields",
                formula="Net Income / Market Cap",
                numerator="Net Income (TTM)",
                denominator="Market Capitalization"
            ))
        
        return multiples
    
    def _calculate_asset_multiples(self) -> List[Optional[ValuationMultiple]]:
        """Calculate asset-based valuation multiples."""
        multiples = []
        
        # P/B Ratio
        book_value = self._get_field(StandardField.TOTAL_EQUITY, 0)
        if self._market_cap and book_value and book_value > 0:
            pb = self._market_cap / book_value
            
            historical_pb = self._calculate_historical_multiple(
                lambda mc, bv: mc / bv if bv and bv > 0 else None,
                StandardField.TOTAL_EQUITY
            )
            
            multiples.append(ValuationMultiple(
                name="Price-to-Book",
                abbreviation="P/B",
                multiple_type=MultipleType.ASSET,
                current_value=pb,
                historical_average=np.mean(historical_pb) if historical_pb else None,
                historical_median=np.median(historical_pb) if historical_pb else None,
                percentile_rank=self._calculate_percentile(pb, historical_pb),
                sector_average=getattr(VALUATION_PARAMS, "sector_pb_average", 3.0),
                quality=MultipleQuality.MODERATE,
                interpretation=self._interpret_pb(pb),
                formula="Market Cap / Book Value",
                numerator="Market Capitalization",
                denominator="Shareholders' Equity"
            ))
        
        # P/TBV (Price to Tangible Book)
        total_assets = self._get_field(StandardField.TOTAL_ASSETS, 0)
        intangibles = self._get_field(StandardField.INTANGIBLES, 0) or 0
        total_liab = self._get_field(StandardField.TOTAL_LIABILITIES, 0) or 0
        
        if total_assets:
            tangible_book = total_assets - intangibles - total_liab
            if self._market_cap and tangible_book > 0:
                ptbv = self._market_cap / tangible_book
                multiples.append(ValuationMultiple(
                    name="Price-to-Tangible Book",
                    abbreviation="P/TBV",
                    multiple_type=MultipleType.ASSET,
                    current_value=ptbv,
                    historical_average=None,
                    historical_median=None,
                    percentile_rank=None,
                    sector_average=None,
                    quality=MultipleQuality.MODERATE,
                    interpretation=self._interpret_ptbv(ptbv, pb if 'pb' in dir() else None),
                    formula="Market Cap / Tangible Book Value",
                    numerator="Market Capitalization",
                    denominator="Total Assets - Intangibles - Liabilities"
                ))
        
        return multiples
    
    def _calculate_revenue_multiples(self) -> List[Optional[ValuationMultiple]]:
        """Calculate revenue-based valuation multiples."""
        multiples = []
        
        revenue = self._get_field(StandardField.REVENUE, 0)
        
        # P/S Ratio
        if self._market_cap and revenue and revenue > 0:
            ps = self._market_cap / revenue
            
            historical_ps = self._calculate_historical_multiple(
                lambda mc, rev: mc / rev if rev and rev > 0 else None,
                StandardField.REVENUE
            )
            
            multiples.append(ValuationMultiple(
                name="Price-to-Sales",
                abbreviation="P/S",
                multiple_type=MultipleType.REVENUE,
                current_value=ps,
                historical_average=np.mean(historical_ps) if historical_ps else None,
                historical_median=np.median(historical_ps) if historical_ps else None,
                percentile_rank=self._calculate_percentile(ps, historical_ps),
                sector_average=getattr(VALUATION_PARAMS, "sector_ps_average", 3.0),
                quality=MultipleQuality.MODERATE,
                interpretation=self._interpret_ps(ps),
                formula="Market Cap / Revenue",
                numerator="Market Capitalization",
                denominator="Revenue (TTM)"
            ))
        
        # EV/Revenue
        if self._enterprise_value and revenue and revenue > 0:
            ev_rev = self._enterprise_value / revenue
            
            multiples.append(ValuationMultiple(
                name="EV/Revenue",
                abbreviation="EV/Rev",
                multiple_type=MultipleType.REVENUE,
                current_value=ev_rev,
                historical_average=None,
                historical_median=None,
                percentile_rank=None,
                sector_average=None,
                quality=MultipleQuality.MODERATE,
                interpretation=f"EV/Revenue of {ev_rev:.2f}x - useful for high-growth or unprofitable companies",
                formula="Enterprise Value / Revenue",
                numerator="Enterprise Value",
                denominator="Revenue (TTM)"
            ))
        
        return multiples
    
    def _calculate_cashflow_multiples(self) -> List[Optional[ValuationMultiple]]:
        """Calculate cash flow-based valuation multiples."""
        multiples = []
        
        # EV/EBITDA
        ebitda = self._get_field(StandardField.EBITDA, 0)
        if self._enterprise_value and ebitda and ebitda > 0:
            ev_ebitda = self._enterprise_value / ebitda
            
            historical_ev_ebitda = self._calculate_historical_ev_multiple(
                StandardField.EBITDA
            )
            
            multiples.append(ValuationMultiple(
                name="EV/EBITDA",
                abbreviation="EV/EBITDA",
                multiple_type=MultipleType.CASH_FLOW,
                current_value=ev_ebitda,
                historical_average=np.mean(historical_ev_ebitda) if historical_ev_ebitda else None,
                historical_median=np.median(historical_ev_ebitda) if historical_ev_ebitda else None,
                percentile_rank=self._calculate_percentile(ev_ebitda, historical_ev_ebitda),
                sector_average=getattr(VALUATION_PARAMS, "sector_ev_ebitda_average", 12.0),
                quality=MultipleQuality.HIGH,
                interpretation=self._interpret_ev_ebitda(ev_ebitda),
                formula="Enterprise Value / EBITDA",
                numerator="Enterprise Value",
                denominator="EBITDA"
            ))
        
        # EV/EBIT
        ebit = self._get_field(StandardField.OPERATING_INCOME, 0)
        if self._enterprise_value and ebit and ebit > 0:
            ev_ebit = self._enterprise_value / ebit
            
            multiples.append(ValuationMultiple(
                name="EV/EBIT",
                abbreviation="EV/EBIT",
                multiple_type=MultipleType.CASH_FLOW,
                current_value=ev_ebit,
                historical_average=None,
                historical_median=None,
                percentile_rank=None,
                sector_average=None,
                quality=MultipleQuality.HIGH,
                interpretation=f"EV/EBIT of {ev_ebit:.1f}x accounts for depreciation policy differences",
                formula="Enterprise Value / Operating Income",
                numerator="Enterprise Value",
                denominator="Operating Income (EBIT)"
            ))
        
        # P/FCF
        fcf = self._get_field(StandardField.FREE_CASH_FLOW, 0)
        if self._market_cap and fcf and fcf > 0:
            p_fcf = self._market_cap / fcf
            
            multiples.append(ValuationMultiple(
                name="Price-to-Free Cash Flow",
                abbreviation="P/FCF",
                multiple_type=MultipleType.CASH_FLOW,
                current_value=p_fcf,
                historical_average=None,
                historical_median=None,
                percentile_rank=None,
                sector_average=None,
                quality=MultipleQuality.HIGH,
                interpretation=self._interpret_p_fcf(p_fcf),
                formula="Market Cap / Free Cash Flow",
                numerator="Market Capitalization",
                denominator="Free Cash Flow"
            ))
        
        # FCF Yield
        if self._market_cap and fcf and fcf > 0:
            fcf_yield = fcf / self._market_cap
            multiples.append(ValuationMultiple(
                name="FCF Yield",
                abbreviation="FCF Yield",
                multiple_type=MultipleType.CASH_FLOW,
                current_value=fcf_yield,
                historical_average=None,
                historical_median=None,
                percentile_rank=None,
                sector_average=None,
                quality=MultipleQuality.HIGH,
                interpretation=f"FCF yield of {fcf_yield*100:.1f}% represents cash return potential",
                formula="Free Cash Flow / Market Cap",
                numerator="Free Cash Flow",
                denominator="Market Capitalization"
            ))
        
        return multiples
    
    # =========================================================================
    # HISTORICAL CALCULATIONS
    # =========================================================================
    
    def _calculate_historical_multiple(
        self,
        calc_func,
        denominator_field: StandardField
    ) -> List[float]:
        """
        Calculate historical values of a multiple.
        
        Note: This is a simplified approach since we don't have historical
        market caps. We use current market cap with historical fundamentals
        as a proxy for how the multiple would compare.
        
        Args:
            calc_func: Function to calculate multiple
            denominator_field: Field for denominator
            
        Returns:
            List of historical multiple values
        """
        if not self._market_cap:
            return []
        
        values = []
        periods = self._data.statements.periods
        
        for i in range(len(periods)):
            denom = self._get_field(denominator_field, i)
            if denom:
                val = calc_func(self._market_cap, denom)
                if val is not None:
                    values.append(val)
        
        return values
    
    def _calculate_historical_ev_multiple(
        self,
        denominator_field: StandardField
    ) -> List[float]:
        """Calculate historical EV-based multiple values."""
        if not self._enterprise_value:
            return []
        
        values = []
        periods = self._data.statements.periods
        
        for i in range(len(periods)):
            denom = self._get_field(denominator_field, i)
            if denom and denom > 0:
                val = self._enterprise_value / denom
                values.append(val)
        
        return values
    
    def _calculate_percentile(
        self, 
        current: float, 
        historical: List[float]
    ) -> Optional[float]:
        """Calculate percentile rank of current vs historical."""
        if not historical or len(historical) < 2:
            return None
        
        # Count how many historical values current exceeds
        count_below = sum(1 for h in historical if current > h)
        percentile = (count_below / len(historical)) * 100
        
        return percentile
    
    # =========================================================================
    # IMPLIED GROWTH ANALYSIS
    # =========================================================================
    
    def _analyze_implied_growth(
        self,
        key_multiples: Dict[str, Optional[float]]
    ) -> Optional[ImpliedGrowthAnalysis]:
        """
        Analyze growth implied by current valuation.
        
        Using Gordon Growth Model rearranged:
        P/E = 1 / (r - g)
        g = r - (1 / P/E)
        
        Where r = cost of equity
        
        Args:
            key_multiples: Dictionary of key multiples
            
        Returns:
            ImpliedGrowthAnalysis or None
        """
        pe = key_multiples.get('P/E')
        if pe is None or pe <= 0:
            return None
        
        # Estimate cost of equity (simplified CAPM)
        # Using risk-free rate + equity risk premium
        risk_free = getattr(VALUATION_PARAMS, "risk_free_rate", 0.04)
        equity_premium = getattr(VALUATION_PARAMS, "equity_risk_premium", 0.05)
        beta = self._data.company_info.beta if self._data.company_info.beta is not None else 1.0
        
        cost_of_equity = risk_free + (beta * equity_premium)
        
        # Calculate implied growth
        # g = r - E/P = r - 1/PE
        earnings_yield = 1 / pe
        implied_growth = cost_of_equity - earnings_yield
        
        # Calculate historical growth
        historical_growth = self._calculate_historical_earnings_growth()
        
        # Calculate gap
        growth_gap = implied_growth - historical_growth
        
        # Generate interpretation
        if growth_gap > 0.05:
            interpretation = (
                f"Market implies {implied_growth*100:.1f}% growth, but historical growth "
                f"is {historical_growth*100:.1f}%. Valuation requires growth acceleration "
                f"of {growth_gap*100:.1f}pp to be justified."
            )
        elif growth_gap < -0.05:
            interpretation = (
                f"Market implies {implied_growth*100:.1f}% growth, below historical "
                f"{historical_growth*100:.1f}%. Valuation may underestimate growth potential."
            )
        else:
            interpretation = (
                f"Market implied growth of {implied_growth*100:.1f}% is aligned with "
                f"historical growth of {historical_growth*100:.1f}%."
            )
        
        return ImpliedGrowthAnalysis(
            current_pe=pe,
            cost_of_equity=cost_of_equity,
            implied_growth_rate=implied_growth,
            historical_growth_rate=historical_growth,
            growth_gap=growth_gap,
            interpretation=interpretation
        )
    
    def _calculate_historical_earnings_growth(self) -> float:
        """Calculate historical earnings CAGR."""
        periods = self._data.statements.periods
        earnings = []
        
        for i in range(len(periods)):
            ni = self._get_field(StandardField.NET_INCOME, i)
            if ni and ni > 0:
                earnings.append(ni)
        
        if len(earnings) < 2:
            return 0.05  # Default assumption
        
        # Calculate CAGR
        n = len(earnings) - 1
        if earnings[-1] > 0 and earnings[0] > 0:
            cagr = (earnings[0] / earnings[-1]) ** (1 / n) - 1
            return max(-0.20, min(0.30, cagr))  # Cap between -20% and 30%
        
        return 0.05
    
    # =========================================================================
    # RELATIVE VALUATION SUMMARY
    # =========================================================================
    
    def _generate_relative_summary(
        self,
        multiples: List[ValuationMultiple],
        implied_growth: Optional[ImpliedGrowthAnalysis]
    ) -> RelativeValuationSummary:
        """
        Generate relative valuation summary.
        
        Args:
            multiples: List of valuation multiples
            implied_growth: Implied growth analysis
            
        Returns:
            RelativeValuationSummary object
        """
        # Find primary multiple (prefer EV/EBITDA, then P/E)
        primary_multiple = None
        primary_value = None
        
        for m in multiples:
            if m.abbreviation == 'EV/EBITDA' and m.current_value:
                primary_multiple = m.abbreviation
                primary_value = m.current_value
                break
        
        if not primary_multiple:
            for m in multiples:
                if m.abbreviation == 'P/E' and m.current_value:
                    primary_multiple = m.abbreviation
                    primary_value = m.current_value
                    break
        
        if not primary_multiple:
            primary_multiple = "N/A"
            primary_value = 0
        
        # Assess vs history
        vs_history = self._assess_vs_history(multiples)
        
        # Determine overall assessment based on multiples
        overall_assessment = self._preliminary_assessment(multiples)
        
        # Confidence level
        high_quality_count = sum(1 for m in multiples if m.quality == MultipleQuality.HIGH)
        if high_quality_count >= 3:
            confidence = "High"
        elif high_quality_count >= 1:
            confidence = "Moderate"
        else:
            confidence = "Low"
        
        # Key considerations
        considerations = self._generate_considerations(multiples, implied_growth)
        
        return RelativeValuationSummary(
            primary_multiple=primary_multiple,
            primary_value=primary_value,
            vs_history=vs_history,
            overall_assessment=overall_assessment,
            confidence_level=confidence,
            key_considerations=considerations
        )
    
    def _assess_vs_history(self, multiples: List[ValuationMultiple]) -> str:
        """Assess current multiples vs historical averages."""
        above_count = 0
        below_count = 0
        
        for m in multiples:
            if m.current_value and m.historical_average:
                if m.current_value > m.historical_average * 1.1:
                    above_count += 1
                elif m.current_value < m.historical_average * 0.9:
                    below_count += 1
        
        if above_count > below_count:
            return "Trading above historical average"
        elif below_count > above_count:
            return "Trading below historical average"
        else:
            return "Trading near historical average"
    
    def _preliminary_assessment(
        self, 
        multiples: List[ValuationMultiple]
    ) -> ValuationAssessment:
        """Generate preliminary assessment from multiples."""
        scores = []
        
        for m in multiples:
            if m.current_value and m.historical_average and m.quality == MultipleQuality.HIGH:
                ratio = m.current_value / m.historical_average
                scores.append(ratio)
        
        if not scores:
            return ValuationAssessment.FAIRLY_VALUED
        
        avg_ratio = np.mean(scores)
        
        if avg_ratio < 0.5:
            return ValuationAssessment.SIGNIFICANTLY_UNDERVALUED
        elif avg_ratio < 0.8:
            return ValuationAssessment.UNDERVALUED
        elif avg_ratio < 1.2:
            return ValuationAssessment.FAIRLY_VALUED
        elif avg_ratio < 1.5:
            return ValuationAssessment.OVERVALUED
        else:
            return ValuationAssessment.SIGNIFICANTLY_OVERVALUED
    
    def _generate_considerations(
        self,
        multiples: List[ValuationMultiple],
        implied_growth: Optional[ImpliedGrowthAnalysis]
    ) -> List[str]:
        """
        Generate key considerations for valuation.
        
        This method has been enhanced to provide more nuanced commentary that
        considers when high multiples might be justified by business quality.
        High-quality businesses with strong cash conversion and returns
        deserve to trade at premium multiples.
        
        Args:
            multiples: List of calculated valuation multiples
            implied_growth: Implied growth analysis if available
            
        Returns:
            List of consideration strings for the valuation memo
        """
        considerations = []
        
        # Get key multiples for analysis
        pe_multiple = next((m for m in multiples if m.abbreviation == 'P/E'), None)
        ev_ebitda = next((m for m in multiples if m.abbreviation == 'EV/EBITDA'), None)
        fcf_yield = next((m for m in multiples if m.abbreviation == 'FCF Yield'), None)
        p_fcf = next((m for m in multiples if m.abbreviation == 'P/FCF'), None)
        
        # Check P/E with nuanced commentary
        if pe_multiple and pe_multiple.current_value:
            pe = pe_multiple.current_value
            if pe > 35:
                considerations.append(
                    f"P/E of {pe:.1f}x is elevated. This may be justified for companies "
                    f"with exceptional cash conversion (>100%) and ROE (>20%). Without "
                    f"these quality characteristics, the multiple appears expensive."
                )
            elif pe > 25:
                considerations.append(
                    f"P/E of {pe:.1f}x reflects growth expectations. Appropriate for "
                    f"high-quality businesses with strong cash conversion; otherwise "
                    f"suggests some overvaluation."
                )
            elif pe < 12:
                considerations.append(
                    f"P/E of {pe:.1f}x is below market average. May indicate value "
                    f"opportunity or reflect concerns about business quality."
                )
        
        # Check implied growth gap with context
        if implied_growth and abs(implied_growth.growth_gap) > 0.05:
            gap = implied_growth.growth_gap
            implied = implied_growth.implied_growth_rate
            historical = implied_growth.historical_growth_rate
            
            if gap > 0:
                considerations.append(
                    f"Market expects {implied*100:.1f}% growth vs historical {historical*100:.1f}%. "
                    f"This premium is reasonable for quality businesses with expanding "
                    f"addressable markets or improving margins."
                )
            else:
                considerations.append(
                    f"Market expects only {implied*100:.1f}% growth vs historical {historical*100:.1f}%. "
                    f"This may present opportunity if business fundamentals remain strong."
                )
        
        # Check EV/EBITDA with quality context
        if ev_ebitda and ev_ebitda.current_value:
            ev_eb = ev_ebitda.current_value
            if ev_eb > 15:
                considerations.append(
                    f"EV/EBITDA of {ev_eb:.1f}x above typical 8-12x range. Premium may "
                    f"be warranted for asset-light, high-ROIC businesses."
                )
            elif ev_eb < 8:
                considerations.append(
                    f"EV/EBITDA of {ev_eb:.1f}x below typical range suggests potential "
                    f"value opportunity or business quality concerns."
                )
        
        # Check FCF yield - strong indicator of quality
        if fcf_yield and fcf_yield.current_value:
            fcf_y = fcf_yield.current_value
            if fcf_y > 0.08:
                considerations.append(
                    f"Strong FCF yield of {fcf_y*100:.1f}% indicates the company generates "
                    f"substantial cash relative to market cap - a quality indicator."
                )
            elif fcf_y > 0.05:
                considerations.append(
                    f"FCF yield of {fcf_y*100:.1f}% provides adequate cash generation "
                    f"to support current valuation."
                )
            elif fcf_y > 0:
                considerations.append(
                    f"FCF yield of {fcf_y*100:.1f}% is low, requiring strong growth "
                    f"to justify current multiple."
                )
        
        # Check P/FCF - complements FCF yield
        if p_fcf and p_fcf.current_value and p_fcf.current_value > 0:
            pfcf = p_fcf.current_value
            if pfcf > 40:
                considerations.append(
                    f"P/FCF of {pfcf:.1f}x is elevated. Verify if FCF is temporarily "
                    f"depressed due to investment cycle or if multiple is truly expensive."
                )
            elif pfcf < 15:
                considerations.append(
                    f"P/FCF of {pfcf:.1f}x is attractive for a quality business."
                )
        
        # Add quality-adjusted valuation reminder
        considerations.append(
            "Note: Valuation multiples should be assessed relative to business quality. "
            "Companies with 100%+ cash conversion and 20%+ ROE justify 25-35x P/E; "
            "average businesses warrant 12-18x."
        )
        
        return considerations
    
    # =========================================================================
    # OVERALL ASSESSMENT
    # =========================================================================
    
    def _determine_overall_assessment(
        self,
        multiples: List[ValuationMultiple],
        implied_growth: Optional[ImpliedGrowthAnalysis],
        relative_summary: RelativeValuationSummary
    ) -> ValuationAssessment:
        """
        Determine overall valuation assessment.
        
        Combines multiple signals to arrive at assessment.
        
        Args:
            multiples: List of valuation multiples
            implied_growth: Implied growth analysis
            relative_summary: Relative valuation summary
            
        Returns:
            ValuationAssessment enum
        """
        # Start with preliminary assessment
        assessment = relative_summary.overall_assessment
        
        # Adjust based on implied growth
        if implied_growth:
            if implied_growth.growth_gap > 0.10:
                # Market expects much more growth than historical
                if assessment == ValuationAssessment.FAIRLY_VALUED:
                    assessment = ValuationAssessment.OVERVALUED
                elif assessment == ValuationAssessment.OVERVALUED:
                    assessment = ValuationAssessment.SIGNIFICANTLY_OVERVALUED
            elif implied_growth.growth_gap < -0.10:
                # Market expects less growth than historical
                if assessment == ValuationAssessment.FAIRLY_VALUED:
                    assessment = ValuationAssessment.UNDERVALUED
                elif assessment == ValuationAssessment.UNDERVALUED:
                    assessment = ValuationAssessment.SIGNIFICANTLY_UNDERVALUED
        
        return assessment
    
    # =========================================================================
    # INTERPRETATION METHODS
    # =========================================================================
    
    def _interpret_pe(self, pe: float) -> str:
        """Interpret P/E ratio."""
        if pe < 10:
            return f"Low P/E of {pe:.1f}x may indicate value opportunity or earnings concerns"
        elif pe < 15:
            return f"P/E of {pe:.1f}x suggests moderate valuation relative to earnings"
        elif pe < 25:
            return f"P/E of {pe:.1f}x reflects growth expectations above market average"
        elif pe < 40:
            return f"High P/E of {pe:.1f}x implies significant growth expectations"
        else:
            return f"Very high P/E of {pe:.1f}x requires exceptional growth to justify"
    
    def _interpret_pb(self, pb: float) -> str:
        """Interpret P/B ratio."""
        if pb < 1.0:
            return f"P/B below 1.0 ({pb:.2f}x) - trading below book value, potential value"
        elif pb < 2.0:
            return f"P/B of {pb:.2f}x suggests reasonable premium to book value"
        elif pb < 4.0:
            return f"P/B of {pb:.2f}x reflects expectation of strong returns on equity"
        else:
            return f"High P/B of {pb:.2f}x assumes exceptional ROE or intangible value"
    
    def _interpret_ptbv(self, ptbv: float, pb: Optional[float]) -> str:
        """Interpret P/TBV ratio."""
        base = f"P/TBV of {ptbv:.2f}x "
        if pb and ptbv > pb * 1.5:
            return base + "significantly higher than P/B, indicating large intangible value"
        elif ptbv < 1.0:
            return base + "below tangible book value - potential deep value"
        else:
            return base + "reflects premium for tangible asset base"
    
    def _interpret_ps(self, ps: float) -> str:
        """Interpret P/S ratio."""
        if ps < 1.0:
            return f"P/S below 1.0 ({ps:.2f}x) - low relative to sales, check profitability"
        elif ps < 3.0:
            return f"P/S of {ps:.2f}x is moderate - appropriate for stable businesses"
        elif ps < 8.0:
            return f"P/S of {ps:.2f}x suggests high growth or margin expectations"
        else:
            return f"Very high P/S of {ps:.2f}x requires exceptional growth to justify"
    
    def _interpret_ev_ebitda(self, ev_ebitda: float) -> str:
        """Interpret EV/EBITDA ratio."""
        if ev_ebitda < 6:
            return f"Low EV/EBITDA of {ev_ebitda:.1f}x may indicate value or concerns"
        elif ev_ebitda < 10:
            return f"EV/EBITDA of {ev_ebitda:.1f}x suggests reasonable valuation"
        elif ev_ebitda < 15:
            return f"EV/EBITDA of {ev_ebitda:.1f}x reflects growth premium"
        else:
            return f"High EV/EBITDA of {ev_ebitda:.1f}x implies significant growth expectations"
    
    def _interpret_p_fcf(self, p_fcf: float) -> str:
        """Interpret P/FCF ratio."""
        if p_fcf < 15:
            return f"P/FCF of {p_fcf:.1f}x indicates strong free cash flow relative to price"
        elif p_fcf < 25:
            return f"P/FCF of {p_fcf:.1f}x is reasonable for quality businesses"
        elif p_fcf < 40:
            return f"P/FCF of {p_fcf:.1f}x requires growth in free cash flow to justify"
        else:
            return f"High P/FCF of {p_fcf:.1f}x - verify FCF sustainability and growth potential"
    
    # =========================================================================
    # INSIGHTS
    # =========================================================================
    
    def _generate_insights(
        self,
        multiples: List[ValuationMultiple],
        ev_breakdown: EnterpriseValueBreakdown,
        implied_growth: Optional[ImpliedGrowthAnalysis],
        assessment: ValuationAssessment
    ) -> List[str]:
        """Generate key valuation insights."""
        insights = []
        
        # Overall assessment insight
        if assessment == ValuationAssessment.SIGNIFICANTLY_UNDERVALUED:
            insights.append(
                "Valuation appears SIGNIFICANTLY UNDERVALUED relative to fundamentals and history"
            )
        elif assessment == ValuationAssessment.UNDERVALUED:
            insights.append(
                "Valuation appears UNDERVALUED - potential opportunity if fundamentals support"
            )
        elif assessment == ValuationAssessment.OVERVALUED:
            insights.append(
                "Valuation appears OVERVALUED relative to historical norms"
            )
        elif assessment == ValuationAssessment.SIGNIFICANTLY_OVERVALUED:
            insights.append(
                "Valuation appears SIGNIFICANTLY OVERVALUED - requires exceptional execution"
            )
        
        # EV composition insight
        if ev_breakdown.net_debt < 0:
            insights.append(
                f"Net cash position of ${abs(ev_breakdown.net_debt):.0f}M provides financial flexibility"
            )
        elif ev_breakdown.ev_to_market_cap > 1.3:
            insights.append(
                f"EV is {ev_breakdown.ev_to_market_cap:.1f}x market cap due to significant debt"
            )
        
        # Multiple-specific insights
        for m in multiples:
            if m.abbreviation == 'P/E' and m.current_value and m.percentile_rank:
                if m.percentile_rank > 80:
                    insights.append(
                        f"P/E at {m.percentile_rank:.0f}th percentile of historical range - elevated"
                    )
                elif m.percentile_rank < 20:
                    insights.append(
                        f"P/E at {m.percentile_rank:.0f}th percentile of historical range - depressed"
                    )
        
        # Implied growth insight
        if implied_growth:
            insights.append(implied_growth.interpretation)
        
        # FCF yield insight
        fcf_yield = next((m for m in multiples if m.abbreviation == 'FCF Yield'), None)
        if fcf_yield and fcf_yield.current_value:
            risk_free = getattr(VALUATION_PARAMS, "risk_free_rate", 0.04)
            if fcf_yield.current_value > risk_free + 0.04:
                insights.append(
                    f"FCF yield of {fcf_yield.current_value*100:.1f}% exceeds risk-free rate "
                    f"by {(fcf_yield.current_value - risk_free)*100:.1f}pp - attractive for income"
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
        """Extract a field value from processed data."""
        return self._processor.get_field(self._data, field, period_index)
    
    def _add_warning(self, message: str) -> None:
        """Add a warning message."""
        self._warnings.append(message)
        logger.warning(message)
    
    def _build_key_multiples_dict(
        self, 
        multiples: List[ValuationMultiple]
    ) -> Dict[str, Optional[float]]:
        """Build dictionary of key multiples."""
        key_dict = {}
        for m in multiples:
            key_dict[m.abbreviation] = m.current_value
        return key_dict
    
    # =========================================================================
    # OUTPUT FORMATTING
    # =========================================================================
    
    def format_multiples_table(
        self, 
        result: ValuationAnalysisResult
    ) -> pd.DataFrame:
        """Format multiples as a DataFrame."""
        data = []
        for m in result.multiples:
            curr = f"{m.current_value:.2f}" if m.current_value else "N/A"
            hist = f"{m.historical_average:.2f}" if m.historical_average else "N/A"
            
            data.append({
                'Multiple': f"{m.name} ({m.abbreviation})",
                'Current': curr,
                'Hist. Avg': hist,
                'Quality': m.quality.value,
                'Type': m.multiple_type.value
            })
        
        return pd.DataFrame(data)
    
    def format_ev_breakdown(
        self, 
        ev: EnterpriseValueBreakdown
    ) -> pd.DataFrame:
        """Format EV breakdown as a DataFrame."""
        data = [
            {'Component': 'Market Cap', 'Value ($M)': f"{ev.market_cap:,.0f}"},
            {'Component': '+ Total Debt', 'Value ($M)': f"{ev.total_debt:,.0f}"},
            {'Component': '- Cash', 'Value ($M)': f"({ev.cash:,.0f})"},
            {'Component': '= Enterprise Value', 'Value ($M)': f"{ev.enterprise_value:,.0f}"},
            {'Component': 'Net Debt', 'Value ($M)': f"{ev.net_debt:,.0f}"},
            {'Component': 'EV / Market Cap', 'Value ($M)': f"{ev.ev_to_market_cap:.2f}x"}
        ]
        return pd.DataFrame(data)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_valuation(processed_data: ProcessedData) -> ValuationAnalysisResult:
    """
    Convenience function to perform valuation analysis.
    
    Args:
        processed_data: ProcessedData from DataProcessor
        
    Returns:
        ValuationAnalysisResult
        
    Example:
        from data_collector import collect_financial_data
        from data_processor import process_financial_data
        
        raw = collect_financial_data("AAPL")
        processed = process_financial_data(raw)
        result = analyze_valuation(processed)
        
        print(f"P/E: {result.key_multiples.get('P/E', 'N/A')}")
        print(f"Assessment: {result.valuation_assessment.value}")
    """
    analyzer = ValuationAnalyzer(processed_data)
    return analyzer.analyze()


def get_valuation_summary(result: ValuationAnalysisResult) -> Dict[str, Any]:
    """
    Extract key valuation metrics as a simple dictionary.
    
    Args:
        result: ValuationAnalysisResult
        
    Returns:
        Dictionary with key metrics
    """
    return {
        'assessment': result.valuation_assessment.value,
        'primary_multiple': result.relative_summary.primary_multiple,
        'primary_value': result.relative_summary.primary_value,
        'pe': result.key_multiples.get('P/E'),
        'pb': result.key_multiples.get('P/B'),
        'ev_ebitda': result.key_multiples.get('EV/EBITDA'),
        'fcf_yield': result.key_multiples.get('FCF Yield'),
        'enterprise_value': result.ev_breakdown.enterprise_value,
        'market_cap': result.ev_breakdown.market_cap
    }


# =============================================================================
# MODULE TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Module test script.
    
    Run this file directly to test valuation analysis:
        python valuation.py [TICKER]
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
    print(f"VALUATION ANALYZER TEST - {test_ticker}")
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
        
        # Step 3: Analyze valuation
        print("Step 3: Analyzing valuation...")
        analyzer = ValuationAnalyzer(processed)
        result = analyzer.analyze()
        print(f"  Analysis complete\n")
        
        # Print Enterprise Value Breakdown
        print("ENTERPRISE VALUE BREAKDOWN")
        print("-" * 70)
        ev = result.ev_breakdown
        print(f"  Market Cap:          ${ev.market_cap:,.0f}M")
        print(f"  + Total Debt:        ${ev.total_debt:,.0f}M")
        print(f"  - Cash:              ${ev.cash:,.0f}M")
        print(f"  = Enterprise Value:  ${ev.enterprise_value:,.0f}M")
        print(f"  Net Debt:            ${ev.net_debt:,.0f}M")
        print(f"  EV / Market Cap:     {ev.ev_to_market_cap:.2f}x")
        
        # Print Valuation Multiples
        print(f"\nVALUATION MULTIPLES")
        print("-" * 70)
        print(f"  {'Multiple':<25} {'Current':>12} {'Hist Avg':>12} {'Quality':<10}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
        for m in result.multiples:
            curr = f"{m.current_value:.2f}" if m.current_value else "N/A"
            hist = f"{m.historical_average:.2f}" if m.historical_average else "N/A"
            print(f"  {m.name:<25} {curr:>12} {hist:>12} {m.quality.value:<10}")
        
        # Print Implied Growth Analysis
        if result.implied_growth:
            print(f"\nIMPLIED GROWTH ANALYSIS")
            print("-" * 70)
            ig = result.implied_growth
            print(f"  Current P/E:           {ig.current_pe:.1f}x")
            print(f"  Cost of Equity:        {ig.cost_of_equity*100:.1f}%")
            print(f"  Implied Growth:        {ig.implied_growth_rate*100:.1f}%")
            print(f"  Historical Growth:     {ig.historical_growth_rate*100:.1f}%")
            print(f"  Growth Gap:            {ig.growth_gap*100:+.1f}pp")
            print(f"  {ig.interpretation}")
        
        # Print Relative Valuation Summary
        print(f"\nRELATIVE VALUATION SUMMARY")
        print("-" * 70)
        rs = result.relative_summary
        print(f"  Primary Multiple:      {rs.primary_multiple} = {rs.primary_value:.2f}x")
        print(f"  vs History:            {rs.vs_history}")
        print(f"  Overall Assessment:    {rs.overall_assessment.value}")
        print(f"  Confidence Level:      {rs.confidence_level}")
        if rs.key_considerations:
            print(f"  Key Considerations:")
            for c in rs.key_considerations:
                print(f"    - {c}")
        
        # Print Overall Assessment
        print(f"\nOVERALL VALUATION ASSESSMENT")
        print("-" * 70)
        print(f"  {result.valuation_assessment.value.upper()}")
        
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
        print(f"Valuation analysis complete for {test_ticker}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)