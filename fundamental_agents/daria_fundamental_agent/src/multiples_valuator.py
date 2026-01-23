"""
Multiples Valuation Module - Phase 7 Relative Valuation
Fundamental Analyst Agent

Implements institutional-grade relative valuation using price and enterprise
value multiples with historical comparison and implied fair value calculation.

Methodology:
    - Calculate current valuation multiples (P/E, P/B, EV/EBITDA, EV/Revenue, P/S, P/FCF)
    - Compute historical multiples using trailing financials
    - Compare current vs historical averages to assess valuation
    - Derive implied fair values from each multiple
    - Generate composite valuation signal

Key Components:
    - Multiple Calculator: Computes current and historical multiples
    - Historical Analyzer: Tracks multiple trends and averages
    - Valuation Assessor: Determines over/under/fair valuation
    - Implied Value Calculator: Derives fair value from each multiple
    - Composite Scorer: Aggregates signals into overall assessment

Inputs: 
    - CollectionResult from Phase 1
    - DCFValuationResult from Phase 5 (for comparison)
    - DDMValuationResult from Phase 6 (for comparison)

Outputs: MultiplesValuationResult with implied values and assessment

MSc Coursework: IFTE0001 AI Agents in Asset Management
Track A: Fundamental Analyst Agent
Phase 7: Relative Valuation (Multiples Analysis)

[REQUIRED BY COURSEWORK: "Basic intrinsic valuation (DCF or multiples)"]

Version: 1.0.0
"""

from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from .config import (
    LOGGER,
    OUTPUT_DIR,
)


__version__ = "1.0.0"


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

class MultiplesConfig:
    """Configuration parameters for multiples valuation."""
    
    # Valuation thresholds (per coursework plan)
    OVERVALUED_THRESHOLD: float = 0.20      # > +20% vs historical = overvalued
    UNDERVALUED_THRESHOLD: float = -0.20    # < -20% vs historical = undervalued
    
    # Multiple validity ranges (to filter outliers)
    PE_MIN: float = 0.0
    PE_MAX: float = 200.0
    PB_MIN: float = 0.0
    PB_MAX: float = 100.0
    EV_EBITDA_MIN: float = 0.0
    EV_EBITDA_MAX: float = 100.0
    EV_REVENUE_MIN: float = 0.0
    EV_REVENUE_MAX: float = 50.0
    PS_MIN: float = 0.0
    PS_MAX: float = 50.0
    PFCF_MIN: float = 0.0
    PFCF_MAX: float = 200.0
    
    # Weighting for composite score
    MULTIPLE_WEIGHTS: Dict[str, float] = {
        "pe_ratio": 0.25,
        "ev_ebitda": 0.25,
        "pb_ratio": 0.15,
        "ev_revenue": 0.15,
        "ps_ratio": 0.10,
        "pfcf_ratio": 0.10,
    }
    
    # Minimum years for historical analysis
    MIN_HISTORICAL_YEARS: int = 3


# =============================================================================
# ENUMERATIONS
# =============================================================================

class MultipleType(Enum):
    """Types of valuation multiples."""
    PE_RATIO = "pe_ratio"                   # Price / Earnings
    PB_RATIO = "pb_ratio"                   # Price / Book Value
    PS_RATIO = "ps_ratio"                   # Price / Sales (Revenue)
    PFCF_RATIO = "pfcf_ratio"               # Price / Free Cash Flow
    EV_EBITDA = "ev_ebitda"                 # Enterprise Value / EBITDA
    EV_REVENUE = "ev_revenue"               # Enterprise Value / Revenue
    EV_FCF = "ev_fcf"                       # Enterprise Value / FCF


class ValuationAssessment(Enum):
    """Valuation assessment based on multiple comparison."""
    SIGNIFICANTLY_OVERVALUED = "significantly_overvalued"       # > +50%
    OVERVALUED = "overvalued"                                   # +20% to +50%
    FAIRLY_VALUED = "fairly_valued"                             # -20% to +20%
    UNDERVALUED = "undervalued"                                 # -50% to -20%
    SIGNIFICANTLY_UNDERVALUED = "significantly_undervalued"     # < -50%


class TrendDirection(Enum):
    """Trend direction for multiples over time."""
    EXPANDING = "expanding"         # Multiple increasing
    STABLE = "stable"               # Multiple flat
    CONTRACTING = "contracting"     # Multiple decreasing


# =============================================================================
# DATA CONTAINERS - INDIVIDUAL MULTIPLE
# =============================================================================

@dataclass
class MultipleValue:
    """Single multiple value with metadata."""
    
    multiple_type: MultipleType
    value: Optional[float] = None
    is_valid: bool = False
    fiscal_year: str = ""
    
    # Components used in calculation
    numerator: Optional[float] = None
    denominator: Optional[float] = None
    numerator_label: str = ""
    denominator_label: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "multiple_type": self.multiple_type.value,
            "value": self.value,
            "is_valid": self.is_valid,
            "fiscal_year": self.fiscal_year,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "numerator_label": self.numerator_label,
            "denominator_label": self.denominator_label,
        }


@dataclass
class MultipleAnalysis:
    """Complete analysis for a single multiple type."""
    
    multiple_type: MultipleType
    
    # Current value
    current_value: Optional[float] = None
    current_valid: bool = False
    
    # Historical values by year
    historical_values: Dict[str, float] = field(default_factory=dict)
    
    # Historical statistics
    historical_average: Optional[float] = None
    historical_median: Optional[float] = None
    historical_min: Optional[float] = None
    historical_max: Optional[float] = None
    historical_std: Optional[float] = None
    years_of_data: int = 0
    
    # Comparison to historical
    premium_to_average: Optional[float] = None      # (Current - Avg) / Avg
    premium_to_median: Optional[float] = None       # (Current - Median) / Median
    
    # Trend
    trend: TrendDirection = TrendDirection.STABLE
    cagr: Optional[float] = None
    
    # Assessment
    assessment: ValuationAssessment = ValuationAssessment.FAIRLY_VALUED
    
    # Implied fair value
    implied_value_from_average: Optional[float] = None
    implied_value_from_median: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "multiple_type": self.multiple_type.value,
            "current_value": self.current_value,
            "current_valid": self.current_valid,
            "historical_values": self.historical_values,
            "historical_average": self.historical_average,
            "historical_median": self.historical_median,
            "historical_min": self.historical_min,
            "historical_max": self.historical_max,
            "historical_std": self.historical_std,
            "years_of_data": self.years_of_data,
            "premium_to_average": self.premium_to_average,
            "premium_to_median": self.premium_to_median,
            "trend": self.trend.value,
            "cagr": self.cagr,
            "assessment": self.assessment.value,
            "implied_value_from_average": self.implied_value_from_average,
            "implied_value_from_median": self.implied_value_from_median,
        }


# =============================================================================
# DATA CONTAINERS - COMPOSITE ANALYSIS
# =============================================================================

@dataclass
class CompositeValuation:
    """Composite valuation from all multiples."""
    
    # Individual multiple scores (-1 to +1, negative = undervalued)
    multiple_scores: Dict[str, float] = field(default_factory=dict)
    
    # Weighted composite score
    composite_score: float = 0.0
    
    # Assessment
    overall_assessment: ValuationAssessment = ValuationAssessment.FAIRLY_VALUED
    
    # Implied fair values
    implied_values: Dict[str, float] = field(default_factory=dict)
    average_implied_value: Optional[float] = None
    median_implied_value: Optional[float] = None
    
    # Upside/downside from composite
    composite_upside: Optional[float] = None
    
    # Confidence
    multiples_used: int = 0
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "multiple_scores": self.multiple_scores,
            "composite_score": self.composite_score,
            "overall_assessment": self.overall_assessment.value,
            "implied_values": self.implied_values,
            "average_implied_value": self.average_implied_value,
            "median_implied_value": self.median_implied_value,
            "composite_upside": self.composite_upside,
            "multiples_used": self.multiples_used,
            "confidence": self.confidence,
        }


# =============================================================================
# DATA CONTAINERS - CROSS-MODEL COMPARISON
# =============================================================================

@dataclass
class CrossModelComparison:
    """Comparison of intrinsic values across all models."""
    
    # Values from each model
    dcf_value: Optional[float] = None
    ddm_value: Optional[float] = None
    multiples_average: Optional[float] = None
    multiples_median: Optional[float] = None
    
    # Current market price
    current_price: Optional[float] = None
    
    # Upside/downside from each
    dcf_upside: Optional[float] = None
    ddm_upside: Optional[float] = None
    multiples_upside: Optional[float] = None
    
    # Consensus
    average_intrinsic_value: Optional[float] = None
    consensus_upside: Optional[float] = None
    ddm_included_in_consensus: bool = True
    ddm_exclusion_reason: str = ""
    
    # Model agreement
    models_bullish: int = 0
    models_bearish: int = 0
    models_neutral: int = 0
    
    consensus_direction: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dcf_value": self.dcf_value,
            "ddm_value": self.ddm_value,
            "multiples_average": self.multiples_average,
            "multiples_median": self.multiples_median,
            "current_price": self.current_price,
            "dcf_upside": self.dcf_upside,
            "ddm_upside": self.ddm_upside,
            "multiples_upside": self.multiples_upside,
            "average_intrinsic_value": self.average_intrinsic_value,
            "consensus_upside": self.consensus_upside,
            "ddm_included_in_consensus": self.ddm_included_in_consensus,
            "ddm_exclusion_reason": self.ddm_exclusion_reason,
            "models_bullish": self.models_bullish,
            "models_bearish": self.models_bearish,
            "models_neutral": self.models_neutral,
            "consensus_direction": self.consensus_direction,
        }


# =============================================================================
# DATA CONTAINERS - FINAL RESULT
# =============================================================================

@dataclass
class MultiplesValuationResult:
    """Complete Phase 7 Multiples Valuation output."""
    
    # Identification
    ticker: str = ""
    company_name: str = ""
    valuation_date: datetime = field(default_factory=datetime.now)
    
    # Current market data
    current_price: Optional[float] = None
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    shares_outstanding: Optional[float] = None
    
    # Individual multiple analyses
    pe_analysis: MultipleAnalysis = field(
        default_factory=lambda: MultipleAnalysis(MultipleType.PE_RATIO)
    )
    pb_analysis: MultipleAnalysis = field(
        default_factory=lambda: MultipleAnalysis(MultipleType.PB_RATIO)
    )
    ps_analysis: MultipleAnalysis = field(
        default_factory=lambda: MultipleAnalysis(MultipleType.PS_RATIO)
    )
    pfcf_analysis: MultipleAnalysis = field(
        default_factory=lambda: MultipleAnalysis(MultipleType.PFCF_RATIO)
    )
    ev_ebitda_analysis: MultipleAnalysis = field(
        default_factory=lambda: MultipleAnalysis(MultipleType.EV_EBITDA)
    )
    ev_revenue_analysis: MultipleAnalysis = field(
        default_factory=lambda: MultipleAnalysis(MultipleType.EV_REVENUE)
    )
    
    # Composite valuation
    composite_valuation: CompositeValuation = field(default_factory=CompositeValuation)
    
    # Cross-model comparison
    cross_model_comparison: CrossModelComparison = field(default_factory=CrossModelComparison)
    
    # Overall outputs
    implied_fair_value: Optional[float] = None
    upside_downside_pct: Optional[float] = None
    overall_assessment: ValuationAssessment = ValuationAssessment.FAIRLY_VALUED
    
    # Validation
    is_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    
    # Metadata
    multiples_calculated: int = 0
    assumptions: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "valuation_date": self.valuation_date.isoformat(),
            "current_price": self.current_price,
            "market_cap": self.market_cap,
            "enterprise_value": self.enterprise_value,
            "shares_outstanding": self.shares_outstanding,
            "pe_analysis": self.pe_analysis.to_dict(),
            "pb_analysis": self.pb_analysis.to_dict(),
            "ps_analysis": self.ps_analysis.to_dict(),
            "pfcf_analysis": self.pfcf_analysis.to_dict(),
            "ev_ebitda_analysis": self.ev_ebitda_analysis.to_dict(),
            "ev_revenue_analysis": self.ev_revenue_analysis.to_dict(),
            "composite_valuation": self.composite_valuation.to_dict(),
            "cross_model_comparison": self.cross_model_comparison.to_dict(),
            "implied_fair_value": self.implied_fair_value,
            "upside_downside_pct": self.upside_downside_pct,
            "overall_assessment": self.overall_assessment.value,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "multiples_calculated": self.multiples_calculated,
            "assumptions": self.assumptions,
        }


# =============================================================================
# MULTIPLE CALCULATOR
# =============================================================================

class MultipleCalculator:
    """
    Calculates valuation multiples from financial data.
    
    Computes both current and historical multiples for comparison.
    """
    
    def calculate_current_multiples(
        self,
        profile: Any,
        derived_metrics: Any,
        statements: Any,
        market_cap: float,
        enterprise_value: float,
        shares: float,
    ) -> Dict[str, MultipleValue]:
        """Calculate current valuation multiples."""
        multiples = {}
        
        # Get latest fiscal year
        periods = statements.fiscal_periods
        latest_year = periods[0] if periods else ""
        
        # P/E Ratio
        pe = MultipleValue(
            multiple_type=MultipleType.PE_RATIO,
            fiscal_year=latest_year,
            numerator_label="Market Cap",
            denominator_label="Net Income",
        )
        net_income = self._get_value(statements.income_statement, "net_income", latest_year)
        if market_cap and net_income and net_income > 0:
            pe.value = market_cap / net_income
            pe.numerator = market_cap
            pe.denominator = net_income
            pe.is_valid = MultiplesConfig.PE_MIN < pe.value < MultiplesConfig.PE_MAX
        multiples["pe_ratio"] = pe
        
        # P/B Ratio
        pb = MultipleValue(
            multiple_type=MultipleType.PB_RATIO,
            fiscal_year=latest_year,
            numerator_label="Market Cap",
            denominator_label="Total Equity",
        )
        total_equity = self._get_value(statements.balance_sheet, "total_equity", latest_year)
        if market_cap and total_equity and total_equity > 0:
            pb.value = market_cap / total_equity
            pb.numerator = market_cap
            pb.denominator = total_equity
            pb.is_valid = MultiplesConfig.PB_MIN < pb.value < MultiplesConfig.PB_MAX
        multiples["pb_ratio"] = pb
        
        # P/S Ratio
        ps = MultipleValue(
            multiple_type=MultipleType.PS_RATIO,
            fiscal_year=latest_year,
            numerator_label="Market Cap",
            denominator_label="Revenue",
        )
        revenue = self._get_value(statements.income_statement, "total_revenue", latest_year)
        if market_cap and revenue and revenue > 0:
            ps.value = market_cap / revenue
            ps.numerator = market_cap
            ps.denominator = revenue
            ps.is_valid = MultiplesConfig.PS_MIN < ps.value < MultiplesConfig.PS_MAX
        multiples["ps_ratio"] = ps
        
        # P/FCF Ratio
        pfcf = MultipleValue(
            multiple_type=MultipleType.PFCF_RATIO,
            fiscal_year=latest_year,
            numerator_label="Market Cap",
            denominator_label="Free Cash Flow",
        )
        fcf = derived_metrics.fcf_calculated.get(latest_year)
        if market_cap and fcf and fcf > 0:
            pfcf.value = market_cap / fcf
            pfcf.numerator = market_cap
            pfcf.denominator = fcf
            pfcf.is_valid = MultiplesConfig.PFCF_MIN < pfcf.value < MultiplesConfig.PFCF_MAX
        multiples["pfcf_ratio"] = pfcf
        
        # EV/EBITDA
        ev_ebitda = MultipleValue(
            multiple_type=MultipleType.EV_EBITDA,
            fiscal_year=latest_year,
            numerator_label="Enterprise Value",
            denominator_label="EBITDA",
        )
        ebitda = derived_metrics.ebitda_calculated.get(latest_year)
        if not ebitda:
            ebitda = self._get_value(statements.income_statement, "ebitda", latest_year)
        if enterprise_value and ebitda and ebitda > 0:
            ev_ebitda.value = enterprise_value / ebitda
            ev_ebitda.numerator = enterprise_value
            ev_ebitda.denominator = ebitda
            ev_ebitda.is_valid = MultiplesConfig.EV_EBITDA_MIN < ev_ebitda.value < MultiplesConfig.EV_EBITDA_MAX
        multiples["ev_ebitda"] = ev_ebitda
        
        # EV/Revenue
        ev_rev = MultipleValue(
            multiple_type=MultipleType.EV_REVENUE,
            fiscal_year=latest_year,
            numerator_label="Enterprise Value",
            denominator_label="Revenue",
        )
        if enterprise_value and revenue and revenue > 0:
            ev_rev.value = enterprise_value / revenue
            ev_rev.numerator = enterprise_value
            ev_rev.denominator = revenue
            ev_rev.is_valid = MultiplesConfig.EV_REVENUE_MIN < ev_rev.value < MultiplesConfig.EV_REVENUE_MAX
        multiples["ev_revenue"] = ev_rev
        
        return multiples
    
    def calculate_historical_multiples(
        self,
        statements: Any,
        derived_metrics: Any,
        market_cap: float,
        enterprise_value: float,
        fiscal_periods: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate implied historical multiples.
        
        Uses current market cap/EV with historical fundamentals to show
        what current valuation implies relative to historical performance.
        """
        historical = {
            "pe_ratio": {},
            "pb_ratio": {},
            "ps_ratio": {},
            "pfcf_ratio": {},
            "ev_ebitda": {},
            "ev_revenue": {},
        }
        
        for year in fiscal_periods:
            # P/E using current market cap, historical earnings
            net_income = self._get_value(statements.income_statement, "net_income", year)
            if market_cap and net_income and net_income > 0:
                pe = market_cap / net_income
                if MultiplesConfig.PE_MIN < pe < MultiplesConfig.PE_MAX:
                    historical["pe_ratio"][year] = pe
            
            # P/B using current market cap, historical book value
            total_equity = self._get_value(statements.balance_sheet, "total_equity", year)
            if market_cap and total_equity and total_equity > 0:
                pb = market_cap / total_equity
                if MultiplesConfig.PB_MIN < pb < MultiplesConfig.PB_MAX:
                    historical["pb_ratio"][year] = pb
            
            # P/S using current market cap, historical revenue
            revenue = self._get_value(statements.income_statement, "total_revenue", year)
            if market_cap and revenue and revenue > 0:
                ps = market_cap / revenue
                if MultiplesConfig.PS_MIN < ps < MultiplesConfig.PS_MAX:
                    historical["ps_ratio"][year] = ps
            
            # P/FCF using current market cap, historical FCF
            fcf = derived_metrics.fcf_calculated.get(year)
            if market_cap and fcf and fcf > 0:
                pfcf = market_cap / fcf
                if MultiplesConfig.PFCF_MIN < pfcf < MultiplesConfig.PFCF_MAX:
                    historical["pfcf_ratio"][year] = pfcf
            
            # EV/EBITDA using current EV, historical EBITDA
            ebitda = derived_metrics.ebitda_calculated.get(year)
            if not ebitda:
                ebitda = self._get_value(statements.income_statement, "ebitda", year)
            if enterprise_value and ebitda and ebitda > 0:
                ev_ebitda = enterprise_value / ebitda
                if MultiplesConfig.EV_EBITDA_MIN < ev_ebitda < MultiplesConfig.EV_EBITDA_MAX:
                    historical["ev_ebitda"][year] = ev_ebitda
            
            # EV/Revenue using current EV, historical revenue
            if enterprise_value and revenue and revenue > 0:
                ev_rev = enterprise_value / revenue
                if MultiplesConfig.EV_REVENUE_MIN < ev_rev < MultiplesConfig.EV_REVENUE_MAX:
                    historical["ev_revenue"][year] = ev_rev
        
        return historical
    
    def _get_value(self, df: pd.DataFrame, field: str, year: str) -> Optional[float]:
        """Safely extract value from DataFrame."""
        if df is None or df.empty:
            return None
        if field not in df.index or year not in df.columns:
            return None
        value = df.loc[field, year]
        return float(value) if pd.notna(value) else None


# =============================================================================
# HISTORICAL ANALYZER
# =============================================================================

class HistoricalMultipleAnalyzer:
    """
    Analyzes historical multiple trends and statistics.
    """
    
    def analyze(
        self,
        multiple_type: MultipleType,
        current_value: Optional[float],
        historical_values: Dict[str, float],
        current_price: float,
        metric_per_share: Optional[float],
    ) -> MultipleAnalysis:
        """Analyze a single multiple with historical comparison."""
        analysis = MultipleAnalysis(multiple_type=multiple_type)
        
        analysis.current_value = current_value
        analysis.current_valid = current_value is not None and current_value > 0
        analysis.historical_values = historical_values
        analysis.years_of_data = len(historical_values)
        
        if analysis.years_of_data < 2:
            return analysis
        
        # Historical statistics
        values = list(historical_values.values())
        analysis.historical_average = np.mean(values)
        analysis.historical_median = np.median(values)
        analysis.historical_min = min(values)
        analysis.historical_max = max(values)
        analysis.historical_std = np.std(values)
        
        # Premium/discount to historical
        if analysis.current_valid and analysis.historical_average > 0:
            analysis.premium_to_average = (
                (current_value - analysis.historical_average) / analysis.historical_average
            )
        
        if analysis.current_valid and analysis.historical_median > 0:
            analysis.premium_to_median = (
                (current_value - analysis.historical_median) / analysis.historical_median
            )
        
        # Trend analysis
        sorted_years = sorted(historical_values.keys())
        if len(sorted_years) >= 2:
            first_val = historical_values[sorted_years[0]]
            last_val = historical_values[sorted_years[-1]]
            
            if first_val > 0:
                n_years = len(sorted_years) - 1
                analysis.cagr = (last_val / first_val) ** (1 / n_years) - 1
                
                if analysis.cagr > 0.05:
                    analysis.trend = TrendDirection.EXPANDING
                elif analysis.cagr < -0.05:
                    analysis.trend = TrendDirection.CONTRACTING
                else:
                    analysis.trend = TrendDirection.STABLE
        
        # Assessment based on premium to average
        if analysis.premium_to_average is not None:
            prem = analysis.premium_to_average
            if prem > 0.50:
                analysis.assessment = ValuationAssessment.SIGNIFICANTLY_OVERVALUED
            elif prem > MultiplesConfig.OVERVALUED_THRESHOLD:
                analysis.assessment = ValuationAssessment.OVERVALUED
            elif prem < -0.50:
                analysis.assessment = ValuationAssessment.SIGNIFICANTLY_UNDERVALUED
            elif prem < MultiplesConfig.UNDERVALUED_THRESHOLD:
                analysis.assessment = ValuationAssessment.UNDERVALUED
            else:
                analysis.assessment = ValuationAssessment.FAIRLY_VALUED
        
        # Implied fair values (using historical average multiple)
        if metric_per_share and metric_per_share > 0:
            if analysis.historical_average and analysis.historical_average > 0:
                analysis.implied_value_from_average = metric_per_share * analysis.historical_average
            if analysis.historical_median and analysis.historical_median > 0:
                analysis.implied_value_from_median = metric_per_share * analysis.historical_median
        
        return analysis


# =============================================================================
# COMPOSITE VALUATOR
# =============================================================================

class CompositeValuator:
    """
    Aggregates individual multiple analyses into composite valuation.
    """
    
    def compute(
        self,
        analyses: Dict[str, MultipleAnalysis],
        current_price: float,
    ) -> CompositeValuation:
        """Compute composite valuation from individual analyses."""
        composite = CompositeValuation()
        
        weights = MultiplesConfig.MULTIPLE_WEIGHTS
        weighted_sum = 0.0
        total_weight = 0.0
        
        for multiple_name, analysis in analyses.items():
            if analysis.premium_to_average is not None:
                # Convert premium to score (-1 to +1)
                # Positive premium = overvalued = positive score
                score = max(-1.0, min(1.0, analysis.premium_to_average))
                composite.multiple_scores[multiple_name] = score
                
                weight = weights.get(multiple_name, 0.1)
                weighted_sum += score * weight
                total_weight += weight
                
                composite.multiples_used += 1
                
                # Collect implied values
                if analysis.implied_value_from_average:
                    composite.implied_values[multiple_name] = analysis.implied_value_from_average
        
        # Composite score
        if total_weight > 0:
            composite.composite_score = weighted_sum / total_weight
        
        # Overall assessment
        score = composite.composite_score
        if score > 0.50:
            composite.overall_assessment = ValuationAssessment.SIGNIFICANTLY_OVERVALUED
        elif score > 0.20:
            composite.overall_assessment = ValuationAssessment.OVERVALUED
        elif score < -0.50:
            composite.overall_assessment = ValuationAssessment.SIGNIFICANTLY_UNDERVALUED
        elif score < -0.20:
            composite.overall_assessment = ValuationAssessment.UNDERVALUED
        else:
            composite.overall_assessment = ValuationAssessment.FAIRLY_VALUED
        
        # Implied value statistics
        if composite.implied_values:
            values = list(composite.implied_values.values())
            composite.average_implied_value = np.mean(values)
            composite.median_implied_value = np.median(values)
            
            if current_price and current_price > 0:
                composite.composite_upside = (
                    (composite.average_implied_value - current_price) / current_price
                )
        
        # Confidence based on number of multiples
        composite.confidence = min(composite.multiples_used / 6, 1.0)
        
        return composite


# =============================================================================
# CROSS-MODEL COMPARATOR
# =============================================================================

class CrossModelComparator:
    """
    Compares intrinsic values across DCF, DDM, and Multiples models.
    """
    
    def compare(
        self,
        multiples_result: CompositeValuation,
        current_price: float,
        dcf_result: Optional[Any] = None,
        ddm_result: Optional[Any] = None,
    ) -> CrossModelComparison:
        """Compare values across all models."""
        comparison = CrossModelComparison()
        comparison.current_price = current_price
        
        # DCF
        if dcf_result and hasattr(dcf_result, 'intrinsic_value_per_share'):
            comparison.dcf_value = dcf_result.intrinsic_value_per_share
            if current_price > 0:
                comparison.dcf_upside = (comparison.dcf_value - current_price) / current_price
        
        # DDM
        if ddm_result and hasattr(ddm_result, 'intrinsic_value_per_share'):
            if ddm_result.is_applicable:
                comparison.ddm_value = ddm_result.intrinsic_value_per_share
                if current_price > 0:
                    comparison.ddm_upside = (comparison.ddm_value - current_price) / current_price
        
        # Multiples
        comparison.multiples_average = multiples_result.average_implied_value
        comparison.multiples_median = multiples_result.median_implied_value
        comparison.multiples_upside = multiples_result.composite_upside
        
        # Consensus calculation
        # CRITICAL: Exclude DDM from consensus for low-dividend companies
        # DDM is inappropriate when payout ratio < 30% (buyback-focused companies)
        values = []
        include_ddm_in_consensus = True
        ddm_exclusion_reason = ""
        
        # Check if DDM should be excluded from consensus
        if ddm_result and hasattr(ddm_result, 'applicability'):
            app = ddm_result.applicability
            payout_ratio = getattr(app, 'average_payout_ratio', None)
            if payout_ratio is not None and payout_ratio < 0.30:
                # Low payout ratio - company prioritizes buybacks over dividends
                include_ddm_in_consensus = False
                ddm_exclusion_reason = f"Low payout ratio ({payout_ratio:.1%} < 30%) - buyback-focused company"
                LOGGER.info(f"Excluding DDM from consensus: payout ratio {payout_ratio:.1%} < 30%")
        
        # Also exclude DDM if it diverges >80% from DCF (clearly inappropriate model)
        if comparison.dcf_value and comparison.ddm_value and include_ddm_in_consensus:
            divergence = abs(comparison.dcf_value - comparison.ddm_value) / comparison.dcf_value
            if divergence > 0.80:
                include_ddm_in_consensus = False
                ddm_exclusion_reason = f"Extreme model divergence ({divergence:.0%} from DCF) - DDM inappropriate"
                LOGGER.info(f"Excluding DDM from consensus: {divergence:.0%} divergence from DCF")
        
        comparison.ddm_included_in_consensus = include_ddm_in_consensus
        comparison.ddm_exclusion_reason = ddm_exclusion_reason
        
        if comparison.dcf_value:
            values.append(comparison.dcf_value)
        if comparison.ddm_value and include_ddm_in_consensus:
            values.append(comparison.ddm_value)
        if comparison.multiples_average:
            values.append(comparison.multiples_average)
        
        if values:
            comparison.average_intrinsic_value = np.mean(values)
            if current_price > 0:
                comparison.consensus_upside = (
                    (comparison.average_intrinsic_value - current_price) / current_price
                )
        
        # Model agreement (respect DDM exclusion)
        threshold_bull = 0.10   # > +10% upside = bullish
        threshold_bear = -0.10  # < -10% upside = bearish
        
        model_upsides = [
            ("dcf", comparison.dcf_upside),
            ("ddm", comparison.ddm_upside if include_ddm_in_consensus else None),
            ("multiples", comparison.multiples_upside),
        ]
        
        for model_name, upside in model_upsides:
            if upside is not None:
                if upside > threshold_bull:
                    comparison.models_bullish += 1
                elif upside < threshold_bear:
                    comparison.models_bearish += 1
                else:
                    comparison.models_neutral += 1
        
        total_models = comparison.models_bullish + comparison.models_bearish + comparison.models_neutral
        if total_models > 0:
            if comparison.models_bearish > comparison.models_bullish:
                comparison.consensus_direction = "bearish"
            elif comparison.models_bullish > comparison.models_bearish:
                comparison.consensus_direction = "bullish"
            else:
                comparison.consensus_direction = "mixed"
        
        return comparison


# =============================================================================
# MAIN VALUATOR CLASS
# =============================================================================

class Phase7Valuator:
    """
    Main orchestrator for Phase 7 Multiples Valuation.
    
    Coordinates all components for relative valuation analysis.
    
    Usage:
        valuator = Phase7Valuator()
        result = valuator.value(collection_result, dcf_result, ddm_result)
    """
    
    def __init__(self):
        self.calculator = MultipleCalculator()
        self.historical_analyzer = HistoricalMultipleAnalyzer()
        self.composite_valuator = CompositeValuator()
        self.cross_model_comparator = CrossModelComparator()
        self.logger = LOGGER
    
    def value(
        self,
        collection_result: Any,
        dcf_result: Optional[Any] = None,
        ddm_result: Optional[Any] = None,
    ) -> MultiplesValuationResult:
        """
        Perform complete multiples valuation.
        
        Args:
            collection_result: CollectionResult from Phase 1
            dcf_result: Optional DCFValuationResult from Phase 5
            ddm_result: Optional DDMValuationResult from Phase 6
            
        Returns:
            MultiplesValuationResult with implied values and assessment
        """
        self.logger.info(f"Phase 7: Starting multiples valuation for {collection_result.ticker}")
        
        result = MultiplesValuationResult(
            ticker=collection_result.ticker,
            company_name=collection_result.company_name,
        )
        
        # Extract data
        profile = collection_result.company_profile
        statements = collection_result.statements
        derived_metrics = collection_result.derived_metrics
        periods = statements.fiscal_periods
        
        # Get market data
        market_cap = getattr(profile, 'market_cap', None)
        shares = getattr(profile, 'shares_outstanding', None)
        
        if not market_cap or market_cap <= 0:
            result.validation_errors.append("Market cap not available")
            return result
        
        if not shares or shares <= 0:
            shares = self._get_value(statements.balance_sheet, "shares_outstanding", periods[0])
        
        if not shares or shares <= 0:
            result.validation_errors.append("Shares outstanding not available")
            return result
        
        result.market_cap = market_cap
        result.shares_outstanding = shares
        result.current_price = market_cap / shares
        
        # Get or calculate enterprise value
        ev = derived_metrics.enterprise_value
        if not ev or ev <= 0:
            net_debt = derived_metrics.net_debt.get(periods[0], 0) or 0
            ev = market_cap + net_debt
        result.enterprise_value = ev
        
        # Step 1: Calculate current multiples
        self.logger.info("  Calculating current multiples")
        current_multiples = self.calculator.calculate_current_multiples(
            profile, derived_metrics, statements, market_cap, ev, shares
        )
        
        # Step 2: Calculate historical multiples
        self.logger.info("  Calculating historical multiples")
        historical_multiples = self.calculator.calculate_historical_multiples(
            statements, derived_metrics, market_cap, ev, periods
        )
        
        # Step 3: Analyze each multiple
        self.logger.info("  Analyzing multiple trends and valuations")
        
        # Get per-share metrics for implied value calculation
        latest_year = periods[0]
        eps = self._get_eps(statements.income_statement, latest_year, shares)
        bvps = self._get_bvps(statements.balance_sheet, latest_year, shares)
        revenue_ps = self._get_rps(statements.income_statement, latest_year, shares)
        fcf_ps = self._get_fcfps(derived_metrics, latest_year, shares)
        
        # P/E Analysis
        result.pe_analysis = self.historical_analyzer.analyze(
            MultipleType.PE_RATIO,
            current_multiples["pe_ratio"].value,
            historical_multiples["pe_ratio"],
            result.current_price,
            eps
        )
        
        # P/B Analysis
        result.pb_analysis = self.historical_analyzer.analyze(
            MultipleType.PB_RATIO,
            current_multiples["pb_ratio"].value,
            historical_multiples["pb_ratio"],
            result.current_price,
            bvps
        )
        
        # P/S Analysis
        result.ps_analysis = self.historical_analyzer.analyze(
            MultipleType.PS_RATIO,
            current_multiples["ps_ratio"].value,
            historical_multiples["ps_ratio"],
            result.current_price,
            revenue_ps
        )
        
        # P/FCF Analysis
        result.pfcf_analysis = self.historical_analyzer.analyze(
            MultipleType.PFCF_RATIO,
            current_multiples["pfcf_ratio"].value,
            historical_multiples["pfcf_ratio"],
            result.current_price,
            fcf_ps
        )
        
        # EV/EBITDA Analysis (implied value requires conversion to equity)
        ebitda_ps = self._get_ebitda_ps(derived_metrics, statements, latest_year, shares)
        result.ev_ebitda_analysis = self.historical_analyzer.analyze(
            MultipleType.EV_EBITDA,
            current_multiples["ev_ebitda"].value,
            historical_multiples["ev_ebitda"],
            result.current_price,
            ebitda_ps
        )
        # Adjust implied value for EV-based multiple
        self._adjust_ev_implied_value(result.ev_ebitda_analysis, market_cap, ev, shares)
        
        # EV/Revenue Analysis
        result.ev_revenue_analysis = self.historical_analyzer.analyze(
            MultipleType.EV_REVENUE,
            current_multiples["ev_revenue"].value,
            historical_multiples["ev_revenue"],
            result.current_price,
            revenue_ps
        )
        self._adjust_ev_implied_value(result.ev_revenue_analysis, market_cap, ev, shares)
        
        # Step 4: Compute composite valuation
        self.logger.info("  Computing composite valuation")
        analyses = {
            "pe_ratio": result.pe_analysis,
            "pb_ratio": result.pb_analysis,
            "ps_ratio": result.ps_analysis,
            "pfcf_ratio": result.pfcf_analysis,
            "ev_ebitda": result.ev_ebitda_analysis,
            "ev_revenue": result.ev_revenue_analysis,
        }
        
        result.composite_valuation = self.composite_valuator.compute(
            analyses, result.current_price
        )
        
        # Step 5: Cross-model comparison
        self.logger.info("  Comparing across valuation models")
        result.cross_model_comparison = self.cross_model_comparator.compare(
            result.composite_valuation,
            result.current_price,
            dcf_result,
            ddm_result,
        )
        
        # Step 6: Overall outputs
        result.implied_fair_value = result.composite_valuation.average_implied_value
        result.upside_downside_pct = result.composite_valuation.composite_upside
        result.overall_assessment = result.composite_valuation.overall_assessment
        result.multiples_calculated = result.composite_valuation.multiples_used
        
        # Step 7: Record assumptions
        result.assumptions = {
            "overvalued_threshold": MultiplesConfig.OVERVALUED_THRESHOLD,
            "undervalued_threshold": MultiplesConfig.UNDERVALUED_THRESHOLD,
            "multiple_weights": MultiplesConfig.MULTIPLE_WEIGHTS,
        }
        
        # Validate
        result.is_valid = self._validate_result(result)
        
        self.logger.info(
            f"Phase 7 complete: Implied Value ${result.implied_fair_value:.2f}, "
            f"Assessment: {result.overall_assessment.value}"
        )
        
        return result
    
    def _get_value(self, df: pd.DataFrame, field: str, year: str) -> Optional[float]:
        """Safely extract value."""
        if df is None or df.empty:
            return None
        if field not in df.index or year not in df.columns:
            return None
        value = df.loc[field, year]
        return float(value) if pd.notna(value) else None
    
    def _get_eps(self, income: pd.DataFrame, year: str, shares: float) -> Optional[float]:
        """Get earnings per share."""
        net_income = self._get_value(income, "net_income", year)
        if net_income and shares > 0:
            return net_income / shares
        return None
    
    def _get_bvps(self, balance: pd.DataFrame, year: str, shares: float) -> Optional[float]:
        """Get book value per share."""
        equity = self._get_value(balance, "total_equity", year)
        if equity and shares > 0:
            return equity / shares
        return None
    
    def _get_rps(self, income: pd.DataFrame, year: str, shares: float) -> Optional[float]:
        """Get revenue per share."""
        revenue = self._get_value(income, "total_revenue", year)
        if revenue and shares > 0:
            return revenue / shares
        return None
    
    def _get_fcfps(self, derived: Any, year: str, shares: float) -> Optional[float]:
        """Get free cash flow per share."""
        fcf = derived.fcf_calculated.get(year)
        if fcf and shares > 0:
            return fcf / shares
        return None
    
    def _get_ebitda_ps(
        self, derived: Any, statements: Any, year: str, shares: float
    ) -> Optional[float]:
        """Get EBITDA per share."""
        ebitda = derived.ebitda_calculated.get(year)
        if not ebitda:
            ebitda = self._get_value(statements.income_statement, "ebitda", year)
        if ebitda and shares > 0:
            return ebitda / shares
        return None
    
    def _adjust_ev_implied_value(
        self,
        analysis: MultipleAnalysis,
        market_cap: float,
        ev: float,
        shares: float,
    ):
        """Adjust implied value from EV-based multiple to equity value."""
        if ev <= 0 or market_cap <= 0:
            return
        
        # Ratio of market cap to EV
        equity_ratio = market_cap / ev
        
        if analysis.implied_value_from_average:
            # implied_value_from_average is EV-implied, convert to equity
            implied_ev = analysis.implied_value_from_average * shares
            implied_equity = implied_ev * equity_ratio
            analysis.implied_value_from_average = implied_equity / shares
        
        if analysis.implied_value_from_median:
            implied_ev = analysis.implied_value_from_median * shares
            implied_equity = implied_ev * equity_ratio
            analysis.implied_value_from_median = implied_equity / shares
    
    def _validate_result(self, result: MultiplesValuationResult) -> bool:
        """Validate multiples result."""
        if result.validation_errors:
            return False
        
        if result.multiples_calculated < 2:
            result.validation_warnings.append("Less than 2 valid multiples calculated")
        
        if result.implied_fair_value and result.implied_fair_value <= 0:
            result.validation_errors.append("Implied fair value is non-positive")
            return False
        
        return True
    
    def save_report(
        self,
        result: MultiplesValuationResult,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Save multiples valuation report."""
        if output_dir is None:
            output_dir = OUTPUT_DIR
        
        ticker_dir = output_dir / result.ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = ticker_dir / f"{result.ticker}_multiples_valuation.json"
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"Saved multiples valuation to {filepath}")
        return filepath


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def value_company_multiples(
    collection_result: Any,
    dcf_result: Optional[Any] = None,
    ddm_result: Optional[Any] = None,
) -> MultiplesValuationResult:
    """Convenience function for multiples valuation."""
    valuator = Phase7Valuator()
    return valuator.value(collection_result, dcf_result, ddm_result)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "__version__",
    "MultiplesConfig",
    "MultipleType",
    "ValuationAssessment",
    "TrendDirection",
    "MultipleValue",
    "MultipleAnalysis",
    "CompositeValuation",
    "CrossModelComparison",
    "MultiplesValuationResult",
    "MultipleCalculator",
    "HistoricalMultipleAnalyzer",
    "CompositeValuator",
    "CrossModelComparator",
    "Phase7Valuator",
    "value_company_multiples",
]