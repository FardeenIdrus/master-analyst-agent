"""
DDM Valuation Module - Phase 6 Dividend Discount Model
Fundamental Analyst Agent

Implements institutional-grade Dividend Discount Model (DDM) valuation using
explicit 5-year dividend projections and Gordon Growth terminal value methodology.

Methodology:
    Equity Value = Sum of PV(DPS) for Years 1-5 + PV(Terminal Value)
    Intrinsic Value per Share = Equity Value / Shares Outstanding
    
    Note: DDM values equity directly (not enterprise value like DCF)

Key Components:
    - DDM Applicability Assessment: Validates dividend-paying status and stability
    - Historical Dividend Analysis: DPS trends, payout ratios, growth patterns
    - Growth Rate Derivation: Historical CAGR, Sustainable Growth, Earnings-Based
    - Cost of Equity: CAPM-based discount rate
    - 5-Year Dividend Projections: Explicit per-share dividend forecasts
    - Terminal Value: Gordon Growth Model
    - DCF Reconciliation: Compares DDM vs DCF intrinsic values

Inputs: 
    - CollectionResult from Phase 1 (with dividend history and company profile)
    - DCFValuationResult from Phase 5 (for reconciliation)

Outputs: DDMValuationResult with intrinsic value and comprehensive analysis

MSc Coursework: IFTE0001 AI Agents in Asset Management
Track A: Fundamental Analyst Agent
Phase 6: DDM Valuation (5-Year Dividend Projections)

[ENHANCEMENT: Second intrinsic valuation method for dividend-paying companies]

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

class DDMConfig:
    """Configuration parameters for DDM valuation."""
    
    # Risk-free rate (10-Year US Treasury approximation)
    RISK_FREE_RATE: float = 0.043  # 4.3%
    
    # Equity Risk Premium (historical average)
    EQUITY_RISK_PREMIUM: float = 0.055  # 5.5%
    
    # Default beta for missing data
    DEFAULT_BETA: float = 1.0
    
    # Terminal growth rate constraints
    MIN_TERMINAL_GROWTH: float = 0.01  # 1% floor
    MAX_TERMINAL_GROWTH: float = 0.03  # 3% cap (long-term GDP)
    DEFAULT_TERMINAL_GROWTH: float = 0.02  # 2% default (conservative for dividends)
    
    # Projection period growth constraints
    MAX_PROJECTION_GROWTH: float = 0.15  # 15% cap (per coursework plan)
    MIN_PROJECTION_GROWTH: float = 0.00  # 0% floor (dividends rarely cut)
    
    # Projection period
    PROJECTION_YEARS: int = 5
    
    # DDM Applicability thresholds
    MIN_DIVIDEND_YEARS: int = 3          # Minimum years of dividend history
    MIN_PAYOUT_RATIO: float = 0.10       # Minimum 10% payout ratio
    MAX_PAYOUT_RATIO: float = 0.95       # Maximum 95% payout ratio
    MAX_DIVIDEND_CV: float = 0.50        # Maximum coefficient of variation
    
    # Sensitivity analysis ranges
    GROWTH_SENSITIVITY_RANGE: List[float] = [-0.02, -0.01, 0, 0.01, 0.02]
    KE_SENSITIVITY_RANGE: List[float] = [-0.02, -0.01, 0, 0.01, 0.02]
    
    # Scenario multipliers
    BEAR_CASE_MULTIPLIER: float = 0.6
    BULL_CASE_MULTIPLIER: float = 1.4


# =============================================================================
# ENUMERATIONS
# =============================================================================

class DDMApplicability(Enum):
    """DDM applicability status for the company."""
    APPLICABLE = "applicable"                      # Full DDM applicable
    PARTIAL = "partial"                            # Limited applicability (short history)
    NOT_APPLICABLE = "not_applicable"              # No dividends or unsuitable
    
    
class DividendQuality(Enum):
    """Quality assessment of dividend history."""
    HIGH = "high"               # Consistent growth, no cuts, stable payout
    MODERATE = "moderate"       # Some variability but generally stable
    LOW = "low"                 # High variability or recent cuts
    INSUFFICIENT = "insufficient"  # Not enough data


class DDMGrowthMethod(Enum):
    """Methods for deriving dividend growth rate."""
    HISTORICAL_CAGR = "historical_cagr"
    SUSTAINABLE_GROWTH = "sustainable_growth"
    EARNINGS_BASED = "earnings_based"
    PAYOUT_ADJUSTED = "payout_adjusted"


class DDMScenarioType(Enum):
    """Valuation scenario types."""
    BEAR = "bear"
    BASE = "base"
    BULL = "bull"


# =============================================================================
# DATA CONTAINERS - APPLICABILITY CHECK
# =============================================================================

@dataclass
class DDMApplicabilityResult:
    """Result of DDM applicability assessment."""
    
    status: DDMApplicability = DDMApplicability.NOT_APPLICABLE
    
    # Criteria checks
    has_dividend_history: bool = False
    years_of_dividends: int = 0
    has_positive_dividends: bool = False
    has_stable_payout: bool = False
    has_no_cuts: bool = False
    payout_ratio_valid: bool = False
    
    # Detailed metrics
    current_payout_ratio: Optional[float] = None
    average_payout_ratio: Optional[float] = None
    dividend_variability: Optional[float] = None
    
    # Reasons
    applicability_reasons: List[str] = field(default_factory=list)
    exclusion_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "has_dividend_history": self.has_dividend_history,
            "years_of_dividends": self.years_of_dividends,
            "has_positive_dividends": self.has_positive_dividends,
            "has_stable_payout": self.has_stable_payout,
            "has_no_cuts": self.has_no_cuts,
            "payout_ratio_valid": self.payout_ratio_valid,
            "current_payout_ratio": self.current_payout_ratio,
            "average_payout_ratio": self.average_payout_ratio,
            "dividend_variability": self.dividend_variability,
            "applicability_reasons": self.applicability_reasons,
            "exclusion_reasons": self.exclusion_reasons,
        }


# =============================================================================
# DATA CONTAINERS - HISTORICAL ANALYSIS
# =============================================================================

@dataclass
class HistoricalDividendMetrics:
    """Historical dividend analysis metrics."""
    
    # Raw data by year
    total_dividends_by_year: Dict[str, float] = field(default_factory=dict)
    dps_by_year: Dict[str, float] = field(default_factory=dict)
    payout_ratio_by_year: Dict[str, float] = field(default_factory=dict)
    
    # Summary statistics
    years_analyzed: int = 0
    total_dividends_paid: float = 0.0
    average_annual_dividend: float = 0.0
    latest_annual_dividend: float = 0.0
    
    # DPS metrics
    current_dps: float = 0.0
    average_dps: float = 0.0
    dps_std_dev: Optional[float] = None
    dps_coefficient_of_variation: Optional[float] = None
    
    # Growth metrics
    dividend_cagr: Optional[float] = None
    dps_cagr: Optional[float] = None
    yoy_growth_rates: Dict[str, float] = field(default_factory=dict)
    average_growth: Optional[float] = None
    
    # Payout analysis
    current_payout_ratio: Optional[float] = None
    average_payout_ratio: Optional[float] = None
    payout_trend: str = "stable"
    
    # Quality
    dividend_quality: DividendQuality = DividendQuality.INSUFFICIENT
    consecutive_years_paid: int = 0
    consecutive_years_increased: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_dividends_by_year": self.total_dividends_by_year,
            "dps_by_year": self.dps_by_year,
            "payout_ratio_by_year": self.payout_ratio_by_year,
            "years_analyzed": self.years_analyzed,
            "total_dividends_paid": self.total_dividends_paid,
            "average_annual_dividend": self.average_annual_dividend,
            "latest_annual_dividend": self.latest_annual_dividend,
            "current_dps": self.current_dps,
            "average_dps": self.average_dps,
            "dps_std_dev": self.dps_std_dev,
            "dps_coefficient_of_variation": self.dps_coefficient_of_variation,
            "dividend_cagr": self.dividend_cagr,
            "dps_cagr": self.dps_cagr,
            "yoy_growth_rates": self.yoy_growth_rates,
            "average_growth": self.average_growth,
            "current_payout_ratio": self.current_payout_ratio,
            "average_payout_ratio": self.average_payout_ratio,
            "payout_trend": self.payout_trend,
            "dividend_quality": self.dividend_quality.value,
            "consecutive_years_paid": self.consecutive_years_paid,
            "consecutive_years_increased": self.consecutive_years_increased,
        }


# =============================================================================
# DATA CONTAINERS - GROWTH RATE DERIVATION
# =============================================================================

@dataclass
class DDMGrowthEstimate:
    """Single dividend growth rate estimate."""
    
    method: DDMGrowthMethod
    rate: Optional[float] = None
    confidence: float = 0.0
    description: str = ""
    is_available: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "rate": self.rate,
            "confidence": self.confidence,
            "description": self.description,
            "is_available": self.is_available,
        }


@dataclass
class DDMGrowthAnalysis:
    """Complete dividend growth rate analysis."""
    
    # Individual estimates
    historical_cagr: DDMGrowthEstimate = field(
        default_factory=lambda: DDMGrowthEstimate(DDMGrowthMethod.HISTORICAL_CAGR)
    )
    sustainable_growth: DDMGrowthEstimate = field(
        default_factory=lambda: DDMGrowthEstimate(DDMGrowthMethod.SUSTAINABLE_GROWTH)
    )
    earnings_based: DDMGrowthEstimate = field(
        default_factory=lambda: DDMGrowthEstimate(DDMGrowthMethod.EARNINGS_BASED)
    )
    
    # Selected rates
    projection_growth_rate: float = 0.0
    terminal_growth_rate: float = DDMConfig.DEFAULT_TERMINAL_GROWTH
    
    # Selection methodology
    selection_method: str = ""
    selection_rationale: str = ""
    
    # Adjustments
    growth_cap_applied: bool = False
    original_uncapped_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "historical_cagr": self.historical_cagr.to_dict(),
            "sustainable_growth": self.sustainable_growth.to_dict(),
            "earnings_based": self.earnings_based.to_dict(),
            "projection_growth_rate": self.projection_growth_rate,
            "terminal_growth_rate": self.terminal_growth_rate,
            "selection_method": self.selection_method,
            "selection_rationale": self.selection_rationale,
            "growth_cap_applied": self.growth_cap_applied,
            "original_uncapped_rate": self.original_uncapped_rate,
        }


# =============================================================================
# DATA CONTAINERS - COST OF EQUITY
# =============================================================================

@dataclass
class DDMCostOfEquity:
    """Cost of Equity calculation for DDM."""
    
    risk_free_rate: float = DDMConfig.RISK_FREE_RATE
    beta: float = DDMConfig.DEFAULT_BETA
    equity_risk_premium: float = DDMConfig.EQUITY_RISK_PREMIUM
    cost_of_equity: float = 0.0
    beta_source: str = "default"
    
    def calculate(self) -> float:
        """Calculate cost of equity using CAPM."""
        self.cost_of_equity = self.risk_free_rate + (self.beta * self.equity_risk_premium)
        return self.cost_of_equity
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_free_rate": self.risk_free_rate,
            "beta": self.beta,
            "equity_risk_premium": self.equity_risk_premium,
            "cost_of_equity": self.cost_of_equity,
            "beta_source": self.beta_source,
            "formula": "Rf + Beta * ERP",
        }


# =============================================================================
# DATA CONTAINERS - DDM PROJECTION
# =============================================================================

@dataclass
class YearlyDividendProjection:
    """Single year dividend projection."""
    
    year: int
    dps: float
    growth_rate: float
    discount_factor: float
    present_value: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "year": self.year,
            "dps": self.dps,
            "growth_rate": self.growth_rate,
            "discount_factor": self.discount_factor,
            "present_value": self.present_value,
        }


@dataclass
class DDMTerminalValue:
    """Terminal value calculation using Gordon Growth Model."""
    
    final_year_dps: float = 0.0
    terminal_growth_rate: float = DDMConfig.DEFAULT_TERMINAL_GROWTH
    cost_of_equity: float = 0.0
    
    terminal_year_dps: float = 0.0
    terminal_value: float = 0.0
    discount_factor: float = 0.0
    present_value: float = 0.0
    
    is_valid: bool = True
    validation_message: str = ""
    
    def calculate(self) -> float:
        """Calculate terminal value using Gordon Growth Model."""
        if self.cost_of_equity <= self.terminal_growth_rate:
            self.is_valid = False
            self.validation_message = "Cost of equity must exceed terminal growth rate"
            return 0.0
        
        self.terminal_year_dps = self.final_year_dps * (1 + self.terminal_growth_rate)
        self.terminal_value = self.terminal_year_dps / (self.cost_of_equity - self.terminal_growth_rate)
        self.discount_factor = 1 / ((1 + self.cost_of_equity) ** DDMConfig.PROJECTION_YEARS)
        self.present_value = self.terminal_value * self.discount_factor
        
        self.is_valid = True
        return self.present_value
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_year_dps": self.final_year_dps,
            "terminal_growth_rate": self.terminal_growth_rate,
            "cost_of_equity": self.cost_of_equity,
            "terminal_year_dps": self.terminal_year_dps,
            "terminal_value_undiscounted": self.terminal_value,
            "discount_factor": self.discount_factor,
            "present_value": self.present_value,
            "is_valid": self.is_valid,
            "validation_message": self.validation_message,
            "formula": "DPS * (1+g) / (Ke - g)",
        }


@dataclass
class DDMProjection:
    """Complete DDM projection."""
    
    base_dps: float = 0.0
    projection_growth_rate: float = 0.0
    terminal_growth_rate: float = DDMConfig.DEFAULT_TERMINAL_GROWTH
    cost_of_equity: float = 0.0
    
    yearly_projections: List[YearlyDividendProjection] = field(default_factory=list)
    terminal_value: DDMTerminalValue = field(default_factory=DDMTerminalValue)
    
    sum_of_pv_dividends: float = 0.0
    pv_terminal_value: float = 0.0
    intrinsic_value_per_share: float = 0.0
    
    terminal_value_pct: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_dps": self.base_dps,
            "projection_growth_rate": self.projection_growth_rate,
            "terminal_growth_rate": self.terminal_growth_rate,
            "cost_of_equity": self.cost_of_equity,
            "yearly_projections": [p.to_dict() for p in self.yearly_projections],
            "terminal_value": self.terminal_value.to_dict(),
            "sum_of_pv_dividends": self.sum_of_pv_dividends,
            "pv_terminal_value": self.pv_terminal_value,
            "intrinsic_value_per_share": self.intrinsic_value_per_share,
            "terminal_value_pct": self.terminal_value_pct,
        }


# =============================================================================
# DATA CONTAINERS - SENSITIVITY ANALYSIS
# =============================================================================

@dataclass
class DDMSensitivityMatrix:
    """Sensitivity analysis matrix."""
    
    growth_rates: List[float] = field(default_factory=list)
    discount_rates: List[float] = field(default_factory=list)
    values: List[List[float]] = field(default_factory=list)
    base_growth_idx: int = 0
    base_ke_idx: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "growth_rates": self.growth_rates,
            "discount_rates": self.discount_rates,
            "values": self.values,
            "base_growth_idx": self.base_growth_idx,
            "base_ke_idx": self.base_ke_idx,
        }


@dataclass
class DDMScenarioValuation:
    """Single scenario valuation."""
    
    scenario: DDMScenarioType
    growth_rate: float
    terminal_growth_rate: float
    intrinsic_value_per_share: float
    upside_downside_pct: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.value,
            "growth_rate": self.growth_rate,
            "terminal_growth_rate": self.terminal_growth_rate,
            "intrinsic_value_per_share": self.intrinsic_value_per_share,
            "upside_downside_pct": self.upside_downside_pct,
        }


@dataclass
class DDMSensitivityAnalysis:
    """Complete sensitivity analysis."""
    
    sensitivity_matrix: DDMSensitivityMatrix = field(default_factory=DDMSensitivityMatrix)
    bear_case: Optional[DDMScenarioValuation] = None
    base_case: Optional[DDMScenarioValuation] = None
    bull_case: Optional[DDMScenarioValuation] = None
    
    min_value: float = 0.0
    max_value: float = 0.0
    value_range_pct: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensitivity_matrix": self.sensitivity_matrix.to_dict(),
            "bear_case": self.bear_case.to_dict() if self.bear_case else None,
            "base_case": self.base_case.to_dict() if self.base_case else None,
            "bull_case": self.bull_case.to_dict() if self.bull_case else None,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "value_range_pct": self.value_range_pct,
        }


# =============================================================================
# DATA CONTAINERS - DCF RECONCILIATION
# =============================================================================

@dataclass
class DCFDDMReconciliation:
    """Reconciliation between DCF and DDM valuations."""
    
    dcf_intrinsic_value: Optional[float] = None
    ddm_intrinsic_value: Optional[float] = None
    
    value_difference: float = 0.0
    percentage_difference: float = 0.0
    
    dcf_higher: bool = False
    convergence_status: str = ""
    
    explanation: List[str] = field(default_factory=list)
    
    # Methodology comparison
    dcf_growth_rate: Optional[float] = None
    ddm_growth_rate: Optional[float] = None
    dcf_discount_rate: Optional[float] = None
    ddm_discount_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dcf_intrinsic_value": self.dcf_intrinsic_value,
            "ddm_intrinsic_value": self.ddm_intrinsic_value,
            "value_difference": self.value_difference,
            "percentage_difference": self.percentage_difference,
            "dcf_higher": self.dcf_higher,
            "convergence_status": self.convergence_status,
            "explanation": self.explanation,
            "dcf_growth_rate": self.dcf_growth_rate,
            "ddm_growth_rate": self.ddm_growth_rate,
            "dcf_discount_rate": self.dcf_discount_rate,
            "ddm_discount_rate": self.ddm_discount_rate,
        }


# =============================================================================
# DATA CONTAINERS - FINAL RESULT
# =============================================================================

@dataclass
class DDMValuationResult:
    """Complete Phase 6 DDM Valuation output."""
    
    # Identification
    ticker: str = ""
    company_name: str = ""
    valuation_date: datetime = field(default_factory=datetime.now)
    
    # Applicability
    applicability: DDMApplicabilityResult = field(default_factory=DDMApplicabilityResult)
    
    # Historical analysis
    historical_dividends: HistoricalDividendMetrics = field(default_factory=HistoricalDividendMetrics)
    
    # Growth analysis
    growth_analysis: DDMGrowthAnalysis = field(default_factory=DDMGrowthAnalysis)
    
    # Cost of equity
    cost_of_equity: DDMCostOfEquity = field(default_factory=DDMCostOfEquity)
    
    # DDM projection
    ddm_projection: DDMProjection = field(default_factory=DDMProjection)
    
    # Valuation outputs
    intrinsic_value_per_share: float = 0.0
    current_price: Optional[float] = None
    upside_downside_pct: float = 0.0
    
    # Yield analysis
    current_dividend_yield: Optional[float] = None
    implied_dividend_yield: Optional[float] = None
    
    # Sensitivity analysis
    sensitivity_analysis: DDMSensitivityAnalysis = field(default_factory=DDMSensitivityAnalysis)
    
    # DCF reconciliation
    dcf_reconciliation: DCFDDMReconciliation = field(default_factory=DCFDDMReconciliation)
    
    # Validation
    is_valid: bool = False
    is_applicable: bool = False
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    
    # Metadata
    assumptions: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "valuation_date": self.valuation_date.isoformat(),
            "applicability": self.applicability.to_dict(),
            "historical_dividends": self.historical_dividends.to_dict(),
            "growth_analysis": self.growth_analysis.to_dict(),
            "cost_of_equity": self.cost_of_equity.to_dict(),
            "ddm_projection": self.ddm_projection.to_dict(),
            "intrinsic_value_per_share": self.intrinsic_value_per_share,
            "current_price": self.current_price,
            "upside_downside_pct": self.upside_downside_pct,
            "current_dividend_yield": self.current_dividend_yield,
            "implied_dividend_yield": self.implied_dividend_yield,
            "sensitivity_analysis": self.sensitivity_analysis.to_dict(),
            "dcf_reconciliation": self.dcf_reconciliation.to_dict(),
            "is_valid": self.is_valid,
            "is_applicable": self.is_applicable,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "assumptions": self.assumptions,
        }


# =============================================================================
# APPLICABILITY CHECKER
# =============================================================================

class DDMApplicabilityChecker:
    """
    Assesses whether DDM is applicable for a given company.
    
    Criteria:
        - Has dividend history (minimum 3 years)
        - Positive dividends in all years
        - Stable payout (no significant cuts)
        - Reasonable payout ratio (10-95%)
    """
    
    def check(
        self,
        dividend_history: Any,
        income_statement: pd.DataFrame,
        fiscal_periods: List[str],
    ) -> DDMApplicabilityResult:
        """Check DDM applicability."""
        result = DDMApplicabilityResult()
        
        # Check dividend history exists
        annual_dividends = dividend_history.annual_dividends
        if not annual_dividends:
            result.exclusion_reasons.append("No dividend history available")
            return result
        
        result.has_dividend_history = True
        result.years_of_dividends = len(annual_dividends)
        
        # Check minimum years
        if result.years_of_dividends < DDMConfig.MIN_DIVIDEND_YEARS:
            result.exclusion_reasons.append(
                f"Insufficient dividend history: {result.years_of_dividends} years "
                f"(minimum {DDMConfig.MIN_DIVIDEND_YEARS})"
            )
            result.status = DDMApplicability.PARTIAL
        else:
            result.applicability_reasons.append(
                f"{result.years_of_dividends} years of dividend history"
            )
        
        # Check positive dividends
        dividend_values = list(annual_dividends.values())
        positive_count = sum(1 for d in dividend_values if d > 0)
        
        if positive_count == len(dividend_values):
            result.has_positive_dividends = True
            result.applicability_reasons.append("All years have positive dividends")
        else:
            result.exclusion_reasons.append(
                f"Non-positive dividends in {len(dividend_values) - positive_count} years"
            )
        
        # Check for dividend cuts
        result.has_no_cuts = not dividend_history.has_dividend_cuts
        if result.has_no_cuts:
            result.applicability_reasons.append("No dividend cuts in history")
        else:
            result.exclusion_reasons.append("Dividend cuts detected")
        
        # Check payout stability
        result.has_stable_payout = dividend_history.payout_stable
        if result.has_stable_payout:
            result.applicability_reasons.append("Stable dividend payout")
        
        # Calculate and check payout ratios
        payout_ratios = self._calculate_payout_ratios(
            annual_dividends, income_statement, fiscal_periods
        )
        
        if payout_ratios:
            result.current_payout_ratio = list(payout_ratios.values())[0]
            result.average_payout_ratio = np.mean(list(payout_ratios.values()))
            
            # Check payout ratio validity
            if DDMConfig.MIN_PAYOUT_RATIO <= result.average_payout_ratio <= DDMConfig.MAX_PAYOUT_RATIO:
                result.payout_ratio_valid = True
                result.applicability_reasons.append(
                    f"Payout ratio {result.average_payout_ratio:.1%} within valid range"
                )
            else:
                result.exclusion_reasons.append(
                    f"Payout ratio {result.average_payout_ratio:.1%} outside valid range "
                    f"({DDMConfig.MIN_PAYOUT_RATIO:.0%}-{DDMConfig.MAX_PAYOUT_RATIO:.0%})"
                )
        
        # Calculate dividend variability
        if len(dividend_values) > 1:
            result.dividend_variability = np.std(dividend_values) / np.mean(dividend_values)
        
        # Determine final status
        critical_checks = [
            result.has_dividend_history,
            result.has_positive_dividends,
            result.years_of_dividends >= DDMConfig.MIN_DIVIDEND_YEARS,
        ]
        
        if all(critical_checks):
            result.status = DDMApplicability.APPLICABLE
        elif result.has_dividend_history and result.has_positive_dividends:
            result.status = DDMApplicability.PARTIAL
        else:
            result.status = DDMApplicability.NOT_APPLICABLE
        
        return result
    
    def _calculate_payout_ratios(
        self,
        annual_dividends: Dict[str, float],
        income_statement: pd.DataFrame,
        fiscal_periods: List[str],
    ) -> Dict[str, float]:
        """Calculate payout ratios by year."""
        ratios = {}
        
        for year in fiscal_periods:
            dividend = annual_dividends.get(year)
            net_income = self._get_value(income_statement, "net_income", year)
            
            if dividend and net_income and net_income > 0:
                ratios[year] = dividend / net_income
        
        return ratios
    
    def _get_value(self, df: pd.DataFrame, field: str, year: str) -> Optional[float]:
        """Safely extract value from DataFrame."""
        if df is None or df.empty:
            return None
        if field not in df.index or year not in df.columns:
            return None
        value = df.loc[field, year]
        return float(value) if pd.notna(value) else None


# =============================================================================
# HISTORICAL DIVIDEND ANALYZER
# =============================================================================

class HistoricalDividendAnalyzer:
    """
    Analyzes historical dividend data for DDM inputs.
    
    Extracts DPS trends, calculates growth rates, and assesses quality.
    """
    
    def analyze(
        self,
        dividend_history: Any,
        income_statement: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        fiscal_periods: List[str],
        shares_outstanding: float,
    ) -> HistoricalDividendMetrics:
        """Analyze historical dividends."""
        metrics = HistoricalDividendMetrics()
        
        annual_dividends = dividend_history.annual_dividends
        sorted_periods = sorted(fiscal_periods, reverse=False)
        
        # Extract total dividends by year
        dividend_values = []
        for year in sorted_periods:
            div = annual_dividends.get(year)
            if div is not None and div > 0:
                metrics.total_dividends_by_year[year] = div
                dividend_values.append(div)
        
        metrics.years_analyzed = len(dividend_values)
        
        if metrics.years_analyzed == 0:
            metrics.dividend_quality = DividendQuality.INSUFFICIENT
            return metrics
        
        # Summary statistics
        metrics.total_dividends_paid = sum(dividend_values)
        metrics.average_annual_dividend = np.mean(dividend_values)
        metrics.latest_annual_dividend = dividend_values[-1]
        
        # Calculate DPS by year
        for year in sorted_periods:
            div = metrics.total_dividends_by_year.get(year)
            shares = self._get_value(balance_sheet, "shares_outstanding", year) or shares_outstanding
            
            if div and shares and shares > 0:
                metrics.dps_by_year[year] = div / shares
        
        # DPS statistics
        if metrics.dps_by_year:
            dps_values = list(metrics.dps_by_year.values())
            metrics.current_dps = dps_values[-1] if dps_values else 0
            metrics.average_dps = np.mean(dps_values)
            
            if len(dps_values) > 1:
                metrics.dps_std_dev = np.std(dps_values)
                if metrics.average_dps > 0:
                    metrics.dps_coefficient_of_variation = metrics.dps_std_dev / metrics.average_dps
        
        # Use dividend CAGR from Phase 1 if available
        metrics.dividend_cagr = dividend_history.dividend_cagr
        
        # Calculate DPS CAGR
        if len(metrics.dps_by_year) >= 2:
            dps_sorted = [metrics.dps_by_year[y] for y in sorted_periods if y in metrics.dps_by_year]
            if dps_sorted[0] > 0 and dps_sorted[-1] > 0:
                n_years = len(dps_sorted) - 1
                metrics.dps_cagr = (dps_sorted[-1] / dps_sorted[0]) ** (1 / n_years) - 1
        
        # YoY growth rates
        dps_list = [(y, metrics.dps_by_year[y]) for y in sorted_periods if y in metrics.dps_by_year]
        for i in range(1, len(dps_list)):
            curr_year, curr_dps = dps_list[i]
            prev_year, prev_dps = dps_list[i - 1]
            if prev_dps > 0:
                metrics.yoy_growth_rates[curr_year] = (curr_dps - prev_dps) / prev_dps
        
        if metrics.yoy_growth_rates:
            metrics.average_growth = np.mean(list(metrics.yoy_growth_rates.values()))
        
        # Payout ratios
        for year in sorted_periods:
            div = metrics.total_dividends_by_year.get(year)
            net_income = self._get_value(income_statement, "net_income", year)
            if div and net_income and net_income > 0:
                metrics.payout_ratio_by_year[year] = div / net_income
        
        if metrics.payout_ratio_by_year:
            payout_values = list(metrics.payout_ratio_by_year.values())
            metrics.current_payout_ratio = payout_values[-1] if payout_values else None
            metrics.average_payout_ratio = np.mean(payout_values)
            
            # Determine payout trend
            if len(payout_values) >= 3:
                if payout_values[-1] > payout_values[0] * 1.1:
                    metrics.payout_trend = "increasing"
                elif payout_values[-1] < payout_values[0] * 0.9:
                    metrics.payout_trend = "decreasing"
                else:
                    metrics.payout_trend = "stable"
        
        # Consecutive years analysis
        metrics.consecutive_years_paid = metrics.years_analyzed
        
        increase_count = 0
        for i in range(1, len(dividend_values)):
            if dividend_values[i] >= dividend_values[i - 1]:
                increase_count += 1
            else:
                break
        metrics.consecutive_years_increased = increase_count
        
        # Quality assessment
        metrics.dividend_quality = self._assess_quality(metrics, dividend_history)
        
        return metrics
    
    def _get_value(self, df: pd.DataFrame, field: str, year: str) -> Optional[float]:
        """Safely extract value."""
        if df is None or df.empty:
            return None
        if field not in df.index or year not in df.columns:
            return None
        value = df.loc[field, year]
        return float(value) if pd.notna(value) else None
    
    def _assess_quality(
        self,
        metrics: HistoricalDividendMetrics,
        dividend_history: Any,
    ) -> DividendQuality:
        """Assess dividend quality."""
        if metrics.years_analyzed < 3:
            return DividendQuality.INSUFFICIENT
        
        # Scoring
        score = 0
        
        # No cuts
        if not dividend_history.has_dividend_cuts:
            score += 2
        
        # Stable payout
        if dividend_history.payout_stable:
            score += 1
        
        # Low variability
        cv = metrics.dps_coefficient_of_variation or 0
        if cv < 0.15:
            score += 2
        elif cv < 0.30:
            score += 1
        
        # Positive growth
        if metrics.dps_cagr and metrics.dps_cagr > 0:
            score += 1
        
        # Reasonable payout ratio
        avg_payout = metrics.average_payout_ratio or 0
        if 0.2 <= avg_payout <= 0.6:
            score += 1
        
        if score >= 6:
            return DividendQuality.HIGH
        elif score >= 4:
            return DividendQuality.MODERATE
        else:
            return DividendQuality.LOW


# =============================================================================
# GROWTH RATE DERIVER
# =============================================================================

class DDMGrowthDeriver:
    """
    Derives dividend growth rates using multiple methodologies.
    
    Methods:
        1. Historical Dividend/DPS CAGR
        2. Sustainable Dividend Growth (ROE * Payout Ratio, capped)
        3. Earnings-Based (EPS growth rate)
    
    Selection uses conservative minimum approach for dividend stability.
    """
    
    def derive(
        self,
        historical_dividends: HistoricalDividendMetrics,
        company_profile: Any,
    ) -> DDMGrowthAnalysis:
        """Derive dividend growth rates."""
        analysis = DDMGrowthAnalysis()
        available_rates = []
        
        # Method A: Historical CAGR
        analysis.historical_cagr = self._calculate_historical_cagr(historical_dividends)
        if analysis.historical_cagr.is_available and analysis.historical_cagr.rate is not None:
            if analysis.historical_cagr.rate >= 0:  # Only use positive growth for dividends
                available_rates.append(analysis.historical_cagr.rate)
        
        # Method B: Sustainable Growth
        analysis.sustainable_growth = self._calculate_sustainable_growth(
            historical_dividends, company_profile
        )
        if analysis.sustainable_growth.is_available and analysis.sustainable_growth.rate is not None:
            if analysis.sustainable_growth.rate >= 0:
                available_rates.append(analysis.sustainable_growth.rate)
        
        # Method C: Earnings-Based
        analysis.earnings_based = self._calculate_earnings_based(company_profile)
        if analysis.earnings_based.is_available and analysis.earnings_based.rate is not None:
            if analysis.earnings_based.rate >= 0:
                available_rates.append(analysis.earnings_based.rate)
        
        # Select projection growth rate (conservative: minimum)
        if available_rates:
            uncapped_rate = min(available_rates)  # Conservative for dividends
            analysis.original_uncapped_rate = uncapped_rate
            
            # Apply constraints
            capped_rate = max(DDMConfig.MIN_PROJECTION_GROWTH,
                            min(DDMConfig.MAX_PROJECTION_GROWTH, uncapped_rate))
            
            analysis.projection_growth_rate = capped_rate
            analysis.growth_cap_applied = (capped_rate != uncapped_rate)
            
            analysis.selection_method = "minimum"
            analysis.selection_rationale = f"Minimum of {len(available_rates)} methods (conservative for dividends)"
        else:
            # Fallback
            analysis.projection_growth_rate = 0.02
            analysis.selection_method = "fallback"
            analysis.selection_rationale = "No reliable estimates, using 2% conservative assumption"
        
        # Terminal growth rate
        analysis.terminal_growth_rate = DDMConfig.DEFAULT_TERMINAL_GROWTH
        
        return analysis
    
    def _calculate_historical_cagr(
        self,
        historical_dividends: HistoricalDividendMetrics,
    ) -> DDMGrowthEstimate:
        """Calculate from historical DPS CAGR."""
        estimate = DDMGrowthEstimate(method=DDMGrowthMethod.HISTORICAL_CAGR)
        
        # Prefer DPS CAGR over total dividend CAGR
        cagr = historical_dividends.dps_cagr or historical_dividends.dividend_cagr
        
        if cagr is not None:
            estimate.rate = cagr
            estimate.is_available = True
            
            # Confidence based on quality
            quality_score = {
                DividendQuality.HIGH: 0.9,
                DividendQuality.MODERATE: 0.7,
                DividendQuality.LOW: 0.4,
                DividendQuality.INSUFFICIENT: 0.2,
            }.get(historical_dividends.dividend_quality, 0.5)
            
            years_score = min(historical_dividends.years_analyzed / 5, 1.0)
            estimate.confidence = quality_score * years_score
            
            estimate.description = f"{historical_dividends.years_analyzed}-year DPS CAGR: {cagr:.2%}"
        else:
            estimate.description = "Insufficient data for CAGR calculation"
        
        return estimate
    
    def _calculate_sustainable_growth(
        self,
        historical_dividends: HistoricalDividendMetrics,
        company_profile: Any,
    ) -> DDMGrowthEstimate:
        """
        Calculate sustainable dividend growth.
        
        For dividends: g = ROE * (1 - Payout Ratio) is internal growth,
        but dividend growth is typically limited by earnings growth.
        Use: g = min(ROE * Retention, Earnings Growth)
        """
        estimate = DDMGrowthEstimate(method=DDMGrowthMethod.SUSTAINABLE_GROWTH)
        
        roe = getattr(company_profile, 'roe', None)
        payout_ratio = historical_dividends.average_payout_ratio
        
        if roe is not None and payout_ratio is not None:
            # Robust ROE conversion
            if roe >= 2:
                roe_decimal = roe / 100
            else:
                roe_decimal = roe
            
            retention = 1 - payout_ratio
            sustainable_rate = roe_decimal * retention
            
            # Cap at reasonable level
            estimate.rate = min(sustainable_rate, 0.15)
            estimate.is_available = True
            estimate.confidence = 0.6
            
            if sustainable_rate > 0.15:
                estimate.description = (
                    f"ROE ({roe_decimal:.1%}) x Retention ({retention:.1%}) = "
                    f"{sustainable_rate:.1%}, capped to 15%"
                )
            else:
                estimate.description = f"ROE ({roe_decimal:.1%}) x Retention ({retention:.1%})"
        else:
            estimate.description = "ROE or payout ratio data not available"
        
        return estimate
    
    def _calculate_earnings_based(self, company_profile: Any) -> DDMGrowthEstimate:
        """Use earnings growth as proxy for dividend growth potential."""
        estimate = DDMGrowthEstimate(method=DDMGrowthMethod.EARNINGS_BASED)
        
        earnings_growth = getattr(company_profile, 'earnings_growth_yoy', None)
        
        if earnings_growth is not None:
            # Robust conversion
            if abs(earnings_growth) >= 2:
                rate = earnings_growth / 100
            else:
                rate = earnings_growth
            
            # Flag anomalies
            if abs(rate) > 0.30:
                estimate.rate = min(max(rate, -0.10), 0.15)  # Cap extreme values
                estimate.is_available = True
                estimate.confidence = 0.3
                estimate.description = f"Earnings growth {rate:.1%} (anomalous, capped)"
            else:
                estimate.rate = rate
                estimate.is_available = True
                estimate.confidence = 0.7
                estimate.description = f"Earnings growth rate: {rate:.2%}"
        else:
            estimate.description = "Earnings growth data not available"
        
        return estimate


# =============================================================================
# DDM PROJECTOR
# =============================================================================

class DDMProjector:
    """
    Projects dividends and calculates intrinsic value.
    
    Methodology:
        1. Project DPS for Years 1-5
        2. Calculate Terminal Value using Gordon Growth
        3. Discount all to present value
        4. Sum for intrinsic value per share
    """
    
    def project(
        self,
        base_dps: float,
        projection_growth_rate: float,
        terminal_growth_rate: float,
        cost_of_equity: float,
    ) -> DDMProjection:
        """Project dividends and value."""
        projection = DDMProjection(
            base_dps=base_dps,
            projection_growth_rate=projection_growth_rate,
            terminal_growth_rate=terminal_growth_rate,
            cost_of_equity=cost_of_equity,
        )
        
        # Project DPS for Years 1-5
        current_dps = base_dps
        sum_pv = 0.0
        
        for year in range(1, DDMConfig.PROJECTION_YEARS + 1):
            projected_dps = current_dps * (1 + projection_growth_rate)
            current_dps = projected_dps
            
            discount_factor = 1 / ((1 + cost_of_equity) ** year)
            present_value = projected_dps * discount_factor
            
            yearly_proj = YearlyDividendProjection(
                year=year,
                dps=projected_dps,
                growth_rate=projection_growth_rate,
                discount_factor=discount_factor,
                present_value=present_value,
            )
            projection.yearly_projections.append(yearly_proj)
            sum_pv += present_value
        
        projection.sum_of_pv_dividends = sum_pv
        
        # Terminal value
        projection.terminal_value = DDMTerminalValue(
            final_year_dps=current_dps,
            terminal_growth_rate=terminal_growth_rate,
            cost_of_equity=cost_of_equity,
        )
        projection.terminal_value.calculate()
        projection.pv_terminal_value = projection.terminal_value.present_value
        
        # Intrinsic value per share
        projection.intrinsic_value_per_share = sum_pv + projection.pv_terminal_value
        
        # Terminal value percentage
        if projection.intrinsic_value_per_share > 0:
            projection.terminal_value_pct = projection.pv_terminal_value / projection.intrinsic_value_per_share
        
        return projection


# =============================================================================
# SENSITIVITY ANALYZER
# =============================================================================

class DDMSensitivityAnalyzer:
    """Performs DDM sensitivity and scenario analysis."""
    
    def __init__(self, projector: DDMProjector):
        self.projector = projector
    
    def analyze(
        self,
        base_dps: float,
        base_growth: float,
        terminal_growth: float,
        base_ke: float,
        current_price: Optional[float],
    ) -> DDMSensitivityAnalysis:
        """Perform sensitivity analysis."""
        analysis = DDMSensitivityAnalysis()
        
        # Sensitivity matrix
        analysis.sensitivity_matrix = self._build_matrix(
            base_dps, base_growth, terminal_growth, base_ke
        )
        
        # Scenarios
        analysis.bear_case = self._calculate_scenario(
            DDMScenarioType.BEAR, base_dps, base_growth, terminal_growth, base_ke,
            current_price, DDMConfig.BEAR_CASE_MULTIPLIER
        )
        
        analysis.base_case = self._calculate_scenario(
            DDMScenarioType.BASE, base_dps, base_growth, terminal_growth, base_ke,
            current_price, 1.0
        )
        
        analysis.bull_case = self._calculate_scenario(
            DDMScenarioType.BULL, base_dps, base_growth, terminal_growth, base_ke,
            current_price, DDMConfig.BULL_CASE_MULTIPLIER
        )
        
        # Value range
        all_values = [v for row in analysis.sensitivity_matrix.values for v in row if v > 0]
        if all_values:
            analysis.min_value = min(all_values)
            analysis.max_value = max(all_values)
            mid = (analysis.min_value + analysis.max_value) / 2
            if mid > 0:
                analysis.value_range_pct = (analysis.max_value - analysis.min_value) / mid
        
        return analysis
    
    def _build_matrix(
        self,
        base_dps: float,
        base_growth: float,
        terminal_growth: float,
        base_ke: float,
    ) -> DDMSensitivityMatrix:
        """Build sensitivity matrix."""
        matrix = DDMSensitivityMatrix()
        
        growth_deltas = DDMConfig.GROWTH_SENSITIVITY_RANGE
        ke_deltas = DDMConfig.KE_SENSITIVITY_RANGE
        
        matrix.growth_rates = [base_growth + d for d in growth_deltas]
        matrix.discount_rates = [base_ke + d for d in ke_deltas]
        matrix.base_growth_idx = growth_deltas.index(0)
        matrix.base_ke_idx = ke_deltas.index(0)
        
        matrix.values = []
        for growth in matrix.growth_rates:
            row = []
            for ke in matrix.discount_rates:
                if ke > terminal_growth:
                    proj = self.projector.project(base_dps, growth, terminal_growth, ke)
                    row.append(proj.intrinsic_value_per_share)
                else:
                    row.append(0.0)
            matrix.values.append(row)
        
        return matrix
    
    def _calculate_scenario(
        self,
        scenario: DDMScenarioType,
        base_dps: float,
        base_growth: float,
        terminal_growth: float,
        base_ke: float,
        current_price: Optional[float],
        growth_multiplier: float,
    ) -> DDMScenarioValuation:
        """Calculate scenario valuation."""
        scenario_growth = base_growth * growth_multiplier
        scenario_growth = max(DDMConfig.MIN_PROJECTION_GROWTH,
                             min(DDMConfig.MAX_PROJECTION_GROWTH, scenario_growth))
        
        if scenario == DDMScenarioType.BEAR:
            scenario_terminal = DDMConfig.MIN_TERMINAL_GROWTH
        elif scenario == DDMScenarioType.BULL:
            scenario_terminal = DDMConfig.MAX_TERMINAL_GROWTH
        else:
            scenario_terminal = terminal_growth
        
        projection = self.projector.project(base_dps, scenario_growth, scenario_terminal, base_ke)
        
        upside = 0.0
        if current_price and current_price > 0:
            upside = (projection.intrinsic_value_per_share - current_price) / current_price
        
        return DDMScenarioValuation(
            scenario=scenario,
            growth_rate=scenario_growth,
            terminal_growth_rate=scenario_terminal,
            intrinsic_value_per_share=projection.intrinsic_value_per_share,
            upside_downside_pct=upside,
        )


# =============================================================================
# DCF RECONCILER
# =============================================================================

class DCFDDMReconciler:
    """
    Reconciles DCF and DDM valuations.
    
    Explains differences between the two methodologies.
    """
    
    def reconcile(
        self,
        ddm_result: DDMValuationResult,
        dcf_result: Optional[Any],
    ) -> DCFDDMReconciliation:
        """Reconcile DCF and DDM values."""
        recon = DCFDDMReconciliation()
        
        recon.ddm_intrinsic_value = ddm_result.intrinsic_value_per_share
        
        if dcf_result is None:
            recon.explanation.append("DCF valuation not available for comparison")
            return recon
        
        recon.dcf_intrinsic_value = dcf_result.intrinsic_value_per_share
        
        # Calculate difference
        recon.value_difference = recon.dcf_intrinsic_value - recon.ddm_intrinsic_value
        
        if recon.ddm_intrinsic_value > 0:
            recon.percentage_difference = recon.value_difference / recon.ddm_intrinsic_value
        
        recon.dcf_higher = recon.dcf_intrinsic_value > recon.ddm_intrinsic_value
        
        # Convergence status
        pct_diff = abs(recon.percentage_difference)
        if pct_diff <= 0.10:
            recon.convergence_status = "strong_convergence"
        elif pct_diff <= 0.25:
            recon.convergence_status = "moderate_convergence"
        elif pct_diff <= 0.50:
            recon.convergence_status = "weak_convergence"
        else:
            recon.convergence_status = "divergent"
        
        # Store methodology parameters
        recon.dcf_growth_rate = dcf_result.growth_analysis.projection_growth_rate
        recon.ddm_growth_rate = ddm_result.growth_analysis.projection_growth_rate
        recon.dcf_discount_rate = dcf_result.wacc_calculation.wacc_constrained
        recon.ddm_discount_rate = ddm_result.cost_of_equity.cost_of_equity
        
        # Generate explanation
        recon.explanation = self._generate_explanation(recon, ddm_result, dcf_result)
        
        return recon
    
    def _generate_explanation(
        self,
        recon: DCFDDMReconciliation,
        ddm_result: DDMValuationResult,
        dcf_result: Any,
    ) -> List[str]:
        """Generate explanation for value differences."""
        explanations = []
        
        # Cash flow basis
        explanations.append(
            f"DCF uses Free Cash Flow (all cash available to firm), "
            f"DDM uses Dividends (cash distributed to shareholders)"
        )
        
        # Growth rate comparison
        if recon.dcf_growth_rate and recon.ddm_growth_rate:
            if recon.dcf_growth_rate > recon.ddm_growth_rate:
                explanations.append(
                    f"DCF growth ({recon.dcf_growth_rate:.1%}) exceeds DDM growth "
                    f"({recon.ddm_growth_rate:.1%}), contributing to higher DCF value"
                )
            elif recon.ddm_growth_rate > recon.dcf_growth_rate:
                explanations.append(
                    f"DDM growth ({recon.ddm_growth_rate:.1%}) exceeds DCF growth "
                    f"({recon.dcf_growth_rate:.1%}), contributing to higher DDM value"
                )
        
        # Discount rate comparison
        if recon.dcf_discount_rate and recon.ddm_discount_rate:
            if recon.dcf_discount_rate < recon.ddm_discount_rate:
                explanations.append(
                    f"DCF uses WACC ({recon.dcf_discount_rate:.1%}) which is lower than "
                    f"DDM's Cost of Equity ({recon.ddm_discount_rate:.1%}), increasing DCF value"
                )
            elif recon.ddm_discount_rate < recon.dcf_discount_rate:
                explanations.append(
                    f"DDM uses Cost of Equity ({recon.ddm_discount_rate:.1%}) which is lower than "
                    f"DCF's WACC ({recon.dcf_discount_rate:.1%}), increasing DDM value"
                )
        
        # Payout ratio impact
        payout = ddm_result.historical_dividends.average_payout_ratio
        if payout:
            retained = 1 - payout
            explanations.append(
                f"Company retains {retained:.1%} of earnings, "
                f"FCF captures this retained value while DDM does not"
            )
        
        return explanations


# =============================================================================
# MAIN VALUATOR CLASS
# =============================================================================

class Phase6Valuator:
    """
    Main orchestrator for Phase 6 DDM Valuation.
    
    Coordinates all components for dividend discount model valuation.
    
    Usage:
        valuator = Phase6Valuator()
        result = valuator.value(collection_result, dcf_result)
    """
    
    def __init__(self):
        self.applicability_checker = DDMApplicabilityChecker()
        self.dividend_analyzer = HistoricalDividendAnalyzer()
        self.growth_deriver = DDMGrowthDeriver()
        self.projector = DDMProjector()
        self.sensitivity_analyzer = DDMSensitivityAnalyzer(self.projector)
        self.reconciler = DCFDDMReconciler()
        self.logger = LOGGER
    
    def value(
        self,
        collection_result: Any,
        dcf_result: Optional[Any] = None,
    ) -> DDMValuationResult:
        """
        Perform complete DDM valuation.
        
        Args:
            collection_result: CollectionResult from Phase 1
            dcf_result: Optional DCFValuationResult from Phase 5
            
        Returns:
            DDMValuationResult with intrinsic value and analysis
        """
        self.logger.info(f"Phase 6: Starting DDM valuation for {collection_result.ticker}")
        
        result = DDMValuationResult(
            ticker=collection_result.ticker,
            company_name=collection_result.company_name,
        )
        
        # Extract data
        profile = collection_result.company_profile
        statements = collection_result.statements
        dividend_history = collection_result.dividend_history
        periods = statements.fiscal_periods
        
        # Get shares outstanding
        shares = getattr(profile, 'shares_outstanding', None)
        if shares is None:
            shares = self._get_value(statements.balance_sheet, "shares_outstanding", periods[0])
        
        if not shares or shares <= 0:
            result.validation_errors.append("Shares outstanding not available")
            return result
        
        # Step 1: Check applicability
        self.logger.info("  Checking DDM applicability")
        result.applicability = self.applicability_checker.check(
            dividend_history, statements.income_statement, periods
        )
        
        result.is_applicable = (result.applicability.status != DDMApplicability.NOT_APPLICABLE)
        
        if not result.is_applicable:
            self.logger.info(f"  DDM not applicable: {result.applicability.exclusion_reasons}")
            return result
        
        # Step 2: Analyze historical dividends
        self.logger.info("  Analyzing historical dividends")
        result.historical_dividends = self.dividend_analyzer.analyze(
            dividend_history, statements.income_statement, statements.balance_sheet,
            periods, shares
        )
        
        if result.historical_dividends.current_dps <= 0:
            result.validation_errors.append("Current DPS is not positive")
            return result
        
        # Step 3: Derive growth rates
        self.logger.info("  Deriving dividend growth rates")
        result.growth_analysis = self.growth_deriver.derive(
            result.historical_dividends, profile
        )
        
        # Step 4: Calculate Cost of Equity
        self.logger.info("  Calculating cost of equity")
        beta = getattr(profile, 'beta', None)
        if beta and 0.1 < beta < 3.0:
            result.cost_of_equity.beta = beta
            result.cost_of_equity.beta_source = "company_data"
        
        ke = result.cost_of_equity.calculate()
        
        # Step 5: Project dividends
        self.logger.info("  Projecting dividends and calculating intrinsic value")
        
        base_dps = result.historical_dividends.current_dps
        proj_growth = result.growth_analysis.projection_growth_rate
        term_growth = result.growth_analysis.terminal_growth_rate
        
        if ke <= term_growth:
            result.validation_errors.append(
                f"Cost of equity ({ke:.2%}) must exceed terminal growth ({term_growth:.2%})"
            )
            return result
        
        result.ddm_projection = self.projector.project(base_dps, proj_growth, term_growth, ke)
        result.intrinsic_value_per_share = result.ddm_projection.intrinsic_value_per_share
        
        # Step 6: Market comparison
        market_cap = getattr(profile, 'market_cap', None)
        if market_cap and shares > 0:
            result.current_price = market_cap / shares
        
        if result.current_price and result.current_price > 0:
            result.upside_downside_pct = (
                (result.intrinsic_value_per_share - result.current_price) / result.current_price
            )
            
            # Dividend yields
            result.current_dividend_yield = base_dps / result.current_price
            result.implied_dividend_yield = base_dps / result.intrinsic_value_per_share
        
        # Step 7: Sensitivity analysis
        self.logger.info("  Running sensitivity analysis")
        result.sensitivity_analysis = self.sensitivity_analyzer.analyze(
            base_dps, proj_growth, term_growth, ke, result.current_price
        )
        
        # Step 8: DCF reconciliation
        if dcf_result:
            self.logger.info("  Reconciling with DCF valuation")
            result.dcf_reconciliation = self.reconciler.reconcile(result, dcf_result)
        
        # Step 9: Record assumptions
        result.assumptions = {
            "risk_free_rate": DDMConfig.RISK_FREE_RATE,
            "equity_risk_premium": DDMConfig.EQUITY_RISK_PREMIUM,
            "projection_years": DDMConfig.PROJECTION_YEARS,
            "terminal_growth_cap": DDMConfig.MAX_TERMINAL_GROWTH,
            "projection_growth_cap": DDMConfig.MAX_PROJECTION_GROWTH,
        }
        
        # Validate
        result.is_valid = self._validate_result(result)
        
        self.logger.info(
            f"Phase 6 complete: DDM Intrinsic Value ${result.intrinsic_value_per_share:.2f}, "
            f"Upside {result.upside_downside_pct:.1%}"
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
    
    def _validate_result(self, result: DDMValuationResult) -> bool:
        """Validate DDM result."""
        if result.validation_errors:
            return False
        
        if result.intrinsic_value_per_share <= 0:
            result.validation_errors.append("Intrinsic value is non-positive")
            return False
        
        tv_pct = result.ddm_projection.terminal_value_pct
        if tv_pct > 0.95:
            result.validation_warnings.append(
                f"Terminal value represents {tv_pct:.0%} of intrinsic value (very high)"
            )
        
        return True
    
    def save_report(
        self,
        result: DDMValuationResult,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Save DDM valuation report."""
        if output_dir is None:
            output_dir = OUTPUT_DIR
        
        ticker_dir = output_dir / result.ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = ticker_dir / f"{result.ticker}_ddm_valuation.json"
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"Saved DDM valuation to {filepath}")
        return filepath


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def value_company_ddm(
    collection_result: Any,
    dcf_result: Optional[Any] = None,
) -> DDMValuationResult:
    """Convenience function for DDM valuation."""
    valuator = Phase6Valuator()
    return valuator.value(collection_result, dcf_result)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "__version__",
    "DDMConfig",
    "DDMApplicability",
    "DividendQuality",
    "DDMGrowthMethod",
    "DDMScenarioType",
    "DDMApplicabilityResult",
    "HistoricalDividendMetrics",
    "DDMGrowthEstimate",
    "DDMGrowthAnalysis",
    "DDMCostOfEquity",
    "YearlyDividendProjection",
    "DDMTerminalValue",
    "DDMProjection",
    "DDMSensitivityMatrix",
    "DDMScenarioValuation",
    "DDMSensitivityAnalysis",
    "DCFDDMReconciliation",
    "DDMValuationResult",
    "DDMApplicabilityChecker",
    "HistoricalDividendAnalyzer",
    "DDMGrowthDeriver",
    "DDMProjector",
    "DDMSensitivityAnalyzer",
    "DCFDDMReconciler",
    "Phase6Valuator",
    "value_company_ddm",
]