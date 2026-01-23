"""
DCF Valuation Module - Phase 5 Intrinsic Valuation
Fundamental Analyst Agent

Implements institutional-grade Discounted Cash Flow (DCF) valuation using
Free Cash Flow projections, WACC-based discounting, and Gordon Growth
terminal value methodology.

Methodology:
    Enterprise Value = Sum of PV(FCF) for Years 1-5 + PV(Terminal Value)
    Equity Value = Enterprise Value - Net Debt + Cash
    Intrinsic Value per Share = Equity Value / Shares Outstanding

Key Components:
    - Historical FCF Analysis: 5-year trend, volatility, and quality assessment
    - Growth Rate Derivation: Historical CAGR, Sustainable Growth, Analyst Consensus
    - WACC Calculation: CAPM-based Cost of Equity, After-tax Cost of Debt
    - DCF Projection: 5-year explicit period with Gordon Growth terminal value
    - Sensitivity Analysis: Growth rate vs discount rate matrix
    - Scenario Analysis: Bull, Base, and Bear case valuations

Inputs: CollectionResult from Phase 1 (with derived metrics and company profile)
Outputs: DCFValuationResult with intrinsic value and comprehensive analysis

MSc Coursework: IFTE0001 AI Agents in Asset Management
Track A: Fundamental Analyst Agent
Phase 5: DCF Valuation (Free Cash Flow Based)

[REQUIRED BY COURSEWORK: "Basic intrinsic valuation (DCF or multiples)"]

Version: 1.0.1
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


__version__ = "1.0.1"


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

class DCFConfig:
    """Configuration parameters for DCF valuation."""
    
    # Risk-free rate (10-Year US Treasury approximation)
    RISK_FREE_RATE: float = 0.043  # 4.3%
    
    # Equity Risk Premium (historical average)
    EQUITY_RISK_PREMIUM: float = 0.055  # 5.5%
    
    # Default beta for missing data
    DEFAULT_BETA: float = 1.0
    
    # Terminal growth rate constraints
    MIN_TERMINAL_GROWTH: float = 0.02  # 2% floor (inflation)
    MAX_TERMINAL_GROWTH: float = 0.03  # 3% cap (long-term GDP)
    DEFAULT_TERMINAL_GROWTH: float = 0.025  # 2.5% default
    
    # Projection period growth constraints
    MAX_PROJECTION_GROWTH: float = 0.15  # 15% cap for projection period
    MIN_PROJECTION_GROWTH: float = -0.05  # -5% floor
    
    # WACC constraints
    MIN_WACC: float = 0.06  # 6% floor
    MAX_WACC: float = 0.20  # 20% cap
    
    # Cost of debt estimation
    DEFAULT_COST_OF_DEBT: float = 0.05  # 5% if cannot calculate
    INVESTMENT_GRADE_SPREAD: float = 0.015  # 1.5% spread over risk-free
    
    # Tax rate
    DEFAULT_TAX_RATE: float = 0.21  # 21% US corporate rate
    
    # Projection period
    PROJECTION_YEARS: int = 5
    
    # Sensitivity analysis ranges
    GROWTH_SENSITIVITY_RANGE: List[float] = [-0.02, -0.01, 0, 0.01, 0.02]
    WACC_SENSITIVITY_RANGE: List[float] = [-0.02, -0.01, 0, 0.01, 0.02]
    
    # Scenario multipliers (applied to base growth)
    BEAR_CASE_MULTIPLIER: float = 0.5
    BULL_CASE_MULTIPLIER: float = 1.5


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ValuationSignal(Enum):
    """Valuation signal based on intrinsic vs market value."""
    SIGNIFICANTLY_UNDERVALUED = "significantly_undervalued"  # >30% upside
    UNDERVALUED = "undervalued"                              # 15-30% upside
    FAIRLY_VALUED = "fairly_valued"                          # -15% to +15%
    OVERVALUED = "overvalued"                                # 15-30% downside
    SIGNIFICANTLY_OVERVALUED = "significantly_overvalued"    # >30% downside


class GrowthMethod(Enum):
    """Methods for deriving growth rate."""
    HISTORICAL_CAGR = "historical_cagr"
    SUSTAINABLE_GROWTH = "sustainable_growth"
    ANALYST_CONSENSUS = "analyst_consensus"
    WEIGHTED_AVERAGE = "weighted_average"


class FCFQuality(Enum):
    """Quality assessment of historical FCF."""
    HIGH = "high"           # Consistent positive, low volatility
    MODERATE = "moderate"   # Mostly positive, moderate volatility
    LOW = "low"             # Inconsistent or high volatility
    NEGATIVE = "negative"   # Negative average FCF


class ScenarioType(Enum):
    """Valuation scenario types."""
    BEAR = "bear"
    BASE = "base"
    BULL = "bull"


# =============================================================================
# DATA CONTAINERS - HISTORICAL FCF ANALYSIS
# =============================================================================

@dataclass
class HistoricalFCFMetrics:
    """Historical Free Cash Flow analysis metrics."""
    
    # Raw FCF data by year
    fcf_by_year: Dict[str, float] = field(default_factory=dict)
    
    # Summary statistics
    years_analyzed: int = 0
    total_fcf: float = 0.0
    average_fcf: float = 0.0
    latest_fcf: float = 0.0
    
    # Growth metrics
    fcf_cagr: Optional[float] = None
    yoy_growth_rates: Dict[str, float] = field(default_factory=dict)
    average_growth: Optional[float] = None
    
    # Volatility metrics
    fcf_std_dev: Optional[float] = None
    coefficient_of_variation: Optional[float] = None
    
    # Margin analysis
    fcf_margins: Dict[str, float] = field(default_factory=dict)
    average_fcf_margin: Optional[float] = None
    
    # Quality assessment
    positive_years: int = 0
    negative_years: int = 0
    fcf_quality: FCFQuality = FCFQuality.LOW
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fcf_by_year": self.fcf_by_year,
            "years_analyzed": self.years_analyzed,
            "total_fcf": self.total_fcf,
            "average_fcf": self.average_fcf,
            "latest_fcf": self.latest_fcf,
            "fcf_cagr": self.fcf_cagr,
            "yoy_growth_rates": self.yoy_growth_rates,
            "average_growth": self.average_growth,
            "fcf_std_dev": self.fcf_std_dev,
            "coefficient_of_variation": self.coefficient_of_variation,
            "fcf_margins": self.fcf_margins,
            "average_fcf_margin": self.average_fcf_margin,
            "positive_years": self.positive_years,
            "negative_years": self.negative_years,
            "fcf_quality": self.fcf_quality.value,
        }


# =============================================================================
# DATA CONTAINERS - GROWTH RATE DERIVATION
# =============================================================================

@dataclass
class GrowthRateEstimate:
    """Single growth rate estimate with source."""
    
    method: GrowthMethod
    rate: Optional[float] = None
    confidence: float = 0.0  # 0.0 to 1.0
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
class GrowthRateAnalysis:
    """Complete growth rate derivation analysis."""
    
    # Individual estimates
    historical_cagr: GrowthRateEstimate = field(default_factory=lambda: GrowthRateEstimate(GrowthMethod.HISTORICAL_CAGR))
    sustainable_growth: GrowthRateEstimate = field(default_factory=lambda: GrowthRateEstimate(GrowthMethod.SUSTAINABLE_GROWTH))
    analyst_consensus: GrowthRateEstimate = field(default_factory=lambda: GrowthRateEstimate(GrowthMethod.ANALYST_CONSENSUS))
    
    # Selected rates
    projection_growth_rate: float = 0.0  # For Years 1-5
    terminal_growth_rate: float = DCFConfig.DEFAULT_TERMINAL_GROWTH
    
    # Selection methodology
    selection_method: str = ""
    selection_rationale: str = ""
    
    # Adjustments applied
    growth_cap_applied: bool = False
    original_uncapped_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "historical_cagr": self.historical_cagr.to_dict(),
            "sustainable_growth": self.sustainable_growth.to_dict(),
            "analyst_consensus": self.analyst_consensus.to_dict(),
            "projection_growth_rate": self.projection_growth_rate,
            "terminal_growth_rate": self.terminal_growth_rate,
            "selection_method": self.selection_method,
            "selection_rationale": self.selection_rationale,
            "growth_cap_applied": self.growth_cap_applied,
            "original_uncapped_rate": self.original_uncapped_rate,
        }


# =============================================================================
# DATA CONTAINERS - WACC CALCULATION
# =============================================================================

@dataclass
class CostOfEquity:
    """Cost of Equity calculation via CAPM."""
    
    risk_free_rate: float = DCFConfig.RISK_FREE_RATE
    beta: float = DCFConfig.DEFAULT_BETA
    equity_risk_premium: float = DCFConfig.EQUITY_RISK_PREMIUM
    cost_of_equity: float = 0.0
    
    # Source flags
    beta_source: str = "default"
    
    def calculate(self) -> float:
        """Calculate cost of equity using CAPM: Ke = Rf + Beta * ERP"""
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


@dataclass
class CostOfDebt:
    """Cost of Debt calculation."""
    
    interest_expense: Optional[float] = None
    total_debt: Optional[float] = None
    pre_tax_cost: float = DCFConfig.DEFAULT_COST_OF_DEBT
    tax_rate: float = DCFConfig.DEFAULT_TAX_RATE
    after_tax_cost: float = 0.0
    
    # Calculation method
    calculation_method: str = "default"
    
    def calculate(self) -> float:
        """Calculate after-tax cost of debt."""
        if self.interest_expense and self.total_debt and self.total_debt > 0:
            self.pre_tax_cost = self.interest_expense / self.total_debt
            self.calculation_method = "actual"
        else:
            self.pre_tax_cost = DCFConfig.RISK_FREE_RATE + DCFConfig.INVESTMENT_GRADE_SPREAD
            self.calculation_method = "estimated"
        
        self.after_tax_cost = self.pre_tax_cost * (1 - self.tax_rate)
        return self.after_tax_cost
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "interest_expense": self.interest_expense,
            "total_debt": self.total_debt,
            "pre_tax_cost": self.pre_tax_cost,
            "tax_rate": self.tax_rate,
            "after_tax_cost": self.after_tax_cost,
            "calculation_method": self.calculation_method,
        }


@dataclass
class WACCCalculation:
    """Weighted Average Cost of Capital calculation."""
    
    # Components
    cost_of_equity: CostOfEquity = field(default_factory=CostOfEquity)
    cost_of_debt: CostOfDebt = field(default_factory=CostOfDebt)
    
    # Capital structure
    market_cap: Optional[float] = None
    total_debt: Optional[float] = None
    equity_weight: float = 0.0
    debt_weight: float = 0.0
    
    # Result
    wacc: float = 0.0
    wacc_constrained: float = 0.0
    constraint_applied: bool = False
    
    def calculate(self) -> float:
        """Calculate WACC with capital structure weights."""
        # Calculate weights
        if self.market_cap and self.total_debt:
            total_capital = self.market_cap + self.total_debt
            if total_capital > 0:
                self.equity_weight = self.market_cap / total_capital
                self.debt_weight = self.total_debt / total_capital
        
        # Default to 100% equity if no debt data
        if self.equity_weight == 0 and self.debt_weight == 0:
            self.equity_weight = 1.0
            self.debt_weight = 0.0
        
        # Calculate component costs
        ke = self.cost_of_equity.calculate()
        kd = self.cost_of_debt.calculate()
        
        # Calculate WACC
        self.wacc = (self.equity_weight * ke) + (self.debt_weight * kd)
        
        # Apply constraints
        self.wacc_constrained = max(DCFConfig.MIN_WACC, min(DCFConfig.MAX_WACC, self.wacc))
        self.constraint_applied = (self.wacc_constrained != self.wacc)
        
        return self.wacc_constrained
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cost_of_equity": self.cost_of_equity.to_dict(),
            "cost_of_debt": self.cost_of_debt.to_dict(),
            "market_cap": self.market_cap,
            "total_debt": self.total_debt,
            "equity_weight": self.equity_weight,
            "debt_weight": self.debt_weight,
            "wacc_raw": self.wacc,
            "wacc_constrained": self.wacc_constrained,
            "constraint_applied": self.constraint_applied,
        }


# =============================================================================
# DATA CONTAINERS - DCF PROJECTION
# =============================================================================

@dataclass
class YearlyProjection:
    """Single year FCF projection."""
    
    year: int
    fcf: float
    growth_rate: float
    discount_factor: float
    present_value: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "year": self.year,
            "fcf": self.fcf,
            "growth_rate": self.growth_rate,
            "discount_factor": self.discount_factor,
            "present_value": self.present_value,
        }


@dataclass
class TerminalValueCalculation:
    """Terminal value calculation using Gordon Growth Model."""
    
    final_year_fcf: float = 0.0
    terminal_growth_rate: float = DCFConfig.DEFAULT_TERMINAL_GROWTH
    discount_rate: float = 0.0
    
    # Calculated values
    terminal_year_fcf: float = 0.0  # FCF in terminal year (Year 6)
    terminal_value: float = 0.0     # Undiscounted terminal value
    discount_factor: float = 0.0
    present_value: float = 0.0
    
    # Validation
    is_valid: bool = True
    validation_message: str = ""
    
    def calculate(self) -> float:
        """Calculate terminal value using Gordon Growth Model."""
        # Validate inputs
        if self.discount_rate <= self.terminal_growth_rate:
            self.is_valid = False
            self.validation_message = "Discount rate must exceed terminal growth rate"
            return 0.0
        
        # Terminal year FCF (Year 6)
        self.terminal_year_fcf = self.final_year_fcf * (1 + self.terminal_growth_rate)
        
        # Gordon Growth: TV = FCF6 / (WACC - g)
        self.terminal_value = self.terminal_year_fcf / (self.discount_rate - self.terminal_growth_rate)
        
        # Discount back to present (5 years)
        self.discount_factor = 1 / ((1 + self.discount_rate) ** DCFConfig.PROJECTION_YEARS)
        self.present_value = self.terminal_value * self.discount_factor
        
        self.is_valid = True
        return self.present_value
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_year_fcf": self.final_year_fcf,
            "terminal_growth_rate": self.terminal_growth_rate,
            "discount_rate": self.discount_rate,
            "terminal_year_fcf": self.terminal_year_fcf,
            "terminal_value_undiscounted": self.terminal_value,
            "discount_factor": self.discount_factor,
            "present_value": self.present_value,
            "is_valid": self.is_valid,
            "validation_message": self.validation_message,
            "formula": "FCF * (1+g) / (WACC - g)",
        }


@dataclass
class DCFProjection:
    """Complete DCF projection with all components."""
    
    # Base inputs
    base_fcf: float = 0.0
    projection_growth_rate: float = 0.0
    terminal_growth_rate: float = DCFConfig.DEFAULT_TERMINAL_GROWTH
    discount_rate: float = 0.0
    
    # Projections
    yearly_projections: List[YearlyProjection] = field(default_factory=list)
    terminal_value: TerminalValueCalculation = field(default_factory=TerminalValueCalculation)
    
    # Calculated values
    sum_of_pv_fcf: float = 0.0
    pv_terminal_value: float = 0.0
    enterprise_value: float = 0.0
    
    # Terminal value as percentage of EV (sanity check)
    terminal_value_pct: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_fcf": self.base_fcf,
            "projection_growth_rate": self.projection_growth_rate,
            "terminal_growth_rate": self.terminal_growth_rate,
            "discount_rate": self.discount_rate,
            "yearly_projections": [p.to_dict() for p in self.yearly_projections],
            "terminal_value": self.terminal_value.to_dict(),
            "sum_of_pv_fcf": self.sum_of_pv_fcf,
            "pv_terminal_value": self.pv_terminal_value,
            "enterprise_value": self.enterprise_value,
            "terminal_value_pct": self.terminal_value_pct,
        }


# =============================================================================
# DATA CONTAINERS - SENSITIVITY ANALYSIS
# =============================================================================

@dataclass
class SensitivityMatrix:
    """Sensitivity analysis matrix for growth and WACC."""
    
    # Axis values
    growth_rates: List[float] = field(default_factory=list)
    discount_rates: List[float] = field(default_factory=list)
    
    # Matrix of intrinsic values per share
    # values[i][j] = value at growth_rates[i] and discount_rates[j]
    values: List[List[float]] = field(default_factory=list)
    
    # Base case indices
    base_growth_idx: int = 0
    base_wacc_idx: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "growth_rates": self.growth_rates,
            "discount_rates": self.discount_rates,
            "values": self.values,
            "base_growth_idx": self.base_growth_idx,
            "base_wacc_idx": self.base_wacc_idx,
        }


@dataclass
class ScenarioValuation:
    """Single scenario valuation result."""
    
    scenario: ScenarioType
    growth_rate: float
    terminal_growth_rate: float
    enterprise_value: float
    equity_value: float
    intrinsic_value_per_share: float
    upside_downside_pct: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.value,
            "growth_rate": self.growth_rate,
            "terminal_growth_rate": self.terminal_growth_rate,
            "enterprise_value": self.enterprise_value,
            "equity_value": self.equity_value,
            "intrinsic_value_per_share": self.intrinsic_value_per_share,
            "upside_downside_pct": self.upside_downside_pct,
        }


@dataclass
class SensitivityAnalysis:
    """Complete sensitivity and scenario analysis."""
    
    # Sensitivity matrix
    sensitivity_matrix: SensitivityMatrix = field(default_factory=SensitivityMatrix)
    
    # Scenario analysis
    bear_case: Optional[ScenarioValuation] = None
    base_case: Optional[ScenarioValuation] = None
    bull_case: Optional[ScenarioValuation] = None
    
    # Value range
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
# DATA CONTAINERS - FINAL VALUATION RESULT
# =============================================================================

@dataclass
class DCFValuationResult:
    """Complete Phase 5 DCF Valuation output."""
    
    # Identification
    ticker: str = ""
    company_name: str = ""
    valuation_date: datetime = field(default_factory=datetime.now)
    
    # Historical analysis
    historical_fcf: HistoricalFCFMetrics = field(default_factory=HistoricalFCFMetrics)
    
    # Growth analysis
    growth_analysis: GrowthRateAnalysis = field(default_factory=GrowthRateAnalysis)
    
    # Cost of capital
    wacc_calculation: WACCCalculation = field(default_factory=WACCCalculation)
    
    # DCF projection
    dcf_projection: DCFProjection = field(default_factory=DCFProjection)
    
    # Valuation outputs
    enterprise_value: float = 0.0
    net_debt: float = 0.0
    cash_and_equivalents: float = 0.0
    equity_value: float = 0.0
    shares_outstanding: float = 0.0
    intrinsic_value_per_share: float = 0.0
    
    # Market comparison
    current_price: Optional[float] = None
    upside_downside_pct: float = 0.0
    valuation_signal: ValuationSignal = ValuationSignal.FAIRLY_VALUED
    
    # Sensitivity analysis
    sensitivity_analysis: SensitivityAnalysis = field(default_factory=SensitivityAnalysis)
    
    # Validation
    is_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    
    # Metadata
    assumptions: Dict[str, Any] = field(default_factory=dict)
    data_sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "valuation_date": self.valuation_date.isoformat(),
            "historical_fcf": self.historical_fcf.to_dict(),
            "growth_analysis": self.growth_analysis.to_dict(),
            "wacc_calculation": self.wacc_calculation.to_dict(),
            "dcf_projection": self.dcf_projection.to_dict(),
            "enterprise_value": self.enterprise_value,
            "net_debt": self.net_debt,
            "cash_and_equivalents": self.cash_and_equivalents,
            "equity_value": self.equity_value,
            "shares_outstanding": self.shares_outstanding,
            "intrinsic_value_per_share": self.intrinsic_value_per_share,
            "current_price": self.current_price,
            "upside_downside_pct": self.upside_downside_pct,
            "valuation_signal": self.valuation_signal.value,
            "sensitivity_analysis": self.sensitivity_analysis.to_dict(),
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "assumptions": self.assumptions,
            "data_sources": self.data_sources,
        }


# =============================================================================
# HISTORICAL FCF ANALYZER
# =============================================================================

class HistoricalFCFAnalyzer:
    """
    Analyzes historical Free Cash Flow data from Phase 1.
    
    Extracts FCF trends, calculates growth rates, assesses volatility,
    and determines FCF quality for projection reliability.
    """
    
    def analyze(
        self,
        derived_metrics: Any,
        income_statement: pd.DataFrame,
        fiscal_periods: List[str],
    ) -> HistoricalFCFMetrics:
        """
        Analyze historical FCF from Phase 1 derived metrics.
        
        Args:
            derived_metrics: DerivedMetrics from CollectionResult
            income_statement: Income statement DataFrame for revenue data
            fiscal_periods: List of fiscal years (descending order)
            
        Returns:
            HistoricalFCFMetrics with complete analysis
        """
        metrics = HistoricalFCFMetrics()
        
        # Extract FCF by year
        fcf_data = derived_metrics.fcf_calculated
        
        # Sort periods chronologically (oldest first)
        sorted_periods = sorted(fiscal_periods, reverse=False)
        
        fcf_values = []
        for year in sorted_periods:
            fcf = fcf_data.get(year)
            if fcf is not None:
                metrics.fcf_by_year[year] = fcf
                fcf_values.append(fcf)
                if fcf > 0:
                    metrics.positive_years += 1
                else:
                    metrics.negative_years += 1
        
        metrics.years_analyzed = len(fcf_values)
        
        if metrics.years_analyzed == 0:
            metrics.fcf_quality = FCFQuality.NEGATIVE
            return metrics
        
        # Summary statistics
        metrics.total_fcf = sum(fcf_values)
        metrics.average_fcf = np.mean(fcf_values)
        metrics.latest_fcf = fcf_values[-1] if fcf_values else 0.0
        
        # Volatility metrics
        if len(fcf_values) > 1:
            metrics.fcf_std_dev = np.std(fcf_values)
            if metrics.average_fcf != 0:
                metrics.coefficient_of_variation = abs(metrics.fcf_std_dev / metrics.average_fcf)
        
        # Calculate CAGR
        if len(fcf_values) >= 2 and fcf_values[0] > 0 and fcf_values[-1] > 0:
            n_years = len(fcf_values) - 1
            metrics.fcf_cagr = (fcf_values[-1] / fcf_values[0]) ** (1 / n_years) - 1
        
        # Calculate YoY growth rates
        for i in range(1, len(sorted_periods)):
            curr_year = sorted_periods[i]
            prev_year = sorted_periods[i - 1]
            curr_fcf = metrics.fcf_by_year.get(curr_year)
            prev_fcf = metrics.fcf_by_year.get(prev_year)
            
            if curr_fcf is not None and prev_fcf is not None and prev_fcf != 0:
                yoy_growth = (curr_fcf - prev_fcf) / abs(prev_fcf)
                metrics.yoy_growth_rates[curr_year] = yoy_growth
        
        if metrics.yoy_growth_rates:
            metrics.average_growth = np.mean(list(metrics.yoy_growth_rates.values()))
        
        # Calculate FCF margins
        for year in sorted_periods:
            fcf = metrics.fcf_by_year.get(year)
            revenue = self._get_value(income_statement, "total_revenue", year)
            if fcf is not None and revenue is not None and revenue > 0:
                metrics.fcf_margins[year] = fcf / revenue
        
        if metrics.fcf_margins:
            metrics.average_fcf_margin = np.mean(list(metrics.fcf_margins.values()))
        
        # Assess FCF quality
        metrics.fcf_quality = self._assess_quality(metrics)
        
        return metrics
    
    def _get_value(self, df: pd.DataFrame, field: str, year: str) -> Optional[float]:
        """Safely extract value from DataFrame."""
        if df is None or df.empty:
            return None
        if field not in df.index or year not in df.columns:
            return None
        value = df.loc[field, year]
        return float(value) if pd.notna(value) else None
    
    def _assess_quality(self, metrics: HistoricalFCFMetrics) -> FCFQuality:
        """Assess FCF quality based on consistency and volatility."""
        if metrics.average_fcf <= 0:
            return FCFQuality.NEGATIVE
        
        # Check consistency (positive years ratio)
        positive_ratio = metrics.positive_years / max(metrics.years_analyzed, 1)
        
        # Check volatility
        cv = metrics.coefficient_of_variation or 0
        
        if positive_ratio >= 0.8 and cv < 0.3:
            return FCFQuality.HIGH
        elif positive_ratio >= 0.6 and cv < 0.5:
            return FCFQuality.MODERATE
        else:
            return FCFQuality.LOW


# =============================================================================
# GROWTH RATE DERIVER
# =============================================================================

class GrowthRateDeriver:
    """
    Derives growth rates using multiple methodologies.
    
    Methods:
        1. Historical FCF CAGR
        2. Sustainable Growth Rate (ROE * Retention Ratio)
        3. Analyst Consensus (if available)
    
    Selection uses median of available methods, capped at maximum thresholds.
    """
    
    def derive(
        self,
        historical_fcf: HistoricalFCFMetrics,
        company_profile: Any,
        ratio_data: Optional[Dict[str, Any]] = None,
    ) -> GrowthRateAnalysis:
        """
        Derive growth rates from multiple sources.
        
        Args:
            historical_fcf: Historical FCF analysis
            company_profile: CompanyProfile from CollectionResult
            ratio_data: Optional ratio analysis data
            
        Returns:
            GrowthRateAnalysis with selected rates
        """
        analysis = GrowthRateAnalysis()
        available_rates = []
        
        # Method A: Historical FCF CAGR
        analysis.historical_cagr = self._calculate_historical_cagr(historical_fcf)
        if analysis.historical_cagr.is_available:
            available_rates.append(analysis.historical_cagr.rate)
        
        # Method B: Sustainable Growth Rate
        analysis.sustainable_growth = self._calculate_sustainable_growth(company_profile, ratio_data)
        if analysis.sustainable_growth.is_available:
            available_rates.append(analysis.sustainable_growth.rate)
        
        # Method C: Analyst Consensus
        analysis.analyst_consensus = self._get_analyst_consensus(company_profile)
        if analysis.analyst_consensus.is_available:
            available_rates.append(analysis.analyst_consensus.rate)
        
        # Select projection growth rate
        if available_rates:
            # Use median of available rates
            uncapped_rate = np.median(available_rates)
            analysis.original_uncapped_rate = uncapped_rate
            
            # Apply constraints
            capped_rate = max(DCFConfig.MIN_PROJECTION_GROWTH, 
                            min(DCFConfig.MAX_PROJECTION_GROWTH, uncapped_rate))
            
            analysis.projection_growth_rate = capped_rate
            analysis.growth_cap_applied = (capped_rate != uncapped_rate)
            
            # Clearly indicate when cap is applied in selection method
            if analysis.growth_cap_applied:
                analysis.selection_method = f"median_capped_to_{DCFConfig.MAX_PROJECTION_GROWTH:.0%}"
                analysis.selection_rationale = (
                    f"Median of {len(available_rates)} methods was {uncapped_rate:.1%}, "
                    f"capped to {DCFConfig.MAX_PROJECTION_GROWTH:.0%} maximum"
                )
            else:
                analysis.selection_method = "median"
                analysis.selection_rationale = f"Median of {len(available_rates)} available methods"
        else:
            # Fallback to conservative estimate
            analysis.projection_growth_rate = 0.03
            analysis.selection_method = "fallback"
            analysis.selection_rationale = "No reliable growth estimates available, using 3% conservative assumption"
        
        # Terminal growth rate (always constrained to GDP growth)
        analysis.terminal_growth_rate = DCFConfig.DEFAULT_TERMINAL_GROWTH
        
        return analysis
    
    def _calculate_historical_cagr(self, historical_fcf: HistoricalFCFMetrics) -> GrowthRateEstimate:
        """Calculate growth from historical FCF CAGR."""
        estimate = GrowthRateEstimate(method=GrowthMethod.HISTORICAL_CAGR)
        
        if historical_fcf.fcf_cagr is not None:
            estimate.rate = historical_fcf.fcf_cagr
            estimate.is_available = True
            
            # Confidence based on data quality and consistency
            quality_score = {
                FCFQuality.HIGH: 0.9,
                FCFQuality.MODERATE: 0.7,
                FCFQuality.LOW: 0.4,
                FCFQuality.NEGATIVE: 0.1,
            }.get(historical_fcf.fcf_quality, 0.5)
            
            years_score = min(historical_fcf.years_analyzed / 5, 1.0)
            estimate.confidence = quality_score * years_score
            
            estimate.description = f"{historical_fcf.years_analyzed}-year FCF CAGR: {estimate.rate:.1%}"
        else:
            estimate.description = "Insufficient positive FCF data for CAGR calculation"
        
        return estimate
    
    def _calculate_sustainable_growth(
        self,
        company_profile: Any,
        ratio_data: Optional[Dict[str, Any]],
    ) -> GrowthRateEstimate:
        """Calculate sustainable growth rate: ROE * (1 - Payout Ratio)."""
        estimate = GrowthRateEstimate(method=GrowthMethod.SUSTAINABLE_GROWTH)
        
        roe = getattr(company_profile, 'roe', None)
        if roe is None and ratio_data:
            roe = ratio_data.get('roe')
        
        # Estimate retention ratio from dividend data
        dividend_yield = getattr(company_profile, 'dividend_yield', None)
        pe_ratio = getattr(company_profile, 'pe_ratio', None)
        
        if roe is not None:
            # Robust ROE conversion to decimal
            # ROE can be stored as: 171.4 (percentage) or 1.714 (decimal) or 0.05 (5% decimal)
            # Rule: If ROE >= 2, assume percentage format (e.g., 171.4 -> 1.714)
            #       If ROE < 2, assume already decimal (e.g., 1.714 stays 1.714, 0.05 stays 0.05)
            # Rationale: No company has 200%+ ROE that would be stored as decimal > 2
            if roe >= 2:
                roe_decimal = roe / 100
            else:
                roe_decimal = roe
            
            # Estimate payout ratio from dividend yield and P/E
            # Payout Ratio = Dividend Yield * P/E (since DPS/P * P/EPS = DPS/EPS)
            if dividend_yield and pe_ratio and pe_ratio > 0:
                # dividend_yield might be percentage (0.4) or decimal (0.004)
                div_yield_decimal = dividend_yield / 100 if dividend_yield > 0.5 else dividend_yield
                payout_ratio = div_yield_decimal * pe_ratio
            else:
                payout_ratio = 0.3  # Default assumption
            
            retention_ratio = 1 - min(payout_ratio, 0.9)  # Cap payout at 90%
            
            # Calculate sustainable growth (may exceed 100% for high-ROE companies)
            sustainable_rate = roe_decimal * retention_ratio
            
            # Cap sustainable growth at reasonable maximum for projection use
            # but preserve original for description
            estimate.rate = min(sustainable_rate, 0.50)  # Cap at 50% for reasonableness
            estimate.is_available = True
            estimate.confidence = 0.7 if sustainable_rate <= 0.50 else 0.4  # Lower confidence if capped
            
            if sustainable_rate > 0.50:
                estimate.description = f"ROE ({roe_decimal:.1%}) x Retention ({retention_ratio:.1%}) = {sustainable_rate:.1%}, capped to 50%"
            else:
                estimate.description = f"ROE ({roe_decimal:.1%}) x Retention ({retention_ratio:.1%})"
        else:
            estimate.description = "ROE data not available"
        
        return estimate
    
    def _get_analyst_consensus(self, company_profile: Any) -> GrowthRateEstimate:
        """Extract analyst growth consensus if available."""
        estimate = GrowthRateEstimate(method=GrowthMethod.ANALYST_CONSENSUS)
        
        # Try to get earnings growth from profile
        earnings_growth = getattr(company_profile, 'earnings_growth_yoy', None)
        revenue_growth = getattr(company_profile, 'revenue_growth_yoy', None)
        
        growth_value = None
        growth_source = None
        
        if earnings_growth is not None:
            growth_value = earnings_growth
            growth_source = "earnings"
        elif revenue_growth is not None:
            growth_value = revenue_growth
            growth_source = "revenue"
        
        if growth_value is not None:
            # Robust conversion: values >= 2 are likely percentages (e.g., 19.5 for 19.5%)
            # values < 2 could be decimals (e.g., 0.195 for 19.5%)
            if abs(growth_value) >= 2:
                rate_decimal = growth_value / 100
            else:
                rate_decimal = growth_value
            
            # Validate: analyst growth estimates > 50% YoY are likely anomalies
            # (one-time earnings spikes, tax benefits, etc.)
            if abs(rate_decimal) > 0.50:
                # Mark as anomaly but still include with low confidence
                estimate.rate = rate_decimal
                estimate.is_available = True
                estimate.confidence = 0.2  # Very low confidence for anomalous values
                estimate.description = f"Analyst {growth_source} growth: {rate_decimal:.1%} (anomalous, low confidence)"
            else:
                estimate.rate = rate_decimal
                estimate.is_available = True
                estimate.confidence = 0.8 if growth_source == "earnings" else 0.6
                estimate.description = f"Analyst {growth_source} growth estimate: {rate_decimal:.1%}"
        else:
            estimate.description = "No analyst consensus available"
        
        return estimate


# =============================================================================
# WACC CALCULATOR
# =============================================================================

class WACCCalculator:
    """
    Calculates Weighted Average Cost of Capital.
    
    Components:
        - Cost of Equity via CAPM: Rf + Beta * ERP
        - Cost of Debt: Interest Expense / Total Debt (after tax)
        - Weights: Market Cap / (Market Cap + Debt)
    """
    
    def calculate(
        self,
        company_profile: Any,
        balance_sheet: pd.DataFrame,
        income_statement: pd.DataFrame,
        latest_year: str,
    ) -> WACCCalculation:
        """
        Calculate WACC using company data.
        
        Args:
            company_profile: CompanyProfile with beta and market cap
            balance_sheet: Balance sheet DataFrame
            income_statement: Income statement DataFrame
            latest_year: Most recent fiscal year
            
        Returns:
            WACCCalculation with complete breakdown
        """
        wacc_calc = WACCCalculation()
        
        # Extract data for Cost of Equity
        beta = getattr(company_profile, 'beta', None)
        if beta is not None and 0.1 < beta < 3.0:
            wacc_calc.cost_of_equity.beta = beta
            wacc_calc.cost_of_equity.beta_source = "company_data"
        else:
            wacc_calc.cost_of_equity.beta = DCFConfig.DEFAULT_BETA
            wacc_calc.cost_of_equity.beta_source = "default"
        
        # Extract data for Cost of Debt
        interest_expense = self._get_value(income_statement, "interest_expense", latest_year)
        total_debt = self._get_value(balance_sheet, "total_debt", latest_year)
        
        wacc_calc.cost_of_debt.interest_expense = interest_expense
        wacc_calc.cost_of_debt.total_debt = total_debt
        
        # Estimate tax rate from actual data if available
        pretax_income = self._get_value(income_statement, "pretax_income", latest_year)
        tax_expense = self._get_value(income_statement, "tax_expense", latest_year)
        
        if pretax_income and pretax_income > 0 and tax_expense:
            effective_tax_rate = tax_expense / pretax_income
            if 0.1 < effective_tax_rate < 0.4:
                wacc_calc.cost_of_debt.tax_rate = effective_tax_rate
        
        # Capital structure weights
        market_cap = getattr(company_profile, 'market_cap', None)
        wacc_calc.market_cap = market_cap
        wacc_calc.total_debt = total_debt
        
        # Calculate WACC
        wacc_calc.calculate()
        
        return wacc_calc
    
    def _get_value(self, df: pd.DataFrame, field: str, year: str) -> Optional[float]:
        """Safely extract value from DataFrame."""
        if df is None or df.empty:
            return None
        if field not in df.index or year not in df.columns:
            return None
        value = df.loc[field, year]
        return float(value) if pd.notna(value) else None


# =============================================================================
# DCF PROJECTOR
# =============================================================================

class DCFProjector:
    """
    Projects Free Cash Flows and calculates intrinsic value.
    
    Methodology:
        1. Project FCF for Years 1-5 using growth rate
        2. Calculate Terminal Value using Gordon Growth Model
        3. Discount all cash flows to present value
        4. Sum to get Enterprise Value
    """
    
    def project(
        self,
        base_fcf: float,
        projection_growth_rate: float,
        terminal_growth_rate: float,
        discount_rate: float,
    ) -> DCFProjection:
        """
        Project FCF and calculate present values.
        
        Args:
            base_fcf: Starting FCF (latest year)
            projection_growth_rate: Growth rate for Years 1-5
            terminal_growth_rate: Long-term growth rate
            discount_rate: WACC for discounting
            
        Returns:
            DCFProjection with yearly projections and terminal value
        """
        projection = DCFProjection(
            base_fcf=base_fcf,
            projection_growth_rate=projection_growth_rate,
            terminal_growth_rate=terminal_growth_rate,
            discount_rate=discount_rate,
        )
        
        # Project FCF for Years 1-5
        current_fcf = base_fcf
        sum_pv = 0.0
        
        for year in range(1, DCFConfig.PROJECTION_YEARS + 1):
            # Project FCF
            projected_fcf = current_fcf * (1 + projection_growth_rate)
            current_fcf = projected_fcf
            
            # Calculate discount factor and present value
            discount_factor = 1 / ((1 + discount_rate) ** year)
            present_value = projected_fcf * discount_factor
            
            yearly_proj = YearlyProjection(
                year=year,
                fcf=projected_fcf,
                growth_rate=projection_growth_rate,
                discount_factor=discount_factor,
                present_value=present_value,
            )
            projection.yearly_projections.append(yearly_proj)
            sum_pv += present_value
        
        projection.sum_of_pv_fcf = sum_pv
        
        # Calculate Terminal Value
        projection.terminal_value = TerminalValueCalculation(
            final_year_fcf=current_fcf,
            terminal_growth_rate=terminal_growth_rate,
            discount_rate=discount_rate,
        )
        projection.terminal_value.calculate()
        projection.pv_terminal_value = projection.terminal_value.present_value
        
        # Enterprise Value
        projection.enterprise_value = projection.sum_of_pv_fcf + projection.pv_terminal_value
        
        # Terminal value as percentage of EV
        if projection.enterprise_value > 0:
            projection.terminal_value_pct = projection.pv_terminal_value / projection.enterprise_value
        
        return projection


# =============================================================================
# SENSITIVITY ANALYZER
# =============================================================================

class SensitivityAnalyzer:
    """
    Performs sensitivity and scenario analysis on DCF valuation.
    
    Sensitivity Matrix: Growth rate vs WACC impact on intrinsic value
    Scenario Analysis: Bear, Base, Bull case valuations
    """
    
    def __init__(self, projector: DCFProjector):
        self.projector = projector
    
    def analyze(
        self,
        base_fcf: float,
        base_growth: float,
        terminal_growth: float,
        base_wacc: float,
        net_debt: float,
        cash: float,
        shares_outstanding: float,
        current_price: Optional[float],
    ) -> SensitivityAnalysis:
        """
        Perform sensitivity and scenario analysis.
        
        Args:
            base_fcf: Latest FCF
            base_growth: Base case projection growth rate
            terminal_growth: Terminal growth rate
            base_wacc: Base case WACC
            net_debt: Net debt for equity bridge
            cash: Cash and equivalents
            shares_outstanding: Shares for per-share calculation
            current_price: Current market price
            
        Returns:
            SensitivityAnalysis with matrix and scenarios
        """
        analysis = SensitivityAnalysis()
        
        # Build sensitivity matrix
        analysis.sensitivity_matrix = self._build_sensitivity_matrix(
            base_fcf, base_growth, terminal_growth, base_wacc,
            net_debt, cash, shares_outstanding
        )
        
        # Scenario analysis
        analysis.bear_case = self._calculate_scenario(
            ScenarioType.BEAR, base_fcf, base_growth, terminal_growth, base_wacc,
            net_debt, cash, shares_outstanding, current_price,
            DCFConfig.BEAR_CASE_MULTIPLIER
        )
        
        analysis.base_case = self._calculate_scenario(
            ScenarioType.BASE, base_fcf, base_growth, terminal_growth, base_wacc,
            net_debt, cash, shares_outstanding, current_price, 1.0
        )
        
        analysis.bull_case = self._calculate_scenario(
            ScenarioType.BULL, base_fcf, base_growth, terminal_growth, base_wacc,
            net_debt, cash, shares_outstanding, current_price,
            DCFConfig.BULL_CASE_MULTIPLIER
        )
        
        # Value range
        all_values = [v for row in analysis.sensitivity_matrix.values for v in row if v > 0]
        if all_values:
            analysis.min_value = min(all_values)
            analysis.max_value = max(all_values)
            mid_value = (analysis.min_value + analysis.max_value) / 2
            if mid_value > 0:
                analysis.value_range_pct = (analysis.max_value - analysis.min_value) / mid_value
        
        return analysis
    
    def _build_sensitivity_matrix(
        self,
        base_fcf: float,
        base_growth: float,
        terminal_growth: float,
        base_wacc: float,
        net_debt: float,
        cash: float,
        shares_outstanding: float,
    ) -> SensitivityMatrix:
        """Build sensitivity matrix varying growth and WACC."""
        matrix = SensitivityMatrix()
        
        # Define ranges
        growth_deltas = DCFConfig.GROWTH_SENSITIVITY_RANGE
        wacc_deltas = DCFConfig.WACC_SENSITIVITY_RANGE
        
        matrix.growth_rates = [base_growth + d for d in growth_deltas]
        matrix.discount_rates = [base_wacc + d for d in wacc_deltas]
        
        # Find base case indices
        matrix.base_growth_idx = growth_deltas.index(0)
        matrix.base_wacc_idx = wacc_deltas.index(0)
        
        # Calculate values for each combination
        matrix.values = []
        for growth in matrix.growth_rates:
            row = []
            for wacc in matrix.discount_rates:
                intrinsic = self._calculate_intrinsic_value(
                    base_fcf, growth, terminal_growth, wacc,
                    net_debt, cash, shares_outstanding
                )
                row.append(intrinsic)
            matrix.values.append(row)
        
        return matrix
    
    def _calculate_scenario(
        self,
        scenario: ScenarioType,
        base_fcf: float,
        base_growth: float,
        terminal_growth: float,
        base_wacc: float,
        net_debt: float,
        cash: float,
        shares_outstanding: float,
        current_price: Optional[float],
        growth_multiplier: float,
    ) -> ScenarioValuation:
        """Calculate single scenario valuation."""
        # Adjust growth rate for scenario
        scenario_growth = base_growth * growth_multiplier
        scenario_growth = max(DCFConfig.MIN_PROJECTION_GROWTH,
                             min(DCFConfig.MAX_PROJECTION_GROWTH, scenario_growth))
        
        # Adjust terminal growth for scenario
        if scenario == ScenarioType.BEAR:
            scenario_terminal = DCFConfig.MIN_TERMINAL_GROWTH
        elif scenario == ScenarioType.BULL:
            scenario_terminal = DCFConfig.MAX_TERMINAL_GROWTH
        else:
            scenario_terminal = terminal_growth
        
        # Project and value
        projection = self.projector.project(
            base_fcf, scenario_growth, scenario_terminal, base_wacc
        )
        
        # Calculate equity value and per-share
        equity_value = projection.enterprise_value - net_debt + cash
        intrinsic_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0
        
        # Upside/downside
        upside_pct = 0.0
        if current_price and current_price > 0:
            upside_pct = (intrinsic_per_share - current_price) / current_price
        
        return ScenarioValuation(
            scenario=scenario,
            growth_rate=scenario_growth,
            terminal_growth_rate=scenario_terminal,
            enterprise_value=projection.enterprise_value,
            equity_value=equity_value,
            intrinsic_value_per_share=intrinsic_per_share,
            upside_downside_pct=upside_pct,
        )
    
    def _calculate_intrinsic_value(
        self,
        base_fcf: float,
        growth: float,
        terminal_growth: float,
        wacc: float,
        net_debt: float,
        cash: float,
        shares_outstanding: float,
    ) -> float:
        """Calculate intrinsic value per share for given parameters."""
        # Ensure WACC > terminal growth
        if wacc <= terminal_growth:
            return 0.0
        
        projection = self.projector.project(base_fcf, growth, terminal_growth, wacc)
        equity_value = projection.enterprise_value - net_debt + cash
        
        if shares_outstanding > 0:
            return equity_value / shares_outstanding
        return 0.0


# =============================================================================
# MAIN VALUATOR CLASS
# =============================================================================

class Phase5Valuator:
    """
    Main orchestrator for Phase 5 DCF Valuation.
    
    Coordinates all components to produce comprehensive DCF valuation
    with sensitivity analysis and scenario modeling.
    
    Usage:
        valuator = Phase5Valuator()
        result = valuator.value(collection_result)
    """
    
    def __init__(self):
        self.fcf_analyzer = HistoricalFCFAnalyzer()
        self.growth_deriver = GrowthRateDeriver()
        self.wacc_calculator = WACCCalculator()
        self.projector = DCFProjector()
        self.sensitivity_analyzer = SensitivityAnalyzer(self.projector)
        self.logger = LOGGER
    
    def value(self, collection_result: Any) -> DCFValuationResult:
        """
        Perform complete DCF valuation from Phase 1 data.
        
        Args:
            collection_result: CollectionResult from Phase 1
            
        Returns:
            DCFValuationResult with intrinsic value and analysis
        """
        self.logger.info(f"Phase 5: Starting DCF valuation for {collection_result.ticker}")
        
        result = DCFValuationResult(
            ticker=collection_result.ticker,
            company_name=collection_result.company_name,
        )
        
        # Extract data
        profile = collection_result.company_profile
        statements = collection_result.statements
        derived = collection_result.derived_metrics
        periods = statements.fiscal_periods
        latest_year = periods[0] if periods else None
        
        if not latest_year:
            result.validation_errors.append("No fiscal periods available")
            return result
        
        # Step 1: Analyze Historical FCF
        self.logger.info("  Analyzing historical FCF")
        result.historical_fcf = self.fcf_analyzer.analyze(
            derived, statements.income_statement, periods
        )
        
        if result.historical_fcf.latest_fcf <= 0:
            result.validation_warnings.append(
                f"Latest FCF is non-positive: ${result.historical_fcf.latest_fcf/1e9:.2f}B"
            )
        
        # Step 2: Derive Growth Rates
        self.logger.info("  Deriving growth rates")
        result.growth_analysis = self.growth_deriver.derive(
            result.historical_fcf, profile
        )
        
        # Step 3: Calculate WACC
        self.logger.info("  Calculating WACC")
        result.wacc_calculation = self.wacc_calculator.calculate(
            profile, statements.balance_sheet, statements.income_statement, latest_year
        )
        
        wacc = result.wacc_calculation.wacc_constrained
        
        # Step 4: Project FCF and Calculate Value
        self.logger.info("  Projecting FCF and calculating enterprise value")
        
        base_fcf = result.historical_fcf.latest_fcf
        proj_growth = result.growth_analysis.projection_growth_rate
        term_growth = result.growth_analysis.terminal_growth_rate
        
        # Validate inputs
        if wacc <= term_growth:
            result.validation_errors.append(
                f"WACC ({wacc:.2%}) must exceed terminal growth ({term_growth:.2%})"
            )
            return result
        
        result.dcf_projection = self.projector.project(
            base_fcf, proj_growth, term_growth, wacc
        )
        
        # Step 5: Calculate Equity Value
        result.enterprise_value = result.dcf_projection.enterprise_value
        
        # Get net debt and cash
        net_debt = derived.net_debt.get(latest_year, 0) or 0
        cash = self._get_value(statements.balance_sheet, "cash_and_equivalents", latest_year) or 0
        
        result.net_debt = net_debt
        result.cash_and_equivalents = cash
        result.equity_value = result.enterprise_value - net_debt + cash
        
        # Get shares outstanding
        shares = getattr(profile, 'shares_outstanding', None)
        if shares is None:
            shares = self._get_value(statements.balance_sheet, "shares_outstanding", latest_year)
        
        result.shares_outstanding = shares or 0
        
        if result.shares_outstanding > 0:
            result.intrinsic_value_per_share = result.equity_value / result.shares_outstanding
        else:
            result.validation_errors.append("Shares outstanding not available")
            return result
        
        # Step 6: Market Comparison
        self._compare_to_market(result, profile)
        
        # Step 7: Sensitivity Analysis
        self.logger.info("  Running sensitivity analysis")
        result.sensitivity_analysis = self.sensitivity_analyzer.analyze(
            base_fcf, proj_growth, term_growth, wacc,
            net_debt, cash, result.shares_outstanding,
            result.current_price
        )
        
        # Step 8: Record Assumptions and Sources
        result.assumptions = {
            "risk_free_rate": DCFConfig.RISK_FREE_RATE,
            "equity_risk_premium": DCFConfig.EQUITY_RISK_PREMIUM,
            "projection_years": DCFConfig.PROJECTION_YEARS,
            "terminal_growth_cap": DCFConfig.MAX_TERMINAL_GROWTH,
            "projection_growth_cap": DCFConfig.MAX_PROJECTION_GROWTH,
            "tax_rate": result.wacc_calculation.cost_of_debt.tax_rate,
        }
        
        result.data_sources = [
            "Phase 1: Alpha Vantage financial statements",
            "Phase 1: Derived FCF metrics",
            "Company beta from market data",
        ]
        
        # Validate result
        result.is_valid = self._validate_result(result)
        
        self.logger.info(
            f"Phase 5 complete: Intrinsic Value ${result.intrinsic_value_per_share:.2f}, "
            f"Signal: {result.valuation_signal.value}"
        )
        
        return result
    
    def _get_value(self, df: pd.DataFrame, field: str, year: str) -> Optional[float]:
        """Safely extract value from DataFrame."""
        if df is None or df.empty:
            return None
        if field not in df.index or year not in df.columns:
            return None
        value = df.loc[field, year]
        return float(value) if pd.notna(value) else None
    
    def _compare_to_market(self, result: DCFValuationResult, profile: Any) -> None:
        """Compare intrinsic value to current market price."""
        # Estimate current price from market cap and shares
        market_cap = getattr(profile, 'market_cap', None)
        if market_cap and result.shares_outstanding > 0:
            result.current_price = market_cap / result.shares_outstanding
        
        # Calculate upside/downside
        if result.current_price and result.current_price > 0:
            result.upside_downside_pct = (
                (result.intrinsic_value_per_share - result.current_price) / result.current_price
            )
            
            # Determine valuation signal
            pct = result.upside_downside_pct
            if pct > 0.30:
                result.valuation_signal = ValuationSignal.SIGNIFICANTLY_UNDERVALUED
            elif pct > 0.15:
                result.valuation_signal = ValuationSignal.UNDERVALUED
            elif pct < -0.30:
                result.valuation_signal = ValuationSignal.SIGNIFICANTLY_OVERVALUED
            elif pct < -0.15:
                result.valuation_signal = ValuationSignal.OVERVALUED
            else:
                result.valuation_signal = ValuationSignal.FAIRLY_VALUED
    
    def _validate_result(self, result: DCFValuationResult) -> bool:
        """Validate the valuation result."""
        # Check for critical errors
        if result.validation_errors:
            return False
        
        # Sanity checks
        if result.intrinsic_value_per_share <= 0:
            result.validation_errors.append("Intrinsic value is non-positive")
            return False
        
        # Check terminal value percentage (should be < 85% typically)
        tv_pct = result.dcf_projection.terminal_value_pct
        if tv_pct > 0.90:
            result.validation_warnings.append(
                f"Terminal value represents {tv_pct:.0%} of enterprise value (high)"
            )
        
        # Check if WACC was constrained
        if result.wacc_calculation.constraint_applied:
            result.validation_warnings.append(
                f"WACC was constrained from {result.wacc_calculation.wacc:.2%} to "
                f"{result.wacc_calculation.wacc_constrained:.2%}"
            )
        
        return True
    
    def save_report(
        self,
        result: DCFValuationResult,
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Save DCF valuation report to JSON file.
        
        Args:
            result: DCFValuationResult from valuation
            output_dir: Optional output directory (defaults to OUTPUT_DIR)
            
        Returns:
            Path to saved JSON file
        """
        if output_dir is None:
            output_dir = OUTPUT_DIR
        
        ticker_dir = output_dir / result.ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = ticker_dir / f"{result.ticker}_dcf_valuation.json"
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"Saved DCF valuation to {filepath}")
        return filepath


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def value_company_dcf(collection_result: Any) -> DCFValuationResult:
    """
    Convenience function for DCF valuation.
    
    Args:
        collection_result: CollectionResult from Phase 1
        
    Returns:
        DCFValuationResult with intrinsic value and analysis
    """
    valuator = Phase5Valuator()
    return valuator.value(collection_result)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Version
    "__version__",
    
    # Configuration
    "DCFConfig",
    
    # Enumerations
    "ValuationSignal",
    "GrowthMethod",
    "FCFQuality",
    "ScenarioType",
    
    # Data Containers
    "HistoricalFCFMetrics",
    "GrowthRateEstimate",
    "GrowthRateAnalysis",
    "CostOfEquity",
    "CostOfDebt",
    "WACCCalculation",
    "YearlyProjection",
    "TerminalValueCalculation",
    "DCFProjection",
    "SensitivityMatrix",
    "ScenarioValuation",
    "SensitivityAnalysis",
    "DCFValuationResult",
    
    # Analyzers
    "HistoricalFCFAnalyzer",
    "GrowthRateDeriver",
    "WACCCalculator",
    "DCFProjector",
    "SensitivityAnalyzer",
    "Phase5Valuator",
    
    # Convenience Function
    "value_company_dcf",
]