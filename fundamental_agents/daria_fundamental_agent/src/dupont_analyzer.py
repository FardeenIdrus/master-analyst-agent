"""
DuPont Analysis Module - Phase 4 ROE Decomposition
Fundamental Analyst Agent

Implements institutional-grade DuPont decomposition providing granular
ROE analysis with driver attribution, variance analysis, and quality assessment.

Three-Factor DuPont Model:
    ROE = Net Profit Margin x Asset Turnover x Equity Multiplier

Five-Factor DuPont Model:
    ROE = Tax Burden x Interest Burden x Operating Margin x Asset Turnover x Equity Multiplier

Features:
    - Traditional 3-factor and extended 5-factor decomposition
    - Year-over-year variance attribution with exact reconciliation
    - Multi-year trend analysis for all components
    - ROE quality and sustainability assessment
    - Professional reporting with actionable insights

Inputs: CollectionResult from Phase 1 (or ValidatedData from Phase 2)
Outputs: DuPontAnalysisResult container with complete decomposition

MSc Coursework: IFTE0001 AI Agents in Asset Management
Track A: Fundamental Analyst Agent
Phase 4: DuPont Decomposition

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
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum

from .config import (
    LOGGER,
    OUTPUT_DIR,
)


__version__ = "1.0.0"


# =============================================================================
# ENUMERATIONS
# =============================================================================

class DuPontDriver(Enum):
    """Primary driver of ROE performance."""
    PROFITABILITY = "profitability"
    EFFICIENCY = "efficiency"
    LEVERAGE = "leverage"
    TAX_MANAGEMENT = "tax_management"
    BALANCED = "balanced"


class ROEQuality(Enum):
    """ROE quality classification."""
    HIGH_QUALITY = "high_quality"
    MODERATE_QUALITY = "moderate_quality"
    LOW_QUALITY = "low_quality"
    LEVERAGE_DRIVEN = "leverage_driven"
    NOT_ASSESSED = "not_assessed"


class TrendDirection(Enum):
    """Trend direction classification."""
    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"
    VOLATILE = "volatile"
    INSUFFICIENT_DATA = "insufficient_data"


class DuPontComponent(Enum):
    """DuPont model components."""
    TAX_BURDEN = "tax_burden"
    INTEREST_BURDEN = "interest_burden"
    OPERATING_MARGIN = "operating_margin"
    NET_PROFIT_MARGIN = "net_profit_margin"
    ASSET_TURNOVER = "asset_turnover"
    EQUITY_MULTIPLIER = "equity_multiplier"
    ROE = "roe"


# Alias for compatibility
ComponentTrend = TrendDirection


# =============================================================================
# DATA CONTAINERS - COMPONENT VALUES
# =============================================================================

@dataclass
class ComponentValue:
    """Single component value with metadata."""
    value: Optional[float] = None
    numerator: Optional[float] = None
    denominator: Optional[float] = None
    is_estimated: bool = False
    note: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "is_estimated": self.is_estimated,
            "note": self.note,
        }


@dataclass
class ComponentTrendInfo:
    """Trend information for a component."""
    trend_direction: TrendDirection = TrendDirection.INSUFFICIENT_DATA
    cagr: Optional[float] = None
    average: Optional[float] = None
    volatility: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trend_direction": self.trend_direction.value,
            "cagr": self.cagr,
            "average": self.average,
            "volatility": self.volatility,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }


# =============================================================================
# DATA CONTAINERS - THREE-FACTOR DUPONT
# =============================================================================

@dataclass
class ThreeFactorDuPont:
    """
    Traditional 3-Factor DuPont decomposition for a single year.
    
    ROE = Net Profit Margin x Asset Turnover x Equity Multiplier
    """
    fiscal_year: str
    
    # Components (each with .value attribute)
    net_profit_margin: ComponentValue = field(default_factory=ComponentValue)
    asset_turnover: ComponentValue = field(default_factory=ComponentValue)
    equity_multiplier: ComponentValue = field(default_factory=ComponentValue)
    
    # Calculated ROE
    roe_calculated: Optional[float] = None
    roe_reported: Optional[float] = None
    
    # Reconciliation
    reconciliation_delta: Optional[float] = None
    is_reconciled: bool = False
    
    # Underlying data
    net_income: Optional[float] = None
    revenue: Optional[float] = None
    avg_assets: Optional[float] = None
    avg_equity: Optional[float] = None
    
    def calculate_roe(self) -> Optional[float]:
        """Calculate ROE as product of three factors."""
        npm = self.net_profit_margin.value
        at = self.asset_turnover.value
        em = self.equity_multiplier.value
        if all(v is not None for v in [npm, at, em]):
            return npm * at * em
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fiscal_year": self.fiscal_year,
            "net_profit_margin": self.net_profit_margin.to_dict(),
            "asset_turnover": self.asset_turnover.to_dict(),
            "equity_multiplier": self.equity_multiplier.to_dict(),
            "roe_calculated": self.roe_calculated,
            "roe_reported": self.roe_reported,
            "reconciliation_delta": self.reconciliation_delta,
            "is_reconciled": self.is_reconciled,
        }


# =============================================================================
# DATA CONTAINERS - FIVE-FACTOR DUPONT
# =============================================================================

@dataclass
class FiveFactorDuPont:
    """
    Extended 5-Factor DuPont decomposition for a single year.
    
    ROE = Tax Burden x Interest Burden x Operating Margin x Asset Turnover x Equity Multiplier
    """
    fiscal_year: str
    
    # Components (each with .value attribute)
    tax_burden: ComponentValue = field(default_factory=ComponentValue)
    interest_burden: ComponentValue = field(default_factory=ComponentValue)
    operating_margin: ComponentValue = field(default_factory=ComponentValue)
    asset_turnover: ComponentValue = field(default_factory=ComponentValue)
    equity_multiplier: ComponentValue = field(default_factory=ComponentValue)
    
    # Calculated ROE
    roe_calculated: Optional[float] = None
    roe_reported: Optional[float] = None
    
    # Derived metrics
    operational_roe: Optional[float] = None  # ROE without leverage effect
    leverage_contribution: Optional[float] = None  # Additional ROE from leverage
    
    # Reconciliation
    reconciliation_delta: Optional[float] = None
    is_reconciled: bool = False
    
    # Underlying data
    net_income: Optional[float] = None
    ebt: Optional[float] = None
    ebit: Optional[float] = None
    revenue: Optional[float] = None
    avg_assets: Optional[float] = None
    avg_equity: Optional[float] = None
    
    def calculate_roe(self) -> Optional[float]:
        """Calculate ROE as product of five factors."""
        components = [
            self.tax_burden.value,
            self.interest_burden.value,
            self.operating_margin.value,
            self.asset_turnover.value,
            self.equity_multiplier.value,
        ]
        if all(v is not None for v in components):
            return math.prod(components)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fiscal_year": self.fiscal_year,
            "tax_burden": self.tax_burden.to_dict(),
            "interest_burden": self.interest_burden.to_dict(),
            "operating_margin": self.operating_margin.to_dict(),
            "asset_turnover": self.asset_turnover.to_dict(),
            "equity_multiplier": self.equity_multiplier.to_dict(),
            "roe_calculated": self.roe_calculated,
            "operational_roe": self.operational_roe,
            "leverage_contribution": self.leverage_contribution,
            "is_reconciled": self.is_reconciled,
        }


# =============================================================================
# DATA CONTAINERS - ROA DECOMPOSITION
# =============================================================================

@dataclass
class ROADecomposition:
    """ROA decomposition for a single year."""
    fiscal_year: str
    net_profit_margin: ComponentValue = field(default_factory=ComponentValue)
    asset_turnover: ComponentValue = field(default_factory=ComponentValue)
    roa_calculated: Optional[float] = None
    roa_reported: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fiscal_year": self.fiscal_year,
            "net_profit_margin": self.net_profit_margin.to_dict(),
            "asset_turnover": self.asset_turnover.to_dict(),
            "roa_calculated": self.roa_calculated,
            "roa_reported": self.roa_reported,
        }


# =============================================================================
# DATA CONTAINERS - VARIANCE ANALYSIS
# =============================================================================

@dataclass
class VarianceComponent:
    """Variance attribution for a single component."""
    prior_value: Optional[float] = None
    current_value: Optional[float] = None
    change: Optional[float] = None
    contribution_to_roe_change: Optional[float] = None
    contribution_percentage: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prior_value": self.prior_value,
            "current_value": self.current_value,
            "change": self.change,
            "contribution_to_roe_change": self.contribution_to_roe_change,
            "contribution_percentage": self.contribution_percentage,
        }


@dataclass
class ROEVarianceAnalysis:
    """Year-over-year ROE variance attribution."""
    prior_year: str
    current_year: str
    
    # ROE values
    prior_roe: Optional[float] = None
    current_roe: Optional[float] = None
    roe_change: Optional[float] = None
    
    # Component contributions
    npm_contribution: VarianceComponent = field(default_factory=VarianceComponent)
    at_contribution: VarianceComponent = field(default_factory=VarianceComponent)
    em_contribution: VarianceComponent = field(default_factory=VarianceComponent)
    
    # Attribution summary
    total_explained: Optional[float] = None
    residual: Optional[float] = None
    is_reconciled: bool = False
    
    # Primary driver
    primary_driver: DuPontDriver = DuPontDriver.BALANCED
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prior_year": self.prior_year,
            "current_year": self.current_year,
            "prior_roe": self.prior_roe,
            "current_roe": self.current_roe,
            "roe_change": self.roe_change,
            "npm_contribution": self.npm_contribution.to_dict(),
            "at_contribution": self.at_contribution.to_dict(),
            "em_contribution": self.em_contribution.to_dict(),
            "total_explained": self.total_explained,
            "is_reconciled": self.is_reconciled,
            "primary_driver": self.primary_driver.value,
        }


# =============================================================================
# DATA CONTAINERS - QUALITY ASSESSMENT
# =============================================================================

@dataclass
class QualityAssessment:
    """ROE quality and sustainability assessment."""
    quality_rating: ROEQuality = ROEQuality.NOT_ASSESSED
    quality_score: float = 0.0  # 0-100
    
    # Qualitative assessments
    operational_strength: str = "not_assessed"
    leverage_dependency: str = "not_assessed"
    sustainability: str = "not_assessed"
    
    # Lists of findings
    strengths: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality_rating": self.quality_rating.value,
            "quality_score": self.quality_score,
            "operational_strength": self.operational_strength,
            "leverage_dependency": self.leverage_dependency,
            "sustainability": self.sustainability,
            "strengths": self.strengths,
            "concerns": self.concerns,
            "risks": self.risks,
        }


# =============================================================================
# MAIN RESULT CONTAINER
# =============================================================================

@dataclass
class DuPontAnalysisResult:
    """
    Complete output of Phase 4 DuPont analysis.
    
    Contains 3-factor and 5-factor decompositions, variance analysis,
    trend analysis, and quality assessment.
    """
    ticker: str
    company_name: str
    fiscal_periods: List[str]
    
    # Decompositions by year
    three_factor: Dict[str, ThreeFactorDuPont] = field(default_factory=dict)
    five_factor: Dict[str, FiveFactorDuPont] = field(default_factory=dict)
    roa_decomposition: Dict[str, ROADecomposition] = field(default_factory=dict)
    
    # Component trends
    component_trends: Dict[str, ComponentTrendInfo] = field(default_factory=dict)
    
    # Variance analysis (year-over-year)
    variance_analysis: List[ROEVarianceAnalysis] = field(default_factory=list)
    
    # ROE summary
    average_roe: Optional[float] = None
    roe_volatility: Optional[float] = None
    roe_trend: TrendDirection = TrendDirection.INSUFFICIENT_DATA
    primary_roe_driver: DuPontDriver = DuPontDriver.BALANCED
    
    # Quality assessment
    quality_assessment: Optional[QualityAssessment] = None
    
    # Metadata
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    data_source: str = "Phase 1/2/3 Pipeline"
    
    @property
    def is_valid(self) -> bool:
        """Check if analysis produced valid results."""
        return len(self.three_factor) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "fiscal_periods": self.fiscal_periods,
            "three_factor": {k: v.to_dict() for k, v in self.three_factor.items()},
            "five_factor": {k: v.to_dict() for k, v in self.five_factor.items()},
            "roa_decomposition": {k: v.to_dict() for k, v in self.roa_decomposition.items()},
            "component_trends": {k: v.to_dict() for k, v in self.component_trends.items()},
            "variance_analysis": [va.to_dict() for va in self.variance_analysis],
            "average_roe": self.average_roe,
            "roe_volatility": self.roe_volatility,
            "roe_trend": self.roe_trend.value,
            "primary_roe_driver": self.primary_roe_driver.value,
            "quality_assessment": self.quality_assessment.to_dict() if self.quality_assessment else None,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "data_source": self.data_source,
        }


# =============================================================================
# CALCULATOR CLASSES
# =============================================================================

class DuPontCalculator:
    """Calculator for DuPont decomposition components."""
    
    VERIFICATION_TOLERANCE = 0.02  # 2% tolerance for reconciliation
    
    def __init__(self):
        self.logger = LOGGER
    
    @staticmethod
    def safe_divide(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
        """Perform division with null and zero checks."""
        if numerator is None or denominator is None:
            return None
        if denominator == 0:
            return None
        return numerator / denominator
    
    @staticmethod
    def safe_average(value1: Optional[float], value2: Optional[float]) -> Optional[float]:
        """Compute average of two values with null handling."""
        if value1 is None and value2 is None:
            return None
        if value1 is None:
            return value2
        if value2 is None:
            return value1
        return (value1 + value2) / 2
    
    def compute_three_factor(
        self,
        net_income: Optional[float],
        revenue: Optional[float],
        avg_assets: Optional[float],
        avg_equity: Optional[float],
        fiscal_year: str
    ) -> ThreeFactorDuPont:
        """Compute 3-factor DuPont decomposition."""
        result = ThreeFactorDuPont(fiscal_year=fiscal_year)
        
        # Store underlying data
        result.net_income = net_income
        result.revenue = revenue
        result.avg_assets = avg_assets
        result.avg_equity = avg_equity
        
        # Net Profit Margin = Net Income / Revenue
        npm = self.safe_divide(net_income, revenue)
        result.net_profit_margin = ComponentValue(
            value=npm,
            numerator=net_income,
            denominator=revenue
        )
        
        # Asset Turnover = Revenue / Average Assets
        at = self.safe_divide(revenue, avg_assets)
        result.asset_turnover = ComponentValue(
            value=at,
            numerator=revenue,
            denominator=avg_assets
        )
        
        # Equity Multiplier = Average Assets / Average Equity
        em = self.safe_divide(avg_assets, avg_equity)
        result.equity_multiplier = ComponentValue(
            value=em,
            numerator=avg_assets,
            denominator=avg_equity
        )
        
        # Calculate ROE
        result.roe_calculated = result.calculate_roe()
        
        # Calculate reported ROE for verification
        result.roe_reported = self.safe_divide(net_income, avg_equity)
        
        # Reconciliation check
        if result.roe_calculated is not None and result.roe_reported is not None:
            result.reconciliation_delta = abs(result.roe_calculated - result.roe_reported)
            result.is_reconciled = result.reconciliation_delta < self.VERIFICATION_TOLERANCE
        
        return result
    
    def compute_five_factor(
        self,
        net_income: Optional[float],
        ebt: Optional[float],
        ebit: Optional[float],
        revenue: Optional[float],
        avg_assets: Optional[float],
        avg_equity: Optional[float],
        fiscal_year: str
    ) -> FiveFactorDuPont:
        """Compute 5-factor DuPont decomposition."""
        result = FiveFactorDuPont(fiscal_year=fiscal_year)
        
        # Store underlying data
        result.net_income = net_income
        result.ebt = ebt
        result.ebit = ebit
        result.revenue = revenue
        result.avg_assets = avg_assets
        result.avg_equity = avg_equity
        
        # Tax Burden = Net Income / EBT
        tb = self.safe_divide(net_income, ebt)
        result.tax_burden = ComponentValue(
            value=tb,
            numerator=net_income,
            denominator=ebt
        )
        
        # Interest Burden = EBT / EBIT
        ib = self.safe_divide(ebt, ebit)
        result.interest_burden = ComponentValue(
            value=ib,
            numerator=ebt,
            denominator=ebit
        )
        
        # Operating Margin = EBIT / Revenue
        om = self.safe_divide(ebit, revenue)
        result.operating_margin = ComponentValue(
            value=om,
            numerator=ebit,
            denominator=revenue
        )
        
        # Asset Turnover = Revenue / Average Assets
        at = self.safe_divide(revenue, avg_assets)
        result.asset_turnover = ComponentValue(
            value=at,
            numerator=revenue,
            denominator=avg_assets
        )
        
        # Equity Multiplier = Average Assets / Average Equity
        em = self.safe_divide(avg_assets, avg_equity)
        result.equity_multiplier = ComponentValue(
            value=em,
            numerator=avg_assets,
            denominator=avg_equity
        )
        
        # Calculate ROE
        result.roe_calculated = result.calculate_roe()
        
        # Calculate reported ROE for verification
        result.roe_reported = self.safe_divide(net_income, avg_equity)
        
        # Calculate operational ROE (without leverage) = Tax x Interest x OpMargin x AT
        if all(v is not None for v in [tb, ib, om, at]):
            result.operational_roe = tb * ib * om * at
        
        # Calculate leverage contribution
        if result.operational_roe is not None and result.roe_calculated is not None:
            result.leverage_contribution = result.roe_calculated - result.operational_roe
        
        # Reconciliation check
        if result.roe_calculated is not None and result.roe_reported is not None:
            result.reconciliation_delta = abs(result.roe_calculated - result.roe_reported)
            result.is_reconciled = result.reconciliation_delta < self.VERIFICATION_TOLERANCE
        
        return result
    
    def compute_roa_decomposition(
        self,
        net_income: Optional[float],
        revenue: Optional[float],
        avg_assets: Optional[float],
        fiscal_year: str
    ) -> ROADecomposition:
        """Compute ROA decomposition."""
        result = ROADecomposition(fiscal_year=fiscal_year)
        
        # Net Profit Margin
        npm = self.safe_divide(net_income, revenue)
        result.net_profit_margin = ComponentValue(
            value=npm,
            numerator=net_income,
            denominator=revenue
        )
        
        # Asset Turnover
        at = self.safe_divide(revenue, avg_assets)
        result.asset_turnover = ComponentValue(
            value=at,
            numerator=revenue,
            denominator=avg_assets
        )
        
        # Calculate ROA
        if npm is not None and at is not None:
            result.roa_calculated = npm * at
        
        # Reported ROA
        result.roa_reported = self.safe_divide(net_income, avg_assets)
        
        return result


class VarianceAnalyzer:
    """Analyzer for year-over-year variance attribution."""
    
    def __init__(self):
        self.logger = LOGGER
    
    def analyze_variance(
        self,
        prior: ThreeFactorDuPont,
        current: ThreeFactorDuPont
    ) -> ROEVarianceAnalysis:
        """
        Analyze ROE variance between two periods using the decomposition method.
        
        Uses the average contribution method for accurate attribution.
        """
        result = ROEVarianceAnalysis(
            prior_year=prior.fiscal_year,
            current_year=current.fiscal_year
        )
        
        # ROE values
        result.prior_roe = prior.roe_calculated
        result.current_roe = current.roe_calculated
        
        if result.prior_roe is not None and result.current_roe is not None:
            result.roe_change = result.current_roe - result.prior_roe
        
        # Get component values
        npm_prior = prior.net_profit_margin.value
        npm_current = current.net_profit_margin.value
        at_prior = prior.asset_turnover.value
        at_current = current.asset_turnover.value
        em_prior = prior.equity_multiplier.value
        em_current = current.equity_multiplier.value
        
        # Calculate contributions using logarithmic method
        if all(v is not None and v > 0 for v in [
            npm_prior, npm_current, at_prior, at_current, em_prior, em_current
        ]) and result.roe_change is not None:
            
            # Log changes
            log_npm_change = math.log(npm_current) - math.log(npm_prior)
            log_at_change = math.log(at_current) - math.log(at_prior)
            log_em_change = math.log(em_current) - math.log(em_prior)
            
            total_log_change = log_npm_change + log_at_change + log_em_change
            
            # Average ROE for scaling
            avg_roe = (result.prior_roe + result.current_roe) / 2
            
            # Scale contributions
            if total_log_change != 0:
                scale = result.roe_change / total_log_change
            else:
                scale = avg_roe
            
            npm_contrib = log_npm_change * scale
            at_contrib = log_at_change * scale
            em_contrib = log_em_change * scale
            
            # NPM contribution
            result.npm_contribution = VarianceComponent(
                prior_value=npm_prior,
                current_value=npm_current,
                change=npm_current - npm_prior,
                contribution_to_roe_change=npm_contrib,
                contribution_percentage=(npm_contrib / result.roe_change * 100) if result.roe_change != 0 else 0
            )
            
            # AT contribution
            result.at_contribution = VarianceComponent(
                prior_value=at_prior,
                current_value=at_current,
                change=at_current - at_prior,
                contribution_to_roe_change=at_contrib,
                contribution_percentage=(at_contrib / result.roe_change * 100) if result.roe_change != 0 else 0
            )
            
            # EM contribution
            result.em_contribution = VarianceComponent(
                prior_value=em_prior,
                current_value=em_current,
                change=em_current - em_prior,
                contribution_to_roe_change=em_contrib,
                contribution_percentage=(em_contrib / result.roe_change * 100) if result.roe_change != 0 else 0
            )
            
            # Total explained
            result.total_explained = npm_contrib + at_contrib + em_contrib
            result.residual = result.roe_change - result.total_explained
            result.is_reconciled = abs(result.residual) < 0.001
            
            # Determine primary driver
            contrib_abs = {
                DuPontDriver.PROFITABILITY: abs(npm_contrib),
                DuPontDriver.EFFICIENCY: abs(at_contrib),
                DuPontDriver.LEVERAGE: abs(em_contrib),
            }
            result.primary_driver = max(contrib_abs, key=contrib_abs.get)
        else:
            # Set empty contributions
            result.npm_contribution = VarianceComponent(
                prior_value=npm_prior, current_value=npm_current
            )
            result.at_contribution = VarianceComponent(
                prior_value=at_prior, current_value=at_current
            )
            result.em_contribution = VarianceComponent(
                prior_value=em_prior, current_value=em_current
            )
        
        return result


class TrendAnalyzer:
    """Analyzer for component trend analysis."""
    
    def __init__(self):
        self.logger = LOGGER
    
    def analyze_trend(
        self,
        values: List[float],
        years: List[str]
    ) -> ComponentTrendInfo:
        """Analyze trend for a component time series."""
        result = ComponentTrendInfo()
        
        if len(values) < 2:
            return result
        
        # Filter out None values
        valid_pairs = [(y, v) for y, v in zip(years, values) if v is not None]
        if len(valid_pairs) < 2:
            return result
        
        valid_values = [v for _, v in valid_pairs]
        
        # Basic statistics
        result.average = sum(valid_values) / len(valid_values)
        result.min_value = min(valid_values)
        result.max_value = max(valid_values)
        
        if result.average != 0:
            result.volatility = np.std(valid_values) / abs(result.average)
        
        # CAGR
        if valid_values[0] > 0 and valid_values[-1] > 0:
            periods = len(valid_values) - 1
            result.cagr = (valid_values[-1] / valid_values[0]) ** (1 / periods) - 1
        
        # Trend direction
        if result.volatility and result.volatility > 0.30:
            result.trend_direction = TrendDirection.VOLATILE
        else:
            # Linear regression slope
            x = np.arange(len(valid_values))
            slope = np.polyfit(x, valid_values, 1)[0]
            
            # Normalize slope
            normalized_slope = slope / abs(result.average) if result.average != 0 else 0
            
            if normalized_slope > 0.03:
                result.trend_direction = TrendDirection.IMPROVING
            elif normalized_slope > -0.03:
                result.trend_direction = TrendDirection.STABLE
            else:
                result.trend_direction = TrendDirection.DETERIORATING
        
        return result


class QualityAssessor:
    """Assessor for ROE quality and sustainability."""
    
    def __init__(self):
        self.logger = LOGGER
    
    def assess_quality(
        self,
        result: DuPontAnalysisResult
    ) -> QualityAssessment:
        """Assess ROE quality based on decomposition analysis."""
        qa = QualityAssessment()
        score = 100.0
        
        # Get latest year data
        if not result.fiscal_periods:
            return qa
        
        latest_year = result.fiscal_periods[0]
        three_factor = result.three_factor.get(latest_year)
        five_factor = result.five_factor.get(latest_year)
        
        if not three_factor:
            return qa
        
        # Evaluate leverage dependency
        em = three_factor.equity_multiplier.value
        npm = three_factor.net_profit_margin.value
        
        if em is not None:
            if em > 5.0:
                qa.leverage_dependency = "high"
                qa.concerns.append("High leverage (EM > 5x) amplifying ROE")
                qa.risks.append("Elevated financial risk from aggressive leverage")
                score -= 25
            elif em > 3.0:
                qa.leverage_dependency = "moderate"
                qa.concerns.append("Moderate leverage contributing to ROE")
                score -= 10
            else:
                qa.leverage_dependency = "low"
                qa.strengths.append("Conservative leverage providing stability")
        
        # Evaluate operational strength
        if npm is not None:
            if npm > 0.15:
                qa.operational_strength = "strong"
                qa.strengths.append("Strong profit margins indicating pricing power")
            elif npm > 0.05:
                qa.operational_strength = "moderate"
            else:
                qa.operational_strength = "weak"
                qa.concerns.append("Weak profit margins limiting ROE quality")
                score -= 15
        
        # Evaluate sustainability based on five-factor
        if five_factor:
            tb = five_factor.tax_burden.value
            ib = five_factor.interest_burden.value
            
            if tb is not None and tb > 0.90:
                qa.concerns.append("Very low effective tax rate may not be sustainable")
                score -= 5
            
            if ib is not None and ib < 0.70:
                qa.risks.append("High interest burden indicates elevated debt costs")
                score -= 10
        
        # Determine overall quality rating
        if em and em > 4.0 and npm and npm < 0.10:
            qa.quality_rating = ROEQuality.LEVERAGE_DRIVEN
            qa.sustainability = "low - primarily leverage driven"
        elif score >= 80:
            qa.quality_rating = ROEQuality.HIGH_QUALITY
            qa.sustainability = "high - driven by operational excellence"
        elif score >= 60:
            qa.quality_rating = ROEQuality.MODERATE_QUALITY
            qa.sustainability = "moderate - mixed drivers"
        else:
            qa.quality_rating = ROEQuality.LOW_QUALITY
            qa.sustainability = "low - concerns identified"
        
        qa.quality_score = max(0, score)
        
        return qa


# =============================================================================
# MAIN ANALYZER CLASS
# =============================================================================

class Phase4Analyzer:
    """
    Main orchestrator for Phase 4 DuPont analysis.
    
    Coordinates decomposition, variance analysis, trend analysis,
    and quality assessment to produce comprehensive ROE analysis.
    
    Usage:
        analyzer = Phase4Analyzer()
        result = analyzer.analyze(collection_result)
    """
    
    def __init__(self):
        self.calculator = DuPontCalculator()
        self.variance_analyzer = VarianceAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        self.quality_assessor = QualityAssessor()
        self.logger = LOGGER
    
    def analyze(self, collection_result: Any) -> DuPontAnalysisResult:
        """
        Perform complete DuPont analysis from CollectionResult.
        
        Args:
            collection_result: CollectionResult from Phase 1/2 pipeline
            
        Returns:
            DuPontAnalysisResult with complete decomposition
        """
        self.logger.info(f"Starting DuPont analysis for {collection_result.ticker}")
        
        # Extract DataFrames
        income_df = collection_result.statements.income_statement
        balance_df = collection_result.statements.balance_sheet
        fiscal_periods = collection_result.statements.fiscal_periods
        
        return self.analyze_from_dataframes(
            income_df=income_df,
            balance_df=balance_df,
            ticker=collection_result.ticker,
            company_name=collection_result.company_name,
            fiscal_periods=fiscal_periods,
        )
    
    def analyze_from_dataframes(
        self,
        income_df: pd.DataFrame,
        balance_df: pd.DataFrame,
        ticker: str,
        company_name: str,
        fiscal_periods: Optional[List[str]] = None
    ) -> DuPontAnalysisResult:
        """Perform complete DuPont analysis from DataFrames."""
        self.logger.info(f"Executing DuPont decomposition for {ticker}")
        
        # Determine fiscal periods
        if fiscal_periods is None:
            fiscal_periods = list(income_df.columns)
        
        # Sort periods (most recent first for result, but oldest first for analysis)
        sorted_periods = sorted(fiscal_periods, reverse=False)
        
        # Initialize result
        result = DuPontAnalysisResult(
            ticker=ticker,
            company_name=company_name,
            fiscal_periods=sorted(fiscal_periods, reverse=True),  # Most recent first
        )
        
        # Compute decompositions for each year
        roe_values = []
        
        for i, year in enumerate(sorted_periods):
            prev_year = sorted_periods[i - 1] if i > 0 else None
            
            # Extract financial data
            data = self._extract_data(income_df, balance_df, year, prev_year)
            
            # Compute 3-factor decomposition
            three_factor = self.calculator.compute_three_factor(
                net_income=data["net_income"],
                revenue=data["revenue"],
                avg_assets=data["avg_assets"],
                avg_equity=data["avg_equity"],
                fiscal_year=year
            )
            result.three_factor[year] = three_factor
            
            # Compute 5-factor decomposition
            five_factor = self.calculator.compute_five_factor(
                net_income=data["net_income"],
                ebt=data["ebt"],
                ebit=data["ebit"],
                revenue=data["revenue"],
                avg_assets=data["avg_assets"],
                avg_equity=data["avg_equity"],
                fiscal_year=year
            )
            result.five_factor[year] = five_factor
            
            # Compute ROA decomposition
            roa = self.calculator.compute_roa_decomposition(
                net_income=data["net_income"],
                revenue=data["revenue"],
                avg_assets=data["avg_assets"],
                fiscal_year=year
            )
            result.roa_decomposition[year] = roa
            
            # Collect ROE values
            if three_factor.roe_calculated is not None:
                roe_values.append(three_factor.roe_calculated)
        
        # Perform variance analysis
        for i in range(1, len(sorted_periods)):
            prior_year = sorted_periods[i - 1]
            current_year = sorted_periods[i]
            
            if prior_year in result.three_factor and current_year in result.three_factor:
                va = self.variance_analyzer.analyze_variance(
                    result.three_factor[prior_year],
                    result.three_factor[current_year]
                )
                result.variance_analysis.append(va)
        
        # Reverse variance analysis to match fiscal_periods order (most recent first)
        result.variance_analysis = list(reversed(result.variance_analysis))
        
        # Compute component trends
        self._compute_trends(result, sorted_periods)
        
        # Compute ROE summary statistics
        if roe_values:
            result.average_roe = sum(roe_values) / len(roe_values)
            if len(roe_values) > 1:
                result.roe_volatility = np.std(roe_values) / abs(result.average_roe) if result.average_roe != 0 else 0
            
            # ROE trend
            roe_trend = self.trend_analyzer.analyze_trend(roe_values, sorted_periods)
            result.roe_trend = roe_trend.trend_direction
        
        # Determine primary ROE driver
        result.primary_roe_driver = self._determine_primary_driver(result)
        
        # Quality assessment
        result.quality_assessment = self.quality_assessor.assess_quality(result)
        
        self.logger.info(
            f"DuPont analysis complete: {len(result.three_factor)}/5 components, "
            f"{len(result.variance_analysis)} contribution periods, "
            f"status={'verified' if all(tf.is_reconciled for tf in result.three_factor.values()) else 'partial'}"
        )
        
        return result
    
    def _extract_data(
        self,
        income_df: pd.DataFrame,
        balance_df: pd.DataFrame,
        year: str,
        prev_year: Optional[str]
    ) -> Dict[str, Optional[float]]:
        """Extract financial data for DuPont calculations."""
        data = {}
        
        # Income statement items
        data["net_income"] = self._get_value(income_df, "net_income", year)
        data["ebt"] = self._get_value(income_df, "pretax_income", year)
        data["ebit"] = self._get_value(income_df, "ebit", year)
        data["revenue"] = self._get_value(income_df, "total_revenue", year)
        
        # Use operating_income if ebit not available
        if data["ebit"] is None:
            data["ebit"] = self._get_value(income_df, "operating_income", year)
        
        # Balance sheet items
        assets_current = self._get_value(balance_df, "total_assets", year)
        equity_current = self._get_value(balance_df, "total_equity", year)
        
        if prev_year:
            assets_prev = self._get_value(balance_df, "total_assets", prev_year)
            equity_prev = self._get_value(balance_df, "total_equity", prev_year)
            data["avg_assets"] = self.calculator.safe_average(assets_current, assets_prev)
            data["avg_equity"] = self.calculator.safe_average(equity_current, equity_prev)
        else:
            data["avg_assets"] = assets_current
            data["avg_equity"] = equity_current
        
        return data
    
    def _get_value(
        self,
        df: pd.DataFrame,
        field: str,
        year: str
    ) -> Optional[float]:
        """Safely extract value from DataFrame."""
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
    
    def _compute_trends(self, result: DuPontAnalysisResult, sorted_periods: List[str]):
        """Compute trends for all components."""
        # Extract component values in chronological order
        npm_values = []
        at_values = []
        em_values = []
        tb_values = []
        ib_values = []
        om_values = []
        
        for year in sorted_periods:
            tf = result.three_factor.get(year)
            ff = result.five_factor.get(year)
            
            if tf:
                npm_values.append(tf.net_profit_margin.value)
                at_values.append(tf.asset_turnover.value)
                em_values.append(tf.equity_multiplier.value)
            
            if ff:
                tb_values.append(ff.tax_burden.value)
                ib_values.append(ff.interest_burden.value)
                om_values.append(ff.operating_margin.value)
        
        # Compute trends
        result.component_trends["Net Profit Margin"] = self.trend_analyzer.analyze_trend(npm_values, sorted_periods)
        result.component_trends["Asset Turnover"] = self.trend_analyzer.analyze_trend(at_values, sorted_periods)
        result.component_trends["Equity Multiplier"] = self.trend_analyzer.analyze_trend(em_values, sorted_periods)
        result.component_trends["Tax Burden"] = self.trend_analyzer.analyze_trend(tb_values, sorted_periods)
        result.component_trends["Interest Burden"] = self.trend_analyzer.analyze_trend(ib_values, sorted_periods)
        result.component_trends["Operating Margin"] = self.trend_analyzer.analyze_trend(om_values, sorted_periods)
        
        # ROE trend
        roe_values = [result.three_factor[y].roe_calculated for y in sorted_periods if y in result.three_factor]
        result.component_trends["ROE"] = self.trend_analyzer.analyze_trend(roe_values, sorted_periods)
    
    def _determine_primary_driver(self, result: DuPontAnalysisResult) -> DuPontDriver:
        """Determine primary ROE driver based on component contributions."""
        if not result.variance_analysis:
            return DuPontDriver.BALANCED
        
        # Count driver occurrences
        driver_counts = {d: 0 for d in DuPontDriver}
        for va in result.variance_analysis:
            driver_counts[va.primary_driver] += 1
        
        # Return most common driver
        return max(driver_counts, key=driver_counts.get)
    
    def save_report(
        self,
        result: DuPontAnalysisResult,
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Save DuPont analysis report to JSON file.
        
        Creates a ticker-specific subdirectory and saves the complete
        analysis result in a format consistent with other phase outputs.
        
        Args:
            result: DuPontAnalysisResult from analysis
            output_dir: Optional output directory (defaults to OUTPUT_DIR)
            
        Returns:
            Path to saved JSON file
        """
        if output_dir is None:
            output_dir = OUTPUT_DIR
        
        # Create ticker-specific directory
        ticker_dir = output_dir / result.ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = ticker_dir / f"{result.ticker}_dupont_analysis.json"
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"Saved DuPont analysis to {filepath}")
        return filepath
    
    def export_to_json(
        self,
        result: DuPontAnalysisResult,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Export analysis result to JSON file (legacy method).
        
        Note: Prefer save_report() for consistency with other phases.
        This method is retained for backward compatibility.
        
        Args:
            result: DuPontAnalysisResult from analysis
            output_path: Optional full path for output file
            
        Returns:
            Path to saved JSON file
        """
        if output_path is None:
            output_path = OUTPUT_DIR / f"{result.ticker}_dupont_analysis.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"Exported DuPont analysis to {output_path}")
        return output_path


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def analyze_dupont(
    collection_result: Any = None,
    income_df: pd.DataFrame = None,
    balance_df: pd.DataFrame = None,
    ticker: str = None,
    company_name: str = None,
    fiscal_periods: List[str] = None,
) -> DuPontAnalysisResult:
    """
    Convenience function for DuPont analysis.
    
    Can be called with either a CollectionResult or individual DataFrames.
    """
    analyzer = Phase4Analyzer()
    
    if collection_result is not None:
        return analyzer.analyze(collection_result)
    elif income_df is not None and balance_df is not None:
        return analyzer.analyze_from_dataframes(
            income_df=income_df,
            balance_df=balance_df,
            ticker=ticker or "UNKNOWN",
            company_name=company_name or "Unknown Company",
            fiscal_periods=fiscal_periods,
        )
    else:
        raise ValueError(
            "Must provide either collection_result or (income_df, balance_df, ticker, company_name)"
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Version
    "__version__",
    
    # Enumerations
    "DuPontDriver",
    "ROEQuality",
    "TrendDirection",
    "DuPontComponent",
    "ComponentTrend",
    
    # Data Containers
    "ComponentValue",
    "ComponentTrendInfo",
    "ThreeFactorDuPont",
    "FiveFactorDuPont",
    "ROADecomposition",
    "VarianceComponent",
    "ROEVarianceAnalysis",
    "QualityAssessment",
    "DuPontAnalysisResult",
    
    # Analyzers
    "DuPontCalculator",
    "VarianceAnalyzer",
    "TrendAnalyzer",
    "QualityAssessor",
    "Phase4Analyzer",
    
    # Convenience Function
    "analyze_dupont",
]