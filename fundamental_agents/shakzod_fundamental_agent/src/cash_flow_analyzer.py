"""
Cash Flow Analyzer Module for Fundamental Analyst Agent

This module performs comprehensive cash flow analysis, tracing the conversion
of accrual-based profits (Net Income) to actual cash generation (Operating
Cash Flow) and ultimately to Free Cash Flow. This analysis is fundamental to
the "Profitability vs Cash Drivers" thesis - assessing whether reported profits
convert to real, distributable cash.

The cash flow bridge is critical for understanding:
    1. Whether earnings are "real" (backed by cash) or "paper" (accrual-based)
    2. The sustainability of profitability (cash-generating businesses are more durable)
    3. Capital allocation capacity (only FCF can fund dividends, buybacks, M&A)
    4. Working capital efficiency (cash tied up vs released from operations)
    5. Capital intensity (CapEx requirements for growth)

Cash Flow Bridge Structure:
    Net Income (Accrual Profit)
      + Depreciation and Amortization    [Non-cash expense add-back]
      + Stock-Based Compensation         [Non-cash expense add-back]
      + Deferred Tax                     [Timing difference]
      + Change in Working Capital        [Cash tied up/released by operations]
      + Other Non-Cash Items             [Other adjustments]
    = Operating Cash Flow (OCF)
      - Capital Expenditure              [Maintenance + Growth investment]
    = Free Cash Flow (FCF)

Key Metrics:
    Cash Conversion Rate = OCF / Net Income
    FCF Margin = FCF / Revenue
    CapEx Intensity = CapEx / Revenue
    CapEx / D&A Ratio

MSc Coursework: AI Agents in Asset Management
Track A: Fundamental Analyst Agent

Author: MSc AI Agents in Asset Management
Version: 1.0.0
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
    EARNINGS_QUALITY,
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

DEFAULT_EXCELLENT_THRESHOLD = 1.10
DEFAULT_GOOD_THRESHOLD = 0.90
DEFAULT_ACCEPTABLE_THRESHOLD = 0.70
DEFAULT_POOR_THRESHOLD = 0.50

CAPEX_GROWTH_THRESHOLD = 1.30
CAPEX_MAINTENANCE_THRESHOLD = 0.80
CAPEX_ASSET_LIGHT_THRESHOLD = 0.02


# =============================================================================
# ENUMERATIONS
# =============================================================================

class CashFlowComponent(Enum):
    """Enumeration of cash flow bridge components."""
    NET_INCOME = "net_income"
    DEPRECIATION = "depreciation"
    STOCK_COMPENSATION = "stock_compensation"
    DEFERRED_TAX = "deferred_tax"
    WORKING_CAPITAL = "working_capital"
    OTHER_OPERATING = "other_operating"
    OPERATING_CASH_FLOW = "operating_cash_flow"
    CAPEX = "capex"
    FREE_CASH_FLOW = "free_cash_flow"


class CashConversionQuality(Enum):
    """Quality rating for cash conversion."""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    ACCEPTABLE = "Acceptable"
    CONCERNING = "Concerning"
    POOR = "Poor"


class CapExProfile(Enum):
    """Capital expenditure profile assessment."""
    GROWTH_INVESTMENT = "Growth Investment"
    MAINTENANCE = "Maintenance"
    UNDERINVESTMENT = "Underinvestment"
    ASSET_LIGHT = "Asset Light"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BridgeItem:
    """A single item in the cash flow bridge."""
    name: str
    component: CashFlowComponent
    amount: float
    as_percent_of_ni: Optional[float] = None
    description: str = ""
    is_subtotal: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "component": self.component.value,
            "amount": self.amount,
            "as_percent_of_ni": self.as_percent_of_ni,
            "description": self.description,
            "is_subtotal": self.is_subtotal
        }


@dataclass
class CashFlowBridge:
    """Complete Net Income to Free Cash Flow bridge."""
    period: str
    net_income: float
    items: List[BridgeItem]
    operating_cash_flow: float
    capital_expenditure: float
    free_cash_flow: float
    ocf_calculated: float
    ocf_reconciliation_diff: float
    is_reconciled: bool
    validation_status: ValidationStatus
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "period": self.period,
            "net_income": self.net_income,
            "items": [item.to_dict() for item in self.items],
            "operating_cash_flow": self.operating_cash_flow,
            "capital_expenditure": self.capital_expenditure,
            "free_cash_flow": self.free_cash_flow,
            "ocf_calculated": self.ocf_calculated,
            "ocf_reconciliation_diff": self.ocf_reconciliation_diff,
            "is_reconciled": self.is_reconciled,
            "validation_status": self.validation_status.value
        }


@dataclass
class CashConversionMetrics:
    """Metrics for assessing cash conversion quality."""
    cash_conversion_rate: float
    fcf_conversion_rate: float
    fcf_margin: float
    quality_rating: CashConversionQuality
    quality_description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cash_conversion_rate": self.cash_conversion_rate,
            "fcf_conversion_rate": self.fcf_conversion_rate,
            "fcf_margin": self.fcf_margin,
            "quality_rating": self.quality_rating.value,
            "quality_description": self.quality_description
        }


@dataclass
class CapExAnalysis:
    """Analysis of capital expenditure patterns."""
    capex_amount: float
    capex_to_revenue: float
    capex_to_depreciation: float
    capex_profile: CapExProfile
    maintenance_capex_estimate: float
    growth_capex_estimate: float
    profile_description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "capex_amount": self.capex_amount,
            "capex_to_revenue": self.capex_to_revenue,
            "capex_to_depreciation": self.capex_to_depreciation,
            "capex_profile": self.capex_profile.value,
            "maintenance_capex_estimate": self.maintenance_capex_estimate,
            "growth_capex_estimate": self.growth_capex_estimate,
            "profile_description": self.profile_description
        }


@dataclass
class WorkingCapitalCashImpact:
    """Working capital's impact on cash flow."""
    total_wc_change: float
    receivables_change: float
    inventory_change: float
    payables_change: float
    cash_impact_description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_wc_change": self.total_wc_change,
            "receivables_change": self.receivables_change,
            "inventory_change": self.inventory_change,
            "payables_change": self.payables_change,
            "cash_impact_description": self.cash_impact_description
        }


@dataclass
class CashFlowAnalysisResult:
    """Complete result of cash flow analysis."""
    ticker: str
    analysis_period: str
    bridge: CashFlowBridge
    conversion_metrics: CashConversionMetrics
    capex_analysis: CapExAnalysis
    working_capital_impact: WorkingCapitalCashImpact
    insights: List[str]
    red_flags: List[str]
    warnings: List[str]
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ticker": self.ticker,
            "analysis_period": self.analysis_period,
            "bridge": self.bridge.to_dict(),
            "conversion_metrics": self.conversion_metrics.to_dict(),
            "capex_analysis": self.capex_analysis.to_dict(),
            "working_capital_impact": self.working_capital_impact.to_dict(),
            "insights": self.insights,
            "red_flags": self.red_flags,
            "warnings": self.warnings,
            "analysis_timestamp": self.analysis_timestamp.isoformat()
        }


# =============================================================================
# CASH FLOW ANALYZER CLASS
# =============================================================================

class CashFlowAnalyzer:
    """
    Comprehensive cash flow analysis with profit-to-cash conversion assessment.
    
    This analyzer builds the cash flow bridge from Net Income to Operating
    Cash Flow to Free Cash Flow, assesses cash conversion quality, analyzes
    capital expenditure patterns, and identifies potential red flags.
    """
    
    def __init__(self, processed_data: ProcessedData) -> None:
        """
        Initialize the CashFlowAnalyzer.
        
        Args:
            processed_data: ProcessedData object from DataProcessor
        """
        self._data = processed_data
        self._processor = DataProcessor()
        self._warnings: List[str] = []
        self._red_flags: List[str] = []
        
        logger.info(f"CashFlowAnalyzer initialized for {processed_data.ticker}")
    
    def analyze(self) -> CashFlowAnalysisResult:
        """
        Perform complete cash flow analysis.
        
        Returns:
            CashFlowAnalysisResult with complete analysis
        """
        logger.info(f"Starting cash flow analysis for {self._data.ticker}")
        
        self._warnings = []
        self._red_flags = []
        
        periods = self._data.statements.periods
        analysis_period = self._format_analysis_period(periods)
        
        bridge = self._build_cash_flow_bridge()
        conversion_metrics = self._calculate_conversion_metrics(bridge)
        capex_analysis = self._analyze_capex(bridge)
        wc_impact = self._analyze_working_capital_impact()
        insights = self._generate_insights(bridge, conversion_metrics, capex_analysis)
        self._identify_red_flags(bridge, conversion_metrics, capex_analysis)
        
        result = CashFlowAnalysisResult(
            ticker=self._data.ticker,
            analysis_period=analysis_period,
            bridge=bridge,
            conversion_metrics=conversion_metrics,
            capex_analysis=capex_analysis,
            working_capital_impact=wc_impact,
            insights=insights,
            red_flags=self._red_flags.copy(),
            warnings=self._warnings.copy()
        )
        
        logger.info(f"Cash flow analysis complete for {self._data.ticker}")
        return result
    
    def _format_analysis_period(self, periods: List[str]) -> str:
        """Format the analysis period string."""
        if not periods:
            return "Current"
        
        period = periods[0]
        if hasattr(period, 'strftime'):
            return f"FY{period.strftime('%Y')}"
        else:
            return f"FY{period}"
    
    def _get_period_label(self) -> str:
        """Get the period label for the current analysis period."""
        periods = self._data.statements.periods
        if not periods:
            return "Current"
        
        period = periods[0]
        if hasattr(period, 'strftime'):
            return period.strftime('%Y')
        else:
            return str(period)
    
    def _build_cash_flow_bridge(self) -> CashFlowBridge:
        """Build the Net Income to Free Cash Flow bridge."""
        logger.debug("Building cash flow bridge")
        
        period_label = self._get_period_label()
        
        net_income = self._get_field(StandardField.NET_INCOME) or 0.0
        depreciation = self._get_field(StandardField.DEPRECIATION_CF) or 0.0
        stock_comp = self._get_field(StandardField.STOCK_COMP) or 0.0
        deferred_tax = self._get_field(StandardField.DEFERRED_TAX) or 0.0
        wc_change = self._get_field(StandardField.CHANGE_IN_WORKING_CAPITAL) or 0.0
        other_operating = self._get_field(StandardField.OTHER_OPERATING) or 0.0
        ocf_reported = self._get_field(StandardField.OPERATING_CASH_FLOW) or 0.0
        capex_raw = self._get_field(StandardField.CAPEX) or 0.0
        
        capex = -abs(capex_raw) if capex_raw != 0 else 0.0
        fcf = ocf_reported + capex
        
        fcf_from_data = self._get_field(StandardField.FREE_CASH_FLOW)
        if fcf_from_data is not None and abs(fcf_from_data - fcf) > 1.0:
            self._add_warning(
                f"FCF calculation ({fcf:.1f}M) differs from reported ({fcf_from_data:.1f}M)"
            )
        
        items = self._create_bridge_items(
            net_income, depreciation, stock_comp, deferred_tax,
            wc_change, other_operating, ocf_reported, capex, fcf
        )
        
        ocf_calculated = (
            net_income + depreciation + stock_comp +
            deferred_tax + wc_change + other_operating
        )
        
        ocf_diff = ocf_calculated - ocf_reported
        tolerance = max(
            abs(ocf_reported) * VALIDATION.ocf_bridge_tolerance_pct,
            100.0
        )
        is_reconciled = abs(ocf_diff) <= tolerance
        
        if not is_reconciled:
            validation_status = ValidationStatus.WARNING
            self._add_warning(
                f"OCF bridge has unexplained difference of ${ocf_diff:.1f}M"
            )
        else:
            validation_status = ValidationStatus.PASSED
        
        return CashFlowBridge(
            period=period_label,
            net_income=net_income,
            items=items,
            operating_cash_flow=ocf_reported,
            capital_expenditure=capex,
            free_cash_flow=fcf,
            ocf_calculated=ocf_calculated,
            ocf_reconciliation_diff=ocf_diff,
            is_reconciled=is_reconciled,
            validation_status=validation_status
        )
    
    def _create_bridge_items(
        self,
        net_income: float,
        depreciation: float,
        stock_comp: float,
        deferred_tax: float,
        wc_change: float,
        other_operating: float,
        ocf_reported: float,
        capex: float,
        fcf: float
    ) -> List[BridgeItem]:
        """Create the list of bridge items for the cash flow bridge."""
        items: List[BridgeItem] = []
        
        def pct_of_ni(amount: float) -> Optional[float]:
            if net_income != 0:
                return (amount / net_income) * 100
            return None
        
        items.append(BridgeItem(
            name="Net Income",
            component=CashFlowComponent.NET_INCOME,
            amount=net_income,
            as_percent_of_ni=100.0 if net_income != 0 else None,
            description="Starting point - accrual profit",
            is_subtotal=True
        ))
        
        items.append(BridgeItem(
            name="Depreciation and Amortization",
            component=CashFlowComponent.DEPRECIATION,
            amount=depreciation,
            as_percent_of_ni=pct_of_ni(depreciation),
            description="Non-cash expense added back"
        ))
        
        items.append(BridgeItem(
            name="Stock-Based Compensation",
            component=CashFlowComponent.STOCK_COMPENSATION,
            amount=stock_comp,
            as_percent_of_ni=pct_of_ni(stock_comp),
            description="Non-cash compensation expense"
        ))
        
        items.append(BridgeItem(
            name="Deferred Tax",
            component=CashFlowComponent.DEFERRED_TAX,
            amount=deferred_tax,
            as_percent_of_ni=pct_of_ni(deferred_tax),
            description="Tax timing differences"
        ))
        
        items.append(BridgeItem(
            name="Change in Working Capital",
            component=CashFlowComponent.WORKING_CAPITAL,
            amount=wc_change,
            as_percent_of_ni=pct_of_ni(wc_change),
            description="Cash tied up in (-) or released from (+) operations"
        ))
        
        items.append(BridgeItem(
            name="Other Operating Items",
            component=CashFlowComponent.OTHER_OPERATING,
            amount=other_operating,
            as_percent_of_ni=pct_of_ni(other_operating),
            description="Other non-cash items and adjustments"
        ))
        
        items.append(BridgeItem(
            name="Operating Cash Flow",
            component=CashFlowComponent.OPERATING_CASH_FLOW,
            amount=ocf_reported,
            as_percent_of_ni=pct_of_ni(ocf_reported),
            description="Cash generated from operations",
            is_subtotal=True
        ))
        
        items.append(BridgeItem(
            name="Capital Expenditure",
            component=CashFlowComponent.CAPEX,
            amount=capex,
            as_percent_of_ni=pct_of_ni(capex),
            description="Investment in property, plant and equipment"
        ))
        
        items.append(BridgeItem(
            name="Free Cash Flow",
            component=CashFlowComponent.FREE_CASH_FLOW,
            amount=fcf,
            as_percent_of_ni=pct_of_ni(fcf),
            description="Cash available for distribution",
            is_subtotal=True
        ))
        
        return items
    
    def _calculate_conversion_metrics(self, bridge: CashFlowBridge) -> CashConversionMetrics:
        """Calculate cash conversion quality metrics."""
        ni = bridge.net_income
        ocf = bridge.operating_cash_flow
        fcf = bridge.free_cash_flow
        
        revenue = self._get_field(StandardField.REVENUE) or 0.0
        
        cash_conversion_rate = self._calculate_safe_ratio(ocf, ni, "cash conversion")
        fcf_conversion_rate = self._calculate_safe_ratio(fcf, ni, "FCF conversion")
        
        if ni < 0 and ocf > 0:
            self._add_warning("Positive operating cash flow despite net loss")
            cash_conversion_rate = 0.0
            fcf_conversion_rate = 0.0
        
        fcf_margin = fcf / revenue if revenue > 0 else 0.0
        
        quality_rating, quality_description = self._assess_conversion_quality(
            cash_conversion_rate, ni
        )
        
        return CashConversionMetrics(
            cash_conversion_rate=cash_conversion_rate,
            fcf_conversion_rate=fcf_conversion_rate,
            fcf_margin=fcf_margin,
            quality_rating=quality_rating,
            quality_description=quality_description
        )
    
    def _calculate_safe_ratio(
        self,
        numerator: float,
        denominator: float,
        ratio_name: str
    ) -> float:
        """Calculate a ratio safely, handling edge cases."""
        if denominator == 0:
            return 0.0
        
        ratio = numerator / denominator
        
        if ratio > 10.0:
            return 9.99
        if ratio < -10.0:
            return -9.99
        
        return ratio
    
    def _assess_conversion_quality(
        self,
        conversion_rate: float,
        net_income: float
    ) -> Tuple[CashConversionQuality, str]:
        """Assess cash conversion quality based on thresholds."""
        if net_income <= 0:
            return (
                CashConversionQuality.CONCERNING,
                "Net income is negative or zero - conversion rate not meaningful"
            )
        
        excellent = getattr(EARNINGS_QUALITY, 'cash_conversion_excellent', DEFAULT_EXCELLENT_THRESHOLD)
        good = getattr(EARNINGS_QUALITY, 'cash_conversion_good', DEFAULT_GOOD_THRESHOLD)
        acceptable = getattr(EARNINGS_QUALITY, 'cash_conversion_acceptable', DEFAULT_ACCEPTABLE_THRESHOLD)
        
        if conversion_rate >= excellent:
            return (
                CashConversionQuality.EXCELLENT,
                f"OCF exceeds net income by {(conversion_rate-1)*100:.0f}% - "
                f"high quality earnings backed by strong cash generation"
            )
        elif conversion_rate >= good:
            return (
                CashConversionQuality.GOOD,
                f"Cash conversion of {conversion_rate*100:.0f}% indicates "
                f"earnings are well-supported by operating cash flow"
            )
        elif conversion_rate >= acceptable:
            return (
                CashConversionQuality.ACCEPTABLE,
                f"Cash conversion of {conversion_rate*100:.0f}% is acceptable "
                f"but warrants monitoring for deterioration"
            )
        elif conversion_rate >= DEFAULT_POOR_THRESHOLD:
            return (
                CashConversionQuality.CONCERNING,
                f"Cash conversion of {conversion_rate*100:.0f}% suggests "
                f"significant gap between earnings and cash - investigate causes"
            )
        else:
            return (
                CashConversionQuality.POOR,
                f"Cash conversion of {conversion_rate*100:.0f}% indicates "
                f"earnings are not being backed by cash - red flag"
            )
    
    def _analyze_capex(self, bridge: CashFlowBridge) -> CapExAnalysis:
        """Analyze capital expenditure patterns."""
        capex = abs(bridge.capital_expenditure)
        
        revenue = self._get_field(StandardField.REVENUE) or 0.0
        depreciation = self._get_field(StandardField.DEPRECIATION_CF) or 0.0
        
        capex_to_revenue = capex / revenue if revenue > 0 else 0.0
        capex_to_depreciation = capex / depreciation if depreciation > 0 else 0.0
        
        maintenance_capex = min(capex, depreciation) if depreciation > 0 else capex
        growth_capex = max(0, capex - depreciation)
        
        profile, description = self._assess_capex_profile(
            capex_to_revenue, capex_to_depreciation
        )
        
        return CapExAnalysis(
            capex_amount=capex,
            capex_to_revenue=capex_to_revenue,
            capex_to_depreciation=capex_to_depreciation,
            capex_profile=profile,
            maintenance_capex_estimate=maintenance_capex,
            growth_capex_estimate=growth_capex,
            profile_description=description
        )
    
    def _assess_capex_profile(
        self,
        capex_to_revenue: float,
        capex_to_depreciation: float
    ) -> Tuple[CapExProfile, str]:
        """Assess the CapEx profile based on ratios."""
        if capex_to_revenue < CAPEX_ASSET_LIGHT_THRESHOLD:
            return (
                CapExProfile.ASSET_LIGHT,
                f"Very low CapEx intensity ({capex_to_revenue*100:.1f}% of revenue) "
                f"indicates asset-light business model"
            )
        elif capex_to_depreciation >= CAPEX_GROWTH_THRESHOLD:
            return (
                CapExProfile.GROWTH_INVESTMENT,
                f"CapEx ({capex_to_depreciation:.1f}x D&A) significantly exceeds depreciation, "
                f"indicating investment for growth beyond maintenance"
            )
        elif capex_to_depreciation >= CAPEX_MAINTENANCE_THRESHOLD:
            return (
                CapExProfile.MAINTENANCE,
                f"CapEx ({capex_to_depreciation:.1f}x D&A) roughly matches depreciation, "
                f"indicating maintenance-level investment"
            )
        else:
            self._add_warning("CapEx below depreciation may indicate underinvestment")
            return (
                CapExProfile.UNDERINVESTMENT,
                f"CapEx ({capex_to_depreciation:.1f}x D&A) below depreciation "
                f"may indicate underinvestment in asset base"
            )
    
    def _analyze_working_capital_impact(self) -> WorkingCapitalCashImpact:
        """Analyze working capital's impact on cash flow."""
        total_wc = self._get_field(StandardField.CHANGE_IN_WORKING_CAPITAL) or 0.0
        ar_change = self._get_field(StandardField.CHANGE_IN_RECEIVABLES) or 0.0
        inv_change = self._get_field(StandardField.CHANGE_IN_INVENTORY) or 0.0
        ap_change = self._get_field(StandardField.CHANGE_IN_PAYABLES) or 0.0
        
        if total_wc > 0:
            description = f"Working capital released ${total_wc:.1f}M of cash"
        elif total_wc < 0:
            description = f"Working capital consumed ${abs(total_wc):.1f}M of cash"
        else:
            description = "Working capital had minimal impact on cash flow"
        
        return WorkingCapitalCashImpact(
            total_wc_change=total_wc,
            receivables_change=ar_change,
            inventory_change=inv_change,
            payables_change=ap_change,
            cash_impact_description=description
        )
    
    def _generate_insights(
        self,
        bridge: CashFlowBridge,
        conversion: CashConversionMetrics,
        capex: CapExAnalysis
    ) -> List[str]:
        """Generate key insights from the analysis."""
        insights: List[str] = []
        
        if conversion.quality_rating in [CashConversionQuality.EXCELLENT, CashConversionQuality.GOOD]:
            insights.append(
                f"Strong cash conversion rate of {conversion.cash_conversion_rate*100:.0f}% "
                f"indicates high-quality earnings backed by real cash generation"
            )
        elif conversion.quality_rating == CashConversionQuality.POOR:
            insights.append(
                f"Weak cash conversion rate of {conversion.cash_conversion_rate*100:.0f}% "
                f"raises concerns about earnings quality and sustainability"
            )
        
        if conversion.fcf_margin > 0.15:
            insights.append(
                f"Strong FCF margin of {conversion.fcf_margin*100:.1f}% provides "
                f"significant capacity for dividends, buybacks, and debt reduction"
            )
        elif conversion.fcf_margin > 0.05:
            insights.append(
                f"Healthy FCF margin of {conversion.fcf_margin*100:.1f}% supports "
                f"moderate capital returns to shareholders"
            )
        elif conversion.fcf_margin > 0:
            insights.append(
                f"Modest FCF margin of {conversion.fcf_margin*100:.1f}% limits "
                f"flexibility for capital returns"
            )
        else:
            insights.append(
                f"Negative FCF indicates cash consumption - company requires "
                f"external financing for operations or investment"
            )
        
        if capex.capex_profile == CapExProfile.GROWTH_INVESTMENT:
            insights.append(
                f"CapEx of {capex.capex_to_depreciation:.1f}x depreciation signals "
                f"investment for growth, with ~${capex.growth_capex_estimate:.0f}M "
                f"above maintenance levels"
            )
        elif capex.capex_profile == CapExProfile.ASSET_LIGHT:
            insights.append(
                f"Asset-light model with CapEx at only {capex.capex_to_revenue*100:.1f}% "
                f"of revenue enables high cash conversion"
            )
        
        return insights
    
    def _identify_red_flags(
        self,
        bridge: CashFlowBridge,
        conversion: CashConversionMetrics,
        capex: CapExAnalysis
    ) -> None:
        """Identify potential red flags in cash flow."""
        if conversion.quality_rating == CashConversionQuality.POOR:
            self._red_flags.append(
                f"POOR CASH CONVERSION: OCF is only {conversion.cash_conversion_rate*100:.0f}% "
                f"of net income - earnings may not be backed by cash"
            )
        elif conversion.quality_rating == CashConversionQuality.CONCERNING:
            self._red_flags.append(
                f"CONCERNING CASH CONVERSION: OCF is {conversion.cash_conversion_rate*100:.0f}% "
                f"of net income - warrants investigation"
            )
        
        if bridge.free_cash_flow < 0 and bridge.net_income > 0:
            self._red_flags.append(
                f"NEGATIVE FCF DESPITE PROFIT: Company generated ${bridge.net_income:.0f}M "
                f"profit but consumed ${abs(bridge.free_cash_flow):.0f}M in cash"
            )
        
        if capex.capex_profile == CapExProfile.UNDERINVESTMENT:
            self._red_flags.append(
                f"POTENTIAL UNDERINVESTMENT: CapEx ({capex.capex_to_depreciation:.1f}x D&A) "
                f"below depreciation may impair future growth"
            )
        
        ni = bridge.net_income
        sbc = 0.0
        for item in bridge.items:
            if item.component == CashFlowComponent.STOCK_COMPENSATION:
                sbc = item.amount
                break
        
        if ni > 0 and sbc > 0:
            sbc_pct = sbc / ni
            if sbc_pct > 0.30:
                self._red_flags.append(
                    f"HIGH STOCK COMPENSATION: SBC at {sbc_pct*100:.0f}% of net income "
                    f"represents significant non-cash earnings component"
                )
        
        if not bridge.is_reconciled:
            self._red_flags.append(
                f"OCF RECONCILIATION GAP: ${bridge.ocf_reconciliation_diff:.0f}M unexplained "
                f"difference between calculated and reported OCF"
            )
    
    def _get_field(self, field: StandardField, period_index: int = 0) -> Optional[float]:
        """Extract a field value from processed data."""
        return self._processor.get_field(self._data, field, period_index)
    
    def _add_warning(self, message: str) -> None:
        """Add a warning message."""
        self._warnings.append(message)
        logger.warning(message)
    
    def format_bridge_table(self, bridge: CashFlowBridge) -> pd.DataFrame:
        """Format the cash flow bridge as a pandas DataFrame for display."""
        data = []
        for item in bridge.items:
            row = {
                'Component': item.name,
                'Amount ($M)': item.amount,
                '% of NI': item.as_percent_of_ni,
                'Description': item.description
            }
            data.append(row)
        return pd.DataFrame(data)
    
    def format_metrics_summary(self, result: CashFlowAnalysisResult) -> pd.DataFrame:
        """Format key metrics as a summary DataFrame."""
        conv = result.conversion_metrics
        capex = result.capex_analysis
        
        data = [
            {'Metric': 'Cash Conversion Rate', 'Value': f"{conv.cash_conversion_rate*100:.1f}%", 'Assessment': conv.quality_rating.value},
            {'Metric': 'FCF Conversion Rate', 'Value': f"{conv.fcf_conversion_rate*100:.1f}%", 'Assessment': ''},
            {'Metric': 'FCF Margin', 'Value': f"{conv.fcf_margin*100:.1f}%", 'Assessment': ''},
            {'Metric': 'CapEx / Revenue', 'Value': f"{capex.capex_to_revenue*100:.1f}%", 'Assessment': ''},
            {'Metric': 'CapEx / D&A', 'Value': f"{capex.capex_to_depreciation:.2f}x", 'Assessment': capex.capex_profile.value}
        ]
        return pd.DataFrame(data)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_cash_flow(processed_data: ProcessedData) -> CashFlowAnalysisResult:
    """Convenience function to perform cash flow analysis."""
    analyzer = CashFlowAnalyzer(processed_data)
    return analyzer.analyze()


def get_cash_flow_summary(result: CashFlowAnalysisResult) -> Dict[str, Any]:
    """Extract key cash flow metrics as a simple dictionary."""
    bridge = result.bridge
    conv = result.conversion_metrics
    capex = result.capex_analysis
    
    return {
        'net_income': bridge.net_income,
        'operating_cash_flow': bridge.operating_cash_flow,
        'free_cash_flow': bridge.free_cash_flow,
        'capital_expenditure': bridge.capital_expenditure,
        'cash_conversion_rate': conv.cash_conversion_rate,
        'fcf_conversion_rate': conv.fcf_conversion_rate,
        'fcf_margin': conv.fcf_margin,
        'capex_to_revenue': capex.capex_to_revenue,
        'capex_to_depreciation': capex.capex_to_depreciation,
        'quality_rating': conv.quality_rating.value
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
    print(f"CASH FLOW ANALYZER MODULE TEST - {test_ticker}")
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
        print(f"  Processing complete")
        print()
        
        print("Step 3: Analyzing cash flow...")
        analyzer = CashFlowAnalyzer(processed)
        result = analyzer.analyze()
        print(f"  Analysis complete")
        print()
        
        print("CASH FLOW BRIDGE")
        print("-" * 70)
        print(f"  Analysis Period: {result.analysis_period}")
        print(f"  Validation Status: {result.bridge.validation_status.value.upper()}")
        
        bridge = result.bridge
        print()
        print(f"  {'Component':<30} {'Amount ($M)':>12} {'% of NI':>10}")
        print(f"  {'-'*30} {'-'*12} {'-'*10}")
        
        for item in bridge.items:
            amount_str = f"${item.amount:,.1f}"
            pct_str = f"{item.as_percent_of_ni:.1f}%" if item.as_percent_of_ni else ""
            if item.is_subtotal:
                print(f"  {item.name:<30} {amount_str:>12} {pct_str:>10}")
            else:
                print(f"    {item.name:<28} {amount_str:>12} {pct_str:>10}")
        
        print()
        print("CASH CONVERSION METRICS")
        print("-" * 70)
        conv = result.conversion_metrics
        print(f"  Cash Conversion Rate:  {conv.cash_conversion_rate*100:.1f}%")
        print(f"  FCF Margin:            {conv.fcf_margin*100:.1f}%")
        print(f"  Quality Rating:        {conv.quality_rating.value}")
        
        print()
        print("=" * 70)
        print(f"Cash flow analysis complete for {test_ticker}")
        print("=" * 70)
        print()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)