"""
Accuracy Checker Module - Phase 8 Advanced Validation
Fundamental Analyst Agent

Implements institutional-grade accuracy verification and quality assurance
across all analysis phases before final investment memo generation.

Methodology:
    - Input Integrity Verification: Validate source data hasn't been corrupted
    - Calculation Recalculation: Re-verify key calculations from each phase
    - Cross-Phase Reconciliation: Ensure data consistency between phases
    - Valuation Consistency: Verify DCF, DDM, Multiples use identical inputs
    - Range Validation: Check all values fall within reasonable bounds
    - Methodology Compliance: Verify adherence to implementation plan
    - Confidence Scoring: Generate reliability scores for each component

Key Components:
    - InputIntegrityChecker: Validates Phase 1 data integrity
    - CalculationVerifier: Re-verifies key calculations from Phases 2-7
    - CrossPhaseReconciler: Validates data consistency across phases
    - ValuationConsistencyChecker: Ensures valuation models use same inputs
    - RangeValidator: Checks values are within reasonable bounds
    - ConfidenceScorer: Generates overall quality/confidence scores
    - AccuracyReportGenerator: Produces comprehensive accuracy report

Inputs:
    - CollectionResult from Phase 1
    - ValidationReport from Phase 2
    - RatioAnalysisResult from Phase 3
    - DuPontAnalysisResult from Phase 4
    - DCFValuationResult from Phase 5
    - DDMValuationResult from Phase 6
    - MultiplesValuationResult from Phase 7

Outputs: AccuracyCheckResult with verification status and confidence scores

MSc Coursework: IFTE0001 AI Agents in Asset Management
Track A: Fundamental Analyst Agent
Phase 8: Advanced Accuracy Verification

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

class AccuracyConfig:
    """Configuration parameters for accuracy checking."""
    
    # Tolerance thresholds for numerical comparisons
    TOLERANCE_STRICT: float = 0.001      # 0.1% for critical calculations
    TOLERANCE_NORMAL: float = 0.01       # 1% for standard calculations
    TOLERANCE_LOOSE: float = 0.05        # 5% for approximate comparisons
    
    # Range validation bounds
    PE_RATIO_MIN: float = 0.0
    PE_RATIO_MAX: float = 500.0
    PB_RATIO_MIN: float = 0.0
    PB_RATIO_MAX: float = 200.0
    MARGIN_MIN: float = -1.0             # -100%
    MARGIN_MAX: float = 1.0              # 100%
    GROWTH_RATE_MIN: float = -0.50       # -50%
    GROWTH_RATE_MAX: float = 1.00        # 100%
    WACC_MIN: float = 0.02               # 2%
    WACC_MAX: float = 0.25               # 25%
    BETA_MIN: float = 0.0
    BETA_MAX: float = 5.0
    TERMINAL_GROWTH_MAX: float = 0.05    # 5%
    INTRINSIC_VALUE_MIN: float = 0.0
    
    # Confidence scoring weights
    WEIGHT_INPUT_INTEGRITY: float = 0.15
    WEIGHT_CALCULATION_ACCURACY: float = 0.25
    WEIGHT_CROSS_PHASE_CONSISTENCY: float = 0.20
    WEIGHT_VALUATION_CONSISTENCY: float = 0.20
    WEIGHT_RANGE_VALIDITY: float = 0.10
    WEIGHT_METHODOLOGY_COMPLIANCE: float = 0.10
    
    # Minimum confidence thresholds
    MIN_CONFIDENCE_CRITICAL: float = 0.95
    MIN_CONFIDENCE_HIGH: float = 0.85
    MIN_CONFIDENCE_ACCEPTABLE: float = 0.70


# =============================================================================
# ENUMERATIONS
# =============================================================================

class VerificationStatus(Enum):
    """Status of a verification check."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


class ConfidenceLevel(Enum):
    """Overall confidence level."""
    VERY_HIGH = "very_high"         # >= 95%
    HIGH = "high"                   # >= 85%
    ACCEPTABLE = "acceptable"       # >= 70%
    LOW = "low"                     # >= 50%
    VERY_LOW = "very_low"           # < 50%


class CheckCategory(Enum):
    """Categories of accuracy checks."""
    INPUT_INTEGRITY = "input_integrity"
    CALCULATION_ACCURACY = "calculation_accuracy"
    CROSS_PHASE_CONSISTENCY = "cross_phase_consistency"
    VALUATION_CONSISTENCY = "valuation_consistency"
    RANGE_VALIDITY = "range_validity"
    METHODOLOGY_COMPLIANCE = "methodology_compliance"


# =============================================================================
# DATA CONTAINERS - INDIVIDUAL CHECK RESULT
# =============================================================================

@dataclass
class CheckResult:
    """Result of a single verification check."""
    
    check_name: str = ""
    category: CheckCategory = CheckCategory.INPUT_INTEGRITY
    status: VerificationStatus = VerificationStatus.SKIPPED
    
    expected_value: Optional[float] = None
    actual_value: Optional[float] = None
    tolerance: float = AccuracyConfig.TOLERANCE_NORMAL
    deviation: Optional[float] = None
    
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_name": self.check_name,
            "category": self.category.value,
            "status": self.status.value,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "tolerance": self.tolerance,
            "deviation": self.deviation,
            "message": self.message,
            "details": self.details,
        }


# =============================================================================
# DATA CONTAINERS - CATEGORY SUMMARY
# =============================================================================

@dataclass
class CategorySummary:
    """Summary of checks within a category."""
    
    category: CheckCategory = CheckCategory.INPUT_INTEGRITY
    total_checks: int = 0
    passed: int = 0
    warnings: int = 0
    failed: int = 0
    skipped: int = 0
    
    pass_rate: float = 0.0
    confidence_score: float = 0.0
    
    checks: List[CheckResult] = field(default_factory=list)
    
    def calculate_metrics(self):
        """Calculate pass rate and confidence score."""
        if self.total_checks > 0:
            self.pass_rate = self.passed / self.total_checks
            # Confidence: passed = 1.0, warning = 0.7, failed = 0.0, skipped = 0.5
            weighted_sum = (
                self.passed * 1.0 +
                self.warnings * 0.7 +
                self.failed * 0.0 +
                self.skipped * 0.5
            )
            self.confidence_score = weighted_sum / self.total_checks
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "total_checks": self.total_checks,
            "passed": self.passed,
            "warnings": self.warnings,
            "failed": self.failed,
            "skipped": self.skipped,
            "pass_rate": self.pass_rate,
            "confidence_score": self.confidence_score,
            "checks": [c.to_dict() for c in self.checks],
        }


# =============================================================================
# DATA CONTAINERS - VALUATION COMPARISON
# =============================================================================

@dataclass
class ValuationInputComparison:
    """Comparison of inputs across valuation models."""
    
    parameter: str = ""
    dcf_value: Optional[float] = None
    ddm_value: Optional[float] = None
    multiples_value: Optional[float] = None
    
    is_consistent: bool = True
    max_deviation: float = 0.0
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter": self.parameter,
            "dcf_value": self.dcf_value,
            "ddm_value": self.ddm_value,
            "multiples_value": self.multiples_value,
            "is_consistent": self.is_consistent,
            "max_deviation": self.max_deviation,
            "notes": self.notes,
        }


# =============================================================================
# DATA CONTAINERS - FINAL RESULT
# =============================================================================

@dataclass
class AccuracyCheckResult:
    """Complete Phase 8 Accuracy Check output."""
    
    # Identification
    ticker: str = ""
    company_name: str = ""
    check_timestamp: datetime = field(default_factory=datetime.now)
    
    # Category summaries
    input_integrity: CategorySummary = field(
        default_factory=lambda: CategorySummary(CheckCategory.INPUT_INTEGRITY)
    )
    calculation_accuracy: CategorySummary = field(
        default_factory=lambda: CategorySummary(CheckCategory.CALCULATION_ACCURACY)
    )
    cross_phase_consistency: CategorySummary = field(
        default_factory=lambda: CategorySummary(CheckCategory.CROSS_PHASE_CONSISTENCY)
    )
    valuation_consistency: CategorySummary = field(
        default_factory=lambda: CategorySummary(CheckCategory.VALUATION_CONSISTENCY)
    )
    range_validity: CategorySummary = field(
        default_factory=lambda: CategorySummary(CheckCategory.RANGE_VALIDITY)
    )
    methodology_compliance: CategorySummary = field(
        default_factory=lambda: CategorySummary(CheckCategory.METHODOLOGY_COMPLIANCE)
    )
    
    # Valuation input comparisons
    valuation_input_comparisons: List[ValuationInputComparison] = field(default_factory=list)
    
    # Overall metrics
    total_checks: int = 0
    total_passed: int = 0
    total_warnings: int = 0
    total_failed: int = 0
    
    overall_pass_rate: float = 0.0
    overall_confidence: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.ACCEPTABLE
    
    # Critical issues
    critical_issues: List[str] = field(default_factory=list)
    warnings_list: List[str] = field(default_factory=list)
    
    # Recommendation
    is_reliable: bool = False
    recommendation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "check_timestamp": self.check_timestamp.isoformat(),
            "input_integrity": self.input_integrity.to_dict(),
            "calculation_accuracy": self.calculation_accuracy.to_dict(),
            "cross_phase_consistency": self.cross_phase_consistency.to_dict(),
            "valuation_consistency": self.valuation_consistency.to_dict(),
            "range_validity": self.range_validity.to_dict(),
            "methodology_compliance": self.methodology_compliance.to_dict(),
            "valuation_input_comparisons": [v.to_dict() for v in self.valuation_input_comparisons],
            "total_checks": self.total_checks,
            "total_passed": self.total_passed,
            "total_warnings": self.total_warnings,
            "total_failed": self.total_failed,
            "overall_pass_rate": self.overall_pass_rate,
            "overall_confidence": self.overall_confidence,
            "confidence_level": self.confidence_level.value,
            "critical_issues": self.critical_issues,
            "warnings_list": self.warnings_list,
            "is_reliable": self.is_reliable,
            "recommendation": self.recommendation,
        }


# =============================================================================
# INPUT INTEGRITY CHECKER
# =============================================================================

class InputIntegrityChecker:
    """
    Verifies integrity of Phase 1 input data.
    
    Checks:
        - Required fields are present
        - Values are non-null where expected
        - Data types are correct
        - No obvious data corruption
    """
    
    def check(
        self,
        collection_result: Any,
    ) -> CategorySummary:
        """Run input integrity checks."""
        summary = CategorySummary(CheckCategory.INPUT_INTEGRITY)
        
        # Check 1: Company profile completeness
        check = self._check_profile_completeness(collection_result.company_profile)
        summary.checks.append(check)
        
        # Check 2: Financial statements availability
        check = self._check_statements_availability(collection_result.statements)
        summary.checks.append(check)
        
        # Check 3: Fiscal periods consistency
        check = self._check_fiscal_periods(collection_result.statements)
        summary.checks.append(check)
        
        # Check 4: Derived metrics availability
        check = self._check_derived_metrics(collection_result.derived_metrics)
        summary.checks.append(check)
        
        # Check 5: Market cap validity
        check = self._check_market_cap(collection_result.company_profile)
        summary.checks.append(check)
        
        # Check 6: Shares outstanding validity
        check = self._check_shares_outstanding(
            collection_result.company_profile,
            collection_result.statements
        )
        summary.checks.append(check)
        
        # Check 7: Revenue continuity
        check = self._check_revenue_continuity(collection_result.statements)
        summary.checks.append(check)
        
        # Check 8: Net income sign consistency
        check = self._check_net_income_signs(collection_result.statements)
        summary.checks.append(check)
        
        # Calculate summary metrics
        self._calculate_summary(summary)
        
        return summary
    
    def _check_profile_completeness(self, profile: Any) -> CheckResult:
        """Check company profile has required fields."""
        check = CheckResult(
            check_name="profile_completeness",
            category=CheckCategory.INPUT_INTEGRITY,
        )
        
        required_fields = ['market_cap', 'beta', 'pe_ratio', 'roe']
        missing = []
        
        for field in required_fields:
            value = getattr(profile, field, None)
            if value is None:
                missing.append(field)
        
        if not missing:
            check.status = VerificationStatus.PASSED
            check.message = "All required profile fields present"
        elif len(missing) <= 1:
            check.status = VerificationStatus.WARNING
            check.message = f"Missing profile field: {missing[0]}"
        else:
            check.status = VerificationStatus.FAILED
            check.message = f"Missing profile fields: {', '.join(missing)}"
        
        check.details = {"missing_fields": missing}
        return check
    
    def _check_statements_availability(self, statements: Any) -> CheckResult:
        """Check all three statements are available."""
        check = CheckResult(
            check_name="statements_availability",
            category=CheckCategory.INPUT_INTEGRITY,
        )
        
        available = []
        missing = []
        
        if statements.income_statement is not None and not statements.income_statement.empty:
            available.append("income_statement")
        else:
            missing.append("income_statement")
        
        if statements.balance_sheet is not None and not statements.balance_sheet.empty:
            available.append("balance_sheet")
        else:
            missing.append("balance_sheet")
        
        if statements.cash_flow is not None and not statements.cash_flow.empty:
            available.append("cash_flow")
        else:
            missing.append("cash_flow")
        
        if not missing:
            check.status = VerificationStatus.PASSED
            check.message = "All three financial statements available"
        else:
            check.status = VerificationStatus.FAILED
            check.message = f"Missing statements: {', '.join(missing)}"
        
        check.details = {"available": available, "missing": missing}
        return check
    
    def _check_fiscal_periods(self, statements: Any) -> CheckResult:
        """Check fiscal periods are consistent and sufficient."""
        check = CheckResult(
            check_name="fiscal_periods",
            category=CheckCategory.INPUT_INTEGRITY,
        )
        
        periods = statements.fiscal_periods
        
        if len(periods) >= 5:
            check.status = VerificationStatus.PASSED
            check.message = f"{len(periods)} years of data available"
        elif len(periods) >= 3:
            check.status = VerificationStatus.WARNING
            check.message = f"Only {len(periods)} years of data (5 preferred)"
        else:
            check.status = VerificationStatus.FAILED
            check.message = f"Insufficient data: only {len(periods)} year(s)"
        
        check.details = {"periods": periods, "count": len(periods)}
        return check
    
    def _check_derived_metrics(self, derived: Any) -> CheckResult:
        """Check derived metrics are calculated."""
        check = CheckResult(
            check_name="derived_metrics",
            category=CheckCategory.INPUT_INTEGRITY,
        )
        
        available = 0
        total = 4
        
        if derived.fcf_calculated:
            available += 1
        if derived.ebitda_calculated:
            available += 1
        if derived.net_debt:
            available += 1
        if derived.enterprise_value:
            available += 1
        
        if available == total:
            check.status = VerificationStatus.PASSED
            check.message = "All derived metrics calculated"
        elif available >= 2:
            check.status = VerificationStatus.WARNING
            check.message = f"{available}/{total} derived metrics available"
        else:
            check.status = VerificationStatus.FAILED
            check.message = f"Only {available}/{total} derived metrics"
        
        check.details = {"available": available, "total": total}
        return check
    
    def _check_market_cap(self, profile: Any) -> CheckResult:
        """Check market cap is valid."""
        check = CheckResult(
            check_name="market_cap_validity",
            category=CheckCategory.INPUT_INTEGRITY,
        )
        
        market_cap = getattr(profile, 'market_cap', None)
        
        if market_cap and market_cap > 0:
            check.status = VerificationStatus.PASSED
            check.actual_value = market_cap
            check.message = f"Market cap valid: ${market_cap/1e9:.2f}B"
        else:
            check.status = VerificationStatus.FAILED
            check.message = "Market cap missing or invalid"
        
        return check
    
    def _check_shares_outstanding(self, profile: Any, statements: Any) -> CheckResult:
        """Check shares outstanding consistency."""
        check = CheckResult(
            check_name="shares_outstanding",
            category=CheckCategory.INPUT_INTEGRITY,
        )
        
        profile_shares = getattr(profile, 'shares_outstanding', None)
        
        # Try to get from balance sheet
        bs_shares = None
        if statements.balance_sheet is not None and not statements.balance_sheet.empty:
            periods = statements.fiscal_periods
            if periods and 'shares_outstanding' in statements.balance_sheet.index:
                bs_shares = statements.balance_sheet.loc['shares_outstanding', periods[0]]
                if pd.notna(bs_shares):
                    bs_shares = float(bs_shares)
                else:
                    bs_shares = None
        
        if profile_shares and bs_shares:
            deviation = abs(profile_shares - bs_shares) / bs_shares
            check.expected_value = bs_shares
            check.actual_value = profile_shares
            check.deviation = deviation
            
            if deviation < 0.10:  # Within 10%
                check.status = VerificationStatus.PASSED
                check.message = "Shares outstanding consistent between sources"
            else:
                check.status = VerificationStatus.WARNING
                check.message = f"Shares differ by {deviation*100:.1f}% between sources"
        elif profile_shares or bs_shares:
            check.status = VerificationStatus.PASSED
            check.message = "Shares outstanding available from one source"
            check.actual_value = profile_shares or bs_shares
        else:
            check.status = VerificationStatus.FAILED
            check.message = "Shares outstanding not available"
        
        return check
    
    def _check_revenue_continuity(self, statements: Any) -> CheckResult:
        """Check revenue is positive for all years."""
        check = CheckResult(
            check_name="revenue_continuity",
            category=CheckCategory.INPUT_INTEGRITY,
        )
        
        if statements.income_statement is None or statements.income_statement.empty:
            check.status = VerificationStatus.SKIPPED
            check.message = "Income statement not available"
            return check
        
        periods = statements.fiscal_periods
        negative_years = []
        
        for period in periods:
            if 'total_revenue' in statements.income_statement.index:
                revenue = statements.income_statement.loc['total_revenue', period]
                if pd.notna(revenue) and float(revenue) <= 0:
                    negative_years.append(period)
        
        if not negative_years:
            check.status = VerificationStatus.PASSED
            check.message = "Revenue positive for all years"
        else:
            check.status = VerificationStatus.WARNING
            check.message = f"Non-positive revenue in: {', '.join(negative_years)}"
        
        check.details = {"negative_years": negative_years}
        return check
    
    def _check_net_income_signs(self, statements: Any) -> CheckResult:
        """Check net income sign pattern."""
        check = CheckResult(
            check_name="net_income_pattern",
            category=CheckCategory.INPUT_INTEGRITY,
        )
        
        if statements.income_statement is None or statements.income_statement.empty:
            check.status = VerificationStatus.SKIPPED
            check.message = "Income statement not available"
            return check
        
        periods = statements.fiscal_periods
        negative_count = 0
        
        for period in periods:
            if 'net_income' in statements.income_statement.index:
                ni = statements.income_statement.loc['net_income', period]
                if pd.notna(ni) and float(ni) < 0:
                    negative_count += 1
        
        if negative_count == 0:
            check.status = VerificationStatus.PASSED
            check.message = "Net income positive all years"
        elif negative_count <= 1:
            check.status = VerificationStatus.WARNING
            check.message = f"{negative_count} year(s) with negative net income"
        else:
            check.status = VerificationStatus.WARNING
            check.message = f"{negative_count} years with negative net income"
        
        check.details = {"negative_years": negative_count}
        return check
    
    def _calculate_summary(self, summary: CategorySummary):
        """Calculate summary metrics."""
        for check in summary.checks:
            summary.total_checks += 1
            if check.status == VerificationStatus.PASSED:
                summary.passed += 1
            elif check.status == VerificationStatus.WARNING:
                summary.warnings += 1
            elif check.status == VerificationStatus.FAILED:
                summary.failed += 1
            else:
                summary.skipped += 1
        
        summary.calculate_metrics()


# =============================================================================
# CALCULATION VERIFIER
# =============================================================================

class CalculationVerifier:
    """
    Re-verifies key calculations from previous phases.
    
    Performs independent recalculation and compares against stored values.
    """
    
    def verify(
        self,
        collection_result: Any,
        dcf_result: Optional[Any] = None,
        ddm_result: Optional[Any] = None,
        multiples_result: Optional[Any] = None,
    ) -> CategorySummary:
        """Run calculation verification checks."""
        summary = CategorySummary(CheckCategory.CALCULATION_ACCURACY)
        
        statements = collection_result.statements
        derived = collection_result.derived_metrics
        profile = collection_result.company_profile
        periods = statements.fiscal_periods
        
        # Check 1: FCF calculation verification
        check = self._verify_fcf(statements, derived, periods)
        summary.checks.append(check)
        
        # Check 2: EBITDA calculation verification
        check = self._verify_ebitda(statements, derived, periods)
        summary.checks.append(check)
        
        # Check 3: Net debt calculation verification
        check = self._verify_net_debt(statements, derived, periods)
        summary.checks.append(check)
        
        # Check 4: DCF intrinsic value verification (if available)
        if dcf_result:
            check = self._verify_dcf_value(dcf_result, derived, profile)
            summary.checks.append(check)
        
        # Check 5: DDM intrinsic value verification (if available)
        if ddm_result and ddm_result.is_applicable:
            check = self._verify_ddm_value(ddm_result)
            summary.checks.append(check)
        
        # Check 6: Cost of Equity (CAPM) verification
        if dcf_result:
            check = self._verify_cost_of_equity(dcf_result)
            summary.checks.append(check)
        
        # Check 7: P/E ratio verification
        if multiples_result:
            check = self._verify_pe_ratio(multiples_result, statements, profile, periods)
            summary.checks.append(check)
        
        # Check 8: Enterprise Value verification
        check = self._verify_enterprise_value(derived, profile)
        summary.checks.append(check)
        
        # Calculate summary metrics
        self._calculate_summary(summary)
        
        return summary
    
    def _verify_fcf(
        self,
        statements: Any,
        derived: Any,
        periods: List[str],
    ) -> CheckResult:
        """Verify FCF = Operating Cash Flow - CapEx."""
        check = CheckResult(
            check_name="fcf_calculation",
            category=CheckCategory.CALCULATION_ACCURACY,
            tolerance=AccuracyConfig.TOLERANCE_STRICT,
        )
        
        if not periods:
            check.status = VerificationStatus.SKIPPED
            check.message = "No periods available"
            return check
        
        latest = periods[0]
        cf = statements.cash_flow
        
        if cf is None or cf.empty:
            check.status = VerificationStatus.SKIPPED
            check.message = "Cash flow statement not available"
            return check
        
        ocf = self._get_value(cf, 'operating_cash_flow', latest)
        capex = self._get_value(cf, 'capital_expenditure', latest)
        stored_fcf = derived.fcf_calculated.get(latest)
        
        if ocf is not None and capex is not None:
            # CapEx is typically positive in cash flow statement
            calculated_fcf = ocf - capex
            check.expected_value = calculated_fcf
            check.actual_value = stored_fcf
            
            if stored_fcf:
                deviation = abs(calculated_fcf - stored_fcf) / abs(stored_fcf) if stored_fcf != 0 else 0
                check.deviation = deviation
                
                if deviation <= check.tolerance:
                    check.status = VerificationStatus.PASSED
                    check.message = f"FCF calculation verified: ${calculated_fcf/1e9:.2f}B"
                else:
                    check.status = VerificationStatus.WARNING
                    check.message = f"FCF deviation: {deviation*100:.2f}%"
            else:
                check.status = VerificationStatus.WARNING
                check.message = "Stored FCF not available for comparison"
        else:
            check.status = VerificationStatus.SKIPPED
            check.message = "OCF or CapEx not available"
        
        return check
    
    def _verify_ebitda(
        self,
        statements: Any,
        derived: Any,
        periods: List[str],
    ) -> CheckResult:
        """Verify EBITDA calculation."""
        check = CheckResult(
            check_name="ebitda_calculation",
            category=CheckCategory.CALCULATION_ACCURACY,
            tolerance=AccuracyConfig.TOLERANCE_NORMAL,
        )
        
        if not periods:
            check.status = VerificationStatus.SKIPPED
            return check
        
        latest = periods[0]
        inc = statements.income_statement
        
        if inc is None or inc.empty:
            check.status = VerificationStatus.SKIPPED
            return check
        
        # Try direct EBITDA first
        stored_ebitda = self._get_value(inc, 'ebitda', latest)
        if not stored_ebitda:
            stored_ebitda = derived.ebitda_calculated.get(latest)
        
        # Calculate from components
        operating_income = self._get_value(inc, 'operating_income', latest)
        da = self._get_value(inc, 'depreciation_amortization', latest)
        
        if not da:
            da = self._get_value(statements.cash_flow, 'depreciation_amortization', latest)
        
        if operating_income and da:
            calculated_ebitda = operating_income + da
            check.expected_value = calculated_ebitda
            check.actual_value = stored_ebitda
            
            if stored_ebitda:
                deviation = abs(calculated_ebitda - stored_ebitda) / abs(stored_ebitda) if stored_ebitda != 0 else 0
                check.deviation = deviation
                
                if deviation <= check.tolerance:
                    check.status = VerificationStatus.PASSED
                    check.message = f"EBITDA verified: ${calculated_ebitda/1e9:.2f}B"
                else:
                    check.status = VerificationStatus.WARNING
                    check.message = f"EBITDA deviation: {deviation*100:.2f}%"
            else:
                check.status = VerificationStatus.PASSED
                check.message = f"EBITDA calculated: ${calculated_ebitda/1e9:.2f}B"
        else:
            check.status = VerificationStatus.SKIPPED
            check.message = "Components not available"
        
        return check
    
    def _verify_net_debt(
        self,
        statements: Any,
        derived: Any,
        periods: List[str],
    ) -> CheckResult:
        """Verify Net Debt = Total Debt - Cash."""
        check = CheckResult(
            check_name="net_debt_calculation",
            category=CheckCategory.CALCULATION_ACCURACY,
            tolerance=AccuracyConfig.TOLERANCE_STRICT,
        )
        
        if not periods:
            check.status = VerificationStatus.SKIPPED
            return check
        
        latest = periods[0]
        bs = statements.balance_sheet
        
        if bs is None or bs.empty:
            check.status = VerificationStatus.SKIPPED
            return check
        
        total_debt = self._get_value(bs, 'total_debt', latest)
        cash = self._get_value(bs, 'cash_and_equivalents', latest)
        stored_net_debt = derived.net_debt.get(latest)
        
        if total_debt is not None and cash is not None:
            calculated_net_debt = total_debt - cash
            check.expected_value = calculated_net_debt
            check.actual_value = stored_net_debt
            
            if stored_net_debt is not None:
                deviation = abs(calculated_net_debt - stored_net_debt) / abs(stored_net_debt) if stored_net_debt != 0 else 0
                check.deviation = deviation
                
                if deviation <= check.tolerance:
                    check.status = VerificationStatus.PASSED
                    check.message = f"Net debt verified: ${calculated_net_debt/1e9:.2f}B"
                else:
                    check.status = VerificationStatus.WARNING
                    check.message = f"Net debt deviation: {deviation*100:.2f}%"
            else:
                check.status = VerificationStatus.PASSED
                check.message = f"Net debt calculated: ${calculated_net_debt/1e9:.2f}B"
        else:
            check.status = VerificationStatus.SKIPPED
            check.message = "Total debt or cash not available"
        
        return check
    
    def _verify_dcf_value(
        self,
        dcf_result: Any,
        derived: Any,
        profile: Any,
    ) -> CheckResult:
        """Verify DCF intrinsic value calculation."""
        check = CheckResult(
            check_name="dcf_intrinsic_value",
            category=CheckCategory.CALCULATION_ACCURACY,
            tolerance=AccuracyConfig.TOLERANCE_NORMAL,
        )
        
        stored_value = dcf_result.intrinsic_value_per_share
        equity_value = dcf_result.equity_value
        shares = dcf_result.shares_outstanding
        
        if equity_value and shares and shares > 0:
            calculated_value = equity_value / shares
            check.expected_value = calculated_value
            check.actual_value = stored_value
            
            deviation = abs(calculated_value - stored_value) / stored_value if stored_value > 0 else 0
            check.deviation = deviation
            
            if deviation <= check.tolerance:
                check.status = VerificationStatus.PASSED
                check.message = f"DCF value verified: ${stored_value:.2f}"
            else:
                check.status = VerificationStatus.WARNING
                check.message = f"DCF value deviation: {deviation*100:.2f}%"
        else:
            check.status = VerificationStatus.SKIPPED
            check.message = "DCF components not available"
        
        return check
    
    def _verify_ddm_value(self, ddm_result: Any) -> CheckResult:
        """Verify DDM intrinsic value calculation."""
        check = CheckResult(
            check_name="ddm_intrinsic_value",
            category=CheckCategory.CALCULATION_ACCURACY,
            tolerance=AccuracyConfig.TOLERANCE_NORMAL,
        )
        
        projection = ddm_result.ddm_projection
        sum_pv = projection.sum_of_pv_dividends
        pv_terminal = projection.pv_terminal_value
        stored_value = ddm_result.intrinsic_value_per_share
        
        if sum_pv and pv_terminal:
            calculated_value = sum_pv + pv_terminal
            check.expected_value = calculated_value
            check.actual_value = stored_value
            
            deviation = abs(calculated_value - stored_value) / stored_value if stored_value > 0 else 0
            check.deviation = deviation
            
            if deviation <= check.tolerance:
                check.status = VerificationStatus.PASSED
                check.message = f"DDM value verified: ${stored_value:.2f}"
            else:
                check.status = VerificationStatus.WARNING
                check.message = f"DDM value deviation: {deviation*100:.2f}%"
        else:
            check.status = VerificationStatus.SKIPPED
            check.message = "DDM projection not available"
        
        return check
    
    def _verify_cost_of_equity(self, dcf_result: Any) -> CheckResult:
        """Verify Cost of Equity = Rf + Beta * ERP."""
        check = CheckResult(
            check_name="cost_of_equity_capm",
            category=CheckCategory.CALCULATION_ACCURACY,
            tolerance=AccuracyConfig.TOLERANCE_STRICT,
        )
        
        wacc_calc = dcf_result.wacc_calculation
        ke_calc = wacc_calc.cost_of_equity
        rf = ke_calc.risk_free_rate
        beta = ke_calc.beta
        erp = ke_calc.equity_risk_premium
        stored_ke = ke_calc.cost_of_equity
        
        if rf and beta and erp:
            calculated_ke = rf + (beta * erp)
            check.expected_value = calculated_ke
            check.actual_value = stored_ke
            
            deviation = abs(calculated_ke - stored_ke) / stored_ke if stored_ke > 0 else 0
            check.deviation = deviation
            
            if deviation <= check.tolerance:
                check.status = VerificationStatus.PASSED
                check.message = f"Cost of Equity verified: {stored_ke*100:.2f}%"
            else:
                check.status = VerificationStatus.FAILED
                check.message = f"Cost of Equity error: expected {calculated_ke*100:.2f}%, got {stored_ke*100:.2f}%"
        else:
            check.status = VerificationStatus.SKIPPED
            check.message = "CAPM components not available"
        
        return check
    
    def _verify_pe_ratio(
        self,
        multiples_result: Any,
        statements: Any,
        profile: Any,
        periods: List[str],
    ) -> CheckResult:
        """Verify P/E = Market Cap / Net Income."""
        check = CheckResult(
            check_name="pe_ratio",
            category=CheckCategory.CALCULATION_ACCURACY,
            tolerance=AccuracyConfig.TOLERANCE_NORMAL,
        )
        
        market_cap = getattr(profile, 'market_cap', None)
        
        if not market_cap or not periods:
            check.status = VerificationStatus.SKIPPED
            return check
        
        latest = periods[0]
        net_income = self._get_value(statements.income_statement, 'net_income', latest)
        stored_pe = multiples_result.pe_analysis.current_value
        
        if net_income and net_income > 0:
            calculated_pe = market_cap / net_income
            check.expected_value = calculated_pe
            check.actual_value = stored_pe
            
            if stored_pe:
                deviation = abs(calculated_pe - stored_pe) / stored_pe
                check.deviation = deviation
                
                if deviation <= check.tolerance:
                    check.status = VerificationStatus.PASSED
                    check.message = f"P/E ratio verified: {stored_pe:.2f}"
                else:
                    check.status = VerificationStatus.WARNING
                    check.message = f"P/E deviation: {deviation*100:.2f}%"
            else:
                check.status = VerificationStatus.WARNING
                check.message = "Stored P/E not available"
        else:
            check.status = VerificationStatus.SKIPPED
            check.message = "Net income not positive"
        
        return check
    
    def _verify_enterprise_value(self, derived: Any, profile: Any) -> CheckResult:
        """Verify EV = Market Cap + Net Debt."""
        check = CheckResult(
            check_name="enterprise_value",
            category=CheckCategory.CALCULATION_ACCURACY,
            tolerance=AccuracyConfig.TOLERANCE_NORMAL,
        )
        
        market_cap = getattr(profile, 'market_cap', None)
        stored_ev = derived.enterprise_value
        
        # Get latest net debt
        net_debt = None
        if derived.net_debt:
            # Get the first (latest) value
            for year in sorted(derived.net_debt.keys(), reverse=True):
                net_debt = derived.net_debt[year]
                if net_debt is not None:
                    break
        
        if market_cap and net_debt is not None:
            calculated_ev = market_cap + net_debt
            check.expected_value = calculated_ev
            check.actual_value = stored_ev
            
            if stored_ev:
                deviation = abs(calculated_ev - stored_ev) / stored_ev if stored_ev > 0 else 0
                check.deviation = deviation
                
                if deviation <= check.tolerance:
                    check.status = VerificationStatus.PASSED
                    check.message = f"Enterprise Value verified: ${stored_ev/1e9:.2f}B"
                else:
                    check.status = VerificationStatus.WARNING
                    check.message = f"EV deviation: {deviation*100:.2f}%"
            else:
                check.status = VerificationStatus.PASSED
                check.message = f"EV calculated: ${calculated_ev/1e9:.2f}B"
        else:
            check.status = VerificationStatus.SKIPPED
            check.message = "Market cap or net debt not available"
        
        return check
    
    def _get_value(self, df: pd.DataFrame, field: str, year: str) -> Optional[float]:
        """Safely extract value from DataFrame."""
        if df is None or df.empty:
            return None
        if field not in df.index or year not in df.columns:
            return None
        value = df.loc[field, year]
        return float(value) if pd.notna(value) else None
    
    def _calculate_summary(self, summary: CategorySummary):
        """Calculate summary metrics."""
        for check in summary.checks:
            summary.total_checks += 1
            if check.status == VerificationStatus.PASSED:
                summary.passed += 1
            elif check.status == VerificationStatus.WARNING:
                summary.warnings += 1
            elif check.status == VerificationStatus.FAILED:
                summary.failed += 1
            else:
                summary.skipped += 1
        
        summary.calculate_metrics()


# =============================================================================
# CROSS-PHASE RECONCILER
# =============================================================================

class CrossPhaseReconciler:
    """
    Validates data consistency between phases.
    
    Ensures values used in later phases match those from earlier phases.
    """
    
    def reconcile(
        self,
        collection_result: Any,
        validation_result: Optional[Any] = None,
        ratio_result: Optional[Any] = None,
        dupont_result: Optional[Any] = None,
        dcf_result: Optional[Any] = None,
        ddm_result: Optional[Any] = None,
    ) -> CategorySummary:
        """Run cross-phase reconciliation checks."""
        summary = CategorySummary(CheckCategory.CROSS_PHASE_CONSISTENCY)
        
        # Check 1: ROE consistency (Phase 3 vs Phase 4)
        if ratio_result and dupont_result:
            check = self._check_roe_consistency(ratio_result, dupont_result)
            summary.checks.append(check)
        
        # Check 2: FCF consistency (Phase 1 vs Phase 5)
        if dcf_result:
            check = self._check_fcf_consistency(collection_result, dcf_result)
            summary.checks.append(check)
        
        # Check 3: Dividend consistency (Phase 1 vs Phase 6)
        if ddm_result and ddm_result.is_applicable:
            check = self._check_dividend_consistency(collection_result, ddm_result)
            summary.checks.append(check)
        
        # Check 4: Market cap usage consistency
        if dcf_result:
            check = self._check_market_cap_consistency(collection_result, dcf_result)
            summary.checks.append(check)
        
        # Check 5: Beta consistency across phases
        if dcf_result and ddm_result:
            check = self._check_beta_consistency(dcf_result, ddm_result)
            summary.checks.append(check)
        
        # Check 6: Shares outstanding consistency
        if dcf_result:
            check = self._check_shares_consistency(collection_result, dcf_result)
            summary.checks.append(check)
        
        # Calculate summary metrics
        self._calculate_summary(summary)
        
        return summary
    
    def _check_roe_consistency(self, ratio_result: Any, dupont_result: Any) -> CheckResult:
        """Check ROE matches between ratio analysis and DuPont."""
        check = CheckResult(
            check_name="roe_consistency",
            category=CheckCategory.CROSS_PHASE_CONSISTENCY,
            tolerance=AccuracyConfig.TOLERANCE_NORMAL,
        )
        
        # Get ROE from ratio analysis (Phase 3)
        ratio_roe = None
        roe_series = ratio_result.profitability.ratios.get("roe")
        if roe_series and roe_series.latest_value is not None:
            ratio_roe = roe_series.latest_value
        
        # Get ROE from DuPont (Phase 4)
        dupont_roe = None
        if dupont_result.three_factor:
            # Get the latest year's ROE
            years = sorted(dupont_result.three_factor.keys(), reverse=True)
            if years:
                latest_dupont = dupont_result.three_factor[years[0]]
                dupont_roe = latest_dupont.roe_calculated
        
        if ratio_roe is not None and dupont_roe is not None:
            check.expected_value = ratio_roe
            check.actual_value = dupont_roe
            
            deviation = abs(ratio_roe - dupont_roe) / abs(ratio_roe) if ratio_roe != 0 else 0
            check.deviation = deviation
            
            if deviation <= check.tolerance:
                check.status = VerificationStatus.PASSED
                check.message = f"ROE consistent: Phase 3 = {ratio_roe*100:.2f}%, Phase 4 = {dupont_roe*100:.2f}%"
            else:
                check.status = VerificationStatus.WARNING
                check.message = f"ROE differs: Phase 3 = {ratio_roe*100:.2f}%, Phase 4 = {dupont_roe*100:.2f}%"
        else:
            check.status = VerificationStatus.SKIPPED
            check.message = "ROE not available from both phases"
        
        return check
    
    def _check_fcf_consistency(self, collection_result: Any, dcf_result: Any) -> CheckResult:
        """Check FCF from Phase 1 matches Phase 5 input."""
        check = CheckResult(
            check_name="fcf_consistency",
            category=CheckCategory.CROSS_PHASE_CONSISTENCY,
            tolerance=AccuracyConfig.TOLERANCE_STRICT,
        )
        
        # Get Phase 1 FCF
        phase1_fcf = None
        periods = collection_result.statements.fiscal_periods
        if periods:
            phase1_fcf = collection_result.derived_metrics.fcf_calculated.get(periods[0])
        
        # Get Phase 5 base FCF
        phase5_fcf = dcf_result.historical_fcf.latest_fcf
        
        if phase1_fcf and phase5_fcf:
            check.expected_value = phase1_fcf
            check.actual_value = phase5_fcf
            
            deviation = abs(phase1_fcf - phase5_fcf) / abs(phase1_fcf) if phase1_fcf != 0 else 0
            check.deviation = deviation
            
            if deviation <= check.tolerance:
                check.status = VerificationStatus.PASSED
                check.message = f"FCF consistent: ${phase1_fcf/1e9:.2f}B"
            else:
                check.status = VerificationStatus.WARNING
                check.message = f"FCF differs: Phase 1 = ${phase1_fcf/1e9:.2f}B, Phase 5 = ${phase5_fcf/1e9:.2f}B"
        else:
            check.status = VerificationStatus.SKIPPED
            check.message = "FCF not available from both phases"
        
        return check
    
    def _check_dividend_consistency(self, collection_result: Any, ddm_result: Any) -> CheckResult:
        """Check dividend data from Phase 1 matches Phase 6."""
        check = CheckResult(
            check_name="dividend_consistency",
            category=CheckCategory.CROSS_PHASE_CONSISTENCY,
            tolerance=AccuracyConfig.TOLERANCE_NORMAL,
        )
        
        # Get Phase 1 current DPS
        phase1_dps = collection_result.dividend_history.current_annual_dps
        
        # Get Phase 6 current DPS
        phase6_dps = ddm_result.historical_dividends.current_dps
        
        if phase1_dps and phase6_dps:
            check.expected_value = phase1_dps
            check.actual_value = phase6_dps
            
            deviation = abs(phase1_dps - phase6_dps) / abs(phase1_dps) if phase1_dps != 0 else 0
            check.deviation = deviation
            
            if deviation <= check.tolerance:
                check.status = VerificationStatus.PASSED
                check.message = f"DPS consistent: ${phase6_dps:.4f}"
            else:
                check.status = VerificationStatus.WARNING
                check.message = f"DPS differs: Phase 1 = ${phase1_dps:.4f}, Phase 6 = ${phase6_dps:.4f}"
        else:
            check.status = VerificationStatus.SKIPPED
            check.message = "DPS not available from both phases"
        
        return check
    
    def _check_market_cap_consistency(self, collection_result: Any, dcf_result: Any) -> CheckResult:
        """Check market cap usage consistency."""
        check = CheckResult(
            check_name="market_cap_consistency",
            category=CheckCategory.CROSS_PHASE_CONSISTENCY,
            tolerance=AccuracyConfig.TOLERANCE_STRICT,
        )
        
        phase1_mc = getattr(collection_result.company_profile, 'market_cap', None)
        phase5_mc = dcf_result.wacc_calculation.market_cap
        
        if phase1_mc and phase5_mc:
            check.expected_value = phase1_mc
            check.actual_value = phase5_mc
            
            deviation = abs(phase1_mc - phase5_mc) / phase1_mc if phase1_mc > 0 else 0
            check.deviation = deviation
            
            if deviation <= check.tolerance:
                check.status = VerificationStatus.PASSED
                check.message = "Market cap consistent across phases"
            else:
                check.status = VerificationStatus.FAILED
                check.message = f"Market cap differs: Phase 1 = ${phase1_mc/1e9:.2f}B, Phase 5 = ${phase5_mc/1e9:.2f}B"
        else:
            check.status = VerificationStatus.SKIPPED
            check.message = "Market cap not available from both phases"
        
        return check
    
    def _check_beta_consistency(self, dcf_result: Any, ddm_result: Any) -> CheckResult:
        """Check beta is consistent between DCF and DDM."""
        check = CheckResult(
            check_name="beta_consistency",
            category=CheckCategory.CROSS_PHASE_CONSISTENCY,
            tolerance=AccuracyConfig.TOLERANCE_STRICT,
        )
        
        dcf_beta = dcf_result.wacc_calculation.cost_of_equity.beta
        ddm_beta = ddm_result.cost_of_equity.beta
        
        if dcf_beta and ddm_beta:
            check.expected_value = dcf_beta
            check.actual_value = ddm_beta
            
            deviation = abs(dcf_beta - ddm_beta) / dcf_beta if dcf_beta > 0 else 0
            check.deviation = deviation
            
            if deviation <= check.tolerance:
                check.status = VerificationStatus.PASSED
                check.message = f"Beta consistent: {dcf_beta:.2f}"
            else:
                check.status = VerificationStatus.FAILED
                check.message = f"Beta differs: DCF = {dcf_beta:.2f}, DDM = {ddm_beta:.2f}"
        else:
            check.status = VerificationStatus.SKIPPED
            check.message = "Beta not available from both phases"
        
        return check
    
    def _check_shares_consistency(self, collection_result: Any, dcf_result: Any) -> CheckResult:
        """Check shares outstanding consistency."""
        check = CheckResult(
            check_name="shares_consistency",
            category=CheckCategory.CROSS_PHASE_CONSISTENCY,
            tolerance=AccuracyConfig.TOLERANCE_NORMAL,
        )
        
        phase1_shares = getattr(collection_result.company_profile, 'shares_outstanding', None)
        phase5_shares = dcf_result.shares_outstanding
        
        if phase1_shares and phase5_shares:
            check.expected_value = phase1_shares
            check.actual_value = phase5_shares
            
            deviation = abs(phase1_shares - phase5_shares) / phase1_shares if phase1_shares > 0 else 0
            check.deviation = deviation
            
            if deviation <= check.tolerance:
                check.status = VerificationStatus.PASSED
                check.message = "Shares outstanding consistent"
            else:
                check.status = VerificationStatus.WARNING
                check.message = f"Shares differ by {deviation*100:.1f}%"
        else:
            check.status = VerificationStatus.SKIPPED
            check.message = "Shares not available from both phases"
        
        return check
    
    def _calculate_summary(self, summary: CategorySummary):
        """Calculate summary metrics."""
        for check in summary.checks:
            summary.total_checks += 1
            if check.status == VerificationStatus.PASSED:
                summary.passed += 1
            elif check.status == VerificationStatus.WARNING:
                summary.warnings += 1
            elif check.status == VerificationStatus.FAILED:
                summary.failed += 1
            else:
                summary.skipped += 1
        
        summary.calculate_metrics()


# =============================================================================
# VALUATION CONSISTENCY CHECKER
# =============================================================================

class ValuationConsistencyChecker:
    """
    Ensures all three valuation models use identical base inputs.
    
    Critical for meaningful cross-model comparison.
    """
    
    def check(
        self,
        dcf_result: Optional[Any] = None,
        ddm_result: Optional[Any] = None,
        multiples_result: Optional[Any] = None,
    ) -> Tuple[CategorySummary, List[ValuationInputComparison]]:
        """Run valuation consistency checks."""
        summary = CategorySummary(CheckCategory.VALUATION_CONSISTENCY)
        comparisons = []
        
        # Check 1: Current price consistency
        check, comp = self._check_price_consistency(dcf_result, ddm_result, multiples_result)
        summary.checks.append(check)
        if comp:
            comparisons.append(comp)
        
        # Check 2: Risk-free rate consistency
        check, comp = self._check_rf_consistency(dcf_result, ddm_result)
        summary.checks.append(check)
        if comp:
            comparisons.append(comp)
        
        # Check 3: Beta consistency
        check, comp = self._check_beta_consistency_val(dcf_result, ddm_result)
        summary.checks.append(check)
        if comp:
            comparisons.append(comp)
        
        # Check 4: Equity risk premium consistency
        check, comp = self._check_erp_consistency(dcf_result, ddm_result)
        summary.checks.append(check)
        if comp:
            comparisons.append(comp)
        
        # Check 5: Terminal growth rate reasonableness
        check = self._check_terminal_growth(dcf_result, ddm_result)
        summary.checks.append(check)
        
        # Check 6: Discount rate relationship (WACC vs Ke)
        check = self._check_discount_rate_relationship(dcf_result, ddm_result)
        summary.checks.append(check)
        
        # Calculate summary metrics
        self._calculate_summary(summary)
        
        return summary, comparisons
    
    def _check_price_consistency(
        self,
        dcf_result: Optional[Any],
        ddm_result: Optional[Any],
        multiples_result: Optional[Any],
    ) -> Tuple[CheckResult, Optional[ValuationInputComparison]]:
        """Check current price is consistent across models."""
        check = CheckResult(
            check_name="current_price_consistency",
            category=CheckCategory.VALUATION_CONSISTENCY,
            tolerance=AccuracyConfig.TOLERANCE_STRICT,
        )
        
        prices = []
        comp = ValuationInputComparison(parameter="current_price")
        
        if dcf_result:
            comp.dcf_value = dcf_result.current_price
            if comp.dcf_value:
                prices.append(comp.dcf_value)
        
        if ddm_result and ddm_result.is_applicable:
            comp.ddm_value = ddm_result.current_price
            if comp.ddm_value:
                prices.append(comp.ddm_value)
        
        if multiples_result:
            comp.multiples_value = multiples_result.current_price
            if comp.multiples_value:
                prices.append(comp.multiples_value)
        
        if len(prices) >= 2:
            max_price = max(prices)
            min_price = min(prices)
            deviation = (max_price - min_price) / min_price if min_price > 0 else 0
            comp.max_deviation = deviation
            
            if deviation <= check.tolerance:
                check.status = VerificationStatus.PASSED
                check.message = f"Current price consistent: ${prices[0]:.2f}"
                comp.is_consistent = True
            else:
                check.status = VerificationStatus.FAILED
                check.message = f"Price inconsistency: range ${min_price:.2f} - ${max_price:.2f}"
                comp.is_consistent = False
        elif len(prices) == 1:
            check.status = VerificationStatus.WARNING
            check.message = "Only one price available for comparison"
        else:
            check.status = VerificationStatus.SKIPPED
            check.message = "No prices available"
            return check, None
        
        return check, comp
    
    def _check_rf_consistency(
        self,
        dcf_result: Optional[Any],
        ddm_result: Optional[Any],
    ) -> Tuple[CheckResult, Optional[ValuationInputComparison]]:
        """Check risk-free rate consistency."""
        check = CheckResult(
            check_name="risk_free_rate_consistency",
            category=CheckCategory.VALUATION_CONSISTENCY,
            tolerance=AccuracyConfig.TOLERANCE_STRICT,
        )
        
        comp = ValuationInputComparison(parameter="risk_free_rate")
        
        dcf_rf = dcf_result.wacc_calculation.cost_of_equity.risk_free_rate if dcf_result else None
        ddm_rf = ddm_result.cost_of_equity.risk_free_rate if ddm_result and ddm_result.is_applicable else None
        
        comp.dcf_value = dcf_rf
        comp.ddm_value = ddm_rf
        
        if dcf_rf and ddm_rf:
            deviation = abs(dcf_rf - ddm_rf) / dcf_rf if dcf_rf > 0 else 0
            comp.max_deviation = deviation
            
            if deviation <= check.tolerance:
                check.status = VerificationStatus.PASSED
                check.message = f"Risk-free rate consistent: {dcf_rf*100:.2f}%"
                comp.is_consistent = True
            else:
                check.status = VerificationStatus.FAILED
                check.message = f"Rf inconsistent: DCF = {dcf_rf*100:.2f}%, DDM = {ddm_rf*100:.2f}%"
                comp.is_consistent = False
        else:
            check.status = VerificationStatus.SKIPPED
            check.message = "Risk-free rate not available from both models"
            return check, None
        
        return check, comp
    
    def _check_beta_consistency_val(
        self,
        dcf_result: Optional[Any],
        ddm_result: Optional[Any],
    ) -> Tuple[CheckResult, Optional[ValuationInputComparison]]:
        """Check beta consistency between valuation models."""
        check = CheckResult(
            check_name="beta_consistency_valuation",
            category=CheckCategory.VALUATION_CONSISTENCY,
            tolerance=AccuracyConfig.TOLERANCE_STRICT,
        )
        
        comp = ValuationInputComparison(parameter="beta")
        
        dcf_beta = dcf_result.wacc_calculation.cost_of_equity.beta if dcf_result else None
        ddm_beta = ddm_result.cost_of_equity.beta if ddm_result and ddm_result.is_applicable else None
        
        comp.dcf_value = dcf_beta
        comp.ddm_value = ddm_beta
        
        if dcf_beta and ddm_beta:
            deviation = abs(dcf_beta - ddm_beta) / dcf_beta if dcf_beta > 0 else 0
            comp.max_deviation = deviation
            
            if deviation <= check.tolerance:
                check.status = VerificationStatus.PASSED
                check.message = f"Beta consistent: {dcf_beta:.2f}"
                comp.is_consistent = True
            else:
                check.status = VerificationStatus.FAILED
                check.message = f"Beta inconsistent: DCF = {dcf_beta:.2f}, DDM = {ddm_beta:.2f}"
                comp.is_consistent = False
        else:
            check.status = VerificationStatus.SKIPPED
            check.message = "Beta not available from both models"
            return check, None
        
        return check, comp
    
    def _check_erp_consistency(
        self,
        dcf_result: Optional[Any],
        ddm_result: Optional[Any],
    ) -> Tuple[CheckResult, Optional[ValuationInputComparison]]:
        """Check equity risk premium consistency."""
        check = CheckResult(
            check_name="equity_risk_premium_consistency",
            category=CheckCategory.VALUATION_CONSISTENCY,
            tolerance=AccuracyConfig.TOLERANCE_STRICT,
        )
        
        comp = ValuationInputComparison(parameter="equity_risk_premium")
        
        dcf_erp = dcf_result.wacc_calculation.cost_of_equity.equity_risk_premium if dcf_result else None
        ddm_erp = ddm_result.cost_of_equity.equity_risk_premium if ddm_result and ddm_result.is_applicable else None
        
        comp.dcf_value = dcf_erp
        comp.ddm_value = ddm_erp
        
        if dcf_erp and ddm_erp:
            deviation = abs(dcf_erp - ddm_erp) / dcf_erp if dcf_erp > 0 else 0
            comp.max_deviation = deviation
            
            if deviation <= check.tolerance:
                check.status = VerificationStatus.PASSED
                check.message = f"ERP consistent: {dcf_erp*100:.2f}%"
                comp.is_consistent = True
            else:
                check.status = VerificationStatus.FAILED
                check.message = f"ERP inconsistent: DCF = {dcf_erp*100:.2f}%, DDM = {ddm_erp*100:.2f}%"
                comp.is_consistent = False
        else:
            check.status = VerificationStatus.SKIPPED
            check.message = "ERP not available from both models"
            return check, None
        
        return check, comp
    
    def _check_terminal_growth(
        self,
        dcf_result: Optional[Any],
        ddm_result: Optional[Any],
    ) -> CheckResult:
        """Check terminal growth rates are reasonable."""
        check = CheckResult(
            check_name="terminal_growth_reasonableness",
            category=CheckCategory.VALUATION_CONSISTENCY,
        )
        
        issues = []
        
        if dcf_result:
            dcf_tg = dcf_result.growth_analysis.terminal_growth_rate
            if dcf_tg and dcf_tg > AccuracyConfig.TERMINAL_GROWTH_MAX:
                issues.append(f"DCF terminal growth {dcf_tg*100:.1f}% exceeds {AccuracyConfig.TERMINAL_GROWTH_MAX*100}%")
        
        if ddm_result and ddm_result.is_applicable:
            ddm_tg = ddm_result.growth_analysis.terminal_growth_rate
            if ddm_tg and ddm_tg > AccuracyConfig.TERMINAL_GROWTH_MAX:
                issues.append(f"DDM terminal growth {ddm_tg*100:.1f}% exceeds {AccuracyConfig.TERMINAL_GROWTH_MAX*100}%")
        
        if not issues:
            check.status = VerificationStatus.PASSED
            check.message = "Terminal growth rates within reasonable bounds"
        else:
            check.status = VerificationStatus.WARNING
            check.message = "; ".join(issues)
        
        check.details = {"issues": issues}
        return check
    
    def _check_discount_rate_relationship(
        self,
        dcf_result: Optional[Any],
        ddm_result: Optional[Any],
    ) -> CheckResult:
        """Check WACC <= Cost of Equity (as expected)."""
        check = CheckResult(
            check_name="discount_rate_relationship",
            category=CheckCategory.VALUATION_CONSISTENCY,
        )
        
        if dcf_result and ddm_result and ddm_result.is_applicable:
            wacc = dcf_result.wacc_calculation.wacc_constrained
            ke = ddm_result.cost_of_equity.cost_of_equity
            
            if wacc and ke:
                if wacc <= ke * 1.01:  # Allow 1% tolerance
                    check.status = VerificationStatus.PASSED
                    check.message = f"WACC ({wacc*100:.2f}%) <= Ke ({ke*100:.2f}%) as expected"
                else:
                    check.status = VerificationStatus.WARNING
                    check.message = f"Unexpected: WACC ({wacc*100:.2f}%) > Ke ({ke*100:.2f}%)"
                
                check.details = {"wacc": wacc, "ke": ke}
            else:
                check.status = VerificationStatus.SKIPPED
                check.message = "WACC or Ke not available"
        else:
            check.status = VerificationStatus.SKIPPED
            check.message = "Both DCF and DDM required for comparison"
        
        return check
    
    def _calculate_summary(self, summary: CategorySummary):
        """Calculate summary metrics."""
        for check in summary.checks:
            summary.total_checks += 1
            if check.status == VerificationStatus.PASSED:
                summary.passed += 1
            elif check.status == VerificationStatus.WARNING:
                summary.warnings += 1
            elif check.status == VerificationStatus.FAILED:
                summary.failed += 1
            else:
                summary.skipped += 1
        
        summary.calculate_metrics()


# =============================================================================
# RANGE VALIDATOR
# =============================================================================

class RangeValidator:
    """
    Validates that all values fall within reasonable bounds.
    
    Catches potential data errors or calculation anomalies.
    """
    
    def validate(
        self,
        collection_result: Any,
        dcf_result: Optional[Any] = None,
        ddm_result: Optional[Any] = None,
        multiples_result: Optional[Any] = None,
    ) -> CategorySummary:
        """Run range validation checks."""
        summary = CategorySummary(CheckCategory.RANGE_VALIDITY)
        
        # Check 1: P/E ratio range
        if multiples_result and multiples_result.pe_analysis.current_valid:
            check = self._check_pe_range(multiples_result.pe_analysis.current_value)
            summary.checks.append(check)
        
        # Check 2: P/B ratio range
        if multiples_result and multiples_result.pb_analysis.current_valid:
            check = self._check_pb_range(multiples_result.pb_analysis.current_value)
            summary.checks.append(check)
        
        # Check 3: WACC range
        if dcf_result:
            check = self._check_wacc_range(dcf_result.wacc_calculation.wacc_constrained)
            summary.checks.append(check)
        
        # Check 4: Beta range
        if dcf_result:
            check = self._check_beta_range(dcf_result.wacc_calculation.cost_of_equity.beta)
            summary.checks.append(check)
        
        # Check 5: Growth rate range (DCF)
        if dcf_result:
            check = self._check_growth_range(
                dcf_result.growth_analysis.projection_growth_rate,
                "DCF projection growth"
            )
            summary.checks.append(check)
        
        # Check 6: Growth rate range (DDM)
        if ddm_result and ddm_result.is_applicable:
            check = self._check_growth_range(
                ddm_result.growth_analysis.projection_growth_rate,
                "DDM projection growth"
            )
            summary.checks.append(check)
        
        # Check 7: Intrinsic value positive
        if dcf_result:
            check = self._check_intrinsic_value_positive(
                dcf_result.intrinsic_value_per_share,
                "DCF"
            )
            summary.checks.append(check)
        
        # Check 8: DDM intrinsic value positive
        if ddm_result and ddm_result.is_applicable:
            check = self._check_intrinsic_value_positive(
                ddm_result.intrinsic_value_per_share,
                "DDM"
            )
            summary.checks.append(check)
        
        # Check 9: Terminal value percentage
        if dcf_result:
            check = self._check_terminal_value_percentage(dcf_result)
            summary.checks.append(check)
        
        # Calculate summary metrics
        self._calculate_summary(summary)
        
        return summary
    
    def _check_pe_range(self, pe: float) -> CheckResult:
        """Check P/E ratio is within reasonable range."""
        check = CheckResult(
            check_name="pe_ratio_range",
            category=CheckCategory.RANGE_VALIDITY,
        )
        check.actual_value = pe
        
        if AccuracyConfig.PE_RATIO_MIN < pe < AccuracyConfig.PE_RATIO_MAX:
            check.status = VerificationStatus.PASSED
            check.message = f"P/E ratio {pe:.2f} within normal range"
        elif pe < 0:
            check.status = VerificationStatus.WARNING
            check.message = f"Negative P/E ratio {pe:.2f} (company has losses)"
        else:
            check.status = VerificationStatus.WARNING
            check.message = f"P/E ratio {pe:.2f} outside typical range (0-{AccuracyConfig.PE_RATIO_MAX})"
        
        return check
    
    def _check_pb_range(self, pb: float) -> CheckResult:
        """Check P/B ratio is within reasonable range."""
        check = CheckResult(
            check_name="pb_ratio_range",
            category=CheckCategory.RANGE_VALIDITY,
        )
        check.actual_value = pb
        
        if AccuracyConfig.PB_RATIO_MIN < pb < AccuracyConfig.PB_RATIO_MAX:
            check.status = VerificationStatus.PASSED
            check.message = f"P/B ratio {pb:.2f} within normal range"
        else:
            check.status = VerificationStatus.WARNING
            check.message = f"P/B ratio {pb:.2f} outside typical range (0-{AccuracyConfig.PB_RATIO_MAX})"
        
        return check
    
    def _check_wacc_range(self, wacc: float) -> CheckResult:
        """Check WACC is within reasonable range."""
        check = CheckResult(
            check_name="wacc_range",
            category=CheckCategory.RANGE_VALIDITY,
        )
        check.actual_value = wacc
        
        if AccuracyConfig.WACC_MIN <= wacc <= AccuracyConfig.WACC_MAX:
            check.status = VerificationStatus.PASSED
            check.message = f"WACC {wacc*100:.2f}% within normal range"
        else:
            check.status = VerificationStatus.WARNING
            check.message = f"WACC {wacc*100:.2f}% outside typical range ({AccuracyConfig.WACC_MIN*100}%-{AccuracyConfig.WACC_MAX*100}%)"
        
        return check
    
    def _check_beta_range(self, beta: float) -> CheckResult:
        """Check beta is within reasonable range."""
        check = CheckResult(
            check_name="beta_range",
            category=CheckCategory.RANGE_VALIDITY,
        )
        check.actual_value = beta
        
        if AccuracyConfig.BETA_MIN < beta < AccuracyConfig.BETA_MAX:
            check.status = VerificationStatus.PASSED
            check.message = f"Beta {beta:.2f} within normal range"
        else:
            check.status = VerificationStatus.WARNING
            check.message = f"Beta {beta:.2f} outside typical range (0-{AccuracyConfig.BETA_MAX})"
        
        return check
    
    def _check_growth_range(self, growth: float, label: str) -> CheckResult:
        """Check growth rate is within reasonable range."""
        check = CheckResult(
            check_name=f"{label.lower().replace(' ', '_')}_range",
            category=CheckCategory.RANGE_VALIDITY,
        )
        check.actual_value = growth
        
        if AccuracyConfig.GROWTH_RATE_MIN <= growth <= AccuracyConfig.GROWTH_RATE_MAX:
            check.status = VerificationStatus.PASSED
            check.message = f"{label} {growth*100:.2f}% within normal range"
        else:
            check.status = VerificationStatus.WARNING
            check.message = f"{label} {growth*100:.2f}% outside typical range"
        
        return check
    
    def _check_intrinsic_value_positive(self, value: float, model: str) -> CheckResult:
        """Check intrinsic value is positive."""
        check = CheckResult(
            check_name=f"{model.lower()}_value_positive",
            category=CheckCategory.RANGE_VALIDITY,
        )
        check.actual_value = value
        
        if value > AccuracyConfig.INTRINSIC_VALUE_MIN:
            check.status = VerificationStatus.PASSED
            check.message = f"{model} intrinsic value ${value:.2f} is positive"
        else:
            check.status = VerificationStatus.FAILED
            check.message = f"{model} intrinsic value ${value:.2f} is non-positive"
        
        return check
    
    def _check_terminal_value_percentage(self, dcf_result: Any) -> CheckResult:
        """Check terminal value isn't excessive percentage of total."""
        check = CheckResult(
            check_name="terminal_value_percentage",
            category=CheckCategory.RANGE_VALIDITY,
        )
        
        tv_pct = dcf_result.dcf_projection.terminal_value_pct
        check.actual_value = tv_pct
        
        if tv_pct:
            if tv_pct < 0.85:  # Less than 85%
                check.status = VerificationStatus.PASSED
                check.message = f"Terminal value {tv_pct*100:.1f}% of EV is reasonable"
            elif tv_pct < 0.95:
                check.status = VerificationStatus.WARNING
                check.message = f"Terminal value {tv_pct*100:.1f}% of EV is high"
            else:
                check.status = VerificationStatus.WARNING
                check.message = f"Terminal value {tv_pct*100:.1f}% of EV is very high - valuation sensitive to assumptions"
        else:
            check.status = VerificationStatus.SKIPPED
            check.message = "Terminal value percentage not available"
        
        return check
    
    def _calculate_summary(self, summary: CategorySummary):
        """Calculate summary metrics."""
        for check in summary.checks:
            summary.total_checks += 1
            if check.status == VerificationStatus.PASSED:
                summary.passed += 1
            elif check.status == VerificationStatus.WARNING:
                summary.warnings += 1
            elif check.status == VerificationStatus.FAILED:
                summary.failed += 1
            else:
                summary.skipped += 1
        
        summary.calculate_metrics()


# =============================================================================
# METHODOLOGY COMPLIANCE CHECKER
# =============================================================================

class MethodologyComplianceChecker:
    """
    Verifies calculations follow the implementation plan methodology.
    """
    
    def check(
        self,
        dcf_result: Optional[Any] = None,
        ddm_result: Optional[Any] = None,
        multiples_result: Optional[Any] = None,
    ) -> CategorySummary:
        """Run methodology compliance checks."""
        summary = CategorySummary(CheckCategory.METHODOLOGY_COMPLIANCE)
        
        # Check 1: DCF uses median for growth selection
        if dcf_result:
            check = self._check_dcf_growth_method(dcf_result)
            summary.checks.append(check)
        
        # Check 2: DDM uses minimum for growth selection
        if ddm_result and ddm_result.is_applicable:
            check = self._check_ddm_growth_method(ddm_result)
            summary.checks.append(check)
        
        # Check 3: DCF terminal growth capped at GDP
        if dcf_result:
            check = self._check_terminal_growth_cap(dcf_result, "DCF")
            summary.checks.append(check)
        
        # Check 4: DDM terminal growth capped at GDP
        if ddm_result and ddm_result.is_applicable:
            check = self._check_terminal_growth_cap(ddm_result, "DDM")
            summary.checks.append(check)
        
        # Check 5: Multiples uses +/-20% thresholds
        if multiples_result:
            check = self._check_multiples_thresholds(multiples_result)
            summary.checks.append(check)
        
        # Check 6: 5-year projection period
        if dcf_result:
            check = self._check_projection_period(dcf_result)
            summary.checks.append(check)
        
        # Calculate summary metrics
        self._calculate_summary(summary)
        
        return summary
    
    def _check_dcf_growth_method(self, dcf_result: Any) -> CheckResult:
        """Check DCF uses median for growth selection (per plan)."""
        check = CheckResult(
            check_name="dcf_growth_selection_method",
            category=CheckCategory.METHODOLOGY_COMPLIANCE,
        )
        
        method = dcf_result.growth_analysis.selection_method
        
        if method == "median":
            check.status = VerificationStatus.PASSED
            check.message = "DCF uses median growth selection (per plan)"
        else:
            check.status = VerificationStatus.WARNING
            check.message = f"DCF uses '{method}' instead of median"
        
        check.details = {"method": method}
        return check
    
    def _check_ddm_growth_method(self, ddm_result: Any) -> CheckResult:
        """Check DDM uses minimum for growth selection (per plan)."""
        check = CheckResult(
            check_name="ddm_growth_selection_method",
            category=CheckCategory.METHODOLOGY_COMPLIANCE,
        )
        
        method = ddm_result.growth_analysis.selection_method
        
        if method == "minimum":
            check.status = VerificationStatus.PASSED
            check.message = "DDM uses minimum growth selection (per plan)"
        else:
            check.status = VerificationStatus.WARNING
            check.message = f"DDM uses '{method}' instead of minimum"
        
        check.details = {"method": method}
        return check
    
    def _check_terminal_growth_cap(self, result: Any, model: str) -> CheckResult:
        """Check terminal growth is capped at GDP growth (~2-3%)."""
        check = CheckResult(
            check_name=f"{model.lower()}_terminal_growth_cap",
            category=CheckCategory.METHODOLOGY_COMPLIANCE,
        )
        
        tg = result.growth_analysis.terminal_growth_rate
        
        if tg and tg <= 0.03:  # 3% per plan
            check.status = VerificationStatus.PASSED
            check.message = f"{model} terminal growth {tg*100:.2f}% capped at GDP (per plan)"
        elif tg:
            check.status = VerificationStatus.WARNING
            check.message = f"{model} terminal growth {tg*100:.2f}% exceeds GDP cap"
        else:
            check.status = VerificationStatus.SKIPPED
            check.message = f"{model} terminal growth not available"
        
        return check
    
    def _check_multiples_thresholds(self, multiples_result: Any) -> CheckResult:
        """Check multiples uses +/-20% thresholds (per plan)."""
        check = CheckResult(
            check_name="multiples_valuation_thresholds",
            category=CheckCategory.METHODOLOGY_COMPLIANCE,
        )
        
        # Check if assumptions contain the thresholds
        assumptions = multiples_result.assumptions
        
        ov_threshold = assumptions.get('overvalued_threshold', 0.20)
        uv_threshold = assumptions.get('undervalued_threshold', -0.20)
        
        if ov_threshold == 0.20 and uv_threshold == -0.20:
            check.status = VerificationStatus.PASSED
            check.message = "Multiples uses +/-20% thresholds (per plan)"
        else:
            check.status = VerificationStatus.WARNING
            check.message = f"Non-standard thresholds: +{ov_threshold*100}%/{uv_threshold*100}%"
        
        return check
    
    def _check_projection_period(self, dcf_result: Any) -> CheckResult:
        """Check 5-year projection period is used."""
        check = CheckResult(
            check_name="projection_period",
            category=CheckCategory.METHODOLOGY_COMPLIANCE,
        )
        
        num_years = len(dcf_result.dcf_projection.yearly_projections)
        
        if num_years == 5:
            check.status = VerificationStatus.PASSED
            check.message = "5-year projection period used (per plan)"
        else:
            check.status = VerificationStatus.WARNING
            check.message = f"{num_years}-year projection instead of 5 years"
        
        return check
    
    def _calculate_summary(self, summary: CategorySummary):
        """Calculate summary metrics."""
        for check in summary.checks:
            summary.total_checks += 1
            if check.status == VerificationStatus.PASSED:
                summary.passed += 1
            elif check.status == VerificationStatus.WARNING:
                summary.warnings += 1
            elif check.status == VerificationStatus.FAILED:
                summary.failed += 1
            else:
                summary.skipped += 1
        
        summary.calculate_metrics()


# =============================================================================
# MAIN VALUATOR CLASS
# =============================================================================

class Phase8AccuracyChecker:
    """
    Main orchestrator for Phase 8 Accuracy Verification.
    
    Coordinates all verification components and generates comprehensive report.
    
    Usage:
        checker = Phase8AccuracyChecker()
        result = checker.check(collection_result, validation_result, ...)
    """
    
    def __init__(self):
        self.input_checker = InputIntegrityChecker()
        self.calculation_verifier = CalculationVerifier()
        self.cross_phase_reconciler = CrossPhaseReconciler()
        self.valuation_checker = ValuationConsistencyChecker()
        self.range_validator = RangeValidator()
        self.methodology_checker = MethodologyComplianceChecker()
        self.logger = LOGGER
    
    def check(
        self,
        collection_result: Any,
        validation_result: Optional[Any] = None,
        ratio_result: Optional[Any] = None,
        dupont_result: Optional[Any] = None,
        dcf_result: Optional[Any] = None,
        ddm_result: Optional[Any] = None,
        multiples_result: Optional[Any] = None,
    ) -> AccuracyCheckResult:
        """
        Perform complete accuracy verification.
        
        Args:
            collection_result: CollectionResult from Phase 1
            validation_result: ValidationReport from Phase 2
            ratio_result: RatioAnalysisResult from Phase 3
            dupont_result: DuPontAnalysisResult from Phase 4
            dcf_result: DCFValuationResult from Phase 5
            ddm_result: DDMValuationResult from Phase 6
            multiples_result: MultiplesValuationResult from Phase 7
            
        Returns:
            AccuracyCheckResult with verification status and confidence scores
        """
        self.logger.info(f"Phase 8: Starting accuracy verification for {collection_result.ticker}")
        
        result = AccuracyCheckResult(
            ticker=collection_result.ticker,
            company_name=collection_result.company_name,
        )
        
        # Category 1: Input Integrity
        self.logger.info("  Checking input integrity")
        result.input_integrity = self.input_checker.check(collection_result)
        
        # Category 2: Calculation Accuracy
        self.logger.info("  Verifying calculations")
        result.calculation_accuracy = self.calculation_verifier.verify(
            collection_result, dcf_result, ddm_result, multiples_result
        )
        
        # Category 3: Cross-Phase Consistency
        self.logger.info("  Reconciling cross-phase data")
        result.cross_phase_consistency = self.cross_phase_reconciler.reconcile(
            collection_result, validation_result, ratio_result,
            dupont_result, dcf_result, ddm_result
        )
        
        # Category 4: Valuation Consistency
        self.logger.info("  Checking valuation consistency")
        result.valuation_consistency, result.valuation_input_comparisons = \
            self.valuation_checker.check(dcf_result, ddm_result, multiples_result)
        
        # Category 5: Range Validity
        self.logger.info("  Validating value ranges")
        result.range_validity = self.range_validator.validate(
            collection_result, dcf_result, ddm_result, multiples_result
        )
        
        # Category 6: Methodology Compliance
        self.logger.info("  Checking methodology compliance")
        result.methodology_compliance = self.methodology_checker.check(
            dcf_result, ddm_result, multiples_result
        )
        
        # Calculate overall metrics
        self._calculate_overall_metrics(result)
        
        # Generate recommendation
        self._generate_recommendation(result)
        
        self.logger.info(
            f"Phase 8 complete: {result.overall_pass_rate*100:.1f}% pass rate, "
            f"Confidence: {result.confidence_level.value}"
        )
        
        return result
    
    def _calculate_overall_metrics(self, result: AccuracyCheckResult):
        """Calculate overall metrics from all categories."""
        categories = [
            (result.input_integrity, AccuracyConfig.WEIGHT_INPUT_INTEGRITY),
            (result.calculation_accuracy, AccuracyConfig.WEIGHT_CALCULATION_ACCURACY),
            (result.cross_phase_consistency, AccuracyConfig.WEIGHT_CROSS_PHASE_CONSISTENCY),
            (result.valuation_consistency, AccuracyConfig.WEIGHT_VALUATION_CONSISTENCY),
            (result.range_validity, AccuracyConfig.WEIGHT_RANGE_VALIDITY),
            (result.methodology_compliance, AccuracyConfig.WEIGHT_METHODOLOGY_COMPLIANCE),
        ]
        
        total_checks = 0
        total_passed = 0
        total_warnings = 0
        total_failed = 0
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for cat, weight in categories:
            total_checks += cat.total_checks
            total_passed += cat.passed
            total_warnings += cat.warnings
            total_failed += cat.failed
            
            if cat.total_checks > 0:
                weighted_confidence += cat.confidence_score * weight
                total_weight += weight
        
        result.total_checks = total_checks
        result.total_passed = total_passed
        result.total_warnings = total_warnings
        result.total_failed = total_failed
        
        if total_checks > 0:
            result.overall_pass_rate = total_passed / total_checks
        
        if total_weight > 0:
            result.overall_confidence = weighted_confidence / total_weight
        
        # Determine confidence level
        if result.overall_confidence >= AccuracyConfig.MIN_CONFIDENCE_CRITICAL:
            result.confidence_level = ConfidenceLevel.VERY_HIGH
        elif result.overall_confidence >= AccuracyConfig.MIN_CONFIDENCE_HIGH:
            result.confidence_level = ConfidenceLevel.HIGH
        elif result.overall_confidence >= AccuracyConfig.MIN_CONFIDENCE_ACCEPTABLE:
            result.confidence_level = ConfidenceLevel.ACCEPTABLE
        elif result.overall_confidence >= 0.50:
            result.confidence_level = ConfidenceLevel.LOW
        else:
            result.confidence_level = ConfidenceLevel.VERY_LOW
        
        # Collect critical issues
        for cat, _ in categories:
            for check in cat.checks:
                if check.status == VerificationStatus.FAILED:
                    result.critical_issues.append(f"[{cat.category.value}] {check.message}")
                elif check.status == VerificationStatus.WARNING:
                    result.warnings_list.append(f"[{cat.category.value}] {check.message}")
        
        # Determine reliability
        result.is_reliable = (
            result.total_failed == 0 and
            result.overall_confidence >= AccuracyConfig.MIN_CONFIDENCE_ACCEPTABLE
        )
    
    def _generate_recommendation(self, result: AccuracyCheckResult):
        """Generate overall recommendation."""
        if result.total_failed > 0:
            result.recommendation = (
                f"REVIEW REQUIRED: {result.total_failed} critical issue(s) detected. "
                f"Verify data sources and recalculate affected components before "
                f"generating investment memo."
            )
        elif result.total_warnings > 5:
            result.recommendation = (
                f"PROCEED WITH CAUTION: {result.total_warnings} warning(s) identified. "
                f"Analysis is generally sound but some assumptions should be reviewed."
            )
        elif result.confidence_level in [ConfidenceLevel.VERY_HIGH, ConfidenceLevel.HIGH]:
            result.recommendation = (
                f"ANALYSIS VERIFIED: {result.overall_confidence*100:.1f}% confidence. "
                f"All checks passed or within acceptable tolerances. "
                f"Ready for investment memo generation."
            )
        else:
            result.recommendation = (
                f"ACCEPTABLE: {result.overall_confidence*100:.1f}% confidence. "
                f"Analysis complete with minor caveats. Review warnings before finalizing."
            )
    
    def save_report(
        self,
        result: AccuracyCheckResult,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Save accuracy check report."""
        if output_dir is None:
            output_dir = OUTPUT_DIR
        
        ticker_dir = output_dir / result.ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = ticker_dir / f"{result.ticker}_accuracy_report.json"
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"Saved accuracy report to {filepath}")
        return filepath


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def check_accuracy(
    collection_result: Any,
    validation_result: Optional[Any] = None,
    ratio_result: Optional[Any] = None,
    dupont_result: Optional[Any] = None,
    dcf_result: Optional[Any] = None,
    ddm_result: Optional[Any] = None,
    multiples_result: Optional[Any] = None,
) -> AccuracyCheckResult:
    """Convenience function for accuracy verification."""
    checker = Phase8AccuracyChecker()
    return checker.check(
        collection_result, validation_result, ratio_result,
        dupont_result, dcf_result, ddm_result, multiples_result
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "__version__",
    "AccuracyConfig",
    "VerificationStatus",
    "ConfidenceLevel",
    "CheckCategory",
    "CheckResult",
    "CategorySummary",
    "ValuationInputComparison",
    "AccuracyCheckResult",
    "InputIntegrityChecker",
    "CalculationVerifier",
    "CrossPhaseReconciler",
    "ValuationConsistencyChecker",
    "RangeValidator",
    "MethodologyComplianceChecker",
    "Phase8AccuracyChecker",
    "check_accuracy",
]