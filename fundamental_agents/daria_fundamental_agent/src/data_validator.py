"""
Data Validation Module - Phase 2 Data Validation & Standardization
Fundamental Analyst Agent

Implements comprehensive data validation pipeline including:
- Cross-statement reconciliation (Net Income, D&A consistency)
- Statistical outlier detection (IQR, Z-score methods)
- Sign convention validation
- Growth rate calculations (YoY, CAGR)
- Trend classification
- Data standardization

Inputs: CollectionResult from Phase 1
Outputs: ValidatedData container with quality assessment

MSc Coursework: IFTE0001 AI Agents in Asset Management
Track A: Fundamental Analyst Agent
Phase 2: Data Validation & Standardization

Version: 1.0.0
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from .config import (
    LOGGER,
    OUTPUT_DIR,
    ReconciliationStatus,
    TrendClassification,
    OutlierSeverity,
    SignConvention,
    RECONCILIATION_CONFIG,
    OUTLIER_CONFIG,
    GROWTH_CONFIG,
    PHASE2_VALIDATION_THRESHOLDS,
    CROSS_STATEMENT_RECONCILIATION_PAIRS,
    KEY_GROWTH_METRICS,
    get_sign_convention,
    classify_trend,
)


__version__ = "1.0.0"


# =============================================================================
# DATA CONTAINERS
# =============================================================================

@dataclass
class ReconciliationResult:
    """Result of a single cross-statement reconciliation check."""
    
    metric_name: str
    statement_a_name: str
    statement_b_name: str
    value_a: Optional[float]
    value_b: Optional[float]
    variance: Optional[float]
    variance_percent: Optional[float]
    status: ReconciliationStatus
    fiscal_year: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "statement_a": self.statement_a_name,
            "statement_b": self.statement_b_name,
            "value_a": self.value_a,
            "value_b": self.value_b,
            "variance": self.variance,
            "variance_percent": self.variance_percent,
            "status": self.status.value,
            "fiscal_year": self.fiscal_year,
        }


@dataclass
class ReconciliationReport:
    """Comprehensive cross-statement reconciliation report."""
    
    results: List[ReconciliationResult] = field(default_factory=list)
    reconciliation_score: float = 0.0
    total_checks: int = 0
    passed_checks: int = 0
    minor_variances: int = 0
    major_variances: int = 0
    failed_checks: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reconciliation_score": self.reconciliation_score,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "minor_variances": self.minor_variances,
            "major_variances": self.major_variances,
            "failed_checks": self.failed_checks,
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class OutlierResult:
    """Result of outlier detection for a single metric."""
    
    field_name: str
    fiscal_year: str
    value: float
    z_score: Optional[float]
    iqr_score: Optional[float]
    severity: OutlierSeverity
    yoy_change: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "fiscal_year": self.fiscal_year,
            "value": self.value,
            "z_score": self.z_score,
            "iqr_score": self.iqr_score,
            "severity": self.severity.value,
            "yoy_change": self.yoy_change,
        }


@dataclass
class OutlierReport:
    """Comprehensive outlier detection report."""
    
    outliers: List[OutlierResult] = field(default_factory=list)
    total_values_analyzed: int = 0
    outlier_count: int = 0
    outlier_percentage: float = 0.0
    mild_outliers: int = 0
    moderate_outliers: int = 0
    extreme_outliers: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_values_analyzed": self.total_values_analyzed,
            "outlier_count": self.outlier_count,
            "outlier_percentage": self.outlier_percentage,
            "mild_outliers": self.mild_outliers,
            "moderate_outliers": self.moderate_outliers,
            "extreme_outliers": self.extreme_outliers,
            "outliers": [o.to_dict() for o in self.outliers],
        }


@dataclass
class GrowthMetric:
    """Growth rate analysis for a single metric."""
    
    field_name: str
    yoy_growth: Dict[str, Optional[float]] = field(default_factory=dict)
    cagr: Optional[float] = None
    average_growth: Optional[float] = None
    growth_volatility: Optional[float] = None
    coefficient_of_variation: Optional[float] = None
    trend: TrendClassification = TrendClassification.INSUFFICIENT_DATA
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "yoy_growth": self.yoy_growth,
            "cagr": self.cagr,
            "average_growth": self.average_growth,
            "growth_volatility": self.growth_volatility,
            "coefficient_of_variation": self.coefficient_of_variation,
            "trend": self.trend.value,
        }


@dataclass
class GrowthReport:
    """Comprehensive growth rate analysis report."""
    
    metrics: Dict[str, GrowthMetric] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {"metrics": {k: v.to_dict() for k, v in self.metrics.items()}}


@dataclass
class SignValidationResult:
    """Result of sign convention validation for a field."""
    
    field_name: str
    statement_type: str
    expected_sign: SignConvention
    violations: List[Tuple[str, float]] = field(default_factory=list)
    is_valid: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "statement_type": self.statement_type,
            "expected_sign": self.expected_sign.value,
            "violations": self.violations,
            "is_valid": self.is_valid,
        }


@dataclass
class SignValidationReport:
    """Comprehensive sign convention validation report."""
    
    results: List[SignValidationResult] = field(default_factory=list)
    total_fields_checked: int = 0
    valid_fields: int = 0
    validation_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_fields_checked": self.total_fields_checked,
            "valid_fields": self.valid_fields,
            "validation_score": self.validation_score,
            "results": [r.to_dict() for r in self.results if not r.is_valid],
        }


@dataclass
class StandardizedStatements:
    """Standardized financial statements with quality annotations."""
    
    income_statement: pd.DataFrame
    balance_sheet: pd.DataFrame
    cash_flow: pd.DataFrame
    quality_flags: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    @property
    def fiscal_periods(self) -> List[str]:
        if self.income_statement is not None and not self.income_statement.empty:
            return list(self.income_statement.columns)
        return []


@dataclass
class ValidationSummary:
    """Overall data validation summary."""
    
    overall_score: float = 0.0
    reconciliation_score: float = 0.0
    consistency_score: float = 0.0
    completeness_score: float = 0.0
    quality_tier: str = "poor"
    
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "reconciliation_score": self.reconciliation_score,
            "consistency_score": self.consistency_score,
            "completeness_score": self.completeness_score,
            "quality_tier": self.quality_tier,
            "issues": self.issues,
            "warnings": self.warnings,
            "info": self.info,
        }


@dataclass
class ValidatedData:
    """Complete output of Phase 2 validation pipeline."""
    
    ticker: str
    company_name: str
    fiscal_periods: List[str]
    statements: StandardizedStatements
    reconciliation_report: ReconciliationReport
    outlier_report: OutlierReport
    growth_report: GrowthReport
    sign_validation_report: SignValidationReport
    validation_summary: ValidationSummary
    validation_timestamp: datetime = field(default_factory=datetime.now)
    phase1_source: str = "Alpha Vantage"
    
    @property
    def is_valid(self) -> bool:
        return self.validation_summary.overall_score >= PHASE2_VALIDATION_THRESHOLDS.acceptable_threshold
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "fiscal_periods": self.fiscal_periods,
            "reconciliation_report": self.reconciliation_report.to_dict(),
            "outlier_report": self.outlier_report.to_dict(),
            "growth_report": self.growth_report.to_dict(),
            "sign_validation_report": self.sign_validation_report.to_dict(),
            "validation_summary": self.validation_summary.to_dict(),
            "validation_timestamp": self.validation_timestamp.isoformat(),
            "phase1_source": self.phase1_source,
        }


# =============================================================================
# CROSS-STATEMENT RECONCILER
# =============================================================================

class CrossStatementReconciler:
    """Validates consistency between financial statements."""
    
    def reconcile(
        self,
        income_statement: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cash_flow: pd.DataFrame,
    ) -> ReconciliationReport:
        """Perform all cross-statement reconciliation checks."""
        report = ReconciliationReport()
        
        statements = {
            "income_statement": income_statement,
            "balance_sheet": balance_sheet,
            "cash_flow": cash_flow,
        }
        
        periods = self._get_common_periods(statements)
        if not periods:
            return report
        
        # Run configured reconciliation checks
        for check in CROSS_STATEMENT_RECONCILIATION_PAIRS:
            stmt_a_name, field_a = check["statement_a"]
            stmt_b_name, field_b = check["statement_b"]
            tolerance = check["tolerance"]
            
            stmt_a = statements.get(stmt_a_name)
            stmt_b = statements.get(stmt_b_name)
            
            if stmt_a is None or stmt_b is None:
                continue
            
            for year in periods:
                result = self._reconcile_field(
                    metric_name=check["name"],
                    df_a=stmt_a,
                    df_b=stmt_b,
                    field_a=field_a,
                    field_b=field_b,
                    stmt_a_name=stmt_a_name,
                    stmt_b_name=stmt_b_name,
                    year=year,
                    tolerance=tolerance,
                )
                report.results.append(result)
                report.total_checks += 1
                
                if result.status == ReconciliationStatus.RECONCILED:
                    report.passed_checks += 1
                elif result.status == ReconciliationStatus.MINOR_VARIANCE:
                    report.minor_variances += 1
                elif result.status == ReconciliationStatus.MAJOR_VARIANCE:
                    report.major_variances += 1
                else:
                    report.failed_checks += 1
        
        # Add accounting equation check
        acct_results = self._check_accounting_equation(balance_sheet, periods)
        for result in acct_results:
            report.results.append(result)
            report.total_checks += 1
            if result.status == ReconciliationStatus.RECONCILED:
                report.passed_checks += 1
            elif result.status == ReconciliationStatus.MINOR_VARIANCE:
                report.minor_variances += 1
            else:
                report.major_variances += 1
        
        # Add cash flow articulation check
        cf_results = self._check_cash_flow_articulation(cash_flow, periods)
        for result in cf_results:
            report.results.append(result)
            report.total_checks += 1
            if result.status == ReconciliationStatus.RECONCILED:
                report.passed_checks += 1
            elif result.status == ReconciliationStatus.MINOR_VARIANCE:
                report.minor_variances += 1
            else:
                report.major_variances += 1
        
        # Calculate reconciliation score
        if report.total_checks > 0:
            score = (
                report.passed_checks * 1.0 +
                report.minor_variances * 0.8 +
                report.major_variances * 0.4
            ) / report.total_checks
            report.reconciliation_score = score
        
        return report
    
    def _get_common_periods(self, statements: Dict[str, pd.DataFrame]) -> List[str]:
        """Get fiscal periods common to all non-empty statements."""
        period_sets = []
        for df in statements.values():
            if df is not None and not df.empty:
                period_sets.append(set(df.columns))
        
        if not period_sets:
            return []
        
        common = period_sets[0]
        for ps in period_sets[1:]:
            common = common.intersection(ps)
        
        return sorted(list(common), reverse=True)
    
    def _reconcile_field(
        self,
        metric_name: str,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        field_a: str,
        field_b: str,
        stmt_a_name: str,
        stmt_b_name: str,
        year: str,
        tolerance: float,
    ) -> ReconciliationResult:
        """Reconcile a single field between two statements."""
        value_a = self._get_value(df_a, field_a, year)
        value_b = self._get_value(df_b, field_b, year)
        
        if value_a is None or value_b is None:
            return ReconciliationResult(
                metric_name=metric_name,
                statement_a_name=stmt_a_name,
                statement_b_name=stmt_b_name,
                value_a=value_a,
                value_b=value_b,
                variance=None,
                variance_percent=None,
                status=ReconciliationStatus.CANNOT_RECONCILE,
                fiscal_year=year,
            )
        
        variance = abs(value_a - value_b)
        base = max(abs(value_a), abs(value_b), 1)
        variance_percent = variance / base
        
        if variance_percent <= tolerance:
            status = ReconciliationStatus.RECONCILED
        elif variance_percent <= RECONCILIATION_CONFIG.minor_variance_threshold:
            status = ReconciliationStatus.MINOR_VARIANCE
        else:
            status = ReconciliationStatus.MAJOR_VARIANCE
        
        return ReconciliationResult(
            metric_name=metric_name,
            statement_a_name=stmt_a_name,
            statement_b_name=stmt_b_name,
            value_a=value_a,
            value_b=value_b,
            variance=variance,
            variance_percent=variance_percent,
            status=status,
            fiscal_year=year,
        )
    
    def _check_accounting_equation(
        self,
        balance_sheet: pd.DataFrame,
        periods: List[str],
    ) -> List[ReconciliationResult]:
        """Verify Assets = Liabilities + Equity for all periods."""
        results = []
        
        if balance_sheet is None or balance_sheet.empty:
            return results
        
        for year in periods:
            assets = self._get_value(balance_sheet, "total_assets", year)
            liabilities = self._get_value(balance_sheet, "total_liabilities", year)
            equity = self._get_value(balance_sheet, "total_equity", year)
            
            if assets is None or liabilities is None or equity is None:
                results.append(ReconciliationResult(
                    metric_name="Accounting Equation",
                    statement_a_name="total_assets",
                    statement_b_name="liabilities + equity",
                    value_a=assets,
                    value_b=(liabilities + equity) if liabilities and equity else None,
                    variance=None,
                    variance_percent=None,
                    status=ReconciliationStatus.CANNOT_RECONCILE,
                    fiscal_year=year,
                ))
                continue
            
            calculated = liabilities + equity
            variance = abs(assets - calculated)
            variance_percent = variance / max(abs(assets), 1)
            
            if variance_percent <= RECONCILIATION_CONFIG.accounting_equation_tolerance:
                status = ReconciliationStatus.RECONCILED
            elif variance_percent <= RECONCILIATION_CONFIG.minor_variance_threshold:
                status = ReconciliationStatus.MINOR_VARIANCE
            else:
                status = ReconciliationStatus.MAJOR_VARIANCE
            
            results.append(ReconciliationResult(
                metric_name="Accounting Equation",
                statement_a_name="total_assets",
                statement_b_name="liabilities + equity",
                value_a=assets,
                value_b=calculated,
                variance=variance,
                variance_percent=variance_percent,
                status=status,
                fiscal_year=year,
            ))
        
        return results
    
    def _check_cash_flow_articulation(
        self,
        cash_flow: pd.DataFrame,
        periods: List[str],
    ) -> List[ReconciliationResult]:
        """Verify cash flow articulation: OCF + ICF + FCF = Change in Cash."""
        results = []
        
        if cash_flow is None or cash_flow.empty:
            return results
        
        for year in periods:
            ocf = self._get_value(cash_flow, "operating_cash_flow", year)
            icf = self._get_value(cash_flow, "investing_cash_flow", year)
            fcf = self._get_value(cash_flow, "financing_cash_flow", year)
            change_cash = self._get_value(cash_flow, "change_in_cash", year)
            
            if any(v is None for v in [ocf, icf, fcf, change_cash]):
                continue
            
            calculated = ocf + icf + fcf
            variance = abs(calculated - change_cash)
            variance_percent = variance / max(abs(change_cash), abs(calculated), 1)
            
            if variance_percent <= RECONCILIATION_CONFIG.cash_flow_articulation_tolerance:
                status = ReconciliationStatus.RECONCILED
            elif variance_percent <= RECONCILIATION_CONFIG.minor_variance_threshold:
                status = ReconciliationStatus.MINOR_VARIANCE
            else:
                status = ReconciliationStatus.MAJOR_VARIANCE
            
            results.append(ReconciliationResult(
                metric_name="Cash Flow Articulation",
                statement_a_name="OCF + ICF + FCF",
                statement_b_name="change_in_cash",
                value_a=calculated,
                value_b=change_cash,
                variance=variance,
                variance_percent=variance_percent,
                status=status,
                fiscal_year=year,
            ))
        
        return results
    
    def _get_value(self, df: pd.DataFrame, field: str, year: str) -> Optional[float]:
        """Safely get value from DataFrame."""
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


# =============================================================================
# OUTLIER DETECTOR
# =============================================================================

class OutlierDetector:
    """Detects statistical outliers in financial time series data."""
    
    def detect(
        self,
        income_statement: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cash_flow: pd.DataFrame,
    ) -> OutlierReport:
        """Detect outliers across all financial statements."""
        report = OutlierReport()
        
        statements = [
            ("income_statement", income_statement),
            ("balance_sheet", balance_sheet),
            ("cash_flow", cash_flow),
        ]
        
        for stmt_name, df in statements:
            if df is None or df.empty:
                continue
            
            outliers, count = self._detect_in_statement(df)
            report.outliers.extend(outliers)
            report.total_values_analyzed += count
        
        report.outlier_count = len(report.outliers)
        if report.total_values_analyzed > 0:
            report.outlier_percentage = report.outlier_count / report.total_values_analyzed
        
        report.mild_outliers = sum(1 for o in report.outliers if o.severity == OutlierSeverity.MILD)
        report.moderate_outliers = sum(1 for o in report.outliers if o.severity == OutlierSeverity.MODERATE)
        report.extreme_outliers = sum(1 for o in report.outliers if o.severity == OutlierSeverity.EXTREME)
        
        return report
    
    def _detect_in_statement(self, df: pd.DataFrame) -> Tuple[List[OutlierResult], int]:
        """Detect outliers in a single statement."""
        outliers = []
        total_values = 0
        
        for field in df.index:
            series = df.loc[field].dropna()
            
            if len(series) < OUTLIER_CONFIG.min_data_points:
                continue
            
            total_values += len(series)
            
            values = series.values.astype(float)
            mean = np.mean(values)
            std = np.std(values)
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            years = list(series.index)
            for i, year in enumerate(years):
                value = values[i]
                
                z_score = (value - mean) / std if std > 0 else 0
                
                if iqr > 0:
                    if value < q1:
                        iqr_score = (q1 - value) / iqr
                    elif value > q3:
                        iqr_score = (value - q3) / iqr
                    else:
                        iqr_score = 0
                else:
                    iqr_score = 0
                
                yoy_change = None
                if i < len(years) - 1:
                    prior_value = values[i + 1]
                    if abs(prior_value) > 0:
                        yoy_change = (value - prior_value) / abs(prior_value)
                
                severity = self._classify_severity(abs(z_score), iqr_score, yoy_change)
                
                if severity != OutlierSeverity.NONE:
                    outliers.append(OutlierResult(
                        field_name=field,
                        fiscal_year=year,
                        value=value,
                        z_score=z_score,
                        iqr_score=iqr_score,
                        severity=severity,
                        yoy_change=yoy_change,
                    ))
        
        return outliers, total_values
    
    def _classify_severity(
        self,
        abs_zscore: float,
        iqr_score: float,
        yoy_change: Optional[float],
    ) -> OutlierSeverity:
        """Classify outlier severity based on statistical measures."""
        
        if abs_zscore >= OUTLIER_CONFIG.zscore_extreme_threshold:
            return OutlierSeverity.EXTREME
        elif abs_zscore >= OUTLIER_CONFIG.zscore_moderate_threshold:
            return OutlierSeverity.MODERATE
        elif abs_zscore >= OUTLIER_CONFIG.zscore_mild_threshold:
            return OutlierSeverity.MILD
        
        if iqr_score >= OUTLIER_CONFIG.iqr_extreme_threshold:
            return OutlierSeverity.EXTREME
        elif iqr_score >= OUTLIER_CONFIG.iqr_moderate_threshold:
            return OutlierSeverity.MODERATE
        elif iqr_score >= OUTLIER_CONFIG.iqr_mild_threshold:
            return OutlierSeverity.MILD
        
        if yoy_change is not None:
            abs_yoy = abs(yoy_change)
            if abs_yoy >= OUTLIER_CONFIG.yoy_change_critical_threshold:
                return OutlierSeverity.MODERATE
            elif abs_yoy >= OUTLIER_CONFIG.yoy_change_flag_threshold:
                return OutlierSeverity.MILD
        
        return OutlierSeverity.NONE


# =============================================================================
# GROWTH ANALYZER
# =============================================================================

class GrowthAnalyzer:
    """Calculates growth rates and trend classifications for financial metrics."""
    
    def analyze(
        self,
        income_statement: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cash_flow: pd.DataFrame,
        derived_metrics: Dict[str, Dict[str, Optional[float]]],
    ) -> GrowthReport:
        """Analyze growth rates for all key metrics."""
        report = GrowthReport()
        
        # Analyze statement metrics
        for field in KEY_GROWTH_METRICS:
            if income_statement is not None and field in income_statement.index:
                series = income_statement.loc[field].dropna()
                report.metrics[field] = self._analyze_series(field, series)
            elif balance_sheet is not None and field in balance_sheet.index:
                series = balance_sheet.loc[field].dropna()
                report.metrics[field] = self._analyze_series(field, series)
            elif cash_flow is not None and field in cash_flow.index:
                series = cash_flow.loc[field].dropna()
                report.metrics[field] = self._analyze_series(field, series)
        
        # Analyze derived metrics
        for metric_name, values in derived_metrics.items():
            if metric_name in KEY_GROWTH_METRICS and metric_name not in report.metrics:
                clean_values = {k: v for k, v in values.items() if v is not None}
                if clean_values:
                    series = pd.Series(clean_values)
                    report.metrics[metric_name] = self._analyze_series(metric_name, series)
        
        return report
    
    def _analyze_series(self, field_name: str, series: pd.Series) -> GrowthMetric:
        """Analyze growth for a single time series."""
        metric = GrowthMetric(field_name=field_name)
        
        if len(series) < 2:
            return metric
        
        series = series.sort_index(ascending=True)
        years = list(series.index)
        values = series.values.astype(float)
        
        # Calculate YoY growth rates
        for i in range(1, len(values)):
            current_year = years[i]
            prior_value = values[i - 1]
            current_value = values[i]
            
            if abs(prior_value) > 0:
                yoy = (current_value - prior_value) / abs(prior_value)
                metric.yoy_growth[current_year] = yoy
        
        # Calculate CAGR
        if len(values) >= GROWTH_CONFIG.min_years_for_cagr:
            start_value = values[0]
            end_value = values[-1]
            n_years = len(values) - 1
            
            if start_value > 0 and end_value > 0:
                metric.cagr = (end_value / start_value) ** (1 / n_years) - 1
            elif start_value < 0 and end_value < 0:
                metric.cagr = -((abs(end_value) / abs(start_value)) ** (1 / n_years) - 1)
        
        # Calculate average growth and volatility
        growth_rates = list(metric.yoy_growth.values())
        if growth_rates:
            metric.average_growth = np.mean(growth_rates)
            metric.growth_volatility = np.std(growth_rates)
        
        # Calculate coefficient of variation
        mean_value = np.mean(values)
        std_value = np.std(values)
        if abs(mean_value) > 0:
            metric.coefficient_of_variation = std_value / abs(mean_value)
        
        # Classify trend
        if len(values) >= GROWTH_CONFIG.min_years_for_trend:
            if metric.cagr is not None and metric.coefficient_of_variation is not None:
                metric.trend = classify_trend(metric.cagr, metric.coefficient_of_variation)
        
        return metric


# =============================================================================
# SIGN VALIDATOR
# =============================================================================

class SignValidator:
    """Validates that financial metrics have expected sign conventions."""
    
    def validate(
        self,
        income_statement: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cash_flow: pd.DataFrame,
    ) -> SignValidationReport:
        """Validate sign conventions across all statements."""
        report = SignValidationReport()
        
        statements = [
            ("income_statement", income_statement),
            ("balance_sheet", balance_sheet),
            ("cash_flow", cash_flow),
        ]
        
        for stmt_name, df in statements:
            if df is None or df.empty:
                continue
            
            for field in df.index:
                expected = get_sign_convention(stmt_name, field)
                result = self._validate_field(df, field, stmt_name, expected)
                report.results.append(result)
                report.total_fields_checked += 1
                
                if result.is_valid:
                    report.valid_fields += 1
        
        if report.total_fields_checked > 0:
            report.validation_score = report.valid_fields / report.total_fields_checked
        
        return report
    
    def _validate_field(
        self,
        df: pd.DataFrame,
        field: str,
        stmt_name: str,
        expected: SignConvention,
    ) -> SignValidationResult:
        """Validate sign convention for a single field."""
        result = SignValidationResult(
            field_name=field,
            statement_type=stmt_name,
            expected_sign=expected,
        )
        
        if expected == SignConvention.EITHER:
            return result
        
        series = df.loc[field].dropna()
        
        for year, value in series.items():
            violation = False
            
            if expected == SignConvention.POSITIVE and value < 0:
                violation = True
            elif expected == SignConvention.NEGATIVE and value > 0:
                violation = True
            elif expected == SignConvention.INFLOW and value < 0:
                violation = True
            elif expected == SignConvention.OUTFLOW and value > 0:
                violation = True
            
            if violation:
                result.violations.append((year, float(value)))
                result.is_valid = False
        
        return result


# =============================================================================
# DATA STANDARDIZER
# =============================================================================

class DataStandardizer:
    """Standardizes financial data for downstream analysis."""
    
    def standardize(
        self,
        income_statement: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cash_flow: pd.DataFrame,
        outlier_report: OutlierReport,
    ) -> StandardizedStatements:
        """Standardize all financial statements."""
        std_income = income_statement.copy() if income_statement is not None else pd.DataFrame()
        std_balance = balance_sheet.copy() if balance_sheet is not None else pd.DataFrame()
        std_cashflow = cash_flow.copy() if cash_flow is not None else pd.DataFrame()
        
        quality_flags = self._build_quality_flags(outlier_report)
        
        return StandardizedStatements(
            income_statement=std_income,
            balance_sheet=std_balance,
            cash_flow=std_cashflow,
            quality_flags=quality_flags,
        )
    
    def _build_quality_flags(self, outlier_report: OutlierReport) -> Dict[str, Dict[str, str]]:
        """Build quality flag dictionary from outlier report."""
        flags = {}
        
        for outlier in outlier_report.outliers:
            field = outlier.field_name
            year = outlier.fiscal_year
            
            if field not in flags:
                flags[field] = {}
            
            if outlier.severity == OutlierSeverity.EXTREME:
                flags[field][year] = "EXTREME_OUTLIER"
            elif outlier.severity == OutlierSeverity.MODERATE:
                flags[field][year] = "MODERATE_OUTLIER"
            elif outlier.severity == OutlierSeverity.MILD:
                flags[field][year] = "MILD_OUTLIER"
        
        return flags


# =============================================================================
# MAIN DATA VALIDATOR (PHASE 2)
# =============================================================================

class Phase2Validator:
    """
    Main orchestrator for Phase 2 data validation pipeline.
    
    Coordinates all validation components and produces ValidatedData
    ready for Phase 3 ratio analysis.
    """
    
    def __init__(self):
        """Initialize validator components."""
        self.reconciler = CrossStatementReconciler()
        self.outlier_detector = OutlierDetector()
        self.growth_analyzer = GrowthAnalyzer()
        self.sign_validator = SignValidator()
        self.standardizer = DataStandardizer()
        
        LOGGER.info(f"Phase2Validator initialized (v{__version__})")
    
    def validate(self, collection_result) -> ValidatedData:
        """
        Perform comprehensive validation on Phase 1 collection result.
        
        Args:
            collection_result: CollectionResult from Phase 1
            
        Returns:
            ValidatedData container with all validation results
        """
        LOGGER.info(f"Phase 2: Starting validation for {collection_result.ticker}")
        
        income = collection_result.statements.income_statement
        balance = collection_result.statements.balance_sheet
        cashflow = collection_result.statements.cash_flow
        
        derived = {
            "fcf_calculated": collection_result.derived_metrics.fcf_calculated,
            "ebitda_calculated": collection_result.derived_metrics.ebitda_calculated,
            "working_capital": collection_result.derived_metrics.working_capital,
            "net_debt": collection_result.derived_metrics.net_debt,
            "invested_capital": collection_result.derived_metrics.invested_capital,
        }
        
        LOGGER.info("  Running cross-statement reconciliation")
        reconciliation_report = self.reconciler.reconcile(income, balance, cashflow)
        
        LOGGER.info("  Running outlier detection")
        outlier_report = self.outlier_detector.detect(income, balance, cashflow)
        
        LOGGER.info("  Running growth analysis")
        growth_report = self.growth_analyzer.analyze(income, balance, cashflow, derived)
        
        LOGGER.info("  Running sign validation")
        sign_report = self.sign_validator.validate(income, balance, cashflow)
        
        LOGGER.info("  Standardizing data")
        standardized = self.standardizer.standardize(income, balance, cashflow, outlier_report)
        
        summary = self._generate_summary(
            reconciliation_report,
            outlier_report,
            sign_report,
            collection_result.quality_metrics.overall_completeness,
        )
        
        LOGGER.info(
            f"Phase 2 complete for {collection_result.ticker}: "
            f"score={summary.overall_score:.2%}, tier={summary.quality_tier}"
        )
        
        return ValidatedData(
            ticker=collection_result.ticker,
            company_name=collection_result.company_name,
            fiscal_periods=collection_result.statements.fiscal_periods,
            statements=standardized,
            reconciliation_report=reconciliation_report,
            outlier_report=outlier_report,
            growth_report=growth_report,
            sign_validation_report=sign_report,
            validation_summary=summary,
            phase1_source=collection_result.data_source,
        )
    
    def _generate_summary(
        self,
        reconciliation_report: ReconciliationReport,
        outlier_report: OutlierReport,
        sign_report: SignValidationReport,
        completeness: float,
    ) -> ValidationSummary:
        """Generate overall validation summary."""
        summary = ValidationSummary()
        
        summary.reconciliation_score = reconciliation_report.reconciliation_score
        summary.completeness_score = completeness
        
        outlier_penalty = min(outlier_report.outlier_percentage * 2, 0.3)
        summary.consistency_score = sign_report.validation_score * (1 - outlier_penalty)
        
        summary.overall_score = (
            summary.reconciliation_score * 0.30 +
            summary.consistency_score * 0.30 +
            summary.completeness_score * 0.40
        )
        
        if summary.overall_score >= PHASE2_VALIDATION_THRESHOLDS.excellent_threshold:
            summary.quality_tier = "excellent"
        elif summary.overall_score >= PHASE2_VALIDATION_THRESHOLDS.good_threshold:
            summary.quality_tier = "good"
        elif summary.overall_score >= PHASE2_VALIDATION_THRESHOLDS.acceptable_threshold:
            summary.quality_tier = "acceptable"
        else:
            summary.quality_tier = "poor"
        
        # Helper for singular/plural
        def pluralize(count, singular, plural):
            return singular if count == 1 else plural
        
        if reconciliation_report.major_variances > 0:
            n = reconciliation_report.major_variances
            summary.warnings.append(
                f"{n} major reconciliation {pluralize(n, 'variance', 'variances')}"
            )
        
        if outlier_report.extreme_outliers > 0:
            n = outlier_report.extreme_outliers
            summary.warnings.append(f"{n} extreme {pluralize(n, 'outlier', 'outliers')}")
        
        if outlier_report.moderate_outliers > 0:
            n = outlier_report.moderate_outliers
            summary.info.append(f"{n} moderate {pluralize(n, 'outlier', 'outliers')}")
        
        if sign_report.validation_score < 1.0:
            n = sign_report.total_fields_checked - sign_report.valid_fields
            summary.info.append(f"{n} {pluralize(n, 'field', 'fields')} with unexpected signs")
        
        return summary
    
    def save_report(self, validated_data: ValidatedData, output_dir: Path = None) -> Path:
        """Save validation report to JSON file."""
        if output_dir is None:
            output_dir = OUTPUT_DIR
        
        ticker_dir = output_dir / validated_data.ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = ticker_dir / f"{validated_data.ticker}_validation_report.json"
        
        with open(filepath, "w") as f:
            json.dump(validated_data.to_dict(), f, indent=2, default=str)
        
        LOGGER.info(f"Saved validation report to {filepath}")
        return filepath


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def validate_financial_data(collection_result) -> ValidatedData:
    """Convenience function to validate Phase 1 data."""
    validator = Phase2Validator()
    return validator.validate(collection_result)