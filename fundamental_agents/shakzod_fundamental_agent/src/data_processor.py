"""
Data Processor Module for Fundamental Analyst Agent

This module transforms raw financial data from Yahoo Finance into a clean,
standardized format suitable for analysis. It serves as the critical bridge
between raw data collection and downstream analytical modules.

Module Responsibilities:
    1. Field Standardization: Map Yahoo Finance field names to consistent internal names
    2. Missing Value Handling: Identify, flag, and where possible, estimate missing values
    3. Derived Field Calculation: Compute fields not directly available (EBITDA, FCF, etc.)
    4. Data Alignment: Ensure all statements cover the same fiscal periods
    5. Unit Normalization: Convert all monetary values to millions USD
    6. Integrity Validation: Check accounting identities and flag anomalies
    7. Audit Trail: Maintain a record of all transformations applied

Design Principles:
    - Robustness: Handle partial data gracefully without crashing
    - Transparency: Maintain detailed audit trail of all transformations
    - Consistency: Use StandardField enum for all field references
    - Quality Metrics: Provide comprehensive data quality assessment
    - Backward Compatibility: Support all downstream module interfaces

MSc Coursework: AI Agents in Asset Management
Track A: Fundamental Analyst Agent

Dependencies:
    - pandas: Data manipulation
    - numpy: Numerical operations
    - config: Configuration constants
    - data_collector: Raw data structures

Example Usage:
    >>> from data_collector import collect_financial_data
    >>> from data_processor import DataProcessor, StandardField
    >>> 
    >>> raw_data = collect_financial_data("AAPL")
    >>> processor = DataProcessor()
    >>> processed = processor.process(raw_data)
    >>> 
    >>> revenue = processor.get_field(processed, StandardField.REVENUE, 0)
    >>> print(f"Revenue: ${revenue:,.1f}M")

Author: MSc AI Agents in Asset Management
Version: 1.0.0
"""

# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from enum import Enum

import pandas as pd
import numpy as np

from config import (
    VALIDATION,
    INCOME_STATEMENT_FIELDS,
    BALANCE_SHEET_FIELDS,
    CASH_FLOW_FIELDS,
    ValidationStatus,
    get_field_alternatives
)
from data_collector import (
    CollectedData,
    FinancialStatements,
    CompanyInfo,
    DataCollector
)


# =============================================================================
# MODULE CONFIGURATION
# =============================================================================

logger = logging.getLogger(__name__)

DEFAULT_SCALE_FACTOR = 1e6
DEFAULT_CURRENCY = "USD"
HIGH_QUALITY_COVERAGE_THRESHOLD = 0.80
MINIMUM_COVERAGE_THRESHOLD = 0.50


# =============================================================================
# STANDARD FIELD ENUMERATION
# =============================================================================

class StandardField(Enum):
    """
    Enumeration of standard field names used throughout the analysis pipeline.
    
    Using an enum ensures consistency across all modules and enables IDE
    autocompletion. The value is the string key used in processed DataFrames.
    """
    # Income Statement Fields
    REVENUE = "revenue"
    COST_OF_REVENUE = "cost_of_revenue"
    GROSS_PROFIT = "gross_profit"
    OPERATING_EXPENSES = "operating_expenses"
    RD_EXPENSE = "rd_expense"
    SGA_EXPENSE = "sga_expense"
    OPERATING_INCOME = "operating_income"
    INTEREST_EXPENSE = "interest_expense"
    PRETAX_INCOME = "pretax_income"
    TAX_EXPENSE = "tax_expense"
    NET_INCOME = "net_income"
    EBITDA = "ebitda"
    DEPRECIATION = "depreciation"
    EPS = "eps"
    SHARES_OUTSTANDING = "shares_outstanding"
    
    # Balance Sheet Fields - Assets
    TOTAL_ASSETS = "total_assets"
    CURRENT_ASSETS = "current_assets"
    CASH = "cash"
    SHORT_TERM_INVESTMENTS = "short_term_investments"
    ACCOUNTS_RECEIVABLE = "accounts_receivable"
    INVENTORY = "inventory"
    OTHER_CURRENT_ASSETS = "other_current_assets"
    PPE_NET = "ppe_net"
    GOODWILL = "goodwill"
    INTANGIBLES = "intangibles"
    
    # Balance Sheet Fields - Liabilities
    TOTAL_LIABILITIES = "total_liabilities"
    CURRENT_LIABILITIES = "current_liabilities"
    ACCOUNTS_PAYABLE = "accounts_payable"
    SHORT_TERM_DEBT = "short_term_debt"
    LONG_TERM_DEBT = "long_term_debt"
    TOTAL_DEBT = "total_debt"
    
    # Balance Sheet Fields - Equity
    TOTAL_EQUITY = "total_equity"
    RETAINED_EARNINGS = "retained_earnings"
    
    # Cash Flow Statement Fields
    OPERATING_CASH_FLOW = "operating_cash_flow"
    DEPRECIATION_CF = "depreciation_cf"
    STOCK_COMP = "stock_compensation"
    DEFERRED_TAX = "deferred_tax"
    CHANGE_IN_WORKING_CAPITAL = "change_in_working_capital"
    CHANGE_IN_RECEIVABLES = "change_in_receivables"
    CHANGE_IN_INVENTORY = "change_in_inventory"
    CHANGE_IN_PAYABLES = "change_in_payables"
    OTHER_OPERATING = "other_operating"
    INVESTING_CASH_FLOW = "investing_cash_flow"
    CAPEX = "capex"
    FINANCING_CASH_FLOW = "financing_cash_flow"
    DIVIDENDS_PAID = "dividends_paid"
    SHARE_REPURCHASES = "share_repurchases"
    FREE_CASH_FLOW = "free_cash_flow"
    
    @classmethod
    def income_statement_fields(cls) -> List[StandardField]:
        """Get all income statement fields."""
        return [
            cls.REVENUE, cls.COST_OF_REVENUE, cls.GROSS_PROFIT,
            cls.OPERATING_EXPENSES, cls.RD_EXPENSE, cls.SGA_EXPENSE,
            cls.OPERATING_INCOME, cls.INTEREST_EXPENSE, cls.PRETAX_INCOME,
            cls.TAX_EXPENSE, cls.NET_INCOME, cls.EBITDA, cls.DEPRECIATION,
            cls.EPS, cls.SHARES_OUTSTANDING
        ]
    
    @classmethod
    def balance_sheet_fields(cls) -> List[StandardField]:
        """Get all balance sheet fields."""
        return [
            cls.TOTAL_ASSETS, cls.CURRENT_ASSETS, cls.CASH,
            cls.SHORT_TERM_INVESTMENTS, cls.ACCOUNTS_RECEIVABLE,
            cls.INVENTORY, cls.OTHER_CURRENT_ASSETS, cls.PPE_NET,
            cls.GOODWILL, cls.INTANGIBLES, cls.TOTAL_LIABILITIES,
            cls.CURRENT_LIABILITIES, cls.ACCOUNTS_PAYABLE,
            cls.SHORT_TERM_DEBT, cls.LONG_TERM_DEBT, cls.TOTAL_DEBT,
            cls.TOTAL_EQUITY, cls.RETAINED_EARNINGS
        ]
    
    @classmethod
    def cash_flow_fields(cls) -> List[StandardField]:
        """Get all cash flow statement fields."""
        return [
            cls.OPERATING_CASH_FLOW, cls.DEPRECIATION_CF, cls.STOCK_COMP,
            cls.DEFERRED_TAX, cls.CHANGE_IN_WORKING_CAPITAL,
            cls.CHANGE_IN_RECEIVABLES, cls.CHANGE_IN_INVENTORY,
            cls.CHANGE_IN_PAYABLES, cls.OTHER_OPERATING,
            cls.INVESTING_CASH_FLOW, cls.CAPEX, cls.FINANCING_CASH_FLOW,
            cls.DIVIDENDS_PAID, cls.SHARE_REPURCHASES, cls.FREE_CASH_FLOW
        ]


# =============================================================================
# FIELD MAPPING CONFIGURATION
# =============================================================================

FIELD_MAPPING: Dict[StandardField, Tuple[str, str]] = {
    StandardField.REVENUE: ("total_revenue", "income"),
    StandardField.COST_OF_REVENUE: ("cost_of_revenue", "income"),
    StandardField.GROSS_PROFIT: ("gross_profit", "income"),
    StandardField.OPERATING_EXPENSES: ("operating_expenses", "income"),
    StandardField.RD_EXPENSE: ("research_development", "income"),
    StandardField.SGA_EXPENSE: ("sga_expense", "income"),
    StandardField.OPERATING_INCOME: ("operating_income", "income"),
    StandardField.INTEREST_EXPENSE: ("interest_expense", "income"),
    StandardField.PRETAX_INCOME: ("pretax_income", "income"),
    StandardField.TAX_EXPENSE: ("tax_provision", "income"),
    StandardField.NET_INCOME: ("net_income", "income"),
    StandardField.EBITDA: ("ebitda", "income"),
    StandardField.DEPRECIATION: ("depreciation_amortization", "income"),
    StandardField.EPS: ("basic_eps", "income"),
    StandardField.SHARES_OUTSTANDING: ("shares_outstanding", "income"),
    StandardField.TOTAL_ASSETS: ("total_assets", "balance"),
    StandardField.CURRENT_ASSETS: ("current_assets", "balance"),
    StandardField.CASH: ("cash_and_equivalents", "balance"),
    StandardField.SHORT_TERM_INVESTMENTS: ("short_term_investments", "balance"),
    StandardField.ACCOUNTS_RECEIVABLE: ("accounts_receivable", "balance"),
    StandardField.INVENTORY: ("inventory", "balance"),
    StandardField.OTHER_CURRENT_ASSETS: ("other_current_assets", "balance"),
    StandardField.PPE_NET: ("ppe_net", "balance"),
    StandardField.GOODWILL: ("goodwill", "balance"),
    StandardField.INTANGIBLES: ("intangible_assets", "balance"),
    StandardField.TOTAL_LIABILITIES: ("total_liabilities", "balance"),
    StandardField.CURRENT_LIABILITIES: ("current_liabilities", "balance"),
    StandardField.ACCOUNTS_PAYABLE: ("accounts_payable", "balance"),
    StandardField.SHORT_TERM_DEBT: ("short_term_debt", "balance"),
    StandardField.LONG_TERM_DEBT: ("long_term_debt", "balance"),
    StandardField.TOTAL_DEBT: ("total_debt", "balance"),
    StandardField.TOTAL_EQUITY: ("total_equity", "balance"),
    StandardField.RETAINED_EARNINGS: ("retained_earnings", "balance"),
    StandardField.OPERATING_CASH_FLOW: ("operating_cash_flow", "cashflow"),
    StandardField.DEPRECIATION_CF: ("depreciation_amortization_cf", "cashflow"),
    StandardField.STOCK_COMP: ("stock_based_compensation", "cashflow"),
    StandardField.DEFERRED_TAX: ("deferred_taxes", "cashflow"),
    StandardField.CHANGE_IN_WORKING_CAPITAL: ("change_in_working_capital", "cashflow"),
    StandardField.CHANGE_IN_RECEIVABLES: ("change_in_receivables", "cashflow"),
    StandardField.CHANGE_IN_INVENTORY: ("change_in_inventory", "cashflow"),
    StandardField.CHANGE_IN_PAYABLES: ("change_in_payables", "cashflow"),
    StandardField.OTHER_OPERATING: ("other_operating_activities", "cashflow"),
    StandardField.INVESTING_CASH_FLOW: ("investing_cash_flow", "cashflow"),
    StandardField.CAPEX: ("capital_expenditure", "cashflow"),
    StandardField.FINANCING_CASH_FLOW: ("financing_cash_flow", "cashflow"),
    StandardField.DIVIDENDS_PAID: ("dividends_paid", "cashflow"),
    StandardField.SHARE_REPURCHASES: ("share_repurchases", "cashflow"),
    StandardField.FREE_CASH_FLOW: ("free_cash_flow", "cashflow"),
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TransformationRecord:
    """
    Record of a single data transformation applied during processing.
    
    Attributes:
        timestamp: When the transformation was applied
        field: Field that was transformed
        transformation_type: Type of transformation
        description: Human-readable description
        original_value: Value before transformation
        new_value: Value after transformation
        period: Fiscal period affected
    """
    timestamp: datetime
    field: str
    transformation_type: str
    description: str
    original_value: Optional[float] = None
    new_value: Optional[float] = None
    period: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "field": self.field,
            "transformation_type": self.transformation_type,
            "description": self.description,
            "original_value": self.original_value,
            "new_value": self.new_value,
            "period": self.period
        }


@dataclass
class DataQualityMetrics:
    """
    Metrics describing the quality and completeness of processed data.
    
    Attributes:
        total_fields_expected: Total number of fields attempted
        fields_found: Number of fields successfully extracted
        fields_derived: Number of fields calculated from other data
        fields_missing: Number of fields that could not be obtained
        coverage_ratio: Proportion of fields successfully obtained
        periods_available: Number of fiscal periods
        accounting_checks_passed: Number of checks that passed
        accounting_checks_failed: Number of checks that failed
        anomalies_detected: List of detected anomalies
    """
    total_fields_expected: int
    fields_found: int
    fields_derived: int
    fields_missing: int
    coverage_ratio: float
    periods_available: int
    accounting_checks_passed: int
    accounting_checks_failed: int
    anomalies_detected: List[str] = field(default_factory=list)
    
    @property
    def anomalies(self) -> List[str]:
        """Alias for anomalies_detected for backward compatibility."""
        return self.anomalies_detected
    
    @property
    def is_high_quality(self) -> bool:
        """Check if data quality meets high quality threshold."""
        return self.coverage_ratio >= HIGH_QUALITY_COVERAGE_THRESHOLD
    
    @property
    def is_usable(self) -> bool:
        """Check if data quality meets minimum usability threshold."""
        return self.coverage_ratio >= MINIMUM_COVERAGE_THRESHOLD
    
    @property
    def has_anomalies(self) -> bool:
        """Check if any anomalies were detected."""
        return len(self.anomalies_detected) > 0
    
    @property
    def accounting_check_pass_rate(self) -> float:
        """Calculate the pass rate for accounting checks."""
        total = self.accounting_checks_passed + self.accounting_checks_failed
        if total == 0:
            return 1.0
        return self.accounting_checks_passed / total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_fields_expected": self.total_fields_expected,
            "fields_found": self.fields_found,
            "fields_derived": self.fields_derived,
            "fields_missing": self.fields_missing,
            "coverage_ratio": self.coverage_ratio,
            "periods_available": self.periods_available,
            "accounting_checks_passed": self.accounting_checks_passed,
            "accounting_checks_failed": self.accounting_checks_failed,
            "anomalies_detected": self.anomalies_detected,
            "is_high_quality": self.is_high_quality,
            "is_usable": self.is_usable
        }


@dataclass
class ProcessedStatements:
    """
    Container for processed and standardized financial statements.
    
    Attributes:
        income_statement: Processed income statement DataFrame
        balance_sheet: Processed balance sheet DataFrame
        cash_flow: Processed cash flow statement DataFrame
        periods: List of fiscal period labels
        currency: Reporting currency code
    """
    income_statement: pd.DataFrame
    balance_sheet: pd.DataFrame
    cash_flow: pd.DataFrame
    periods: List[str]
    currency: str = DEFAULT_CURRENCY
    
    @property
    def num_periods(self) -> int:
        """Get the number of fiscal periods available."""
        return len(self.periods)
    
    @property
    def is_empty(self) -> bool:
        """Check if all statements are empty."""
        return (
            self.income_statement.empty and
            self.balance_sheet.empty and
            self.cash_flow.empty
        )
    
    def get_statement(self, statement_type: str) -> pd.DataFrame:
        """Get a statement by type name."""
        mapping = {
            "income": self.income_statement,
            "balance": self.balance_sheet,
            "cashflow": self.cash_flow
        }
        if statement_type not in mapping:
            raise ValueError(f"Unknown statement type: {statement_type}")
        return mapping[statement_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "income_statement": self.income_statement.to_dict() if not self.income_statement.empty else {},
            "balance_sheet": self.balance_sheet.to_dict() if not self.balance_sheet.empty else {},
            "cash_flow": self.cash_flow.to_dict() if not self.cash_flow.empty else {},
            "periods": self.periods,
            "currency": self.currency,
            "num_periods": self.num_periods
        }


@dataclass
class ProcessedData:
    """
    Complete container for all processed financial data.
    
    This is the primary output of the DataProcessor class.
    
    Attributes:
        company_info: Company metadata
        statements: Processed financial statements
        quality_metrics: Data quality assessment
        transformations: Audit trail of transformations
        processing_timestamp: When processing was performed
        warnings: List of warning messages
        raw_data: Reference to original raw data
    """
    company_info: CompanyInfo
    statements: ProcessedStatements
    quality_metrics: DataQualityMetrics
    transformations: List[TransformationRecord]
    processing_timestamp: datetime = field(default_factory=datetime.now)
    warnings: List[str] = field(default_factory=list)
    raw_data: Optional[CollectedData] = None
    
    @property
    def ticker(self) -> str:
        """Get the stock ticker symbol."""
        return self.company_info.ticker
    
    @property
    def company_name(self) -> str:
        """Get the company name."""
        return self.company_info.name
    
    @property
    def is_valid(self) -> bool:
        """Check if processed data is valid for analysis."""
        return self.quality_metrics.is_usable
    
    @property
    def is_high_quality(self) -> bool:
        """Check if processed data is high quality."""
        return self.quality_metrics.is_high_quality
    
    @property
    def periods(self) -> List[str]:
        """Get list of fiscal periods."""
        return self.statements.periods
    
    @property
    def num_periods(self) -> int:
        """Get number of fiscal periods."""
        return self.statements.num_periods
    
    @property
    def currency(self) -> str:
        """Get reporting currency."""
        return self.statements.currency
    
    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-like access for compatibility."""
        key_mapping = {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "name": self.company_name,
            "currency": self.currency,
            "periods": self.periods,
            "num_periods": self.num_periods,
            "is_valid": self.is_valid,
            "is_high_quality": self.is_high_quality,
            "warnings": self.warnings,
            "processing_timestamp": self.processing_timestamp
        }
        return key_mapping.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "company_info": self.company_info.to_dict(),
            "statements": self.statements.to_dict(),
            "quality_metrics": self.quality_metrics.to_dict(),
            "transformations": [t.to_dict() for t in self.transformations],
            "processing_timestamp": self.processing_timestamp.isoformat(),
            "warnings": self.warnings,
            "periods": self.periods,
            "currency": self.currency
        }


# =============================================================================
# DATA PROCESSOR CLASS
# =============================================================================

class DataProcessor:
    """
    Financial data processing and standardization pipeline.
    
    This class transforms raw financial data into a clean, standardized format
    suitable for downstream analysis.
    """
    
    def __init__(self, scale_to_millions: bool = True) -> None:
        """
        Initialize the data processor.
        
        Args:
            scale_to_millions: If True, convert monetary values to millions.
        """
        self._transformations: List[TransformationRecord] = []
        self._warnings: List[str] = []
        self._scale_factor = DEFAULT_SCALE_FACTOR if scale_to_millions else 1.0
        self._scale_label = "millions" if scale_to_millions else "original units"
        logger.info(f"DataProcessor initialized (scale: {self._scale_label})")
    
    def process(self, raw_data: CollectedData) -> ProcessedData:
        """
        Process raw financial data into standardized format.
        
        Args:
            raw_data: CollectedData object from DataCollector.
            
        Returns:
            ProcessedData object with standardized data.
            
        Raises:
            ValueError: If raw_data is invalid or missing critical components.
        """
        self._transformations = []
        self._warnings = []
        
        logger.info(f"Starting data processing for {raw_data.company_info.ticker}")
        
        self._validate_input(raw_data)
        
        periods = self._align_periods(raw_data.statements)
        period_labels = [self._format_period(p) for p in periods]
        
        income_stmt = self._process_income_statement(
            raw_data.statements.income_statement, periods
        )
        balance_sheet = self._process_balance_sheet(
            raw_data.statements.balance_sheet, periods
        )
        cash_flow = self._process_cash_flow(
            raw_data.statements.cash_flow, periods
        )
        
        income_stmt, balance_sheet = self._calculate_derived_income_fields(
            income_stmt, balance_sheet
        )
        cash_flow = self._calculate_derived_cash_flow_fields(cash_flow, income_stmt)
        
        self._validate_accounting_identities(income_stmt, balance_sheet, cash_flow)
        
        quality_metrics = self._compute_quality_metrics(
            income_stmt, balance_sheet, cash_flow, periods
        )
        
        processed_statements = ProcessedStatements(
            income_statement=income_stmt,
            balance_sheet=balance_sheet,
            cash_flow=cash_flow,
            periods=period_labels,
            currency=raw_data.company_info.currency
        )
        
        logger.info(
            f"Processing complete for {raw_data.company_info.ticker}: "
            f"{quality_metrics.fields_found} fields found, "
            f"{quality_metrics.fields_derived} derived, "
            f"{quality_metrics.fields_missing} missing"
        )
        
        return ProcessedData(
            company_info=raw_data.company_info,
            statements=processed_statements,
            quality_metrics=quality_metrics,
            transformations=self._transformations,
            processing_timestamp=datetime.now(),
            warnings=self._warnings,
            raw_data=raw_data
        )
    
    def _validate_input(self, raw_data: CollectedData) -> None:
        """Validate that raw data contains minimum required components."""
        if raw_data is None:
            raise ValueError("Raw data cannot be None")
        
        if raw_data.statements is None:
            raise ValueError("Financial statements container cannot be None")
        
        if raw_data.statements.income_statement is None or \
           raw_data.statements.income_statement.empty:
            raise ValueError("Income statement is required but empty or None")
        
        if raw_data.statements.balance_sheet is None or \
           raw_data.statements.balance_sheet.empty:
            self._add_warning("Balance sheet is empty - some analyses will be limited")
        
        if raw_data.statements.cash_flow is None or \
           raw_data.statements.cash_flow.empty:
            self._add_warning("Cash flow statement is empty - some analyses will be limited")
    
    def _align_periods(self, statements: FinancialStatements) -> List[Any]:
        """Determine common fiscal periods across all statements."""
        if statements.income_statement is None or statements.income_statement.empty:
            return []
        
        reference_periods = list(statements.income_statement.columns)
        
        if statements.balance_sheet is not None and not statements.balance_sheet.empty:
            bs_periods = set(statements.balance_sheet.columns)
            missing_bs = set(reference_periods) - bs_periods
            if missing_bs:
                self._add_warning(f"Balance sheet missing periods: {[self._format_period(p) for p in missing_bs]}")
        
        if statements.cash_flow is not None and not statements.cash_flow.empty:
            cf_periods = set(statements.cash_flow.columns)
            missing_cf = set(reference_periods) - cf_periods
            if missing_cf:
                self._add_warning(f"Cash flow missing periods: {[self._format_period(p) for p in missing_cf]}")
        
        self._record_transformation(
            field="periods",
            transformation_type="aligned",
            description=f"Aligned {len(reference_periods)} fiscal periods"
        )
        
        return reference_periods
    
    def _format_period(self, period: Any) -> str:
        """Format a fiscal period as a human-readable string label."""
        if hasattr(period, 'strftime'):
            return period.strftime('%Y')
        return str(period)
    
    def _process_income_statement(self, raw_stmt: pd.DataFrame, periods: List[Any]) -> pd.DataFrame:
        """Process raw income statement into standardized format."""
        if raw_stmt is None or raw_stmt.empty:
            return pd.DataFrame()
        return self._extract_fields(raw_stmt, periods, StandardField.income_statement_fields(), "income")
    
    def _process_balance_sheet(self, raw_stmt: pd.DataFrame, periods: List[Any]) -> pd.DataFrame:
        """Process raw balance sheet into standardized format."""
        if raw_stmt is None or raw_stmt.empty:
            return pd.DataFrame()
        return self._extract_fields(raw_stmt, periods, StandardField.balance_sheet_fields(), "balance")
    
    def _process_cash_flow(self, raw_stmt: pd.DataFrame, periods: List[Any]) -> pd.DataFrame:
        """Process raw cash flow statement into standardized format."""
        if raw_stmt is None or raw_stmt.empty:
            return pd.DataFrame()
        return self._extract_fields(raw_stmt, periods, StandardField.cash_flow_fields(), "cashflow")
    
    def _extract_fields(
        self,
        raw_stmt: pd.DataFrame,
        periods: List[Any],
        fields: List[StandardField],
        statement_type: str
    ) -> pd.DataFrame:
        """Extract and standardize fields from a raw financial statement."""
        result_data: Dict[str, List[Optional[float]]] = {}
        period_labels = [self._format_period(p) for p in periods]
        non_scaled_fields = {StandardField.EPS}
        
        for std_field in fields:
            field_values: List[Optional[float]] = []
            
            for period in periods:
                value = self._get_field_value(raw_stmt, std_field, statement_type, period)
                
                if value is not None and std_field not in non_scaled_fields:
                    value = value / self._scale_factor
                
                field_values.append(value)
            
            result_data[std_field.value] = field_values
        
        result_df = pd.DataFrame(result_data, index=period_labels).T
        result_df.index.name = 'field'
        
        return result_df
    
    def _get_field_value(
        self,
        statement: pd.DataFrame,
        std_field: StandardField,
        statement_type: str,
        period: Any
    ) -> Optional[float]:
        """Extract a single field value from a raw statement."""
        if statement is None or statement.empty:
            return None
        
        if period not in statement.columns:
            return None
        
        if std_field not in FIELD_MAPPING:
            return None
        
        config_key, _ = FIELD_MAPPING[std_field]
        alternatives = get_field_alternatives(statement_type, config_key)
        
        for alt_name in alternatives:
            if alt_name in statement.index:
                try:
                    value = statement.loc[alt_name, period]
                    if pd.notna(value):
                        return float(value)
                except (KeyError, TypeError, ValueError):
                    continue
        
        return None
    
    def _calculate_derived_income_fields(
        self,
        income_stmt: pd.DataFrame,
        balance_sheet: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate derived income statement and balance sheet fields."""
        if income_stmt.empty:
            return income_stmt, balance_sheet
        
        income_stmt = self._derive_gross_profit(income_stmt)
        income_stmt = self._derive_operating_expenses(income_stmt)
        income_stmt = self._derive_ebitda(income_stmt)
        
        if not balance_sheet.empty:
            balance_sheet = self._derive_total_debt(balance_sheet)
        
        return income_stmt, balance_sheet
    
    def _derive_gross_profit(self, income_stmt: pd.DataFrame) -> pd.DataFrame:
        """Derive Gross Profit = Revenue - Cost of Revenue."""
        if not self._is_missing(income_stmt, StandardField.GROSS_PROFIT.value):
            return income_stmt
        
        revenue = self._get_row_safe(income_stmt, StandardField.REVENUE.value)
        cogs = self._get_row_safe(income_stmt, StandardField.COST_OF_REVENUE.value)
        
        if revenue is not None and cogs is not None:
            gross_profit = revenue - cogs
            income_stmt.loc[StandardField.GROSS_PROFIT.value] = gross_profit
            self._record_transformation(
                field=StandardField.GROSS_PROFIT.value,
                transformation_type="derived",
                description="Gross Profit = Revenue - Cost of Revenue"
            )
        
        return income_stmt
    
    def _derive_operating_expenses(self, income_stmt: pd.DataFrame) -> pd.DataFrame:
        """Derive Operating Expenses = Gross Profit - Operating Income."""
        if not self._is_missing(income_stmt, StandardField.OPERATING_EXPENSES.value):
            return income_stmt
        
        gross_profit = self._get_row_safe(income_stmt, StandardField.GROSS_PROFIT.value)
        operating_income = self._get_row_safe(income_stmt, StandardField.OPERATING_INCOME.value)
        
        if gross_profit is not None and operating_income is not None:
            opex = gross_profit - operating_income
            income_stmt.loc[StandardField.OPERATING_EXPENSES.value] = opex
            self._record_transformation(
                field=StandardField.OPERATING_EXPENSES.value,
                transformation_type="derived",
                description="Operating Expenses = Gross Profit - Operating Income"
            )
        
        return income_stmt
    
    def _derive_ebitda(self, income_stmt: pd.DataFrame) -> pd.DataFrame:
        """Derive EBITDA = Operating Income + Depreciation & Amortization."""
        if not self._is_missing(income_stmt, StandardField.EBITDA.value):
            return income_stmt
        
        operating_income = self._get_row_safe(income_stmt, StandardField.OPERATING_INCOME.value)
        depreciation = self._get_row_safe(income_stmt, StandardField.DEPRECIATION.value)
        
        if operating_income is not None and depreciation is not None:
            ebitda = operating_income + depreciation.abs()
            income_stmt.loc[StandardField.EBITDA.value] = ebitda
            self._record_transformation(
                field=StandardField.EBITDA.value,
                transformation_type="derived",
                description="EBITDA = Operating Income + D&A"
            )
        elif operating_income is not None:
            income_stmt.loc[StandardField.EBITDA.value] = operating_income
            self._record_transformation(
                field=StandardField.EBITDA.value,
                transformation_type="estimated",
                description="EBITDA estimated as Operating Income (D&A unavailable)"
            )
            self._add_warning("EBITDA estimated without D&A - may be understated")
        
        return income_stmt
    
    def _derive_total_debt(self, balance_sheet: pd.DataFrame) -> pd.DataFrame:
        """Derive Total Debt = Short-term Debt + Long-term Debt."""
        if not self._is_missing(balance_sheet, StandardField.TOTAL_DEBT.value):
            return balance_sheet
        
        st_debt = self._get_row_safe(balance_sheet, StandardField.SHORT_TERM_DEBT.value)
        lt_debt = self._get_row_safe(balance_sheet, StandardField.LONG_TERM_DEBT.value)
        
        if st_debt is not None or lt_debt is not None:
            if st_debt is None:
                st_debt = pd.Series(0, index=balance_sheet.columns)
            if lt_debt is None:
                lt_debt = pd.Series(0, index=balance_sheet.columns)
            
            total_debt = st_debt.fillna(0) + lt_debt.fillna(0)
            balance_sheet.loc[StandardField.TOTAL_DEBT.value] = total_debt
            self._record_transformation(
                field=StandardField.TOTAL_DEBT.value,
                transformation_type="derived",
                description="Total Debt = Short-term Debt + Long-term Debt"
            )
        
        return balance_sheet
    
    def _calculate_derived_cash_flow_fields(
        self,
        cash_flow: pd.DataFrame,
        income_stmt: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate derived cash flow statement fields."""
        if cash_flow.empty:
            return cash_flow
        
        cash_flow = self._derive_free_cash_flow(cash_flow)
        cash_flow = self._copy_depreciation_from_income(cash_flow, income_stmt)
        
        return cash_flow
    
    def _derive_free_cash_flow(self, cash_flow: pd.DataFrame) -> pd.DataFrame:
        """
        Derive Free Cash Flow = Operating Cash Flow - |CapEx|.
        
        CRITICAL: Always recalculate FCF to ensure consistency.
        Some data providers incorrectly calculate FCF as OCF + CapEx.
        We ALWAYS use OCF - |CapEx| regardless of what raw data provides.
        """
        ocf = self._get_row_safe(cash_flow, StandardField.OPERATING_CASH_FLOW.value)
        capex = self._get_row_safe(cash_flow, StandardField.CAPEX.value)
        
        if ocf is not None and capex is not None:
            # ALWAYS calculate FCF = OCF - |CapEx|
            fcf = ocf - capex.abs()
            
            # Log if raw FCF differs significantly (for debugging)
            if not self._is_missing(cash_flow, StandardField.FREE_CASH_FLOW.value):
                existing_fcf = self._get_row_safe(cash_flow, StandardField.FREE_CASH_FLOW.value)
                if existing_fcf is not None:
                    diff_pct = ((existing_fcf - fcf) / fcf.abs()).mean()
                    if abs(diff_pct) > 0.05:
                        logger.warning(
                            f"Raw FCF differs from calculated by {diff_pct*100:.1f}%. "
                            f"Using standardized FCF = OCF - |CapEx|."
                        )
            
            # Always overwrite with correctly calculated FCF
            cash_flow.loc[StandardField.FREE_CASH_FLOW.value] = fcf
            self._record_transformation(
                field=StandardField.FREE_CASH_FLOW.value,
                transformation_type="derived",
                description="Free Cash Flow = Operating Cash Flow - |CapEx| (standardized)"
            )
        
        return cash_flow
    
    def _copy_depreciation_from_income(
        self,
        cash_flow: pd.DataFrame,
        income_stmt: pd.DataFrame
    ) -> pd.DataFrame:
        """Copy Depreciation from income statement to cash flow if missing."""
        if not self._is_missing(cash_flow, StandardField.DEPRECIATION_CF.value):
            return cash_flow
        
        if income_stmt.empty:
            return cash_flow
        
        income_da = self._get_row_safe(income_stmt, StandardField.DEPRECIATION.value)
        
        if income_da is not None:
            cash_flow.loc[StandardField.DEPRECIATION_CF.value] = income_da
            self._record_transformation(
                field=StandardField.DEPRECIATION_CF.value,
                transformation_type="derived",
                description="D&A copied from income statement"
            )
        
        return cash_flow
    
    def _validate_accounting_identities(
        self,
        income_stmt: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cash_flow: pd.DataFrame
    ) -> None:
        """Validate fundamental accounting identities."""
        self._check_balance_sheet_identity(balance_sheet)
        self._check_gross_profit_consistency(income_stmt)
    
    def _check_balance_sheet_identity(self, balance_sheet: pd.DataFrame) -> None:
        """Check that Assets = Liabilities + Equity for all periods."""
        if balance_sheet.empty:
            return
        
        assets = self._get_row_safe(balance_sheet, StandardField.TOTAL_ASSETS.value)
        liabilities = self._get_row_safe(balance_sheet, StandardField.TOTAL_LIABILITIES.value)
        equity = self._get_row_safe(balance_sheet, StandardField.TOTAL_EQUITY.value)
        
        if assets is None or liabilities is None or equity is None:
            return
        
        for period in balance_sheet.columns:
            try:
                a = self._safe_get_value(assets, period)
                l = self._safe_get_value(liabilities, period)
                e = self._safe_get_value(equity, period)
                
                if a is None or l is None or e is None:
                    continue
                
                diff = abs(a - (l + e))
                tolerance = abs(a) * 0.01
                
                if diff > tolerance:
                    self._add_warning(
                        f"Balance sheet identity violation in {period}: "
                        f"Assets ({a:,.1f}M) != Liabilities ({l:,.1f}M) + Equity ({e:,.1f}M)"
                    )
            except (KeyError, TypeError, ValueError):
                continue
    
    def _check_gross_profit_consistency(self, income_stmt: pd.DataFrame) -> None:
        """Check that Gross Profit = Revenue - COGS for all periods."""
        if income_stmt.empty:
            return
        
        revenue = self._get_row_safe(income_stmt, StandardField.REVENUE.value)
        cogs = self._get_row_safe(income_stmt, StandardField.COST_OF_REVENUE.value)
        gross_profit = self._get_row_safe(income_stmt, StandardField.GROSS_PROFIT.value)
        
        if revenue is None or cogs is None or gross_profit is None:
            return
        
        for period in income_stmt.columns:
            try:
                r = self._safe_get_value(revenue, period)
                c = self._safe_get_value(cogs, period)
                gp = self._safe_get_value(gross_profit, period)
                
                if r is None or c is None or gp is None:
                    continue
                
                expected_gp = r - c
                diff = abs(gp - expected_gp)
                tolerance = abs(r) * 0.01
                
                if diff > tolerance:
                    self._add_warning(
                        f"Gross profit inconsistency in {period}: "
                        f"Reported ({gp:,.1f}M) vs Calculated ({expected_gp:,.1f}M)"
                    )
            except (KeyError, TypeError, ValueError):
                continue
    
    def _compute_quality_metrics(
        self,
        income_stmt: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cash_flow: pd.DataFrame,
        periods: List[Any]
    ) -> DataQualityMetrics:
        """Compute comprehensive data quality metrics."""
        all_fields = list(StandardField)
        total_expected = len(all_fields)
        
        found = 0
        for std_field in all_fields:
            if self._field_is_present(std_field.value, income_stmt, balance_sheet, cash_flow):
                found += 1
        
        derived = sum(
            1 for t in self._transformations
            if t.transformation_type in ['derived', 'estimated']
        )
        
        fields_missing = total_expected - found
        
        checks_failed = sum(
            1 for w in self._warnings
            if 'identity' in w.lower() or 'inconsistency' in w.lower()
        )
        checks_passed = max(0, len(periods) * 2 - checks_failed)
        
        anomalies = self._detect_anomalies(income_stmt, balance_sheet, cash_flow)
        coverage_ratio = found / total_expected if total_expected > 0 else 0.0
        
        return DataQualityMetrics(
            total_fields_expected=total_expected,
            fields_found=found,
            fields_derived=derived,
            fields_missing=fields_missing,
            coverage_ratio=coverage_ratio,
            periods_available=len(periods),
            accounting_checks_passed=checks_passed,
            accounting_checks_failed=checks_failed,
            anomalies_detected=anomalies
        )
    
    def _field_is_present(
        self,
        field_name: str,
        income_stmt: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cash_flow: pd.DataFrame
    ) -> bool:
        """Check if a field is present in any statement."""
        for stmt in [income_stmt, balance_sheet, cash_flow]:
            if stmt.empty:
                continue
            if field_name in stmt.index:
                if not stmt.loc[field_name].isna().all():
                    return True
        return False
    
    def _detect_anomalies(
        self,
        income_stmt: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cash_flow: pd.DataFrame
    ) -> List[str]:
        """Detect potential data anomalies."""
        anomalies: List[str] = []
        
        if income_stmt.empty:
            anomalies.append("Income statement is empty")
            return anomalies
        
        revenue = self._get_row_safe(income_stmt, StandardField.REVENUE.value)
        if revenue is not None and (revenue < 0).any():
            anomalies.append("Negative revenue detected in one or more periods")
        
        gross_profit = self._get_row_safe(income_stmt, StandardField.GROSS_PROFIT.value)
        if revenue is not None and gross_profit is not None:
            safe_revenue = revenue.replace(0, np.nan)
            gross_margin = gross_profit / safe_revenue
            
            if (gross_margin < VALIDATION.min_reasonable_gross_margin).any():
                anomalies.append(f"Gross margin below {VALIDATION.min_reasonable_gross_margin*100:.0f}%")
            if (gross_margin > VALIDATION.max_reasonable_gross_margin).any():
                anomalies.append(f"Gross margin above {VALIDATION.max_reasonable_gross_margin*100:.0f}%")
        
        operating_income = self._get_row_safe(income_stmt, StandardField.OPERATING_INCOME.value)
        if revenue is not None and operating_income is not None:
            safe_revenue = revenue.replace(0, np.nan)
            op_margin = operating_income / safe_revenue
            
            if (op_margin < VALIDATION.min_reasonable_operating_margin).any():
                anomalies.append(f"Operating margin below {VALIDATION.min_reasonable_operating_margin*100:.0f}%")
            if (op_margin > VALIDATION.max_reasonable_operating_margin).any():
                anomalies.append(f"Operating margin above {VALIDATION.max_reasonable_operating_margin*100:.0f}%")
        
        if not balance_sheet.empty:
            assets = self._get_row_safe(balance_sheet, StandardField.TOTAL_ASSETS.value)
            if assets is not None and (assets == 0).any():
                anomalies.append("Zero total assets detected")
        
        return anomalies
    
    def _is_missing(self, df: pd.DataFrame, field_name: str) -> bool:
        """Check if a field is missing or entirely null."""
        if df.empty:
            return True
        if field_name not in df.index:
            return True
        return df.loc[field_name].isna().all()
    
    def _get_row_safe(self, df: pd.DataFrame, field_name: str) -> Optional[pd.Series]:
        """Safely extract a row from a DataFrame."""
        if df.empty:
            return None
        if field_name not in df.index:
            return None
        return df.loc[field_name]
    
    def _safe_get_value(self, series: pd.Series, key: Any) -> Optional[float]:
        """Safely get a value from a Series."""
        if series is None:
            return None
        try:
            value = series.get(key) if isinstance(series, pd.Series) else series
            if pd.notna(value):
                return float(value)
        except (KeyError, TypeError, ValueError):
            pass
        return None
    
    def _record_transformation(
        self,
        field: str,
        transformation_type: str,
        description: str,
        original_value: Optional[float] = None,
        new_value: Optional[float] = None,
        period: Optional[str] = None
    ) -> None:
        """Record a transformation in the audit trail."""
        record = TransformationRecord(
            timestamp=datetime.now(),
            field=field,
            transformation_type=transformation_type,
            description=description,
            original_value=original_value,
            new_value=new_value,
            period=period
        )
        self._transformations.append(record)
        logger.debug(f"Transformation: {field} - {description}")
    
    def _add_warning(self, message: str) -> None:
        """Add a warning message."""
        self._warnings.append(message)
        logger.warning(message)
    
    def get_field(
        self,
        processed_data: ProcessedData,
        field: StandardField,
        period_index: int = 0
    ) -> Optional[float]:
        """Extract a single field value from processed data."""
        field_name = field.value
        statements = processed_data.statements
        
        for stmt in [statements.income_statement, statements.balance_sheet, statements.cash_flow]:
            if stmt.empty:
                continue
            if field_name in stmt.index:
                try:
                    value = stmt.loc[field_name].iloc[period_index]
                    if pd.notna(value):
                        return float(value)
                except (IndexError, KeyError):
                    continue
        
        return None
    
    def get_field_series(
        self,
        processed_data: ProcessedData,
        field: StandardField
    ) -> Optional[pd.Series]:
        """Extract a complete time series for a field."""
        field_name = field.value
        statements = processed_data.statements
        
        for stmt in [statements.income_statement, statements.balance_sheet, statements.cash_flow]:
            if stmt.empty:
                continue
            if field_name in stmt.index:
                return stmt.loc[field_name].copy()
        
        return None
    
    def get_all_fields(
        self,
        processed_data: ProcessedData,
        period_index: int = 0
    ) -> Dict[str, Optional[float]]:
        """Extract all field values for a specific period."""
        result: Dict[str, Optional[float]] = {}
        
        for std_field in StandardField:
            value = self.get_field(processed_data, std_field, period_index)
            result[std_field.value] = value
        
        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def process_financial_data(raw_data: CollectedData) -> ProcessedData:
    """Convenience function to process financial data."""
    processor = DataProcessor()
    return processor.process(raw_data)


def get_standard_field_value(
    processed_data: ProcessedData,
    field: StandardField,
    period_index: int = 0
) -> Optional[float]:
    """Convenience function to get a single field value."""
    processor = DataProcessor()
    return processor.get_field(processed_data, field, period_index)


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
    print(f"DATA PROCESSOR MODULE TEST - {test_ticker}")
    print("=" * 70)
    print()
    
    try:
        print("Step 1: Collecting raw data...")
        collector = DataCollector()
        raw_data = collector.collect(test_ticker)
        print(f"  Raw data collected: {raw_data.validation.years_available} years")
        print()
        
        print("Step 2: Processing data...")
        processor = DataProcessor()
        processed = processor.process(raw_data)
        print(f"  Processing complete")
        print()
        
        print("PROCESSED DATA INFO")
        print("-" * 70)
        print(f"  Ticker:          {processed.ticker}")
        print(f"  Company:         {processed.company_name}")
        print(f"  Currency:        {processed.currency}")
        print(f"  Periods:         {processed.periods}")
        print(f"  Is Valid:        {processed.is_valid}")
        print(f"  Is High Quality: {processed.is_high_quality}")
        print()
        
        print("DATA QUALITY METRICS")
        print("-" * 70)
        qm = processed.quality_metrics
        print(f"  Fields Expected:   {qm.total_fields_expected}")
        print(f"  Fields Found:      {qm.fields_found}")
        print(f"  Fields Derived:    {qm.fields_derived}")
        print(f"  Fields Missing:    {qm.fields_missing}")
        print(f"  Coverage Ratio:    {qm.coverage_ratio:.1%}")
        print(f"  Periods Available: {qm.periods_available}")
        print(f"  Is High Quality:   {qm.is_high_quality}")
        print(f"  Is Usable:         {qm.is_usable}")
        
        if qm.anomalies:
            print(f"\n  Anomalies ({len(qm.anomalies)}):")
            for a in qm.anomalies:
                print(f"    - {a}")
        
        print()
        print("=" * 70)
        print(f"Test complete for {test_ticker}")
        print("=" * 70)
        print()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)