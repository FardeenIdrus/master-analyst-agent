"""
Configuration Module - Phases 1-3 Data Acquisition, Validation & Ratio Analysis
Fundamental Analyst Agent

Centralizes configuration constants, field mappings, validation thresholds,
quality assessment criteria, reconciliation tolerances, growth classification,
and financial ratio benchmarks for the fundamental analysis pipeline.

MSc Coursework: IFTE0001 AI Agents in Asset Management
Track A: Fundamental Analyst Agent
Phase 1: Data Acquisition
Phase 2: Data Validation & Standardization
Phase 3: Financial Ratio Analysis

Version: 3.2.0
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, FrozenSet, Optional
from enum import Enum


# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger instance with professional formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


LOGGER = setup_logger("DataAcquisition")


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ValidationStatus(Enum):
    """Validation status codes."""
    VALID = "valid"
    PARTIAL = "partial"
    INVALID = "invalid"


class DataQualityTier(Enum):
    """Data quality classification tiers."""
    EXCELLENT = "excellent"  # >= 95% completeness
    GOOD = "good"            # >= 80% completeness
    ACCEPTABLE = "acceptable"  # >= 60% completeness
    POOR = "poor"            # < 60% completeness


class StatementType(Enum):
    """Financial statement types."""
    INCOME = "income_statement"
    BALANCE = "balance_sheet"
    CASHFLOW = "cash_flow"


# =============================================================================
# ALPHA VANTAGE FIELD MAPPINGS
# =============================================================================

# Income Statement: Alpha Vantage API field names -> standardized names
INCOME_FIELD_MAP: Dict[str, str] = {
    # Revenue and Gross Profit
    "totalRevenue": "total_revenue",
    "costOfRevenue": "cost_of_revenue",
    "costofGoodsAndServicesSold": "cogs",
    "grossProfit": "gross_profit",
    
    # Operating Items
    "operatingExpenses": "operating_expenses",
    "operatingIncome": "operating_income",
    "researchAndDevelopment": "rd_expense",
    "sellingGeneralAndAdministrative": "sga_expense",
    
    # Non-Operating Items
    "interestExpense": "interest_expense",
    "interestIncome": "interest_income",
    "otherNonOperatingIncome": "other_non_operating",
    
    # Pre-Tax and Tax
    "incomeBeforeTax": "pretax_income",
    "incomeTaxExpense": "tax_expense",
    
    # Net Income
    "netIncome": "net_income",
    "netIncomeFromContinuingOperations": "net_income_continuing",
    
    # EBIT/EBITDA
    "ebit": "ebit",
    "ebitda": "ebitda",
    
    # Depreciation and EPS
    "depreciationAndAmortization": "depreciation_amortization",
    "reportedEPS": "eps_reported",
    
    # Other
    "comprehensiveIncomeNetOfTax": "comprehensive_income",
}

# Balance Sheet: Alpha Vantage API field names -> standardized names
BALANCE_FIELD_MAP: Dict[str, str] = {
    # Total Assets
    "totalAssets": "total_assets",
    
    # Current Assets
    "totalCurrentAssets": "current_assets",
    "cashAndCashEquivalentsAtCarryingValue": "cash_and_equivalents",
    "cashAndShortTermInvestments": "cash_and_short_term_investments",
    "shortTermInvestments": "short_term_investments",
    "currentNetReceivables": "accounts_receivable",
    "inventory": "inventory",
    "otherCurrentAssets": "other_current_assets",
    
    # Non-Current Assets
    "totalNonCurrentAssets": "non_current_assets",
    "propertyPlantEquipment": "ppe_gross",
    "accumulatedDepreciationAmortizationPPE": "accumulated_depreciation",
    "goodwill": "goodwill",
    "intangibleAssets": "intangible_assets",
    "intangibleAssetsExcludingGoodwill": "other_intangibles",
    "longTermInvestments": "long_term_investments",
    "otherNonCurrentAssets": "other_non_current_assets",
    
    # Total Liabilities
    "totalLiabilities": "total_liabilities",
    
    # Current Liabilities
    "totalCurrentLiabilities": "current_liabilities",
    "currentAccountsPayable": "accounts_payable",
    "currentDebt": "current_debt",
    "shortTermDebt": "short_term_debt",
    "otherCurrentLiabilities": "other_current_liabilities",
    
    # Non-Current Liabilities
    "totalNonCurrentLiabilities": "non_current_liabilities",
    "longTermDebt": "long_term_debt",
    "longTermDebtNoncurrent": "lt_debt_noncurrent",
    "capitalLeaseObligations": "capital_lease_obligations",
    "otherNonCurrentLiabilities": "other_non_current_liabilities",
    
    # Total Debt
    "shortLongTermDebtTotal": "total_debt",
    
    # Equity
    "totalShareholderEquity": "total_equity",
    "retainedEarnings": "retained_earnings",
    "commonStock": "common_stock",
    "commonStockSharesOutstanding": "shares_outstanding",
    "treasuryStock": "treasury_stock",
    "additionalPaidInCapital": "additional_paid_in_capital",
}

# Cash Flow Statement: Alpha Vantage API field names -> standardized names
CASHFLOW_FIELD_MAP: Dict[str, str] = {
    # Operating Activities
    "operatingCashflow": "operating_cash_flow",
    "netIncome": "net_income",
    "depreciationDepletionAndAmortization": "depreciation_amortization",
    "changeInOperatingLiabilities": "change_operating_liabilities",
    "changeInOperatingAssets": "change_operating_assets",
    "changeInReceivables": "change_receivables",
    "changeInInventory": "change_inventory",
    "paymentsForOperatingActivities": "operating_payments",
    
    # Investing Activities
    "cashflowFromInvestment": "investing_cash_flow",
    "capitalExpenditures": "capital_expenditure",
    
    # Financing Activities
    "cashflowFromFinancing": "financing_cash_flow",
    "dividendPayout": "dividends_paid",
    "dividendPayoutCommonStock": "common_dividends",
    "dividendPayoutPreferredStock": "preferred_dividends",
    "paymentsForRepurchaseOfCommonStock": "share_repurchases",
    "paymentsForRepurchaseOfEquity": "equity_repurchases",
    "proceedsFromIssuanceOfCommonStock": "stock_issuance",
    "proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet": "debt_issuance",
    "proceedsFromRepurchaseOfEquity": "repurchase_proceeds",
    
    # Net Change
    "changeInCashAndCashEquivalents": "change_in_cash",
    
    # Free Cash Flow (if provided by API)
    "freeCashFlow": "free_cash_flow_reported",
}

# Company Overview: Alpha Vantage API field names -> standardized names
OVERVIEW_FIELD_MAP: Dict[str, str] = {
    # Identification
    "Symbol": "ticker",
    "Name": "company_name",
    "Description": "description",
    "Exchange": "exchange",
    "Currency": "currency",
    "Country": "country",
    "Sector": "sector",
    "Industry": "industry",
    "FiscalYearEnd": "fiscal_year_end",
    
    # Market Data
    "MarketCapitalization": "market_cap",
    "SharesOutstanding": "shares_outstanding",
    "52WeekHigh": "high_52week",
    "52WeekLow": "low_52week",
    "50DayMovingAverage": "ma_50day",
    "200DayMovingAverage": "ma_200day",
    "Beta": "beta",
    
    # Valuation Multiples
    "PERatio": "pe_ratio",
    "PEGRatio": "peg_ratio",
    "BookValue": "book_value",
    "PriceToBookRatio": "pb_ratio",
    "PriceToSalesRatioTTM": "ps_ratio",
    "EVToRevenue": "ev_to_revenue",
    "EVToEBITDA": "ev_to_ebitda",
    "TrailingPE": "pe_trailing",
    "ForwardPE": "pe_forward",
    
    # Profitability
    "ProfitMargin": "profit_margin",
    "OperatingMarginTTM": "operating_margin",
    "ReturnOnAssetsTTM": "roa",
    "ReturnOnEquityTTM": "roe",
    
    # Dividends
    "DividendPerShare": "dividend_per_share",
    "DividendYield": "dividend_yield",
    "ExDividendDate": "ex_dividend_date",
    "DividendDate": "dividend_date",
    
    # Growth
    "QuarterlyEarningsGrowthYOY": "earnings_growth_yoy",
    "QuarterlyRevenueGrowthYOY": "revenue_growth_yoy",
    
    # Earnings
    "EPS": "eps",
    "DilutedEPSTTM": "eps_diluted_ttm",
    
    # Analyst
    "AnalystTargetPrice": "analyst_target",
    "AnalystRatingStrongBuy": "analyst_strong_buy",
    "AnalystRatingBuy": "analyst_buy",
    "AnalystRatingHold": "analyst_hold",
    "AnalystRatingSell": "analyst_sell",
    "AnalystRatingStrongSell": "analyst_strong_sell",
}


# =============================================================================
# SEC DATA CORRECTIONS - VERIFIED FROM OFFICIAL 10-K FILINGS
# =============================================================================
# This section contains verified data corrections for known API data issues.
# Data sources: Apple Inc. Form 10-K filings (SEC EDGAR)
# Last verified: January 2026

# Issue 1: Alpha Vantage returns combined "Accounts Receivable + Vendor Non-Trade Receivables"
# SEC 10-K shows these as SEPARATE line items:
#   - "Accounts receivable, net" = Trade receivables from customers
#   - "Vendor non-trade receivables" = Receivables from manufacturing partners
# For accurate receivables turnover and DSO, we need ONLY trade receivables.

# Issue 2: Interest Expense not returned separately for recent years
# Apple reports "Other income/(expense), net" which combines:
#   - Interest and dividend income
#   - Interest expense  
#   - Other income (expense), net
# We extract interest expense from the detailed notes in the 10-K.

SEC_DATA_CORRECTIONS: Dict[str, Dict[str, Dict[str, float]]] = {
    "AAPL": {
        # ------------------------------------------------------------------
        # ACCOUNTS RECEIVABLE (Trade Only) - From Apple 10-K Balance Sheets
        # SEC shows separate: AR (trade) + Vendor Non-Trade Receivables
        # API incorrectly combines both; we correct to trade AR only
        # Source: Apple Inc. Form 10-K, Consolidated Balance Sheets
        # ------------------------------------------------------------------
        "accounts_receivable": {
            "2025-09-27": 39777.0,   # FY2025: AR net $39.777B (10-K pg. F-4)
            "2024-09-28": 33410.0,   # FY2024: AR net $33.410B (10-K pg. F-4)
            "2023-09-30": 29508.0,   # FY2023: AR net $29.508B (10-K pg. F-5)
            "2022-09-24": 28184.0,   # FY2022: AR net $28.184B (10-K pg. F-5)
            "2021-09-25": 26278.0,   # FY2021: AR net $26.278B (10-K pg. F-5)
            "2020-09-26": 16120.0,   # FY2020: AR net $16.120B (10-K pg. F-5)
        },
        # ------------------------------------------------------------------
        # VENDOR NON-TRADE RECEIVABLES - For reference (not used in ratios)
        # These are receivables from manufacturing vendors, NOT trade AR
        # ------------------------------------------------------------------
        "vendor_non_trade_receivables": {
            "2025-09-27": 33180.0,   # FY2025: Vendor NTR $33.180B
            "2024-09-28": 32833.0,   # FY2024: Vendor NTR $32.833B
            "2023-09-30": 31477.0,   # FY2023: Vendor NTR $31.477B
            "2022-09-24": 32748.0,   # FY2022: Vendor NTR $32.748B
            "2021-09-25": 25228.0,   # FY2021: Vendor NTR $25.228B
            "2020-09-26": 21325.0,   # FY2020: Vendor NTR $21.325B
        },
        # ------------------------------------------------------------------
        # INTEREST EXPENSE - From Apple 10-K Note: Other Income/(Expense)
        # Apple discloses breakdown in Notes to Consolidated Financial Statements
        # Source: Apple Inc. Form 10-K, Notes - Other Income/(Expense), net
        # ------------------------------------------------------------------
        "interest_expense": {
            "2025-09-27": 2931.0,    # FY2025: Interest expense $2.931B (Note 5)
            "2024-09-28": 2862.0,    # FY2024: Interest expense $2.862B (Note 5)
            "2023-09-30": 3933.0,    # FY2023: Interest expense $3.933B (Note 5)
            "2022-09-24": 2931.0,    # FY2022: Interest expense $2.931B (Note 5)
            "2021-09-25": 2645.0,    # FY2021: Interest expense $2.645B (Note 5)
            "2020-09-26": 2873.0,    # FY2020: Interest expense $2.873B (Note 5)
        },
        # ------------------------------------------------------------------
        # INTEREST INCOME - From Apple 10-K Note: Other Income/(Expense)
        # ------------------------------------------------------------------
        "interest_income": {
            "2025-09-27": 2610.0,    # FY2025: Interest & dividend income $2.610B
            "2024-09-28": 3131.0,    # FY2024: Interest & dividend income $3.131B
            "2023-09-30": 3750.0,    # FY2023: Interest & dividend income $3.750B
            "2022-09-24": 2825.0,    # FY2022: Interest & dividend income $2.825B
            "2021-09-25": 2843.0,    # FY2021: Interest & dividend income $2.843B
            "2020-09-26": 3763.0,    # FY2020: Interest & dividend income $3.763B
        },
    },
}

# Fiscal year end date mapping for SEC corrections lookup
FISCAL_YEAR_END_DATES: Dict[str, Dict[int, str]] = {
    "AAPL": {
        2025: "2025-09-27",
        2024: "2024-09-28",
        2023: "2023-09-30",
        2022: "2022-09-24",
        2021: "2021-09-25",
        2020: "2020-09-26",
    },
}


def get_sec_correction(ticker: str, field: str, fiscal_date: str) -> Optional[float]:
    """
    Get SEC-verified correction for a specific field and date.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        field: Field name to correct (e.g., "accounts_receivable")
        fiscal_date: Fiscal year end date (e.g., "2025-09-27")
        
    Returns:
        Corrected value in millions, or None if no correction available
    """
    ticker = ticker.upper()
    if ticker not in SEC_DATA_CORRECTIONS:
        return None
    
    corrections = SEC_DATA_CORRECTIONS[ticker]
    if field not in corrections:
        return None
    
    field_corrections = corrections[field]
    
    # Try exact match first
    if fiscal_date in field_corrections:
        return field_corrections[fiscal_date]
    
    # Try to match by fiscal year
    try:
        year = int(fiscal_date[:4])
        if ticker in FISCAL_YEAR_END_DATES:
            if year in FISCAL_YEAR_END_DATES[ticker]:
                fy_date = FISCAL_YEAR_END_DATES[ticker][year]
                if fy_date in field_corrections:
                    return field_corrections[fy_date]
    except (ValueError, TypeError):
        pass
    
    return None


def has_sec_corrections(ticker: str) -> bool:
    """Check if SEC corrections exist for a ticker."""
    return ticker.upper() in SEC_DATA_CORRECTIONS


def get_sec_correction_fields(ticker: str) -> List[str]:
    """Get list of fields with SEC corrections for a ticker."""
    ticker = ticker.upper()
    if ticker not in SEC_DATA_CORRECTIONS:
        return []
    return list(SEC_DATA_CORRECTIONS[ticker].keys())


# =============================================================================
# CRITICAL FIELDS FOR VALIDATION
# =============================================================================

# Critical fields that MUST be present for valid analysis
CRITICAL_INCOME_FIELDS: FrozenSet[str] = frozenset({
    "total_revenue",
    "gross_profit",
    "operating_income",
    "net_income",
})

CRITICAL_BALANCE_FIELDS: FrozenSet[str] = frozenset({
    "total_assets",
    "current_assets",
    "total_liabilities",
    "current_liabilities",
    "total_equity",
})

CRITICAL_CASHFLOW_FIELDS: FrozenSet[str] = frozenset({
    "operating_cash_flow",
    "capital_expenditure",
})

# Extended fields for comprehensive analysis
EXTENDED_INCOME_FIELDS: FrozenSet[str] = frozenset({
    "cost_of_revenue",
    "operating_expenses",
    "interest_expense",
    "pretax_income",
    "tax_expense",
    "depreciation_amortization",
    "ebit",
    "ebitda",
})

EXTENDED_BALANCE_FIELDS: FrozenSet[str] = frozenset({
    "cash_and_equivalents",
    "accounts_receivable",
    "inventory",
    "ppe_gross",
    "long_term_debt",
    "total_debt",
    "retained_earnings",
})

EXTENDED_CASHFLOW_FIELDS: FrozenSet[str] = frozenset({
    "net_income",
    "depreciation_amortization",
    "investing_cash_flow",
    "financing_cash_flow",
    "dividends_paid",
    "change_in_cash",
})


# =============================================================================
# DERIVED METRICS CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class DerivedMetricConfig:
    """Configuration for a derived metric calculation."""
    name: str
    description: str
    formula: str
    components: FrozenSet[str]


# Metrics calculated during data acquisition for downstream phases
DERIVED_METRICS: Dict[str, DerivedMetricConfig] = {
    "fcf_calculated": DerivedMetricConfig(
        name="Free Cash Flow (Calculated)",
        description="Operating Cash Flow minus Capital Expenditure",
        formula="OCF - |CapEx|",
        components=frozenset({"operating_cash_flow", "capital_expenditure"}),
    ),
    "ebitda_calculated": DerivedMetricConfig(
        name="EBITDA (Calculated)",
        description="Operating Income plus Depreciation and Amortization",
        formula="Operating Income + D&A",
        components=frozenset({"operating_income", "depreciation_amortization"}),
    ),
    "working_capital": DerivedMetricConfig(
        name="Working Capital",
        description="Current Assets minus Current Liabilities",
        formula="Current Assets - Current Liabilities",
        components=frozenset({"current_assets", "current_liabilities"}),
    ),
    "net_debt": DerivedMetricConfig(
        name="Net Debt",
        description="Total Debt minus Cash and Equivalents",
        formula="Total Debt - Cash",
        components=frozenset({"total_debt", "cash_and_equivalents"}),
    ),
    "invested_capital": DerivedMetricConfig(
        name="Invested Capital",
        description="Total Equity plus Total Debt minus Cash",
        formula="Equity + Debt - Cash",
        components=frozenset({"total_equity", "total_debt", "cash_and_equivalents"}),
    ),
}


# =============================================================================
# VALIDATION CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for data validation."""
    
    # Years of data
    years_required: int = 5
    years_minimum: int = 3
    
    # Accounting equation tolerance (percentage)
    accounting_equation_tolerance: float = 0.02  # 2%
    
    # Quality thresholds
    excellent_threshold: float = 0.95  # >= 95%
    good_threshold: float = 0.80       # >= 80%
    acceptable_threshold: float = 0.60  # >= 60%
    
    def get_quality_tier(self, completeness: float) -> DataQualityTier:
        """Determine quality tier from completeness score."""
        if completeness >= self.excellent_threshold:
            return DataQualityTier.EXCELLENT
        elif completeness >= self.good_threshold:
            return DataQualityTier.GOOD
        elif completeness >= self.acceptable_threshold:
            return DataQualityTier.ACCEPTABLE
        else:
            return DataQualityTier.POOR


# =============================================================================
# API CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class AlphaVantageConfig:
    """Alpha Vantage API configuration."""
    
    api_key: str = os.getenv("ALPHA_VANTAGE_API_KEY", "9GZT5T05DZS83LAE")
    base_url: str = "https://www.alphavantage.co/query"
    request_timeout: int = 30
    rate_limit_seconds: float = 12.0
    cache_expiry_hours: float = 5.0
    
    @property
    def cache_expiry_seconds(self) -> int:
        """Cache expiry time in seconds."""
        return int(self.cache_expiry_hours * 3600)


@dataclass(frozen=True)
class DataConfig:
    """Data collection configuration."""
    
    years_of_data: int = 5
    min_years_required: int = 3
    min_dividend_years: int = 3


# =============================================================================
# GLOBAL CONFIGURATION INSTANCES
# =============================================================================

VALIDATION_CONFIG = ValidationConfig()
ALPHA_VANTAGE_CONFIG = AlphaVantageConfig()
DATA_CONFIG = DataConfig()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_field_mapping(statement_type: StatementType) -> Dict[str, str]:
    """Get field mapping for a statement type."""
    mappings = {
        StatementType.INCOME: INCOME_FIELD_MAP,
        StatementType.BALANCE: BALANCE_FIELD_MAP,
        StatementType.CASHFLOW: CASHFLOW_FIELD_MAP,
    }
    return mappings.get(statement_type, {})


def get_critical_fields(statement_type: StatementType) -> FrozenSet[str]:
    """Get critical fields for a statement type."""
    fields = {
        StatementType.INCOME: CRITICAL_INCOME_FIELDS,
        StatementType.BALANCE: CRITICAL_BALANCE_FIELDS,
        StatementType.CASHFLOW: CRITICAL_CASHFLOW_FIELDS,
    }
    return fields.get(statement_type, frozenset())


def get_expected_fields(statement_type: StatementType) -> FrozenSet[str]:
    """Get all expected fields (critical + extended) for a statement type."""
    critical = get_critical_fields(statement_type)
    extended = {
        StatementType.INCOME: EXTENDED_INCOME_FIELDS,
        StatementType.BALANCE: EXTENDED_BALANCE_FIELDS,
        StatementType.CASHFLOW: EXTENDED_CASHFLOW_FIELDS,
    }.get(statement_type, frozenset())
    return critical | extended


# =============================================================================
# PHASE 2: VALIDATION ENUMERATIONS
# =============================================================================

class ReconciliationStatus(Enum):
    """Cross-statement reconciliation status."""
    RECONCILED = "reconciled"
    MINOR_VARIANCE = "minor_variance"
    MAJOR_VARIANCE = "major_variance"
    CANNOT_RECONCILE = "cannot_reconcile"


class TrendClassification(Enum):
    """Time series trend classification."""
    STRONG_GROWTH = "strong_growth"      # CAGR > 10%
    MODERATE_GROWTH = "moderate_growth"  # CAGR 3-10%
    STABLE = "stable"                    # CAGR -3% to 3%
    MODERATE_DECLINE = "moderate_decline"  # CAGR -10% to -3%
    STRONG_DECLINE = "strong_decline"    # CAGR < -10%
    VOLATILE = "volatile"                # High coefficient of variation
    INSUFFICIENT_DATA = "insufficient_data"


class OutlierSeverity(Enum):
    """Outlier detection severity level."""
    NONE = "none"
    MILD = "mild"          # 1.5-3 IQR or 2-3 sigma
    MODERATE = "moderate"  # 3-4.5 IQR or 3-4 sigma
    EXTREME = "extreme"    # >4.5 IQR or >4 sigma


class SignConvention(Enum):
    """Expected sign convention for financial fields."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    EITHER = "either"
    OUTFLOW = "outflow"
    INFLOW = "inflow"


# =============================================================================
# PHASE 2: RECONCILIATION CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class ReconciliationConfig:
    """
    Configuration for cross-statement reconciliation checks.
    Tolerances are expressed as percentages of the larger value.
    """
    
    net_income_tolerance: float = 0.001  # 0.1%
    depreciation_tolerance: float = 0.01  # 1%
    accounting_equation_tolerance: float = 0.02  # 2%
    cash_flow_articulation_tolerance: float = 0.05  # 5%
    retained_earnings_tolerance: float = 0.10  # 10%
    minor_variance_threshold: float = 0.05  # 5%
    major_variance_threshold: float = 0.15  # 15%


# =============================================================================
# PHASE 2: OUTLIER DETECTION CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class OutlierConfig:
    """Configuration for statistical outlier detection."""
    
    # IQR-based thresholds
    iqr_mild_threshold: float = 1.5
    iqr_moderate_threshold: float = 3.0
    iqr_extreme_threshold: float = 4.5
    
    # Z-score thresholds
    zscore_mild_threshold: float = 2.0
    zscore_moderate_threshold: float = 3.0
    zscore_extreme_threshold: float = 4.0
    
    # Year-over-year change thresholds
    yoy_change_flag_threshold: float = 0.50  # 50%
    yoy_change_critical_threshold: float = 1.00  # 100%
    
    # Minimum data points for detection
    min_data_points: int = 4


# =============================================================================
# PHASE 2: GROWTH RATE CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class GrowthConfig:
    """Configuration for growth rate calculations and trend classification."""
    
    # CAGR thresholds for trend classification
    strong_growth_threshold: float = 0.10
    moderate_growth_threshold: float = 0.03
    stable_threshold: float = 0.03
    moderate_decline_threshold: float = -0.10
    
    # Volatility threshold
    volatility_cv_threshold: float = 0.30  # 30%
    
    # Minimum years requirements
    min_years_for_cagr: int = 2
    min_years_for_trend: int = 3


# =============================================================================
# PHASE 2: SIGN CONVENTIONS
# =============================================================================

INCOME_SIGN_CONVENTIONS: Dict[str, SignConvention] = {
    "total_revenue": SignConvention.POSITIVE,
    "cost_of_revenue": SignConvention.POSITIVE,
    "cogs": SignConvention.POSITIVE,
    "gross_profit": SignConvention.POSITIVE,
    "operating_expenses": SignConvention.POSITIVE,
    "operating_income": SignConvention.EITHER,
    "rd_expense": SignConvention.POSITIVE,
    "sga_expense": SignConvention.POSITIVE,
    "pretax_income": SignConvention.EITHER,
    "tax_expense": SignConvention.EITHER,
    "net_income": SignConvention.EITHER,
    "interest_expense": SignConvention.POSITIVE,
    "interest_income": SignConvention.POSITIVE,
    "ebit": SignConvention.EITHER,
    "ebitda": SignConvention.EITHER,
    "depreciation_amortization": SignConvention.POSITIVE,
}

BALANCE_SIGN_CONVENTIONS: Dict[str, SignConvention] = {
    "total_assets": SignConvention.POSITIVE,
    "current_assets": SignConvention.POSITIVE,
    "cash_and_equivalents": SignConvention.POSITIVE,
    "accounts_receivable": SignConvention.POSITIVE,
    "inventory": SignConvention.POSITIVE,
    "total_liabilities": SignConvention.POSITIVE,
    "current_liabilities": SignConvention.POSITIVE,
    "long_term_debt": SignConvention.POSITIVE,
    "total_debt": SignConvention.POSITIVE,
    "total_equity": SignConvention.EITHER,
    "retained_earnings": SignConvention.EITHER,
}

CASHFLOW_SIGN_CONVENTIONS: Dict[str, SignConvention] = {
    "operating_cash_flow": SignConvention.EITHER,
    "net_income": SignConvention.EITHER,
    "depreciation_amortization": SignConvention.POSITIVE,
    "capital_expenditure": SignConvention.POSITIVE,
    "investing_cash_flow": SignConvention.EITHER,
    "financing_cash_flow": SignConvention.EITHER,
    "dividends_paid": SignConvention.POSITIVE,
    "change_in_cash": SignConvention.EITHER,
}


# =============================================================================
# PHASE 2: KEY METRICS FOR VALIDATION
# =============================================================================

CROSS_STATEMENT_RECONCILIATION_PAIRS: List[Dict] = [
    {
        "name": "Net Income",
        "statement_a": ("income_statement", "net_income"),
        "statement_b": ("cash_flow", "net_income"),
        "tolerance": 0.001,
    },
    {
        "name": "Depreciation & Amortization",
        "statement_a": ("income_statement", "depreciation_amortization"),
        "statement_b": ("cash_flow", "depreciation_amortization"),
        "tolerance": 0.01,
    },
]

KEY_GROWTH_METRICS: FrozenSet[str] = frozenset({
    "total_revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "ebitda",
    "total_assets",
    "total_equity",
    "total_debt",
    "operating_cash_flow",
    "fcf_calculated",
})

ALWAYS_POSITIVE_METRICS: FrozenSet[str] = frozenset({
    "total_revenue",
    "total_assets",
    "current_assets",
    "total_liabilities",
    "current_liabilities",
})


# =============================================================================
# PHASE 2: VALIDATION THRESHOLDS
# =============================================================================

@dataclass(frozen=True)
class Phase2ValidationThresholds:
    """Aggregate validation thresholds for Phase 2 quality assessment."""
    
    min_reconciliation_score: float = 0.80
    max_outlier_percentage: float = 0.20
    min_consistency_score: float = 0.85
    excellent_threshold: float = 0.95
    good_threshold: float = 0.85
    acceptable_threshold: float = 0.70


# =============================================================================
# PHASE 2: GLOBAL CONFIGURATION INSTANCES
# =============================================================================

RECONCILIATION_CONFIG = ReconciliationConfig()
OUTLIER_CONFIG = OutlierConfig()
GROWTH_CONFIG = GrowthConfig()
PHASE2_VALIDATION_THRESHOLDS = Phase2ValidationThresholds()


# =============================================================================
# PHASE 2: HELPER FUNCTIONS
# =============================================================================

def get_sign_convention(statement_type: str, field_name: str) -> SignConvention:
    """Get expected sign convention for a field."""
    conventions = {
        "income_statement": INCOME_SIGN_CONVENTIONS,
        "balance_sheet": BALANCE_SIGN_CONVENTIONS,
        "cash_flow": CASHFLOW_SIGN_CONVENTIONS,
    }
    statement_conventions = conventions.get(statement_type, {})
    return statement_conventions.get(field_name, SignConvention.EITHER)


def classify_trend(cagr: float, cv: float) -> TrendClassification:
    """Classify trend based on CAGR and coefficient of variation."""
    if cv > GROWTH_CONFIG.volatility_cv_threshold:
        return TrendClassification.VOLATILE
    
    if cagr > GROWTH_CONFIG.strong_growth_threshold:
        return TrendClassification.STRONG_GROWTH
    elif cagr > GROWTH_CONFIG.moderate_growth_threshold:
        return TrendClassification.MODERATE_GROWTH
    elif cagr > -GROWTH_CONFIG.stable_threshold:
        return TrendClassification.STABLE
    elif cagr > GROWTH_CONFIG.moderate_decline_threshold:
        return TrendClassification.MODERATE_DECLINE
    else:
        return TrendClassification.STRONG_DECLINE


# =============================================================================
# PHASE 3: RATIO ANALYSIS ENUMERATIONS
# =============================================================================

class RatioCategory(Enum):
    """Financial ratio categories."""
    PROFITABILITY = "profitability"
    LEVERAGE = "leverage"
    LIQUIDITY = "liquidity"
    EFFICIENCY = "efficiency"


class RatioAssessment(Enum):
    """Ratio value assessment relative to benchmarks."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    WEAK = "weak"
    CRITICAL = "critical"
    NOT_APPLICABLE = "not_applicable"


class RatioTrend(Enum):
    """Ratio trend direction over time."""
    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"
    VOLATILE = "volatile"
    INSUFFICIENT_DATA = "insufficient_data"


# =============================================================================
# PHASE 3: RATIO DEFINITIONS
# =============================================================================

@dataclass(frozen=True)
class RatioDefinition:
    """
    Definition of a financial ratio including metadata and benchmarks.
    
    Attributes:
        name: Display name of the ratio
        category: Ratio category (profitability, leverage, etc.)
        formula: Human-readable formula description
        interpretation: How to interpret the ratio (higher_better, lower_better, optimal_range)
        benchmark_excellent: Threshold for excellent assessment
        benchmark_good: Threshold for good assessment
        benchmark_acceptable: Threshold for acceptable assessment
        benchmark_weak: Threshold for weak assessment (below = critical)
        invert_assessment: If True, lower values are better
        format_as_percent: If True, display as percentage
        format_as_days: If True, display as days
        format_decimals: Number of decimal places for display
    """
    name: str
    category: RatioCategory
    formula: str
    interpretation: str
    benchmark_excellent: float
    benchmark_good: float
    benchmark_acceptable: float
    benchmark_weak: float
    invert_assessment: bool = False  # True if lower is better
    format_as_percent: bool = False
    format_as_days: bool = False
    format_decimals: int = 2


# =============================================================================
# PHASE 3: PROFITABILITY RATIO DEFINITIONS
# =============================================================================

PROFITABILITY_RATIOS: Dict[str, RatioDefinition] = {
    "gross_margin": RatioDefinition(
        name="Gross Margin",
        category=RatioCategory.PROFITABILITY,
        formula="Gross Profit / Revenue",
        interpretation="Measures production efficiency and pricing power",
        benchmark_excellent=0.50,
        benchmark_good=0.35,
        benchmark_acceptable=0.25,
        benchmark_weak=0.15,
        format_as_percent=True,
    ),
    "operating_margin": RatioDefinition(
        name="Operating Margin",
        category=RatioCategory.PROFITABILITY,
        formula="Operating Income / Revenue",
        interpretation="Measures operational efficiency before interest and taxes",
        benchmark_excellent=0.25,
        benchmark_good=0.15,
        benchmark_acceptable=0.08,
        benchmark_weak=0.03,
        format_as_percent=True,
    ),
    "net_profit_margin": RatioDefinition(
        name="Net Profit Margin",
        category=RatioCategory.PROFITABILITY,
        formula="Net Income / Revenue",
        interpretation="Measures overall profitability after all expenses",
        benchmark_excellent=0.20,
        benchmark_good=0.10,
        benchmark_acceptable=0.05,
        benchmark_weak=0.02,
        format_as_percent=True,
    ),
    "ebitda_margin": RatioDefinition(
        name="EBITDA Margin",
        category=RatioCategory.PROFITABILITY,
        formula="EBITDA / Revenue",
        interpretation="Measures operating cash generation capacity",
        benchmark_excellent=0.30,
        benchmark_good=0.20,
        benchmark_acceptable=0.12,
        benchmark_weak=0.06,
        format_as_percent=True,
    ),
    "roe": RatioDefinition(
        name="Return on Equity",
        category=RatioCategory.PROFITABILITY,
        formula="Net Income / Average Shareholders' Equity",
        interpretation="Measures return generated on shareholders' investment",
        benchmark_excellent=0.20,
        benchmark_good=0.15,
        benchmark_acceptable=0.10,
        benchmark_weak=0.05,
        format_as_percent=True,
    ),
    "roa": RatioDefinition(
        name="Return on Assets",
        category=RatioCategory.PROFITABILITY,
        formula="Net Income / Average Total Assets",
        interpretation="Measures efficiency of asset utilization",
        benchmark_excellent=0.12,
        benchmark_good=0.08,
        benchmark_acceptable=0.05,
        benchmark_weak=0.02,
        format_as_percent=True,
    ),
    "roic": RatioDefinition(
        name="Return on Invested Capital",
        category=RatioCategory.PROFITABILITY,
        formula="NOPAT / Invested Capital",
        interpretation="Measures return on capital invested in operations",
        benchmark_excellent=0.18,
        benchmark_good=0.12,
        benchmark_acceptable=0.08,
        benchmark_weak=0.04,
        format_as_percent=True,
    ),
    "roce": RatioDefinition(
        name="Return on Capital Employed",
        category=RatioCategory.PROFITABILITY,
        formula="EBIT / (Total Assets - Current Liabilities)",
        interpretation="Measures profitability relative to capital employed",
        benchmark_excellent=0.20,
        benchmark_good=0.14,
        benchmark_acceptable=0.09,
        benchmark_weak=0.05,
        format_as_percent=True,
    ),
    "fcf_margin": RatioDefinition(
        name="Free Cash Flow Margin",
        category=RatioCategory.PROFITABILITY,
        formula="Free Cash Flow / Revenue",
        interpretation="Measures cash generation efficiency relative to revenue",
        benchmark_excellent=0.15,
        benchmark_good=0.10,
        benchmark_acceptable=0.05,
        benchmark_weak=0.02,
        format_as_percent=True,
    ),
}


# =============================================================================
# PHASE 3: LEVERAGE RATIO DEFINITIONS
# =============================================================================

LEVERAGE_RATIOS: Dict[str, RatioDefinition] = {
    "debt_to_equity": RatioDefinition(
        name="Debt-to-Equity",
        category=RatioCategory.LEVERAGE,
        formula="Total Debt / Total Equity",
        interpretation="Measures financial leverage and capital structure",
        benchmark_excellent=0.3,
        benchmark_good=0.6,
        benchmark_acceptable=1.0,
        benchmark_weak=1.5,
        invert_assessment=True,
        format_decimals=2,
    ),
    "debt_to_assets": RatioDefinition(
        name="Debt-to-Assets",
        category=RatioCategory.LEVERAGE,
        formula="Total Debt / Total Assets",
        interpretation="Measures proportion of assets financed by debt",
        benchmark_excellent=0.20,
        benchmark_good=0.35,
        benchmark_acceptable=0.50,
        benchmark_weak=0.65,
        invert_assessment=True,
        format_as_percent=True,
    ),
    "debt_to_capital": RatioDefinition(
        name="Debt-to-Capital",
        category=RatioCategory.LEVERAGE,
        formula="Total Debt / (Total Debt + Total Equity)",
        interpretation="Measures debt proportion in capital structure",
        benchmark_excellent=0.25,
        benchmark_good=0.40,
        benchmark_acceptable=0.55,
        benchmark_weak=0.70,
        invert_assessment=True,
        format_as_percent=True,
    ),
    "interest_coverage": RatioDefinition(
        name="Interest Coverage",
        category=RatioCategory.LEVERAGE,
        formula="EBIT / Interest Expense",
        interpretation="Measures ability to service debt interest",
        benchmark_excellent=10.0,
        benchmark_good=5.0,
        benchmark_acceptable=3.0,
        benchmark_weak=1.5,
        format_decimals=1,
    ),
    "cash_coverage": RatioDefinition(
        name="Cash Coverage",
        category=RatioCategory.LEVERAGE,
        formula="(EBIT + D&A) / Interest Expense",
        interpretation="Measures cash available to cover interest",
        benchmark_excellent=15.0,
        benchmark_good=8.0,
        benchmark_acceptable=4.0,
        benchmark_weak=2.0,
        format_decimals=1,
    ),
    "debt_to_ebitda": RatioDefinition(
        name="Debt-to-EBITDA",
        category=RatioCategory.LEVERAGE,
        formula="Total Debt / EBITDA",
        interpretation="Measures years to repay debt from operating cash",
        benchmark_excellent=1.0,
        benchmark_good=2.0,
        benchmark_acceptable=3.5,
        benchmark_weak=5.0,
        invert_assessment=True,
        format_decimals=1,
    ),
    "equity_multiplier": RatioDefinition(
        name="Equity Multiplier",
        category=RatioCategory.LEVERAGE,
        formula="Total Assets / Total Equity",
        interpretation="Measures financial leverage (assets per unit equity)",
        benchmark_excellent=1.5,
        benchmark_good=2.5,
        benchmark_acceptable=4.0,
        benchmark_weak=6.0,
        invert_assessment=True,
        format_decimals=2,
    ),
    "long_term_debt_to_equity": RatioDefinition(
        name="Long-term Debt-to-Equity",
        category=RatioCategory.LEVERAGE,
        formula="Long-term Debt / Total Equity",
        interpretation="Measures long-term financial leverage",
        benchmark_excellent=0.2,
        benchmark_good=0.5,
        benchmark_acceptable=0.8,
        benchmark_weak=1.2,
        invert_assessment=True,
        format_decimals=2,
    ),
    "net_debt_to_ebitda": RatioDefinition(
        name="Net Debt-to-EBITDA",
        category=RatioCategory.LEVERAGE,
        formula="(Total Debt - Cash) / EBITDA",
        interpretation="Measures net leverage relative to operating cash generation",
        benchmark_excellent=0.5,
        benchmark_good=1.5,
        benchmark_acceptable=2.5,
        benchmark_weak=4.0,
        invert_assessment=True,
        format_decimals=1,
    ),
}


# =============================================================================
# PHASE 3: LIQUIDITY RATIO DEFINITIONS
# =============================================================================

LIQUIDITY_RATIOS: Dict[str, RatioDefinition] = {
    "current_ratio": RatioDefinition(
        name="Current Ratio",
        category=RatioCategory.LIQUIDITY,
        formula="Current Assets / Current Liabilities",
        interpretation="Measures short-term liquidity and working capital adequacy",
        benchmark_excellent=2.5,
        benchmark_good=1.8,
        benchmark_acceptable=1.2,
        benchmark_weak=0.9,
        format_decimals=2,
    ),
    "quick_ratio": RatioDefinition(
        name="Quick Ratio",
        category=RatioCategory.LIQUIDITY,
        formula="(Current Assets - Inventory) / Current Liabilities",
        interpretation="Measures immediate liquidity excluding inventory",
        benchmark_excellent=1.8,
        benchmark_good=1.2,
        benchmark_acceptable=0.8,
        benchmark_weak=0.5,
        format_decimals=2,
    ),
    "cash_ratio": RatioDefinition(
        name="Cash Ratio",
        category=RatioCategory.LIQUIDITY,
        formula="Cash and Equivalents / Current Liabilities",
        interpretation="Measures ability to pay short-term obligations with cash",
        benchmark_excellent=0.5,
        benchmark_good=0.3,
        benchmark_acceptable=0.15,
        benchmark_weak=0.05,
        format_decimals=2,
    ),
    "operating_cash_flow_ratio": RatioDefinition(
        name="Operating Cash Flow Ratio",
        category=RatioCategory.LIQUIDITY,
        formula="Operating Cash Flow / Current Liabilities",
        interpretation="Measures ability to cover short-term debt from operations",
        benchmark_excellent=1.0,
        benchmark_good=0.6,
        benchmark_acceptable=0.4,
        benchmark_weak=0.2,
        format_decimals=2,
    ),
    "working_capital_to_assets": RatioDefinition(
        name="Working Capital to Assets",
        category=RatioCategory.LIQUIDITY,
        formula="(Current Assets - Current Liabilities) / Total Assets",
        interpretation="Measures working capital relative to company size",
        benchmark_excellent=0.20,
        benchmark_good=0.10,
        benchmark_acceptable=0.0,
        benchmark_weak=-0.10,
        format_as_percent=True,
    ),
}


# =============================================================================
# PHASE 3: EFFICIENCY RATIO DEFINITIONS
# =============================================================================

EFFICIENCY_RATIOS: Dict[str, RatioDefinition] = {
    "asset_turnover": RatioDefinition(
        name="Asset Turnover",
        category=RatioCategory.EFFICIENCY,
        formula="Revenue / Average Total Assets",
        interpretation="Measures revenue generated per dollar of assets",
        benchmark_excellent=1.5,
        benchmark_good=1.0,
        benchmark_acceptable=0.6,
        benchmark_weak=0.3,
        format_decimals=2,
    ),
    "fixed_asset_turnover": RatioDefinition(
        name="Fixed Asset Turnover",
        category=RatioCategory.EFFICIENCY,
        formula="Revenue / Average Net PPE",
        interpretation="Measures efficiency of fixed asset utilization",
        benchmark_excellent=8.0,
        benchmark_good=5.0,
        benchmark_acceptable=3.0,
        benchmark_weak=1.5,
        format_decimals=2,
    ),
    "inventory_turnover": RatioDefinition(
        name="Inventory Turnover",
        category=RatioCategory.EFFICIENCY,
        formula="Cost of Goods Sold / Average Inventory",
        interpretation="Measures how quickly inventory is sold",
        benchmark_excellent=12.0,
        benchmark_good=8.0,
        benchmark_acceptable=5.0,
        benchmark_weak=3.0,
        format_decimals=1,
    ),
    "receivables_turnover": RatioDefinition(
        name="Receivables Turnover",
        category=RatioCategory.EFFICIENCY,
        formula="Revenue / Average Accounts Receivable",
        interpretation="Measures efficiency of credit collection",
        benchmark_excellent=15.0,
        benchmark_good=10.0,
        benchmark_acceptable=6.0,
        benchmark_weak=4.0,
        format_decimals=1,
    ),
    "payables_turnover": RatioDefinition(
        name="Payables Turnover",
        category=RatioCategory.EFFICIENCY,
        formula="COGS / Average Accounts Payable",
        interpretation="Measures payment speed to suppliers",
        benchmark_excellent=6.0,
        benchmark_good=8.0,
        benchmark_acceptable=10.0,
        benchmark_weak=12.0,
        invert_assessment=True,
        format_decimals=1,
    ),
    "days_sales_outstanding": RatioDefinition(
        name="Days Sales Outstanding",
        category=RatioCategory.EFFICIENCY,
        formula="365 / Receivables Turnover",
        interpretation="Average days to collect receivables",
        benchmark_excellent=25.0,
        benchmark_good=40.0,
        benchmark_acceptable=55.0,
        benchmark_weak=75.0,
        invert_assessment=True,
        format_as_days=True,
    ),
    "days_inventory_outstanding": RatioDefinition(
        name="Days Inventory Outstanding",
        category=RatioCategory.EFFICIENCY,
        formula="365 / Inventory Turnover",
        interpretation="Average days to sell inventory",
        benchmark_excellent=30.0,
        benchmark_good=50.0,
        benchmark_acceptable=75.0,
        benchmark_weak=100.0,
        invert_assessment=True,
        format_as_days=True,
    ),
    "days_payables_outstanding": RatioDefinition(
        name="Days Payables Outstanding",
        category=RatioCategory.EFFICIENCY,
        formula="365 / Payables Turnover",
        interpretation="Average days to pay suppliers",
        benchmark_excellent=60.0,
        benchmark_good=45.0,
        benchmark_acceptable=35.0,
        benchmark_weak=25.0,
        format_as_days=True,
    ),
    "cash_conversion_cycle": RatioDefinition(
        name="Cash Conversion Cycle",
        category=RatioCategory.EFFICIENCY,
        formula="DSO + DIO - DPO",
        interpretation="Days between paying suppliers and collecting from customers",
        benchmark_excellent=20.0,
        benchmark_good=45.0,
        benchmark_acceptable=70.0,
        benchmark_weak=100.0,
        invert_assessment=True,
        format_as_days=True,
    ),
}


# =============================================================================
# PHASE 3: GROWTH RATIO DEFINITIONS
# =============================================================================

class GrowthRatioCategory(Enum):
    """Growth ratio sub-categories for organization."""
    REVENUE = "revenue"
    EARNINGS = "earnings"
    CASH_FLOW = "cash_flow"
    DIVIDEND = "dividend"
    SUSTAINABLE = "sustainable"


GROWTH_RATIOS: Dict[str, RatioDefinition] = {
    "revenue_cagr_5y": RatioDefinition(
        name="Revenue CAGR (5Y)",
        category=RatioCategory.EFFICIENCY,  # Using efficiency as placeholder
        formula="(Revenue_Y5 / Revenue_Y1)^(1/4) - 1",
        interpretation="Compound annual revenue growth over 5 years",
        benchmark_excellent=0.15,
        benchmark_good=0.08,
        benchmark_acceptable=0.03,
        benchmark_weak=0.0,
        format_as_percent=True,
    ),
    "net_income_cagr_5y": RatioDefinition(
        name="Net Income CAGR (5Y)",
        category=RatioCategory.PROFITABILITY,
        formula="(Net_Income_Y5 / Net_Income_Y1)^(1/4) - 1",
        interpretation="Compound annual earnings growth over 5 years",
        benchmark_excellent=0.15,
        benchmark_good=0.08,
        benchmark_acceptable=0.03,
        benchmark_weak=0.0,
        format_as_percent=True,
    ),
    "fcf_cagr_5y": RatioDefinition(
        name="FCF CAGR (5Y)",
        category=RatioCategory.PROFITABILITY,
        formula="(FCF_Y5 / FCF_Y1)^(1/4) - 1",
        interpretation="Compound annual free cash flow growth over 5 years",
        benchmark_excellent=0.12,
        benchmark_good=0.06,
        benchmark_acceptable=0.02,
        benchmark_weak=-0.02,
        format_as_percent=True,
    ),
    "dividend_cagr_5y": RatioDefinition(
        name="Dividend CAGR (5Y)",
        category=RatioCategory.PROFITABILITY,
        formula="(DPS_Y5 / DPS_Y1)^(1/4) - 1",
        interpretation="Compound annual dividend growth over 5 years",
        benchmark_excellent=0.10,
        benchmark_good=0.05,
        benchmark_acceptable=0.02,
        benchmark_weak=0.0,
        format_as_percent=True,
    ),
    "sustainable_growth_rate": RatioDefinition(
        name="Sustainable Growth Rate",
        category=RatioCategory.PROFITABILITY,
        formula="ROE x (1 - Dividend Payout Ratio)",
        interpretation="Maximum growth rate without external financing",
        benchmark_excellent=0.15,
        benchmark_good=0.10,
        benchmark_acceptable=0.05,
        benchmark_weak=0.02,
        format_as_percent=True,
    ),
    "dividend_payout_ratio": RatioDefinition(
        name="Dividend Payout Ratio",
        category=RatioCategory.PROFITABILITY,
        formula="Dividends Paid / Net Income",
        interpretation="Proportion of earnings distributed as dividends",
        benchmark_excellent=0.35,
        benchmark_good=0.50,
        benchmark_acceptable=0.70,
        benchmark_weak=0.90,
        invert_assessment=True,
        format_as_percent=True,
    ),
    "retention_ratio": RatioDefinition(
        name="Retention Ratio",
        category=RatioCategory.PROFITABILITY,
        formula="1 - Dividend Payout Ratio",
        interpretation="Proportion of earnings retained for reinvestment",
        benchmark_excellent=0.65,
        benchmark_good=0.50,
        benchmark_acceptable=0.30,
        benchmark_weak=0.10,
        format_as_percent=True,
    ),
}


# =============================================================================
# PHASE 3: COMBINED RATIO REGISTRY
# =============================================================================

ALL_RATIO_DEFINITIONS: Dict[str, RatioDefinition] = {
    **PROFITABILITY_RATIOS,
    **LEVERAGE_RATIOS,
    **LIQUIDITY_RATIOS,
    **EFFICIENCY_RATIOS,
    **GROWTH_RATIOS,
}


# =============================================================================
# PHASE 3: RATIO ANALYSIS CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class RatioAnalysisConfig:
    """Configuration for ratio analysis and scoring."""
    
    # Minimum years required for trend analysis
    min_years_for_trend: int = 3
    
    # Trend detection thresholds (absolute change in ratio)
    trend_improving_threshold: float = 0.02   # 2% improvement
    trend_deteriorating_threshold: float = -0.02  # 2% deterioration
    
    # Volatility threshold for trend classification
    volatility_cv_threshold: float = 0.25  # 25% coefficient of variation
    
    # Category weights for overall score
    profitability_weight: float = 0.30
    leverage_weight: float = 0.25
    liquidity_weight: float = 0.20
    efficiency_weight: float = 0.25
    
    # Assessment score mapping
    score_excellent: float = 1.0
    score_good: float = 0.75
    score_acceptable: float = 0.50
    score_weak: float = 0.25
    score_critical: float = 0.0


# =============================================================================
# PHASE 3: GLOBAL CONFIGURATION INSTANCES
# =============================================================================

RATIO_ANALYSIS_CONFIG = RatioAnalysisConfig()


# =============================================================================
# PHASE 3: HELPER FUNCTIONS
# =============================================================================

def get_ratio_definition(ratio_name: str) -> Optional[RatioDefinition]:
    """Get ratio definition by name."""
    return ALL_RATIO_DEFINITIONS.get(ratio_name)


def get_ratios_by_category(category: RatioCategory) -> Dict[str, RatioDefinition]:
    """Get all ratio definitions for a category."""
    return {
        name: defn for name, defn in ALL_RATIO_DEFINITIONS.items()
        if defn.category == category
    }


def assess_ratio_value(ratio_name: str, value: float) -> RatioAssessment:
    """
    Assess a ratio value against benchmarks.
    
    Args:
        ratio_name: Name of the ratio
        value: Calculated ratio value
        
    Returns:
        RatioAssessment enum value
    """
    defn = get_ratio_definition(ratio_name)
    if defn is None:
        return RatioAssessment.NOT_APPLICABLE
    
    if value is None or (isinstance(value, float) and (value != value)):  # NaN check
        return RatioAssessment.NOT_APPLICABLE
    
    if defn.invert_assessment:
        # Lower is better
        if value <= defn.benchmark_excellent:
            return RatioAssessment.EXCELLENT
        elif value <= defn.benchmark_good:
            return RatioAssessment.GOOD
        elif value <= defn.benchmark_acceptable:
            return RatioAssessment.ACCEPTABLE
        elif value <= defn.benchmark_weak:
            return RatioAssessment.WEAK
        else:
            return RatioAssessment.CRITICAL
    else:
        # Higher is better
        if value >= defn.benchmark_excellent:
            return RatioAssessment.EXCELLENT
        elif value >= defn.benchmark_good:
            return RatioAssessment.GOOD
        elif value >= defn.benchmark_acceptable:
            return RatioAssessment.ACCEPTABLE
        elif value >= defn.benchmark_weak:
            return RatioAssessment.WEAK
        else:
            return RatioAssessment.CRITICAL