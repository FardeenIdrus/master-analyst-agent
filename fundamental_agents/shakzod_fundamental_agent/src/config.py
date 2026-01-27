"""
Configuration Module for Fundamental Analyst Agent

This module centralizes all configuration constants, field mappings,
validation thresholds, and settings used throughout the analysis pipeline.

All "magic numbers" and configuration values are defined here to ensure:
1. Single source of truth for all constants
2. Easy modification without touching analysis code
3. Transparency in assumptions and thresholds
4. Consistency across all modules

MSc Coursework: AI Agents in Asset Management
Track A: Fundamental Analyst Agent
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ValidationStatus(Enum):
    """Status codes for validation results."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


class AnalysisStep(Enum):
    """Enumeration of analysis pipeline steps."""
    DATA_COLLECTION = "Data Collection"
    DATA_PROCESSING = "Data Processing"
    PROFITABILITY_ANALYSIS = "Profitability Analysis"
    CASH_FLOW_ANALYSIS = "Cash Flow Analysis"
    EARNINGS_QUALITY = "Earnings Quality Assessment"
    WORKING_CAPITAL = "Working Capital Analysis"
    RATIO_CALCULATION = "Financial Ratio Calculation"
    VALUATION = "Valuation Analysis"
    MEMO_GENERATION = "Investment Memo Generation"


class EarningsQualityRating(Enum):
    """Earnings quality classification."""
    HIGH = "High Quality"
    MODERATE = "Moderate Quality"
    LOW = "Low Quality"
    CONCERN = "Quality Concern"


class ValuationAssessment(Enum):
    """
    Valuation classification based on relative multiple analysis.
    
    Assessment is determined by comparing current valuation multiples to
    historical averages and implied growth rates:
    
    - SIGNIFICANTLY_UNDERVALUED: Multiple <50% of fair value estimate
    - UNDERVALUED: Multiple 50-80% of fair value estimate
    - FAIRLY_VALUED: Multiple 80-120% of fair value estimate
    - OVERVALUED: Multiple 120-150% of fair value estimate
    - SIGNIFICANTLY_OVERVALUED: Multiple >150% of fair value estimate
    """
    SIGNIFICANTLY_UNDERVALUED = "Significantly Undervalued"
    UNDERVALUED = "Undervalued"
    FAIRLY_VALUED = "Fairly Valued"
    OVERVALUED = "Overvalued"
    SIGNIFICANTLY_OVERVALUED = "Significantly Overvalued"


# =============================================================================
# FIELD NAME MAPPINGS
# =============================================================================

# Yahoo Finance field names can vary. These mappings provide alternatives
# to try when extracting data from financial statements.

INCOME_STATEMENT_FIELDS: Dict[str, List[str]] = {
    # Revenue
    "total_revenue": [
        "Total Revenue",
        "TotalRevenue",
        "Revenue",
        "Revenues",
        "Net Sales",
        "Sales"
    ],
    
    # Cost of Revenue
    "cost_of_revenue": [
        "Cost Of Revenue",
        "CostOfRevenue",
        "Cost of Goods Sold",
        "COGS",
        "Cost Of Goods Sold"
    ],
    
    # Gross Profit
    "gross_profit": [
        "Gross Profit",
        "GrossProfit",
        "Gross Income"
    ],
    
    # Operating Expenses
    "operating_expenses": [
        "Operating Expense",
        "OperatingExpense",
        "Operating Expenses",
        "Total Operating Expenses"
    ],
    
    # Research and Development
    "research_development": [
        "Research And Development",
        "ResearchAndDevelopment",
        "R&D Expense",
        "Research & Development"
    ],
    
    # Selling, General & Administrative
    "sga_expense": [
        "Selling General And Administration",
        "SellingGeneralAndAdministration",
        "SG&A",
        "Selling General Administrative"
    ],
    
    # Operating Income (EBIT)
    "operating_income": [
        "Operating Income",
        "OperatingIncome",
        "EBIT",
        "Earnings Before Interest and Taxes"
    ],
    
    # Interest Expense
    "interest_expense": [
        "Interest Expense",
        "InterestExpense",
        "Interest Expense Non Operating"
    ],
    
    # Pretax Income
    "pretax_income": [
        "Pretax Income",
        "PretaxIncome",
        "Income Before Tax",
        "EBT"
    ],
    
    # Tax Provision
    "tax_provision": [
        "Tax Provision",
        "TaxProvision",
        "Income Tax Expense",
        "Provision For Income Taxes"
    ],
    
    # Net Income
    "net_income": [
        "Net Income",
        "NetIncome",
        "Net Income Common Stockholders",
        "Net Profit"
    ],
    
    # EBITDA
    "ebitda": [
        "EBITDA",
        "Ebitda",
        "Normalized EBITDA"
    ],
    
    # Depreciation and Amortization (from income statement)
    "depreciation_amortization": [
        "Depreciation And Amortization",
        "DepreciationAndAmortization",
        "Depreciation",
        "D&A"
    ],
    
    # Basic EPS
    "basic_eps": [
        "Basic EPS",
        "BasicEPS",
        "Diluted EPS",
        "EPS"
    ],
    
    # Shares Outstanding
    "shares_outstanding": [
        "Basic Average Shares",
        "BasicAverageShares",
        "Diluted Average Shares",
        "Weighted Average Shares Outstanding"
    ]
}


BALANCE_SHEET_FIELDS: Dict[str, List[str]] = {
    # Assets
    "total_assets": [
        "Total Assets",
        "TotalAssets"
    ],
    
    "current_assets": [
        "Current Assets",
        "CurrentAssets",
        "Total Current Assets"
    ],
    
    "cash_and_equivalents": [
        "Cash And Cash Equivalents",
        "CashAndCashEquivalents",
        "Cash",
        "Cash Financial"
    ],
    
    "short_term_investments": [
        "Other Short Term Investments",
        "OtherShortTermInvestments",
        "Short Term Investments",
        "Marketable Securities"
    ],
    
    "accounts_receivable": [
        "Accounts Receivable",
        "AccountsReceivable",
        "Receivables",
        "Net Receivables"
    ],
    
    "inventory": [
        "Inventory",
        "Inventories",
        "Total Inventory"
    ],
    
    "other_current_assets": [
        "Other Current Assets",
        "OtherCurrentAssets"
    ],
    
    # Non-current Assets
    "ppe_net": [
        "Net PPE",
        "NetPPE",
        "Property Plant And Equipment Net",
        "Property Plant Equipment"
    ],
    
    "goodwill": [
        "Goodwill",
        "GoodWill"
    ],
    
    "intangible_assets": [
        "Intangible Assets",
        "IntangibleAssets",
        "Other Intangible Assets"
    ],
    
    # Liabilities
    "total_liabilities": [
        "Total Liabilities Net Minority Interest",
        "TotalLiabilitiesNetMinorityInterest",
        "Total Liabilities"
    ],
    
    "current_liabilities": [
        "Current Liabilities",
        "CurrentLiabilities",
        "Total Current Liabilities"
    ],
    
    "accounts_payable": [
        "Accounts Payable",
        "AccountsPayable",
        "Payables"
    ],
    
    "short_term_debt": [
        "Current Debt",
        "CurrentDebt",
        "Short Term Debt",
        "Current Portion Long Term Debt"
    ],
    
    "long_term_debt": [
        "Long Term Debt",
        "LongTermDebt",
        "Long Term Debt And Capital Lease Obligation"
    ],
    
    "total_debt": [
        "Total Debt",
        "TotalDebt"
    ],
    
    # Equity
    "total_equity": [
        "Total Equity Gross Minority Interest",
        "TotalEquityGrossMinorityInterest",
        "Stockholders Equity",
        "Total Stockholders Equity",
        "Total Equity"
    ],
    
    "retained_earnings": [
        "Retained Earnings",
        "RetainedEarnings"
    ],
    
    "common_stock": [
        "Common Stock",
        "CommonStock",
        "Common Stock Equity"
    ]
}


CASH_FLOW_FIELDS: Dict[str, List[str]] = {
    # Operating Cash Flow
    "operating_cash_flow": [
        "Operating Cash Flow",
        "OperatingCashFlow",
        "Cash Flow From Continuing Operating Activities",
        "Net Cash Provided By Operating Activities"
    ],
    
    # Net Income (starting point)
    "net_income_cf": [
        "Net Income From Continuing Operations",
        "NetIncomeFromContinuingOperations",
        "Net Income"
    ],
    
    # Depreciation & Amortization
    "depreciation_amortization_cf": [
        "Depreciation And Amortization",
        "DepreciationAndAmortization",
        "Depreciation Amortization Depletion"
    ],
    
    # Stock Based Compensation
    "stock_based_compensation": [
        "Stock Based Compensation",
        "StockBasedCompensation",
        "Share Based Compensation"
    ],
    
    # Deferred Taxes
    "deferred_taxes": [
        "Deferred Tax",
        "DeferredTax",
        "Deferred Income Tax",
        "Change In Deferred Tax"
    ],
    
    # Change in Working Capital
    "change_in_working_capital": [
        "Change In Working Capital",
        "ChangeInWorkingCapital",
        "Changes In Working Capital"
    ],
    
    # Change in Receivables
    "change_in_receivables": [
        "Change In Receivables",
        "ChangeInReceivables",
        "Changes In Account Receivables"
    ],
    
    # Change in Inventory
    "change_in_inventory": [
        "Change In Inventory",
        "ChangeInInventory",
        "Changes In Inventories"
    ],
    
    # Change in Payables
    "change_in_payables": [
        "Change In Payables And Accrued Expense",
        "ChangeInPayablesAndAccruedExpense",
        "Change In Payables",
        "Changes In Account Payables"
    ],
    
    # Other Operating Activities
    "other_operating_activities": [
        "Other Non Cash Items",
        "OtherNonCashItems",
        "Other Operating Cash Flow Items"
    ],
    
    # Investing Cash Flow
    "investing_cash_flow": [
        "Investing Cash Flow",
        "InvestingCashFlow",
        "Cash Flow From Continuing Investing Activities"
    ],
    
    # Capital Expenditure
    "capital_expenditure": [
        "Capital Expenditure",
        "CapitalExpenditure",
        "Purchase Of PPE",
        "Capital Expenditure Reported Negative"
    ],
    
    # Financing Cash Flow
    "financing_cash_flow": [
        "Financing Cash Flow",
        "FinancingCashFlow",
        "Cash Flow From Continuing Financing Activities"
    ],
    
    # Dividends Paid
    "dividends_paid": [
        "Cash Dividends Paid",
        "CashDividendsPaid",
        "Common Stock Dividend Paid",
        "Payment Of Dividends"
    ],
    
    # Share Repurchases
    "share_repurchases": [
        "Repurchase Of Capital Stock",
        "RepurchaseOfCapitalStock",
        "Common Stock Payments",
        "Buyback Of Shares"
    ],
    
    # Free Cash Flow
    "free_cash_flow": [
        "Free Cash Flow",
        "FreeCashFlow"
    ]
}


# =============================================================================
# VALIDATION THRESHOLDS
# =============================================================================

@dataclass(frozen=True)
class ValidationThresholds:
    """Thresholds for validation checks throughout the analysis."""
    
    # Reconciliation tolerances (in millions USD)
    ebit_bridge_tolerance_mm: float = 1.0  # Bridge must reconcile within $1M
    ocf_bridge_tolerance_pct: float = 0.02  # OCF reconciliation within 2%
    
    # Data requirements - Updated to require 5 complete years as per coursework
    minimum_years_required: int = 5  # Minimum years for analysis (coursework requirement)
    preferred_years: int = 5  # Preferred years of data
    minimum_usable_years: int = 3  # Fallback minimum if 5 years unavailable
    
    # Reasonableness checks for working capital days
    max_reasonable_dso: int = 180  # Days - flag if exceeded
    max_reasonable_dio: int = 365  # Days - flag if exceeded
    max_reasonable_dpo: int = 180  # Days - flag if exceeded
    
    # Margin bounds (flag if outside)
    min_reasonable_gross_margin: float = -0.50  # -50%
    max_reasonable_gross_margin: float = 0.95   # 95%
    min_reasonable_operating_margin: float = -1.0  # -100%
    max_reasonable_operating_margin: float = 0.80  # 80%


# =============================================================================
# EARNINGS QUALITY THRESHOLDS
# =============================================================================

@dataclass(frozen=True)
class EarningsQualityThresholds:
    """Thresholds for earnings quality assessment."""
    
    # Accruals ratio thresholds
    # Accruals = (Net Income - OCF) / Average Total Assets
    accruals_high_quality: float = 0.05    # <5% = high quality
    accruals_moderate_quality: float = 0.10  # 5-10% = moderate
    accruals_low_quality: float = 0.15     # 10-15% = low quality
    # >15% = quality concern
    
    # Cash conversion rate thresholds
    # Cash Conversion = OCF / Net Income
    cash_conversion_excellent: float = 1.10  # >110% = excellent
    cash_conversion_good: float = 0.90       # 90-110% = good
    cash_conversion_acceptable: float = 0.70  # 70-90% = acceptable
    # <70% = concern
    
    # Growth comparison thresholds (for red flag detection)
    # Flag if AR growth exceeds revenue growth by this margin
    ar_vs_revenue_threshold: float = 0.10  # 10 percentage points
    
    # Flag if inventory growth exceeds COGS growth by this margin
    inventory_vs_cogs_threshold: float = 0.10  # 10 percentage points


# =============================================================================
# VALUATION PARAMETERS
# =============================================================================

@dataclass(frozen=True)
class ValuationParameters:
    """
    Parameters for valuation multiples assessment and implied growth analysis.
    
    Contains thresholds for classifying valuation multiples as undervalued,
    fairly valued, or overvalued, as well as parameters for cost of equity
    estimation used in implied growth calculations.
    """
    
    # Cost of equity parameters (used for implied growth analysis via CAPM)
    risk_free_rate: float = 0.04           # 4% risk-free rate assumption
    equity_risk_premium: float = 0.05      # 5% equity risk premium
    
    # P/E ratio bands for valuation assessment
    pe_significantly_undervalued: float = 8.0    # P/E below this is significantly undervalued
    pe_undervalued: float = 12.0                 # P/E below this is undervalued
    pe_fairly_valued_low: float = 12.0           # Lower bound of fair value range
    pe_fairly_valued_high: float = 25.0          # Upper bound of fair value range
    pe_overvalued: float = 25.0                  # P/E above this is overvalued
    pe_significantly_overvalued: float = 40.0    # P/E above this is significantly overvalued
    
    # EV/EBITDA bands for valuation assessment
    ev_ebitda_significantly_undervalued: float = 5.0    # Below this is significantly undervalued
    ev_ebitda_undervalued: float = 8.0                  # Below this is undervalued
    ev_ebitda_fairly_valued_low: float = 8.0            # Lower bound of fair value range
    ev_ebitda_fairly_valued_high: float = 15.0          # Upper bound of fair value range
    ev_ebitda_overvalued: float = 15.0                  # Above this is overvalued
    ev_ebitda_significantly_overvalued: float = 20.0    # Above this is significantly overvalued
    
    # Price/FCF bands for valuation assessment
    price_fcf_significantly_undervalued: float = 10.0   # Below this is significantly undervalued
    price_fcf_undervalued: float = 15.0                 # Below this is undervalued
    price_fcf_fairly_valued_low: float = 15.0           # Lower bound of fair value range
    price_fcf_fairly_valued_high: float = 30.0          # Upper bound of fair value range
    price_fcf_overvalued: float = 30.0                  # Above this is overvalued
    price_fcf_significantly_overvalued: float = 50.0    # Above this is significantly overvalued


# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class OutputConfig:
    """Configuration for output generation."""
    
    # Enabled output formats
    json_enabled: bool = True
    excel_enabled: bool = True
    markdown_enabled: bool = True
    html_enabled: bool = False  # Optional
    
    # Number formatting
    currency_decimal_places: int = 2
    percentage_decimal_places: int = 2
    ratio_decimal_places: int = 2
    days_decimal_places: int = 1
    
    # Display settings
    values_in_millions: bool = True
    show_validation_details: bool = True


# =============================================================================
# GLOBAL CONFIGURATION INSTANCES
# =============================================================================

# Create singleton instances of configuration classes
VALIDATION = ValidationThresholds()
EARNINGS_QUALITY = EarningsQualityThresholds()
VALUATION_PARAMS = ValuationParameters()
OUTPUT = OutputConfig()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_field_alternatives(statement_type: str, field_name: str) -> List[str]:
    """
    Get list of alternative field names for a given field.
    
    Args:
        statement_type: One of 'income', 'balance', 'cashflow'
        field_name: Standardized field name (e.g., 'total_revenue')
        
    Returns:
        List of alternative field names to try
    """
    mappings = {
        'income': INCOME_STATEMENT_FIELDS,
        'balance': BALANCE_SHEET_FIELDS,
        'cashflow': CASH_FLOW_FIELDS
    }
    
    statement_fields = mappings.get(statement_type, {})
    return statement_fields.get(field_name, [field_name])


def get_all_required_fields() -> Dict[str, List[str]]:
    """
    Get all required fields for minimum viable analysis.
    
    Returns:
        Dictionary mapping statement type to required field names
    """
    return {
        'income': [
            'total_revenue',
            'cost_of_revenue',
            'operating_income',
            'net_income'
        ],
        'balance': [
            'total_assets',
            'accounts_receivable',
            'inventory',
            'accounts_payable',
            'total_debt',
            'total_equity'
        ],
        'cashflow': [
            'operating_cash_flow',
            'capital_expenditure',
            'depreciation_amortization_cf'
        ]
    }