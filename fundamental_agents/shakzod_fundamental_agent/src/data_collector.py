"""
Data Collector Module for Fundamental Analyst Agent

This module implements the data ingestion pipeline that fetches financial
statements and company information. It uses Alpha Vantage as the primary
data source (provides 5+ years of complete annual data) with Yahoo Finance
as a fallback for company metadata.

Module Responsibilities:
    1. Fetch 5 years of annual financial statements:
       - Income Statement (P&L)
       - Balance Sheet
       - Cash Flow Statement
    2. Fetch company metadata
    3. Validate data completeness and quality
    4. Return structured data containers for downstream analysis

Data Sources:
    - Primary: Alpha Vantage API (free, 5+ years of annual data)
    - Secondary: Yahoo Finance (for company metadata/fallback)

IMPORTANT: Requires Alpha Vantage API key. Get free key at:
    https://www.alphavantage.co/support/#api-key
    
Set environment variable: ALPHA_VANTAGE_API_KEY=your_key_here

MSc Coursework: AI Agents in Asset Management
Track A: Fundamental Analyst Agent

Author: MSc AI Agents in Asset Management
Version: 4.0.0 (Alpha Vantage integration)
"""

# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations

import logging
import math
import os
import time
import requests
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np

# Conditional import of yfinance for company metadata
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

# Local imports
from config import (
    VALIDATION,
    ValidationStatus,
    get_field_alternatives,
    get_all_required_fields
)


# =============================================================================
# MODULE CONFIGURATION
# =============================================================================

logger = logging.getLogger(__name__)

DEFAULT_DATA_SOURCE = "Alpha Vantage"
TICKER_MAX_LENGTH = 10

# Alpha Vantage Configuration
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
ALPHA_VANTAGE_TIMEOUT = 30
ALPHA_VANTAGE_RATE_LIMIT_DELAY = 12  # seconds between calls (5 calls/min limit)


# =============================================================================
# ALPHA VANTAGE FIELD MAPPINGS
# =============================================================================

# Map Alpha Vantage field names to standardized names
AV_INCOME_FIELD_MAP = {
    "totalRevenue": "Total Revenue",
    "costOfRevenue": "Cost Of Revenue",
    "costofGoodsAndServicesSold": "Cost Of Revenue",
    "grossProfit": "Gross Profit",
    "operatingIncome": "Operating Income",
    "operatingExpenses": "Operating Expense",
    "researchAndDevelopment": "Research And Development",
    "sellingGeneralAndAdministrative": "Selling General And Administration",
    "interestExpense": "Interest Expense",
    "interestIncome": "Interest Income",
    "incomeBeforeTax": "Pretax Income",
    "incomeTaxExpense": "Tax Provision",
    "netIncome": "Net Income",
    "netIncomeFromContinuingOperations": "Net Income Continuous Operations",
    "ebit": "EBIT",
    "ebitda": "EBITDA",
    "depreciationAndAmortization": "Depreciation And Amortization",
    "comprehensiveIncomeNetOfTax": "Comprehensive Income",
}

AV_BALANCE_FIELD_MAP = {
    "totalAssets": "Total Assets",
    "totalCurrentAssets": "Current Assets",
    "cashAndCashEquivalentsAtCarryingValue": "Cash And Cash Equivalents",
    "cashAndShortTermInvestments": "Cash Cash Equivalents And Short Term Investments",
    "shortTermInvestments": "Other Short Term Investments",
    "currentNetReceivables": "Accounts Receivable",
    "inventory": "Inventory",
    "otherCurrentAssets": "Other Current Assets",
    "totalNonCurrentAssets": "Total Non Current Assets",
    "propertyPlantEquipment": "Net PPE",
    "accumulatedDepreciationAmortizationPPE": "Accumulated Depreciation",
    "goodwill": "Goodwill",
    "intangibleAssets": "Other Intangible Assets",
    "intangibleAssetsExcludingGoodwill": "Other Intangible Assets",
    "longTermInvestments": "Investments And Advances",
    "otherNonCurrentAssets": "Other Non Current Assets",
    "totalLiabilities": "Total Liabilities Net Minority Interest",
    "totalCurrentLiabilities": "Current Liabilities",
    "currentAccountsPayable": "Accounts Payable",
    "currentDebt": "Current Debt",
    "shortTermDebt": "Current Debt",
    "otherCurrentLiabilities": "Other Current Liabilities",
    "totalNonCurrentLiabilities": "Total Non Current Liabilities Net Minority Interest",
    "longTermDebt": "Long Term Debt",
    "longTermDebtNoncurrent": "Long Term Debt",
    "otherNonCurrentLiabilities": "Other Non Current Liabilities",
    "totalShareholderEquity": "Stockholders Equity",
    "retainedEarnings": "Retained Earnings",
    "commonStock": "Common Stock",
    "commonStockSharesOutstanding": "Ordinary Shares Number",
    "treasuryStock": "Treasury Stock",
}

AV_CASHFLOW_FIELD_MAP = {
    "operatingCashflow": "Operating Cash Flow",
    "paymentsForOperatingActivities": "Payments For Operating Activities",
    "changeInOperatingLiabilities": "Change In Operating Liabilities",
    "changeInOperatingAssets": "Change In Operating Assets",
    "depreciationDepletionAndAmortization": "Depreciation And Amortization",
    "capitalExpenditures": "Capital Expenditure",
    "changeInReceivables": "Change In Receivables",
    "changeInInventory": "Change In Inventory",
    "profitLoss": "Net Income",
    "cashflowFromInvestment": "Investing Cash Flow",
    "cashflowFromFinancing": "Financing Cash Flow",
    "dividendPayout": "Cash Dividends Paid",
    "dividendPayoutCommonStock": "Common Stock Dividend Paid",
    "dividendPayoutPreferredStock": "Preferred Stock Dividend Paid",
    "paymentsForRepurchaseOfCommonStock": "Repurchase Of Capital Stock",
    "paymentsForRepurchaseOfEquity": "Repurchase Of Capital Stock",
    "paymentsForRepurchaseOfPreferredStock": "Repurchase Of Preferred Stock",
    "proceedsFromIssuanceOfCommonStock": "Common Stock Issuance",
    "proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet": "Long Term Debt Issuance",
    "proceedsFromIssuanceOfPreferredStock": "Preferred Stock Issuance",
    "proceedsFromRepurchaseOfEquity": "Proceeds From Repurchase Of Equity",
    "proceedsFromSaleOfTreasuryStock": "Proceeds From Sale Of Treasury Stock",
    "changeInCashAndCashEquivalents": "Changes In Cash",
    "netIncome": "Net Income From Continuing Operations",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CompanyInfo:
    """Container for company metadata and market information."""
    ticker: str
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    currency: str = "USD"
    shares_outstanding: Optional[float] = None
    market_cap: Optional[float] = None
    current_price: Optional[float] = None
    beta: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    dividend_yield: Optional[float] = None
    fetch_timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self) -> None:
        self.ticker = self.ticker.upper().strip()
        if not self.name or self.name.strip() == "":
            self.name = self.ticker

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "sector": self.sector,
            "industry": self.industry,
            "currency": self.currency,
            "shares_outstanding": self.shares_outstanding,
            "market_cap": self.market_cap,
            "current_price": self.current_price,
            "beta": self.beta,
            "fifty_two_week_high": self.fifty_two_week_high,
            "fifty_two_week_low": self.fifty_two_week_low,
            "dividend_yield": self.dividend_yield,
            "fetch_timestamp": self.fetch_timestamp.isoformat()
        }


@dataclass
class FinancialStatements:
    """Container for financial statement data."""
    income_statement: pd.DataFrame
    balance_sheet: pd.DataFrame
    cash_flow: pd.DataFrame
    income_statement_quarterly: Optional[pd.DataFrame] = None
    balance_sheet_quarterly: Optional[pd.DataFrame] = None
    cash_flow_quarterly: Optional[pd.DataFrame] = None
    
    def get_years_available(self) -> int:
        if self.income_statement is None or self.income_statement.empty:
            return 0
        return len(self.income_statement.columns)
    
    def get_fiscal_periods(self) -> List[Any]:
        if self.income_statement is None or self.income_statement.empty:
            return []
        return list(self.income_statement.columns)
    
    def is_empty(self) -> bool:
        return (
            (self.income_statement is None or self.income_statement.empty) and
            (self.balance_sheet is None or self.balance_sheet.empty) and
            (self.cash_flow is None or self.cash_flow.empty)
        )


@dataclass
class DataValidationResult:
    """Result of data validation checks."""
    status: ValidationStatus
    years_available: int
    missing_statements: List[str] = field(default_factory=list)
    missing_fields: Dict[str, List[str]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        return self.status != ValidationStatus.FAILED
    
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0
    
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "years_available": self.years_available,
            "missing_statements": self.missing_statements,
            "missing_fields": self.missing_fields,
            "warnings": self.warnings,
            "errors": self.errors
        }


@dataclass
class CollectedData:
    """Complete container for all collected financial data."""
    company_info: CompanyInfo
    statements: FinancialStatements
    validation: DataValidationResult
    collection_timestamp: datetime = field(default_factory=datetime.now)
    data_source: str = DEFAULT_DATA_SOURCE
    
    def is_valid(self) -> bool:
        return self.validation.is_valid()
    
    def get_ticker(self) -> str:
        return self.company_info.ticker
    
    def get_company_name(self) -> str:
        return self.company_info.name
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "company_info": self.company_info.to_dict(),
            "validation": self.validation.to_dict(),
            "collection_timestamp": self.collection_timestamp.isoformat(),
            "data_source": self.data_source,
            "years_available": self.statements.get_years_available()
        }


# =============================================================================
# ALPHA VANTAGE FETCHER
# =============================================================================

class AlphaVantageFetcher:
    """
    Fetches financial data from Alpha Vantage API.
    
    Alpha Vantage provides:
    - 5+ years of annual financial statements
    - Free API with rate limits (5 calls/minute)
    - High quality, standardized data
    
    Get free API key: https://www.alphavantage.co/support/#api-key
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Alpha Vantage API key required. "
                "Set ALPHA_VANTAGE_API_KEY environment variable or pass api_key parameter. "
                "Get free key at: https://www.alphavantage.co/support/#api-key"
            )
        self._last_call_time = 0
        logger.info("AlphaVantageFetcher initialized")
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_call_time
        if elapsed < ALPHA_VANTAGE_RATE_LIMIT_DELAY:
            sleep_time = ALPHA_VANTAGE_RATE_LIMIT_DELAY - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self._last_call_time = time.time()
    
    def _make_request(self, function: str, symbol: str) -> Optional[Dict]:
        """Make API request to Alpha Vantage."""
        self._rate_limit()
        
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        try:
            logger.info(f"Fetching {function} for {symbol} from Alpha Vantage")
            response = requests.get(
                ALPHA_VANTAGE_BASE_URL,
                params=params,
                timeout=ALPHA_VANTAGE_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return None
            if "Note" in data:
                logger.warning(f"Alpha Vantage note (rate limit?): {data['Note']}")
                return None
            if "Information" in data:
                logger.warning(f"Alpha Vantage info: {data['Information']}")
                return None
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Alpha Vantage request failed: {e}")
            return None
    
    def get_company_overview(self, symbol: str) -> Optional[Dict]:
        """Fetch company overview/metadata."""
        return self._make_request("OVERVIEW", symbol)
    
    def get_income_statement(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch income statement data.
        
        Returns:
            Tuple of (annual_df, quarterly_df)
        """
        data = self._make_request("INCOME_STATEMENT", symbol)
        if not data:
            return pd.DataFrame(), pd.DataFrame()
        
        annual_df = self._parse_reports(data.get("annualReports", []), AV_INCOME_FIELD_MAP)
        quarterly_df = self._parse_reports(data.get("quarterlyReports", []), AV_INCOME_FIELD_MAP)
        
        return annual_df, quarterly_df
    
    def get_balance_sheet(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch balance sheet data.
        
        Returns:
            Tuple of (annual_df, quarterly_df)
        """
        data = self._make_request("BALANCE_SHEET", symbol)
        if not data:
            return pd.DataFrame(), pd.DataFrame()
        
        annual_df = self._parse_reports(data.get("annualReports", []), AV_BALANCE_FIELD_MAP)
        quarterly_df = self._parse_reports(data.get("quarterlyReports", []), AV_BALANCE_FIELD_MAP)
        
        return annual_df, quarterly_df
    
    def get_cash_flow(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch cash flow statement data.
        
        Returns:
            Tuple of (annual_df, quarterly_df)
        """
        data = self._make_request("CASH_FLOW", symbol)
        if not data:
            return pd.DataFrame(), pd.DataFrame()
        
        annual_df = self._parse_reports(data.get("annualReports", []), AV_CASHFLOW_FIELD_MAP)
        quarterly_df = self._parse_reports(data.get("quarterlyReports", []), AV_CASHFLOW_FIELD_MAP)
        
        return annual_df, quarterly_df
    
    def _parse_reports(self, reports: List[Dict], field_map: Dict[str, str]) -> pd.DataFrame:
        """
        Parse Alpha Vantage reports into a DataFrame.
        
        Args:
            reports: List of report dictionaries from API
            field_map: Mapping of AV field names to standardized names
            
        Returns:
            DataFrame with dates as columns and metrics as rows
        """
        if not reports:
            return pd.DataFrame()
        
        data_dict = {}
        
        for report in reports:
            # Get fiscal date
            fiscal_date = report.get("fiscalDateEnding")
            if not fiscal_date:
                continue
            
            try:
                date_col = pd.Timestamp(fiscal_date)
            except:
                continue
            
            # Extract values
            col_data = {}
            for av_field, std_field in field_map.items():
                value = report.get(av_field)
                if value is not None and value != "None":
                    try:
                        col_data[std_field] = float(value)
                    except (ValueError, TypeError):
                        pass
            
            # Also include unmapped fields with original names
            for key, value in report.items():
                if key not in ["fiscalDateEnding", "reportedCurrency"] and key not in field_map:
                    if value is not None and value != "None":
                        try:
                            col_data[key] = float(value)
                        except (ValueError, TypeError):
                            pass
            
            if col_data:
                data_dict[date_col] = col_data
        
        if not data_dict:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(data_dict)
        
        # Sort columns by date (most recent first)
        df = df.reindex(sorted(df.columns, reverse=True), axis=1)
        
        return df


# =============================================================================
# DATA COLLECTOR CLASS
# =============================================================================

class DataCollector:
    """
    Financial data collection pipeline for fundamental analysis.
    
    Uses Alpha Vantage as primary data source for complete 5-year coverage.
    Falls back to Yahoo Finance for company metadata.
    """
    
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize the DataCollector.
        
        Args:
            api_key: Alpha Vantage API key (or set ALPHA_VANTAGE_API_KEY env var)
        """
        self._api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self._av_fetcher: Optional[AlphaVantageFetcher] = None
        self._ticker_cache: Dict[str, Any] = {}
        
        # Initialize Alpha Vantage fetcher
        if self._api_key:
            try:
                self._av_fetcher = AlphaVantageFetcher(self._api_key)
                logger.info("DataCollector initialized with Alpha Vantage")
            except Exception as e:
                logger.warning(f"Alpha Vantage init failed: {e}")
        else:
            logger.warning(
                "No Alpha Vantage API key found. "
                "Set ALPHA_VANTAGE_API_KEY environment variable. "
                "Get free key at: https://www.alphavantage.co/support/#api-key"
            )
    
    # =========================================================================
    # MAIN PUBLIC METHOD
    # =========================================================================
    
    def collect(self, ticker: str) -> CollectedData:
        """
        Collect all financial data for a given ticker symbol.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
        
        Returns:
            CollectedData object containing all collected data
        """
        ticker = self._validate_ticker(ticker)
        logger.info(f"Starting data collection for ticker: {ticker}")
        
        # Fetch company information (from Yahoo Finance or Alpha Vantage)
        company_info = self._fetch_company_info(ticker)
        
        # Fetch financial statements from Alpha Vantage
        if self._av_fetcher:
            statements = self._fetch_from_alpha_vantage(ticker)
            data_source = "Alpha Vantage"
        else:
            # Fallback to Yahoo Finance if no API key
            statements = self._fetch_from_yahoo_finance(ticker)
            data_source = "Yahoo Finance"
        
        # Validate collected data
        validation = self._validate_data(statements)
        
        # Log summary
        self._log_collection_summary(ticker, company_info, validation, data_source)
        
        return CollectedData(
            company_info=company_info,
            statements=statements,
            validation=validation,
            collection_timestamp=datetime.now(),
            data_source=data_source
        )
    
    # =========================================================================
    # ALPHA VANTAGE DATA FETCHING
    # =========================================================================
    
    def _fetch_from_alpha_vantage(self, ticker: str) -> FinancialStatements:
        """Fetch all financial statements from Alpha Vantage."""
        if not self._av_fetcher:
            return FinancialStatements(
                income_statement=pd.DataFrame(),
                balance_sheet=pd.DataFrame(),
                cash_flow=pd.DataFrame()
            )
        
        # Fetch income statement
        income_annual, income_quarterly = self._av_fetcher.get_income_statement(ticker)
        logger.info(f"Income Statement: {len(income_annual.columns)} annual periods")
        
        # Fetch balance sheet
        balance_annual, balance_quarterly = self._av_fetcher.get_balance_sheet(ticker)
        logger.info(f"Balance Sheet: {len(balance_annual.columns)} annual periods")
        
        # Fetch cash flow
        cashflow_annual, cashflow_quarterly = self._av_fetcher.get_cash_flow(ticker)
        logger.info(f"Cash Flow: {len(cashflow_annual.columns)} annual periods")
        
        # Limit to 5 years for consistency
        if len(income_annual.columns) > 5:
            income_annual = income_annual.iloc[:, :5]
        if len(balance_annual.columns) > 5:
            balance_annual = balance_annual.iloc[:, :5]
        if len(cashflow_annual.columns) > 5:
            cashflow_annual = cashflow_annual.iloc[:, :5]
        
        # Calculate derived fields
        income_annual = self._calculate_derived_fields(income_annual, balance_annual, cashflow_annual)
        
        return FinancialStatements(
            income_statement=income_annual,
            balance_sheet=balance_annual,
            cash_flow=cashflow_annual,
            income_statement_quarterly=income_quarterly,
            balance_sheet_quarterly=balance_quarterly,
            cash_flow_quarterly=cashflow_quarterly
        )
    
    def _calculate_derived_fields(
        self, 
        income: pd.DataFrame, 
        balance: pd.DataFrame, 
        cashflow: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate derived fields that may be missing."""
        if income.empty:
            return income
        
        # Calculate Gross Profit if missing
        if "Gross Profit" not in income.index:
            if "Total Revenue" in income.index and "Cost Of Revenue" in income.index:
                income.loc["Gross Profit"] = income.loc["Total Revenue"] - income.loc["Cost Of Revenue"]
        
        # Calculate Operating Income if missing
        if "Operating Income" not in income.index:
            if "Gross Profit" in income.index and "Operating Expense" in income.index:
                income.loc["Operating Income"] = income.loc["Gross Profit"] - income.loc["Operating Expense"]
        
        # Calculate EBIT if missing (use Operating Income as proxy)
        if "EBIT" not in income.index and "Operating Income" in income.index:
            income.loc["EBIT"] = income.loc["Operating Income"]
        
        # Calculate Free Cash Flow if missing
        # CRITICAL FIX: FCF = OCF - |CapEx|
        # Some data providers (like Alpha Vantage) report CapEx as POSITIVE
        # Others report it as negative. Using abs() handles both conventions.
        if not cashflow.empty:
            if "Free Cash Flow" not in cashflow.index:
                if "Operating Cash Flow" in cashflow.index and "Capital Expenditure" in cashflow.index:
                    capex = cashflow.loc["Capital Expenditure"]
                    # Always subtract absolute value of CapEx to ensure correct calculation
                    cashflow.loc["Free Cash Flow"] = (
                        cashflow.loc["Operating Cash Flow"] - capex.abs()
                    )
        
        return income
    
    # =========================================================================
    # YAHOO FINANCE FALLBACK
    # =========================================================================
    
    def _fetch_from_yahoo_finance(self, ticker: str) -> FinancialStatements:
        """Fallback to Yahoo Finance if Alpha Vantage unavailable."""
        if not YFINANCE_AVAILABLE:
            logger.error("Neither Alpha Vantage nor Yahoo Finance available")
            return FinancialStatements(
                income_statement=pd.DataFrame(),
                balance_sheet=pd.DataFrame(),
                cash_flow=pd.DataFrame()
            )
        
        yf_ticker = self._get_yf_ticker(ticker)
        
        income = self._safe_fetch_yf_statement(yf_ticker, "income_stmt")
        balance = self._safe_fetch_yf_statement(yf_ticker, "balance_sheet")
        cashflow = self._safe_fetch_yf_statement(yf_ticker, "cash_flow")
        
        return FinancialStatements(
            income_statement=income,
            balance_sheet=balance,
            cash_flow=cashflow
        )
    
    def _get_yf_ticker(self, ticker: str) -> Any:
        """Get or create Yahoo Finance ticker object."""
        if ticker not in self._ticker_cache:
            self._ticker_cache[ticker] = yf.Ticker(ticker)
        return self._ticker_cache[ticker]
    
    def _safe_fetch_yf_statement(self, yf_ticker: Any, attr: str) -> pd.DataFrame:
        """Safely fetch a Yahoo Finance statement."""
        try:
            stmt = getattr(yf_ticker, attr, None)
            if stmt is not None and isinstance(stmt, pd.DataFrame) and not stmt.empty:
                return stmt
        except Exception as e:
            logger.error(f"Yahoo Finance fetch failed for {attr}: {e}")
        return pd.DataFrame()
    
    # =========================================================================
    # COMPANY INFO
    # =========================================================================
    
    def _fetch_company_info(self, ticker: str) -> CompanyInfo:
        """Fetch company metadata from available sources."""
        info = {}
        
        # Try Alpha Vantage first
        if self._av_fetcher:
            try:
                av_overview = self._av_fetcher.get_company_overview(ticker)
                if av_overview:
                    info = {
                        "name": av_overview.get("Name", ticker),
                        "sector": av_overview.get("Sector"),
                        "industry": av_overview.get("Industry"),
                        "market_cap": self._safe_float(av_overview.get("MarketCapitalization")),
                        "beta": self._safe_float(av_overview.get("Beta")),
                        "dividend_yield": self._safe_float(av_overview.get("DividendYield")),
                        "shares_outstanding": self._safe_float(av_overview.get("SharesOutstanding")),
                        "52_week_high": self._safe_float(av_overview.get("52WeekHigh")),
                        "52_week_low": self._safe_float(av_overview.get("52WeekLow")),
                    }
            except Exception as e:
                logger.warning(f"Alpha Vantage overview failed: {e}")
        
        # Supplement with Yahoo Finance
        if YFINANCE_AVAILABLE:
            try:
                yf_ticker = self._get_yf_ticker(ticker)
                yf_info = yf_ticker.info or {}
                
                if not info.get("name"):
                    info["name"] = yf_info.get("longName") or yf_info.get("shortName") or ticker
                if not info.get("sector"):
                    info["sector"] = yf_info.get("sector")
                if not info.get("industry"):
                    info["industry"] = yf_info.get("industry")
                if not info.get("market_cap"):
                    info["market_cap"] = self._safe_float(yf_info.get("marketCap"))
                if not info.get("beta"):
                    info["beta"] = self._safe_float(yf_info.get("beta"))
                if not info.get("shares_outstanding"):
                    info["shares_outstanding"] = self._safe_float(yf_info.get("sharesOutstanding"))
                
                # Current price from Yahoo
                info["current_price"] = self._safe_float(
                    yf_info.get("currentPrice") or 
                    yf_info.get("regularMarketPrice") or 
                    yf_info.get("previousClose")
                )
                info["52_week_high"] = info.get("52_week_high") or self._safe_float(yf_info.get("fiftyTwoWeekHigh"))
                info["52_week_low"] = info.get("52_week_low") or self._safe_float(yf_info.get("fiftyTwoWeekLow"))
                info["dividend_yield"] = info.get("dividend_yield") or self._safe_float(yf_info.get("dividendYield"))
                
            except Exception as e:
                logger.warning(f"Yahoo Finance info failed: {e}")
        
        return CompanyInfo(
            ticker=ticker,
            name=info.get("name", ticker),
            sector=info.get("sector"),
            industry=info.get("industry"),
            currency="USD",
            shares_outstanding=info.get("shares_outstanding"),
            market_cap=info.get("market_cap"),
            current_price=info.get("current_price"),
            beta=info.get("beta"),
            fifty_two_week_high=info.get("52_week_high"),
            fifty_two_week_low=info.get("52_week_low"),
            dividend_yield=info.get("dividend_yield"),
            fetch_timestamp=datetime.now()
        )
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    def _validate_ticker(self, ticker: str) -> str:
        """Validate and normalize ticker symbol."""
        if ticker is None:
            raise ValueError("Ticker symbol cannot be None")
        if not isinstance(ticker, str):
            raise ValueError(f"Ticker must be string, got {type(ticker).__name__}")
        
        ticker = ticker.strip().upper()
        if not ticker:
            raise ValueError("Ticker symbol cannot be empty")
        if len(ticker) > TICKER_MAX_LENGTH:
            raise ValueError(f"Ticker '{ticker}' too long (max {TICKER_MAX_LENGTH})")
        
        return ticker
    
    def _validate_data(self, statements: FinancialStatements) -> DataValidationResult:
        """Validate completeness and quality of collected data."""
        errors: List[str] = []
        warnings: List[str] = []
        missing_statements: List[str] = []
        missing_fields: Dict[str, List[str]] = {}
        
        checks = [
            (statements.income_statement, "Income Statement", "income"),
            (statements.balance_sheet, "Balance Sheet", "balance"),
            (statements.cash_flow, "Cash Flow Statement", "cashflow")
        ]
        
        years_available = 0
        
        for stmt, name, stmt_type in checks:
            if stmt is None or stmt.empty:
                missing_statements.append(name)
                errors.append(f"Missing required statement: {name}")
            else:
                years_available = max(years_available, len(stmt.columns))
                missing = self._check_required_fields(stmt, stmt_type)
                if missing:
                    missing_fields[name] = missing
                    for f in missing:
                        warnings.append(f"{name}: Missing field '{f}'")
        
        # Check year coverage
        complete_years = self._count_complete_years(statements)
        
        if complete_years < VALIDATION.minimum_years_required:
            if complete_years >= 3:
                warnings.append(
                    f"Limited data: {complete_years} complete years, "
                    f"{VALIDATION.minimum_years_required} preferred"
                )
            else:
                errors.append(
                    f"Insufficient data: {complete_years} years, minimum 3 required"
                )
        
        if errors:
            status = ValidationStatus.FAILED
        elif warnings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.PASSED
        
        return DataValidationResult(
            status=status,
            years_available=complete_years,
            missing_statements=missing_statements,
            missing_fields=missing_fields,
            warnings=warnings,
            errors=errors
        )
    
    def _count_complete_years(self, statements: FinancialStatements) -> int:
        """Count years with complete key financial data."""
        if statements.is_empty():
            return 0
        
        income = statements.income_statement
        if income is None or income.empty:
            return 0
        
        revenue_fields = ['Total Revenue', 'totalRevenue', 'Revenue']
        net_income_fields = ['Net Income', 'netIncome', 'Net Income Continuous Operations']
        
        complete = 0
        for col in income.columns:
            has_revenue = any(f in income.index and pd.notna(income.loc[f, col]) for f in revenue_fields)
            has_net_income = any(f in income.index and pd.notna(income.loc[f, col]) for f in net_income_fields)
            
            if has_revenue and has_net_income:
                complete += 1
        
        return complete
    
    def _check_required_fields(self, stmt: pd.DataFrame, stmt_type: str) -> List[str]:
        """Check for missing required fields."""
        required = get_all_required_fields().get(stmt_type, [])
        missing = []
        
        for field in required:
            alternatives = get_field_alternatives(stmt_type, field)
            if not any(alt in stmt.index for alt in alternatives):
                missing.append(field)
        
        return missing
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float."""
        if value is None:
            return None
        try:
            result = float(value)
            return None if math.isnan(result) or math.isinf(result) else result
        except (ValueError, TypeError):
            return None
    
    def _log_collection_summary(
        self, ticker: str, company_info: CompanyInfo, 
        validation: DataValidationResult, data_source: str
    ) -> None:
        """Log collection summary."""
        logger.info(
            f"Data collection complete for {company_info.name} ({ticker}): "
            f"Status={validation.status.value.upper()}, "
            f"Years={validation.years_available}, Source={data_source}"
        )
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_field_value(
        self, statement: pd.DataFrame, field_name: str, 
        statement_type: str, period_index: int = 0, 
        default: Optional[float] = None
    ) -> Optional[float]:
        """Extract field value from statement."""
        if statement is None or statement.empty:
            return default
        
        alternatives = get_field_alternatives(statement_type, field_name)
        for alt in alternatives:
            if alt in statement.index:
                try:
                    value = statement.loc[alt].iloc[period_index]
                    if pd.notna(value):
                        return float(value)
                except:
                    continue
        return default
    
    def get_field_series(
        self, statement: pd.DataFrame, field_name: str, statement_type: str
    ) -> Optional[pd.Series]:
        """Extract field time series."""
        if statement is None or statement.empty:
            return None
        
        alternatives = get_field_alternatives(statement_type, field_name)
        for alt in alternatives:
            if alt in statement.index:
                return statement.loc[alt].copy()
        return None
    
    def get_available_periods(self, statements: FinancialStatements) -> List[Any]:
        """Get available fiscal periods."""
        return statements.get_fiscal_periods()
    
    def get_period_labels(self, statements: FinancialStatements) -> List[str]:
        """Get human-readable period labels."""
        periods = self.get_available_periods(statements)
        return [p.strftime('%Y') if hasattr(p, 'strftime') else str(p) for p in periods]
    
    def clear_cache(self) -> None:
        """Clear ticker cache."""
        self._ticker_cache.clear()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def collect_financial_data(ticker: str, api_key: Optional[str] = None) -> CollectedData:
    """Convenience function to collect financial data."""
    collector = DataCollector(api_key=api_key)
    return collector.collect(ticker)


def validate_ticker_data(ticker: str, api_key: Optional[str] = None) -> Tuple[bool, List[str]]:
    """Quick validation check for ticker data availability."""
    try:
        data = collect_financial_data(ticker, api_key)
        is_valid = data.validation.status != ValidationStatus.FAILED
        issues = data.validation.errors + data.validation.warnings
        return is_valid, issues
    except Exception as e:
        return False, [f"Collection failed: {str(e)}"]


# =============================================================================
# MODULE SELF-TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    test_ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    print()
    print("=" * 70)
    print(f"DATA COLLECTOR TEST (Alpha Vantage) - {test_ticker}")
    print("=" * 70)
    print()
    
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        print("WARNING: ALPHA_VANTAGE_API_KEY not set!")
        print("Get free key at: https://www.alphavantage.co/support/#api-key")
        print()
    
    try:
        collector = DataCollector()
        print(f"Collecting data for {test_ticker}...")
        print()
        
        data = collector.collect(test_ticker)
        
        print("COMPANY INFORMATION")
        print("-" * 70)
        print(f"  Name:           {data.company_info.name}")
        print(f"  Ticker:         {data.company_info.ticker}")
        print(f"  Sector:         {data.company_info.sector or 'N/A'}")
        print(f"  Industry:       {data.company_info.industry or 'N/A'}")
        print(f"  Data Source:    {data.data_source}")
        
        if data.company_info.current_price:
            print(f"  Current Price:  ${data.company_info.current_price:,.2f}")
        if data.company_info.market_cap:
            print(f"  Market Cap:     ${data.company_info.market_cap/1e9:,.2f}B")
        
        print()
        print("DATA AVAILABILITY")
        print("-" * 70)
        
        for name, stmt in [
            ("Income Statement", data.statements.income_statement),
            ("Balance Sheet", data.statements.balance_sheet),
            ("Cash Flow", data.statements.cash_flow)
        ]:
            if stmt is not None and not stmt.empty:
                years = [str(col)[:4] if hasattr(col, 'year') else str(col)[:4] for col in stmt.columns]
                print(f"  {name:20s}: {len(stmt.columns)} years ({', '.join(years)})")
            else:
                print(f"  {name:20s}: NOT AVAILABLE")
        
        print()
        print("VALIDATION")
        print("-" * 70)
        print(f"  Status:         {data.validation.status.value.upper()}")
        print(f"  Complete Years: {data.validation.years_available}")
        
        if data.validation.errors:
            print("  Errors:")
            for e in data.validation.errors:
                print(f"    - {e}")
        
        if data.validation.warnings:
            print("  Warnings:")
            for w in data.validation.warnings[:5]:
                print(f"    - {w}")
        
        print()
        print("=" * 70)
        print(f"Collection complete. Source: {data.data_source}")
        print("=" * 70)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)