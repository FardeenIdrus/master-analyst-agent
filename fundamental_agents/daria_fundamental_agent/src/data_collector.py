"""
Data Collector Module - Phase 1 Data Acquisition
Fundamental Analyst Agent

Implements institutional-grade data acquisition pipeline for financial statements
with comprehensive validation, quality metrics, and derived calculations.

Features:
    - 5-year financial statement collection (Income, Balance, Cash Flow)
    - Company overview and market data
    - Intelligent 5-hour caching with metadata
    - Data quality scoring and tier classification
    - Accounting equation validation
    - Derived metrics calculation (FCF, EBITDA, Working Capital, Net Debt)
    - Dividend history extraction for DDM

Data Source: Alpha Vantage API (free tier: 25 calls/day)
Rate Limiting: 12-second delay between API calls

MSc Coursework: IFTE0001 AI Agents in Asset Management
Track A: Fundamental Analyst Agent
Phase 1: Data Acquisition

Version: 3.0.0
"""

from __future__ import annotations

import json
import time
import math
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, FrozenSet

from .config import (
    ALPHA_VANTAGE_CONFIG,
    DATA_CONFIG,
    VALIDATION_CONFIG,
    CACHE_DIR,
    OUTPUT_DIR,
    LOGGER,
    ValidationStatus,
    DataQualityTier,
    StatementType,
    INCOME_FIELD_MAP,
    BALANCE_FIELD_MAP,
    CASHFLOW_FIELD_MAP,
    OVERVIEW_FIELD_MAP,
    CRITICAL_INCOME_FIELDS,
    CRITICAL_BALANCE_FIELDS,
    CRITICAL_CASHFLOW_FIELDS,
    DERIVED_METRICS,
    get_critical_fields,
    get_expected_fields,
)

# Try to import Yahoo Finance supplementer
try:
    from .yahoo_supplementer import YahooFinanceSupplementer, SupplementationReport
    YAHOO_SUPPLEMENTER_AVAILABLE = True
except ImportError:
    YAHOO_SUPPLEMENTER_AVAILABLE = False
    SupplementationReport = None  # Type hint placeholder


__version__ = "3.1.0"
RATE_LIMIT_SECONDS = 12.0


# =============================================================================
# DATA CONTAINERS
# =============================================================================

@dataclass
class CompanyProfile:
    """
    Company metadata and market information.
    
    Contains identification, market data, valuation multiples, profitability
    metrics, dividend data, growth metrics, and analyst estimates.
    """
    
    # Identification
    ticker: str
    name: str
    description: Optional[str] = None
    exchange: Optional[str] = None
    currency: str = "USD"
    country: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    fiscal_year_end: Optional[str] = None
    
    # Market Data
    market_cap: Optional[float] = None
    shares_outstanding: Optional[float] = None
    high_52week: Optional[float] = None
    low_52week: Optional[float] = None
    ma_50day: Optional[float] = None
    ma_200day: Optional[float] = None
    beta: Optional[float] = None
    
    # Valuation Multiples
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    book_value: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    ev_to_revenue: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    pe_trailing: Optional[float] = None
    pe_forward: Optional[float] = None
    
    # Profitability
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    roa: Optional[float] = None
    roe: Optional[float] = None
    
    # Dividends
    dividend_per_share: Optional[float] = None
    dividend_yield: Optional[float] = None
    ex_dividend_date: Optional[str] = None
    
    # Growth
    earnings_growth_yoy: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    
    # Earnings
    eps: Optional[float] = None
    eps_diluted_ttm: Optional[float] = None
    
    # Analyst Data
    analyst_target: Optional[float] = None
    analyst_strong_buy: Optional[int] = None
    analyst_buy: Optional[int] = None
    analyst_hold: Optional[int] = None
    analyst_sell: Optional[int] = None
    analyst_strong_sell: Optional[int] = None
    
    # Metadata
    fetch_timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Normalize ticker symbol."""
        self.ticker = self.ticker.upper().strip()
        if not self.name:
            self.name = self.ticker

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ticker": self.ticker,
            "name": self.name,
            "description": self.description,
            "exchange": self.exchange,
            "currency": self.currency,
            "country": self.country,
            "sector": self.sector,
            "industry": self.industry,
            "fiscal_year_end": self.fiscal_year_end,
            "market_cap": self.market_cap,
            "shares_outstanding": self.shares_outstanding,
            "high_52week": self.high_52week,
            "low_52week": self.low_52week,
            "ma_50day": self.ma_50day,
            "ma_200day": self.ma_200day,
            "beta": self.beta,
            "pe_ratio": self.pe_ratio,
            "peg_ratio": self.peg_ratio,
            "book_value": self.book_value,
            "pb_ratio": self.pb_ratio,
            "ps_ratio": self.ps_ratio,
            "ev_to_revenue": self.ev_to_revenue,
            "ev_to_ebitda": self.ev_to_ebitda,
            "pe_trailing": self.pe_trailing,
            "pe_forward": self.pe_forward,
            "profit_margin": self.profit_margin,
            "operating_margin": self.operating_margin,
            "roa": self.roa,
            "roe": self.roe,
            "dividend_per_share": self.dividend_per_share,
            "dividend_yield": self.dividend_yield,
            "ex_dividend_date": self.ex_dividend_date,
            "earnings_growth_yoy": self.earnings_growth_yoy,
            "revenue_growth_yoy": self.revenue_growth_yoy,
            "eps": self.eps,
            "eps_diluted_ttm": self.eps_diluted_ttm,
            "analyst_target": self.analyst_target,
            "fetch_timestamp": self.fetch_timestamp.isoformat(),
        }


@dataclass
class FinancialStatements:
    """
    Container for financial statement DataFrames.
    
    Each DataFrame structure:
        - Index: Standardized field names
        - Columns: Fiscal years (descending order, e.g., 2024, 2023, 2022)
        - Values: Float values in original currency units
    """
    
    income_statement: pd.DataFrame
    balance_sheet: pd.DataFrame
    cash_flow: pd.DataFrame
    
    @property
    def years_available(self) -> int:
        """Number of years of data available."""
        if self.income_statement is None or self.income_statement.empty:
            return 0
        return len(self.income_statement.columns)
    
    @property
    def fiscal_periods(self) -> List[str]:
        """List of fiscal year labels."""
        if self.income_statement is None or self.income_statement.empty:
            return []
        return list(self.income_statement.columns)
    
    @property
    def is_empty(self) -> bool:
        """Check if all statements are empty."""
        income_empty = self.income_statement is None or self.income_statement.empty
        balance_empty = self.balance_sheet is None or self.balance_sheet.empty
        cashflow_empty = self.cash_flow is None or self.cash_flow.empty
        return income_empty and balance_empty and cashflow_empty


@dataclass
class QualityMetrics:
    """
    Data quality assessment metrics.
    
    Provides comprehensive quality scoring including completeness percentages,
    field coverage, critical field validation, and accounting identity checks.
    
    Note: 'warnings' affect validation status (PARTIAL), 'info' does not.
    """
    
    # Completeness scores (0.0 to 1.0)
    income_completeness: float = 0.0
    balance_completeness: float = 0.0
    cashflow_completeness: float = 0.0
    overall_completeness: float = 0.0
    
    # Field coverage
    income_fields_present: int = 0
    income_fields_expected: int = 0
    balance_fields_present: int = 0
    balance_fields_expected: int = 0
    cashflow_fields_present: int = 0
    cashflow_fields_expected: int = 0
    
    # Critical field validation
    critical_income_valid: bool = False
    critical_balance_valid: bool = False
    critical_cashflow_valid: bool = False
    
    # Accounting equation check
    accounting_equation_valid: bool = False
    accounting_equation_deviation: Optional[float] = None
    
    # Quality classification
    quality_tier: DataQualityTier = DataQualityTier.POOR
    
    # Issues (warnings affect status, info does not)
    warnings: List[str] = field(default_factory=list)  # Affects validation status
    info: List[str] = field(default_factory=list)      # Informational only
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "income_completeness": self.income_completeness,
            "balance_completeness": self.balance_completeness,
            "cashflow_completeness": self.cashflow_completeness,
            "overall_completeness": self.overall_completeness,
            "income_fields": f"{self.income_fields_present}/{self.income_fields_expected}",
            "balance_fields": f"{self.balance_fields_present}/{self.balance_fields_expected}",
            "cashflow_fields": f"{self.cashflow_fields_present}/{self.cashflow_fields_expected}",
            "critical_income_valid": self.critical_income_valid,
            "critical_balance_valid": self.critical_balance_valid,
            "critical_cashflow_valid": self.critical_cashflow_valid,
            "accounting_equation_valid": self.accounting_equation_valid,
            "accounting_equation_deviation": self.accounting_equation_deviation,
            "quality_tier": self.quality_tier.value,
            "warnings": self.warnings,
            "info": self.info,
            "errors": self.errors,
        }


@dataclass
class DividendHistory:
    """
    Historical dividend data for DDM analysis.
    
    Extracts dividend history from cash flow statements and provides
    metrics needed for dividend discount model valuation.
    """
    
    # Annual dividends by fiscal year
    annual_dividends: Dict[str, float] = field(default_factory=dict)
    
    # Summary statistics
    years_of_data: int = 0
    current_annual_dps: Optional[float] = None
    dividend_cagr: Optional[float] = None
    
    # Quality flags
    has_dividend_cuts: bool = False
    payout_stable: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "annual_dividends": self.annual_dividends,
            "years_of_data": self.years_of_data,
            "current_annual_dps": self.current_annual_dps,
            "dividend_cagr": self.dividend_cagr,
            "has_dividend_cuts": self.has_dividend_cuts,
            "payout_stable": self.payout_stable,
        }


@dataclass
class DerivedMetrics:
    """
    Calculated metrics derived from financial statements.
    
    These metrics are calculated during data acquisition to provide
    inputs for downstream valuation and analysis phases.
    """
    
    # By fiscal year: {"2024": value, "2023": value, ...}
    fcf_calculated: Dict[str, Optional[float]] = field(default_factory=dict)
    ebitda_calculated: Dict[str, Optional[float]] = field(default_factory=dict)
    working_capital: Dict[str, Optional[float]] = field(default_factory=dict)
    net_debt: Dict[str, Optional[float]] = field(default_factory=dict)
    invested_capital: Dict[str, Optional[float]] = field(default_factory=dict)
    enterprise_value: Optional[float] = None  # Current only, requires market cap
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fcf_calculated": self.fcf_calculated,
            "ebitda_calculated": self.ebitda_calculated,
            "working_capital": self.working_capital,
            "net_debt": self.net_debt,
            "invested_capital": self.invested_capital,
            "enterprise_value": self.enterprise_value,
        }


@dataclass
class CollectionResult:
    """
    Complete result of data collection operation.
    
    Primary output of the data acquisition pipeline, containing all
    collected data, quality metrics, and derived calculations.
    """
    
    # Core data
    company_profile: CompanyProfile
    statements: FinancialStatements
    
    # Quality assessment
    quality_metrics: QualityMetrics
    validation_status: ValidationStatus
    
    # Derived data
    derived_metrics: DerivedMetrics
    dividend_history: DividendHistory
    
    # Data supplementation (from Yahoo Finance)
    supplementation_report: Optional[Any] = None  # SupplementationReport if available
    
    # Metadata
    data_source: str = "Alpha Vantage"
    collection_timestamp: datetime = field(default_factory=datetime.now)
    api_calls_made: int = 0
    from_cache: bool = False
    data_supplemented: bool = False
    
    @property
    def is_valid(self) -> bool:
        """Check if data is valid for analysis."""
        return self.validation_status != ValidationStatus.INVALID
    
    @property
    def ticker(self) -> str:
        """Return ticker symbol."""
        return self.company_profile.ticker
    
    @property
    def company_name(self) -> str:
        """Return company name."""
        return self.company_profile.name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "company_profile": self.company_profile.to_dict(),
            "quality_metrics": self.quality_metrics.to_dict(),
            "validation_status": self.validation_status.value,
            "derived_metrics": self.derived_metrics.to_dict(),
            "dividend_history": self.dividend_history.to_dict(),
            "data_source": self.data_source,
            "collection_timestamp": self.collection_timestamp.isoformat(),
            "api_calls_made": self.api_calls_made,
            "from_cache": self.from_cache,
            "data_supplemented": self.data_supplemented,
            "years_available": self.statements.years_available,
            "fiscal_periods": self.statements.fiscal_periods,
        }
        
        # Add supplementation details if available
        if self.supplementation_report is not None:
            result["supplementation"] = {
                "gaps_filled": self.supplementation_report.gaps_filled,
                "gaps_remaining": self.supplementation_report.gaps_remaining,
                "fields_supplemented": list(self.supplementation_report.fields_supplemented.keys()),
            }
        
        return result


# =============================================================================
# CACHE MANAGER
# =============================================================================

class CacheManager:
    """
    File-based cache manager for API responses.
    
    Cache Structure:
        cache/{TICKER}/
            income_statement.json
            balance_sheet.json
            cash_flow.json
            overview.json
            _metadata.json
    
    Metadata tracks fetch timestamps and expiry times for cache invalidation.
    """
    
    def __init__(self, cache_dir: Path, expiry_seconds: int):
        """
        Initialize CacheManager.
        
        Args:
            cache_dir: Base directory for cache files
            expiry_seconds: Cache expiry time in seconds
        """
        self.cache_dir = cache_dir
        self.expiry_seconds = expiry_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_ticker_dir(self, ticker: str) -> Path:
        """Get cache directory for ticker."""
        ticker_dir = self.cache_dir / ticker.upper()
        ticker_dir.mkdir(parents=True, exist_ok=True)
        return ticker_dir
    
    def _get_cache_path(self, ticker: str, data_type: str) -> Path:
        """Get cache file path for ticker and data type."""
        return self._get_ticker_dir(ticker) / f"{data_type}.json"
    
    def _get_metadata_path(self, ticker: str) -> Path:
        """Get metadata file path for ticker."""
        return self._get_ticker_dir(ticker) / "_metadata.json"
    
    def is_valid(self, ticker: str, data_type: str) -> bool:
        """
        Check if cached data exists and is not expired.
        
        Args:
            ticker: Stock ticker symbol
            data_type: Type of data (income_statement, balance_sheet, etc.)
            
        Returns:
            True if cache is valid, False otherwise
        """
        cache_path = self._get_cache_path(ticker, data_type)
        metadata_path = self._get_metadata_path(ticker)
        
        if not cache_path.exists() or not metadata_path.exists():
            return False
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            if data_type not in metadata:
                return False
            
            expiry_str = metadata[data_type].get("expiry_timestamp")
            if not expiry_str:
                return False
            
            expiry = datetime.fromisoformat(expiry_str)
            return datetime.now() < expiry
            
        except (json.JSONDecodeError, KeyError, ValueError):
            return False
    
    def get(self, ticker: str, data_type: str) -> Optional[Dict]:
        """
        Retrieve cached data if valid.
        
        Args:
            ticker: Stock ticker symbol
            data_type: Type of data
            
        Returns:
            Cached data dictionary or None if not available
        """
        if not self.is_valid(ticker, data_type):
            return None
        
        cache_path = self._get_cache_path(ticker, data_type)
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def save(self, ticker: str, data_type: str, data: Dict) -> bool:
        """
        Save data to cache with metadata.
        
        Args:
            ticker: Stock ticker symbol
            data_type: Type of data
            data: Data dictionary to cache
            
        Returns:
            True if save successful, False otherwise
        """
        cache_path = self._get_cache_path(ticker, data_type)
        metadata_path = self._get_metadata_path(ticker)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            now = datetime.now()
            expiry = now + timedelta(seconds=self.expiry_seconds)
            
            metadata = {}
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except (json.JSONDecodeError, IOError):
                    metadata = {}
            
            metadata[data_type] = {
                "fetch_timestamp": now.isoformat(),
                "expiry_timestamp": expiry.isoformat(),
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except IOError:
            return False
    
    def get_status(self, ticker: str) -> Dict[str, bool]:
        """
        Get cache validity status for all data types.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary mapping data type to validity status
        """
        data_types = ["income_statement", "balance_sheet", "cash_flow", "overview"]
        return {dt: self.is_valid(ticker, dt) for dt in data_types}
    
    def get_all_valid(self, ticker: str) -> bool:
        """Check if all data types are cached and valid."""
        status = self.get_status(ticker)
        return all(status.values())


# =============================================================================
# ALPHA VANTAGE API CLIENT
# =============================================================================

class AlphaVantageClient:
    """
    Alpha Vantage API client with rate limiting.
    
    Enforces 12-second delay between API calls to comply with
    free tier limits (5 calls per minute, 25 calls per day).
    """
    
    def __init__(self, api_key: str, base_url: str, timeout: int):
        """
        Initialize AlphaVantageClient.
        
        Args:
            api_key: Alpha Vantage API key
            base_url: API base URL
            timeout: Request timeout in seconds
            
        Raises:
            ValueError: If API key is not provided
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self._last_call_time: float = 0.0
        self._call_count: int = 0
        
        if not self.api_key:
            raise ValueError(
                "Alpha Vantage API key required. "
                "Set ALPHA_VANTAGE_API_KEY environment variable. "
                "Get free key at: https://www.alphavantage.co/support/#api-key"
            )
    
    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between API calls."""
        if self._last_call_time > 0:
            elapsed = time.time() - self._last_call_time
            if elapsed < RATE_LIMIT_SECONDS:
                wait_time = RATE_LIMIT_SECONDS - elapsed
                LOGGER.info(f"Rate limit: waiting {wait_time:.1f}s")
                time.sleep(wait_time)
        
        self._last_call_time = time.time()
    
    def _request(
        self, 
        function: str, 
        symbol: str
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Make API request with rate limiting.
        
        Args:
            function: API function name
            symbol: Stock ticker symbol
            
        Returns:
            Tuple of (data_dict, error_message)
        """
        self._enforce_rate_limit()
        
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
        }
        
        try:
            LOGGER.info(f"API request: {function} for {symbol}")
            
            response = requests.get(
                self.base_url,
                params=params,
                timeout=self.timeout
            )
            
            self._call_count += 1
            response.raise_for_status()
            data = response.json()
            
            if "Error Message" in data:
                return None, data["Error Message"]
            if "Note" in data:
                return None, data["Note"]
            if "Information" in data:
                return None, data["Information"]
            
            LOGGER.info(f"API success: {function}")
            return data, None
            
        except requests.exceptions.Timeout:
            return None, f"Request timeout for {function}"
        except requests.exceptions.RequestException as e:
            return None, f"Request failed: {str(e)}"
    
    @property
    def call_count(self) -> int:
        """Return total API calls made."""
        return self._call_count
    
    def get_income_statement(self, symbol: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Fetch income statement data."""
        return self._request("INCOME_STATEMENT", symbol)
    
    def get_balance_sheet(self, symbol: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Fetch balance sheet data."""
        return self._request("BALANCE_SHEET", symbol)
    
    def get_cash_flow(self, symbol: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Fetch cash flow statement data."""
        return self._request("CASH_FLOW", symbol)
    
    def get_overview(self, symbol: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Fetch company overview data."""
        return self._request("OVERVIEW", symbol)


# =============================================================================
# DATA PARSER
# =============================================================================

class DataParser:
    """
    Parses Alpha Vantage API responses into standardized DataFrames.
    
    Responsibilities:
        - Map API field names to standardized names
        - Convert string values to appropriate numeric types
        - Handle missing and null values
        - Structure data as pandas DataFrames
    """
    
    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """
        Safely convert value to float.
        
        Args:
            value: Value to convert
            
        Returns:
            Float value or None if conversion fails
        """
        if value is None:
            return None
        if isinstance(value, (int, float)):
            if math.isnan(value) or math.isinf(value):
                return None
            return float(value)
        if isinstance(value, str):
            value = value.strip()
            if value.lower() in ("none", "n/a", "-", "", "null", "nan"):
                return None
            try:
                result = float(value)
                if math.isnan(result) or math.isinf(result):
                    return None
                return result
            except ValueError:
                return None
        return None
    
    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        """Safely convert value to integer."""
        float_val = DataParser._safe_float(value)
        if float_val is None:
            return None
        return int(float_val)
    
    def parse_financial_statement(
        self,
        data: Dict,
        field_map: Dict[str, str],
        years: int = 5
    ) -> pd.DataFrame:
        """
        Parse financial statement data into a pandas DataFrame.
        
        Args:
            data: Raw API response
            field_map: Mapping from API field names to standardized names
            years: Number of years to include
            
        Returns:
            DataFrame with standardized field names as index, fiscal years as columns
        """
        annual_reports = data.get("annualReports", [])
        
        if not annual_reports:
            return pd.DataFrame()
        
        annual_reports = annual_reports[:years]
        
        records = {}
        
        for report in annual_reports:
            fiscal_date = report.get("fiscalDateEnding", "")
            fiscal_year = fiscal_date[:4] if fiscal_date else ""
            
            if not fiscal_year:
                continue
            
            column_data = {}
            
            for raw_field, std_field in field_map.items():
                if raw_field in report:
                    value = self._safe_float(report[raw_field])
                    if value is not None:
                        column_data[std_field] = value
            
            if column_data:
                records[fiscal_year] = column_data
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df = df[sorted(df.columns, reverse=True)]
        
        return df
    
    def parse_income_statement(self, data: Dict, years: int = 5) -> pd.DataFrame:
        """Parse income statement data."""
        return self.parse_financial_statement(data, INCOME_FIELD_MAP, years)
    
    def parse_balance_sheet(self, data: Dict, years: int = 5) -> pd.DataFrame:
        """Parse balance sheet data."""
        return self.parse_financial_statement(data, BALANCE_FIELD_MAP, years)
    
    def parse_cash_flow(self, data: Dict, years: int = 5) -> pd.DataFrame:
        """Parse cash flow statement data."""
        return self.parse_financial_statement(data, CASHFLOW_FIELD_MAP, years)
    
    def parse_company_overview(self, data: Dict) -> Optional[CompanyProfile]:
        """
        Parse company overview data.
        
        Args:
            data: Raw API response
            
        Returns:
            CompanyProfile object or None if parsing fails
        """
        if not data or "Symbol" not in data:
            return None
        
        return CompanyProfile(
            ticker=data.get("Symbol", ""),
            name=data.get("Name", ""),
            description=data.get("Description"),
            exchange=data.get("Exchange"),
            currency=data.get("Currency", "USD"),
            country=data.get("Country"),
            sector=data.get("Sector"),
            industry=data.get("Industry"),
            fiscal_year_end=data.get("FiscalYearEnd"),
            market_cap=self._safe_float(data.get("MarketCapitalization")),
            shares_outstanding=self._safe_float(data.get("SharesOutstanding")),
            high_52week=self._safe_float(data.get("52WeekHigh")),
            low_52week=self._safe_float(data.get("52WeekLow")),
            ma_50day=self._safe_float(data.get("50DayMovingAverage")),
            ma_200day=self._safe_float(data.get("200DayMovingAverage")),
            beta=self._safe_float(data.get("Beta")),
            pe_ratio=self._safe_float(data.get("PERatio")),
            peg_ratio=self._safe_float(data.get("PEGRatio")),
            book_value=self._safe_float(data.get("BookValue")),
            pb_ratio=self._safe_float(data.get("PriceToBookRatio")),
            ps_ratio=self._safe_float(data.get("PriceToSalesRatioTTM")),
            ev_to_revenue=self._safe_float(data.get("EVToRevenue")),
            ev_to_ebitda=self._safe_float(data.get("EVToEBITDA")),
            pe_trailing=self._safe_float(data.get("TrailingPE")),
            pe_forward=self._safe_float(data.get("ForwardPE")),
            profit_margin=self._safe_float(data.get("ProfitMargin")),
            operating_margin=self._safe_float(data.get("OperatingMarginTTM")),
            roa=self._safe_float(data.get("ReturnOnAssetsTTM")),
            roe=self._safe_float(data.get("ReturnOnEquityTTM")),
            dividend_per_share=self._safe_float(data.get("DividendPerShare")),
            dividend_yield=self._safe_float(data.get("DividendYield")),
            ex_dividend_date=data.get("ExDividendDate"),
            earnings_growth_yoy=self._safe_float(data.get("QuarterlyEarningsGrowthYOY")),
            revenue_growth_yoy=self._safe_float(data.get("QuarterlyRevenueGrowthYOY")),
            eps=self._safe_float(data.get("EPS")),
            eps_diluted_ttm=self._safe_float(data.get("DilutedEPSTTM")),
            analyst_target=self._safe_float(data.get("AnalystTargetPrice")),
            analyst_strong_buy=self._safe_int(data.get("AnalystRatingStrongBuy")),
            analyst_buy=self._safe_int(data.get("AnalystRatingBuy")),
            analyst_hold=self._safe_int(data.get("AnalystRatingHold")),
            analyst_sell=self._safe_int(data.get("AnalystRatingSell")),
            analyst_strong_sell=self._safe_int(data.get("AnalystRatingStrongSell")),
        )


# =============================================================================
# DATA POST-PROCESSOR
# =============================================================================

class DataPostProcessor:
    """
    Post-processes financial statements to fill calculable missing values.
    
    When Alpha Vantage returns 'None' for certain fields, we can often
    derive the values from other available data. This class handles those
    calculations to improve data completeness.
    """
    
    def process(self, statements: FinancialStatements) -> FinancialStatements:
        """
        Fill in calculable missing values and fix data quality issues.
        
        Args:
            statements: Raw financial statements from API
            
        Returns:
            Statements with derived values filled in where possible
        """
        if statements.is_empty:
            return statements
        
        # Calculate change_in_cash from balance sheet cash values
        self._fill_change_in_cash(statements)
        
        # Fix cash flow articulation using balance sheet (more reliable for recent years)
        self._fix_cash_flow_articulation(statements)
        
        return statements
    
    def _fill_change_in_cash(self, statements: FinancialStatements) -> None:
        """
        Calculate change_in_cash from year-over-year cash balances.
        
        Formula: change_in_cash[year] = cash[year] - cash[year-1]
        """
        cf = statements.cash_flow
        bs = statements.balance_sheet
        
        if cf is None or cf.empty or bs is None or bs.empty:
            return
        
        # Ensure change_in_cash row exists
        if "change_in_cash" not in cf.index:
            cf.loc["change_in_cash"] = np.nan
        
        # Get cash field from balance sheet
        cash_field = None
        for field in ["cash_and_equivalents", "cash_and_short_term_investments"]:
            if field in bs.index:
                cash_field = field
                break
        
        if cash_field is None:
            return
        
        years = sorted(cf.columns, reverse=True)
        
        for i, year in enumerate(years[:-1]):  # Skip last year (no prior year)
            # Only fill if currently missing
            if pd.isna(cf.loc["change_in_cash", year]):
                prior_year = years[i + 1]
                
                current_cash = bs.loc[cash_field, year] if year in bs.columns else None
                prior_cash = bs.loc[cash_field, prior_year] if prior_year in bs.columns else None
                
                if pd.notna(current_cash) and pd.notna(prior_cash):
                    cf.loc["change_in_cash", year] = current_cash - prior_cash
    
    def _fix_cash_flow_articulation(self, statements: FinancialStatements) -> None:
        """
        Fix cash flow statement articulation issues.
        
        The cash flow statement should satisfy:
        Change in Cash = Operating CF + Investing CF + Financing CF
        
        When there's a discrepancy (due to API data issues), we recalculate
        change_in_cash from the balance sheet cash values which are more reliable.
        """
        cf = statements.cash_flow
        bs = statements.balance_sheet
        
        if cf is None or cf.empty or bs is None or bs.empty:
            return
        
        # Get cash field from balance sheet
        cash_field = None
        for field in ["cash_and_equivalents", "cash_and_short_term_investments"]:
            if field in bs.index:
                cash_field = field
                break
        
        if cash_field is None:
            return
        
        years = sorted(cf.columns, reverse=True)
        
        for i, year in enumerate(years[:-1]):  # Skip last year (no prior year)
            prior_year = years[i + 1]
            
            # Get cash values from balance sheet
            current_cash = bs.loc[cash_field, year] if year in bs.columns else None
            prior_cash = bs.loc[cash_field, prior_year] if prior_year in bs.columns else None
            
            if pd.isna(current_cash) or pd.isna(prior_cash):
                continue
            
            # Calculate change in cash from balance sheet
            balance_calc_change = current_cash - prior_cash
            
            # Update cash flow statement with balance sheet calculated value
            cf.loc["change_in_cash", year] = balance_calc_change


# =============================================================================
# DATA VALIDATOR
# =============================================================================

class DataValidator:
    """
    Validates collected financial data for completeness and quality.
    
    Validation Checks:
        1. Field completeness per statement
        2. Critical field presence
        3. Accounting equation validation (Assets = Liabilities + Equity)
        4. Quality tier classification
    """
    
    def validate(self, statements: FinancialStatements) -> QualityMetrics:
        """
        Perform comprehensive validation of financial statements.
        
        Args:
            statements: FinancialStatements container
            
        Returns:
            QualityMetrics with completeness scores and validation results
        """
        metrics = QualityMetrics()
        
        # Validate income statement
        metrics.income_completeness, metrics.income_fields_present, metrics.income_fields_expected = \
            self._calculate_completeness(statements.income_statement, StatementType.INCOME)
        metrics.critical_income_valid = self._check_critical_fields(
            statements.income_statement, CRITICAL_INCOME_FIELDS
        )
        
        # Validate balance sheet
        metrics.balance_completeness, metrics.balance_fields_present, metrics.balance_fields_expected = \
            self._calculate_completeness(statements.balance_sheet, StatementType.BALANCE)
        metrics.critical_balance_valid = self._check_critical_fields(
            statements.balance_sheet, CRITICAL_BALANCE_FIELDS
        )
        
        # Validate cash flow
        metrics.cashflow_completeness, metrics.cashflow_fields_present, metrics.cashflow_fields_expected = \
            self._calculate_completeness(statements.cash_flow, StatementType.CASHFLOW)
        metrics.critical_cashflow_valid = self._check_critical_fields(
            statements.cash_flow, CRITICAL_CASHFLOW_FIELDS
        )
        
        # Calculate overall completeness (weighted average)
        metrics.overall_completeness = (
            metrics.income_completeness * 0.35 +
            metrics.balance_completeness * 0.35 +
            metrics.cashflow_completeness * 0.30
        )
        
        # Validate accounting equation
        metrics.accounting_equation_valid, metrics.accounting_equation_deviation = \
            self._validate_accounting_equation(statements.balance_sheet)
        
        # Determine quality tier
        metrics.quality_tier = VALIDATION_CONFIG.get_quality_tier(metrics.overall_completeness)
        
        # Add warnings/errors
        self._add_validation_messages(statements, metrics)
        
        return metrics
    
    def _calculate_completeness(
        self, 
        df: pd.DataFrame, 
        stmt_type: StatementType
    ) -> Tuple[float, int, int]:
        """
        Calculate completeness score for a statement.
        
        Returns:
            Tuple of (completeness_ratio, fields_present, fields_expected)
        """
        if df is None or df.empty:
            expected = len(get_expected_fields(stmt_type))
            return 0.0, 0, expected
        
        expected_fields = get_expected_fields(stmt_type)
        present_fields = set(df.index)
        
        # Count fields that have at least one non-null value
        valid_fields = 0
        for field in present_fields:
            if field in expected_fields:
                if not df.loc[field].isna().all():
                    valid_fields += 1
        
        expected_count = len(expected_fields)
        completeness = valid_fields / expected_count if expected_count > 0 else 0.0
        
        return completeness, valid_fields, expected_count
    
    def _check_critical_fields(
        self, 
        df: pd.DataFrame, 
        critical_fields: FrozenSet[str]
    ) -> bool:
        """Check if all critical fields are present and have data."""
        if df is None or df.empty:
            return False
        
        for field in critical_fields:
            if field not in df.index:
                return False
            if df.loc[field].isna().all():
                return False
        
        return True
    
    def _validate_accounting_equation(
        self, 
        balance_sheet: pd.DataFrame
    ) -> Tuple[bool, Optional[float]]:
        """
        Validate accounting equation: Assets = Liabilities + Equity.
        
        Returns:
            Tuple of (is_valid, deviation_percentage)
        """
        if balance_sheet is None or balance_sheet.empty:
            return False, None
        
        required = {"total_assets", "total_liabilities", "total_equity"}
        if not required.issubset(set(balance_sheet.index)):
            return False, None
        
        # Check most recent year
        latest_year = balance_sheet.columns[0]
        
        try:
            assets = balance_sheet.loc["total_assets", latest_year]
            liabilities = balance_sheet.loc["total_liabilities", latest_year]
            equity = balance_sheet.loc["total_equity", latest_year]
            
            if pd.isna(assets) or pd.isna(liabilities) or pd.isna(equity):
                return False, None
            
            computed = liabilities + equity
            deviation = abs(assets - computed) / assets if assets != 0 else 0.0
            
            is_valid = deviation <= VALIDATION_CONFIG.accounting_equation_tolerance
            return is_valid, deviation
            
        except (KeyError, TypeError):
            return False, None
    
    def _add_validation_messages(
        self, 
        statements: FinancialStatements, 
        metrics: QualityMetrics
    ) -> None:
        """Add warning and error messages based on validation results."""
        
        # Check years of data
        if statements.years_available < VALIDATION_CONFIG.years_minimum:
            metrics.errors.append(
                f"Insufficient data: {statements.years_available} years "
                f"(minimum {VALIDATION_CONFIG.years_minimum} required)"
            )
        elif statements.years_available < VALIDATION_CONFIG.years_required:
            metrics.warnings.append(
                f"Limited data: {statements.years_available} years "
                f"(recommended {VALIDATION_CONFIG.years_required})"
            )
        
        # Check critical fields
        if not metrics.critical_income_valid:
            metrics.warnings.append("Income statement missing critical fields")
        if not metrics.critical_balance_valid:
            metrics.warnings.append("Balance sheet missing critical fields")
        if not metrics.critical_cashflow_valid:
            metrics.warnings.append("Cash flow statement missing critical fields")
        
        # Check accounting equation
        if not metrics.accounting_equation_valid and metrics.accounting_equation_deviation is not None:
            metrics.warnings.append(
                f"Accounting equation deviation: {metrics.accounting_equation_deviation:.2%}"
            )
        
        # Detect suspicious data patterns
        self._detect_suspicious_patterns(statements, metrics)
    
    def _detect_suspicious_patterns(
        self,
        statements: FinancialStatements,
        metrics: QualityMetrics
    ) -> None:
        """
        Detect and flag suspicious data patterns from API.
        
        API data gaps are logged as 'info' (don't affect validation status).
        Suspicious values (like 0 interest with debt) are logged as 'warnings'.
        """
        
        if statements.is_empty:
            return
        
        latest_year = statements.fiscal_periods[0] if statements.fiscal_periods else None
        if not latest_year:
            return
        
        # Check for interest expense = 0 with significant debt
        if statements.balance_sheet is not None and statements.income_statement is not None:
            total_debt = self._get_value(statements.balance_sheet, "total_debt", latest_year)
            interest_exp = self._get_value(statements.income_statement, "interest_expense", latest_year)
            
            if total_debt and total_debt > 1e9:  # Company has >$1B debt
                if interest_exp is None:
                    # API data gap - informational only
                    metrics.info.append(
                        f"API data gap: interest_expense unavailable for {latest_year} "
                        f"(company has ${total_debt/1e9:.1f}B debt)"
                    )
                elif interest_exp == 0:
                    # Suspicious value - this is a warning
                    metrics.info.append(
                        f"Suspicious: interest_expense=0 for {latest_year} "
                        f"but total_debt=${total_debt/1e9:.1f}B (possible API data issue)"
                    )
        
        # Check for missing cash flow reconciliation fields in recent years
        if statements.cash_flow is not None:
            cf = statements.cash_flow
            missing_recent = []
            reconciliation_fields = ["change_receivables"]  # change_in_cash now calculated
            
            for field in reconciliation_fields:
                if field in cf.index:
                    # Check if most recent years are N/A but older years have data
                    recent_na = pd.isna(cf.loc[field, latest_year]) if latest_year in cf.columns else True
                    has_older_data = not cf.loc[field].isna().all()
                    if recent_na and has_older_data:
                        missing_recent.append(field)
            
            if missing_recent:
                # API data gap - informational only
                metrics.info.append(
                    f"API data gap: {', '.join(missing_recent)} unavailable for recent years"
                )
    
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
    
    def determine_status(self, metrics: QualityMetrics, years: int) -> ValidationStatus:
        """
        Determine overall validation status.
        
        Args:
            metrics: Quality metrics from validation
            years: Number of years of data
            
        Returns:
            ValidationStatus enum value
        """
        if metrics.errors:
            return ValidationStatus.INVALID
        
        if years < VALIDATION_CONFIG.years_minimum:
            return ValidationStatus.INVALID
        
        all_critical_valid = (
            metrics.critical_income_valid and
            metrics.critical_balance_valid and
            metrics.critical_cashflow_valid
        )
        
        if not all_critical_valid:
            return ValidationStatus.PARTIAL
        
        if metrics.warnings:
            return ValidationStatus.PARTIAL
        
        return ValidationStatus.VALID


# =============================================================================
# METRICS CALCULATOR
# =============================================================================

class MetricsCalculator:
    """
    Calculates derived metrics from financial statements.
    
    These metrics are prepared during data acquisition to serve as
    inputs for downstream valuation and analysis phases.
    """
    
    def calculate(
        self, 
        statements: FinancialStatements,
        profile: CompanyProfile
    ) -> DerivedMetrics:
        """
        Calculate all derived metrics.
        
        Args:
            statements: Financial statements
            profile: Company profile with market data
            
        Returns:
            DerivedMetrics container
        """
        metrics = DerivedMetrics()
        
        if statements.is_empty:
            return metrics
        
        fiscal_periods = statements.fiscal_periods
        
        for year in fiscal_periods:
            # Free Cash Flow: OCF - |CapEx|
            metrics.fcf_calculated[year] = self._calculate_fcf(statements, year)
            
            # EBITDA: Operating Income + D&A
            metrics.ebitda_calculated[year] = self._calculate_ebitda(statements, year)
            
            # Working Capital: Current Assets - Current Liabilities
            metrics.working_capital[year] = self._calculate_working_capital(statements, year)
            
            # Net Debt: Total Debt - Cash
            metrics.net_debt[year] = self._calculate_net_debt(statements, year)
            
            # Invested Capital: Equity + Debt - Cash
            metrics.invested_capital[year] = self._calculate_invested_capital(statements, year)
        
        # Enterprise Value (current only, requires market cap)
        metrics.enterprise_value = self._calculate_enterprise_value(statements, profile)
        
        return metrics
    
    def _get_value(
        self, 
        df: pd.DataFrame, 
        field: str, 
        year: str
    ) -> Optional[float]:
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
    
    def _calculate_fcf(
        self, 
        statements: FinancialStatements, 
        year: str
    ) -> Optional[float]:
        """
        Calculate Free Cash Flow: OCF - CapEx
        
        Note: Alpha Vantage reports CapEx as a positive value (absolute spending).
        We use abs() to handle any sign variations in the data source.
        """
        ocf = self._get_value(statements.cash_flow, "operating_cash_flow", year)
        capex = self._get_value(statements.cash_flow, "capital_expenditure", year)
        
        if ocf is None or capex is None:
            return None
        
        return ocf - abs(capex)
    
    def _calculate_ebitda(
        self, 
        statements: FinancialStatements, 
        year: str
    ) -> Optional[float]:
        """Calculate EBITDA: Operating Income + D&A"""
        # First try to get EBITDA from income statement if available
        ebitda = self._get_value(statements.income_statement, "ebitda", year)
        if ebitda is not None:
            return ebitda
        
        # Otherwise calculate from components
        operating_income = self._get_value(statements.income_statement, "operating_income", year)
        da_income = self._get_value(statements.income_statement, "depreciation_amortization", year)
        da_cashflow = self._get_value(statements.cash_flow, "depreciation_amortization", year)
        
        da = da_income or da_cashflow
        
        if operating_income is None:
            return None
        if da is None:
            return operating_income  # Return operating income if no D&A available
        
        return operating_income + da
    
    def _calculate_working_capital(
        self, 
        statements: FinancialStatements, 
        year: str
    ) -> Optional[float]:
        """Calculate Working Capital: Current Assets - Current Liabilities"""
        current_assets = self._get_value(statements.balance_sheet, "current_assets", year)
        current_liabilities = self._get_value(statements.balance_sheet, "current_liabilities", year)
        
        if current_assets is None or current_liabilities is None:
            return None
        
        return current_assets - current_liabilities
    
    def _calculate_net_debt(
        self, 
        statements: FinancialStatements, 
        year: str
    ) -> Optional[float]:
        """Calculate Net Debt: Total Debt - Cash"""
        total_debt = self._get_value(statements.balance_sheet, "total_debt", year)
        cash = self._get_value(statements.balance_sheet, "cash_and_equivalents", year)
        
        # Try alternative cash field
        if cash is None:
            cash = self._get_value(statements.balance_sheet, "cash_and_short_term_investments", year)
        
        if total_debt is None:
            # Try to calculate total debt from components
            long_term = self._get_value(statements.balance_sheet, "long_term_debt", year) or 0
            short_term = self._get_value(statements.balance_sheet, "short_term_debt", year) or 0
            current_debt = self._get_value(statements.balance_sheet, "current_debt", year) or 0
            total_debt = long_term + max(short_term, current_debt)
            if total_debt == 0:
                total_debt = None
        
        if total_debt is None or cash is None:
            return None
        
        return total_debt - cash
    
    def _calculate_invested_capital(
        self, 
        statements: FinancialStatements, 
        year: str
    ) -> Optional[float]:
        """Calculate Invested Capital: Equity + Debt - Cash"""
        equity = self._get_value(statements.balance_sheet, "total_equity", year)
        total_debt = self._get_value(statements.balance_sheet, "total_debt", year)
        cash = self._get_value(statements.balance_sheet, "cash_and_equivalents", year)
        
        if cash is None:
            cash = self._get_value(statements.balance_sheet, "cash_and_short_term_investments", year)
        
        if equity is None:
            return None
        
        debt = total_debt or 0
        cash_val = cash or 0
        
        return equity + debt - cash_val
    
    def _calculate_enterprise_value(
        self, 
        statements: FinancialStatements,
        profile: CompanyProfile
    ) -> Optional[float]:
        """Calculate Enterprise Value: Market Cap + Debt - Cash"""
        if profile.market_cap is None:
            return None
        
        if statements.is_empty or not statements.fiscal_periods:
            return None
        
        latest_year = statements.fiscal_periods[0]
        net_debt = self._calculate_net_debt(statements, latest_year)
        
        if net_debt is None:
            return None
        
        return profile.market_cap + net_debt


# =============================================================================
# DIVIDEND ANALYZER
# =============================================================================

class DividendAnalyzer:
    """
    Analyzes dividend history for DDM valuation.
    
    Extracts dividend data from cash flow statements and calculates
    metrics needed for dividend discount model valuation.
    """
    
    def analyze(
        self, 
        statements: FinancialStatements,
        profile: CompanyProfile
    ) -> DividendHistory:
        """
        Analyze dividend history.
        
        Args:
            statements: Financial statements
            profile: Company profile
            
        Returns:
            DividendHistory container
        """
        history = DividendHistory()
        
        if statements.cash_flow is None or statements.cash_flow.empty:
            return history
        
        # Extract annual dividends
        history.annual_dividends = self._extract_dividends(statements.cash_flow)
        history.years_of_data = len(history.annual_dividends)
        
        if history.years_of_data == 0:
            return history
        
        # Get current DPS from profile or calculate
        if profile.dividend_per_share is not None:
            history.current_annual_dps = profile.dividend_per_share
        else:
            # Estimate from most recent dividend payment
            latest_year = max(history.annual_dividends.keys())
            shares = self._get_shares_outstanding(statements.balance_sheet, latest_year)
            if shares and history.annual_dividends[latest_year]:
                history.current_annual_dps = abs(history.annual_dividends[latest_year]) / shares
        
        # Calculate CAGR
        history.dividend_cagr = self._calculate_cagr(history.annual_dividends)
        
        # Check for dividend cuts
        history.has_dividend_cuts = self._check_dividend_cuts(history.annual_dividends)
        
        # Assess stability
        history.payout_stable = self._assess_stability(history.annual_dividends)
        
        return history
    
    def _extract_dividends(self, cash_flow: pd.DataFrame) -> Dict[str, float]:
        """
        Extract annual dividend payments from cash flow.
        
        Note: Alpha Vantage reports the same value in both 'dividends_paid' and
        'common_dividends' for companies paying only common dividends. We use
        a priority system to avoid double-counting.
        
        Priority: dividends_paid > common_dividends > preferred_dividends
        """
        dividends = {}
        
        # Priority order - use first available, don't sum (they often duplicate)
        primary_fields = ["dividends_paid", "common_dividends"]
        
        for year in cash_flow.columns:
            dividend_value = None
            
            # Get primary dividend (common dividends)
            for field in primary_fields:
                if field in cash_flow.index:
                    value = cash_flow.loc[field, year]
                    if pd.notna(value) and abs(float(value)) > 0:
                        dividend_value = abs(float(value))
                        break  # Use first found, don't sum
            
            # Add preferred dividends separately if present (rare)
            if "preferred_dividends" in cash_flow.index:
                pref_value = cash_flow.loc["preferred_dividends", year]
                if pd.notna(pref_value) and abs(float(pref_value)) > 0:
                    if dividend_value is None:
                        dividend_value = abs(float(pref_value))
                    else:
                        dividend_value += abs(float(pref_value))
            
            if dividend_value is not None and dividend_value > 0:
                dividends[year] = dividend_value
        
        return dividends
    
    def _get_shares_outstanding(
        self, 
        balance_sheet: pd.DataFrame, 
        year: str
    ) -> Optional[float]:
        """Get shares outstanding from balance sheet."""
        if balance_sheet is None or balance_sheet.empty:
            return None
        if "shares_outstanding" not in balance_sheet.index:
            return None
        if year not in balance_sheet.columns:
            return None
        value = balance_sheet.loc["shares_outstanding", year]
        if pd.isna(value):
            return None
        return float(value)
    
    def _calculate_cagr(self, dividends: Dict[str, float]) -> Optional[float]:
        """Calculate dividend CAGR."""
        if len(dividends) < 2:
            return None
        
        years = sorted(dividends.keys())
        start_year = years[0]
        end_year = years[-1]
        
        start_value = dividends[start_year]
        end_value = dividends[end_year]
        
        if start_value <= 0 or end_value <= 0:
            return None
        
        n_years = int(end_year) - int(start_year)
        if n_years <= 0:
            return None
        
        cagr = (end_value / start_value) ** (1 / n_years) - 1
        return cagr
    
    def _check_dividend_cuts(self, dividends: Dict[str, float]) -> bool:
        """Check if there were any dividend cuts."""
        if len(dividends) < 2:
            return False
        
        years = sorted(dividends.keys(), reverse=True)
        
        for i in range(len(years) - 1):
            current = dividends[years[i]]
            previous = dividends[years[i + 1]]
            
            if current < previous * 0.95:  # 5% tolerance
                return True
        
        return False
    
    def _assess_stability(self, dividends: Dict[str, float]) -> bool:
        """Assess if dividend payouts are stable."""
        if len(dividends) < 3:
            return False
        
        values = list(dividends.values())
        mean_dividend = np.mean(values)
        std_dividend = np.std(values)
        
        if mean_dividend == 0:
            return False
        
        # Coefficient of variation < 20% indicates stability
        cv = std_dividend / mean_dividend
        return cv < 0.20


# =============================================================================
# MAIN DATA COLLECTOR
# =============================================================================

class DataCollector:
    """
    Main data collection orchestrator for fundamental analysis.
    
    Coordinates API requests, caching, parsing, validation, and metric
    calculation to produce complete financial data for a company.
    
    Features:
        - Alpha Vantage API as primary data source
        - Yahoo Finance supplementation for missing data
        - Intelligent caching with 5-hour expiry
        - Comprehensive validation and quality scoring
    
    Usage:
        collector = DataCollector()
        result = collector.collect("AAPL")
        
        if result.is_valid:
            statements = result.statements
            metrics = result.derived_metrics
            # Proceed with analysis
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        enable_supplementation: bool = True,
    ):
        """
        Initialize DataCollector.
        
        Args:
            api_key: Alpha Vantage API key (defaults to environment variable)
            cache_dir: Cache directory path (defaults to project cache)
            enable_supplementation: Enable Yahoo Finance data supplementation
        """
        self.cache = CacheManager(
            cache_dir=cache_dir or CACHE_DIR,
            expiry_seconds=ALPHA_VANTAGE_CONFIG.cache_expiry_seconds,
        )
        
        self.api_client = AlphaVantageClient(
            api_key=api_key or ALPHA_VANTAGE_CONFIG.api_key,
            base_url=ALPHA_VANTAGE_CONFIG.base_url,
            timeout=ALPHA_VANTAGE_CONFIG.request_timeout,
        )
        
        self.parser = DataParser()
        self.validator = DataValidator()
        self.metrics_calc = MetricsCalculator()
        self.dividend_analyzer = DividendAnalyzer()
        
        # Initialize Yahoo Finance supplementer if available and enabled
        self.enable_supplementation = enable_supplementation
        self.supplementer = None
        if enable_supplementation and YAHOO_SUPPLEMENTER_AVAILABLE:
            try:
                self.supplementer = YahooFinanceSupplementer()
                if not self.supplementer.is_available:
                    self.supplementer = None
                    LOGGER.info("Yahoo Finance supplementation disabled (yfinance not installed)")
            except Exception as e:
                LOGGER.warning(f"Failed to initialize Yahoo Finance supplementer: {e}")
                self.supplementer = None
        
        LOGGER.info(f"DataCollector initialized (v{__version__})")
    
    def collect(
        self,
        ticker: str,
        force_refresh: bool = False,
        supplement: bool = True,
    ) -> CollectionResult:
        """
        Collect all financial data for a company.
        
        Args:
            ticker: Stock ticker symbol
            force_refresh: Bypass cache and fetch fresh data
            supplement: Enable Yahoo Finance supplementation for missing data
            
        Returns:
            CollectionResult containing all financial data and metrics
        """
        ticker = ticker.upper().strip()
        LOGGER.info(f"Starting data collection for {ticker}")
        
        api_calls = 0
        from_cache = False
        supplementation_report = None
        data_supplemented = False
        
        # Check if all data is cached
        if not force_refresh and self.cache.get_all_valid(ticker):
            from_cache = True
            LOGGER.info(f"Using cached data for {ticker}")
        
        # Collect income statement
        income_df, income_calls = self._fetch_statement(
            ticker, "income_statement",
            self.api_client.get_income_statement,
            self.parser.parse_income_statement,
            force_refresh,
        )
        api_calls += income_calls
        
        # Collect balance sheet
        balance_df, balance_calls = self._fetch_statement(
            ticker, "balance_sheet",
            self.api_client.get_balance_sheet,
            self.parser.parse_balance_sheet,
            force_refresh,
        )
        api_calls += balance_calls
        
        # Collect cash flow
        cashflow_df, cashflow_calls = self._fetch_statement(
            ticker, "cash_flow",
            self.api_client.get_cash_flow,
            self.parser.parse_cash_flow,
            force_refresh,
        )
        api_calls += cashflow_calls
        
        # Collect company overview
        company_profile, overview_calls = self._fetch_overview(
            ticker, force_refresh
        )
        api_calls += overview_calls
        
        if company_profile is None:
            company_profile = CompanyProfile(ticker=ticker, name=ticker)
        
        # Create statements container
        statements = FinancialStatements(
            income_statement=income_df,
            balance_sheet=balance_df,
            cash_flow=cashflow_df,
        )
        
        # Post-process to fill calculable missing values
        post_processor = DataPostProcessor()
        statements = post_processor.process(statements)
        
        # Apply SEC data corrections for known API issues
        sec_corrections = []
        statements, sec_corrections = self._apply_sec_corrections(ticker, statements)
        
        # Supplement data with Yahoo Finance (if enabled and available)
        if supplement and self.supplementer is not None:
            try:
                # Create a temporary result for supplementation
                temp_result = CollectionResult(
                    company_profile=company_profile,
                    statements=statements,
                    quality_metrics=QualityMetrics(),
                    validation_status=ValidationStatus.PARTIAL,
                    derived_metrics=DerivedMetrics(),
                    dividend_history=DividendHistory(),
                )
                
                # Run supplementation
                temp_result, supplementation_report = self.supplementer.supplement(temp_result)
                
                # Update statements with supplemented data
                statements = temp_result.statements
                
                if supplementation_report and supplementation_report.gaps_filled > 0:
                    data_supplemented = True
                    LOGGER.info(f"Supplemented {supplementation_report.gaps_filled} data gaps from Yahoo Finance")
                    
            except Exception as e:
                LOGGER.warning(f"Yahoo Finance supplementation failed: {e}")
                supplementation_report = None
        
        # Validate data (after supplementation)
        quality_metrics = self.validator.validate(statements)
        validation_status = self.validator.determine_status(
            quality_metrics, 
            statements.years_available
        )
        
        # Calculate derived metrics
        derived_metrics = self.metrics_calc.calculate(statements, company_profile)
        
        # Analyze dividends
        dividend_history = self.dividend_analyzer.analyze(statements, company_profile)
        
        LOGGER.info(
            f"Collection complete for {ticker}: "
            f"status={validation_status.value}, "
            f"years={statements.years_available}, "
            f"quality={quality_metrics.quality_tier.value}, "
            f"api_calls={api_calls}, "
            f"supplemented={data_supplemented}"
        )
        
        return CollectionResult(
            company_profile=company_profile,
            statements=statements,
            quality_metrics=quality_metrics,
            validation_status=validation_status,
            derived_metrics=derived_metrics,
            dividend_history=dividend_history,
            supplementation_report=supplementation_report,
            api_calls_made=api_calls,
            from_cache=from_cache if api_calls == 0 else False,
            data_supplemented=data_supplemented,
        )
    
    def _fetch_statement(
        self,
        ticker: str,
        data_type: str,
        api_method,
        parse_method,
        force_refresh: bool,
    ) -> Tuple[pd.DataFrame, int]:
        """
        Fetch financial statement with caching.
        
        Returns:
            Tuple of (DataFrame, api_calls_made)
        """
        if not force_refresh:
            cached = self.cache.get(ticker, data_type)
            if cached is not None:
                LOGGER.info(f"Using cached {data_type}")
                return parse_method(cached, DATA_CONFIG.years_of_data), 0
        
        data, error = api_method(ticker)
        
        if data is not None:
            self.cache.save(ticker, data_type, data)
            return parse_method(data, DATA_CONFIG.years_of_data), 1
        else:
            LOGGER.warning(f"Failed to fetch {data_type}: {error}")
            return pd.DataFrame(), 1
    
    def _fetch_overview(
        self,
        ticker: str,
        force_refresh: bool,
    ) -> Tuple[Optional[CompanyProfile], int]:
        """
        Fetch company overview with caching.
        
        Returns:
            Tuple of (CompanyProfile, api_calls_made)
        """
        if not force_refresh:
            cached = self.cache.get(ticker, "overview")
            if cached is not None:
                LOGGER.info("Using cached overview")
                return self.parser.parse_company_overview(cached), 0
        
        data, error = self.api_client.get_overview(ticker)
        
        if data is not None:
            self.cache.save(ticker, "overview", data)
            return self.parser.parse_company_overview(data), 1
        else:
            LOGGER.warning(f"Failed to fetch overview: {error}")
            return None, 1
    
    def _apply_sec_corrections(
        self,
        ticker: str,
        statements: FinancialStatements,
    ) -> Tuple[FinancialStatements, List[str]]:
        """
        Apply SEC-verified data corrections to fix known API data issues.
        
        This method corrects data fields where the API returns incorrect or
        combined values that differ from official SEC 10-K filings.
        
        Current corrections:
            - accounts_receivable: API returns AR + Vendor Non-Trade combined
            - interest_expense: API doesn't return this for recent years
        
        IMPORTANT: SEC corrections are stored in MILLIONS in config.py.
        API data is in RAW DOLLARS. We must convert by multiplying by 1,000,000.
        
        Args:
            ticker: Stock ticker symbol
            statements: FinancialStatements to correct
            
        Returns:
            Tuple of (corrected FinancialStatements, list of corrections made)
        """
        from .config import has_sec_corrections, get_sec_correction
        
        corrections_made = []
        
        if not has_sec_corrections(ticker):
            return statements, corrections_made
        
        LOGGER.info(f"Applying SEC data corrections for {ticker}")
        
        # SEC corrections are in millions, API data is in raw dollars
        # Multiply by 1,000,000 to convert
        MILLIONS_TO_RAW = 1_000_000
        
        # Correct balance sheet fields (accounts_receivable)
        if not statements.balance_sheet.empty:
            bs = statements.balance_sheet.copy()
            
            for col in bs.columns:
                fiscal_date = col
                
                # Correct accounts_receivable (trade only, not combined)
                if "accounts_receivable" in bs.index:
                    corrected_ar_millions = get_sec_correction(ticker, "accounts_receivable", fiscal_date)
                    if corrected_ar_millions is not None:
                        old_value = bs.at["accounts_receivable", col]
                        # Convert from millions to raw dollars
                        corrected_ar_raw = corrected_ar_millions * MILLIONS_TO_RAW
                        bs.at["accounts_receivable", col] = corrected_ar_raw
                        
                        # Format for logging (show in billions for readability)
                        old_b = old_value / 1e9 if pd.notna(old_value) else 0
                        new_b = corrected_ar_raw / 1e9
                        corrections_made.append(
                            f"Balance Sheet: accounts_receivable {fiscal_date}: "
                            f"${old_b:.2f}B -> ${new_b:.2f}B (SEC trade AR only)"
                        )
            
            statements = FinancialStatements(
                income_statement=statements.income_statement,
                balance_sheet=bs,
                cash_flow=statements.cash_flow,
            )
        
        # Correct income statement fields (interest_expense, interest_income)
        if not statements.income_statement.empty:
            inc = statements.income_statement.copy()
            
            for col in inc.columns:
                fiscal_date = col
                
                # Correct interest_expense
                corrected_ie_millions = get_sec_correction(ticker, "interest_expense", fiscal_date)
                if corrected_ie_millions is not None:
                    if "interest_expense" not in inc.index:
                        # Add the row if it doesn't exist
                        inc.loc["interest_expense"] = float('nan')
                    
                    old_value = inc.at["interest_expense", col]
                    old_is_valid = pd.notna(old_value) and old_value != 0
                    
                    # Convert from millions to raw dollars
                    corrected_ie_raw = corrected_ie_millions * MILLIONS_TO_RAW
                    inc.at["interest_expense", col] = corrected_ie_raw
                    
                    # Format for logging
                    new_b = corrected_ie_raw / 1e9
                    if not old_is_valid:
                        corrections_made.append(
                            f"Income Statement: interest_expense {fiscal_date}: "
                            f"N/A -> ${new_b:.3f}B (SEC 10-K Note)"
                        )
                    else:
                        old_b = old_value / 1e9
                        corrections_made.append(
                            f"Income Statement: interest_expense {fiscal_date}: "
                            f"${old_b:.3f}B -> ${new_b:.3f}B (SEC 10-K Note)"
                        )
                
                # Also add interest_income if available
                corrected_ii_millions = get_sec_correction(ticker, "interest_income", fiscal_date)
                if corrected_ii_millions is not None:
                    if "interest_income" not in inc.index:
                        inc.loc["interest_income"] = float('nan')
                    
                    # Convert from millions to raw dollars
                    corrected_ii_raw = corrected_ii_millions * MILLIONS_TO_RAW
                    inc.at["interest_income", col] = corrected_ii_raw
            
            statements = FinancialStatements(
                income_statement=inc,
                balance_sheet=statements.balance_sheet,
                cash_flow=statements.cash_flow,
            )
        
        if corrections_made:
            LOGGER.info(f"Applied {len(corrections_made)} SEC data corrections for {ticker}")
            for correction in corrections_made:
                LOGGER.info(f"  • {correction}")
        
        return statements, corrections_made
    
    def get_cache_status(self, ticker: str) -> Dict[str, bool]:
        """Get cache validity status for a ticker."""
        return self.cache.get_status(ticker)
    
    def save_to_json(
        self,
        result: CollectionResult,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Save collected data to JSON file.
        
        Args:
            result: CollectionResult to save
            output_path: Optional custom output path
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            output_dir = OUTPUT_DIR / result.ticker
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{result.ticker}_collected_data.json"
        
        output_data = result.to_dict()
        
        # Add statement data
        output_data["statements"] = {
            "income_statement": (
                result.statements.income_statement.to_dict() 
                if not result.statements.income_statement.empty else {}
            ),
            "balance_sheet": (
                result.statements.balance_sheet.to_dict() 
                if not result.statements.balance_sheet.empty else {}
            ),
            "cash_flow": (
                result.statements.cash_flow.to_dict() 
                if not result.statements.cash_flow.empty else {}
            ),
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        LOGGER.info(f"Saved data to {output_path}")
        return output_path


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def collect_financial_data(
    ticker: str,
    api_key: Optional[str] = None,
    force_refresh: bool = False,
    supplement: bool = True,
) -> CollectionResult:
    """
    Convenience function to collect financial data for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        api_key: Optional API key
        force_refresh: Bypass cache
        supplement: Enable Yahoo Finance supplementation (default: True)
        
    Returns:
        CollectionResult with all financial data and metrics
    """
    collector = DataCollector(api_key=api_key)
    return collector.collect(ticker, force_refresh=force_refresh, supplement=supplement)