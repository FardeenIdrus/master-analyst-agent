"""
Data Collector - Alpha Vantage API
Fetches and cleans financial statements (5 years)

Usage:
    from datacollector import DataCollector
    
    collector = DataCollector()
    data = collector.collect("AAPL")
"""

import requests
import pandas as pd
from pathlib import Path

try:
    from config import ALPHA_VANTAGE_KEY
except ImportError:
    ALPHA_VANTAGE_KEY = ""


class DataCollector:
    """Collects financial data from Alpha Vantage API."""
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    # Field mappings: Alpha Vantage → Standard names for ratios
    INCOME_FIELDS = {
        'fiscalDateEnding': 'Date',
        'totalRevenue': 'Total Revenue',
        'grossProfit': 'Gross Profit',
        'costOfRevenue': 'Cost Of Revenue',
        'costofGoodsAndServicesSold': 'Cost Of Revenue',
        'operatingIncome': 'Operating Income',
        'operatingExpenses': 'Operating Expense',
        'netIncome': 'Net Income',
        'ebitda': 'EBITDA',
        'ebit': 'EBIT',
        'interestExpense': 'Interest Expense',
        'interestIncome': 'Interest Income',
        'incomeTaxExpense': 'Tax Provision',
        'incomeBeforeTax': 'Pretax Income',
        'researchAndDevelopment': 'Research And Development',
        'sellingGeneralAndAdministrative': 'Selling General And Administration',
        'depreciationAndAmortization': 'Depreciation And Amortization',
    }
    
    BALANCE_FIELDS = {
        'fiscalDateEnding': 'Date',
        'totalAssets': 'Total Assets',
        'totalCurrentAssets': 'Current Assets',
        'totalNonCurrentAssets': 'Total Non Current Assets',
        'totalLiabilities': 'Total Liabilities Net Minority Interest',
        'totalCurrentLiabilities': 'Current Liabilities',
        'totalNonCurrentLiabilities': 'Total Non Current Liabilities',
        'totalShareholderEquity': 'Stockholders Equity',
        'retainedEarnings': 'Retained Earnings',
        'commonStock': 'Common Stock',
        'cashAndCashEquivalentsAtCarryingValue': 'Cash And Cash Equivalents',
        'cashAndShortTermInvestments': 'Cash And Cash Equivalents',
        'inventory': 'Inventory',
        'currentNetReceivables': 'Accounts Receivable',
        'accountsPayable': 'Accounts Payable',
        'shortTermDebt': 'Current Debt',
        'currentLongTermDebt': 'Current Debt',
        'longTermDebt': 'Long Term Debt',
        'longTermDebtNoncurrent': 'Long Term Debt',
        'shortLongTermDebtTotal': 'Total Debt',
        'propertyPlantEquipment': 'Net PPE',
        'commonStockSharesOutstanding': 'Shares Outstanding',
    }
    
    CASHFLOW_FIELDS = {
        'fiscalDateEnding': 'Date',
        'operatingCashflow': 'Operating Cash Flow',
        'capitalExpenditures': 'Capital Expenditure',
        'cashflowFromInvestment': 'Investing Cash Flow',
        'cashflowFromFinancing': 'Financing Cash Flow',
        'dividendPayout': 'Cash Dividends Paid',
        'dividendPayoutCommonStock': 'Cash Dividends Paid',
        'dividendPayoutPreferredStock': 'Preferred Dividends Paid',
        'paymentsForRepurchaseOfCommonStock': 'Stock Repurchase',
        'changeInCashAndCashEquivalents': 'Changes In Cash',
        'netIncome': 'Net Income From Continuing Operations',
        'depreciationDepletionAndAmortization': 'Depreciation And Amortization',
        'changeInReceivables': 'Change In Receivables',
        'changeInInventory': 'Change In Inventory',
    }
    
    def __init__(self, api_key: str = None):
        """
        Initialize collector.
        
        Args:
            api_key: Alpha Vantage API key (or set in config.py)
        """
        self.api_key = api_key or ALPHA_VANTAGE_KEY
        
        if not self.api_key or self.api_key == "YOUR_ALPHA_VANTAGE_KEY":
            raise ValueError(
                "Alpha Vantage API key required!\n"
                "  1. Get free key: https://www.alphavantage.co/support/#api-key\n"
                "  2. Add to config.py: ALPHA_VANTAGE_KEY = 'your_key'"
            )
        
        self.ticker = None
        self.data = {}
    
    def _request(self, function: str, ticker: str) -> dict:
        """Make API request."""
        import time
        
        params = {
            'function': function,
            'symbol': ticker,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"API request failed: {e}")
        
        # Check for API errors
        if "Error Message" in data:
            raise ValueError(f"Invalid ticker symbol: {ticker}")
        
        if "Note" in data:
            raise ValueError(
                "Alpha Vantage rate limit reached (25 requests/day).\n"
                "Wait 24 hours or get premium key."
            )
        
        if "Information" in data:
            # Rate limit - wait and retry once
            print(f"    ⏳ Rate limit hit, waiting 60 seconds...")
            time.sleep(60)
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            data = response.json()
            
            if "Information" in data:
                raise ValueError("Still rate limited. Wait a few minutes and try again.")
        
        # Wait 1.5 seconds between requests (Alpha Vantage requires 1/sec)
        time.sleep(1.5)
        
        return data
    
    def _parse_reports(self, data: dict, report_key: str, field_map: dict, years: int = 5) -> pd.DataFrame:
        """Parse API response into clean DataFrame."""
        if report_key not in data:
            raise ValueError(f"No '{report_key}' in API response")
        
        reports = data[report_key]
        if not reports:
            raise ValueError(f"Empty {report_key}")
        
        # Limit to specified years
        reports = reports[:years]
        
        # Build records: {date: {field: value}}
        records = {}
        for report in reports:
            date = report.get('fiscalDateEnding', 'Unknown')
            records[date] = {}
            
            for api_field, value in report.items():
                # Skip non-data fields
                if api_field in ['fiscalDateEnding', 'reportedCurrency']:
                    continue
                
                # Map to standard name
                std_name = field_map.get(api_field, api_field)
                
                # Convert to numeric
                if value and value != 'None':
                    try:
                        records[date][std_name] = float(value)
                    except ValueError:
                        records[date][std_name] = None
                else:
                    records[date][std_name] = None
        
        # Create DataFrame (metrics as rows, dates as columns)
        df = pd.DataFrame(records)
        
        # Sort columns by date (newest first)
        df = df.reindex(sorted(df.columns, reverse=True), axis=1)
        
        return df
    
    def _clean_income(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate income statement."""
        # Add EBIT if missing
        if 'EBIT' not in df.index and 'Operating Income' in df.index:
            df.loc['EBIT'] = df.loc['Operating Income']
        
        # Calculate EBITDA if missing
        if 'EBITDA' not in df.index:
            if 'EBIT' in df.index and 'Depreciation And Amortization' in df.index:
                df.loc['EBITDA'] = df.loc['EBIT'] + df.loc['Depreciation And Amortization'].abs()
        
        return df
    
    def _clean_balance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate balance sheet."""
        # Calculate Total Debt if missing
        if 'Total Debt' not in df.index:
            lt_debt = df.loc['Long Term Debt'] if 'Long Term Debt' in df.index else 0
            st_debt = df.loc['Current Debt'] if 'Current Debt' in df.index else 0
            
            # Handle Series/scalar
            if isinstance(lt_debt, pd.Series) or isinstance(st_debt, pd.Series):
                lt_debt = lt_debt if isinstance(lt_debt, pd.Series) else pd.Series([lt_debt] * len(df.columns), index=df.columns)
                st_debt = st_debt if isinstance(st_debt, pd.Series) else pd.Series([st_debt] * len(df.columns), index=df.columns)
                df.loc['Total Debt'] = lt_debt.fillna(0) + st_debt.fillna(0)
        
        return df
    
    def _clean_cashflow(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate cash flow statement."""
        # Calculate Free Cash Flow if missing
        if 'Free Cash Flow' not in df.index:
            if 'Operating Cash Flow' in df.index and 'Capital Expenditure' in df.index:
                ocf = df.loc['Operating Cash Flow'].fillna(0)
                capex = df.loc['Capital Expenditure'].fillna(0).abs()
                df.loc['Free Cash Flow'] = ocf - capex
        
        return df
    
    def collect(self, ticker: str, output_dir: str = None, save: bool = True, years: int = 5) -> dict:
        """
        Collect all financial statements for a company.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL', 'MSFT')
            output_dir: Where to save CSVs (default: {TICKER}_Data/)
            save: Whether to save CSV files
            years: Number of years to fetch (default: 5)
        
        Returns:
            dict with keys: income_stmt, balance_sheet, cash_flow, output_dir
        """
        self.ticker = ticker.upper()
        
        if output_dir is None:
            output_dir = f"{self.ticker}_Data"
        
        print(f"\n  [DataCollector] Fetching {self.ticker} from Alpha Vantage ({years} years)...")
        
        # Fetch income statement
        print(f"    → Income Statement...")
        income_data = self._request('INCOME_STATEMENT', self.ticker)
        income_df = self._parse_reports(income_data, 'annualReports', self.INCOME_FIELDS, years)
        income_df = self._clean_income(income_df)
        
        # Fetch balance sheet
        print(f"    → Balance Sheet...")
        balance_data = self._request('BALANCE_SHEET', self.ticker)
        balance_df = self._parse_reports(balance_data, 'annualReports', self.BALANCE_FIELDS, years)
        balance_df = self._clean_balance(balance_df)
        
        # Fetch cash flow
        print(f"    → Cash Flow Statement...")
        cashflow_data = self._request('CASH_FLOW', self.ticker)
        cashflow_df = self._parse_reports(cashflow_data, 'annualReports', self.CASHFLOW_FIELDS, years)
        cashflow_df = self._clean_cashflow(cashflow_df)
        
        # Save to CSV
        if save:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            income_df.to_csv(f"{output_dir}/{self.ticker}_income_statement.csv")
            balance_df.to_csv(f"{output_dir}/{self.ticker}_balance_sheet.csv")
            cashflow_df.to_csv(f"{output_dir}/{self.ticker}_cash_flow.csv")
            
            print(f"    ✓ Saved {len(income_df.columns)} years to {output_dir}/")
        
        # Store and return
        self.data = {
            'income_stmt': income_df,
            'balance_sheet': balance_df,
            'cash_flow': cashflow_df,
            'output_dir': output_dir,
            'ticker': self.ticker,
            'years': len(income_df.columns)
        }
        
        return self.data
    
    def load(self, ticker: str, data_dir: str = None) -> dict:
        """
        Load existing data from CSV files.
        
        Args:
            ticker: Stock symbol
            data_dir: Directory with CSVs (default: {TICKER}_Data/)
        
        Returns:
            dict with DataFrames, or None if files don't exist
        """
        self.ticker = ticker.upper()
        
        if data_dir is None:
            data_dir = f"{self.ticker}_Data"
        
        files = {
            'income_stmt': Path(data_dir) / f"{self.ticker}_income_statement.csv",
            'balance_sheet': Path(data_dir) / f"{self.ticker}_balance_sheet.csv",
            'cash_flow': Path(data_dir) / f"{self.ticker}_cash_flow.csv"
        }
        
        # Check all files exist
        if not all(f.exists() for f in files.values()):
            return None
        
        print(f"\n  [DataCollector] Loading {self.ticker} from cache...")
        
        self.data = {
            'income_stmt': pd.read_csv(files['income_stmt'], index_col=0),
            'balance_sheet': pd.read_csv(files['balance_sheet'], index_col=0),
            'cash_flow': pd.read_csv(files['cash_flow'], index_col=0),
            'output_dir': data_dir,
            'ticker': self.ticker
        }
        
        self.data['years'] = len(self.data['income_stmt'].columns)
        print(f"    ✓ Loaded {self.data['years']} years from cache")
        
        return self.data
    
    def get(self, ticker: str, refresh: bool = False) -> dict:
        """
        Smart fetch: load from cache if exists, else collect from API.
        
        Args:
            ticker: Stock symbol
            refresh: Force new API call even if cache exists
        
        Returns:
            dict with DataFrames
        """
        self.ticker = ticker.upper()
        
        # Try cache first
        if not refresh:
            cached = self.load(self.ticker)
            if cached:
                return cached
        
        # Fetch from API
        return self.collect(self.ticker)
    
    def print_summary(self):
        """Print summary of collected data."""
        if not self.data:
            print("No data collected yet.")
            return
        
        print(f"\n  {'='*50}")
        print(f"  DATA SUMMARY: {self.ticker}")
        print(f"  {'='*50}")
        print(f"  Years: {self.data['years']}")
        print(f"  Columns: {list(self.data['income_stmt'].columns)}")
        
        inc = self.data['income_stmt']
        bal = self.data['balance_sheet']
        cf = self.data['cash_flow']
        
        print(f"\n  Latest Year Metrics:")
        
        def _val(df, key):
            if key in df.index:
                v = df.loc[key].iloc[0]
                if pd.notna(v):
                    return f"${v/1e9:.2f}B"
            return "N/A"
        
        print(f"    Revenue:           {_val(inc, 'Total Revenue')}")
        print(f"    Gross Profit:      {_val(inc, 'Gross Profit')}")
        print(f"    Net Income:        {_val(inc, 'Net Income')}")
        print(f"    EBITDA:            {_val(inc, 'EBITDA')}")
        print(f"    Total Assets:      {_val(bal, 'Total Assets')}")
        print(f"    Total Liabilities: {_val(bal, 'Total Liabilities Net Minority Interest')}")
        print(f"    Equity:            {_val(bal, 'Stockholders Equity')}")
        print(f"    Total Debt:        {_val(bal, 'Total Debt')}")
        print(f"    Operating CF:      {_val(cf, 'Operating Cash Flow')}")
        print(f"    Free Cash Flow:    {_val(cf, 'Free Cash Flow')}")


# ==================== TEST ====================

if __name__ == "__main__":
    import sys
    
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    print(f"\n{'='*50}")
    print(f"  DATACOLLECTOR TEST: {ticker}")
    print(f"{'='*50}")
    
    try:
        collector = DataCollector()
        data = collector.get(ticker)
        collector.print_summary()
        print(f"\n  ✓ Success!")
        
    except ValueError as e:
        print(f"\n  ✗ Error: {e}")
    except ConnectionError as e:
        print(f"\n  ✗ Connection Error: {e}")