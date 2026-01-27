import pandas as pd
from pathlib import Path
from datetime import datetime
import os
import time
from dotenv import load_dotenv
from alpha_vantage.fundamentaldata import FundamentalData

# Load environment variables from master repo root
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent.parent
load_dotenv(REPO_ROOT / ".env")

# Use script-relative paths
RAW_DATA_PATH = SCRIPT_DIR / "data" / "raw"
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)


def get_api_key():
    """Get Alpha Vantage API key."""
    api_key = os.getenv("ALPHA_VANTAGE_KEY_MOHAMED") or os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_KEY_MOHAMED or ALPHA_VANTAGE_API_KEY not found in .env")
    return api_key


def get_financial_statements(ticker_symbol, api_key):
    """
    Pull annual financial statements from Alpha Vantage.
    Returns: (income_stmt, balance_sheet, cash_flow) as DataFrames.
    """
    fd = FundamentalData(key=api_key, output_format='pandas')

    print(f"  Fetching income statement for {ticker_symbol}...")
    income_stmt, _ = fd.get_income_statement_annual(ticker_symbol)

    print("  Waiting 12 seconds (API rate limit)...")
    time.sleep(12)

    print(f"  Fetching balance sheet for {ticker_symbol}...")
    balance_sheet, _ = fd.get_balance_sheet_annual(ticker_symbol)

    print("  Waiting 12 seconds (API rate limit)...")
    time.sleep(12)

    print(f"  Fetching cash flow for {ticker_symbol}...")
    cash_flow, _ = fd.get_cash_flow_annual(ticker_symbol)

    return income_stmt, balance_sheet, cash_flow


def transform_alpha_vantage_to_yfinance_format(df, statement_type):
    """
    Transform Alpha Vantage format to match yfinance format.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    if 'fiscalDateEnding' in df.columns:
        df['year'] = pd.to_datetime(df['fiscalDateEnding']).dt.year
    else:
        raise ValueError("fiscalDateEnding column not found")

    df = df.drop('fiscalDateEnding', axis=1)
    df = df.set_index('year')
    df = df.T
    df = df[sorted(df.columns)]

    if len(df.columns) > 5:
        df = df[df.columns[-5:]]

    df = df.apply(pd.to_numeric, errors='coerce')

    field_mapping = get_field_mapping(statement_type)
    df = df.rename(index=field_mapping)

    return df


def get_field_mapping(statement_type):
    """Map Alpha Vantage field names to yfinance-compatible names."""

    if statement_type == 'income':
        return {
            'totalRevenue': 'Total Revenue',
            'costOfRevenue': 'Cost Of Revenue',
            'grossProfit': 'Gross Profit',
            'operatingIncome': 'EBIT',
            'ebit': 'Operating Income',
            'netIncome': 'Net Income',
            'researchAndDevelopment': 'Research And Development',
            'sellingGeneralAndAdministrative': 'Selling General And Administration',
            'incomeTaxExpense': 'Tax Provision',
            'incomeBeforeTax': 'Pretax Income',
            'operatingExpenses': 'Operating Expense',
            'interestExpense': 'Interest Expense',
            'interestIncome': 'Interest Income',
            'netInterestIncome': 'Net Interest Income',
        }

    elif statement_type == 'balance':
        return {
            'totalAssets': 'Total Assets',
            'totalLiabilities': 'Total Liabilities Net Minority Interest',
            'totalCurrentAssets': 'Total Current Assets',
            'totalCurrentLiabilities': 'Total Current Liabilities',
            'cashAndCashEquivalentsAtCarryingValue': 'Cash And Cash Equivalents',
            'cashAndShortTermInvestments': 'Cash And Short Term Investments',
            'longTermDebt': 'Long Term Debt',
            'longTermDebtNoncurrent': 'Long Term Debt And Capital Lease Obligation',
            'shortTermDebt': 'Short Term Debt',
            'currentDebt': 'Current Debt',
            'currentLongTermDebt': 'Current Debt And Capital Lease Obligation',
            'shortLongTermDebtTotal': 'Total Debt',
            'totalShareholderEquity': 'Total Stockholders Equity',
        }

    elif statement_type == 'cashflow':
        return {
            'operatingCashflow': 'Operating Cash Flow',
            'capitalExpenditures': 'Capital Expenditure',
            'depreciationDepletionAndAmortization': 'Depreciation And Amortization',
            'depreciation': 'Depreciation',
            'dividendPayout': 'Cash Dividends Paid',
            'netIncome': 'Net Income From Continuing Operations',
            'changeInWorkingCapital': 'Change In Working Capital',
        }

    return {}


def collect_data(ticker_symbol: str) -> Path:
    """
    Main function to collect data for a ticker.
    Returns the path to the data directory.
    """
    # Create ticker-specific data folder
    ticker_data_path = RAW_DATA_PATH / ticker_symbol
    ticker_data_path.mkdir(parents=True, exist_ok=True)

    csv_files = [
        ticker_data_path / "income_statement.csv",
        ticker_data_path / "balance_sheet.csv",
        ticker_data_path / "cash_flow.csv",
    ]

    # Check if cached data exists
    if all(f.exists() for f in csv_files):
        print(f"  ✓ Using cached CSV data for {ticker_symbol}")
        return ticker_data_path

    print(f"  Fetching financial data for {ticker_symbol} from Alpha Vantage...")
    print("  Note: This will take ~30 seconds due to API rate limits...")

    api_key = get_api_key()

    # Fetch statements
    income, balance, cashflow = get_financial_statements(ticker_symbol, api_key)

    # Transform to yfinance-compatible format
    print("  Transforming data...")
    income_clean = transform_alpha_vantage_to_yfinance_format(income, 'income')
    balance_clean = transform_alpha_vantage_to_yfinance_format(balance, 'balance')
    cashflow_clean = transform_alpha_vantage_to_yfinance_format(cashflow, 'cashflow')

    # Save
    income_clean.to_csv(ticker_data_path / "income_statement.csv")
    balance_clean.to_csv(ticker_data_path / "balance_sheet.csv")
    cashflow_clean.to_csv(ticker_data_path / "cash_flow.csv")

    # Save metadata
    meta = {
        "ticker": ticker_symbol,
        "download_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "source": "Alpha Vantage",
    }
    pd.DataFrame([meta]).to_csv(ticker_data_path / "metadata.csv", index=False)

    print(f"  ✓ Data collection complete for {ticker_symbol}")
    return ticker_data_path


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"
    collect_data(ticker)
