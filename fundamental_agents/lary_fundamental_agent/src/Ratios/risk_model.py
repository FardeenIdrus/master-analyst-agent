"""Risk & Leverage Model - Functions Only"""

import numpy as np
import pandas as pd

# Constants
DEBT_EQUITY_LOW_RISK = 1.0
DEBT_EQUITY_HIGH_RISK = 2.0
INTEREST_COVERAGE_LOW_RISK = 3.0
INTEREST_COVERAGE_HIGH_RISK = 1.5
ALTMAN_SAFE_ZONE = 2.99
ALTMAN_DISTRESS_ZONE = 1.81

ALTMAN_WEIGHTS = {
    'working_capital': 1.2,
    'retained_earnings': 1.4,
    'ebit': 3.3,
    'market_cap': 0.6,
    'sales': 1.0
}


def get_debt_to_equity(balance_sheet: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Calculate Debt-to-Equity Ratio."""
    if isinstance(balance_sheet, pd.DataFrame):
        total_debt = balance_sheet.loc['Total Debt']
        equity = balance_sheet.loc['Stockholders Equity']
        return total_debt / equity
    raise TypeError("Expects pd.DataFrame, no other value.")


def get_debt_to_assets(balance_sheet: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Calculate Debt-to-Assets Ratio."""
    if isinstance(balance_sheet, pd.DataFrame):
        total_debt = balance_sheet.loc['Total Debt']
        total_assets = balance_sheet.loc['Total Assets']
        return total_debt / total_assets
    raise TypeError("Expects pd.DataFrame, no other value.")


def get_equity_multiplier(balance_sheet: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Calculate Equity Multiplier."""
    if isinstance(balance_sheet, pd.DataFrame):
        total_assets = balance_sheet.loc['Total Assets']
        equity = balance_sheet.loc['Stockholders Equity']
        return total_assets / equity
    raise TypeError("Expects pd.DataFrame, no other value.")


def get_interest_coverage(income_statement: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Calculate Interest Coverage Ratio."""
    if isinstance(income_statement, pd.DataFrame):
        ebit = income_statement.loc['EBIT']
        interest_expense = income_statement.loc['Interest Expense']
        return ebit / abs(interest_expense)
    raise TypeError("Expects pd.DataFrame, no other value.")


def get_cash_flow_to_debt(cash_flow: pd.DataFrame, balance_sheet: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Calculate Cash Flow to Debt Ratio."""
    if isinstance(cash_flow, pd.DataFrame) and isinstance(balance_sheet, pd.DataFrame):
        operating_cf = cash_flow.loc['Operating Cash Flow']
        total_debt = balance_sheet.loc['Total Debt']
        return operating_cf / total_debt
    raise TypeError("Expects pd.DataFrame, no other value.")


def get_current_ratio(balance_sheet: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Calculate Current Ratio."""
    if isinstance(balance_sheet, pd.DataFrame):
        current_assets = balance_sheet.loc['Current Assets']
        current_liabilities = balance_sheet.loc['Current Liabilities']
        return current_assets / current_liabilities
    raise TypeError("Expects pd.DataFrame, no other value.")


def get_quick_ratio(balance_sheet: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Calculate Quick Ratio."""
    if isinstance(balance_sheet, pd.DataFrame):
        current_assets = balance_sheet.loc['Current Assets']
        inventory = balance_sheet.loc['Inventory']
        current_liabilities = balance_sheet.loc['Current Liabilities']
        return (current_assets - inventory) / current_liabilities
    raise TypeError("Expects pd.DataFrame, no other value.")


def get_cash_ratio(balance_sheet: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Calculate Cash Ratio."""
    if isinstance(balance_sheet, pd.DataFrame):
        cash = balance_sheet.loc['Cash And Cash Equivalents']
        current_liabilities = balance_sheet.loc['Current Liabilities']
        return cash / current_liabilities
    raise TypeError("Expects pd.DataFrame, no other value.")


def get_gross_margin(income_statement: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Calculate Gross Profit Margin."""
    if isinstance(income_statement, pd.DataFrame):
        revenue = income_statement.loc['Total Revenue']
        gross_profit = income_statement.loc['Gross Profit']
        return (gross_profit / revenue) * 100
    raise TypeError("Expects pd.DataFrame, no other value.")


def get_operating_margin(income_statement: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Calculate Operating Profit Margin."""
    if isinstance(income_statement, pd.DataFrame):
        revenue = income_statement.loc['Total Revenue']
        operating_income = income_statement.loc['Operating Income']
        return (operating_income / revenue) * 100
    raise TypeError("Expects pd.DataFrame, no other value.")


def get_net_margin(income_statement: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Calculate Net Profit Margin."""
    if isinstance(income_statement, pd.DataFrame):
        revenue = income_statement.loc['Total Revenue']
        net_income = income_statement.loc['Net Income']
        return (net_income / revenue) * 100
    raise TypeError("Expects pd.DataFrame, no other value.")


def get_roe(income_statement: pd.DataFrame, balance_sheet: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Calculate Return on Equity."""
    if isinstance(income_statement, pd.DataFrame) and isinstance(balance_sheet, pd.DataFrame):
        net_income = income_statement.loc['Net Income']
        equity = balance_sheet.loc['Stockholders Equity']
        return (net_income / equity) * 100
    raise TypeError("Expects pd.DataFrame, no other value.")


def get_roa(income_statement: pd.DataFrame, balance_sheet: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Calculate Return on Assets."""
    if isinstance(income_statement, pd.DataFrame) and isinstance(balance_sheet, pd.DataFrame):
        net_income = income_statement.loc['Net Income']
        total_assets = balance_sheet.loc['Total Assets']
        return (net_income / total_assets) * 100
    raise TypeError("Expects pd.DataFrame, no other value.")


def get_asset_turnover(income_statement: pd.DataFrame, balance_sheet: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Calculate Asset Turnover Ratio."""
    if isinstance(income_statement, pd.DataFrame) and isinstance(balance_sheet, pd.DataFrame):
        revenue = income_statement.loc['Total Revenue']
        total_assets = balance_sheet.loc['Total Assets']
        return revenue / total_assets
    raise TypeError("Expects pd.DataFrame, no other value.")


def get_revenue_growth(income_statement: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Calculate Revenue Growth Rate."""
    if isinstance(income_statement, pd.DataFrame):
        revenue = income_statement.loc['Total Revenue']
        return revenue.pct_change() * 100
    if isinstance(income_statement, pd.Series):
        return income_statement.pct_change() * 100
    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_altman_z_score(income_statement: pd.DataFrame, balance_sheet: pd.DataFrame, market_cap: float) -> pd.Series | pd.DataFrame:
    """Calculate Altman Z-Score for bankruptcy prediction."""
    if isinstance(income_statement, pd.DataFrame) and isinstance(balance_sheet, pd.DataFrame):
        current_assets = balance_sheet.loc['Current Assets']
        current_liabilities = balance_sheet.loc['Current Liabilities']
        working_capital = current_assets - current_liabilities
        total_assets = balance_sheet.loc['Total Assets']
        x1 = working_capital / total_assets
        
        retained_earnings = balance_sheet.loc['Retained Earnings']
        x2 = retained_earnings / total_assets
        
        ebit = income_statement.loc['EBIT']
        x3 = ebit / total_assets
        
        total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest']
        x4 = market_cap / total_liabilities
        
        sales = income_statement.loc['Total Revenue']
        x5 = sales / total_assets
        
        z_score = (
            ALTMAN_WEIGHTS['working_capital'] * x1 +
            ALTMAN_WEIGHTS['retained_earnings'] * x2 +
            ALTMAN_WEIGHTS['ebit'] * x3 +
            ALTMAN_WEIGHTS['market_cap'] * x4 +
            ALTMAN_WEIGHTS['sales'] * x5
        )
        return z_score
    raise TypeError("Expects pd.DataFrame, no other value.")