"""
Leverage Ratios Module
Calculates: Debt-to-Equity, Debt-to-Assets, Equity Multiplier, Interest Coverage,
            Cash Flow to Debt, Net Debt/EBITDA, Debt Service Coverage, Long-term Debt Ratio
"""

import pandas as pd
import numpy as np


class LeverageAnalysis:
    """Analyses company leverage and solvency."""
    
    def __init__(self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame, cash_flow: pd.DataFrame = None):
        self.income_stmt = income_stmt
        self.balance_sheet = balance_sheet
        self.cash_flow = cash_flow
        self.ratios = {}
    
    def get_debt_to_equity(self) -> pd.Series:
        """Total Debt / Stockholders Equity"""
        try:
            result = self.balance_sheet.loc['Total Debt'] / self.balance_sheet.loc['Stockholders Equity']
            self.ratios['debt_to_equity'] = result
            return result
        except KeyError as e:
            print(f"  Error calculating debt-to-equity: {e}")
            return None
    
    def get_debt_to_assets(self) -> pd.Series:
        """Total Debt / Total Assets"""
        try:
            result = self.balance_sheet.loc['Total Debt'] / self.balance_sheet.loc['Total Assets']
            self.ratios['debt_to_assets'] = result
            return result
        except KeyError as e:
            print(f"  Error calculating debt-to-assets: {e}")
            return None
    
    def get_equity_multiplier(self) -> pd.Series:
        """Total Assets / Stockholders Equity"""
        try:
            result = self.balance_sheet.loc['Total Assets'] / self.balance_sheet.loc['Stockholders Equity']
            self.ratios['equity_multiplier'] = result
            return result
        except KeyError as e:
            print(f"  Error calculating equity multiplier: {e}")
            return None
    
    def get_interest_coverage(self) -> pd.Series:
        """EBIT / Interest Expense - Ability to pay interest"""
        try:
            interest = abs(self.income_stmt.loc['Interest Expense'])
            # Avoid division by zero
            interest = interest.replace(0, np.nan)
            result = self.income_stmt.loc['EBIT'] / interest
            self.ratios['interest_coverage'] = result
            return result
        except KeyError as e:
            print(f"  Error calculating interest coverage: {e}")
            return None
    
    def get_cash_flow_to_debt(self) -> pd.Series:
        """Operating Cash Flow / Total Debt"""
        if self.cash_flow is None:
            return None
        try:
            result = self.cash_flow.loc['Operating Cash Flow'] / self.balance_sheet.loc['Total Debt']
            self.ratios['cash_flow_to_debt'] = result
            return result
        except KeyError as e:
            print(f"  Error calculating cash flow to debt: {e}")
            return None
    
    def get_net_debt_to_ebitda(self) -> pd.Series:
        """
        (Total Debt - Cash) / EBITDA
        Key metric for credit analysis - measures years to repay debt
        < 2x is healthy, > 4x is concerning
        """
        try:
            total_debt = self.balance_sheet.loc['Total Debt']
            cash = self.balance_sheet.loc['Cash And Cash Equivalents']
            ebitda = self.income_stmt.loc['EBITDA']
            
            net_debt = total_debt - cash
            # Avoid division by zero
            ebitda = ebitda.replace(0, np.nan)
            result = net_debt / ebitda
            self.ratios['net_debt_to_ebitda'] = result
            return result
        except KeyError as e:
            print(f"  Error calculating Net Debt/EBITDA: {e}")
            return None
    
    def get_debt_service_coverage(self) -> pd.Series:
        """
        Operating Cash Flow / (Interest + Principal Payments)
        Measures ability to service all debt obligations
        > 1.25x is generally acceptable
        """
        if self.cash_flow is None:
            return None
        try:
            ocf = self.cash_flow.loc['Operating Cash Flow']
            interest = abs(self.income_stmt.loc['Interest Expense'])
            
            # Use current debt as proxy for principal payments
            current_debt = self.balance_sheet.loc['Current Debt']
            
            debt_service = interest + current_debt
            # Avoid division by zero
            debt_service = debt_service.replace(0, np.nan)
            result = ocf / debt_service
            self.ratios['debt_service_coverage'] = result
            return result
        except KeyError as e:
            print(f"  Error calculating debt service coverage: {e}")
            return None
    
    def get_long_term_debt_ratio(self) -> pd.Series:
        """
        Long Term Debt / Total Assets
        Measures long-term financial leverage
        """
        try:
            result = self.balance_sheet.loc['Long Term Debt'] / self.balance_sheet.loc['Total Assets']
            self.ratios['long_term_debt_ratio'] = result
            return result
        except KeyError as e:
            print(f"  Error calculating long-term debt ratio: {e}")
            return None
    
    def get_financial_leverage(self) -> pd.Series:
        """
        Total Liabilities / Stockholders Equity
        Broader measure than debt-to-equity (includes all liabilities)
        """
        try:
            result = self.balance_sheet.loc['Total Liabilities Net Minority Interest'] / self.balance_sheet.loc['Stockholders Equity']
            self.ratios['financial_leverage'] = result
            return result
        except KeyError as e:
            print(f"  Error calculating financial leverage: {e}")
            return None
    
    def calculate_all(self) -> dict:
        """Calculate all leverage ratios."""
        self.get_debt_to_equity()
        self.get_debt_to_assets()
        self.get_equity_multiplier()
        self.get_interest_coverage()
        self.get_cash_flow_to_debt()
        self.get_net_debt_to_ebitda()
        self.get_debt_service_coverage()
        self.get_long_term_debt_ratio()
        self.get_financial_leverage()
        return self.ratios
    
    def get_latest_values(self) -> dict:
        """Get most recent year's values."""
        return {name: series.iloc[0] for name, series in self.ratios.items() if series is not None}
    
    def print_summary(self):
        """Print formatted summary."""
        if not self.ratios:
            print("No ratios calculated. Run calculate_all() first.")
            return
        
        print("\n" + "=" * 67)
        print("LEVERAGE ANALYSIS")
        print("=" * 67)
        
        # Get years from first available ratio
        first_ratio = next((v for v in self.ratios.values() if v is not None), None)
        if first_ratio is None:
            print("No data available")
            return
        
        years = first_ratio.index.tolist()
        
        header = f"{'Metric':<25}"
        for year in years[:5]:
            header += f"{str(year)[:4]:>12}"
        print(header)
        print("-" * 67)
        
        display_names = {
            'debt_to_equity': 'Debt-to-Equity',
            'debt_to_assets': 'Debt-to-Assets',
            'equity_multiplier': 'Equity Multiplier',
            'financial_leverage': 'Financial Leverage',
            'long_term_debt_ratio': 'LT Debt Ratio',
            'interest_coverage': 'Interest Coverage',
            'debt_service_coverage': 'Debt Service Cover',
            'cash_flow_to_debt': 'Cash Flow to Debt',
            'net_debt_to_ebitda': 'Net Debt/EBITDA'
        }
        
        for key, name in display_names.items():
            if key in self.ratios and self.ratios[key] is not None:
                row = f"{name:<25}"
                for i in range(min(5, len(years))):
                    val = self.ratios[key].iloc[i]
                    row += f"{val:>12.2f}" if pd.notna(val) else f"{'N/A':>12}"
                print(row)
        
        print("-" * 67)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert ratios to DataFrame."""
        valid = {k: v for k, v in self.ratios.items() if v is not None}
        return pd.DataFrame(valid).T


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import os
    
    paths = ["../APPL_Data/", "APPL_Data/", ""]
    data_dir = next((p for p in paths if os.path.exists(f"{p}AAPL_balance_sheet.csv")), None)
    
    if data_dir is not None:
        print("Testing Leverage Analysis...\n")
        
        income_stmt = pd.read_csv(f"{data_dir}AAPL_income_statement.csv", index_col=0)
        balance_sheet = pd.read_csv(f"{data_dir}AAPL_balance_sheet.csv", index_col=0)
        cash_flow = pd.read_csv(f"{data_dir}AAPL_cash_flow.csv", index_col=0)
        
        analysis = LeverageAnalysis(income_stmt, balance_sheet, cash_flow)
        analysis.calculate_all()
        analysis.print_summary()
        
        print("\n✅ Test passed!")
    else:
        print("No test data found. Place AAPL CSVs in APPL_Data/ folder.")