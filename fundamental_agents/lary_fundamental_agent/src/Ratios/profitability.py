"""
Profitability Ratios Module
Calculates: Gross Margin, Operating Margin, Net Margin, ROE, ROA, ROCE, EBITDA Margin
"""

import pandas as pd
import numpy as np


class ProfitabilityAnalysis:
    """Analyses company profitability using key financial ratios."""
    
    def __init__(self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame):
        self.income_stmt = income_stmt
        self.balance_sheet = balance_sheet
        self.ratios = {}
    
    def get_gross_margin(self) -> pd.Series:
        """(Gross Profit / Total Revenue) × 100"""
        try:
            result = (self.income_stmt.loc['Gross Profit'] / self.income_stmt.loc['Total Revenue']) * 100
            self.ratios['gross_margin'] = result
            return result
        except KeyError as e:
            print(f"  Error calculating gross margin: {e}")
            return None
    
    def get_operating_margin(self) -> pd.Series:
        """(Operating Income / Total Revenue) × 100"""
        try:
            result = (self.income_stmt.loc['Operating Income'] / self.income_stmt.loc['Total Revenue']) * 100
            self.ratios['operating_margin'] = result
            return result
        except KeyError as e:
            print(f"  Error calculating operating margin: {e}")
            return None
    
    def get_net_margin(self) -> pd.Series:
        """(Net Income / Total Revenue) × 100"""
        try:
            result = (self.income_stmt.loc['Net Income'] / self.income_stmt.loc['Total Revenue']) * 100
            self.ratios['net_margin'] = result
            return result
        except KeyError as e:
            print(f"  Error calculating net margin: {e}")
            return None
    
    def get_ebitda_margin(self) -> pd.Series:
        """(EBITDA / Total Revenue) × 100 - Measures cash profitability"""
        try:
            result = (self.income_stmt.loc['EBITDA'] / self.income_stmt.loc['Total Revenue']) * 100
            self.ratios['ebitda_margin'] = result
            return result
        except KeyError as e:
            print(f"  Error calculating EBITDA margin: {e}")
            return None
    
    def get_roe(self) -> pd.Series:
        """(Net Income / Stockholders Equity) × 100"""
        try:
            result = (self.income_stmt.loc['Net Income'] / self.balance_sheet.loc['Stockholders Equity']) * 100
            self.ratios['roe'] = result
            return result
        except KeyError as e:
            print(f"  Error calculating ROE: {e}")
            return None
    
    def get_roa(self) -> pd.Series:
        """(Net Income / Total Assets) × 100"""
        try:
            result = (self.income_stmt.loc['Net Income'] / self.balance_sheet.loc['Total Assets']) * 100
            self.ratios['roa'] = result
            return result
        except KeyError as e:
            print(f"  Error calculating ROA: {e}")
            return None
    
    def get_roce(self) -> pd.Series:
        """
        Return on Capital Employed (ROCE)
        Formula: (EBIT / Capital Employed) × 100
        Capital Employed = Total Assets - Current Liabilities
        Measures efficiency of capital usage
        """
        try:
            ebit = self.income_stmt.loc['EBIT']
            total_assets = self.balance_sheet.loc['Total Assets']
            current_liabilities = self.balance_sheet.loc['Current Liabilities']
            
            capital_employed = total_assets - current_liabilities
            result = (ebit / capital_employed) * 100
            self.ratios['roce'] = result
            return result
        except KeyError as e:
            print(f"  Error calculating ROCE: {e}")
            return None
    
    def get_roic(self) -> pd.Series:
        """
        Return on Invested Capital (ROIC)
        Formula: (EBIT × (1 - Tax Rate)) / Invested Capital
        Invested Capital = Total Debt + Stockholders Equity - Cash
        """
        try:
            ebit = self.income_stmt.loc['EBIT']
            
            # Calculate effective tax rate
            tax = self.income_stmt.loc['Tax Provision']
            pretax = self.income_stmt.loc['Pretax Income']
            tax_rate = tax / pretax
            
            # NOPAT (Net Operating Profit After Tax)
            nopat = ebit * (1 - tax_rate)
            
            # Invested Capital
            total_debt = self.balance_sheet.loc['Total Debt']
            equity = self.balance_sheet.loc['Stockholders Equity']
            cash = self.balance_sheet.loc['Cash And Cash Equivalents']
            invested_capital = total_debt + equity - cash
            
            result = (nopat / invested_capital) * 100
            self.ratios['roic'] = result
            return result
        except KeyError as e:
            print(f"  Error calculating ROIC: {e}")
            return None
    
    def calculate_all(self) -> dict:
        """Calculate all profitability ratios."""
        print("Calculating profitability ratios...")
        
        self.get_gross_margin()
        self.get_operating_margin()
        self.get_net_margin()
        self.get_ebitda_margin()
        self.get_roe()
        self.get_roa()
        self.get_roce()
        self.get_roic()
        
        print(f"  Completed: {len(self.ratios)} ratios calculated")
        return self.ratios
    
    def get_latest_values(self) -> dict:
        """Get most recent year's values."""
        latest = {}
        for name, series in self.ratios.items():
            if series is not None and len(series) > 0:
                latest[name] = series.iloc[0]
        return latest
    
    def print_summary(self):
        """Print formatted summary."""
        if not self.ratios:
            print("No ratios calculated. Run calculate_all() first.")
            return
        
        print("\n" + "=" * 67)
        print("PROFITABILITY ANALYSIS")
        print("=" * 67)
        
        # Get years
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
            'gross_margin': 'Gross Margin (%)',
            'operating_margin': 'Operating Margin (%)',
            'ebitda_margin': 'EBITDA Margin (%)',
            'net_margin': 'Net Margin (%)',
            'roe': 'Return on Equity (%)',
            'roa': 'Return on Assets (%)',
            'roce': 'ROCE (%)',
            'roic': 'ROIC (%)'
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
    data_dir = next((p for p in paths if os.path.exists(f"{p}AAPL_income_statement.csv")), None)
    
    if data_dir is not None:
        print("Testing Profitability Analysis...\n")
        
        income_stmt = pd.read_csv(f"{data_dir}AAPL_income_statement.csv", index_col=0)
        balance_sheet = pd.read_csv(f"{data_dir}AAPL_balance_sheet.csv", index_col=0)
        
        analysis = ProfitabilityAnalysis(income_stmt, balance_sheet)
        analysis.calculate_all()
        analysis.print_summary()
        
        print("\n✅ Test passed!")
    else:
        print("No test data found. Place AAPL CSVs in APPL_Data/ folder.")