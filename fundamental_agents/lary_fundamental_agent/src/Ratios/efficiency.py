"""
Efficiency Ratios Module
Calculates: Asset Turnover, Receivables Turnover, Payables Turnover, Inventory Turnover
"""

import pandas as pd
import numpy as np


class EfficiencyAnalysis:
    """Analyses how efficiently a company uses its assets."""
    
    def __init__(self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame):
        self.income_stmt = income_stmt
        self.balance_sheet = balance_sheet
        self.ratios = {}
    
    def get_asset_turnover(self) -> pd.Series:
        """Total Revenue / Total Assets - Measures how efficiently assets generate revenue"""
        try:
            result = self.income_stmt.loc['Total Revenue'] / self.balance_sheet.loc['Total Assets']
            self.ratios['asset_turnover'] = result
            return result
        except KeyError as e:
            print(f"  Error calculating asset turnover: {e}")
            return None
    
    def get_receivables_turnover(self) -> pd.Series:
        """Total Revenue / Accounts Receivable - How quickly company collects payments"""
        try:
            result = self.income_stmt.loc['Total Revenue'] / self.balance_sheet.loc['Accounts Receivable']
            self.ratios['receivables_turnover'] = result
            return result
        except KeyError as e:
            print(f"  Error calculating receivables turnover: {e}")
            return None
    
    def get_days_sales_outstanding(self) -> pd.Series:
        """365 / Receivables Turnover - Average days to collect payment"""
        try:
            rec_turnover = self.ratios.get('receivables_turnover')
            if rec_turnover is None:
                self.get_receivables_turnover()
                rec_turnover = self.ratios.get('receivables_turnover')
            
            if rec_turnover is not None:
                result = 365 / rec_turnover
                self.ratios['days_sales_outstanding'] = result
                return result
            return None
        except Exception as e:
            print(f"  Error calculating DSO: {e}")
            return None
    
    def get_payables_turnover(self) -> pd.Series:
        """Cost of Revenue / Accounts Payable - How quickly company pays suppliers"""
        try:
            # Try different field names for accounts payable
            payables = None
            for field in ['Accounts Payable', 'currentAccountsPayable', 'accountPayables']:
                if field in self.balance_sheet.index:
                    payables = self.balance_sheet.loc[field]
                    break
            
            if payables is None:
                print(f"  Error calculating payables turnover: 'Accounts Payable' not found")
                return None
            
            result = self.income_stmt.loc['Cost Of Revenue'] / payables
            self.ratios['payables_turnover'] = result
            return result
        except KeyError as e:
            print(f"  Error calculating payables turnover: {e}")
            return None
    
    def get_days_payables_outstanding(self) -> pd.Series:
        """365 / Payables Turnover - Average days to pay suppliers"""
        try:
            pay_turnover = self.ratios.get('payables_turnover')
            if pay_turnover is None:
                self.get_payables_turnover()
                pay_turnover = self.ratios.get('payables_turnover')
            
            if pay_turnover is not None:
                result = 365 / pay_turnover
                self.ratios['days_payables_outstanding'] = result
                return result
            return None
        except Exception as e:
            print(f"  Error calculating DPO: {e}")
            return None
    
    def get_inventory_turnover(self) -> pd.Series:
        """Cost of Revenue / Inventory - How quickly inventory is sold"""
        try:
            inventory = self.balance_sheet.loc['Inventory']
            # Handle companies with zero or no inventory
            if (inventory == 0).all() or inventory.isna().all():
                print(f"  Note: Company has minimal/no inventory")
                return None
            
            result = self.income_stmt.loc['Cost Of Revenue'] / inventory
            self.ratios['inventory_turnover'] = result
            return result
        except KeyError as e:
            print(f"  Error calculating inventory turnover: {e}")
            return None
    
    def get_days_inventory_outstanding(self) -> pd.Series:
        """365 / Inventory Turnover - Average days to sell inventory"""
        try:
            inv_turnover = self.ratios.get('inventory_turnover')
            if inv_turnover is None:
                self.get_inventory_turnover()
                inv_turnover = self.ratios.get('inventory_turnover')
            
            if inv_turnover is not None:
                result = 365 / inv_turnover
                self.ratios['days_inventory_outstanding'] = result
                return result
            return None
        except Exception as e:
            print(f"  Error calculating DIO: {e}")
            return None
    
    def get_cash_conversion_cycle(self) -> pd.Series:
        """DSO + DIO - DPO - Days to convert inventory to cash"""
        try:
            dso = self.ratios.get('days_sales_outstanding')
            dio = self.ratios.get('days_inventory_outstanding')
            dpo = self.ratios.get('days_payables_outstanding')
            
            if dso is not None and dpo is not None:
                if dio is not None:
                    result = dso + dio - dpo
                else:
                    result = dso - dpo  # For companies without inventory
                self.ratios['cash_conversion_cycle'] = result
                return result
            return None
        except Exception as e:
            print(f"  Error calculating CCC: {e}")
            return None
    
    def calculate_all(self) -> dict:
        """Calculate all efficiency ratios."""
        self.get_asset_turnover()
        self.get_receivables_turnover()
        self.get_days_sales_outstanding()
        self.get_payables_turnover()
        self.get_days_payables_outstanding()
        self.get_inventory_turnover()
        self.get_days_inventory_outstanding()
        self.get_cash_conversion_cycle()
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
        print("EFFICIENCY ANALYSIS")
        print("=" * 67)
        
        # Get years from first available ratio
        first_ratio = next((v for v in self.ratios.values() if v is not None), None)
        if first_ratio is None:
            print("No ratios available.")
            return
        
        years = first_ratio.index.tolist()
        
        header = f"{'Metric':<25}"
        for year in years[:5]:
            header += f"{str(year)[:4]:>12}"
        print(header)
        print("-" * 67)
        
        display_names = {
            'asset_turnover': 'Asset Turnover',
            'receivables_turnover': 'Receivables Turnover',
            'days_sales_outstanding': 'Days Sales Out (DSO)',
            'payables_turnover': 'Payables Turnover',
            'days_payables_outstanding': 'Days Payables Out (DPO)',
            'inventory_turnover': 'Inventory Turnover',
            'days_inventory_outstanding': 'Days Inventory (DIO)',
            'cash_conversion_cycle': 'Cash Conv. Cycle (days)'
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
        print("Testing Efficiency Analysis...\n")
        
        income_stmt = pd.read_csv(f"{data_dir}AAPL_income_statement.csv", index_col=0)
        balance_sheet = pd.read_csv(f"{data_dir}AAPL_balance_sheet.csv", index_col=0)
        
        analysis = EfficiencyAnalysis(income_stmt, balance_sheet)
        analysis.calculate_all()
        analysis.print_summary()
        
        print("\n✅ Test passed!")
    else:
        print("No test data found. Place AAPL CSVs in APPL_Data/ folder.")