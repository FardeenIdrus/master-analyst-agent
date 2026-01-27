"""
Growth Ratios Module
Calculates: Revenue Growth, Net Income Growth
"""

import pandas as pd
import numpy as np


class GrowthAnalysis:
    """Analyses company growth trends."""
    
    def __init__(self, income_stmt: pd.DataFrame):
        self.income_stmt = income_stmt
        self.ratios = {}
    
    def get_revenue_growth(self) -> pd.Series:
        """Year-over-year revenue growth (%)"""
        try:
            revenue = self.income_stmt.loc['Total Revenue']
            result = revenue.pct_change(periods=-1) * 100  # Negative because newest is first
            self.ratios['revenue_growth'] = result
            return result
        except KeyError as e:
            print(f"  Error calculating revenue growth: {e}")
            return None
    
    def get_net_income_growth(self) -> pd.Series:
        """Year-over-year net income growth (%)"""
        try:
            net_income = self.income_stmt.loc['Net Income']
            result = net_income.pct_change(periods=-1) * 100
            self.ratios['net_income_growth'] = result
            return result
        except KeyError as e:
            print(f"  Error calculating net income growth: {e}")
            return None
    
    def calculate_all(self) -> dict:
        """Calculate all growth ratios."""
        self.get_revenue_growth()
        self.get_net_income_growth()
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
        print("GROWTH ANALYSIS")
        print("=" * 67)
        
        years = list(self.ratios.values())[0].index.tolist()
        
        header = f"{'Metric':<25}"
        for year in years[:5]:
            header += f"{str(year)[:4]:>12}"
        print(header)
        print("-" * 67)
        
        display_names = {
            'revenue_growth': 'Revenue Growth (%)',
            'net_income_growth': 'Net Income Growth (%)'
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
        print("Testing Growth Analysis...\n")
        
        income_stmt = pd.read_csv(f"{data_dir}AAPL_income_statement.csv", index_col=0)
        
        analysis = GrowthAnalysis(income_stmt)
        analysis.calculate_all()
        analysis.print_summary()
        
        print("\n Test passed!")
    else:
        print("No test data found. Place AAPL CSVs in APPL_Data/ folder.")