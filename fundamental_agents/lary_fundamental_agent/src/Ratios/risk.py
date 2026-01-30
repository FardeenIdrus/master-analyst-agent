"""
Risk Analysis Module
Calculates: Altman Z-Score (bankruptcy prediction)
"""

import pandas as pd
import numpy as np


# Altman Z-Score constants
ALTMAN_WEIGHTS = {
    'working_capital': 1.2,
    'retained_earnings': 1.4,
    'ebit': 3.3,
    'market_cap': 0.6,
    'sales': 1.0
}

SAFE_ZONE = 2.99
GREY_ZONE = 1.81


class RiskAnalysis:
    """Analyses company bankruptcy risk using Altman Z-Score."""
    
    def __init__(self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame, market_cap: float):
        self.income_stmt = income_stmt
        self.balance_sheet = balance_sheet
        self.market_cap = market_cap
        self.ratios = {}
    
    def get_altman_z_score(self) -> pd.Series:
        """
        Altman Z-Score = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
        
        X1 = Working Capital / Total Assets
        X2 = Retained Earnings / Total Assets
        X3 = EBIT / Total Assets
        X4 = Market Cap / Total Liabilities
        X5 = Sales / Total Assets
        
        > 2.99 = Safe Zone
        1.81 - 2.99 = Grey Zone
        < 1.81 = Distress Zone
        """
        try:
            total_assets = self.balance_sheet.loc['Total Assets']
            
            # X1: Working Capital / Total Assets
            working_capital = (self.balance_sheet.loc['Current Assets'] - 
                             self.balance_sheet.loc['Current Liabilities'])
            x1 = working_capital / total_assets
            
            # X2: Retained Earnings / Total Assets
            x2 = self.balance_sheet.loc['Retained Earnings'] / total_assets
            
            # X3: EBIT / Total Assets
            x3 = self.income_stmt.loc['EBIT'] / total_assets
            
            # X4: Market Cap / Total Liabilities
            x4 = self.market_cap / self.balance_sheet.loc['Total Liabilities Net Minority Interest']
            
            # X5: Sales / Total Assets
            x5 = self.income_stmt.loc['Total Revenue'] / total_assets
            
            # Calculate Z-Score
            z_score = (
                ALTMAN_WEIGHTS['working_capital'] * x1 +
                ALTMAN_WEIGHTS['retained_earnings'] * x2 +
                ALTMAN_WEIGHTS['ebit'] * x3 +
                ALTMAN_WEIGHTS['market_cap'] * x4 +
                ALTMAN_WEIGHTS['sales'] * x5
            )
            
            self.ratios['altman_z_score'] = z_score
            return z_score
            
        except KeyError as e:
            print(f"  Error calculating Altman Z-Score: {e}")
            return None
    
    def calculate_all(self) -> dict:
        """Calculate all risk ratios."""
        self.get_altman_z_score()
        return self.ratios
    
    def get_interpretation(self, z_score: float) -> str:
        """Interpret Z-Score value."""
        if z_score > SAFE_ZONE:
            return "Safe Zone - Low bankruptcy risk"
        elif z_score > GREY_ZONE:
            return "Grey Zone - Moderate risk"
        else:
            return "Distress Zone - High bankruptcy risk"
    
    def get_latest_values(self) -> dict:
        """Get most recent year's values."""
        return {name: series.iloc[0] for name, series in self.ratios.items() if series is not None}
    
    def print_summary(self):
        """Print formatted summary."""
        if not self.ratios:
            print("No ratios calculated. Run calculate_all() first.")
            return
        
        print("\n" + "=" * 67)
        print("RISK ANALYSIS (Altman Z-Score)")
        print("=" * 67)
        
        years = list(self.ratios.values())[0].index.tolist()
        
        header = f"{'Metric':<25}"
        for year in years[:5]:
            header += f"{str(year)[:4]:>12}"
        print(header)
        print("-" * 67)
        
        if 'altman_z_score' in self.ratios and self.ratios['altman_z_score'] is not None:
            row = f"{'Altman Z-Score':<25}"
            for i in range(min(5, len(years))):
                val = self.ratios['altman_z_score'].iloc[i]
                row += f"{val:>12.2f}" if pd.notna(val) else f"{'N/A':>12}"
            print(row)
        
        print("-" * 67)
        
        # Interpretation
        latest = self.get_latest_values()
        if 'altman_z_score' in latest:
            z = latest['altman_z_score']
            print(f"\nLatest Z-Score: {z:.2f}")
            print(f"Interpretation: {self.get_interpretation(z)}")
            print(f"\nBenchmarks: >2.99 Safe | 1.81-2.99 Grey | <1.81 Distress")
    
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
        print("Testing Risk Analysis...\n")
        
        income_stmt = pd.read_csv(f"{data_dir}AAPL_income_statement.csv", index_col=0)
        balance_sheet = pd.read_csv(f"{data_dir}AAPL_balance_sheet.csv", index_col=0)
        
        # Apple's approximate market cap (you can update this)
        APPLE_MARKET_CAP = 3_500_000_000_000  # $3.5 trillion
        
        analysis = RiskAnalysis(income_stmt, balance_sheet, APPLE_MARKET_CAP)
        analysis.calculate_all()
        analysis.print_summary()
        
        print("\n Test passed!")
    else:
        print("No test data found. Place AAPL CSVs in APPL_Data/ folder.")