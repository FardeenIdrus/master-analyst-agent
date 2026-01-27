"""
Liquidity Ratios Module
=======================
Calculates liquidity metrics to assess a company's ability
to meet short-term obligations with its current assets.

Ratios Included:
    - Current Ratio
    - Quick Ratio
    - Cash Ratio

"""

import pandas as pd
import numpy as np


class LiquidityAnalysis:
    """
    Analyses company liquidity using key financial ratios.
    
    Liquidity ratios measure a company's ability to pay off
    short-term debts using its most liquid assets.
    
    Attributes:
        balance_sheet (DataFrame): Balance sheet data
        ratios (dict): Calculated liquidity ratios
    
    Example:
        >>> liquidity = LiquidityAnalysis(balance_sheet)
        >>> results = liquidity.calculate_all()
        >>> liquidity.print_summary()
    """
    
    def __init__(self, balance_sheet: pd.DataFrame):
        """
        Initialise with balance sheet data.
        
        Args:
            balance_sheet: Balance sheet DataFrame with years as columns
        """
        self.balance_sheet = balance_sheet
        self.ratios = {}
    
    # =========================================================================
    # Individual Ratio Calculations
    # =========================================================================
    
    def get_current_ratio(self) -> pd.Series:
        """
        Calculate Current Ratio.
        
        Formula: Current Assets / Current Liabilities
        
        Interpretation:
            - Measures basic liquidity
            - > 1.0 means company can cover short-term debts
            - Benchmark: 1.5 - 2.0 is generally healthy
            - Too high may indicate inefficient asset use
        
        Returns:
            Series of current ratios by year
        """
        try:
            current_assets = self.balance_sheet.loc['Current Assets']
            current_liabilities = self.balance_sheet.loc['Current Liabilities']
            
            current_ratio = current_assets / current_liabilities
            self.ratios['current_ratio'] = current_ratio
            return current_ratio
            
        except KeyError as e:
            print(f"  Error: Missing data field for current ratio - {e}")
            return None
    
    def get_quick_ratio(self) -> pd.Series:
        """
        Calculate Quick Ratio (Acid-Test Ratio).
        
        Formula: (Current Assets - Inventory) / Current Liabilities
        
        Interpretation:
            - Stricter liquidity test than current ratio
            - Excludes inventory (harder to liquidate quickly)
            - Benchmark: > 1.0 is generally good
            - Important for companies with slow-moving inventory
        
        Returns:
            Series of quick ratios by year
        """
        try:
            current_assets = self.balance_sheet.loc['Current Assets']
            inventory = self.balance_sheet.loc['Inventory']
            current_liabilities = self.balance_sheet.loc['Current Liabilities']
            
            quick_ratio = (current_assets - inventory) / current_liabilities
            self.ratios['quick_ratio'] = quick_ratio
            return quick_ratio
            
        except KeyError as e:
            print(f"  Error: Missing data field for quick ratio - {e}")
            return None
    
    def get_cash_ratio(self) -> pd.Series:
        """
        Calculate Cash Ratio.
        
        Formula: Cash and Cash Equivalents / Current Liabilities
        
        Interpretation:
            - Most conservative liquidity measure
            - Shows ability to pay debts with cash only
            - Benchmark: 0.2 - 0.5 is typical
            - Too high may indicate excess idle cash
        
        Returns:
            Series of cash ratios by year
        """
        try:
            cash = self.balance_sheet.loc['Cash And Cash Equivalents']
            current_liabilities = self.balance_sheet.loc['Current Liabilities']
            
            cash_ratio = cash / current_liabilities
            self.ratios['cash_ratio'] = cash_ratio
            return cash_ratio
            
        except KeyError as e:
            print(f"  Error: Missing data field for cash ratio - {e}")
            return None
    
    # =========================================================================
    # Aggregate Functions
    # =========================================================================
    
    def calculate_all(self) -> dict:
        """
        Calculate all liquidity ratios.
        
        Returns:
            Dictionary containing all liquidity metrics
        """
        print("Calculating liquidity ratios...")
        
        self.get_current_ratio()
        self.get_quick_ratio()
        self.get_cash_ratio()
        
        print(f"  Completed: {len(self.ratios)} ratios calculated")
        return self.ratios
    
    def get_latest_values(self) -> dict:
        """
        Get the most recent year's values for all ratios.
        
        Returns:
            Dictionary with ratio names and latest values
        """
        latest = {}
        for name, series in self.ratios.items():
            if series is not None and len(series) > 0:
                latest[name] = series.iloc[0]  # First column is most recent
        return latest
    
    def print_summary(self):
        """Print a formatted summary of liquidity ratios."""
        if not self.ratios:
            print("No ratios calculated. Run calculate_all() first.")
            return
        
        print("\n" + "=" * 55)
        print("LIQUIDITY ANALYSIS")
        print("=" * 55)
        
        # Get years from first available ratio
        years = None
        for ratio in self.ratios.values():
            if ratio is not None:
                years = ratio.index.tolist()
                break
        
        if years is None:
            print("No data available")
            return
        
        # Print header
        header = f"{'Metric':<25}"
        for year in years[:4]:  # Show up to 4 years
            year_str = str(year)[:4] if len(str(year)) >= 4 else str(year)
            header += f"{year_str:>12}"
        print(header)
        print("-" * 55)
        
        # Print each ratio
        ratio_display_names = {
            'current_ratio': 'Current Ratio',
            'quick_ratio': 'Quick Ratio',
            'cash_ratio': 'Cash Ratio'
        }
        
        for key, display_name in ratio_display_names.items():
            if key in self.ratios and self.ratios[key] is not None:
                row = f"{display_name:<25}"
                for i, year in enumerate(years[:4]):
                    value = self.ratios[key].iloc[i]
                    row += f"{value:>12.2f}"
                print(row)
            else:
                print(f"{display_name:<25}{'N/A':>12}")
        
        print("-" * 55)
        
        # Add interpretation for latest values
        self._print_interpretation()
    
    def _print_interpretation(self):
        """Print interpretation of the latest liquidity ratios."""
        latest = self.get_latest_values()
        
        if not latest:
            return
        
        print("\nInterpretation (Latest Year):")
        
        if 'current_ratio' in latest:
            cr = latest['current_ratio']
            if cr >= 1.5:
                status = "Healthy"
            elif cr >= 1.0:
                status = "Adequate"
            else:
                status = "Concern"
            print(f"  Current Ratio: {cr:.2f} - {status}")
        
        if 'quick_ratio' in latest:
            qr = latest['quick_ratio']
            if qr >= 1.0:
                status = "Strong"
            elif qr >= 0.5:
                status = "Acceptable"
            else:
                status = "Weak"
            print(f"  Quick Ratio: {qr:.2f} - {status}")
        
        if 'cash_ratio' in latest:
            cash = latest['cash_ratio']
            if cash >= 0.5:
                status = "Very Liquid"
            elif cash >= 0.2:
                status = "Adequate"
            else:
                status = "Low Cash"
            print(f"  Cash Ratio: {cash:.2f} - {status}")
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all ratios to a single DataFrame.
        
        Returns:
            DataFrame with ratios as rows and years as columns
        """
        if not self.ratios:
            return pd.DataFrame()
        
        valid_ratios = {k: v for k, v in self.ratios.items() if v is not None}
        return pd.DataFrame(valid_ratios).T


# =============================================================================
# Standalone Functions (for direct use without class)
# =============================================================================

def calculate_current_ratio(balance_sheet: pd.DataFrame) -> pd.Series:
    """Calculate Current Ratio from balance sheet."""
    current_assets = balance_sheet.loc['Current Assets']
    current_liabilities = balance_sheet.loc['Current Liabilities']
    return current_assets / current_liabilities


def calculate_quick_ratio(balance_sheet: pd.DataFrame) -> pd.Series:
    """Calculate Quick Ratio from balance sheet."""
    current_assets = balance_sheet.loc['Current Assets']
    inventory = balance_sheet.loc['Inventory']
    current_liabilities = balance_sheet.loc['Current Liabilities']
    return (current_assets - inventory) / current_liabilities


def calculate_cash_ratio(balance_sheet: pd.DataFrame) -> pd.Series:
    """Calculate Cash Ratio from balance sheet."""
    cash = balance_sheet.loc['Cash And Cash Equivalents']
    current_liabilities = balance_sheet.loc['Current Liabilities']
    return cash / current_liabilities


# =============================================================================
# Main - Test the module
# =============================================================================

if __name__ == "__main__":
    import os
    
    # Check multiple possible data locations
    possible_paths = [
        "../APPL_Data/AAPL_balance_sheet.csv",  # Running from Ratios folder
        "APPL_Data/AAPL_balance_sheet.csv",     # Running from project root
        "AAPL_balance_sheet.csv",                # Data in same folder
    ]
    
    test_file = None
    for path in possible_paths:
        if os.path.exists(path):
            test_file = path
            break
    
    if test_file:
        print(f"Testing Liquidity Analysis with Apple data...")
        print(f"Data found at: {test_file}\n")
        
        # Load data
        balance_sheet = pd.read_csv(test_file, index_col=0)
        
        # Run analysis
        analysis = LiquidityAnalysis(balance_sheet)
        results = analysis.calculate_all()
        analysis.print_summary()
        
        print("\n Liquidity module test passed!")
    else:
        print("Liquidity Analysis Module")
        print("=" * 40)
        print("\nNo test data found.")
        print("Looked in:", possible_paths)
        print("\nExample usage:")
        print("  from ratios.liquidity import LiquidityAnalysis")
        print("  analysis = LiquidityAnalysis(balance_sheet)")
        print("  results = analysis.calculate_all()")
        print("  analysis.print_summary()")











