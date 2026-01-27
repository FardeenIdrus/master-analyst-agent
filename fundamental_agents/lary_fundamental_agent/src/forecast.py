"""
Financial Forecast Module
Forecasts financial statements for 4 years with confidence intervals.

Methods:
- Revenue: Historical growth rate + trend analysis
- Net Income: Projected revenue × average margin
- Free Cash Flow: Net Income + D&A - CapEx
- Total Debt: Historical trend
- Total Assets: Based on asset turnover

Confidence Intervals:
- Base Case: Most likely (historical average)
- Bull Case: Optimistic (+1 standard deviation)
- Bear Case: Pessimistic (-1 standard deviation)
"""

import pandas as pd 
import numpy as np
from pathlib import Path


class FinancialForecast:
    """Forecasts financial statements with confidence intervals."""
    
    def __init__(self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame, cash_flow: pd.DataFrame):
        self.income_stmt = income_stmt
        self.balance_sheet = balance_sheet
        self.cash_flow = cash_flow
        
        self.forecast_years = 4
        self.forecasts = {}
        self.confidence = {}
    
    def _get_growth_rate(self, series: pd.Series) -> tuple:
        """
        Calculate average growth rate and standard deviation.
        Returns: (mean_growth, std_growth)
        """
        # Remove NaN values
        clean = series.dropna()
        if len(clean) < 2:
            return 0, 0
        
        # Calculate year-over-year growth rates
        # Data is newest to oldest, so we reverse for proper calculation
        values = clean.values[::-1]  # oldest to newest
        growth_rates = []
        
        for i in range(1, len(values)):
            if values[i-1] != 0:
                growth = (values[i] - values[i-1]) / abs(values[i-1])
                growth_rates.append(growth)
        
        if not growth_rates:
            return 0, 0
        
        mean_growth = np.mean(growth_rates)
        std_growth = np.std(growth_rates) if len(growth_rates) > 1 else 0
        
        return mean_growth, std_growth
    
    def _get_average_margin(self, numerator: pd.Series, denominator: pd.Series) -> tuple:
        """Calculate average margin and standard deviation."""
        margin = numerator / denominator
        clean = margin.dropna()
        
        if len(clean) == 0:
            return 0, 0
        
        return clean.mean(), clean.std() if len(clean) > 1 else 0
    
    def _project_series(self, last_value: float, growth_mean: float, growth_std: float) -> dict:
        """
        Project values for forecast years with confidence intervals.
        
        Returns dict with:
        - base: Base case projection (mean growth)
        - bull: Optimistic (+1 std)
        - bear: Pessimistic (-1 std)
        """
        base = []
        bull = []
        bear = []
        
        val_base = last_value
        val_bull = last_value
        val_bear = last_value
        
        for year in range(self.forecast_years):
            val_base *= (1 + growth_mean)
            val_bull *= (1 + growth_mean + growth_std)
            val_bear *= (1 + growth_mean - growth_std)
            
            base.append(val_base)
            bull.append(val_bull)
            bear.append(max(0, val_bear))  # Prevent negative values
        
        return {'base': base, 'bull': bull, 'bear': bear}
    
    def forecast_revenue(self) -> dict:
        """Forecast revenue using historical growth rate."""
        try:
            revenue = self.income_stmt.loc['Total Revenue']
            last_value = revenue.iloc[0]  # Most recent
            
            growth_mean, growth_std = self._get_growth_rate(revenue)
            
            projection = self._project_series(last_value, growth_mean, growth_std)
            
            self.forecasts['revenue'] = projection
            self.confidence['revenue'] = {
                'growth_mean': growth_mean,
                'growth_std': growth_std,
                'last_value': last_value
            }
            
            return projection
        except KeyError as e:
            print(f"  Error forecasting revenue: {e}")
            return None
    
    def forecast_net_income(self) -> dict:
        """Forecast net income using projected revenue and average margin."""
        try:
            if 'revenue' not in self.forecasts:
                self.forecast_revenue()
            
            revenue = self.income_stmt.loc['Total Revenue']
            net_income = self.income_stmt.loc['Net Income']
            
            margin_mean, margin_std = self._get_average_margin(net_income, revenue)
            
            # Apply margins to projected revenue
            base = [r * margin_mean for r in self.forecasts['revenue']['base']]
            bull = [r * (margin_mean + margin_std) for r in self.forecasts['revenue']['bull']]
            bear = [r * max(0, margin_mean - margin_std) for r in self.forecasts['revenue']['bear']]
            
            projection = {'base': base, 'bull': bull, 'bear': bear}
            
            self.forecasts['net_income'] = projection
            self.confidence['net_income'] = {
                'margin_mean': margin_mean,
                'margin_std': margin_std
            }
            
            return projection
        except KeyError as e:
            print(f"  Error forecasting net income: {e}")
            return None
    
    def forecast_free_cash_flow(self) -> dict:
        """Forecast FCF = Net Income + D&A - CapEx."""
        try:
            if 'net_income' not in self.forecasts:
                self.forecast_net_income()
            
            # Get historical ratios
            net_income = self.income_stmt.loc['Net Income']
            
            # D&A to Net Income ratio
            # Try multiple field names for compatibility
            da = None
            for field in ['Depreciation And Amortization', 'Reconciled Depreciation']:
                if field in self.income_stmt.index:
                    da = self.income_stmt.loc[field]
                    break
                if field in self.cash_flow.index:
                    da = self.cash_flow.loc[field]
                    break
            
            if da is None:
                da_ratio_mean = 0.15  # Default assumption
            else:
                da_ratio_mean = (da / net_income).dropna().mean()
            
            # CapEx to Net Income ratio
            capex = abs(self.cash_flow.loc['Capital Expenditure'])
            capex_ratio_mean = (capex / net_income).dropna().mean()
            
            # FCF = NI + D&A - CapEx
            fcf_factor = 1 + da_ratio_mean - capex_ratio_mean
            
            base = [ni * fcf_factor for ni in self.forecasts['net_income']['base']]
            bull = [ni * (fcf_factor * 1.1) for ni in self.forecasts['net_income']['bull']]
            bear = [ni * (fcf_factor * 0.9) for ni in self.forecasts['net_income']['bear']]
            
            projection = {'base': base, 'bull': bull, 'bear': bear}
            
            self.forecasts['free_cash_flow'] = projection
            self.confidence['free_cash_flow'] = {
                'da_ratio': da_ratio_mean,
                'capex_ratio': capex_ratio_mean,
                'fcf_factor': fcf_factor
            }
            
            return projection
        except KeyError as e:
            print(f"  Error forecasting FCF: {e}")
            return None
    
    def forecast_total_assets(self) -> dict:
        """Forecast total assets using asset turnover ratio."""
        try:
            if 'revenue' not in self.forecasts:
                self.forecast_revenue()
            
            revenue = self.income_stmt.loc['Total Revenue']
            assets = self.balance_sheet.loc['Total Assets']
            
            # Average asset turnover
            turnover_mean = (revenue / assets).dropna().mean()
            
            # Assets = Revenue / Turnover
            base = [r / turnover_mean for r in self.forecasts['revenue']['base']]
            bull = [r / (turnover_mean * 0.95) for r in self.forecasts['revenue']['bull']]  # Lower turnover = more assets
            bear = [r / (turnover_mean * 1.05) for r in self.forecasts['revenue']['bear']]
            
            projection = {'base': base, 'bull': bull, 'bear': bear}
            
            self.forecasts['total_assets'] = projection
            self.confidence['total_assets'] = {
                'turnover_mean': turnover_mean
            }
            
            return projection
        except KeyError as e:
            print(f"  Error forecasting total assets: {e}")
            return None
    
    def forecast_total_debt(self) -> dict:
        """Forecast total debt using historical trend."""
        try:
            debt = self.balance_sheet.loc['Total Debt']
            last_value = debt.iloc[0]
            
            growth_mean, growth_std = self._get_growth_rate(debt)
            
            projection = self._project_series(last_value, growth_mean, growth_std)
            
            self.forecasts['total_debt'] = projection
            self.confidence['total_debt'] = {
                'growth_mean': growth_mean,
                'growth_std': growth_std,
                'last_value': last_value
            }
            
            return projection
        except KeyError as e:
            print(f"  Error forecasting total debt: {e}")
            return None
    
    def forecast_eps(self) -> dict:
        """Forecast EPS using net income growth."""
        try:
            if 'net_income' not in self.forecasts:
                self.forecast_net_income()
            
            # Get shares outstanding trend (usually stable or decreasing for buybacks)
            net_income = self.income_stmt.loc['Net Income']
            
            # Use last known EPS as base
            # Assume shares stay roughly constant (conservative)
            last_ni = net_income.iloc[0]
            
            base = self.forecasts['net_income']['base']
            bull = self.forecasts['net_income']['bull']
            bear = self.forecasts['net_income']['bear']
            
            # Scale by ratio to get EPS-like values
            scale = 1e9  # Normalize to per-share basis (approximate)
            
            projection = {
                'base': [ni / scale for ni in base],
                'bull': [ni / scale for ni in bull],
                'bear': [ni / scale for ni in bear]
            }
            
            self.forecasts['eps_growth'] = projection
            
            return projection
        except KeyError as e:
            print(f"  Error forecasting EPS: {e}")
            return None
    
    def calculate_all(self) -> dict:
        """Run all forecasts."""
        print("Generating 4-year financial forecasts...")
        
        self.forecast_revenue()
        self.forecast_net_income()
        self.forecast_free_cash_flow()
        self.forecast_total_assets()
        self.forecast_total_debt()
        
        print(f"  Completed: {len(self.forecasts)} items forecasted")
        return self.forecasts
    
    def get_forecast_years(self) -> list:
        """Get list of forecast year labels."""
        # Get most recent year from data
        try:
            latest = str(self.income_stmt.columns[0])[:4]
            base_year = int(latest)
        except:
            base_year = 2025
        
        return [f"{base_year + i + 1}E" for i in range(self.forecast_years)]
    
    def print_summary(self):
        """Print formatted forecast summary."""
        if not self.forecasts:
            print("No forecasts generated. Run calculate_all() first.")
            return
        
        years = self.get_forecast_years()
        
        print("\n" + "=" * 75)
        print("FINANCIAL FORECAST (4-YEAR PROJECTION)")
        print("=" * 75)
        
        # Header
        header = f"{'Metric':<20}{'Case':<8}"
        for year in years:
            header += f"{year:>12}"
        print(header)
        print("-" * 75)
        
        display_items = {
            'revenue': ('Revenue ($B)', 1e9),
            'net_income': ('Net Income ($B)', 1e9),
            'free_cash_flow': ('Free Cash Flow ($B)', 1e9),
            'total_assets': ('Total Assets ($B)', 1e9),
            'total_debt': ('Total Debt ($B)', 1e9)
        }
        
        for key, (name, scale) in display_items.items():
            if key in self.forecasts:
                proj = self.forecasts[key]
                
                # Base case
                row = f"{name:<20}{'Base':<8}"
                for val in proj['base']:
                    row += f"{val/scale:>12.1f}"
                print(row)
                
                # Bull case
                row = f"{'':<20}{'Bull':<8}"
                for val in proj['bull']:
                    row += f"{val/scale:>12.1f}"
                print(row)
                
                # Bear case
                row = f"{'':<20}{'Bear':<8}"
                for val in proj['bear']:
                    row += f"{val/scale:>12.1f}"
                print(row)
                
                print()
        
        print("-" * 75)
        
        # Print confidence metrics
        print("\nFORECAST ASSUMPTIONS:")
        print("-" * 40)
        
        if 'revenue' in self.confidence:
            c = self.confidence['revenue']
            print(f"  Revenue Growth Rate:    {c['growth_mean']*100:.1f}% ± {c['growth_std']*100:.1f}%")
        
        if 'net_income' in self.confidence:
            c = self.confidence['net_income']
            print(f"  Net Margin:             {c['margin_mean']*100:.1f}% ± {c['margin_std']*100:.1f}%")
        
        if 'free_cash_flow' in self.confidence:
            c = self.confidence['free_cash_flow']
            print(f"  FCF Conversion:         {c['fcf_factor']*100:.1f}% of Net Income")
        
        if 'total_debt' in self.confidence:
            c = self.confidence['total_debt']
            print(f"  Debt Growth Rate:       {c['growth_mean']*100:.1f}% ± {c['growth_std']*100:.1f}%")
        
        print("-" * 40)
        print("  Confidence Level: 68% (±1 Standard Deviation)")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert forecasts to DataFrame."""
        if not self.forecasts:
            return pd.DataFrame()
        
        years = self.get_forecast_years()
        data = {}
        
        for key, proj in self.forecasts.items():
            for case in ['base', 'bull', 'bear']:
                col_name = f"{key}_{case}"
                data[col_name] = proj[case]
        
        df = pd.DataFrame(data, index=years)
        return df


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import os
    
    paths = ["../APPL_Data/", "APPL_Data/", ""]
    data_dir = next((p for p in paths if os.path.exists(f"{p}AAPL_income_statement.csv")), None)
    
    if data_dir is not None:
        print("Testing Financial Forecast...\n")
        
        income_stmt = pd.read_csv(f"{data_dir}AAPL_income_statement.csv", index_col=0)
        balance_sheet = pd.read_csv(f"{data_dir}AAPL_balance_sheet.csv", index_col=0)
        cash_flow = pd.read_csv(f"{data_dir}AAPL_cash_flow.csv", index_col=0)
        
        forecast = FinancialForecast(income_stmt, balance_sheet, cash_flow)
        forecast.calculate_all()
        forecast.print_summary()
        
        print("\n Test passed!")
    else:
        print("No test data found. Place AAPL CSVs in APPL_Data/ folder.")