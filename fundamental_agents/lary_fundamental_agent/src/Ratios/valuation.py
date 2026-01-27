"""
Valuation Ratios Module
Calculates: P/E Ratio, P/B Ratio, P/S Ratio, EV/EBITDA
"""

import pandas as pd
import yfinance as yf


class ValuationAnalysis:
    """Analyses company valuation multiples."""
    
    def __init__(self, ticker: str, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame):
        self.ticker = ticker.upper()
        self.income_stmt = income_stmt
        self.balance_sheet = balance_sheet
        self.ratios = {}
        
        # Get live market data
        self.stock = yf.Ticker(self.ticker)
        self.market_cap = self.stock.info.get('marketCap', 0)
        self.current_price = self.stock.info.get('currentPrice', 0)
        self.shares_outstanding = self.stock.info.get('sharesOutstanding', 0)
    
    def get_pe_ratio(self) -> float:
        """Price-to-Earnings Ratio = Market Cap / Net Income"""
        try:
            net_income = self.income_stmt.loc['Net Income'].iloc[0]
            if net_income > 0:
                result = self.market_cap / net_income
                self.ratios['pe_ratio'] = result
                return result
            return None
        except KeyError as e:
            print(f"  Error calculating P/E ratio: {e}")
            return None
    
    def get_pb_ratio(self) -> float:
        """Price-to-Book Ratio = Market Cap / Stockholders Equity"""
        try:
            equity = self.balance_sheet.loc['Stockholders Equity'].iloc[0]
            if equity > 0:
                result = self.market_cap / equity
                self.ratios['pb_ratio'] = result
                return result
            return None
        except KeyError as e:
            print(f"  Error calculating P/B ratio: {e}")
            return None
    
    def get_ps_ratio(self) -> float:
        """Price-to-Sales Ratio = Market Cap / Total Revenue"""
        try:
            revenue = self.income_stmt.loc['Total Revenue'].iloc[0]
            if revenue > 0:
                result = self.market_cap / revenue
                self.ratios['ps_ratio'] = result
                return result
            return None
        except KeyError as e:
            print(f"  Error calculating P/S ratio: {e}")
            return None
    
    def get_ev_ebitda(self) -> float:
        """EV/EBITDA = Enterprise Value / EBITDA"""
        try:
            # Enterprise Value = Market Cap + Total Debt - Cash
            total_debt = self.balance_sheet.loc['Total Debt'].iloc[0]
            cash = self.balance_sheet.loc['Cash And Cash Equivalents'].iloc[0]
            ev = self.market_cap + total_debt - cash
            
            ebitda = self.income_stmt.loc['EBITDA'].iloc[0]
            if ebitda > 0:
                result = ev / ebitda
                self.ratios['ev_ebitda'] = result
                return result
            return None
        except KeyError as e:
            print(f"  Error calculating EV/EBITDA: {e}")
            return None
    
    def get_eps(self) -> float:
        """Earnings Per Share = Net Income / Shares Outstanding"""
        try:
            net_income = self.income_stmt.loc['Net Income'].iloc[0]
            if self.shares_outstanding > 0:
                result = net_income / self.shares_outstanding
                self.ratios['eps'] = result
                return result
            return None
        except KeyError as e:
            print(f"  Error calculating EPS: {e}")
            return None
    
    def calculate_all(self) -> dict:
        """Calculate all valuation ratios."""
        self.get_pe_ratio()
        self.get_pb_ratio()
        self.get_ps_ratio()
        self.get_ev_ebitda()
        self.get_eps()
        return self.ratios
    
    def get_latest_values(self) -> dict:
        """Get all calculated ratios."""
        return self.ratios
    
    def _format(self, value, suffix=""):
        """Format value, handling None."""
        if value is None or pd.isna(value):
            return "N/A"
        return f"{value:.2f}{suffix}"
    
    def print_summary(self):
        """Print formatted summary."""
        if not self.ratios:
            print("No ratios calculated. Run calculate_all() first.")
            return
        
        print("\n" + "=" * 55)
        print("VALUATION ANALYSIS")
        print("=" * 55)
        print(f"  {'Stock Price:':<25} ${self.current_price:,.2f}")
        print(f"  {'Market Cap:':<25} ${self.market_cap/1e9:,.1f}B")
        print("-" * 55)
        print(f"  {'P/E Ratio:':<25} {self._format(self.ratios.get('pe_ratio'))}x")
        print(f"  {'P/B Ratio:':<25} {self._format(self.ratios.get('pb_ratio'))}x")
        print(f"  {'P/S Ratio:':<25} {self._format(self.ratios.get('ps_ratio'))}x")
        print(f"  {'EV/EBITDA:':<25} {self._format(self.ratios.get('ev_ebitda'))}x")
        print(f"  {'EPS:':<25} ${self._format(self.ratios.get('eps'))}")
        print("-" * 55)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import os
    
    paths = ["../APPL_Data/", "APPL_Data/", ""]
    data_dir = next((p for p in paths if os.path.exists(f"{p}AAPL_income_statement.csv")), None)
    
    if data_dir is not None:
        print("Testing Valuation Analysis...\n")
        
        income_stmt = pd.read_csv(f"{data_dir}AAPL_income_statement.csv", index_col=0)
        balance_sheet = pd.read_csv(f"{data_dir}AAPL_balance_sheet.csv", index_col=0)
        
        analysis = ValuationAnalysis("AAPL", income_stmt, balance_sheet)
        analysis.calculate_all()
        analysis.print_summary()
        
        print("\n✅ Test passed!")
    else:
        print("No test data found. Place AAPL CSVs in APPL_Data/ folder.")