#!/usr/bin/env python3
"""
Run Demo Script - Fundamental Analyst AI Agent
===============================================================================

MSc Coursework: IFTE0001 - AI Agents in Asset Management
Track A: Fundamental Analyst Agent

Main entry point for the Fundamental Analyst Agent demonstration.
Orchestrates the complete 10-step analysis pipeline with Claude AI integration.

Usage:
    python run_demo.py                    # Analyze AAPL (default)
    python run_demo.py MSFT               # Analyze Microsoft
    python run_demo.py GOOGL -o ./reports # Custom output directory
    python run_demo.py TSLA --quiet       # Minimal output

Pipeline Steps:
    1. Data Collection      - Fetch from Alpha Vantage (5 years)
    2. Data Processing      - Standardize and validate
    3. Profitability        - EBIT variance bridge
    4. Cash Flow            - Quality and conversion
    5. Earnings Quality     - Accruals and red flags
    6. Working Capital      - DSO, DIO, DPO, CCC
    7. Financial Ratios     - ROE, ROA, ROIC
    8. Valuation            - P/E, EV/EBITDA, P/FCF
    9. Memo Generation      - Claude AI integration
    10. Validation          - Accuracy verification

Output:
    - JSON: Comprehensive structured data
    - Markdown: Investment memorandum
    - HTML: Professional web report
    - PDF: Institutional-quality document

Author: MSc AI Agents in Asset Management
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'src'))

# =============================================================================
# CONFIGURATION
# =============================================================================

# Alpha Vantage API Key - Provides 5+ years of complete annual financial data
# Get free key at: https://www.alphavantage.co/support/#api-key
# Alpha Vantage API Key - Provides 5+ years of complete annual financial data
# Get free key at: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY = "PZ3ALS5PSLJZS0W6"

# Set Alpha Vantage API key in environment
if ALPHA_VANTAGE_API_KEY and not os.getenv("ALPHA_VANTAGE_API_KEY"):
    os.environ["ALPHA_VANTAGE_API_KEY"] = ALPHA_VANTAGE_API_KEY

# Claude API configuration - Anthropic API key for memo generation
# Get your API key from: https://console.anthropic.com/
# Set this in your environment variables or config file
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


# =============================================================================
# DISPLAY UTILITIES
# =============================================================================

def print_banner():
    """Display application banner."""
    print("""
================================================================================
                    FUNDAMENTAL ANALYST AI AGENT
================================================================================

    MSc Coursework: AI Agents in Asset Management
    Track A: Fundamental Analyst Agent

    Features:
      - Deep Claude AI Integration (Anthropic)
      - 10-Step Financial Analysis Pipeline
      - Institutional-Quality Investment Memoranda
      - Comprehensive Multi-Format Output

    Configuration:
      - Data Source: Alpha Vantage (5 years of annual data)

================================================================================
""")


def print_section(title: str):
    """Print section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_subsection(title: str):
    """Print subsection header."""
    print(f"\n{'-' * 50}")
    print(f"  {title}")
    print(f"{'-' * 50}")


def fmt_time(ms: float) -> str:
    """Format milliseconds to readable string."""
    return f"{ms/1000:.1f}s" if ms >= 1000 else f"{ms:.0f}ms"


def fmt_size(bytes: int) -> str:
    """Format bytes to readable string."""
    return f"{bytes/1024/1024:.1f}MB" if bytes >= 1024*1024 else f"{bytes/1024:.1f}KB"


def print_data_collection_details(agent):
    """Print detailed data collection results."""
    print_subsection("Step 1 Data Collection Details")
    
    if not hasattr(agent, '_collected_data') or agent._collected_data is None:
        print("  No collected data available.")
        return
    
    data = agent._collected_data
    
    # Company Info
    info = data.company_info
    print(f"\n  COMPANY INFORMATION:")
    print(f"    Name:               {info.name}")
    print(f"    Ticker:             {info.ticker}")
    print(f"    Sector:             {info.sector or 'N/A'}")
    print(f"    Industry:           {info.industry or 'N/A'}")
    print(f"    Currency:           {info.currency}")
    print(f"    Current Price:      ${info.current_price:,.2f}" if info.current_price else "    Current Price:      N/A")
    print(f"    Market Cap:         ${info.market_cap/1e9:,.1f}B" if info.market_cap else "    Market Cap:         N/A")
    print(f"    Shares Outstanding: {info.shares_outstanding/1e9:.2f}B" if info.shares_outstanding else "    Shares Outstanding: N/A")
    print(f"    Beta:               {info.beta:.2f}" if info.beta else "    Beta:               N/A")
    print(f"    52-Week High:       ${info.fifty_two_week_high:,.2f}" if info.fifty_two_week_high else "    52-Week High:       N/A")
    print(f"    52-Week Low:        ${info.fifty_two_week_low:,.2f}" if info.fifty_two_week_low else "    52-Week Low:        N/A")
    
    # Financial Statements Summary
    stmts = data.statements
    print(f"\n  FINANCIAL STATEMENTS SUMMARY:")
    
    # Income Statement
    is_df = stmts.income_statement
    if is_df is not None and not is_df.empty:
        print(f"\n    INCOME STATEMENT:")
        print(f"      Periods: {len(is_df.columns)} years")
        print(f"      Line Items: {len(is_df)} rows")
        fiscal_years = [str(c.year) if hasattr(c, 'year') else str(c)[:4] for c in is_df.columns]
        print(f"      Fiscal Years: {', '.join(fiscal_years)}")
        
        # Check completeness per year
        key_fields_to_check = ['Total Revenue', 'Net Income', 'Operating Income', 'Gross Profit']
        print(f"\n      Data Completeness by Year:")
        for i, col in enumerate(is_df.columns):
            year_label = fiscal_years[i]
            fields_present = 0
            fields_missing = []
            for field in key_fields_to_check:
                if field in is_df.index:
                    val = is_df.loc[field, col]
                    if pd.notna(val) and val != 0:
                        fields_present += 1
                    else:
                        fields_missing.append(field)
                else:
                    fields_missing.append(field)
            
            status = "COMPLETE" if fields_present >= 3 else "INCOMPLETE"
            status_icon = "[OK]" if fields_present >= 3 else "[!!]"
            missing_str = f" (missing: {', '.join(fields_missing)})" if fields_missing else ""
            print(f"        {status_icon} {year_label}: {fields_present}/{len(key_fields_to_check)} key fields{missing_str}")
        
        print(f"\n      Key Line Items (latest year):")
        key_items = ['Total Revenue', 'Gross Profit', 'Operating Income', 'EBIT', 'Net Income', 
                     'Cost Of Revenue', 'Operating Expense', 'Interest Expense', 'Tax Provision']
        for item in key_items:
            if item in is_df.index:
                vals = is_df.loc[item]
                latest = vals.iloc[0] if len(vals) > 0 else None
                if latest is not None and not pd.isna(latest):
                    print(f"        {item:25s}: ${latest/1e6:>12,.0f}M (latest)")
        
        print(f"\n      All Available Line Items ({len(is_df)} total):")
        for i, item in enumerate(is_df.index):
            print(f"        {i+1:3d}. {item}")
    else:
        print(f"\n    INCOME STATEMENT: NOT AVAILABLE")
    
    # Balance Sheet
    bs_df = stmts.balance_sheet
    if bs_df is not None and not bs_df.empty:
        print(f"\n    BALANCE SHEET:")
        print(f"      Periods: {len(bs_df.columns)} years")
        print(f"      Line Items: {len(bs_df)} rows")
        fiscal_years_bs = [str(c.year) if hasattr(c, 'year') else str(c)[:4] for c in bs_df.columns]
        print(f"      Fiscal Years: {', '.join(fiscal_years_bs)}")
        
        # Check completeness per year
        key_fields_bs = ['Total Assets', 'Total Liabilities Net Minority Interest', 'Stockholders Equity']
        print(f"\n      Data Completeness by Year:")
        for i, col in enumerate(bs_df.columns):
            year_label = fiscal_years_bs[i]
            fields_present = 0
            fields_missing = []
            for field in key_fields_bs:
                if field in bs_df.index:
                    val = bs_df.loc[field, col]
                    if pd.notna(val) and val != 0:
                        fields_present += 1
                    else:
                        fields_missing.append(field)
                else:
                    fields_missing.append(field)
            
            status_icon = "[OK]" if fields_present >= 2 else "[!!]"
            missing_str = f" (missing: {', '.join(fields_missing[:2])})" if fields_missing else ""
            print(f"        {status_icon} {year_label}: {fields_present}/{len(key_fields_bs)} key fields{missing_str}")
        
        print(f"\n      Key Line Items (latest year):")
        key_items = ['Total Assets', 'Total Liabilities Net Minority Interest', 'Stockholders Equity',
                     'Cash And Cash Equivalents', 'Total Debt', 'Current Assets', 'Current Liabilities',
                     'Accounts Receivable', 'Inventory', 'Accounts Payable']
        for item in key_items:
            if item in bs_df.index:
                vals = bs_df.loc[item]
                latest = vals.iloc[0] if len(vals) > 0 else None
                if latest is not None and not pd.isna(latest):
                    print(f"        {item:45s}: ${latest/1e6:>12,.0f}M")
        
        print(f"\n      All Available Line Items ({len(bs_df)} total):")
        for i, item in enumerate(bs_df.index):
            print(f"        {i+1:3d}. {item}")
    else:
        print(f"\n    BALANCE SHEET: NOT AVAILABLE")
    
    # Cash Flow Statement
    cf_df = stmts.cash_flow
    if cf_df is not None and not cf_df.empty:
        print(f"\n    CASH FLOW STATEMENT:")
        print(f"      Periods: {len(cf_df.columns)} years")
        print(f"      Line Items: {len(cf_df)} rows")
        fiscal_years_cf = [str(c.year) if hasattr(c, 'year') else str(c)[:4] for c in cf_df.columns]
        print(f"      Fiscal Years: {', '.join(fiscal_years_cf)}")
        
        # Check completeness per year
        key_fields_cf = ['Operating Cash Flow', 'Free Cash Flow', 'Capital Expenditure']
        print(f"\n      Data Completeness by Year:")
        for i, col in enumerate(cf_df.columns):
            year_label = fiscal_years_cf[i]
            fields_present = 0
            fields_missing = []
            for field in key_fields_cf:
                if field in cf_df.index:
                    val = cf_df.loc[field, col]
                    if pd.notna(val) and val != 0:
                        fields_present += 1
                    else:
                        fields_missing.append(field)
                else:
                    fields_missing.append(field)
            
            status_icon = "[OK]" if fields_present >= 2 else "[!!]"
            missing_str = f" (missing: {', '.join(fields_missing)})" if fields_missing else ""
            print(f"        {status_icon} {year_label}: {fields_present}/{len(key_fields_cf)} key fields{missing_str}")
        
        print(f"\n      Key Line Items (latest year):")
        key_items = ['Operating Cash Flow', 'Free Cash Flow', 'Capital Expenditure', 
                     'Depreciation And Amortization', 'Net Income From Continuing Operations',
                     'Change In Working Capital', 'Investing Cash Flow', 'Financing Cash Flow']
        for item in key_items:
            if item in cf_df.index:
                vals = cf_df.loc[item]
                latest = vals.iloc[0] if len(vals) > 0 else None
                if latest is not None and not pd.isna(latest):
                    print(f"        {item:45s}: ${latest/1e6:>12,.0f}M")
        
        print(f"\n      All Available Line Items ({len(cf_df)} total):")
        for i, item in enumerate(cf_df.index):
            print(f"        {i+1:3d}. {item}")
    else:
        print(f"\n    CASH FLOW STATEMENT: NOT AVAILABLE")
    
    # Print full data tables for each statement
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(f"\n  FULL DATA TABLES (values in original currency):")
    
    if is_df is not None and not is_df.empty:
        print(f"\n  === INCOME STATEMENT (5 Years) ===")
        # Format for display
        display_df = is_df.copy()
        display_df.columns = [str(c.year) if hasattr(c, 'year') else str(c)[:4] for c in display_df.columns]
        for col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"${x/1e6:,.0f}M" if pd.notna(x) and x != 0 else "N/A")
        print(display_df.to_string())
    
    if bs_df is not None and not bs_df.empty:
        print(f"\n  === BALANCE SHEET (5 Years) ===")
        display_df = bs_df.copy()
        display_df.columns = [str(c.year) if hasattr(c, 'year') else str(c)[:4] for c in display_df.columns]
        for col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"${x/1e6:,.0f}M" if pd.notna(x) and x != 0 else "N/A")
        print(display_df.to_string())
    
    if cf_df is not None and not cf_df.empty:
        print(f"\n  === CASH FLOW STATEMENT (5 Years) ===")
        display_df = cf_df.copy()
        display_df.columns = [str(c.year) if hasattr(c, 'year') else str(c)[:4] for c in display_df.columns]
        for col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"${x/1e6:,.0f}M" if pd.notna(x) and x != 0 else "N/A")
        print(display_df.to_string())
    
    # Validation Summary
    val = data.validation
    print(f"\n  VALIDATION RESULTS:")
    print(f"    Status:           {val.status.value.upper()}")
    print(f"    Years Available:  {val.years_available} (complete data)")
    
    # Count total columns for comparison
    total_cols = len(data.statements.income_statement.columns) if data.statements.income_statement is not None and not data.statements.income_statement.empty else 0
    if total_cols > val.years_available:
        print(f"    Total Columns:    {total_cols} (includes incomplete)")
    
    if val.errors:
        print(f"    Errors:           {len(val.errors)}")
        for e in val.errors:
            print(f"      - {e}")
    if val.warnings:
        print(f"    Warnings:         {len(val.warnings)}")
        for w in val.warnings:
            print(f"      - {w}")
    if val.missing_statements:
        print(f"    Missing Statements: {', '.join(val.missing_statements)}")
    
    print("")


# =============================================================================
# CORE EXECUTION
# =============================================================================

def run_demo(ticker: str, output_dir: str, api_key: Optional[str] = None, verbose: bool = True) -> Tuple[bool, Optional[Any]]:
    """Execute complete analysis pipeline."""
    from agent import FundamentalAnalystAgent
    
    print_section(f"ANALYSIS: {ticker}")
    print(f"  Configuration:")
    print(f"    Ticker:      {ticker}")
    print(f"    Output:      {os.path.abspath(output_dir)}")
    print(f"    Start Time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"    Data Source: Alpha Vantage (5 years of annual data)")
    print(f"    LLM:         Claude AI (claude-sonnet-4-20250514)")
    
    # Phase 1: Initialize
    print_subsection("Phase 1: Initialization")
    try:
        agent = FundamentalAnalystAgent(ticker, api_key=api_key or ANTHROPIC_API_KEY)
        print(f"  [OK] Agent initialized")
        print(f"  [OK] Claude AI integration active")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False, None
    
    # Phase 2: Execute Pipeline
    print_subsection("Phase 2: Analysis Pipeline")
    print("  Executing 10-step pipeline...")
    
    try:
        results = agent.run_analysis()
        
        # Print detailed data collection results
        if verbose:
            print_data_collection_details(agent)
        
        print(f"\n  {'Step':<6} {'Module':<28} {'Status':<10} {'Time':<10}")
        print(f"  {'-'*54}")
        for step in results.steps.values():
            status = {'success': 'OK', 'warning': 'WARN', 'failed': 'FAIL', 'skipped': 'SKIP'}.get(step.status.value, '???')
            print(f"  {step.step_number:<6} {step.step_name:<28} {status:<10} {fmt_time(step.execution_time_ms):<10}")
        print(f"  {'-'*54}")
        
        print(f"\n  Total Time: {fmt_time(results.execution_time_total_ms)}")
        print(f"  Status: {results.overall_status.value.upper()}")
        
        if results.validation_report:
            vr = results.validation_report
            print(f"\n  Validation: {vr.validation_score:.0f}/100 ({vr.passed_checks}/{vr.total_checks} checks)")
        
        if results.memo:
            print(f"\n  Recommendation: {results.memo.recommendation.value}")
            print(f"  Confidence: {results.memo.confidence_score:.0f}%")
            if results.memo.target_price:
                print(f"  Target Price: ${results.memo.target_price:.2f}")
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        return False, None
    
    # Phase 3: Export
    print_subsection("Phase 3: Export Results")
    try:
        files = agent.export_results(output_dir)
        print(f"  Generated {len(files)} files:\n")
        
        for fmt, path in files.items():
            size = os.path.getsize(path)
            print(f"    {fmt.upper():<10} {os.path.basename(path):<45} {fmt_size(size)}")
    except Exception as e:
        print(f"  [WARNING] {e}")
    
    # Phase 4: Summary
    print_subsection("Phase 4: Summary")
    if verbose:
        agent.print_summary()
    
    return True, results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fundamental Analyst AI Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_demo.py               Analyze Apple (default)
    python run_demo.py MSFT          Analyze Microsoft
    python run_demo.py GOOGL -o out  Custom output directory
    python run_demo.py --quiet       Minimal output

Pipeline:
    1. Data Collection    6. Working Capital
    2. Data Processing    7. Financial Ratios
    3. Profitability      8. Valuation
    4. Cash Flow          9. Memo Generation
    5. Earnings Quality   10. Validation
"""
    )
    
    parser.add_argument('ticker', nargs='?', default='AAPL', help='Stock ticker (default: AAPL)')
    parser.add_argument('--output', '-o', default='outputs', help='Output directory')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    if not args.quiet:
        print_banner()
    
    ticker = args.ticker.upper().strip()
    if not ticker.isalpha() or len(ticker) > 5:
        print(f"[ERROR] Invalid ticker: {args.ticker}")
        sys.exit(1)
    
    output_dir = os.path.join(args.output, ticker)
    os.makedirs(output_dir, exist_ok=True)
    
    success, results = run_demo(ticker, output_dir, verbose=not args.quiet)
    
    print_section("COMPLETE")
    
    if success:
        print("  Status: SUCCESS")
        print(f"\n  Output: {os.path.abspath(output_dir)}")
        print(f"\n  Files:")
        print(f"    - {ticker}_analysis_*.json")
        print(f"    - {ticker}_investment_memo_*.md")
        print(f"    - {ticker}_investment_memo_*.html")
        print(f"    - {ticker}_report_*.pdf")
        
        if results and results.memo:
            print(f"\n  Recommendation: {results.memo.recommendation.value}")
            print(f"  Confidence: {results.memo.confidence_score:.0f}%")
        
        print("\n" + "=" * 70)
        sys.exit(0)
    else:
        print("  Status: FAILED")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)