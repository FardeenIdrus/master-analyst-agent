#!/usr/bin/env python3
"""
Fundamental Analyst Agent - Phases 1-9 Demo
============================================

IFTE0001: AI Agents in Asset Management - Track A

Demonstrates the complete fundamental analysis pipeline from data acquisition
through LLM-generated investment memo:

Phase 1: Data Acquisition
- 5 years of financial statements from Alpha Vantage
- Comprehensive validation and derived metrics

Phase 2: Data Validation & Standardization
- Cross-statement reconciliation
- Outlier detection
- Growth rate analysis
- Sign validation

Phase 3: Financial Ratio Analysis
- 39 ratios across 5 categories
- Benchmark-based assessment
- Overall financial health scoring

Phase 4: DuPont Analysis
- 3-Factor DuPont decomposition (NPM x AT x EM)
- 5-Factor DuPont decomposition (Tax x Interest x Operating x AT x EM)
- Year-over-year variance attribution
- ROE quality and sustainability assessment

Phase 5: DCF Valuation
- Historical FCF analysis and quality assessment
- Multi-method growth rate derivation (CAGR, Sustainable, Analyst)
- WACC calculation via CAPM
- 5-year FCF projection with Gordon Growth terminal value
- Sensitivity analysis (growth vs WACC matrix)
- Scenario analysis (Bear, Base, Bull cases)
- Intrinsic value per share with market comparison

Phase 6: DDM Valuation
- DDM applicability assessment
- Historical dividend analysis and quality assessment
- Dividend growth rate derivation (CAGR, Sustainable, Earnings-based)
- Cost of Equity calculation via CAPM
- 5-year dividend projection with Gordon Growth terminal value
- Sensitivity analysis (growth vs Ke matrix)
- Scenario analysis (Bear, Base, Bull cases)
- DCF vs DDM reconciliation

Phase 7: Multiples Valuation
- 6 valuation multiples (P/E, P/B, P/S, P/FCF, EV/EBITDA, EV/Revenue)
- Historical average comparisons
- Cross-model consensus valuation

Phase 8: Accuracy Verification
- 43 verification checks across 6 categories
- Confidence scoring and reliability assessment
- Methodology compliance verification

Phase 9: LLM Investment Memo
- Claude Sonnet 4.5 integration
- Professional 2-page investment memo generation
- Multi-format output (JSON, Markdown, HTML, PDF)
- Data-driven recommendation synthesis

Usage:
    python run_demo.py              # Analyze Apple (default)
    python run_demo.py MSFT         # Analyze Microsoft  
    python run_demo.py AAPL --refresh  # Force refresh (bypass cache)
    python run_demo.py AAPL --phase1-only  # Phase 1 only
    python run_demo.py AAPL --phase8-only  # Phases 1-8 only (skip memo)
    python run_demo.py AAPL --api-key YOUR_KEY  # With Anthropic API key

Output:
    - Company Profile and Financial Statements
    - Financial Ratio Analysis (39 ratios)
    - DuPont Decomposition and ROE Quality
    - DCF, DDM, and Multiples Valuations
    - Accuracy Verification Report
    - Investment Memo (JSON, MD, HTML, PDF)

Version: 4.0.0
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.data_collector import DataCollector, CollectionResult
from src.data_validator import Phase2Validator, ValidatedData
from src.ratio_analyzer import Phase3Analyzer, RatioAnalysisResult, RatioAssessment
from src.dupont_analyzer import (
    Phase4Analyzer, 
    DuPontAnalysisResult, 
    DuPontDriver, 
    ROEQuality,
    TrendDirection,
)
from src import __version__

# Try to import Phase 5 DCF Valuator
try:
    from src.dcf_valuator import (
        Phase5Valuator,
        DCFValuationResult,
        ValuationSignal,
        FCFQuality,
        ScenarioType,
    )
    DCF_AVAILABLE = True
except ImportError:
    DCF_AVAILABLE = False

# Try to import Phase 6 DDM Valuator
try:
    from src.ddm_valuator import (
        Phase6Valuator,
        DDMValuationResult,
        DDMApplicability,
        DividendQuality,
        DDMScenarioType,
    )
    DDM_AVAILABLE = True
except ImportError:
    DDM_AVAILABLE = False

# Try to import Phase 7 Multiples Valuator
try:
    from src.multiples_valuator import (
        Phase7Valuator,
        MultiplesValuationResult,
        MultipleType,
        ValuationAssessment,
    )
    MULTIPLES_AVAILABLE = True
except ImportError:
    MULTIPLES_AVAILABLE = False

# Try to import Phase 8 Accuracy Checker
try:
    from src.accuracy_checker import (
        Phase8AccuracyChecker,
        AccuracyCheckResult,
        VerificationStatus,
        ConfidenceLevel,
        CheckCategory,
    )
    ACCURACY_AVAILABLE = True
except ImportError:
    ACCURACY_AVAILABLE = False

# Try to import Phase 9 Memo Generator
try:
    from src.memo_generator import (
        Phase9MemoGenerator,
        InvestmentMemoResult,
        MemoStatus,
        RecommendationType,
        ConvictionLevel as MemoConvictionLevel,
    )
    MEMO_AVAILABLE = True
except ImportError as e:
    print(f"  [DEBUG] Memo import error: {e}")
    MEMO_AVAILABLE = False

# Try to import Yahoo Finance supplementer
try:
    from src.yahoo_supplementer import YahooFinanceSupplementer, SupplementationReport
    YAHOO_AVAILABLE = True
except ImportError:
    YAHOO_AVAILABLE = False


# =============================================================================
# FORMATTING UTILITIES
# =============================================================================

def format_currency(value, scale=1e9, suffix="B"):
    """Format value as currency with scale."""
    if value is None:
        return "N/A"
    return f"${value/scale:,.2f}{suffix}"


def format_number(value, decimals=2):
    """Format numeric value."""
    if value is None:
        return "N/A"
    if abs(value) >= 1e9:
        return f"{value/1e9:,.2f}B"
    elif abs(value) >= 1e6:
        return f"{value/1e6:,.2f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:,.2f}K"
    else:
        return f"{value:,.{decimals}f}"


def format_percent(value, decimals=2):
    """Format value as percentage."""
    if value is None:
        return "N/A"
    return f"{value*100:.{decimals}f}%"


def print_line(char="=", length=80):
    """Print separator line."""
    print(char * length)


def print_header(title):
    """Print section header."""
    print()
    print_line("=")
    print(f"  {title}")
    print_line("=")


def print_subheader(title):
    """Print subsection header."""
    print()
    print(f"  {title}")
    print_line("-", 50)


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def print_banner():
    """Print application banner."""
    print()
    print_line()
    print("  FUNDAMENTAL ANALYST AGENT")
    print("  Phases 1-8: Data, Validation, Ratios, DuPont, DCF, DDM, Multiples & Accuracy")
    print_line()
    print("  IFTE0001: AI Agents in Asset Management - Track A")
    print(f"  Version: {__version__}")
    print_line()


def print_company_profile(profile):
    """Print company profile section."""
    print_header("COMPANY PROFILE")
    
    print(f"  Ticker:           {profile.ticker}")
    print(f"  Name:             {profile.name}")
    print(f"  Sector:           {profile.sector or 'N/A'}")
    print(f"  Industry:         {profile.industry or 'N/A'}")
    print(f"  Exchange:         {profile.exchange or 'N/A'}")
    print(f"  Currency:         {profile.currency}")
    print(f"  Fiscal Year End:  {profile.fiscal_year_end or 'N/A'}")
    
    print_subheader("Market Data")
    print(f"  Market Cap:       {format_currency(profile.market_cap)}")
    print(f"  Shares Out:       {format_number(profile.shares_outstanding)}")
    print(f"  52-Week High:     ${profile.high_52week:,.2f}" if profile.high_52week else "  52-Week High:     N/A")
    print(f"  52-Week Low:      ${profile.low_52week:,.2f}" if profile.low_52week else "  52-Week Low:      N/A")
    print(f"  Beta:             {profile.beta:.2f}" if profile.beta else "  Beta:             N/A")
    
    print_subheader("Valuation Multiples")
    print(f"  P/E Ratio:        {profile.pe_ratio:.2f}" if profile.pe_ratio else "  P/E Ratio:        N/A")
    print(f"  P/B Ratio:        {profile.pb_ratio:.2f}" if profile.pb_ratio else "  P/B Ratio:        N/A")
    print(f"  EV/EBITDA:        {profile.ev_to_ebitda:.2f}" if profile.ev_to_ebitda else "  EV/EBITDA:        N/A")
    print(f"  EV/Revenue:       {profile.ev_to_revenue:.2f}" if profile.ev_to_revenue else "  EV/Revenue:       N/A")
    
    print_subheader("Profitability")
    print(f"  Profit Margin:    {format_percent(profile.profit_margin)}")
    print(f"  Operating Margin: {format_percent(profile.operating_margin)}")
    print(f"  ROA:              {format_percent(profile.roa)}")
    print(f"  ROE:              {format_percent(profile.roe)}")
    
    print_subheader("Dividends")
    print(f"  Dividend/Share:   ${profile.dividend_per_share:.2f}" if profile.dividend_per_share else "  Dividend/Share:   N/A")
    print(f"  Dividend Yield:   {format_percent(profile.dividend_yield)}")


def print_quality_metrics(metrics):
    """Print data quality metrics section."""
    print_header("DATA QUALITY METRICS")
    
    print_subheader("Completeness Scores")
    print(f"  Income Statement: {format_percent(metrics.income_completeness)} ({metrics.income_fields_present}/{metrics.income_fields_expected} fields)")
    print(f"  Balance Sheet:    {format_percent(metrics.balance_completeness)} ({metrics.balance_fields_present}/{metrics.balance_fields_expected} fields)")
    print(f"  Cash Flow:        {format_percent(metrics.cashflow_completeness)} ({metrics.cashflow_fields_present}/{metrics.cashflow_fields_expected} fields)")
    print(f"  Overall:          {format_percent(metrics.overall_completeness)}")
    
    print_subheader("Validation Status")
    print(f"  Critical Income Fields:  {'PASS' if metrics.critical_income_valid else 'FAIL'}")
    print(f"  Critical Balance Fields: {'PASS' if metrics.critical_balance_valid else 'FAIL'}")
    print(f"  Critical Cashflow Fields: {'PASS' if metrics.critical_cashflow_valid else 'FAIL'}")
    
    acct_status = "PASS" if metrics.accounting_equation_valid else "FAIL"
    acct_dev = f" (deviation: {format_percent(metrics.accounting_equation_deviation)})" if metrics.accounting_equation_deviation else ""
    print(f"  Accounting Equation:     {acct_status}{acct_dev}")
    
    print_subheader("Quality Classification")
    print(f"  Quality Tier: {metrics.quality_tier.value.upper()}")
    
    if metrics.warnings:
        print_subheader("Warnings")
        for warning in metrics.warnings:
            print(f"  - {warning}")
    
    if metrics.info:
        print_subheader("Data Notes")
        for info in metrics.info:
            print(f"  - {info}")
    
    if metrics.errors:
        print_subheader("Errors")
        for error in metrics.errors:
            print(f"  - {error}")


def print_financial_statement(df, title, years_display=5):
    """Print financial statement data."""
    print_header(title)
    
    if df is None or df.empty:
        print("  No data available")
        return
    
    years = list(df.columns)[:years_display]
    
    # Print header row
    header = f"  {'Field':<35}"
    for year in years:
        header += f"{year:>15}"
    print(header)
    print_line("-", len(header))
    
    # Print data rows
    for field in df.index:
        row = f"  {field:<35}"
        for year in years:
            value = df.loc[field, year] if year in df.columns else None
            if value is None or (isinstance(value, float) and (value != value)):  # NaN check
                row += f"{'N/A':>15}"
            else:
                row += f"{format_number(value):>15}"
        print(row)


def print_derived_metrics(metrics, years):
    """Print derived metrics section."""
    print_header("DERIVED METRICS")
    
    if not years:
        print("  No data available")
        return
    
    years = years[:5]
    
    # Print header
    header = f"  {'Metric':<25}"
    for year in years:
        header += f"{year:>15}"
    print(header)
    print_line("-", len(header))
    
    # FCF
    row = f"  {'Free Cash Flow':<25}"
    for year in years:
        value = metrics.fcf_calculated.get(year)
        row += f"{format_number(value):>15}"
    print(row)
    
    # EBITDA
    row = f"  {'EBITDA':<25}"
    for year in years:
        value = metrics.ebitda_calculated.get(year)
        row += f"{format_number(value):>15}"
    print(row)
    
    # Working Capital
    row = f"  {'Working Capital':<25}"
    for year in years:
        value = metrics.working_capital.get(year)
        row += f"{format_number(value):>15}"
    print(row)
    
    # Net Debt
    row = f"  {'Net Debt':<25}"
    for year in years:
        value = metrics.net_debt.get(year)
        row += f"{format_number(value):>15}"
    print(row)
    
    # Invested Capital
    row = f"  {'Invested Capital':<25}"
    for year in years:
        value = metrics.invested_capital.get(year)
        row += f"{format_number(value):>15}"
    print(row)
    
    # Enterprise Value (current only)
    print()
    print(f"  Current Enterprise Value: {format_currency(metrics.enterprise_value)}")


def print_dividend_history(dividend):
    """Print dividend history section."""
    print_header("DIVIDEND HISTORY")
    
    if dividend.years_of_data == 0:
        print("  No dividend history available")
        return
    
    print_subheader("Annual Dividends Paid")
    
    years = sorted(dividend.annual_dividends.keys(), reverse=True)
    for year in years:
        value = dividend.annual_dividends[year]
        print(f"  {year}: {format_number(value)}")
    
    print_subheader("Dividend Analysis")
    print(f"  Years of Data:      {dividend.years_of_data}")
    print(f"  Current DPS:        ${dividend.current_annual_dps:.2f}" if dividend.current_annual_dps else "  Current DPS:        N/A")
    print(f"  Dividend CAGR:      {format_percent(dividend.dividend_cagr)}")
    print(f"  Has Dividend Cuts:  {'Yes' if dividend.has_dividend_cuts else 'No'}")
    print(f"  Payout Stable:      {'Yes' if dividend.payout_stable else 'No'}")


def print_collection_summary(result):
    """Print collection summary section."""
    print_header("COLLECTION SUMMARY")
    
    print(f"  Ticker:             {result.ticker}")
    print(f"  Company:            {result.company_name}")
    print(f"  Years Available:    {result.statements.years_available}")
    print(f"  Fiscal Periods:     {', '.join(result.statements.fiscal_periods)}")
    print(f"  Validation Status:  {result.validation_status.value.upper()}")
    print(f"  Quality Tier:       {result.quality_metrics.quality_tier.value.upper()}")
    print(f"  Data Source:        {result.data_source}")
    print(f"  API Calls Made:     {result.api_calls_made}")
    print(f"  From Cache:         {'Yes' if result.from_cache else 'No'}")
    print(f"  Collection Time:    {result.collection_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")


# =============================================================================
# PHASE 2 OUTPUT FUNCTIONS
# =============================================================================

def print_reconciliation_report(report):
    """Print reconciliation report section."""
    print_header("CROSS-STATEMENT RECONCILIATION")
    
    print_subheader("Summary")
    print(f"  Total Checks:      {report.total_checks}")
    print(f"  Passed:            {report.passed_checks}")
    print(f"  Minor Variances:   {report.minor_variances}")
    print(f"  Major Variances:   {report.major_variances}")
    print(f"  Cannot Reconcile:  {report.failed_checks}")
    print(f"  Score:             {format_percent(report.reconciliation_score)}")
    
    variances = [r for r in report.results if r.status.value != "reconciled"]
    if variances:
        print_subheader("Variances")
        for v in variances[:8]:
            print(f"  {v.metric_name} ({v.fiscal_year}): {format_percent(v.variance_percent)} [{v.status.value}]")


def print_outlier_report(report):
    """Print outlier detection report section."""
    print_header("OUTLIER DETECTION")
    
    print_subheader("Summary")
    print(f"  Values Analyzed:   {report.total_values_analyzed}")
    print(f"  Outliers Found:    {report.outlier_count}")
    print(f"  Outlier Rate:      {format_percent(report.outlier_percentage)}")
    print(f"  Mild:              {report.mild_outliers}")
    print(f"  Moderate:          {report.moderate_outliers}")
    print(f"  Extreme:           {report.extreme_outliers}")
    
    # Sort by severity (extreme first) then by absolute z-score
    severity_order = {"extreme": 0, "moderate": 1, "mild": 2}
    significant = [o for o in report.outliers if o.severity.value in ("moderate", "extreme")]
    significant.sort(key=lambda x: (severity_order.get(x.severity.value, 3), -abs(x.z_score or 0)))
    
    if significant:
        print_subheader("Significant Outliers")
        for o in significant[:8]:
            yoy = f", YoY: {format_percent(o.yoy_change)}" if o.yoy_change else ""
            severity_tag = " [EXTREME]" if o.severity.value == "extreme" else ""
            print(f"  {o.field_name} ({o.fiscal_year}): Z={o.z_score:.2f}{yoy}{severity_tag}")


def print_growth_report(report):
    """Print growth analysis report section."""
    print_header("GROWTH RATE ANALYSIS")
    
    if not report.metrics:
        print("  No growth metrics available")
        return
    
    key_metrics = [
        "total_revenue", "gross_profit", "operating_income", "net_income",
        "operating_cash_flow", "fcf_calculated", "total_assets", "total_equity"
    ]
    
    print_subheader("CAGR and Trend")
    header = f"  {'Metric':<22}{'CAGR':>10}{'Avg':>10}{'Trend':>18}"
    print(header)
    print("  " + "-" * 58)
    
    for metric_name in key_metrics:
        if metric_name in report.metrics:
            m = report.metrics[metric_name]
            cagr = format_percent(m.cagr) if m.cagr is not None else "N/A"
            avg = format_percent(m.average_growth) if m.average_growth is not None else "N/A"
            trend = m.trend.value.replace("_", " ").title()
            print(f"  {metric_name:<22}{cagr:>10}{avg:>10}{trend:>18}")


def print_supplementation_report(report):
    """Print data supplementation report."""
    if not report or report.gaps_filled == 0:
        return
    
    print_header("DATA SUPPLEMENTATION (Yahoo Finance)")
    
    print_subheader("Summary")
    print(f"  Gaps Filled:       {report.gaps_filled}")
    print(f"  Gaps Remaining:    {report.gaps_remaining}")
    
    if report.fields_supplemented:
        print_subheader("Supplemented Fields")
        for field, years in report.fields_supplemented.items():
            for year, sup_val in years.items():
                source = sup_val.source.value.replace("_", " ").title()
                conf = f"{sup_val.confidence*100:.0f}%"
                print(f"  {field} ({year}): {format_number(sup_val.value)} [{source}, {conf}]")
                if sup_val.notes:
                    print(f"    Note: {sup_val.notes}")
    
    if report.warnings:
        print_subheader("Warnings")
        for w in report.warnings:
            print(f"  ! {w}")


def print_validation_summary(summary):
    """Print Phase 2 validation summary section."""
    print_header("PHASE 2 VALIDATION SUMMARY")
    
    print_subheader("Quality Scores")
    print(f"  Reconciliation:    {format_percent(summary.reconciliation_score)}")
    print(f"  Consistency:       {format_percent(summary.consistency_score)}")
    print(f"  Completeness:      {format_percent(summary.completeness_score)}")
    print(f"  Overall Score:     {format_percent(summary.overall_score)}")
    print(f"  Quality Tier:      {summary.quality_tier.upper()}")
    
    if summary.warnings:
        print_subheader("Warnings")
        for w in summary.warnings:
            print(f"  - {w}")
    
    if summary.info:
        print_subheader("Info")
        for i in summary.info:
            print(f"  - {i}")


# =============================================================================
# PHASE 3 OUTPUT FUNCTIONS
# =============================================================================

def format_ratio_value(value, definition):
    """Format ratio value according to its definition."""
    if value is None:
        return "N/A"
    
    if definition.format_as_percent:
        return f"{value * 100:.2f}%"
    elif definition.format_as_days:
        return f"{value:.1f} days"
    else:
        return f"{value:.{definition.format_decimals}f}"


def get_assessment_symbol(assessment):
    """Get symbol for assessment level."""
    symbols = {
        RatioAssessment.EXCELLENT: "[E]",
        RatioAssessment.GOOD: "[G]",
        RatioAssessment.ACCEPTABLE: "[A]",
        RatioAssessment.WEAK: "[W]",
        RatioAssessment.CRITICAL: "[C]",
        RatioAssessment.NOT_APPLICABLE: "[-]",
    }
    return symbols.get(assessment, "[-]")


def print_ratio_category(category_name, category_analysis, years):
    """Print ratios for a single category."""
    print_subheader(f"{category_name.upper()} RATIOS")
    
    if not category_analysis.ratios:
        print("  No ratios calculated")
        return
    
    # Identify CAGR ratios (period-spanning metrics)
    cagr_ratios = {"revenue_cagr_5y", "net_income_cagr_5y", "fcf_cagr_5y", "dividend_cagr_5y"}
    
    # Header row with years
    year_cols = "  " + " ".join(f"{y[:4]:>10}" for y in years[:5])
    print(f"  {'Ratio':<28} {year_cols}  {'Latest':>8}  {'Trend':<12}")
    print("  " + "-" * 100)
    
    for ratio_name, series in category_analysis.ratios.items():
        if not series.values:
            continue
        
        # For CAGR ratios, show the value across all years (it's a period metric)
        is_cagr = ratio_name in cagr_ratios
        
        # Get values for each year
        values = []
        for i, y in enumerate(years[:5]):
            rv = series.values.get(y)
            if rv and rv.value is not None:
                formatted = format_ratio_value(rv.value, series.definition)
                values.append(f"{formatted:>10}")
            elif is_cagr and series.latest_value is not None:
                # For CAGR, show dashes for non-latest years (it's a period metric)
                if i == 0:
                    formatted = format_ratio_value(series.latest_value, series.definition)
                    values.append(f"{formatted:>10}")
                else:
                    values.append(f"{'---':>10}")
            else:
                values.append(f"{'N/A':>10}")
        
        value_str = " ".join(values)
        
        # Latest assessment
        assessment_str = get_assessment_symbol(series.latest_assessment)
        
        # Trend
        trend_str = series.trend.value if series.trend else "N/A"
        
        print(f"  {series.definition.name:<28} {value_str}  {assessment_str:>8}  {trend_str:<12}")
    
    # Category score
    print("  " + "-" * 100)
    print(f"  {'Category Score:':<28} {format_percent(category_analysis.category_score)}")
    
    if category_analysis.strengths:
        print(f"\n  Strengths:")
        for s in category_analysis.strengths[:3]:
            print(f"    + {s}")
    
    if category_analysis.weaknesses:
        print(f"\n  Weaknesses:")
        for w in category_analysis.weaknesses[:3]:
            print(f"    - {w}")


def print_ratio_analysis(result: RatioAnalysisResult):
    """Print complete Phase 3 ratio analysis."""
    print_header("PHASE 3: FINANCIAL RATIO ANALYSIS")
    
    years = result.fiscal_periods
    
    # Print each category
    print_ratio_category("Profitability", result.profitability, years)
    print()
    print_ratio_category("Leverage", result.leverage, years)
    print()
    print_ratio_category("Liquidity", result.liquidity, years)
    print()
    print_ratio_category("Efficiency", result.efficiency, years)
    print()
    print_ratio_category("Growth", result.growth, years)


def print_overall_assessment(result: RatioAnalysisResult):
    """Print overall financial health assessment."""
    print_header("OVERALL FINANCIAL ASSESSMENT")
    
    overall = result.overall_assessment
    
    print_subheader("Category Scores")
    print(f"  Profitability:     {format_percent(overall.profitability_score):>10}")
    print(f"  Leverage:          {format_percent(overall.leverage_score):>10}")
    print(f"  Liquidity:         {format_percent(overall.liquidity_score):>10}")
    print(f"  Efficiency:        {format_percent(overall.efficiency_score):>10}")
    
    print_subheader("Overall Score")
    print(f"  Score:             {format_percent(overall.overall_score):>10}")
    print(f"  Assessment:        {overall.overall_assessment.value.upper():>10}")
    
    if overall.key_strengths:
        print_subheader("Key Strengths")
        for s in overall.key_strengths[:5]:
            print(f"  + {s}")
    
    if overall.key_weaknesses:
        print_subheader("Key Weaknesses")
        for w in overall.key_weaknesses[:5]:
            print(f"  - {w}")
    
    if overall.key_risks:
        print_subheader("Key Risks")
        for r in overall.key_risks:
            print(f"  ! {r}")


def print_ratio_summary(result: RatioAnalysisResult):
    """Print condensed ratio analysis summary."""
    print_header("PHASE 3 RATIO ANALYSIS SUMMARY")
    
    overall = result.overall_assessment
    
    print_subheader("Category Scores")
    print(f"  Profitability:     {format_percent(overall.profitability_score):>10}  ({result.profitability.assessment_distribution})")
    print(f"  Leverage:          {format_percent(overall.leverage_score):>10}  ({result.leverage.assessment_distribution})")
    print(f"  Liquidity:         {format_percent(overall.liquidity_score):>10}  ({result.liquidity.assessment_distribution})")
    print(f"  Efficiency:        {format_percent(overall.efficiency_score):>10}  ({result.efficiency.assessment_distribution})")
    
    print()
    print(f"  Overall Score:     {format_percent(overall.overall_score):>10}")
    print(f"  Assessment:        {overall.overall_assessment.value.upper():>10}")
    print(f"  Ratios Calculated: {result.ratios_calculated}")


# =============================================================================
# PHASE 4 OUTPUT FUNCTIONS
# =============================================================================

def get_driver_symbol(driver: DuPontDriver) -> str:
    """Get compact symbol for DuPont driver."""
    symbols = {
        DuPontDriver.NET_PROFIT_MARGIN: "NPM",
        DuPontDriver.ASSET_TURNOVER: "AT",
        DuPontDriver.EQUITY_MULTIPLIER: "EM",
        DuPontDriver.TAX_BURDEN: "TAX",
        DuPontDriver.INTEREST_BURDEN: "INT",
        DuPontDriver.OPERATING_MARGIN: "OM",
        DuPontDriver.MIXED: "MIX",
    }
    return symbols.get(driver, "---")


def get_quality_symbol(quality: ROEQuality) -> str:
    """Get compact symbol for ROE quality rating."""
    symbols = {
        ROEQuality.HIGH_QUALITY: "[HQ]",
        ROEQuality.GOOD_QUALITY: "[GQ]",
        ROEQuality.MODERATE_QUALITY: "[MQ]",
        ROEQuality.LOW_QUALITY: "[LQ]",
        ROEQuality.LEVERAGE_DEPENDENT: "[LD]",
    }
    return symbols.get(quality, "[-]")


def get_trend_symbol(trend: TrendDirection) -> str:
    """Get compact symbol for trend direction."""
    symbols = {
        TrendDirection.IMPROVING: "improving",
        TrendDirection.STABLE: "stable",
        TrendDirection.DETERIORATING: "deteriorating",
        TrendDirection.VOLATILE: "volatile",
    }
    return symbols.get(trend, "---")


def print_three_factor_dupont(result: DuPontAnalysisResult):
    """Print 3-Factor DuPont decomposition table."""
    print_header("3-FACTOR DUPONT DECOMPOSITION")
    
    years = result.fiscal_periods[:5]
    
    print("  ROE = Net Profit Margin x Asset Turnover x Equity Multiplier")
    print()
    
    # Header
    year_cols = "  ".join(f"{y[:4]:>10}" for y in years)
    print(f"  {'Component':<24} {year_cols}    {'Trend':<12}")
    print("  " + "-" * 90)
    
    # Get values for each component
    components = [
        ("Net Profit Margin", "net_profit_margin", True),
        ("Asset Turnover", "asset_turnover", False),
        ("Equity Multiplier", "equity_multiplier", False),
    ]
    
    for name, attr, as_pct in components:
        values = []
        for y in years:
            tf = result.three_factor.get(y)
            if tf:
                comp = getattr(tf, attr)
                if comp.value is not None:
                    if as_pct:
                        values.append(f"{comp.value*100:>9.2f}%")
                    else:
                        values.append(f"{comp.value:>10.2f}")
                else:
                    values.append(f"{'N/A':>10}")
            else:
                values.append(f"{'N/A':>10}")
        
        value_str = "  ".join(values)
        
        # Get trend
        trend = result.component_trends.get(name)
        trend_str = get_trend_symbol(trend.trend_direction) if trend else "---"
        
        print(f"  {name:<24} {value_str}    {trend_str:<12}")
    
    print("  " + "-" * 90)
    
    # ROE row
    roe_values = []
    for y in years:
        tf = result.three_factor.get(y)
        if tf and tf.roe_calculated is not None:
            roe_values.append(f"{tf.roe_calculated*100:>9.2f}%")
        else:
            roe_values.append(f"{'N/A':>10}")
    
    roe_str = "  ".join(roe_values)
    roe_trend = result.component_trends.get("ROE")
    roe_trend_str = get_trend_symbol(roe_trend.trend_direction) if roe_trend else "---"
    
    print(f"  {'ROE (Calculated)':<24} {roe_str}    {roe_trend_str:<12}")
    
    # Reconciliation status
    print()
    recon_status = []
    for y in years:
        tf = result.three_factor.get(y)
        if tf:
            status = "OK" if tf.is_reconciled else "VAR"
            recon_status.append(f"{status:>10}")
        else:
            recon_status.append(f"{'---':>10}")
    
    print(f"  {'Reconciliation':<24} {'  '.join(recon_status)}")
    
    # Methodology note
    print()
    print("  Note: DuPont uses AVERAGE assets/equity for consistency with ROE calculation")
    print("        (ROE = NI / Avg Equity). Phase 3 ratios use point-in-time values.")


def print_five_factor_dupont(result: DuPontAnalysisResult):
    """Print 5-Factor DuPont decomposition table."""
    print_header("5-FACTOR DUPONT DECOMPOSITION")
    
    years = result.fiscal_periods[:5]
    
    print("  ROE = Tax Burden x Interest Burden x Operating Margin x Asset Turnover x Equity Multiplier")
    print()
    
    # Header
    year_cols = "  ".join(f"{y[:4]:>10}" for y in years)
    print(f"  {'Component':<24} {year_cols}    {'Trend':<12}")
    print("  " + "-" * 90)
    
    # Components: (display name, attribute name, format as percent)
    components = [
        ("Tax Burden", "tax_burden", True),
        ("Interest Burden", "interest_burden", True),
        ("Operating Margin", "operating_margin", True),
        ("Asset Turnover", "asset_turnover", False),
        ("Equity Multiplier", "equity_multiplier", False),
    ]
    
    for name, attr, as_pct in components:
        values = []
        for y in years:
            ff = result.five_factor.get(y)
            if ff:
                comp = getattr(ff, attr)
                if comp.value is not None:
                    if as_pct:
                        values.append(f"{comp.value*100:>9.2f}%")
                    else:
                        values.append(f"{comp.value:>10.2f}")
                else:
                    values.append(f"{'N/A':>10}")
            else:
                values.append(f"{'N/A':>10}")
        
        value_str = "  ".join(values)
        
        # Get trend
        trend = result.component_trends.get(name)
        trend_str = get_trend_symbol(trend.trend_direction) if trend else "---"
        
        print(f"  {name:<24} {value_str}    {trend_str:<12}")
    
    print("  " + "-" * 90)
    
    # Operational ROE (without leverage)
    op_roe_values = []
    for y in years:
        ff = result.five_factor.get(y)
        if ff and ff.operational_roe is not None:
            op_roe_values.append(f"{ff.operational_roe*100:>9.2f}%")
        else:
            op_roe_values.append(f"{'N/A':>10}")
    
    op_roe_str = "  ".join(op_roe_values)
    print(f"  {'Operational ROE':<24} {op_roe_str}")
    
    # Leverage contribution
    lev_values = []
    for y in years:
        ff = result.five_factor.get(y)
        if ff and ff.leverage_contribution is not None:
            lev_values.append(f"{ff.leverage_contribution*100:>+9.2f}%")
        else:
            lev_values.append(f"{'N/A':>10}")
    
    lev_str = "  ".join(lev_values)
    print(f"  {'Leverage Contribution':<24} {lev_str}")


def print_variance_analysis(result: DuPontAnalysisResult):
    """Print year-over-year ROE variance attribution."""
    print_header("ROE VARIANCE ATTRIBUTION")
    
    if not result.variance_analysis:
        print("  Insufficient data for variance analysis")
        return
    
    for va in result.variance_analysis:
        print_subheader(f"{va.prior_year} to {va.current_year}")
        
        # ROE change summary
        if va.roe_change is not None:
            direction = "increased" if va.roe_change > 0 else "decreased"
            print(f"  ROE {direction} from {format_percent(va.prior_roe)} to {format_percent(va.current_roe)}")
            print(f"  Total Change: {va.roe_change*100:+.2f} percentage points")
            print()
        
        # Attribution table
        print(f"  {'Factor':<24} {'Prior':>10} {'Current':>10} {'Change':>10} {'Contrib':>12} {'% of Change':>12}")
        print("  " + "-" * 80)
        
        # NPM contribution
        npm = va.npm_contribution
        print(f"  {'Net Profit Margin':<24} "
              f"{format_percent(npm.prior_value):>10} "
              f"{format_percent(npm.current_value):>10} "
              f"{npm.change*100 if npm.change else 0:>+9.2f}% "
              f"{npm.contribution_to_roe_change*100 if npm.contribution_to_roe_change else 0:>+11.2f}% "
              f"{npm.contribution_percentage if npm.contribution_percentage else 0:>11.1f}%")
        
        # AT contribution
        at = va.at_contribution
        print(f"  {'Asset Turnover':<24} "
              f"{at.prior_value if at.prior_value else 0:>10.2f} "
              f"{at.current_value if at.current_value else 0:>10.2f} "
              f"{at.change if at.change else 0:>+10.2f} "
              f"{at.contribution_to_roe_change*100 if at.contribution_to_roe_change else 0:>+11.2f}% "
              f"{at.contribution_percentage if at.contribution_percentage else 0:>11.1f}%")
        
        # EM contribution
        em = va.em_contribution
        print(f"  {'Equity Multiplier':<24} "
              f"{em.prior_value if em.prior_value else 0:>10.2f} "
              f"{em.current_value if em.current_value else 0:>10.2f} "
              f"{em.change if em.change else 0:>+10.2f} "
              f"{em.contribution_to_roe_change*100 if em.contribution_to_roe_change else 0:>+11.2f}% "
              f"{em.contribution_percentage if em.contribution_percentage else 0:>11.1f}%")
        
        print("  " + "-" * 80)
        
        # Totals
        print(f"  {'Total Explained':<24} {'':>10} {'':>10} {'':>10} "
              f"{va.total_explained*100 if va.total_explained else 0:>+11.2f}% "
              f"{'100.0%' if va.is_reconciled else 'VAR':>12}")
        
        # Primary driver
        print()
        print(f"  Primary Driver: {va.primary_driver.value}")
        print()


def print_quality_assessment(result: DuPontAnalysisResult):
    """Print ROE quality and sustainability assessment."""
    print_header("ROE QUALITY ASSESSMENT")
    
    qa = result.quality_assessment
    if not qa:
        print("  Quality assessment not available")
        return
    
    print_subheader("Quality Rating")
    print(f"  Rating:              {qa.quality_rating.value}")
    print(f"  Score:               {qa.quality_score:.0f}/100")
    print()
    print(f"  Operational Strength: {qa.operational_strength}")
    print(f"  Leverage Dependency:  {qa.leverage_dependency}")
    print(f"  Sustainability:       {qa.sustainability}")
    
    if qa.strengths:
        print_subheader("Strengths")
        for s in qa.strengths:
            print(f"  + {s}")
    
    if qa.concerns:
        print_subheader("Concerns")
        for c in qa.concerns:
            print(f"  - {c}")
    
    if qa.risks:
        print_subheader("Risks")
        for r in qa.risks:
            print(f"  ! {r}")


def print_dupont_summary(result: DuPontAnalysisResult):
    """Print condensed DuPont analysis summary."""
    print_header("PHASE 4 DUPONT ANALYSIS SUMMARY")
    
    print_subheader("ROE Overview")
    print(f"  Average ROE:         {format_percent(result.average_roe):>10}")
    print(f"  ROE Volatility:      {result.roe_volatility:.2f}" if result.roe_volatility else "  ROE Volatility:          N/A")
    print(f"  ROE Trend:           {result.roe_trend.value:>10}")
    print(f"  Primary Driver:      {result.primary_roe_driver.value:>10}")
    
    if result.quality_assessment:
        qa = result.quality_assessment
        print()
        print_subheader("Quality Assessment")
        print(f"  Quality Rating:      {qa.quality_rating.value}")
        print(f"  Quality Score:       {qa.quality_score:.0f}/100")
        print(f"  Sustainability:      {qa.sustainability}")


def print_dupont_analysis(result: DuPontAnalysisResult):
    """Print full DuPont analysis output."""
    print()
    print_three_factor_dupont(result)
    print()
    print_five_factor_dupont(result)
    print()
    print_variance_analysis(result)
    print()
    print_quality_assessment(result)


# =============================================================================
# PHASE 5 OUTPUT FUNCTIONS - DCF VALUATION
# =============================================================================

def print_historical_fcf(result: 'DCFValuationResult'):
    """Print historical FCF analysis."""
    print_header("HISTORICAL FREE CASH FLOW ANALYSIS")
    
    fcf = result.historical_fcf
    
    print_subheader("FCF by Year")
    for year in sorted(fcf.fcf_by_year.keys(), reverse=True):
        value = fcf.fcf_by_year[year]
        print(f"  {year}: {format_currency(value)}")
    
    print()
    print_subheader("FCF Statistics")
    print(f"  Years Analyzed:       {fcf.years_analyzed}")
    print(f"  Average FCF:          {format_currency(fcf.average_fcf)}")
    print(f"  Latest FCF:           {format_currency(fcf.latest_fcf)}")
    print(f"  FCF CAGR:             {format_percent(fcf.fcf_cagr) if fcf.fcf_cagr else 'N/A':>10}")
    print(f"  Average Growth:       {format_percent(fcf.average_growth) if fcf.average_growth else 'N/A':>10}")
    print(f"  Volatility (CV):      {fcf.coefficient_of_variation:.2f}" if fcf.coefficient_of_variation else "  Volatility (CV):          N/A")
    print(f"  FCF Quality:          {fcf.fcf_quality.value}")
    
    print()
    print_subheader("FCF Margins")
    for year in sorted(fcf.fcf_margins.keys(), reverse=True):
        margin = fcf.fcf_margins[year]
        print(f"  {year}: {format_percent(margin)}")
    if fcf.average_fcf_margin:
        print(f"  Average: {format_percent(fcf.average_fcf_margin)}")


def print_growth_analysis(result: 'DCFValuationResult'):
    """Print growth rate derivation analysis."""
    print_header("GROWTH RATE DERIVATION")
    
    ga = result.growth_analysis
    
    print_subheader("Growth Rate Estimates")
    
    # Historical CAGR
    hc = ga.historical_cagr
    status = "[USED]" if hc.is_available else "[N/A]"
    rate = format_percent(hc.rate) if hc.rate is not None else "N/A"
    print(f"  Historical FCF CAGR:     {rate:>10}  {status}")
    if hc.description:
        print(f"    {hc.description}")
    
    # Sustainable Growth
    sg = ga.sustainable_growth
    status = "[USED]" if sg.is_available else "[N/A]"
    rate = format_percent(sg.rate) if sg.rate is not None else "N/A"
    print(f"  Sustainable Growth:      {rate:>10}  {status}")
    if sg.description:
        print(f"    {sg.description}")
    
    # Analyst Consensus
    ac = ga.analyst_consensus
    status = "[USED]" if ac.is_available else "[N/A]"
    rate = format_percent(ac.rate) if ac.rate is not None else "N/A"
    print(f"  Analyst Consensus:       {rate:>10}  {status}")
    if ac.description:
        print(f"    {ac.description}")
    
    print()
    print_subheader("Selected Growth Rates")
    print(f"  Projection Growth (Y1-5): {format_percent(ga.projection_growth_rate)}")
    print(f"  Terminal Growth:          {format_percent(ga.terminal_growth_rate)}")
    print(f"  Selection Method:         {ga.selection_method}")
    if ga.growth_cap_applied and ga.original_uncapped_rate is not None:
        print(f"  ⚠ Cap Applied:            {format_percent(ga.original_uncapped_rate)} → {format_percent(ga.projection_growth_rate)}")
        print(f"    Rationale: {ga.selection_rationale}")


def print_wacc_calculation(result: 'DCFValuationResult'):
    """Print WACC calculation breakdown."""
    print_header("WEIGHTED AVERAGE COST OF CAPITAL (WACC)")
    
    wacc = result.wacc_calculation
    
    print_subheader("Cost of Equity (CAPM)")
    coe = wacc.cost_of_equity
    print(f"  Risk-Free Rate (Rf):      {format_percent(coe.risk_free_rate)}")
    print(f"  Beta:                     {coe.beta:.2f} ({coe.beta_source})")
    print(f"  Equity Risk Premium:      {format_percent(coe.equity_risk_premium)}")
    print(f"  Cost of Equity:           {format_percent(coe.cost_of_equity)}")
    print(f"  Formula: Rf + Beta * ERP = {coe.risk_free_rate:.2%} + {coe.beta:.2f} * {coe.equity_risk_premium:.2%}")
    
    print()
    print_subheader("Cost of Debt")
    cod = wacc.cost_of_debt
    print(f"  Interest Expense:         {format_currency(cod.interest_expense) if cod.interest_expense else 'N/A'}")
    print(f"  Total Debt:               {format_currency(cod.total_debt) if cod.total_debt else 'N/A'}")
    print(f"  Pre-Tax Cost of Debt:     {format_percent(cod.pre_tax_cost)}")
    print(f"  Tax Rate:                 {format_percent(cod.tax_rate)}")
    print(f"  After-Tax Cost of Debt:   {format_percent(cod.after_tax_cost)}")
    print(f"  Calculation Method:       {cod.calculation_method}")
    
    print()
    print_subheader("Capital Structure & WACC")
    print(f"  Market Cap:               {format_currency(wacc.market_cap) if wacc.market_cap else 'N/A'}")
    print(f"  Total Debt:               {format_currency(wacc.total_debt) if wacc.total_debt else 'N/A'}")
    print(f"  Equity Weight:            {format_percent(wacc.equity_weight)}")
    print(f"  Debt Weight:              {format_percent(wacc.debt_weight)}")
    print_line("-", 50)
    print(f"  WACC:                     {format_percent(wacc.wacc_constrained)}")
    if wacc.constraint_applied:
        print(f"  (Constrained from {format_percent(wacc.wacc)})")


def print_dcf_projection(result: 'DCFValuationResult'):
    """Print DCF projection details."""
    print_header("DCF PROJECTION (5-YEAR)")
    
    proj = result.dcf_projection
    
    print_subheader("Projected Free Cash Flows")
    print(f"  {'Year':<6} {'FCF':>14} {'Growth':>10} {'Discount':>10} {'PV':>14}")
    print_line("-", 60)
    
    for yp in proj.yearly_projections:
        print(f"  {yp.year:<6} {format_currency(yp.fcf):>14} {format_percent(yp.growth_rate):>10} "
              f"{yp.discount_factor:>10.4f} {format_currency(yp.present_value):>14}")
    
    print_line("-", 60)
    print(f"  {'Sum PV FCF':<6} {'':<14} {'':<10} {'':<10} {format_currency(proj.sum_of_pv_fcf):>14}")
    
    print()
    print_subheader("Terminal Value (Gordon Growth Model)")
    tv = proj.terminal_value
    print(f"  Final Year FCF (Y5):      {format_currency(tv.final_year_fcf)}")
    print(f"  Terminal Growth Rate:     {format_percent(tv.terminal_growth_rate)}")
    print(f"  Terminal Year FCF (Y6):   {format_currency(tv.terminal_year_fcf)}")
    print(f"  Terminal Value:           {format_currency(tv.terminal_value)}")
    print(f"  Discount Factor (Y5):     {tv.discount_factor:.4f}")
    print(f"  PV of Terminal Value:     {format_currency(tv.present_value)}")
    
    print()
    print_subheader("Enterprise Value")
    print(f"  Sum of PV (FCF Y1-5):     {format_currency(proj.sum_of_pv_fcf)}")
    print(f"  PV of Terminal Value:     {format_currency(proj.pv_terminal_value)}")
    print_line("-", 50)
    print(f"  Enterprise Value:         {format_currency(proj.enterprise_value)}")
    print(f"  Terminal Value % of EV:   {format_percent(proj.terminal_value_pct)}")


def print_valuation_summary(result: 'DCFValuationResult'):
    """Print final valuation summary."""
    print_header("DCF VALUATION SUMMARY")
    
    print_subheader("Equity Bridge")
    print(f"  Enterprise Value:         {format_currency(result.enterprise_value)}")
    print(f"  Less: Net Debt:           {format_currency(result.net_debt)}")
    print(f"  Plus: Cash:               {format_currency(result.cash_and_equivalents)}")
    print_line("-", 50)
    print(f"  Equity Value:             {format_currency(result.equity_value)}")
    
    print()
    print_subheader("Per Share Value")
    print(f"  Shares Outstanding:       {format_number(result.shares_outstanding)}")
    print(f"  Intrinsic Value/Share:    ${result.intrinsic_value_per_share:.2f}")
    
    print()
    print_subheader("Market Comparison")
    if result.current_price:
        print(f"  Current Market Price:     ${result.current_price:.2f}")
        print(f"  Upside/(Downside):        {format_percent(result.upside_downside_pct)}")
        print(f"  Valuation Signal:         {result.valuation_signal.value.upper()}")
    else:
        print("  Current price not available")


def print_sensitivity_analysis(result: 'DCFValuationResult'):
    """Print sensitivity analysis matrix and scenarios."""
    print_header("SENSITIVITY ANALYSIS")
    
    sa = result.sensitivity_analysis
    matrix = sa.sensitivity_matrix
    
    print_subheader("Growth Rate vs WACC Sensitivity Matrix")
    print("  Intrinsic Value per Share at various Growth and WACC combinations:")
    print()
    
    # Header row
    header = "  Growth \\ WACC"
    for wacc in matrix.discount_rates:
        header += f"  {format_percent(wacc):>8}"
    print(header)
    print_line("-", len(header))
    
    # Data rows
    for i, growth in enumerate(matrix.growth_rates):
        row = f"  {format_percent(growth):>12}"
        for j, value in enumerate(matrix.values[i]):
            marker = " *" if (i == matrix.base_growth_idx and j == matrix.base_wacc_idx) else "  "
            row += f"  ${value:>6.0f}{marker}"
        print(row)
    
    print()
    print("  * = Base case")
    
    print()
    print_subheader("Scenario Analysis")
    print(f"  {'Scenario':<12} {'Growth':>10} {'Term.G':>10} {'EV':>14} {'Value/Shr':>12} {'Upside':>10}")
    print_line("-", 76)
    
    for scenario in [sa.bear_case, sa.base_case, sa.bull_case]:
        if scenario:
            print(f"  {scenario.scenario.value.upper():<12} "
                  f"{format_percent(scenario.growth_rate):>10} "
                  f"{format_percent(scenario.terminal_growth_rate):>10} "
                  f"{format_currency(scenario.enterprise_value):>14} "
                  f"${scenario.intrinsic_value_per_share:>10.2f} "
                  f"{format_percent(scenario.upside_downside_pct):>10}")
    
    print()
    print_subheader("Value Range")
    print(f"  Minimum Value:            ${sa.min_value:.2f}")
    print(f"  Maximum Value:            ${sa.max_value:.2f}")
    print(f"  Range:                    {format_percent(sa.value_range_pct)}")


def print_dcf_warnings(result: 'DCFValuationResult'):
    """Print validation warnings and assumptions."""
    if result.validation_warnings:
        print_subheader("Warnings")
        for warning in result.validation_warnings:
            print(f"  - {warning}")
    
    if result.validation_errors:
        print_subheader("Errors")
        for error in result.validation_errors:
            print(f"  ! {error}")


def print_dcf_valuation(result: 'DCFValuationResult'):
    """Print full DCF valuation output."""
    print()
    print_historical_fcf(result)
    print()
    print_growth_analysis(result)
    print()
    print_wacc_calculation(result)
    print()
    print_dcf_projection(result)
    print()
    print_valuation_summary(result)
    print()
    print_sensitivity_analysis(result)
    print()
    if result.validation_warnings or result.validation_errors:
        print_dcf_warnings(result)


def print_dcf_summary(result: 'DCFValuationResult'):
    """Print condensed DCF valuation summary."""
    print_header("PHASE 5 DCF VALUATION SUMMARY")
    
    print_subheader("Key Inputs")
    print(f"  Base FCF:                 {format_currency(result.historical_fcf.latest_fcf)}")
    print(f"  Projection Growth:        {format_percent(result.growth_analysis.projection_growth_rate)}")
    print(f"  Terminal Growth:          {format_percent(result.growth_analysis.terminal_growth_rate)}")
    print(f"  WACC:                     {format_percent(result.wacc_calculation.wacc_constrained)}")
    
    print()
    print_subheader("Valuation Result")
    print(f"  Enterprise Value:         {format_currency(result.enterprise_value)}")
    print(f"  Equity Value:             {format_currency(result.equity_value)}")
    print(f"  Intrinsic Value/Share:    ${result.intrinsic_value_per_share:.2f}")
    
    if result.current_price:
        print()
        print_subheader("Market Comparison")
        print(f"  Current Price:            ${result.current_price:.2f}")
        print(f"  Upside/(Downside):        {format_percent(result.upside_downside_pct)}")
        print(f"  Signal:                   {result.valuation_signal.value.upper()}")


# =============================================================================
# PHASE 6: DDM VALUATION OUTPUT
# =============================================================================

def print_ddm_applicability(result: 'DDMValuationResult'):
    """Print DDM applicability assessment."""
    print_header("DDM APPLICABILITY ASSESSMENT")
    
    app = result.applicability
    print()
    print_subheader("Applicability Status")
    print(f"  Status:                   {app.status.value.upper()}")
    print(f"  Years of Dividends:       {app.years_of_dividends}")
    print(f"  Has Positive Dividends:   {'Yes' if app.has_positive_dividends else 'No'}")
    print(f"  No Dividend Cuts:         {'Yes' if app.has_no_cuts else 'No'}")
    print(f"  Stable Payout:            {'Yes' if app.has_stable_payout else 'No'}")
    
    if app.current_payout_ratio:
        print(f"  Current Payout Ratio:     {format_percent(app.current_payout_ratio)}")
    if app.average_payout_ratio:
        print(f"  Average Payout Ratio:     {format_percent(app.average_payout_ratio)}")
    
    if app.applicability_reasons:
        print()
        print_subheader("Applicability Criteria Met")
        for reason in app.applicability_reasons:
            print(f"  + {reason}")
    
    if app.exclusion_reasons:
        print()
        print_subheader("Concerns")
        for reason in app.exclusion_reasons:
            print(f"  - {reason}")


def print_ddm_historical_dividends(result: 'DDMValuationResult'):
    """Print historical dividend analysis."""
    print_header("HISTORICAL DIVIDEND ANALYSIS")
    
    hist = result.historical_dividends
    
    # DPS by year
    print()
    print_subheader("Dividend Per Share by Year")
    for year, dps in sorted(hist.dps_by_year.items(), reverse=True):
        print(f"  {year}: ${dps:.4f}")
    
    # Statistics
    print()
    print_subheader("DPS Statistics")
    print(f"  Years Analyzed:           {hist.years_analyzed}")
    print(f"  Current DPS:              ${hist.current_dps:.4f}")
    print(f"  Average DPS:              ${hist.average_dps:.4f}")
    if hist.dps_cagr is not None:
        print(f"  DPS CAGR:                 {format_percent(hist.dps_cagr)}")
    if hist.average_growth is not None:
        print(f"  Average Growth:           {format_percent(hist.average_growth)}")
    if hist.dps_coefficient_of_variation is not None:
        print(f"  Volatility (CV):          {hist.dps_coefficient_of_variation:.2f}")
    print(f"  Dividend Quality:         {hist.dividend_quality.value}")
    
    # Payout ratios
    if hist.payout_ratio_by_year:
        print()
        print_subheader("Payout Ratios by Year")
        for year, ratio in sorted(hist.payout_ratio_by_year.items(), reverse=True):
            print(f"  {year}: {format_percent(ratio)}")
        if hist.average_payout_ratio:
            print(f"  Average: {format_percent(hist.average_payout_ratio)}")
        print(f"  Trend: {hist.payout_trend}")


def print_ddm_growth_analysis(result: 'DDMValuationResult'):
    """Print DDM growth rate derivation."""
    print_header("DIVIDEND GROWTH RATE DERIVATION")
    
    growth = result.growth_analysis
    
    print()
    print_subheader("Growth Rate Estimates")
    
    # Historical CAGR
    hist = growth.historical_cagr
    status = "[USED]" if hist.is_available else "[N/A]"
    if hist.rate is not None:
        print(f"  Historical DPS CAGR:      {format_percent(hist.rate):>8}  {status}")
    else:
        print(f"  Historical DPS CAGR:           N/A  {status}")
    if hist.description:
        print(f"    {hist.description}")
    
    # Sustainable growth
    sust = growth.sustainable_growth
    status = "[USED]" if sust.is_available else "[N/A]"
    if sust.rate is not None:
        print(f"  Sustainable Growth:       {format_percent(sust.rate):>8}  {status}")
    else:
        print(f"  Sustainable Growth:            N/A  {status}")
    if sust.description:
        print(f"    {sust.description}")
    
    # Earnings-based
    earn = growth.earnings_based
    status = "[USED]" if earn.is_available else "[N/A]"
    if earn.rate is not None:
        print(f"  Earnings-Based:           {format_percent(earn.rate):>8}  {status}")
    else:
        print(f"  Earnings-Based:                N/A  {status}")
    if earn.description:
        print(f"    {earn.description}")
    
    # Selected rates
    print()
    print_subheader("Selected Growth Rates")
    print(f"  Projection Growth (Y1-5): {format_percent(growth.projection_growth_rate)}")
    print(f"  Terminal Growth:          {format_percent(growth.terminal_growth_rate)}")
    print(f"  Selection Method:         {growth.selection_method}")
    
    if growth.growth_cap_applied and growth.original_uncapped_rate is not None:
        print(f"  ⚠ Cap Applied:            {format_percent(growth.original_uncapped_rate)} → {format_percent(growth.projection_growth_rate)}")
        if hasattr(growth, 'selection_rationale') and growth.selection_rationale:
            print(f"    Rationale: {growth.selection_rationale}")


def print_ddm_cost_of_equity(result: 'DDMValuationResult'):
    """Print cost of equity calculation."""
    print_header("COST OF EQUITY (CAPM)")
    
    ke = result.cost_of_equity
    
    print()
    print_subheader("CAPM Components")
    print(f"  Risk-Free Rate (Rf):      {format_percent(ke.risk_free_rate)}")
    print(f"  Beta:                     {ke.beta:.2f} ({ke.beta_source})")
    print(f"  Equity Risk Premium:      {format_percent(ke.equity_risk_premium)}")
    print(f"  Cost of Equity (Ke):      {format_percent(ke.cost_of_equity)}")
    print(f"  Formula: Rf + Beta * ERP = {format_percent(ke.risk_free_rate)} + {ke.beta:.2f} * {format_percent(ke.equity_risk_premium)}")


def print_ddm_projection(result: 'DDMValuationResult'):
    """Print DDM dividend projections."""
    print_header("DDM PROJECTION (5-YEAR)")
    
    proj = result.ddm_projection
    
    # Yearly projections
    print()
    print_subheader("Projected Dividends Per Share")
    print(f"  {'Year':<6} {'DPS':>12} {'Growth':>10} {'Discount':>10} {'PV':>12}")
    print("-" * 60)
    
    for yp in proj.yearly_projections:
        print(f"  {yp.year:<6} ${yp.dps:>10.4f} {format_percent(yp.growth_rate):>10} {yp.discount_factor:>10.4f} ${yp.present_value:>10.4f}")
    
    print("-" * 60)
    print(f"  Sum PV Dividends{' ' * 34}${proj.sum_of_pv_dividends:>10.4f}")
    
    # Terminal value
    tv = proj.terminal_value
    print()
    print_subheader("Terminal Value (Gordon Growth Model)")
    print(f"  Final Year DPS (Y5):      ${tv.final_year_dps:.4f}")
    print(f"  Terminal Growth Rate:     {format_percent(tv.terminal_growth_rate)}")
    print(f"  Terminal Year DPS (Y6):   ${tv.terminal_year_dps:.4f}")
    print(f"  Terminal Value:           ${tv.terminal_value:.2f}")
    print(f"  Discount Factor (Y5):     {tv.discount_factor:.4f}")
    print(f"  PV of Terminal Value:     ${tv.present_value:.2f}")
    
    # Intrinsic value
    print()
    print_subheader("Intrinsic Value Per Share")
    print(f"  Sum of PV (DPS Y1-5):     ${proj.sum_of_pv_dividends:.4f}")
    print(f"  PV of Terminal Value:     ${proj.pv_terminal_value:.2f}")
    print("-" * 50)
    print(f"  Intrinsic Value/Share:    ${proj.intrinsic_value_per_share:.2f}")
    print(f"  Terminal Value % of IV:   {format_percent(proj.terminal_value_pct)}")


def print_ddm_market_comparison(result: 'DDMValuationResult'):
    """Print DDM market comparison."""
    print_header("DDM VALUATION SUMMARY")
    
    print()
    print_subheader("Intrinsic Value")
    print(f"  DDM Intrinsic Value/Share:  ${result.intrinsic_value_per_share:.2f}")
    
    if result.current_price:
        print()
        print_subheader("Market Comparison")
        print(f"  Current Market Price:       ${result.current_price:.2f}")
        print(f"  Upside/(Downside):          {format_percent(result.upside_downside_pct)}")
    
    if result.current_dividend_yield and result.implied_dividend_yield:
        print()
        print_subheader("Dividend Yield Analysis")
        print(f"  Current Dividend Yield:     {format_percent(result.current_dividend_yield)}")
        print(f"  Implied Dividend Yield:     {format_percent(result.implied_dividend_yield)}")


def print_ddm_sensitivity(result: 'DDMValuationResult'):
    """Print DDM sensitivity analysis."""
    print_header("DDM SENSITIVITY ANALYSIS")
    
    sens = result.sensitivity_analysis
    matrix = sens.sensitivity_matrix
    
    # Sensitivity matrix
    print()
    print_subheader("Growth Rate vs Cost of Equity Sensitivity Matrix")
    print("  Intrinsic Value per Share at various Growth and Ke combinations:")
    print()
    
    # Header row
    header = f"  {'Growth \\ Ke':<12}"
    for ke in matrix.discount_rates:
        header += f"{format_percent(ke):>10}"
    print(header)
    print("-" * (12 + len(matrix.discount_rates) * 10 + 2))
    
    # Data rows
    for i, growth in enumerate(matrix.growth_rates):
        row = f"  {format_percent(growth):>12}"
        for j, val in enumerate(matrix.values[i]):
            marker = " *" if i == matrix.base_growth_idx and j == matrix.base_ke_idx else "  "
            if val > 0:
                row += f"${val:>7.0f}{marker}"
            else:
                row += f"{'N/A':>9} "
        print(row)
    
    print()
    print("  * = Base case")
    
    # Scenario analysis
    print()
    print_subheader("Scenario Analysis")
    print(f"  {'Scenario':<10} {'Growth':>10} {'Term.G':>10} {'Value/Shr':>12} {'Upside':>10}")
    print("-" * 60)
    
    for scenario in [sens.bear_case, sens.base_case, sens.bull_case]:
        if scenario:
            print(f"  {scenario.scenario.value.upper():<10} "
                  f"{format_percent(scenario.growth_rate):>10} "
                  f"{format_percent(scenario.terminal_growth_rate):>10} "
                  f"${scenario.intrinsic_value_per_share:>10.2f} "
                  f"{format_percent(scenario.upside_downside_pct):>10}")
    
    # Value range
    print()
    print_subheader("Value Range")
    print(f"  Minimum Value:            ${sens.min_value:.2f}")
    print(f"  Maximum Value:            ${sens.max_value:.2f}")
    print(f"  Range:                    {format_percent(sens.value_range_pct)}")


def print_dcf_ddm_reconciliation(result: 'DDMValuationResult'):
    """Print DCF vs DDM reconciliation."""
    recon = result.dcf_reconciliation
    
    if recon.dcf_intrinsic_value is None:
        return
    
    print_header("DCF VS DDM RECONCILIATION")
    
    print()
    print_subheader("Intrinsic Value Comparison")
    print(f"  DCF Intrinsic Value:      ${recon.dcf_intrinsic_value:.2f}")
    print(f"  DDM Intrinsic Value:      ${recon.ddm_intrinsic_value:.2f}")
    print(f"  Value Difference:         ${recon.value_difference:.2f}")
    print(f"  Percentage Difference:    {format_percent(recon.percentage_difference)}")
    print(f"  Convergence Status:       {recon.convergence_status.upper().replace('_', ' ')}")
    
    print()
    print_subheader("Methodology Comparison")
    if recon.dcf_growth_rate is not None and recon.ddm_growth_rate is not None:
        print(f"  DCF Projection Growth:    {format_percent(recon.dcf_growth_rate)}")
        print(f"  DDM Projection Growth:    {format_percent(recon.ddm_growth_rate)}")
    if recon.dcf_discount_rate is not None and recon.ddm_discount_rate is not None:
        print(f"  DCF Discount (WACC):      {format_percent(recon.dcf_discount_rate)}")
        print(f"  DDM Discount (Ke):        {format_percent(recon.ddm_discount_rate)}")
    
    if recon.explanation:
        print()
        print_subheader("Explanation")
        for exp in recon.explanation:
            print(f"  - {exp}")


def print_ddm_warnings(result: 'DDMValuationResult'):
    """Print DDM validation warnings."""
    if result.validation_warnings:
        print()
        print_subheader("Warnings")
        for warning in result.validation_warnings:
            print(f"  ! {warning}")
    
    if result.validation_errors:
        print()
        print_subheader("Errors")
        for error in result.validation_errors:
            print(f"  X {error}")


def print_ddm_valuation(result: 'DDMValuationResult'):
    """Print complete DDM valuation output."""
    if not result.is_applicable:
        print_header("DDM VALUATION")
        print()
        print("  DDM is not applicable for this company.")
        if result.applicability.exclusion_reasons:
            print()
            print_subheader("Reasons")
            for reason in result.applicability.exclusion_reasons:
                print(f"  - {reason}")
        return
    
    print_ddm_applicability(result)
    print_ddm_historical_dividends(result)
    print_ddm_growth_analysis(result)
    print_ddm_cost_of_equity(result)
    print_ddm_projection(result)
    print_ddm_market_comparison(result)
    print_ddm_sensitivity(result)
    print_dcf_ddm_reconciliation(result)
    print_ddm_warnings(result)


def print_ddm_summary(result: 'DDMValuationResult'):
    """Print condensed DDM summary."""
    print_header("DDM VALUATION SUMMARY")
    
    if not result.is_applicable:
        print()
        print(f"  Status: DDM Not Applicable")
        if result.applicability.exclusion_reasons:
            print(f"  Reason: {result.applicability.exclusion_reasons[0]}")
        return
    
    print()
    print_subheader("Dividend Metrics")
    print(f"  Current DPS:              ${result.historical_dividends.current_dps:.4f}")
    print(f"  DPS CAGR:                 {format_percent(result.historical_dividends.dps_cagr)}")
    print(f"  Dividend Quality:         {result.historical_dividends.dividend_quality.value}")
    
    print()
    print_subheader("Valuation Inputs")
    print(f"  Projection Growth:        {format_percent(result.growth_analysis.projection_growth_rate)}")
    print(f"  Terminal Growth:          {format_percent(result.growth_analysis.terminal_growth_rate)}")
    print(f"  Cost of Equity:           {format_percent(result.cost_of_equity.cost_of_equity)}")
    
    print()
    print_subheader("Valuation Result")
    print(f"  DDM Intrinsic Value:      ${result.intrinsic_value_per_share:.2f}")
    
    if result.current_price:
        print(f"  Current Price:            ${result.current_price:.2f}")
        print(f"  Upside/(Downside):        {format_percent(result.upside_downside_pct)}")
    
    # DCF comparison if available
    if result.dcf_reconciliation.dcf_intrinsic_value:
        print()
        print_subheader("DCF Comparison")
        print(f"  DCF Intrinsic Value:      ${result.dcf_reconciliation.dcf_intrinsic_value:.2f}")
        print(f"  Difference:               {format_percent(result.dcf_reconciliation.percentage_difference)}")
        print(f"  Convergence:              {result.dcf_reconciliation.convergence_status.upper().replace('_', ' ')}")


# =============================================================================
# PHASE 7: MULTIPLES VALUATION OUTPUT
# =============================================================================

def print_multiples_current(result: 'MultiplesValuationResult'):
    """Print current multiples."""
    print_header("CURRENT VALUATION MULTIPLES")
    
    print()
    print_subheader("Price Multiples")
    
    analyses = [
        ("P/E Ratio", result.pe_analysis),
        ("P/B Ratio", result.pb_analysis),
        ("P/S Ratio", result.ps_analysis),
        ("P/FCF Ratio", result.pfcf_analysis),
    ]
    
    for name, analysis in analyses:
        if analysis.current_valid:
            print(f"  {name:<15} {analysis.current_value:>8.2f}")
        else:
            print(f"  {name:<15}      N/A")
    
    print()
    print_subheader("Enterprise Value Multiples")
    
    ev_analyses = [
        ("EV/EBITDA", result.ev_ebitda_analysis),
        ("EV/Revenue", result.ev_revenue_analysis),
    ]
    
    for name, analysis in ev_analyses:
        if analysis.current_valid:
            print(f"  {name:<15} {analysis.current_value:>8.2f}")
        else:
            print(f"  {name:<15}      N/A")


def print_multiples_historical(result: 'MultiplesValuationResult'):
    """Print historical multiple comparison."""
    print_header("HISTORICAL MULTIPLE COMPARISON")
    
    all_analyses = [
        ("P/E Ratio", result.pe_analysis),
        ("P/B Ratio", result.pb_analysis),
        ("P/S Ratio", result.ps_analysis),
        ("P/FCF Ratio", result.pfcf_analysis),
        ("EV/EBITDA", result.ev_ebitda_analysis),
        ("EV/Revenue", result.ev_revenue_analysis),
    ]
    
    print()
    print_subheader("Current vs Historical Average")
    print(f"  {'Multiple':<12} {'Current':>10} {'Avg':>10} {'Premium':>10} {'Assessment':<20}")
    print("-" * 70)
    
    for name, analysis in all_analyses:
        if analysis.current_valid and analysis.historical_average:
            current = f"{analysis.current_value:.2f}"
            avg = f"{analysis.historical_average:.2f}"
            premium = format_percent(analysis.premium_to_average) if analysis.premium_to_average else "N/A"
            assessment = analysis.assessment.value.replace("_", " ").title()
            print(f"  {name:<12} {current:>10} {avg:>10} {premium:>10} {assessment:<20}")
    
    print()
    print_subheader("Historical Ranges (5-Year)")
    print(f"  {'Multiple':<12} {'Min':>10} {'Max':>10} {'Std Dev':>10} {'Trend':<15}")
    print("-" * 65)
    
    for name, analysis in all_analyses:
        if analysis.years_of_data >= 2:
            min_val = f"{analysis.historical_min:.2f}" if analysis.historical_min else "N/A"
            max_val = f"{analysis.historical_max:.2f}" if analysis.historical_max else "N/A"
            std_val = f"{analysis.historical_std:.2f}" if analysis.historical_std else "N/A"
            trend = analysis.trend.value if analysis.trend else "N/A"
            print(f"  {name:<12} {min_val:>10} {max_val:>10} {std_val:>10} {trend:<15}")


def print_multiples_implied_values(result: 'MultiplesValuationResult'):
    """Print implied fair values from each multiple."""
    print_header("IMPLIED FAIR VALUES")
    
    print()
    print_subheader("Implied Value from Historical Averages")
    print(f"  {'Multiple':<12} {'Implied Value':>15} {'vs Current':>12}")
    print("-" * 45)
    
    current_price = result.current_price
    
    all_analyses = [
        ("P/E Ratio", result.pe_analysis),
        ("P/B Ratio", result.pb_analysis),
        ("P/S Ratio", result.ps_analysis),
        ("P/FCF Ratio", result.pfcf_analysis),
        ("EV/EBITDA", result.ev_ebitda_analysis),
        ("EV/Revenue", result.ev_revenue_analysis),
    ]
    
    for name, analysis in all_analyses:
        if analysis.implied_value_from_average:
            implied = f"${analysis.implied_value_from_average:.2f}"
            if current_price:
                upside = (analysis.implied_value_from_average - current_price) / current_price
                vs_current = format_percent(upside)
            else:
                vs_current = "N/A"
            print(f"  {name:<12} {implied:>15} {vs_current:>12}")
    
    # Composite
    comp = result.composite_valuation
    if comp.average_implied_value:
        print("-" * 45)
        print(f"  {'AVERAGE':<12} ${comp.average_implied_value:>13.2f} {format_percent(comp.composite_upside):>12}")
        print(f"  {'MEDIAN':<12} ${comp.median_implied_value:>13.2f}")


def print_multiples_composite(result: 'MultiplesValuationResult'):
    """Print composite valuation assessment."""
    print_header("COMPOSITE VALUATION ASSESSMENT")
    
    comp = result.composite_valuation
    
    print()
    print_subheader("Multiple Scores (Negative = Undervalued)")
    for multiple, score in comp.multiple_scores.items():
        bar_len = int(abs(score) * 20)
        if score >= 0:
            bar = "+" * bar_len
            print(f"  {multiple:<12} {score:>+6.2f}  |{bar}")
        else:
            bar = "-" * bar_len
            print(f"  {multiple:<12} {score:>+6.2f}  {bar}|")
    
    print()
    print_subheader("Composite Result")
    print(f"  Composite Score:          {comp.composite_score:>+.2f}")
    print(f"  Overall Assessment:       {comp.overall_assessment.value.replace('_', ' ').upper()}")
    print(f"  Multiples Used:           {comp.multiples_used}")
    print(f"  Confidence:               {format_percent(comp.confidence)}")
    
    if comp.average_implied_value:
        print()
        print_subheader("Implied Fair Value")
        print(f"  Average Implied Value:    ${comp.average_implied_value:.2f}")
        print(f"  Median Implied Value:     ${comp.median_implied_value:.2f}")
        if result.current_price:
            print(f"  Current Price:            ${result.current_price:.2f}")
            print(f"  Upside/(Downside):        {format_percent(comp.composite_upside)}")


def print_cross_model_comparison(result: 'MultiplesValuationResult'):
    """Print cross-model valuation comparison."""
    comp = result.cross_model_comparison
    
    print_header("CROSS-MODEL VALUATION COMPARISON")
    
    print()
    print_subheader("Intrinsic Values by Model")
    print(f"  {'Model':<20} {'Intrinsic Value':>15} {'Upside':>12}")
    print("-" * 55)
    
    if comp.dcf_value:
        upside = format_percent(comp.dcf_upside) if comp.dcf_upside else "N/A"
        print(f"  {'DCF (FCF-Based)':<20} ${comp.dcf_value:>13.2f} {upside:>12}")
    
    if comp.ddm_value:
        upside = format_percent(comp.ddm_upside) if comp.ddm_upside else "N/A"
        print(f"  {'DDM (Dividend-Based)':<20} ${comp.ddm_value:>13.2f} {upside:>12}")
    
    if comp.multiples_average:
        upside = format_percent(comp.multiples_upside) if comp.multiples_upside else "N/A"
        print(f"  {'Multiples (Relative)':<20} ${comp.multiples_average:>13.2f} {upside:>12}")
    
    print("-" * 55)
    if comp.average_intrinsic_value:
        upside = format_percent(comp.consensus_upside) if comp.consensus_upside else "N/A"
        print(f"  {'CONSENSUS AVERAGE':<20} ${comp.average_intrinsic_value:>13.2f} {upside:>12}")
    
    if comp.current_price:
        print(f"\n  Current Market Price:     ${comp.current_price:.2f}")
    
    print()
    print_subheader("Model Agreement")
    print(f"  Models Bullish:           {comp.models_bullish}")
    print(f"  Models Neutral:           {comp.models_neutral}")
    print(f"  Models Bearish:           {comp.models_bearish}")
    print(f"  Consensus Direction:      {comp.consensus_direction.upper()}")


def print_multiples_warnings(result: 'MultiplesValuationResult'):
    """Print multiples validation warnings."""
    if result.validation_warnings:
        print()
        print_subheader("Warnings")
        for warning in result.validation_warnings:
            print(f"  ! {warning}")
    
    if result.validation_errors:
        print()
        print_subheader("Errors")
        for error in result.validation_errors:
            print(f"  X {error}")


def print_multiples_valuation(result: 'MultiplesValuationResult'):
    """Print complete multiples valuation output."""
    print_multiples_current(result)
    print_multiples_historical(result)
    print_multiples_implied_values(result)
    print_multiples_composite(result)
    print_cross_model_comparison(result)
    print_multiples_warnings(result)


def print_multiples_summary(result: 'MultiplesValuationResult'):
    """Print condensed multiples summary."""
    print_header("MULTIPLES VALUATION SUMMARY")
    
    print()
    print_subheader("Key Multiples")
    
    multiples_display = [
        ("P/E", result.pe_analysis),
        ("EV/EBITDA", result.ev_ebitda_analysis),
        ("P/B", result.pb_analysis),
    ]
    
    for name, analysis in multiples_display:
        if analysis.current_valid:
            premium = format_percent(analysis.premium_to_average) if analysis.premium_to_average else "N/A"
            print(f"  {name:<12} Current: {analysis.current_value:>6.1f}  Premium: {premium}")
    
    print()
    print_subheader("Valuation Result")
    print(f"  Implied Fair Value:       ${result.implied_fair_value:.2f}")
    print(f"  Current Price:            ${result.current_price:.2f}")
    print(f"  Upside/(Downside):        {format_percent(result.upside_downside_pct)}")
    print(f"  Assessment:               {result.overall_assessment.value.replace('_', ' ').upper()}")
    
    # Cross-model comparison
    comp = result.cross_model_comparison
    if comp.consensus_upside is not None:
        print()
        print_subheader("Cross-Model Consensus")
        print(f"  Consensus Value:          ${comp.average_intrinsic_value:.2f}")
        print(f"  Consensus Upside:         {format_percent(comp.consensus_upside)}")
        print(f"  Direction:                {comp.consensus_direction.upper()}")


# =============================================================================
# PHASE 8: ACCURACY CHECKER OUTPUT
# =============================================================================

def get_status_symbol(status: 'VerificationStatus') -> str:
    """Get display symbol for verification status."""
    symbols = {
        VerificationStatus.PASSED: "PASS",
        VerificationStatus.WARNING: "WARN",
        VerificationStatus.FAILED: "FAIL",
        VerificationStatus.SKIPPED: "SKIP",
    }
    return symbols.get(status, "----")


def print_accuracy_category(summary: 'CategorySummary', category_name: str):
    """Print a single accuracy category."""
    print_subheader(f"{category_name}")
    print(f"  Checks: {summary.total_checks}  |  "
          f"Passed: {summary.passed}  |  "
          f"Warnings: {summary.warnings}  |  "
          f"Failed: {summary.failed}")
    print(f"  Pass Rate: {summary.pass_rate*100:.1f}%  |  "
          f"Confidence: {summary.confidence_score*100:.1f}%")
    
    # Show individual checks
    if summary.checks:
        print()
        for check in summary.checks:
            status_sym = get_status_symbol(check.status)
            print(f"    [{status_sym}] {check.check_name}: {check.message}")


def print_accuracy_input_integrity(result: 'AccuracyCheckResult'):
    """Print input integrity checks."""
    print_header("INPUT INTEGRITY VERIFICATION")
    print()
    print_accuracy_category(result.input_integrity, "Data Integrity Checks")


def print_accuracy_calculations(result: 'AccuracyCheckResult'):
    """Print calculation verification results."""
    print_header("CALCULATION ACCURACY VERIFICATION")
    print()
    print_accuracy_category(result.calculation_accuracy, "Calculation Re-verification")


def print_accuracy_cross_phase(result: 'AccuracyCheckResult'):
    """Print cross-phase consistency results."""
    print_header("CROSS-PHASE CONSISTENCY")
    print()
    print_accuracy_category(result.cross_phase_consistency, "Data Consistency Across Phases")


def print_accuracy_valuation(result: 'AccuracyCheckResult'):
    """Print valuation consistency results."""
    print_header("VALUATION MODEL CONSISTENCY")
    print()
    print_accuracy_category(result.valuation_consistency, "Valuation Input Consistency")
    
    if result.valuation_input_comparisons:
        print()
        print_subheader("Input Parameter Comparison")
        print(f"  {'Parameter':<25} {'DCF':>12} {'DDM':>12} {'Multiples':>12} {'Consistent':<10}")
        print("-" * 80)
        
        for comp in result.valuation_input_comparisons:
            dcf_val = f"{comp.dcf_value:.4f}" if comp.dcf_value else "N/A"
            ddm_val = f"{comp.ddm_value:.4f}" if comp.ddm_value else "N/A"
            mult_val = f"{comp.multiples_value:.4f}" if comp.multiples_value else "N/A"
            
            # Format based on parameter type
            if comp.parameter == "current_price":
                dcf_val = f"${comp.dcf_value:.2f}" if comp.dcf_value else "N/A"
                ddm_val = f"${comp.ddm_value:.2f}" if comp.ddm_value else "N/A"
                mult_val = f"${comp.multiples_value:.2f}" if comp.multiples_value else "N/A"
            elif comp.parameter in ["risk_free_rate", "equity_risk_premium"]:
                dcf_val = f"{comp.dcf_value*100:.2f}%" if comp.dcf_value else "N/A"
                ddm_val = f"{comp.ddm_value*100:.2f}%" if comp.ddm_value else "N/A"
            elif comp.parameter == "beta":
                dcf_val = f"{comp.dcf_value:.2f}" if comp.dcf_value else "N/A"
                ddm_val = f"{comp.ddm_value:.2f}" if comp.ddm_value else "N/A"
            
            consistent = "YES" if comp.is_consistent else "NO"
            print(f"  {comp.parameter:<25} {dcf_val:>12} {ddm_val:>12} {mult_val:>12} {consistent:<10}")


def print_accuracy_range(result: 'AccuracyCheckResult'):
    """Print range validity results."""
    print_header("RANGE VALIDITY CHECKS")
    print()
    print_accuracy_category(result.range_validity, "Value Range Validation")


def print_accuracy_methodology(result: 'AccuracyCheckResult'):
    """Print methodology compliance results."""
    print_header("METHODOLOGY COMPLIANCE")
    print()
    print_accuracy_category(result.methodology_compliance, "Implementation Plan Compliance")


def print_accuracy_summary_section(result: 'AccuracyCheckResult'):
    """Print overall accuracy summary."""
    print_header("ACCURACY CHECK SUMMARY")
    
    print()
    print_subheader("Category Scores")
    
    categories = [
        ("Input Integrity", result.input_integrity),
        ("Calculation Accuracy", result.calculation_accuracy),
        ("Cross-Phase Consistency", result.cross_phase_consistency),
        ("Valuation Consistency", result.valuation_consistency),
        ("Range Validity", result.range_validity),
        ("Methodology Compliance", result.methodology_compliance),
    ]
    
    print(f"  {'Category':<25} {'Checks':>8} {'Passed':>8} {'Score':>10}")
    print("-" * 55)
    
    for name, cat in categories:
        score = f"{cat.confidence_score*100:.1f}%"
        print(f"  {name:<25} {cat.total_checks:>8} {cat.passed:>8} {score:>10}")
    
    print("-" * 55)
    print(f"  {'OVERALL':<25} {result.total_checks:>8} {result.total_passed:>8} "
          f"{result.overall_confidence*100:.1f}%")
    
    print()
    print_subheader("Overall Assessment")
    print(f"  Total Checks:             {result.total_checks}")
    print(f"  Passed:                   {result.total_passed}")
    print(f"  Warnings:                 {result.total_warnings}")
    print(f"  Failed:                   {result.total_failed}")
    print(f"  Overall Pass Rate:        {result.overall_pass_rate*100:.1f}%")
    print(f"  Overall Confidence:       {result.overall_confidence*100:.1f}%")
    print(f"  Confidence Level:         {result.confidence_level.value.upper().replace('_', ' ')}")
    print(f"  Analysis Reliable:        {'YES' if result.is_reliable else 'NO'}")


def print_accuracy_issues(result: 'AccuracyCheckResult'):
    """Print critical issues and warnings."""
    if result.critical_issues:
        print()
        print_subheader("Critical Issues")
        for issue in result.critical_issues:
            print(f"  [FAIL] {issue}")
    
    if result.warnings_list:
        print()
        print_subheader("Warnings")
        for i, warning in enumerate(result.warnings_list[:10]):
            print(f"  [WARN] {warning}")
        if len(result.warnings_list) > 10:
            print(f"  ... and {len(result.warnings_list) - 10} more warnings")


def print_accuracy_recommendation(result: 'AccuracyCheckResult'):
    """Print final recommendation."""
    print()
    print_subheader("Recommendation")
    print(f"  {result.recommendation}")


def print_accuracy_check(result: 'AccuracyCheckResult'):
    """Print complete accuracy check output."""
    print_accuracy_input_integrity(result)
    print_accuracy_calculations(result)
    print_accuracy_cross_phase(result)
    print_accuracy_valuation(result)
    print_accuracy_range(result)
    print_accuracy_methodology(result)
    print_accuracy_summary_section(result)
    print_accuracy_issues(result)
    print_accuracy_recommendation(result)


def print_accuracy_summary(result: 'AccuracyCheckResult'):
    """Print condensed accuracy summary."""
    print_header("ACCURACY VERIFICATION SUMMARY")
    
    print()
    print_subheader("Quick Assessment")
    
    categories = [
        ("Input Integrity", result.input_integrity),
        ("Calculations", result.calculation_accuracy),
        ("Cross-Phase", result.cross_phase_consistency),
        ("Valuation", result.valuation_consistency),
        ("Ranges", result.range_validity),
        ("Methodology", result.methodology_compliance),
    ]
    
    for name, cat in categories:
        status = "PASS" if cat.failed == 0 else "FAIL" if cat.failed > 0 else "WARN"
        print(f"  {name:<20} [{status}] {cat.passed}/{cat.total_checks} checks passed")
    
    print()
    print_subheader("Overall Result")
    print(f"  Confidence Level:         {result.confidence_level.value.upper().replace('_', ' ')}")
    print(f"  Overall Score:            {result.overall_confidence*100:.1f}%")
    print(f"  Analysis Reliable:        {'YES' if result.is_reliable else 'NO'}")
    
    if result.critical_issues:
        print()
        print_subheader("Critical Issues")
        for issue in result.critical_issues[:3]:
            print(f"  [!] {issue}")
        if len(result.critical_issues) > 3:
            print(f"  ... and {len(result.critical_issues) - 3} more issues")
    
    print()
    print_subheader("Recommendation")
    # Truncate recommendation for summary
    rec = result.recommendation
    if len(rec) > 100:
        rec = rec[:97] + "..."
    print(f"  {rec}")


# =============================================================================
# PHASE 9 OUTPUT FUNCTIONS - INVESTMENT MEMO
# =============================================================================

def print_memo_header(result: 'InvestmentMemoResult'):
    """Print memo generation header."""
    print_header("INVESTMENT MEMO GENERATION")
    print()
    
    memo = result.memo
    print(f"  Company:            {memo.company_name} ({memo.ticker})")
    print(f"  Analysis Date:      {memo.analysis_date}")
    print(f"  Analyst:            {memo.analyst}")
    print(f"  Model Used:         {result.model_used}")


def print_memo_recommendation_box(result: 'InvestmentMemoResult'):
    """Print recommendation summary box."""
    print()
    print_subheader("Recommendation Summary")
    
    rec = result.memo.recommendation
    print()
    print(f"  {'Recommendation:':<20} {rec.recommendation.to_display()}")
    print(f"  {'Conviction:':<20} {rec.conviction.value.upper()}")
    if rec.current_price:
        print(f"  {'Current Price:':<20} ${rec.current_price:.2f}")
    if rec.target_price:
        print(f"  {'Target Price:':<20} ${rec.target_price:.2f}")
    if rec.upside_potential:
        print(f"  {'Upside Potential:':<20} {rec.upside_potential*100:+.1f}%")


def print_memo_section(title: str, content: str, max_preview: int = 500):
    """Print a memo section with optional preview truncation."""
    print()
    print_subheader(title)
    
    # Clean up content for display
    clean_content = content.strip()
    
    if len(clean_content) <= max_preview:
        print(f"  {clean_content}")
    else:
        preview = clean_content[:max_preview].rsplit(' ', 1)[0]
        print(f"  {preview}...")
        remaining = len(clean_content) - len(preview)
        print(f"  [+{remaining} more characters]")


def print_memo_sections(result: 'InvestmentMemoResult'):
    """Print all memo sections."""
    memo = result.memo
    
    print_memo_section("Executive Summary", memo.executive_summary.content)
    print_memo_section("Company Overview", memo.company_overview.content, 300)
    print_memo_section("Financial Analysis", memo.financial_analysis.content, 400)
    print_memo_section("Valuation Analysis", memo.valuation_analysis.content, 400)
    print_memo_section("Risk Assessment", memo.risk_assessment.content, 300)
    print_memo_section("Investment Thesis", memo.investment_thesis.content, 300)


def print_memo_outputs(result: 'InvestmentMemoResult'):
    """Print output file paths."""
    print()
    print_subheader("Generated Files")
    
    if result.json_path:
        print(f"  JSON:     {result.json_path}")
    if result.md_path:
        print(f"  Markdown: {result.md_path}")
    if result.pdf_path:
        print(f"  PDF:      {result.pdf_path}")


def print_memo_metadata(result: 'InvestmentMemoResult'):
    """Print generation metadata."""
    print()
    print_subheader("Generation Metadata")
    
    print(f"  Generation Time:    {result.generation_duration_seconds:.2f}s")
    print(f"  Tokens Used:        {result.llm_tokens_used}")


def print_memo_full(result: 'InvestmentMemoResult'):
    """Print complete memo output."""
    print_memo_header(result)
    print_memo_recommendation_box(result)
    print_memo_sections(result)
    print_memo_outputs(result)
    print_memo_metadata(result)


def print_memo_summary(result: 'InvestmentMemoResult'):
    """Print condensed memo summary."""
    print_header("INVESTMENT MEMO SUMMARY")
    
    memo = result.memo
    rec = memo.recommendation
    print()
    print(f"  {memo.company_name} ({memo.ticker})")
    
    rec_str = rec.recommendation.to_display()
    target_str = f"${rec.target_price:.2f}" if rec.target_price else "N/A"
    upside_str = f"{rec.upside_potential*100:+.1f}%" if rec.upside_potential else "N/A"
    print(f"  Recommendation: {rec_str} | Target: {target_str} | Upside: {upside_str}")
    print()
    print(f"  Generated in {result.generation_duration_seconds:.1f}s | Tokens: {result.llm_tokens_used}")
    
    if result.json_path:
        print()
        print(f"  Files saved to: {result.json_path.parent}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fundamental Analyst Agent - Phases 1-9"
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        default="AAPL",
        help="Stock ticker symbol (default: AAPL)"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh (bypass cache)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )
    parser.add_argument(
        "--phase1-only",
        action="store_true",
        help="Run Phase 1 only (skip validation and ratio analysis)"
    )
    parser.add_argument(
        "--phase2-only",
        action="store_true",
        help="Run Phases 1-2 only (skip ratio analysis)"
    )
    parser.add_argument(
        "--phase3-only",
        action="store_true",
        help="Run Phases 1-3 only (skip DuPont analysis)"
    )
    parser.add_argument(
        "--phase4-only",
        action="store_true",
        help="Run Phases 1-4 only (skip DCF valuation)"
    )
    parser.add_argument(
        "--phase5-only",
        action="store_true",
        help="Run Phases 1-5 only (skip DDM valuation)"
    )
    parser.add_argument(
        "--phase6-only",
        action="store_true",
        help="Run Phases 1-6 only (skip Multiples valuation)"
    )
    parser.add_argument(
        "--phase7-only",
        action="store_true",
        help="Run Phases 1-7 only (skip Accuracy verification)"
    )
    parser.add_argument(
        "--phase8-only",
        action="store_true",
        help="Run Phases 1-8 only (skip LLM memo generation)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key for Phase 9 memo generation"
    )
    parser.add_argument(
        "--no-supplement",
        action="store_true",
        help="Disable Yahoo Finance data supplementation"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Initialize collector (with or without supplementation)
    try:
        collector = DataCollector(enable_supplementation=not args.no_supplement)
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nTo get a free API key:")
        print("  1. Visit https://www.alphavantage.co/support/#api-key")
        print("  2. Register for a free API key")
        print("  3. Set environment variable: export ALPHA_VANTAGE_API_KEY=your_key")
        sys.exit(1)
    
    # Phase 1: Collect data (includes automatic supplementation if enabled)
    print(f"\nPhase 1: Collecting data for {args.ticker.upper()}...")
    print("This may take up to 60 seconds for fresh data (rate limiting).\n")
    
    result = collector.collect(args.ticker, force_refresh=args.refresh)
    
    # Print Phase 1 results
    if not args.quiet:
        print_company_profile(result.company_profile)
        print_quality_metrics(result.quality_metrics)
        
        # Print supplementation report if any gaps were filled
        if result.supplementation_report and result.supplementation_report.gaps_filled > 0:
            print_supplementation_report(result.supplementation_report)
        
        print_financial_statement(
            result.statements.income_statement, 
            "INCOME STATEMENT (5 Years)"
        )
        print_financial_statement(
            result.statements.balance_sheet, 
            "BALANCE SHEET (5 Years)"
        )
        print_financial_statement(
            result.statements.cash_flow, 
            "CASH FLOW STATEMENT (5 Years)"
        )
        print_derived_metrics(result.derived_metrics, result.statements.fiscal_periods)
        print_dividend_history(result.dividend_history)
    
    print_collection_summary(result)
    
    # Save Phase 1 JSON
    output_path = collector.save_to_json(result)
    print(f"\n  Data saved to: {output_path}")
    
    # Phase 2: Validation (unless --phase1-only)
    if not args.phase1_only and result.is_valid:
        print()
        print_line()
        print("  Phase 1 Complete - Starting Phase 2")
        print_line()
        
        validator = Phase2Validator()
        validated = validator.validate(result)
        
        if not args.quiet:
            print_reconciliation_report(validated.reconciliation_report)
            print_outlier_report(validated.outlier_report)
            print_growth_report(validated.growth_report)
        
        print_validation_summary(validated.validation_summary)
        
        # Save Phase 2 report
        report_path = validator.save_report(validated)
        print(f"\n  Validation report saved to: {report_path}")
        
        # Phase 3: Ratio Analysis (unless --phase2-only)
        if not args.phase2_only and validated.is_valid:
            print()
            print_line()
            print("  Phase 2 Complete - Starting Phase 3")
            print_line()
            
            analyzer = Phase3Analyzer()
            ratio_result = analyzer.analyze(result)
            
            if not args.quiet:
                print_ratio_analysis(ratio_result)
                print()
                print_overall_assessment(ratio_result)
            else:
                print_ratio_summary(ratio_result)
            
            # Save Phase 3 report
            ratio_path = analyzer.save_report(ratio_result)
            print(f"\n  Ratio analysis saved to: {ratio_path}")
            
            # Phase 4: DuPont Analysis (unless --phase3-only)
            if not args.phase3_only and ratio_result.is_valid:
                print()
                print_line()
                print("  Phase 3 Complete - Starting Phase 4")
                print_line()
                
                dupont_analyzer = Phase4Analyzer()
                dupont_result = dupont_analyzer.analyze(result)
                
                if not args.quiet:
                    print_dupont_analysis(dupont_result)
                else:
                    print_dupont_summary(dupont_result)
                
                # Save Phase 4 report
                dupont_path = dupont_analyzer.save_report(dupont_result)
                print(f"\n  DuPont analysis saved to: {dupont_path}")
                
                # Phase 5: DCF Valuation (unless --phase4-only or DCF not available)
                if not args.phase4_only and DCF_AVAILABLE and dupont_result.is_valid:
                    print()
                    print_line()
                    print("  Phase 4 Complete - Starting Phase 5")
                    print_line()
                    
                    dcf_valuator = Phase5Valuator()
                    dcf_result = dcf_valuator.value(result)
                    
                    if not args.quiet:
                        print_dcf_valuation(dcf_result)
                    else:
                        print_dcf_summary(dcf_result)
                    
                    # Save Phase 5 report
                    dcf_path = dcf_valuator.save_report(dcf_result)
                    print(f"\n  DCF valuation saved to: {dcf_path}")
                    
                    # Phase 6: DDM Valuation (unless --phase5-only or DDM not available)
                    if not args.phase5_only and DDM_AVAILABLE and dcf_result.is_valid:
                        print()
                        print_line()
                        print("  Phase 5 Complete - Starting Phase 6")
                        print_line()
                        
                        ddm_valuator = Phase6Valuator()
                        ddm_result = ddm_valuator.value(result, dcf_result)
                        
                        if not args.quiet:
                            print_ddm_valuation(ddm_result)
                        else:
                            print_ddm_summary(ddm_result)
                        
                        # Save Phase 6 report
                        if ddm_result.is_applicable:
                            ddm_path = ddm_valuator.save_report(ddm_result)
                            print(f"\n  DDM valuation saved to: {ddm_path}")
                        
                        print()
                        print_line()
                        print(f"  Phase 6 DDM Valuation Complete - Starting Phase 7")
                        if ddm_result.is_applicable:
                            print(f"  DDM Intrinsic Value: ${ddm_result.intrinsic_value_per_share:.2f}")
                            print(f"  DCF Intrinsic Value: ${dcf_result.intrinsic_value_per_share:.2f}")
                            if ddm_result.current_price:
                                print(f"  Current Price: ${ddm_result.current_price:.2f}")
                        else:
                            print(f"  DDM Status: Not Applicable")
                            print(f"  DCF Intrinsic Value: ${dcf_result.intrinsic_value_per_share:.2f}")
                        print_line()
                        print()
                        
                        # Phase 7: Multiples Valuation (unless --phase6-only or MULTIPLES not available)
                        if not args.phase6_only and MULTIPLES_AVAILABLE:
                            multiples_valuator = Phase7Valuator()
                            multiples_result = multiples_valuator.value(result, dcf_result, ddm_result)
                            
                            # Print Phase 7 output
                            if not args.quiet:
                                print_multiples_valuation(multiples_result)
                            else:
                                print_multiples_summary(multiples_result)
                            
                            # Save Phase 7 report
                            if multiples_result.is_valid:
                                multiples_path = multiples_valuator.save_report(multiples_result)
                                print(f"\n  Multiples valuation saved to: {multiples_path}")
                            
                            print()
                            print_line()
                            print(f"  Phase 7 Multiples Valuation Complete - Ready for Phase 8")
                            print(f"  Implied Fair Value: ${multiples_result.implied_fair_value:.2f}")
                            print(f"  Assessment: {multiples_result.overall_assessment.value.replace('_', ' ').upper()}")
                            
                            # Cross-model summary
                            comp = multiples_result.cross_model_comparison
                            if comp.average_intrinsic_value:
                                print(f"  Cross-Model Consensus: ${comp.average_intrinsic_value:.2f}")
                                print(f"  Consensus Direction: {comp.consensus_direction.upper()}")
                            print_line()
                            print()
                            
                            # Phase 8: Accuracy Verification (unless --phase7-only or ACCURACY not available)
                            if not args.phase7_only and ACCURACY_AVAILABLE:
                                accuracy_checker = Phase8AccuracyChecker()
                                accuracy_result = accuracy_checker.check(
                                    result,
                                    validated,
                                    ratio_result,
                                    dupont_result,
                                    dcf_result,
                                    ddm_result,
                                    multiples_result,
                                )
                                
                                # Print Phase 8 output
                                if not args.quiet:
                                    print_accuracy_check(accuracy_result)
                                else:
                                    print_accuracy_summary(accuracy_result)
                                
                                # Save Phase 8 report
                                accuracy_path = accuracy_checker.save_report(accuracy_result)
                                print(f"\n  Accuracy report saved to: {accuracy_path}")
                                
                                print()
                                print_line()
                                print(f"  Phase 8 Accuracy Verification Complete - Ready for Phase 9 (LLM Memo)")
                                print(f"  Overall Confidence: {accuracy_result.overall_confidence*100:.1f}%")
                                print(f"  Confidence Level: {accuracy_result.confidence_level.value.upper().replace('_', ' ')}")
                                print(f"  Analysis Reliable: {'YES' if accuracy_result.is_reliable else 'NO'}")
                                print_line()
                                print()
                                
                                # Phase 9: LLM Investment Memo (unless --phase8-only or MEMO not available)
                                if not args.phase8_only and MEMO_AVAILABLE:
                                    # Get API key from argument, environment, or use built-in default
                                    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
                                    # Note: If api_key is None, Phase9MemoGenerator will use the built-in default
                                    
                                    memo_generator = Phase9MemoGenerator(api_key)
                                    memo_result = memo_generator.generate(
                                        result,
                                        validated,
                                        ratio_result,
                                        dupont_result,
                                        dcf_result,
                                        ddm_result,
                                        multiples_result,
                                        accuracy_result,
                                    )
                                    
                                    # Print Phase 9 output
                                    if memo_result.is_valid:
                                        if not args.quiet:
                                            print_memo_full(memo_result)
                                        else:
                                            print_memo_summary(memo_result)
                                        
                                        # Save report
                                        report_path = memo_generator.save_report(memo_result)
                                        print(f"\n  Memo generation report saved to: {report_path}")
                                    else:
                                        print()
                                        print_header("MEMO GENERATION FAILED")
                                        print(f"  Status: {memo_result.status.value}")
                                        print(f"  Error: {memo_result.error_message}")
                                    
                                    print()
                                    print_line()
                                    print("=" * 80)
                                    print(f"  PHASE 9 INVESTMENT MEMO COMPLETE")
                                    print("=" * 80)
                                    if memo_result.is_valid:
                                        rec = memo_result.memo.recommendation
                                        print(f"  Recommendation: {rec.recommendation.to_display()}")
                                        if rec.target_price:
                                            print(f"  Target Price: ${rec.target_price:.2f}")
                                        if rec.upside_potential:
                                            print(f"  Upside: {rec.upside_potential*100:+.1f}%")
                                        print(f"  Tokens Used: {memo_result.llm_tokens_used}")
                                    print("=" * 80)
                                    print()
                                    
                                    return 0 if memo_result.is_valid else 1
                                    
                                elif not MEMO_AVAILABLE:
                                    print()
                                    print_line()
                                    print(f"  Phase 8 Accuracy Verification Complete")
                                    print(f"  Note: Phase 9 Memo Generation not available (module not found)")
                                    print_line()
                                    print()
                                    
                                    return 0 if accuracy_result.is_reliable else 1
                                else:
                                    # --phase8-only specified
                                    return 0 if accuracy_result.is_reliable else 1
                            elif not ACCURACY_AVAILABLE:
                                print()
                                print_line()
                                print(f"  Phase 7 Multiples Valuation Complete")
                                print(f"  Note: Phase 8 Accuracy Verification not available (module not found)")
                                print_line()
                                print()
                                
                                return 0 if multiples_result.is_valid else 1
                            else:
                                # --phase7-only specified
                                return 0 if multiples_result.is_valid else 1
                        elif not MULTIPLES_AVAILABLE:
                            print()
                            print_line()
                            print(f"  Phase 6 DDM Valuation Complete")
                            print(f"  Note: Phase 7 Multiples Valuation not available (module not found)")
                            print_line()
                            print()
                            
                            return 0 if ddm_result.is_valid or not ddm_result.is_applicable else 1
                        else:
                            # --phase6-only specified
                            return 0 if ddm_result.is_valid or not ddm_result.is_applicable else 1
                    elif not DDM_AVAILABLE:
                        print()
                        print_line()
                        print(f"  Phase 5 DCF Valuation Complete")
                        print(f"  Note: Phase 6 DDM Valuation not available (module not found)")
                        print(f"  DCF Intrinsic Value: ${dcf_result.intrinsic_value_per_share:.2f}")
                        print_line()
                        print()
                        
                        return 0 if dcf_result.is_valid else 1
                    else:
                        print()
                        print_line()
                        print(f"  Phase 5 DCF Valuation Complete - Ready for Phase 6")
                        print(f"  Intrinsic Value: ${dcf_result.intrinsic_value_per_share:.2f}")
                        if dcf_result.current_price:
                            print(f"  Current Price: ${dcf_result.current_price:.2f}")
                            print(f"  Upside/(Downside): {dcf_result.upside_downside_pct:.1%}")
                        print(f"  Signal: {dcf_result.valuation_signal.value.upper()}")
                        print_line()
                        print()
                        
                        return 0 if dcf_result.is_valid else 1
                elif not DCF_AVAILABLE:
                    print()
                    print_line()
                    print("  Phase 4 DuPont Analysis Complete")
                    print("  Note: Phase 5 DCF Valuation not available (module not found)")
                    print_line()
                    print()
                    
                    return 0 if dupont_result.is_valid else 1
                else:
                    print()
                    print_line()
                    print(f"  Phase 4 DuPont Analysis Complete - Ready for Phase 5")
                    print(f"  ROE Quality: {dupont_result.quality_assessment.quality_rating.value}")
                    print(f"  Primary Driver: {dupont_result.primary_roe_driver.value}")
                    print(f"  Average ROE: {dupont_result.average_roe*100:.1f}%")
                    print_line()
                    print()
                    
                    return 0 if dupont_result.is_valid else 1
            else:
                print()
                print_line()
                print(f"  Phase 3 Ratio Analysis Complete - Ready for Phase 4 (DuPont)")
                print(f"  Overall Assessment: {ratio_result.overall_assessment.overall_assessment.value.upper()}")
                print(f"  Overall Score: {ratio_result.overall_assessment.overall_score:.1%}")
                print_line()
                print()
                
                return 0 if ratio_result.is_valid else 1
        else:
            print()
            print_line()
            print(f"  Phase 2 Validation Complete")
            print(f"  Valid for Ratio Analysis: {'Yes' if validated.is_valid else 'No'}")
            print_line()
            print()
            
            return 0 if validated.is_valid else 1
    else:
        print()
        print_line()
        print("  Phase 1 Data Acquisition Complete")
        print_line()
        print()
        
        return 0 if result.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())