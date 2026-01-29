#!/usr/bin/env python3
"""
Professional Report Generator for Technical Analysis Pipeline

MSc AI Agents in Asset Management - Track B: Technical Analyst Agent

Generates publication-quality reports in multiple native formats:
    - HTML: Interactive web report with SVG visualizations
    - PDF: Native PDF using reportlab (not conversion)
    - JSON: Comprehensive machine-readable structured data
    - Markdown: Documentation-ready format

Author: Tamer
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

VERSION = "1.0.0"


# =============================================================================
# HTML REPORT GENERATOR
# =============================================================================

def generate_html_report(
    symbol: str,
    phase1_output: Any,
    phase2_output: Any,
    output_path: Path
) -> None:
    """Generate professional HTML report with embedded SVG visualizations."""
    p1 = phase1_output
    p2 = phase2_output
    analysis = p2.current_analysis
    quality = p1.quality
    profile = p1.profile
    levels = analysis.key_levels
    
    # Extract nested objects
    tail_risk = profile.tail_risk
    stats = profile.statistical_tests
    vol_profile = profile.volatility_profile
    
    # Color mappings
    signal_colors = {
        "STRONG_BUY": "#10b981", "BUY": "#34d399",
        "NEUTRAL": "#6b7280", "SELL": "#f87171", "STRONG_SELL": "#ef4444"
    }
    quality_colors = {
        "EXCELLENT": "#10b981", "GOOD": "#34d399",
        "ACCEPTABLE": "#fbbf24", "POOR": "#f87171", "UNUSABLE": "#ef4444"
    }
    
    sig_color = signal_colors.get(analysis.overall_signal.value, "#6b7280")
    qual_color = quality_colors.get(quality.grade.value, "#6b7280")
    
    # Generate confidence gauge SVG
    conf_pct = analysis.overall_confidence * 100
    conf_arc = 2 * math.pi * 54 * (conf_pct / 100)
    conf_gauge = f'''<svg viewBox="0 0 140 140" class="gauge">
        <circle cx="70" cy="70" r="54" fill="none" stroke="#1f2937" stroke-width="12"/>
        <circle cx="70" cy="70" r="54" fill="none" stroke="{sig_color}" stroke-width="12"
            stroke-dasharray="{conf_arc} 999" stroke-linecap="round" transform="rotate(-90 70 70)"/>
        <text x="70" y="65" class="gauge-value">{conf_pct:.1f}%</text>
        <text x="70" y="85" class="gauge-label">Confidence</text>
    </svg>'''
    
    # Generate quality gauge SVG
    qual_pct = quality.overall
    qual_arc = 2 * math.pi * 54 * (qual_pct / 100)
    qual_gauge = f'''<svg viewBox="0 0 140 140" class="gauge">
        <circle cx="70" cy="70" r="54" fill="none" stroke="#1f2937" stroke-width="12"/>
        <circle cx="70" cy="70" r="54" fill="none" stroke="{qual_color}" stroke-width="12"
            stroke-dasharray="{qual_arc} 999" stroke-linecap="round" transform="rotate(-90 70 70)"/>
        <text x="70" y="65" class="gauge-value">{qual_pct:.1f}</text>
        <text x="70" y="85" class="gauge-label">Quality</text>
    </svg>'''
    
    # Build indicator family sections
    families_html = ""
    for fname, fdata in analysis.families.items():
        fc = signal_colors.get(fdata.aggregate_signal.value, "#6b7280")
        
        indicators_html = ""
        for iname, sig in fdata.indicators.items():
            ic = signal_colors.get(sig.direction.value, "#6b7280")
            factors = sig.factors[:2] if sig.factors else ["-"]
            factors_str = "; ".join(factors)
            indicators_html += f'''
                <div class="indicator-row">
                    <span class="ind-name">{iname.upper()}</span>
                    <span class="ind-signal" style="color:{ic}">{sig.direction.value.replace("_"," ")}</span>
                    <span class="ind-conf">{sig.confidence:.0%}</span>
                    <span class="ind-zone">{sig.zone}</span>
                    <span class="ind-factors">{factors_str}</span>
                </div>'''
        
        families_html += f'''
            <div class="family-card">
                <div class="family-header">
                    <div class="family-badge" style="background:{fc}">{fdata.aggregate_signal.value.replace("_"," ")}</div>
                    <div class="family-meta">
                        <span class="family-name">{fname.upper()}</span>
                        <span class="family-conf">{fdata.aggregate_confidence:.0%} confidence</span>
                    </div>
                    <div class="family-weight">{fdata.weight:.0%}</div>
                </div>
                <div class="indicators-list">{indicators_html}</div>
            </div>'''
    
    # Build divergences section
    div_html = ""
    if analysis.divergences:
        for d in analysis.divergences:
            dc = "#10b981" if "BULLISH" in d.divergence_type.value else "#ef4444"
            div_html += f'''
                <div class="divergence-item" style="border-color:{dc}">
                    <span class="div-type" style="color:{dc}">{d.divergence_type.value.replace("_"," ")}</span>
                    <span class="div-ind">{d.indicator_name}</span>
                    <span class="div-strength">{d.strength:.0%}</span>
                    <span class="div-duration">{d.bars_duration} bars</span>
                </div>'''
    else:
        div_html = '<div class="empty-state">No divergences detected</div>'
    
    # Build risk factors
    risk_html = ""
    if analysis.risk_factors:
        for r in analysis.risk_factors:
            risk_html += f'<div class="risk-item"><span class="risk-dot"></span>{r}</div>'
    else:
        risk_html = '<div class="empty-state">No significant risk factors</div>'
    
    # Build statistical tests
    tests = [
        ("Jarque-Bera", "Normality", not stats.is_normal, stats.is_normal),
        ("ADF", "Stationarity", stats.is_stationary, stats.is_stationary),
        ("KPSS", "Trend Stationary", not stats.is_stationary_kpss, stats.is_stationary_kpss),
        ("Ljung-Box", "Autocorrelation", stats.has_autocorrelation, None),
        ("ARCH", "Vol Clustering", stats.has_arch_effects, None),
    ]
    tests_html = ""
    for name, desc, detected, is_pass in tests:
        if is_pass is None:
            status = "detected" if detected else "none"
            status_text = "DETECTED" if detected else "NONE"
        else:
            status = "pass" if is_pass else "fail"
            status_text = "PASS" if is_pass else "FAIL"
        tests_html += f'''
            <div class="test-row">
                <span class="test-name">{name}</span>
                <span class="test-desc">{desc}</span>
                <span class="test-status {status}">{status_text}</span>
            </div>'''
    
    # Build volatility estimators
    vol_html = ""
    vol_items = [
        ("Close to Close", vol_profile.close_to_close),
        ("Parkinson", vol_profile.parkinson),
        ("Garman-Klass", vol_profile.garman_klass),
        ("Rogers-Satchell", vol_profile.rogers_satchell),
        ("Yang-Zhang", vol_profile.yang_zhang),
        ("GKYZ", vol_profile.gkyz),
        ("Hodges-Tompkins", vol_profile.hodges_tompkins),
        ("Composite", vol_profile.composite),
    ]
    for vname, vval in vol_items:
        if vval is not None:
            vol_html += f'''
                <div class="vol-row">
                    <span class="vol-name">{vname}</span>
                    <span class="vol-value">{vval:.2%}</span>
                </div>'''
    
    # Build key levels
    level_items = [
        ("BB Upper", levels.get('bb_upper')), ("BB Middle", levels.get('bb_middle')),
        ("BB Lower", levels.get('bb_lower')), ("KC Upper", levels.get('kc_upper')),
        ("KC Lower", levels.get('kc_lower')), ("Supertrend", levels.get('supertrend')),
        ("Cloud Top", levels.get('ichimoku_cloud_top')), ("Cloud Bottom", levels.get('ichimoku_cloud_bottom')),
    ]
    levels_html = ""
    for lname, lval in level_items:
        if lval and not (isinstance(lval, float) and math.isnan(lval)):
            levels_html += f'''
                <div class="level-item">
                    <span class="level-name">{lname}</span>
                    <span class="level-value">${lval:.2f}</span>
                </div>'''
    
    # Full HTML document
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{symbol} Technical Analysis Report</title>
    <style>
        :root {{
            --bg-0: #0a0a0f;
            --bg-1: #111118;
            --bg-2: #1a1a24;
            --bg-3: #252532;
            --border: #2d2d3a;
            --text-1: #f4f4f6;
            --text-2: #a1a1aa;
            --text-3: #71717a;
            --green: #10b981;
            --red: #ef4444;
            --yellow: #fbbf24;
        }}
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-0);
            color: var(--text-1);
            line-height: 1.5;
            font-size: 14px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 32px; }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding: 32px;
            background: linear-gradient(135deg, var(--bg-1), var(--bg-2));
            border: 1px solid var(--border);
            border-radius: 16px;
            margin-bottom: 24px;
        }}
        .header-left h1 {{ font-size: 36px; font-weight: 700; margin-bottom: 4px; }}
        .header-left .company {{ font-size: 18px; color: var(--text-2); margin-bottom: 16px; }}
        .header-meta {{ display: flex; gap: 24px; font-size: 13px; color: var(--text-3); }}
        .quality-badge {{
            display: inline-block;
            padding: 8px 16px;
            background: {qual_color}22;
            color: {qual_color};
            border-radius: 8px;
            font-weight: 600;
        }}
        .signal-hero {{
            display: grid;
            grid-template-columns: 1fr auto auto;
            gap: 32px;
            padding: 32px;
            background: var(--bg-1);
            border: 1px solid var(--border);
            border-radius: 16px;
            margin-bottom: 24px;
            align-items: center;
        }}
        .signal-content .label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-3); }}
        .signal-content .value {{ font-size: 56px; font-weight: 800; color: {sig_color}; line-height: 1.1; }}
        .signal-content .strength {{ 
            display: inline-block;
            margin-top: 12px;
            padding: 8px 16px;
            background: var(--bg-2);
            border-radius: 8px;
            font-size: 14px;
            color: var(--text-2);
        }}
        .gauge {{ width: 140px; height: 140px; }}
        .gauge-value {{ font-size: 24px; font-weight: 700; fill: var(--text-1); text-anchor: middle; }}
        .gauge-label {{ font-size: 11px; fill: var(--text-3); text-anchor: middle; }}
        .recommendation {{
            grid-column: 1 / -1;
            padding: 20px 24px;
            background: var(--bg-2);
            border-radius: 12px;
            margin-top: 8px;
        }}
        .recommendation .rec-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-3); margin-bottom: 8px; }}
        .recommendation .rec-text {{ font-size: 15px; line-height: 1.6; }}
        .grid-2 {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 24px; margin-bottom: 24px; }}
        .grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }}
        .section {{
            background: var(--bg-1);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        .section-title {{
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-2);
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }}
        .metric {{
            background: var(--bg-2);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        .metric .value {{ font-size: 28px; font-weight: 700; margin-bottom: 4px; }}
        .metric .value.positive {{ color: var(--green); }}
        .metric .value.negative {{ color: var(--red); }}
        .metric .label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-3); }}
        .family-card {{
            background: var(--bg-1);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
        }}
        .family-header {{ display: flex; align-items: center; gap: 16px; margin-bottom: 16px; }}
        .family-badge {{ padding: 6px 14px; border-radius: 6px; font-size: 12px; font-weight: 600; color: white; }}
        .family-meta {{ flex: 1; }}
        .family-name {{ font-weight: 600; font-size: 13px; }}
        .family-conf {{ font-size: 12px; color: var(--text-3); display: block; }}
        .family-weight {{ font-size: 13px; color: var(--text-3); }}
        .indicator-row {{
            display: grid;
            grid-template-columns: 90px 100px 70px 100px 1fr;
            gap: 12px;
            padding: 10px 12px;
            background: var(--bg-2);
            border-radius: 8px;
            font-size: 13px;
            margin-bottom: 6px;
            align-items: center;
        }}
        .ind-name {{ font-weight: 500; }}
        .ind-signal {{ font-weight: 600; }}
        .ind-conf {{ color: var(--text-2); }}
        .ind-zone {{ color: var(--text-3); font-size: 12px; }}
        .ind-factors {{ color: var(--text-3); font-size: 11px; }}
        .divergence-item {{
            display: grid;
            grid-template-columns: 140px 80px 80px 80px;
            gap: 16px;
            padding: 14px 16px;
            background: var(--bg-2);
            border-radius: 8px;
            border-left: 3px solid;
            margin-bottom: 8px;
            font-size: 13px;
        }}
        .div-type {{ font-weight: 600; }}
        .div-ind {{ color: var(--text-2); }}
        .div-strength, .div-duration {{ color: var(--text-3); }}
        .level-item {{ display: flex; justify-content: space-between; padding: 12px 16px; background: var(--bg-2); border-radius: 8px; }}
        .level-name {{ color: var(--text-3); font-size: 12px; }}
        .level-value {{ font-weight: 600; font-family: 'SF Mono', Monaco, monospace; }}
        .risk-item {{ display: flex; align-items: center; gap: 12px; padding: 14px 16px; background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 8px; margin-bottom: 8px; }}
        .risk-dot {{ width: 8px; height: 8px; background: var(--red); border-radius: 50%; flex-shrink: 0; }}
        .test-row {{ display: grid; grid-template-columns: 120px 1fr 100px; gap: 16px; padding: 12px 16px; border-bottom: 1px solid var(--border); font-size: 13px; }}
        .test-row:last-child {{ border-bottom: none; }}
        .test-name {{ font-weight: 500; }}
        .test-desc {{ color: var(--text-3); }}
        .test-status {{ text-align: right; font-weight: 600; }}
        .test-status.pass {{ color: var(--green); }}
        .test-status.fail {{ color: var(--red); }}
        .test-status.detected {{ color: var(--yellow); }}
        .test-status.none {{ color: var(--text-3); }}
        .vol-row {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid var(--border); font-size: 13px; }}
        .vol-row:last-child {{ border-bottom: none; }}
        .vol-name {{ color: var(--text-2); }}
        .vol-value {{ font-weight: 500; font-family: 'SF Mono', Monaco, monospace; }}
        .empty-state {{ padding: 24px; text-align: center; color: var(--text-3); font-style: italic; }}
        .prov-table {{ width: 100%; }}
        .prov-table tr {{ border-bottom: 1px solid var(--border); }}
        .prov-table tr:last-child {{ border-bottom: none; }}
        .prov-table td {{ padding: 12px 0; font-size: 13px; }}
        .prov-table td:first-child {{ color: var(--text-3); width: 180px; }}
        .prov-table td:last-child {{ font-family: 'SF Mono', Monaco, monospace; font-size: 12px; }}
        .footer {{ text-align: center; padding: 32px; color: var(--text-3); font-size: 12px; }}
        @media (max-width: 1200px) {{ .indicator-row {{ grid-template-columns: 1fr 1fr; }} .grid-4 {{ grid-template-columns: repeat(2, 1fr); }} }}
        @media (max-width: 768px) {{ .signal-hero {{ grid-template-columns: 1fr; }} .grid-2 {{ grid-template-columns: 1fr; }} .header {{ flex-direction: column; gap: 16px; }} }}
        @media print {{ body {{ background: white; color: black; }} .section, .family-card, .signal-hero, .header {{ background: white; border-color: #ddd; }} }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-left">
                <h1>{symbol}</h1>
                <div class="company">{p1.company_name}</div>
                <div class="header-meta">
                    <span>{p1.sector}</span>
                    <span>{p1.industry}</span>
                    <span>{p1.period[0]} to {p1.period[1]}</span>
                    <span>{len(p1.daily):,} records</span>
                </div>
            </div>
            <div class="header-right">
                <div class="quality-badge">{quality.grade.value}</div>
                <div style="margin-top:8px;font-size:12px;color:var(--text-3)">
                    Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}
                </div>
            </div>
        </div>
        
        <div class="signal-hero">
            <div class="signal-content">
                <div class="label">Overall Signal</div>
                <div class="value">{analysis.overall_signal.value.replace("_", " ")}</div>
                <div class="strength">Strength: {analysis.signal_strength.value.replace("_", " ")}</div>
            </div>
            {conf_gauge}
            {qual_gauge}
            <div class="recommendation">
                <div class="rec-label">Trading Recommendation</div>
                <div class="rec-text">{analysis.recommendation}</div>
            </div>
        </div>
        
        <div class="grid-2">
            <div class="section">
                <div class="section-title">Data Quality Assessment</div>
                <div class="grid-4">
                    <div class="metric"><div class="value">{quality.completeness:.1f}</div><div class="label">Completeness</div></div>
                    <div class="metric"><div class="value">{quality.accuracy:.1f}</div><div class="label">Accuracy</div></div>
                    <div class="metric"><div class="value">{quality.consistency:.1f}</div><div class="label">Consistency</div></div>
                    <div class="metric"><div class="value">{quality.timeliness:.1f}</div><div class="label">Timeliness</div></div>
                </div>
            </div>
            <div class="section">
                <div class="section-title">Market Profile</div>
                <div class="grid-4">
                    <div class="metric"><div class="value {'positive' if profile.annualized_return > 0 else 'negative'}">{profile.annualized_return:+.1%}</div><div class="label">Annual Return</div></div>
                    <div class="metric"><div class="value">{profile.annualized_volatility:.1%}</div><div class="label">Volatility</div></div>
                    <div class="metric"><div class="value">{profile.sharpe_ratio:.2f}</div><div class="label">Sharpe</div></div>
                    <div class="metric"><div class="value">{profile.sortino_ratio:.2f}</div><div class="label">Sortino</div></div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">Indicator Family Analysis</div>
            <div class="grid-2" style="margin-bottom:0">{families_html}</div>
        </div>
        
        <div class="grid-2">
            <div class="section">
                <div class="section-title">Key Technical Levels</div>
                <div class="grid-4">{levels_html}</div>
            </div>
            <div class="section">
                <div class="section-title">Detected Divergences</div>
                {div_html}
            </div>
        </div>
        
        <div class="grid-2">
            <div class="section">
                <div class="section-title">Statistical Tests</div>
                {tests_html}
            </div>
            <div class="section">
                <div class="section-title">Volatility Estimators (7)</div>
                {vol_html}
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">Risk Factors</div>
            {risk_html}
        </div>
        
        <div class="section">
            <div class="section-title">Data Provenance</div>
            <table class="prov-table">
                <tr><td>Data Source</td><td>{p1.provenance.source}</td></tr>
                <tr><td>Fetch Timestamp</td><td>{p1.provenance.fetch_timestamp}</td></tr>
                <tr><td>Data Hash (SHA-256)</td><td>{p1.provenance.data_hash}</td></tr>
                <tr><td>Records</td><td>{len(p1.daily):,} daily | {len(p1.weekly):,} weekly | {len(p1.monthly):,} monthly</td></tr>
                <tr><td>Engine Version</td><td>{p2.version}</td></tr>
            </table>
        </div>
        
        <div class="footer">
            <strong>Quantitative Technical Analysis Agent v{VERSION}</strong><br>
            MSc AI Agents in Asset Management - Track B<br>
            Report generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>'''
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"Generated HTML: {output_path}")


# =============================================================================
# MARKDOWN REPORT GENERATOR
# =============================================================================

def generate_markdown_report(
    symbol: str,
    phase1_output: Any,
    phase2_output: Any,
    output_path: Path
) -> None:
    """Generate comprehensive Markdown documentation report."""
    p1 = phase1_output
    p2 = phase2_output
    analysis = p2.current_analysis
    quality = p1.quality
    profile = p1.profile
    levels = analysis.key_levels
    
    # Extract nested objects
    tail_risk = profile.tail_risk
    stats = profile.statistical_tests
    vol_profile = profile.volatility_profile
    
    # Build family sections
    families_md = ""
    for fname, fdata in analysis.families.items():
        families_md += f"\n### {fname.upper()} (Weight: {fdata.weight:.0%})\n\n"
        families_md += f"**Signal:** {fdata.aggregate_signal.value} | **Confidence:** {fdata.aggregate_confidence:.0%}\n\n"
        families_md += "| Indicator | Signal | Confidence | Zone | Key Factor |\n"
        families_md += "|-----------|--------|------------|------|------------|\n"
        for iname, sig in fdata.indicators.items():
            factor = sig.factors[0] if sig.factors else "-"
            families_md += f"| {iname.upper()} | {sig.direction.value} | {sig.confidence:.0%} | {sig.zone} | {factor} |\n"
    
    # Build divergences
    div_md = ""
    if analysis.divergences:
        div_md = "\n## Divergence Analysis\n\n"
        div_md += "| Type | Indicator | Strength | Duration |\n|------|-----------|----------|----------|\n"
        for d in analysis.divergences:
            div_md += f"| {d.divergence_type.value} | {d.indicator_name} | {d.strength:.0%} | {d.bars_duration} bars |\n"
    
    # Build volatility
    vol_md = "\n## Volatility Analysis (7 Estimators)\n\n| Estimator | Value |\n|-----------|-------|\n"
    vol_items = [
        ("Close to Close", vol_profile.close_to_close),
        ("Parkinson", vol_profile.parkinson),
        ("Garman-Klass", vol_profile.garman_klass),
        ("Rogers-Satchell", vol_profile.rogers_satchell),
        ("Yang-Zhang", vol_profile.yang_zhang),
        ("GKYZ", vol_profile.gkyz),
        ("Hodges-Tompkins", vol_profile.hodges_tompkins),
        ("Composite", vol_profile.composite),
    ]
    for vn, vv in vol_items:
        if vv is not None:
            vol_md += f"| {vn} | {vv:.2%} |\n"
    
    md = f'''# Technical Analysis Report: {symbol}

## {p1.company_name}

| Field | Value |
|-------|-------|
| Sector | {p1.sector} |
| Industry | {p1.industry} |
| Analysis Period | {p1.period[0]} to {p1.period[1]} |
| Total Records | {len(p1.daily):,} daily |
| Report Generated | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |

---

## Executive Summary

### Overall Signal

| Metric | Value |
|--------|-------|
| Direction | **{analysis.overall_signal.value}** |
| Confidence | {analysis.overall_confidence:.1%} |
| Strength | {analysis.signal_strength.value} |
| Volatility Regime | {p2.volatility_regime.value} |

### Trading Recommendation

> {analysis.recommendation}

### Signal Distribution

- Bullish Families: {analysis.bullish_count}
- Bearish Families: {analysis.bearish_count}
- Neutral Families: {analysis.neutral_count}

---

## Data Quality Assessment

| Dimension | Score | Weight | Status |
|-----------|-------|--------|--------|
| Completeness | {quality.completeness:.1f} | 40% | {'Pass' if quality.completeness >= 95 else 'Review'} |
| Accuracy | {quality.accuracy:.1f} | 30% | {'Pass' if quality.accuracy >= 90 else 'Review'} |
| Consistency | {quality.consistency:.1f} | 20% | {'Pass' if quality.consistency >= 95 else 'Review'} |
| Timeliness | {quality.timeliness:.1f} | 10% | {'Pass' if quality.timeliness >= 90 else 'Review'} |
| **Overall** | **{quality.overall:.1f}/100** | - | **{quality.grade.value}** |

---

## Market Profile

### Returns and Risk

| Metric | Value |
|--------|-------|
| Annualized Return | {profile.annualized_return:+.2%} |
| Annualized Volatility | {profile.annualized_volatility:.2%} |
| Sharpe Ratio | {profile.sharpe_ratio:.3f} |
| Sortino Ratio | {profile.sortino_ratio:.3f} |
| Maximum Drawdown | {tail_risk.max_drawdown:.2%} |
| Calmar Ratio | {profile.calmar_ratio:.3f} |

### Distribution Properties

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Skewness | {profile.skewness:+.3f} | {'Left-tailed' if profile.skewness < -0.5 else 'Right-tailed' if profile.skewness > 0.5 else 'Symmetric'} |
| Kurtosis | {profile.kurtosis:.3f} | {'Fat tails (leptokurtic)' if profile.kurtosis > 3 else 'Normal tails'} |
| Hurst Exponent | {profile.hurst_exponent:.3f} | {profile.trend_character.value} |

---

## Statistical Tests

| Test | Hypothesis | Result | Interpretation |
|------|------------|--------|----------------|
| Jarque-Bera | Normality | {'PASS' if stats.is_normal else 'FAIL'} | {'Normal distribution' if stats.is_normal else 'Non-normal distribution'} |
| ADF | Unit Root | {'PASS' if stats.is_stationary else 'FAIL'} | {'Stationary' if stats.is_stationary else 'Non-stationary'} |
| KPSS | Trend Stationary | {'PASS' if stats.is_stationary_kpss else 'FAIL'} | {'Trend stationary' if stats.is_stationary_kpss else 'Non-stationary'} |
| Ljung-Box | Autocorrelation | {'DETECTED' if stats.has_autocorrelation else 'NONE'} | {'Predictable patterns' if stats.has_autocorrelation else 'No autocorrelation'} |
| ARCH | Vol Clustering | {'DETECTED' if stats.has_arch_effects else 'NONE'} | {'Use GARCH' if stats.has_arch_effects else 'Homoscedastic'} |

---

## Indicator Family Analysis

{families_md}

---

## Key Technical Levels

| Level | Price |
|-------|-------|
| Bollinger Upper | ${levels.get('bb_upper', 0):.2f} |
| Bollinger Middle | ${levels.get('bb_middle', 0):.2f} |
| Bollinger Lower | ${levels.get('bb_lower', 0):.2f} |
| Keltner Upper | ${levels.get('kc_upper', 0):.2f} |
| Keltner Lower | ${levels.get('kc_lower', 0):.2f} |
| Supertrend | ${levels.get('supertrend', 0):.2f} |
| Ichimoku Cloud Top | ${levels.get('ichimoku_cloud_top', 0):.2f} |
| Ichimoku Cloud Bottom | ${levels.get('ichimoku_cloud_bottom', 0):.2f} |

{div_md}

## Risk Assessment

{chr(10).join(['- ' + r for r in analysis.risk_factors]) if analysis.risk_factors else 'No significant risk factors identified.'}

{vol_md}

---

## Data Provenance

| Field | Value |
|-------|-------|
| Source | {p1.provenance.source} |
| Fetch Timestamp | {p1.provenance.fetch_timestamp} |
| Data Hash (SHA-256) | `{p1.provenance.data_hash}` |
| Daily Records | {len(p1.daily):,} |
| Weekly Records | {len(p1.weekly):,} |
| Monthly Records | {len(p1.monthly):,} |
| Pipeline Version | {p1.provenance.version} |
| Engine Version | {p2.version} |

---

*Generated by Quantitative Technical Analysis Agent v{VERSION}*

*MSc AI Agents in Asset Management - Track B: Technical Analyst Agent*
'''
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)
    logger.info(f"Generated Markdown: {output_path}")


# =============================================================================
# JSON REPORT GENERATOR
# =============================================================================

def generate_json_report(
    symbol: str,
    phase1_output: Any,
    phase2_output: Any,
    output_path: Path
) -> None:
    """Generate comprehensive JSON report for programmatic consumption."""
    p1 = phase1_output
    p2 = phase2_output
    analysis = p2.current_analysis
    quality = p1.quality
    profile = p1.profile
    
    # Extract nested objects
    tail_risk = profile.tail_risk
    stats = profile.statistical_tests
    vol_profile = profile.volatility_profile
    
    report = {
        "metadata": {
            "symbol": symbol,
            "company_name": p1.company_name,
            "sector": p1.sector,
            "industry": p1.industry,
            "generated_at": datetime.now().isoformat(),
            "report_version": VERSION,
            "engine_version": p2.version,
            "pipeline_version": p1.provenance.version
        },
        "executive_summary": {
            "overall_signal": {
                "direction": analysis.overall_signal.value,
                "confidence": round(analysis.overall_confidence, 4),
                "strength": analysis.signal_strength.value
            },
            "volatility_regime": p2.volatility_regime.value,
            "recommendation": analysis.recommendation,
            "signal_distribution": {
                "bullish": analysis.bullish_count,
                "bearish": analysis.bearish_count,
                "neutral": analysis.neutral_count
            }
        },
        "analysis_period": {
            "start": p1.period[0],
            "end": p1.period[1],
            "daily_records": len(p1.daily),
            "weekly_records": len(p1.weekly) if p1.weekly is not None else 0,
            "monthly_records": len(p1.monthly) if p1.monthly is not None else 0
        },
        "data_quality": {
            "overall_score": round(quality.overall, 2),
            "grade": quality.grade.value,
            "dimensions": {
                "completeness": {"score": round(quality.completeness, 2), "weight": 0.40},
                "accuracy": {"score": round(quality.accuracy, 2), "weight": 0.30},
                "consistency": {"score": round(quality.consistency, 2), "weight": 0.20},
                "timeliness": {"score": round(quality.timeliness, 2), "weight": 0.10}
            }
        },
        "market_profile": {
            "returns": {
                "annualized": round(profile.annualized_return, 6)
            },
            "risk": {
                "annualized_volatility": round(profile.annualized_volatility, 6),
                "sharpe_ratio": round(profile.sharpe_ratio, 4),
                "sortino_ratio": round(profile.sortino_ratio, 4),
                "calmar_ratio": round(profile.calmar_ratio, 4),
                "max_drawdown": round(tail_risk.max_drawdown, 6)
            },
            "distribution": {
                "skewness": round(profile.skewness, 6),
                "kurtosis": round(profile.kurtosis, 6),
                "is_normal": stats.is_normal
            },
            "trend": {
                "hurst_exponent": round(profile.hurst_exponent, 6),
                "character": profile.trend_character.value
            }
        },
        "statistical_tests": {
            "jarque_bera": {"is_normal": stats.is_normal, "test": "normality"},
            "adf": {"is_stationary": stats.is_stationary, "test": "unit_root"},
            "kpss": {"is_stationary": stats.is_stationary_kpss, "test": "trend_stationarity"},
            "ljung_box": {"has_autocorrelation": stats.has_autocorrelation, "test": "autocorrelation"},
            "arch": {"has_arch_effects": stats.has_arch_effects, "test": "volatility_clustering"}
        },
        "indicator_families": {},
        "key_levels": {},
        "divergences": [],
        "risk_factors": analysis.risk_factors,
        "volatility_estimators": {
            "close_to_close": round(vol_profile.close_to_close, 6),
            "parkinson": round(vol_profile.parkinson, 6),
            "garman_klass": round(vol_profile.garman_klass, 6),
            "rogers_satchell": round(vol_profile.rogers_satchell, 6),
            "yang_zhang": round(vol_profile.yang_zhang, 6),
            "gkyz": round(vol_profile.gkyz, 6),
            "hodges_tompkins": round(vol_profile.hodges_tompkins, 6),
            "composite": round(vol_profile.composite, 6)
        },
        "provenance": {
            "source": p1.provenance.source,
            "fetch_timestamp": p1.provenance.fetch_timestamp,
            "data_hash": p1.provenance.data_hash
        }
    }
    
    # Add families
    for fname, fdata in analysis.families.items():
        report["indicator_families"][fname] = {
            "signal": fdata.aggregate_signal.value,
            "confidence": round(fdata.aggregate_confidence, 4),
            "weight": fdata.weight,
            "indicators": {}
        }
        for iname, sig in fdata.indicators.items():
            report["indicator_families"][fname]["indicators"][iname] = {
                "direction": sig.direction.value,
                "confidence": round(sig.confidence, 4),
                "value": round(float(sig.value), 6),
                "zone": sig.zone,
                "factors": sig.factors
            }
    
    # Add levels
    for ln, lv in analysis.key_levels.items():
        if lv is not None and not (isinstance(lv, float) and math.isnan(lv)):
            report["key_levels"][ln] = round(float(lv), 4)
    
    # Add divergences
    for d in analysis.divergences:
        report["divergences"].append({
            "type": d.divergence_type.value,
            "indicator": d.indicator_name,
            "strength": round(d.strength, 4),
            "duration_bars": d.bars_duration
        })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Generated JSON: {output_path}")


# =============================================================================
# PDF REPORT GENERATOR - COMMENTED OUT (JSON only)
# =============================================================================
# PDF generation has been disabled. Uncomment the function below if needed.
"""
def generate_pdf_report(
    symbol: str,
    phase1_output: Any,
    phase2_output: Any,
    output_path: Path
) -> bool:
    # Generate native PDF report using reportlab.
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )
        from reportlab.lib.enums import TA_CENTER
    except ImportError:
        logger.warning("reportlab not installed. Run: pip install reportlab")
        return False

    p1 = phase1_output
    p2 = phase2_output
    analysis = p2.current_analysis
    quality = p1.quality
    profile = p1.profile
    levels = analysis.key_levels

    # Extract nested objects
    tail_risk = profile.tail_risk
    stats = profile.statistical_tests

    # Colors
    NAVY = colors.HexColor("#1a1a2e")
    GREEN = colors.HexColor("#10b981")
    RED = colors.HexColor("#ef4444")
    GRAY = colors.HexColor("#6b7280")
    LIGHT = colors.HexColor("#f3f4f6")

    signal_colors = {
        "STRONG_BUY": GREEN, "BUY": colors.HexColor("#34d399"),
        "NEUTRAL": GRAY, "SELL": colors.HexColor("#f87171"), "STRONG_SELL": RED
    }
    sig_color = signal_colors.get(analysis.overall_signal.value, GRAY)

    # Create document
    doc = SimpleDocTemplate(
        str(output_path), pagesize=letter,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch
    )

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=26, spaceAfter=4, textColor=NAVY)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=12, textColor=GRAY, spaceAfter=16)
    section_style = ParagraphStyle('Section', parent=styles['Heading2'], fontSize=13, spaceBefore=20, spaceAfter=10, textColor=NAVY)
    normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontSize=10, leading=14)

    story = []

    # Title
    story.append(Paragraph(f"<b>{symbol}</b> Technical Analysis Report", title_style))
    story.append(Paragraph(f"{p1.company_name} | {p1.sector} | {p1.industry}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=LIGHT, spaceAfter=16))

    # Executive Summary
    story.append(Paragraph("Executive Summary", section_style))
    summary_data = [
        ["Overall Signal", analysis.overall_signal.value.replace("_", " ")],
        ["Confidence", f"{analysis.overall_confidence:.1%}"],
        ["Signal Strength", analysis.signal_strength.value.replace("_", " ")],
        ["Volatility Regime", p2.volatility_regime.value],
        ["Data Quality", f"{quality.overall:.1f}/100 ({quality.grade.value})"],
    ]
    t = Table(summary_data, colWidths=[2.2*inch, 4.3*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), LIGHT),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (1, 0), (1, 0), sig_color),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Recommendation:</b> {analysis.recommendation}", normal_style))
    story.append(Spacer(1, 16))

    # Data Quality
    story.append(Paragraph("Data Quality Assessment", section_style))
    q_data = [
        ["Dimension", "Score", "Weight", "Status"],
        ["Completeness", f"{quality.completeness:.1f}", "40%", "Pass" if quality.completeness >= 95 else "Review"],
        ["Accuracy", f"{quality.accuracy:.1f}", "30%", "Pass" if quality.accuracy >= 90 else "Review"],
        ["Consistency", f"{quality.consistency:.1f}", "20%", "Pass" if quality.consistency >= 95 else "Review"],
        ["Timeliness", f"{quality.timeliness:.1f}", "10%", "Pass" if quality.timeliness >= 90 else "Review"],
    ]
    t = Table(q_data, colWidths=[1.8*inch, 1.5*inch, 1.5*inch, 1.7*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    # Market Profile
    story.append(Paragraph("Market Profile", section_style))
    m_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Annual Return", f"{profile.annualized_return:+.2%}", "Volatility", f"{profile.annualized_volatility:.2%}"],
        ["Sharpe Ratio", f"{profile.sharpe_ratio:.3f}", "Sortino Ratio", f"{profile.sortino_ratio:.3f}"],
        ["Max Drawdown", f"{tail_risk.max_drawdown:.2%}", "Calmar Ratio", f"{profile.calmar_ratio:.3f}"],
        ["Hurst Exponent", f"{profile.hurst_exponent:.3f}", "Trend", profile.trend_character.value],
        ["Skewness", f"{profile.skewness:+.3f}", "Kurtosis", f"{profile.kurtosis:.3f}"],
    ]
    t = Table(m_data, colWidths=[1.6*inch, 1.5*inch, 1.6*inch, 1.8*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 1), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
        ('ALIGN', (3, 1), (3, -1), 'RIGHT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    # Indicator Families
    story.append(Paragraph("Indicator Family Summary", section_style))
    f_data = [["Family", "Signal", "Confidence", "Weight"]]
    for fn, fd in analysis.families.items():
        f_data.append([fn.upper(), fd.aggregate_signal.value.replace("_", " "), f"{fd.aggregate_confidence:.0%}", f"{fd.weight:.0%}"])
    t = Table(f_data, colWidths=[1.8*inch, 2*inch, 1.5*inch, 1.2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    # Statistical Tests
    story.append(Paragraph("Statistical Tests", section_style))
    test_data = [
        ["Test", "Result", "Interpretation"],
        ["Jarque-Bera", "PASS" if stats.is_normal else "FAIL", "Normal" if stats.is_normal else "Non-normal"],
        ["ADF", "PASS" if stats.is_stationary else "FAIL", "Stationary" if stats.is_stationary else "Unit root"],
        ["KPSS", "PASS" if stats.is_stationary_kpss else "FAIL", "Trend stationary" if stats.is_stationary_kpss else "Non-stationary"],
        ["Ljung-Box", "DETECTED" if stats.has_autocorrelation else "NONE", "Autocorrelation" if stats.has_autocorrelation else "No autocorrelation"],
        ["ARCH", "DETECTED" if stats.has_arch_effects else "NONE", "Vol clustering" if stats.has_arch_effects else "Homoscedastic"],
    ]
    t = Table(test_data, colWidths=[1.8*inch, 1.5*inch, 3.2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    # Risk Factors
    story.append(Paragraph("Risk Factors", section_style))
    if analysis.risk_factors:
        for r in analysis.risk_factors:
            story.append(Paragraph(f"- {r}", normal_style))
    else:
        story.append(Paragraph("No significant risk factors identified.", normal_style))
    story.append(Spacer(1, 16))

    # Footer
    story.append(HRFlowable(width="100%", thickness=1, color=LIGHT, spaceBefore=16))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=GRAY, alignment=TA_CENTER)
    story.append(Paragraph(
        f"Quantitative Technical Analysis Agent v{VERSION} | MSc AI Agents in Asset Management - Track B | "
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        footer_style
    ))

    # Build PDF
    doc.build(story)
    logger.info(f"Generated PDF: {output_path}")
    return True
"""


# =============================================================================
# MAIN GENERATOR
# =============================================================================

def generate_all_reports(
    symbol: str,
    phase1_output: Any,
    phase2_output: Any,
    output_dir: Path
) -> Dict[str, Optional[Path]]:
    """Generate all report formats: HTML, Markdown, JSON, PDF."""
    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    sym = symbol.lower()
    outputs = {}
    
    # HTML
    html_path = reports_dir / f"{sym}_analysis.html"
    try:
        generate_html_report(symbol, phase1_output, phase2_output, html_path)
        outputs['html'] = html_path
    except Exception as e:
        logger.error(f"HTML failed: {e}")
        outputs['html'] = None
    
    # Markdown
    md_path = reports_dir / f"{sym}_analysis.md"
    try:
        generate_markdown_report(symbol, phase1_output, phase2_output, md_path)
        outputs['md'] = md_path
    except Exception as e:
        logger.error(f"Markdown failed: {e}")
        outputs['md'] = None
    
    # JSON
    json_path = reports_dir / f"{sym}_analysis.json"
    try:
        generate_json_report(symbol, phase1_output, phase2_output, json_path)
        outputs['json'] = json_path
    except Exception as e:
        logger.error(f"JSON failed: {e}")
        outputs['json'] = None
    
    # PDF generation commented out - JSON only
    # pdf_path = reports_dir / f"{sym}_analysis.pdf"
    # try:
    #     if generate_pdf_report(symbol, phase1_output, phase2_output, pdf_path):
    #         outputs['pdf'] = pdf_path
    #     else:
    #         outputs['pdf'] = None
    # except Exception as e:
    #     logger.error(f"PDF failed: {e}")
    #     outputs['pdf'] = None

    return outputs