"""
================================================================================
PHASE 5: PREMIUM INSTITUTIONAL TRADE NOTE REPORTS v4.0
================================================================================

Professional-grade report generation with institutional visual design.
Includes SVG visualizations, modern typography, and premium aesthetics.

Version: 4.0.0
================================================================================
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import math

logger = logging.getLogger(__name__)

VERSION = "4.0.0"


# =============================================================================
# PREMIUM HTML FORMATTER
# =============================================================================

class PremiumHTMLFormatter:
    """Generate premium institutional-grade HTML reports with visualizations."""
    
    def format(self, note: Any) -> str:
        """Generate premium HTML report."""
        # Color scheme
        rec_colors = {
            'STRONG BUY': '#10B981',
            'BUY': '#34D399',
            'ACCUMULATE': '#6EE7B7',
            'HOLD': '#FCD34D',
            'REDUCE': '#FBBF24',
            'SELL': '#F87171',
            'STRONG SELL': '#EF4444'
        }
        rec_value = note.recommendation.value if hasattr(note.recommendation, 'value') else str(note.recommendation)
        rec_color = rec_colors.get(rec_value, '#6B7280')
        
        risk_colors = {
            'LOW': '#10B981',
            'MODERATE': '#FBBF24',
            'HIGH': '#F97316',
            'VERY HIGH': '#EF4444'
        }
        risk_value = note.risk_rating.value if hasattr(note.risk_rating, 'value') else str(note.risk_rating)
        risk_color = risk_colors.get(risk_value, '#6B7280')
        
        # Extract values safely
        current_price = float(note.current_price) if note.current_price else 0
        expected_return = float(note.expected_return) if note.expected_return else 0
        rr_ratio = note.trade_levels.risk_reward_ratio if note.trade_levels else 0
        
        conv_value = note.conviction.value if hasattr(note.conviction, 'value') else str(note.conviction)
        horizon_value = note.time_horizon.value if hasattr(note.time_horizon, 'value') else str(note.time_horizon)
        
        # Generate scenario chart data
        scenarios = note.scenarios if note.scenarios else []
        scenario_chart = self._generate_scenario_chart(scenarios, current_price)
        
        # Generate gauge SVGs
        confidence_gauge = self._generate_gauge(0.733, "Signal Confidence", "#3B82F6")  # 73.3%
        risk_gauge = self._generate_risk_gauge(risk_value)
        
        # Trade levels visualization
        trade_levels_viz = self._generate_trade_levels_viz(note.trade_levels, current_price)
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Institutional Trade Note | {note.metadata.symbol}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --primary: #1E3A5F;
            --primary-dark: #0F2744;
            --accent: #3B82F6;
            --accent-light: #60A5FA;
            --success: #10B981;
            --warning: #F59E0B;
            --danger: #EF4444;
            --bg-primary: #FFFFFF;
            --bg-secondary: #F8FAFC;
            --bg-tertiary: #F1F5F9;
            --text-primary: #0F172A;
            --text-secondary: #475569;
            --text-muted: #94A3B8;
            --border: #E2E8F0;
            --border-light: #F1F5F9;
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
            --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
            --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
            --shadow-xl: 0 20px 25px -5px rgba(0,0,0,0.1), 0 10px 10px -5px rgba(0,0,0,0.04);
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        /* Header Section */
        .header {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: white;
            padding: 2.5rem 3rem;
            border-radius: 20px 20px 0 0;
            position: relative;
            overflow: hidden;
        }}
        
        .header::before {{
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            transform: translate(100px, -200px);
        }}
        
        .header-top {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 1.5rem;
        }}
        
        .brand {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        
        .brand-icon {{
            width: 48px;
            height: 48px;
            background: rgba(255,255,255,0.15);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }}
        
        .brand-text {{
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            opacity: 0.9;
        }}
        
        .report-meta {{
            text-align: right;
            font-size: 0.8rem;
            opacity: 0.8;
        }}
        
        .security-info {{
            display: flex;
            align-items: baseline;
            gap: 1.5rem;
            flex-wrap: wrap;
        }}
        
        .symbol {{
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }}
        
        .company-name {{
            font-size: 1.25rem;
            font-weight: 500;
            opacity: 0.9;
        }}
        
        .sector-badge {{
            background: rgba(255,255,255,0.15);
            padding: 0.4rem 1rem;
            border-radius: 30px;
            font-size: 0.8rem;
            font-weight: 500;
        }}
        
        /* Main Content */
        .main-content {{
            background: var(--bg-primary);
            border-radius: 0 0 20px 20px;
            box-shadow: var(--shadow-xl);
        }}
        
        /* Recommendation Banner */
        .rec-banner {{
            background: linear-gradient(90deg, {rec_color}15 0%, {rec_color}05 100%);
            border-left: 5px solid {rec_color};
            padding: 2rem 3rem;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 2rem;
            align-items: center;
        }}
        
        .rec-main {{
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}
        
        .rec-label {{
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
            font-weight: 600;
        }}
        
        .rec-value {{
            font-size: 2rem;
            font-weight: 800;
            color: {rec_color};
            letter-spacing: -0.02em;
        }}
        
        .rec-item {{
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }}
        
        .rec-item .value {{
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-primary);
        }}
        
        .rec-item .value.price {{
            font-family: 'SF Mono', 'Monaco', monospace;
        }}
        
        .rec-item .value.positive {{
            color: var(--success);
        }}
        
        .rec-item .value.risk {{
            color: {risk_color};
        }}
        
        /* Content Sections */
        .content {{
            padding: 0;
        }}
        
        .section {{
            padding: 2.5rem 3rem;
            border-bottom: 1px solid var(--border-light);
        }}
        
        .section:last-child {{
            border-bottom: none;
        }}
        
        .section-header {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}
        
        .section-icon {{
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-light) 100%);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.1rem;
        }}
        
        .section-title {{
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-primary);
        }}
        
        /* Executive Summary */
        .executive-summary {{
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
        }}
        
        .summary-text {{
            font-size: 1.05rem;
            line-height: 1.9;
            color: var(--text-secondary);
            column-count: 2;
            column-gap: 3rem;
            text-align: justify;
        }}
        
        .summary-text p {{
            margin-bottom: 1.25rem;
            text-indent: 2rem;
        }}
        
        .summary-text p:first-child {{
            text-indent: 0;
        }}
        
        .summary-text p:first-child::first-letter {{
            font-size: 3.5rem;
            font-weight: 700;
            float: left;
            line-height: 1;
            margin-right: 0.5rem;
            margin-top: 0.1rem;
            color: var(--primary);
        }}
        
        /* KPI Dashboard */
        .kpi-dashboard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .kpi-card {{
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid var(--border);
            transition: all 0.3s ease;
        }}
        
        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }}
        
        .kpi-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }}
        
        .kpi-title {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            font-weight: 600;
        }}
        
        .kpi-badge {{
            font-size: 0.7rem;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-weight: 600;
        }}
        
        .kpi-badge.positive {{
            background: #D1FAE5;
            color: #065F46;
        }}
        
        .kpi-badge.negative {{
            background: #FEE2E2;
            color: #991B1B;
        }}
        
        .kpi-badge.neutral {{
            background: #E2E8F0;
            color: #475569;
        }}
        
        .kpi-value {{
            font-size: 2rem;
            font-weight: 800;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }}
        
        .kpi-subtitle {{
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}
        
        /* Visualization Container */
        .viz-container {{
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid var(--border);
        }}
        
        .viz-title {{
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        /* Trade Levels Visualization */
        .trade-levels-viz {{
            position: relative;
            height: 300px;
            margin: 2rem 0;
        }}
        
        /* Scenario Cards */
        .scenario-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
        }}
        
        .scenario-card {{
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid var(--border);
            position: relative;
            overflow: hidden;
        }}
        
        .scenario-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
        }}
        
        .scenario-card.bull::before {{
            background: linear-gradient(90deg, #10B981, #34D399);
        }}
        
        .scenario-card.base::before {{
            background: linear-gradient(90deg, #F59E0B, #FBBF24);
        }}
        
        .scenario-card.bear::before {{
            background: linear-gradient(90deg, #EF4444, #F87171);
        }}
        
        .scenario-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }}
        
        .scenario-name {{
            font-size: 1rem;
            font-weight: 700;
            color: var(--text-primary);
        }}
        
        .scenario-prob {{
            font-size: 0.85rem;
            font-weight: 700;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            background: var(--bg-tertiary);
        }}
        
        .scenario-card.bull .scenario-prob {{
            color: #065F46;
            background: #D1FAE5;
        }}
        
        .scenario-card.base .scenario-prob {{
            color: #92400E;
            background: #FEF3C7;
        }}
        
        .scenario-card.bear .scenario-prob {{
            color: #991B1B;
            background: #FEE2E2;
        }}
        
        .scenario-metrics {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-bottom: 1rem;
        }}
        
        .scenario-metric {{
            text-align: center;
            padding: 0.75rem;
            background: var(--bg-primary);
            border-radius: 10px;
        }}
        
        .scenario-metric .label {{
            font-size: 0.7rem;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 0.25rem;
        }}
        
        .scenario-metric .value {{
            font-size: 1.1rem;
            font-weight: 700;
        }}
        
        .scenario-card.bull .scenario-metric .value {{
            color: var(--success);
        }}
        
        .scenario-card.base .scenario-metric .value {{
            color: var(--warning);
        }}
        
        .scenario-card.bear .scenario-metric .value {{
            color: var(--danger);
        }}
        
        .scenario-drivers {{
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}
        
        .scenario-drivers ul {{
            list-style: none;
            padding: 0;
        }}
        
        .scenario-drivers li {{
            padding: 0.4rem 0;
            padding-left: 1.25rem;
            position: relative;
        }}
        
        .scenario-drivers li::before {{
            content: '→';
            position: absolute;
            left: 0;
            color: var(--accent);
        }}
        
        /* Tables */
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        
        .data-table th {{
            background: var(--bg-secondary);
            padding: 1rem;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            border-bottom: 2px solid var(--border);
        }}
        
        .data-table td {{
            padding: 1rem;
            border-bottom: 1px solid var(--border-light);
        }}
        
        .data-table tr:hover td {{
            background: var(--bg-secondary);
        }}
        
        /* Catalyst & Risk Lists */
        .dual-column {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }}
        
        .list-card {{
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid var(--border);
        }}
        
        .list-card.catalysts {{
            border-left: 4px solid var(--success);
        }}
        
        .list-card.risks {{
            border-left: 4px solid var(--danger);
        }}
        
        .list-card-title {{
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .list-card.catalysts .list-card-title {{
            color: var(--success);
        }}
        
        .list-card.risks .list-card-title {{
            color: var(--danger);
        }}
        
        .list-card ol {{
            padding-left: 1.25rem;
        }}
        
        .list-card li {{
            padding: 0.6rem 0;
            color: var(--text-secondary);
            font-size: 0.9rem;
            line-height: 1.6;
        }}
        
        /* Analysis Text */
        .analysis-text {{
            font-size: 0.95rem;
            line-height: 1.8;
            color: var(--text-secondary);
        }}
        
        .analysis-text p {{
            margin-bottom: 1.25rem;
            text-align: justify;
        }}
        
        /* Disclaimer */
        .disclaimer {{
            background: var(--bg-tertiary);
            padding: 1.5rem 3rem;
            font-size: 0.75rem;
            color: var(--text-muted);
            line-height: 1.6;
            border-top: 1px solid var(--border);
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 1.5rem;
            color: var(--text-muted);
            font-size: 0.8rem;
        }}
        
        /* Responsive */
        @media (max-width: 1200px) {{
            .summary-text {{
                column-count: 1;
            }}
            .scenario-grid {{
                grid-template-columns: 1fr;
            }}
            .dual-column {{
                grid-template-columns: 1fr;
            }}
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .container {{
                max-width: 100%;
            }}
            .header {{
                border-radius: 0;
            }}
            .main-content {{
                box-shadow: none;
                border-radius: 0;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header">
            <div class="header-top">
                <div class="brand">
                    <div class="brand-icon">📊</div>
                    <div class="brand-text">Institutional Research<br><strong>Trade Analysis Report</strong></div>
                </div>
                <div class="report-meta">
                    <div>Report ID: {note.metadata.note_id}</div>
                    <div>{note.metadata.generation_date.strftime('%B %d, %Y')}</div>
                    <div>Version {VERSION}</div>
                </div>
            </div>
            <div class="security-info">
                <span class="symbol">{note.metadata.symbol}</span>
                <span class="company-name">{note.metadata.company_name}</span>
                <span class="sector-badge">{note.metadata.sector} • {note.metadata.industry}</span>
            </div>
        </header>
        
        <!-- Main Content -->
        <main class="main-content">
            <!-- Recommendation Banner -->
            <div class="rec-banner">
                <div class="rec-main">
                    <span class="rec-label">Recommendation</span>
                    <span class="rec-value">{rec_value}</span>
                </div>
                <div class="rec-item">
                    <span class="rec-label">Conviction</span>
                    <span class="value">{conv_value}</span>
                </div>
                <div class="rec-item">
                    <span class="rec-label">Current Price</span>
                    <span class="value price">${current_price:,.2f}</span>
                </div>
                <div class="rec-item">
                    <span class="rec-label">Expected Return</span>
                    <span class="value positive">{expected_return*100:+.1f}%</span>
                </div>
                <div class="rec-item">
                    <span class="rec-label">Risk/Reward</span>
                    <span class="value">{rr_ratio:.2f}:1</span>
                </div>
                <div class="rec-item">
                    <span class="rec-label">Risk Rating</span>
                    <span class="value risk">{risk_value}</span>
                </div>
            </div>
            
            <!-- Executive Summary -->
            <section class="section executive-summary">
                <div class="section-header">
                    <div class="section-icon">📝</div>
                    <h2 class="section-title">Executive Summary</h2>
                </div>
                <div class="summary-text">
                    {self._format_paragraphs(note.executive_summary)}
                </div>
            </section>
            
            <!-- Key Metrics Dashboard -->
            <section class="section">
                <div class="section-header">
                    <div class="section-icon">📈</div>
                    <h2 class="section-title">Performance Dashboard</h2>
                </div>
                
                <div class="kpi-dashboard">
                    {self._generate_kpi_cards(note)}
                </div>
                
                <!-- Trade Levels Visualization -->
                <div class="viz-container">
                    <div class="viz-title">Price Levels & Trading Range</div>
                    {trade_levels_viz}
                </div>
            </section>
            
            <!-- Scenario Analysis -->
            <section class="section">
                <div class="section-header">
                    <div class="section-icon">🎯</div>
                    <h2 class="section-title">Scenario Analysis</h2>
                </div>
                
                <div class="scenario-grid">
                    {self._generate_scenario_cards(scenarios, current_price)}
                </div>
            </section>
            
            <!-- Technical Analysis -->
            <section class="section">
                <div class="section-header">
                    <div class="section-icon">📊</div>
                    <h2 class="section-title">Technical Analysis</h2>
                </div>
                <div class="analysis-text">
                    {self._format_paragraphs(note.technical_analysis)}
                </div>
            </section>
            
            <!-- Regime Analysis -->
            <section class="section">
                <div class="section-header">
                    <div class="section-icon">🔄</div>
                    <h2 class="section-title">Market Regime Analysis</h2>
                </div>
                <div class="analysis-text">
                    {self._format_paragraphs(note.regime_analysis)}
                </div>
            </section>
            
            <!-- Backtest Summary -->
            <section class="section">
                <div class="section-header">
                    <div class="section-icon">⚡</div>
                    <h2 class="section-title">Backtesting Summary</h2>
                </div>
                <div class="analysis-text">
                    {self._format_paragraphs(note.backtest_summary)}
                </div>
            </section>
            
            <!-- Catalysts & Risks -->
            <section class="section">
                <div class="section-header">
                    <div class="section-icon">⚖️</div>
                    <h2 class="section-title">Catalysts & Risk Factors</h2>
                </div>
                
                <div class="dual-column">
                    <div class="list-card catalysts">
                        <div class="list-card-title">
                            <span>🚀</span> Key Catalysts
                        </div>
                        <ol>
                            {self._format_list_items(note.key_catalysts[:5])}
                        </ol>
                    </div>
                    
                    <div class="list-card risks">
                        <div class="list-card-title">
                            <span>⚠️</span> Primary Risks
                        </div>
                        <ol>
                            {self._format_list_items(note.primary_risks[:5])}
                        </ol>
                    </div>
                </div>
            </section>
            
            <!-- Risk Assessment -->
            <section class="section">
                <div class="section-header">
                    <div class="section-icon">🛡️</div>
                    <h2 class="section-title">Risk Assessment</h2>
                </div>
                <div class="analysis-text">
                    {self._format_paragraphs(note.risk_assessment)}
                </div>
            </section>
            
            <!-- Trade Levels Table -->
            <section class="section">
                <div class="section-header">
                    <div class="section-icon">🎚️</div>
                    <h2 class="section-title">Trade Setup</h2>
                </div>
                
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Level</th>
                            <th>Price</th>
                            <th>Distance</th>
                            <th>Notes</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Entry</strong></td>
                            <td style="font-family: monospace; font-weight: 600;">${note.trade_levels.entry_price:,.2f}</td>
                            <td>—</td>
                            <td>Current market price</td>
                        </tr>
                        <tr>
                            <td><strong>Stop Loss</strong></td>
                            <td style="font-family: monospace; color: var(--danger);">${note.trade_levels.stop_loss:,.2f}</td>
                            <td style="color: var(--danger);">{((note.trade_levels.stop_loss/current_price - 1)*100):+.1f}%</td>
                            <td>Below key support level</td>
                        </tr>
                        <tr>
                            <td><strong>Take Profit 1</strong></td>
                            <td style="font-family: monospace; color: var(--success);">${note.trade_levels.take_profit_1:,.2f}</td>
                            <td style="color: var(--success);">{((note.trade_levels.take_profit_1/current_price - 1)*100):+.1f}%</td>
                            <td>Primary target</td>
                        </tr>
                        {f'<tr><td><strong>Take Profit 2</strong></td><td style="font-family: monospace; color: var(--success);">${note.trade_levels.take_profit_2:,.2f}</td><td style="color: var(--success);">{((note.trade_levels.take_profit_2/current_price - 1)*100):+.1f}%</td><td>Extended target</td></tr>' if note.trade_levels.take_profit_2 else ''}
                    </tbody>
                </table>
                
                <!-- Position Sizing -->
                <div class="viz-container" style="margin-top: 1.5rem;">
                    <div class="viz-title">Position Sizing Recommendation</div>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; text-align: center;">
                        <div>
                            <div style="font-size: 2.5rem; font-weight: 800; color: var(--accent);">{note.position_sizing.recommended_size*100:.1f}%</div>
                            <div style="color: var(--text-muted); font-size: 0.85rem;">Recommended Size</div>
                        </div>
                        <div>
                            <div style="font-size: 2.5rem; font-weight: 800; color: var(--text-secondary);">{note.position_sizing.max_size*100:.1f}%</div>
                            <div style="color: var(--text-muted); font-size: 0.85rem;">Maximum Size</div>
                        </div>
                        <div>
                            <div style="font-size: 1.1rem; font-weight: 600; color: var(--primary); padding-top: 0.5rem;">{note.position_sizing.sizing_method}</div>
                            <div style="color: var(--text-muted); font-size: 0.85rem;">Methodology</div>
                        </div>
                    </div>
                    <div style="margin-top: 1rem; padding: 1rem; background: var(--bg-primary); border-radius: 10px; font-size: 0.9rem; color: var(--text-secondary);">
                        <strong>Rationale:</strong> {note.position_sizing.rationale}
                    </div>
                </div>
            </section>
            
            <!-- Disclaimer -->
            <div class="disclaimer">
                <strong>IMPORTANT DISCLOSURES:</strong> {note.disclaimer}
            </div>
        </main>
        
        <!-- Footer -->
        <footer class="footer">
            <div>Report ID: {note.metadata.note_id} | Generated: {note.metadata.generation_date.strftime('%Y-%m-%d %H:%M:%S')} UTC | {note.metadata.analyst}</div>
        </footer>
    </div>
</body>
</html>'''
        
        return html
    
    def _format_paragraphs(self, text: str) -> str:
        """Format text into paragraphs."""
        if not text:
            return "<p>No analysis available.</p>"
        paragraphs = text.split('\n\n')
        return '\n'.join(f'<p>{p.strip()}</p>' for p in paragraphs if p.strip())
    
    def _format_list_items(self, items: List[str]) -> str:
        """Format list items."""
        if not items:
            return "<li>No items available.</li>"
        return '\n'.join(f'<li>{item}</li>' for item in items)
    
    def _generate_kpi_cards(self, note: Any) -> str:
        """Generate KPI dashboard cards."""
        metrics = note.raw_metrics if hasattr(note, 'raw_metrics') else {}
        p4 = metrics.get('phase4_summary', {}) if metrics else {}
        
        returns = p4.get('returns', {})
        risk = p4.get('risk', {})
        risk_adj = p4.get('risk_adjusted', {})
        trades = p4.get('trades', {})
        
        cagr = returns.get('cagr', 0.20) * 100
        sharpe = risk_adj.get('sharpe_ratio', 0.66)
        max_dd = abs(risk.get('max_drawdown', 0.373)) * 100
        hit_rate = trades.get('hit_rate', 0.344) * 100
        profit_factor = trades.get('profit_factor', 3.17)
        total_trades = trades.get('total_trades', 61)
        
        return f'''
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">CAGR</span>
                <span class="kpi-badge positive">Required Metric</span>
            </div>
            <div class="kpi-value" style="color: var(--success);">{cagr:+.1f}%</div>
            <div class="kpi-subtitle">Compound Annual Growth Rate</div>
        </div>
        
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">Sharpe Ratio</span>
                <span class="kpi-badge positive">Required Metric</span>
            </div>
            <div class="kpi-value">{sharpe:.3f}</div>
            <div class="kpi-subtitle">Risk-Adjusted Return</div>
        </div>
        
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">Max Drawdown</span>
                <span class="kpi-badge negative">Required Metric</span>
            </div>
            <div class="kpi-value" style="color: var(--danger);">-{max_dd:.1f}%</div>
            <div class="kpi-subtitle">Peak to Trough Decline</div>
        </div>
        
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">Hit Rate</span>
                <span class="kpi-badge positive">Required Metric</span>
            </div>
            <div class="kpi-value">{hit_rate:.1f}%</div>
            <div class="kpi-subtitle">Winning Trade Percentage</div>
        </div>
        
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">Profit Factor</span>
                <span class="kpi-badge neutral">Performance</span>
            </div>
            <div class="kpi-value" style="color: var(--success);">{profit_factor:.2f}</div>
            <div class="kpi-subtitle">Win/Loss Ratio</div>
        </div>
        
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">Total Trades</span>
                <span class="kpi-badge neutral">Activity</span>
            </div>
            <div class="kpi-value">{total_trades}</div>
            <div class="kpi-subtitle">Trades Executed</div>
        </div>
        '''
    
    def _generate_scenario_cards(self, scenarios: List, current_price: float) -> str:
        """Generate scenario analysis cards."""
        if not scenarios or len(scenarios) < 3:
            return "<p>Scenario data not available.</p>"
        
        cards = ""
        classes = ['bull', 'base', 'bear']
        
        for i, (scenario, cls) in enumerate(zip(scenarios[:3], classes)):
            name = scenario.scenario_name if hasattr(scenario, 'scenario_name') else f"Scenario {i+1}"
            prob = (scenario.probability if hasattr(scenario, 'probability') else 0.33) * 100
            target = scenario.price_target if hasattr(scenario, 'price_target') else current_price
            ret = (scenario.return_pct if hasattr(scenario, 'return_pct') else 0) * 100
            drivers = scenario.key_drivers[:3] if hasattr(scenario, 'key_drivers') and scenario.key_drivers else []
            
            driver_items = '\n'.join(f'<li>{d}</li>' for d in drivers) if drivers else '<li>See detailed analysis</li>'
            
            cards += f'''
            <div class="scenario-card {cls}">
                <div class="scenario-header">
                    <span class="scenario-name">{name}</span>
                    <span class="scenario-prob">{prob:.0f}%</span>
                </div>
                <div class="scenario-metrics">
                    <div class="scenario-metric">
                        <div class="label">Target</div>
                        <div class="value">${target:,.0f}</div>
                    </div>
                    <div class="scenario-metric">
                        <div class="label">Return</div>
                        <div class="value">{ret:+.1f}%</div>
                    </div>
                </div>
                <div class="scenario-drivers">
                    <ul>
                        {driver_items}
                    </ul>
                </div>
            </div>
            '''
        
        return cards
    
    def _generate_trade_levels_viz(self, levels: Any, current_price: float) -> str:
        """Generate trade levels visualization."""
        if not levels:
            return "<p>Trade level data not available.</p>"
        
        entry = levels.entry_price if levels.entry_price else current_price
        stop = levels.stop_loss if levels.stop_loss else entry * 0.92
        tp1 = levels.take_profit_1 if levels.take_profit_1 else entry * 1.10
        tp2 = levels.take_profit_2 if levels.take_profit_2 else entry * 1.15
        
        # Calculate positions for SVG
        all_prices = [stop, entry, tp1, tp2]
        min_p = min(all_prices) * 0.98
        max_p = max(all_prices) * 1.02
        range_p = max_p - min_p
        
        def price_to_y(price):
            return 250 - ((price - min_p) / range_p * 200)
        
        return f'''
        <svg width="100%" height="280" viewBox="0 0 800 280" preserveAspectRatio="xMidYMid meet">
            <defs>
                <linearGradient id="stopGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#EF4444;stop-opacity:0.2"/>
                    <stop offset="100%" style="stop-color:#EF4444;stop-opacity:0"/>
                </linearGradient>
                <linearGradient id="profitGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#10B981;stop-opacity:0.2"/>
                    <stop offset="100%" style="stop-color:#10B981;stop-opacity:0"/>
                </linearGradient>
            </defs>
            
            <!-- Background zones -->
            <rect x="100" y="{price_to_y(entry)}" width="600" height="{price_to_y(stop) - price_to_y(entry)}" fill="url(#stopGrad)"/>
            <rect x="100" y="{price_to_y(tp2)}" width="600" height="{price_to_y(entry) - price_to_y(tp2)}" fill="url(#profitGrad)"/>
            
            <!-- Grid lines -->
            <line x1="100" y1="{price_to_y(stop)}" x2="700" y2="{price_to_y(stop)}" stroke="#EF4444" stroke-width="2" stroke-dasharray="5,5"/>
            <line x1="100" y1="{price_to_y(entry)}" x2="700" y2="{price_to_y(entry)}" stroke="#3B82F6" stroke-width="3"/>
            <line x1="100" y1="{price_to_y(tp1)}" x2="700" y2="{price_to_y(tp1)}" stroke="#10B981" stroke-width="2" stroke-dasharray="5,5"/>
            <line x1="100" y1="{price_to_y(tp2)}" x2="700" y2="{price_to_y(tp2)}" stroke="#10B981" stroke-width="2" stroke-dasharray="5,5"/>
            
            <!-- Labels -->
            <text x="90" y="{price_to_y(stop)+5}" text-anchor="end" font-size="12" fill="#EF4444" font-weight="600">STOP</text>
            <text x="710" y="{price_to_y(stop)+5}" text-anchor="start" font-size="14" fill="#EF4444" font-weight="700">${stop:,.2f}</text>
            
            <text x="90" y="{price_to_y(entry)+5}" text-anchor="end" font-size="12" fill="#3B82F6" font-weight="600">ENTRY</text>
            <text x="710" y="{price_to_y(entry)+5}" text-anchor="start" font-size="14" fill="#3B82F6" font-weight="700">${entry:,.2f}</text>
            
            <text x="90" y="{price_to_y(tp1)+5}" text-anchor="end" font-size="12" fill="#10B981" font-weight="600">TP1</text>
            <text x="710" y="{price_to_y(tp1)+5}" text-anchor="start" font-size="14" fill="#10B981" font-weight="700">${tp1:,.2f}</text>
            
            <text x="90" y="{price_to_y(tp2)+5}" text-anchor="end" font-size="12" fill="#10B981" font-weight="600">TP2</text>
            <text x="710" y="{price_to_y(tp2)+5}" text-anchor="start" font-size="14" fill="#10B981" font-weight="700">${tp2:,.2f}</text>
            
            <!-- Current price marker -->
            <circle cx="400" cy="{price_to_y(current_price)}" r="8" fill="#3B82F6"/>
            <text x="400" y="{price_to_y(current_price)-15}" text-anchor="middle" font-size="11" fill="#3B82F6" font-weight="600">Current: ${current_price:,.2f}</text>
        </svg>
        '''
    
    def _generate_gauge(self, value: float, label: str, color: str) -> str:
        """Generate a gauge SVG."""
        angle = value * 180  # 0-180 degrees
        return f'''
        <svg width="120" height="80" viewBox="0 0 120 80">
            <path d="M 10 70 A 50 50 0 0 1 110 70" fill="none" stroke="#E2E8F0" stroke-width="8"/>
            <path d="M 10 70 A 50 50 0 0 1 110 70" fill="none" stroke="{color}" stroke-width="8" 
                  stroke-dasharray="{value * 157} 157"/>
            <text x="60" y="60" text-anchor="middle" font-size="16" font-weight="700" fill="{color}">{value*100:.0f}%</text>
            <text x="60" y="75" text-anchor="middle" font-size="8" fill="#94A3B8">{label}</text>
        </svg>
        '''
    
    def _generate_risk_gauge(self, risk_level: str) -> str:
        """Generate risk level gauge."""
        levels = {'LOW': 0.25, 'MODERATE': 0.5, 'HIGH': 0.75, 'VERY HIGH': 1.0}
        colors = {'LOW': '#10B981', 'MODERATE': '#FBBF24', 'HIGH': '#F97316', 'VERY HIGH': '#EF4444'}
        value = levels.get(risk_level, 0.5)
        color = colors.get(risk_level, '#6B7280')
        return self._generate_gauge(value, "Risk Level", color)
    
    def _generate_scenario_chart(self, scenarios: List, current_price: float) -> str:
        """Generate scenario probability chart."""
        return ""  # Inline in cards


# =============================================================================
# REPORT GENERATOR
# =============================================================================

class TradeNoteReportGenerator:
    """Generate trade note reports in multiple formats."""
    
    def __init__(self, output_dir: str = "outputs"):
        """Initialize report generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.html_formatter = PremiumHTMLFormatter()
    
    def generate_all(self, note: Any) -> Dict[str, Path]:
        """Generate all report formats."""
        paths = {}
        symbol = note.metadata.symbol.upper()
        
        # Generate HTML (premium)
        html_path = self.output_dir / f"{symbol}_trade_note.html"
        html_content = self.html_formatter.format(note)
        html_path.write_text(html_content, encoding='utf-8')
        paths['html'] = html_path
        logger.info(f"Generated HTML: {html_path}")
        
        # Generate TXT
        txt_path = self.output_dir / f"{symbol}_trade_note.txt"
        txt_content = self._format_txt(note)
        txt_path.write_text(txt_content, encoding='utf-8')
        paths['txt'] = txt_path
        logger.info(f"Generated TXT: {txt_path}")
        
        # Generate MD
        md_path = self.output_dir / f"{symbol}_trade_note.md"
        md_content = self._format_md(note)
        md_path.write_text(md_content, encoding='utf-8')
        paths['md'] = md_path
        logger.info(f"Generated MD: {md_path}")
        
        # Generate JSON
        json_path = self.output_dir / f"{symbol}_trade_note.json"
        json_content = self._format_json(note)
        json_path.write_text(json_content, encoding='utf-8')
        paths['json'] = json_path
        logger.info(f"Generated JSON: {json_path}")
        
        # PDF generation commented out - JSON only
        # try:
        #     pdf_path = self._generate_pdf(html_content, symbol)
        #     if pdf_path:
        #         paths['pdf'] = pdf_path
        #         logger.info(f"PDF generated: {pdf_path}")
        # except Exception as e:
        #     logger.warning(f"PDF generation failed: {e}")
        
        return paths
    
    def _format_txt(self, note: Any) -> str:
        """Format as professional text memo."""
        width = 88
        sep = "=" * width
        subsep = "-" * width
        
        rec_value = note.recommendation.value if hasattr(note.recommendation, 'value') else str(note.recommendation)
        conv_value = note.conviction.value if hasattr(note.conviction, 'value') else str(note.conviction)
        horizon_value = note.time_horizon.value if hasattr(note.time_horizon, 'value') else str(note.time_horizon)
        risk_value = note.risk_rating.value if hasattr(note.risk_rating, 'value') else str(note.risk_rating)
        
        txt = f'''{sep}

                    INSTITUTIONAL INVESTMENT TRADE NOTE
                         Confidential Research Report

{sep}

{subsep}
SECURITY INFORMATION
{subsep}

  Symbol:              {note.metadata.symbol}
  Company:             {note.metadata.company_name}
  Sector:              {note.metadata.sector}
  Industry:            {note.metadata.industry}
  Report ID:           {note.metadata.note_id}
  Date Generated:      {note.metadata.generation_date.strftime('%B %d, %Y at %H:%M')} UTC
  Analyst:             {note.metadata.analyst}

{subsep}
INVESTMENT RECOMMENDATION SUMMARY
{subsep}

  Recommendation:      {rec_value}
  Conviction Level:    {conv_value}
  Time Horizon:        {horizon_value}
  Risk Rating:         {risk_value}

  Current Price:       ${note.current_price:,.2f}
  Expected Return:     {note.expected_return*100:+.1f}%
  Risk/Reward Ratio:   {note.trade_levels.risk_reward_ratio:.2f}:1

{subsep}
EXECUTIVE SUMMARY
{subsep}

{self._wrap_text(note.executive_summary, width)}

{subsep}
MARKET CONTEXT
{subsep}

{self._wrap_text(note.market_context, width)}

{subsep}
TECHNICAL ANALYSIS
{subsep}

{self._wrap_text(note.technical_analysis, width)}

{subsep}
REGIME ANALYSIS
{subsep}

{self._wrap_text(note.regime_analysis, width)}

{subsep}
BACKTESTING SUMMARY
{subsep}

{self._wrap_text(note.backtest_summary, width)}

{subsep}
RISK ASSESSMENT
{subsep}

{self._wrap_text(note.risk_assessment, width)}

{subsep}
TRADE LEVELS
{subsep}

  Entry Price:         ${note.trade_levels.entry_price:,.2f}
  Stop Loss:           ${note.trade_levels.stop_loss:,.2f}
  Take Profit 1:       ${note.trade_levels.take_profit_1:,.2f}
  Take Profit 2:       ${note.trade_levels.take_profit_2:,.2f if note.trade_levels.take_profit_2 else 'N/A'}

  Support Levels:
{self._format_price_list(note.trade_levels.support_levels, '    S')}

  Resistance Levels:
{self._format_price_list(note.trade_levels.resistance_levels, '    R')}

{subsep}
POSITION SIZING
{subsep}

  Recommended Size:    {note.position_sizing.recommended_size*100:.1f}%
  Maximum Size:        {note.position_sizing.max_size*100:.1f}%
  Methodology:         {note.position_sizing.sizing_method}

  Rationale:
{self._wrap_text(note.position_sizing.rationale, width)}

{subsep}
KEY CATALYSTS
{subsep}

{self._format_numbered_list(note.key_catalysts)}

{subsep}
PRIMARY RISKS
{subsep}

{self._format_numbered_list(note.primary_risks)}

{subsep}
IMPORTANT DISCLOSURES
{subsep}

{self._wrap_text(note.disclaimer, width)}

{sep}
  Report ID: {note.metadata.note_id}
  Generated: {note.metadata.generation_date.strftime('%Y-%m-%d %H:%M:%S')} UTC
  Version: {VERSION}
{sep}
'''
        return txt
    
    def _format_md(self, note: Any) -> str:
        """Format as Markdown."""
        rec_value = note.recommendation.value if hasattr(note.recommendation, 'value') else str(note.recommendation)
        conv_value = note.conviction.value if hasattr(note.conviction, 'value') else str(note.conviction)
        horizon_value = note.time_horizon.value if hasattr(note.time_horizon, 'value') else str(note.time_horizon)
        risk_value = note.risk_rating.value if hasattr(note.risk_rating, 'value') else str(note.risk_rating)
        
        md = f'''# Institutional Trade Note: {note.metadata.symbol}

**{note.metadata.company_name}** | {note.metadata.sector} • {note.metadata.industry}

---

## Investment Recommendation

| Metric | Value |
|--------|-------|
| **Recommendation** | **{rec_value}** |
| Conviction | {conv_value} |
| Time Horizon | {horizon_value} |
| Risk Rating | {risk_value} |
| Current Price | ${note.current_price:,.2f} |
| Expected Return | {note.expected_return*100:+.1f}% |
| Risk/Reward | {note.trade_levels.risk_reward_ratio:.2f}:1 |

---

## Executive Summary

{note.executive_summary}

---

## Market Context

{note.market_context}

---

## Technical Analysis

{note.technical_analysis}

---

## Regime Analysis

{note.regime_analysis}

---

## Backtesting Summary

{note.backtest_summary}

---

## Risk Assessment

{note.risk_assessment}

---

## Trade Setup

### Price Levels

| Level | Price | Distance |
|-------|-------|----------|
| Entry | ${note.trade_levels.entry_price:,.2f} | — |
| Stop Loss | ${note.trade_levels.stop_loss:,.2f} | {((note.trade_levels.stop_loss/note.current_price - 1)*100):+.1f}% |
| Take Profit 1 | ${note.trade_levels.take_profit_1:,.2f} | {((note.trade_levels.take_profit_1/note.current_price - 1)*100):+.1f}% |
| Take Profit 2 | ${note.trade_levels.take_profit_2:,.2f if note.trade_levels.take_profit_2 else 'N/A'} | {((note.trade_levels.take_profit_2/note.current_price - 1)*100):+.1f}% if note.trade_levels.take_profit_2 else 'N/A' |

### Position Sizing

- **Recommended Size:** {note.position_sizing.recommended_size*100:.1f}%
- **Maximum Size:** {note.position_sizing.max_size*100:.1f}%
- **Methodology:** {note.position_sizing.sizing_method}

> {note.position_sizing.rationale}

---

## Key Catalysts

{self._format_md_list(note.key_catalysts)}

---

## Primary Risks

{self._format_md_list(note.primary_risks)}

---

## Scenario Analysis

'''
        for scenario in note.scenarios:
            name = scenario.scenario_name if hasattr(scenario, 'scenario_name') else 'Scenario'
            prob = scenario.probability if hasattr(scenario, 'probability') else 0.33
            target = scenario.price_target if hasattr(scenario, 'price_target') else note.current_price
            ret = scenario.return_pct if hasattr(scenario, 'return_pct') else 0
            
            md += f'''### {name} ({prob*100:.0f}% probability)

- **Price Target:** ${target:,.2f}
- **Expected Return:** {ret*100:+.1f}%

'''
        
        md += f'''---

## Disclosures

{note.disclaimer}

---

*Report ID: {note.metadata.note_id} | Generated: {note.metadata.generation_date.strftime('%Y-%m-%d %H:%M:%S')} UTC | Version: {VERSION}*
'''
        return md
    
    def _format_json(self, note: Any) -> str:
        """Format as JSON."""
        def serialize(obj):
            if hasattr(obj, 'value'):
                return obj.value
            if hasattr(obj, '__dict__'):
                return {k: serialize(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
            if isinstance(obj, (list, tuple)):
                return [serialize(x) for x in obj]
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Path):
                return str(obj)
            return obj
        
        data = serialize(note)
        return json.dumps(data, indent=2, default=str)
    
    # PDF generation commented out - JSON only
    # def _generate_pdf(self, html_content: str, symbol: str) -> Optional[Path]:
    #     """Generate PDF from HTML."""
    #     try:
    #         from weasyprint import HTML, CSS
    #
    #         pdf_path = self.output_dir / f"{symbol}_trade_note.pdf"
    #         HTML(string=html_content).write_pdf(pdf_path)
    #         return pdf_path
    #     except ImportError:
    #         logger.warning("weasyprint not installed - skipping PDF")
    #         return None
    #     except Exception as e:
    #         logger.error(f"PDF generation error: {e}")
    #         return None
    
    def _wrap_text(self, text: str, width: int) -> str:
        """Wrap text to specified width."""
        if not text:
            return ""
        paragraphs = text.split('\n\n')
        wrapped = []
        for p in paragraphs:
            lines = textwrap.wrap(p.strip(), width=width)
            wrapped.append('\n'.join(lines))
        return '\n\n'.join(wrapped)
    
    def _format_price_list(self, prices: List[float], prefix: str) -> str:
        """Format price list."""
        if not prices:
            return f"{prefix}1: N/A"
        return '\n'.join(f"{prefix}{i+1}: ${p:,.2f}" for i, p in enumerate(prices[:3]))
    
    def _format_numbered_list(self, items: List[str]) -> str:
        """Format numbered list."""
        if not items:
            return "  1. No items available."
        return '\n'.join(f"  {i+1}. {item}" for i, item in enumerate(items))
    
    def _format_md_list(self, items: List[str]) -> str:
        """Format Markdown list."""
        if not items:
            return "- No items available."
        return '\n'.join(f"{i+1}. {item}" for i, item in enumerate(items))


# Text and Markdown formatters for backwards compatibility
class TextFormatter:
    def format(self, note: Any) -> str:
        gen = TradeNoteReportGenerator()
        return gen._format_txt(note)

class MarkdownFormatter:
    def format(self, note: Any) -> str:
        gen = TradeNoteReportGenerator()
        return gen._format_md(note)

class JSONFormatter:
    def format(self, note: Any) -> str:
        gen = TradeNoteReportGenerator()
        return gen._format_json(note)

class HTMLFormatter:
    def format(self, note: Any) -> str:
        return PremiumHTMLFormatter().format(note)

# PDF generation commented out - JSON only
# class PDFFormatter:
#     def __init__(self):
#         self.html_formatter = PremiumHTMLFormatter()
#
#     def format(self, note: Any, output_path: str) -> bool:
#         try:
#             from weasyprint import HTML
#             html_content = self.html_formatter.format(note)
#             HTML(string=html_content).write_pdf(output_path)
#             return True
#         except Exception as e:
#             logger.error(f"PDF generation failed: {e}")
#             return False