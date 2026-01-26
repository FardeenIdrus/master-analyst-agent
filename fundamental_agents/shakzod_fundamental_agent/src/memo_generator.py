#!/usr/bin/env python3
"""
Memo Generator - Investment Memorandum Generation with Claude AI
MSc Coursework: AI Agents in Asset Management - Step 9
"""

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Claude API configuration - Anthropic API key for memo generation
# Get your API key from: https://console.anthropic.com/
# Set this in your environment variables or config file
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEFAULT_MODEL = "claude-sonnet-4-20250514"


class InvestmentRecommendation(Enum):
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"


class MemoSection(Enum):
    INVESTMENT_THESIS = "investment_thesis"
    EXECUTIVE_SUMMARY = "executive_summary"
    PROFITABILITY = "profitability"
    CASH_FLOW = "cash_flow"
    EARNINGS_QUALITY = "earnings_quality"
    WORKING_CAPITAL = "working_capital"
    VALUATION = "valuation"
    RISKS = "risks"
    CATALYSTS = "catalysts"
    CONCLUSION = "conclusion"


class MemoGenerationMode(Enum):
    LLM = "llm"
    RULE_BASED = "rule_based"


@dataclass
class MemoSectionContent:
    section: MemoSection
    title: str
    content: str
    key_points: List[str] = field(default_factory=list)


@dataclass
class AnalysisSummary:
    ticker: str
    company_name: str
    sector: str = "Technology"
    industry: str = "Unknown"
    current_price: float = 0.0
    market_cap: float = 0.0
    revenue_current: float = 0.0
    revenue_prior: float = 0.0
    revenue_growth: float = 0.0
    ebit_current: float = 0.0
    ebit_prior: float = 0.0
    ebit_change: float = 0.0
    ebit_change_pct: float = 0.0
    volume_effect: float = 0.0
    margin_effect: float = 0.0
    gross_margin_current: float = 0.0
    gross_margin_prior: float = 0.0
    gross_margin_change: float = 0.0
    operating_margin_current: float = 0.0
    operating_margin_prior: float = 0.0
    operating_margin_change: float = 0.0
    net_margin_current: float = 0.0
    primary_profit_driver: str = "Volume"
    net_income: float = 0.0
    operating_cash_flow: float = 0.0
    free_cash_flow: float = 0.0
    capex: float = 0.0
    depreciation: float = 0.0
    cash_conversion_rate: float = 0.0
    fcf_conversion_rate: float = 0.0
    fcf_margin: float = 0.0
    capex_to_revenue: float = 0.0
    capex_to_da: float = 0.0
    cash_flow_quality: str = "Adequate"
    accruals_ratio: float = 0.0
    earnings_quality_score: float = 70.0
    earnings_quality_rating: str = "Good"
    red_flags: List[str] = field(default_factory=list)
    ar_divergence: float = 0.0
    inventory_divergence: float = 0.0
    dso: float = 0.0
    dso_change: float = 0.0
    dio: float = 0.0
    dio_change: float = 0.0
    dpo: float = 0.0
    dpo_change: float = 0.0
    ccc: float = 0.0
    ccc_change: float = 0.0
    working_capital_trend: str = "Stable"
    wc_cash_impact: float = 0.0
    net_working_capital: float = 0.0
    current_ratio: float = 1.0
    quick_ratio: float = 1.0
    roe: float = 0.0
    roa: float = 0.0
    roic: float = 0.0
    debt_to_equity: float = 0.0
    debt_to_assets: float = 0.0
    interest_coverage: float = 0.0
    ni_growth: float = 0.0
    asset_turnover: float = 0.0
    equity_multiplier: float = 0.0
    pe_ratio: float = 0.0
    forward_pe: float = 0.0
    ev_ebitda: float = 0.0
    ev_revenue: float = 0.0
    price_to_fcf: float = 0.0
    price_to_book: float = 0.0
    price_to_sales: float = 0.0
    peg_ratio: float = 0.0
    dividend_yield: float = 0.0
    valuation_assessment: str = "Fairly Valued"
    implied_growth: float = 0.0
    enterprise_value: float = 0.0
    current_period: str = "2024"
    prior_period: str = "2023"


@dataclass
class InvestmentMemo:
    ticker: str
    company_name: str
    recommendation: InvestmentRecommendation
    confidence_score: float
    target_price: Optional[float]
    sections: List[MemoSectionContent]
    generation_mode: MemoGenerationMode
    generated_at: datetime
    summary: Optional[AnalysisSummary] = None
    score_breakdown: Optional[Dict[str, Any]] = None
    llm_raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        s = self.summary
        upside = ((self.target_price / s.current_price) - 1) * 100 if s and s.current_price and self.target_price else 0
        result = {
            "document_info": {"title": f"Investment Memorandum: {self.company_name}", "ticker": self.ticker, "report_date": self.generated_at.isoformat(), "generation_mode": self.generation_mode.value},
            "recommendation": {"rating": self.recommendation.value, "confidence_score": self.confidence_score, "target_price": self.target_price, "current_price": s.current_price if s else 0, "upside_potential": upside},
            "score_breakdown": self.score_breakdown or {},
            "company_profile": {"name": self.company_name, "ticker": self.ticker, "sector": s.sector if s else "Unknown", "industry": s.industry if s else "Unknown", "market_cap": s.market_cap if s else 0},
            "financial_metrics": {
                "profitability": {"revenue_current": s.revenue_current if s else 0, "revenue_growth_pct": s.revenue_growth * 100 if s else 0, "ebit_current": s.ebit_current if s else 0, "ebit_change_pct": s.ebit_change_pct * 100 if s else 0, "gross_margin": s.gross_margin_current * 100 if s else 0, "operating_margin": s.operating_margin_current * 100 if s else 0, "net_margin": s.net_margin_current * 100 if s else 0},
                "cash_flow": {"operating_cash_flow": s.operating_cash_flow if s else 0, "free_cash_flow": s.free_cash_flow if s else 0, "cash_conversion_rate": s.cash_conversion_rate * 100 if s else 0, "fcf_margin": s.fcf_margin * 100 if s else 0, "quality_rating": s.cash_flow_quality if s else "Unknown"},
                "earnings_quality": {"quality_score": s.earnings_quality_score if s else 0, "quality_rating": s.earnings_quality_rating if s else "Unknown", "accruals_ratio": s.accruals_ratio * 100 if s else 0, "red_flags": s.red_flags if s else []},
                "working_capital": {"dso": s.dso if s else 0, "dio": s.dio if s else 0, "dpo": s.dpo if s else 0, "ccc": s.ccc if s else 0, "current_ratio": s.current_ratio if s else 0},
                "return_metrics": {"roe": s.roe * 100 if s else 0, "roa": s.roa * 100 if s else 0, "roic": s.roic * 100 if s else 0},
                "valuation": {"pe_ratio": s.pe_ratio if s else 0, "ev_ebitda": s.ev_ebitda if s else 0, "price_to_fcf": s.price_to_fcf if s else 0, "price_to_book": s.price_to_book if s else 0, "price_to_sales": s.price_to_sales if s else 0, "assessment": s.valuation_assessment if s else "Unknown"}
            },
            "analysis_sections": [{"section_id": sec.section.value, "title": sec.title, "content": sec.content, "key_points": sec.key_points} for sec in self.sections]
        }
        return result

    def to_markdown(self) -> str:
        s = self.summary
        md = [f"# Investment Memorandum: {self.company_name} ({self.ticker})", "", "---", "", "## Document Information", "", "| Field | Value |", "|-------|-------|"]
        md.append(f"| **Report Date** | {self.generated_at.strftime('%B %d, %Y')} |")
        md.append(f"| **Recommendation** | **{self.recommendation.value}** |")
        md.append(f"| **Confidence** | {self.confidence_score:.0f}% |")
        if self.target_price: md.append(f"| **Price Target** | ${self.target_price:.2f} |")
        if s and s.current_price: md.append(f"| **Current Price** | ${s.current_price:.2f} |")
        md.append(f"| **Generation Mode** | {self.generation_mode.value.upper()} |")
        md.append("")
        
        if self.score_breakdown:
            md.extend(["---", "", "## Recommendation Score Breakdown", "", "| Factor | Score | Weight | Contribution |", "|--------|-------|--------|--------------|"])
            scores, weights = self.score_breakdown.get('scores', {}), self.score_breakdown.get('weights', {})
            for f in ['cash_flow', 'earnings_quality', 'profitability', 'valuation', 'growth']:
                sc, wt = scores.get(f, 0), weights.get(f, 0)
                md.append(f"| {f.replace('_', ' ').title()} | {sc:.0f}/100 | {wt:.0%} | {sc * wt:.1f} |")
            md.append(f"| **TOTAL** | | | **{self.score_breakdown.get('total_score', 0):.1f}** |")
            md.append("")
        
        if s:
            md.extend(["---", "", "## Key Financial Metrics", "", "### Profitability", "", "| Metric | Current | Prior | Change |", "|--------|---------|-------|--------|"])
            md.append(f"| Revenue | ${s.revenue_current:,.0f}M | ${s.revenue_prior:,.0f}M | {s.revenue_growth*100:+.1f}% |")
            md.append(f"| EBIT | ${s.ebit_current:,.0f}M | ${s.ebit_prior:,.0f}M | {s.ebit_change_pct*100:+.1f}% |")
            md.append(f"| Gross Margin | {s.gross_margin_current*100:.1f}% | {s.gross_margin_prior*100:.1f}% | {s.gross_margin_change*100:+.1f}pp |")
            md.append(f"| Operating Margin | {s.operating_margin_current*100:.1f}% | {s.operating_margin_prior*100:.1f}% | {s.operating_margin_change*100:+.1f}pp |")
            md.extend(["", "### Cash Flow", "", "| Metric | Value |", "|--------|-------|"])
            md.append(f"| Operating Cash Flow | ${s.operating_cash_flow:,.0f}M |")
            md.append(f"| Free Cash Flow | ${s.free_cash_flow:,.0f}M |")
            md.append(f"| Cash Conversion Rate | {s.cash_conversion_rate*100:.0f}% |")
            md.append(f"| FCF Margin | {s.fcf_margin*100:.1f}% |")
            md.append(f"| Quality Rating | {s.cash_flow_quality} |")
            md.extend(["", "### Return Metrics", "", "| Metric | Value |", "|--------|-------|"])
            md.append(f"| ROE | {s.roe*100:.1f}% |")
            md.append(f"| ROA | {s.roa*100:.1f}% |")
            md.append(f"| ROIC | {s.roic*100:.1f}% |")
            md.extend(["", "### Valuation", "", "| Multiple | Value |", "|----------|-------|"])
            md.append(f"| P/E | {s.pe_ratio:.1f}x |")
            md.append(f"| EV/EBITDA | {s.ev_ebitda:.1f}x |")
            md.append(f"| P/FCF | {s.price_to_fcf:.1f}x |")
            md.append(f"| Assessment | {s.valuation_assessment} |")
            md.extend(["", "### Working Capital", "", "| Metric | Days | YoY |", "|--------|------|-----|"])
            md.append(f"| DSO | {s.dso:.0f} | {s.dso_change:+.0f} |")
            md.append(f"| DIO | {s.dio:.0f} | {s.dio_change:+.0f} |")
            md.append(f"| DPO | {s.dpo:.0f} | {s.dpo_change:+.0f} |")
            md.append(f"| CCC | {s.ccc:.0f} | {s.ccc_change:+.0f} |")
            md.append("")
        
        md.extend(["---", "", "## Detailed Analysis", ""])
        for sec in self.sections:
            md.extend([f"### {sec.title}", "", sec.content, ""])
            if sec.key_points:
                md.append("**Key Points:**")
                for p in sec.key_points: md.append(f"- {p}")
                md.append("")
            md.extend(["---", ""])
        
        md.extend(["", f"*Generated by Fundamental Analyst AI Agent | Mode: {self.generation_mode.value.upper()} | Powered by Claude AI*", "", "**Disclaimer:** This analysis is for educational purposes only."])
        return "\n".join(md)

    def to_html(self) -> str:
        s = self.summary
        colors = {"STRONG BUY": ("#059669", "#d1fae5", "#065f46"), "BUY": ("#10b981", "#d1fae5", "#047857"), "HOLD": ("#f59e0b", "#fef3c7", "#b45309"), "SELL": ("#ef4444", "#fee2e2", "#b91c1c"), "STRONG SELL": ("#dc2626", "#fee2e2", "#991b1b")}
        accent, accent_bg, accent_dark = colors.get(self.recommendation.value, ("#6b7280", "#f3f4f6", "#374151"))
        
        def chg_cls(v): return "positive" if v >= 0 else "negative"
        
        score_html = ""
        if self.score_breakdown:
            scores, weights, total = self.score_breakdown.get('scores', {}), self.score_breakdown.get('weights', {}), self.score_breakdown.get('total_score', 0)
            rows = "".join(f'<div class="score-row"><span class="factor">{f.replace("_", " ").title()}</span><div class="bar-wrap"><div class="bar" style="width:{min(scores.get(f,0),100)}%"></div></div><span class="sc">{scores.get(f,0):.0f}</span><span class="wt">x{weights.get(f,0):.0%}</span><span class="ct">={scores.get(f,0)*weights.get(f,0):.1f}</span></div>' for f in ['cash_flow','earnings_quality','profitability','valuation','growth'])
            score_html = f'<section class="score-sec"><h2>Score Breakdown</h2><div class="score-grid">{rows}</div><div class="total"><span>Total</span><span class="tval">{total:.1f}</span></div><div class="legend"><span class="sb">Strong Buy 78+</span><span class="b">Buy 65-77</span><span class="h">Hold 50-64</span><span class="s">Sell 38-49</span><span class="ss">Strong Sell &lt;38</span></div></section>'
        
        metrics_html = ""
        if s:
            metrics_html = f'''<section class="metrics-sec"><h2>Financial Metrics</h2><div class="mgrid">
            <div class="mcard"><h3>Profitability</h3><div class="mr"><span>Revenue</span><span>${s.revenue_current:,.0f}M</span></div><div class="mr"><span>Revenue Growth</span><span class="{chg_cls(s.revenue_growth)}">{s.revenue_growth*100:+.1f}%</span></div><div class="mr"><span>EBIT</span><span>${s.ebit_current:,.0f}M</span></div><div class="mr"><span>EBIT Growth</span><span class="{chg_cls(s.ebit_change_pct)}">{s.ebit_change_pct*100:+.1f}%</span></div><div class="mr"><span>Gross Margin</span><span>{s.gross_margin_current*100:.1f}%</span></div><div class="mr"><span>Operating Margin</span><span>{s.operating_margin_current*100:.1f}%</span></div><div class="mr"><span>Net Margin</span><span>{s.net_margin_current*100:.1f}%</span></div></div>
            <div class="mcard"><h3>Cash Flow</h3><div class="mr"><span>Operating CF</span><span>${s.operating_cash_flow:,.0f}M</span></div><div class="mr"><span>Free Cash Flow</span><span>${s.free_cash_flow:,.0f}M</span></div><div class="mr"><span>Cash Conversion</span><span>{s.cash_conversion_rate*100:.0f}%</span></div><div class="mr"><span>FCF Margin</span><span>{s.fcf_margin*100:.1f}%</span></div><div class="mr"><span>CapEx</span><span>${s.capex:,.0f}M</span></div><div class="mr"><span>Quality</span><span class="badge">{s.cash_flow_quality}</span></div></div>
            <div class="mcard"><h3>Returns</h3><div class="mr"><span>ROE</span><span>{s.roe*100:.1f}%</span></div><div class="mr"><span>ROA</span><span>{s.roa*100:.1f}%</span></div><div class="mr"><span>ROIC</span><span>{s.roic*100:.1f}%</span></div><div class="mr"><span>Asset Turnover</span><span>{s.asset_turnover:.2f}x</span></div></div>
            <div class="mcard"><h3>Valuation</h3><div class="mr"><span>P/E</span><span>{s.pe_ratio:.1f}x</span></div><div class="mr"><span>EV/EBITDA</span><span>{s.ev_ebitda:.1f}x</span></div><div class="mr"><span>P/FCF</span><span>{s.price_to_fcf:.1f}x</span></div><div class="mr"><span>P/Book</span><span>{s.price_to_book:.2f}x</span></div><div class="mr"><span>Assessment</span><span class="badge">{s.valuation_assessment}</span></div></div>
            <div class="mcard"><h3>Working Capital</h3><div class="mr"><span>DSO</span><span>{s.dso:.0f} days</span></div><div class="mr"><span>DIO</span><span>{s.dio:.0f} days</span></div><div class="mr"><span>DPO</span><span>{s.dpo:.0f} days</span></div><div class="mr"><span>CCC</span><span>{s.ccc:.0f} days</span></div><div class="mr"><span>Current Ratio</span><span>{s.current_ratio:.2f}x</span></div></div>
            <div class="mcard"><h3>Earnings Quality</h3><div class="mr"><span>Score</span><span>{s.earnings_quality_score:.0f}/100</span></div><div class="mr"><span>Rating</span><span class="badge">{s.earnings_quality_rating}</span></div><div class="mr"><span>Accruals</span><span>{s.accruals_ratio*100:.1f}%</span></div><div class="mr"><span>Red Flags</span><span>{len(s.red_flags) if s.red_flags else 0}</span></div><div class="mr"><span>Debt/Equity</span><span>{s.debt_to_equity:.2f}x</span></div></div>
            </div></section>'''
        
        sections_html = "".join(f'<section class="asec"><h2>{sec.title}</h2><div class="content"><p>{sec.content.replace(chr(10)+chr(10), "</p><p>")}</p></div>{"<div class=kp><h4>Key Points</h4><ul>" + "".join(f"<li>{p}</li>" for p in sec.key_points) + "</ul></div>" if sec.key_points else ""}</section>' for sec in self.sections)
        
        return f'''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Investment Memo: {self.ticker}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@600;700&display=swap" rel="stylesheet">
<style>:root{{--accent:{accent};--accent-bg:{accent_bg};--accent-dark:{accent_dark}}}*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:'Inter',sans-serif;background:linear-gradient(135deg,#f8fafc,#e2e8f0);color:#1e293b;line-height:1.7}}.container{{max-width:1100px;margin:0 auto;padding:40px 20px 80px}}.card{{background:#fff;border-radius:20px;box-shadow:0 20px 40px rgba(0,0,0,0.1);overflow:hidden}}header{{background:linear-gradient(135deg,#0f172a,#1e293b);color:#fff;padding:50px;position:relative}}header::after{{content:'';position:absolute;top:0;right:0;width:40%;height:100%;background:linear-gradient(135deg,transparent,rgba(255,255,255,0.03))}}.doc-label{{display:inline-block;background:rgba(255,255,255,0.1);padding:8px 16px;border-radius:50px;font-size:0.75rem;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:20px}}.ticker{{font-family:'Playfair Display',serif;font-size:3.5rem;font-weight:700;margin-bottom:8px}}.company{{font-size:1.4rem;font-weight:300;opacity:0.9;margin-bottom:6px}}.sector{{font-size:0.9rem;opacity:0.7;margin-bottom:30px}}.rec-box{{display:flex;align-items:center;gap:30px;flex-wrap:wrap}}.rec-badge{{background:var(--accent-bg);color:var(--accent-dark);font-size:1.4rem;font-weight:800;padding:16px 36px;border-radius:50px;letter-spacing:1.5px;box-shadow:0 4px 15px rgba(0,0,0,0.1)}}.conf-box{{text-align:center}}.conf-val{{font-size:2rem;font-weight:700}}.conf-lbl{{font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;opacity:0.8}}.target{{background:rgba(255,255,255,0.1);padding:12px 24px;border-radius:12px}}.target-lbl{{font-size:0.65rem;text-transform:uppercase;letter-spacing:1px;opacity:0.7}}.target-val{{font-size:1.4rem;font-weight:700}}.meta{{display:flex;gap:40px;margin-top:30px;padding-top:25px;border-top:1px solid rgba(255,255,255,0.15);flex-wrap:wrap}}.meta-item span{{display:block}}.meta-item .lbl{{font-size:0.65rem;text-transform:uppercase;letter-spacing:1px;opacity:0.6}}.meta-item .val{{font-size:0.95rem;font-weight:600}}.score-sec,.metrics-sec{{padding:40px 50px;background:#f8fafc;border-bottom:1px solid #e2e8f0}}.score-sec h2,.metrics-sec h2{{font-family:'Playfair Display',serif;font-size:1.4rem;color:#0f172a;margin-bottom:25px;text-align:center}}.score-grid{{max-width:700px;margin:0 auto}}.score-row{{display:grid;grid-template-columns:140px 1fr 50px 60px 60px;align-items:center;gap:12px;padding:12px 0;border-bottom:1px solid #e2e8f0}}.factor{{font-size:0.9rem;font-weight:500}}.bar-wrap{{height:10px;background:#e2e8f0;border-radius:5px;overflow:hidden}}.bar{{height:100%;background:linear-gradient(90deg,#3b82f6,#60a5fa);border-radius:5px}}.sc{{font-weight:600;text-align:right}}.wt{{color:#64748b;font-size:0.85rem;text-align:right}}.ct{{font-weight:600;color:#3b82f6;text-align:right}}.total{{display:flex;justify-content:space-between;max-width:700px;margin:20px auto 0;padding-top:20px;border-top:2px solid #0f172a;font-weight:700}}.tval{{font-size:1.5rem;color:#0f172a}}.legend{{display:flex;justify-content:center;gap:12px;margin-top:20px;flex-wrap:wrap}}.legend span{{font-size:0.7rem;padding:4px 10px;border-radius:4px}}.sb{{background:#d1fae5;color:#065f46}}.b{{background:#dcfce7;color:#166534}}.h{{background:#fef3c7;color:#92400e}}.s{{background:#fee2e2;color:#991b1b}}.ss{{background:#fecaca;color:#7f1d1d}}.mgrid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}}.mcard{{background:#fff;border-radius:12px;padding:24px;box-shadow:0 2px 8px rgba(0,0,0,0.05);border:1px solid #e2e8f0}}.mcard h3{{font-size:0.85rem;font-weight:700;color:#3b82f6;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:16px;padding-bottom:12px;border-bottom:2px solid #3b82f6}}.mr{{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid #f1f5f9}}.mr:last-child{{border-bottom:none}}.mr span:first-child{{font-size:0.85rem;color:#64748b}}.mr span:last-child{{font-size:0.95rem;font-weight:600;color:#1e293b}}.mr .positive{{color:#059669}}.mr .negative{{color:#dc2626}}.mr .badge{{background:#f1f5f9;padding:2px 10px;border-radius:4px;font-size:0.8rem}}main{{padding:50px}}.asec{{margin-bottom:40px;padding-bottom:40px;border-bottom:1px solid #e2e8f0}}.asec:last-child{{border-bottom:none;margin-bottom:0;padding-bottom:0}}.asec h2{{font-family:'Playfair Display',serif;font-size:1.3rem;color:#0f172a;margin-bottom:20px;padding-bottom:10px;border-bottom:3px solid #3b82f6;display:inline-block}}.content{{font-size:1rem;line-height:1.85}}.content p{{margin-bottom:16px}}.kp{{margin-top:25px;padding:25px;background:#f8fafc;border-radius:12px;border-left:4px solid #3b82f6}}.kp h4{{font-size:0.85rem;font-weight:700;color:#3b82f6;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:15px}}.kp ul{{list-style:none}}.kp li{{margin-bottom:10px;padding-left:20px;position:relative;font-size:0.95rem}}.kp li::before{{content:'';position:absolute;left:0;top:8px;width:8px;height:8px;background:#3b82f6;border-radius:50%}}footer{{background:#f8fafc;padding:35px 50px;border-top:1px solid #e2e8f0}}.footer-row{{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:20px}}.branding{{font-size:0.9rem;color:#64748b}}.branding strong{{color:#1e293b;display:block;margin-bottom:4px}}.gen-badge{{display:inline-flex;align-items:center;gap:8px;background:#0f172a;color:#fff;padding:10px 20px;border-radius:50px;font-size:0.8rem;font-weight:600}}.disclaimer{{margin-top:20px;padding:20px;background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.1);border-radius:8px;font-size:0.75rem;color:#64748b;line-height:1.6}}.disclaimer strong{{color:#dc2626}}@media print{{body{{background:#fff}}.card{{box-shadow:none}}}}@media(max-width:768px){{header{{padding:30px}}.ticker{{font-size:2.5rem}}main{{padding:30px}}.score-row{{grid-template-columns:1fr}}}}</style>
</head><body><div class="container"><article class="card">
<header><span class="doc-label">Investment Memorandum</span><h1 class="ticker">{self.ticker}</h1><p class="company">{self.company_name}</p><p class="sector">{s.sector if s else 'N/A'} | {s.industry if s else 'N/A'}</p>
<div class="rec-box"><div class="rec-badge">{self.recommendation.value}</div><div class="conf-box"><div class="conf-val">{self.confidence_score:.0f}%</div><div class="conf-lbl">Confidence</div></div>{f'<div class="target"><div class="target-lbl">Price Target</div><div class="target-val">${self.target_price:.2f}</div></div>' if self.target_price else ''}</div>
<div class="meta"><div class="meta-item"><span class="lbl">Report Date</span><span class="val">{self.generated_at.strftime("%B %d, %Y")}</span></div><div class="meta-item"><span class="lbl">Current Price</span><span class="val">${s.current_price:.2f}</span></div><div class="meta-item"><span class="lbl">Market Cap</span><span class="val">${s.market_cap/1e9:.1f}B</span></div><div class="meta-item"><span class="lbl">Period</span><span class="val">FY{s.prior_period} - FY{s.current_period}</span></div><div class="meta-item"><span class="lbl">Mode</span><span class="val">{self.generation_mode.value.upper()}</span></div></div></header>
{score_html}{metrics_html}<main>{sections_html}</main>
<footer><div class="footer-row"><div class="branding"><strong>Fundamental Analyst AI Agent</strong>MSc AI Agents in Asset Management</div><span class="gen-badge">Claude AI Powered</span></div><div class="disclaimer"><strong>Disclaimer:</strong> This investment memorandum is for educational purposes only. It does not constitute investment advice. Past performance does not guarantee future results.</div></footer>
</article></div></body></html>'''


class MemoGenerator:
    """Investment Memo Generator with Claude AI integration."""
    
    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv('ANTHROPIC_API_KEY') or ANTHROPIC_API_KEY
        self._use_llm = bool(self._api_key)
        self._model = DEFAULT_MODEL
        self._client = None
        self._score_breakdown = None
    
    def generate(self, summary: AnalysisSummary) -> InvestmentMemo:
        rec, conf = self._calculate_recommendation(summary)
        
        # Calculate target price first (before generating sections)
        target = self._calculate_target_price(summary, rec)
        
        # CRITICAL: Validate recommendation against target price
        # A BUY with negative upside is illogical and must be corrected
        rec, conf = self._validate_and_adjust_recommendation(rec, conf, target, summary.current_price)
        
        # Recalculate target if recommendation was adjusted
        target = self._calculate_target_price(summary, rec)
        
        if self._use_llm:
            try:
                sections, raw = self._generate_llm_sections(summary, rec)
                mode = MemoGenerationMode.LLM
            except Exception as e:
                logger.warning(f"LLM failed: {e}")
                sections = self._generate_rule_based_sections(summary, rec)
                raw, mode = "", MemoGenerationMode.RULE_BASED
        else:
            sections, raw, mode = self._generate_rule_based_sections(summary, rec), "", MemoGenerationMode.RULE_BASED
        
        return InvestmentMemo(ticker=summary.ticker, company_name=summary.company_name, recommendation=rec, confidence_score=conf, target_price=target, sections=sections, generation_mode=mode, generated_at=datetime.now(), summary=summary, score_breakdown=self._score_breakdown, llm_raw_response=raw)
    
    def _calculate_recommendation(self, s: AnalysisSummary) -> Tuple[InvestmentRecommendation, float]:
        scores, weights = {}, {'cash_flow': 0.35, 'earnings_quality': 0.25, 'profitability': 0.20, 'valuation': 0.15, 'growth': 0.05}
        
        ccr = s.cash_conversion_rate
        scores['cash_flow'] = 95 if ccr >= 1.10 else 88 if ccr >= 1.00 else 78 if ccr >= 0.90 else 65 if ccr >= 0.75 else 45 if ccr >= 0.50 else 25
        scores['cash_flow'] = min(100, scores['cash_flow'] + 5) if s.free_cash_flow > 0 else max(0, scores['cash_flow'] - 15)
        
        scores['earnings_quality'] = max(0, s.earnings_quality_score - min(10, len(s.red_flags) * 2 if s.red_flags else 0))
        
        ebit_chg = s.ebit_change_pct
        scores['profitability'] = 90 if ebit_chg > 0.15 else 78 if ebit_chg > 0.08 else 68 if ebit_chg > 0.02 else 55 if ebit_chg > -0.02 else 40 if ebit_chg > -0.10 else 25
        if s.operating_margin_change > 0.02: scores['profitability'] = min(100, scores['profitability'] + 10)
        elif s.operating_margin_change > 0: scores['profitability'] = min(100, scores['profitability'] + 5)
        
        pe = s.pe_ratio
        tier = "premium" if ccr >= 1.0 and s.roe >= 0.20 else "high" if ccr >= 0.90 and s.roe >= 0.15 else "average" if ccr >= 0.75 and s.roe >= 0.10 else "low"
        pe_ranges = {"premium": (25, 40), "high": (18, 30), "average": (12, 22), "low": (8, 15)}
        pe_lo, pe_hi = pe_ranges.get(tier, (12, 22))
        scores['valuation'] = 50 if pe <= 0 or pe > 200 else 92 if pe < pe_lo * 0.7 else 78 if pe < pe_lo else 62 if pe <= pe_hi else 45 if pe <= pe_hi * 1.3 else 28
        
        rg = s.revenue_growth
        scores['growth'] = 90 if rg > 0.15 else 75 if rg > 0.08 else 62 if rg > 0.03 else 50 if rg > -0.02 else 35 if rg > -0.08 else 20
        
        total = sum(scores[k] * weights[k] for k in weights)
        self._score_breakdown = {'scores': scores.copy(), 'weights': weights.copy(), 'total_score': total}
        
        rec = InvestmentRecommendation.STRONG_BUY if total >= 78 else InvestmentRecommendation.BUY if total >= 65 else InvestmentRecommendation.HOLD if total >= 50 else InvestmentRecommendation.SELL if total >= 38 else InvestmentRecommendation.STRONG_SELL
        dists = [abs(total - t) for t in [38, 50, 65, 78]]
        conf = min(95, total + 20) if min(dists) > 10 else min(85, total + 10) if min(dists) > 5 else min(75, total + 5)
        return rec, max(35, min(95, conf))
    
    def _calculate_target_price(self, s: AnalysisSummary, rec: InvestmentRecommendation) -> Optional[float]:
        if s.pe_ratio <= 0 or s.current_price <= 0: return None
        eps = s.current_price / s.pe_ratio
        mult = 1.15 if s.earnings_quality_score >= 80 else 1.05 if s.earnings_quality_score >= 60 else 0.95
        rec_mult = {InvestmentRecommendation.STRONG_BUY: 1.20, InvestmentRecommendation.BUY: 1.10, InvestmentRecommendation.HOLD: 1.00, InvestmentRecommendation.SELL: 0.90, InvestmentRecommendation.STRONG_SELL: 0.80}
        return round(eps * s.pe_ratio * mult * rec_mult[rec], 2)
    
    def _validate_and_adjust_recommendation(
        self, 
        rec: InvestmentRecommendation, 
        conf: float,
        target: Optional[float],
        current_price: float
    ) -> Tuple[InvestmentRecommendation, float]:
        """
        Validate recommendation against target price and adjust if contradictory.
        
        A BUY/STRONG BUY with negative implied upside is illogical.
        A SELL/STRONG SELL with positive implied upside is illogical.
        """
        if target is None or current_price <= 0:
            return rec, conf
        
        upside = (target / current_price) - 1
        
        # CRITICAL: BUY recommendations MUST have positive upside
        if rec in (InvestmentRecommendation.STRONG_BUY, InvestmentRecommendation.BUY):
            if upside < 0:
                # Downgrade: Negative upside cannot be a BUY
                rec = InvestmentRecommendation.HOLD
                conf = min(conf, 60)  # Lower confidence due to contradiction
            elif upside < 0.10 and rec == InvestmentRecommendation.STRONG_BUY:
                # Downgrade: Less than 10% upside shouldn't be STRONG BUY
                rec = InvestmentRecommendation.BUY
                conf = min(conf, 70)
        
        # SELL recommendations MUST have negative upside (or minimal upside)
        elif rec in (InvestmentRecommendation.STRONG_SELL, InvestmentRecommendation.SELL):
            if upside > 0.10:
                # Upgrade: Significant upside cannot be a SELL
                rec = InvestmentRecommendation.HOLD
                conf = min(conf, 60)
        
        return rec, conf
    
    def _generate_llm_sections(self, s: AnalysisSummary, rec: InvestmentRecommendation) -> Tuple[List[MemoSectionContent], str]:
        import importlib.util
        if importlib.util.find_spec('anthropic') is None:
            raise RuntimeError("anthropic not installed")
        import anthropic
        if not self._client: self._client = anthropic.Anthropic(api_key=self._api_key)
        
        prompt = f'''You are a senior equity analyst at Goldman Sachs writing an institutional investment memorandum.

COMPANY: {s.company_name} ({s.ticker}) | RECOMMENDATION: {rec.value}

FINANCIAL DATA:
- Revenue: ${s.revenue_current:,.0f}M (growth: {s.revenue_growth*100:+.1f}%)
- EBIT: ${s.ebit_current:,.0f}M (change: {s.ebit_change_pct*100:+.1f}%)
- Gross Margin: {s.gross_margin_current*100:.1f}% | Operating Margin: {s.operating_margin_current*100:.1f}%
- Operating CF: ${s.operating_cash_flow:,.0f}M | FCF: ${s.free_cash_flow:,.0f}M
- Cash Conversion: {s.cash_conversion_rate*100:.0f}% | FCF Margin: {s.fcf_margin*100:.1f}%
- ROE: {s.roe*100:.1f}% | ROA: {s.roa*100:.1f}% | ROIC: {s.roic*100:.1f}%
- P/E: {s.pe_ratio:.1f}x | EV/EBITDA: {s.ev_ebitda:.1f}x | Valuation: {s.valuation_assessment}
- Earnings Quality: {s.earnings_quality_score:.0f}/100 ({s.earnings_quality_rating})
- DSO: {s.dso:.0f} | DIO: {s.dio:.0f} | DPO: {s.dpo:.0f} | CCC: {s.ccc:.0f}

IMPORTANT: Write in PLAIN TEXT only. Do NOT use any markdown formatting such as asterisks for bold (**text**), underscores for italic (_text_), or hash symbols for headers (#). The output will be rendered in a PDF document where such formatting will appear as literal characters.

Write comprehensive analysis using this EXACT XML format. Each section should be 2-4 paragraphs with specific numbers.

<memo>
<investment_thesis>2-3 paragraphs explaining the core investment case</investment_thesis>
<executive_summary>3-4 paragraphs covering company overview, financial highlights, and conclusion</executive_summary>
<profitability_analysis>3-4 paragraphs on EBIT, margins, volume/margin effects</profitability_analysis>
<cash_flow_quality>3-4 paragraphs on OCF, FCF, cash conversion, quality</cash_flow_quality>
<earnings_quality>2-3 paragraphs on accruals, quality score, red flags</earnings_quality>
<working_capital>2-3 paragraphs on DSO/DIO/DPO/CCC, liquidity</working_capital>
<valuation>3-4 paragraphs on multiples, quality-adjusted assessment</valuation>
<risks>5 numbered specific risk factors, 2-3 sentences each</risks>
<catalysts>4 numbered potential catalysts, 2-3 sentences each</catalysts>
<conclusion>2 paragraphs summarizing the investment case</conclusion>
</memo>'''
        
        msg = self._client.messages.create(model=self._model, max_tokens=8000, temperature=0.2, messages=[{"role": "user", "content": prompt}])
        response = msg.content[0].text
        return self._parse_xml_response(response, s, rec), response
    
    def _parse_xml_response(self, response: str, s: AnalysisSummary, rec: InvestmentRecommendation) -> List[MemoSectionContent]:
        sections = []
        mapping = [("investment_thesis", MemoSection.INVESTMENT_THESIS, "Investment Thesis"), ("executive_summary", MemoSection.EXECUTIVE_SUMMARY, "Executive Summary"), ("profitability_analysis", MemoSection.PROFITABILITY, "Profitability Analysis"), ("cash_flow_quality", MemoSection.CASH_FLOW, "Cash Flow Quality"), ("earnings_quality", MemoSection.EARNINGS_QUALITY, "Earnings Quality Assessment"), ("working_capital", MemoSection.WORKING_CAPITAL, "Working Capital Efficiency"), ("valuation", MemoSection.VALUATION, "Valuation Analysis"), ("risks", MemoSection.RISKS, "Key Investment Risks"), ("catalysts", MemoSection.CATALYSTS, "Potential Catalysts"), ("conclusion", MemoSection.CONCLUSION, "Investment Conclusion")]
        
        for tag, sec_enum, title in mapping:
            match = re.search(f"<{tag}>(.*?)</{tag}>", response, re.DOTALL | re.IGNORECASE)
            if match:
                # Clean content: remove extra newlines and markdown formatting
                content = re.sub(r'\n{3,}', '\n\n', match.group(1).strip())
                content = self._clean_markdown(content)
                kp = []
                if tag in ['risks', 'catalysts']:
                    kp = [self._clean_markdown(m.strip()) for m in re.findall(r'\d+\.\s*\*?\*?([^:\n]+?)(?:\*?\*?:|\n)', content)][:5]
                sections.append(MemoSectionContent(section=sec_enum, title=title, content=content, key_points=kp))
        
        return sections if len(sections) >= 5 else self._generate_rule_based_sections(s, rec)
    
    def _clean_markdown(self, text: str) -> str:
        """Remove markdown formatting from text for clean PDF rendering."""
        if not text:
            return ""
        # Remove bold+italic (triple asterisks or underscores)
        text = re.sub(r'\*\*\*([^*]+)\*\*\*', r'\1', text)
        text = re.sub(r'___([^_]+)___', r'\1', text)
        # Remove bold (double asterisks or double underscores)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        # Remove italic (single asterisks or underscores around words)
        text = re.sub(r'\*([^*\n]+)\*', r'\1', text)
        text = re.sub(r'(?<!\w)_([^_\n]+)_(?!\w)', r'\1', text)
        # Remove any remaining standalone asterisks
        text = re.sub(r'(?<![*\w])\*(?![*\s])', '', text)
        text = re.sub(r'(?<![*\s])\*(?![*\w])', '', text)
        # Remove markdown headers
        text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
        # Clean up HTML entities
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        # Fix D&A; -> D&A artifacts
        text = re.sub(r'D&A;', 'D&A', text)
        text = re.sub(r'R&D;', 'R&D', text)
        # Clean up multiple spaces
        text = re.sub(r'  +', ' ', text)
        return text.strip()
    
    def _generate_rule_based_sections(self, s: AnalysisSummary, rec: InvestmentRecommendation) -> List[MemoSectionContent]:
        sections = []
        
        # Investment Thesis
        if rec in [InvestmentRecommendation.STRONG_BUY, InvestmentRecommendation.BUY]:
            thesis = f"{s.company_name} represents an attractive investment with {s.cash_conversion_rate*100:.0f}% cash conversion, demonstrating high-quality earnings. EBIT of ${s.ebit_current:,.0f}M grew {s.ebit_change_pct*100:+.1f}% with operating margins of {s.operating_margin_current*100:.1f}%. Returns are exceptional: ROE {s.roe*100:.1f}%, ROIC {s.roic*100:.1f}%. At {s.pe_ratio:.1f}x P/E ({s.valuation_assessment.lower()}), we see an attractive entry point."
        elif rec == InvestmentRecommendation.HOLD:
            thesis = f"{s.company_name} presents balanced risk/reward. Operating margins of {s.operating_margin_current*100:.1f}% and {s.cash_conversion_rate*100:.0f}% cash conversion are solid, but {s.pe_ratio:.1f}x P/E reflects these positives. FCF of ${s.free_cash_flow:,.0f}M supports operations. We recommend holding while monitoring for better entry points."
        else:
            thesis = f"{s.company_name} faces challenges with {s.cash_conversion_rate*100:.0f}% cash conversion raising earnings quality concerns. EBIT {'declined' if s.ebit_change_pct < 0 else 'grew'} {s.ebit_change_pct*100:+.1f}% to ${s.ebit_current:,.0f}M. At {s.pe_ratio:.1f}x P/E, risks are not adequately discounted."
        sections.append(MemoSectionContent(MemoSection.INVESTMENT_THESIS, "Investment Thesis", thesis, [f"Cash conversion: {s.cash_conversion_rate*100:.0f}%", f"EBIT growth: {s.ebit_change_pct*100:+.1f}%", f"Valuation: {s.pe_ratio:.1f}x P/E"]))
        
        # Executive Summary
        exec_sum = f"{s.company_name} ({s.ticker}) operates in {s.sector} with ${s.market_cap/1e9:.1f}B market cap. Revenue of ${s.revenue_current:,.0f}M grew {s.revenue_growth*100:+.1f}%, with gross margins of {s.gross_margin_current*100:.1f}% and operating margins of {s.operating_margin_current*100:.1f}%.\n\nCash flow generation is {'strong' if s.cash_conversion_rate >= 1.0 else 'solid' if s.cash_conversion_rate >= 0.85 else 'adequate'}: OCF ${s.operating_cash_flow:,.0f}M, FCF ${s.free_cash_flow:,.0f}M, {s.cash_conversion_rate*100:.0f}% conversion. Returns: ROE {s.roe*100:.1f}%, ROA {s.roa*100:.1f}%, ROIC {s.roic*100:.1f}%.\n\nWe rate {rec.value} with {self._score_breakdown['total_score']:.0f}/100 composite score. Key strengths: {s.cash_flow_quality.lower()} cash flow, {s.earnings_quality_rating.lower()} earnings quality."
        sections.append(MemoSectionContent(MemoSection.EXECUTIVE_SUMMARY, "Executive Summary", exec_sum, [f"Revenue: ${s.revenue_current:,.0f}M ({s.revenue_growth*100:+.1f}%)", f"FCF: ${s.free_cash_flow:,.0f}M", f"ROE/ROIC: {s.roe*100:.1f}%/{s.roic*100:.1f}%"]))
        
        # Profitability
        prof = f"EBIT {'increased' if s.ebit_change >= 0 else 'decreased'} ${abs(s.ebit_change):,.0f}M ({s.ebit_change_pct*100:+.1f}%) from ${s.ebit_prior:,.0f}M to ${s.ebit_current:,.0f}M. Volume effect: ${s.volume_effect:,.0f}M, margin effect: ${s.margin_effect:,.0f}M. Primary driver: {s.primary_profit_driver.lower()}.\n\nGross margin {'expanded' if s.gross_margin_change >= 0 else 'contracted'} {abs(s.gross_margin_change)*100:.1f}pp to {s.gross_margin_current*100:.1f}%. Operating margin {'improved' if s.operating_margin_change >= 0 else 'declined'} {abs(s.operating_margin_change)*100:.1f}pp to {s.operating_margin_current*100:.1f}%. Net margin: {s.net_margin_current*100:.1f}%."
        sections.append(MemoSectionContent(MemoSection.PROFITABILITY, "Profitability Analysis", prof, [f"EBIT change: ${s.ebit_change:+,.0f}M", f"Driver: {s.primary_profit_driver}", f"Op margin: {s.operating_margin_current*100:.1f}%"]))
        
        # Cash Flow
        cf = f"Net income of ${s.net_income:,.0f}M converted to ${s.operating_cash_flow:,.0f}M OCF ({s.cash_conversion_rate*100:.0f}% rate). This {'excellent' if s.cash_conversion_rate >= 1.0 else 'solid' if s.cash_conversion_rate >= 0.85 else 'adequate'} conversion indicates {'cash exceeds earnings' if s.cash_conversion_rate >= 1.0 else 'earnings largely cash-backed'}.\n\nFCF of ${s.free_cash_flow:,.0f}M ({s.fcf_margin*100:.1f}% margin) after ${s.capex:,.0f}M CapEx ({s.capex_to_revenue*100:.1f}% of revenue). CapEx/D&A of {s.capex_to_da:.2f}x suggests {'growth' if s.capex_to_da > 1.2 else 'maintenance'}-level investment. Quality: {s.cash_flow_quality}."
        sections.append(MemoSectionContent(MemoSection.CASH_FLOW, "Cash Flow Quality", cf, [f"Cash conversion: {s.cash_conversion_rate*100:.0f}%", f"FCF margin: {s.fcf_margin*100:.1f}%", f"Quality: {s.cash_flow_quality}"]))
        
        # Earnings Quality
        eq = f"Earnings quality rated {s.earnings_quality_rating} ({s.earnings_quality_score:.0f}/100). Accruals ratio of {s.accruals_ratio*100:.1f}% {'is within bounds' if abs(s.accruals_ratio) < 0.10 else 'is elevated'}.\n\n{f'Red flags: {", ".join(s.red_flags)}.' if s.red_flags else 'No significant red flags identified.'} Combined with {s.cash_conversion_rate*100:.0f}% cash conversion, earnings provide a {'reliable' if s.earnings_quality_score >= 75 else 'reasonable'} valuation basis."
        sections.append(MemoSectionContent(MemoSection.EARNINGS_QUALITY, "Earnings Quality Assessment", eq, [f"Score: {s.earnings_quality_score:.0f}/100", f"Accruals: {s.accruals_ratio*100:.1f}%", f"Red flags: {len(s.red_flags) if s.red_flags else 0}"]))
        
        # Working Capital
        wc = f"DSO: {s.dso:.0f} days, DIO: {s.dio:.0f} days, DPO: {s.dpo:.0f} days. Cash conversion cycle: {s.ccc:.0f} days ({'negative CCC finances operations via suppliers' if s.ccc < 0 else 'efficient' if s.ccc < 30 else 'typical'}).\n\nCCC {'improved' if s.ccc_change < 0 else 'deteriorated'} {abs(s.ccc_change):.0f} days YoY. Current ratio: {s.current_ratio:.2f}x, quick ratio: {s.quick_ratio:.2f}x. Net WC: ${s.net_working_capital:,.0f}M."
        sections.append(MemoSectionContent(MemoSection.WORKING_CAPITAL, "Working Capital Efficiency", wc, [f"CCC: {s.ccc:.0f} days", f"Current ratio: {s.current_ratio:.2f}x", f"DSO/DIO/DPO: {s.dso:.0f}/{s.dio:.0f}/{s.dpo:.0f}"]))
        
        # Valuation
        tier = "premium" if s.cash_conversion_rate >= 1.0 and s.roe >= 0.20 else "high" if s.cash_conversion_rate >= 0.90 else "average"
        val = f"Trading at {s.pe_ratio:.1f}x P/E, {s.ev_ebitda:.1f}x EV/EBITDA, {s.price_to_fcf:.1f}x P/FCF. Additional: P/B {s.price_to_book:.2f}x, P/S {s.price_to_sales:.2f}x.\n\nGiven {tier} quality tier ({s.cash_conversion_rate*100:.0f}% CCR, {s.roe*100:.1f}% ROE), valuation is {s.valuation_assessment.lower()}. Implied growth of {s.implied_growth*100:.1f}% {'exceeds' if s.implied_growth > s.revenue_growth * 1.5 else 'is consistent with'} historical {s.revenue_growth*100:.1f}% revenue growth."
        sections.append(MemoSectionContent(MemoSection.VALUATION, "Valuation Analysis", val, [f"P/E: {s.pe_ratio:.1f}x", f"EV/EBITDA: {s.ev_ebitda:.1f}x", f"Assessment: {s.valuation_assessment}"]))
        
        # Risks
        risks = []
        if s.cash_conversion_rate < 0.80: risks.append(f"Cash Conversion: {s.cash_conversion_rate*100:.0f}% rate suggests some earnings may not be sustainable.")
        if s.debt_to_equity > 1.5: risks.append(f"Leverage: {s.debt_to_equity:.2f}x D/E increases financial risk.")
        if s.ebit_change_pct < -0.05: risks.append(f"Profitability: EBIT declined {abs(s.ebit_change_pct)*100:.1f}%.")
        if s.pe_ratio > 35: risks.append(f"Valuation: {s.pe_ratio:.1f}x P/E prices in high growth expectations.")
        risks.extend(["Market: Broader volatility could impact valuation.", "Execution: Strategic initiative success is uncertain.", "Competition: Industry pressures could affect share.", "Macro: Economic slowdown could impact demand.", "Regulatory: Changes could affect operations."])
        risk_content = "\n\n".join(f"{i+1}. {r}" for i, r in enumerate(risks[:5]))
        sections.append(MemoSectionContent(MemoSection.RISKS, "Key Investment Risks", risk_content, [r.split(':')[0] for r in risks[:5]]))
        
        # Catalysts
        catalysts = [f"Margin Expansion: Cost discipline could drive margins above {s.operating_margin_current*100:.1f}%.", f"Capital Return: ${s.free_cash_flow:,.0f}M FCF supports dividends/buybacks.", f"Growth: New products could accelerate beyond {s.revenue_growth*100:.1f}% growth.", "Multiple Expansion: Quality recognition could drive re-rating."]
        cat_content = "\n\n".join(f"{i+1}. {c}" for i, c in enumerate(catalysts))
        sections.append(MemoSectionContent(MemoSection.CATALYSTS, "Potential Catalysts", cat_content, [c.split(':')[0] for c in catalysts]))
        
        # Conclusion
        target = self._calculate_target_price(s, rec)
        conclusion = f"We rate {s.company_name} ({s.ticker}) {rec.value}. Investment supported by {s.cash_flow_quality.lower()} cash flow ({s.cash_conversion_rate*100:.0f}% conversion), {s.earnings_quality_rating.lower()} earnings quality ({s.earnings_quality_score:.0f}/100), and returns of {s.roe*100:.1f}% ROE / {s.roic*100:.1f}% ROIC.\n\nValuation of {s.pe_ratio:.1f}x P/E and {s.ev_ebitda:.1f}x EV/EBITDA is {s.valuation_assessment.lower()}. Price target: ${target:.2f}." if target else f"We rate {s.company_name} ({s.ticker}) {rec.value}. Investment supported by {s.cash_flow_quality.lower()} cash flow ({s.cash_conversion_rate*100:.0f}% conversion), {s.earnings_quality_rating.lower()} earnings quality ({s.earnings_quality_score:.0f}/100), and returns of {s.roe*100:.1f}% ROE / {s.roic*100:.1f}% ROIC.\n\nValuation of {s.pe_ratio:.1f}x P/E and {s.ev_ebitda:.1f}x EV/EBITDA is {s.valuation_assessment.lower()}."
        sections.append(MemoSectionContent(MemoSection.CONCLUSION, "Investment Conclusion", conclusion, [f"Recommendation: {rec.value}", f"Cash flow: {s.cash_flow_quality}", f"Valuation: {s.valuation_assessment}"]))
        
        return sections


def generate_memo(summary: AnalysisSummary, api_key: Optional[str] = None) -> InvestmentMemo:
    return MemoGenerator(api_key=api_key).generate(summary)