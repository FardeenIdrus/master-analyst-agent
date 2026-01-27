"""
Master Agent - Orchestrates parallel execution of analyst agents and synthesizes their outputs.

Pipeline:
1. Run all registered analyst agents concurrently as subprocesses
2. Collect JSON outputs from shared_outputs/ directory
3. Send outputs to LLM for synthesis into a unified recommendation
4. Save and display the final investment recommendation
"""

import asyncio
import subprocess
import json
from dotenv import load_dotenv
load_dotenv()
import os
from pathlib import Path
from datetime import datetime
import sys

# Ticker from command line or default 
TICKER = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"

# === CONFIGURATION ===

REPO_ROOT = Path(__file__).parent.parent
SHARED_OUTPUTS = REPO_ROOT / "shared_outputs"  # Directory where agents write their JSON results

# Agent registry: (display_name, entry_script_path)
# Each agent runs independently and writes output to SHARED_OUTPUTS
AGENTS = [
    ("technical_fardeen", REPO_ROOT / "technical_agents" / "fardeen_technical_agent" / "src" / "llm_agent.py"),
    ("fundamental_daria", REPO_ROOT / "fundamental_agents" / "daria_fundamental_agent" / "run_demo.py"),
    ("fundamental_shakzod", REPO_ROOT / "fundamental_agents" / "shakzod_fundamental_agent" / "run_demo.py"),
    ("fundamental_lary", REPO_ROOT / "fundamental_agents" / "lary_fundamental_agent" / "agents.py"),
]


# === AGENT EXECUTION ===

async def run_agent(name: str, script_path: Path) -> dict:
    """
    Execute a single agent as an async subprocess.
    """
    print(f"[{name}] Starting...")
    start = datetime.now()

    try:
        # Pass ticker to all agents
        cmd = ["python", str(script_path), TICKER]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=script_path.parent
        )

        stdout, stderr = await process.communicate()
        duration = (datetime.now() - start).total_seconds()

        if process.returncode == 0:
            print(f"[{name}] ✓ Completed in {duration:.1f}s")
            return {"agent": name, "status": "success", "duration": duration}
        else:
            print(f"[{name}] ✗ Failed: {stderr.decode()[:200]}")
            return {"agent": name, "status": "failed", "error": stderr.decode()[:500]}

    except Exception as e:
        print(f"[{name}] ✗ Error: {e}")
        return {"agent": name, "status": "error", "error": str(e)}


async def run_all_agents():
    """
    Run all registered agents concurrently using asyncio.gather.
    Only runs agents whose script paths exist on disk.

    Returns:
        List of result dicts from each agent
    """
    print("=" * 60)
    print("MASTER AGENT - Parallel Execution")
    print("=" * 60)

    SHARED_OUTPUTS.mkdir(exist_ok=True)

    # Filter to existing paths and run concurrently
    tasks = [run_agent(name, path) for name, path in AGENTS if path.exists()]
    results = await asyncio.gather(*tasks)

    # Print summary
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    for r in results:
        status = "✓" if r["status"] == "success" else "✗"
        print(f"  {status} {r['agent']}: {r['status']}")

    return results


# === OUTPUT COLLECTION ===

def load_all_outputs() -> list[dict]:
    """
    Load all JSON files from shared_outputs/ directory.
    Adds '_source_file' field to each output for traceability.

    Returns:
        List of parsed JSON dicts from each agent's output file
    """
    outputs = []
    for json_file in SHARED_OUTPUTS.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                data["_source_file"] = json_file.name  # Track origin
                outputs.append(data)
                print(f"  Loaded: {json_file.name}")
        except Exception as e:
            print(f"  Failed to load {json_file.name}: {e}")
    return outputs


# === LLM SYNTHESIS ===

def synthesize_with_llm(outputs: list[dict]) -> dict:
    """
    Send all agent outputs to GPT-4o for synthesis into a unified recommendation.

    Args:
        outputs: List of JSON dicts from all agents

    Returns:
        Synthesized recommendation dict with fields: ticker, recommendation,
        target_price, confidence, technical_summary, fundamental_summary,
        key_risks, investment_thesis
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Construct prompt with all analyst outputs (truncated to 15k chars for token limits)
    prompt = f"""You are a senior portfolio manager at a top-tier investment firm synthesizing research from multiple analysts into a comprehensive investment memo.

You have received {len(outputs)} analyst reports for {TICKER}:

{json.dumps(outputs, indent=2, default=str)[:15000]}

Create a detailed, institutional-grade investment recommendation memo. The investment_thesis should be 2-3 pages long (approximately 1500-2500 words) and include:

1. EXECUTIVE SUMMARY (1 paragraph)
   - Clear recommendation with conviction level
   - Target price and expected return
   - Investment timeframe

2. COMPANY OVERVIEW (1 paragraph)
   - Business description and competitive positioning
   - Key products/services and revenue drivers
   - Market position and competitive advantages

3. TECHNICAL ANALYSIS SUMMARY (2-3 paragraphs)
   - Current price action and trend analysis
   - Key support/resistance levels
   - Momentum indicators and what they signal
   - Short-term vs long-term technical outlook

4. FUNDAMENTAL ANALYSIS SUMMARY (3-4 paragraphs)
   - Financial health assessment (profitability, margins, growth)
   - Valuation analysis (P/E, DCF, comparables)
   - Balance sheet strength and cash flow quality
   - Earnings quality and sustainability

5. INVESTMENT THESIS (3-4 paragraphs)
   - Core reasons to buy/sell/hold
   - Key catalysts that could drive the stock
   - Competitive moat analysis
   - Management quality assessment

6. RISK ASSESSMENT (2-3 paragraphs)
   - Key risks to the investment thesis
   - Downside scenarios and price targets
   - Mitigating factors

7. CONCLUSION & RECOMMENDATION
   - Final recommendation with price target
   - Suggested position sizing
   - Key metrics to monitor

Output as JSON with the following structure:
{{
    "ticker": "{TICKER}",
    "company_name": "<company name>",
    "analysis_date": "{datetime.now().strftime('%Y-%m-%d')}",
    "recommendation": "STRONG BUY/BUY/HOLD/SELL/STRONG SELL",
    "target_price": float,
    "current_price": float,
    "expected_return_pct": float,
    "confidence": float (0-100),
    "investment_horizon": "short-term/medium-term/long-term",
    "technical_summary": "Detailed technical analysis summary (2-3 paragraphs)",
    "fundamental_summary": "Detailed fundamental analysis summary (3-4 paragraphs)",
    "key_catalysts": ["catalyst1", "catalyst2", "catalyst3"],
    "key_risks": ["risk1", "risk2", "risk3", "risk4", "risk5"],
    "investment_thesis": "FULL 2-3 PAGE INVESTMENT THESIS WITH ALL SECTIONS ABOVE"
}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8000,
        temperature=0.2  # Low temperature for consistent, factual output
    )

    # Extract JSON from response (handles cases where LLM adds extra text)
    text = response.choices[0].message.content
    import re
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        return json.loads(match.group())
    return {"raw_response": text}  # Fallback if JSON extraction fails


def save_final_report(synthesis: dict):
    """Save synthesized report to FINAL_RECOMMENDATION.json and PDF."""
    # Save JSON
    json_path = SHARED_OUTPUTS / "FINAL_RECOMMENDATION.json"
    with open(json_path, 'w') as f:
        json.dump(synthesis, f, indent=2)
    print(f"\nFinal report saved: {json_path}")

    # Generate PDF
    pdf_path = generate_pdf_report(synthesis)
    if pdf_path:
        print(f"PDF report saved: {pdf_path}")

    return json_path


def generate_pdf_report(synthesis: dict) -> Path:
    """Generate a professional PDF report from the synthesis."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    except ImportError:
        print("  Warning: reportlab not installed, skipping PDF generation")
        return None

    pdf_path = SHARED_OUTPUTS / "FINAL_RECOMMENDATION.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter,
                           rightMargin=0.75*inch, leftMargin=0.75*inch,
                           topMargin=0.75*inch, bottomMargin=0.75*inch)

    # Colors
    NAVY = HexColor('#0f172a')
    BLUE = HexColor('#1e40af')
    GREEN = HexColor('#059669')
    RED = HexColor('#dc2626')
    GRAY = HexColor('#64748b')

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=24,
                                  textColor=NAVY, spaceAfter=20, alignment=TA_CENTER)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14,
                                    textColor=BLUE, spaceBefore=20, spaceAfter=10)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10,
                                 leading=14, alignment=TA_JUSTIFY, spaceAfter=8)

    # Determine recommendation color
    rec = synthesis.get('recommendation', 'HOLD').upper()
    if 'BUY' in rec:
        rec_color = GREEN
    elif 'SELL' in rec:
        rec_color = RED
    else:
        rec_color = GRAY

    rec_style = ParagraphStyle('Rec', parent=styles['Heading1'], fontSize=18,
                                textColor=rec_color, alignment=TA_CENTER)

    story = []

    # Title
    ticker = synthesis.get('ticker', TICKER)
    company = synthesis.get('company_name', 'Apple Inc.')
    story.append(Paragraph(f"{ticker} Investment Memo", title_style))
    story.append(Paragraph(f"{company}", ParagraphStyle('Sub', fontSize=14, textColor=GRAY, alignment=TA_CENTER)))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"Analysis Date: {synthesis.get('analysis_date', datetime.now().strftime('%Y-%m-%d'))}",
                          ParagraphStyle('Date', fontSize=10, textColor=GRAY, alignment=TA_CENTER)))
    story.append(Spacer(1, 20))

    # Recommendation Box
    story.append(Paragraph(f"Recommendation: {rec}", rec_style))
    story.append(Spacer(1, 15))

    # Key Metrics Table
    target = synthesis.get('target_price', 0)
    current = synthesis.get('current_price', 0)
    confidence = synthesis.get('confidence', 0)
    expected_return = synthesis.get('expected_return_pct', 0)
    horizon = synthesis.get('investment_horizon', 'medium-term')

    metrics_data = [
        ['Target Price', f'${target:.2f}' if target else 'N/A'],
        ['Current Price', f'${current:.2f}' if current else 'N/A'],
        ['Expected Return', f'{expected_return:.1f}%' if expected_return else 'N/A'],
        ['Confidence', f'{confidence:.0f}%' if confidence else 'N/A'],
        ['Investment Horizon', horizon.title()],
    ]

    metrics_table = Table(metrics_data, colWidths=[2*inch, 2*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), HexColor('#f1f5f9')),
        ('TEXTCOLOR', (0, 0), (-1, -1), NAVY),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cbd5e1')),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 20))

    # Technical Summary
    story.append(Paragraph("Technical Analysis Summary", heading_style))
    tech_summary = synthesis.get('technical_summary', 'N/A')
    for para in tech_summary.split('\n\n'):
        if para.strip():
            story.append(Paragraph(para.strip(), body_style))

    # Fundamental Summary
    story.append(Paragraph("Fundamental Analysis Summary", heading_style))
    fund_summary = synthesis.get('fundamental_summary', 'N/A')
    for para in fund_summary.split('\n\n'):
        if para.strip():
            story.append(Paragraph(para.strip(), body_style))

    # Key Catalysts
    catalysts = synthesis.get('key_catalysts', [])
    if catalysts:
        story.append(Paragraph("Key Catalysts", heading_style))
        for catalyst in catalysts:
            story.append(Paragraph(f"• {catalyst}", body_style))

    # Key Risks
    risks = synthesis.get('key_risks', [])
    if risks:
        story.append(Paragraph("Key Risks", heading_style))
        for risk in risks:
            story.append(Paragraph(f"• {risk}", body_style))

    # Investment Thesis (main content)
    story.append(PageBreak())
    story.append(Paragraph("Investment Thesis", title_style))
    thesis = synthesis.get('investment_thesis', 'N/A')
    for para in thesis.split('\n\n'):
        if para.strip():
            # Check if it's a section header (all caps or starts with number)
            if para.strip().isupper() or (para.strip()[0].isdigit() and '.' in para.strip()[:3]):
                story.append(Paragraph(para.strip(), heading_style))
            else:
                story.append(Paragraph(para.strip(), body_style))

    # Build PDF
    doc.build(story)
    return pdf_path


# === MAIN PIPELINE ===

async def main():
    """
    Main pipeline: run agents -> collect outputs -> synthesize -> save & display.
    """
    # Step 1: Execute all agents in parallel
    print(f"\nAnalyzing: {TICKER}\n")  # Add this line
    
    await run_all_agents()

    # Step 2: Collect outputs from shared directory
    print("\n" + "=" * 60)
    print("LOADING AGENT OUTPUTS")
    print("=" * 60)
    outputs = load_all_outputs()

    if not outputs:
        print("No outputs found! Check if agents ran correctly.")
        return

    # Step 3: Synthesize all reports via LLM
    print("\n" + "=" * 60)
    print("SYNTHESIZING WITH LLM")
    print("=" * 60)
    synthesis = synthesize_with_llm(outputs)

    # Step 4: Persist final report
    save_final_report(synthesis)

    # Step 5: Display summary
    print("\n" + "=" * 60)
    print("FINAL RECOMMENDATION")
    print("=" * 60)
    print(f"  Ticker: {synthesis.get('ticker', TICKER)}")
    print(f"  Recommendation: {synthesis.get('recommendation', 'N/A')}")
    print(f"  Target Price: ${synthesis.get('target_price', 0):.2f}")
    print(f"  Confidence: {synthesis.get('confidence', 0):.0f}%")
    print(f"\n  Thesis: {synthesis.get('investment_thesis', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())
