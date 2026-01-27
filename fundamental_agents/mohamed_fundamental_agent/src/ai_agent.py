import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from fundamental_analyzer import FundamentalAnalyzer
from report_generator import create_analysis_data_structure
from Data_collector import collect_data

# Load environment variables from master repo root
# Path: src -> mohamed_fundamental_agent -> fundamental_agents -> master-analyst-agent
REPO_ROOT = Path(__file__).parent.parent.parent.parent
load_dotenv(REPO_ROOT / ".env")

# Shared outputs directory
SHARED_OUTPUTS = REPO_ROOT / "shared_outputs"
SHARED_OUTPUTS.mkdir(exist_ok=True)

# Get ticker from command line
TICKER = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"


class InvestmentAIAgent:
    def __init__(self):
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with OpenAI fallback."""
        response = None

        # Try Anthropic first
        if self.anthropic_key:
            try:
                from anthropic import Anthropic
                client = Anthropic(api_key=self.anthropic_key)
                message = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                response = message.content[0].text
                print("  Using Anthropic API")
            except Exception as e:
                print(f"  Anthropic failed: {e}")

        # Fallback to OpenAI
        if response is None and self.openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_key)
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                response = completion.choices[0].message.content
                print("  Using OpenAI API")
            except Exception as e:
                print(f"  OpenAI failed: {e}")

        if response is None:
            raise ValueError("No LLM API available or all failed")

        return response

    def generate_investment_report(self, ticker: str):
        """Generate a comprehensive AI-powered investment report."""

        print(f"Analyzing {ticker}...")

        # Step 1: Collect data from Alpha Vantage (or use cache)
        print("Step 1: Collecting financial data...")
        data_path = collect_data(ticker)

        # Step 2: Run fundamental analysis
        print("Step 2: Running fundamental analysis...")
        analyzer = FundamentalAnalyzer(ticker, data_path=str(data_path))

        # Get DCF results
        dcf_result = analyzer.dcf_valuation()
        sensitivity = analyzer.dcf_terminal_growth_sensitivity()

        # Get multiples results
        multiples = analyzer.multiples_valuation()

        # Calculate blended recommendation
        blended_price = dcf_result['intrinsic_price'] * 0.6 + multiples['average_implied_price'] * 0.4
        current_price = dcf_result['current_price']
        blended_upside = ((blended_price - current_price) / current_price) * 100

        # Prepare data for LLM
        analysis_data = f"""
Stock: {ticker}
Current Price: ${current_price:.2f}

DCF VALUATION:
- Intrinsic Price: ${dcf_result['intrinsic_price']:.2f}
- Forecast Growth (CAGR): {dcf_result['forecast_growth']:.2%}
- WACC: {dcf_result['wacc']:.2%}
- Terminal Growth: {dcf_result['terminal_growth']:.2%}
- Base FCFF: ${dcf_result['fcff_base']:,.0f}
- Enterprise Value: ${dcf_result['enterprise_value']:,.0f}
- Upside: {((dcf_result['intrinsic_price'] - current_price) / current_price * 100):+.2f}%

MULTIPLES VALUATION:
Current Multiples:
- P/E: {multiples['current_multiples']['P/E']:.2f}
- P/B: {multiples['current_multiples']['P/B']:.2f}
- P/S: {multiples['current_multiples']['P/S']:.2f}
- EV/EBITDA: {multiples['current_multiples']['EV/EBITDA']:.2f}

Peer Median Multiples:
- P/E: {multiples['peer_median_multiples']['P/E']:.2f}
- P/B: {multiples['peer_median_multiples']['P/B']:.2f}
- P/S: {multiples['peer_median_multiples']['P/S']:.2f}
- EV/EBITDA: {multiples['peer_median_multiples']['EV/EBITDA']:.2f}

Implied Prices:
- P/E: ${multiples['implied_prices']['P/E']:.2f}
- P/B: ${multiples['implied_prices']['P/B']:.2f}
- P/S: ${multiples['implied_prices']['P/S']:.2f}
- EV/EBITDA: ${multiples['implied_prices']['EV/EBITDA']:.2f}
- Average: ${multiples['average_implied_price']:.2f}
- Upside: {((multiples['average_implied_price'] - current_price) / current_price * 100):+.2f}%

BLENDED RECOMMENDATION:
- Target Price: ${blended_price:.2f} (60% DCF, 40% Multiples)
- Upside: {blended_upside:+.2f}%
"""

        # Create prompt for LLM
        prompt = f"""You are a professional equity research analyst. Based on the following fundamental analysis, write a comprehensive investment report for {ticker}.

{analysis_data}

Please provide:
1. **Executive Summary** - One paragraph overview with clear BUY/HOLD/SELL recommendation
2. **Valuation Analysis** - Discuss the DCF and multiples approaches, highlighting key insights, concisely describe the rationale behind any outliers.
3. **Key Risks** - What could make this analysis wrong?
4. **Investment Thesis** - The core argument for or against investing
5. **Price Target** - Your recommended target price with reasoning

Be specific, professional, and data-driven. Use the numbers provided to support your analysis."""

        # Call LLM API
        print("Generating AI report...")
        report = self._call_llm(prompt)

        # Create analysis data structure
        analysis_data = create_analysis_data_structure(
            ticker=ticker,
            dcf_result=dcf_result,
            multiples=multiples,
            blended_target=blended_price,
            blended_upside=blended_upside,
            sensitivity=sensitivity,
            ai_report=report,
        )

        # Save JSON to shared_outputs (no PDF)
        output_path = SHARED_OUTPUTS / f"fundamental_mohamed_{ticker}.json"
        with open(output_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        print(f"✓ JSON saved: {output_path}")

        return analysis_data


if __name__ == "__main__":
    agent = InvestmentAIAgent()

    print("="*70)
    print(f"MOHAMED'S AI-POWERED INVESTMENT ANALYSIS - {TICKER}")
    print("="*70)

    result = agent.generate_investment_report(TICKER)

    print("\n" + "="*70)
    print("QUANTITATIVE SUMMARY")
    print("="*70)
    print(f"Current Price:        ${result['current_price']:.2f}")
    print(f"DCF Target:           ${result['dcf_target']:.2f}")
    print(f"Multiples Target:     ${result['multiples_target']:.2f}")
    print(f"Blended Target:       ${result['blended_target']:.2f}")
    print(f"Upside Potential:     {result['upside_potential']:+.2f}%")
    print(f"Recommendation:       {result['recommendation']}")
