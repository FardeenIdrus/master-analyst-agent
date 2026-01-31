#!/usr/bin/env python3
"""
Master Analyst Agent - Demo Script
===================================

This script demonstrates the full AI Analyst Agent pipeline:
1. Executes 6 specialized analyst agents in parallel (2 Technical + 4 Fundamental)
2. Collects and processes their JSON outputs
3. Synthesizes recommendations using an LLM (Claude/GPT-4)
4. Generates a professional PDF investment memo with charts

Usage:
    python run_demo.py                  # Analyze AAPL (default)
    python run_demo.py MSFT             # Analyze Microsoft
    python run_demo.py TSLA --verbose   # Analyze Tesla with verbose output

Requirements:
    - Python 3.10+
    - Dependencies: pip install -r requirements.txt
    - API Keys in .env file (OPENAI_API_KEY and/or ANTHROPIC_API_KEY)

Outputs:
    - shared_outputs/FINAL_RECOMMENDATION.json  (structured analysis)
    - shared_outputs/FINAL_RECOMMENDATION.pdf   (professional report)
    - shared_outputs/analyst_targets_chart.png  (target comparison)
    - shared_outputs/price_action_chart.png     (6-month price chart)

Module: IFTE0001 - Introduction to Financial Markets
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 70)
    print("  MASTER ANALYST AGENT - AI-Powered Investment Analysis")
    print("  UCL MSc Financial Technology - IFTE0001")
    print("=" * 70)


def print_architecture():
    """Print system architecture overview."""
    print("""
    SYSTEM ARCHITECTURE
    -------------------

    ┌─────────────────────────────────────────────────────────────────┐
    │                      MASTER AGENT                               │
    │  (Orchestration, Consensus Building, LLM Synthesis)             │
    └─────────────────────────────────────────────────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
    ┌─────────▼─────────┐                   ┌──────────▼──────────┐
    │  TECHNICAL (50%)  │                   │  FUNDAMENTAL (50%)  │
    └───────────────────┘                   └─────────────────────┘
         │       │                            │    │    │    │
    ┌────▼──┐ ┌──▼───┐              ┌────────▼┐ ┌─▼──┐ ┌▼───┐ ┌▼──────┐
    │Fardeen│ │Tamer │              │ Daria   │ │Shak│ │Lary│ │Mohamed│
    │ (20%) │ │(20%) │              │ (15%)   │ │(15)│ │(15)│ │ (15%) │
    └───────┘ └──────┘              └─────────┘ └────┘ └────┘ └───────┘

    Technical Agents:
      - RSI, MACD, ADX, Bollinger Bands
      - Market regime detection (Hurst exponent)
      - Backtesting with Monte Carlo simulation
      - Position sizing (Kelly criterion)

    Fundamental Agents:
      - Financial ratio analysis (profitability, leverage, growth)
      - DCF valuation models
      - Peer comparison and multiples analysis
      - Earnings quality assessment
    """)


def check_environment():
    """Check that required environment variables and dependencies are set."""
    print("\nChecking environment...")

    # Check .env file
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        print("  [WARNING] .env file not found. API calls may fail.")
        print("            Create .env with OPENAI_API_KEY or ANTHROPIC_API_KEY")
    else:
        print("  [OK] .env file found")

    # Check for API keys
    from dotenv import load_dotenv
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if anthropic_key:
        print("  [OK] ANTHROPIC_API_KEY configured (primary)")
    if openai_key:
        print("  [OK] OPENAI_API_KEY configured" + (" (fallback)" if anthropic_key else " (primary)"))

    if not openai_key and not anthropic_key:
        print("  [ERROR] No API keys found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env")
        return False

    # Check shared_outputs directory
    outputs_dir = Path(__file__).parent / "shared_outputs"
    outputs_dir.mkdir(exist_ok=True)
    print(f"  [OK] Output directory: {outputs_dir}")

    return True


async def run_full_pipeline(ticker: str, verbose: bool = False):
    """Run the complete master agent pipeline with all 6 agents."""
    from master_agent import main

    # Override ticker via sys.argv (master_agent reads from sys.argv[1])
    original_argv = sys.argv.copy()
    sys.argv = ["run_demo.py", ticker]

    try:
        await main()
    finally:
        sys.argv = original_argv

    # Print output locations
    outputs_dir = Path(__file__).parent / "shared_outputs"
    print("\n" + "=" * 70)
    print("OUTPUT FILES GENERATED")
    print("=" * 70)

    output_files = [
        ("Final JSON", "FINAL_RECOMMENDATION.json"),
        ("Final PDF", "FINAL_RECOMMENDATION.pdf"),
        ("Targets Chart", "analyst_targets_chart.png"),
        ("Price Chart", "price_action_chart.png"),
    ]

    for name, filename in output_files:
        filepath = outputs_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"  {name}: {filepath} ({size_kb:.1f} KB)")
        else:
            print(f"  {name}: [not generated]")


def main():
    """Main entry point for demo script."""
    parser = argparse.ArgumentParser(
        description="Master Analyst Agent - AI-Powered Investment Analysis Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_demo.py              # Analyze AAPL (default)
  python run_demo.py MSFT         # Analyze Microsoft
  python run_demo.py TSLA -v      # Analyze Tesla with verbose output
  python run_demo.py --info       # Show system architecture
  python run_demo.py --check      # Check environment setup
        """
    )

    parser.add_argument(
        "ticker",
        nargs="?",
        default="AAPL",
        help="Stock ticker symbol to analyze (default: AAPL)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show system architecture and exit"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check environment setup and exit"
    )

    args = parser.parse_args()

    print_banner()

    if args.info:
        print_architecture()
        return 0

    if args.check:
        if check_environment():
            print("\n[OK] Environment check passed. Ready to run.")
            return 0
        else:
            print("\n[FAIL] Environment check failed. Fix issues above.")
            return 1

    # Check environment before running
    if not check_environment():
        print("\nRun with --check for more details on environment setup.")
        return 1

    ticker = args.ticker.upper()
    print(f"\nTarget: {ticker}")

    # Run full pipeline (all 6 agents)
    print_architecture()
    print("\nStarting full analysis pipeline...")
    print("This will run all 6 agents in parallel and synthesize results.\n")

    asyncio.run(run_full_pipeline(ticker, args.verbose))

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
