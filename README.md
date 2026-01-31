# Multi-Agent Stock Analysis System

**UCL MSc DIGITAL FINANCE AND BANKING | IFTE0001: AI Analyst Agents in Asset Management**

A multi-agent system that combines technical and fundamental analysis to generate institutional-quality investment recommendations.

## Overview

This system orchestrates 6 specialized AI agents (2 technical + 4 fundamental) through a master agent that aggregates their signals using weighted voting to produce a final investment recommendation.

```
                           MASTER AGENT
                               |
              +----------------+----------------+
              |                                 |
      TECHNICAL (50%)                   FUNDAMENTAL (50%)
           |    |                        |    |    |    |
        +--+    +--+                  +--+ +--+ +--+ +--+
        |          |                  |    |    |    |
     Fardeen    Tamer              Daria Shak Lary Mohamed
      (20%)    (20%)               (15%) (15%) (15%) (15%)
```

## Features

**Technical Analysis (Track B)**
- Momentum Indicators: RSI, MACD, Stochastic, Williams %R, Rate of Change
- Trend Analysis: SMA 50/200 crossovers, ADX/DMI system, Aroon, Supertrend, Ichimoku
- Volatility Systems: Bollinger Bands, Keltner Channels, Donchian Channels, ATR
- Volume Analysis: OBV, Chaikin Money Flow, Money Flow Index, VWMA
- Market Regime Detection: Hurst exponent classification (trending vs mean-reverting)
- Backtesting Engine: 10-year historical testing with 500 Monte Carlo simulations
- Risk Metrics: Sharpe, Sortino, Calmar ratios, VaR, CVaR, max drawdown
- Position Sizing: Fractional Kelly criterion with GARCH volatility forecasting

**Fundamental Analysis (Track A)**
- DCF Valuation: 5-year FCFF projections, WACC calculation (CAPM), Gordon Growth terminal value
- Multiples Valuation: P/E, P/B, P/S, EV/EBITDA with peer median comparison
- Dividend Discount Model: For income-generating stocks
- DuPont Analysis: ROE decomposition into margin, turnover, and leverage
- Financial Ratios: Profitability, liquidity, leverage, efficiency, growth metrics
- Earnings Quality: Accruals ratio, cash conversion analysis
- Scenario Analysis: Bull/Base/Bear cases with probability-weighted targets
- Sensitivity Analysis: Terminal growth rate and WACC variations

**Master Agent Orchestration**
- Weighted Signal Aggregation: Deterministic recommendation via plurality voting
- Weight Distribution: Technical 50% (2x20%), Fundamental 50% (4x12.5%)
- Dynamic Weight Renormalization: Adjusts when agents fail or timeout
- Conflict Detection: Identifies and explains disagreements between analysts
- LLM Synthesis: Claude/GPT generates institutional-quality investment memos
- Binding Recommendations: Pre-calculated signals override LLM suggestions
- PDF Report Generation: Professional investment committee briefing documents

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd master-analyst-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the project root with your API keys:

```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Alpha Vantage keys (one per fundamental agent to avoid rate limits)
ALPHA_VANTAGE_KEY_LARY=your_key
ALPHA_VANTAGE_KEY_SHAKZOD=your_key
ALPHA_VANTAGE_KEY_DARIA=your_key
ALPHA_VANTAGE_KEY_MOHAMED=your_key
```

Note: Alpha Vantage free tier allows 5 API calls per minute. Each agent uses a separate key to parallelize data collection.

## Usage

Run the full multi-agent analysis:

```bash
python run_demo.py AAPL
```

Or run the master agent directly:

```bash
python src/master_agent.py AAPL
```

## Project Structure

```
master-analyst-agent/
├── src/
│   └── master_agent.py          # Main orchestration logic
├── technical_agents/
│   ├── fardeen_technical_agent/ # Technical analysis with backtesting
│   └── tamer_technical_agent/   # Technical indicators
├── fundamental_agents/
│   ├── daria_fundamental_agent/ # Fundamental analysis
│   ├── shakzod_fundamental_agent/
│   ├── lary_fundamental_agent/
│   └── mohamed_fundamental_agent/
│       └── src/
│           ├── ai_agent.py           # Agent entry point
│           ├── fundamental_analyzer.py # DCF and multiples
│           ├── Data_collector.py     # Alpha Vantage data fetching
│           └── report_generator.py   # Output formatting
├── shared_outputs/              # Agent JSON outputs and final PDF
├── run_demo.py                  # Demo script to run all agents
├── requirements.txt
└── .env                         # API keys (not committed)
```

## Output Files

After running, outputs are saved to `shared_outputs/`:

| File | Description |
|------|-------------|
| `technical_fardeen_<TICKER>.json` | Fardeen's technical analysis |
| `technical_tamer_<TICKER>.json` | Tamer's technical analysis |
| `fundamental_daria_<TICKER>.json` | Daria's fundamental analysis |
| `fundamental_shakzod_<TICKER>.json` | Shakzod's fundamental analysis |
| `fundamental_lary_<TICKER>.json` | Lary's fundamental analysis |
| `fundamental_mohamed_<TICKER>.json` | Mohamed's fundamental analysis |
| `FINAL_RECOMMENDATION.json` | Aggregated master agent output |
| `FINAL_RECOMMENDATION.pdf` | Investment committee report |

## Weighted Signal Aggregation

The final recommendation is determined algorithmically:

1. Each agent outputs a signal: BUY, HOLD, or SELL
2. Signals are normalized (STRONG_BUY -> BUY, STRONG_SELL -> SELL)
3. Agent weights are summed for each signal type
4. The signal with the highest total weight wins (plurality)
5. The LLM explains the recommendation but cannot override it

Example output:
```
Weighted Signal Aggregation:
  BUY:  35.0%
  HOLD: 50.0%
  SELL: 15.0%
BINDING RECOMMENDATION: HOLD (50.0% weight)
```

## API Rate Limiting

The system includes caching to minimize API calls:
- Alpha Vantage data cached for 72 hours
- Financial statements cached after first fetch
- Peer multiples cached to avoid repeated calls

## Agent Methodologies

### Technical Agents

**Fardeen (20% weight)**
- Momentum: RSI (14-period), MACD (12/26/9)
- Trend: SMA 50/200, ADX for trend strength
- Volatility: ATR, Bollinger Bands (20-period, 2 std dev)
- Regime Detection: Hurst exponent to classify trending vs mean-reverting markets
- Backtesting: 10-year historical test with 500 Monte Carlo simulations
- Position Sizing: Fractional Kelly criterion (0.25x), GARCH volatility adjustment
- Output: Signal with confidence score, entry/exit levels, scenario analysis

**Tamer (20% weight)**
- Momentum Oscillators: RSI, Stochastic %K/%D, Williams %R, Rate of Change
- Trend Indicators: MACD with histogram, ADX/DMI system, Aroon, Supertrend
- Volatility Systems: Bollinger Bands, Keltner Channels, Donchian Channels
- Volume Analysis: OBV, Chaikin Money Flow, Money Flow Index, VWMA
- Trading Systems: Ichimoku Kinko Hyo (5 components)
- Regime-aware thresholds that adapt to volatility conditions

### Fundamental Agents

**Daria (15% weight)**
- DCF Valuation: 5-year FCF projection, WACC discount rate, Gordon Growth terminal value
- Multiples: P/E, P/B, EV/EBITDA with peer comparison
- DDM: Dividend Discount Model for income stocks
- DuPont Analysis: ROE decomposition into margin, turnover, leverage
- Scenario Analysis: Bull/Base/Bear cases with probability weights

**Shakzod (15% weight)**
- 10-step analysis pipeline from data collection to memo generation
- Profitability Analysis: Margins, returns, operating efficiency
- Cash Flow Analysis: Operating, investing, financing flows
- Earnings Quality: Accruals ratio, cash conversion
- Working Capital: Liquidity, receivables/payables management
- Valuation: Multiple methodologies with cross-validation

**Lary (15% weight)**
- Financial Ratios: Profitability, liquidity, leverage, efficiency, growth
- Risk Assessment: Quantitative risk scoring model
- Financial Forecasting: Revenue and earnings projections
- Valuation: DCF and multiples with sensitivity tables
- PDF Reports: Institutional-quality formatted output

**Mohamed (15% weight)**
- DCF Valuation: 5-year FCFF projection, WACC (CAPM-based), terminal value
- Multiples Valuation: P/E, P/B, P/S, EV/EBITDA vs tech sector peers
- Blended Target: 60% DCF weight + 40% multiples weight
- Sensitivity Analysis: Terminal growth rate variations
- Data: Alpha Vantage API with 72-hour caching

## Requirements

- Python 3.10+
- See `requirements.txt` for full dependency list
- API keys: OpenAI or Anthropic (for LLM), Alpha Vantage (for market data)
