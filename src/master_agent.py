"""
Master Agent - Runs all agents in parallel and synthesizes results
"""

import asyncio
import subprocess
import json
import os
from pathlib import Path
from datetime import datetime

# Paths
REPO_ROOT = Path(__file__).parent.parent
SHARED_OUTPUTS = REPO_ROOT / "shared_outputs"

# Agent configurations: (name, script_path)
AGENTS = [
    ("technical_fardeen", REPO_ROOT / "technical_agents" / "fardeen_technical_agent" / "src" / "llm_agent.py"),  # adjust entry point
    ("fundamental_daria", REPO_ROOT / "fundamental_agents" / "daria_fundamental_agent" / "run_demo.py"),  # adjust entry point
    # Add more agents here as teammates join
]


async def run_agent(name: str, script_path: Path) -> dict:
    """Run a single agent asynchronously."""
    print(f"[{name}] Starting...")
    start = datetime.now()
    
    try:
        # Run the agent script as subprocess
        process = await asyncio.create_subprocess_exec(
            "python", str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=script_path.parent  # Run from agent's directory
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
    """Run all agents in parallel."""
    print("=" * 60)
    print("MASTER AGENT - Parallel Execution")
    print("=" * 60)
    
    # Clear old outputs
    SHARED_OUTPUTS.mkdir(exist_ok=True)
    
    # Run all agents concurrently
    tasks = [run_agent(name, path) for name, path in AGENTS if path.exists()]
    results = await asyncio.gather(*tasks)
    
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    for r in results:
        status = "✓" if r["status"] == "success" else "✗"
        print(f"  {status} {r['agent']}: {r['status']}")
    
    return results


def load_all_outputs() -> list[dict]:
    """Load all JSON outputs from shared_outputs/."""
    outputs = []
    for json_file in SHARED_OUTPUTS.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                data["_source_file"] = json_file.name
                outputs.append(data)
                print(f"  Loaded: {json_file.name}")
        except Exception as e:
            print(f"  Failed to load {json_file.name}: {e}")
    return outputs


def synthesize_with_llm(outputs: list[dict]) -> dict:
    """Send all outputs to LLM for synthesis."""
    from openai import OpenAI  # or use anthropic
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = f"""You are a senior portfolio manager synthesizing research from multiple analysts.

You have received {len(outputs)} analyst reports for AAPL:

{json.dumps(outputs, indent=2, default=str)[:15000]}  # Truncate if too long

Synthesize these into a final investment recommendation:

1. What is the consensus recommendation? (STRONG BUY / BUY / HOLD / SELL / STRONG SELL)
2. What is the weighted average target price?
3. What is the overall confidence level? (0-100%)
4. Key supporting points from technical analysis
5. Key supporting points from fundamental analysis
6. Main risks identified
7. Final investment thesis (2-3 sentences)

Output as JSON:
{{
    "ticker": "AAPL",
    "recommendation": "BUY/SELL/HOLD",
    "target_price": float,
    "confidence": float,
    "technical_summary": "...",
    "fundamental_summary": "...", 
    "key_risks": ["...", "..."],
    "investment_thesis": "..."
}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    
    # Parse JSON from response
    text = response.choices[0].message.content
    # Extract JSON
    import re
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        return json.loads(match.group())
    return {"raw_response": text}


def save_final_report(synthesis: dict):
    """Save final synthesized report."""
    output_path = SHARED_OUTPUTS / "FINAL_RECOMMENDATION.json"
    with open(output_path, 'w') as f:
        json.dump(synthesis, f, indent=2)
    print(f"\nFinal report saved: {output_path}")
    return output_path


async def main():
    # Step 1: Run all agents in parallel
    await run_all_agents()
    
    # Step 2: Load all outputs
    print("\n" + "=" * 60)
    print("LOADING AGENT OUTPUTS")
    print("=" * 60)
    outputs = load_all_outputs()
    
    if not outputs:
        print("No outputs found! Check if agents ran correctly.")
        return
    
    # Step 3: Synthesize with LLM
    print("\n" + "=" * 60)
    print("SYNTHESIZING WITH LLM")
    print("=" * 60)
    synthesis = synthesize_with_llm(outputs)
    
    # Step 4: Save final report
    save_final_report(synthesis)
    
    # Step 5: Print summary
    print("\n" + "=" * 60)
    print("FINAL RECOMMENDATION")
    print("=" * 60)
    print(f"  Ticker: {synthesis.get('ticker', 'AAPL')}")
    print(f"  Recommendation: {synthesis.get('recommendation', 'N/A')}")
    print(f"  Target Price: ${synthesis.get('target_price', 0):.2f}")
    print(f"  Confidence: {synthesis.get('confidence', 0):.0f}%")
    print(f"\n  Thesis: {synthesis.get('investment_thesis', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())