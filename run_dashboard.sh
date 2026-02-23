#!/bin/bash
# Launch the Master Analyst Agent Dashboard
# Auto-opens browser after startup

cd "$(dirname "$0")"
DIR="$(pwd)"

# Activate virtual environment (try both common names)
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: No virtual environment found."
    echo "Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Exclude non-dashboard directories from hot-reload file watcher.
# Without this, the pipeline subprocess writing to shared_outputs/ (and importing
# from src/, fundamental_agents/, technical_agents/) triggers a backend restart
# + frontend recompile, killing the live timer after ~7 seconds.
export REFLEX_HOT_RELOAD_EXCLUDE_PATHS="$DIR/shared_outputs:$DIR/src:$DIR/fundamental_agents:$DIR/technical_agents:$DIR/data:$DIR/venv"

# Kill any existing Reflex / Node processes on our ports
lsof -ti:3000 2>/dev/null | xargs kill -9 2>/dev/null
lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null
sleep 1

# Open browser once the frontend port is ready (poll every 2s, max 120s)
(
    for i in $(seq 1 60); do
        sleep 2
        if curl -s -o /dev/null -w '' http://localhost:3000 2>/dev/null; then
            if command -v open &> /dev/null; then
                open "http://localhost:3000"
            elif command -v xdg-open &> /dev/null; then
                xdg-open "http://localhost:3000"
            fi
            exit 0
        fi
    done
) &

echo "================================================"
echo "  Master Analyst Agent — Dashboard"
echo "  Browser will open automatically once ready."
echo "================================================"
echo ""

exec reflex run
