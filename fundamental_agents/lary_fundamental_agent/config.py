# config.py
import os
from dotenv import load_dotenv
from pathlib import Path

# Load from master repo .env
repo_root = Path(__file__).parent.parent.parent
load_dotenv(repo_root / ".env")

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY_LARY", "9GZT5T05DZS83LAE")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")