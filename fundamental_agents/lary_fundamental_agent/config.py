# config.py
import os
from dotenv import load_dotenv

# Load from master repo .env
load_dotenv()

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY_LARY", "9GZT5T05DZS83LAE")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")