# app/config.py

import os
from dotenv import load_dotenv

load_dotenv()  # Load .env

# AI Proxy token is mandatory
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN is not set in environment variables (.env).")

# OpenAI API key is optional. If present, we'll prefer it for chat/embeddings.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

FALLBACK_EMAIL = "21f3003062@ds.study.iitm.ac.in"

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
