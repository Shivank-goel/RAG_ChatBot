# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")

# Embeddings / generation defaults (we’ll plug these in later steps)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "google/flan-t5-base")

# Vector store and retrieval settings
CHROMA_DIR = os.getenv("CHROMA_DIR", "storage")
TOP_K = int(os.getenv("TOP_K", 4))

# Chunking defaults (for later)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 600))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))

# Alpha Vantage rate limit controls
AV_RATE_LIMIT_SLEEP = int(os.getenv("AV_RATE_LIMIT_SLEEP", 13))  # free tier ~5 req/min
AV_BASE_URL = "https://www.alphavantage.co/query"
