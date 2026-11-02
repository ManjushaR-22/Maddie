import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
PDF_DIR = DATA_DIR / "pdf"
TEXT_DIR = DATA_DIR / "text_files"

DEFAULT_COLLECTION = "pdf_documents"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "openai/gpt-oss-20b"

AVAILABLE_MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
]

def get_groq_api_key() -> str:
    """Retrieve Groq API key from environment variables, checking both standard and user variations."""
    key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
    if not key:
        raise ValueError(
            "Groq API Key not found. Please set GROQ_API_KEY or GROK_API_KEY in your .env file."
        )
    return key
