import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (two levels up from configs/)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# Data directories — override via .env if your layout differs
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))

NOTEBOOKS_DIR     = PROJECT_ROOT / "notebooks"
RAW_HTML_DIR      = DATA_DIR / "raw"  / "html_reports"
JSON_REPORTS_DIR  = DATA_DIR / "processed" / "json_reports"
JSON_CHUNKS_DIR   = DATA_DIR / "processed" / "json_chunks"
FAISS_INDEX_DIR   = DATA_DIR / "processed" / "index" / "FAISS"


def set_environment():
    """Inject config values into os.environ so libraries pick them up."""
    env_vars = {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "HUGGINGFACEHUB_API_TOKEN": HUGGINGFACEHUB_API_TOKEN,
    }
    for key, value in env_vars.items():
        if value:
            os.environ[key] = value
