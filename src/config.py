import os
from pathlib import Path

ROOT = Path(__file__).parent.parent

DATA_DIR = ROOT / "data" / "MMDocIR_Evaluation"
ANNOTATIONS_PATH = DATA_DIR / "MMDocIR_annotations.jsonl"
PAGES_PATH = DATA_DIR / "MMDocIR_pages.parquet"
LAYOUTS_PATH = DATA_DIR / "MMDocIR_layouts.parquet"

CHROMA_PATH = str(ROOT / "chroma_db")

SUBSET_SIZE = 5
TOP_K_PAGES = 10
TOP_K_LAYOUTS = 5
MERGE_TOKEN_THRESHOLD = 400
HEADING_MAX_TOKENS = 10
BRIDGE_OVERLAP_TOKENS = 100

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

EMBED_MODEL = "all-MiniLM-L6-v2"  # used by Chroma's DefaultEmbeddingFunction
CLAUDE_MODEL = "claude-sonnet-4-6"

COLLECTION_LAYOUTS = "mmdocir_layouts"
COLLECTION_PAGES = "mmdocir_pages"
