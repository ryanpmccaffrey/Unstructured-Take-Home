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
TOP_K_LAYOUTS = 5       # final layout chunks returned after re-ranking
TOP_K_RETRIEVE = 10     # initial layout candidates fetched before re-ranking
HEADING_MAX_TOKENS = 10

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

EMBED_MODEL = "bge-base-en-v1.5"  # Xenova ONNX quantized, loaded via onnxruntime
BGE_ONNX_DIR = str(ROOT / "bge_base_onnx_cache")
RERANKER_ONNX_DIR = str(ROOT / "bge_reranker_base_cache")
CLAUDE_MODEL = "claude-sonnet-4-6"

COLLECTION_LAYOUTS = "mmdocir_layouts"
COLLECTION_PAGES = "mmdocir_pages"
