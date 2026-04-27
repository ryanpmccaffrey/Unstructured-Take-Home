# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for package management (Python 3.12).

```bash
make install                          # uv sync
make format                           # ruff format
make ingest                           # chunk → embed → upsert to ChromaDB
make eval                             # run agent over all questions, write results
uv run ruff check                     # lint
uv run jupyter notebook               # launch notebooks
uv add <package>                      # add a dependency
export $(cat .env | xargs) && make eval  # load API key from .env then eval
```

`.env` must contain `ANTHROPIC_API_KEY=sk-...`.

There are no automated tests. The eval pipeline (`make eval`) is the validation mechanism.

## Architecture

A multimodal agentic RAG system over the **MMDocIR** benchmark dataset. The goal is multi-modal retrieval and question answering over long PDF documents (313 docs, avg 65 pages). The prototype evaluates a 5-document pinned baseline set.

### Dataset (`data/MMDocIR_Evaluation/`)

Three core files define two retrieval granularities:

| File | Rows | Description |
|------|------|-------------|
| `MMDocIR_annotations.jsonl` | 313 | One row per document; contains 1,658 QA pairs with ground-truth page and layout labels |
| `MMDocIR_pages.parquet` | 20,395 | One row per page; contains full-page screenshots (`image_binary`), `ocr_text`, and `vlm_text` |
| `MMDocIR_layouts.parquet` | 170,338 | One row per layout element (text/image/table/equation) within a page; contains cropped `image_binary`, `bbox`, `ocr_text`/`vlm_text` for non-text types |

**Joining annotations → pages/layouts**: `annotations.page_indices` and `annotations.layout_indices` are `[start, end]` *row index* ranges into the respective parquet files — not ID-based lookups. Use `.iloc[start:end+1]` to slice.

**Layout types**: `text`, `image`, `table`, `equation`. Only `image`/`table` have `ocr_text` and `vlm_text`; `text`/`equation` have raw `text`.

### Retrieval task structure

Questions (`annotations.questions`) have `type` (modality), `page_id` (list of ground-truth page IDs), and `layout_mapping` (list of `{page, page_size, bbox}` dicts identifying the specific layout element(s)). Evaluation is at two levels:
- **Page retrieval**: retrieve the correct page(s)
- **Layout retrieval**: retrieve the specific layout element(s) within a page

### Two agent code paths

Both are fully wired LangGraph `StateGraph` compilations sharing the same `RAGState` TypedDict and the same `graph.invoke(state)` interface. To switch, change the import in `evaluate.py` from `src.agent.graph` to `src.agent.graph_react`.

**Default path — Planner-Executor** (`graph.py` → `nodes.py` + `dspy_modules.py`): A DSPy planner classifies the query and produces a retrieval plan; LangGraph executor nodes carry out the plan; a sufficiency check can loop back to the planner (max 2 retries); a validator can loop back to the generator (max 1 retry). This is what `make eval` runs.

**Alternate path — ReAct** (`graph_react.py` → `react.py` + `tools.py`): Claude drives the retrieval loop by calling typed tools (search, search_table, search_figure, search_pages, get_document_metadata), inspecting results including images, and deciding when it has enough evidence to answer. Capped at `MAX_TURNS = 8`.

### `src/` — Implementation

#### `src/config.py`
Single source of truth for all constants and env vars: paths, ChromaDB collection names, `SUBSET_SIZE` (currently 5), `TOP_K_PAGES` (10), `TOP_K_LAYOUTS` (5), `TOP_K_RETRIEVE` (10, pre-rerank candidate count), `HEADING_MAX_TOKENS` (10), model names, ONNX cache dirs.

#### `src/embedder.py`
`BGEEmbedder` — loads `Xenova/bge-base-en-v1.5` as a quantized ONNX model via `onnxruntime`. No PyTorch required. Uses CLS-token pooling + L2 normalization. Asymmetric retrieval: queries get the instruction prefix `"Represent this sentence for searching relevant passages: "`, documents are embedded as-is. Singleton via `get_embedder()`.

#### `src/ingest/loader.py`
`load_subset(seed=42)` — if `INCLUDED_DOCS` is non-empty, loads exactly those documents (used for reproducible baseline comparisons). Otherwise does stratified random sampling: one document guaranteed per domain, remaining slots filled from a shuffled pool. `EXCLUDED_DOCS` always skipped. Returns `(doc_metadata_list, pages_df, layouts_df)`. To pin a different document set, edit the `INCLUDED_DOCS` list in this file.

#### `src/ingest/chunker.py`
`chunk_document()` — produces Tier 1 layout chunks and Tier 2 page chunks.
- **Tier 1 — element-level**: one chunk per layout element. No merging — the document parser already segmented at paragraph granularity. Merging would lower embedding precision and hurt IoU recall. chunk_id: `{doc_name}_page{page_id}_layout{layout_id}`.
- **Metadata chunks**: one per document (`element_type="metadata"`), synthesised from first page VLM text. Handles title/author/date questions.
- **Bridge chunks**: full text of last prose element on page N + full text of first prose element on page N+1 (`element_type="bridge"`). Whole elements used — no token slices — to avoid mid-sentence cuts.
- **section_heading**: tracks nearest short element (≤ HEADING_MAX_TOKENS) as metadata; stored on subsequent chunks but not merged into them.
- **Tier 2**: one entry per page using `vlm_text` (falls back to `ocr_text`).

#### `src/ingest/indexer.py`
`build_index()` — embeds chunks in batches of 64 using `BGEEmbedder`, upserts to two ChromaDB collections: `mmdocir_layouts` (Tier 1) and `mmdocir_pages` (Tier 2). Raw embeddings stored (no built-in EF).

#### `src/retrieval/retriever.py`
Two retrieval entry points depending on which agent is active:
- `retrieve_candidates(query, chroma, granularity, modality, top_k)` — used by the Planner-Executor (`node_retrieve`). Returns raw candidates without reranking; `node_rerank` handles reranking separately. `granularity="layout"` runs the coarse page pass → fine layout pass; `granularity="page"` queries Tier 2 directly.
- `retrieve(query, chroma, strategy, doc_filter, modality)` — used by the ReAct tools. Dispatches to `hierarchical_retrieve` (coarse page → fine layout → rerank → context expansion) or `direct_layout_retrieve` (skips page pass).
- `_MODALITY_ELEMENT_FILTER`: maps `"table"` → `"table"`, `"figure"` → `"image"`, `"metadata"` → `"metadata"` for element_type filtering in both paths.

#### `src/retrieval/reranker.py`
`CrossEncoderReranker` — loads `Xenova/bge-reranker-base` (`model_quantized.onnx`, 279 MB, XLM-RoBERTa backbone) via `onnxruntime`. Scores `(query, document)` pairs jointly. Automatically detects whether the model requires `token_type_ids` (BERT-based models do, RoBERTa-based models don't). Falls back to `model_quint8_avx2.onnx` if present (for ms-marco-MiniLM compatibility). Singleton via `get_reranker()`.

#### `src/agent/state.py`
`RAGState` TypedDict — shared interface for both agent implementations and the eval harness. Three field groups: (1) shared outputs (`answer`, `cited_chunk_ids`, `retrieved_chunks` with `_safe_add` dedup reducer); (2) Planner-Executor fields (`modality`, `granularity`, `rewritten_query`, `top_k`, `candidate_chunks`, `reranked_chunks`, sufficiency/validation loop state); (3) ReAct field (`messages` with `operator.add` append reducer for conversation history). Both graphs use the same TypedDict; each only reads/writes its own group.

#### `src/agent/image_store.py`
`LayoutImageStore` — lazily loads all `image` and `table` rows from the layouts parquet into a `(doc_name, layout_id) → bytes` dict on first access. All 14,902 visual rows have non-empty `image_binary`. Used by `dspy_modules.py` (`_chunk_images`) to build `dspy.Image` objects for the Planner-Executor generator, and by `tools.py` to attach JPEG images to ReAct tool results.

#### `src/agent/dspy_modules.py`
DSPy signatures and configured predictors for the Planner-Executor pipeline. Four `dspy.Signature` classes: `QueryAnalysis` (modality, granularity, rewritten_query, top_k outputs), `SufficiencyCheck` (is_sufficient + insufficiency_reason), `AnswerGeneration` (answer + cited_chunk_ids, accepts `list[dspy.Image]`), `AnswerValidation` (is_valid + feedback, also multimodal). Predictors are `dspy.Predict` instances configured against `claude-sonnet-4-6` via LiteLLM. Helper functions: `_chunk_images(chunks)` resizes and base64-encodes figure/table images into `dspy.Image` objects; `_format_context(chunks)` formats chunk text with `[chunk_id]` prefixes.

#### `src/agent/nodes.py`
LangGraph node functions for the Planner-Executor: `node_query_analyzer` (calls `analyze_query`, clamps `top_k` to [5,20], forces `granularity="layout"` for table/figure modalities regardless of DSPy output — critical override), `node_retrieve` (calls `retrieve_candidates` with `2× top_k`), `node_rerank` (cross-encoder rerank + context expansion for layout granularity), `node_sufficiency_check`, `node_generate`, `node_validate`. Routing functions: `route_after_sufficiency` (max 2 retries), `route_after_validation` (max 1 retry).

#### `src/agent/tools.py`
Tool definitions and execution for the ReAct agent. Five tools:
- `search` — hierarchical retrieval, no element_type filter
- `search_table` — hierarchical retrieval, `element_type="table"`
- `search_figure` — hierarchical retrieval, `element_type="image"`
- `search_pages` — queries `mmdocir_pages` directly (Tier 2, top-15), for aggregation across many pages
- `get_document_metadata` — `direct_layout_retrieve` with `element_type="metadata"`

`_chunks_to_content_blocks()` converts chunks to Anthropic content blocks — text + base64 image (resized to ≤ 1568 px) for figure/table chunks.

#### `src/agent/react.py`
LangGraph node functions for the ReAct agent: `call_llm` (Anthropic API call with full message history), `execute_tools` (runs all tool_use blocks from the last assistant message, accumulates chunks), `finalize` (extracts answer from last text block — MAX_TURNS fallback), `should_continue` (routing: "tools" if tool_use blocks present and under MAX_TURNS, else "end"). Wired into `graph_react.py`.

#### `src/agent/graph_react.py`
Compiles the ReAct `StateGraph`: `START → call_llm → should_continue → execute_tools → call_llm` (loop) or `→ finalize → END`. Exports `graph` with the same `.invoke(state)` interface as the Planner-Executor graph.

#### `src/agent/graph.py`
Compiles the Planner-Executor `StateGraph` and exports `graph`. This is what `evaluate.py` imports by default. See `graph_react.py` for the ReAct alternative.

#### `src/eval/evaluate.py`
`run_evaluation()` — runs the full agent over all questions and computes:
- `page_recall@5`: ground-truth page in first 5 `retrieved_chunks` (note: undercounts for ReAct since it accumulates chunks across many turns — `cited_page_grounded` is the more reliable signal for both paths)
- `layout_recall@5`: IoU > 0.5 between retrieved chunk bbox and ground-truth layout bbox
- `cited_page_grounded`: Claude's cited chunks include the ground-truth page
- `cited_layout_grounded`: Claude's cited chunks overlap the ground-truth layout element
- `token_f1`, `fuzzy_match`, `exact_containment`, `llm_judge`
- `modality_breakdown`: per question type (Chart, Table, Figure, Pure-text, etc.)

Writes timestamped output to `results/eval_YYYYMMDD_HHMMSS.csv` and `.json`.

### ONNX model caches

All gitignored, downloaded via `huggingface_hub.hf_hub_download` on first use:
- `bge_base_onnx_cache/` — `Xenova/bge-base-en-v1.5` (105 MB, `onnx/model_quantized.onnx`)
- `bge_reranker_base_cache/` — `Xenova/bge-reranker-base` (279 MB, `onnx/model_quantized.onnx`)

### Hardware constraints

Developed on MacBook Pro 13-inch (2017): Intel Core i5, 8 GB RAM, no discrete GPU, macOS Ventura. Key constraints:
- PyTorch >= 2.4 not available (dropped Intel Mac wheels) → all model inference via ONNX
- `transformers 5.x` requires PyTorch >= 2.4 → can't use standard HuggingFace model loading
- No GPU → ColPali/ColQwen visual retrievers impractical
- 8 GB RAM → BGE-large (321 MB) was too slow; BGE-base (105 MB) is the current embedder

### Notebooks

`notebooks/eda.ipynb` — exploratory analysis; loads all three dataset files. Run notebook cells from the `data/` directory or adjust paths accordingly (the notebook uses relative paths like `'MMDocIR_Evaluation/MMDocIR_annotations.jsonl'`).
