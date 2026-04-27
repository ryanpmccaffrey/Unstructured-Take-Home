# Multimodal Agentic RAG — MMDocIR Take-Home

A prototype multimodal retrieval-augmented generation (RAG) system built over the [MMDocIR](https://arxiv.org/abs/2501.08828) benchmark dataset. The system answers complex, multi-hop questions grounded in long PDF documents containing interleaved text, tables, and figures.

**Stack:** Python 3.12 · Anthropic Claude · LangGraph · DSPy · ChromaDB · `bge-base-en-v1.5` · `bge-reranker-base`

> **Development environment:** This prototype was built and evaluated on a MacBook Pro 13-inch (2017) — 3.1 GHz Dual-Core Intel Core i5, 8 GB RAM, Intel Iris Plus Graphics 650 (1536 MB), macOS Ventura. The machine has no discrete GPU and less than 16 GB of RAM, which made several otherwise attractive approaches impractical: GPU-backed visual retrievers (ColPali/ColQwen require 3B+ parameter VLMs), PyTorch-based fine-tuned checkpoint loading (dependency conflicts on Intel Mac), and full-dataset ingestion at 170K+ chunks in a single run. Design choices were made with these constraints in mind; see [Future Improvements](#future-improvements) for notes on what would be unlocked on better hardware.

---

## Getting Started

### Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- `make` (pre-installed on macOS; `brew install make` if missing)
- An Anthropic API key (`ANTHROPIC_API_KEY`)
- The MMDocIR dataset parquet files placed at `data/MMDocIR_Evaluation/`

### Install

```bash
git clone https://github.com/ryanpmccaffrey/Unstructured-Take-Home.git
cd Unstructured-Take-Home
make install
```

### Download the dataset

The dataset is hosted on Hugging Face: [MMDocIR/MMDocIR_Evaluation_Dataset](https://huggingface.co/datasets/MMDocIR/MMDocIR_Evaluation_Dataset). Place the three parquet/jsonl files under `data/MMDocIR_Evaluation/`:

```
data/MMDocIR_Evaluation/
├── MMDocIR_annotations.jsonl
├── MMDocIR_layouts.parquet
└── MMDocIR_pages.parquet
```

### Build the index

```bash
export ANTHROPIC_API_KEY=sk-ant-...
make ingest
```

This selects the 5 pinned baseline documents (see `INCLUDED_DOCS` in `src/ingest/loader.py`), chunks each layout element into its own chunk, embeds with `bge-base-en-v1.5`, and persists to `chroma_db/`. Expect ~1,590 layout chunks and ~131 page chunks.

### Run evaluation

```bash
make eval
```

Results are written to `results/eval_YYYYMMDD_HHMMSS.csv` (per-question) and `results/eval_YYYYMMDD_HHMMSS.json` (aggregate) — each run gets a unique timestamped file.

---

## Project Layout

```
.
├── data/                        # Dataset files (gitignored)
│   └── MMDocIR_Evaluation/
├── chroma_db/                   # Persisted vector index (gitignored, rebuild with make ingest)
├── results/                     # Timestamped eval output files
├── notebooks/
│   └── eda.ipynb                # Exploratory analysis of dataset schema (gitignored)
├── scripts/
│   ├── ingest.py                # CLI: chunk → embed → upsert to Chroma
│   └── run_eval.py              # CLI: run agent over all questions, write metrics
├── src/
│   ├── config.py                # All constants and env var loading (single source of truth)
│   ├── embedder.py              # BGE-base ONNX embedder (CLS pooling, asymmetric retrieval)
│   ├── ingest/
│   │   ├── loader.py            # Document selection
│   │   ├── chunker.py           # Three-tier chunking algorithm (see Design Choices)
│   │   └── indexer.py           # Batch embed with BGE-base, upsert to Chroma collections
│   ├── retrieval/
│   │   ├── retriever.py         # Hierarchical retrieval + context expansion
│   │   └── reranker.py          # Cross-encoder re-ranker (bge-reranker-base ONNX)
│   ├── agent/
│   │   ├── state.py             # RAGState TypedDict shared by both agent implementations
│   │   ├── dspy_modules.py      # DSPy signatures + predictors (query analysis, sufficiency, generate, validate)
│   │   ├── nodes.py             # Planner-Executor LangGraph node functions + routing logic
│   │   ├── graph.py             # Planner-Executor StateGraph (default; used by make eval)
│   │   ├── tools.py             # ReAct tool definitions + execution
│   │   ├── react.py             # ReAct LangGraph node functions + routing logic
│   │   ├── graph_react.py       # ReAct StateGraph (alternate implementation)
│   │   └── image_store.py       # Lazy loader for layout image binaries from parquet
│   └── eval/
│       └── evaluate.py          # Metrics: page recall, layout recall (IoU), token F1,
│                                 #   fuzzy match, LLM-as-judge, exact containment,
│                                 #   citation grounding, per-modality breakdown
├── .env                         # ANTHROPIC_API_KEY (gitignored)
├── Makefile                     # install / format / ingest / eval
└── pyproject.toml
```

---

## Tool & Framework Rationale

### LangGraph

Both agent implementations (Planner-Executor and ReAct) are compiled `StateGraph` instances. LangGraph was chosen over custom orchestration because the two conditional loops — the sufficiency retry loop and the validation retry loop — are first-class graph constructs (conditional edges + cycle detection), not ad-hoc `while` loops embedded in business logic. This makes the control flow inspectable, the retry caps easy to enforce, and the implementations swappable: both share the same `RAGState` TypedDict and the same `.invoke(state)` interface. Switching the eval harness from one implementation to the other is a one-line import change.

### DSPy

Used for all four LLM calls in the Planner-Executor pipeline (query analysis, sufficiency check, answer generation, answer validation). DSPy was chosen over raw f-string prompts for two reasons: (1) each prompt is a typed Python `Signature` class, making inputs/outputs explicit and testable rather than buried in multiline strings; (2) DSPy's `MIPROv2` optimizer can improve prompt quality from a small labeled set without manual tuning — a concrete future upgrade path once a training split is available. `dspy.Image` also handles multimodal inputs cleanly without bespoke base64 plumbing in the prompt layer.

### ChromaDB

The vector store for both the layout-level (`mmdocir_layouts`) and page-level (`mmdocir_pages`) collections. ChromaDB is embedded — no server to run, no Docker, no connection strings — which makes local development and CI trivially simple. It supports metadata filtering (used to scope layout queries to candidate pages), stores raw embeddings (no built-in embedding function needed), and persists to disk via SQLite + Parquet. At the prototype scale (~1,600 chunks across 5 documents), it is effectively instantaneous. The architecture is unchanged if you swap it for Qdrant or Pinecone at production scale; only the client initialization and collection API calls need to change.

### BGE-base-en-v1.5 + BGE-reranker-base (ONNX via Xenova)

Both models run as quantized ONNX models via `onnxruntime` with no PyTorch dependency — a hard requirement on the development machine (Intel Core i5, macOS Ventura), where PyTorch 2.4+ has no available wheels. The BGE-base embedder (105 MB) uses asymmetric retrieval: queries get an instruction prefix (`"Represent this sentence for searching relevant passages: "`) while documents are embedded as-is, which improves precision for question-to-document retrieval. BGE-reranker-base is the highest-leverage single retrieval improvement after decent embeddings: as a cross-encoder it sees both the query and document together, catching semantic matches that cosine similarity misses. It is purpose-built to complement BGE embeddings and consistently outperforms lighter rerankers (e.g. MiniLM) on diverse retrieval benchmarks.

---

## Critical Design Choices

### 1. Three-Tier Indexing Schema

The dataset's documents are already segmented into semantically meaningful layout elements — text blocks, figures, tables, equations — which makes naive fixed-size token splitting not just unnecessary but counterproductive (it would cut tables in half or separate captions from their figures). Instead, each layout element becomes a chunk, preserving the natural structure of the document. These chunks are organized into a three-tier schema across two ChromaDB collections, with raw embeddings upserted in batches of 64.

#### Tier 1 — Layout chunks (`mmdocir_layouts` collection)

The primary retrieval unit. One chunk = one layout element. `chunk_id` format: `{doc_name}_page{page_id}_layout{layout_id}`.

| Field | Type | Description |
|---|---|---|
| `chunk_id` | `str` | Unique identifier: `{doc_name}_page{page_id}_layout{layout_id}` |
| `doc_name` | `str` | Source document name |
| `domain` | `str` | Dataset domain (finance, science, …) |
| `page_id` | `int` | Page within the document |
| `layout_ids` | `JSON str` | Source layout row IDs (`[layout_id]` for single elements; `[]` for bridge and metadata chunks) |
| `element_type` | `str` | `text`, `equation`, `figure`, `table`, `bridge`, or `metadata` |
| `document` | `str` | ChromaDB document field — the text embedded into the vector index and returned at query time (see Type-Aware Text Selection). Stored here rather than in metadata so ChromaDB indexes it natively; **not** present in the metadata dict. |
| `image_path` | `str` | Path to the cropped element image (figures and tables only) |
| `page_image_path` | `str` | Path to the full-page screenshot |
| `prev_chunk_id` | `str` | Previous chunk on the same page (linked list for context expansion) |
| `next_chunk_id` | `str` | Next chunk on the same page |
| `page_position` | `str` | `top` / `middle` / `bottom` — derived from the element's bbox y-center relative to page height, split at one-third intervals |
| `section_heading` | `str` | Text of the nearest short element (≤ 10 tokens) that preceded this chunk on the page — used as a section-context signal |
| `page_ids` | `JSON str` | Only set for bridge chunks: `[page_N, page_N+1]` |

**Chunking algorithm (per page):**

1. Walk layout elements in `layout_id` order.
2. Each element becomes its own chunk directly — no merging. The document parser has already segmented at paragraph granularity; merging would lower embedding precision and hurt IoU-based layout recall.
3. Any `text`/`equation` element ≤ 10 tokens is treated as a heading and stored in `section_heading` for subsequent chunks on the page.
4. `prev_chunk_id` / `next_chunk_id` pointers are assigned in a final pass to form a linked list within each page.

**Bridge chunks** (`element_type="bridge"`) are also stored in `mmdocir_layouts`. After each page is processed, the full text of the last prose element on page N is concatenated with the full text of the first prose element on page N+1. Whole elements are used (not fixed token slices) so bridge text is always semantically complete with no mid-sentence cuts. Bridge `chunk_id` format: `{doc_name}_bridge_{N}-{N+1}`.

**Metadata chunks** (`element_type="metadata"`) — one per document, synthesised from the first page's VLM description prefixed with `doc_name` and `domain`. Surfaces title, authors, and document-level context for questions about the document itself rather than its content.

#### Tier 2 — Page summaries (`mmdocir_pages` collection)

One entry per page, embedded from the pre-computed `vlm_text` (a holistic VLM description of the full page). Used for coarse candidate retrieval before drilling into Tier 1. `chunk_id` format: `{doc_name}_page_{page_id}`.

| Field | Type | Description |
|---|---|---|
| `chunk_id` | `str` | `{doc_name}_page_{page_id}` |
| `doc_name` | `str` | Source document name |
| `domain` | `str` | Dataset domain |
| `page_id` | `int` | Page within the document |
| `document` | `str` | ChromaDB document field — `vlm_text` for the page (falls back to `ocr_text` if absent); embedded and returned at query time, **not** in the metadata dict. |
| `image_path` | `str` | Path to the full-page screenshot |
| `child_chunk_ids` | `JSON str` | List of Tier 1 `chunk_id`s whose `page_id` matches this page |

#### Tier 3 — Document metadata (in-memory dict)

Derived from `MMDocIR_annotations.jsonl` at query time. Holds ground-truth QA pairs, page/layout index ranges, and domain labels. Used for metadata filtering when scoping retrieval to a single document, and for evaluation scoring — never embedded into the vector index.

### 2. Type-Aware Text Selection

Rather than treating all layout elements uniformly, the embedding text is chosen per element type:

| Layout type | `text_for_embedding` | Rationale |
|---|---|---|
| `text` / `equation` | `text` field (fallback: `ocr_text`) | Pre-parsed, clean |
| `figure` / `image` | `vlm_text` (fallback: `ocr_text`) | OCR on figures is mostly noise; fallback only when VLM description is absent |
| `table` | `ocr_text + "\n" + vlm_text` | OCR captures cell values; VLM captures structure |

This matters because VLM-generated text substantially outperforms OCR for semantic retrieval on visual content — a key finding from the MMDocIR paper.

### 3. Hierarchical Retrieval with Context Expansion and Re-ranking

A flat top-k query over layout chunks alone often returns the map *or* the table, but not both when evidence is scattered. Retrieval runs in four steps:

1. **Coarse**: Query Tier 2 (pages) for the top `TOP_K_PAGES = 10` candidate pages (fixed)
2. **Fine**: Query Tier 1 (layouts) with a metadata filter scoping to those page IDs — fetches `2× top_k` candidates for reranker headroom
3. **Re-rank**: `bge-reranker-base` (XLM-RoBERTa backbone, 279 MB quantized ONNX) scores each `(query, chunk)` pair jointly and keeps the top `top_k`. Unlike the bi-encoder which embeds query and document independently, the cross-encoder sees both together — significantly improving precision at the cost of not being pre-computable. BGE-reranker is purpose-built to complement BGE-base embeddings and outperforms lighter alternatives (e.g. MiniLM) on diverse retrieval benchmarks.
4. **Context expansion**: For each surviving layout chunk, fetch its `prev_chunk_id` and `next_chunk_id` siblings by ID — ensuring the model has surrounding context without additional vector queries

### 4. Planner-Executor Agent

Rather than a fixed pipeline, the system uses an adaptive **Planner-Executor** pattern implemented with LangGraph and DSPy. A lightweight planner classifies each query and produces a retrieval plan; a series of executor nodes carry out that plan; a sufficiency check can loop back to the planner on failure; and a validator can loop back to the generator on grounding failures. See [Agentic Design Pattern](#agentic-design-pattern) for the full graph structure.

### 5. Multimodal Evidence at Generation Time

Embeddings are text-only (`bge-base-en-v1.5` on VLM/OCR text). At generation time, for any `image` or `table` chunk in the reranked evidence, the system loads the original JPEG binary from the layouts parquet, resizes it to ≤ 1568 px on any dimension, and passes it to the DSPy generator as a `dspy.Image` alongside the text description. This gives Claude the ability to directly read chart values, table cells, and spatial relationships from the raw image — rather than relying solely on the pre-computed VLM description.

### 6. BGE-Base Embeddings via ONNX

Text chunks are embedded using `bge-base-en-v1.5` (768-dim), loaded as a quantized ONNX model via `onnxruntime` — no PyTorch or external embedding API required. The Xenova ONNX export of the model is downloaded once and cached locally under `bge_base_onnx_cache/`. BGE-large-en-v1.5 (1024-dim) was evaluated first but was prohibitively slow on the development machine (Intel Core i5, 8 GB RAM); the base variant is 3× smaller (105 MB vs 321 MB) with only a modest quality reduction.

BGE uses asymmetric retrieval: queries are prefixed with `"Represent this sentence for searching relevant passages: "` while documents are embedded as-is, which improves precision for question-to-document retrieval.

The `MMDocIR/MMDocIR_Retrievers` collection on HuggingFace contains fine-tuned versions of BGE and other text retrievers trained specifically on the MMDocIR document retrieval task. These would provide additional gains over the base model (see Future Improvements).

---

## Agentic Design Pattern

The system implements **Adaptive Retrieval with a Planner-Executor Architecture**. Unlike a static pipeline, the agent uses conditional logic to adapt retrieval strategy based on the query and to verify answer quality before returning.

### The Graph

**Node 1 — Query Analyzer (the "Planner")**

A lightweight DSPy call that classifies the incoming query along two dimensions:

- **Modality signal**: Is the answer likely in text, a figure/chart, a table, or requires cross-modal reasoning? This routes retrieval toward the right element types.
- **Granularity signal**: Does the query target a specific element ("what's the value in Figure 3") or a broad theme ("summarize the methodology")? This determines whether to do layout-level or page-level retrieval.

The planner outputs a retrieval plan: `{modality, granularity, rewritten_query, top_k}`. On retry, it receives the `insufficiency_reason` from the previous attempt and adapts the plan accordingly — switching granularity, adjusting element-type focus, or broadening the query.

**Node 2 — Retrieve**

Embeds the rewritten query with BGE-base-en-v1.5 and runs a vector search based on the plan:
- `layout` granularity: coarse Tier 2 page pass → fine Tier 1 layout pass with optional element-type filter
- `page` granularity: direct Tier 2 query, used for aggregation or broad thematic questions

Fetches `2× top_k` raw candidates to give the reranker sufficient headroom.

**Node 3 — Rerank**

Cross-encoder reranker (`bge-reranker-base`) scores each `(query, chunk)` pair jointly and keeps the top `top_k`. For layout-granularity results, also fetches prev/next linked-list siblings for context expansion.

**Node 4 — Sufficiency Check (the "agentic" part)**

A DSPy call that evaluates: *"Given these retrieved chunks, do I have enough evidence to answer the query?"* Two outcomes:

- **Sufficient** → proceed to generation
- **Insufficient** → emit an `insufficiency_reason` and loop back to the Query Analyzer with updated state so the planner can revise the strategy

Capped at 2 retries (3 total retrieval attempts) to bound cost.

**Node 5 — Generator**

A DSPy call that generates a concise, cited answer from the final reranked evidence. For figure and table chunks, raw images are passed alongside text descriptions as multimodal content. On re-generation, the validator's feedback is included so the model can correct specific grounding failures.

**Node 6 — Answer Validator**

A DSPy call that checks whether the answer is grounded in the retrieved evidence and flags hallucinations. If validation fails, loops back to the generator with specific feedback. Capped at 1 retry (2 total generation attempts).

### Graph Structure

```
Query → [Query Analyzer] → [Retrieve] → [Rerank] → [Sufficiency Check]
              ↑                                           ↙         ↘
              └──── (replan, max 2 retries) ─────────────       [Generate]
                                                                      ↓
                                                               [Validate]
                                                               ↙       ↘
                                                        (retry, max 1)  Final Answer
```

### Why This Pattern Over Alternatives

**Why not a simple sequential chain?** The sufficiency loop and the modality/granularity routing are inherently conditional — a fixed chain can't adapt when the first retrieval attempt returns the wrong element types or insufficient evidence. You'd either over-retrieve on every query (wasteful) or accept bad evidence without recourse.

**Why not a full autonomous agent (ReAct-style)?** ReAct is well-suited when the action space is open-ended and the agent must discover what to do. Here, the retrieval strategies are well-defined and enumerable. The Planner-Executor gives you the same adaptivity — it can switch granularity, adjust modality focus, and retry — with explicit retry caps, predictable token costs, and no risk of the model going off-script. The ReAct implementation is included in the codebase as a reference (see `graph_react.py`), and the results table shows both approaches side by side.

**Why not a simple router?** A pure router (classify → pick one strategy) has no recovery path. If the classification is wrong or the retrieved evidence is insufficient, the answer is just wrong. The sufficiency loop is what separates this from a router — it gives the system a mechanism to detect and correct bad initial routing rather than committing to it.

---

## Evaluation Dataset

The full MMDocIR benchmark comprises **313 documents** averaging 65 pages each (~20,000 pages and ~170,000 layout elements total). Due to hardware and time constraints on the development machine (Intel Core i5, 8 GB RAM — see note at top), evaluation was run on a 5-document prototype subset.  The author acknowledges that the small sample size may not be representative of a full benchmark and could introduce bias. Scope was significantly limited due to hardware and time constraints.

| | Baseline eval set | Full benchmark |
|---|---|---|
| Documents | 5 | 313 |
| Pages | 131 | ~20,395 |
| Layout elements | ~1,590 chunks | ~170,338 |
| Questions | 29 | 1,658 |
| Domains covered | 5 / 10 | 10 / 10 |

The 5-document baseline set is pinned via `INCLUDED_DOCS` in `src/ingest/loader.py` for reproducible comparison across configurations. It covers Research report, Tutorial/Workshop, Academic paper, Guidebook, and Brochure domains. To evaluate across all 10 domains, clear `INCLUDED_DOCS` and set `SUBSET_SIZE = 15` (or higher) in `src/config.py`.

---

## Results

All results on the 5-document, 29-question baseline set:

| Configuration | Page R@5 | Layout R@5 | Cited Page | Cited Layout | LLM Judge | Exact Contain. |
|---|---|---|---|---|---|---|
| ReAct Agent | 79.3% | 58.6% | 96.6% | **82.8%** | **82.8%** | 51.7% |
| **Planner-Executor** | **89.7%** | **62.1%** | **96.6%** | 65.5% | 79.3% | **55.2%** |

The biggest jump in **Page Recall** came from the Planner-Executor's query rewriting and adaptive retrieval — 79.3% → 89.7%. The biggest jump in **Layout Recall** (55.2% → 62.1%) came from element-level chunking. **Exact Containment** improved to a new best of 55.2%, reflecting the structured DSPy generation prompts.

**On Cited Layout Grounding**: The Planner-Executor scores 65.5% vs the prior best of 82.8%. The gap is partly structural — the sufficiency check sometimes accepts page-level evidence as sufficient for questions that require a specific layout element, leading to page-chunk citations that can't be IoU-matched. Forcing `granularity=layout` for table/figure modalities recovered the initial 34.5% to 65.5%; further improvement would come from tightening the sufficiency check for visual modalities.

**On Token F1 and Fuzzy Match**: Misleadingly low — ground-truth answers are short phrases while generated answers are full sentences. LLM Judge is the most meaningful quality indicator and aligns with MMDocRAG's evaluation approach.

Overall, I believe the Planner-Executor routing approach is preferred over the ReAct approach because it achieves comparable retrieval and answer quality with lower expected token usage and latency.  

---
## Future Improvements

### Retrieval Quality

- **Visual retrievers (ColPali / ColQwen)**: The MMDocIR paper's headline finding is that visual retrievers significantly outperform text retrievers. `MMDocIR/MMDocIR_Retrievers/colpali-v1.1` and `colqwen2-v1.0` embed raw page screenshots directly into vectors using a vision-language model backbone — no OCR or VLM text extraction needed. These require GPU memory (3B+ parameter VLMs) and a complete re-ingestion with image embeddings instead of text. A **hybrid retriever** combining text and visual embeddings provides the best overall recall across all question types.
- **MMDocIR fine-tuned text retrievers**: The `MMDocIR/MMDocIR_Retrievers` collection on HuggingFace contains checkpoints for BGE, E5, GTE, and ColBERT fine-tuned specifically on the MMDocIR document retrieval training set. Swapping in `MMDocIR/MMDocIR_Retrievers/bge-base-en-v1.5` or `bge-large-en-v1.5` would be the easiest next retrieval improvement. Loading requires PyTorch >= 2.4, which is unavailable on Intel Mac (PyTorch dropped x86 macOS wheels before reaching 2.4) — not a constraint on Linux or Apple Silicon.
- **Stronger text embeddings for production**: `voyage-3` or `voyage-multimodal-3` (Anthropic's embedding partner via Voyage AI) provide state-of-the-art retrieval quality and native multimodal support via a managed API, removing the need for local ONNX model management.
- **Larger context window usage**: Retrieve more candidates (top-20 or top-50) and let Claude select relevant chunks directly, trading latency for recall.

### Answer Quality

- **Extractive answer spans**: Post-process generated answers to extract the shortest answer span, which would make Token F1 a more meaningful metric and align with benchmark scoring conventions.
- **DSPy optimization**: Compile the `QueryAnalysis`, `SufficiencyCheck`, `AnswerGeneration`, and `AnswerValidation` modules with DSPy's `MIPROv2` optimizer using a small labeled set, improving prompt quality without manual tuning.
- **Structured answer output**: Enforce structured answer format (e.g. answer + confidence + supporting quotes) to make evaluation and downstream use more reliable.

### Scale and Coverage

- **Full dataset indexing**: Extend from 5 to all 313 documents. The architecture is unchanged; only `SUBSET_SIZE` in `src/config.py` needs updating. At full scale, expect ~170K layout chunks and ~20K page chunks — well within ChromaDB's capacity for a single-node deployment.
- **Incremental ingestion**: Add support for adding new documents to an existing index without full re-ingestion, using Chroma's `upsert` semantics.

---

## Path to Productionalization

Scaling this prototype to a production system serving 100,000+ documents requires
changes at every layer of the stack. The sections below address storage, ingestion,
serving, cost, and observability.

### Storage and Indexing

Replace ChromaDB (local, file-based) with a managed vector database. The choice
depends on retrieval strategy:

- **If using ColQwen2 / late-interaction retrieval**: the vector DB must support
  multi-vector representations per document (one embedding per token, scored via
  MaxSim). **Qdrant** and **Vespa** support this natively. Pinecone and Weaviate
  do not — you'd need to flatten to single-vector (CLS pooling), which sacrifices
  the late-interaction advantage that makes visual retrieval competitive.
- **If using single-vector retrieval only** (BGE/GTE): any managed vector DB works.
  Pinecone, Weaviate, and Qdrant are all viable.

**Sizing estimate**: At 100K documents, chunk counts depend heavily on document
length. MMDocIR's evaluation documents average 65 pages × 8.3 layouts per page
(~540 chunks per doc, ~54M chunks total). Shorter documents (10–20 pages) bring
this down to 8–16M. At this scale, sharding and approximate nearest neighbor (ANN)
indexing (HNSW) are necessary — managed services handle both transparently.

Raw document storage (PDFs, page screenshots, layout image crops) should live in
object storage (S3 or GCS), referenced by URI in the vector DB metadata. Only
embeddings and lightweight metadata belong in the vector DB itself.

Chunk and document metadata (provenance, ingestion status, domain, page count)
should live in a relational store (Postgres) for filtering, audit trails, and
lineage queries that vector DBs handle poorly.

### Ingestion Pipeline

The prototype processes documents sequentially — parse, embed, upsert, repeat.
At scale, this breaks in two ways: throughput (VLM description generation runs
2–5 seconds per layout element) and reliability (a mid-batch failure means
reprocessing everything). Production ingestion splits into two operational modes.

#### Bulk Backfill (onboarding existing document corpus)

For initial onboarding of 100K documents, the bottleneck is VLM enrichment.
At ~540 layouts per document and ~3 seconds per VLM call, a single worker would
take ~5 years. This requires distributed compute:

1. Stage documents in S3. Build a manifest (document ID → S3 path).
2. Run MinerU layout detection in parallel across a worker pool (**Ray**,
   **AWS Batch**, or **Kubernetes Jobs**). This is CPU-bound and embarrassingly
   parallel across documents.
3. Distribute VLM enrichment calls across a **Ray** cluster or an async worker
   pool. If self-hosting the VLM (e.g., Qwen2-VL via vLLM), Ray Serve is the
   natural fit — it manages GPU allocation, batching, and autoscaling in one
   framework. If using an external API (GPT-4o, Claude), use an async HTTP
   client with rate limiting and exponential backoff. Aim for 100–500 concurrent
   calls depending on API tier.
4. Embed in batches (text embeddings are fast; visual embeddings via ColQwen2
   require GPU) and bulk-upsert to the vector DB.

#### Steady-State Ingestion (new documents arriving continuously)

For ongoing ingestion (tens to hundreds of documents per day):

1. Document uploaded to S3 → emits an event to **SQS** (simpler) or **Kafka**
   (if you need ordering guarantees, replay, and backpressure handling).
2. A worker pool consumes events, runs the same parse → enrich → embed → upsert
   pipeline per document.
3. Each stage separated by a durable queue so that a spike in uploads doesn't
   cascade into VLM API rate limits or vector DB write pressure.

Failed documents are dead-lettered and retried independently.

### Serving

The multi-hop agent (Planner → parallel retrieval → fusion → reranking →
sufficiency check → generation) is not a single request-response cycle. A single
query may trigger 2–3 retrieval iterations, each involving vector DB lookups and
LLM calls. Expected latency: 10–30 seconds per query.

Wrap the agent in a **FastAPI** service:

- `POST /query` — accepts a question and optional filters. Returns a streaming
  response (SSE or WebSocket) so the user sees incremental progress rather than
  waiting 30 seconds for a complete answer. Include retrieval metadata (which
  chunks were retrieved, confidence scores, hop count) in the response for
  transparency.
- `POST /ingest` — triggers async ingestion, returns a job ID. Status polled
  via `GET /ingest/{job_id}`.
- `GET /health` — for load balancer health checks.

**Caching**: For multi-hop queries, caching the final answer on question
similarity is low-hit-rate — the planner generates dynamic intermediate queries
unlikely to repeat verbatim. More effective: cache retrieval results at the chunk
level (chunk ID → embedding, metadata, VLM description) and cache reranker scores
for (query, chunk) pairs seen recently. Use **Redis** with TTL-based eviction.

**Timeouts and circuit breakers**: Cap agent loops at 3 iterations. Set per-call
timeouts on LLM and vector DB requests. If the VLM API is degraded, fall back to
text-only retrieval rather than failing the entire query.

### Cost Estimation

At 100K documents, the major cost drivers are:

| Component | Estimate |
|---|---|
| VLM enrichment (one-time backfill) | ~54M layout images × $0.01–0.03/call = **$540K–$1.6M** (API pricing) or significantly less self-hosted |
| Text embeddings (one-time) | ~54M chunks × BGE = negligible (runs on CPU in hours) |
| Visual embeddings (one-time) | ~54M page/layout images × ColQwen2 = significant GPU time, days on a multi-GPU node |
| Vector DB hosting | Qdrant Cloud or equivalent: ~$500–2,000/month at this scale depending on replication |
| Per-query LLM cost | ~5–10 LLM calls per multi-hop query × $0.01–0.05/call = $0.05–0.50 per query |

The VLM enrichment cost dominates. Self-hosting Qwen2-VL on GPU instances
(~$2–4/hr per A100) with Ray Serve can reduce this by 5–10x compared to API
pricing, at the cost of operational complexity.

### Observability

At scale, you need to trace retrieval quality, not just uptime:

- **Query-level tracing**: Log each hop's retrieval results, reranker scores,
  sufficiency check outcome, and final generation. Tools: **LangSmith** or
  **Arize Phoenix** (both integrate with LangGraph).
- **Retrieval quality monitoring**: Sample queries periodically, compare
  retrieved chunks against known-good answers (using MMDocIR's evaluation set as
  a regression suite). Alert on recall degradation.
- **Cost tracking**: Log token usage and LLM calls per query. Set per-user and
  per-hour budgets to prevent runaway agent loops from burning through API credits.
- **Latency dashboards**: P50/P95/P99 for each stage (retrieval, reranking,
  generation) independently, not just end-to-end. The bottleneck shifts depending
  on load.
