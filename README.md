# Multimodal Agentic RAG — MMDocIR Take-Home

A prototype multimodal retrieval-augmented generation (RAG) system built over the [MMDocIR](https://arxiv.org/abs/2501.08828) benchmark dataset. The system answers complex, multi-hop questions grounded in long PDF documents containing interleaved text, tables, and figures.

**Stack:** Python 3.12 · LangGraph · DSPy · Anthropic Claude · ChromaDB · `all-MiniLM-L6-v2`

---

## Getting Started

### Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) for dependency management
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

This selects 5 documents (one per domain), chunks them into a three-tier schema, embeds with `all-MiniLM-L6-v2`, and persists to `chroma_db/`. Expect ~500 layout chunks and ~130 page chunks.

### Run evaluation

```bash
make eval
```

Results are written to `eval_results.csv` (per-question) and `eval_summary.json` (aggregate).

### Ask a question interactively

```python
from src.agent.graph import graph
from src.agent.state import RAGState

result = graph.invoke({
    "question": "Which subgroup among Hispanics gained most confidence from 2008 to 2015?",
    "modality": "", "is_multi_hop": False, "retrieval_strategy": "hierarchical",
    "rewritten_query": "", "candidate_page_ids": [], "retrieved_chunks": [],
    "is_sufficient": False, "answer": "", "cited_chunk_ids": [], "retry_count": 0,
})
print(result["answer"])
print(result["cited_chunk_ids"])
```

---

## Project Layout

```
.
├── data/                        # Dataset files (gitignored)
│   └── MMDocIR_Evaluation/
├── chroma_db/                   # Persisted vector index (gitignored, rebuild with make ingest)
├── notebooks/
│   └── eda.ipynb                # Exploratory analysis of dataset schema
├── scripts/
│   ├── ingest.py                # CLI: chunk → embed → upsert to Chroma
│   └── run_eval.py              # CLI: run agent over all questions, write metrics
├── src/
│   ├── config.py                # All constants and env var loading (single source of truth)
│   ├── ingest/
│   │   ├── loader.py            # Load parquets, select document subset
│   │   ├── chunker.py           # Core chunking algorithm (see Design Choices)
│   │   └── indexer.py           # Embed with DefaultEmbeddingFunction, upsert to Chroma
│   ├── retrieval/
│   │   └── retriever.py        # Hierarchical retrieval + context expansion
│   ├── agent/
│   │   ├── state.py            # LangGraph TypedDict state schema
│   │   ├── dspy_modules.py     # DSPy signatures: QueryAnalysis, SufficiencyCheck
│   │   ├── image_store.py      # Lazy loader for layout image binaries from parquet
│   │   ├── nodes.py            # LangGraph node functions
│   │   └── graph.py            # Graph assembly and compilation
│   └── eval/
│       └── evaluate.py         # Metrics: page recall, layout recall (IoU), token F1,
│                                #          fuzzy match, LLM-as-judge, exact containment
├── eval_results.csv             # Per-question evaluation output
├── eval_summary.json            # Aggregate metrics
├── Makefile                     # install / format / ingest / eval
└── pyproject.toml
```

---

## Critical Design Choices

### 1. Three-Tier Indexing Schema

The dataset provides layout elements (text blocks, figures, tables, equations) that are already semantically segmented — far better than naive fixed-size token splitting, which would cut tables in half or separate captions from their figures. The schema has three tiers:

**Tier 1 — Layout chunks** (Chroma collection: `mmdocir_layouts`): The primary retrieval unit. Adjacent small text/equation elements within a page are merged until a 400-token threshold is reached; figures and tables are always kept standalone. Each chunk carries:
- `layout_ids[]` — which source layout rows were merged in
- `prev_chunk_id` / `next_chunk_id` — linked list for context expansion
- `section_heading` — nearest ancestor heading element
- `page_position` — top / middle / bottom (derived from bbox)

**Tier 2 — Page summaries** (Chroma collection: `mmdocir_pages`): One entry per page using the pre-computed `vlm_text` (holistic VLM description of the full page). Used for coarse candidate retrieval before drilling into layout chunks. Each page chunk stores `child_chunk_ids[]` pointing to its Tier 1 children.

**Tier 3 — Document metadata** (in-memory dict): Derived from `MMDocIR_annotations.jsonl`. Holds ground-truth QA pairs and index ranges. Used for metadata filtering and evaluation — not embedded.

**Bridge chunks** are also stored in the `mmdocir_layouts` collection (`element_type="bridge"`). Each bridge contains the last ~100 tokens of page N concatenated with the first ~100 tokens of page N+1, specifically to handle multi-hop questions where evidence spans a page boundary.

### 2. Type-Aware Text Selection

Rather than treating all layout elements uniformly, the embedding text is chosen per element type:

| Layout type | `text_for_embedding` | Rationale |
|---|---|---|
| `text` / `equation` | `text` field (fallback: `ocr_text`) | Pre-parsed, clean |
| `figure` / `image` | `vlm_text` only | OCR on figures is mostly noise |
| `table` | `ocr_text + "\n" + vlm_text` | OCR captures cell values; VLM captures structure |

This matters because VLM-generated text substantially outperforms OCR for semantic retrieval on visual content — a key finding from the MMDocIR paper.

### 3. Hierarchical Retrieval with Context Expansion

A flat top-k query over layout chunks alone often returns the map *or* the table, but not both when evidence is scattered. Retrieval runs in three steps:

1. **Coarse**: Query Tier 2 (pages) for top-10 candidate pages
2. **Fine**: Query Tier 1 (layouts) with a metadata filter scoping to those page IDs
3. **Context expansion**: For each retrieved layout chunk, fetch its `prev_chunk_id` and `next_chunk_id` siblings by ID — ensuring Claude has surrounding context without additional vector queries

The agent's `analyze_query` node can also route directly to Tier 1 (`layout_direct` strategy) for highly specific, single-chunk questions, skipping the coarse page pass.

### 4. Multimodal Generation with Original Images

Embeddings are text-only (`all-MiniLM-L6-v2` on VLM/OCR text), but at generation time the system passes the original JPEG image binary alongside the text description for any retrieved `image` or `table` chunk. The `image_binary` column is already present in the layouts parquet — no external image files need to be unpacked.

This gives Claude the ability to directly interpret charts, read table cell values, and reason about spatial relationships in figures, rather than relying solely on the pre-computed VLM description. The content blocks sent to Claude are interleaved: `[chunk_id] (table):` → image block → VLM text description.

### 5. LangGraph Orchestration with Agentic Retry

The LangGraph graph implements a conditional feedback loop:

```
analyze_query → retrieve → assess_sufficiency → generate_answer
                    ↑              |
                    └─ expand ─────┘ (if insufficient, once)
```

`analyze_query` (DSPy `Predict`) classifies the question modality, estimates whether it's multi-hop, selects retrieval strategy, and rewrites the query for better semantic search. `assess_sufficiency` (DSPy `Predict`) decides whether the retrieved context is enough to answer the question. If not, `expand_retrieval` broadens the query and forces hierarchical retrieval before a single retry. The retry is capped at 1 to avoid loops.

### 6. DSPy for Structured LM Calls

DSPy is used for the two text-only LM calls in the pipeline (`QueryAnalysis` and `SufficiencyCheck`) where typed input/output signatures enforce structured outputs (enums, booleans). Answer generation bypasses DSPy and calls the Anthropic SDK directly to support multimodal content blocks, since DSPy's multimodal support is less mature.

### 7. Embedding Without an External API

`ChromaDB`'s built-in `DefaultEmbeddingFunction` (backed by `all-MiniLM-L6-v2` via `onnxruntime`) was chosen to eliminate the need for a separate embedding API key, keeping the system self-contained. The model is downloaded once on first use and cached locally. For production, this should be replaced with a stronger embedding model (see Future Improvements).

---

## Results

Evaluated on 5 documents (one per domain) covering 29 questions from the MMDocIR benchmark.

| Metric | Score | Notes |
|---|---|---|
| **Page Recall@5** | **82.8%** | Ground-truth page found in top-5 retrieved pages |
| **Layout Recall@5** | **55.2%** | Ground-truth layout element overlaps retrieved chunk (IoU > 0.5) |
| **LLM Judge** | **62.1%** | Claude rates generated answer as semantically correct |
| **Exact Containment** | **44.8%** | Ground-truth string present in generated answer |
| Token F1 | 9.0% | Word-overlap F1 — low due to length mismatch (see below) |
| Fuzzy Match | 11.6% | Character-level SequenceMatcher ratio |

**On Token F1 and Fuzzy Match**: These numbers are misleadingly low. Ground-truth answers are short phrases (e.g. `"Less well-off"`, `"Some college or more"`) while generated answers are full explanatory sentences. Token F1 and character-level similarity penalize this length mismatch heavily even when the answer is semantically correct. **LLM Judge at 62.1% is the most meaningful answer quality indicator** and aligns with what MMDocRAG uses (they report answer quality scores and F1 on extractive spans, not full sentences).

**On Layout Recall**: The 55.2% layout recall reflects the genuine difficulty of the task — many questions require multi-layout or cross-page evidence, and our retrieval is constrained to top-5 results. Layout recall is also harder than page recall by design: finding the right page is a necessary but not sufficient condition for finding the right element within it.

---

## Future Improvements

### Retrieval Quality

- **Visual embeddings**: Replace text-only retrieval with a hybrid approach using CLIP or [ColPali](https://github.com/illuin-tech/colpali) for image embedding. The MMDocIR paper shows visual retrievers significantly outperform text retrievers on image-type questions; a hybrid retriever is the best overall strategy.
- **Stronger text embeddings**: Swap `all-MiniLM-L6-v2` for `voyage-3` or `voyage-multimodal-3` (Anthropic's embedding partner) for substantially better semantic retrieval, especially on technical and domain-specific content.
- **Cross-encoder re-ranking**: Add a re-ranking step after initial retrieval (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-score retrieved chunks before passing to the LLM. This significantly improves precision without changing the retrieval architecture.
- **Query decomposition**: For explicitly multi-hop questions, decompose the query into sub-questions, retrieve for each, and merge evidence before generation.
- **Larger context window usage**: Retrieve more candidates (top-20 or top-50) and let Claude select relevant chunks directly, trading latency for recall.

### Answer Quality

- **Extractive answer spans**: Post-process generated answers to extract the shortest answer span, which would make Token F1 a more meaningful metric and align with benchmark scoring conventions.
- **DSPy optimization**: Compile the `QueryAnalysis` and `SufficiencyCheck` modules with DSPy's `MIPROv2` optimizer using a small labeled set, improving prompt quality without manual tuning.
- **Structured answer output**: Enforce structured answer format (e.g. answer + confidence + supporting quotes) to make evaluation and downstream use more reliable.

### Scale and Coverage

- **Full dataset indexing**: Extend from 5 to all 313 documents. The architecture is unchanged; only `SUBSET_SIZE` in `src/config.py` needs updating. At full scale, expect ~170K layout chunks and ~20K page chunks — well within ChromaDB's capacity for a single-node deployment.
- **Incremental ingestion**: Add support for adding new documents to an existing index without full re-ingestion, using Chroma's `upsert` semantics.

---

## Path to Productionalization

Scaling this prototype to a production system serving 100,000+ documents requires changes at every layer of the stack.

### Storage and Indexing

Replace ChromaDB (local file-based) with a managed vector database: **Qdrant Cloud**, **Pinecone**, or **Weaviate**. These provide horizontal sharding, replication, filtered search at scale, and persistent storage decoupled from the application. At 100K documents (~3.4M layout chunks), a single-node vector DB becomes a bottleneck; managed services handle this transparently.

Raw document storage (PDFs, images) should live in object storage (S3 or GCS), with only the embeddings and metadata in the vector DB.

### Ingestion Pipeline

Replace the batch script with an event-driven pipeline:
1. Document uploaded to S3 → triggers an ingestion job (Lambda, Celery worker, or a dedicated service)
2. Document parsed (OCR + VLM extraction, or use Unstructured's own pipeline) → layout elements extracted
3. Chunks embedded and upserted to the vector DB
4. Document metadata registered in a relational store (Postgres) for filtering and audit

For documents not already pre-processed (unlike MMDocIR), this stage requires OCR and VLM inference at scale, which warrants a GPU-backed inference service or API (e.g. Unstructured's hosted API, Azure Document Intelligence, or a self-hosted VLM).

### Serving

Wrap the LangGraph graph in a **FastAPI** service:
- `POST /query` — takes a question and optional filters, returns answer + sources + retrieval metadata
- `POST /ingest` — triggers async ingestion for a new document
- `GET /health` — for load balancer health checks

Add a **caching layer** (Redis) keyed on (question embedding, metadata filters) to avoid redundant LLM calls for repeated or near-duplicate queries.

### Observability and Evaluation

- **Tracing**: Instrument with [LangSmith](https://smith.langchain.com/) or [Arize Phoenix](https://phoenix.arize.com/) to log every retrieval call, LLM prompt, and latency at the node level.
- **Continuous evaluation**: Run the evaluation harness on a held-out question set on every deployment. Alert on regression in page recall or LLM judge score.
- **Cost tracking**: Log token usage per query. At scale, sufficiency checking (an extra LLM call per question) and LLM-as-judge evaluation are the main cost drivers — both can be replaced with lighter models (Haiku) or removed if latency/cost is critical.

### Reliability and Safety

- **Async retrieval**: Run Tier 1 and Tier 2 queries concurrently rather than sequentially.
- **Timeouts and fallbacks**: Set per-node timeouts in LangGraph; fall back to Tier 1 direct retrieval if Tier 2 is slow.
- **Rate limiting and auth**: API key authentication and per-tenant rate limits if serving multiple users.
- **PII and content filtering**: Scan retrieved chunks and generated answers for sensitive content before returning to users, especially for legal, financial, or medical document domains.
