import operator
from typing import Annotated

from typing_extensions import TypedDict


def _safe_add(left: list | None, right: list | None) -> list:
    """Append right to left, deduplicating by chunk_id."""
    existing = left or []
    seen = {c["chunk_id"] for c in existing}
    return existing + [c for c in (right or []) if c["chunk_id"] not in seen]


class RAGState(TypedDict):
    question: str

    # ── Shared outputs ─────────────────────────────────────────────────────────
    answer: str
    cited_chunk_ids: list[str]
    # Annotated reducer: each node returns a list slice; state accumulates the
    # deduplicated union across all retrieval attempts / tool calls.
    retrieved_chunks: Annotated[list[dict], _safe_add]

    # ── Planner-Executor fields (graph.py) ────────────────────────────────────
    modality: str        # 'text', 'table', 'figure', 'mixed'
    granularity: str     # 'layout' or 'page'
    rewritten_query: str
    top_k: int
    candidate_chunks: list[dict]   # raw embedding results (overwritten each attempt)
    reranked_chunks: list[dict]    # post-rerank + context-expanded (overwritten each attempt)
    is_sufficient: bool
    insufficiency_reason: str
    retry_count: int
    is_validated: bool
    validation_feedback: str
    validation_attempts: int

    # ── ReAct fields (graph_react.py) ─────────────────────────────────────────
    # Annotated reducer: each node returns a list of new message dicts;
    # operator.add concatenates them to build the full conversation history.
    messages: Annotated[list[dict], operator.add]
