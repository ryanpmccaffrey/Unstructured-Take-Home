"""
LangGraph node functions for the Planner-Executor RAG pipeline.

Nodes:
  node_query_analyzer    — DSPy: classify query, produce retrieval plan; adapts on retry
  node_retrieve          — vector search using the retrieval plan
  node_rerank            — cross-encoder rerank + context expansion (layout granularity only)
  node_sufficiency_check — DSPy: assess whether context is sufficient to answer
  node_generate          — DSPy: generate cited answer with multimodal evidence
  node_validate          — DSPy: verify answer is grounded in retrieved evidence

Routing:
  route_after_sufficiency — loops back to planner on failure (max 2 retries → 3 attempts total)
  route_after_validation  — loops back to generator on failure (max 1 retry → 2 attempts total)
"""

from __future__ import annotations

import chromadb

from src.agent.dspy_modules import (
    _chunk_images,
    _format_context,
    analyze_query,
    check_sufficiency,
    generate_answer,
    validate_answer,
)
from src.agent.state import RAGState
from src.config import CHROMA_PATH, COLLECTION_LAYOUTS, TOP_K_LAYOUTS
from src.retrieval.reranker import get_reranker
from src.retrieval.retriever import _expand_context, retrieve_candidates

_chroma_client: chromadb.ClientAPI | None = None


def _get_chroma() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma_client


def node_query_analyzer(state: RAGState) -> dict:
    """Classify query and produce retrieval plan; increments retry_count when replanning."""
    insufficiency_reason = state.get("insufficiency_reason") or ""
    result = analyze_query(
        question=state["question"],
        insufficiency_reason=insufficiency_reason,
    )
    # table/figure questions always need a specific layout element — force layout granularity
    # regardless of what the planner output; page granularity is only valid for broad
    # aggregation questions (text/mixed modality)
    granularity = result.granularity
    if result.modality in ("table", "figure"):
        granularity = "layout"
    return {
        "modality": result.modality,
        "granularity": granularity,
        "rewritten_query": result.rewritten_query,
        "top_k": max(5, min(20, int(result.top_k))),
        # Increment only on replanning (non-empty reason means a prior attempt failed)
        "retry_count": (state.get("retry_count") or 0) + (1 if insufficiency_reason else 0),
    }


def node_retrieve(state: RAGState) -> dict:
    """Run vector search using the retrieval plan; fetches 2× top_k for reranker headroom."""
    top_k = state.get("top_k") or TOP_K_LAYOUTS
    candidates = retrieve_candidates(
        query=state.get("rewritten_query") or state["question"],
        chroma=_get_chroma(),
        granularity=state.get("granularity") or "layout",
        modality=state.get("modality") or "",
        top_k=top_k * 2,
    )
    return {"candidate_chunks": candidates}


def node_rerank(state: RAGState) -> dict:
    """Cross-encoder rerank candidates; expand prev/next context for layout granularity."""
    candidates = state.get("candidate_chunks") or []
    query = state.get("rewritten_query") or state["question"]
    top_k = state.get("top_k") or TOP_K_LAYOUTS

    reranked = get_reranker().rerank(query, candidates, top_k=min(top_k, len(candidates)))

    if state.get("granularity") != "page":
        layouts_col = _get_chroma().get_collection(COLLECTION_LAYOUTS)
        reranked = _expand_context(layouts_col, reranked)

    return {
        "reranked_chunks": reranked,
        "retrieved_chunks": reranked,  # accumulated via _safe_add dedup reducer
    }


def node_sufficiency_check(state: RAGState) -> dict:
    """Assess whether reranked chunks are sufficient to answer the question."""
    result = check_sufficiency(
        question=state["question"],
        context=_format_context(state.get("reranked_chunks") or []),
    )
    return {
        "is_sufficient": result.is_sufficient,
        "insufficiency_reason": "" if result.is_sufficient else result.insufficiency_reason,
    }


def node_generate(state: RAGState) -> dict:
    """Generate a cited answer from reranked evidence; incorporates validation_feedback on retry."""
    chunks = state.get("reranked_chunks") or []
    result = generate_answer(
        question=state["question"],
        context=_format_context(chunks),
        images=_chunk_images(chunks),
        validation_feedback=state.get("validation_feedback") or "",
    )
    cited = result.cited_chunk_ids if isinstance(result.cited_chunk_ids, list) else []
    return {"answer": result.answer, "cited_chunk_ids": cited}


def node_validate(state: RAGState) -> dict:
    """Validate that the generated answer is grounded in the retrieved evidence."""
    chunks = state.get("reranked_chunks") or []
    result = validate_answer(
        question=state["question"],
        answer=state.get("answer") or "",
        context=_format_context(chunks),
        images=_chunk_images(chunks),
    )
    return {
        "is_validated": result.is_valid,
        "validation_feedback": "" if result.is_valid else result.feedback,
        "validation_attempts": (state.get("validation_attempts") or 0) + 1,
    }


def route_after_sufficiency(state: RAGState) -> str:
    """Loop to planner if insufficient (max 2 retries = 3 total attempts), else generate."""
    if state.get("is_sufficient"):
        return "generate"
    if (state.get("retry_count") or 0) >= 2:
        return "generate"
    return "replan"


def route_after_validation(state: RAGState) -> str:
    """Loop to generator if invalid (max 1 retry = 2 total attempts), else end."""
    if state.get("is_validated"):
        return "end"
    if (state.get("validation_attempts") or 0) >= 2:
        return "end"
    return "regenerate"
