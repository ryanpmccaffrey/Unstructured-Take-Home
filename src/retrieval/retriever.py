"""
Hierarchical retrieval using BGE-base-en-v1.5 (ONNX) embeddings:
  Step 1 — Coarse: query Tier 2 (pages) for candidate page IDs
  Step 2 — Fine: query Tier 1 (layouts) with TOP_K_RETRIEVE candidates,
            filtered by element_type when modality is table or figure
  Step 3 — Re-rank: cross-encoder scores each (query, chunk) pair, keeps TOP_K_LAYOUTS
  Step 4 — Context expansion: fetch prev/next siblings for each surviving chunk
"""

import chromadb

from src.config import (
    CHROMA_PATH,
    COLLECTION_LAYOUTS,
    COLLECTION_PAGES,
    TOP_K_LAYOUTS,
    TOP_K_PAGES,
    TOP_K_RETRIEVE,
)
from src.embedder import get_embedder
from src.retrieval.reranker import get_reranker

# Maps analyze_query modality labels to element_type values in the layout collection.
# table/figure questions are routed to only that element type for higher precision.
# text and mixed questions use no filter — text evidence can appear in any element type.
_MODALITY_ELEMENT_FILTER: dict[str, str] = {
    "table": "table",
    "figure": "image",
    "metadata": "metadata",
}


def _get_chroma() -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client for the configured CHROMA_PATH."""
    return chromadb.PersistentClient(path=CHROMA_PATH)


def _chroma_results_to_list(results: dict) -> list[dict]:
    """Flatten a ChromaDB query result dict into a list of chunk dicts."""
    chunks = []
    for i, chunk_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i]
        distance = results["distances"][0][i] if results.get("distances") else None
        chunks.append({"chunk_id": chunk_id, "document": doc, "distance": distance, **meta})
    return chunks


def _fetch_by_ids(collection: chromadb.Collection, ids: list[str]) -> list[dict]:
    """Fetch chunks by explicit IDs without a vector query."""
    if not ids:
        return []
    result = collection.get(ids=ids, include=["documents", "metadatas"])
    return [
        {"chunk_id": chunk_id, "document": result["documents"][i], **result["metadatas"][i]}
        for i, chunk_id in enumerate(result["ids"])
    ]


def _expand_context(collection: chromadb.Collection, chunks: list[dict]) -> list[dict]:
    """Append prev/next linked-list siblings for each chunk to widen context."""
    sibling_ids: list[str] = []
    seen = {c["chunk_id"] for c in chunks}
    for c in chunks:
        for key in ("prev_chunk_id", "next_chunk_id"):
            sid = c.get(key) or ""
            if sid and sid not in seen:
                sibling_ids.append(sid)
                seen.add(sid)
    return chunks + _fetch_by_ids(collection, sibling_ids)


def _build_layout_filter(
    candidate_page_ids: list[int],
    doc_filter: str | None,
    modality: str,
) -> dict | None:
    """Build the ChromaDB where-clause combining page, doc, and element_type filters."""
    clauses: list[dict] = []
    if candidate_page_ids:
        clauses.append({"page_id": {"$in": candidate_page_ids}})
    if doc_filter:
        clauses.append({"doc_name": {"$eq": doc_filter}})
    element_type = _MODALITY_ELEMENT_FILTER.get(modality)
    if element_type:
        clauses.append({"element_type": {"$eq": element_type}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def hierarchical_retrieve(
    query: str,
    chroma: chromadb.ClientAPI,
    top_k_pages: int = TOP_K_PAGES,
    top_k_layouts: int = TOP_K_LAYOUTS,
    doc_filter: str | None = None,
    query_embedding: list[float] | None = None,
    modality: str = "",
) -> list[dict]:
    """Coarse page retrieval → fine layout retrieval → cross-encoder rerank → context expansion."""
    embedder = get_embedder()
    query_emb = query_embedding if query_embedding is not None else embedder.embed_query(query)

    pages_col = chroma.get_collection(COLLECTION_PAGES)
    layouts_col = chroma.get_collection(COLLECTION_LAYOUTS)

    where_doc = {"doc_name": {"$eq": doc_filter}} if doc_filter else None

    page_results = pages_col.query(
        query_embeddings=[query_emb],
        n_results=min(top_k_pages, pages_col.count()),
        where=where_doc,
        include=["metadatas", "documents", "distances"],
    )
    candidate_pages = _chroma_results_to_list(page_results)
    candidate_page_ids = [int(p["page_id"]) for p in candidate_pages]

    layout_filter = _build_layout_filter(candidate_page_ids, doc_filter, modality)
    n_candidates = min(TOP_K_RETRIEVE, layouts_col.count())

    layout_results = layouts_col.query(
        query_embeddings=[query_emb],
        n_results=n_candidates,
        where=layout_filter,
        include=["metadatas", "documents", "distances"],
    )

    layout_chunks = _chroma_results_to_list(layout_results)
    reranked = get_reranker().rerank(query, layout_chunks, top_k=top_k_layouts)
    return _expand_context(layouts_col, reranked)


def direct_layout_retrieve(
    query: str,
    chroma: chromadb.ClientAPI,
    top_k: int = TOP_K_LAYOUTS,
    doc_filter: str | None = None,
    query_embedding: list[float] | None = None,
    modality: str = "",
) -> list[dict]:
    """Query the layout collection directly without a coarse page pass, then rerank and expand."""
    embedder = get_embedder()
    query_emb = query_embedding if query_embedding is not None else embedder.embed_query(query)
    layouts_col = chroma.get_collection(COLLECTION_LAYOUTS)

    layout_filter = _build_layout_filter([], doc_filter, modality)
    results = layouts_col.query(
        query_embeddings=[query_emb],
        n_results=min(top_k, layouts_col.count()),
        where=layout_filter,
        include=["metadatas", "documents", "distances"],
    )
    chunks = _chroma_results_to_list(results)
    reranked = get_reranker().rerank(query, chunks, top_k=top_k)
    return _expand_context(layouts_col, reranked)


def retrieve(
    query: str,
    chroma: chromadb.ClientAPI,
    strategy: str = "hierarchical",
    doc_filter: str | None = None,
    query_embedding: list[float] | None = None,
    modality: str = "",
) -> tuple[list[dict], list[int]]:
    """Dispatch to hierarchical or direct retrieval and return (chunks, page_ids)."""
    if strategy == "hierarchical":
        chunks = hierarchical_retrieve(
            query, chroma, doc_filter=doc_filter,
            query_embedding=query_embedding, modality=modality,
        )
    else:
        chunks = direct_layout_retrieve(
            query, chroma, doc_filter=doc_filter,
            query_embedding=query_embedding, modality=modality,
        )

    page_ids = sorted({int(c["page_id"]) for c in chunks if c.get("page_id") not in ("", None)})
    return chunks, page_ids


def retrieve_candidates(
    query: str,
    chroma: chromadb.ClientAPI,
    granularity: str = "layout",
    modality: str = "",
    top_k: int = TOP_K_RETRIEVE,
    doc_filter: str | None = None,
) -> list[dict]:
    """
    Embed query and run vector search, returning raw candidates without reranking.

    granularity='layout': coarse page pass → fine layout pass (Tier 1 chunks)
    granularity='page':   direct Tier 2 page query (page-level VLM chunks)
    """
    embedder = get_embedder()
    query_emb = embedder.embed_query(query)

    if granularity == "page":
        pages_col = chroma.get_collection(COLLECTION_PAGES)
        where_doc = {"doc_name": {"$eq": doc_filter}} if doc_filter else None
        results = pages_col.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, pages_col.count()),
            where=where_doc,
            include=["metadatas", "documents", "distances"],
        )
        return _chroma_results_to_list(results)

    # layout granularity: coarse page pass → fine layout pass
    pages_col = chroma.get_collection(COLLECTION_PAGES)
    layouts_col = chroma.get_collection(COLLECTION_LAYOUTS)

    where_doc = {"doc_name": {"$eq": doc_filter}} if doc_filter else None
    page_results = pages_col.query(
        query_embeddings=[query_emb],
        n_results=min(TOP_K_PAGES, pages_col.count()),
        where=where_doc,
        include=["metadatas", "documents", "distances"],
    )
    candidate_page_ids = [int(p["page_id"]) for p in _chroma_results_to_list(page_results)]

    layout_filter = _build_layout_filter(candidate_page_ids, doc_filter, modality)
    results = layouts_col.query(
        query_embeddings=[query_emb],
        n_results=min(top_k, layouts_col.count()),
        where=layout_filter,
        include=["metadatas", "documents", "distances"],
    )
    return _chroma_results_to_list(results)
