"""
Hierarchical retrieval:
  Step 1 — Coarse: query Tier 2 (pages) for candidate page IDs
  Step 2 — Fine: query Tier 1 (layouts) filtered to those pages
  Step 3 — Context expansion: fetch prev/next siblings for each layout chunk

Direct layout retrieval skips Step 1 (used for low-ambiguity, single-chunk questions).

Embeddings are handled by Chroma's DefaultEmbeddingFunction (onnxruntime-backed
all-MiniLM-L6-v2) — no external embedding API needed.
"""

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from src.config import (
    CHROMA_PATH,
    COLLECTION_LAYOUTS,
    COLLECTION_PAGES,
    TOP_K_LAYOUTS,
    TOP_K_PAGES,
)

_ef = DefaultEmbeddingFunction()


def _get_chroma() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=CHROMA_PATH)


def _chroma_results_to_list(results: dict) -> list[dict]:
    chunks = []
    for i, chunk_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i]
        distance = results["distances"][0][i] if results.get("distances") else None
        chunks.append({"chunk_id": chunk_id, "document": doc, "distance": distance, **meta})
    return chunks


def _fetch_by_ids(collection: chromadb.Collection, ids: list[str]) -> list[dict]:
    if not ids:
        return []
    result = collection.get(ids=ids, include=["documents", "metadatas"])
    return [
        {"chunk_id": chunk_id, "document": result["documents"][i], **result["metadatas"][i]}
        for i, chunk_id in enumerate(result["ids"])
    ]


def _expand_context(collection: chromadb.Collection, chunks: list[dict]) -> list[dict]:
    sibling_ids: list[str] = []
    seen = {c["chunk_id"] for c in chunks}
    for c in chunks:
        for key in ("prev_chunk_id", "next_chunk_id"):
            sid = c.get(key) or ""
            if sid and sid not in seen:
                sibling_ids.append(sid)
                seen.add(sid)
    return chunks + _fetch_by_ids(collection, sibling_ids)


def hierarchical_retrieve(
    query: str,
    chroma: chromadb.ClientAPI,
    top_k_pages: int = TOP_K_PAGES,
    top_k_layouts: int = TOP_K_LAYOUTS,
    doc_filter: str | None = None,
) -> list[dict]:
    pages_col = chroma.get_collection(COLLECTION_PAGES, embedding_function=_ef)
    layouts_col = chroma.get_collection(COLLECTION_LAYOUTS, embedding_function=_ef)

    where_doc = {"doc_name": {"$eq": doc_filter}} if doc_filter else None

    page_results = pages_col.query(
        query_texts=[query],
        n_results=min(top_k_pages, pages_col.count()),
        where=where_doc,
        include=["metadatas", "documents", "distances"],
    )
    candidate_pages = _chroma_results_to_list(page_results)
    candidate_page_ids = [int(p["page_id"]) for p in candidate_pages]

    if candidate_page_ids:
        page_filter: dict = {"page_id": {"$in": candidate_page_ids}}
        if doc_filter:
            page_filter = {"$and": [{"doc_name": {"$eq": doc_filter}}, page_filter]}
        layout_results = layouts_col.query(
            query_texts=[query],
            n_results=min(top_k_layouts, layouts_col.count()),
            where=page_filter,
            include=["metadatas", "documents", "distances"],
        )
    else:
        layout_results = layouts_col.query(
            query_texts=[query],
            n_results=min(top_k_layouts, layouts_col.count()),
            where=where_doc,
            include=["metadatas", "documents", "distances"],
        )

    layout_chunks = _chroma_results_to_list(layout_results)
    return _expand_context(layouts_col, layout_chunks)


def direct_layout_retrieve(
    query: str,
    chroma: chromadb.ClientAPI,
    top_k: int = TOP_K_LAYOUTS,
    doc_filter: str | None = None,
) -> list[dict]:
    layouts_col = chroma.get_collection(COLLECTION_LAYOUTS, embedding_function=_ef)
    where = {"doc_name": {"$eq": doc_filter}} if doc_filter else None
    results = layouts_col.query(
        query_texts=[query],
        n_results=min(top_k, layouts_col.count()),
        where=where,
        include=["metadatas", "documents", "distances"],
    )
    chunks = _chroma_results_to_list(results)
    return _expand_context(layouts_col, chunks)


def retrieve(
    query: str,
    chroma: chromadb.ClientAPI,
    strategy: str = "hierarchical",
    doc_filter: str | None = None,
) -> tuple[list[dict], list[int]]:
    if strategy == "hierarchical":
        chunks = hierarchical_retrieve(query, chroma, doc_filter=doc_filter)
    else:
        chunks = direct_layout_retrieve(query, chroma, doc_filter=doc_filter)

    page_ids = sorted({int(c["page_id"]) for c in chunks if c.get("page_id") not in ("", None)})
    return chunks, page_ids
