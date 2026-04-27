"""
Retrieval tool definitions and execution for the ReAct agent.

Five tools expose the index to Claude:
  search            — general vector search across all element types
  search_table      — scoped to table elements only
  search_figure     — scoped to image/figure elements only
  get_document_metadata — scoped to the per-document metadata chunk
  search_pages      — queries Tier 2 page-level VLM descriptions directly;
                      returns up to 15 pages, useful for aggregation questions
                      that require scanning many pages rather than finding one
                      specific element

Claude calls whichever combination it needs; each call goes through the full
hierarchical retrieval + cross-encoder rerank pipeline (except search_pages
which queries the pages collection directly).
"""

from __future__ import annotations

import base64
import io
from typing import Any

import chromadb
from PIL import Image

from src.agent.image_store import image_store
from src.config import COLLECTION_PAGES
from src.embedder import get_embedder
from src.retrieval.retriever import _chroma_results_to_list, direct_layout_retrieve, retrieve

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "search",
        "description": (
            "Search the document for relevant passages. Use for questions about "
            "written content, explanations, definitions, or any text-based evidence. "
            "Also use when the relevant element type is unknown."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_table",
        "description": (
            "Search specifically for tables. Use for questions about structured data, "
            "comparisons, statistics, or numerical information presented in tabular form."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query for table content"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_figure",
        "description": (
            "Search specifically for figures, charts, diagrams, and images. Use for "
            "visual questions about graphs, illustrations, colours, or spatial relationships."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query for visual content"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_pages",
        "description": (
            "Search across full-page descriptions rather than individual elements. "
            "Returns up to 15 pages, each with a holistic VLM description of everything "
            "on that page. Use this for aggregation questions that require scanning many "
            "pages (e.g. 'which pages mention X?', 'how many chapters discuss Y?', "
            "'which sections contain Z across the document?'). Prefer the other tools "
            "for single-element questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_document_metadata",
        "description": (
            "Get document-level information: title, authors, publication date, organisation, "
            "and a high-level overview of the document. Use for questions about who wrote the "
            "document, when it was published, what organisation produced it, or what the "
            "document is generally about."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What document-level information you need"}
            },
            "required": ["query"],
        },
    },
]

_TOOL_MODALITY: dict[str, str] = {
    "search": "",
    "search_table": "table",
    "search_figure": "figure",
    "get_document_metadata": "metadata",
}

_SEARCH_PAGES_TOP_K = 15  # broader than layout tools to support aggregation


_MAX_IMAGE_DIM = 1568  # Anthropic's safe limit for multi-image requests


def _resize_image(image_bytes: bytes) -> bytes:
    """Resize image to fit within _MAX_IMAGE_DIM on any dimension, preserving aspect ratio."""
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail((_MAX_IMAGE_DIM, _MAX_IMAGE_DIM), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _chunks_to_content_blocks(chunks: list[dict]) -> list[dict]:
    """Convert retrieved chunks into Anthropic tool-result content blocks."""
    if not chunks:
        return [{"type": "text", "text": "No relevant content found."}]

    blocks: list[dict] = []
    for chunk in chunks:
        cid = chunk.get("chunk_id", "unknown")
        el_type = chunk.get("element_type", "")
        page_id = chunk.get("page_id", "?")
        text = chunk.get("document") or ""

        blocks.append({
            "type": "text",
            "text": f"[{cid}] ({el_type}, page {page_id}):\n{text}",
        })

        image_bytes = image_store.get_for_chunk(chunk)
        if image_bytes:
            image_bytes = _resize_image(image_bytes)
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(image_bytes).decode("utf-8"),
                },
            })
    return blocks


def execute_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    chroma: chromadb.ClientAPI,
) -> tuple[list[dict], list[dict]]:
    """
    Execute a named tool call and return (raw_chunks, content_blocks).

    raw_chunks: chunk dicts accumulated for retrieved_chunks tracking.
    content_blocks: formatted Anthropic tool-result content blocks.
    """
    query = tool_input.get("query", "")
    modality = _TOOL_MODALITY.get(tool_name, "")

    if tool_name == "get_document_metadata":
        # Skip the coarse page pass — metadata chunks aren't page-specific.
        chunks = direct_layout_retrieve(query, chroma, modality="metadata")

    elif tool_name == "search_pages":
        # Query Tier 2 (page-level VLM descriptions) directly, returning more
        # results than layout tools to support aggregation across many pages.
        pages_col = chroma.get_collection(COLLECTION_PAGES)
        query_emb = get_embedder().embed_query(query)
        results = pages_col.query(
            query_embeddings=[query_emb],
            n_results=min(_SEARCH_PAGES_TOP_K, pages_col.count()),
            include=["metadatas", "documents", "distances"],
        )
        chunks = _chroma_results_to_list(results)

    else:
        chunks, _ = retrieve(query, chroma, strategy="hierarchical", modality=modality)

    return chunks, _chunks_to_content_blocks(chunks)
