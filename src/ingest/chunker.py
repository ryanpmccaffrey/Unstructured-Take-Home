"""
Chunking pipeline. Produces Tier 1 (layout) and Tier 2 (page) chunks.

Tier 1 algorithm — one chunk per layout element:
  - Each layout element (paragraph, figure, table, equation) becomes its own chunk.
    The document parser has already segmented at paragraph granularity; merging
    would lower embedding precision and hurt IoU-based layout recall.
  - section_heading tracks the most recent short (≤ HEADING_MAX_TOKENS) text element
    seen on the page — stored as metadata on subsequent chunks, not merged into them.
  - prev/next pointers form a linked list within each page for context expansion.
  - Bridge chunks are emitted between consecutive pages using the full text of the
    last prose element on page N and the full text of the first prose element on
    page N+1. Whole elements are used rather than fixed token slices so that bridge
    text is always semantically complete (no mid-sentence cuts).
  - One metadata chunk per document (element_type="metadata") synthesised from the
    first page's VLM description — captures title, author, and document-level context
    that metadata questions need.

Tier 2: one chunk per page using pages_df.vlm_text (holistic VLM description).
"""

import json
from dataclasses import dataclass, field

import pandas as pd
import tiktoken

from src.config import HEADING_MAX_TOKENS

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Return the cl100k token count for the given text."""
    return len(_enc.encode(text or ""))


def _page_position(bbox: list[float], page_size: list[float]) -> str:
    """Classify vertical position of a bbox as top / middle / bottom within its page."""
    if not bbox or not page_size:
        return "unknown"
    y_center = (bbox[1] + bbox[3]) / 2
    height = page_size[1]
    if y_center < height / 3:
        return "top"
    elif y_center < 2 * height / 3:
        return "middle"
    return "bottom"


def _text_for_element(row: pd.Series) -> str:
    """Best text representation for a layout element, by type."""
    element_type = row.get("type", "text")
    if element_type in ("text", "equation"):
        return row.get("text") or row.get("ocr_text") or ""
    elif element_type == "table":
        ocr = row.get("ocr_text") or ""
        vlm = row.get("vlm_text") or ""
        return (ocr + "\n" + vlm).strip()
    else:  # figure / image
        return row.get("vlm_text") or row.get("ocr_text") or ""


def _is_heading(text: str) -> bool:
    """Return True if text is short enough to be treated as a section heading."""
    return 0 < _count_tokens(text) <= HEADING_MAX_TOKENS


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    domain: str
    page_id: int
    layout_ids: list[int]
    element_type: str
    text_for_embedding: str
    image_path: str | None
    page_image_path: str | None
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    page_position: str = "unknown"
    section_heading: str = ""
    # bridge-specific
    page_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a flat dict suitable for ChromaDB metadata storage."""
        return {
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "domain": self.domain,
            "page_id": self.page_id,
            "layout_ids": json.dumps(self.layout_ids),
            "element_type": self.element_type,
            "text_for_embedding": self.text_for_embedding,
            "image_path": self.image_path or "",
            "page_image_path": self.page_image_path or "",
            "prev_chunk_id": self.prev_chunk_id or "",
            "next_chunk_id": self.next_chunk_id or "",
            "page_position": self.page_position,
            "section_heading": self.section_heading,
            "page_ids": json.dumps(self.page_ids),
        }


def chunk_document(
    doc_meta: dict,
    pages_df: pd.DataFrame,
    layouts_df: pd.DataFrame,
) -> tuple[list[Chunk], list[dict]]:
    """
    Returns (tier1_chunks, tier2_page_chunks).
    tier2_page_chunks are plain dicts (used by indexer directly).
    """
    doc_name = doc_meta["doc_name"]
    domain = doc_meta["domain"]

    doc_layouts = layouts_df[layouts_df["doc_name"] == doc_name].copy()
    doc_pages = pages_df[pages_df["doc_name"] == doc_name].copy()

    all_tier1: list[Chunk] = []
    tier2_chunks: list[dict] = []

    # ── Metadata chunk (one per document) ──────────────────────────────────────
    # Synthesised from the first page's VLM description. Captures title, authors,
    # and document-level context that metadata questions need but layout elements
    # don't surface.
    first_page = doc_pages.iloc[0] if len(doc_pages) > 0 else None
    if first_page is not None:
        first_vlm = str(first_page.get("vlm_text") or first_page.get("ocr_text") or "")
        meta_text = f"Document: {doc_name}\nDomain: {domain}\n\n{first_vlm}"
        all_tier1.append(Chunk(
            chunk_id=f"{doc_name}_metadata",
            doc_name=doc_name,
            domain=domain,
            page_id=0,
            layout_ids=[],
            element_type="metadata",
            text_for_embedding=meta_text,
            image_path=str(first_page.get("image_path") or "") or None,
            page_image_path=str(first_page.get("image_path") or "") or None,
        ))

    prev_page_prose_tail: str = ""  # last BRIDGE_OVERLAP_TOKENS of last prose on previous page

    for _, page_row in doc_pages.iterrows():
        page_id = int(page_row["passage_id"])
        page_image_path = str(page_row.get("image_path") or "")
        page_layouts = doc_layouts[doc_layouts["page_id"] == page_id].sort_values("layout_id")

        page_chunks: list[Chunk] = []
        last_heading: str = ""

        for _, el in page_layouts.iterrows():
            el_type = el.get("type", "text")
            text = _text_for_element(el)
            bbox_raw = el.get("bbox")
            bbox = list(bbox_raw) if bbox_raw is not None else []
            ps_raw = el.get("page_size")
            page_size = list(ps_raw) if ps_raw is not None else []
            layout_id = int(el["layout_id"])

            if el_type in ("text", "equation") and _is_heading(text):
                last_heading = text.strip()

            img_path = str(el.get("image_path") or "") or None
            if el_type in ("text", "equation"):
                img_path = None  # text elements have no meaningful image

            page_chunks.append(Chunk(
                chunk_id=f"{doc_name}_page{page_id}_layout{layout_id}",
                doc_name=doc_name,
                domain=domain,
                page_id=page_id,
                layout_ids=[layout_id],
                element_type=el_type,
                text_for_embedding=text,
                image_path=img_path,
                page_image_path=page_image_path,
                page_position=_page_position(bbox, page_size),
                section_heading=last_heading,
            ))

        # ── assign prev/next pointers ──
        for i, c in enumerate(page_chunks):
            c.prev_chunk_id = page_chunks[i - 1].chunk_id if i > 0 else None
            c.next_chunk_id = page_chunks[i + 1].chunk_id if i < len(page_chunks) - 1 else None

        # ── bridge chunk with previous page ──
        # Use whole prose elements (text/equation) rather than fixed token slices so
        # bridge text is always a complete semantic unit with no mid-sentence cuts.
        # figure/table VLM descriptions are excluded — they don't carry narrative
        # continuity across page breaks.
        prose_chunks = [c for c in page_chunks if c.element_type in ("text", "equation")]
        if prev_page_prose_tail and prose_chunks:
            bridge_text = prev_page_prose_tail + "\n" + prose_chunks[0].text_for_embedding
            prev_page_id = page_id - 1
            all_tier1.append(Chunk(
                chunk_id=f"{doc_name}_bridge_{prev_page_id}-{page_id}",
                doc_name=doc_name,
                domain=domain,
                page_id=page_id,
                layout_ids=[],
                element_type="bridge",
                text_for_embedding=bridge_text,
                image_path=None,
                page_image_path=page_image_path,
                page_ids=[prev_page_id, page_id],
            ))

        # store full last prose element for the next page's bridge
        prev_page_prose_tail = prose_chunks[-1].text_for_embedding if prose_chunks else ""

        all_tier1.extend(page_chunks)

        # ── Tier 2 page chunk ──
        vlm_text = str(page_row.get("vlm_text") or page_row.get("ocr_text") or "")
        tier2_chunks.append({
            "chunk_id": f"{doc_name}_page_{page_id}",
            "doc_name": doc_name,
            "domain": domain,
            "page_id": page_id,
            "text_for_embedding": vlm_text,
            "image_path": page_image_path,
            "child_chunk_ids": json.dumps([c.chunk_id for c in page_chunks]),
        })

    return all_tier1, tier2_chunks
