"""
Chunking pipeline. Produces Tier 1 (layout) and Tier 2 (page) chunks.

Tier 1 algorithm per page:
  - Walk layout elements in layout_id order
  - Accumulate consecutive text/equation rows into a merged chunk until
    MERGE_TOKEN_THRESHOLD is reached or a non-text element is encountered
  - Figures and tables are always standalone chunks
  - Each chunk gets: prev/next pointers, section_heading, page_position
  - Bridge chunks are emitted for every consecutive page pair

Tier 2: one chunk per page using pages_df.vlm_text (holistic VLM description).
"""

import json
from dataclasses import dataclass, field

import pandas as pd
import tiktoken

from src.config import BRIDGE_OVERLAP_TOKENS, HEADING_MAX_TOKENS, MERGE_TOKEN_THRESHOLD

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text or ""))


def _first_n_tokens(text: str, n: int) -> str:
    tokens = _enc.encode(text or "")
    return _enc.decode(tokens[:n])


def _last_n_tokens(text: str, n: int) -> str:
    tokens = _enc.encode(text or "")
    return _enc.decode(tokens[-n:])


def _page_position(bbox: list[float], page_size: list[float]) -> str:
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
    text_for_generation: str
    image_path: str | None
    page_image_path: str | None
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    page_position: str = "unknown"
    section_heading: str = ""
    # bridge-specific
    page_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "domain": self.domain,
            "page_id": self.page_id,
            "layout_ids": json.dumps(self.layout_ids),
            "element_type": self.element_type,
            "text_for_embedding": self.text_for_embedding,
            "text_for_generation": self.text_for_generation,
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
    prev_page_tail: str = ""  # last BRIDGE_OVERLAP_TOKENS of the previous page

    for _, page_row in doc_pages.iterrows():
        page_id = int(page_row["passage_id"])
        page_image_path = str(page_row.get("image_path") or "")
        page_layouts = doc_layouts[doc_layouts["page_id"] == page_id].sort_values(
            "layout_id"
        )

        page_chunks: list[Chunk] = []
        chunk_counter = 0

        # ── accumulator for merging consecutive text/equation elements ──
        acc_texts: list[str] = []
        acc_layout_ids: list[int] = []
        acc_bboxes: list[list[float]] = []
        acc_page_sizes: list[list[float]] = []
        last_heading: str = ""

        def flush_text_acc() -> None:
            nonlocal chunk_counter, last_heading
            if not acc_texts:
                return
            merged = "\n".join(acc_texts)
            first_bbox = acc_bboxes[0] if acc_bboxes else []
            first_ps = acc_page_sizes[0] if acc_page_sizes else []
            cid = f"{doc_name}_page{page_id}_chunk{chunk_counter}"
            chunk_counter += 1
            page_chunks.append(
                Chunk(
                    chunk_id=cid,
                    doc_name=doc_name,
                    domain=domain,
                    page_id=page_id,
                    layout_ids=list(acc_layout_ids),
                    element_type="text",
                    text_for_embedding=merged,
                    text_for_generation=merged,
                    image_path=None,
                    page_image_path=page_image_path,
                    page_position=_page_position(first_bbox, first_ps),
                    section_heading=last_heading,
                )
            )
            acc_texts.clear()
            acc_layout_ids.clear()
            acc_bboxes.clear()
            acc_page_sizes.clear()

        for _, el in page_layouts.iterrows():
            el_type = el.get("type", "text")
            text = _text_for_element(el)
            bbox_raw = el.get("bbox")
            bbox = list(bbox_raw) if bbox_raw is not None else []
            ps_raw = el.get("page_size")
            page_size = list(ps_raw) if ps_raw is not None else []
            layout_id = int(el["layout_id"])

            if el_type in ("text", "equation"):
                if _is_heading(text):
                    last_heading = text.strip()

                acc_texts.append(text)
                acc_layout_ids.append(layout_id)
                acc_bboxes.append(bbox)
                acc_page_sizes.append(page_size)

                if _count_tokens("\n".join(acc_texts)) >= MERGE_TOKEN_THRESHOLD:
                    flush_text_acc()
            else:
                # flush pending text before emitting non-text chunk
                flush_text_acc()

                img_path = str(el.get("image_path") or "")
                cid = f"{doc_name}_page{page_id}_chunk{chunk_counter}"
                chunk_counter += 1
                page_chunks.append(
                    Chunk(
                        chunk_id=cid,
                        doc_name=doc_name,
                        domain=domain,
                        page_id=page_id,
                        layout_ids=[layout_id],
                        element_type=el_type,
                        text_for_embedding=text,
                        text_for_generation=text,
                        image_path=img_path if img_path else None,
                        page_image_path=page_image_path,
                        page_position=_page_position(bbox, page_size),
                        section_heading=last_heading,
                    )
                )

        flush_text_acc()

        # ── assign prev/next pointers ──
        for i, c in enumerate(page_chunks):
            c.prev_chunk_id = page_chunks[i - 1].chunk_id if i > 0 else None
            c.next_chunk_id = (
                page_chunks[i + 1].chunk_id if i < len(page_chunks) - 1 else None
            )

        # ── bridge chunk with previous page ──
        if prev_page_tail and page_chunks:
            page_head = _first_n_tokens(
                page_chunks[0].text_for_embedding, BRIDGE_OVERLAP_TOKENS
            )
            bridge_text = prev_page_tail + " " + page_head
            prev_page_id = page_id - 1
            bridge_chunk = Chunk(
                chunk_id=f"{doc_name}_bridge_{prev_page_id}-{page_id}",
                doc_name=doc_name,
                domain=domain,
                page_id=page_id,
                layout_ids=[],
                element_type="bridge",
                text_for_embedding=bridge_text,
                text_for_generation=bridge_text,
                image_path=None,
                page_image_path=page_image_path,
                page_ids=[prev_page_id, page_id],
            )
            all_tier1.append(bridge_chunk)

        # update tail for next page's bridge
        if page_chunks:
            prev_page_tail = _last_n_tokens(
                page_chunks[-1].text_for_embedding, BRIDGE_OVERLAP_TOKENS
            )
        else:
            prev_page_tail = ""

        all_tier1.extend(page_chunks)

        # ── Tier 2 page chunk ──
        vlm_text = str(page_row.get("vlm_text") or page_row.get("ocr_text") or "")
        tier2_chunks.append(
            {
                "chunk_id": f"{doc_name}_page_{page_id}",
                "doc_name": doc_name,
                "domain": domain,
                "page_id": page_id,
                "text_for_embedding": vlm_text,
                "image_path": page_image_path,
                "child_chunk_ids": json.dumps([c.chunk_id for c in page_chunks]),
            }
        )

    return all_tier1, tier2_chunks
