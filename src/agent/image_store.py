"""
Lazy-loaded lookup for layout image binaries.

Loads only image/table rows from the layouts parquet (not text/equation),
keyed by (doc_name, layout_id). Built once on first access.
"""

from __future__ import annotations

import json

import pandas as pd

from src.config import LAYOUTS_PATH


class LayoutImageStore:
    def __init__(self) -> None:
        """Initialise with an empty store; data is loaded lazily on first access."""
        self._store: dict[tuple[str, int], bytes] | None = None

    def _load(self) -> None:
        """Load all image/table image_binary rows from the layouts parquet into memory."""
        df = pd.read_parquet(
            LAYOUTS_PATH,
            columns=["doc_name", "layout_id", "type", "image_binary"],
        )
        df = df[df["type"].isin(("image", "table"))]
        self._store = {
            (row["doc_name"], int(row["layout_id"])): row["image_binary"]
            for _, row in df.iterrows()
        }

    def get(self, doc_name: str, layout_id: int) -> bytes | None:
        """Return image bytes for a specific (doc_name, layout_id) pair, or None."""
        if self._store is None:
            self._load()
        return self._store.get((doc_name, layout_id))

    def get_for_chunk(self, chunk: dict) -> bytes | None:
        """Return image bytes for a chunk if it's an image/table element."""
        if chunk.get("element_type") not in ("image", "table"):
            return None
        doc_name = chunk.get("doc_name", "")
        layout_ids_raw = chunk.get("layout_ids", "[]")
        try:
            layout_ids = json.loads(layout_ids_raw) if isinstance(layout_ids_raw, str) else layout_ids_raw
        except (json.JSONDecodeError, TypeError):
            return None
        if not layout_ids:
            return None
        return self.get(doc_name, int(layout_ids[0]))


image_store = LayoutImageStore()
