"""
ONNX-based BGE embedder — no torch or transformers required.

Uses the quantized Xenova/bge-base-en-v1.5 ONNX weights with onnxruntime
and the HuggingFace tokenizers library for tokenization.

BGE models use CLS-token pooling + L2 normalization for sentence embeddings.
For asymmetric retrieval (query vs document), BGE recommends prepending an
instruction to queries: "Represent this sentence for searching relevant passages: "
"""

from __future__ import annotations

import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from src.config import BGE_ONNX_DIR

_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
_BATCH_SIZE = 64


class BGEEmbedder:
    def __init__(self, onnx_dir: str = BGE_ONNX_DIR) -> None:
        """Load tokenizer and ONNX inference session from the cache directory."""
        tok_path = os.path.join(onnx_dir, "tokenizer.json")
        model_path = os.path.join(onnx_dir, "onnx", "model_quantized.onnx")
        self._tok = Tokenizer.from_file(tok_path)
        self._tok.enable_padding(pad_token="[PAD]", pad_id=0)
        self._tok.enable_truncation(max_length=512)
        self._sess = ort.InferenceSession(model_path)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Run a single ONNX forward pass and return L2-normalised CLS embeddings."""
        encodings = self._tok.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)
        outputs = self._sess.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        cls_emb = outputs[0][:, 0, :]
        return cls_emb / np.linalg.norm(cls_emb, axis=1, keepdims=True)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts in batches; no query instruction prefix."""
        all_embs: list[np.ndarray] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            all_embs.append(self._embed_batch(batch))
        return np.vstack(all_embs).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query with the BGE instruction prefix for asymmetric retrieval."""
        instructed = _QUERY_INSTRUCTION + text
        return self._embed_batch([instructed])[0].tolist()


_embedder: BGEEmbedder | None = None


def get_embedder() -> BGEEmbedder:
    """Return the process-level BGEEmbedder singleton, initialising it on first call."""
    global _embedder
    if _embedder is None:
        _embedder = BGEEmbedder()
    return _embedder
