"""
Cross-encoder re-ranker using cross-encoder/ms-marco-MiniLM-L6-v2 (ONNX).

Unlike bi-encoders, a cross-encoder sees the query and document together:
  [CLS] query [SEP] document [SEP]
and outputs a single relevance logit. This is slower than a bi-encoder
(can't pre-compute document embeddings) but significantly more accurate
at distinguishing truly relevant chunks from near-miss retrievals.

Used after the initial bi-encoder retrieval to re-score and reorder candidates
before context expansion and answer generation.
"""

from __future__ import annotations

import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from src.config import RERANKER_ONNX_DIR


class CrossEncoderReranker:
    def __init__(self, onnx_dir: str = RERANKER_ONNX_DIR) -> None:
        """Load tokenizer and quantized ONNX cross-encoder from the cache directory."""
        tok_path = os.path.join(onnx_dir, "tokenizer.json")
        # Try AVX2-quantized first (ms-marco-MiniLM), fall back to standard quantized (BGE)
        avx2_path = os.path.join(onnx_dir, "onnx", "model_quint8_avx2.onnx")
        default_path = os.path.join(onnx_dir, "onnx", "model_quantized.onnx")
        model_path = avx2_path if os.path.exists(avx2_path) else default_path
        self._tok = Tokenizer.from_file(tok_path)
        self._tok.enable_truncation(max_length=512)
        self._tok.enable_padding(pad_token="[PAD]", pad_id=0)
        self._sess = ort.InferenceSession(model_path)
        self._input_names = {inp.name for inp in self._sess.get_inputs()}

    def _score(self, query: str, texts: list[str]) -> np.ndarray:
        """Return raw relevance logits for each (query, text) pair."""
        encodings = self._tok.encode_batch([(query, t) for t in texts])
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        # BERT-based models use token_type_ids; RoBERTa-based models (e.g. BGE-reranker) do not
        if "token_type_ids" in self._input_names:
            inputs["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
        outputs = self._sess.run(None, inputs)
        return outputs[0][:, 0]  # shape (batch,) — single relevance logit per pair

    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        """Score all chunks against the query and return the top_k by relevance."""
        if not chunks:
            return chunks
        texts = [c.get("document") or c.get("text_for_embedding") or "" for c in chunks]
        scores = self._score(query, texts)
        ranked = sorted(zip(scores.tolist(), chunks), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_k]]


_reranker: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker:
    """Return the process-level CrossEncoderReranker singleton, initialising it on first call."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker
