from __future__ import annotations

import base64

import anthropic
import chromadb

from src.agent.dspy_modules import analyze_query, check_sufficiency
from src.agent.image_store import image_store
from src.agent.state import RAGState
from src.config import ANTHROPIC_API_KEY, CHROMA_PATH, CLAUDE_MODEL
from src.retrieval.retriever import retrieve

_chroma: chromadb.ClientAPI | None = None
_anthropic: anthropic.Anthropic | None = None


def _get_chroma() -> chromadb.ClientAPI:
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma


def _get_anthropic() -> anthropic.Anthropic:
    global _anthropic
    if _anthropic is None:
        _anthropic = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        cid = c.get("chunk_id", "unknown")
        text = c.get("document") or c.get("text_for_generation") or ""
        el_type = c.get("element_type", "")
        parts.append(f"[{cid}] ({el_type})\n{text}")
    return "\n\n---\n\n".join(parts)


def _build_multimodal_content(question: str, chunks: list[dict]) -> list[dict]:
    """
    Build Anthropic message content blocks. For image/table chunks, attach
    the original image alongside the text description so Claude can reason
    directly from the visual. Text chunks are passed as plain text blocks.
    """
    content: list[dict] = [
        {"type": "text", "text": f"Answer the following question using only the provided context. "
                                  f"Cite the chunk IDs (e.g. [chunk_id]) that support your answer.\n\n"
                                  f"Question: {question}\n\nContext:"}
    ]

    for chunk in chunks:
        cid = chunk.get("chunk_id", "unknown")
        text = chunk.get("document") or chunk.get("text_for_generation") or ""
        el_type = chunk.get("element_type", "")

        content.append({"type": "text", "text": f"\n\n[{cid}] ({el_type}):"})

        image_bytes = image_store.get_for_chunk(chunk)
        if image_bytes:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(image_bytes).decode("utf-8"),
                },
            })

        if text:
            content.append({"type": "text", "text": text})

    content.append({
        "type": "text",
        "text": "\n\nProvide a concise answer and list the cited chunk IDs on a final line "
                "prefixed with 'Cited:' (comma-separated)."
    })
    return content


def _parse_cited_chunks(raw: str) -> tuple[str, list[str]]:
    """Split the model response into answer text and cited chunk IDs."""
    lines = raw.strip().splitlines()
    cited: list[str] = []
    answer_lines: list[str] = []
    for line in lines:
        if line.strip().lower().startswith("cited:"):
            cited = [c.strip() for c in line.split(":", 1)[1].split(",") if c.strip()]
        else:
            answer_lines.append(line)
    return "\n".join(answer_lines).strip(), cited


def node_analyze_query(state: RAGState) -> dict:
    result = analyze_query(question=state["question"])
    return {
        "modality": result.modality,
        "is_multi_hop": result.is_multi_hop,
        "retrieval_strategy": result.retrieval_strategy,
        "rewritten_query": result.rewritten_query,
    }


def node_retrieve(state: RAGState) -> dict:
    chroma = _get_chroma()
    query = state.get("rewritten_query") or state["question"]
    strategy = state.get("retrieval_strategy") or "hierarchical"

    if state.get("retry_count", 0) > 0:
        strategy = "hierarchical"

    chunks, page_ids = retrieve(query, chroma, strategy=strategy)
    return {
        "retrieved_chunks": chunks,
        "candidate_page_ids": page_ids,
    }


def node_assess_sufficiency(state: RAGState) -> dict:
    context = _format_context(state["retrieved_chunks"])
    result = check_sufficiency(question=state["question"], context=context)
    return {"is_sufficient": result.is_sufficient}


def node_generate_answer(state: RAGState) -> dict:
    client = _get_anthropic()
    content = _build_multimodal_content(state["question"], state["retrieved_chunks"])
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )
    raw = response.content[0].text
    answer, cited = _parse_cited_chunks(raw)
    return {"answer": answer, "cited_chunk_ids": cited}


def node_expand_retrieval(state: RAGState) -> dict:
    broadened = f"{state['question']} {state.get('rewritten_query', '')}".strip()
    return {
        "rewritten_query": broadened,
        "retrieval_strategy": "hierarchical",
        "retry_count": state.get("retry_count", 0) + 1,
    }


def route_after_sufficiency(state: RAGState) -> str:
    if state.get("is_sufficient"):
        return "generate"
    if state.get("retry_count", 0) >= 1:
        return "generate"
    return "expand"
