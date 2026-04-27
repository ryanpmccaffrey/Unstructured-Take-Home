import base64
import io
from typing import Literal

import dspy
from PIL import Image as PILImage

from src.agent.image_store import image_store
from src.config import ANTHROPIC_API_KEY, CLAUDE_MODEL

lm = dspy.LM(
    model=f"anthropic/{CLAUDE_MODEL}",
    api_key=ANTHROPIC_API_KEY,
    max_tokens=1024,
)
dspy.configure(lm=lm)

_MAX_IMAGE_DIM = 1568


def _chunk_images(chunks: list[dict]) -> list[dspy.Image]:
    """Return dspy.Image objects for all figure/table chunks that have image binaries."""
    images = []
    for chunk in chunks:
        raw = image_store.get_for_chunk(chunk)
        if raw is None:
            continue
        img = PILImage.open(io.BytesIO(raw))
        img.thumbnail((_MAX_IMAGE_DIM, _MAX_IMAGE_DIM), PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
        images.append(dspy.Image(url=f"data:image/jpeg;base64,{b64}"))
    return images


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as a single context string with chunk IDs."""
    parts = []
    for c in chunks:
        cid = c.get("chunk_id", "unknown")
        el_type = c.get("element_type", "")
        page_id = c.get("page_id", "?")
        text = c.get("document") or ""
        parts.append(f"[{cid}] ({el_type}, page {page_id}):\n{text}")
    return "\n\n---\n\n".join(parts)


# ── Signatures ─────────────────────────────────────────────────────────────────

class QueryAnalysis(dspy.Signature):
    """
    Analyze a document retrieval question to produce an optimal retrieval plan.
    When retrying, adapt the plan based on why the previous attempt was insufficient.
    """
    question: str = dspy.InputField()
    insufficiency_reason: str = dspy.InputField(
        desc="Why previous retrieval failed; empty string on first attempt."
    )
    modality: Literal["text", "table", "figure", "mixed"] = dspy.OutputField(
        desc="Primary evidence modality needed to answer the question."
    )
    granularity: Literal["layout", "page"] = dspy.OutputField(
        desc="'layout' for specific element questions; 'page' for broad or thematic questions."
    )
    rewritten_query: str = dspy.OutputField(
        desc="Optimized search query for semantic retrieval."
    )
    top_k: int = dspy.OutputField(
        desc="Number of final chunks to return after reranking; between 5 and 20."
    )


class SufficiencyCheck(dspy.Signature):
    """
    Assess whether retrieved context is sufficient to answer the question.
    Be conservative: if key evidence is missing or context is too vague, mark insufficient.
    """
    question: str = dspy.InputField()
    context: str = dspy.InputField(desc="Retrieved chunks with [chunk_id] prefixes.")
    is_sufficient: bool = dspy.OutputField(
        desc="True only if the context contains enough evidence to answer the question."
    )
    insufficiency_reason: str = dspy.OutputField(
        desc="If insufficient: specific reason such as 'wrong modality', 'need page-level context', "
             "or 'query too narrow'. Empty if sufficient."
    )


class AnswerGeneration(dspy.Signature):
    """
    Generate a concise, cited answer from retrieved document evidence.
    On re-generation, address the validation feedback provided.
    """
    question: str = dspy.InputField()
    context: str = dspy.InputField(
        desc="Retrieved chunks with [chunk_id] prefixes and element types."
    )
    images: list[dspy.Image] = dspy.InputField(
        desc="Visual content from figure and table chunks; may be empty."
    )
    validation_feedback: str = dspy.InputField(
        desc="Feedback from a failed prior validation attempt; empty string on first attempt."
    )
    answer: str = dspy.OutputField(desc="Concise direct answer to the question.")
    cited_chunk_ids: list[str] = dspy.OutputField(
        desc="List of chunk IDs from context that directly support the answer."
    )


class AnswerValidation(dspy.Signature):
    """
    Validate whether the generated answer is grounded in the retrieved evidence.
    Flag hallucinations or claims not supported by the provided context.
    """
    question: str = dspy.InputField()
    answer: str = dspy.InputField()
    context: str = dspy.InputField(desc="Retrieved chunks with [chunk_id] prefixes.")
    images: list[dspy.Image] = dspy.InputField(
        desc="Visual content from figure and table chunks; may be empty."
    )
    is_valid: bool = dspy.OutputField(
        desc="True if the answer is grounded in the evidence with no hallucinations."
    )
    feedback: str = dspy.OutputField(
        desc="If invalid: specific feedback on what is wrong or ungrounded. Empty if valid."
    )


# ── Configured predictors ──────────────────────────────────────────────────────
analyze_query = dspy.Predict(QueryAnalysis)
check_sufficiency = dspy.Predict(SufficiencyCheck)
generate_answer = dspy.Predict(AnswerGeneration)
validate_answer = dspy.Predict(AnswerValidation)
