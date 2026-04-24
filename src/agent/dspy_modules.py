from typing import Literal

import dspy

from src.config import ANTHROPIC_API_KEY, CLAUDE_MODEL

lm = dspy.LM(
    model=f"anthropic/{CLAUDE_MODEL}",
    api_key=ANTHROPIC_API_KEY,
    max_tokens=1024,
)
dspy.configure(lm=lm)


class QueryAnalysis(dspy.Signature):
    """
    Analyze a document retrieval question to plan the optimal retrieval strategy.
    retrieval_strategy should be 'hierarchical' when the question is broad or multi-hop,
    or 'layout_direct' when it's very specific and a single layout element is likely sufficient.
    """

    question: str = dspy.InputField()
    modality: Literal["text", "table", "figure", "mixed"] = dspy.OutputField(
        desc="The modality type of evidence needed to answer this question."
    )
    is_multi_hop: bool = dspy.OutputField(
        desc="True if answering requires evidence from multiple pages or multiple layout elements."
    )
    retrieval_strategy: Literal["hierarchical", "layout_direct"] = dspy.OutputField(
        desc="'hierarchical' for broad/multi-hop questions; 'layout_direct' for specific single-chunk questions."
    )
    rewritten_query: str = dspy.OutputField(
        desc="Expanded or clarified version of the question, optimized for semantic search."
    )


class SufficiencyCheck(dspy.Signature):
    """
    Assess whether the retrieved context is sufficient to answer the question.
    Be conservative: if key evidence is missing or the context is too vague, mark as insufficient.
    """

    question: str = dspy.InputField()
    context: str = dspy.InputField(
        desc="Concatenated text from retrieved document chunks."
    )
    is_sufficient: bool = dspy.OutputField(
        desc="True if the context contains enough information to answer the question."
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of why the context is or isn't sufficient."
    )



analyze_query = dspy.Predict(QueryAnalysis)
check_sufficiency = dspy.Predict(SufficiencyCheck)
# generate_answer is handled directly via Anthropic SDK in nodes.py
# to support multimodal content (image blocks for figure/table chunks)
