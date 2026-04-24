from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class RAGState(TypedDict):
    question: str
    modality: str
    is_multi_hop: bool
    retrieval_strategy: str
    rewritten_query: str
    candidate_page_ids: list[int]
    retrieved_chunks: list[dict]
    is_sufficient: bool
    answer: str
    cited_chunk_ids: list[str]
    retry_count: int
