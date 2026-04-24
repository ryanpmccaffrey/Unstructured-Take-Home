from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    node_analyze_query,
    node_assess_sufficiency,
    node_expand_retrieval,
    node_generate_answer,
    node_retrieve,
    route_after_sufficiency,
)
from src.agent.state import RAGState

_builder = StateGraph(RAGState)

_builder.add_node("analyze_query", node_analyze_query)
_builder.add_node("retrieve", node_retrieve)
_builder.add_node("assess_sufficiency", node_assess_sufficiency)
_builder.add_node("generate_answer", node_generate_answer)
_builder.add_node("expand_retrieval", node_expand_retrieval)

_builder.add_edge(START, "analyze_query")
_builder.add_edge("analyze_query", "retrieve")
_builder.add_edge("retrieve", "assess_sufficiency")
_builder.add_conditional_edges(
    "assess_sufficiency",
    route_after_sufficiency,
    {"generate": "generate_answer", "expand": "expand_retrieval"},
)
_builder.add_edge("expand_retrieval", "retrieve")
_builder.add_edge("generate_answer", END)

graph = _builder.compile()
