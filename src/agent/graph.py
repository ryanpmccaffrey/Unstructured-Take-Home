"""
Planner-Executor agent entry point (default for make eval).

A lightweight DSPy planner classifies the query and produces a retrieval plan;
LangGraph executor nodes carry out that plan; a sufficiency check can loop back
to the planner on failure; a validator can loop back to the generator on
grounding failures.

Graph structure:

  START → query_analyzer → retrieve → rerank → sufficiency_check
              ↑ [replan: max 2 retries]               ↓ [generate]
              └──────────────────────────────────── generate → validate → END
                                                         ↑ [regenerate: max 1 retry]
                                                         └──────────────────────────┘

Alternative: see graph_react.py for the ReAct (tool-use loop) implementation.
Both expose the same graph.invoke(state) interface.
"""

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    node_generate,
    node_query_analyzer,
    node_rerank,
    node_retrieve,
    node_sufficiency_check,
    node_validate,
    route_after_sufficiency,
    route_after_validation,
)
from src.agent.state import RAGState

_workflow = StateGraph(RAGState)

_workflow.add_node("query_analyzer", node_query_analyzer)
_workflow.add_node("retrieve", node_retrieve)
_workflow.add_node("rerank", node_rerank)
_workflow.add_node("sufficiency_check", node_sufficiency_check)
_workflow.add_node("generate", node_generate)
_workflow.add_node("validate", node_validate)

_workflow.add_edge(START, "query_analyzer")
_workflow.add_edge("query_analyzer", "retrieve")
_workflow.add_edge("retrieve", "rerank")
_workflow.add_edge("rerank", "sufficiency_check")
_workflow.add_conditional_edges(
    "sufficiency_check",
    route_after_sufficiency,
    {"replan": "query_analyzer", "generate": "generate"},
)
_workflow.add_edge("generate", "validate")
_workflow.add_conditional_edges(
    "validate",
    route_after_validation,
    {"end": END, "regenerate": "generate"},
)

graph = _workflow.compile()
