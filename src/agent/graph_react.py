"""
ReAct agent entry point.

Claude drives retrieval by calling typed tools, inspecting results (including
images), and deciding when it has enough evidence to answer. The loop is capped
at MAX_TURNS = 8 to bound cost.

Graph structure:

  START → call_llm → should_continue
                          ├─ "tools" → execute_tools → call_llm  (loop)
                          └─ "end"   → finalize → END

Use graph_react.graph.invoke(state) — same interface as the Planner-Executor
graph in graph.py. To switch the eval harness to ReAct, change the import in
evaluate.py from `src.agent.graph` to `src.agent.graph_react`.
"""

from langgraph.graph import END, START, StateGraph

from src.agent.react import call_llm, execute_tools, finalize, should_continue
from src.agent.state import RAGState

_workflow = StateGraph(RAGState)

_workflow.add_node("call_llm", call_llm)
_workflow.add_node("execute_tools", execute_tools)
_workflow.add_node("finalize", finalize)

_workflow.add_edge(START, "call_llm")
_workflow.add_conditional_edges(
    "call_llm",
    should_continue,
    {"tools": "execute_tools", "end": "finalize"},
)
_workflow.add_edge("execute_tools", "call_llm")
_workflow.add_edge("finalize", END)

graph = _workflow.compile()
