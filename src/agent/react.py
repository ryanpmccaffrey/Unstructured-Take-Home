"""
LangGraph node functions for the ReAct agent.

call_llm:       calls Anthropic API with the current conversation history.
execute_tools:  executes all tool_use blocks from the last assistant message.
finalize:       extracts answer and cited chunk IDs once the loop ends.
should_continue: routing function — "tools" → execute_tools, "end" → finalize.
"""

from __future__ import annotations

import anthropic
import chromadb

from src.agent.state import RAGState
from src.agent.tools import TOOL_DEFINITIONS, execute_tool
from src.config import ANTHROPIC_API_KEY, CHROMA_PATH, CLAUDE_MODEL

MAX_TURNS = 8

_SYSTEM_PROMPT = """You are a precise document question-answering assistant with access to retrieval tools.

Instructions:
1. Use the tools to find evidence that answers the question. Choose the most specific tool available.
2. You may call tools multiple times with different queries if the first result is insufficient.
3. Once you have enough evidence, provide a concise, direct answer.
4. In your final answer, cite the exact chunk IDs (e.g. [chunk_id]) from the retrieved content.
5. End your answer with a line in this exact format:
   Cited: chunk_id_1, chunk_id_2, ...

Only cite chunk IDs that actually appear in the retrieved results."""

_chroma: chromadb.ClientAPI | None = None
_client: anthropic.Anthropic | None = None


def _get_chroma() -> chromadb.ClientAPI:
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def _parse_cited_chunks(text: str) -> tuple[str, list[str]]:
    lines = text.strip().splitlines()
    cited: list[str] = []
    answer_lines: list[str] = []
    for line in lines:
        if line.strip().lower().startswith("cited:"):
            cited = [c.strip() for c in line.split(":", 1)[1].split(",") if c.strip()]
        else:
            answer_lines.append(line)
    return "\n".join(answer_lines).strip(), cited


def call_llm(state: RAGState) -> dict:
    """Call Anthropic API with current messages; initialize conversation on the first turn."""
    messages = list(state.get("messages") or [])

    if not messages:
        messages = [{"role": "user", "content": state["question"]}]

    response = _get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        tools=TOOL_DEFINITIONS,
        messages=messages,
    )

    # Prepend the user message on the first turn so the reducer sees the full exchange.
    new_messages: list[dict] = []
    if not state.get("messages"):
        new_messages.append({"role": "user", "content": state["question"]})
    new_messages.append({"role": "assistant", "content": response.content})

    update: dict = {"messages": new_messages}

    if response.stop_reason == "end_turn":
        final_text = next((b.text for b in response.content if hasattr(b, "text")), "")
        answer, cited_ids = _parse_cited_chunks(final_text)
        update["answer"] = answer
        update["cited_chunk_ids"] = cited_ids

    return update


def execute_tools(state: RAGState) -> dict:
    """Execute all tool_use blocks from the last assistant message."""
    chroma = _get_chroma()

    last_assistant = next(
        (m for m in reversed(state["messages"]) if m["role"] == "assistant"),
        None,
    )
    if not last_assistant:
        return {}

    seen_ids = {c["chunk_id"] for c in state.get("retrieved_chunks") or []}
    tool_results: list[dict] = []
    new_chunks: list[dict] = []

    for block in last_assistant["content"]:
        if not hasattr(block, "type") or block.type != "tool_use":
            continue
        chunks, content_blocks = execute_tool(block.name, block.input, chroma)

        for c in chunks:
            if c["chunk_id"] not in seen_ids:
                seen_ids.add(c["chunk_id"])
                new_chunks.append(c)

        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": content_blocks,
        })

    return {
        "messages": [{"role": "user", "content": tool_results}],
        "retrieved_chunks": new_chunks,
    }


def finalize(state: RAGState) -> dict:
    """Extract answer from the last assistant text block if not already set (MAX_TURNS fallback)."""
    if state.get("answer"):
        return {}

    final_text = next(
        (
            b.text
            for m in reversed(state.get("messages", []))
            if m["role"] == "assistant"
            for b in (m["content"] if isinstance(m["content"], list) else [])
            if hasattr(b, "text")
        ),
        "Unable to answer within the allowed number of retrieval steps.",
    )
    answer, cited_ids = _parse_cited_chunks(final_text)
    return {"answer": answer, "cited_chunk_ids": cited_ids}


def should_continue(state: RAGState) -> str:
    """Route to execute_tools if Claude requested tools, or end otherwise."""
    if state.get("answer"):
        return "end"

    assistant_turns = sum(1 for m in state.get("messages", []) if m["role"] == "assistant")
    if assistant_turns >= MAX_TURNS:
        return "end"

    last_assistant = next(
        (m for m in reversed(state.get("messages", [])) if m["role"] == "assistant"),
        None,
    )
    if last_assistant and any(
        hasattr(b, "type") and b.type == "tool_use"
        for b in (last_assistant.get("content") or [])
    ):
        return "tools"

    return "end"
