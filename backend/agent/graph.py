"""
agent/graph.py — LangGraph ReAct agent for life insurance support.

Implements a manual ReAct loop using StateGraph, compatible with langgraph>=1.x.
Flow: llm_node → (tool call?) → tool_executor_node → llm_node → ... → END
"""
import json
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

from agent.tools import search_knowledge_base, get_claims_process
from config import get_llm

# ── System prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = SystemMessage(content="""You are a knowledgeable and empathetic Life Insurance Support Assistant.

Your role is to help users understand:
- Life insurance policy types (Term, Whole Life, ULIP, Endowment, Money-Back)
- Policy benefits, riders, and coverage details
- Eligibility criteria (age, health, occupation)
- The claims process and required documents
- Premium payment options and policy renewal

Guidelines:
- Use the `search_knowledge_base` tool when the user asks about policy types, eligibility, benefits, riders, or exclusions.
- Use the `get_claims_process` tool when the user asks about filing a claim or required documents.
- Only call a tool when it is clearly needed; do not call tools for greetings or simple clarifications.
- Be clear, concise, and empathetic.
- If you don't know something, say so honestly.
- Do NOT give financial or legal advice.
""")

TOOLS = [search_knowledge_base, get_claims_process]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: str


def llm_node(state: AgentState) -> dict:
    """Invoke the LLM with tools bound; may produce tool calls or a final answer."""
    llm = get_llm().bind_tools(TOOLS, tool_choice="auto")
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def tool_executor_node(state: AgentState) -> dict:
    """Execute all tool calls requested by the last LLM message."""
    last_message = state["messages"][-1]
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool = TOOLS_BY_NAME[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": tool_messages}


def should_continue(state: AgentState) -> str:
    """Route: if last message has tool calls → execute tools, else → END."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def build_graph(checkpointer):
    """Build and compile the LangGraph StateGraph with SQLite checkpointing."""
    graph = StateGraph(AgentState)

    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_executor_node)

    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")

    return graph.compile(checkpointer=checkpointer)
