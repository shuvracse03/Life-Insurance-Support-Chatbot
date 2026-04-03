"""
app/routes/chat.py — POST /chat endpoint.
"""
from fastapi import APIRouter, Depends
from langchain_core.messages import HumanMessage, AIMessage

from app.schemas.chat import ChatRequest, ChatResponse, MessageItem
from app.dependencies import get_agent

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, agent=Depends(get_agent)) -> ChatResponse:
    """
    Accept a user message and session_id, invoke the LangGraph agent,
    and return the assistant's response along with the full conversation history.
    """
    config = {"configurable": {"thread_id": request.session_id}}

    # Invoke the agent with the new user message
    result = agent.invoke(
        {"messages": [HumanMessage(content=request.message)],
         "session_id": request.session_id},
        config=config,
    )

    # Extract the latest assistant response
    all_messages = result["messages"]
    assistant_response = ""
    for msg in reversed(all_messages):
        if isinstance(msg, AIMessage) and msg.content:
            assistant_response = msg.content
            break

    # Build history for the UI
    history: list[MessageItem] = []
    for msg in all_messages:
        if isinstance(msg, HumanMessage):
            history.append(MessageItem(role="user", content=msg.content))
        elif isinstance(msg, AIMessage) and msg.content:
            history.append(MessageItem(role="assistant", content=msg.content))

    return ChatResponse(
        session_id=request.session_id,
        response=assistant_response,
        history=history,
    )
