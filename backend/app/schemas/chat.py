"""
app/schemas/chat.py — Pydantic request/response models for the /chat endpoint.
"""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique session/thread ID (UUID)")
    message: str = Field(..., min_length=1, description="User's message text")


class MessageItem(BaseModel):
    role: str          # "user" or "assistant"
    content: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    history: list[MessageItem]
