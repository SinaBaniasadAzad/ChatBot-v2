"""
لایهٔ FastAPI روی همان هستهٔ مشترک.

اجرا:  uvicorn src.api.app:app --reload
سپس مستندات تعاملی:  http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.conversation.manager import ConversationManager

app = FastAPI(title="Ticket Triage Chatbot", version="0.1.0")

# یک نمونهٔ مشترک (prompt و few-shot یک‌بار ساخته می‌شوند).
_manager: ConversationManager | None = None


def get_manager() -> ConversationManager:
    global _manager
    if _manager is None:
        _manager = ConversationManager()
    return _manager


class StartRequest(BaseModel):
    summary: str
    description: str


class AnswerRequest(BaseModel):
    session_id: str
    answer: str


@app.post("/classify/start")
def classify_start(req: StartRequest) -> dict:
    return get_manager().start(req.summary, req.description)


@app.post("/classify/answer")
def classify_answer(req: AnswerRequest) -> dict:
    try:
        return get_manager().answer(req.session_id, req.answer)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
