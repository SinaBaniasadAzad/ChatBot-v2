"""لایهٔ FastAPI روی همان هستهٔ مشترک + سروِ رابطِ وب (SPA در پوشهٔ web/).

اجرا:  uvicorn src.api.app:app --reload
  رابط کاربری:        http://127.0.0.1:8000/
  مستندات تعاملی:     http://127.0.0.1:8000/docs

اندپوینت‌ها:
  POST /classify/start   شروع دسته‌بندی (بدون تغییر نسبت به قبل)
  POST /classify/answer  پاسخ به سوال تکمیلی (بدون تغییر)
  GET  /api/faq          قالب‌های FAQ از data/faq.json
  POST /api/tickets      ثبت نهایی تیکت → شمارهٔ پیگیری TKT-YYYY-NNNNN
  GET  /api/logo         لوگوی شرکت (data/logo.png)
"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.settings import PROJECT_ROOT, settings
from src.conversation.manager import ConversationManager
from src.tickets.store import TicketStore

WEB_DIR = PROJECT_ROOT / "web"
FAQ_PATH = PROJECT_ROOT / "data" / "faq.json"
LOGO_PATH = PROJECT_ROOT / "data" / "logo.png"

app = FastAPI(title="Ticket Triage Chatbot", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# یک نمونهٔ مشترک (prompt و few-shot یک‌بار ساخته می‌شوند).
_manager: ConversationManager | None = None
_store: TicketStore | None = None


def get_manager() -> ConversationManager:
    global _manager
    if _manager is None:
        _manager = ConversationManager()
    return _manager


def get_store() -> TicketStore:
    global _store
    if _store is None:
        _store = TicketStore(settings.tickets_log_path)
    return _store


class StartRequest(BaseModel):
    summary: str
    description: str


class AnswerRequest(BaseModel):
    session_id: str
    answer: str


class TicketRequest(BaseModel):
    employee_id: str = Field(min_length=1, max_length=20)
    first_name: str = Field(min_length=1, max_length=80)
    last_name: str = Field(min_length=1, max_length=80)
    summary: str = ""
    description: str = ""
    labels: dict[str, str | None] = Field(default_factory=dict)
    needs_review: bool = False
    session_id: str | None = None


@app.post("/classify/start")
def classify_start(req: StartRequest) -> dict:
    return get_manager().start(req.summary, req.description)


@app.post("/classify/answer")
def classify_answer(req: AnswerRequest) -> dict:
    try:
        return get_manager().answer(req.session_id, req.answer)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/faq")
def faq() -> dict:
    try:
        return json.loads(FAQ_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"categories": [], "items": []}


@app.post("/api/tickets")
def submit_ticket(req: TicketRequest) -> dict:
    if not (req.summary.strip() or req.description.strip()):
        raise HTTPException(status_code=422, detail="summary or description is required")
    return get_store().submit(
        employee_id=req.employee_id.strip(),
        first_name=req.first_name.strip(),
        last_name=req.last_name.strip(),
        summary=req.summary.strip(),
        description=req.description.strip(),
        labels=req.labels,
        needs_review=req.needs_review,
        session_id=req.session_id,
    )


@app.get("/api/logo")
def logo() -> FileResponse:
    if not LOGO_PATH.exists():
        raise HTTPException(status_code=404, detail="logo not found")
    return FileResponse(LOGO_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
