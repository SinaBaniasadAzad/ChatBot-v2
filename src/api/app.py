"""لایهٔ FastAPI روی همان هستهٔ مشترک + سروِ رابطِ وب (SPA در پوشهٔ web/).

اجرا:  uvicorn src.api.app:app --reload
  رابط کاربری:        http://127.0.0.1:8000/
  صفِ بازبینی:        http://127.0.0.1:8000/review
  مستندات تعاملی:     http://127.0.0.1:8000/docs

اندپوینت‌ها:
  POST /classify/start   شروع دسته‌بندی (بدون تغییر نسبت به قبل)
  POST /classify/answer  پاسخ به سوال تکمیلی (بدون تغییر)
  GET  /api/faq          قالب‌های FAQ از data/faq.json
  POST /api/tickets      ثبت نهایی تیکت → شمارهٔ پیگیری TKT-YYYY-NNNNN
  GET  /api/logo         لوگوی شرکت (data/logo.png)
  /api/review/*          صفِ بازبینیِ انسانی (annotation & review queue)
  GET  /api/debug/config پیکربندیِ مؤثرِ classifier (اثرانگشت + few-shot)
"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.settings import PROJECT_ROOT, settings
from src import observability as obs
from src.conversation.manager import ConversationManager
from src.review.store import ReviewStore, maybe_build_review_store
from src.taxonomy import load_taxonomy
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
_review_store: ReviewStore | None = None
_review_store_init = False


def get_manager() -> ConversationManager:
    global _manager
    if _manager is None:
        _manager = ConversationManager(review_store=get_review_store())
    return _manager


def get_store() -> TicketStore:
    global _store
    if _store is None:
        _store = TicketStore(settings.tickets_log_path)
    return _store


def get_review_store() -> ReviewStore | None:
    """صفِ بازبینی — مستقل از classifier تا بدونِ کلیدِ API هم کار کند."""
    global _review_store, _review_store_init
    if not _review_store_init:
        _review_store = maybe_build_review_store()
        _review_store_init = True
    return _review_store


def _require_review_store() -> ReviewStore:
    store = get_review_store()
    if store is None:
        raise HTTPException(status_code=503, detail="review queue is disabled")
    return store


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


# ---------------------------------------------------------------------------
# صفِ بازبینیِ انسانی (Annotation & Review Queue)
# ---------------------------------------------------------------------------
class ResolveRequest(BaseModel):
    labels: dict[str, str]            # {layer_id: label_id} — برچسبِ طلاییِ انسانی
    reviewer: str = Field(min_length=1, max_length=80)
    notes: str | None = None


class DismissRequest(BaseModel):
    reviewer: str = Field(min_length=1, max_length=80)
    notes: str | None = None


def _with_trace_url(item: dict) -> dict:
    item["trace_url"] = obs.trace_url(item.get("trace_id"))
    return item


def _validate_labels(labels: dict[str, str]) -> None:
    tax = load_taxonomy()
    for lid, label in labels.items():
        layer = tax.get_layer(lid)
        if layer is None:
            raise HTTPException(status_code=422, detail=f"unknown layer: {lid}")
        if label not in layer.label_ids:
            raise HTTPException(status_code=422, detail=f"unknown label '{label}' for {lid}")


@app.get("/api/review/queue")
def review_queue(status: str | None = "pending", source: str | None = None,
                 limit: int = 50, offset: int = 0) -> dict:
    store = _require_review_store()
    items = store.list_items(status=status or None, source=source, limit=limit, offset=offset)
    return {"items": [_with_trace_url(i) for i in items], "stats": store.stats()}


@app.get("/api/review/items/{item_id}")
def review_item(item_id: int) -> dict:
    item = _require_review_store().get(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="review item not found")
    return _with_trace_url(item)


@app.post("/api/review/items/{item_id}/resolve")
def review_resolve(item_id: int, req: ResolveRequest) -> dict:
    _validate_labels(req.labels)
    store = _require_review_store()
    item = store.resolve(item_id, req.labels, req.reviewer.strip(), req.notes)
    if item is None:
        raise HTTPException(status_code=409, detail="item not found or already closed")
    # برچسبِ انسانی → score روی traceِ Langfuse (حلقهٔ بازخوردِ ارزیابیِ آنلاین).
    trace_id = item.get("trace_id")
    if trace_id:
        pred = item.get("predicted_labels") or {}
        matches = [int(pred.get(lid) == lbl) for lid, lbl in req.labels.items()]
        obs.score_trace(trace_id, "human_reviewed", 1, comment=f"reviewer: {req.reviewer}")
        if matches:
            obs.score_trace(trace_id, "human_correct", sum(matches) / len(matches),
                            comment=req.notes)
        for lid, lbl in req.labels.items():
            obs.score_trace(trace_id, f"human_label_{lid}", lbl, data_type="CATEGORICAL")
        obs.flush()
    return _with_trace_url(item)


@app.post("/api/review/items/{item_id}/dismiss")
def review_dismiss(item_id: int, req: DismissRequest) -> dict:
    item = _require_review_store().dismiss(item_id, req.reviewer.strip(), req.notes)
    if item is None:
        raise HTTPException(status_code=409, detail="item not found or already closed")
    return _with_trace_url(item)


@app.get("/api/review/stats")
def review_stats() -> dict:
    return _require_review_store().stats()


@app.get("/api/review/export")
def review_export() -> dict:
    """برچسب‌های طلاییِ تاییدشده — خوراکِ Gold Set / کاندیدِ few-shot (با گاردِ نشت در اسکریپت)."""
    rows = _require_review_store().export_gold()
    return {"count": len(rows), "rows": rows}


@app.get("/api/review/taxonomy")
def review_taxonomy() -> dict:
    """لایه‌ها و برچسب‌های مجاز برای UIِ بازبینی (پویا از taxonomy)."""
    tax = load_taxonomy()
    return {
        "layers": [
            {
                "id": layer.id,
                "name": layer.name,
                "labels": [{"id": lbl.id, "name": lbl.name} for lbl in layer.labels],
            }
            for layer in tax.layers
        ]
    }


@app.get("/api/debug/config")
def debug_config() -> dict:
    """پیکربندیِ مؤثرِ classifier: اثرانگشت، مثال‌های few-shot، وضعیتِ retrieval."""
    return get_manager().classifier.config_snapshot()


@app.get("/review", include_in_schema=False)
def review_page() -> FileResponse:
    return FileResponse(WEB_DIR / "review.html")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
