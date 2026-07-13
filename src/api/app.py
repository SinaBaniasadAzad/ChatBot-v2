"""لایهٔ FastAPI روی همان هستهٔ مشترک + سروِ رابطِ وب (SPA در پوشهٔ web/).

اجرا (production — الزام: یک worker، جلسه‌ها درون‌حافظه‌ای‌اند):
  uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 1
  رابط کاربری:        /            مستندات تعاملی:  /docs

اندپوینت‌ها:
  POST /classify/start   شروع دسته‌بندی (در قطعیِ LLM: پاسخ 503 با کد llm_unavailable)
  POST /classify/answer  پاسخ به سوال تکمیلی
  GET  /api/faq          قالب‌های FAQ از data/faq.json
  POST /api/tickets      ثبت نهایی تیکت → شمارهٔ پیگیری TKT-YYYY-NNNNN (بدونِ LLM کار می‌کند)
  GET  /api/tickets/...  جستجو/بازیابی برای ادمین (فقط با ADMIN_API_TOKEN)
  GET  /health           liveness (پروسه بالاست؟)
  GET  /ready            readiness (DB سالم؟ LLM پیکربندی شده؟ retrieval بارگذاری شد؟)
  GET  /metrics          شمارنده‌ها و زمان‌سنج‌های درون‌پروسه‌ای (JSON)

Degradation: اگر DeepSeek در دسترس نباشد (مدارشکنِ باز/خطای پیاپی)، دسته‌بندی 503
می‌دهد اما ثبتِ تیکت مستقل است — SPA همان لحظه «ثبتِ بدونِ دسته‌بندی» را پیشنهاد
می‌کند و تیکت با needs_review=true برای مسیریابیِ دستی ذخیره می‌شود.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.settings import PROJECT_ROOT, settings
from src.conversation.manager import ConversationManager
from src.db.database import Database, get_database
from src.db.maintenance import apply_retention_from_settings
from src.llm.client import LLMUnavailableError
from src.tickets.store import TicketStore
from src.utils.logging import get_logger
from src.utils.metrics import metrics

log = get_logger("api")

WEB_DIR = PROJECT_ROOT / "web"
FAQ_PATH = PROJECT_ROOT / "data" / "faq.json"
LOGO_PATH = PROJECT_ROOT / "data" / "logo.png"
APP_VERSION = os.getenv("APP_VERSION", "dev")

# نمونه‌های مشترک (prompt و few-shot یک‌بار ساخته می‌شوند). تنبل، تا تست‌های آفلاین
# و ثبتِ دستیِ تیکت بدونِ کلیدِ API هم کار کنند.
_manager: ConversationManager | None = None
_store: TicketStore | None = None
_retention_stop = threading.Event()


def get_manager() -> ConversationManager:
    global _manager
    if _manager is None:
        _manager = ConversationManager()
    return _manager


def get_store() -> TicketStore:
    global _store
    if _store is None:
        _store = TicketStore(get_database())
    return _store


def _retention_loop(db: Database) -> None:
    """اعمالِ روزانهٔ سیاستِ نگه‌داری/ناشناس‌سازیِ PII (src/db/maintenance.py)."""
    while not _retention_stop.wait(0):
        try:
            apply_retention_from_settings(db)
        except Exception:
            log.exception("اجرای retention ناموفق بود؛ فردا دوباره تلاش می‌شود.")
        if _retention_stop.wait(24 * 3600):
            return


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = get_database()  # ساخت + migrate در همین‌جا (startup تمیز)
    log.info("startup: db=%s schema=v%d version=%s", db.path, db.schema_version(), APP_VERSION)
    thread = None
    if settings.retention_enabled:
        _retention_stop.clear()
        thread = threading.Thread(target=_retention_loop, args=(db,), daemon=True, name="retention")
        thread.start()
    yield
    _retention_stop.set()
    if thread is not None:
        thread.join(timeout=5)
    db.close_all()  # checkpoint و بستنِ اتصال‌ها (shutdown تمیز)
    log.info("shutdown: پایانِ تمیز.")


app = FastAPI(title="Ticket Triage Chatbot", version=APP_VERSION, lifespan=lifespan)

# SPA هم‌مبدأ سرو می‌شود → CORS پیش‌فرض لازم نیست؛ فقط با CORS_ORIGINS روشن می‌شود.
if settings.cors_origins:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    if request.url.path.startswith(("/classify", "/api", "/health", "/ready")):
        # الگویِ route (نه مسیرِ خام) تا cardinality متریک‌ها محدود بماند
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        metrics.inc(f"http_{request.method}_{path}_{response.status_code}_total")
        metrics.observe_ms(f"http_{request.method}_{path}", (time.perf_counter() - t0) * 1000)
    return response


class StartRequest(BaseModel):
    summary: str = Field(max_length=300)
    description: str = Field(max_length=4000)


class AnswerRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    answer: str = Field(max_length=2000)


class TicketRequest(BaseModel):
    employee_id: str = Field(min_length=1, max_length=20)
    first_name: str = Field(min_length=1, max_length=80)
    last_name: str = Field(min_length=1, max_length=80)
    summary: str = Field(default="", max_length=300)
    description: str = Field(default="", max_length=4000)
    labels: dict[str, str | None] = Field(default_factory=dict)
    needs_review: bool = False
    session_id: str | None = Field(default=None, max_length=64)


def _llm_unavailable(exc: Exception) -> JSONResponse:
    metrics.inc("classify_unavailable_total")
    log.warning("دسته‌بندی در دسترس نیست: %s", exc)
    return JSONResponse(
        status_code=503,
        headers={"Retry-After": str(int(settings.cb_cooldown_seconds))},
        content={
            "detail": {
                "code": "llm_unavailable",
                "message": "سرویسِ دسته‌بندی موقتاً در دسترس نیست؛ تیکت را می‌توانید بدونِ دسته‌بندی ثبت کنید.",
            }
        },
    )


@app.post("/classify/start")
def classify_start(req: StartRequest):
    try:
        return get_manager().start(req.summary, req.description)
    except LLMUnavailableError as e:
        return _llm_unavailable(e)
    except RuntimeError as e:  # مثلاً DEEPSEEK_API_KEY تنظیم نشده
        return _llm_unavailable(e)


@app.post("/classify/answer")
def classify_answer(req: AnswerRequest):
    try:
        return get_manager().answer(req.session_id, req.answer)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except LLMUnavailableError as e:
        return _llm_unavailable(e)
    except RuntimeError as e:
        return _llm_unavailable(e)


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
    record = get_store().submit(
        employee_id=req.employee_id.strip(),
        first_name=req.first_name.strip(),
        last_name=req.last_name.strip(),
        summary=req.summary.strip(),
        description=req.description.strip(),
        labels=req.labels,
        needs_review=req.needs_review,
        session_id=req.session_id,
    )
    metrics.inc("tickets_submitted_total")
    if req.needs_review:
        metrics.inc("tickets_needs_review_total")
    return record


# ---- اندپوینت‌های ادمین (جستجو/بازیابی؛ حاوی PII → فقط با توکن) ----
def _require_admin(token: str | None) -> None:
    if not settings.admin_api_token:
        raise HTTPException(status_code=404)  # غیرفعال — وجودش را هم لو نده
    if not (token and secrets.compare_digest(token, settings.admin_api_token)):
        raise HTTPException(status_code=401, detail="invalid admin token")


@app.get("/api/tickets/{reference}")
def get_ticket(reference: str, x_admin_token: str | None = Header(default=None)) -> dict:
    _require_admin(x_admin_token)
    record = get_store().get(reference)
    if record is None:
        raise HTTPException(status_code=404, detail="ticket not found")
    return record


@app.get("/api/tickets")
def search_tickets(
    q: str = "",
    needs_review: bool | None = None,
    limit: int = 50,
    offset: int = 0,
    x_admin_token: str | None = Header(default=None),
) -> dict:
    _require_admin(x_admin_token)
    items = get_store().search(q, needs_review=needs_review, limit=limit, offset=offset)
    return {"items": items, "count": len(items)}


@app.get("/api/logo")
def logo() -> FileResponse:
    if not LOGO_PATH.exists():
        raise HTTPException(status_code=404, detail="logo not found")
    return FileResponse(LOGO_PATH)


# ---- سلامت و پایش ----
@app.get("/health")
def health() -> dict:
    """Liveness: پروسه زنده و پاسخ‌گوست."""
    return {"status": "ok", "version": APP_VERSION}


@app.get("/ready")
def ready():
    """Readiness: DB قابلِ استفاده است؟ (LLM و retrieval گزارش می‌شوند ولی مانع نیستند —
    ثبتِ دستیِ تیکت بدونِ هر دو کار می‌کند.)"""
    checks: dict[str, str] = {}
    ok = True
    try:
        db = get_database()
        db.check()
        checks["db"] = f"ok (schema v{db.schema_version()})"
    except Exception as e:
        ok = False
        checks["db"] = f"error: {e}"

    if not settings.deepseek_api_key:
        checks["llm"] = "unconfigured (DEEPSEEK_API_KEY خالی است — فقط ثبتِ دستی)"
    elif _manager is not None:
        checks["llm"] = f"configured (circuit={_manager.classifier.client.breaker.state})"
    else:
        checks["llm"] = "configured (not yet used)"

    if not settings.retrieval_enabled:
        checks["retrieval"] = "disabled"
    elif _manager is not None:
        checks["retrieval"] = "loaded" if _manager.classifier.retriever else "unavailable (fallback)"
    else:
        checks["retrieval"] = "enabled (not yet loaded)"

    body = {"status": "ready" if ok else "unavailable", "checks": checks}
    return body if ok else JSONResponse(status_code=503, content=body)


@app.get("/metrics")
def get_metrics() -> dict:
    """متریک‌های درون‌پروسه‌ای (JSON). برای Prometheus: src/utils/metrics.py را جایگزین کنید."""
    return metrics.snapshot()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
