"""
Facade ردیابی روی Langfuse (SDK v4، مبتنی بر OpenTelemetry) — اختیاری و شکست‌ناپذیر.

اصلِ طراحی (همان الگوی retriever): نبودِ بستهٔ `langfuse`، نبودِ کلیدها، یا
OBSERVABILITY_ENABLED=false فقط ردیابی را غیرفعال می‌کند؛ مسیرِ اصلیِ دسته‌بندی
هرگز به‌خاطرِ observability نمی‌افتد و هیچ تستی به سرورِ Langfuse نیاز ندارد.

چرا facade و نه استفادهٔ مستقیم از SDK در کد؟
  ۱) هر نقطهٔ instrument فقط با همین ماژول کار می‌کند → خاموش/روشن‌کردن، ارتقای
     نسخهٔ SDK (مثل v3→v4 که API عوض شد) و حتی تعویضِ backend یک‌جا انجام می‌شود.
  ۲) همهٔ فراخوانی‌ها داخلِ try/except هستند: خطای شبکه/سریال‌سازی در tracing
     نباید جریانِ کاربر را بشکند (اصلِ «observer must not affect the observed»).

الگوی استفاده:
    with trace_context(session_id=..., trace_name="ticket-classification"):
        with span("classification-round", input={...}) as root:
            ...
            root.update(output={...})
    with generation("deepseek-completion", model=..., input=messages) as g:
        g.update(output=..., usage_details=..., cost_details=...)
"""
from __future__ import annotations

import atexit
from contextlib import contextmanager

from config.settings import settings
from src.utils.logging import get_logger

log = get_logger("observability")

_client = None
_client_init_done = False


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------
def get_client():
    """کلاینتِ Langfuse (singleton) یا None. هرگز استثنا نمی‌دهد."""
    global _client, _client_init_done
    if _client_init_done:
        return _client
    _client_init_done = True

    if not settings.observability_enabled:
        log.info("ردیابی خاموش است (OBSERVABILITY_ENABLED=false).")
        return None
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        log.info("کلیدهای Langfuse تنظیم نشده‌اند — ردیابی غیرفعال شد (مسیرِ اصلی بدونِ تغییر).")
        return None
    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            environment=settings.observability_environment,
            sample_rate=settings.langfuse_sample_rate,
        )
        atexit.register(_shutdown)
        log.info(
            "ردیابیِ Langfuse فعال شد (host=%s, env=%s)",
            settings.langfuse_host, settings.observability_environment,
        )
    except Exception as e:  # نبودِ بسته/پیکربندیِ خراب → فقط ردیابی خاموش می‌شود
        log.warning("راه‌اندازیِ Langfuse ناموفق بود؛ ردیابی غیرفعال شد: %s", e)
        _client = None
    return _client


def enabled() -> bool:
    return get_client() is not None


def _shutdown() -> None:
    global _client
    if _client is not None:
        try:
            _client.shutdown()
        except Exception:
            pass


def _reset_for_tests() -> None:
    """فقط برای تست: singleton را ریست می‌کند تا تغییرِ settings اثر کند."""
    global _client, _client_init_done
    _shutdown()
    _client = None
    _client_init_done = False


# ---------------------------------------------------------------------------
# spanهای امن (no-op وقتی ردیابی خاموش است؛ ضدِ استثنا وقتی روشن است)
# ---------------------------------------------------------------------------
class _NoopSpan:
    """جایگزینِ بی‌اثرِ span — هر متد/صفتی را می‌پذیرد و هیچ‌کاری نمی‌کند."""

    trace_id = None
    id = None

    def __getattr__(self, name):
        return self._noop

    @staticmethod
    def _noop(*args, **kwargs):
        return None


NOOP_SPAN = _NoopSpan()


class _SafeSpan:
    """Wrapper دورِ spanِ واقعی: هیچ خطای tracing به کدِ اصلی نشت نمی‌کند."""

    __slots__ = ("_span",)

    def __init__(self, span) -> None:
        self._span = span

    @property
    def trace_id(self):
        try:
            return self._span.trace_id
        except Exception:
            return None

    @property
    def id(self):
        try:
            return self._span.id
        except Exception:
            return None

    def __getattr__(self, name):
        attr = getattr(self._span, name, None)
        if not callable(attr):
            return attr

        def _safe(*args, **kwargs):
            try:
                return attr(*args, **kwargs)
            except Exception as e:
                log.debug("خطای tracing در %s نادیده گرفته شد: %s", name, e)
                return None

        return _safe


@contextmanager
def span(name: str, *, input=None, metadata=None, as_type: str = "span"):
    """یک observationِ عمومی. as_type: span | chain | retriever | tool | ... (آیکونِ UI)."""
    lf = get_client()
    if lf is None:
        yield NOOP_SPAN
        return
    try:
        cm = lf.start_as_current_observation(
            name=name, as_type=as_type, input=input, metadata=metadata
        )
    except Exception as e:
        log.debug("شروعِ span ناموفق بود (%s): %s", name, e)
        yield NOOP_SPAN
        return
    with cm as s:
        yield _SafeSpan(s)


@contextmanager
def generation(name: str, *, model=None, input=None, model_parameters=None, metadata=None):
    """یک observation از نوعِ generation (فراخوانیِ LLM) با usage/cost."""
    lf = get_client()
    if lf is None:
        yield NOOP_SPAN
        return
    try:
        cm = lf.start_as_current_observation(
            name=name, as_type="generation", model=model, input=input,
            model_parameters=model_parameters, metadata=metadata,
        )
    except Exception as e:
        log.debug("شروعِ generation ناموفق بود (%s): %s", name, e)
        yield NOOP_SPAN
        return
    with cm as g:
        yield _SafeSpan(g)


@contextmanager
def trace_context(*, session_id=None, user_id=None, trace_name=None, metadata=None, tags=None):
    """صفاتِ trace-سطح (گروه‌بندیِ session و متادیتای پیکربندی) برای spanهای داخلش.

    در SDK v4 صفاتِ trace با propagate_attributes «قبل از» ساختِ span تعیین می‌شوند؛
    خروجی‌های پایانِ کار (برچسب‌ها/اکشن) را روی spanِ ریشه یا score بگذارید.
    """
    lf = get_client()
    if lf is None:
        yield
        return
    try:
        from langfuse import propagate_attributes

        cm = propagate_attributes(
            session_id=session_id, user_id=user_id, trace_name=trace_name,
            metadata=metadata, tags=tags,
        )
    except Exception as e:
        log.debug("propagate_attributes ناموفق بود: %s", e)
        yield
        return
    with cm:
        yield


# ---------------------------------------------------------------------------
# ابزارهای trace-سطح
# ---------------------------------------------------------------------------
def current_trace_id() -> str | None:
    lf = get_client()
    if lf is None:
        return None
    try:
        return lf.get_current_trace_id()
    except Exception:
        return None


def score_current_trace(name: str, value, *, comment: str | None = None, data_type=None) -> None:
    """ثبتِ score روی traceِ جاری (مثلاً ambiguity/needs_review/top_similarity)."""
    lf = get_client()
    if lf is None:
        return
    try:
        kwargs = {"name": name, "value": value, "comment": comment}
        if data_type:
            kwargs["data_type"] = data_type
        lf.score_current_trace(**kwargs)
    except Exception as e:
        log.debug("score_current_trace نادیده گرفته شد: %s", e)


def score_trace(trace_id: str, name: str, value, *, comment: str | None = None, data_type=None) -> None:
    """ثبتِ score روی یک traceِ مشخص (مسیرِ صفِ بازبینی: برچسبِ انسانی → trace)."""
    lf = get_client()
    if lf is None or not trace_id:
        return
    try:
        kwargs = {"trace_id": trace_id, "name": name, "value": value, "comment": comment}
        if data_type:
            kwargs["data_type"] = data_type
        lf.create_score(**kwargs)
    except Exception as e:
        log.debug("score_trace نادیده گرفته شد: %s", e)


def trace_url(trace_id: str | None) -> str | None:
    """لینکِ trace در UIِ Langfuse (برای صفِ بازبینی)."""
    if not trace_id:
        return None
    lf = get_client()
    if lf is not None:
        try:
            url = lf.get_trace_url(trace_id=trace_id)
            if url:
                return url
        except Exception:
            pass
    # fallbackِ بدونِ کلاینت: ریدایرکتِ /trace/{id} به پروژهٔ مربوطه
    return f"{settings.langfuse_host.rstrip('/')}/trace/{trace_id}"


def flush() -> None:
    """ارسالِ فوریِ بافر — برای اسکریپت‌های کوتاه‌عمر (eval/experiment) ضروری است."""
    lf = get_client()
    if lf is None:
        return
    try:
        lf.flush()
    except Exception as e:
        log.debug("flush نادیده گرفته شد: %s", e)
