"""
Wrapper نازک روی DeepSeek (سازگار با OpenAI SDK).

مسئولیت‌ها: فراخوانی API، حالت JSON، retry فقط روی خطاهای *گذرا* (شبکه/timeout/
429/5xx/JSON خراب) با backoff نمایی + jitter، مدارشکن (circuit breaker) برای
degradationِ سریع وقتی DeepSeek قطع است، self-consistency، و متادیتا (latency و
مصرف توکن) برای لاگ/متریک. هیچ منطق تجاری اینجا نیست.

مدارشکن: پس از N چرخهٔ ناموفقِ پیاپی، تا مدتِ cooldown هر فراخوان بلافاصله
LLMUnavailableError می‌دهد (به‌جای ۳×timeout انتظار)؛ بعد از cooldown یک تلاشِ
آزمایشی (half-open) عبور می‌کند و موفقیتش مدار را می‌بندد.
"""
from __future__ import annotations

import json
import random
import threading
import time
from collections import Counter
from dataclasses import dataclass, field

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from config.settings import settings
from src.utils.logging import get_logger
from src.utils.metrics import metrics

log = get_logger("llm.client")


class LLMUnavailableError(RuntimeError):
    """سرویسِ دسته‌بندی فعلاً در دسترس نیست (خطای پیاپی یا مدارشکنِ باز).

    لایهٔ API این را به 503 ترجمه می‌کند؛ ثبتِ دستیِ تیکت همچنان ممکن است."""


def _is_retryable(err: Exception) -> bool:
    """فقط خطاهای گذرا ارزشِ retry دارند؛ خطای پیکربندی (401/400) بلافاصله بالا برود."""
    if isinstance(err, (APIConnectionError, APITimeoutError, RateLimitError, json.JSONDecodeError)):
        return True
    if isinstance(err, APIStatusError):
        return err.status_code >= 500
    return False


class _CircuitBreaker:
    def __init__(self, threshold: int, cooldown: float) -> None:
        self.threshold = max(1, threshold)
        self.cooldown = cooldown
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return True
            if time.monotonic() - self._opened_at >= self.cooldown:
                # half-open: یک تلاشِ آزمایشی عبور کند؛ پنجره جلو می‌رود تا رگبار نشود
                self._opened_at = time.monotonic() - self.cooldown + min(self.cooldown, 5.0)
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.threshold and self._opened_at is None:
                self._opened_at = time.monotonic()
                log.error("مدارشکنِ LLM باز شد (پس از %d خطای پیاپی).", self._failures)

    @property
    def state(self) -> str:
        with self._lock:
            return "open" if self._opened_at is not None else "closed"


@dataclass
class LLMResponse:
    """خروجی یک فراخوانی + متادیتا (برای لاگ)."""
    data: dict
    model: str
    latency_ms: float
    usage: dict = field(default_factory=dict)
    raw: str = ""


class DeepSeekClient:
    def __init__(self) -> None:
        settings.require_api_key()
        self._client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            timeout=settings.request_timeout,
            max_retries=0,  # retry این‌جا مدیریت می‌شود (لاگ + متریک + مدارشکن)
            http_client=httpx.Client(
                timeout=settings.request_timeout,
                limits=httpx.Limits(
                    max_connections=settings.llm_max_connections,
                    max_keepalive_connections=min(8, settings.llm_max_connections),
                ),
            ),
        )
        self.breaker = _CircuitBreaker(
            settings.cb_failure_threshold, settings.cb_cooldown_seconds
        )

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """یک فراخوانی در حالت JSON؛ روی خطای گذرا retry می‌کند.

        Raises:
            LLMUnavailableError: مدارشکن باز است یا همهٔ تلاش‌ها شکست خورد.
        """
        if not self.breaker.allow():
            metrics.inc("llm_rejected_circuit_open_total")
            raise LLMUnavailableError("مدارشکنِ LLM باز است؛ سرویسِ دسته‌بندی موقتاً در دسترس نیست.")

        model = model or settings.model
        temperature = settings.temperature if temperature is None else temperature
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        last_err: Exception | None = None
        for attempt in range(1, settings.max_retries + 1):
            try:
                t0 = time.perf_counter()
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                content = resp.choices[0].message.content or ""
                data = json.loads(content)  # JSON mode فقط نحو معتبر را تضمین می‌کند
                usage = resp.usage.model_dump() if resp.usage else {}
                self.breaker.record_success()
                metrics.inc("llm_calls_total")
                metrics.observe_ms("llm_latency", latency_ms)
                return LLMResponse(
                    data=data, model=model, latency_ms=latency_ms, usage=usage, raw=content
                )
            except Exception as e:
                last_err = e
                metrics.inc("llm_errors_total")
                if not _is_retryable(e):
                    log.error("خطای غیرقابلِ‌retry از API (پیکربندی/احراز هویت؟): %s", e)
                    self.breaker.record_failure()
                    raise LLMUnavailableError(f"خطای غیرگذرا از سرویسِ LLM: {e}") from e
                log.warning("خطای گذرای API (تلاش %d/%d): %s", attempt, settings.max_retries, e)
                if attempt < settings.max_retries:
                    # backoff نمایی + jitter (نصف تا یک‌ونیم برابر) — بدون همگاییِ رگباری
                    time.sleep(min(2 ** attempt, 8) * (0.5 + random.random()))

        self.breaker.record_failure()
        metrics.inc("llm_exhausted_retries_total")
        raise LLMUnavailableError(
            f"فراخوانی DeepSeek پس از {settings.max_retries} تلاش ناموفق بود: {last_err}"
        ) from last_err

    def majority_vote(
        self,
        system: str,
        user: str,
        *,
        key_fn,
        n: int = 3,
        temperature: float = 0.5,
    ) -> tuple[LLMResponse, float]:
        """
        Self-consistency: n بار نمونه‌گیری و رأی‌گیری روی data هر پاسخ.
        key_fn(dict) -> کلیدِ قابل‌مقایسه (مثلاً تاپلِ برچسب‌ها).
        خروجی: (پاسخِ برنده، نسبت توافق در بازهٔ 0..1).
        """
        responses: list[LLMResponse] = []
        keys: list = []
        for _ in range(n):
            r = self.complete_json(system, user, temperature=temperature)
            responses.append(r)
            keys.append(key_fn(r.data))
        winner_key, count = Counter(keys).most_common(1)[0]
        winner = next(r for r, k in zip(responses, keys) if k == winner_key)
        return winner, count / n
