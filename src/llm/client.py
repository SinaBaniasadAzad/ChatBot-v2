"""
Wrapper نازک روی DeepSeek (سازگار با OpenAI SDK).

مسئولیت‌ها: فراخوانی API، حالت JSON، retry روی خطای شبکه/parse، self-consistency،
و برگرداندن متادیتا (latency و مصرف توکن) برای لاگ و تحلیل هزینه.
هیچ منطق تجاری اینجا نیست.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field

from openai import OpenAI

from config.settings import settings
from src.observability import generation as obs_generation
from src.observability.cost import cost_details, usage_details
from src.utils.logging import get_logger

log = get_logger("llm.client")


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
        )

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """یک فراخوانی در حالت JSON. خروجی: LLMResponse. در صورت خطا retry می‌کند."""
        model = model or settings.model
        temperature = settings.temperature if temperature is None else temperature
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # spanِ generation: پیام‌ها، usage با تفکیکِ cache hit/miss، هزینهٔ دلاری و retryها.
        with obs_generation(
            "deepseek-completion",
            model=model,
            input=messages,
            model_parameters={"temperature": temperature, "response_format": "json_object"},
        ) as gen:
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
                    gen.update(
                        output=data,
                        usage_details=usage_details(usage),
                        cost_details=cost_details(usage),
                        metadata={"latency_ms": round(latency_ms, 1), "attempts": attempt},
                    )
                    return LLMResponse(
                        data=data, model=model, latency_ms=latency_ms, usage=usage, raw=content
                    )
                except json.JSONDecodeError as e:
                    last_err = e
                    log.warning("خروجی JSON معتبر نبود (تلاش %d/%d).", attempt, settings.max_retries)
                except Exception as e:  # خطای شبکه/API
                    last_err = e
                    log.warning("خطای فراخوانی API (تلاش %d/%d): %s", attempt, settings.max_retries, e)
                time.sleep(min(2 ** attempt, 8))  # backoff نمایی ساده

            gen.update(
                level="ERROR",
                status_message=str(last_err),
                metadata={"attempts": settings.max_retries},
            )
            raise RuntimeError(
                f"فراخوانی DeepSeek پس از {settings.max_retries} تلاش ناموفق بود: {last_err}"
            )

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
