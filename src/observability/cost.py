"""
تبدیلِ usageِ خامِ DeepSeek به usage/costِ استانداردِ Langfuse.

منبعِ حقیقتِ قیمت‌ها همان `src/reporting/cost.py` (Pricing) است؛ این ماژول فقط
نگاشت انجام می‌دهد تا اعدادِ داشبوردِ Langfuse با گزارش‌های موجود یکسان باشند.

نگاشتِ توکن‌ها (کلیدهای usage_details در Langfuse):
  input             = prompt_cache_miss_tokens  (ورودیِ گران)
  input_cache_read  = prompt_cache_hit_tokens   (ورودیِ کش‌شدهٔ ارزان)
  output            = completion_tokens
اگر تفکیکِ hit/miss گزارش نشده باشد، کلِ prompt محافظه‌کارانه cache-miss فرض
می‌شود (هم‌رفتار با eval_incdb و reporting/cost).
"""
from __future__ import annotations

from config.settings import settings
from src.reporting.cost import Pricing

_MILLION = 1_000_000


def pricing_from_settings() -> Pricing:
    return Pricing(
        input_per_m=settings.price_input_per_1m,
        cache_hit_per_m=settings.price_cache_hit_per_1m,
        output_per_m=settings.price_output_per_1m,
    )


def split_usage(usage: dict | None) -> tuple[int, int, int]:
    """(cache_hit, cache_miss, completion) از usageِ خامِ OpenAI-سازگارِ DeepSeek."""
    u = usage or {}
    prompt = int(u.get("prompt_tokens", 0) or 0)
    completion = int(u.get("completion_tokens", 0) or 0)
    hit = u.get("prompt_cache_hit_tokens")
    miss = u.get("prompt_cache_miss_tokens")
    if hit is None or miss is None:
        hit, miss = 0, prompt
    return int(hit or 0), int(miss or 0), completion


def usage_details(usage: dict | None) -> dict:
    hit, miss, completion = split_usage(usage)
    return {
        "input": miss,
        "input_cache_read": hit,
        "output": completion,
        "total": hit + miss + completion,
    }


def cost_details(usage: dict | None, pricing: Pricing | None = None) -> dict:
    """هزینهٔ دلاری به تفکیک — کلیدها آینهٔ usage_details تا UI درست تجمیع کند."""
    p = pricing or pricing_from_settings()
    hit, miss, completion = split_usage(usage)
    input_cost = miss * p.input_per_m / _MILLION
    cache_cost = hit * p.cache_hit_per_m / _MILLION
    output_cost = completion * p.output_per_m / _MILLION
    return {
        "input": round(input_cost, 8),
        "input_cache_read": round(cache_cost, 8),
        "output": round(output_cost, 8),
        "total": round(input_cost + cache_cost + output_cost, 8),
    }
