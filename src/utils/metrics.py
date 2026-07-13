"""شمارنده‌ها و زمان‌سنج‌های درون‌پروسه‌ای (thread-safe) — بدون وابستگیِ خارجی.

با GET /metrics به‌صورت JSON خوانده می‌شوند؛ برای پایشِ دستی/اسکریپتی کافی است.
اگر روزی Prometheus در دسترس بود، همین نقاطِ instrument شده با prometheus_client
قابلِ جایگزینی‌اند (فقط این ماژول عوض می‌شود).
"""
from __future__ import annotations

import threading
import time


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._timings: dict[str, dict[str, float]] = {}
        self.started_at = time.time()

    def inc(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def observe_ms(self, name: str, ms: float) -> None:
        """ثبتِ یک اندازه‌گیریِ زمانی (میلی‌ثانیه): count/sum/min/max/last."""
        with self._lock:
            t = self._timings.get(name)
            if t is None:
                self._timings[name] = {"count": 1, "sum_ms": ms, "min_ms": ms, "max_ms": ms, "last_ms": ms}
            else:
                t["count"] += 1
                t["sum_ms"] += ms
                t["min_ms"] = min(t["min_ms"], ms)
                t["max_ms"] = max(t["max_ms"], ms)
                t["last_ms"] = ms

    def snapshot(self) -> dict:
        with self._lock:
            timings = {
                name: {**t, "avg_ms": round(t["sum_ms"] / t["count"], 1)}
                for name, t in ((n, dict(v)) for n, v in self._timings.items())
            }
            return {
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "counters": dict(self._counters),
                "timings": timings,
            }


metrics = Metrics()
