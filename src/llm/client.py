"""
Thin wrapper around DeepSeek (OpenAI-SDK compatible).

Responsibilities: API calls, JSON mode, retry on network/parse errors,
self-consistency, and returning metadata (latency and token usage) for
logging and cost analysis. No business logic lives here.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field

from openai import OpenAI

from config.settings import settings
from src.utils.logging import get_logger

log = get_logger("llm.client")


@dataclass
class LLMResponse:
    """The result of a single call plus metadata (for logging)."""
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
        """One JSON-mode call. Returns an LLMResponse; retries on failure."""
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
                data = json.loads(content)  # JSON mode only guarantees valid syntax
                usage = resp.usage.model_dump() if resp.usage else {}
                return LLMResponse(
                    data=data, model=model, latency_ms=latency_ms, usage=usage, raw=content
                )
            except json.JSONDecodeError as e:
                last_err = e
                log.warning("Output was not valid JSON (attempt %d/%d).", attempt, settings.max_retries)
            except Exception as e:  # network/API error
                last_err = e
                log.warning("API call error (attempt %d/%d): %s", attempt, settings.max_retries, e)
            time.sleep(min(2 ** attempt, 8))  # simple exponential backoff

        raise RuntimeError(f"DeepSeek call failed after {settings.max_retries} attempts: {last_err}")

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
        Self-consistency: sample n times and vote over each response's data.
        key_fn(dict) -> a comparable key (e.g. the tuple of labels).
        Returns: (winning response, agreement ratio in 0..1).
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
