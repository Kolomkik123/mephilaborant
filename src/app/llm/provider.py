"""Подключение внешних моделей для расчётчика.

mock — работа без сети; openrouter — внешний API; local — явно настроенный
OpenAI-совместимый сервер. Автоматического переключения между ними нет.
Числа считает Python, модели используются для распознавания и поиска."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import settings

# Приблизительные тарифы за миллион токенов, USD: (вход, выход).
# Это локальная оценка, не актуальный счёт провайдера. Команда provider
# проверяет текстовый запрос, но не обновляет тарифы.
PRICE_PER_MTOK_USD: dict[str, tuple[float, float]] = {
    "qwen/qwen3-235b-a22b-2507": (0.09, 0.55),
    "qwen/qwen3-235b-a22b-thinking-2507": (0.45, 3.50),
    "qwen/qwen3-vl-235b-a22b-instruct": (0.21, 1.90),
}
DEFAULT_PRICE_USD = (0.20, 0.80)


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    latency_s: float = 0.0
    cost_rub: float = 0.0
    raw: Any = None

    def json(self) -> Any:
        """Разбор ответа как JSON с терпимостью к обёртке ```json."""
        text = self.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())


def _cost_rub(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price_in, price_out = PRICE_PER_MTOK_USD.get(model, DEFAULT_PRICE_USD)
    usd = (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000
    return usd * settings.usd_rub


class LLMError(RuntimeError):
    pass


class BaseProvider:
    name = "base"

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        images: list[bytes] | None = None,
        json_out: bool = False,
        temperature: float = 0.2,
    ) -> LLMResult:
        raise NotImplementedError


class OpenAICompatibleProvider(BaseProvider):
    """OpenRouter и llama.cpp — оба говорят протоколом OpenAI."""

    def __init__(self, *, base_url: str, api_key: str, model: str, vision_model: str, name: str):
        from openai import OpenAI  # импорт внутри: в mock-режиме пакет не нужен

        self.name = name
        self.model = model
        self.vision_model = vision_model
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",
            timeout=settings.request_timeout_s,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        images: list[bytes] | None = None,
        json_out: bool = False,
        temperature: float = 0.2,
    ) -> LLMResult:
        import base64

        model = self.vision_model if images else self.model
        payload = [dict(m) for m in messages]
        if images:
            last = payload[-1]
            content: list[dict[str, Any]] = [{"type": "text", "text": str(last.get("content", ""))}]
            for blob in images:
                mime = "image/jpeg" if blob.startswith(b"\xff\xd8\xff") else "image/png"
                b64 = base64.b64encode(blob).decode("ascii")
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                )
            last["content"] = content

        kwargs: dict[str, Any] = {"model": model, "messages": payload, "temperature": temperature}
        if json_out:
            kwargs["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # сеть, лимит, ключ — наверх одним типом
            raise LLMError(f"{self.name}: {exc}") from exc
        latency = time.perf_counter() - started

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        return LLMResult(
            text=response.choices[0].message.content or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model,
            latency_s=latency,
            cost_rub=_cost_rub(model, prompt_tokens, completion_tokens),
            raw=response,
        )


class FallbackProvider(BaseProvider):
    """Основной провайдер с подстраховкой.

    Цепочка используется только при явном создании этого класса.
    get_provider не включает её автоматически.
    """

    def __init__(self, primary: BaseProvider, reserves: list[BaseProvider]):
        self.primary = primary
        self.reserves = reserves
        self.name = primary.name
        self.fallbacks_used = 0

    def chat(self, messages, *, images=None, json_out=False, temperature=0.2) -> LLMResult:
        chain = [self.primary, *self.reserves]
        last_error: Exception | None = None
        for provider in chain:
            try:
                result = provider.chat(
                    messages, images=images, json_out=json_out, temperature=temperature
                )
                if provider is not self.primary:
                    self.fallbacks_used += 1
                return result
            except LLMError as exc:
                last_error = exc
        raise LLMError(f"все провайдеры недоступны: {last_error}")


_cache: dict[str, BaseProvider] = {}


def get_provider(force: str | None = None) -> BaseProvider:
    """Провайдер по настройке. Кэшируется: клиент создаётся один раз."""
    kind = force or settings.provider
    if kind in _cache:
        return _cache[kind]

    if kind == "mock":
        from app.llm.mock import MockProvider

        provider: BaseProvider = MockProvider()
    elif kind == "openrouter":
        if not settings.openrouter_key:
            raise LLMError(
                "LLM_PROVIDER=openrouter, но OPENROUTER_API_KEY пуст. "
                "Положите ключ в .env или поставьте LLM_PROVIDER=mock."
            )
        from app.llm.mock import MockProvider

        primary = OpenAICompatibleProvider(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_key,
            model=settings.model,
            vision_model=settings.vision_model,
            name="openrouter",
        )
        # A real model failure must not silently turn into a mock result.
        provider = primary
    elif kind == "local":
        provider = OpenAICompatibleProvider(
            base_url=settings.local_base_url,
            api_key="",
            model=settings.local_model,
            vision_model=settings.local_model,
            name="local",
        )
    else:
        raise LLMError(f"неизвестный LLM_PROVIDER: {kind}")

    _cache[kind] = provider
    return provider
