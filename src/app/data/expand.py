"""Переформулировка запроса для поиска формулы через текстовый провайдер.

Используется при неудачном первом поиске. В mock-режиме отключена."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.llm.provider import LLMError, get_provider

INSTRUCTION = """[[TASK:expand]] Студент спрашивает про лабораторную работу
по физике своими словами. Перепиши его вопрос терминами, которыми это
называется в методичке, чтобы по нему нашёлся нужный раздел.

Правила:
- дай от одной до трёх переформулировок, от самой близкой к исходной;
- заменяй бытовые слова на принятые: «довесок» — перегрузок, «колесо» —
  маховик, «батарейка» — источник тока; это примеры, а не весь список;
- сохраняй смысл вопроса, ничего не додумывай и не отвечай на него;
- если вопрос и так написан терминами, верни пустой список.

Верни строго JSON: {"queries": ["...", "..."]}."""


@dataclass
class Expansion:
    original: str
    variants: list[str] = field(default_factory=list)
    source: str = ""  # llm | none
    error: str = ""

    @property
    def available(self) -> bool:
        return bool(self.variants)


def expand_query(query: str, limit: int = 3) -> Expansion:
    """Переформулировки запроса терминами методички."""
    provider = get_provider()
    if provider.name == "mock":
        return Expansion(
            original=query,
            source="none",
            error="mock не умеет переформулировать: нужен реальный провайдер",
        )

    try:
        result = provider.chat(
            [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": query},
            ],
            json_out=True,
        )
        payload = result.json()
    except (LLMError, ValueError) as exc:
        return Expansion(original=query, source="none", error=str(exc))

    variants = [
        str(item).strip()
        for item in (payload.get("queries") or [])
        if str(item).strip() and str(item).strip().lower() != query.strip().lower()
    ]
    return Expansion(original=query, variants=variants[:limit], source="llm")
