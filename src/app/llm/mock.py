"""Провайдер для работы расчётчика без сети и ключей.

Вычисления выполняет Python. Mock возвращает явный отказ для фотографий
и неподдерживаемых запросов; качество распознавания он не имитирует.
"""
from __future__ import annotations

import json
import re
from typing import Any
from app.llm.provider import BaseProvider, LLMResult

TASK_RE = re.compile(r"\[\[TASK:(\w+)\]\]")


class MockProvider(BaseProvider):
    name = "mock"

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        images: list[bytes] | None = None,
        json_out: bool = False,
        temperature: float = 0.2,
    ) -> LLMResult:
        system = " ".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "system"
        )
        user = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content")
                user = content if isinstance(content, str) else str(content)
                break

        match = TASK_RE.search(system)
        task = match.group(1) if match else "answer"
        handler = getattr(self, f"_task_{task}", self._unsupported)
        text = handler(user, system, images)

        # Токены считаем грубо, по длине: нужен не точный счёт, а работающая
        # обвязка метрик времени и цены.
        prompt_tokens = max(1, (len(system) + len(user)) // 4)
        completion_tokens = max(1, len(text) // 4)
        return LLMResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model="mock",
            latency_s=0.0,
            cost_rub=0.0,
        )

    def _unsupported(self, user: str, system: str, images) -> str:
        return json.dumps({"error": "mock_unavailable",
                           "message": "Mock не генерирует ответы модели. Используйте расчёт по числовой таблице."},
                          ensure_ascii=False)

    def _task_vision_table(self, user: str, system: str, images) -> str:
        """Разбор фото таблицы.

        Без модели зрения честно отказываемся: выдумать числа здесь — худшее,
        что может сделать расчётный контур.
        """
        return json.dumps(
            {
                "error": "vision_unavailable",
                "message": (
                    "Mock-режим не распознаёт фото. Введите таблицу числами "
                    "или настройте провайдер с моделью зрения."
                ),
            },
            ensure_ascii=False,
        )
