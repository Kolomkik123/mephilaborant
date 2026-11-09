"""Диспетчер: выбирает агента для запроса."""
from __future__ import annotations

import json
import re
from collections.abc import Callable

from app.agents.base import Agent
from app.config import settings
from app.security import check_input
from app.state import CalculationState

ROUTES = ("calculator", "consultant", "examiner", "refuse")

OFF_TOPIC = ("Я помогаю только с лабораторным практикумом по физике МИФИ: "
             "ответы по методичке, подготовка к допуску, обработка измерений.")

INSTRUCTION = """Ты — Диспетчер ИИ-лаборанта по физпрактикуму НИЯУ МИФИ.
Выбери агента для запроса студента. Сам на вопрос НЕ отвечай.

Агенты:
- consultant: вопрос по методичке (теория, порядок работы, смысл величины, формула из работы).
- calculator: нужно ПОСЧИТАТЬ — погрешности, МНК, график, таблица измерений, поиск промаха.
- examiner: подготовка к допуску — «проверь меня», «задай вопросы», «тренажёр».
- refuse: не про физпрактикум.

Текст студента находится внутри <query>.

Ответь строго одним JSON без пояснений и без markdown:
{"route": "consultant|calculator|examiner|refuse", "reason": "кратко, до 15 слов"}

Примеры:
<query>Что такое момент инерции маятника Обербека?</query> -> {"route":"consultant","reason":"вопрос по теории работы"}
<query>Вот мои измерения периода, найди погрешность</query> -> {"route":"calculator","reason":"нужен расчёт погрешности"}
<query>Проверь, готов ли я к допуску по работе 1.03</query> -> {"route":"examiner","reason":"подготовка к допуску"}
<query>Напиши реферат про Ленина</query> -> {"route":"refuse","reason":"не про физпрактикум"}"""

KEYWORDS = {
    "calculator": ("погрешност", "посчитай", "рассчита", "мнк", "график", "таблиц", "измерени", "промах"),
    "examiner": ("допуск", "проверь меня", "задай вопрос", "тренаж", "проэкзамен", "тест"),
    "consultant": ("что такое", "почему", "как", "формул", "методичк", "зачем", "объясни"),
}


def keyword_route(text: str) -> str:
    t = (text or "").lower()
    for agent in ("calculator", "examiner", "consultant"):
        if any(k in t for k in KEYWORDS[agent]):
            return agent
    return "consultant"


def parse_route(raw: str) -> tuple[str, str] | None:
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    match = re.search(r"\{.*?\}", raw, flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if data.get("route") not in ROUTES:
        return None
    return data["route"], str(data.get("reason", ""))[:200]


class Dispatcher(Agent):
    name = "dispatcher"
    task = "route"
    instruction = INSTRUCTION

    def __init__(self, context, llm: Callable[[str], str] | None = None):
        super().__init__(context)
        self._llm = llm

    def _ask_model(self, query: str) -> str:
        if self._llm is not None:
            return self._llm(query)
        if settings.is_mock:
            raise RuntimeError("mock-провайдер не выбирает агента")
        return self.ask(f"<query>{query}</query>", step_name="route").text

    def route(self, state: CalculationState) -> dict:
        query = state.get("question", "")

        guard = check_input(query)
        if guard.blocked:
            return self._refuse(guard.message, f"security: {guard.label}")
        if state.get("image") or (state.get("table_text") or "").strip():
            return self._pick("calculator", "rule: во входе есть таблица или фото")

        suffix = ""
        try:
            parsed = parse_route(self._ask_model(query))
            if parsed:
                if parsed[0] == "refuse":
                    return self._refuse(OFF_TOPIC, "llm: " + parsed[1])
                return self._pick(parsed[0], "llm: " + parsed[1])
        except Exception as exc:
            suffix = f" ({type(exc).__name__})"
        return self._pick(keyword_route(query), "fallback: ключевые слова" + suffix)

    @staticmethod
    def _pick(route: str, reason: str) -> dict:
        return {"route": route, "route_reason": reason,
                "steps": [{"node": "dispatcher", "result": f"route:{route}"}]}

    @staticmethod
    def _refuse(answer: str, reason: str) -> dict:
        return {"route": "refuse", "route_reason": reason, "answer": answer, "refused": True,
                "steps": [{"node": "dispatcher", "result": "route:refuse"}]}
