"""Вход и результат самостоятельного расчётчика."""
from __future__ import annotations

from typing import Any, Literal, TypedDict


class CalculationState(TypedDict, total=False):
    question: str
    student_id: str
    image: bytes | None
    table_text: str
    mode: Literal["repeat", "fit"]
    x_column: str
    y_column: str
    instrument_errors: dict[str, dict[str, Any]]
    work: str
    answer: str
    citations: list[str]
    attachments: list[str]
    payload: dict[str, Any]
    steps: list[dict[str, Any]]
    refused: bool
    error: str
