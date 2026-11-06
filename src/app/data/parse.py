"""Парсинг методичек: PDF → фрагменты с номером страницы.

Номер страницы — не украшение: на нём стоит обещание «ссылка на страницу»
и метрика опоры ответа на методичку. Поэтому страница тянется через всю
цепочку от PyMuPDF до ответа студенту.

Marker (заявлен в стеке) подключается опционально: если пакет установлен,
он даёт более чистый текст на сложной вёрстке. Без него работает PyMuPDF.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

CHUNK_CHARS = 900
CHUNK_OVERLAP = 150

# «Работа 1.4», «Лабораторная работа № 3», «2.7. Измерение ...»
WORK_RE = re.compile(
    r"(?:лабораторная\s+работа|работа)\s*№?\s*(\d+[.\-]?\d*)", re.IGNORECASE
)
QUESTIONS_HEAD_RE = re.compile(r"контрольные\s+вопрос", re.IGNORECASE)
QUESTION_ITEM_RE = re.compile(r"^\s*(?:\d+[.)]|[-–—•])\s*(.+)$")


@dataclass
class Fragment:
    """Кусок текста методички с адресом, по которому его можно показать."""

    text: str
    source: str
    page: int
    work: str = ""
    kind: str = "text"  # text | question

    @property
    def id(self) -> str:
        return f"{self.source}|p{self.page}|{self.kind}|{abs(hash(self.text)) % 10**8}"

    @property
    def citation(self) -> str:
        work = f", работа {self.work}" if self.work else ""
        return f"{self.source}{work}, с. {self.page}"


@dataclass
class ParsedDoc:
    source: str
    fragments: list[Fragment] = field(default_factory=list)

    @property
    def questions(self) -> list[Fragment]:
        return [f for f in self.fragments if f.kind == "question"]


def _page_texts(path: Path) -> list[str]:
    """Текст по страницам. Порядок страниц сохраняется."""
    import fitz  # pymupdf

    with fitz.open(path) as doc:
        return [page.get_text("text") for page in doc]


def _chunk(text: str) -> list[str]:
    """Нарезка с перехлёстом, по возможности по границам абзацев."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    if len(text) <= CHUNK_CHARS:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + CHUNK_CHARS)
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if cut > CHUNK_CHARS // 2:
                end = start + cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - CHUNK_OVERLAP)
    return chunks


def _extract_questions(page_text: str) -> list[str]:
    """Контрольные вопросы со страницы, если на ней есть такой раздел."""
    lines = page_text.splitlines()
    out: list[str] = []
    collecting = False
    for line in lines:
        if QUESTIONS_HEAD_RE.search(line):
            collecting = True
            continue
        if not collecting:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        # Новый крупный раздел обрывает список вопросов.
        if WORK_RE.search(stripped) or stripped.isupper() and len(stripped) > 12:
            break
        match = QUESTION_ITEM_RE.match(line)
        if match:
            out.append(match.group(1).strip())
        elif out and len(stripped) > 3 and not stripped[0].isdigit():
            # Продолжение предыдущего вопроса, перенесённое на новую строку.
            out[-1] = f"{out[-1]} {stripped}"
    return [q for q in out if len(q) > 10]


def parse_pdf(path: Path) -> ParsedDoc:
    """PDF → фрагменты текста и контрольные вопросы, с номерами страниц."""
    source = path.stem
    doc = ParsedDoc(source=source)
    current_work = ""

    for page_number, page_text in enumerate(_page_texts(path), start=1):
        if not page_text.strip():
            continue
        work_match = WORK_RE.search(page_text)
        if work_match:
            current_work = work_match.group(1)

        questions = _extract_questions(page_text)

        # Страница, которая почти целиком занята списком контрольных вопросов,
        # не попадает в текстовый индекс: для расчёта нужны определения
        # и формулы, а не повтор вопроса без пояснений.
        questions_share = (
            sum(len(q) for q in questions) / max(1, len(page_text.strip()))
            if questions
            else 0.0
        )
        if questions_share < 0.5:
            for chunk in _chunk(page_text):
                doc.fragments.append(
                    Fragment(
                        text=chunk,
                        source=source,
                        page=page_number,
                        work=current_work,
                        kind="text",
                    )
                )

        for question in questions:
            doc.fragments.append(
                Fragment(
                    text=question,
                    source=source,
                    page=page_number,
                    work=current_work,
                    kind="question",
                )
            )
    return doc


def parse_dir(directory: Path) -> list[ParsedDoc]:
    """Все PDF в каталоге. Пустой каталог — пустой список, не ошибка."""
    if not directory.exists():
        return []
    return [parse_pdf(path) for path in sorted(directory.glob("*.pdf"))]
