"""Встроенный поиск расчётчика по тексту PDF и точному номеру работы.

Релевантность поиска по теме оценивается по баллу BM25 и покрытию слов."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.data.index import Hit, content_terms, load_index, tokenize
from app.security import sanitize_fragment

# «работа 1.3», «работе №2.1», «лаба 1.3»: номер выделяется целиком.
# По номеру фрагменты выбираются напрямую из метаданных PDF.
WORK_RE = re.compile(r"(?:работ\w*|лаб\w*)\s*№?\s*(\d+[.\-]\d+|\d+)", re.IGNORECASE)


def work_from_question(text: str, fallback: str = "") -> str:
    match = WORK_RE.search(text or "")
    return match.group(1) if match else fallback


# Порог BM25 и доля совпавших значимых слов запроса.
MIN_SCORE = 1.5
MIN_COVERAGE = 0.34


@dataclass
class Retrieval:
    query: str
    hits: list[Hit] = field(default_factory=list)
    relevant: bool = False
    coverage: float = 0.0
    dense_score: float = 0.0

    @property
    def citations(self) -> list[str]:
        return [hit.fragment.citation for hit in self.hits]

    @property
    def pages(self) -> list[int]:
        return [hit.fragment.page for hit in self.hits]

    @property
    def works(self) -> list[str]:
        return [hit.fragment.work for hit in self.hits if hit.fragment.work]

    def as_context(self) -> str:
        """Фрагменты в формате, который понимают и реальная модель, и mock."""
        parts = []
        for hit in self.hits:
            parts.append(
                f"[[FRAGMENT page={hit.fragment.page}]]"
                f"{sanitize_fragment(hit.fragment.text)}[[/FRAGMENT]]"
            )
        return "\n".join(parts)


def coverage(query: str, text: str) -> float:
    """Доля значимых слов запроса, встретившихся во фрагменте."""
    terms = content_terms(query)
    if not terms:
        return 0.0
    found = set(tokenize(text))
    return len(terms & found) / len(terms)


class Retriever:
    def __init__(
        self,
        backend: str | None = None,
        min_score: float = MIN_SCORE,
        min_coverage: float = MIN_COVERAGE,
    ):
        self.index = load_index(backend)
        self.min_score = min_score
        self.min_coverage = min_coverage

    def search(self, query: str, top_k: int = 5, kind: str | None = None) -> Retrieval:
        """Найти фрагменты BM25 и проверить покрытие слов запроса."""
        hits = self.index.search(query, top_k=top_k, kind=kind)
        if not hits:
            return Retrieval(query=query, hits=[], relevant=False)
        best_coverage = max(coverage(query, hit.fragment.text) for hit in hits)
        best_lexical = max(hit.score for hit in hits)
        return Retrieval(
            query=query,
            hits=hits,
            relevant=best_lexical >= self.min_score and best_coverage >= self.min_coverage,
            coverage=best_coverage,
        )

    def fragments_for(
        self,
        work_or_topic: str,
        limit: int = 5,
        work: str = "",
        kind: str = "text",
    ) -> Retrieval:
        """Фрагменты методички по номеру работы или по теме.

        Номер работы сильнее лексической близости: это точное совпадение по
        метаданным, и порог к нему не применяется — работу студент назвал
        сам. Без номера остаётся поиск с порогом.
        """
        if work:
            by_work = self.index.by_work(work, kind=kind)
            if by_work:
                return Retrieval(
                    query=work_or_topic, hits=by_work[:limit], relevant=True
                )
        return self.search(work_or_topic, top_k=limit, kind=kind)

    def questions_for(
        self, work_or_topic: str, limit: int = 5, work: str = ""
    ) -> Retrieval:
        """Контрольные вопросы методички по работе или теме.

        Если номер работы известен, он важнее лексической близости: вопросы
        чужой работы на допуске бесполезны. Поэтому при заданном work сначала
        берутся вопросы этой работы, и только при их нехватке — остальные.

        Возвращается Retrieval, а не список: лексический поиск отдаёт топ-k
        всегда, даже когда ни одно слово запроса не встретилось толком, и
        отличить вопросы по делу от шума можно только порогом. По количеству
        найденного это не отличается: на «спроси меня про довесок» индекс
        честно вернул пять вопросов из пяти разных работ, ни одного по теме.
        """
        return self.fragments_for(
            work_or_topic, limit=limit, work=work, kind="question"
        )