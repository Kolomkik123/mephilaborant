"""Локальный индекс PDF для расчётчика: BM25 без нейросетевой модели.

Это встроенный поиск по словам, не модуль поиска другого участника команды."""

from __future__ import annotations

import json
import math
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.config import settings
from app.data.parse import Fragment, ParsedDoc, parse_dir

TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)

# Грубое отсечение русских окончаний: без морфологии, но заметно поднимает
# совпадения («погрешности» ↔ «погрешность»).
SUFFIXES = (
    "ами", "ями", "ость", "ости", "ения", "ение", "ении", "иями", "ого", "ему",
    "ыми", "ими", "ах", "ях", "ам", "ям", "ой", "ый", "ая", "ое", "ые", "ие",
    "ов", "ев", "ей", "ию", "ия", "ию", "ем", "ом", "ух", "и", "ы", "а", "о",
    "е", "у", "ю", "я", "ь",
)


def normalize(token: str) -> str:
    token = token.lower()
    if len(token) <= 4:
        return token
    for suffix in SUFFIXES:
        if len(token) - len(suffix) >= 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


# Служебные слова. Без их отсечения BM25 даёт постороннему запросу
# ощутимый балл за совпадение предлогов: «рецепт борща с говядиной» получал
# 2.98 — выше, чем некоторые настоящие вопросы по методичке.
STOPWORDS = frozenset(
    """
    и а но или да же ли бы не ни то как что чем чего чему кто кого кому
    в во на за по из от до у о об обо при над под про без для через между
    это этот эта эти тот та те так там тут где когда куда откуда почему зачем
    если чтобы также тоже есть был была были быть будет можно нужно надо
    мой моя мои наш наша наши ваш ваша ваши его её их свой своя свои
    я ты он она оно мы вы они себя сам сама сами весь вся все всё
    """.split()
)


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    tokens = [normalize(t) for t in TOKEN_RE.findall(text)]
    if keep_stopwords:
        return tokens
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def content_terms(text: str) -> set[str]:
    """Значимые слова запроса — для проверки покрытия найденным фрагментом."""
    return {t for t in tokenize(text) if len(t) > 2}


@dataclass
class Hit:
    fragment: Fragment
    score: float  # лексический балл BM25; у чисто векторного поиска 0
    dense_score: float = 0.0  # косинус к запросу, если считался вектор


class BM25Index:
    """BM25 Okapi. Хранится одним pickle-файлом рядом с индексом."""

    k1 = 1.5
    b = 0.75

    def __init__(self) -> None:
        self.fragments: list[Fragment] = []
        self.tokens: list[Counter[str]] = []
        self.lengths: list[int] = []
        self.df: Counter[str] = Counter()
        self.avg_len = 0.0

    def add(self, fragments: list[Fragment]) -> None:
        for fragment in fragments:
            counts = Counter(tokenize(fragment.text))
            if not counts:
                continue
            self.fragments.append(fragment)
            self.tokens.append(counts)
            self.lengths.append(sum(counts.values()))
            for term in counts:
                self.df[term] += 1
        self.avg_len = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0

    def search(self, query: str, top_k: int = 5, kind: str | None = None) -> list[Hit]:
        terms = tokenize(query)
        if not terms or not self.fragments:
            return []
        total = len(self.fragments)
        scored: list[Hit] = []
        for position, counts in enumerate(self.tokens):
            fragment = self.fragments[position]
            if kind and fragment.kind != kind:
                continue
            length = self.lengths[position]
            score = 0.0
            for term in terms:
                freq = counts.get(term, 0)
                if not freq:
                    continue
                idf = math.log(1 + (total - self.df[term] + 0.5) / (self.df[term] + 0.5))
                denom = freq + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                score += idf * freq * (self.k1 + 1) / denom
            if score > 0:
                scored.append(Hit(fragment=fragment, score=score))
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]

    def by_work(self, work: str, kind: str | None = None) -> list[Hit]:
        """Фрагменты конкретной работы, в порядке методички.

        Лексическим поиском номер работы не найти: «2.1» распадается на
        токены «2» и «1», а в тексте самих контрольных вопросов номера нет.
        Поэтому отбор по работе идёт напрямую по метаданным, а порядок
        сохраняется таким же, как в исходном PDF.
        """
        return [
            Hit(fragment=fragment, score=0.0)
            for fragment in self.fragments
            if fragment.work == work and (kind is None or fragment.kind == kind)
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "BM25Index":
        return pickle.loads(path.read_bytes())


def _bm25_path() -> Path:
    return settings.index_dir / "bm25.pkl"


def _manifest_path() -> Path:
    return settings.index_dir / "manifest.json"


def build_index(
    methodics_dir: Path | None = None, backend: str | None = None
) -> dict[str, object]:
    """Собрать индекс из PDF. Возвращает манифест: что проиндексировано."""
    source_dir = methodics_dir or settings.methodics_dir
    docs: list[ParsedDoc] = parse_dir(source_dir)
    fragments = [f for doc in docs for f in doc.fragments]

    chosen = backend or settings.index_backend
    if chosen not in ("bm25", "auto"):
        raise ValueError("В этой версии доступен INDEX_BACKEND=bm25")

    bm25 = BM25Index()
    bm25.add(fragments)
    bm25.save(_bm25_path())
    manifest = {
        "sources": [doc.source for doc in docs],
        "documents": len(docs),
        "fragments": len(fragments),
        "questions": sum(len(doc.questions) for doc in docs),
        "backends": ["bm25"],
    }
    _manifest_path().parent.mkdir(parents=True, exist_ok=True)
    _manifest_path().write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def load_index(backend: str | None = None) -> BM25Index:
    """Загрузить BM25; при первом запуске собрать индекс из PDF.

    Значение auto из прежних локальных настроек также означает BM25.
    """
    chosen = backend or settings.index_backend
    if chosen not in ("bm25", "auto"):
        raise ValueError("В этой версии доступен INDEX_BACKEND=bm25")
    if not _bm25_path().exists():
        build_index(backend=chosen)
    return BM25Index.load(_bm25_path())


def index_manifest() -> dict[str, object]:
    if _manifest_path().exists():
        return json.loads(_manifest_path().read_text(encoding="utf-8"))
    return {"documents": 0, "fragments": 0, "questions": 0, "backends": []}
