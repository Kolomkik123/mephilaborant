"""Result names grounded in explicit symbol definitions in the methodic.

Symbols alone are ambiguous: E may be Young's modulus or electromotive
force. A name is used only when the text links the term to that symbol.
Unsupported or conflicting definitions leave the name empty.
"""
import re
from dataclasses import dataclass

@dataclass(frozen=True)
class QuantityName:
    name: str = ""
    source: str = ""
    evidence: str = ""

# Canonical nominative forms; the source may use another grammatical case.
# These terms do not bind to any particular letter or lab number.
TERMS = (
    ("Ускорение свободного падения", r"ускорени[еяю]\s+свободного\s+падения"),
    ("Ускорение", r"ускорени[еяю]"),
    ("Модуль Юнга", r"модул(?:ь|я|ю|е|ем)\s+Юнга"),
    ("Момент инерции", r"момент(?:ом|а|у|е)?\s+инерции"),
    ("Коэффициент динамической вязкости", r"коэффициент(?:ом|а|у|е)?\s+динамической\s+вязкости"),
    ("Коэффициент вязкости", r"коэффициент(?:ом|а|у|е)?\s+вязкости"),
    ("Динамическая вязкость", r"динамическ(?:ая|ой|ую)\s+вязкост[ьи]"),
    ("Плотность", r"плотност[ьи](?:ю)?"),
    ("Внутреннее сопротивление", r"внутренн(?:ее|его|ему|им|ем)\s+сопротивлени[еяю]"),
    ("Сопротивление", r"сопротивлени[еяю]"),
    ("ЭДС", r"ЭДС|электродвижущ(?:ая|ей|ую)\s+сил[аыуе]"),
    ("Показатель адиабаты", r"показател(?:ь|я|ю|е|ем)\s+адиабаты|отношени[еяю]\s+тепло[её]мкостей"),
    ("Жёсткость", r"ж[её]сткост[ьи](?:ю)?"),
    ("Скорость", r"скорост[ьи](?:ю)?"),
    ("Период", r"период(?:ом|а|у|е)?"),
    ("Энергия", r"энерги[яиюей]+"),
    ("Сила", r"сил[аыуе]"),
    ("Масса", r"масс[аыуе](?:ой)?"),
    ("Объём", r"объ[её]м(?:ом|а|у|е)?"),
    ("Давление", r"давлени[еяю]"),
    ("Температура", r"температур[аыуе](?:ой)?"),
    ("Напряжение", r"напряжени[еяю]"),
)

def infer_quantity_name(symbol: str, texts: list[str], sources: list[str] | None = None) -> QuantityName:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", symbol or ""):
        return QuantityName()
    sources = sources or [""] * len(texts)
    token = re.compile(r"(?<![A-Za-z0-9_])"+re.escape(symbol)+r"(?![A-Za-z0-9_])")
    candidates = []
    for text, source in zip(texts, sources):
        text = " ".join(text.split())
        terms = [(canonical, match) for canonical, pattern in TERMS
                 for match in re.finditer(r"(?<![А-Яа-яЁё])(?:"+pattern+r")(?![А-Яа-яЁё])", text, re.I)]
        for canonical, term in terms:
            if any(other.start() <= term.start() and term.end() <= other.end()
                   and (other.start(),other.end()) != (term.start(),term.end()) for _, other in terms):
                continue
            for match in token.finditer(text):
                if term.end() <= match.start():
                    bridge = text[term.end():match.start()]
                    if len(bridge) > 110 or re.search(r"[A-Za-z,.!?;=]", bridge):
                        continue
                    start, end = term.start(), match.end()
                elif match.end() <= term.start():
                    bridge = text[match.end():term.start()]
                    if not re.fullmatch(r"\s*(?:[—–:-]|это|обозначает)\s*", bridge):
                        continue
                    start, end = match.start(), term.end()
                else:
                    continue
                candidates.append((canonical, len(bridge), source, text[start:end]))
    if not candidates:
        return QuantityName()
    # Suppress a generic term when its more specific definition is present.
    names = {item[0] for item in candidates}
    specific = {name for name in names if not any(name != other and name.casefold() in other.casefold() for other in names)}
    if len(specific) != 1:
        return QuantityName()
    name = next(iter(specific))
    best = min((item for item in candidates if item[0] == name), key=lambda item: item[1])
    return QuantityName(name, best[2], best[3])
