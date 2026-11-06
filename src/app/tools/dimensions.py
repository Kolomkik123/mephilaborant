"""Размерность результата: что получается из формулы.

Зачем. Расчётчик до этого говорил «все величины расчёта приведены к СИ,
значит и результат в СИ» — правда, но студенту в отчёт нужна единица:
«(3.90 ± 0.11)e+11 Па». Единица не приписывается по работе и не угадывается
по порядку величины, а выводится из самой формулы и единиц того, что в неё
подставлено.

Заодно это проверка. Размерности в сумме должны сходиться: в «g/a - 1»
отношение ускорений безразмерно, и единица вычитается из единицы. Если не
сходится — неверна либо формула, либо единицы столбцов, и расчётчик об этом
говорит. Проверяется и аргумент синуса: он обязан быть безразмерным, иначе
формула взята не та.

Как. Единица разбирается в вектор показателей по базису (кг, м, с, А, К) —
«м/Н» это s²/kg, — а выражение обходится по дереву SymPy: произведение
складывает векторы, степень умножает, сумма требует равенства, функция
требует безразмерного аргумента. Моль и канделу в базис не включал: в
механическом и электрическом практикуме они не встречаются, а неполный
базис лучше неверного — незнакомая единица даёт «не определить», а не
молчаливую ошибку.

Чего модуль не делает. Он не приводит единицу к той, которую ждёт
методичка: «кг·м²» и «кг·м^2» для него одно и то же, а «Дж» и «Н·м» —
одинаковая размерность с разными названиями, и выбор между ними наш, а не
выведенный.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

# Базис: килограмм, метр, секунда, ампер, кельвин.
BASE = ("кг", "м", "с", "А", "К")
ZERO = (Fraction(0),) * len(BASE)

Exponents = tuple[Fraction, ...]


def _vector(**powers: int) -> Exponents:
    return tuple(Fraction(powers.get(name, 0)) for name in ("кг", "м", "с", "А", "К"))


# Единица → показатели. Ключи — те единицы СИ, которые отдаёт
# app/tools/units.py, плюс названные производные, которые встречаются в
# практикуме. Порядок важен только для показа: из нескольких названий с
# одной размерностью берётся первое совпадение в NAMED.
UNITS: dict[str, Exponents] = {
    "кг": _vector(кг=1),
    "м": _vector(м=1),
    "с": _vector(с=1),
    "А": _vector(А=1),
    "К": _vector(К=1),
    # безразмерные
    "рад": ZERO,
    "": ZERO,
    # геометрия и кинематика
    "м²": _vector(м=2),
    "м³": _vector(м=3),
    "м/с": _vector(м=1, с=-1),
    "м/с²": _vector(м=1, с=-2),
    "кг/м³": _vector(кг=1, м=-3),
    "кг·м²": _vector(кг=1, м=2),
    # механика
    "Н": _vector(кг=1, м=1, с=-2),
    "Па": _vector(кг=1, м=-1, с=-2),
    "Дж": _vector(кг=1, м=2, с=-2),
    "Вт": _vector(кг=1, м=2, с=-3),
    "Гц": _vector(с=-1),
    "Н/м": _vector(кг=1, с=-2),
    "Па·с": _vector(кг=1, м=-1, с=-1),
    # электричество
    "Кл": _vector(с=1, А=1),
    "В": _vector(кг=1, м=2, с=-3, А=-1),
    "Ом": _vector(кг=1, м=2, с=-3, А=-2),
    "Ф": _vector(кг=-1, м=-2, с=4, А=2),
    "Тл": _vector(кг=1, с=-2, А=-1),
    "Вб": _vector(кг=1, м=2, с=-2, А=-1),
    "Ом·м": _vector(кг=1, м=3, с=-3, А=-2),
    "В/м": _vector(кг=1, м=1, с=-3, А=-1),
    # теплота
    "Дж/К": _vector(кг=1, м=2, с=-2, К=-1),
    "Дж/(кг·К)": _vector(м=2, с=-2, К=-1),
}

# Названия, которыми стоит подписывать результат, если размерность совпала
# в точности. Проверяются в этом порядке: «Дж» раньше «Н·м» (их размерность
# одна), «Па» раньше «Н/м²».
NAMED = (
    "м",
    "кг",
    "с",
    "А",
    "К",
    "Н",
    "Па",
    "Дж",
    "Вт",
    "Гц",
    "В",
    "Ом",
    "Кл",
    "Ф",
    "Тл",
    "Вб",
    "м²",
    "м³",
    "м/с",
    "м/с²",
    "кг/м³",
    "кг·м²",
    "Н/м",
    "Па·с",
    "Ом·м",
    "В/м",
    "Дж/К",
    "Дж/(кг·К)",
)

SUPERSCRIPT = {"²": "2", "³": "3", "⁻": "-"}
FACTOR_RE = re.compile(r"([^·*\s]+)")


def _parse_power(token: str) -> tuple[str, Fraction] | None:
    """«м2», «м²», «с^-1» → (единица, показатель)."""
    token = "".join(SUPERSCRIPT.get(char, char) for char in token.strip())
    if not token:
        return None
    match = re.fullmatch(r"(.+?)\^?(-?\d+(?:/\d+)?)", token)
    if match and match.group(1) in UNITS:
        return match.group(1), Fraction(match.group(2))
    if token in UNITS:
        return token, Fraction(1)
    return None


def unit_exponents(unit: str) -> Exponents | None:
    """Показатели составной единицы: «м/Н» → с²/кг.

    Разбирается произведение в числителе и знаменателе: «кг·м2/(с3·А)».
    Незнакомая единица даёт None — и выше это становится «размерность не
    определить», а не молчаливым «наверное, безразмерная».
    """
    text = (unit or "").strip()
    if text in UNITS:
        return UNITS[text]
    numerator, _, denominator = text.partition("/")
    result = list(ZERO)
    for part, sign in ((numerator, 1), (denominator, -1)):
        for token in FACTOR_RE.findall(part.replace("(", "").replace(")", "")):
            parsed = _parse_power(token)
            if parsed is None:
                return None
            name, power = parsed
            for index, value in enumerate(UNITS[name]):
                result[index] += sign * power * value
    return tuple(result)


@dataclass
class Dimension:
    """Размерность величины в базисе (кг, м, с, А, К)."""

    exponents: Exponents

    @property
    def is_dimensionless(self) -> bool:
        return all(power == 0 for power in self.exponents)

    @property
    def as_text(self) -> str:
        """Единица: названная, если совпала точно, иначе составленная."""
        if self.is_dimensionless:
            return ""
        for name in NAMED:
            if UNITS[name] == self.exponents:
                return name
        return self._composed

    @property
    def _composed(self) -> str:
        def power(name: str, value: Fraction) -> str:
            value = abs(value)
            if value == 1:
                return name
            if value.denominator == 1:
                return f"{name}{self._superscript(value.numerator)}"
            return f"{name}^{value}"

        top = [power(n, p) for n, p in zip(BASE, self.exponents) if p > 0]
        bottom = [power(n, p) for n, p in zip(BASE, self.exponents) if p < 0]
        head = "·".join(top) if top else "1"
        if not bottom:
            return head
        tail = "·".join(bottom)
        return f"{head}/{tail}" if len(bottom) == 1 else f"{head}/({tail})"

    @staticmethod
    def _superscript(value: int) -> str:
        digits = {"2": "²", "3": "³", "4": "⁴", "5": "⁵"}
        return "".join(digits.get(char, f"^{char}") for char in str(value))


@dataclass
class UnitResult:
    """Что удалось сказать о размерности результата."""

    dimension: Dimension | None = None
    problem: str = ""

    @property
    def known(self) -> bool:
        return self.dimension is not None and not self.problem

    @property
    def as_text(self) -> str:
        if not self.known:
            return ""
        return self.dimension.as_text


class _Inconsistent(Exception):
    """Размерности в формуле не сходятся — это находка, а не сбой."""


FUNCTIONS = ("sin", "cos", "tan", "cot", "asin", "acos", "atan", "log", "exp")


def _walk(node, units: dict[str, str]) -> Exponents | None:
    import sympy as sp

    # is_number, а не is_Number: pi у SymPy не Number, а NumberSymbol, и по
    # строгой проверке размерность «4*pi**2*L/T**2» не определялась вовсе.
    if node.is_number:
        return ZERO
    if node.is_Symbol:
        name = str(node)
        if name not in units:
            return None
        return unit_exponents(units[name])
    if node.is_Mul:
        total = list(ZERO)
        for argument in node.args:
            part = _walk(argument, units)
            if part is None:
                return None
            total = [a + b for a, b in zip(total, part)]
        return tuple(total)
    if node.is_Pow:
        base = _walk(node.base, units)
        if base is None:
            return None
        if not node.exp.is_Rational:
            if all(power == 0 for power in base):
                return ZERO
            raise _Inconsistent(
                "размерная величина возводится в степень, которая не "
                "рациональное число — проверьте формулу"
            )
        power = Fraction(int(node.exp.p), int(node.exp.q))
        return tuple(value * power for value in base)
    if node.is_Add:
        parts = [_walk(argument, units) for argument in node.args]
        if any(part is None for part in parts):
            return None
        first = parts[0]
        for part, argument in zip(parts[1:], node.args[1:]):
            if part != first:
                left = Dimension(first).as_text or "безразмерная"
                right = Dimension(part).as_text or "безразмерная"
                raise _Inconsistent(
                    f"складываются {left} и {right} ({node.args[0]} и {argument})"
                )
        return first
    if isinstance(node, sp.Function) and type(node).__name__ in FUNCTIONS:
        argument = _walk(node.args[0], units)
        if argument is None:
            return None
        if any(power != 0 for power in argument):
            raise _Inconsistent(
                f"{type(node).__name__} считается от размерной величины "
                f"({Dimension(argument).as_text})"
            )
        return ZERO
    return None


def result_unit(expression: str, units: dict[str, str]) -> UnitResult:
    """Размерность того, что даёт выражение, по единицам его величин.

    units — единица каждой величины выражения в СИ. Если единицы какой-то
    величины нет или она незнакомая, размерность не определяется: пустой
    ответ честнее угаданного.
    """
    from app.tools.formulas import _parse

    try:
        parsed = _parse(expression)
    except Exception:
        return UnitResult()
    try:
        exponents = _walk(parsed, units)
    except _Inconsistent as problem:
        return UnitResult(problem=str(problem))
    if exponents is None:
        return UnitResult()
    return UnitResult(dimension=Dimension(exponents))
