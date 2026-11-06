"""Рабочая формула из текста методички.

Зачем. Студент приносит таблицу и номер работы, а формулу обработки не
помнит — она в методичке. Раньше расчётчик в этом случае считал только
прямые измерения, а косвенное требовало, чтобы формулу написал сам студент.

Почему формулу выбирает Python, а не модель. Формула — это то, из чего
получается ответ работы; подсказанная наугад, она даёт правдоподобное
число с правильной размерностью, и проверить его студенту нечем. Поэтому
кандидаты берутся регулярным выражением из найденного фрагмента и
проходят проверку, которую нельзя пройти догадкой:

- выражение разбирается SymPy;
- все свободные символы входят в измеренные столбцы или в известные
  величины (константы вроде pi SymPy знает сама);
- хотя бы один ИЗМЕРЕННЫЙ столбец в формуле есть — формула, собранная из
  одних известных величин, считает теоретическое значение, а не результат
  измерения («a = m*g/(2*M + m)» при заданных массах не использует ни пути,
  ни времени);
- величина слева сама не измерена — иначе это не косвенное измерение, а
  тождество.

Наклон прямой как измеренная величина. В половине работ практикума
ответ получается не из столбца, а из наклона графика: «E = L/(S*k), где
k — наклон графика dL(F)». Наклон измерен не хуже столбца — у него есть
значение и погрешность из МНК, — поэтому расчётчик передаёт его сюда
наравне с измеренными величинами (параметр from_fit). Но буква k в
методичке означает и постоянную Больцмана, и жёсткость пружины, поэтому
формула с наклоном берётся только из фрагмента, где наклон назван
словами: «наклон», «угловой коэффициент», «тангенс угла наклона». Без
этого условия «n = p/(k*T)» молча посчиталось бы через наклон графика.

Известные величины — это g и то, что студент написал в запросе словами:
«R0 = 10 Ом», «rho_s = 7800». В методичке они лежат в таблице установки, а
не в столбцах замеров, и без них не берётся ни мост Уитстона, ни Стокс.
Расчётчик приводит их к СИ до выбора формулы (app/tools/units.py), как и
столбцы таблицы, и печатает переведённое в ответе.

Формулы извлекаются из текстового слоя PDF. Если запись не распознана,
расчётчик просит уточнение; формулы-картинки этот модуль не распознаёт.

Единица результата, объявленная в методичке. В примере обработки рядом с
формулой напечатано, что получилось у авторов: «J = 3.7e-3 кг·м²»,
«g = 9.87 м/с²». Числа оттуда брать нельзя — это чужой опыт, — а вот
единицу можно: она говорит, какой размерности должен быть ответ. Расчётчик
сверяет её с размерностью, которую сам вывел из формулы, и ловит ошибку,
которую больше ловить нечем: единицу в шапке таблицы, перепутанную
согласованно («L, с» вместо «L, м» — размерность сходится, а g выходит в
герцах).

Текст методички — данные, а не код, и в SymPy он не попадает без
фильтра: кандидат с кавычками, двойным подчёркиванием или точкой вне
числа отбрасывается до разбора.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Левая часть — обозначение величины, правая — выражение из латиницы,
# цифр и знаков. Кириллица в набор не входит, поэтому «gamma = Cp/Cv
# называют показателем адиабаты» обрывается перед словом «называют».
# Запятой в наборе тоже нет: в методичке она всегда начинает пояснение
# («rho = 4*m/(pi*d**2*h), где d — диаметр»), а в выражении для SymPy не
# встречается. Заодно негодный кандидат не проглатывает следующий за ним.
ASSIGNMENT_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_]{0,15})\s*=\s*([A-Za-z0-9_+\-*/().\s]{2,80})"
)

# Точка допустима только внутри числа (2.5), двойное подчёркивание и
# кавычки — не допустимы вовсе.
UNSAFE_RE = re.compile(r"__|['\"]|(?<![0-9])\.|\.(?![0-9])")


# Константы, которые не измеряют. Погрешность g (9.78–9.83 в зависимости
# от широты) здесь считается нулевой: на фоне погрешности периода в
# полпроцента она не видна, а подставлять её как измеренную величину
# значит записать результат точнее, чем он есть.
KNOWN_CONSTANTS: dict[str, float] = {"g": 9.81}

# Единицы констант нужны, чтобы вывести размерность результата
# (app/tools/dimensions.py): без «м/с²» у g не считается ни модуль Юнга,
# ни момент инерции.
CONSTANT_UNITS: dict[str, str] = {"g": "м/с²"}

# «R0 = 10 Ом», «rho_s = 7800 кг/м3». Число должно быть отдельным токеном:
# отрицательный просмотр отсекает «g = 4*pi**2*L/T**2», где за четвёркой
# идёт знак умножения, а не единица измерения.
GIVEN_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_]{0,15})\s*=\s*"
    r"([-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)"
    r"(?!\s*[*/^(]|\s*\*\*)"
    r"\s*([A-Za-zА-Яа-я°][A-Za-zА-Яа-я°/0-9\u00b2\u00b3]{0,9})?"
)

# Наклон и свободный член прямой МНК: имена, под которыми они входят в
# формулы методички, и слова, по которым фрагмент опознаётся. Маркер
# обязателен: см. про постоянную Больцмана в начале модуля. Свободный член
# встречается реже наклона, а буква b в методичке чаще означает толщину
# или ширину — поэтому и слова для него отдельные, узкие.
SLOPE_RE = re.compile(r"наклон|угл\w*\s+коэффициент|тангенс\s+угла", re.IGNORECASE)
INTERCEPT_RE = re.compile(
    r"свободн\w*\s+член|отрезок,?\s+отсека\w*|пересечени\w*\s+с\s+ос\w",
    re.IGNORECASE,
)
FIT_MARKERS: dict[str, re.Pattern[str]] = {"k": SLOPE_RE, "b": INTERCEPT_RE}

TRIG_RE = re.compile(r"\b(?:sin|cos|tan|cot|ctg|asin|acos|atan)\s*\(")
DEGREE_RE = re.compile(r"град|°|deg", re.IGNORECASE)


@dataclass
class Given:
    """Величина, которую студент задал в запросе."""

    name: str
    value: float
    unit: str = ""

    @property
    def as_text(self) -> str:
        return f"{self.name} = {self.value:g}{(' ' + self.unit) if self.unit else ''}"


def given_values(text: str) -> list[Given]:
    """Величины, названные в запросе: «R0 = 10 Ом».

    Разбирается только проза: в таблице замеров знака равенства нет, а
    формулу студента из текста убирает сам расчётчик — иначе её левая
    часть попала бы сюда как заданная величина.
    """
    found: dict[str, Given] = {}
    for match in GIVEN_RE.finditer(text or ""):
        name = match.group(1)
        value = float(match.group(2).replace(",", "."))
        found[name] = Given(name=name, value=value, unit=(match.group(3) or "").strip())
    return list(found.values())


# «J = 3.7e-3 кг·м²», «g = 9.87 ± 0.12 м/с²». Погрешность между числом и
# единицей необязательна, а перенос строки между любыми частями —
# обязателен к допущению: в PDF запись рвётся где угодно. Сама единица
# проверяется таблицей размерностей, поэтому набор символов здесь можно
# держать широким: что не разберётся — отбросится.
DECLARED_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_]{0,15})\s*=\s*"
    r"[-+]?\d[\d.,]*(?:[eE][-+]?\d+)?"
    r"(?:\s*±\s*[-+]?\d[\d.,]*(?:[eE][-+]?\d+)?)?"
    r"\s*([А-Яа-яA-Za-z°][А-Яа-яA-Za-z°·/()\^²³0-9-]{0,14})"
)


@dataclass
class Declared:
    """Единица величины, объявленная в методичке, и где это написано."""

    name: str
    unit: str
    source: str = ""


def declared_units(
    texts: list[str], sources: list[str] | None = None
) -> dict[str, Declared]:
    """Единицы величин, объявленные в примерах обработки методички.

    Берётся только то, что разбирается таблицей размерностей: «Основной»
    из «J = 3.7e-3 кг·м². Основной вклад...» не единица и отбрасывается.
    Если одну величину методичка объявляет в двух разных размерностях —
    такая же буква в другом разделе, — сверять по ней нельзя, и она из
    ответа убирается: ложная тревога хуже отсутствия проверки.

    Ссылка хранится отдельно от ссылки на формулу: пример обработки обычно
    на другой странице, и подсунуть студенту не ту страницу нельзя.
    """
    from app.tools.dimensions import unit_exponents

    sources = sources or [""] * len(texts or [])
    found: dict[str, Declared] = {}
    conflicting: set[str] = set()
    for text, source in zip(texts or [], sources):
        for match in DECLARED_RE.finditer(text or ""):
            name = match.group(1)
            unit = match.group(2).strip(" .,;·/-")
            exponents = unit_exponents(unit)
            if exponents is None:
                continue
            if name in found and unit_exponents(found[name].unit) != exponents:
                conflicting.add(name)
                continue
            found.setdefault(name, Declared(name=name, unit=unit, source=source))
    for name in conflicting:
        found.pop(name, None)
    return found


def _trig_over_degrees(expression: str, units: dict[str, str]) -> bool:
    """Угол в градусах под синусом — молча неверный результат.

    SymPy считает тригонометрию в радианах, а студент пишет градусы, как
    они и записаны в методичке. Обычные «град» и «°» расчётчик переводит в
    радианы до выбора формулы, и тогда эта проверка не срабатывает. Она
    остаётся для единицы, которая на градусы похожа, но в таблице перевода
    не нашлась («град (прибл.)»): такую формулу лучше не брать вовсе —
    расхождение не бросится в глаза, а проверить его нечем.
    """
    if not TRIG_RE.search(expression):
        return False
    return any(
        name in expression and DEGREE_RE.search(unit or "")
        for name, unit in (units or {}).items()
    )


@dataclass
class Formula:
    """Разобранная рабочая формула."""

    quantity: str  # что считается: g, rho, E
    expression: str  # то, что считается: 4*pi**2*L/T**2
    symbols: tuple[str, ...]  # измеренные величины, которые в неё входят
    constants: tuple[str, ...] = ()  # известные величины: g, R0, плотности
    source: str = ""  # ссылка на страницу методички
    written: str = ""  # как записано в методичке, до подстановки ступеней
    steps: tuple[tuple[str, str], ...] = ()  # промежуточные: («a», «2*h/t**2»)
    declared_unit: str = ""  # единица результата из примера обработки
    declared_source: str = ""  # где этот пример напечатан
    quantity_name: str = ""
    quantity_name_source: str = ""
    quantity_name_evidence: str = ""

    @property
    def as_text(self) -> str:
        head = f"{self.quantity} = {self.written or self.expression}"
        if not self.steps:
            return head
        chain = ", ".join(f"{name} = {expression}" for name, expression in self.steps)
        return f"{head}, где {chain}"


def _clean(expression: str) -> str:
    """Выражение в одну строку.

    В PDF формула переносится по строкам, и «sqrt(2*g*L*(1 -» без
    склейки обрывается на переносе: именно так и терялась формула
    работы 1.5.
    """
    return " ".join(expression.split()).rstrip(",;. ")


def candidates(text: str) -> list[tuple[str, str]]:
    """Все пары «величина = выражение» из текста, без проверки смысла."""
    found: list[tuple[str, str]] = []
    for match in ASSIGNMENT_RE.finditer(text or ""):
        expression = _clean(match.group(2))
        if not expression or UNSAFE_RE.search(expression):
            continue
        found.append((match.group(1), expression))
    # Some methodics state the processing formula in words and put a
    # numerical example after the colon. Recover only the explicit relation,
    # requiring the slope's symbol k in the same fragment.
    if re.search(r"наклон\w*\s+k\b", text or "", re.I):
        for match in re.finditer(r"равно\s+удвоенному\s+наклону\s*:\s*([A-Za-z][A-Za-z0-9_]*)\s*=", text or "", re.I):
            found.append((match.group(1), "2*k"))
    return found


# Имена, которые остаются за SymPy: функции и pi. Всё остальное в формуле
# методички — обозначение величины, и разбирать его надо как символ, а не
# как объект SymPy с тем же именем. Иначе «E = L/(S*k)» не разбирается
# вовсе: S у SymPy — реестр синглтонов, E — основание натурального
# логарифма, I — мнимая единица. Площадь, путь, ЭДС и ток в методичке
# обозначены именно так, поэтому формулы с ними раньше молча терялись.
SYMPY_NAMES = frozenset(
    "sqrt sin cos tan cot asin acos atan sinh cosh tanh log ln exp abs pi".split()
)

NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _parse(expression: str):
    """Выражение как объект SymPy, где величины — символы, а не константы."""
    import sympy as sp

    names = {
        name for name in NAME_RE.findall(expression) if name not in SYMPY_NAMES
    }
    return sp.sympify(expression, locals={name: sp.Symbol(name) for name in names})


def _free_symbols(expression: str) -> set[str] | None:
    """Свободные символы выражения или None, если SymPy его не разобрал."""
    try:
        parsed = _parse(expression)
    except Exception:
        return None
    if parsed.is_number:
        return set()
    return {str(symbol) for symbol in parsed.free_symbols}


def usable(
    expression: str,
    measured: list[str],
    known: dict[str, float] | None = None,
) -> bool:
    """Можно ли посчитать это выражение по измеренным столбцам.

    Нужно и для формулы студента: «формулу не помню» регулярное выражение
    в запросе ловит как формулу, а SymPy на ней падает. Проверка одна и та
    же — тогда непригодная формула студента не оборачивается ошибкой, а
    уступает место формуле из методички.
    """
    if not expression or UNSAFE_RE.search(expression):
        return False
    symbols = _free_symbols(expression)
    allowed = set(measured) | set(known or KNOWN_CONSTANTS)
    return bool(symbols is not None and symbols <= allowed)


def symbols_of(expression: str) -> list[str]:
    """Свободные символы выражения: что в него вообще входит.

    pi и e SymPy знает сама, поэтому в список они не попадают.
    """
    return sorted(_free_symbols(expression) or ())


def used_constants(expression: str, measured: list[str]) -> list[str]:
    """Известные величины, которые вошли в выражение.

    Нужно, чтобы напечатать подставленное: единицы при подстановке не
    переводятся, и увидеть, что подставилось, должен студент.
    """
    symbols = _free_symbols(expression)
    if not symbols:
        return []
    return sorted(symbols - set(measured))


def _fit_symbol_is_confirmed(text: str, symbols: set[str], from_fit: set[str]) -> bool:
    """Назван ли в этом фрагменте наклон, если формула его использует.

    Проверяется текст того самого фрагмента, из которого взята формула:
    «E = L/(S*k), где k — наклон графика dL(F)» — наклон назван, а в
    «n = p/(k*T)» из раздела про идеальный газ k означает постоянную
    Больцмана, которой у нас нет.
    """
    for name in symbols & from_fit:
        marker = FIT_MARKERS.get(name)
        if marker is None or not marker.search(text or ""):
            return False
    return True


# Сколько промежуточных ступеней подставляется. Одна: «J через a, a через
# h и t» — так записана обработка в методичке, и каждая следующая ступень
# умножает риск подставить не ту величину. Третьей ступени в методичках
# практикума не встретилось.
MAX_CHAIN_STEPS = 1


def _substitute(expression: str, name: str, replacement: str) -> str | None:
    """Выражение, в котором name заменён на replacement.

    Подстановка идёт через SymPy, а не по тексту: «g/a - 1» при
    a = 2*h/t**2 должно стать «g*t**2/(2*h) - 1», а текстовая замена дала
    бы «g/2*h/t**2 - 1» — то же самое с точностью до порядка действий, то
    есть неверно.
    """
    import sympy as sp

    try:
        composed = _parse(expression).subs(sp.Symbol(name), _parse(replacement))
    except Exception:
        return None
    return str(composed)


def _resolve(
    expression: str,
    *,
    allowed: set[str],
    measured_set: set[str],
    pool: list[tuple[str, str, str]],
    fit_set: set[str],
    depth: int,
    resolving: tuple[str, ...] = (),
) -> tuple[str, list[tuple[str, str]]] | None:
    """Выражение без неизвестных символов и ступени, через которые оно получено.

    Момент инерции маховика записан в методичке как J = m*r**2*(g/a - 1), и
    ускорения a в таблице замеров нет: оно само считается из высоты и
    времени падения, a = 2*h/t**2. Такая величина подставляется — но
    только если её собственная формула проходит те же проверки, что и
    рабочая. Главная из них: в ступени должна быть хотя бы одна измеренная
    величина. Без этого подставилось бы «a = 0.119» из примера обработки в
    методичке — число чужого опыта вместо результата студента.
    """
    symbols = _free_symbols(expression)
    if symbols is None:
        return None
    unknown = symbols - allowed
    if not unknown:
        return expression, []
    if depth <= 0:
        return None

    current = expression
    steps: list[tuple[str, str]] = []
    for name in sorted(unknown):
        if name in resolving:
            # Цикл: a выражено через b, b через a. Подставлять нечего.
            return None
        best: tuple[int, str, str, list[tuple[str, str]]] | None = None
        for quantity, candidate, text in pool:
            if quantity != name:
                continue
            inner = _resolve(
                candidate,
                allowed=allowed,
                measured_set=measured_set,
                pool=pool,
                fit_set=fit_set,
                depth=depth - 1,
                resolving=resolving + (name,),
            )
            if inner is None:
                continue
            inner_expression, inner_steps = inner
            inner_symbols = _free_symbols(inner_expression) or set()
            if not inner_symbols & measured_set:
                continue
            if fit_set and not _fit_symbol_is_confirmed(text, inner_symbols, fit_set):
                continue
            weight = len(inner_symbols & measured_set)
            if best is None or weight > best[0]:
                best = (weight, candidate, inner_expression, inner_steps)
        if best is None:
            return None
        _, candidate, inner_expression, inner_steps = best
        composed = _substitute(current, name, inner_expression)
        if composed is None:
            return None
        current = composed
        steps.extend(inner_steps)
        steps.append((name, candidate))

    final = _free_symbols(current)
    if final is None or not final <= allowed:
        return None
    return current, steps


def pick_formula(
    texts: list[str],
    measured: list[str],
    sources: list[str] | None = None,
    known: dict[str, float] | None = None,
    units: dict[str, str] | None = None,
    notes: list[str] | None = None,
    from_fit: tuple[str, ...] | list[str] = (),
) -> Formula | None:
    """Формула, которую можно посчитать по измеренному и известному.

    Из подходящих выбирается та, что использует больше измеренных величин:
    формула со всеми измеренными столбцами — и есть рабочая формула работы,
    а короткая вроде «a = 2*h/t**2» может быть промежуточной.

    from_fit — измеренные величины, полученные не из столбца, а из МНК.
    Они входят в measured, но формула с ними берётся только из фрагмента,
    где наклон назван словами.
    """
    measured_set = set(measured)
    if not measured_set:
        return None
    fit_set = set(from_fit) & measured_set
    known = KNOWN_CONSTANTS if known is None else known
    allowed = measured_set | set(known)

    sources = sources or [""] * len(texts)
    # Все пары «величина = выражение» из найденного: из них берётся и
    # рабочая формула, и промежуточная ступень к ней.
    pool = [
        (quantity, expression, text)
        for text in texts
        for quantity, expression in candidates(text)
    ]

    declared = declared_units(texts, sources)

    best: Formula | None = None
    for text, source in zip(texts, sources):
        for quantity, expression in candidates(text):
            if quantity in measured_set:
                # Слева измеренная величина: это проверочное соотношение,
                # а не то, что нужно получить.
                continue
            own = _free_symbols(expression)
            if own is None:
                continue
            if fit_set and not _fit_symbol_is_confirmed(text, own, fit_set):
                # Буква наклона в формуле есть, а слова «наклон» во
                # фрагменте нет: скорее всего это другая величина.
                continue
            resolved = _resolve(
                expression,
                allowed=allowed,
                measured_set=measured_set,
                pool=pool,
                fit_set=fit_set,
                depth=MAX_CHAIN_STEPS,
            )
            if resolved is None:
                continue
            computable, steps = resolved
            symbols = _free_symbols(computable) or set()
            # Формула без ни одной измеренной величины считает
            # теоретическое значение по заданным — это не обработка
            # измерений.
            from_table = symbols & measured_set
            if not from_table:
                continue
            if _trig_over_degrees(computable, units or {}):
                # Молчать тут нельзя: формула в методичке есть, и студент
                # вправе знать, почему она не взята.
                if notes is not None:
                    notes.append(
                        f"{quantity} = {expression}: в таблице угол в градусах, "
                        "а в формуле он под синусом — SymPy считает радианы. "
                        "Переведите угол или напишите формулу сами"
                    )
                continue
            from app.tools.quantities import infer_quantity_name
            label = infer_quantity_name(quantity, [text], [source])
            if not label.name:
                label = infer_quantity_name(quantity, texts, sources)
            formula = Formula(
                quantity=quantity,
                expression=computable,
                symbols=tuple(sorted(from_table)),
                constants=tuple(sorted(symbols - measured_set)),
                source=source,
                written=expression,
                steps=tuple(steps),
                declared_unit=(
                    declared[quantity].unit if quantity in declared else ""
                ),
                declared_source=(
                    declared[quantity].source if quantity in declared else ""
                ),
                quantity_name=label.name,
                quantity_name_source=label.source,
                quantity_name_evidence=label.evidence,
            )
            if best is None or len(formula.symbols) > len(best.symbols):
                best = formula
    return best
