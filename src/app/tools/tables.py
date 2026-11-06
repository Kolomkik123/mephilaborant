"""Разбор таблицы измерений из текста и из фото.

Текстовый путь — основной и всегда рабочий: студент вставляет таблицу в чат.
Путь через фото (Qwen3-VL) — удобство, но не единственный вход. Это закрытие
риска «фото таблиц распознаётся плохо» из программы работ: при отказе зрения
контур Расчётчика продолжает работать.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

SEPARATOR_RE = re.compile(r"[\t;|]|\s{2,}|,(?!\d)")
# Шапка делится только по явным разделителям: запятая в ней — часть
# обозначения единицы («F, Н»), а не граница столбца.
HEADER_SEPARATOR_RE = re.compile(r"[\t;|]|\s{2,}")
NUMBER_RE = re.compile(r"^[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?$")


@dataclass
class Table:
    columns: dict[str, list[float]] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    source: str = "text"
    warnings: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not any(self.columns.values())

    def summary(self) -> str:
        parts = []
        for name, values in self.columns.items():
            unit = f" [{self.units.get(name, '')}]" if self.units.get(name) else ""
            parts.append(f"{name}{unit}: {len(values)} знач.")
        return "; ".join(parts) if parts else "таблица пуста"


def _to_float(token: str) -> float | None:
    token = token.strip().replace(" ", "").replace(" ", "").strip(".;:")
    if not token or not NUMBER_RE.match(token):
        return None
    return float(token.replace(",", "."))


def _split(line: str, header: bool = False) -> list[str]:
    # Preserve empty cells: dropping one would shift subsequent measurements.
    if any(separator in line for separator in ("\t", ";", "|")):
        return [cell.strip() for cell in re.split(r"[\t;|]", line.strip().strip("|"))]
    if not header:
        # Строка из одних чисел может быть разделена одиночными пробелами:
        # «1.0 2.0» — это две колонки, а не одно значение.
        plain = [cell for cell in re.split(r"\s+|;|\|", line.strip()) if cell]
        if len(plain) > 1 and all(_to_float(cell) is not None for cell in plain):
            return plain
    pattern = HEADER_SEPARATOR_RE if header else SEPARATOR_RE
    return [cell.strip() for cell in pattern.split(line.strip()) if cell.strip()]


def _parse_header_cell(cell: str) -> tuple[str, str]:
    """«T, с» или «T [с]» → ('T', 'с')."""
    match = re.match(r"^(.+?)\s*[,\[(]\s*([^\])]+)[\])]?$", cell)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return cell.strip(), ""


def resolve_column_name(requested: str, available: list[str]) -> str:
    """Resolve display variants without changing measured values or symbols.

    Exact matches win (T and t may be separate quantities). Case-insensitive
    fallback is used only when there is exactly one possible column.
    """
    if requested in available:
        return requested
    def normalized(name):
        symbol, _ = _parse_header_cell(name)
        return re.sub(r"\s+", "", symbol).translate(str.maketrans({"²":"^2","³":"^3"})).replace("**", "^")
    wanted = normalized(requested)
    matches = [name for name in available if normalized(name) == wanted]
    if not matches:
        matches = [name for name in available if normalized(name).casefold() == wanted.casefold()]
    if len(matches) == 1:
        return matches[0]
    message = "неоднозначное обозначение" if matches else "столбец не найден"
    raise ValueError(f"{requested!r}: {message}. Доступные столбцы: {', '.join(available)}. Укажите точное обозначение.")


PROSE_PREFIX_RE = re.compile(r"^[^\d\n]{0,60}?:\s*(?=\S)")


def _strip_prose_prefix(line: str) -> str:
    """Убрать обращение перед таблицей: «Обработай таблицу: T, с 2.01 ...».

    Отсекается только то, что стоит до двоеточия и само не содержит цифр, —
    иначе можно срезать данные.
    """
    match = PROSE_PREFIX_RE.match(line)
    return line[match.end():] if match else line


def count_numbers(line: str) -> int:
    """Сколько чисел в строке. Нужно и здесь, и расчётчику: строка с тремя
    числами — это данные, а не обращение к ассистенту."""
    return sum(1 for cell in re.split(r"[\s;|	]+|,(?!\d)", line) if _to_float(cell) is not None)


def max_number_run(line: str) -> int:
    """Самая длинная череда идущих подряд чисел.

    Отличает строку данных от фразы, в которой числа просто упомянуты:
    «обработай T, с 2.01 1.99 2.00» — череда из трёх, а «работу 2.2,
    rho_s = 7800, rho_f = 1000» — три числа, но вразбивку, по одному между
    словами. Для обеих count_numbers даёт три, и этого мало.
    """
    best = run = 0
    for cell in re.split(r"[\s;|\t]+|,(?!\d)", line):
        if cell and _to_float(cell) is not None:
            run += 1
            best = max(best, run)
        elif cell:
            run = 0
    return best


def _looks_like_prose(line: str) -> bool:
    """Строка — обращение к ассистенту, а не шапка таблицы.

    Шапка состоит из коротких обозначений: «S, м», «t², с²». Фраза
    «Обработай измерения для работы 1.3» в шапку не укладывается ни одной
    ячейкой. По двоеточию это не отличить: номер работы — цифра, и
    PROSE_PREFIX_RE такую строку не трогает.
    """
    cells = _split(line, header=True)
    if not cells:
        return True
    return any(len(cell) > 25 or len(cell.split()) > 3 for cell in cells)


def _data_width(lines: list[str]) -> int:
    """Сколько ячеек в строках данных — по самой частой ширине."""
    widths: dict[int, int] = {}
    for line in lines:
        if count_numbers(line) < 2:
            continue
        width = len(_split(line, header=True))
        widths[width] = widths.get(width, 0) + 1
    if not widths:
        return 0
    return max(widths, key=lambda width: widths[width])


def _wrong_width(lines: list[str]) -> bool:
    """Первая строка не похожа на шапку этих данных, а вторая похожа.

    Признак структурный, и языка не знает: «Обработай, формула m*g*h» —
    одна ячейка, а в данных их две, как и в настоящей шапке «L, м | T, с».
    Поэтому срабатывает и там, где длина строки и число слов ни о чём не
    говорят.
    """
    if len(lines) < 3:
        return False
    if any(_to_float(cell) is not None for cell in _split(lines[1])):
        return False
    width = _data_width(lines[1:])
    if width < 2:
        return False
    return (
        len(_split(lines[0], header=True)) != width
        and len(_split(lines[1], header=True)) == width
    )


def _drop_prose_lines(lines: list[str]) -> list[str]:
    """Снять фразы перед таблицей, оставив сами данные.

    Строку, в которой три числа идут подряд, не снимаем никогда: в ней
    может лежать сама таблица, написанная внутри фразы.
    """
    while len(lines) > 1 and max_number_run(lines[0]) < 3:
        if not any(count_numbers(line) for line in lines[1:]):
            break
        if not (_wrong_width(lines) or _looks_like_prose(lines[0])):
            break
        lines = lines[1:]
    return lines


def _one_line_table(cells: list[str]) -> Table | None:
    """Одна строка вида «T, с 2.01 1.99 2.00» — величина и её измерения."""
    numbers: list[float] = []
    head: list[str] = []
    for cell in cells:
        value = _to_float(cell)
        if value is None:
            if numbers:  # текст после чисел — не наш случай
                return None
            head.append(cell)
        else:
            numbers.append(value)
    if len(numbers) < 3:
        return None

    table = Table()
    if head:
        name, unit = _parse_header_cell(" ".join(head))
        if not unit and len(head) >= 2:
            name, unit = " ".join(head[:-1]), head[-1]
    else:
        name, unit = "x", ""
    table.columns[name] = numbers
    if unit:
        table.units[name] = unit
    return table


def _longest_number_run(text: str, minimum: int = 3) -> Table | None:
    """Самая длинная череда чисел в произвольном тексте.

    Студент пишет в чат фразой: «обработай T, с 2.01 1.99 2.00 2.02, приборная
    погрешность 0.01, формула 4*pi**2*L/T**2». Измерения здесь — самая длинная
    непрерывная череда чисел; одиночные числа в словах про погрешность и
    формулу в неё не попадают.
    """
    tokens = re.split(r"[\s;|\t]+|,(?!\d)", text)
    tokens = [t for t in tokens if t]

    best: tuple[int, int] = (0, 0)  # (начало, длина)
    position = 0
    while position < len(tokens):
        if _to_float(tokens[position]) is None:
            position += 1
            continue
        end = position
        while end < len(tokens) and _to_float(tokens[end]) is not None:
            end += 1
        if end - position > best[1]:
            best = (position, end - position)
        position = end

    start, length = best
    if length < minimum:
        return None

    numbers = [_to_float(t) for t in tokens[start : start + length]]
    head = [t for t in tokens[max(0, start - 2) : start] if _to_float(t) is None]
    name, unit = _parse_header_cell(" ".join(head)) if head else ("x", "")
    if not unit and len(head) >= 2:
        name, unit = " ".join(head[:-1]), head[-1]
    name = re.sub(r"^[^\wа-яё]+|[^\wа-яё]+$", "", name, flags=re.IGNORECASE) or "x"

    table = Table(source="inline")
    table.columns[name] = [n for n in numbers if n is not None]
    if unit:
        table.units[name] = unit
    return table


def parse_text_table(text: str) -> Table:
    """Таблица из текста. Поддержаны столбцы, строки и одна строка чисел."""
    lines = [line for line in text.splitlines() if line.strip()]
    lines = [line for line in lines if not set(line.strip()) <= {"-", "|", "+", " ", "="}]
    if not lines:
        return Table(warnings=["во входе нет строк"])

    lines = _drop_prose_lines(lines)
    lines[0] = _strip_prose_prefix(lines[0])
    rows = [_split(line) for line in lines if line.strip()]
    rows = [row for row in rows if row]
    if not rows:
        return Table(warnings=["во входе нет строк"])

    if len(rows) == 1:
        single = _one_line_table(rows[0])
        if single is not None:
            return single
    first_numeric = all(_to_float(cell) is not None for cell in rows[0])

    table = Table()
    if first_numeric:
        width = max(len(row) for row in rows)
        names = [f"c{i + 1}" for i in range(width)]
        data_rows = rows
    else:
        names, units = [], {}
        for cell in _split(lines[0], header=True):
            name, unit = _parse_header_cell(cell)
            names.append(name)
            if unit:
                units[name] = unit
        table.units = units
        data_rows = rows[1:]

    # Строчная раскладка: «T, с | 2.01 2.00 1.99» — величина в первой ячейке.
    row_layout = (
        not first_numeric
        and len(names) <= 2
        and bool(data_rows)
        and all(len(row) > 2 for row in data_rows if row)
        and all(_to_float(row[0]) is None for row in data_rows if row)
    )
    if row_layout:
        table.columns = {}
        for row in data_rows:
            name, unit = _parse_header_cell(row[0])
            values = [v for v in (_to_float(cell) for cell in row[1:]) if v is not None]
            if values:
                table.columns[name] = values
                if unit:
                    table.units[name] = unit
        return table

    if len(set(names)) != len(names):
        raise ValueError("Повторяются имена столбцов: задайте уникальные обозначения.")
    if len(names) > 1:
        for row_number, row in enumerate(data_rows, start=2):
            if len(row) != len(names) or any(_to_float(cell) is None for cell in row):
                raise ValueError(
                    f"Строка {row_number}: пропуск или нечисловая ячейка. "
                    "Исправьте строку целиком; пары x/y нельзя сдвигать."
                )
    for position, name in enumerate(names):
        values: list[float] = []
        for row in data_rows:
            if position >= len(row):
                continue
            value = _to_float(row[position])
            if value is not None:
                values.append(value)
        table.columns[name] = values

    if not any(table.columns.values()):
        # Последняя попытка: таблица написана внутри фразы.
        inline = _longest_number_run(text)
        if inline is not None:
            return inline

    lengths = {len(v) for v in table.columns.values() if v}
    if len(lengths) > 1:
        table.warnings.append(
            f"столбцы разной длины: {sorted(lengths)} — проверьте пропуски"
        )
    table.columns = {name: values for name, values in table.columns.items() if values}
    if not table.columns:
        table.warnings.append("не найдено ни одного числового столбца")
    return table


def parse_photo_table(image: bytes, hint: str = "") -> Table:
    """Фото таблицы → Table через модель зрения (Qwen3-VL).

    При любом сбое возвращается пустая таблица с предупреждением: расчёт на
    выдуманных числах недопустим, лучше попросить ввести таблицу руками.
    """
    from app.llm.provider import LLMError, get_provider

    system = (
        "[[TASK:vision_table]] Ты распознаёшь таблицы измерений с фотографии "
        "лабораторного журнала. Верни строго JSON вида "
        '{"columns": {"T": [2.01, 2.00]}, "units": {"T": "с"}} '
        "без пояснений. Числа переноси как есть, ничего не досчитывай. "
        "В columns ключи — только обозначения величин, без единиц: "
        "заголовок 'F, Н' запиши как columns.F и units.F='Н'. "
        "Силу в ньютонах обозначай кириллической Н или международной N, "
        "не заменяй кириллическую Н на похожую латинскую H. "
        "Если таблица нечитаема, верни "
        '{"error": "unreadable", "message": "<что мешает>"}.'
    )
    user = hint or "Распознай таблицу измерений с фото."

    try:
        result = get_provider().chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            images=[image],
            json_out=True,
        )
        payload = result.json()
    except (LLMError, json.JSONDecodeError, ValueError) as exc:
        return Table(source="photo", warnings=[f"распознавание не удалось: {exc}"])

    if not isinstance(payload, dict):
        return Table(source="photo", warnings=["Ожидался объект JSON с таблицей"])
    if "error" in payload:
        return Table(
            source="photo",
            warnings=[str(payload.get("message", payload["error"]))],
        )

    columns: dict[str, list[float]] = {}
    raw_columns = payload.get("columns")
    if not isinstance(raw_columns, dict):
        return Table(source="photo", warnings=["Нет объекта columns в ответе модели"])
    for name, values in raw_columns.items():
        if not isinstance(values, list):
            return Table(source="photo", warnings=[f"Столбец {name}: ожидался список"])
        numbers = [_to_float(str(v)) if not isinstance(v, bool) else None for v in values]
        if any(v is None or not math.isfinite(v) for v in numbers):
            return Table(source="photo", warnings=[f"Столбец {name}: нечитаемая ячейка; введите таблицу текстом"])
        if numbers:
            columns[str(name)] = numbers
    raw_units = payload.get("units") or {}
    if not isinstance(raw_units, dict):
        return Table(source="photo", warnings=["units должен быть объектом JSON"])
    if len({len(v) for v in columns.values()}) > 1:
        return Table(source="photo", warnings=["Столбцы на фото разной длины; проверьте пропуски"])
    units = {str(k): str(v) for k, v in raw_units.items()}
    from app.tools.units import to_si
    clean_columns, clean_units, warnings = {}, {}, []
    for original_name, values in columns.items():
        name, header_unit = _parse_header_cell(original_name)
        unit = units.get(original_name, units.get(name, header_unit)).strip()
        # A common visual confusion in a force column only. H elsewhere
        # may mean henry and must never be globally treated as newtons.
        if re.fullmatch(r"F(?:_?\d+)?", name) and unit == "H":
            unit = "Н"
            warnings.append(f"{name}: латинская H распознана как Н (ньютон); сверьте единицу на фото")
        if re.fullmatch(r"F(?:_?\d+)?", name) and header_unit == "H":
            header_unit = "Н"
        if header_unit and not unit:
            unit = header_unit
        if header_unit and unit and to_si(header_unit) != to_si(unit):
            return Table(source="photo", warnings=[f"{name}: разные единицы в заголовке и units; проверьте фото"])
        if name in clean_columns:
            return Table(source="photo", warnings=[f"После разбора заголовка повторяется столбец {name}"])
        clean_columns[name] = values
        if unit:
            clean_units[name] = unit
    table = Table(columns=clean_columns, units=clean_units, source="photo", warnings=warnings)
    if table.is_empty:
        table.warnings.append("модель не вернула ни одного числового столбца")
    return table
