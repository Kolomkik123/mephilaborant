"""Агент-Расчётчик: таблица и формула → погрешности, график, промахи.

Главное правило: числа считает Python, не модель. LLM здесь нужна только
чтобы разобрать фото таблицы. Формулу обработки, если студент её не назвал,
расчётчик поднимает из методички — но выбирает её не модель, а проверка:
выражение должно разбираться SymPy и считаться по измеренным величинам
(см. app/tools/formulas.py). Измеренной величиной здесь считается и наклон
прямой МНК: в работах, где строят график, ответ получается именно из него
(«E = L/(S*k)»), а значение и погрешность у наклона есть. Перед расчётом
таблица и заданные величины переводятся в СИ (app/tools/units.py): формула
методички записана в СИ, а студент пишет то, что показывает прибор.
Единица результата выводится из формулы (app/tools/dimensions.py), а не
приписывается по работе, и сверяется с той, которую объявляет сама
методичка в примере обработки. Всё остальное — вызовы app.tools.measurements.

Целевые метрики проверяются отдельно на примерах из реальных методичек.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass

from app.agents.base import Agent, StepLimitExceeded
from app.data.retriever import Retrieval, Retriever, work_from_question
from app.state import CalculationState
from app.tools import measurements as m
from app.tools.dimensions import result_unit, unit_exponents
from app.tools.formulas import (
    CONSTANT_UNITS,
    KNOWN_CONSTANTS,
    Formula,
    given_values,
    pick_formula,
    symbols_of,
    usable,
    used_constants,
)
from app.tools.tables import (
    Table,
    count_numbers,
    parse_photo_table,
    parse_text_table,
    resolve_column_name,
)
from app.tools.units import Conversion, convert, convert_columns, to_si
from app.tools.quantities import infer_quantity_name

INSTRUMENT_RE = re.compile(
    r"приборн\w*\s*(?:погрешност\w*)?\s*[:=]?\s*([-+]?\d+(?:[.,]\d+)?)"
    r"\s*([А-Яа-яA-Za-z°]{1,6})?",
    re.IGNORECASE,
)
FORMULA_RE = re.compile(r"(?:формула|по формуле)\s*[:=]?\s*([^\n;]+)", re.IGNORECASE)


WORDY_RE = re.compile(r"[A-Za-zА-Яа-яЁё]{4,}")


@dataclass
class FitQuantity:
    """Коэффициент прямой МНК как измеренная величина.

    Наклон измерен не хуже столбца: у него есть значение и погрешность, и
    в формулу он входит так же. Единица собирается из единиц столбцов, а
    они к этому моменту уже приведены к СИ, поэтому наклон выходит в
    единицах СИ без отдельного пересчёта.
    """

    name: str
    value: float
    error: float
    unit: str = ""
    comment: str = ""
    standard_error: float = 0.0
    degrees_freedom: float = math.inf

    @property
    def as_text(self) -> str:
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.name} = {self.value:.4g} ± {self.error:.2g}{unit}"


def _prose(question: str) -> str:
    """Слова запроса без таблицы.

    Искать формулу по таблице чисел нечем: числа студента в индексе
    методички не встречаются, а переформулировать модели предлагается
    вопрос, а не пять строк замеров. Остаются строки, в которых есть
    настоящее слово и числа не составляют половину — шапка «L, м T, с» и
    строки замеров отсеиваются обеими проверками. Если не осталось
    ничего, возвращается исходный запрос.
    """
    kept = []
    for line in question.splitlines():
        if not WORDY_RE.search(line):
            continue
        if count_numbers(line) * 2 >= max(len(line.split()), 1):
            continue
        kept.append(line.strip())
    return " ".join(kept) or question


class Calculator(Agent):
    name = "calculator"
    task = "answer"

    def __init__(self, context, retriever: Retriever | None = None):
        super().__init__(context)
        self._retriever = retriever
        self._formula_notes: list[str] = []
        self._methodics_fragments: list[dict[str, str]] = []

    @property
    def retriever(self) -> Retriever:
        """Индекс грузится при первом обращении, а не на каждый расчёт.

        Нужен он только когда формулы в запросе нет: большинство расчётов
        обходится без поиска, и платить за загрузку индекса на каждом из
        них незачем.
        """
        if self._retriever is None:
            from app.data.search_backend import make_retriever
            self._retriever = make_retriever()
        return self._retriever

    def _lookup_formula(
        self,
        question: str,
        measured: list[str],
        work: str = "",
        known: dict[str, float] | None = None,
        units: dict[str, str] | None = None,
        from_fit: tuple[str, ...] = (),
    ) -> Formula | None:
        """Рабочая формула из методички, когда студент её не назвал.

        Если номер работы в запросе есть — фрагменты берутся по нему, и
        поиск не нужен. Если нет, запрос идёт в поиск и, не найдя формулы,
        переформулируется через настроенный текстовый провайдер.
        Критерий удачи здесь не релевантность выдачи, а
        формула, которая считается по измеренным столбцам: выдача может
        быть релевантной и без формулы вовсе.
        """
        self._formula_notes = []
        self._methodics_fragments = []
        query = _prose(question)
        with self.context.trace.step("calculator.search_formula", "retrieval") as step:
            found = self.retriever.fragments_for(query, limit=5, work=work)
            step.meta.update(
                {
                    "work": work,
                    "query": query,
                    "hits": len(found.hits),
                    "relevant": found.relevant,
                }
            )

        def pick(retrieval: Retrieval) -> Formula | None:
            if not retrieval.hits:
                return None
            selected = pick_formula(
                [hit.fragment.text for hit in retrieval.hits],
                measured,
                [hit.fragment.citation for hit in retrieval.hits],
                known=known,
                units=units,
                notes=self._formula_notes,
                from_fit=from_fit,
            )
            if selected is not None:
                self._methodics_fragments = [
                    {"citation": hit.fragment.citation, "text": hit.fragment.text}
                    for hit in retrieval.hits
                ]
            return selected

        with self.context.trace.step("calculator.pick_formula", "tool") as step:
            formula = pick(found)
            # Формулы в найденном может не быть — это свойство методички,
            # а не сбой инструмента: step.ok не трогаем, иначе метрика
            # ошибок вызова инструментов считает чужие промахи.
            step.meta.update(
                {
                    "found": bool(formula),
                    "formula": formula.as_text if formula else "",
                    "measured": measured,
                    "known": sorted(known or {}),
                    "from_fit": list(from_fit),
                }
            )

        if formula is None and not work:
            retried, _ = self.search_with_expansion(
                query,
                found,
                search=lambda variant: self.retriever.fragments_for(variant, limit=5),
                good=lambda result: pick(result) is not None,
                better=lambda candidate, best: bool(candidate.hits)
                and (not best.hits or candidate.hits[0].score > best.hits[0].score),
                describe=lambda result: {"hits": len(result.hits)},
            )
            formula = pick(retried)

        return formula

    def _get_table(self, state: CalculationState) -> Table:
        if state.get("image"):
            self.context.guard_steps()
            self.context.trace.llm_steps += 1
            with self.context.trace.step("calculator.vision", "tool") as step:
                table = parse_photo_table(state["image"], state.get("question", ""))
                step.meta.update(
                    {
                        "source": "photo",
                        "columns": list(table.columns),
                        "found": not table.is_empty,
                    }
                )
                # Нераспознанное фото — это сбой инструмента: данные были.
                step.ok = not table.is_empty
                if table.is_empty:
                    step.error = "; ".join(table.warnings)
            return table

        text = state.get("table_text") or state.get("question", "")
        with self.context.trace.step("calculator.parse_table", "tool") as step:
            table = parse_text_table(text)
            # Отсутствие таблицы во входе — не сбой инструмента, а повод
            # попросить данные. Инструмент отработал и честно сказал «пусто».
            # Иначе метрика ошибок вызова инструментов считает чужие промахи.
            step.meta.update(
                {
                    "source": "text",
                    "columns": list(table.columns),
                    "found": not table.is_empty,
                }
            )
            if table.is_empty:
                step.meta["reason"] = "; ".join(table.warnings) or "данных во входе нет"
        return table

    def process(self, state: CalculationState) -> CalculationState:
        from app.llm.provider import LLMError
        try:
            return self._process(state)
        except (ValueError, LLMError, StepLimitExceeded) as exc:
            return {"answer": str(exc), "refused": True,
                    "steps": [{"node": "calculator", "result": "needs_correction"}]}

    def _process(self, state: CalculationState) -> CalculationState:
        question = state.get("question", "")
        table = self._get_table(state)

        if table.is_empty:
            hint = " ".join(table.warnings) if table.warnings else ""
            return {
                "answer": (
                    "Не вижу таблицы измерений. Вставьте числа в чат — столбцами "
                    "с шапкой, например:\n\nt, с\tx, м\n1.0\t2.1\n2.0\t3.9\n\n"
                    + hint
                ),
                "refused": True,
                "steps": [{"node": "calculator", "result": "no_table"}],
            }

        # Перевод в СИ — до всего остального: и промахи, и МНК, и график,
        # и формула работают уже с приведёнными числами, поэтому наклон
        # прямой получается сразу в единицах СИ.
        with self.context.trace.step("calculator.units", "tool") as step:
            columns, si_units, conversions, unknown_units = convert_columns(
                table.columns, table.units
            )
            table = Table(
                columns=columns,
                units=si_units,
                source=table.source,
                warnings=table.warnings,
            )
            # Заданные величины переводятся так же: «m = 200 г» в формуле
            # СИ без перевода даёт результат в тысячу раз больше.
            given = given_values(_prose(FORMULA_RE.sub(" ", question)))
            known: dict[str, float] = dict(KNOWN_CONSTANTS)
            given_units: dict[str, str] = {}
            for item in given:
                si = to_si(item.unit) if item.unit else None
                if si is None:
                    known[item.name] = item.value
                    given_units[item.name] = item.unit
                    if item.unit:
                        unknown_units.append(f"{item.name} [{item.unit}]")
                    continue
                si_unit, factor, offset = si
                known[item.name] = convert(item.value, factor, offset)
                given_units[item.name] = si_unit
                conversion = Conversion(
                    name=item.name,
                    unit=item.unit,
                    si_unit=si_unit,
                    factor=factor,
                    offset=offset,
                )
                if conversion.changes_anything:
                    conversions.append(conversion)
            step.meta.update(
                {
                    "converted": [item.as_text for item in conversions],
                    "unknown": unknown_units,
                }
            )
        factors = {item.name: item.factor for item in conversions}

        instrument_match = INSTRUMENT_RE.search(question)
        instrument_error = (
            float(instrument_match.group(1).replace(",", ".")) if instrument_match else 0.0
        )
        # Приборную погрешность студент пишет в единицах прибора: «0.01 с».
        # Если единица названа и знакома — переводим по ней, иначе считаем,
        # что она в единицах своего столбца, и берём его множитель.
        instrument_si = to_si(instrument_match.group(2) or "") if instrument_match else None
        per_column_errors = state.get("instrument_errors", {})
        if not isinstance(per_column_errors, dict):
            raise ValueError("Приборные погрешности задаются объектом по именам столбцов")
        if instrument_match and len(table.columns) > 1:
            raise ValueError("Для нескольких столбцов задайте приборные погрешности отдельно по именам величин.")
        instrument_by_name = {}
        for name, spec in per_column_errors.items():
            if name not in table.columns or not isinstance(spec, dict):
                raise ValueError(f"Неизвестный столбец или неверная погрешность: {name}")
            value = float(spec.get("value", 0))
            original_unit = spec.get("unit", "")
            conversion = to_si(original_unit) if original_unit else None
            if original_unit and (conversion is None or unit_exponents(conversion[0]) != unit_exponents(table.units.get(name, ""))):
                raise ValueError(f"Приборная погрешность {name}: единица не совпадает со столбцом")
            factor = conversion[1] if conversion else factors.get(name, 1.0)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Погрешность {name} должна быть конечной и неотрицательной")
            instrument_by_name[name] = value * factor
        if instrument_match:
            name = next(iter(table.columns))
            if instrument_si and unit_exponents(instrument_si[0]) != unit_exponents(table.units.get(name, "")):
                raise ValueError("Единица приборной погрешности не совпадает со столбцом")
            if instrument_match.group(2) and not instrument_si:
                raise ValueError("Неизвестная единица приборной погрешности")
            instrument_by_name[name] = instrument_error * (instrument_si[1] if instrument_si else factors.get(name, 1.0))

        lines: list[str] = []
        if conversions:
            lines.append(
                "Перевёл в СИ: "
                + "; ".join(item.as_text for item in conversions)
                + " — формула методички записана в СИ."
            )
        if unknown_units:
            raise ValueError(
                "Единицы не знаю: "
                + ", ".join(unknown_units)
                + " — укажите поддерживаемые единицы перед расчётом."
            )
        attachments: list[str] = []
        payload: dict[str, object] = {"columns": {k: v for k, v in table.columns.items()}}
        results: dict[str, m.DirectResult] = {}
        derived: dict[str, FitQuantity] = {}

        # The user selects the experiment type; row order has no semantics.
        numeric_columns = [
            (name, values) for name, values in table.columns.items() if len(values) >= 3
        ]
        mode = state.get("mode", "")
        if not mode:
            mode = "fit" if re.search(r"режим\s*:\s*(?:мнк|fit)", question, re.I) else ""
        if not mode and len(table.columns) == 1:
            mode = "repeat"
        if mode not in {"repeat", "fit"}:
            raise ValueError("Выберите режим: repeat — повторные измерения, fit — зависимость и МНК. Режим нельзя надёжно угадать по порядку строк.")
        is_dependence = mode == "fit"
        if is_dependence:
            names = list(table.columns)
            x_name = resolve_column_name(state.get("x_column") or names[0], names)
            y_name = resolve_column_name(state.get("y_column") or next((n for n in names if n != x_name), ""), names)
            if x_name == y_name:
                raise ValueError("Оси x и y должны быть разными. Доступные столбцы: " + ", ".join(names))
            lines.append(f"Оси МНК: x = {x_name}, y = {y_name}.")
            numeric_columns = [(x_name, table.columns[x_name]), (y_name, table.columns[y_name])]
            if len(numeric_columns[0][1]) != len(numeric_columns[1][1]):
                raise ValueError("Столбцы x и y разной длины: исправьте неполные строки")
            if instrument_by_name.get(x_name, 0):
                raise ValueError("Обычный МНК считает x точным. Для погрешности x нужен метод с ошибками обеих осей; этот режим пока не поддерживается.")
        lines.append("Погрешности: расширенная неопределённость при P ≈ 0.95; независимые входы, линейное распространение, приборные границы — равномерное распределение.")

        # 1. Промахи и прямые измерения по каждому столбцу — только для
        #    повторных измерений одной величины.
        for name, values in ({} if is_dependence else table.columns).items():
            if len(values) < 2:
                raise ValueError(f"{name}: для оценки случайной погрешности нужно хотя бы два измерения. Постоянные задайте в запросе: L = 1 м.")
            with self.context.trace.step(f"calculator.outliers[{name}]", "tool") as step:
                outliers = m.find_outliers(values)
                step.meta.update(
                    {"outliers": outliers.outlier_indices, "method": outliers.method}
                )
            if outliers.has_outliers:
                flagged = ", ".join(
                    f"№{i + 1} ({values[i]:g})" for i in outliers.outlier_indices
                )
                lines.append(f"Промахи в «{name}»: {flagged} — {outliers.comment}, исключены.")

            with self.context.trace.step(f"calculator.direct[{name}]", "tool") as step:
                result = m.process_direct(
                    outliers.kept,
                    name=name,
                    unit=table.units.get(name, ""),
                    instrument_error=instrument_by_name.get(name, 0.0),
                )
                step.meta.update({"mean": result.mean, "error": result.total_error})
            results[name] = result
            lines.append(result.as_text())

        payload["direct"] = {
            name: {
                "mean": r.mean,
                "error": r.total_error,
                "n": r.n,
                "relative": r.relative_error if math.isfinite(r.relative_error) else None,
                "standard_error": r.standard_error,
            }
            for name, r in results.items()
        }

        # 2. МНК и график — только для зависимости. Для двух наборов
        #    повторных измерений прямая бессмысленна: «T(L)» по пяти почти
        #    одинаковым L даёт R² = 0.38 и график ни о чём. Столбцы разной
        #    длины — скорее всего зависимость с пропуском, их тоже строим.
        if is_dependence:
            (x_name, x_values), (y_name, y_values) = numeric_columns[0], numeric_columns[1]
            size = len(x_values)
            if size < 3:
                raise ValueError("Для МНК нужно не менее трёх полных пар измерений")
            if size >= 3:
                with self.context.trace.step("calculator.least_squares", "tool") as step:
                    fit = m.least_squares(x_values, y_values, y_error=instrument_by_name.get(y_name, 0.0) / math.sqrt(3))
                    step.meta.update({"slope": fit.slope, "r2": fit.r_squared})
                lines.append(f"МНК {y_name}({x_name}): {fit.as_text()}")
                payload["fit"] = {
                    "x_column": x_name,
                    "y_column": y_name,
                    "slope": fit.slope,
                    "slope_error": fit.slope_error,
                    "intercept": fit.intercept,
                    "intercept_error": fit.intercept_error,
                    "r_squared": fit.r_squared,
                }
                # Коэффициенты прямой — измеренные величины: именно из
                # наклона получается ответ работ, где строят график.
                # Столбец с таким же именем важнее: подменять измеренную
                # величину коэффициентом нельзя.
                x_unit = table.units.get(x_name, "")
                y_unit = table.units.get(y_name, "")
                slope_unit = f"{y_unit}/{x_unit}" if x_unit and y_unit else ""
                for alias, value, error, unit, comment in (
                    ("k", fit.slope, fit.slope_error, slope_unit, f"наклон {y_name}({x_name})"),
                    ("b", fit.intercept, fit.intercept_error, y_unit, "свободный член"),
                ):
                    if alias in table.columns:
                        continue
                    derived[alias] = FitQuantity(
                        name=alias, value=value, error=error, unit=unit, comment=comment,
                        standard_error=fit.slope_standard_error if alias == "k" else fit.intercept_standard_error,
                        degrees_freedom=fit.degrees_freedom,
                    )
                suspected = m.fit_outliers(x_values, y_values)
                payload["fit"]["suspected_outlier_indices"] = suspected
                lines.append("МНК: x считается точным, ошибки y независимы и одинаковы по дисперсии.")
                if suspected:
                    lines.append("Подозрительные точки по остаткам МНК: " + ", ".join(str(i+1) for i in suspected) + ". Проверьте журнал; точки автоматически не удалены.")
                payload["fit"]["as_measured"] = sorted(derived)

                with self.context.trace.step("calculator.plot", "tool") as step:
                    path = m.plot_fit(
                        x_values[:size],
                        y_values[:size],
                        fit,
                        x_label=f"{x_name}, {table.units.get(x_name, '')}".strip(", "),
                        y_label=f"{y_name}, {table.units.get(y_name, '')}".strip(", "),
                        title=f"{y_name} от {x_name}",
                        filename=f"fit_{self.context.trace.request_id}.png",
                        y_errors=[instrument_by_name.get(y_name, 0)] * size if instrument_by_name.get(y_name) else None,
                    )
                    step.meta["path"] = str(path)
                attachments.append(str(path))
                lines.append(f"График: {path.name}")

        # 3. Косвенное измерение. Формулу называет студент, а если не
        #    назвал — её поднимает из методички _lookup_formula.
        formula_match = FORMULA_RE.search(question)
        formula = formula_match.group(1).strip().rstrip(".") if formula_match else ""
        result_symbol = ""
        if "=" in formula:
            left, right = formula.split("=", 1)
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", left.strip()):
                raise ValueError("Слева от знака = должно быть обозначение результата")
            formula = right.strip()
            result_symbol = left.strip()
        # «формулу не помню» регулярное выражение ловит как формулу. Такая
        # формула не считается, и упираться в неё незачем — пусть уступит
        # место формуле из методички, но студенту об этом надо сказать.
        # Известные величины: константы плюс то, что студент назвал в
        # запросе («R0 = 10 Ом»). Формулу из запроса из текста убираем —
        # иначе её левая часть попадёт сюда как заданная величина.
        # Измеренные величины: столбцы плюс коэффициенты прямой.
        measured_names = list(results) + list(derived)

        rejected = ""
        if formula and measured_names and not usable(formula, measured_names, known):
            rejected, formula = formula, ""
        formula_source = "запрос" if formula else ""
        citation = ""
        work = work_from_question(question, state.get("work", ""))
        if work and not re.fullmatch(r"\d+(?:[.\-]\d+)?", work):
            raise ValueError("Укажите номер работы в виде 1.3 или напишите его в задании: Работа 1.3")
        # Искать формулу стоит, когда есть за что зацепиться: названа работа
        # или измерено несколько величин. По одному столбцу и без номера
        # работы искать нечем — в запросе «обработай: T, с 2.01 1.99 2.00»
        # для поиска нет ничего, кроме слова «обработай».
        from_methodics = None
        lookup_requested = bool(work) or (
            len(measured_names) >= 2 and (
                not is_dependence or re.search(r"формул|модул|вычисл|рассчит", question, re.I)
            )
        )
        if not formula and measured_names and lookup_requested:
            from_methodics = self._lookup_formula(
                question,
                measured_names,
                work,
                known=known,
                units={**table.units, **given_units},
                from_fit=tuple(derived),
            )

        if from_methodics is not None:
            payload["methodics"] = self._methodics_fragments
            formula = from_methodics.expression
            formula_source = "методичка"
            citation = from_methodics.source
            result_symbol = from_methodics.quantity
            preamble = (
                f"Формулу «{rejected}» посчитать по вашим столбцам нельзя"
                if rejected
                else "Формулы в запросе нет"
            )
            lines.append(
                f"{preamble}, взял из методички: "
                f"{from_methodics.as_text} ({citation})."
            )
            if from_methodics.steps:
                # Подстановка ступени — не только удобство: производная по
                # измеренной величине берётся сразу, без промежуточного
                # округления, и одна и та же величина в двух местах
                # выражения учитывается как одна.
                lines.append(
                    "Промежуточную величину подставил в формулу, а не считал "
                    "отдельно: погрешность тогда берётся по составленному "
                    f"выражению {from_methodics.expression}."
                )
        elif self._formula_notes:
            lines.append("Формулу из методички не взял: " + self._formula_notes[0] + ".")
        elif rejected:
            lines.append(
                f"Формулу «{rejected}» по вашим столбцам посчитать нельзя. "
                "Подходящей в методичке тоже не нашлось — напишите формулу "
                "сами, в виде, понятном SymPy, например 4*pi**2*L/T**2."
            )
        elif not formula:
            lines.append(
                "Подходящая формула для косвенного результата не найдена. "
                "Укажите номер работы, обозначения величин и недостающие постоянные "
                "(например L = 1 м). Ниже доступны только результаты обработки самой таблицы."
            )

        quantity_name = quantity_name_source = quantity_name_evidence = ""
        if from_methodics is not None:
            quantity_name = from_methodics.quantity_name
            quantity_name_source = from_methodics.quantity_name_source
            quantity_name_evidence = from_methodics.quantity_name_evidence
        elif result_symbol:
            label = infer_quantity_name(result_symbol, [question], ["запрос"])
            if work:
                definitions = self.retriever.fragments_for(_prose(question), limit=5, work=work)
                documented = infer_quantity_name(result_symbol,
                    [hit.fragment.text for hit in definitions.hits],
                    [hit.fragment.citation for hit in definitions.hits])
                if documented.name:
                    label = documented
            quantity_name, quantity_name_source, quantity_name_evidence = label.name, label.source, label.evidence

        if formula and measured_names:
            values = {name: r.mean for name, r in results.items()}
            errors = {name: r.standard_error for name, r in results.items()}
            degrees_freedom = {name: r.degrees_freedom for name, r in results.items()}
            # Наклон входит в расчёт со своей погрешностью из МНК, поэтому
            # в разборе вкладов он виден наравне со столбцами.
            in_formula = set(symbols_of(formula))
            used_fit = [name for name in derived if name in in_formula]
            values.update({name: derived[name].value for name in used_fit})
            if len(used_fit) > 1:
                raise ValueError("Наклон k и сдвиг b коррелированы. Формула с обоими требует учёта ковариации; этот случай пока не поддерживается.")
            errors.update({name: derived[name].standard_error for name in used_fit})
            degrees_freedom.update({name: derived[name].degrees_freedom for name in used_fit})
            if used_fit:
                shown = "; ".join(
                    f"{derived[name].as_text} ({derived[name].comment})" for name in used_fit
                )
                lines.append(
                    f"Коэффициенты прямой подставил как измеренные величины: {shown}."
                )
                payload["substituted_fit"] = {
                    name: {"value": derived[name].value, "error": derived[name].error}
                    for name in used_fit
                }
            # Известные величины идут в расчёт с нулевой погрешностью:
            # вклада в погрешность результата они не дают, и в разбор
            # вкладов не попадают.
            substituted = [
                name for name in used_constants(formula, measured_names) if name in known
            ]
            values.update({name: known[name] for name in substituted})
            if substituted:
                shown = "; ".join(
                    f"{name} = {known[name]:g}"
                    + (f" {given_units[name]}" if given_units.get(name) else "")
                    for name in substituted
                )
                lines.append(f"Подставил известные величины: {shown}.")
                payload["substituted"] = {name: known[name] for name in substituted}
            # Единица результата выводится из формулы и единиц того, что
            # в неё подставлено. Единицы коэффициентов прямой собраны из
            # единиц столбцов, единицы констант — в CONSTANT_UNITS.
            unit_by_name = {
                **CONSTANT_UNITS,
                **{name: table.units.get(name, "") for name in table.columns},
                **{name: item.unit for name, item in derived.items()},
                **given_units,
            }
            declared = from_methodics.declared_unit if from_methodics else ""
            declared_source = (
                from_methodics.declared_source if from_methodics else ""
            ) or citation
            matches: bool | None = None
            with self.context.trace.step("calculator.dimensions", "tool") as step:
                unit = result_unit(formula, unit_by_name)
                expected = unit_exponents(declared) if declared else None
                if unit.known and expected is not None:
                    matches = expected == unit.dimension.exponents
                step.meta.update(
                    {
                        "unit": unit.as_text,
                        "known": unit.known,
                        "problem": unit.problem,
                        "declared": declared,
                        "matches_methodics": matches,
                        "units": {
                            name: unit_by_name.get(name, "")
                            for name in symbols_of(formula)
                        },
                    }
                )
            if unit.problem:
                # Размерность не сошлась: неверна либо формула, либо
                # единицы столбцов. Промолчать здесь значит выдать
                # правдоподобное число неизвестно чего.
                raise ValueError(
                    f"Размерность не сходится: {unit.problem}. "
                    "Проверьте единицы в шапке таблицы и формулу — "
                    "число ниже считать нельзя."
                )
            if matches is True:
                lines.append(
                    f"Размерность сошлась с методичкой: {from_methodics.quantity} "
                    f"там в {declared} ({declared_source})."
                )
            elif matches is False:
                # Единственная ошибка, которую больше ловить нечем:
                # согласованно перепутанная единица в шапке таблицы.
                # Размерность формулы при ней сходится, а результат выходит
                # не той величиной — поймать это можно только сверкой с
                # методичкой.
                raise ValueError(
                    "Размерность не сошлась с методичкой: получилось "
                    f"{unit.as_text or 'безразмерная величина'}, а "
                    f"{from_methodics.quantity} в методичке в {declared} "
                    f"({declared_source}). Скорее всего, в шапке таблицы не та "
                    "единица — проверьте и посчитайте заново."
                )
            suffix = f" {unit.as_text}" if unit.as_text else ""

            with self.context.trace.step("calculator.propagate", "tool") as step:
                try:
                    value, error, contributions = m.propagate_error(formula, values, errors, degrees_freedom)
                    step.meta.update({"formula": formula, "value": value, "error": error})
                    result_label = " ".join(part for part in (quantity_name, result_symbol) if part) or "Результат"
                    result_text = f"{result_label} = {m.as_measurement(value, error)}{suffix}"
                    relation = f"{result_symbol} = {formula}" if result_symbol else formula
                    relative_text = f" Относительная погрешность δ = {error / abs(value) * 100:.1f}%." if value else ""
                    lines.append(f"Расчёт по формуле {relation}.{relative_text}")
                    payload["indirect"] = {
                        "quantity": result_symbol,
                        "quantity_name": quantity_name,
                        "quantity_name_source": quantity_name_source,
                        "quantity_name_evidence": quantity_name_evidence,
                        "result_text": result_text,
                        "formula": formula,
                        "unit": unit.as_text,
                        "declared_unit": declared,
                        "unit_matches_methodics": matches,
                        "written": (
                            from_methodics.written if from_methodics else formula
                        ),
                        "steps": (
                            [list(step) for step in from_methodics.steps]
                            if from_methodics
                            else []
                        ),
                        "source": formula_source,
                        "citation": citation,
                        "value": value,
                        "error": error,
                        "contributions": contributions,
                    }
                    if unit.known and not unit.as_text:
                        lines.append(
                            "Величина безразмерная — так и должно быть: "
                            "единицы в формуле сокращаются."
                        )
                    elif not unit.known and not unit.problem:
                        # Размерность не вывести: у какой-то величины
                        # единица не названа или незнакомая.
                        lines.append(
                            "Единицу результата не вывел — не у всех величин "
                            "формулы известна единица."
                            + (
                                " Все переведённые величины в СИ."
                                if conversions and not unknown_units
                                else ""
                            )
                        )
                    dominant = max(contributions, key=contributions.get) if contributions else ""
                    if dominant:
                        share = contributions[dominant] / sum(contributions.values())
                        lines.append(
                            f"Основной вклад в погрешность даёт {dominant}: {share:.0%}. "
                            "Эту величину и стоит мерить точнее."
                        )
                except Exception as exc:
                    step.ok = False
                    step.error = f"{type(exc).__name__}: {exc}"
                    raise ValueError(f"Не удалось вычислить формулу «{formula}»: {exc}") from exc

        if table.warnings:
            lines.append("Замечания по таблице: " + "; ".join(table.warnings))
        if table.source == "photo":
            lines.append("Данные распознаны с фото — сверьте числа с журналом.")

        result_text = payload.get("indirect", {}).get("result_text", "")
        citations = list(dict.fromkeys(source for source in (citation, quantity_name_source) if source and source != "запрос"))
        return {
            "answer": (result_text + "\n\n" if result_text else "") + "\n\n".join(lines),
            "attachments": attachments,
            "citations": citations,
            "payload": payload,
            "steps": [{"node": "calculator", "result": "calculated"}],
        }


def calculator_node(state: CalculationState, context) -> CalculationState:
    return Calculator(context).process(state)
