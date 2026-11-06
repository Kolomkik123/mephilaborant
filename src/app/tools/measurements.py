"""Обработка измерений: погрешности, промахи, МНК, график.

Это инструменты, а не промпты. Значения и неопределённости считает Python.
Качество на лабораторных примерах проверяется отдельным набором данных.

Методика — стандартная для физпрактикума:
прямые измерения — выборочное СКО, СКО среднего, коэффициент Стьюдента;
косвенные — перенос погрешности в квадратурах через частные производные;
промахи — критерий Шовене и критерий Граббса;
линейная зависимость — МНК с погрешностями наклона и сдвига.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from app.config import settings


@dataclass
class DirectResult:
    """Результат обработки прямых измерений одной величины."""

    name: str
    n: int
    mean: float
    sd: float  # выборочное СКО единичного измерения
    sem: float  # СКО среднего
    t_factor: float
    random_error: float  # случайная погрешность при заданной вероятности
    instrument_error: float
    total_error: float
    confidence: float
    relative_error: float
    unit: str = ""
    standard_error: float = 0.0
    degrees_freedom: float = math.inf
    coverage_factor: float = 1.0

    def as_text(self) -> str:
        digits = _digits_for(self.total_error)
        unit = f" {self.unit}" if self.unit else ""
        return (
            f"{self.name} = {self.mean:.{digits}f} ± {self.total_error:.{digits}f}{unit} "
            f"(P = {self.confidence:.2f}, n = {self.n}, "
            f"δ = {self.relative_error * 100:.1f}%)"
        )


@dataclass
class OutlierReport:
    values: list[float]
    kept: list[float]
    outlier_indices: list[int] = field(default_factory=list)
    method: str = "chauvenet"
    comment: str = ""

    @property
    def has_outliers(self) -> bool:
        return bool(self.outlier_indices)


@dataclass
class FitResult:
    slope: float
    slope_error: float
    intercept: float
    intercept_error: float
    r_squared: float
    n: int
    slope_standard_error: float = 0.0
    intercept_standard_error: float = 0.0
    degrees_freedom: float = math.inf
    confidence: float = 0.95

    def as_text(self) -> str:
        return (
            f"k = {self.slope:.5g} ± {self.slope_error:.3g}; "
            f"b = {self.intercept:.5g} ± {self.intercept_error:.3g}; "
            f"R² = {self.r_squared:.4f} (n = {self.n}, P ≈ {self.confidence:.2f})"
        )


def _digits_for(error: float) -> int:
    """Разрядность по погрешности: округляем результат по её порядку."""
    if error <= 0 or not math.isfinite(error):
        return 3
    order = math.floor(math.log10(abs(error)))
    return max(0, -order + 1)


def as_measurement(value: float, error: float) -> str:
    """Значение с погрешностью, округлённые по порядку погрешности.

    Модуль Юнга получается из наклона прямой и равен 3.9e11 Па. Запись
    «390462307372 ± 11032052789» формально верна, но читать её нельзя, и
    двенадцать значащих цифр в ней ничего не значат. Поэтому очень большие
    и очень малые результаты записываются в показательной форме с общим
    порядком у значения и погрешности — как это и принято в обработке
    измерений.
    """
    if value and (abs(value) >= 1e5 or abs(value) < 1e-3):
        exponent = math.floor(math.log10(abs(value)))
        scale = 10.0**exponent
        digits = _digits_for(error / scale) if error > 0 else 3
        return f"({value / scale:.{digits}f} ± {error / scale:.{digits}f})e{exponent:+03d}"
    digits = _digits_for(error)
    return f"{value:.{digits}f} ± {error:.{digits}f}"


def student_t(n: int, confidence: float = 0.95) -> float:
    """Коэффициент Стьюдента для n измерений."""
    if n < 2:
        return 0.0
    try:
        from scipy import stats

        return float(stats.t.ppf(1 - (1 - confidence) / 2, n - 1))
    except ImportError:
        # Таблица для P = 0.95 как резерв, если scipy нет.
        table = {
            2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36,
            9: 2.31, 10: 2.26, 15: 2.14, 20: 2.09, 30: 2.05,
        }
        keys = sorted(table)
        for key in keys:
            if n <= key:
                return table[key]
        return 1.96


def expanded_error(parts: Sequence[tuple[float, float]], confidence: float = 0.95):
    """Combine standard components, then apply approximate t coverage.

    Welch–Satterthwaite effective degrees of freedom; independent inputs.
    Type B rectangular limits are converted to standard uncertainty by √3.
    Coverage is approximate, especially for dominant rectangular components.
    """
    from scipy.stats import t, norm
    variance = sum(u*u for u, _ in parts)
    denominator = sum(u**4 / df for u, df in parts if math.isfinite(df) and df > 0)
    df = variance**2 / denominator if denominator else math.inf
    k = float(t.ppf((1+confidence)/2, df) if math.isfinite(df) else norm.ppf((1+confidence)/2))
    return math.sqrt(variance), df, k, k * math.sqrt(variance)


def _finite_data(values: Sequence[float]) -> np.ndarray:
    data = np.asarray(list(values), dtype=float)
    if data.ndim != 1 or data.size == 0 or not np.isfinite(data).all():
        raise ValueError("Нужен непустой одномерный набор конечных чисел")
    return data


def process_direct(
    values: Sequence[float],
    *,
    name: str = "x",
    unit: str = "",
    instrument_error: float = 0.0,
    confidence: float = 0.95,
) -> DirectResult:
    """Прямые измерения: среднее, СКО, случайная и полная погрешность."""
    data = _finite_data(values)
    if not 0 < confidence < 1 or not math.isfinite(instrument_error) or instrument_error < 0:
        raise ValueError("Некорректная вероятность или приборная погрешность")
    if data.size == 0:
        raise ValueError("пустой набор измерений")
    n = int(data.size)
    mean = float(data.mean())
    sd = float(data.std(ddof=1)) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n > 1 else 0.0
    t_factor = student_t(n, confidence)
    random_error = t_factor * sem
    # Приборная погрешность входит как равномерное распределение: Δ/√3.
    systematic = instrument_error / math.sqrt(3) if instrument_error else 0.0
    standard, df, coverage, total = expanded_error([(sem, n-1), (systematic, math.inf)], confidence)
    return DirectResult(
        name=name,
        n=n,
        mean=mean,
        sd=sd,
        sem=sem,
        t_factor=t_factor,
        random_error=random_error,
        instrument_error=instrument_error,
        total_error=total,
        confidence=confidence,
        relative_error=(total / abs(mean)) if mean else float("inf"),
        unit=unit,
        standard_error=standard,
        degrees_freedom=df,
        coverage_factor=coverage,
    )


def find_outliers(values: Sequence[float], method: str = "chauvenet") -> OutlierReport:
    """Поиск промахов. Шовене по умолчанию, Граббс — по запросу."""
    data = np.asarray(list(values), dtype=float)
    n = int(data.size)
    report = OutlierReport(values=[float(v) for v in data], kept=[], method=method)
    if n < 3:
        report.kept = report.values
        report.comment = "менее трёх измерений: промахи не ищутся"
        return report

    mean = float(data.mean())
    sd = float(data.std(ddof=1))
    if sd == 0:
        report.kept = report.values
        report.comment = "все значения совпали: разброса нет"
        return report

    deviations = np.abs(data - mean) / sd

    if method == "grubbs":
        try:
            from scipy import stats

            alpha = 0.05
            t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
            critical = (n - 1) / math.sqrt(n) * math.sqrt(
                t_crit**2 / (n - 2 + t_crit**2)
            )
        except ImportError:
            critical = 2.5
        flagged = [i for i in range(n) if deviations[i] > critical]
        report.comment = f"критерий Граббса, порог {critical:.2f}σ"
    else:
        # Шовене: промах, если ожидаемое число таких отклонений меньше 0.5.
        try:
            from scipy import stats

            probability = 2 * (1 - stats.norm.cdf(deviations))
        except ImportError:
            probability = np.array(
                [2 * (1 - 0.5 * (1 + math.erf(d / math.sqrt(2)))) for d in deviations]
            )
        flagged = [i for i in range(n) if probability[i] * n < 0.5]
        report.comment = "критерий Шовене"

    report.outlier_indices = flagged
    report.kept = [float(v) for i, v in enumerate(data) if i not in flagged]
    return report


def propagate_error(
    formula: str,
    values: dict[str, float],
    errors: dict[str, float],
    degrees_freedom: dict[str, float] | None = None,
    confidence: float = 0.95,
) -> tuple[float, float, dict[str, float]]:
    """Косвенное измерение: значение, погрешность и вклад каждой переменной.

    formula — выражение на языке SymPy, например "4*pi**2*L/T**2".
    Возвращает (значение, погрешность, вклады в квадрате погрешности).
    """
    import sympy as sp

    symbols = {name: sp.Symbol(name) for name in values}
    from app.tools.formulas import usable
    if not usable(formula, list(values), {}):
        raise ValueError("Недопустимое выражение или неизвестные переменные")
    expression = sp.sympify(formula, locals=symbols)
    substitution = {symbols[name]: value for name, value in values.items()}

    result = float(expression.subs(substitution).evalf())

    contributions: dict[str, float] = {}
    total_square = 0.0
    for name, symbol in symbols.items():
        error = errors.get(name, 0.0)
        if not error:
            continue
        derivative = float(sp.diff(expression, symbol).subs(substitution).evalf())
        part = (derivative * error) ** 2
        contributions[name] = part
        total_square += part

    if not math.isfinite(result):
        raise ValueError("Результат не является конечным числом")
    parts = [(math.sqrt(part), (degrees_freedom or {}).get(name, math.inf)) for name, part in contributions.items()]
    _, _, _, expanded = expanded_error(parts, confidence)
    return result, expanded, contributions


def least_squares(
    x: Sequence[float], y: Sequence[float], *, through_origin: bool = False, y_error: float = 0.0
) -> FitResult:
    """МНК для линейной зависимости с погрешностями коэффициентов."""
    x_array = _finite_data(x)
    y_array = _finite_data(y)
    if x_array.size != y_array.size:
        raise ValueError("длины x и y не совпадают")
    n = int(x_array.size)
    if n < 3:
        raise ValueError("для МНК нужно не менее трёх точек")
    if not math.isfinite(y_error) or y_error < 0:
        raise ValueError("Некорректная стандартная неопределённость y")
    if np.ptp(x_array) == 0:
        raise ValueError("Для МНК нужны различные значения x")

    if through_origin:
        slope = float((x_array * y_array).sum() / (x_array**2).sum())
        residuals = y_array - slope * x_array
        variance = max(float((residuals**2).sum() / (n - 1)), y_error**2)
        slope_error = math.sqrt(variance / float((x_array**2).sum()))
        intercept = intercept_error = 0.0
    else:
        slope, intercept = np.polyfit(x_array, y_array, 1)
        slope, intercept = float(slope), float(intercept)
        residuals = y_array - (slope * x_array + intercept)
        variance = max(float((residuals**2).sum() / (n - 2)), y_error**2)
        sxx = float(((x_array - x_array.mean()) ** 2).sum())
        slope_error = math.sqrt(variance / sxx)
        intercept_error = math.sqrt(variance * (1 / n + x_array.mean() ** 2 / sxx))

    total_square = float(((y_array - y_array.mean()) ** 2).sum())
    residual_square = float((residuals**2).sum())
    r_squared = 1 - residual_square / total_square if total_square else 1.0

    slope_standard, intercept_standard = slope_error, intercept_error
    df = n - (1 if through_origin else 2)
    coverage = student_t(df+1)
    return FitResult(
        slope=slope,
        slope_error=slope_error * coverage,
        intercept=intercept,
        intercept_error=intercept_error * coverage,
        r_squared=r_squared,
        n=n,
        slope_standard_error=slope_standard,
        intercept_standard_error=intercept_standard,
        degrees_freedom=df,
    )


def fit_outliers(x: Sequence[float], y: Sequence[float]) -> list[int]:
    """Externally studentized residuals, Bonferroni two-sided alpha=0.05.

    Flags are suggestions only. High leverage and multiple outliers can mask
    errors; no universal recall is claimed.
    """
    from scipy.stats import t
    xa, ya = _finite_data(x), _finite_data(y)
    if len(xa) != len(ya) or len(xa) < 4 or np.ptp(xa) == 0:
        return []
    n = len(xa)
    slope, intercept = np.polyfit(xa, ya, 1)
    residual = ya - (slope * xa + intercept)
    leverage = 1/n + (xa-xa.mean())**2 / ((xa-xa.mean())**2).sum()
    sse = float((residual**2).sum())
    if sse <= np.finfo(float).eps * max(float((ya**2).sum()), 1e-300):
        return []
    threshold = float(t.ppf(1-0.05/(2*n), n-3))
    flagged = []
    for i in range(n):
        if 1-leverage[i] <= 1e-12:
            continue
        deleted_variance = max(0, (sse-residual[i]**2/(1-leverage[i]))/(n-3))
        denominator = math.sqrt(deleted_variance*(1-leverage[i]))
        score = abs(residual[i])/denominator if denominator else (math.inf if residual[i] else 0)
        if score > threshold:
            flagged.append(i)
    return flagged


def plot_fit(
    x: Sequence[float],
    y: Sequence[float],
    fit: FitResult,
    *,
    x_label: str = "x",
    y_label: str = "y",
    y_errors: Sequence[float] | None = None,
    title: str = "",
    filename: str = "fit.png",
) -> Path:
    """График с точками, планками погрешностей и прямой МНК."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_array = np.asarray(list(x), dtype=float)
    y_array = np.asarray(list(y), dtype=float)

    figure, axes = plt.subplots(figsize=(7, 4.5), dpi=150)
    if y_errors is not None:
        axes.errorbar(
            x_array, y_array, yerr=list(y_errors), fmt="o", capsize=3,
            color="#1f3b73", label="измерения",
        )
    else:
        axes.plot(x_array, y_array, "o", color="#1f3b73", label="измерения")

    grid = np.linspace(x_array.min(), x_array.max(), 100)
    axes.plot(
        grid,
        fit.slope * grid + fit.intercept,
        "-",
        color="#c0392b",
        label=f"МНК: k = {fit.slope:.4g} ± {fit.slope_error:.2g}",
    )
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    if title:
        axes.set_title(title)
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = settings.artifacts_dir / filename
    figure.savefig(path)
    plt.close(figure)
    return path


def check_plausibility(value: float, unit: str = "") -> list[str]:
    """Вспомогательная проверка правдоподобия числа.

    Ловит не физику, а явные признаки сбоя расчёта: не-число, нулевую или
    абсурдную величину для известных единиц.
    """
    problems: list[str] = []
    if not math.isfinite(value):
        problems.append("результат не является числом")
        return problems

    known_ranges = {
        "м/с^2": (0.1, 100.0),
        "м/с²": (0.1, 100.0),
        "с": (1e-6, 1e5),
        "м": (1e-6, 1e4),
        "кг": (1e-6, 1e4),
    }
    bounds = known_ranges.get(unit)
    if bounds and not (bounds[0] <= abs(value) <= bounds[1]):
        problems.append(
            f"значение {value:.4g} {unit} вне разумного диапазона "
            f"{bounds[0]:g}–{bounds[1]:g} {unit}"
        )
    if value == 0:
        problems.append("результат равен нулю — проверьте подстановку")
    return problems
