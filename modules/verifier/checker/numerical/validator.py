import re
import math
from typing import Any
from checker.schemas import NumericValidationResult, NumericIssue

try:
    import pint
    UNIT_REGISTRY = pint.UnitRegistry()
except Exception:
    UNIT_REGISTRY = None

NUMBER_RE = re.compile(
    r'(?<![\w])[-+]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][-+]?\d+)?'
)

def _to_float(token: str) -> float:
    return float(token.replace(",", "."))

def _convert_expected(value: float, unit: str | None, target_unit: str | None):
    if not unit or not target_unit or unit == target_unit:
        return value
    if UNIT_REGISTRY is None:
        raise ValueError("Pint недоступен")
    q = value * UNIT_REGISTRY(unit)
    return q.to(target_unit).magnitude

def validate_numbers(answer: str, numbers: dict[str, Any] | None) -> dict:
    numbers = numbers or {}
    if not numbers:
        return NumericValidationResult(
            status="needs_review",
            warnings=["Поле numbers пусто: численная сверка невозможна."]
        ).model_dump()

    issues = []
    checked = []
    warnings = []

    for key, item in numbers.items():
        if not isinstance(item, dict) or "value" not in item:
            warnings.append(f"Некорректная запись numbers[{key}].")
            continue

        expected = float(item["value"])
        expected_unit = item.get("unit")
        display_name = item.get("display_name", key)
        abs_tol = float(item.get("tolerance_abs", 1e-6))
        rel_tol = float(item.get("tolerance_rel", 1e-6))

        # Basic strategy: only compare if the quantity name/key is present nearby.
        # A production version should use a quantity extractor/entity matcher.
        name_present = key.lower() in answer.lower() or display_name.lower() in answer.lower()
        if not name_present:
            warnings.append(
                f"Не удалось однозначно сопоставить величину '{display_name}' с текстом ответа."
            )
            continue

        checked.append(key)
        # Extract numbers only in the local text following the quantity label.
        # This avoids accepting a wrong current merely because another quantity
        # elsewhere in the answer happens to have the expected numeric value.
        label = display_name if display_name.lower() in answer.lower() else key
        match = re.search(re.escape(label), answer, flags=re.IGNORECASE)
        local_text = answer[match.end():match.end() + 100] if match else ""
        observed_tokens = NUMBER_RE.findall(local_text)
        observed = []
        for token in observed_tokens:
            try:
                observed.append(_to_float(token))
            except ValueError:
                pass

        if not observed:
            issues.append(NumericIssue(
                quantity_key=key,
                expected_value=expected,
                expected_unit=expected_unit,
                observed_values=[],
                message=f"В ответе не найдено числовое значение для {display_name}."
            ))
            continue

        # Conservative initial implementation: accept if any numeric token matches.
        # It does not yet parse units attached to each token or disambiguate formulas.
        matching = any(
            math.isclose(v, expected, rel_tol=rel_tol, abs_tol=abs_tol)
            for v in observed
        )

        # Unit conversion is handled when structured observed values are later supplied.
        # Free-text conversion such as 100 cm == 1 m requires quantity-aware parsing.
        if not matching:
            issues.append(NumericIssue(
                quantity_key=key,
                expected_value=expected,
                expected_unit=expected_unit,
                observed_values=observed,
                message=(
                    f"Число для '{display_name}' не совпадает с ожидаемым "
                    f"значением {expected} {expected_unit or ''}."
                ).strip()
            ))

    status = "revise" if issues else ("needs_review" if warnings and not checked else "pass")
    return NumericValidationResult(
        status=status,
        checked_quantities=checked,
        issues=issues,
        warnings=warnings
    ).model_dump()
