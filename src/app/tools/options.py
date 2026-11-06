"""Shared parsing of per-column instrument bounds."""
import re

def parse_instrument_errors(text: str) -> dict:
    result = {}
    for item in text.split(";"):
        if not item.strip():
            continue
        match = re.fullmatch(r"\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*([+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)\s*(.*?)\s*", item)
        if not match:
            raise ValueError("Погрешности задаются так: T = 0.01 с; L = 1 мм")
        name, value, unit = match.groups()
        if name in result:
            raise ValueError(f"Погрешность {name} указана дважды")
        result[name] = {"value": float(value.replace(",", ".")), "unit": unit}
    return result
