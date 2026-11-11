"""Минимальный локальный runner для эвристического Input Guard."""
import json
from pathlib import Path
from checker.security.detector import detect_injection
from evaluation.metrics import binary_metrics

ROOT = Path(__file__).parent
DATASET = ROOT / "datasets" / "security.jsonl"


def run_security_benchmark():
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    truth, predicted = [], []
    for row in rows:
        result = detect_injection(row["text"])
        truth.append(bool(row["is_attack"]))
        predicted.append(result.suspicious)
        print(row["id"], "expected=", row["is_attack"], "detected=", result.suspicious,
              "patterns=", result.matched_patterns)
    m = binary_metrics(truth, predicted)
    print("\nSecurity benchmark")
    print(f"precision: {m.precision:.3f}")
    print(f"recall: {m.recall:.3f}")
    print(f"f1: {m.f1:.3f}")
    print(f"false_positive_rate: {m.false_positive_rate:.3f}")
    print(f"attack_miss_rate: {m.false_acceptance_rate:.3f}")
    return m


if __name__ == "__main__":
    run_security_benchmark()
