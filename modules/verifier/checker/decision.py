def decide(state: dict) -> dict:
    results = state.get("check_results", {})
    feedback = []
    critical = []

    for name, result in results.items():
        if not isinstance(result, dict):
            continue
        feedback.extend(result.get("issues", []))
        feedback.extend(result.get("warnings", []))
        if result.get("critical_error"):
            critical.extend(result.get("issues", [f"Критическая ошибка: {name}"]))

    if critical:
        return {
            "verdict": "refuse",
            "feedback": feedback,
            "refusal_reason": "; ".join(critical),
        }

    statuses = [
        r.get("overall_status", r.get("status"))
        for r in results.values()
        if isinstance(r, dict)
    ]

    # Missing judge configuration/data is not silently treated as a pass.
    if "needs_review" in statuses:
        return {
            "verdict": "needs_review",
            "feedback": feedback or ["Для завершения проверки требуется дополнительная проверка."],
            "refusal_reason": "",
        }

    if any(s == "revise" for s in statuses):
        retry_count = int(state.get("retry_count", 0))
        max_retries = min(int(state.get("max_retries", 1)), 1)
        if retry_count < max_retries:
            return {
                "verdict": "revise",
                "feedback": feedback or ["Обнаружены ошибки, требующие исправления."],
                "retry_count": retry_count + 1,
                "refusal_reason": "",
            }
        return {
            "verdict": "refuse",
            "feedback": feedback,
            "refusal_reason": "Ошибки не устранены после единственной повторной попытки.",
        }

    return {
        "verdict": "pass",
        "feedback": [],
        "refusal_reason": "",
    }
