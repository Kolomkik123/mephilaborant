from datetime import datetime, timezone
from langgraph.graph import StateGraph, END

from checker.state import PhysicsLabState
from checker.security.detector import detect_injection
from checker.security.sanitizer import wrap_untrusted
from checker.grounding.source_judge import judge_sources
from checker.numerical.validator import validate_numbers
from checker.examiner.validator import validate_examiner
from checker.decision import decide


def _audit(state: dict, event: str, details: dict | None = None) -> list[dict]:
    log = list(state.get("audit_log", []))
    log.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "details": details or {},
    })
    return log


def input_guard_node(state: PhysicsLabState) -> dict:
    """Выполняется до Dispatcher. Не удаляет сообщение, а изолирует его как данные."""
    raw = state.get("student_message", "")
    security = detect_injection(raw)
    wrapped = wrap_untrusted(raw, "student_message")

    untrusted = list(state.get("untrusted_inputs", []))
    untrusted.append({"source": "student_message", "text": raw})
    sanitized_items = [
        {
            "source": str(item.get("source", "external_data")),
            "content": wrap_untrusted(str(item.get("text", "")), str(item.get("source", "external_data")))
        }
        for item in untrusted
    ]

    return {
        "sanitized_student_message": wrapped,
        "untrusted_inputs": untrusted,
        "sanitized_untrusted_inputs": sanitized_items,
        "security_result": security.model_dump(),
        "audit_log": _audit(state, "input_guard", security.model_dump()),
    }


def source_check_node(state: PhysicsLabState) -> dict:
    result = judge_sources(
        state.get("agent_response", ""),
        state.get("sources", []),
        state.get("numbers", {}),
    )
    return {"source_result": result}


def numerical_check_node(state: PhysicsLabState) -> dict:
    agent_name = state.get("agent_name", "").lower()
    numbers = state.get("numbers", {})
    is_calculator = agent_name in {"calculator", "расчётчик", "расчетчик"}
    if not numbers and not is_calculator:
        return {"numerical_result": {"status": "pass", "issues": [], "warnings": [
            "Численная проверка пропущена: агент не передал численные результаты."
        ]}}
    result = validate_numbers(state.get("agent_response", ""), numbers)
    return {"numerical_result": result}


def examiner_check_node(state: PhysicsLabState) -> dict:
    # Only run when the current agent is Examiner or payload was explicitly supplied.
    if state.get("agent_name", "").lower() != "examiner" and not state.get("examiner_payload"):
        return {"examiner_result": {"overall_status": "pass", "issues": []}}

    result = validate_examiner(
        state.get("examiner_payload", {}),
        state.get("context", {}),
        state.get("sources", []),
    )
    return {"examiner_result": result}


def aggregate_checks_node(state: PhysicsLabState) -> dict:
    results = {
        "security": state.get("security_result", {}),
        "sources": state.get("source_result", {}),
        "numbers": state.get("numerical_result", {}),
        "examiner": state.get("examiner_result", {}),
    }
    return {
        "check_results": results,
        "audit_log": _audit(state, "checks_completed", {
            "agent_name": state.get("agent_name", "unknown"),
            "check_names": list(results.keys()),
        }),
    }


def decision_node(state: PhysicsLabState) -> dict:
    update = decide(state)
    new_state = {**state, **update}
    update["audit_log"] = _audit(new_state, "decision", {
        "verdict": update["verdict"],
        "retry_count": update.get("retry_count", state.get("retry_count", 0)),
    })
    return update


def route_after_decision(state: PhysicsLabState) -> str:
    verdict = state.get("verdict", "needs_review")
    if verdict == "pass":
        return "pass"
    if verdict == "revise":
        return "revise"
    return "stop"


def build_checker_graph():
    """Подграф проверки результата агента. Input Guard вызывается отдельно перед Dispatcher."""
    builder = StateGraph(PhysicsLabState)
    builder.add_node("source_check", source_check_node)
    builder.add_node("numerical_check", numerical_check_node)
    builder.add_node("examiner_check", examiner_check_node)
    builder.add_node("aggregate", aggregate_checks_node)
    builder.add_node("decision", decision_node)

    builder.set_entry_point("source_check")
    builder.add_edge("source_check", "numerical_check")
    builder.add_edge("numerical_check", "examiner_check")
    builder.add_edge("examiner_check", "aggregate")
    builder.add_edge("aggregate", "decision")
    builder.add_edge("decision", END)

    return builder.compile()


def validate_agent_output_node(state: PhysicsLabState) -> dict:
    """Удобная функция-адаптер для вызова из основного графа команды."""
    graph = build_checker_graph()
    result = graph.invoke(state)
    return result


def build_input_guard_graph():
    builder = StateGraph(PhysicsLabState)
    builder.add_node("input_guard", input_guard_node)
    builder.set_entry_point("input_guard")
    builder.add_edge("input_guard", END)
    return builder.compile()
