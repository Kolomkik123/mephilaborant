"""Пример подключения Checker к основному графу команды.

В реальном проекте замените dispatcher_node и physics_agent_node своими узлами.
После каждого агента вызывайте checker. Если verdict=revise, передавайте feedback
агенту для одной повторной генерации и снова проверяйте результат.
"""

from langgraph.graph import StateGraph, END
from checker.state import PhysicsLabState
from checker.graph import input_guard_node, validate_agent_output_node


def dispatcher_node(state: PhysicsLabState):
    # Диспетчер должен использовать sanitized_student_message как данные,
    # а не воспринимать содержимое untrusted_data как системные инструкции.
    return {"agent_name": "calculator"}


def physics_agent_node(state: PhysicsLabState):
    # Заглушка вместо настоящего агента.
    return {
        "agent_response": "Сила тока равна 2 А.",
        "sources": [
            {"id": "src1", "text": "Закон Ома: I = U / R.", "origin": "lab_manual"}
        ],
        "numbers": {
            "current": {
                "value": 2.0,
                "unit": "ampere",
                "display_name": "сила тока",
                "tolerance_abs": 0.001,
                "tolerance_rel": 0.0001
            }
        }
    }


def checker_after_agent_node(state: PhysicsLabState):
    return validate_agent_output_node(state)


def route_verdict(state: PhysicsLabState):
    if state.get("verdict") == "pass":
        return "continue"
    if state.get("verdict") == "revise":
        return "retry"
    return "stop"


def retry_agent_node(state: PhysicsLabState):
    # Здесь нужно вызвать того же агента с feedback.
    # Это заглушка; в production агент должен получить замечания.
    return physics_agent_node(state)


def build_main_graph():
    builder = StateGraph(PhysicsLabState)
    builder.add_node("input_guard", input_guard_node)
    builder.add_node("dispatcher", dispatcher_node)
    builder.add_node("agent", physics_agent_node)
    builder.add_node("checker", checker_after_agent_node)
    builder.add_node("retry_agent", retry_agent_node)

    builder.set_entry_point("input_guard")
    builder.add_edge("input_guard", "dispatcher")
    builder.add_edge("dispatcher", "agent")
    builder.add_edge("agent", "checker")
    builder.add_conditional_edges(
        "checker",
        route_verdict,
        {
            "continue": END,
            "retry": "retry_agent",
            "stop": END,
        }
    )
    builder.add_edge("retry_agent", "checker")
    return builder.compile()


if __name__ == "__main__":
    app = build_main_graph()
    output = app.invoke({
        "student_message": "Рассчитай силу тока при U=10 В и R=5 Ом.",
        "context": {"lab_topic": "Закон Ома"},
        "retry_count": 0,
        "max_retries": 1,
        "audit_log": [],
    })
    print(output.get("verdict"))
    print(output.get("check_results"))
