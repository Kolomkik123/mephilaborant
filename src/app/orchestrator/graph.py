"""Путь запроса: Диспетчер -> агент -> Проверяющий -> ответ.

Агенты подключаются в словаре AGENTS: функция (state, context) -> dict с изменениями.
Проверяющий подключается через VERIFIER и возвращает {"verdict": "ok" | "retry" | "fail"}.
"""
from __future__ import annotations

from collections.abc import Callable

from app.agents.base import AgentContext, DummyTrace
from app.agents.calculator import calculator_node
from app.orchestrator.dispatcher import Dispatcher

END = "__end__"
MAX_HOPS = 8
MAX_RETRIES = 1
AGENT_NAMES = ("calculator", "consultant", "examiner")

FAIL_TEXT = "Не смог дать надёжный ответ. Уточните номер работы или вопрос."


def _not_connected(title: str) -> Callable:
    def node(state, context):
        return {"answer": f"{title} ещё не подключён к системе.", "refused": True}
    return node


AGENTS: dict[str, Callable] = {
    "calculator": calculator_node,
    "consultant": _not_connected("Консультант"),
    "examiner": _not_connected("Экзаменатор"),
}


def pass_through_verifier(state, context):
    return {"verdict": "ok"}


VERIFIER: Callable = pass_through_verifier


def after_dispatcher(state) -> str:
    return state["route"] if state.get("route") in AGENT_NAMES else "finalize"


def after_verifier(state) -> str:
    verdict = state.get("verdict", "ok")
    if verdict == "ok":
        return "finalize"
    if verdict == "retry" and state.get("retries", 0) <= MAX_RETRIES and state.get("hops", 0) < MAX_HOPS:
        return state["route"]
    return "fail"


NEXT: dict[str, Callable] = {
    "dispatcher": after_dispatcher,
    "calculator": lambda s: "verifier",
    "consultant": lambda s: "verifier",
    "examiner": lambda s: "verifier",
    "verifier": after_verifier,
    "fail": lambda s: "finalize",
    "finalize": lambda s: END,
}


def make_nodes(context: AgentContext, llm=None, agents=None, verifier=None) -> dict[str, Callable]:
    agents = {**AGENTS, **(agents or {})}
    verify = verifier or VERIFIER

    def wrap(fn: Callable) -> Callable:
        def node(state):
            update = dict(fn(state) or {})
            update["steps"] = list(state.get("steps", [])) + list(update.get("steps", []))
            update["hops"] = state.get("hops", 0) + 1
            return update
        return node

    def verifier_node(state):
        update = dict(verify(state, context) or {})
        if update.get("verdict") == "retry":
            update["retries"] = state.get("retries", 0) + 1
        return update

    def fail_node(state):
        return {"verdict": "fail", "answer": FAIL_TEXT, "refused": True}

    nodes = {"dispatcher": wrap(lambda s: Dispatcher(context, llm).route(s)),
             "verifier": wrap(verifier_node),
             "fail": wrap(fail_node),
             "finalize": wrap(lambda s: {})}
    for name in AGENT_NAMES:
        nodes[name] = wrap(lambda s, fn=agents[name]: fn(s, context))
    return nodes


def run_request(state: dict, *, student_id: str = "anon", llm=None, agents=None,
                verifier=None, use_langgraph: bool = False) -> dict:
    context = AgentContext(DummyTrace(), student_id=student_id)
    nodes = make_nodes(context, llm, agents, verifier)
    state = {**state}
    if use_langgraph:
        return build_graph(context, llm, agents, verifier).invoke(state)
    node = "dispatcher"
    while node != END:
        state.update(nodes[node](state))
        node = "fail" if state.get("hops", 0) >= MAX_HOPS and node != "finalize" else NEXT[node](state)
    return state


def build_graph(context: AgentContext, llm=None, agents=None, verifier=None):
    from langgraph.graph import END as LG_END, START, StateGraph

    nodes = make_nodes(context, llm, agents, verifier)
    graph = StateGraph(dict)
    for name, fn in nodes.items():
        graph.add_node(name, fn)
    graph.add_edge(START, "dispatcher")
    graph.add_conditional_edges("dispatcher", after_dispatcher,
                                {**{n: n for n in AGENT_NAMES}, "finalize": "finalize"})
    for name in AGENT_NAMES:
        graph.add_edge(name, "verifier")
    graph.add_conditional_edges("verifier", after_verifier,
                                {**{n: n for n in AGENT_NAMES}, "finalize": "finalize", "fail": "fail"})
    graph.add_edge("fail", "finalize")
    graph.add_edge("finalize", LG_END)
    return graph.compile()
