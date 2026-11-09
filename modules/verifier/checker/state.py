from typing import Any, TypedDict, Literal


Verdict = Literal["pass", "revise", "refuse", "needs_review"]


class PhysicsLabState(TypedDict, total=False):
    # User input and experiment context
    student_message: str
    context: dict[str, Any]
    untrusted_inputs: list[dict[str, Any]]
    sanitized_untrusted_inputs: list[dict[str, Any]]
    sanitized_student_message: str

    # Agent output being checked
    agent_name: str
    agent_response: str
    sources: list[dict[str, Any]]
    numbers: dict[str, Any]

    # Examiner-specific payload
    examiner_payload: dict[str, Any]

    # Checker results
    security_result: dict[str, Any]
    source_result: dict[str, Any]
    numerical_result: dict[str, Any]
    examiner_result: dict[str, Any]
    check_results: dict[str, Any]

    # Decision and retry control
    verdict: Verdict
    feedback: list[str]
    retry_count: int
    max_retries: int
    final_response: str
    refusal_reason: str

    # Observability
    audit_log: list[dict[str, Any]]
    run_id: str
