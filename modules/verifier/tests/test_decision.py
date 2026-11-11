from checker.decision import decide

def test_pass_when_all_checks_pass():
    state = {
        "check_results": {
            "sources": {"overall_status": "pass", "issues": []},
            "numbers": {"status": "pass", "issues": []},
            "examiner": {"overall_status": "pass", "issues": []},
        },
        "retry_count": 0,
        "max_retries": 1,
    }
    assert decide(state)["verdict"] == "pass"

def test_one_retry_then_refuse():
    state = {
        "check_results": {"numbers": {"status": "revise", "issues": ["Неверное число"]}},
        "retry_count": 0,
        "max_retries": 1,
    }
    first = decide(state)
    assert first["verdict"] == "revise"
    state["retry_count"] = first["retry_count"]
    second = decide(state)
    assert second["verdict"] == "refuse"

def test_missing_judge_is_not_pass():
    state = {
        "check_results": {"sources": {"overall_status": "needs_review", "issues": ["No LLM"]}},
        "retry_count": 0,
        "max_retries": 1,
    }
    assert decide(state)["verdict"] == "needs_review"
