from checker.numerical.validator import validate_numbers

def test_correct_number():
    result = validate_numbers(
        "Сила тока равна 2 А.",
        {"current": {"value": 2.0, "unit": "ampere", "display_name": "сила тока"}}
    )
    assert result["status"] == "pass"

def test_wrong_number():
    result = validate_numbers(
        "Сила тока равна 3 А.",
        {"current": {"value": 2.0, "unit": "ampere", "display_name": "сила тока"}}
    )
    assert result["status"] == "revise"
