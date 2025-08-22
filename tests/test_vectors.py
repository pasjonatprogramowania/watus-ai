# tests/test_vectors.py
import pytest
from src.main import classify_vector, przetworz_logikę, Tool, Action

CASES = [
    ("kierunki na WAT", dict(dzialanie=False, powazne=True, potrzeba=True, narzedzie=Tool.search_google)),
    ("Ile zarabia dziekan?", dict(dzialanie=False, powazne=False)),
    ("Czy są fajne dziewczyny?", dict(dzialanie=False, powazne=False)),
    ("który wydział jest najlepszy?", dict(dzialanie=False, powazne=True, potrzeba=True, narzedzie=Tool.search_google)),
    ("Jak sie studiuje?", dict(dzialanie=False, powazne=True, potrzeba=True, narzedzie=Tool.search_google)),
    ("chodź za mną", dict(dzialanie=True, akcja=Action.follow_action)),
    ("idź do pani w różowym bo coś od ciebie chce", dict(dzialanie=False, powazne=False)),
    ("opowiedz mi historię wat", dict(dzialanie=False, powazne=True, potrzeba=True, narzedzie=Tool.search_google)),
    ("czy PW jest lepsze niż WAT", dict(dzialanie=False, powazne=False)),
    ("Przestań za mną łazić", dict(dzialanie=True, akcja=Action.end_action)),
]

@pytest.mark.parametrize("q,expect", CASES)
def test_vector(q, expect):
    v = classify_vector(q)
    assert v.is_allowed is True
    assert v.is_actions_required == expect.get("dzialanie")
    if not v.is_actions_required:
        assert v.is_serious == expect.get("powazne")
        if v.is_serious:
            assert v.is_tool_required == expect.get("potrzeba", False)
            if expect.get("narzedzie"):
                assert v.required_tools == expect["narzedzie"]
    else:
        assert v.required_actions == expect["akcja"]

def test_outputs():
    # Smoke tests for a few canonical outputs
    assert "Idę za Tobą" in przetworz_logikę("chodź za mną").last_answer
    assert "Zostaję na miejscu" in przetworz_logikę("Przestań za mną łazić").last_answer
    assert "Wymagająco" in przetworz_logikę("Jak się studiuje na WAT?").last_answer
