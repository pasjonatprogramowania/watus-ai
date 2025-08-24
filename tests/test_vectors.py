# tests/test_vectors.py
import pytest
from main import (
    klasyfikuj_wektor,
    przetworz_logikę,
    DecisionVector, Tool, Action,
)

# --- Table-driven expectations (Pydantic used for shape/typing) ---

class VExp(DecisionVector):
    """Expected vector shape for comparison in tests."""

CASES = [
    # (prompt, expected DecisionVector, substrings that must be in answer)
    ("Jakie są kierunki na WAT?",
     VExp(dozwolone=True, czy_dzialanie=False, powazne=True, potrzeba_info=True, narzedzie=Tool.watoznawca),
     ["WAT", "kryptologię", "informatykę"]),

    ("Ile zarabia dziekan?",
     VExp(dozwolone=True, czy_dzialanie=False, powazne=False),
     ["komputer", "kawy"]),  # humor

    ("Czy na WAT są fajne dziewczyny?",
     VExp(dozwolone=True, czy_dzialanie=False, powazne=False),
     ["robotem"]),

    ("Który wydział na WAT jest najlepszy?",
     VExp(dozwolone=True, czy_dzialanie=False, powazne=True, potrzeba_info=True, narzedzie=Tool.watoznawca),
     ["Wydział", "Cybernetyki", "Mechatroniki"]),

    ("Jak sie studiuje na WAT?",
     VExp(dozwolone=True, czy_dzialanie=False, powazne=True, potrzeba_info=True, narzedzie=Tool.watoznawca),
     ["Wymagająco", "atmosfera"]),

    ("Chodź za mną.",
     VExp(dozwolone=True, czy_dzialanie=True, akcja=Action.sledzenie),
     ["Tryb śledzenia włączony", "Idę za Tobą"]),

    ("idź do pani w różowym bo coś od ciebie chce",
     VExp(dozwolone=True, czy_dzialanie=False, powazne=False),
     ["zajęty"]),

    ("Opowiedz mi historię WAT",
     VExp(dozwolone=True, czy_dzialanie=False, powazne=True, potrzeba_info=True, narzedzie=Tool.watoznawca),
     ["1951"]),

    ("czy PW jest lepsze niż WAT",
     VExp(dozwolone=True, czy_dzialanie=False, powazne=False),
     ["zależy", "studenci"]),

    ("Przestań za mną łazić",
     VExp(dozwolone=True, czy_dzialanie=True, akcja=Action.koniec_sledzenia),
     ["Tryb śledzenia wyłączony", "Zostaję"]),
]

@pytest.mark.parametrize("prompt, expected_vec, must", CASES)
def test_vector_and_answer(prompt, expected_vec, must):
    vec = klasyfikuj_wektor(prompt)
    # Compare field-by-field using Pydantic model semantics
    assert vec.dozwolone == expected_vec.dozwolone
    assert vec.czy_dzialanie == expected_vec.czy_dzialanie
    if vec.czy_dzialanie:
        assert vec.akcja == expected_vec.akcja
    else:
        assert vec.powazne == expected_vec.powazne
        if expected_vec.powazne:
            assert vec.potrzeba_info == expected_vec.potrzeba_info
            assert vec.narzedzie == expected_vec.narzedzie

    # Also check the final answer string
    out = przetworz_logikę(prompt).ostateczna_odpowiedz
    for s in must:
        assert s.lower() in out.lower(), f"Missing '{s}' in: {out}"
