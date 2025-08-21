# tests/test_vectors.py
import pytest
from main import klasyfikuj_wektor, przetworz_logikę, Tool, Action

CASES = [
    ("kierunki na WAT", dict(dzialanie=False, powazne=True, potrzeba=True, narzedzie=Tool.watoznawca)),
    ("Ile zarabia dziekan?", dict(dzialanie=False, powazne=False)),
    ("Czy są fajne dziewczyny?", dict(dzialanie=False, powazne=False)),
    ("który wydział jest najlepszy?", dict(dzialanie=False, powazne=True, potrzeba=True, narzedzie=Tool.watoznawca)),
    ("Jak sie studiuje?", dict(dzialanie=False, powazne=True, potrzeba=True, narzedzie=Tool.watoznawca)),
    ("chodź za mną", dict(dzialanie=True, akcja=Action.sledzenie)),
    ("idź do pani w różowym bo coś od ciebie chce", dict(dzialanie=False, powazne=False)),
    ("opowiedz mi historię wat", dict(dzialanie=False, powazne=True, potrzeba=True, narzedzie=Tool.watoznawca)),
    ("czy PW jest lepsze niż WAT", dict(dzialanie=False, powazne=False)),
    ("Przestań za mną łazić", dict(dzialanie=True, akcja=Action.koniec_sledzenia)),
]

@pytest.mark.parametrize("q,expect", CASES)
def test_vector(q, expect):
    v = klasyfikuj_wektor(q)
    assert v.dozwolone is True
    assert v.czy_dzialanie == expect.get("dzialanie")
    if not v.czy_dzialanie:
        assert v.powazne == expect.get("powazne")
        if v.powazne:
            assert v.potrzeba_info == expect.get("potrzeba", False)
            if expect.get("narzedzie"):
                assert v.narzedzie == expect["narzedzie"]
    else:
        assert v.akcja == expect["akcja"]

def test_outputs():
    # Smoke tests for a few canonical outputs
    assert "Idę za Tobą" in przetworz_logikę("chodź za mną").ostateczna_odpowiedz
    assert "Zostaję na miejscu" in przetworz_logikę("Przestań za mną łazić").ostateczna_odpowiedz
    assert "Wymagająco" in przetworz_logikę("Jak się studiuje na WAT?").ostateczna_odpowiedz
