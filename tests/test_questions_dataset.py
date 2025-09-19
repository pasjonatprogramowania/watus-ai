import json
import pathlib
import pytest
import requests
import random

# Ścieżka do pliku .jsonl
DATA_FILE = pathlib.Path(__file__).parent.parent / "data" / "questions.jsonl"

# Twój endpoint API (importujemy z __init__.py żeby nie duplikować)
from src import END_POINT_PROCESS_QUESTION


# Wczytujemy pytania/odpowiedzi z pliku .jsonl
# Wczytujemy pytania/odpowiedzi z pliku .jsonl
def load_questions(sample_size: int = 20):
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f if line.strip()]
    return random.sample(all_items, min(sample_size, len(all_items)))


QUESTIONS = load_questions(20)


@pytest.mark.parametrize("item", QUESTIONS)
def test_question_answer(item):
    """Testuje czy API zwraca sensowną odpowiedź na znane pytanie."""
    q = item["question"]
    expected = item["answer"]

    r = requests.post(
        END_POINT_PROCESS_QUESTION,
        json={"content": q},
        timeout=30
    )
    assert r.status_code == 200, f"HTTP {r.status_code} for question: {q}"

    data = r.json()
    answer = data.get("answer", "").lower()

    # Sprawdzamy, czy chociaż fragment oczekiwanej odpowiedzi jest w odpowiedzi modelu
    # (często wystarczy 1-2 słowa kluczowe)
    key_tokens = expected.lower().split()
    matched = any(tok in answer for tok in key_tokens)

    assert matched, f"Expected fragment of '{expected}' in '{answer}'"
