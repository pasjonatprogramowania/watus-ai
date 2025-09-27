import json
import pathlib
import pytest
import requests
import random
from src import END_POINT_PROCESS_QUESTION,QUESTION_FILES_PATH

ITEM = "item"
CONTENT = "content"
ANSWER = "answer"
QUESTION = "question"


# Wczytujemy pytania/odpowiedzi z pliku .jsonl
def load_questions(sample_size: int = 20):
    with open(QUESTION_FILES_PATH, "r", encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f if line.strip()]
    return random.sample(all_items, min(sample_size, len(all_items)))


QUESTIONS = load_questions(20)


@pytest.mark.parametrize(ITEM, QUESTIONS)
def test_question_answer(item):
    """Testuje czy API zwraca sensowną odpowiedź na znane pytanie."""
    q = item[QUESTION]
    expected = item[ANSWER]

    r = requests.post(
        END_POINT_PROCESS_QUESTION,
        json={CONTENT: q},
        timeout=30
    )
    assert r.status_code == 200, f"HTTP {r.status_code} for question: {q}"

    data = r.json()
    answer = data.get(ANSWER, "").lower()

    # Sprawdzamy, czy chociaż fragment oczekiwanej odpowiedzi jest w odpowiedzi modelu
    # (często wystarczy 1-2 słowa kluczowe)
    key_tokens = expected.lower().split()
    matched = any(tok in answer for tok in key_tokens)

    assert matched, f"Expected fragment of '{expected}' in '{answer}'"
