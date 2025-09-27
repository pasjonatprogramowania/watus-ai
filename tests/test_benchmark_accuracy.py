import sys
import json
import random
import requests
import pathlib

# Dodaj katalog główny projektu do ścieżek importu, jeśli nie już
ROOT = pathlib.Path(__file__).parent.parent.absolute()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import END_POINT_PROCESS_QUESTION, QUESTION_FILES_PATH

MATCHED = "matched"
STATUS = "status"
GOT = "got"
EXPECTED = "expected"
ANSWER = "answer"
QUESTION = "question"

# --- Konfiguracja ---
random.seed(42)  # stały seed, by test był powtarzalny
N = 30  # Ile losowych pytań testujemy

# --- Wczytanie pytań ---
with open(QUESTION_FILES_PATH, "r", encoding="utf-8") as f:
    all_items = [json.loads(line) for line in f if line.strip()]

if not all_items:
    raise RuntimeError(f"No items in {QUESTION_FILES_PATH}")

sample = random.sample(all_items, min(N, len(all_items)))

results = []
correct = 0

# --- Główna pętla benchmarku ---
for item in sample:
    q = item.get(QUESTION)
    expected = item.get(ANSWER)
    try:
        r = requests.post(
            END_POINT_PROCESS_QUESTION,
            json={"content": q},
            timeout=30
        )
        status = r.status_code
        text = r.text
        if status == 200:
            data = r.json()
            # jeśli odpowiedź to struktura z kluczami "answer" i "decisionVector"
            answer = data.get(ANSWER, "")
            if answer is None:
                answer = ""
            answer_lower = answer.lower()
            key_tokens = expected.lower().split()
            matched = any(tok in answer_lower for tok in key_tokens)
            if matched:
                correct += 1
            results.append({
                QUESTION: q,
                EXPECTED: expected,
                GOT: answer,
                STATUS: status,
                MATCHED: matched
            })
        else:
            # log surowego response dla debugu
            print(f"Benchmark: got non-200 status {status}, body: {text}")
            results.append({
                QUESTION: q,
                EXPECTED: expected,
                GOT: f"HTTP {status}: {text}",
                STATUS: status,
                MATCHED: False
            })
    except Exception as e:
        print(f"Benchmark: exception for query '{q}': {e}")
        results.append({
            QUESTION: q,
            EXPECTED: expected,
            GOT: str(e),
            STATUS: "error",
            MATCHED: False
        })

# --- Podsumowanie ---
accuracy = correct / len(sample) * 100 if sample else 0.0

report = {
    "total": len(sample),
    "correct": correct,
    "accuracy_percent": accuracy,
    "results": results
}

with open("benchmark_results.json", "w", encoding="utf-8") as out:
    json.dump(report, out, ensure_ascii=False, indent=2)

print(f"Benchmark finished: {correct}/{len(sample)} correct ({accuracy:.1f}%)")
print("Full report saved to benchmark_results.json")
