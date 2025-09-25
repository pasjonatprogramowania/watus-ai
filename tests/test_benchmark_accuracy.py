import json
import random
import requests
from src import END_POINT_PROCESS_QUESTION,QUESTION_FILES_PATH

MATCHED = "matched"
STATUS = "status"
GOT = "got"
EXPECTED = "expected"
ANSWER = "answer"
QUESTION = "question"

# --- Konfiguracja ---

N = 30  # Ile losowych pytań testujemy

# --- Wczytanie pytań ---
with open(QUESTION_FILES_PATH, "r", encoding="utf-8") as f:
    all_items = [json.loads(line) for line in f if line.strip()]

sample = random.sample(all_items, min(N, len(all_items)))

results = []
correct = 0

# --- Główna pętla benchmarku ---
for item in sample:
    q = item[QUESTION]
    expected = item[ANSWER]
    try:
        r = requests.post(
            END_POINT_PROCESS_QUESTION,
            json={"content": q},
            timeout=30
        )
        status = r.status_code
        if status == 200:
            data = r.json()
            answer = data.get(ANSWER, "").lower()
            key_tokens = expected.lower().split()
            matched = any(tok in answer for tok in key_tokens)
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
            results.append({
                QUESTION: q,
                EXPECTED: expected,
                GOT: f"HTTP {status}",
                STATUS: status,
                MATCHED: False
            })
    except Exception as e:
        results.append({
            QUESTION: q,
            EXPECTED: expected,
            GOT: str(e),
            STATUS: "error",
            MATCHED: False
        })

# --- Podsumowanie ---
accuracy = correct / len(sample) * 100

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
