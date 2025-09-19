import sys
import json
import random
import pathlib
import requests

# --- Naprawa ścieżek importu ---
ROOT = pathlib.Path(__file__).parent.parent.absolute()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import END_POINT_PROCESS_QUESTION

# --- Konfiguracja ---
DATA_FILE = ROOT / "data" / "questions.jsonl"
N = 30  # Ile losowych pytań testujemy

# --- Wczytanie pytań ---
with open(DATA_FILE, "r", encoding="utf-8") as f:
    all_items = [json.loads(line) for line in f if line.strip()]

sample = random.sample(all_items, min(N, len(all_items)))

results = []
correct = 0

# --- Główna pętla benchmarku ---
for item in sample:
    q = item["question"]
    expected = item["answer"]
    try:
        r = requests.post(
            END_POINT_PROCESS_QUESTION,
            json={"content": q},
            timeout=30
        )
        status = r.status_code
        if status == 200:
            data = r.json()
            answer = data.get("answer", "").lower()
            key_tokens = expected.lower().split()
            matched = any(tok in answer for tok in key_tokens)
            if matched:
                correct += 1
            results.append({
                "question": q,
                "expected": expected,
                "got": answer,
                "status": status,
                "matched": matched
            })
        else:
            results.append({
                "question": q,
                "expected": expected,
                "got": f"HTTP {status}",
                "status": status,
                "matched": False
            })
    except Exception as e:
        results.append({
            "question": q,
            "expected": expected,
            "got": str(e),
            "status": "error",
            "matched": False
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

print(f"✅ Benchmark finished: {correct}/{len(sample)} correct ({accuracy:.1f}%)")
print("📁 Full report saved to benchmark_results.json")
