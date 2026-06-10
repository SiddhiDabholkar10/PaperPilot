import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

with open(BASE_DIR / "eval/gold_questions_expanded.json"
          if (BASE_DIR / "eval/gold_questions_expanded.json").exists()
          else BASE_DIR / "eval/gold_questions.json") as f:
    gold = json.load(f)

with open(BASE_DIR / "eval/agent_eval_results.json") as f:
    results = json.load(f)

print("Agent eval — citation recall vs Phase 2 baseline")
print(f"{'ID':<8} {'Raw':>6} {'Adj':>6} {'Perf':>5} {'Status':<20} {'Type':<15} {'Cited':>5}")
print("-" * 80)

total_raw = 0
total_adj = 0
perfect_raw = 0
perfect_adj = 0
passed = needs_revision = failed = errors = 0

for r in results:
    qid = r["id"]
    raw = r.get("citation_recall", 0.0)
    adj = r.get("adjusted_recall", raw)
    is_perfect_raw = r.get("perfect_recall", False)
    is_perfect_adj = r.get("adjusted_perfect", False)
    status = r.get("verification_status", "unknown")
    qt = r.get("question_type", "?")[:14]
    cited = r.get("citations_count", len(r.get("cited_papers", [])))
    not_in_corpus = r.get("gold_not_in_corpus", [])

    total_raw += raw
    total_adj += adj
    if is_perfect_raw:
        perfect_raw += 1
    if is_perfect_adj:
        perfect_adj += 1
    if status == "passed":
        passed += 1
    elif status == "needs_revision":
        needs_revision += 1
    elif status == "failed":
        failed += 1
    elif status == "error":
        errors += 1

    flag = "[OK]" if is_perfect_adj else "[FAIL]"
    corpus_note = f" [WARNING] {len(not_in_corpus)} missing" if not_in_corpus else ""
    print(f"{qid:<8} {raw:>6.2f} {adj:>6.2f} {flag:>5} {status:<20} {qt:<15} {cited:>5}{corpus_note}")

n = len(results)
print("-" * 80)
print(f"{'AVG':<8} {total_raw/n:>6.2f} {total_adj/n:>6.2f}")
print()
print(f"Phase 2 retrieval baseline:  0.675")
print(f"Agent citation recall (raw): {total_raw/n:.3f}")
print(f"Agent adjusted recall:       {total_adj/n:.3f}  ← apples-to-apples comparison")
print(f"Perfect (raw):               {perfect_raw}/{n}")
print(f"Perfect (adjusted):          {perfect_adj}/{n}")
print()
print(f"Verification — passed:{passed}  needs_revision:{needs_revision}  failed:{failed}  errors:{errors}")