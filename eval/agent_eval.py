import sys
import json
import time
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "agents"))
sys.path.insert(0, str(BASE_DIR / "src" / "retrieval"))

from graph import run_agent

# ── Corpus inventory ──────────────────────────────────────────────────────────
def load_corpus_ids() -> set[str]:
    db_path = BASE_DIR / "data" / "paperpilot.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT arxiv_id FROM papers")
    ids = {row[0] for row in cursor.fetchall()}
    conn.close()
    return ids

corpus_ids = load_corpus_ids()
print(f"Corpus size: {len(corpus_ids)} papers")

# ── Gold questions ─────────────────────────────────────────────────────────────
expanded_path = BASE_DIR / "eval" / "gold_questions_expanded.json"
original_path = BASE_DIR / "eval" / "gold_questions.json"

if expanded_path.exists():
    gold_path = expanded_path
    print(f"Using expanded gold questions: {gold_path}")
else:
    gold_path = original_path
    print(f"Using original gold questions: {gold_path}")

with open(gold_path) as f:
    gold_questions = json.load(f)

def get_query(item: dict) -> str:
    return item.get("question") or item.get("query", "")

# ── Resume from existing results ───────────────────────────────────────────────
out_path = BASE_DIR / "eval" / "agent_eval_results.json"
completed_ids = set()

passed = 0
needs_revision = 0
failed = 0
errors = 0

if out_path.exists():
    with open(out_path) as f:
        results = json.load(f)
    completed_ids = {r["id"] for r in results}
    print(f"Resuming — {len(completed_ids)} questions already completed")
    # Restore counters from existing results
    for r in results:
        status = r.get("verification_status", "unknown")
        if status == "passed":
            passed += 1
        elif status == "needs_revision":
            needs_revision += 1
        elif status == "failed":
            failed += 1
        elif status == "error":
            errors += 1
else:
    results = []

print(f"Running agent eval on {len(gold_questions)} questions...")
print(f"{'='*60}")

for i, item in enumerate(gold_questions):
    qid = item["id"]
    query = get_query(item)
    gold_papers = set(item.get("required_papers", []))

    # Skip already completed
    if qid in completed_ids:
        print(f"[{i+1}/{len(gold_questions)}] {qid}: already done — skipping")
        continue

    # Split gold papers into in-corpus and out-of-corpus
    gold_in_corpus = gold_papers & corpus_ids
    gold_not_in_corpus = gold_papers - corpus_ids

    print(f"\n[{i+1}/{len(gold_questions)}] {qid}: {query[:60]}...")
    if gold_not_in_corpus:
        print(f"  [WARN]  {len(gold_not_in_corpus)} gold paper(s) not in corpus: {gold_not_in_corpus}")

    try:
        result = run_agent(query)

        cited = set(result.get("citations", []))
        hits = cited & gold_papers

        citation_recall = len(hits) / len(gold_papers) if gold_papers else 0.0
        adjusted_recall = len(hits) / len(gold_in_corpus) if gold_in_corpus else 0.0

        perfect = citation_recall == 1.0
        adjusted_perfect = adjusted_recall == 1.0

        status = result.get("verification_status", "unknown")
        if status == "passed":
            passed += 1
        elif status == "needs_revision":
            needs_revision += 1
        elif status == "failed":
            failed += 1

        results.append({
            "id": qid,
            "query": query,
            "verification_status": status,
            "citation_recall": citation_recall,
            "perfect_recall": perfect,
            "adjusted_recall": adjusted_recall,
            "adjusted_perfect": adjusted_perfect,
            "cited_papers": sorted(cited),
            "gold_papers": sorted(gold_papers),
            "gold_in_corpus": sorted(gold_in_corpus),
            "gold_not_in_corpus": sorted(gold_not_in_corpus),
            "hits": sorted(hits),
            "answer": result.get("answer", ""),
            "answer_length": len(result.get("answer", "")),
            "iterations_used": result.get("iterations_used", 0),
            "citations_count": len(cited),
            "question_type": item.get("question_type", "unknown"),
            "difficulty": item.get("difficulty", "unknown"),
            "hops": item.get("hops", 0)
        })

        print(f"  Status:   {status}")
        print(f"  Recall:   {citation_recall:.2f} raw | {adjusted_recall:.2f} adjusted")
        print(f"  Perfect:  {perfect} raw | {adjusted_perfect} adjusted")

        # Save incrementally every 5 queries
        if len(results) % 5 == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  [Saved {len(results)} results]")

        time.sleep(3)

    except Exception as e:
        print(f"  ERROR: {e}")
        errors += 1
        results.append({
            "id": qid,
            "query": query,
            "error": str(e),
            "verification_status": "error",
            "citation_recall": 0.0,
            "adjusted_recall": 0.0,
            "perfect_recall": False,
            "adjusted_perfect": False,
            "cited_papers": [],
            "gold_papers": sorted(gold_papers),
            "gold_in_corpus": sorted(gold_in_corpus),
            "gold_not_in_corpus": sorted(gold_not_in_corpus),
            "hits": [],
            "question_type": item.get("question_type", "unknown"),
            "difficulty": item.get("difficulty", "unknown"),
        })
        time.sleep(3)

# ── Final save ─────────────────────────────────────────────────────────────────
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

# ── Summary ────────────────────────────────────────────────────────────────────
valid = [r for r in results if r.get("verification_status") != "error"]

avg_citation_recall = sum(r["citation_recall"] for r in valid) / len(valid) if valid else 0
avg_adjusted_recall = sum(r["adjusted_recall"] for r in valid) / len(valid) if valid else 0
perfect_count = sum(1 for r in valid if r.get("perfect_recall", False))
adjusted_perfect_count = sum(1 for r in valid if r.get("adjusted_perfect", False))

type_stats = {}
for r in valid:
    qt = r.get("question_type", "unknown")
    if qt not in type_stats:
        type_stats[qt] = {"count": 0, "recall_sum": 0.0, "adjusted_sum": 0.0}
    type_stats[qt]["count"] += 1
    type_stats[qt]["recall_sum"] += r["citation_recall"]
    type_stats[qt]["adjusted_sum"] += r["adjusted_recall"]

print(f"\n{'='*60}")
print(f"EVAL COMPLETE")
print(f"{'='*60}")
print(f"Total:              {len(gold_questions)}")
print(f"Errors:             {errors}")
print(f"Passed:             {passed}")
print(f"Needs revision:     {needs_revision}")
print(f"Failed:             {failed}")
print(f"")
print(f"Avg citation recall:  {avg_citation_recall:.3f}  (Phase 2 baseline: 0.675)")
print(f"Avg adjusted recall:  {avg_adjusted_recall:.3f}  (corpus-aware)")
print(f"Perfect (raw):        {perfect_count}/{len(valid)}")
print(f"Perfect (adjusted):   {adjusted_perfect_count}/{len(valid)}")
print(f"")
print(f"Recall by question type:")
for qt, stats in sorted(type_stats.items()):
    n = stats["count"]
    avg_r = stats["recall_sum"] / n
    avg_a = stats["adjusted_sum"] / n
    print(f"  {qt:<15} n={n}  raw={avg_r:.2f}  adjusted={avg_a:.2f}")

print(f"\nResults saved to {out_path}")