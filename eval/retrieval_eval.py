import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "retrieval"))

from retriever import HybridRetriever

GOLD_FILE = BASE_DIR / "eval/gold_questions.json"


def evaluate_retrieval(
    questions: list[dict],
    retriever: HybridRetriever,
    top_k: int = 5
) -> dict:

    results = []

    for q in questions:
        qid = q["id"]
        question = q["question"]
        required_papers = set(q["required_papers"])
        hops = q["required_hops"]

        retrieved = retriever.retrieve(
            question,
            top_k=top_k,
            include_abstracts=True
        )
        retrieved_ids = {r.arxiv_id for r in retrieved}

        found = required_papers & retrieved_ids
        recall = len(found) / len(required_papers)

        result = {
            "id": qid,
            "question": question[:80],
            "type": q["type"],
            "difficulty": q["difficulty"],
            "required_papers": list(required_papers),
            "retrieved_papers": list(retrieved_ids),
            "found": list(found),
            "missed": list(required_papers - retrieved_ids),
            "recall": recall,
            "hops": hops,
            "top_results": [
                {
                    "arxiv_id": r.arxiv_id,
                    "title": r.title[:50],
                    "rrf_score": round(r.rrf_score, 4),
                    "citation_count": r.citation_count,
                    "retrieval_method": r.retrieval_method
                }
                for r in retrieved
            ]
        }
        results.append(result)

    avg_recall = sum(r["recall"] for r in results) / len(results)
    perfect_recall = sum(1 for r in results if r["recall"] == 1.0)
    zero_recall = sum(1 for r in results if r["recall"] == 0.0)

    return {
        "total_questions": len(results),
        "avg_recall": round(avg_recall, 3),
        "perfect_recall": perfect_recall,
        "zero_recall": zero_recall,
        "results": results
    }


if __name__ == "__main__":
    with open(GOLD_FILE, encoding="utf-8") as f:
        gold_questions = json.load(f)

    print(f"Loaded {len(gold_questions)} gold questions")

    retriever = HybridRetriever()

    # Test all 20 questions
    print(f"\nEvaluating retrieval on all {len(gold_questions)} questions...")
    eval_results = evaluate_retrieval(gold_questions, retriever, top_k=10)

    print(f"\n{'='*60}")
    print(f"Retrieval Eval Results")
    print(f"{'='*60}")
    print(f"Questions:      {eval_results['total_questions']}")
    print(f"Avg Recall:     {eval_results['avg_recall']}")
    print(f"Perfect Recall: {eval_results['perfect_recall']}/{eval_results['total_questions']}")
    print(f"Zero Recall:    {eval_results['zero_recall']}/{eval_results['total_questions']}")

    print(f"\nPer-question breakdown:")
    for r in eval_results["results"]:
        status = "[OK]" if r["recall"] == 1.0 else "[WARNING]" if r["recall"] > 0 else "[FAIL]"
        print(f"\n{status} {r['id']} [{r['type']}, {r['difficulty']}]")
        print(f"   Q: {r['question'][:70]}")
        print(f"   Required: {r['required_papers']}")
        print(f"   Found:    {r['found']}")
        print(f"   Missed:   {r['missed']}")
        print(f"   Recall:   {r['recall']:.2f}")
        print(f"   Top results:")
        for t in r["top_results"]:
            print(f"     [{t['rrf_score']}] {t['arxiv_id']} — {t['title'][:45]}")

    # Save
    output_file = BASE_DIR / "eval/retrieval_eval_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nSaved → {output_file}")