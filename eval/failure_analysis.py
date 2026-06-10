import csv
import json
import re
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_FILE = BASE_DIR / "eval" / "retrieval_eval_results.json"
ENRICHED_FILE = BASE_DIR / "data" / "raw" / "papers_enriched.json"
OUT_CSV = BASE_DIR / "eval" / "failure_analysis.csv"
OUT_JSON = BASE_DIR / "eval" / "failure_analysis_summary.json"

LOW_RECALL_THRESHOLD = 0.4


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("-", " ")
    parts = re.split(r"[^a-z0-9]+", text)
    return [p for p in parts if p]


def load_title_map() -> dict[str, str]:
    with open(ENRICHED_FILE, encoding="utf-8") as f:
        papers = json.load(f)
    return {p["arxiv_id"]: p.get("title", "") for p in papers}


def classify_case(result: dict, title_map: dict[str, str]) -> dict:
    recall = result.get("recall", 0.0)
    required = set(result.get("required_papers", []))
    retrieved = set(result.get("retrieved_papers", []))
    found = set(result.get("found", []))
    missed = list(set(result.get("missed", [])))

    if recall == 0:
        severity = "zero"
    elif recall < LOW_RECALL_THRESHOLD:
        severity = "low"
    else:
        severity = "ok"

    top = result.get("top_results", [])
    hybrid = [t for t in top if t.get("retrieval_method") == "hybrid"]
    supplement = [t for t in top if t.get("retrieval_method") == "abstract_supplement"]
    hybrid_ids = {t.get("arxiv_id", "") for t in hybrid}
    supplement_ids = {t.get("arxiv_id", "") for t in supplement}

    # Token-overlap heuristic between missed titles and retrieved titles
    retrieved_title_tokens = set()
    for arxiv_id in retrieved:
        retrieved_title_tokens.update(tokenize(title_map.get(arxiv_id, "")))

    max_overlap = 0.0
    missed_best = ""
    for m in missed:
        m_tokens = set(tokenize(title_map.get(m, "")))
        if not m_tokens:
            continue
        overlap = len(m_tokens & retrieved_title_tokens) / max(1, len(m_tokens))
        if overlap > max_overlap:
            max_overlap = overlap
            missed_best = m

    if not missed:
        hint = "no_issue"
    elif not hybrid:
        hint = "dense_or_fusion_empty"
    elif required & supplement_ids and not (required & hybrid_ids):
        hint = "hybrid_miss_abstract_recovery"
    elif max_overlap < 0.2:
        hint = "query_mismatch_or_sparse_vocab"
    elif max_overlap < 0.5:
        hint = "rerank_or_topk_limit"
    else:
        hint = "near_miss_expand_topk"

    return {
        "id": result.get("id", ""),
        "type": result.get("type", ""),
        "difficulty": result.get("difficulty", ""),
        "recall": round(recall, 3),
        "severity": severity,
        "required_count": len(required),
        "retrieved_count": len(retrieved),
        "found_count": len(found),
        "missed_count": len(missed),
        "required_papers": ";".join(sorted(required)),
        "retrieved_papers": ";".join(sorted(retrieved)),
        "found_papers": ";".join(sorted(found)),
        "missed_papers": ";".join(sorted(missed)),
        "hybrid_hits": len(hybrid_ids & required),
        "abstract_supplement_hits": len(supplement_ids & required),
        "max_missed_title_overlap": round(max_overlap, 3),
        "closest_missed_paper": missed_best,
        "root_cause_hint": hint,
    }


def main():
    with open(EVAL_FILE, encoding="utf-8") as f:
        eval_data = json.load(f)
    title_map = load_title_map()

    rows = [
        classify_case(r, title_map)
        for r in eval_data.get("results", [])
    ]

    rows_sorted = sorted(
        rows,
        key=lambda x: (x["severity"] != "zero", x["severity"] != "low", x["recall"])
    )

    fieldnames = list(rows_sorted[0].keys()) if rows_sorted else []
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)

    counts = Counter(r["root_cause_hint"] for r in rows_sorted if r["severity"] != "ok")
    summary = {
        "total_questions": len(rows_sorted),
        "zero_recall_cases": sum(1 for r in rows_sorted if r["severity"] == "zero"),
        "low_recall_cases": sum(1 for r in rows_sorted if r["severity"] == "low"),
        "hints_breakdown": dict(counts),
        "csv_file": str(OUT_CSV),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Failure analysis complete")
    print(f"  CSV: {OUT_CSV}")
    print(f"  Summary: {OUT_JSON}")
    print(f"  Zero recall: {summary['zero_recall_cases']}")
    print(f"  Low recall:  {summary['low_recall_cases']}")
    if counts:
        print("  Root-cause hints:")
        for k, v in counts.items():
            print(f"    - {k}: {v}")


if __name__ == "__main__":
    main()
