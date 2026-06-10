import json
import re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHUNKS_DIR = BASE_DIR / "data/chunks"


def validate_chunks(
    chunks_dir: Path = CHUNKS_DIR,
) -> dict:

    print("=" * 60)
    print("PaperPilot Data Validator")
    print("=" * 60)

    # Load all chunk types
    chunk_files = {
        "abstract": chunks_dir / "chunks_abstract.json",
        "parent": chunks_dir / "chunks_parent.json",
        "child": chunks_dir / "chunks_child.json"
    }

    all_chunks = {}
    for chunk_type, path in chunk_files.items():
        if not path.exists():
            print(f"ERROR: {path} not found")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            all_chunks[chunk_type] = json.load(f)
        print(f"Loaded {len(all_chunks[chunk_type])} {chunk_type} chunks")

    issues = defaultdict(list)
    stats = defaultdict(int)

    # Required metadata fields for Pinecone
    REQUIRED_FIELDS = [
        "chunk_id", "arxiv_id", "title", "year",
        "section", "chunk_type", "text"
    ]

    # Required for child chunks specifically
    CHILD_REQUIRED = ["parent_chunk_id"]

    # Collect all chunk IDs and parent IDs for cross-reference checks
    all_chunk_ids = set()
    all_parent_ids = set()

    for chunk_type, chunks in all_chunks.items():
        print(f"\n--- Validating {chunk_type} chunks ---")

        for chunk in tqdm(chunks, desc=f"Checking {chunk_type}"):
            chunk_id = chunk.get("chunk_id", "MISSING_ID")
            stats["total"] += 1

            # 1. Duplicate chunk IDs
            if chunk_id in all_chunk_ids:
                issues["duplicate_ids"].append(chunk_id)
                stats["duplicate_ids"] += 1
            else:
                all_chunk_ids.add(chunk_id)

            if chunk_type == "parent":
                all_parent_ids.add(chunk_id)

            # 2. Missing required fields
            for field in REQUIRED_FIELDS:
                if field not in chunk or chunk[field] is None:
                    issues["missing_fields"].append(
                        f"{chunk_id}: missing '{field}'"
                    )
                    stats["missing_fields"] += 1

            # 3. Child chunks must have parent_chunk_id
            if chunk_type == "child":
                for field in CHILD_REQUIRED:
                    if not chunk.get(field):
                        issues["missing_parent_id"].append(chunk_id)
                        stats["missing_parent_id"] += 1

            # 4. Empty text
            text = chunk.get("text", "")
            if not text or not text.strip():
                issues["empty_text"].append(chunk_id)
                stats["empty_text"] += 1
                continue

            # 5. Very short chunks (< 50 chars)
            if len(text.strip()) < 50:
                issues["too_short"].append(
                    f"{chunk_id}: {len(text)} chars — '{text[:50]}'"
                )
                stats["too_short"] += 1

            # 6. Symbol-only chunks (< 40% alphabetic)
            alpha_ratio = sum(c.isalpha() for c in text) / len(text)
            if alpha_ratio < 0.40:
                issues["symbol_heavy"].append(
                    f"{chunk_id}: {alpha_ratio:.2f} alpha ratio"
                )
                stats["symbol_heavy"] += 1

            # 7. Missing title
            if not chunk.get("title", "").strip():
                issues["missing_title"].append(chunk_id)
                stats["missing_title"] += 1

            # 8. Missing section
            if not chunk.get("section", "").strip():
                issues["missing_section"].append(chunk_id)
                stats["missing_section"] += 1

            # 9. Invalid year
            year = chunk.get("year", "")
            if not str(year).isdigit() or not (2014 <= int(str(year)) <= 2027):
                issues["invalid_year"].append(
                    f"{chunk_id}: year='{year}'"
                )
                stats["invalid_year"] += 1

            # 10. Metadata type checks
            if not isinstance(chunk.get("authors", []), list):
                issues["invalid_authors_type"].append(chunk_id)
                stats["invalid_authors_type"] += 1

            if not isinstance(chunk.get("categories", []), list):
                issues["invalid_categories_type"].append(chunk_id)
                stats["invalid_categories_type"] += 1

    # 11. Orphan child chunks — child points to non-existent parent
    print("\n--- Checking orphan child chunks ---")
    orphan_count = 0
    for chunk in tqdm(all_chunks["child"], desc="Checking orphans"):
        parent_id = chunk.get("parent_chunk_id", "")
        if parent_id and parent_id not in all_parent_ids:
            issues["orphan_children"].append(
                f"{chunk['chunk_id']} → {parent_id}"
            )
            orphan_count += 1
    stats["orphan_children"] = orphan_count

    # --- Print report ---
    print(f"\n{'=' * 60}")
    print("Validation Report")
    print(f"{'=' * 60}")
    print(f"Total chunks checked: {stats['total']}")

    critical_issues = [
        "duplicate_ids", "missing_fields", "missing_parent_id",
        "empty_text", "orphan_children"
    ]
    warning_issues = [
        "too_short", "symbol_heavy", "missing_title",
        "missing_section", "invalid_year",
        "invalid_authors_type", "invalid_categories_type"
    ]

    print(f"\nCRITICAL issues (must fix before embedding):")
    has_critical = False
    for issue in critical_issues:
        count = stats.get(issue, 0)
        status = "[FAIL]" if count > 0 else "[OK]"
        print(f"  {status} {issue}: {count}")
        if count > 0:
            has_critical = True

    print(f"\nWARNINGS (review before embedding):")
    has_warnings = False
    for issue in warning_issues:
        count = stats.get(issue, 0)
        status = "[WARNING]" if count > 0 else "[OK]"
        print(f"  {status} {issue}: {count}")
        if count > 0:
            has_warnings = True

    # Show samples of issues
    for issue_type, issue_list in issues.items():
        if issue_list:
            print(f"\nSample {issue_type} (first 3):")
            for item in issue_list[:3]:
                print(f"  {item}")

    # Save report
    report = {
        "total_chunks": stats["total"],
        "critical_issues": {k: stats.get(k, 0) for k in critical_issues},
        "warnings": {k: stats.get(k, 0) for k in warning_issues},
        "issue_samples": {k: v[:5] for k, v in issues.items()}
    }

    report_file = CHUNKS_DIR / "validation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nReport saved → {report_file}")

    if not has_critical and not has_warnings:
        print("\n✓ All checks passed — ready to embed")
    elif not has_critical:
        print("\n‼  Warnings found — review before embedding")
    else:
        print("\n✕ Critical issues found — fix before embedding")

    return report


if __name__ == "__main__":
    report = validate_chunks()