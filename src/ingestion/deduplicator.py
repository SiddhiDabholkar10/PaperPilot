import json
import re
from pathlib import Path
from tqdm import tqdm
BASE_DIR = Path(__file__).resolve().parent.parent.parent
METADATA_FILE = BASE_DIR / "data/raw/papers_metadata.json"


def normalize_title(title: str) -> str:
    """Normalize title for comparison."""
    # Lowercase
    title = title.lower()
    # Remove punctuation
    title = re.sub(r'[^\w\s]', '', title)
    # Remove extra whitespace
    title = re.sub(r'\s+', ' ', title).strip()
    return title


def title_similarity(t1: str, t2: str) -> float:
    """Simple word overlap similarity between two titles."""
    words1 = set(normalize_title(t1).split())
    words2 = set(normalize_title(t2).split())

    # Remove common stop words
    stop_words = {
        'a', 'an', 'the', 'of', 'in', 'on', 'for',
        'to', 'and', 'or', 'with', 'via', 'using',
        'towards', 'toward', 'from', 'is', 'are'
    }
    words1 = words1 - stop_words
    words2 = words2 - stop_words

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


def abstract_similarity(a1: str, a2: str) -> float:
    """Simple word overlap similarity between two abstracts."""
    # Use first 200 chars only — enough to detect duplicates, fast
    a1 = a1[:200].lower()
    a2 = a2[:200].lower()

    words1 = set(re.findall(r'\w+', a1))
    words2 = set(re.findall(r'\w+', a2))

    stop_words = {
        'a', 'an', 'the', 'of', 'in', 'on', 'for',
        'to', 'and', 'or', 'with', 'we', 'our',
        'this', 'that', 'is', 'are', 'paper', 'propose'
    }
    words1 = words1 - stop_words
    words2 = words2 - stop_words

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


def deduplicate_papers(
    metadata_file: Path = METADATA_FILE,
    title_threshold: float = 0.85,
    abstract_threshold: float = 0.80,
) -> list[dict]:
    """
    Remove duplicate papers from metadata.
    
    A paper is considered duplicate if:
    - Same arxiv ID (exact duplicate)
    - Title similarity > title_threshold AND abstract similarity > abstract_threshold
    """

    with open(metadata_file, "r", encoding="utf-8") as f:
        papers = json.load(f)

    print(f"Starting deduplication: {len(papers)} papers")

    # Step 1: Deduplicate by exact arxiv ID
    seen_ids = {}
    id_deduped = []
    for paper in papers:
        arxiv_id = paper["arxiv_id"]
        if arxiv_id not in seen_ids:
            seen_ids[arxiv_id] = True
            id_deduped.append(paper)

    id_dupes_removed = len(papers) - len(id_deduped)
    print(f"  Exact ID duplicates removed: {id_dupes_removed}")

    # Step 2: Deduplicate by title + abstract similarity
    unique_papers = []
    removed = []

    for i, paper in enumerate(tqdm(id_deduped, desc="Checking near-duplicates")):
        is_duplicate = False

        for unique_paper in unique_papers:
            t_sim = title_similarity(paper["title"], unique_paper["title"])

            # Only check abstract if title is similar (saves computation)
            if t_sim > title_threshold:
                a_sim = abstract_similarity(
                    paper.get("abstract", ""),
                    unique_paper.get("abstract", "")
                )
                if a_sim > abstract_threshold:
                    is_duplicate = True
                    removed.append({
                        "removed": paper["arxiv_id"],
                        "kept": unique_paper["arxiv_id"],
                        "title_sim": round(t_sim, 3),
                        "abstract_sim": round(a_sim, 3),
                        "title": paper["title"]
                    })
                    break

        if not is_duplicate:
            unique_papers.append(paper)

    near_dupes_removed = len(id_deduped) - len(unique_papers)
    print(f"  Near-duplicate papers removed: {near_dupes_removed}")
    print(f"  Final paper count: {len(unique_papers)}")

    if removed:
        print(f"\nRemoved near-duplicates:")
        for r in removed:
            print(f"  {r['removed']} ~ {r['kept']}")
            print(f"    Title sim: {r['title_sim']}, Abstract sim: {r['abstract_sim']}")
            print(f"    Title: {r['title'][:70]}")

    # Save deduplicated metadata back
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(unique_papers, f, indent=2)

    # Save dedup report
    report_file = metadata_file.parent / "dedup_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "original_count": len(papers),
            "final_count": len(unique_papers),
            "exact_dupes_removed": id_dupes_removed,
            "near_dupes_removed": near_dupes_removed,
            "removed_papers": removed
        }, f, indent=2)

    print(f"\nDedup report saved to {report_file}")
    return unique_papers


if __name__ == "__main__":
    papers = deduplicate_papers()