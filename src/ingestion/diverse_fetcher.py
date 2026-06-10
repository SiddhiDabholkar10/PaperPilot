import arxiv
import json
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from ids import make_paper_id

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "data/raw"
CATEGORIES = ["cs.LG", "cs.CL"]
DELAY_SECONDS = 5

# Target papers per year band
YEAR_BANDS = [
    {"start": "2019", "end": "2020", "target": 120, "label": "2019-2020-foundational"},
    {"start": "2021", "end": "2022", "target": 180, "label": "2021-2022-improvements"},
    {"start": "2023", "end": "2024", "target": 380, "label": "2023-2024-llm-rag"},
    {"start": "2025", "end": "2026", "target": 250, "label": "2025-2026-latest"},
]
# Total target: 750 new papers → ~1000+ with existing 274
# Total = 200 papers across all bands


def get_clean_id(entry_id: str) -> str:
    return make_paper_id(entry_id.split("/abs/")[-1].split("v")[0])


def safe_fetch(client, search, max_results, retries=3, wait=15):
    """Fetch with retry logic."""
    for attempt in range(retries):
        try:
            results = []
            for paper in client.results(search):
                results.append(paper)
                if len(results) >= max_results:
                    break
            return results
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                print(f"  Waiting {wait}s...")
                time.sleep(wait)
    return []


def fetch_band(
    client,
    band: dict,
    existing_ids: set,
    target: int
) -> list[dict]:
    """Fetch papers for a single year band."""

    start = band["start"]
    end = band["end"]
    label = band["label"]

    cat_query = " OR ".join([f"cat:{c}" for c in CATEGORIES])
    query = (
        f"({cat_query}) AND "
        f"submittedDate:[{start}01010000 TO {end}12312359]"
    )

    print(f"\n  Fetching {label} (target: {target} papers)...")
    print(f"  Query: {query}")

    search = arxiv.Search(
        query=query,
        max_results=target * 3,  # fetch 3x to account for filtering
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []
    raw_results = safe_fetch(client, search, max_results=target * 3)

    for paper in raw_results:
        clean_id = get_clean_id(paper.entry_id)

        # Skip if already in corpus
        if clean_id in existing_ids:
            continue

        # Verify year range
        year = paper.published.year
        if not (int(start) <= year <= int(end)):
            continue

        # Must have target category
        if not any(cat in paper.categories for cat in CATEGORIES):
            continue

        # Skip papers with very short abstracts (likely metadata issues)
        if len(paper.summary) < 100:
            continue

        paper_data = {
            "arxiv_id": clean_id,
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "published": str(paper.published),
            "updated": str(paper.updated),
            "categories": paper.categories,
            "primary_category": paper.primary_category,
            "pdf_url": paper.pdf_url,
            "abstract": paper.summary,
            "comment": paper.comment,
            "year_band": label
        }

        papers.append(paper_data)
        existing_ids.add(clean_id)

        if len(papers) >= target:
            break

    print(f"  Fetched: {len(papers)} papers")
    return papers


def fetch_diverse_corpus(
    output_dir: Path = OUTPUT_DIR,
    year_bands: list = YEAR_BANDS,
) -> list[dict]:

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = output_dir / "papers_metadata.json"

    # Load existing papers
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing_ids = {p["arxiv_id"] for p in existing}
        print(f"Existing corpus: {len(existing)} papers")
    else:
        existing = []
        existing_ids = set()

    client = arxiv.Client(
        page_size=100,
        delay_seconds=DELAY_SECONDS,
        num_retries=5
    )

    all_new_papers = []

    for band in year_bands:
        # Check how many we already have for this band
        already_have = sum(
            1 for p in existing
            if p.get("year_band") == band["label"]
        )
        still_need = band["target"] - already_have

        if still_need <= 0:
            print(f"\n  {band['label']}: already have {already_have} papers, skipping")
            continue

        print(f"\n  {band['label']}: have {already_have}, need {still_need} more")

        new_papers = fetch_band(
            client=client,
            band=band,
            existing_ids=existing_ids,
            target=still_need
        )

        all_new_papers.extend(new_papers)

        # Save incrementally after each band
        combined = existing + all_new_papers
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)

        print(f"  Saved {len(combined)} total papers so far")

        # Respect rate limits between bands
        time.sleep(10)

    # Final save
    final = existing + all_new_papers
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Corpus fetch complete")
    print(f"  Previous papers: {len(existing)}")
    print(f"  New papers:      {len(all_new_papers)}")
    print(f"  Total papers:    {len(final)}")
    print(f"\nYear band breakdown:")
    from collections import Counter
    band_counts = Counter(p.get("year_band", "unknown") for p in final)
    for band in year_bands:
        count = band_counts.get(band["label"], 0)
        print(f"  {band['label']}: {count} papers")

    return final


if __name__ == "__main__":
    papers = fetch_diverse_corpus()