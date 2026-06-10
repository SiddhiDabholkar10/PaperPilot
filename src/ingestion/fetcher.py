import arxiv
import json
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Config
CATEGORIES = ["cs.LG", "cs.CL"]
START_YEAR = 2019
MAX_PAPERS = 1000
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "data/raw"
DELAY_SECONDS = 5

def get_clean_id(entry_id: str) -> str:
    """Extract clean arxiv ID from entry_id URL."""
    # e.g. http://arxiv.org/abs/2605.22786v1 -> 2605.22786
    return entry_id.split("/abs/")[-1].split("v")[0]

def fetch_papers(
    categories: list[str] = CATEGORIES,
    max_papers: int = MAX_PAPERS,
    start_year: int = START_YEAR,
    output_dir: Path = OUTPUT_DIR
) -> list[dict]:
    
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = output_dir / "papers_metadata.json"

    # Load existing if resuming
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing_ids = {p["arxiv_id"] for p in existing}
        print(f"Resuming — {len(existing)} papers already fetched")
    else:
        existing = []
        existing_ids = set()

    # Build category query
    cat_query = " OR ".join([f"cat:{c}" for c in categories])
    query = f"({cat_query}) AND submittedDate:[{start_year}01010000 TO 99991231235959]"

    client = arxiv.Client(
        page_size=100,
        delay_seconds=DELAY_SECONDS,
        num_retries=5
    )

    search = arxiv.Search(
        query=query,
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    papers = list(existing)
    
    try:
        with tqdm(total=max_papers, desc="Fetching papers") as pbar:
            pbar.update(len(existing))
            
            for paper in client.results(search):
                clean_id = get_clean_id(paper.entry_id)
                
                # Skip if already fetched
                if clean_id in existing_ids:
                    continue

                # Filter by year
                if paper.published.year < start_year:
                    continue

                # Filter: must have cs.LG or cs.CL
                if not any(cat in paper.categories for cat in categories):
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
                }

                papers.append(paper_data)
                existing_ids.add(clean_id)
                pbar.update(1)

                # Save incrementally every 50 papers
                if len(papers) % 50 == 0:
                    with open(metadata_file, "w", encoding="utf-8") as f:
                        json.dump(papers, f, indent=2)

                if len(papers) >= max_papers:
                    break

    except KeyboardInterrupt:
        print("\nInterrupted — saving progress...")

    # Final save
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2)

    print(f"\nDone. Fetched {len(papers)} papers -> {metadata_file}")
    return papers


if __name__ == "__main__":
    papers = fetch_papers(max_papers=50)  # Start small — 50 papers to test
    
    # Preview
    print("\nSample paper:")
    print(json.dumps(papers[0], indent=2))
    print(f"\nYear distribution:")
    from collections import Counter
    years = Counter(p["published"][:4] for p in papers)
    for year, count in sorted(years.items()):
        print(f"  {year}: {count} papers")