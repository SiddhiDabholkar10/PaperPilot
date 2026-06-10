import json
import time
import requests
from pathlib import Path
from tqdm import tqdm
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent.parent
METADATA_FILE = BASE_DIR / "data/raw/papers_metadata.json"
ENRICHED_FILE = BASE_DIR / "data/raw/papers_enriched.json"
REFERENCES_FILE = BASE_DIR / "data/raw/paper_references.json"

OPENALEX_BASE = "https://api.openalex.org/works"
# Polite pool — add email for faster rate limits
HEADERS = {
    "User-Agent": "PaperPilot/1.0 (mailto:dabholkarsiddhi10@gmail.com)"
}
DELAY = 0.15  # ~7 req/sec, well under 10/sec limit


def get_openalex_data(arxiv_id: str, title: str, retries: int = 3) -> dict:
    """Fetch citation count from OpenAlex by searching title."""
    
    # Clean title for search
    clean_title = title.strip()
    if not clean_title:
        return {"error": "no_title"}

    for attempt in range(retries):
        try:
            resp = requests.get(
                OPENALEX_BASE,
                params={
                    "search": clean_title,
                    "select": "id,title,cited_by_count,referenced_works,concepts,publication_year",
                    "per-page": 3
                },
                headers=HEADERS,
                timeout=15
            )

            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])

                # Find best match by title similarity
                for result in results:
                    result_title = (result.get("title") or "").lower().strip()
                    query_title = clean_title.lower().strip()

                    # Check if titles are close enough
                    query_words = set(query_title.split())
                    result_words = set(result_title.split())
                    if not query_words:
                        continue

                    overlap = len(query_words & result_words) / len(query_words)
                    if overlap > 0.7:
                        return result

                return {"error": "not_found"}

            elif resp.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"\n  Rate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                return {"error": f"http_{resp.status_code}"}

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5)
            else:
                return {"error": str(e)}

    return {"error": "max_retries"}


def get_referenced_arxiv_ids(referenced_works: list) -> list:
    """
    Fetch arxiv IDs for referenced works.
    OpenAlex returns work IDs like W2741809807 — we need to 
    batch fetch to get arxiv IDs.
    """
    if not referenced_works:
        return []

    # Batch fetch up to 50 referenced works at a time
    arxiv_refs = []
    batch_size = 50

    for i in range(0, min(len(referenced_works), 200), batch_size):
        batch = referenced_works[i:i + batch_size]
        ids_filter = "|".join(batch)

        try:
            resp = requests.get(
                OPENALEX_BASE,
                params={
                    "filter": f"openalex_id:{ids_filter}",
                    "select": "id,title,publication_year,ids",
                    "per-page": batch_size
                },
                headers=HEADERS,
                timeout=15
            )

            if resp.status_code == 200:
                data = resp.json()
                for work in data.get("results", []):
                    ids = work.get("ids", {})
                    arxiv_url = ids.get("arxiv", "")
                    if arxiv_url:
                        # Extract clean arxiv ID from URL
                        arxiv_id = arxiv_url.replace(
                            "https://arxiv.org/abs/", ""
                        ).split("v")[0]
                        arxiv_refs.append({
                            "arxiv_id": arxiv_id,
                            "title": work.get("title", ""),
                            "year": work.get("publication_year", "")
                        })

            time.sleep(DELAY)

        except Exception as e:
            print(f"\n  Batch fetch failed: {e}")
            continue

    return arxiv_refs


def enrich_corpus(
    metadata_file: Path = METADATA_FILE,
    enriched_file: Path = ENRICHED_FILE,
    references_file: Path = REFERENCES_FILE,
    max_papers: int = None
) -> tuple[list, dict]:

    with open(metadata_file, "r", encoding="utf-8") as f:
        papers = json.load(f)

    if max_papers:
        papers = papers[:max_papers]

    # Resume if partially done
    if enriched_file.exists():
        with open(enriched_file, "r", encoding="utf-8") as f:
            enriched_list = json.load(f)
        enriched_map = {p["arxiv_id"]: p for p in enriched_list}
        print(f"Resuming — {len(enriched_map)} papers already enriched")
    else:
        enriched_map = {}

    if references_file.exists():
        with open(references_file, "r", encoding="utf-8") as f:
            all_references = json.load(f)
    else:
        all_references = {}

    enriched_papers = []
    stats = {"found": 0, "not_found": 0, "failed": 0, "total_refs": 0}

    for paper in tqdm(papers, desc="Enriching via OpenAlex"):
        arxiv_id = paper["arxiv_id"]

        # Skip if already enriched
        if arxiv_id in enriched_map:
            enriched_papers.append(enriched_map[arxiv_id])
            continue

        data = get_openalex_data(arxiv_id, paper.get("title", ""))

        if "error" in data:
            enriched = {
                **paper,
                "citation_count": 0,
                "influential_citation_count": 0,
                "openalex_status": data["error"],
                "arxiv_reference_count": 0
            }
            if data["error"] == "not_found":
                stats["not_found"] += 1
            else:
                stats["failed"] += 1
        else:
            citation_count = data.get("cited_by_count", 0) or 0

            # Get concepts (topic tags from OpenAlex)
            concepts = [
                c["display_name"]
                for c in data.get("concepts", [])[:5]
                if c.get("score", 0) > 0.3
            ]

            # Fetch referenced works with arxiv IDs
            referenced_works = data.get("referenced_works", [])
            arxiv_refs = get_referenced_arxiv_ids(referenced_works)

            enriched = {
                **paper,
                "citation_count": citation_count,
                "influential_citation_count": 0,  # OpenAlex doesn't have this
                "openalex_status": "found",
                "openalex_id": data.get("id", ""),
                "openalex_concepts": concepts,
                "arxiv_reference_count": len(arxiv_refs)
            }

            all_references[arxiv_id] = {
                "arxiv_id": arxiv_id,
                "title": paper["title"],
                "citation_count": citation_count,
                "references": arxiv_refs
            }

            stats["found"] += 1
            stats["total_refs"] += len(arxiv_refs)

        enriched_papers.append(enriched)
        enriched_map[arxiv_id] = enriched

        # Save every 20 papers
        if len(enriched_papers) % 20 == 0:
            with open(enriched_file, "w", encoding="utf-8") as f:
                json.dump(enriched_papers, f, ensure_ascii=False, indent=2)
            with open(references_file, "w", encoding="utf-8") as f:
                json.dump(all_references, f, ensure_ascii=False, indent=2)

        time.sleep(DELAY)

    # Final save
    with open(enriched_file, "w", encoding="utf-8") as f:
        json.dump(enriched_papers, f, ensure_ascii=False, indent=2)
    with open(references_file, "w", encoding="utf-8") as f:
        json.dump(all_references, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"\nEnrichment complete:")
    print(f"  Found on OpenAlex:  {stats['found']}")
    print(f"  Not found:          {stats['not_found']}")
    print(f"  Failed:             {stats['failed']}")
    print(f"  Total references:   {stats['total_refs']}")

    # Citation distribution
    counts = sorted(
        [p.get("citation_count", 0) for p in enriched_papers],
        reverse=True
    )
    print(f"\nCitation count distribution:")
    print(f"  Top 5:   {counts[:5]}")
    print(f"  Median:  {counts[len(counts)//2]}")
    print(f"  Zero:    {sum(1 for c in counts if c == 0)}")

    # Top cited papers
    print(f"\nTop 5 most cited papers:")
    top = sorted(enriched_papers, key=lambda x: x.get("citation_count", 0), reverse=True)[:5]
    for p in top:
        print(f"  [{p.get('citation_count', 0)}] {p['title'][:60]}")

    return enriched_papers, all_references


if __name__ == "__main__":
    enriched, refs = enrich_corpus()