import json
import time
import requests
from pathlib import Path
from tqdm import tqdm
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")


METADATA_FILE = BASE_DIR / "data/raw/papers_metadata.json"
ENRICHED_FILE = BASE_DIR / "data/raw/papers_enriched.json"
REFERENCES_FILE = BASE_DIR / "data/raw/paper_references.json"

S2_BASE = "https://api.semanticscholar.org/graph/v1/paper"
S2_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")


# Try richer field sets first, then gracefully fall back if S2 rejects a field.
FIELDS = (
    "citationCount,"
    "influentialCitationCount,"
    "references.externalIds,"
    "references.title,"
    "references.year,"
    "references.citationCount,"
    "tldr"
)
DELAY = 1.0


def get_s2_data(arxiv_id: str, retries: int = 3) -> dict:
    """Fetch citation data from Semantic Scholar."""
    url = f"{S2_BASE}/arXiv:{arxiv_id}"
    params = {"fields": FIELDS}
    headers = {}

    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY

    for attempt in range(retries):
        try:
            resp = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=15
            )

            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 404:
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

    # Resume logic
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
    stats = {
        "found": 0,
        "not_found": 0,
        "failed": 0,
        "total_references": 0,
        "total_influential": 0
    }

    for paper in tqdm(papers, desc="Enriching with Semantic Scholar"):
        arxiv_id = paper["arxiv_id"]

        # Skip if already enriched with S2
        if arxiv_id in enriched_map:
            existing = enriched_map[arxiv_id]
            if existing.get("s2_status") == "found":
                enriched_papers.append(existing)
                stats["found"] += 1
                continue

        data = get_s2_data(arxiv_id)

        if "error" in data:
            enriched = {
                **paper,
                "citation_count": 0,
                "influential_citation_count": 0,
                "s2_status": data["error"],
                "s2_tldr": "",
                "arxiv_reference_count": 0
            }
            if data["error"] == "not_found":
                stats["not_found"] += 1
            else:
                stats["failed"] += 1
        else:
            citation_count = data.get("citationCount", 0) or 0
            influential = data.get("influentialCitationCount", 0) or 0

            # TLDR
            tldr = ""
            if data.get("tldr"):
                tldr = data["tldr"].get("text", "")

            # Topics
            topics = [t["topic"]["name"] for t in data.get("topics", [])[:5]]

            # References with arxiv IDs
            raw_refs = data.get("references", []) or []
            arxiv_refs = []
            for ref in raw_refs:
                ext_ids = ref.get("externalIds") or {}
                ref_arxiv = ext_ids.get("ArXiv", "")
                if ref_arxiv:
                    arxiv_refs.append({
                        "arxiv_id": ref_arxiv.split("v")[0],
                        "title": ref.get("title", ""),
                        "year": ref.get("year", ""),
                        "citation_count": ref.get("citationCount", 0)
                    })

            enriched = {
                **paper,
                "citation_count": citation_count,
                "influential_citation_count": influential,
                "s2_status": "found",
                "s2_tldr": tldr,
                "s2_topics": topics,
                "arxiv_reference_count": len(arxiv_refs)
            }

            all_references[arxiv_id] = {
                "arxiv_id": arxiv_id,
                "title": paper["title"],
                "citation_count": citation_count,
                "influential_citation_count": influential,
                "references": arxiv_refs
            }

            stats["found"] += 1
            stats["total_references"] += len(arxiv_refs)
            stats["total_influential"] += influential

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
    print(f"  Found on S2:        {stats['found']}")
    print(f"  Not found:          {stats['not_found']}")
    print(f"  Failed:             {stats['failed']}")
    print(f"  Total references:   {stats['total_references']}")
    print(f"  Total influential:  {stats['total_influential']}")

    counts = sorted(
        [p.get("citation_count", 0) for p in enriched_papers],
        reverse=True
    )
    print(f"\nCitation count distribution:")
    print(f"  Top 5:   {counts[:5]}")
    print(f"  Median:  {counts[len(counts)//2]}")
    print(f"  Zero:    {sum(1 for c in counts if c == 0)}")

    print(f"\nTop 5 most cited papers:")
    top = sorted(
        enriched_papers,
        key=lambda x: x.get("citation_count", 0),
        reverse=True
    )[:5]
    for p in top:
        print(f"  [{p.get('citation_count', 0):,}] "
              f"[influential: {p.get('influential_citation_count', 0):,}] "
              f"{p['title'][:55]}")

    return enriched_papers, all_references


if __name__ == "__main__":
    # Test with known older papers
    import json
    from pathlib import Path
    
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    with open(BASE_DIR / "data/raw/papers_metadata.json", encoding="utf-8") as f:
        all_papers = json.load(f)
    
    # Get 5 seed papers
    seeds = [p for p in all_papers if p.get("is_seed")][:5]
    print(f"Testing with {len(seeds)} seed papers:")
    for p in seeds:
        print(f"  {p['arxiv_id']}: {p['title'][:50]}")
    
    enriched, refs = enrich_corpus(max_papers=None)
