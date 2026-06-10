# notebooks/01_arxiv_exploration.py
import arxiv
import json

client = arxiv.Client(
    page_size=10,
    delay_seconds=5,
    num_retries=5
)


import time

def safe_search(client, search, retries=3, wait=10):
    for attempt in range(retries):
        try:
            results = list(client.results(search))
            return results
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                print(f"Waiting {wait}s before retry...")
                time.sleep(wait)
    return []


# Test 1: Basic search with category filter in query
print("=== TEST 1: Basic search ===")
search = arxiv.Search(
    query="cat:cs.LG AND attention mechanism transformer",
    max_results=5,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

for paper in safe_search(client, search):
    print(f"ID: {paper.entry_id}")
    print(f"Title: {paper.title}")
    print(f"Authors: {[a.name for a in paper.authors[:3]]}")
    print(f"Published: {paper.published}")
    print(f"Categories: {paper.categories}")
    print(f"PDF URL: {paper.pdf_url}")
    print(f"Abstract (first 200 chars): {paper.summary[:200]}")
    print("---")

# Test 2: Fetch a specific paper by ID
print("\n=== TEST 2: Fetch specific paper ===")
search2 = arxiv.Search(id_list=["1706.03762"])  # Attention is All You Need
for paper in safe_search(client, search2):
    print(f"Title: {paper.title}")
    print(f"Published: {paper.published}")
    print(f"Categories: {paper.categories}")

# Test 3: Check what fields are available
print("\n=== TEST 3: Available fields on a paper object ===")
search3 = arxiv.Search(
    query="(cat:cs.CL OR cat:cs.LG) AND flash attention",
    max_results=1
)
for paper in safe_search(client, search):
    print(json.dumps({
        "entry_id": paper.entry_id,
        "title": paper.title,
        "published": str(paper.published),
        "updated": str(paper.updated),
        "categories": paper.categories,
        "primary_category": paper.primary_category,
        "pdf_url": paper.pdf_url,
        "doi": paper.doi,
        "comment": paper.comment,
        "journal_ref": paper.journal_ref,
    }, indent=2))