# check_internal_refs.py
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "data/raw/paper_references.json", encoding="utf-8") as f:
    refs = json.load(f)

with open(BASE_DIR / "data/raw/papers_metadata.json", encoding="utf-8") as f:
    papers = json.load(f)

corpus_ids = {p["arxiv_id"] for p in papers}

internal_links = 0
total_links = 0
papers_with_internal = 0

for arxiv_id, data in refs.items():
    paper_refs = data.get("references", [])
    total_links += len(paper_refs)
    internal = [r for r in paper_refs if r["arxiv_id"] in corpus_ids]
    internal_links += len(internal)
    if internal:
        papers_with_internal += 1

print(f"Total reference links: {total_links}")
print(f"Internal links (both papers in corpus): {internal_links}")
print(f"Papers with at least one internal reference: {papers_with_internal}")
print(f"Internal link ratio: {internal_links/max(1,total_links):.2%}")