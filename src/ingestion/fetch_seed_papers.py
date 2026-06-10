import arxiv
import json
import time
from pathlib import Path
from ids import make_paper_id

BASE_DIR = Path(__file__).resolve().parent.parent.parent
METADATA_FILE = BASE_DIR / "data/raw/papers_metadata.json"

SEED_PAPERS = [
    {"id": "1706.03762", "label": "Attention is All You Need"},
    {"id": "1810.04805", "label": "BERT"},
    {"id": "1910.10683", "label": "T5"},
    {"id": "1910.13461", "label": "BART"},
    {"id": "1907.11692", "label": "RoBERTa"},
    {"id": "1901.02860", "label": "Transformer-XL"},
    {"id": "2005.14165", "label": "GPT-3"},
    {"id": "2005.11401", "label": "RAG"},
    {"id": "2005.00796", "label": "Dense Passage Retrieval"},
    {"id": "2112.09332", "label": "InstructGPT"},
    {"id": "1904.10509", "label": "Sparse Transformers"},
    {"id": "1809.09600", "label": "HotpotQA"},
    {"id": "1704.00051", "label": "DrQA"},
    {"id": "1906.00300", "label": "Natural Questions"},
    {"id": "1606.05250", "label": "SQuAD"},
    {"id": "1806.03822", "label": "SQuAD 2.0"},
    {"id": "1901.04085", "label": "Passage Re-ranking with BERT"},
    {"id": "1906.02916", "label": "Multi-hop QA through Question Decomposition"},
    {"id": "1409.0473",  "label": "Neural Machine Translation Attention"},
    {"id": "1409.3215",  "label": "Sequence to Sequence Learning"},
    {"id": "1508.04025", "label": "Effective Approaches to Attention-based NMT"},
    {"id": "1412.6980",  "label": "Adam Optimizer"},
    {"id": "1502.03167", "label": "Batch Normalization"},
    {"id": "1512.03385", "label": "ResNet"},
]


def fetch_seed_papers():
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        existing = json.load(f)
    existing_ids = {p["arxiv_id"] for p in existing}

    print(f"Existing corpus: {len(existing)} papers")
    print(f"Seed papers to check: {len(SEED_PAPERS)}")
    print(f"Sample existing IDs: {list(existing_ids)[:3]}")

    client = arxiv.Client(
        page_size=10,
        delay_seconds=3,
        num_retries=5
    )

    new_papers = []

    for seed in SEED_PAPERS:
        clean_id = make_paper_id(seed["id"])

        if clean_id in existing_ids:
            print(f"  Already in corpus: {seed['label']} ({clean_id})")
            continue

        print(f"  Fetching: {seed['label']} ({clean_id})...")

        try:
            search = arxiv.Search(id_list=[seed["id"]])
            results = list(client.results(search))

            if not results:
                print(f"    Not found on arXiv")
                continue

            paper = results[0]
            clean_id = make_paper_id(paper.entry_id.split("/abs/")[-1])

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
                "year_band": "foundational",
                "is_seed": True
            }

            new_papers.append(paper_data)
            existing_ids.add(clean_id)
            print(f"    Added: {paper.title[:60]}")
            time.sleep(3)

        except Exception as e:
            print(f"    Failed: {e}")

    if new_papers:
        combined = existing + new_papers
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        print(f"\nAdded {len(new_papers)} seed papers")
        print(f"Total corpus: {len(combined)} papers")
    else:
        print("\nNo new papers added — all already in corpus or failed")

    return new_papers


if __name__ == "__main__":
    papers = fetch_seed_papers()