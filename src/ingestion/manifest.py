import json
import hashlib
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHUNKS_DIR = BASE_DIR / "data/chunks"
METADATA_FILE = BASE_DIR / "data/raw/papers_metadata.json"
MANIFEST_FILE = BASE_DIR / "data/manifest.json"


def generate_manifest(
    embedding_model: str = "BAAI/bge-large-en-v1.5",
    notes: str = ""
) -> dict:

    # Load metadata
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        papers = json.load(f)

    # Load chunks
    chunk_counts = {}
    chunk_files = {}
    for chunk_type in ["abstract", "parent", "child"]:
        path = CHUNKS_DIR / f"chunks_{chunk_type}.json"
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        chunk_counts[chunk_type] = len(chunks)
        chunk_files[chunk_type] = str(path)

    # Year band distribution
    from collections import Counter
    band_dist = Counter(p.get("year_band", "unknown") for p in papers)
    year_dist = Counter(p["published"][:4] for p in papers)

    # Extraction method breakdown
    child_path = CHUNKS_DIR / "chunks_child.json"
    with open(child_path, "r", encoding="utf-8") as f:
        children = json.load(f)
    method_dist = Counter(c.get("extraction_method", "unknown") for c in children)

    # Generate run ID from timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "pipeline_version": "1.0.0",

        "corpus": {
            "total_papers": len(papers),
            "year_band_distribution": dict(band_dist),
            "year_distribution": dict(sorted(year_dist.items())),
            "categories": ["cs.LG", "cs.CL"],
            "date_range": "2020-2026",
        },

        "extraction": {
            "primary_method": "grobid_0.8.0",
            "fallback_method": "pymupdf_1.24.0",
            "grobid_papers": method_dist.get("grobid", 0) // max(1, chunk_counts["child"] // len(papers)),
            "fallback_papers": 1,
        },

        "chunks": {
            "abstract_chunks": chunk_counts["abstract"],
            "parent_chunks": chunk_counts["parent"],
            "child_chunks": chunk_counts["child"],
            "total_chunks": sum(chunk_counts.values()),
            "extraction_method_breakdown": dict(method_dist),
            "chunk_files": chunk_files,
        },

        "embedding": {
            "model": embedding_model,
            "dimensions": 1024,
            "index_type": "pinecone_serverless",
            "status": "pending",
        },

        "validation": {
            "critical_issues": 0,
            "warnings": 25,
            "warning_type": "symbol_heavy_chunks",
            "decision": "accepted — table/equation blocks, negligible impact on retrieval"
        },

       "enrichment": {
            "source": "openalex",
            "status": "pending",
            "enriched_papers": 0,
            "provides": ["citation_count", "reference_links", "concepts"]
        },

        "notes": notes
    }

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Manifest saved → {MANIFEST_FILE}")
    print(f"\nRun ID: {run_id}")
    print(f"Papers: {manifest['corpus']['total_papers']}")
    print(f"Total chunks: {manifest['chunks']['total_chunks']}")
    print(f"Embedding model: {manifest['embedding']['model']}")
    print(f"\nYear band distribution:")
    for band, count in band_dist.items():
        print(f"  {band}: {count}")

    return manifest


if __name__ == "__main__":
    manifest = generate_manifest(
        embedding_model="BAAI/bge-large-en-v1.5",
        notes="Initial corpus — 250 papers, pre-embedding state"
    )