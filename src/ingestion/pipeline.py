import json
import time
from pathlib import Path
from tqdm import tqdm

from grobid_extractor import extract_all_with_grobid, check_grobid_alive
from grobid_chunker import chunk_all as grobid_chunk_all
from pdf_downloader import download_pdfs
from pymupdf_fallback import run_pymupdf_fallback

BASE_DIR = Path(__file__).resolve().parent.parent.parent
METADATA_FILE = BASE_DIR / "data/raw/papers_metadata.json"
PDF_DIR = BASE_DIR / "data/raw/pdfs"
GROBID_DIR = BASE_DIR / "data/processed/grobid"
CHUNKS_DIR = BASE_DIR / "data/chunks"

def run_full_pipeline(
    max_papers: int = None,
    skip_download: bool = False
):
    """Run the complete ingestion pipeline with fallback."""

    print("=" * 60)
    print("PaperPilot Ingestion Pipeline")
    print("=" * 60)

    # --- Load metadata ---
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        all_metadata = json.load(f)

    if max_papers:
        all_metadata = all_metadata[:max_papers]

    metadata_map = {p["arxiv_id"]: p for p in all_metadata}
    target_ids = {
    p["arxiv_id"] for p in all_metadata
    if (PDF_DIR / f"{p['arxiv_id']}.pdf").exists()
}
    print(f"  PDFs available on disk: {len(target_ids)}")
    print(f"\nTarget papers: {len(target_ids)}")

    # --- Step 1: Download PDFs ---
    if not skip_download:
        print("\n[1/4] Downloading PDFs...")
        download_pdfs(
            metadata_file=METADATA_FILE,
            pdf_dir=PDF_DIR,
            max_papers=max_papers
        )
    else:
        print("\n[1/4] Skipping download (skip_download=True)")

    # --- Step 2: Grobid extraction ---
    print("\n[2/4] Grobid extraction...")

    if not check_grobid_alive():
        print("WARNING: Grobid not running — all papers will use PyMuPDF fallback")
        grobid_results = []
        successful_grobid_ids = set()
        failed_ids = list(target_ids)
    else:
        grobid_results = extract_all_with_grobid(
            pdf_dir=PDF_DIR,
            processed_dir=GROBID_DIR,
            max_papers=max_papers,
            delay=1.0
        )

        successful_grobid_ids = {
            r["arxiv_id"] for r in grobid_results
            if r.get("extraction_status") == "success"
            and r["arxiv_id"] in target_ids
        }

        failed_ids = [
            arxiv_id for arxiv_id in target_ids
            if arxiv_id not in successful_grobid_ids
        ]

    print(f"  Grobid success: {len(successful_grobid_ids)}")
    print(f"  Needs fallback: {len(failed_ids)}")

    # --- Step 3: Grobid chunking ---
    print("\n[3/4] Chunking Grobid extractions...")

    if successful_grobid_ids:
        grobid_chunks = grobid_chunk_all(
            grobid_dir=GROBID_DIR,
            chunks_dir=CHUNKS_DIR,
            metadata_file=METADATA_FILE,
            max_papers=max_papers
        )
    else:
        print("  No Grobid extractions to chunk")
        grobid_chunks = {
            "abstract_chunks": [],
            "parent_chunks": [],
            "child_chunks": []
        }

    # --- Step 4: PyMuPDF fallback ---
    fallback_abstract = []
    fallback_parent = []
    fallback_child = []

    if failed_ids:
        print(f"\n[4/4] PyMuPDF fallback for {len(failed_ids)} papers...")
        fallback_abstract, fallback_parent, fallback_child = run_pymupdf_fallback(
            failed_ids=failed_ids,
            metadata_map=metadata_map,
            pdf_dir=PDF_DIR
        )
    else:
        print("\n[4/4] No fallback needed — all papers processed by Grobid")

    # --- Merge all chunks ---
    print("\nMerging chunks...")

    final_abstract = grobid_chunks["abstract_chunks"] + fallback_abstract
    final_parent = grobid_chunks["parent_chunks"] + fallback_parent
    final_child = grobid_chunks["child_chunks"] + fallback_child

    # Deduplicate by chunk_id (safety net)
    def dedup_chunks(chunks: list) -> list:
        seen = set()
        deduped = []
        for chunk in chunks:
            if chunk["chunk_id"] not in seen:
                seen.add(chunk["chunk_id"])
                deduped.append(chunk)
        return deduped

    final_abstract = dedup_chunks(final_abstract)
    final_parent = dedup_chunks(final_parent)
    final_child = dedup_chunks(final_child)

    # --- Save merged chunks ---
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    for name, chunks in [
        ("abstract", final_abstract),
        ("parent", final_parent),
        ("child", final_child)
    ]:
        out_file = CHUNKS_DIR / f"chunks_{name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(chunks)} {name} chunks → {out_file}")

    # --- Extraction method breakdown ---
    grobid_child_count = sum(
        1 for c in final_child
        if c.get("extraction_method") == "grobid"
    )
    fallback_child_count = sum(
        1 for c in final_child
        if c.get("extraction_method") == "pymupdf_fallback"
    )

    # --- Final summary ---
    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(f"  Papers processed:    {len(target_ids)}")
    print(f"  Grobid papers:       {len(successful_grobid_ids)}")
    print(f"  Fallback papers:     {len(failed_ids)}")
    print(f"  Abstract chunks:     {len(final_abstract)}")
    print(f"  Parent chunks:       {len(final_parent)}")
    print(f"  Child chunks:        {len(final_child)}")
    print(f"    ↳ Grobid:          {grobid_child_count}")
    print(f"    ↳ PyMuPDF fallback:{fallback_child_count}")
    print(f"  Total chunks:        {len(final_abstract) + len(final_parent) + len(final_child)}")
    print("=" * 60)
    

    return {
        "abstract_chunks": final_abstract,
        "parent_chunks": final_parent,
        "child_chunks": final_child,
        "stats": {
            "total_papers": len(target_ids),
            "grobid_papers": len(successful_grobid_ids),
            "fallback_papers": len(failed_ids),
            "abstract_chunks": len(final_abstract),
            "parent_chunks": len(final_parent),
            "child_chunks": len(final_child),
            "grobid_child_chunks": grobid_child_count,
            "fallback_child_chunks": fallback_child_count
        }
    }


if __name__ == "__main__":
      result = run_full_pipeline(
        max_papers=None,      # process all 50
        skip_download=True    # PDFs already downloaded
    )