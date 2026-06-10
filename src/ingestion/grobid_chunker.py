import json
import re
from pathlib import Path
from tqdm import tqdm
from ids import make_paper_id, make_abstract_id, make_parent_id, make_child_id

BASE_DIR = Path(__file__).resolve().parent.parent.parent
GROBID_DIR = BASE_DIR / "data/processed/grobid"
CHUNKS_DIR = BASE_DIR / "data/chunks"
METADATA_FILE = BASE_DIR / "data/raw/papers_metadata.json"

# Parent chunk — larger, given to synthesizer for context
PARENT_CHUNK_SIZE = 1500  # chars
# Child chunk — smaller, used for retrieval matching
CHILD_CHUNK_SIZE = 400    # chars
CHUNK_OVERLAP = 50        # chars


def clean_section_text(text: str) -> str:
    """Clean text from a Grobid-extracted section."""
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove leftover XML artifacts
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()


def make_chunks_from_text(
    text: str,
    chunk_size: int,
    overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into chunks of approximately chunk_size chars with overlap."""
    if not text.strip():
        return []

    # Split by paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current = []
    current_size = 0

    for para in paragraphs:
        if current_size + len(para) > chunk_size and current:
            chunks.append('\n\n'.join(current))
            # Overlap: keep last paragraph
            last = current[-1]
            current = [last, para] if len(last) <= overlap * 3 else [para]
            current_size = sum(len(p) for p in current)
        else:
            current.append(para)
            current_size += len(para)

    if current:
        chunks.append('\n\n'.join(current))

    return [c for c in chunks if len(c) > 80]


def chunk_paper(paper: dict, metadata: dict) -> dict:
    """
    Convert a Grobid-extracted paper into parent + child chunks.
    
    Structure:
    - 1 abstract chunk (special, always retrieved with parent)
    - N parent chunks per section (large, for synthesis context)
    - M child chunks per parent (small, for retrieval matching)
    """

    arxiv_id = paper["arxiv_id"]
    base_meta = {
        "arxiv_id": arxiv_id,
        "title": metadata.get("title", paper.get("title", "")),
        "authors": metadata.get("authors", paper.get("authors", [])),
        "published": metadata.get("published", ""),
        "year": metadata.get("published", "")[:4],
        "categories": metadata.get("categories", []),
        "primary_category": metadata.get("primary_category", ""),
        "extraction_method": paper.get("extraction_method", "grobid"),
    }

    abstract_chunks = []
    parent_chunks = []
    child_chunks = []

    # --- Abstract chunk (special) ---
    abstract_text = clean_section_text(paper.get("abstract", ""))
    if abstract_text:
        abstract_chunk = {
            **base_meta,
            "chunk_id": make_abstract_id(make_paper_id(arxiv_id)),
            "chunk_type": "abstract",
            "section": "ABSTRACT",
            "section_heading": "Abstract",
            "text": abstract_text,
            "parent_chunk_id": None,
            "char_count": len(abstract_text)
        }
        abstract_chunks.append(abstract_chunk)

    # --- Section chunks ---
    parent_idx = 0
    child_idx = 0

    for section in paper.get("sections", []):
        heading = section.get("heading", "")
        section_num = section.get("section_num", "")
        
        # Skip empty or noise sections
        section_text = clean_section_text(section.get("text", ""))
        if not section_text or len(section_text) < 100:
            continue

        # Skip reference/acknowledgment sections
        heading_lower = heading.lower()
        if any(skip in heading_lower for skip in [
            "acknowledg", "appendix", "funding", "declaration"
        ]):
            continue

        # Normalize section name
        section_label = heading.upper() if heading else f"SECTION_{section_num}"

        # --- Parent chunks (large, for synthesis) ---
        parent_texts = make_chunks_from_text(section_text, PARENT_CHUNK_SIZE)

        for parent_text in parent_texts:
            parent_chunk_id = make_parent_id(make_paper_id(arxiv_id), parent_idx)

            parent_chunk = {
                **base_meta,
                "chunk_id": parent_chunk_id,
                "chunk_type": "parent",
                "section": section_label,
                "section_heading": heading,
                "section_num": section_num,
                "text": parent_text,
                "parent_chunk_id": None,
                "char_count": len(parent_text)
            }
            parent_chunks.append(parent_chunk)

            # --- Child chunks (small, for retrieval) ---
            child_texts = make_chunks_from_text(parent_text, CHILD_CHUNK_SIZE)

            for child_text in child_texts:
                child_chunk = {
                    **base_meta,
                    "chunk_id": make_child_id(parent_chunk_id, child_idx),
                    "chunk_type": "child",
                    "section": section_label,
                    "section_heading": heading,
                    "section_num": section_num,
                    "text": child_text,
                    "parent_chunk_id": parent_chunk_id,
                    "char_count": len(child_text)
                }
                child_chunks.append(child_chunk)
                child_idx += 1

            parent_idx += 1

    return {
        "arxiv_id": arxiv_id,
        "abstract_chunks": abstract_chunks,
        "parent_chunks": parent_chunks,
        "child_chunks": child_chunks,
        "stats": {
            "abstract_chunks": len(abstract_chunks),
            "parent_chunks": len(parent_chunks),
            "child_chunks": len(child_chunks),
            "sections_processed": len(paper.get("sections", [])),
            "references": len(paper.get("references", []))
        }
    }


def chunk_all(
    grobid_dir: Path = GROBID_DIR,
    chunks_dir: Path = CHUNKS_DIR,
    metadata_file: Path = METADATA_FILE,
    max_papers: int = None
) -> dict:

    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(metadata_file, "r", encoding="utf-8") as f:
        all_metadata = json.load(f)
    metadata_map = {p["arxiv_id"]: p for p in all_metadata}

    grobid_files = sorted(grobid_dir.glob("*.json"))
    if max_papers:
        grobid_files = grobid_files[:max_papers]

    all_abstract_chunks = []
    all_parent_chunks = []
    all_child_chunks = []
    paper_stats = []

    for grobid_file in tqdm(grobid_files, desc="Chunking papers"):
        with open(grobid_file, "r", encoding="utf-8") as f:
            paper = json.load(f)

        if paper.get("extraction_status") != "success":
            print(f"Skipping {paper['arxiv_id']} — {paper.get('extraction_status')}")
            continue

        metadata = metadata_map.get(paper["arxiv_id"], {})
        result = chunk_paper(paper, metadata)

        all_abstract_chunks.extend(result["abstract_chunks"])
        all_parent_chunks.extend(result["parent_chunks"])
        all_child_chunks.extend(result["child_chunks"])
        paper_stats.append({
            "arxiv_id": result["arxiv_id"],
            "title": metadata.get("title", "")[:60],
            **result["stats"]
        })

    # Save all chunk types separately
    for chunk_type, chunks in [
        ("abstract", all_abstract_chunks),
        ("parent", all_parent_chunks),
        ("child", all_child_chunks)
    ]:
        out_file = chunks_dir / f"chunks_{chunk_type}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

    # Print stats
    print(f"\nChunking complete:")
    print(f"  Papers:          {len(paper_stats)}")
    print(f"  Abstract chunks: {len(all_abstract_chunks)}")
    print(f"  Parent chunks:   {len(all_parent_chunks)}")
    print(f"  Child chunks:    {len(all_child_chunks)}")
    print(f"\nPer-paper breakdown:")
    for s in paper_stats:
        print(f"  {s['arxiv_id']} | "
              f"{s['parent_chunks']} parents | "
              f"{s['child_chunks']} children | "
              f"{s['references']} refs | "
              f"{s['title'][:50]}")

    # Show sample child chunk
    if all_child_chunks:
        sample = all_child_chunks[10]
        print(f"\n--- Sample child chunk ({sample['chunk_id']}) ---")
        print(f"Section: {sample['section_heading']}")
        print(f"Parent:  {sample['parent_chunk_id']}")
        print(f"Text: {sample['text'][:300]}")

    return {
        "abstract_chunks": all_abstract_chunks,
        "parent_chunks": all_parent_chunks,
        "child_chunks": all_child_chunks
    }


if __name__ == "__main__":
    result = chunk_all(max_papers=3)