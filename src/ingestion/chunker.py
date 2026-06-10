import json
import re
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data/processed"
CHUNKS_DIR = BASE_DIR / "data/chunks"
CHUNK_SIZE = 512        # tokens (approx — we'll use chars * 0.75)
CHUNK_OVERLAP = 50      # tokens overlap between chunks
CHARS_PER_TOKEN = 4     # rough approximation

def clean_text(text: str) -> str:
    """Clean extracted PDF text."""
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove hyphenation at line breaks
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove lines that are just numbers (page numbers)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # Remove lines that are just special characters/math noise
    text = re.sub(r'^[^a-zA-Z0-9\s]{3,}$', '', text, flags=re.MULTILINE)
    # Remove references section (not useful for retrieval)
    text = re.sub(r'\n(References|Bibliography)\n.*$', '', text, flags=re.DOTALL)
    return text.strip()

def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks targeting chunk_size tokens."""

    # Use 3 chars per token — more accurate for ML papers with math/symbols
    chunk_size_chars = chunk_size * 3
    overlap_chars = chunk_overlap * 3

    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    # Filter out very short paragraphs (headers, lone numbers, etc.)
    paragraphs = [p for p in paragraphs if len(p) > 50]

    chunks = []
    current_parts = []
    current_size = 0

    for para in paragraphs:
        para_size = len(para)

        # If paragraph alone exceeds chunk size, split by sentences
        if para_size > chunk_size_chars:
            # Flush current chunk first
            if current_parts:
                chunks.append('\n\n'.join(current_parts))
                current_parts = []
                current_size = 0

            # Split oversized paragraph by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sent_buffer = []
            sent_size = 0

            for sent in sentences:
                if sent_size + len(sent) > chunk_size_chars and sent_buffer:
                    chunks.append(' '.join(sent_buffer))
                    # Overlap: keep last sentence
                    sent_buffer = [sent_buffer[-1], sent]
                    sent_size = len(sent_buffer[-2]) + len(sent)
                else:
                    sent_buffer.append(sent)
                    sent_size += len(sent)

            if sent_buffer:
                chunks.append(' '.join(sent_buffer))

        elif current_size + para_size > chunk_size_chars and current_parts:
            # Current chunk is full — save it
            chunks.append('\n\n'.join(current_parts))

            # Overlap: carry last paragraph into next chunk
            last_para = current_parts[-1]
            if len(last_para) <= overlap_chars:
                current_parts = [last_para, para]
                current_size = len(last_para) + para_size
            else:
                current_parts = [para]
                current_size = para_size
        else:
            current_parts.append(para)
            current_size += para_size

    # Last chunk
    if current_parts:
        chunks.append('\n\n'.join(current_parts))

    # Filter noise chunks
    chunks = [c for c in chunks if len(c) > 150]

    return chunks


def chunk_paper(paper: dict, metadata: dict) -> list[dict]:
    """Chunk a single paper's extracted text into indexed chunks."""
    
    clean = clean_text(paper["full_text"])
    chunks = split_into_chunks(clean)

    chunk_objects = []
    for i, chunk_text in enumerate(chunks):
        chunk_objects.append({
            "chunk_id": f"{paper['arxiv_id']}_{i:04d}",
            "arxiv_id": paper["arxiv_id"],
            "chunk_index": i,
            "total_chunks": len(chunks),
            "text": chunk_text,
            "char_count": len(chunk_text),
            # Metadata for filtering
            "title": metadata.get("title", ""),
            "authors": metadata.get("authors", []),
            "published": metadata.get("published", ""),
            "year": metadata.get("published", "")[:4],
            "categories": metadata.get("categories", []),
            "primary_category": metadata.get("primary_category", ""),
            "abstract": metadata.get("abstract", ""),
        })

    return chunk_objects


def chunk_all(
    processed_dir: Path = PROCESSED_DIR,
    chunks_dir: Path = CHUNKS_DIR,
    metadata_file: Path = Path("data/raw/papers_metadata.json"),
    max_papers: int = None
) -> list[dict]:

    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata for all papers
    with open(metadata_file, "r", encoding="utf-8") as f:
        all_metadata = json.load(f)
    metadata_map = {p["arxiv_id"]: p for p in all_metadata}

    # Get processed files
    processed_files = sorted(processed_dir.glob("*.json"))
    if max_papers:
        processed_files = processed_files[:max_papers]

    all_chunks = []
    stats = []

    for proc_file in tqdm(processed_files, desc="Chunking papers"):
        with open(proc_file, "r", encoding="utf-8") as f:
            paper = json.load(f)

        if paper["extraction_status"] != "success":
            continue

        arxiv_id = paper["arxiv_id"]
        metadata = metadata_map.get(arxiv_id, {})

        chunks = chunk_paper(paper, metadata)
        all_chunks.extend(chunks)

        stats.append({
            "arxiv_id": arxiv_id,
            "title": metadata.get("title", "")[:60],
            "num_chunks": len(chunks),
            "avg_chunk_chars": int(sum(c["char_count"] for c in chunks) / len(chunks)) if chunks else 0
        })

    # Save all chunks
    output_file = chunks_dir / "all_chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # Print stats
    print(f"\nChunking complete:")
    print(f"  Papers processed: {len(stats)}")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"\nPer-paper breakdown:")
    for s in stats:
        print(f"  {s['arxiv_id']} | {s['num_chunks']} chunks | "
              f"avg {s['avg_chunk_chars']} chars | {s['title']}")

    # Show a sample chunk
    if all_chunks:
        sample = all_chunks[5]
        print(f"\n--- Sample chunk ({sample['chunk_id']}) ---")
        print(f"Text: {sample['text'][:300]}")

    return all_chunks


if __name__ == "__main__":
    chunks = chunk_all()