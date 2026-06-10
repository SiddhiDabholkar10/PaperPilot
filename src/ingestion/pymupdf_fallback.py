import re
import json
import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm
from ids import make_paper_id, make_abstract_id, make_parent_id, make_child_id

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PYMUPDF_DIR = BASE_DIR / "data/processed/pymupdf"
PDF_DIR = BASE_DIR / "data/raw/pdfs"

PARENT_CHUNK_SIZE = 1500
CHILD_CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Section header patterns common in ML papers
SECTION_PATTERN = re.compile(
    r'^(\d+\.?\d*\.?\s+)?'  # optional numbering like "1." or "1.1"
    r'(Abstract|Introduction|Related Work|Background|'
    r'Preliminaries|Problem Formulation|'
    r'Method|Methods|Methodology|Approach|Model|Framework|'
    r'Experiment|Experiments|Experimental Setup|Evaluation|'
    r'Results|Analysis|Discussion|'
    r'Conclusion|Conclusions|Future Work|'
    r'References|Appendix|Acknowledgment|Acknowledgements)'
    r'(\s|$)',
    re.IGNORECASE
)

SKIP_SECTIONS = {
    "references", "acknowledgment", "acknowledgements",
    "appendix", "funding"
}


def clean_text(text: str) -> str:
    """Clean raw PDF text."""
    # Fix hyphenation at line breaks
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove lone numbers (page numbers)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # Normalize spaces
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove lines that are mostly non-alphabetic (equations, tables)
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            clean_lines.append('')
            continue
        alpha_chars = sum(c.isalpha() for c in stripped)
        # Keep line if >40% alphabetic or very short (headings)
        if len(stripped) == 0 or alpha_chars / len(stripped) > 0.40:
            clean_lines.append(line)
    return '\n'.join(clean_lines).strip()


def detect_sections(full_text: str) -> list[dict]:
    """
    Detect sections from raw text using regex.
    Returns list of dicts with heading and text.
    """
    lines = full_text.split('\n')
    sections = []
    current_heading = "HEADER"
    current_lines = []

    for line in lines:
        stripped = line.strip()
        if SECTION_PATTERN.match(stripped) and len(stripped) < 80:
            # Save previous section
            section_text = '\n'.join(current_lines).strip()
            if section_text:
                sections.append({
                    "heading": current_heading,
                    "text": section_text
                })
            current_heading = stripped
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    if current_lines:
        section_text = '\n'.join(current_lines).strip()
        if section_text:
            sections.append({
                "heading": current_heading,
                "text": section_text
            })

    return sections


def make_chunks(text: str, chunk_size: int) -> list[str]:
    """Split text into chunks of approximately chunk_size chars."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current = []
    current_size = 0

    for para in paragraphs:
        if current_size + len(para) > chunk_size and current:
            chunks.append('\n\n'.join(current))
            last = current[-1]
            current = [last, para] if len(last) <= CHUNK_OVERLAP * 3 else [para]
            current_size = sum(len(p) for p in current)
        else:
            current.append(para)
            current_size += len(para)

    if current:
        chunks.append('\n\n'.join(current))

    return [c for c in chunks if len(c) > 100]


def extract_and_chunk(
    pdf_path: Path,
    metadata: dict
) -> dict:
    """
    Full PyMuPDF fallback pipeline for a single paper.
    Produces same chunk structure as Grobid chunker.
    """
    arxiv_id = pdf_path.stem

    base_meta = {
        "arxiv_id": arxiv_id,
        "title": metadata.get("title", ""),
        "authors": metadata.get("authors", []),
        "published": metadata.get("published", ""),
        "year": metadata.get("published", "")[:4],
        "categories": metadata.get("categories", []),
        "primary_category": metadata.get("primary_category", ""),
        "extraction_method": "pymupdf_fallback"
    }

    abstract_chunks = []
    parent_chunks = []
    child_chunks = []

    # --- Extract raw text ---
    try:
        doc = fitz.open(pdf_path)
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text("text"))
        doc.close()
        full_text = clean_text('\n'.join(pages_text))
    except Exception as e:
        return {
            "arxiv_id": arxiv_id,
            "status": f"extraction_failed: {e}",
            "abstract_chunks": [],
            "parent_chunks": [],
            "child_chunks": []
        }

    # --- Abstract from metadata (always clean) ---
    abstract_text = metadata.get("abstract", "")
    if abstract_text:
        abstract_chunks.append({
            **base_meta,
            "chunk_id": make_abstract_id(make_paper_id(arxiv_id)),
            "chunk_type": "abstract",
            "section": "ABSTRACT",
            "section_heading": "Abstract",
            "text": abstract_text,
            "parent_chunk_id": None,
            "char_count": len(abstract_text)
        })

    # --- Detect sections ---
    sections = detect_sections(full_text)

    parent_idx = 0
    child_idx = 0

    for section in sections:
        heading = section["heading"].strip()
        heading_lower = heading.lower()

        # Skip non-content sections
        if any(skip in heading_lower for skip in SKIP_SECTIONS):
            continue

        section_text = section["text"].strip()
        if len(section_text) < 150:
            continue

        # Normalize section label
        section_label = re.sub(
            r'^\d+\.?\d*\.?\s*', '', heading
        ).upper() or "BODY"

        # --- Parent chunks ---
        parent_texts = make_chunks(section_text, PARENT_CHUNK_SIZE)

        for parent_text in parent_texts:
            parent_chunk_id = make_parent_id(make_paper_id(arxiv_id), parent_idx)

            parent_chunks.append({
                **base_meta,
                "chunk_id": parent_chunk_id,
                "chunk_type": "parent",
                "section": section_label,
                "section_heading": heading,
                "text": parent_text,
                "parent_chunk_id": None,
                "char_count": len(parent_text)
            })

            # --- Child chunks ---
            child_texts = make_chunks(parent_text, CHILD_CHUNK_SIZE)

            for child_text in child_texts:
                child_chunks.append({
                    **base_meta,
                    "chunk_id": make_child_id(parent_chunk_id, child_idx),
                    "chunk_type": "child",
                    "section": section_label,
                    "section_heading": heading,
                    "text": child_text,
                    "parent_chunk_id": parent_chunk_id,
                    "char_count": len(child_text)
                })
                child_idx += 1

            parent_idx += 1

    return {
        "arxiv_id": arxiv_id,
        "status": "success",
        "abstract_chunks": abstract_chunks,
        "parent_chunks": parent_chunks,
        "child_chunks": child_chunks,
        "stats": {
            "sections_detected": len(sections),
            "parent_chunks": len(parent_chunks),
            "child_chunks": len(child_chunks)
        }
    }


def run_pymupdf_fallback(
    failed_ids: list[str],
    metadata_map: dict,
    pdf_dir: Path = PDF_DIR
) -> tuple[list, list, list]:
    """
    Run proper PyMuPDF fallback for a list of arxiv IDs.
    Returns (abstract_chunks, parent_chunks, child_chunks)
    """
    all_abstract = []
    all_parent = []
    all_child = []

    print(f"\nPyMuPDF fallback for {len(failed_ids)} papers...")

    for arxiv_id in tqdm(failed_ids, desc="PyMuPDF fallback"):
        pdf_path = pdf_dir / f"{arxiv_id}.pdf"
        if not pdf_path.exists():
            print(f"  PDF not found: {arxiv_id}")
            continue

        metadata = metadata_map.get(arxiv_id, {})
        result = extract_and_chunk(pdf_path, metadata)

        if result["status"] == "success":
            all_abstract.extend(result["abstract_chunks"])
            all_parent.extend(result["parent_chunks"])
            all_child.extend(result["child_chunks"])

            print(f"  {arxiv_id}: "
                  f"{result['stats']['sections_detected']} sections, "
                  f"{result['stats']['parent_chunks']} parents, "
                  f"{result['stats']['child_chunks']} children")
        else:
            print(f"  {arxiv_id}: FAILED — {result['status']}")

    return all_abstract, all_parent, all_child


if __name__ == "__main__":
    # Test on one paper directly
    from pathlib import Path
    import json

    metadata_file = Path("../../data/raw/papers_metadata.json")
    with open(metadata_file) as f:
        all_meta = json.load(f)
    metadata_map = {p["arxiv_id"]: p for p in all_meta}

    # Pick the first available PDF
    pdf_files = sorted(Path("../../data/raw/pdfs").glob("*.pdf"))
    if pdf_files:
        test_pdf = pdf_files[0]
        arxiv_id = test_pdf.stem
        metadata = metadata_map.get(arxiv_id, {})

        result = extract_and_chunk(test_pdf, metadata)

        print(f"Status: {result['status']}")
        print(f"Stats: {result['stats']}")
        print(f"\nSections detected: {result['stats']['sections_detected']}")
        print(f"Parent chunks: {result['stats']['parent_chunks']}")
        print(f"Child chunks: {result['stats']['child_chunks']}")

        if result["child_chunks"]:
            sample = result["child_chunks"][5]
            print(f"\nSample child chunk:")
            print(f"  Section: {sample['section_heading']}")
            print(f"  Text: {sample['text'][:300]}")