import fitz  # PyMuPDF
import json
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PDF_DIR = BASE_DIR / "data/raw/pdfs"
PROCESSED_DIR = BASE_DIR / "data/processed"


def extract_text_from_pdf(pdf_path: Path) -> dict:
    """Extract text from a single PDF with basic structure preservation."""
    result = {
        "arxiv_id": pdf_path.stem,
        "pages": [],
        "full_text": "",
        "page_count": 0,
        "extraction_status": "success"
    }

    try:
        doc = fitz.open(pdf_path)
        result["page_count"] = len(doc)
        full_text_parts = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text")  # plain text extraction
            
            # Basic cleanup
            text = text.strip()
            if text:
                result["pages"].append({
                    "page_num": page_num + 1,
                    "text": text,
                    "char_count": len(text)
                })
                full_text_parts.append(text)

        result["full_text"] = "\n\n".join(full_text_parts)
        doc.close()

    except Exception as e:
        result["extraction_status"] = f"failed: {str(e)}"

    return result


def extract_all(
    pdf_dir: Path = PDF_DIR,
    processed_dir: Path = PROCESSED_DIR,
    max_papers: int = None
) -> list[dict]:

    processed_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if max_papers:
        pdf_files = pdf_files[:max_papers]

    results = []

    for pdf_path in tqdm(pdf_files, desc="Extracting text"):
        output_file = processed_dir / f"{pdf_path.stem}.json"

        # Skip if already processed
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                results.append(json.load(f))
            continue

        extracted = extract_text_from_pdf(pdf_path)
        results.append(extracted)

        # Save individual file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(extracted, f, ensure_ascii=False, indent=2)

    # Summary
    successful = [r for r in results if r["extraction_status"] == "success"]
    failed = [r for r in results if r["extraction_status"] != "success"]

    print(f"\nExtraction complete:")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed:     {len(failed)}")
    print(f"\nText stats:")
    for r in successful:
        char_count = len(r["full_text"])
        print(f"  {r['arxiv_id']}: {r['page_count']} pages, {char_count:,} chars")

    return results


if __name__ == "__main__":
    results = extract_all()

    # Inspect one paper's extracted text
    if results:
        sample = results[0]
        print(f"\n--- Sample extraction: {sample['arxiv_id']} ---")
        print(f"Pages: {sample['page_count']}")
        print(f"First 500 chars of page 1:")
        if sample["pages"]:
            print(sample["pages"][0]["text"][:500])