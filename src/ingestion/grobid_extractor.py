import requests
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent.parent
GROBID_URL = "http://localhost:8070"
PDF_DIR = BASE_DIR / "data/raw/pdfs"
PROCESSED_DIR = BASE_DIR / "data/processed/grobid"


# TEI XML namespaces used by Grobid
NS = {
    "tei": "http://www.tei-c.org/ns/1.0",
    "xlink": "http://www.w3.org/1999/xlink"
}


def check_grobid_alive() -> bool:
    """Check if Grobid is running."""
    try:
        resp = requests.get(f"{GROBID_URL}/api/isalive", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def parse_tei_xml(xml_str: str, arxiv_id: str) -> dict:
    """Parse Grobid's TEI XML output into structured dict."""
    
    result = {
        "arxiv_id": arxiv_id,
        "extraction_method": "grobid",
        "title": "",
        "authors": [],
        "abstract": "",
        "sections": [],
        "references": [],
        "extraction_status": "success"
    }

    try:
        root = ET.fromstring(xml_str)

        # --- Extract title ---
        title_el = root.find(".//tei:titleStmt/tei:title[@type='main']", NS)
        if title_el is not None and title_el.text:
            result["title"] = title_el.text.strip()

        # --- Extract authors ---
        for author in root.findall(".//tei:fileDesc//tei:author", NS):
            forename = author.find(".//tei:forename", NS)
            surname = author.find(".//tei:surname", NS)
            if forename is not None and surname is not None:
                name = f"{forename.text} {surname.text}".strip()
                result["authors"].append(name)

        # --- Extract abstract ---
        abstract_el = root.find(".//tei:abstract", NS)
        if abstract_el is not None:
            abstract_text = " ".join(abstract_el.itertext()).strip()
            result["abstract"] = abstract_text

        # --- Extract sections from body ---
        body = root.find(".//tei:body", NS)
        if body is not None:
            for div in body.findall("tei:div", NS):
                section = extract_section(div)
                if section and section["text"].strip():
                    result["sections"].append(section)

        # --- Extract references ---
        ref_list = root.find(".//tei:listBibl", NS)
        if ref_list is not None:
            for bibl in ref_list.findall("tei:biblStruct", NS):
                ref = extract_reference(bibl)
                if ref:
                    result["references"].append(ref)

    except ET.ParseError as e:
        result["extraction_status"] = f"xml_parse_error: {str(e)}"

    return result


def extract_section(div_el) -> dict:
    """Extract a single section from a TEI div element."""
    
    # Get section heading
    head_el = div_el.find("tei:head", NS)
    heading = ""
    if head_el is not None and head_el.text:
        heading = head_el.text.strip()

    # Get section number if available
    section_num = head_el.get("n", "") if head_el is not None else ""

    # Get all text content (including nested paragraphs)
    paragraphs = []
    for p in div_el.findall(".//tei:p", NS):
        text = " ".join(p.itertext()).strip()
        if text:
            paragraphs.append(text)

    # Also get nested divs (subsections)
    subsections = []
    for subdiv in div_el.findall("tei:div", NS):
        sub = extract_section(subdiv)
        if sub:
            subsections.append(sub)

    return {
        "heading": heading,
        "section_num": section_num,
        "paragraphs": paragraphs,
        "subsections": subsections,
        "text": "\n\n".join(paragraphs)
    }


def extract_reference(bibl_el) -> dict:
    """Extract a single reference from a biblStruct element."""
    ref = {}

    # Title
    title_el = bibl_el.find(".//tei:title[@level='a']", NS)
    if title_el is None:
        title_el = bibl_el.find(".//tei:title", NS)
    if title_el is not None and title_el.text:
        ref["title"] = title_el.text.strip()

    # Authors
    authors = []
    for author in bibl_el.findall(".//tei:author", NS):
        surname = author.find(".//tei:surname", NS)
        if surname is not None and surname.text:
            authors.append(surname.text.strip())
    ref["authors"] = authors

    # Year
    date_el = bibl_el.find(".//tei:date[@type='published']", NS)
    if date_el is not None:
        ref["year"] = date_el.get("when", "")[:4]

    return ref if ref else None


def process_pdf_with_grobid(
    pdf_path: Path,
    timeout: int = 120
) -> dict:
    """Send a PDF to Grobid and get structured extraction back."""

    arxiv_id = pdf_path.stem

    try:
        with open(pdf_path, "rb") as f:
            response = requests.post(
                f"{GROBID_URL}/api/processFulltextDocument",
                files={"input": (pdf_path.name, f, "application/pdf")},
                data={
                    "consolidateHeader": "0",
                    "consolidateCitations": "0",
                    "includeRawAffiliations": "0",
                    "includeRawCitations": "0",
                    "segmentSentences": "0",
                },
                timeout=timeout
            )

        if response.status_code == 200:
            return parse_tei_xml(response.text, arxiv_id)
        elif response.status_code == 503:
            return {
                "arxiv_id": arxiv_id,
                "extraction_status": "grobid_busy",
                "extraction_method": "grobid"
            }
        else:
            return {
                "arxiv_id": arxiv_id,
                "extraction_status": f"http_error_{response.status_code}",
                "extraction_method": "grobid"
            }

    except requests.Timeout:
        return {
            "arxiv_id": arxiv_id,
            "extraction_status": "timeout",
            "extraction_method": "grobid"
        }
    except Exception as e:
        return {
            "arxiv_id": arxiv_id,
            "extraction_status": f"error: {str(e)}",
            "extraction_method": "grobid"
        }


def extract_all_with_grobid(
    pdf_dir: Path = PDF_DIR,
    processed_dir: Path = PROCESSED_DIR,
    max_papers: int = None,
    delay: float = 1.0
) -> list[dict]:

    processed_dir.mkdir(parents=True, exist_ok=True)

    # Check Grobid is alive
    if not check_grobid_alive():
        print("ERROR: Grobid is not running at localhost:8070")
        print("Start it with: docker run --rm --name grobid -p 8070:8070 lfoppiano/grobid:0.8.0")
        return []

    print(f"Grobid is running at {GROBID_URL}")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if max_papers:
        pdf_files = pdf_files[:max_papers]

    results = []
    stats = {"success": 0, "failed": 0, "fallback_needed": []}

    for pdf_path in tqdm(pdf_files, desc="Processing with Grobid"):
        output_file = processed_dir / f"{pdf_path.stem}.json"

        # Skip if already processed
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                result = json.load(f)
            results.append(result)
            stats["success"] += 1
            continue

        result = process_pdf_with_grobid(pdf_path)

        if result["extraction_status"] == "success":
            stats["success"] += 1
        else:
            stats["failed"] += 1
            stats["fallback_needed"].append(pdf_path.stem)
            print(f"\nFailed: {pdf_path.stem} — {result['extraction_status']}")

        # Save result
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        results.append(result)
        time.sleep(delay)

    # Summary
    print(f"\nGrobid extraction complete:")
    print(f"  Success:  {stats['success']}")
    print(f"  Failed:   {stats['failed']}")
    if stats["fallback_needed"]:
        print(f"  Need PyMuPDF fallback: {stats['fallback_needed']}")

    return results


if __name__ == "__main__":
    # Test on 3 papers first
    results = extract_all_with_grobid(max_papers=3)

    if results:
        sample = results[0]
        print(f"\n--- Sample: {sample['arxiv_id']} ---")
        print(f"Title: {sample.get('title', 'N/A')}")
        print(f"Authors: {sample.get('authors', [])[:3]}")
        print(f"Sections found: {len(sample.get('sections', []))}")
        print(f"References found: {len(sample.get('references', []))}")
        
        if sample.get("sections"):
            first_section = sample["sections"][0]
            print(f"\nFirst section heading: '{first_section['heading']}'")
            print(f"First section text (300 chars):")
            print(first_section["text"][:300])