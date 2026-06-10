import json
import time
import requests
from pathlib import Path
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent.parent
PDF_DIR = BASE_DIR / "data/raw/pdfs"
METADATA_FILE = BASE_DIR / "data/raw/papers_metadata.json"
DELAY_SECONDS = 3

def download_pdfs(
    metadata_file: Path = METADATA_FILE,
    pdf_dir: Path = PDF_DIR,
    max_papers: int = None,
    delay: float = DELAY_SECONDS
) -> dict:
    
    pdf_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_file, "r", encoding="utf-8") as f:
        papers = json.load(f)

    if max_papers:
        papers = papers[:max_papers]

    results = {"success": [], "failed": []}

    for paper in tqdm(papers, desc="Downloading PDFs"):
        arxiv_id = paper["arxiv_id"]
        pdf_path = pdf_dir / f"{arxiv_id}.pdf"

        # Skip if already downloaded
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            results["success"].append(arxiv_id)
            continue

        try:
            response = requests.get(
                paper["pdf_url"],
                timeout=30,
                headers={"User-Agent": "PaperPilot/1.0 (research project)"}
            )
            response.raise_for_status()

            with open(pdf_path, "wb") as f:
                f.write(response.content)

            results["success"].append(arxiv_id)
            time.sleep(delay)

        except Exception as e:
            print(f"\nFailed {arxiv_id}: {e}")
            results["failed"].append(arxiv_id)
            time.sleep(delay)

    # Summary
    print(f"\nDownload complete:")
    print(f"  Success: {len(results['success'])}")
    print(f"  Failed:  {len(results['failed'])}")
    if results["failed"]:
        print(f"  Failed IDs: {results['failed']}")

    # Save results
    results_file = pdf_dir / "download_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    # Test with just 5 papers first
    results = download_pdfs()
    
    # Check file sizes
    print("\nDownloaded files:")
    for pdf_path in sorted(Path(PDF_DIR).glob("*.pdf")):
        size_kb = pdf_path.stat().st_size / 1024
        print(f"  {pdf_path.name}: {size_kb:.1f} KB")
