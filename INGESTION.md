# PaperPilot Ingestion Pipeline

## 1. Overview
PaperPilot's ingestion pipeline builds a research-ready corpus from arXiv NLP/ML papers and transforms it into structured, validated, retrieval-friendly chunks. The pipeline is designed as a staged system so each step is inspectable and restartable: discovery and curation of papers, PDF acquisition, structure-aware parsing (GROBID with PyMuPDF fallback), parent-child chunking, quality validation, and metadata enrichment (Semantic Scholar). The main outputs are corpus metadata, chunk files for indexing/retrieval, and enrichment files with citation and topic signals.

## 2. Architecture
High-level flow:

```text
fetcher/diverse_fetcher + fetch_seed_papers
                |
                v
      data/raw/papers_metadata.json
                |
                v
          pdf_downloader
                |
                v
           data/raw/pdfs/*.pdf
                |
                v
         grobid_extractor (TEI)
                |
      success -> data/processed/grobid/*.json
      failure -> pymupdf_fallback (direct PDF text)
                |
                v
            grobid_chunker
         (plus fallback chunks)
                |
                v
 data/chunks/chunks_abstract.json
 data/chunks/chunks_parent.json
 data/chunks/chunks_child.json
                |
                +--> validator -> data/chunks/validation_report.json
                |
                +--> semantic_scholar ->
                     data/raw/papers_enriched.json
                     data/raw/paper_references.json
```

Orchestration is handled by `src/ingestion/pipeline.py`, which runs extraction + chunking + fallback and merges all chunk outputs.

## 3. Dataset
- Source: arXiv API (`cs.LG`, `cs.CL`) + a curated foundational seed list.
- Date range: 2014-09-01 to 2026-05-21 (from `published` timestamps in current corpus).
- Corpus composition: 274 papers total.
- Year band breakdown:
  - `foundational`: 24 (manual seed papers)
  - `2020-foundational`: 30
  - `2021-2022-improvements`: 40
  - `2023-2024-llm-rag`: 80
  - `2025-2026-latest`: 100 (50 from 2025 + 50 from 2026)
- Why these categories and date range:
  - `cs.LG` + `cs.CL` focuses the corpus on modern ML/NLP + retrieval/generation work.
  - Banded sampling prevents over-weighting only the newest papers and preserves historical context.
  - Seed papers ensure canonical milestones are included even if not surfaced by recency/relevance queries.

## 4. Ingestion Pipeline
### `fetcher.py` / `diverse_fetcher.py` / `fetch_seed_papers.py`
- `fetcher.py`: baseline fetch using arXiv query + start year filter (resume-safe).
- `diverse_fetcher.py`: balanced corpus collection by year bands and targets; filters very short abstracts.
- `fetch_seed_papers.py`: injects 24 foundational papers (e.g., Transformer, BERT, GPT-3, RAG, SQuAD family).
- Why: combines broad coverage (automated fetch) with controlled temporal diversity and guaranteed canonical papers.

### `pdf_downloader.py`
- Downloads paper PDFs from `pdf_url` into `data/raw/pdfs` with retry-safe looping and skip-if-exists behavior.
- Why: separates acquisition from parsing so extraction steps can be rerun without redownloading.

### `grobid_extractor.py`
- Sends PDFs to local GROBID (`/api/processFulltextDocument`) and parses TEI XML into title/authors/abstract/sections/references.
- Records extraction status per paper (`success`, timeout, `http_error_*`, etc.).
- Why: TEI structure gives cleaner section boundaries than raw PDF text extraction.

### `pymupdf_fallback.py`
- Fallback path for papers where GROBID fails.
- Uses PyMuPDF text extraction + regex section detection + same chunk schema as GROBID path.
- Why: guarantees near-complete coverage even when parser service fails.

### `grobid_chunker.py`
- Converts sectioned paper content into abstract/parent/child chunks with stable IDs and metadata.
- Skips low-value sections (e.g., acknowledgements/appendix/funding) and tiny/noisy content.
- Why: retrieval needs smaller chunks; synthesis needs larger context windows.

### `deduplicator.py`
- Deduplicates corpus first by exact arXiv ID, then by title/abstract similarity thresholds.
- Why: avoids duplicate semantic content in retrieval and evaluation.

### `pipeline.py`
- End-to-end orchestrator for download (optional), GROBID extraction, chunking, fallback, merge, dedup-by-chunk-id, and save.
- Why: one command to run core ingestion reliably while preserving individual script modularity.

## 5. Chunking Strategy
- Why parent-child chunking:
  - Parent chunks preserve broader reasoning context for synthesis.
  - Child chunks improve retrieval precision for embedding/search.
- Chunk sizes:
  - Parent: 1500 chars
  - Child: 400 chars
  - Overlap: 50 chars
- Abstract as special chunk:
  - Stored separately (`::abstract`) so high-signal summary text can be prioritized.
- Section-aware extraction via GROBID:
  - Uses TEI sections/headings instead of flat page text whenever possible.
- Stable ID format:
  - `arxiv_id::parent::XXXX::child::XXXX`
  - Plus `arxiv_id::abstract` and `arxiv_id::parent::XXXX`

## 6. Data Validation
Checks run by `validator.py`:
- Duplicate chunk IDs
- Missing required fields
- Missing `parent_chunk_id` in child chunks
- Empty text
- Orphan child chunks
- Very short chunks
- Symbol-heavy chunks
- Missing title/section
- Invalid year
- Invalid metadata types (`authors`, `categories`)

Severity model:
- Critical: duplicate IDs, missing required fields, missing parent IDs, empty text, orphan children
- Warning: short/symbol-heavy/missing optional metadata/type anomalies

Final corpus state (`data/chunks/validation_report.json`):
- Critical: 0
- Warnings: 27 symbol-heavy chunks accepted

## 7. Metadata Enrichment
- Enricher: `semantic_scholar.py` (Semantic Scholar Graph API).
- Captured fields:
  - `citation_count`
  - `influential_citation_count`
  - `s2_tldr`
  - `s2_topics`
- Coverage (current run):
  - 224/274 found
  - 50/274 not found (mostly very new papers not yet indexed)
- Why Semantic Scholar over OpenAlex:
  - Native `influentialCitationCount` and TLDR signals are available.
  - ArXiv-ID direct lookup is cleaner than title-match heuristics.
  - OpenAlex path in this repo is title matching and does not provide influential citation counts.

## 8. Corpus Statistics
Current final numbers:
- Papers: 274
- Abstract chunks: 273
- Parent chunks: 9,310
- Child chunks: 16,739
- Total chunks: 26,322

Extraction breakdown:
- GROBID child chunks: 16,714
- PyMuPDF fallback child chunks: 25

Top cited papers (from `papers_enriched.json`):
1. Deep Residual Learning for Image Recognition (228,510)
2. Attention Is All You Need (177,283)
3. Adam: A Method for Stochastic Optimization (166,266)
4. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (114,811)
5. Language Models are Few-Shot Learners (58,240)

## 9. Design Decisions
- GROBID over pure PyMuPDF:
  - Better structural parsing (sections/references) through TEI.
- Parent-child over flat chunking:
  - Improves retrieval precision and downstream synthesis quality.
- S2 over OpenAlex:
  - Better aligned enrichment signals (`influentialCitationCount`, TLDR, direct arXiv lookup).
- Keeping symbol-heavy chunks (27 warnings accepted):
  - Some technical passages contain math/table notation but remain useful.
- Seed papers strategy:
  - Guarantees foundational coverage independent of recency search bias.

## 10. How to Reproduce
From repo root (`PaperPilot`):

1. Create environment and install dependencies.
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
py -3.11 -m venv venv
pip install -r requirements.txt
```

2. Add environment variables in `.env`.
```text
SEMANTIC_SCHOLAR_API_KEY=your_key_here
```

3. Build metadata corpus.
```powershell
python src/ingestion/diverse_fetcher.py
python src/ingestion/fetch_seed_papers.py
python src/ingestion/deduplicator.py
```

4. Download PDFs.
```powershell
python src/ingestion/pdf_downloader.py
```

5. Start GROBID locally (Docker) before extraction.
```powershell
docker run --rm --name grobid -p 8070:8070 lfoppiano/grobid:0.8.0
```

6. Run core ingestion pipeline (GROBID + fallback + chunk merge).
```powershell
python src/ingestion/pipeline.py
```

7. Validate chunks.
```powershell
python src/ingestion/validator.py
```

8. Enrich with Semantic Scholar.
```powershell
python src/ingestion/semantic_scholar.py
```

## 11. Known Limitations
- 50 papers are too new for current Semantic Scholar indexing (`s2_status=not_found`).
- - 50 papers too new for S2 indexing have citation_count=0 and no reference graph entries
- One paper (`2605.22748`) has GROBID `http_error_500` and is handled by PyMuPDF fallback.
- A small set of symbol-heavy chunks remains (27 warnings) due to technical math/table text.
- 
## 12. Files and Outputs
- `data/raw/papers_metadata.json`: canonical paper-level metadata corpus.
- `data/raw/pdfs/*.pdf`: downloaded source PDFs.
- `data/raw/pdfs/download_results.json`: PDF download success/failure report.
- `data/raw/dedup_report.json`: deduplication summary.
- `data/processed/grobid/*.json`: per-paper GROBID extraction outputs and statuses.
- `data/chunks/chunks_abstract.json`: one abstract chunk per paper when available.
- `data/chunks/chunks_parent.json`: larger synthesis-context chunks.
- `data/chunks/chunks_child.json`: retrieval chunks linked to parents via `parent_chunk_id`.
- `data/chunks/validation_report.json`: validation metrics and issue samples.
- `data/raw/papers_enriched.json`: corpus metadata plus S2 enrichment fields.
- `data/raw/paper_references.json`: per-paper reference payload currently used for graph expansion work.


## 13. Future Work
- Scale corpus to 1,000+ papers
- Fix reference graph extraction for citation-aware retrieval
- Add S2 enrichment for 2025-2026 papers once indexed
- Chunk quality scoring (citation boost, section weighting)
