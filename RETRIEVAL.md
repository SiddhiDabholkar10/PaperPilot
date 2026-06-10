# PaperPilot Retrieval (Phase 2)

## 1. Overview
Phase 2 builds the retrieval foundation that the agent layer depends on: dense embeddings, sparse BM25 search, hybrid fusion, and context recovery through parent chunks. The output is a queryable stack that returns high-signal evidence chunks with paper metadata, citation signals, and optional abstract supplementation.

## 2. Architecture
High-level flow:

```text
User Query
   |
   +--> Dense path: BGE query embedding -> Pinecone top-k child/abstract vectors
   |
   +--> Sparse path: BM25 query scoring over child chunk corpus
   |
   +--> Reciprocal Rank Fusion (RRF)
   |
   +--> Dedup + parent expansion from SQLite
   |
   +--> Optional abstract supplement from papers table
   |
RetrievedChunk list (text + metadata + scores)
```

Core files:
- `src/retrieval/embedder.py`
- `src/retrieval/pinecone_store.py`
- `src/retrieval/bm25_index.py`
- `src/retrieval/retriever.py`
- `src/retrieval/utils.py`

## 3. Embedding
- Model: `BAAI/bge-large-en-v1.5`
- Why this model:
  - Strong retrieval quality for English technical text.
  - 1024-dim embeddings align with current Pinecone index settings.
- Runtime config:
  - Batch size: `64` (tuned for your GPU comment in code).
  - Normalization: `normalize_embeddings=True` for cosine retrieval.
  - Device: CUDA if available, else CPU.
- What gets embedded:
  - `chunks_child.json`
  - `chunks_abstract.json`
  - Parent chunks are not embedded.
- Cache/resume strategy:
  - Writes to `data/embeddings/embeddings_child.json` and `embeddings_abstract.json`.
  - Existing cached IDs are skipped.
  - Incremental flush every ~10 batches + final flush.

## 4. Vector Store (Pinecone)
- Current expected index config:
  - Dimension: `1024`
  - Similarity: `cosine`
  - Deployment: serverless (configured in Pinecone, not created in repo code)
- Stored metadata per vector (lightweight):
  - `chunk_id`, `arxiv_id`, `title`, `year`
  - `section`, `section_heading`, `chunk_type`
  - `parent_chunk_id`, `extraction_method`
  - `citation_count`, `influential_citation_count`, `primary_category`
- Why metadata is lightweight:
  - Avoid metadata size bloat and Pinecone constraints.
  - Keep vector payload efficient and stable.
- Upsert + resume logic:
  - Batch upsert (`UPSERT_BATCH_SIZE=100`), retries (`MAX_BATCH_RETRIES=3`).
  - `upserted_chunks` tracks progress per `pinecone_index`.
  - `failed_chunks` captures errors for postmortem/retry.
  - Null metadata is sanitized before upsert.

## 5. Local Store (SQLite)
DB file: `data/paperpilot.db`

Tables used by retrieval:
- `papers`
- `parent_chunks`
- `upserted_chunks`
- `failed_chunks`

Why parents are in SQLite (not Pinecone):
- Parent text is large and used for synthesis context, not initial retrieval matching.
- Child vectors retrieve precisely; parent lookup restores broader context cheaply.

Parent-child pattern:
- Retrieve child-level candidates.
- Resolve `parent_chunk_id` in SQLite.
- Return parent text as final context window when available.

## 6. BM25 Index
- Built over child chunks only.
- Tokenizer is centralized in `src/retrieval/utils.py`:
  - lowercase
  - hyphen split
  - regex non-alphanumeric split
  - removes single-char tokens except `"a"`
- Why regex tokenizer over simple split:
  - Better handling of scientific punctuation/hyphenation (`self-attention`, `kv-cache`, etc.).
- Why single source of truth matters:
  - Same tokenizer is used for index construction and query tokenization.
  - Prevents train/query token mismatch drift.
- Index artifacts:
  - `data/index/bm25_index.pkl`
  - `data/index/bm25_chunk_ids.json`

## 7. Hybrid Retrieval
- Combines:
  - Dense search (Pinecone)
  - Sparse search (BM25)
  - RRF fusion
- RRF:
  - Score formula: `sum(1 / (K + rank))`
  - Constant: `K = 60`
- Dedup strategy:
  - Parent dedup when parent context is used.
  - Paper-level cap: max 2 chunks per paper in current retriever.
- Abstract supplement:
  - Optional (`include_abstracts=True`)
  - Adds abstract context from `papers.abstract` for retrieved papers.
- BM25-only fallback:
  - If a fused candidate is not in dense metadata, retriever falls back to SQLite-based reconstruction.

## 8. Retrieval Eval Baseline
- Evaluation set: `eval/gold_questions.json` (20 questions).
- Baseline from `eval/retrieval_eval_results.json`:
  - Avg recall: `0.675`
  - Perfect recall: `9/20`
  - Zero recall: `2/20`
- Failure analysis (`eval/failure_analysis_summary.json`):
  - Zero recall cases: `2`
  - Low recall cases: `1`
  - Hint breakdown (full CSV picture):
    - `rerank_or_topk_limit`: 3 (`Q014`, `Q015`, `Q010`)
    - `near_miss_expand_topk`: 4 (`Q006`, `Q012`, `Q013`, `Q003`)
    - `query_mismatch_or_sparse_vocab`: 4 (`Q004`, `Q017`, `Q018`, `Q019`)
- What agent layer is expected to improve:
  - Query decomposition/rewrite for harder bridge questions.
  - Better candidate reranking and evidence selection.
  - Multi-step retrieval planning when one-pass top-k misses.

### Expected agent improvements per failure type

| Hint | Root cause | Agent fix |
|---|---|---|
| query_mismatch_or_sparse_vocab | Query vocab != paper vocab | Decomposer generates targeted sub-queries |
| near_miss_expand_topk | Paper just outside top_k | Reformulator retries with top_k=15 |
| rerank_or_topk_limit | Paper in corpus but not retrieved | Critic triggers reformulation |

## 9. Design Decisions
- Child chunks for retrieval, parent chunks for synthesis:
  - Precision first, then context expansion.
- Cosine similarity:
  - Matches normalized BGE embeddings (`normalize_embeddings=True`).
- Paper-level dedup cap:
  - Prevents one paper from flooding top results.
- Abstract supplement:
  - Increases paper-level grounding with compact summaries.
- Reranker deferred:
  - Held for post-Phase 3 ablation so baseline impact can be measured cleanly.

## 10. Known Limitations
- Remaining zero-recall failures in eval (currently 2/20).
- BGE query-format consistency can still be tuned (query instruction style vs current prefix choice).
- BM25 pickle loading assumes trusted local artifacts.
- No cross-encoder reranker yet.
- Index creation itself is external (Pinecone UI/API), only verification/upsert is in repo scripts.

## 11. Files and Outputs
- Embeddings:
  - `data/embeddings/embeddings_child.json`
  - `data/embeddings/embeddings_abstract.json`
- BM25:
  - `data/index/bm25_index.pkl`
  - `data/index/bm25_chunk_ids.json`
- SQLite:
  - `data/paperpilot.db`
- Pinecone verification:
  - `src/retrieval/verify_pinecone.py`
- Evaluation:
  - `eval/gold_questions.json`
  - `eval/retrieval_eval.py`
  - `eval/retrieval_eval_results.json`
  - `eval/failure_analysis.py`
  - `eval/failure_analysis.csv`
  - `eval/failure_analysis_summary.json`

## 12. How to Rebuild
From repo root:

1. Ensure ingestion artifacts exist:
```powershell
# Must already exist from ingestion phase:
# data/chunks/chunks_child.json
# data/chunks/chunks_abstract.json
# data/chunks/chunks_parent.json
# data/raw/papers_enriched.json
```

2. Build embeddings:
```powershell
python src/retrieval/embedder.py
```

3. Build BM25 index:
```powershell
python src/retrieval/bm25_index.py
```

4. Ensure Pinecone index exists (external setup):
- Name: from `PINECONE_INDEX_NAME` in `.env`
- Dimension: 1024
- Metric: cosine

5. Upsert vectors + populate SQLite:
```powershell
python src/retrieval/pinecone_store.py
```

6. Verify Pinecone connectivity and sample retrieval:
```powershell
python src/retrieval/verify_pinecone.py
```

7. Run retrieval evaluation:
```powershell
python eval/retrieval_eval.py
python eval/failure_analysis.py
```

8. Optional sanity check:
```powershell
python scripts/retrieval_sanity_check.py
```
