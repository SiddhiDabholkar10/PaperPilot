import json
import pickle
import sqlite3
import os
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
from retriever_utils import tokenize
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

INDEX_DIR = BASE_DIR / "data/index"
DB_PATH = BASE_DIR / "data/paperpilot.db"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "paperpilot")
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K_DENSE = 15   
TOP_K_BM25  = 10  
TOP_K_FINAL = 12    
RRF_K       = 60   


@dataclass
class RetrievedChunk:
    """A retrieved chunk with full context."""
    chunk_id: str
    arxiv_id: str
    title: str
    section: str
    section_heading: str
    year: int
    text: str
    char_count: int
    parent_chunk_id: str
    extraction_method: str
    citation_count: int
    influential_citation_count: int
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    retrieval_method: str = "hybrid"
    s2_tldr: str = ""


class HybridRetriever:
    """
    Hybrid retriever combining dense (Pinecone) and sparse (BM25)
    search with Reciprocal Rank Fusion.
    """

    def __init__(self):
        print("Initializing HybridRetriever...")

        self.model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
        print(f"  Embedding model loaded on {DEVICE}")

        pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = pc.Index(PINECONE_INDEX)
        print(f"  Pinecone connected: {PINECONE_INDEX}")

        with open(INDEX_DIR / "bm25_index.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        with open(INDEX_DIR / "bm25_chunk_ids.json", encoding="utf-8") as f:
            self.bm25_chunk_ids = json.load(f)
        print(f"  BM25 loaded: {len(self.bm25_chunk_ids)} documents")

        self.db_path = DB_PATH
        print(f"  SQLite: {DB_PATH.name}")
        print("HybridRetriever ready\n")

    def _get_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query with BGE query prefix."""
        bge_query = f"BGE query: {query}"
        return self.model.encode(
            bge_query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

    def _dense_search(
        self,
        query_emb: np.ndarray,
        top_k: int = TOP_K_DENSE,
        retries: int = 2
    ) -> list[dict]:
        """Search Pinecone with dense embeddings."""
        for attempt in range(retries):
            try:
                results = self.index.query(
                    vector=query_emb.tolist(),
                    top_k=top_k,
                    include_metadata=True
                )
                return [
                    {
                        "chunk_id": m.id,
                        "score": m.score,
                        "metadata": m.metadata
                    }
                    for m in results.matches
                ]
            except Exception as e:
                if attempt < retries - 1:
                    print(f"  Pinecone query attempt {attempt+1} failed: {e}, retrying...")
                    time.sleep(2)
                else:
                    raise
        return []



    def _bm25_search(
        self,
        query: str,
        top_k: int = TOP_K_BM25
    ) -> list[dict]:
        """Search BM25 index with keyword matching."""
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "chunk_id": self.bm25_chunk_ids[idx],
                "score": float(scores[idx])
            }
            for idx in top_idx
            if scores[idx] > 0
        ]

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[dict],
        bm25_results: list[dict],
        k: int = RRF_K
    ) -> dict[str, float]:
        """
        Combine dense and BM25 results using Reciprocal Rank Fusion.
        RRF score = sum(1 / (k + rank)) for each result list.
        """
        rrf_scores = {}

        # Dense gets 2x weight
        for rank, result in enumerate(dense_results):
            chunk_id = result["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 2.0 / (k + rank + 1)

        # BM25 gets 1x weight
        for rank, result in enumerate(bm25_results):
            chunk_id = result["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)

        return rrf_scores

    def _fetch_child_chunk_from_db(
        self,
        conn: sqlite3.Connection,
        chunk_id: str
    ) -> Optional[dict]:
        """
        Fallback: fetch chunk metadata from SQLite
        when dense search didn't return it (BM25-only result).
        """
        cursor = conn.cursor()

        # Try parent_chunks table first
        cursor.execute(
            """SELECT chunk_id, arxiv_id, title, section,
                      section_heading, year, text, char_count,
                      extraction_method, parent_chunk_id
               FROM parent_chunks WHERE chunk_id = ?""",
            (chunk_id,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)

        # Derive from stable ID format
        parts = chunk_id.split("::")
        if len(parts) >= 3:
            arxiv_id = parts[0]
            parent_chunk_id = ""
            if "child" in parts:
                child_pos = parts.index("child")
                parent_chunk_id = "::".join(parts[:child_pos])

            if parent_chunk_id:
                cursor.execute(
                    """SELECT chunk_id, arxiv_id, title, section,
                              section_heading, year, text, char_count,
                              extraction_method
                       FROM parent_chunks WHERE chunk_id = ?""",
                    (parent_chunk_id,)
                )
                parent_row = cursor.fetchone()
                if parent_row:
                    result = dict(parent_row)
                    result["parent_chunk_id"] = parent_chunk_id
                    return result

            # Minimal fallback
            return {
                "chunk_id": chunk_id,
                "arxiv_id": arxiv_id,
                "title": "",
                "section": "",
                "section_heading": "",
                "year": 0,
                "text": "",
                "char_count": 0,
                "extraction_method": "",
                "parent_chunk_id": parent_chunk_id
            }

        return None

    def _fetch_parent_chunk(
        self,
        conn: sqlite3.Connection,
        parent_chunk_id: str
    ) -> Optional[dict]:
        """Fetch parent chunk from SQLite."""
        if not parent_chunk_id:
            return None

        cursor = conn.cursor()
        cursor.execute(
            """SELECT chunk_id, arxiv_id, title, section,
                      section_heading, year, text, char_count,
                      extraction_method
               FROM parent_chunks WHERE chunk_id = ?""",
            (parent_chunk_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def _fetch_paper_metadata(
        self,
        conn: sqlite3.Connection,
        arxiv_id: str
    ) -> dict:
        """Fetch paper metadata from SQLite."""
        cursor = conn.cursor()
        cursor.execute(
            """SELECT citation_count, influential_citation_count, s2_tldr, abstract
               FROM papers WHERE arxiv_id = ?""",
            (arxiv_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else {
            "citation_count": 0,
            "influential_citation_count": 0,
            "s2_tldr": "",
            "abstract": ""
        }

    def _fetch_abstract_for_paper(
        self,
        conn: sqlite3.Connection,
        arxiv_id: str
    ) -> Optional[RetrievedChunk]:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT title, abstract, citation_count,
                    influential_citation_count, s2_tldr, year
            FROM papers WHERE arxiv_id = ?""",
            (arxiv_id,)
        )
        row = cursor.fetchone()
        if not row or not row["abstract"]:
            return None

        return RetrievedChunk(
            chunk_id=f"{arxiv_id}::abstract",
            arxiv_id=arxiv_id,
            title=row["title"] or "",
            section="ABSTRACT",
            section_heading="Abstract",
            year=int(row["year"] or 0),        # ← fixed
            text=row["abstract"] or "",
            char_count=len(row["abstract"] or ""),
            parent_chunk_id="",
            extraction_method="metadata",
            citation_count=int(row["citation_count"] or 0),
            influential_citation_count=int(
                row["influential_citation_count"] or 0
            ),
            dense_score=0.0,
            bm25_score=0.0,
            rrf_score=0.0,
            retrieval_method="abstract_supplement",
            s2_tldr=row["s2_tldr"] or ""
        )
    
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_FINAL,
        top_k_dense: int = TOP_K_DENSE,
        top_k_bm25: int = TOP_K_BM25,
        use_parent: bool = True,
        include_abstracts: bool = False
    ) -> list[RetrievedChunk]:
        """
        Full hybrid retrieval pipeline.

        1. Embed query
        2. Dense search (Pinecone) — TOP_K_DENSE results
        3. BM25 search — TOP_K_BM25 results
        4. RRF fusion
        5. Fetch parent chunks from SQLite
        6. Return top_k RetrievedChunk objects
        """
        conn = self._get_db()

        try:
            query_emb = self._embed_query(query)

            dense_results = self._dense_search(query_emb, top_k=top_k_dense)
            dense_score_map = {r["chunk_id"]: r["score"] for r in dense_results}
            dense_meta_map = {r["chunk_id"]: r["metadata"] for r in dense_results}

            bm25_results = self._bm25_search(query, top_k=top_k_bm25)
            bm25_score_map = {r["chunk_id"]: r["score"] for r in bm25_results}

            rrf_scores = self._reciprocal_rank_fusion(dense_results, bm25_results)
            sorted_ids = sorted(
                rrf_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            retrieved = []
            seen_parents = set()
            seen_paper_counts = {}

            for chunk_id, rrf_score in sorted_ids:
                if len(retrieved) >= top_k:
                    break

                # Get metadata from dense results or fall back to SQLite
                dense_meta = dense_meta_map.get(chunk_id)

                if dense_meta:
                    arxiv_id = dense_meta.get("arxiv_id", chunk_id.split("::")[0])
                    parent_chunk_id = dense_meta.get("parent_chunk_id", "")
                    title = dense_meta.get("title", "")
                    section = dense_meta.get("section", "")
                    section_heading = dense_meta.get("section_heading", "")
                    year = int(dense_meta.get("year", 0) or 0)
                    extraction_method = dense_meta.get("extraction_method", "")
                else:
                    # BM25-only result — fall back to SQLite
                    db_chunk = self._fetch_child_chunk_from_db(conn, chunk_id)
                    if not db_chunk:
                        continue
                    arxiv_id = db_chunk.get("arxiv_id", chunk_id.split("::")[0])
                    parent_chunk_id = db_chunk.get("parent_chunk_id", "")
                    title = db_chunk.get("title", "")
                    section = db_chunk.get("section", "")
                    section_heading = db_chunk.get("section_heading", "")
                    year = int(db_chunk.get("year", 0) or 0)
                    extraction_method = db_chunk.get("extraction_method", "")
    

                # Deduplicate by paper — max 2 chunks per paper
               
                arxiv_count = seen_paper_counts.get(arxiv_id, 0)
                if arxiv_count >= 1:
                    continue

                # Also deduplicate by parent
                if use_parent and parent_chunk_id:
                    if parent_chunk_id in seen_parents:
                        continue
                    seen_parents.add(parent_chunk_id)

                seen_paper_counts[arxiv_id] = arxiv_count + 1

                # Fetch parent chunk for full context
                if use_parent and parent_chunk_id:
                    parent = self._fetch_parent_chunk(conn, parent_chunk_id)
                    text = parent["text"] if parent else ""
                    char_count = parent["char_count"] if parent else 0
                    if parent:
                        section = parent.get("section", section)
                        section_heading = parent.get(
                            "section_heading", section_heading
                        )
                else:
                    # No parent — likely an abstract chunk
                    # Fall back to abstract text from papers table
                    paper = self._fetch_paper_metadata(conn, arxiv_id)
                    text = paper.get("abstract", "") or ""
                    char_count = len(text)

                paper_meta = self._fetch_paper_metadata(conn, arxiv_id)

                retrieved.append(RetrievedChunk(
                    chunk_id=chunk_id,
                    arxiv_id=arxiv_id,
                    title=title,
                    section=section,
                    section_heading=section_heading,
                    year=year,
                    text=text,
                    char_count=char_count,
                    parent_chunk_id=parent_chunk_id,
                    extraction_method=extraction_method,
                    citation_count=int(
                        paper_meta.get("citation_count", 0) or 0
                    ),
                    influential_citation_count=int(
                        paper_meta.get("influential_citation_count", 0) or 0
                    ),
                    dense_score=dense_score_map.get(chunk_id, 0.0),
                    bm25_score=bm25_score_map.get(chunk_id, 0.0),
                    rrf_score=rrf_score,
                    s2_tldr=paper_meta.get("s2_tldr", "") or ""
                ))
            # Optionally supplement with abstract chunks
            if include_abstracts:
                seen_arxiv = {r.arxiv_id for r in retrieved}
                for arxiv_id in seen_arxiv:
                    abstract = self._fetch_abstract_for_paper(conn, arxiv_id)
                    if abstract:
                        retrieved.append(abstract)
        finally:
            conn.close()

        return retrieved


if __name__ == "__main__":
    retriever = HybridRetriever()

    test_queries = [
        "how does the attention mechanism work in transformers",
        "what are the trade-offs between LoRA and full fine-tuning",
        "how did BERT improve on previous language models",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        results = retriever.retrieve(query, top_k=3, include_abstracts=True)

        for i, chunk in enumerate(results):
            print(f"\n[{i+1}] {chunk.title[:55]}")
            print(f"     Section: {chunk.section_heading[:50]}")
            print(f"     Year: {chunk.year} | Citations: {chunk.citation_count}")
            print(f"     RRF: {chunk.rrf_score:.4f} | Dense: {chunk.dense_score:.4f} | BM25: {chunk.bm25_score:.4f}")
            print(f"     Text ({chunk.char_count} chars): {chunk.text[:150]}")
