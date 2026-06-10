import json
import pickle
import sqlite3
import os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

print("=" * 60)
print("Retrieval Stack Sanity Check")
print("=" * 60)

# --- 1. Load BM25 ---
print("\n[1] BM25 Index")
with open(BASE_DIR / "data/index/bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)
with open(BASE_DIR / "data/index/bm25_chunk_ids.json", encoding="utf-8") as f:
    bm25_chunk_ids = json.load(f)
print(f"  Loaded: {len(bm25_chunk_ids)} documents")

# --- 2. Load Pinecone ---
print("\n[2] Pinecone")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "paperpilot"))
stats = index.describe_index_stats()
print(f"  Vectors: {stats.total_vector_count}")

# --- 3. Load SQLite ---
print("\n[3] SQLite")
conn = sqlite3.connect(BASE_DIR / "data/paperpilot.db")
conn.row_factory = sqlite3.Row
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM parent_chunks")
parent_count = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM papers")
paper_count = cursor.fetchone()[0]
print(f"  Parent chunks: {parent_count}")
print(f"  Papers: {paper_count}")

# --- 4. Load embedding model ---
print("\n[4] Embedding Model")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
print(f"  Loaded on {device}")

# --- 5. Run test query through full stack ---
print("\n[5] Full Stack Query Test")
query = "how does the attention mechanism work in transformers"
bge_query = f"BGE query: {query}"

# Embed query
query_emb = model.encode(bge_query, normalize_embeddings=True)

# Dense retrieval
dense_results = index.query(
    vector=query_emb.tolist(),
    top_k=5,
    include_metadata=True
)

print(f"\n  Query: '{query}'")
print(f"\n  Dense (Pinecone) top 5:")
for match in dense_results.matches:
    print(f"    [{match.score:.4f}] {match.metadata.get('title', '')[:45]}")
    print(f"             {match.metadata.get('section', '')[:45]}")

# BM25 retrieval
bm25_tokens = query.lower().split()
bm25_scores = bm25.get_scores(bm25_tokens)
top5_idx = np.argsort(bm25_scores)[::-1][:5]

print(f"\n  BM25 top 5:")
for idx in top5_idx:
    chunk_id = bm25_chunk_ids[idx]
    print(f"    [{bm25_scores[idx]:.4f}] {chunk_id}")

# --- 6. Parent lookup test ---
print("\n[6] Parent Lookup Test")
sample_child_id = dense_results.matches[0].metadata.get("parent_chunk_id", "")
if sample_child_id:
    cursor.execute(
        "SELECT chunk_id, title, section, char_count FROM parent_chunks WHERE chunk_id = ?",
        (sample_child_id,)
    )
    parent = cursor.fetchone()
    if parent:
        print(f"  Child matched parent: {parent['chunk_id']}")
        print(f"  Title: {parent['title'][:50]}")
        print(f"  Section: {parent['section']}")
        print(f"  Size: {parent['char_count']} chars")
    else:
        print(f"   ✕  Parent not found: {sample_child_id}")
else:
    print("  No parent_chunk_id in result (abstract chunk)")

# --- 7. Paper metadata lookup ---
print("\n[7] Paper Metadata Lookup")
sample_arxiv_id = dense_results.matches[0].metadata.get("arxiv_id", "")
cursor.execute(
    "SELECT title, citation_count, influential_citation_count, s2_tldr FROM papers WHERE arxiv_id = ?",
    (sample_arxiv_id,)
)
paper = cursor.fetchone()
if paper:
    print(f"  arxiv_id: {sample_arxiv_id}")
    print(f"  Title: {paper['title'][:50]}")
    print(f"  Citations: {paper['citation_count']}")
    print(f"  Influential: {paper['influential_citation_count']}")
    print(f"  TLDR: {(paper['s2_tldr'] or '')[:100]}")
else:
    print(f"  ✕ Paper not found: {sample_arxiv_id}")

conn.close()
print("\n✓ Retrieval stack sanity check complete")