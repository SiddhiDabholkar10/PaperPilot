import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EMBEDDINGS_DIR = BASE_DIR / "data/embeddings"
CHUNKS_DIR = BASE_DIR / "data/chunks"

print("=== Embedding Sanity Check ===\n")

# Load embeddings
print("Loading embedding files...")
with open(EMBEDDINGS_DIR / "embeddings_child.json", encoding="utf-8") as f:
    child_emb = json.load(f)

with open(EMBEDDINGS_DIR / "embeddings_abstract.json", encoding="utf-8") as f:
    abstract_emb = json.load(f)

# Load chunks for cross-reference
with open(CHUNKS_DIR / "chunks_child.json", encoding="utf-8") as f:
    child_chunks = json.load(f)

with open(CHUNKS_DIR / "chunks_abstract.json", encoding="utf-8") as f:
    abstract_chunks = json.load(f)

# 1. Count check
print("=== Count Check ===")
print(f"  Child chunks:      {len(child_chunks)}")
print(f"  Child embeddings:  {len(child_emb)}")
print(f"  Match: {len(child_chunks) == len(child_emb)}")
print(f"  Abstract chunks:      {len(abstract_chunks)}")
print(f"  Abstract embeddings:  {len(abstract_emb)}")
print(f"  Match: {len(abstract_chunks) == len(abstract_emb)}")

# 2. Dimension check
print("\n=== Dimension Check ===")
sample_child = list(child_emb.values())[0]
sample_abstract = list(abstract_emb.values())[0]
print(f"  Child embedding dim:    {len(sample_child)}")
print(f"  Abstract embedding dim: {len(sample_abstract)}")
print(f"  Expected: 1024")

# 3. Normalization check
print("\n=== Normalization Check ===")
vec = np.array(sample_child)
norm = np.linalg.norm(vec)
print(f"  Sample vector norm: {norm:.6f} (should be ~1.0)")

# 4. Missing embeddings check
print("\n=== Missing Embeddings Check ===")
child_chunk_ids = {c["chunk_id"] for c in child_chunks}
embedded_ids = set(child_emb.keys())
missing = child_chunk_ids - embedded_ids
extra = embedded_ids - child_chunk_ids
print(f"  Missing embeddings: {len(missing)}")
print(f"  Extra embeddings:   {len(extra)}")
if missing:
    print(f"  Sample missing: {list(missing)[:3]}")

# 5. Similarity sanity check
print("\n=== Similarity Sanity Check ===")
# Find two chunks from the same paper — should be more similar
# than two chunks from different papers
child_chunk_map = {c["chunk_id"]: c for c in child_chunks}

# Get two chunks from same paper
same_paper_chunks = [
    c for c in child_chunks
    if c["arxiv_id"] == "1706.03762"
][:2]

# Get two chunks from different papers
diff_paper_chunks = [
    child_chunks[0],
    next(c for c in child_chunks if c["arxiv_id"] != child_chunks[0]["arxiv_id"])
]

if len(same_paper_chunks) >= 2:
    v1 = np.array(child_emb[same_paper_chunks[0]["chunk_id"]])
    v2 = np.array(child_emb[same_paper_chunks[1]["chunk_id"]])
    same_sim = np.dot(v1, v2)
    print(f"  Same paper similarity:      {same_sim:.4f}")

v3 = np.array(child_emb[diff_paper_chunks[0]["chunk_id"]])
v4 = np.array(child_emb[diff_paper_chunks[1]["chunk_id"]])
diff_sim = np.dot(v3, v4)
print(f"  Different paper similarity: {diff_sim:.4f}")
print(f"  Same > Different: {same_sim > diff_sim if len(same_paper_chunks) >= 2 else 'N/A'}")

# 6. Query similarity test
print("\n=== Query Similarity Test ===")
from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer(
    "BAAI/bge-large-en-v1.5",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# BGE uses a query prefix for queries
query = "BGE query: what is the attention mechanism in transformers?"
query_emb = model.encode(query, normalize_embeddings=True)

# Find top 3 most similar child chunks
scores = {}
for chunk_id, emb in child_emb.items():
    score = np.dot(query_emb, np.array(emb))
    scores[chunk_id] = score

top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
print(f"  Query: '{query[:60]}'")
print(f"  Top 3 results:")
for chunk_id, score in top3:
    chunk = child_chunk_map.get(chunk_id, {})
    print(f"    [{score:.4f}] {chunk.get('title', '')[:50]}")
    print(f"             {chunk.get('text', '')[:80]}")

print("\n[OKJ] Embedding check complete")