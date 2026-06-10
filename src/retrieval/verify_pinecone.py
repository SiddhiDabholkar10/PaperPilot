# verify_pinecone.py
import os
from pinecone import Pinecone
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "paperpilot"))

stats = index.describe_index_stats()
print(f"Total vectors: {stats.total_vector_count}")
print(f"Dimension: {stats.dimension}")
print(f"Index fullness: {stats.index_fullness}")

# Test a sample query
import json
from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer(
    "BAAI/bge-large-en-v1.5",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

query = "BGE query: what is the attention mechanism in transformers?"
query_emb = model.encode(query, normalize_embeddings=True).tolist()

results = index.query(
    vector=query_emb,
    top_k=5,
    include_metadata=True
)

print(f"\nTest query: '{query[:60]}'")
print(f"Top 5 results:")
for match in results.matches:
    print(f"  [{match.score:.4f}] {match.metadata.get('title', '')[:50]}")
    print(f"           section: {match.metadata.get('section', '')[:50]}")
    print(f"           year: {match.metadata.get('year')}")
    print(f"           citations: {match.metadata.get('citation_count')}")