import json
import pickle
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from retriever_utils import tokenize

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHUNKS_DIR = BASE_DIR / "data/chunks"
INDEX_DIR = BASE_DIR / "data/index"


def build_bm25_index(
    chunks_dir: Path = CHUNKS_DIR,
    index_dir: Path = INDEX_DIR
):
    index_dir.mkdir(parents=True, exist_ok=True)

    print("Loading child chunks...")
    with open(chunks_dir / "chunks_child.json", encoding="utf-8") as f:
        child_chunks = json.load(f)

    print(f"Building BM25 index over {len(child_chunks)} chunks...")

    corpus = []
    chunk_ids = []

    for chunk in tqdm(child_chunks, desc="Tokenizing"):
        tokens = tokenize(chunk["text"])
        corpus.append(tokens)
        chunk_ids.append(chunk["chunk_id"])

    print("Building BM25 index...")
    bm25 = BM25Okapi(corpus)

    index_file = index_dir / "bm25_index.pkl"
    ids_file = index_dir / "bm25_chunk_ids.json"

    with open(index_file, "wb") as f:
        pickle.dump(bm25, f)

    with open(ids_file, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f)

    print(f"BM25 index saved → {index_file}")
    print(f"Chunk IDs saved  → {ids_file}")
    print(f"Corpus size: {len(corpus)} documents")

    return bm25, chunk_ids


def load_bm25_index(
    index_dir: Path = INDEX_DIR
) -> tuple:
    """Load BM25 index and chunk IDs from disk."""
    with open(index_dir / "bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)

    with open(index_dir / "bm25_chunk_ids.json", encoding="utf-8") as f:
        chunk_ids = json.load(f)

    return bm25, chunk_ids


if __name__ == "__main__":
    bm25, chunk_ids = build_bm25_index()

    print("\nTesting BM25...")
    query_tokens = tokenize("attention mechanism transformer")
    scores = bm25.get_scores(query_tokens)
    top5_idx = np.argsort(scores)[::-1][:5]

    print(f"Query: 'attention mechanism transformer'")
    print(f"Top 5 BM25 results:")
    for idx in top5_idx:
        print(f"  [{scores[idx]:.4f}] {chunk_ids[idx]}")