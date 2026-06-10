import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHUNKS_DIR = BASE_DIR / "data/chunks"
EMBEDDINGS_DIR = BASE_DIR / "data/embeddings"

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 64  # optimal for RTX 5070 8GB
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model() -> SentenceTransformer:
    print(f"Loading {EMBED_MODEL} on {DEVICE}...")
    model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def embed_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
    cache_file: Path,
    batch_size: int = BATCH_SIZE
) -> dict[str, list[float]]:
    """
    Embed a list of chunks and cache to disk.
    Returns dict of chunk_id -> embedding.
    """

    # Load existing cache if resuming
    if cache_file.exists():
        print(f"Loading cached embeddings from {cache_file.name}...")
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        print(f"  Cached: {len(cached)} embeddings")
    else:
        cached = {}

    # Filter chunks that need embedding
    to_embed = [c for c in chunks if c["chunk_id"] not in cached]
    print(f"  Need to embed: {len(to_embed)} chunks")

    if not to_embed:
        print("  All chunks already cached")
        return cached

    # BGE models need a query prefix for retrieval
    # For document embedding (not query), use empty prefix
    texts = [c["text"] for c in to_embed]
    chunk_ids = [c["chunk_id"] for c in to_embed]

    embeddings = {}

    for i in tqdm(
        range(0, len(texts), batch_size),
        desc="Embedding batches"
    ):
        batch_texts = texts[i:i + batch_size]
        batch_ids = chunk_ids[i:i + batch_size]

        # Encode batch
        batch_embeddings = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # cosine similarity
            convert_to_numpy=True
        )

        for chunk_id, embedding in zip(batch_ids, batch_embeddings):
            embeddings[chunk_id] = embedding.tolist()

        # Save cache every 10 batches
        if (i // batch_size) % 10 == 0:
            combined = {**cached, **embeddings}
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(combined, f)

    # Final cache save
    combined = {**cached, **embeddings}
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(combined, f)

    print(f"  Embeddings cached to {cache_file.name}")
    return combined


def run_embedding(
    chunks_dir: Path = CHUNKS_DIR,
    embeddings_dir: Path = EMBEDDINGS_DIR,
    batch_size: int = BATCH_SIZE
):
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Load model once
    model = load_model()

    # Embed child chunks
    print(f"\n--- Embedding child chunks ---")
    with open(chunks_dir / "chunks_child.json", encoding="utf-8") as f:
        child_chunks = json.load(f)
    print(f"Total child chunks: {len(child_chunks)}")

    child_embeddings = embed_chunks(
        chunks=child_chunks,
        model=model,
        cache_file=embeddings_dir / "embeddings_child.json",
        batch_size=batch_size
    )

    # Embed abstract chunks
    print(f"\n--- Embedding abstract chunks ---")
    with open(chunks_dir / "chunks_abstract.json", encoding="utf-8") as f:
        abstract_chunks = json.load(f)
    print(f"Total abstract chunks: {len(abstract_chunks)}")

    abstract_embeddings = embed_chunks(
        chunks=abstract_chunks,
        model=model,
        cache_file=embeddings_dir / "embeddings_abstract.json",
        batch_size=batch_size
    )

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Embedding complete:")
    print(f"  Child embeddings:    {len(child_embeddings)}")
    print(f"  Abstract embeddings: {len(abstract_embeddings)}")
    print(f"  Total vectors:       {len(child_embeddings) + len(abstract_embeddings)}")
    print(f"  Cached to:           {embeddings_dir}")
    print(f"{'=' * 50}")

    return child_embeddings, abstract_embeddings


if __name__ == "__main__":
    child_embeddings, abstract_embeddings = run_embedding()