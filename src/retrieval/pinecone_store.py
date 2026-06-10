import json
import os
import sqlite3
import time
from pathlib import Path
from tqdm import tqdm
from pinecone import Pinecone
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

EMBEDDINGS_DIR = BASE_DIR / "data/embeddings"
CHUNKS_DIR = BASE_DIR / "data/chunks"
ENRICHED_FILE = BASE_DIR / "data/raw/papers_enriched.json"
DB_PATH = BASE_DIR / "data/paperpilot.db"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "paperpilot")
UPSERT_BATCH_SIZE = 100
MAX_BATCH_RETRIES = 3


# --- Preflight checks ---

def preflight_checks():
    """Validate config before doing anything."""
    errors = []

    if not PINECONE_API_KEY:
        errors.append("PINECONE_API_KEY not found in .env")

    if not PINECONE_INDEX:
        errors.append("PINECONE_INDEX_NAME not found in .env")

    if not EMBEDDINGS_DIR.exists():
        errors.append(f"Embeddings directory not found: {EMBEDDINGS_DIR}")

    if not (EMBEDDINGS_DIR / "embeddings_child.json").exists():
        errors.append("embeddings_child.json not found — run embedder.py first")

    if not (EMBEDDINGS_DIR / "embeddings_abstract.json").exists():
        errors.append("embeddings_abstract.json not found — run embedder.py first")

    if not ENRICHED_FILE.exists():
        errors.append("papers_enriched.json not found — run semantic_scholar.py first")

    if errors:
        print(" Preflight checks failed:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("[OKJ] Preflight checks passed")


# --- Safe helpers ---

def safe_int(value, fallback: int = 0) -> int:
    """Safely parse an integer with fallback."""
    try:
        if value is None or value == "":
            return fallback
        return int(str(value)[:4])
    except (ValueError, TypeError):
        return fallback


def safe_year(chunk: dict) -> int:
    """Extract year safely from chunk or published date."""
    year = chunk.get("year")
    if year:
        return safe_int(year)
    published = chunk.get("published", "")
    if published and len(published) >= 4:
        return safe_int(published[:4])
    return 0


def sanitize_metadata_value(value):
    """Ensure Pinecone metadata value is a valid primitive/list type."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        # Pinecone allows list[str] metadata values.
        return [str(v) for v in value if v is not None]
    # Fallback for unexpected types (e.g., dict)
    return str(value)


def sanitize_metadata(metadata: dict) -> dict:
    """Remove/convert unsupported metadata values before Pinecone upsert."""
    return {k: sanitize_metadata_value(v) for k, v in metadata.items()}


def get_index_stats(index) -> dict:
    """Safely get Pinecone index stats."""
    try:
        stats = index.describe_index_stats()
        # Handle both attribute and dict access
        if hasattr(stats, "total_vector_count"):
            return {"total_vector_count": stats.total_vector_count}
        elif isinstance(stats, dict):
            return {"total_vector_count": stats.get("total_vector_count", 0)}
        else:
            return {"total_vector_count": str(stats)}
    except Exception as e:
        return {"total_vector_count": f"error: {e}"}


# --- Database ---

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection):
    """Initialize SQLite tables."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS upserted_chunks (
            chunk_id TEXT NOT NULL,
            arxiv_id TEXT,
            chunk_type TEXT,
            upserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            pinecone_index TEXT NOT NULL,
            PRIMARY KEY (chunk_id, pinecone_index)
        );

        CREATE TABLE IF NOT EXISTS failed_chunks (
            chunk_id TEXT,
            arxiv_id TEXT,
            pinecone_index TEXT,
            error TEXT,
            failed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS parent_chunks (
            chunk_id TEXT PRIMARY KEY,
            arxiv_id TEXT,
            title TEXT,
            section TEXT,
            section_heading TEXT,
            year INTEGER,
            text TEXT,
            char_count INTEGER,
            extraction_method TEXT,
            parent_chunk_id TEXT
        );

        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT,
            published TEXT,
            year INTEGER,
            categories TEXT,
            primary_category TEXT,
            citation_count INTEGER DEFAULT 0,
            influential_citation_count INTEGER DEFAULT 0,
            s2_tldr TEXT,
            abstract TEXT
        );
    """)
    conn.commit()


# --- Data loading ---

def load_enriched_map() -> dict:
    """Load enrichment data keyed by arxiv_id."""
    with open(ENRICHED_FILE, encoding="utf-8") as f:
        enriched = json.load(f)
    return {p["arxiv_id"]: p for p in enriched}


# --- SQLite storage ---

def store_parent_chunks(conn: sqlite3.Connection):
    """Store all parent chunks in SQLite."""
    with open(CHUNKS_DIR / "chunks_parent.json", encoding="utf-8") as f:
        parents = json.load(f)

    cursor = conn.cursor()
    inserted = 0
    skipped = 0

    for chunk in tqdm(parents, desc="Storing parent chunks"):
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO parent_chunks
                (chunk_id, arxiv_id, title, section, section_heading,
                 year, text, char_count, extraction_method, parent_chunk_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk["chunk_id"],
                chunk["arxiv_id"],
                chunk.get("title", ""),
                chunk.get("section", ""),
                chunk.get("section_heading", ""),
                safe_year(chunk),
                chunk["text"],
                chunk.get("char_count", 0),
                chunk.get("extraction_method", ""),
                chunk.get("parent_chunk_id", "")
            ))
            if cursor.rowcount > 0:
                inserted += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"Error inserting {chunk['chunk_id']}: {e}")

    conn.commit()
    print(f"  Parent chunks — inserted: {inserted}, skipped: {skipped}")


def store_papers(conn: sqlite3.Connection, enriched_map: dict):
    """Store paper metadata in SQLite."""
    cursor = conn.cursor()
    inserted = 0
    skipped = 0

    for arxiv_id, paper in tqdm(enriched_map.items(), desc="Storing papers"):
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO papers
                (arxiv_id, title, authors, published, year,
                 categories, primary_category, citation_count,
                 influential_citation_count, s2_tldr, abstract)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                arxiv_id,
                paper.get("title", ""),
                json.dumps(paper.get("authors", [])),
                paper.get("published", ""),
                safe_int(
                    paper.get("year",
                    paper.get("published", "0")[:4])
                ),
                json.dumps(paper.get("categories", [])),
                paper.get("primary_category", ""),
                safe_int(paper.get("citation_count", 0)),
                safe_int(paper.get("influential_citation_count", 0)),
                paper.get("s2_tldr", ""),
                paper.get("abstract", "")
            ))
            if cursor.rowcount > 0:
                inserted += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"Error inserting paper {arxiv_id}: {e}")

    conn.commit()
    print(f"  Papers — inserted: {inserted}, skipped: {skipped}")


# --- Pinecone upsert ---

def build_pinecone_metadata(
    chunk: dict,
    enriched_map: dict
) -> dict:
    """Build lightweight Pinecone metadata for a chunk."""
    arxiv_id = chunk["arxiv_id"]
    enriched = enriched_map.get(arxiv_id, {})

    metadata = {
        "chunk_id": chunk["chunk_id"],
        "arxiv_id": arxiv_id,
        "title": chunk.get("title", "")[:200],
        "year": safe_year(chunk),
        "section": chunk.get("section", "")[:100],
        "section_heading": chunk.get("section_heading", "")[:100],
        "chunk_type": chunk.get("chunk_type", "child"),
        "parent_chunk_id": chunk.get("parent_chunk_id", ""),
        "extraction_method": chunk.get("extraction_method", ""),
        "citation_count": safe_int(enriched.get("citation_count", 0)),
        "influential_citation_count": safe_int(
            enriched.get("influential_citation_count", 0)
        ),
        "primary_category": chunk.get("primary_category", ""),
    }
    return sanitize_metadata(metadata)


def upsert_to_pinecone(
    chunks: list[dict],
    embeddings: dict[str, list],
    enriched_map: dict,
    index,
    conn: sqlite3.Connection,
    chunk_type: str = "child"
):
    """Upsert chunks to Pinecone with retry and failure tracking."""
    cursor = conn.cursor()

    # Get already upserted IDs — filtered by current index
    cursor.execute(
        "SELECT chunk_id FROM upserted_chunks WHERE pinecone_index = ?",
        (PINECONE_INDEX,)
    )
    already_upserted = {row[0] for row in cursor.fetchall()}
    print(f"  Already upserted to '{PINECONE_INDEX}': {len(already_upserted)}")

    # Filter to only new chunks that have embeddings
    new_chunks = [
        c for c in chunks
        if c["chunk_id"] not in already_upserted
        and c["chunk_id"] in embeddings
    ]
    print(f"  New chunks to upsert: {len(new_chunks)}")

    if not new_chunks:
        print("  Nothing to upsert")
        return

    upserted_count = 0
    failed_count = 0

    for i in tqdm(
        range(0, len(new_chunks), UPSERT_BATCH_SIZE),
        desc=f"Upserting {chunk_type} chunks"
    ):
        batch = new_chunks[i:i + UPSERT_BATCH_SIZE]

        vectors = []
        for chunk in batch:
            chunk_id = chunk["chunk_id"]
            embedding = embeddings[chunk_id]
            metadata = build_pinecone_metadata(chunk, enriched_map)
            vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": metadata
            })

        # Retry loop per batch
        success = False
        last_error = None

        for attempt in range(MAX_BATCH_RETRIES):
            try:
                index.upsert(vectors=vectors)

                # Record successful upserts in SQLite
                cursor.executemany("""
                    INSERT OR IGNORE INTO upserted_chunks
                    (chunk_id, arxiv_id, chunk_type, pinecone_index)
                    VALUES (?, ?, ?, ?)
                """, [
                    (c["chunk_id"], c["arxiv_id"], chunk_type, PINECONE_INDEX)
                    for c in batch
                ])
                conn.commit()
                upserted_count += len(batch)
                success = True
                break

            except Exception as e:
                last_error = str(e)
                wait = 5 * (attempt + 1)
                print(f"\n  Batch {i} attempt {attempt+1} failed: {e}")
                if attempt < MAX_BATCH_RETRIES - 1:
                    print(f"  Retrying in {wait}s...")
                    time.sleep(wait)

        if not success:
            # Record failed chunks
            failed_count += len(batch)
            print(f"\n  Batch {i} failed after {MAX_BATCH_RETRIES} attempts")
            cursor.executemany("""
                INSERT INTO failed_chunks
                (chunk_id, arxiv_id, pinecone_index, error)
                VALUES (?, ?, ?, ?)
            """, [
                (c["chunk_id"], c["arxiv_id"], PINECONE_INDEX, last_error)
                for c in batch
            ])
            conn.commit()

    print(f"  Upserted: {upserted_count} | Failed: {failed_count}")

    if failed_count > 0:
        print(f"  !!  {failed_count} chunks failed — check failed_chunks table in SQLite")


# --- Main ---

def run_store(
    embeddings_dir: Path = EMBEDDINGS_DIR,
    chunks_dir: Path = CHUNKS_DIR
):
    print("=" * 60)
    print("PaperPilot — Pinecone Store")
    print("=" * 60)

    # Preflight
    preflight_checks()

    # Connect to Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    print(f"Connected to Pinecone index: {PINECONE_INDEX}")

    # Connect to SQLite
    conn = get_db_connection()
    init_db(conn)
    print(f"SQLite initialized: {DB_PATH}")

    # Load enrichment data
    enriched_map = load_enriched_map()
    print(f"Enrichment loaded: {len(enriched_map)} papers")

    # Store in SQLite
    print("\n--- Storing in SQLite ---")
    store_papers(conn, enriched_map)
    store_parent_chunks(conn)

    # Load embeddings
    print("\n--- Loading embeddings ---")
    with open(embeddings_dir / "embeddings_child.json", encoding="utf-8") as f:
        child_embeddings = json.load(f)
    print(f"  Child embeddings: {len(child_embeddings)}")

    with open(embeddings_dir / "embeddings_abstract.json", encoding="utf-8") as f:
        abstract_embeddings = json.load(f)
    print(f"  Abstract embeddings: {len(abstract_embeddings)}")

    # Load chunks
    with open(chunks_dir / "chunks_child.json", encoding="utf-8") as f:
        child_chunks = json.load(f)

    with open(chunks_dir / "chunks_abstract.json", encoding="utf-8") as f:
        abstract_chunks = json.load(f)

    # Upsert to Pinecone
    print("\n--- Upserting to Pinecone ---")
    print("Child chunks:")
    upsert_to_pinecone(
        chunks=child_chunks,
        embeddings=child_embeddings,
        enriched_map=enriched_map,
        index=index,
        conn=conn,
        chunk_type="child"
    )

    print("\nAbstract chunks:")
    upsert_to_pinecone(
        chunks=abstract_chunks,
        embeddings=abstract_embeddings,
        enriched_map=enriched_map,
        index=index,
        conn=conn,
        chunk_type="abstract"
    )

    # Final stats
    stats = get_index_stats(index)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM parent_chunks")
    parent_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM papers")
    paper_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM failed_chunks")
    failed_count = cursor.fetchone()[0]

    print(f"\n{'=' * 60}")
    print("Store Complete")
    print(f"{'=' * 60}")
    print(f"  Pinecone vectors: {stats['total_vector_count']}")
    print(f"  SQLite parents:   {parent_count}")
    print(f"  SQLite papers:    {paper_count}")
    print(f"  Failed chunks:    {failed_count}")
    print(f"{'=' * 60}")

    conn.close()
    return stats


if __name__ == "__main__":
    run_store()
