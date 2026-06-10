# sanity_check.py
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

checks = {
    "chunks_abstract": BASE_DIR / "data/chunks/chunks_abstract.json",
    "chunks_parent": BASE_DIR / "data/chunks/chunks_parent.json",
    "chunks_child": BASE_DIR / "data/chunks/chunks_child.json",
    "papers_metadata": BASE_DIR / "data/raw/papers_metadata.json",
    "papers_enriched": BASE_DIR / "data/raw/papers_enriched.json",
    "paper_references": BASE_DIR / "data/raw/paper_references.json",
    "manifest": BASE_DIR / "data/manifest.json",
}

print("=== File Checks ===")
for name, path in checks.items():
    exists = path.exists()
    size_kb = path.stat().st_size / 1024 if exists else 0
    print(f"  {'[OK]' if exists else '[FAIL]'} {name}: {size_kb:.1f} KB")

print("\n=== Chunk Counts ===")
for chunk_type in ["abstract", "parent", "child"]:
    path = BASE_DIR / f"data/chunks/chunks_{chunk_type}.json"
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  {chunk_type}: {len(chunks)} chunks")

print("\n=== Enrichment Coverage ===")
with open(BASE_DIR / "data/raw/papers_enriched.json", encoding="utf-8") as f:
    enriched = json.load(f)

found = [p for p in enriched if p.get("s2_status") == "found"]
with_tldr = [p for p in found if p.get("s2_tldr")]
with_refs = [p for p in found if p.get("arxiv_reference_count", 0) > 0]

print(f"  Total papers: {len(enriched)}")
print(f"  Found on S2:  {len(found)}")
print(f"  With TLDR:    {len(with_tldr)}")
print(f"  With refs:    {len(with_refs)}")

print("\n=== Sample Child Chunk ===")
with open(BASE_DIR / "data/chunks/chunks_child.json", encoding="utf-8") as f:
    children = json.load(f)

sample = children[100]
print(f"  chunk_id: {sample['chunk_id']}")
print(f"  arxiv_id: {sample['arxiv_id']}")
print(f"  section:  {sample['section']}")
print(f"  year:     {sample['year']}")
print(f"  chars:    {sample['char_count']}")
print(f"  text:     {sample['text'][:100]}")

print("\n=== ID Format Check ===")
# Verify all child chunks have correct ID format
malformed = [
    c for c in children
    if "::" not in c["chunk_id"]
]
print(f"  Malformed IDs: {len(malformed)}")

print("\n=== Parent-Child Integrity ===")
with open(BASE_DIR / "data/chunks/chunks_parent.json", encoding="utf-8") as f:
    parents = json.load(f)

parent_ids = {p["chunk_id"] for p in parents}
orphans = [
    c for c in children
    if c.get("parent_chunk_id") not in parent_ids
]
print(f"  Total parents: {len(parents)}")
print(f"  Total children: {len(children)}")
print(f"  Orphan children: {len(orphans)}")

print("\n=== Enrichment-Chunk Join Test ===")
# Verify we can join enriched metadata to chunks
enriched_map = {p["arxiv_id"]: p for p in enriched}
chunk_ids = {c["arxiv_id"] for c in children}
enrichment_coverage = sum(
    1 for aid in chunk_ids
    if aid in enriched_map
)
print(f"  Unique papers in chunks: {len(chunk_ids)}")
print(f"  Papers with enrichment:  {enrichment_coverage}")
print(f"  Coverage: {enrichment_coverage/len(chunk_ids)*100:.1f}%")

print("\n=== GPU Check ===")
import torch
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n[OK] Sanity check complete")