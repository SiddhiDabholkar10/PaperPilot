"""
Stable ID system for PaperPilot.

paper_id:        2401.12345
abstract_id:     2401.12345::abstract
parent_id:       2401.12345::parent::0004
child_id:        2401.12345::parent::0004::child::002
"""

def make_paper_id(arxiv_id: str) -> str:
    """Clean arxiv ID — strip version suffix if present."""
    return arxiv_id.split("v")[0]


def make_abstract_id(paper_id: str) -> str:
    return f"{paper_id}::abstract"


def make_parent_id(paper_id: str, parent_idx: int) -> str:
    return f"{paper_id}::parent::{parent_idx:04d}"


def make_child_id(parent_id: str, child_idx: int) -> str:
    return f"{parent_id}::child::{child_idx:04d}"


def parse_chunk_id(chunk_id: str) -> dict:
    """
    Parse a chunk ID back into its components.
    
    Returns dict with paper_id, chunk_type, parent_idx, child_idx.
    """
    parts = chunk_id.split("::")

    if len(parts) == 1:
        return {"paper_id": parts[0], "chunk_type": "paper"}

    if parts[1] == "abstract":
        return {"paper_id": parts[0], "chunk_type": "abstract"}

    # Check child BEFORE parent — child has more parts
    if "child" in parts:
        child_pos = parts.index("child")
        return {
            "paper_id": parts[0],
            "chunk_type": "child",
            "parent_idx": int(parts[2]),
            "child_idx": int(parts[child_pos + 1])
        }

    if parts[1] == "parent":
        return {
            "paper_id": parts[0],
            "chunk_type": "parent",
            "parent_idx": int(parts[2])
        }

    return {"paper_id": parts[0], "chunk_type": "unknown"}

def get_parent_id_from_child(child_id: str) -> str:
    """Extract parent_id from a child_id."""
    # 2401.12345::parent::0004::child::002
    # → 2401.12345::parent::0004
    parts = child_id.split("::")
    if len(parts) >= 3 and "child" in parts:
        child_pos = parts.index("child")
        return "::".join(parts[:child_pos])
    return None


if __name__ == "__main__":
    # Test
    paper_id = make_paper_id("2401.12345v2")
    abstract_id = make_abstract_id(paper_id)
    parent_id = make_parent_id(paper_id, 4)
    child_id = make_child_id(parent_id, 2)

    print(f"paper_id:    {paper_id}")
    print(f"abstract_id: {abstract_id}")
    print(f"parent_id:   {parent_id}")
    print(f"child_id:    {child_id}")

    print(f"\nParsed child: {parse_chunk_id(child_id)}")
    print(f"Parent from child: {get_parent_id_from_child(child_id)}")