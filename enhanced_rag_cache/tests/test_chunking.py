"""
Chunking Layer Tests  (2 test cases)
=====================================

Case 1 — parent_child_chunk
  Verify the two-level hierarchy: parent IDs, child IDs, parent_id
  cross-reference, and that every child carries required Pinecone fields.

Case 2 — structure_recursive_chunk
  Verify that H1/H2 heading metadata is extracted correctly and that
  oversized sections produce sub-indexed chunks (sub_index > 0).
"""

import pytest

# ---------------------------------------------------------------------------
# The chunkers use only stdlib + langchain (no Redis/Pinecone/OpenAI calls)
# so we can import them directly without heavy mocking.
# ---------------------------------------------------------------------------
from src.chunking.parent_child import parent_child_chunk
from src.chunking.structure_recursive import structure_recursive_chunk


# ── Helpers ──────────────────────────────────────────────────────────────────

SIMPLE_TEXT = (
    "# Financial Overview\n\n"
    "Revenue increased significantly over the year. "
    "Total income reached record levels.\n\n"
    "## Revenue Breakdown\n\n"
    "Product revenue: $500M. Service revenue: $200M.\n\n"
    "## Cost Analysis\n\n"
    "Operating costs remained flat. Margin improved by 3pp. "
    "Central administration costs were reduced through automation.\n\n"
    "# Risk Factors\n\n"
    "Market volatility poses the primary risk to near-term guidance. "
    "Currency headwinds may impact international revenues adversely."
)

# A small document that forces recursive splitting when section limit is tight
DENSE_SECTION = "# Section A\n\n" + ("word " * 400)   # ~2000 chars in one section


# ── Case 1: parent_child_chunk structure ─────────────────────────────────────

def test_parent_child_chunk_ids_and_metadata():
    """
    Parent IDs must follow '<doc_id>::parent_<n>' pattern.
    Child IDs must reference their parent and carry all required Pinecone fields.
    """
    doc_id = "report.pdf"
    parents, children = parent_child_chunk(
        SIMPLE_TEXT,
        doc_id=doc_id,
        source="report.pdf",
        parent_chunk_size=500,
        parent_overlap=50,
        child_chunk_size=150,
        child_overlap=20,
    )

    assert len(parents) > 0,  "Expected at least one parent chunk"
    assert len(children) > 0, "Expected at least one child chunk"

    # Validate parent ID format
    for i, p in enumerate(parents):
        assert p.parent_id == f"{doc_id}::parent_{i}", (
            f"Expected parent_id '{doc_id}::parent_{i}', got '{p.parent_id}'"
        )
        assert p.text, "Parent text must not be empty"
        rdict = p.to_redis_dict()
        assert "parent_id" in rdict and "text" in rdict, "Redis dict missing required keys"

    # Validate child → parent cross-reference and Pinecone record
    parent_ids = {p.parent_id for p in parents}
    for child in children:
        assert child.parent_id in parent_ids, (
            f"Child '{child.chunk_id}' has unknown parent_id '{child.parent_id}'"
        )
        record = child.to_pinecone_record()
        for field in ("_id", "chunk_text", "source", "doc_id", "parent_id", "level"):
            assert field in record, f"Pinecone record missing field '{field}'"
        assert record["level"] == "child"


# ── Case 2: structure_recursive heading metadata + sub-indexing ───────────────

def test_structure_recursive_heading_meta_and_sub_index():
    """
    H1/H2 headings must appear in chunk heading_meta.
    A dense single section must be recursively split (sub_index > 0 for at least one chunk).
    """
    doc_id = "dense_report.pdf"
    chunks = structure_recursive_chunk(
        DENSE_SECTION,
        doc_id=doc_id,
        source="dense_report.pdf",
        max_section_size=300,       # force recursive split
        recursive_chunk_size=200,
        recursive_overlap=30,
    )

    assert len(chunks) > 1, "Dense section should produce multiple chunks"

    # At least one chunk should have been sub-indexed (recursive split happened)
    sub_indexed = [c for c in chunks if c.sub_index > 0]
    assert sub_indexed, "Expected at least one recursively split sub-chunk (sub_index > 0)"

    # All records must be valid Pinecone dicts
    for chunk in chunks:
        record = chunk.to_pinecone_record()
        for field in ("_id", "chunk_text", "source", "doc_id", "strategy"):
            assert field in record, f"Record missing '{field}'"
        assert record["strategy"] == "structure_recursive"

    # Test heading metadata on SIMPLE_TEXT (has clear H1 and H2)
    rich_chunks = structure_recursive_chunk(
        SIMPLE_TEXT,
        doc_id="simple.pdf",
        source="simple.pdf",
        max_section_size=1200,
    )
    heading_chunks = [c for c in rich_chunks if c.heading_meta]
    assert heading_chunks, "Expected chunks with heading metadata from H1/H2 sections"
    # At least one chunk should have 'h1' key
    h1_chunks = [c for c in heading_chunks if "h1" in c.heading_meta]
    assert h1_chunks, "Expected h1 metadata from '# Financial Overview' header"
