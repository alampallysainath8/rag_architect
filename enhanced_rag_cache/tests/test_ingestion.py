"""
Lightweight chunking checks for the ingestion pipeline.

These tests exercise the unified structure_recursive_chunk, which now handles
structure-aware splitting AND table/image block preservation in one pass.

Run directly with `python tests/test_ingestion.py` or with `pytest`.
"""

from pathlib import Path
import sys
import os

# Ensure the repo package root is on sys.path so `src` imports work when
# running this script directly (e.g. `python tests/test_ingestion.py`).
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.chunking.structure_recursive import (
    extract_special_blocks,
    structure_recursive_chunk,
)
from src.chunking.parent_child import parent_child_chunk
from src.utils.pdf_to_markdown import file_to_text
from src.utils.image_enricher import enrich_markdown_images


def test_chunking_preserves_tables_and_images():
    # Sample markdown with heading, a table and an image-description block
    md = """
# Revenue

Revenue grew by 15%.

| Quarter | Revenue |
|---------|---------|
| Q1      | 100     |
| Q2      | 115     |

**[Image 1 Description]:**

Figure: Revenue trend chart.

Additional narrative text under the same section.
"""

    chunks = structure_recursive_chunk(md, doc_id="sample.md", source="sample.md")

    n_table = sum(1 for c in chunks if c.chunk_type == "table")
    n_image = sum(1 for c in chunks if c.chunk_type == "image")
    n_text  = sum(1 for c in chunks if c.chunk_type == "text")

    assert n_table >= 1, f"Expected at least 1 table chunk, got {n_table}"
    assert n_image >= 1, f"Expected at least 1 image chunk, got {n_image}"
    assert n_text  >= 1, f"Expected at least 1 text chunk, got {n_text}"


def test_parent_child_basic():
    text = """
# Intro

This is an intro paragraph that should be part of the first parent.

# Details

Here are details that may be split into children.

# Conclusion

Short conclusion.
"""
    parents, children = parent_child_chunk(text, doc_id="doc.txt", source="doc.txt")
    assert len(parents) > 0
    assert len(children) > 0


def main():
    # Allow a file path to be supplied as the first CLI arg or via FILEPATH env
    filepath = None
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = os.environ.get("FILEPATH")

    # Default to the sample PDF in the repo if not provided
    if not filepath:
        filepath = str(Path(__file__).parents[1] / "docs" / "2602.03442v1_rag_pdf-1-9.pdf")

    p = Path(filepath)
    if not p.exists():
        print(f"File not found: {filepath}")
        sys.exit(4)

    print(f"Using file: {filepath}")

    try:
        # Convert PDF → Markdown with embedded images when applicable
        md = file_to_text(str(p), embed_images=True) if p.suffix.lower() == ".pdf" else file_to_text(str(p))

        # Enrich base64 images via Groq vision if API key is present
        image_blocks: list = []
        if os.environ.get("GROQ_API_KEY"):
            print("GROQ_API_KEY detected — enriching images via Groq vision (may take time)")
            try:
                md, image_blocks = enrich_markdown_images(md)
            except Exception as e:
                print("Image enrichment failed (continuing without enrichment):", e)

        # ── Unified structure-recursive (table + image aware) ─────────────
        chunks = structure_recursive_chunk(
            md, doc_id=p.name, source=str(p),
            image_blocks=image_blocks or None,
        )
        n_text  = sum(1 for c in chunks if c.chunk_type == "text")
        n_table = sum(1 for c in chunks if c.chunk_type == "table")
        n_image = sum(1 for c in chunks if c.chunk_type == "image")
        print(
            f"Structure-Recursive chunks: total={len(chunks)}, "
            f"text={n_text}, table={n_table}, image={n_image}"
        )

        out_dir = Path(__file__).parent / "output"
        out_dir.mkdir(exist_ok=True)

        sr_file = out_dir / f"structure_recursive_{p.stem}.txt"
        with open(sr_file, "w", encoding="utf-8") as f:
            for i, c in enumerate(chunks, start=1):
                f.write("=" * 80 + "\n")
                f.write(f"CHUNK {i} | type={c.chunk_type}\n")
                f.write(f"heading_meta: {c.heading_meta}\n")
                f.write("-" * 40 + "\n")
                f.write(c.text or "")
                f.write("\n\n")
        print(f"Wrote structure-recursive chunks to: {sr_file}")

        print("All chunking checks completed successfully.")
        sys.exit(0)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Error while running chunking tests:", e)
        sys.exit(3)


if __name__ == "__main__":
    main()
