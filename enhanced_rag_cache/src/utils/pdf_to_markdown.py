"""
PDF → Markdown converter using pymupdf4llm.

Bordered tables are extracted as markdown pipe tables (table_strategy="lines").
Borderless plain-text tables are handled downstream in the chunker.
"""

from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


def pdf_to_markdown(file_path: str, embed_images: bool = False) -> str:
    """
    Convert a PDF file to a Markdown string.

    Uses pymupdf4llm which produces LLM-friendly Markdown while preserving
    heading hierarchy, bold/italic, tables, and reasonable paragraph breaks.

    Args:
        file_path:     Absolute or relative path to a PDF file.
        embed_images:  If True, images are extracted and embedded as
                       base64 data-URLs directly in the Markdown.
                       Set to True for the pdf_rich ingestion strategy.

    Returns:
        Markdown text of the entire document.
    """
    try:
        import pymupdf4llm  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pymupdf4llm is required for PDF conversion. "
            "Install it with: pip install pymupdf4llm"
        ) from exc

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Converting PDF → Markdown: {path.name}  (embed_images={embed_images})")

    common = {"table_strategy": "lines"}

    if embed_images:
        md_text: str = pymupdf4llm.to_markdown(
            str(path), write_images=True, image_format="png",
            embed_images=True, **common,
        )
    else:
        md_text = pymupdf4llm.to_markdown(str(path), **common)

    logger.info(f"Converted {path.name} — {len(md_text):,} chars of Markdown")
    return md_text


def file_to_text(file_path: str, embed_images: bool = False) -> str:
    """
    Load any supported document and return its text/markdown content.

    Supported types:
      - .pdf  → converted via pymupdf4llm
      - .txt  → raw text
      - .md   → raw markdown

    Args:
        file_path:    Path to the document.
        embed_images: Passed through to pdf_to_markdown for PDF files.

    Returns:
        Text content of the document.
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return pdf_to_markdown(file_path, embed_images=embed_images)
    elif ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: .pdf, .txt, .md"
        )
