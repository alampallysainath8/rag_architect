"""
Generation Layer Tests  (2 test cases)
========================================

Case 1 — format_sources deduplication
  The same (source, section) pair appearing in multiple chunks must produce
  only one entry in the sources list.

Case 2 — generate_answer with empty chunk list
  When no chunks are provided the function must return the fallback message
  without calling the OpenAI API at all.
"""

import pytest
from unittest.mock import MagicMock, patch


# ── Case 1: format_sources deduplication ─────────────────────────────────────

def test_format_sources_deduplication():
    """
    Two chunks from the same source + section must yield exactly one source entry.
    A third chunk from a different section must add a second entry.
    """
    from src.generation.generator import format_sources

    chunks = [
        # Two chunks — same source, same h1/h2 heading
        {
            "score": 0.92, "source": "report.pdf", "doc_id": "report.pdf",
            "chunk_text": "Revenue grew 15%.",
            "h1": "Financial Overview", "h2": "Revenue", "h3": "", "h4": "",
        },
        {
            "score": 0.89, "source": "report.pdf", "doc_id": "report.pdf",
            "chunk_text": "Product revenue: $300M.",
            "h1": "Financial Overview", "h2": "Revenue", "h3": "", "h4": "",
        },
        # Different section — must be a separate entry
        {
            "score": 0.75, "source": "report.pdf", "doc_id": "report.pdf",
            "chunk_text": "Costs remained flat.",
            "h1": "Financial Overview", "h2": "Costs", "h3": "", "h4": "",
        },
    ]

    sources = format_sources(chunks)

    assert len(sources) == 2, (
        f"Expected 2 unique sources (two sections), got {len(sources)}"
    )

    sections = [s["section"] for s in sources]
    assert "Financial Overview > Revenue" in sections
    assert "Financial Overview > Costs"  in sections


# ── Case 2: generate_answer with empty chunks ─────────────────────────────────

def test_generate_answer_returns_fallback_for_empty_chunks():
    """
    When chunks=[] the LLM must NOT be called and the predefined
    fallback message must be returned verbatim.
    """
    with patch("src.generation.generator._client") as mock_openai:
        from src.generation.generator import generate_answer

        answer = generate_answer("What is the revenue?", chunks=[])

    # OpenAI client should not have been invoked
    mock_openai.chat.completions.create.assert_not_called()

    assert "don't have enough information" in answer.lower() or \
           "i don" in answer.lower(), (
        f"Expected fallback message, got: {answer!r}"
    )
