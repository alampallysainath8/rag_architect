"""
Retrieval Layer Tests  (2 test cases)
=======================================

Case 1 — rerank_preloaded: parent context injection
  When a child chunk carries a parent_id and the parent exists in Redis,
  rerank_preloaded() must set context_text to the parent's full text.

Case 2 — retriever.search: hit dict structure normalisation
  Mocks the Pinecone search response and verifies that .search() returns
  correctly structured hit dicts with all expected keys.
"""

import pytest
from unittest.mock import MagicMock, patch


# ── Case 1: rerank_preloaded injects parent context ──────────────────────────

def test_rerank_preloaded_injects_parent_context():
    """
    If parent_id is set and get_parent() returns a parent dict,
    context_text must equal the parent's full text — not the child text.
    """
    child_chunk = {
        "id": "report.pdf::parent_0::child_0",
        "score": 0.95,
        "chunk_text": "Revenue increased in Q3.",
        "source": "report.pdf",
        "doc_id": "report.pdf",
        "parent_id": "report.pdf::parent_0",
        "context_text": "Revenue increased in Q3.",   # default: same as child
        "level": "child",
        "strategy": "parent_child",
        "h1": "", "h2": "", "h3": "", "h4": "",
    }

    parent_payload = {
        "parent_id": "report.pdf::parent_0",
        "doc_id": "report.pdf",
        "source": "report.pdf",
        "text": (
            "Full parent section: Revenue increased significantly in Q3, "
            "driven by strong product adoption and improved ASP pricing."
        ),
    }

    with patch("src.retrieval.reranker.get_parent", return_value=parent_payload):
        from src.retrieval.reranker import rerank_preloaded

        enriched = rerank_preloaded("What drove revenue growth?", [child_chunk])

    assert len(enriched) == 1
    hit = enriched[0]
    assert hit["context_text"] == parent_payload["text"], (
        "context_text should be the full parent text, not the child snippet"
    )
    assert hit["chunk_text"] == child_chunk["chunk_text"], (
        "chunk_text (original child) must remain unchanged"
    )


# ── Case 2: retriever.search returns normalised hits ─────────────────────────

def test_search_returns_normalised_hit_dicts():
    """
    Mocks the Pinecone index.search() response and verifies that search()
    returns list of dicts with all expected keys (no KeyError downstream).
    """
    required_keys = {
        "id", "score", "chunk_text", "source", "doc_id",
        "parent_id", "level", "strategy", "h1", "h2", "h3", "h4",
    }

    fake_response = {
        "result": {
            "hits": [
                {
                    "_id": "report.pdf::struct_0",
                    "_score": 0.88,
                    "fields": {
                        "chunk_text": "Operating costs remained flat year-over-year.",
                        "source":     "report.pdf",
                        "doc_id":     "report.pdf",
                        "parent_id":  "",
                        "level":      "",
                        "chunk_index": "0",
                        "strategy":   "structure_recursive",
                        "h1": "Financial Overview",
                        "h2": "Cost Analysis",
                        "h3": "",
                        "h4": "",
                    },
                }
            ]
        }
    }

    mock_index = MagicMock()
    mock_index.search.return_value = fake_response
    mock_pc = MagicMock()
    mock_pc.Index.return_value = mock_index

    with patch("src.retrieval.retriever._pc", mock_pc):
        from src.retrieval.retriever import search

        hits = search("What happened to costs?", top_k=5)

    assert len(hits) == 1, "Expected exactly one hit"
    hit = hits[0]

    missing = required_keys - hit.keys()
    assert not missing, f"Hit dict missing keys: {missing}"

    assert hit["id"]         == "report.pdf::struct_0"
    assert hit["score"]      == 0.88
    assert hit["chunk_text"] == "Operating costs remained flat year-over-year."
    assert hit["h1"]         == "Financial Overview"
