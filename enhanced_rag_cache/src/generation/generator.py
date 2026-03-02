"""
Generator — LLM-based answer synthesis using OpenAI GPT-4o-mini.

Receives a list of context chunks (each with `context_text` and metadata)
and a user query. Produces a cited, grounded answer.

`context_text` field is preferred over `chunk_text` when present — this
ensures the LLM sees full parent chunks for parent-child strategy results.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from openai import OpenAI

from src.utils.config_loader import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)

_OCFG = cfg["openai"]
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

SYSTEM_PROMPT = """You are a precise, helpful assistant that answers questions using ONLY the context provided below.

Rules:
1. Base your answer strictly on the provided context.
2. If the context does not contain enough information, say: "I don't have enough information to answer that based on the provided documents."
3. Cite sources inline using [n] notation (e.g., "Revenue increased [1]").
4. At the end of your answer, include a "References" section listing each cited source.
   Format: [n] <source>, heading: <section> (if available)
5. Be concise but complete. Prefer bullet points for lists.

Context:
{context}"""


def _build_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Format chunk list into a numbered context string.

    Prefers `context_text` (full parent) over `chunk_text` (narrow child).
    """
    parts = []
    for i, c in enumerate(chunks, 1):
        text = c.get("context_text") or c.get("chunk_text", "")
        source = c.get("source", "unknown")
        heading = " > ".join(
            filter(None, [c.get("h1"), c.get("h2"), c.get("h3"), c.get("h4")])
        )
        label = f"source: {source}"
        if heading:
            label += f", section: {heading}"
        parts.append(f"[{i}] ({label})\n{text}")
    return "\n\n---\n\n".join(parts)


def generate_answer(
    query: str,
    chunks: List[Dict[str, Any]],
) -> str:
    """
    Generate a grounded, cited answer for *query* using *chunks* as context.

    Args:
        query:  User question string.
        chunks: Reranked/retrieved chunk dicts (with optional context_text).

    Returns:
        LLM answer string.
    """
    if not chunks:
        return "I don't have enough information to answer that based on the provided documents."

    context_block = _build_context(chunks)
    system_msg = SYSTEM_PROMPT.format(context=context_block)

    try:
        response = _client.chat.completions.create(
            model=_OCFG["llm_model"],
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": query},
            ],
            max_tokens=_OCFG["max_tokens"],
            temperature=_OCFG["temperature"],
        )
        answer: str = response.choices[0].message.content or ""
        logger.info(f"Generated answer ({len(answer)} chars) for query='{query[:60]}'")
        return answer
    except Exception as exc:
        logger.error(f"OpenAI generation error: {exc}")
        raise


def format_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build a deduplicated, user-friendly source list from the chunk list.

    Returns a list of dicts suitable for serialisation in the API response.
    """
    seen: set[str] = set()
    sources: List[Dict[str, Any]] = []
    for i, c in enumerate(chunks, 1):
        source = c.get("source", "unknown")
        heading = " > ".join(
            filter(None, [c.get("h1"), c.get("h2"), c.get("h3"), c.get("h4")])
        )
        key = f"{source}|{heading}"
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "citation": f"[{i}]",
                    "source": source,
                    "section": heading,
                    "doc_id": c.get("doc_id", ""),
                    "score": round(c.get("score", 0.0), 4),
                    "chunk_preview": (c.get("chunk_text", ""))[:200],
                }
            )
    return sources
