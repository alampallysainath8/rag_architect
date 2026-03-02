"""
Image Enricher — replaces base64 images in Markdown with LLM-generated descriptions.

Pipeline per image:
  1. Extract base64 image + its nearby context (headers / captions / legends)
     from the surrounding Markdown lines.
  2. Send image + context to Groq vision model.
  3. Parse model output into three sections:
       ---DESCRIPTION---  prose paragraph
       ---TABLE---        Markdown table extracted from chart/image
       ---JSON---         Structured JSON for programmatic ingestion
  4. Inline-replace the base64 blob with the enriched text block.

Typical usage:
    from src.utils.image_enricher import enrich_markdown_images
    enriched_md = enrich_markdown_images(raw_md_with_base64)
"""

from __future__ import annotations

import os
import re
from typing import Optional

from groq import Groq

from src.utils.logger import get_logger
from src.utils.exceptions import ImageEnrichmentException

logger = get_logger(__name__)

# ── Groq client ───────────────────────────────────────────────────────────────
_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Add it to your .env file or environment."
            )
        _client = Groq(api_key=api_key)
    return _client


# ── Vision prompt ─────────────────────────────────────────────────────────────
_VISION_PROMPT = """\
You are an AI assistant specialised in analysing financial documents and data visualisations.

You will receive:
- `context`: surrounding Markdown text (headers, captions, legends) that names the chart/data.
- `image`: the chart or visualisation.

Return your answer in exactly three clearly-labelled sections (in this order):

---DESCRIPTION---
One concise paragraph: chart type, what is compared, axis labels, units, and the key insight.

---TABLE---
If the image contains tabular data (bar chart, line chart, stacked chart, table)
extract the underlying numbers as a Markdown table.
- Use legend labels as column headers and series names as row labels.
- Use exact numbers when visible; mark approximations with ~.
- If the image has no extractable numbers write: None

---JSON---
A JSON object:
{ "columns": [...], "rows": [...], "values": [[...], ...] }
where "values" is a list of rows, each a list aligned to "columns" (numbers preferred).
Write null if the image has no extractable data.

Rules:
- Preserve legend labels exactly so downstream code can match them.
- When the context mentions quarter labels (Q4'22, Q1'23 …) use them as column headers.
- Be concise; do not repeat yourself across sections.
"""

_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
_TEMPERATURE = 0.2
_MAX_TOKENS = 1500


# ── Base64 extraction ─────────────────────────────────────────────────────────

def _extract_base64_images(markdown: str) -> list[tuple[str, str]]:
    """
    Return list of (full_match, clean_base64) tuples from a Markdown string.
    Base64 data returned has newlines stripped.
    """
    full_pattern = r"(!\[\]\(data:image/png;base64,[A-Za-z0-9+/=\n\r]+\))"
    b64_pattern = r"!\[\]\(data:image/png;base64,([A-Za-z0-9+/=\n\r]+)\)"

    full_matches = re.findall(full_pattern, markdown)
    b64_matches = [m.replace("\n", "").replace("\r", "") for m in re.findall(b64_pattern, markdown)]
    return list(zip(full_matches, b64_matches))


# ── Context extraction ────────────────────────────────────────────────────────

def _get_nearby_context(
    markdown: str,
    full_match: str,
    before_lines: int = 10,
    after_lines: int = 6,
) -> str:
    """
    Extract surrounding text (headers, captions, legend labels) around an image match.
    Keeps:
      • Heading lines (start with #)
      • Short lines containing $, %, quarter references, or legend keywords
    """
    lines = markdown.splitlines()
    joined = "\n".join(lines)
    idx = joined.find(full_match)
    if idx == -1:
        return ""

    start_line = joined[:idx].count("\n")
    match_line_span = full_match.count("\n") + 1

    before = lines[max(0, start_line - before_lines): start_line]
    after = lines[start_line + match_line_span: start_line + match_line_span + after_lines]

    kept = []
    for line in before + after:
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            kept.append(s)
        elif len(s) < 150 and (
            ":" in s
            or "$" in s
            or "%" in s
            or re.search(r"Q\d['′\"]?\d{2}", s)
            or any(kw in s.lower() for kw in ("legend", "unit", "million", "billion",
                                               "revenue", "income", "expense", "margin"))
        ):
            kept.append(s)
    return "\n".join(kept)


# ── Vision call ──────────────────────────────────────────────────────────────

def _call_vision_model(base64_data: str, context: str) -> str:
    """Send image + context to the Groq vision model and return raw text."""
    content = []

    # Prepend context if available
    if context.strip():
        content.append({
            "type": "text",
            "text": f"Context (section headers, captions, and legends):\n{context}",
        })

    content.append({"type": "text", "text": _VISION_PROMPT})
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64_data}"},
    })

    try:
        response = _get_client().chat.completions.create(
            messages=[{"role": "user", "content": content}],
            model=_MODEL,
            temperature=_TEMPERATURE,
            max_tokens=_MAX_TOKENS,
        )
        raw = response.choices[0].message.content
        return raw if isinstance(raw, str) else (raw.get("text") or str(raw))
    except Exception as exc:
        logger.error(f"Vision model call failed: {exc}")
        raise ImageEnrichmentException(
            f"Groq vision call failed for model '{_MODEL}': {exc}", exc
        ) from exc


# ── Output parser ─────────────────────────────────────────────────────────────

def _parse_model_output(text: str) -> tuple[str, str | None, str | None]:
    """
    Parse raw model output into (description, markdown_table, json_str).
    Falls back to heuristics if the markers are absent.
    """
    if not text:
        return "", None, None

    # --- DESCRIPTION ---
    desc = ""
    m = re.search(r"---DESCRIPTION---\s*(.*?)\s*(?:---TABLE---|---JSON---|\Z)", text, re.S)
    if m:
        desc = m.group(1).strip()
    else:
        paragraphs = text.strip().split("\n\n")
        desc = paragraphs[0].strip() if paragraphs else text.strip()

    # --- TABLE ---
    table: str | None = None
    m2 = re.search(r"---TABLE---\s*(.*?)(?=---JSON---|\Z)", text, re.S)
    if m2:
        candidate = m2.group(1).strip()
        table = None if candidate.lower() == "none" else candidate
    else:
        # fall back: find any markdown table in the output
        tbl = re.search(r"(\|.+\|\n\|[\s\-:]+\|[^\n]*(?:\n\|.+\|)*)", text)
        if tbl:
            table = tbl.group(0).strip()

    # --- JSON ---
    json_str: str | None = None
    m3 = re.search(r"---JSON---\s*(\{.*?\})\s*$", text, re.S)
    if m3:
        candidate = m3.group(1).strip()
        json_str = None if candidate.lower() == "null" else candidate
    else:
        # fall back: find first JSON-like object with "columns" key
        jm = re.search(r"(\{\s*\"columns\".*?\})", text, re.S)
        if jm:
            json_str = jm.group(0).strip()

    return desc, table, json_str


# ── Replacement block builder ─────────────────────────────────────────────────

def _build_replacement(idx: int, desc: str, table: str | None, json_str: str | None) -> str:
    """Build the Markdown block that replaces a base64 image."""
    parts: list[str] = []

    parts.append(f"**[Image {idx} Description]:**\n\n{desc}")

    if table:
        parts.append(f"**[Image {idx} Table]:**\n\n{table}")

    if json_str:
        parts.append(f"**[Image {idx} JSON Extraction]:**\n\n```json\n{json_str}\n```")

    return "\n\n" + "\n\n".join(parts) + "\n\n"


# ── Public API ────────────────────────────────────────────────────────────────

def enrich_markdown_images(markdown: str) -> tuple[str, list[dict]]:
    """
    Find all base64 images in *markdown*, send each to the Groq vision model,
    replace each blob with a placeholder token, and return the block dicts.

    Args:
        markdown: Raw Markdown containing ``![](data:image/png;base64,…)`` blobs.

    Returns:
        (cleaned_markdown, image_blocks)
          cleaned_markdown — original markdown with each base64 blob replaced
                             by ``__IMG_BLOCK_N__`` placeholder tokens.
          image_blocks     — list of block dicts (one per image), compatible
                             with extract_special_blocks output:
                             { "type": "image", "placeholder": str,
                               "content": str, "image_number": int }
    """
    images = _extract_base64_images(markdown)
    if not images:
        logger.info("No base64 images found — skipping enrichment.")
        return markdown, []

    logger.info(f"Enriching {len(images)} image(s)…")
    cleaned = markdown
    image_blocks: list[dict] = []

    for idx, (full_match, b64) in enumerate(images, start=1):
        logger.info(f"  Processing image {idx}/{len(images)}")
        context = _get_nearby_context(cleaned, full_match)
        try:
            raw_output = _call_vision_model(b64, context)
            desc, table, json_str = _parse_model_output(raw_output)
        except ImageEnrichmentException as exc:
            logger.warning(f"  Image {idx}: enrichment failed — using placeholder. {exc}")
            desc, table, json_str = "[Image description unavailable — vision model error]", None, None

        if not desc:
            desc = "[Image description unavailable]"
            logger.warning(f"  Image {idx}: vision model returned empty response.")

        content = _build_replacement(idx, desc, table, json_str).strip()
        token   = f"__IMG_BLOCK_{idx - 1}__"

        image_blocks.append({
            "type":         "image",
            "placeholder":  token,
            "content":      content,
            "image_number": idx,
        })

        # Replace the raw blob with the placeholder; the chunker restores
        # the full description block when it processes each chunk.
        cleaned = cleaned.replace(full_match, f"\n\n{token}\n\n", 1)
        logger.info(f"  Image {idx} ✓ (table={'yes' if table else 'no'}, json={'yes' if json_str else 'no'})")

    return cleaned, image_blocks
