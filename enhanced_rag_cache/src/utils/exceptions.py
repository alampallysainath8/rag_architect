"""
Custom exceptions for the Enhanced RAG Cache pipeline.

PdfExtractionException wraps any error that occurs during the ingestion
flow (PDF parsing, image enrichment, chunking, Pinecone upsert) and
captures file/line location plus full traceback for structured logging.

Usage:
    try:
        ...
    except Exception:
        raise PdfExtractionException("Chunking failed for doc.pdf", sys)
"""

import sys
import traceback
from typing import Optional


class RagPipelineException(Exception):
    """
    Base exception for all RAG pipeline errors.

    Captures the originating file, line number, and full traceback
    automatically from the current exception context.

    Args:
        error_message: Human-readable message or Exception instance.
        error_details: Optional — pass ``sys`` to capture current exc_info,
            or pass an Exception instance directly.
    """

    def __init__(self, error_message, error_details: Optional[object] = None):
        # Normalise message
        norm_msg = str(error_message)

        # Resolve exc_info
        exc_type = exc_value = exc_tb = None
        if error_details is None:
            exc_type, exc_value, exc_tb = sys.exc_info()
        elif hasattr(error_details, "exc_info"):          # sys module passed
            exc_type, exc_value, exc_tb = error_details.exc_info()
        elif isinstance(error_details, BaseException):
            exc_type  = type(error_details)
            exc_value = error_details
            exc_tb    = error_details.__traceback__
        else:
            exc_type, exc_value, exc_tb = sys.exc_info()

        # Walk to the deepest frame for location reporting
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        self.file_name     = last_tb.tb_frame.f_code.co_filename if last_tb else "<unknown>"
        self.lineno        = last_tb.tb_lineno if last_tb else -1
        self.error_message = norm_msg

        self.traceback_str = (
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            if exc_type and exc_tb else ""
        )

        super().__init__(str(self))

    def __str__(self) -> str:
        base = (
            f"[{self.__class__.__name__}] in [{self.file_name}] "
            f"at line [{self.lineno}] | {self.error_message}"
        )
        return f"{base}\nTraceback:\n{self.traceback_str}" if self.traceback_str else base

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(file={self.file_name!r}, "
            f"line={self.lineno}, message={self.error_message!r})"
        )


class IngestionException(RagPipelineException):
    """Raised for errors in the ingestion layer (PDF load, chunking, upsert)."""


class ImageEnrichmentException(RagPipelineException):
    """Raised when the Groq vision pipeline fails for an image."""


class CacheException(RagPipelineException):
    """Raised for unrecoverable errors in any Redis cache tier."""


class RetrievalException(RagPipelineException):
    """Raised for errors during Pinecone search or reranking."""


class GenerationException(RagPipelineException):
    """Raised for errors during LLM answer generation."""
