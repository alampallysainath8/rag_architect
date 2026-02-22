"""
Metadata Builder — creates structured metadata for each chunk.

Metadata structure:
{
    "source": filename,
    "username": username,
    "version": version,
    "page_number": page_number,
    "content_type": "text" | "table",
    "date": ingestion_date,
    "chunk_id": deterministic_id
}
"""

import logging
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class MetadataBuilder:
    """Builds structured metadata for document chunks."""
    
    @staticmethod
    def create_chunk_id(
        document_name: str,
        version: int,
        chunk_index: int
    ) -> str:
        """
        Create deterministic chunk ID.
        
        Format: {document_name}_{version}_chunk{00}
        Example: invoice_v2_chunk03
        
        Args:
            document_name: Document name (without extension).
            version: Document version number.
            chunk_index: Chunk index (0-based).
        
        Returns:
            Deterministic chunk ID.
        """
        # Remove file extension and sanitize document name
        doc_name = Path(document_name).stem
        doc_name = doc_name.replace(" ", "_").replace("-", "_")
        
        # Format: documentname_v{version}_chunk{padded_index}
        chunk_id = f"{doc_name}_v{version}_chunk{chunk_index:03d}"
        
        return chunk_id
    
    @staticmethod
    def build_metadata(
        chunk: Dict[str, Any],
        filename: str,
        username: str,
        version: int,
        ingestion_date: str = None
    ) -> Dict[str, Any]:
        """
        Build complete metadata for a chunk.
        
        Args:
            chunk: Chunk dictionary with chunk_text, page_number, content_type, chunk_index.
            filename: Source filename.
            username: Username (lowercase).
            version: Document version.
            ingestion_date: Date string (ISO format). If None, uses current date.
        
        Returns:
            Complete chunk dictionary with metadata.
        """
        if ingestion_date is None:
            ingestion_date = datetime.now().isoformat()
        
        # Create chunk ID
        chunk_id = MetadataBuilder.create_chunk_id(
            filename,
            version,
            chunk["chunk_index"]
        )
        
        # Build complete metadata
        metadata = {
            "id": chunk_id,
            "chunk_text": chunk["chunk_text"],
            "source": filename,
            "username": username.lower(),
            "version": version,
            "page_number": chunk["page_number"],
            "content_type": chunk["content_type"],
            "date": ingestion_date,
        }
        
        # Add optional fields
        if "chunk_hash" in chunk:
            metadata["chunk_hash"] = chunk["chunk_hash"]
        
        if "table_index" in chunk:
            metadata["table_index"] = chunk["table_index"]
        
        if "sub_chunk_index" in chunk:
            metadata["sub_chunk_index"] = chunk["sub_chunk_index"]
        
        return metadata
    
    @staticmethod
    def build_all_metadata(
        chunks: List[Dict[str, Any]],
        filename: str,
        username: str,
        version: int
    ) -> List[Dict[str, Any]]:
        """
        Build metadata for all chunks.
        
        Args:
            chunks: List of chunk dictionaries.
            filename: Source filename.
            username: Username.
            version: Document version.
        
        Returns:
            List of chunks with complete metadata.
        """
        ingestion_date = datetime.now().isoformat()
        
        chunks_with_metadata = []
        
        for chunk in chunks:
            metadata = MetadataBuilder.build_metadata(
                chunk,
                filename,
                username,
                version,
                ingestion_date
            )
            chunks_with_metadata.append(metadata)
        
        logger.info(
            f"Built metadata for {len(chunks_with_metadata)} chunks "
            f"(file: {filename}, user: {username}, version: {version})"
        )
        
        return chunks_with_metadata


if __name__ == "__main__":
    print("=== Metadata Builder Test ===\n")
    
    # Test chunk ID creation
    chunk_id = MetadataBuilder.create_chunk_id("test_document.pdf", 2, 5)
    print(f"✅ Chunk ID: {chunk_id}")
    assert chunk_id == "test_document_v2_chunk005", "Chunk ID format incorrect"
    
    # Test metadata building
    test_chunk = {
        "chunk_text": "This is a test chunk.",
        "page_number": 1,
        "content_type": "text",
        "chunk_index": 0
    }
    
    metadata = MetadataBuilder.build_metadata(
        test_chunk,
        "test.pdf",
        "alice",
        1
    )
    
    print(f"\n✅ Metadata created:")
    for key, value in metadata.items():
        if key != "chunk_text":
            print(f"   {key}: {value}")
    
    # Test batch metadata building
    test_chunks = [
        {
            "chunk_text": "Chunk 1",
            "page_number": 1,
            "content_type": "text",
            "chunk_index": 0
        },
        {
            "chunk_text": "Chunk 2",
            "page_number": 1,
            "content_type": "text",
            "chunk_index": 1
        },
    ]
    
    all_metadata = MetadataBuilder.build_all_metadata(
        test_chunks,
        "test.pdf",
        "alice",
        1
    )
    
    print(f"\n✅ Built metadata for {len(all_metadata)} chunks")
    print(f"   IDs: {[m['id'] for m in all_metadata]}")
    
    print("\n✅ All tests passed!")
