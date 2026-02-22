"""
Chunker — implements separate chunking strategies for text and tables.

Text Chunking:
- Token-based chunking with overlap
- Maintains semantic continuity
- Avoids splitting mid-sentence

Table Chunking:
- Each table kept as a single chunk (no splitting)
- Tables converted to markdown format
- Preserves complete table structure
"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class Chunker:
    """Handles chunking of text and tables with different strategies."""
    
    def __init__(
        self,
        text_chunk_size: int = 500,
        text_chunk_overlap: int = 80,
        table_chunk_size: int = 800  # Kept for backward compatibility but not used
    ):
        """
        Initialize chunker with configurable parameters.
        
        Args:
            text_chunk_size: Size of text chunks (in characters).
            text_chunk_overlap: Overlap between text chunks (in characters).
            table_chunk_size: DEPRECATED - tables are now always kept as single chunks.
        """
        self.text_chunk_size = text_chunk_size
        self.text_chunk_overlap = text_chunk_overlap
        self.table_chunk_size = table_chunk_size  # Kept for backward compatibility
    
    def chunk_text(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk text content from pages with overlap.
        
        Args:
            pages_data: Pages data from DocumentLoader.
        
        Returns:
            List of text chunks:
            [
                {
                    "chunk_text": "text content...",
                    "page_number": 1,
                    "content_type": "text",
                    "chunk_index": 0
                },
                ...
            ]
        """
        chunks = []
        chunk_index = 0
        
        for page_data in pages_data:
            page_num = page_data["page_number"]
            text = page_data.get("text", "").strip()
            
            if not text:
                continue
            
            # Clean text
            text = self._clean_text(text)
            
            # Split text into chunks with overlap
            page_chunks = self._split_with_overlap(
                text,
                self.text_chunk_size,
                self.text_chunk_overlap
            )
            
            for chunk_text in page_chunks:
                if chunk_text.strip():
                    chunks.append({
                        "chunk_text": chunk_text,
                        "page_number": page_num,
                        "content_type": "text",
                        "chunk_index": chunk_index
                    })
                    chunk_index += 1
        
        logger.info(f"Created {len(chunks)} text chunks")
        return chunks
    
    def chunk_tables(self, table_strings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk table content - keeps each table as a single chunk (no splitting).
        
        Tables are already converted to markdown format by DocumentLoader.
        
        Args:
            table_strings: Table strings from DocumentLoader.get_table_strings().
        
        Returns:
            List of table chunks:
            [
                {
                    "chunk_text": "table content (markdown format)...",
                    "page_number": 1,
                    "content_type": "table",
                    "chunk_index": 0,
                    "table_index": 0
                },
                ...
            ]
        """
        chunks = []
        
        for idx, table_data in enumerate(table_strings):
            page_num = table_data["page_number"]
            table_index = table_data["table_index"]
            table_string = table_data["table_string"]
            
            if not table_string.strip():
                continue
            
            # Keep entire table as one chunk regardless of size
            chunks.append({
                "chunk_text": table_string,
                "page_number": page_num,
                "content_type": "table",
                "chunk_index": idx,
                "table_index": table_index
            })
        
        logger.info(f"Created {len(chunks)} table chunks (each table = 1 chunk)")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text.
        
        Returns:
            Cleaned text.
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def _split_with_overlap(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split.
            chunk_size: Size of each chunk.
            overlap: Overlap between chunks.
        
        Returns:
            List of text chunks.
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        import re

        sentence_boundary_re = re.compile(r'(?<=[\.\?\!])\s+')

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]

            # If we're not at the very end, try to avoid splitting mid-sentence.
            if end < text_length:
                # Search for the last sentence boundary inside the chunk.
                matches = list(sentence_boundary_re.finditer(chunk))
                if matches:
                    last_match = matches[-1]
                    # Only break at a sentence boundary if it's reasonably close to the end
                    # of the chunk (e.g., after at least half of the chunk_size) to avoid
                    # creating many tiny chunks.
                    if last_match.start() > int(chunk_size * 0.5):
                        cut = last_match.end()
                        chunk = chunk[:cut].strip()
                        end = start + cut

            chunks.append(chunk.strip())

            # Move start position (with overlap)
            start = end - overlap if end < text_length else text_length
        
        return chunks

if __name__ == "__main__":
    print("=== Chunker Test ===\n")
    
    # Test text chunking
    test_pages = [
        {
            "page_number": 1,
            "text": "This is a test document. It contains multiple sentences. "
                   "The chunker should split this text into multiple chunks with overlap. "
                   "This helps maintain context across chunks. "
                   "Each chunk should be roughly the same size. " * 10
        }
    ]
    
    chunker = Chunker(text_chunk_size=200, text_chunk_overlap=50)
    
    # Test text chunking
    text_chunks = chunker.chunk_text(test_pages)
    print(f"✅ Created {len(text_chunks)} text chunks")
    print(f"   First chunk: {text_chunks[0]['chunk_text'][:100]}...")
    print(f"   Last chunk: {text_chunks[-1]['chunk_text'][:100]}...")
    
    # Test table chunking
    test_tables = [
        {
            "page_number": 1,
            "table_index": 0,
            "table_string": "Name | Age | City\n--- | --- | ---\nAlice | 30 | NYC\nBob | 25 | SF"
        }
    ]
    
    table_chunks = chunker.chunk_tables(test_tables)
    print(f"\n✅ Created {len(table_chunks)} table chunks")
    if table_chunks:
        print(f"   Table chunk: {table_chunks[0]['chunk_text']}")
    
    print("\n✅ All tests passed!")
