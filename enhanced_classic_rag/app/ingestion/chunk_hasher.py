"""
Chunk Hasher — computes SHA256 hashes for normalized chunk content.

Normalization rules:
- Convert to lowercase
- Normalize whitespace (multiple spaces -> single space)
- Strip leading/trailing whitespace
"""

import hashlib
import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ChunkHasher:
    """Handles chunk content normalization and hashing."""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for consistent hashing.
        
        Normalization:
        - Convert to lowercase
        - Replace multiple whitespace with single space
        - Strip leading/trailing whitespace
        
        Args:
            text: Raw text content.
        
        Returns:
            Normalized text.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        
        # Strip
        text = text.strip()
        
        return text
    
    @staticmethod
    def compute_chunk_hash(text: str) -> str:
        """
        Compute SHA256 hash of normalized chunk content.
        
        Args:
            text: Chunk text content.
        
        Returns:
            SHA256 hash as hexadecimal string.
        """
        normalized = ChunkHasher.normalize_text(text)
        sha256 = hashlib.sha256()
        sha256.update(normalized.encode('utf-8'))
        return sha256.hexdigest()
    
    @staticmethod
    def add_hashes_to_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add chunk_hash field to all chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'chunk_text' field.
        
        Returns:
            Chunks with 'chunk_hash' field added.
        """
        for chunk in chunks:
            chunk_text = chunk.get('chunk_text', '')
            chunk['chunk_hash'] = ChunkHasher.compute_chunk_hash(chunk_text)
        
        logger.info(f"Computed hashes for {len(chunks)} chunks")
        return chunks
    
    @staticmethod
    def create_hash_map(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Create a mapping of chunk_hash -> chunk_data.
        
        Args:
            chunks: List of chunks with 'chunk_hash' field.
        
        Returns:
            Dictionary mapping chunk_hash to full chunk data.
        """
        hash_map = {}
        for chunk in chunks:
            chunk_hash = chunk.get('chunk_hash')
            if chunk_hash:
                hash_map[chunk_hash] = chunk
        
        return hash_map


if __name__ == "__main__":
    print("=== Chunk Hasher Test ===\n")
    
    # Test normalization
    test_text = "  This  is   a  TEST   chunk.  \n\n  With   multiple   spaces.  "
    normalized = ChunkHasher.normalize_text(test_text)
    print(f"Original : '{test_text}'")
    print(f"Normalized: '{normalized}'")
    print()
    
    # Test hashing
    hash1 = ChunkHasher.compute_chunk_hash("This is a test chunk.")
    hash2 = ChunkHasher.compute_chunk_hash("  THIS  is   a TEST   chunk.  ")
    hash3 = ChunkHasher.compute_chunk_hash("This is a different chunk.")
    
    print(f"Hash 1: {hash1}")
    print(f"Hash 2: {hash2}")
    print(f"Hash 3: {hash3}")
    print()
    
    if hash1 == hash2:
        print("✅ Same content produces same hash (normalization works)")
    else:
        print("❌ Normalization failed")
    
    if hash1 != hash3:
        print("✅ Different content produces different hash")
    else:
        print("❌ Hashing failed")
    
    # Test batch hashing
    test_chunks = [
        {"chunk_text": "First chunk", "page": 1},
        {"chunk_text": "Second chunk", "page": 2},
        {"chunk_text": "Third chunk", "page": 3},
    ]
    
    chunks_with_hashes = ChunkHasher.add_hashes_to_chunks(test_chunks)
    print(f"\n✅ Added hashes to {len(chunks_with_hashes)} chunks")
    
    hash_map = ChunkHasher.create_hash_map(chunks_with_hashes)
    print(f"✅ Created hash map with {len(hash_map)} entries")
    
    print("\n✅ All tests passed!")
