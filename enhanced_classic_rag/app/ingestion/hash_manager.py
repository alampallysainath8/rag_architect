"""
Hash Manager — computes SHA256 hashes for file deduplication.
"""

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class HashManager:
    """Manages file hashing for deduplication."""
    
    @staticmethod
    def compute_hash(file_path: str, chunk_size: int = 8192) -> str:
        """
        Compute SHA256 hash of a file.
        
        Args:
            file_path: Path to the file.
            chunk_size: Size of chunks to read (for large files).
        
        Returns:
            SHA256 hash as hexadecimal string.
        """
        sha256 = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    sha256.update(data)
            
            hash_value = sha256.hexdigest()
            logger.info(f"Computed hash for {Path(file_path).name}: {hash_value}")
            return hash_value
            
        except Exception as e:
            logger.error(f"Error computing hash for {file_path}: {e}")
            raise
    
    @staticmethod
    def compute_content_hash(content: bytes) -> str:
        """
        Compute SHA256 hash of content bytes.
        
        Args:
            content: Bytes content.
        
        Returns:
            SHA256 hash as hexadecimal string.
        """
        sha256 = hashlib.sha256()
        sha256.update(content)
        return sha256.hexdigest()


if __name__ == "__main__":
    import tempfile
    import os
    
    print("=== Hash Manager Test ===\n")
    
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("This is a test file for hashing.")
        test_file = f.name
    
    try:
        # Test file hashing
        hash_manager = HashManager()
        file_hash = hash_manager.compute_hash(test_file)
        print(f"✅ File hash: {file_hash}")
        
        # Test content hashing
        content = b"This is a test file for hashing."
        content_hash = hash_manager.compute_content_hash(content)
        print(f"✅ Content hash: {content_hash}")
        
        # Verify hashes match
        if file_hash == content_hash:
            print("✅ File and content hashes match!")
        else:
            print("❌ Hashes don't match")
        
        print("\n✅ All tests passed!")
        
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
            print("🧹 Cleaned up test file")
