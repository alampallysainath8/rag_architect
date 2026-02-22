"""
Incremental Diff — computes differences between old and new document versions.

Uses set operations to identify:
- chunks_to_add: New chunks not in old version
- chunks_to_delete: Old chunks not in new version  
- unchanged_chunks: Chunks present in both versions
"""

import logging
from typing import Dict, Set, Tuple, List, Any

logger = logging.getLogger(__name__)


class IncrementalDiff:
    """Computes incremental differences between document versions."""
    
    @staticmethod
    def compute_diff(
        old_hash_map: Dict[str, str],
        new_hash_map: Dict[str, Any]
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Compute differences between old and new chunk sets.
        
        Args:
            old_hash_map: Mapping of chunk_hash -> chunk_id from previous version.
            new_hash_map: Mapping of chunk_hash -> chunk_data from new version.
        
        Returns:
            Tuple of (chunks_to_add, chunks_to_delete, unchanged_chunks)
            where each is a set of chunk hashes.
        """
        old_hashes = set(old_hash_map.keys())
        new_hashes = set(new_hash_map.keys())
        
        # Compute set differences
        chunks_to_add = new_hashes - old_hashes
        chunks_to_delete = old_hashes - new_hashes
        unchanged_chunks = new_hashes & old_hashes
        
        logger.info(
            f"Diff computed: {len(chunks_to_add)} to add, "
            f"{len(chunks_to_delete)} to delete, "
            f"{len(unchanged_chunks)} unchanged"
        )
        
        return chunks_to_add, chunks_to_delete, unchanged_chunks
    
    @staticmethod
    def get_chunks_to_process(
        chunks_to_add: Set[str],
        new_hash_map: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Get the full chunk data for chunks that need to be added.
        
        Args:
            chunks_to_add: Set of chunk hashes to add.
            new_hash_map: Mapping of chunk_hash -> chunk_data.
        
        Returns:
            List of chunk dictionaries to process.
        """
        chunks = []
        for chunk_hash in chunks_to_add:
            if chunk_hash in new_hash_map:
                chunks.append(new_hash_map[chunk_hash])
        
        return chunks
    
    @staticmethod
    def get_chunk_ids_to_delete(
        chunks_to_delete: Set[str],
        old_hash_map: Dict[str, str]
    ) -> List[str]:
        """
        Get chunk IDs that need to be deleted from Pinecone.
        
        Args:
            chunks_to_delete: Set of chunk hashes to delete.
            old_hash_map: Mapping of chunk_hash -> chunk_id.
        
        Returns:
            List of chunk IDs to delete from Pinecone.
        """
        chunk_ids = []
        for chunk_hash in chunks_to_delete:
            if chunk_hash in old_hash_map:
                chunk_ids.append(old_hash_map[chunk_hash])
        
        return chunk_ids
    
    @staticmethod
    def create_diff_report(
        chunks_to_add: Set[str],
        chunks_to_delete: Set[str],
        unchanged_chunks: Set[str]
    ) -> Dict[str, Any]:
        """
        Create a summary report of the diff operation.
        
        Args:
            chunks_to_add: Set of chunk hashes to add.
            chunks_to_delete: Set of chunk hashes to delete.
            unchanged_chunks: Set of unchanged chunk hashes.
        
        Returns:
            Dictionary with diff statistics.
        """
        total_old = len(chunks_to_delete) + len(unchanged_chunks)
        total_new = len(chunks_to_add) + len(unchanged_chunks)
        
        return {
            "chunks_to_add": len(chunks_to_add),
            "chunks_to_delete": len(chunks_to_delete),
            "unchanged_chunks": len(unchanged_chunks),
            "total_old_chunks": total_old,
            "total_new_chunks": total_new,
            "change_percentage": round(
                (len(chunks_to_add) + len(chunks_to_delete)) / max(total_old, 1) * 100, 2
            ) if total_old > 0 else 0
        }


if __name__ == "__main__":
    print("=== Incremental Diff Test ===\n")
    
    # Simulate old version chunks
    old_hash_map = {
        "hash1": "doc_v1_chunk000",
        "hash2": "doc_v1_chunk001",
        "hash3": "doc_v1_chunk002",
        "hash4": "doc_v1_chunk003",
    }
    
    # Simulate new version chunks
    new_hash_map = {
        "hash2": {"chunk_text": "unchanged chunk 1", "page": 1},
        "hash3": {"chunk_text": "unchanged chunk 2", "page": 1},
        "hash5": {"chunk_text": "new chunk 1", "page": 2},
        "hash6": {"chunk_text": "new chunk 2", "page": 2},
    }
    
    # Compute diff
    chunks_to_add, chunks_to_delete, unchanged = IncrementalDiff.compute_diff(
        old_hash_map, new_hash_map
    )
    
    print(f"Chunks to add: {chunks_to_add}")
    print(f"Chunks to delete: {chunks_to_delete}")
    print(f"Unchanged: {unchanged}")
    print()
    
    # Get chunks to process
    new_chunks = IncrementalDiff.get_chunks_to_process(chunks_to_add, new_hash_map)
    print(f"New chunks to embed: {len(new_chunks)}")
    
    # Get IDs to delete
    delete_ids = IncrementalDiff.get_chunk_ids_to_delete(chunks_to_delete, old_hash_map)
    print(f"Chunk IDs to delete: {delete_ids}")
    print()
    
    # Generate report
    report = IncrementalDiff.create_diff_report(chunks_to_add, chunks_to_delete, unchanged)
    print("Diff Report:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    # Validate results
    assert len(chunks_to_add) == 2, "Should have 2 chunks to add"
    assert len(chunks_to_delete) == 2, "Should have 2 chunks to delete"
    assert len(unchanged) == 2, "Should have 2 unchanged chunks"
    assert len(delete_ids) == 2, "Should have 2 IDs to delete"
    
    print("\n✅ All tests passed!")
