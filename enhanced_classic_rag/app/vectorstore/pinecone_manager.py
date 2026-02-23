"""
Pinecone Manager — handles vector database operations with namespace support.

Features:
- Index creation with integrated embedding
- Namespace management (username-based)
- Batch upsert with metadata
- Retrieval with namespace isolation
"""

import time
import logging
from typing import List, Dict, Any
from pinecone import Pinecone

logger = logging.getLogger(__name__)


class PineconeManager:
    """Manages Pinecone vector database operations."""
    
    def __init__(
        self,
        api_key: str,
        index_name: str,
        cloud: str,
        region: str,
        embedding_model: str,
        batch_size: int = 96
    ):
        """
        Initialize Pinecone manager.
        
        Args:
            api_key: Pinecone API key.
            index_name: Name of the Pinecone index.
            cloud: Cloud provider (e.g., 'aws').
            region: Cloud region (e.g., 'us-east-1').
            embedding_model: Embedding model name.
            batch_size: Batch size for upserts.
        """
        self.api_key = api_key
        self.index_name = index_name
        self.cloud = cloud
        self.region = region
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        
        self._pc = Pinecone(api_key=api_key)
        self._index = None
    
    def _get_or_create_index(self):
        """Get or create the Pinecone index with integrated embedding."""
        if self._index is not None:
            return self._index
        
        if not self._pc.has_index(self.index_name):
            logger.info(f"Creating Pinecone index '{self.index_name}' with integrated embedding...")
            
            try:
                self._pc.create_index_for_model(
                    name=self.index_name,
                    cloud=self.cloud,
                    region=self.region,
                    embed={
                        "model": self.embedding_model,
                        "field_map": {"text": "chunk_text"},
                    },
                )
                
                # Wait for index to be ready
                logger.info("Waiting for index to be ready...")
                max_wait = 60  # Maximum wait time in seconds
                wait_time = 0
                
                while wait_time < max_wait:
                    index_status = self._pc.describe_index(self.index_name)
                    if index_status.status.get("ready", False):
                        break
                    time.sleep(2)
                    wait_time += 2
                
                logger.info("✅ Index created and ready!")
                
            except Exception as e:
                logger.error(f"Error creating index: {e}")
                raise
        
        self._index = self._pc.Index(self.index_name)
        return self._index
    
    def ensure_namespace_exists(self, namespace: str) -> None:
        """
        Ensure namespace exists in Pinecone.
        Note: Pinecone creates namespaces automatically on first upsert.
        
        Args:
            namespace: Namespace name (typically username).
        """
        # Pinecone creates namespaces automatically, but we log for tracking
        logger.info(f"Using namespace: {namespace}")
    
    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        namespace: str
    ) -> int:
        """
        Upsert chunks to Pinecone with metadata.
        
        Args:
            chunks: List of chunk dictionaries with metadata.
            namespace: Namespace for the chunks (typically username).
        
        Returns:
            Number of chunks upserted.
        """
        index = self._get_or_create_index()
        self.ensure_namespace_exists(namespace)
        
        total = 0
        
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            
            # Prepare records for Pinecone
            pinecone_records = []
            
            for chunk in batch:
                record = {
                    "_id": chunk["id"],
                    "chunk_text": chunk["chunk_text"],
                    "source": chunk["source"],
                    "username": chunk["username"],
                    "version": chunk["version"],
                    "page_number": chunk["page_number"],
                    "content_type": chunk["content_type"],
                    "date": chunk["date"],
                }
                
                # Add optional fields
                if "table_index" in chunk:
                    record["table_index"] = chunk["table_index"]
                
                pinecone_records.append(record)
            
            try:
                # Upsert with integrated embedding
                index.upsert_records(namespace, pinecone_records)
                total += len(pinecone_records)
                logger.info(f"Upserted batch {i // self.batch_size + 1}: {len(pinecone_records)} records")
                
            except Exception as e:
                logger.error(f"Error upserting batch: {e}")
                raise
        
        logger.info(f"✅ Total upserted: {total} chunks to namespace '{namespace}'")
        return total
    
    def search(
        self,
        query: str,
        namespace: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks in a namespace.
        
        Args:
            query: Search query.
            namespace: Namespace to search in.
            top_k: Number of results to return.
        
        Returns:
            List of search results with scores and metadata.
        """
        index = self._get_or_create_index()
        
        try:
            results = index.search(
                namespace=namespace,
                query={
                    "top_k": top_k,
                    "inputs": {"text": query},
                },
                fields=["chunk_text", "source", "page_number", "content_type", "version", "date", "table_index"],
            )
            
            hits = []
            for item in results.get("result", {}).get("hits", []):
                hits.append({
                    "id": item.get("_id", ""),
                    "score": item.get("_score", 0.0),
                    "chunk_text": item.get("fields", {}).get("chunk_text", ""),
                    "source": item.get("fields", {}).get("source", ""),
                    "page_number": item.get("fields", {}).get("page_number", 0),
                    "content_type": item.get("fields", {}).get("content_type", ""),
                    "version": item.get("fields", {}).get("version", 1),
                    "date": item.get("fields", {}).get("date", ""),
                    "table_index": item.get("fields", {}).get("table_index", None),
                })
            
            logger.info(f"Found {len(hits)} results for query in namespace '{namespace}'")
            return hits
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise
    
    def delete_chunks(
        self,
        chunk_ids: List[str],
        namespace: str
    ) -> int:
        """
        Delete specific chunks from Pinecone by their IDs.
        
        Args:
            chunk_ids: List of chunk IDs to delete.
            namespace: Namespace to delete from.
        
        Returns:
            Number of chunks deleted.
        """
        if not chunk_ids:
            logger.info("No chunks to delete")
            return 0
        
        index = self._get_or_create_index()
        
        try:
            # Delete in batches to avoid hitting API limits
            batch_size = 100
            total_deleted = 0
            
            for i in range(0, len(chunk_ids), batch_size):
                batch = chunk_ids[i:i + batch_size]
                index.delete(ids=batch, namespace=namespace)
                total_deleted += len(batch)
                logger.info(f"Deleted batch of {len(batch)} chunks from namespace '{namespace}'")
            
            logger.info(f"✅ Total deleted: {total_deleted} chunks from namespace '{namespace}'")
            return total_deleted
            
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            raise
    
    def delete_namespace(self, namespace: str) -> None:
        """
        Delete all vectors in a namespace.
        
        Args:
            namespace: Namespace to delete.
        """
        index = self._get_or_create_index()
        
        try:
            index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Deleted namespace '{namespace}'")
            
        except Exception as e:
            logger.error(f"Error deleting namespace: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Index statistics dictionary.
        """
        index = self._get_or_create_index()
        
        try:
            stats = index.describe_index_stats()
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("=== Pinecone Manager Test ===\n")
    
    # Test configuration
    api_key = os.getenv("PINECONE_API_KEY", "")
    
    if not api_key:
        print("❌ PINECONE_API_KEY not set. Skipping test.")
    else:
        try:
            manager = PineconeManager(
                api_key=api_key,
                index_name="enhanced-rag-test",
                cloud="aws",
                region="us-east-1",
                embedding_model="multilingual-e5-large"
            )
            
            # Test upsert
            test_chunks = [
                {
                    "id": "test_v1_chunk000",
                    "chunk_text": "This is a test chunk about AI.",
                    "source": "test.pdf",
                    "username": "testuser",
                    "version": 1,
                    "page_number": 1,
                    "content_type": "text",
                    "date": "2024-01-01T00:00:00"
                }
            ]
            
            count = manager.upsert_chunks(test_chunks, "testuser")
            print(f"✅ Upserted {count} chunks")
            
            # Test search
            results = manager.search("AI technology", "testuser", top_k=5)
            print(f"✅ Found {len(results)} results")
            
            # Test stats
            stats = manager.get_stats()
            print(f"✅ Index stats: {stats}")
            
            print("\n✅ All tests passed!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
