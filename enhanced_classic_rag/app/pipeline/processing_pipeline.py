"""
Processing Pipeline — orchestrates incremental document ingestion with chunk diffing.

Flow:
1. Normalize username and filename
2. Compute document SHA256 hash
3. Check for duplicates or existing versions
4. If new document → Full ingestion
5. If same hash → Mark duplicate, stop
6. If different hash → Incremental diff
   - Load and chunk document
   - Hash all chunks
   - Compare with previous version
   - Add new chunks
   - Delete removed chunks
   - Skip unchanged chunks
7. Update database with version tracking
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

from app.core.config import settings
from app.db.sqlite_manager import SQLiteManager
from app.ingestion.hash_manager import HashManager
from app.ingestion.document_loader import DocumentLoader
from app.ingestion.chunker import Chunker
from app.ingestion.chunk_hasher import ChunkHasher
from app.ingestion.incremental_diff import IncrementalDiff
from app.ingestion.metadata_builder import MetadataBuilder
from app.vectorstore.pinecone_manager import PineconeManager

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """Orchestrates incremental document processing with chunk-level diffing."""
    
    def __init__(self):
        """Initialize pipeline with all required components."""
        # Initialize components
        self.db_manager = SQLiteManager(settings.SQLITE_DB_PATH)
        self.hash_manager = HashManager()
        self.document_loader = DocumentLoader()
        self.chunker = Chunker(
            text_chunk_size=settings.TEXT_CHUNK_SIZE,
            text_chunk_overlap=settings.TEXT_CHUNK_OVERLAP,
            table_chunk_size=settings.TABLE_CHUNK_SIZE
        )
        self.chunk_hasher = ChunkHasher()
        self.incremental_diff = IncrementalDiff()
        self.metadata_builder = MetadataBuilder()
        self.pinecone_manager = PineconeManager(
            api_key=settings.PINECONE_API_KEY,
            index_name=settings.PINECONE_INDEX_NAME,
            cloud=settings.PINECONE_CLOUD,
            region=settings.PINECONE_REGION,
            embedding_model=settings.EMBEDDING_MODEL,
            batch_size=settings.UPSERT_BATCH_SIZE
        )
        
        logger.info("Processing pipeline initialized with incremental diff support")
    
    def process_document(
        self,
        file_path: str,
        username: str,
        original_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a document with incremental diff support.
        
        Args:
            file_path: Path to the document file.
            username: Username who uploaded the document.
        
        Returns:
            Processing result dictionary with status and details.
        """
        # Step 1: Normalize username and filename
        username = username.lower().strip()
        # Prefer the original filename (from upload) if provided so metadata
        # and `source` fields reflect the user's file name rather than a
        # temporary path name.
        if original_filename:
            filename = Path(original_filename).name.strip()
        else:
            filename = Path(file_path).name.strip()
        
        logger.info(f"Starting processing for {filename} (user: {username})")
        
        try:
            # Step 2: Compute document hash
            logger.info("Computing document hash...")
            document_hash = self.hash_manager.compute_hash(file_path)
            logger.info(f"Document hash: {document_hash}")
            
            # Step 3: Query latest document version
            logger.info("Checking for existing versions...")
            latest_doc = self.db_manager.get_latest_document(filename, username)
            
            # Determine processing path
            if latest_doc is None:
                # Case 1: New document
                logger.info("New document detected - full ingestion")
                return self._process_new_document(file_path, filename, username, document_hash)
            
            elif latest_doc['document_hash'] == document_hash:
                # Case 2: Same hash - duplicate
                logger.warning(f"Duplicate detected: {filename} (v{latest_doc['version']})")
                return self._handle_duplicate(filename, username, latest_doc)
            
            else:
                # Case 3: Different hash - incremental update
                logger.info(f"Document updated detected - incremental diff (old v{latest_doc['version']})")
                return self._process_incremental_update(
                    file_path, filename, username, document_hash, latest_doc
                )
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {
                "status": "error",
                "message": str(e),
                "filename": filename,
                "username": username
            }
    
    def _process_new_document(
        self,
        file_path: str,
        filename: str,
        username: str,
        document_hash: str
    ) -> Dict[str, Any]:
        """Process a brand new document (version 1)."""
        version = 1
        
        # Insert document record
        doc_id = self.db_manager.insert_document(
            filename=filename,
            username=username,
            document_hash=document_hash,
            version=version,
            status="processing",
            is_active=True
        )
        
        try:
            # Ensure namespace exists
            self.pinecone_manager.ensure_namespace_exists(username)
            
            # Load and chunk document
            logger.info("Loading document...")
            pages_data = self.document_loader.load_pdf(file_path)
            table_strings = self.document_loader.get_table_strings(pages_data)
            
            logger.info("Chunking content...")
            text_chunks = self.chunker.chunk_text(pages_data)
            table_chunks = self.chunker.chunk_tables(table_strings)
            all_chunks = text_chunks + table_chunks
            
            if not all_chunks:
                raise ValueError("No chunks created from document")
            
            logger.info(f"Created {len(all_chunks)} chunks ({len(text_chunks)} text, {len(table_chunks)} table)")
            
            # Add chunk hashes
            logger.info("Computing chunk hashes...")
            all_chunks = self.chunk_hasher.add_hashes_to_chunks(all_chunks)
            
            # Build metadata
            logger.info("Building metadata...")
            chunks_with_metadata = self.metadata_builder.build_all_metadata(
                all_chunks, filename, username, version
            )
            
            # Upsert to Pinecone
            logger.info("Upserting to Pinecone...")
            upserted_count = self.pinecone_manager.upsert_chunks(chunks_with_metadata, username)
            
            # Insert chunks to database
            logger.info("Saving chunks to database...")
            chunk_records = []
            for chunk in chunks_with_metadata:
                chunk_records.append({
                    'document_id': doc_id,
                    'filename': filename,
                    'username': username,
                    'version': version,
                    'chunk_id': chunk['id'],
                    'chunk_hash': chunk['chunk_hash'],
                    'page_number': chunk['page_number'],
                    'content_type': chunk['content_type'],
                    'is_active': True
                })
            
            self.db_manager.insert_chunks_batch(chunk_records)
            
            # Update status
            self.db_manager.update_status(doc_id, "processed")
            
            logger.info(f"✅ New document processed: {filename} v{version}")
            
            return {
                "status": "success",
                "message": "New document processed successfully",
                "filename": filename,
                "username": username,
                "version": version,
                "doc_id": doc_id,
                "total_chunks": len(all_chunks),
                "text_chunks": len(text_chunks),
                "table_chunks": len(table_chunks),
                "chunks_added": upserted_count,
                "chunks_deleted": 0,
                "unchanged_chunks": 0
            }
            
        except Exception as e:
            logger.error(f"Error in new document processing: {e}")
            self.db_manager.update_status(doc_id, "failed")
            raise
    
    def _handle_duplicate(
        self,
        filename: str,
        username: str,
        latest_doc: Dict
    ) -> Dict[str, Any]:
        """Handle duplicate document upload."""
        # Insert duplicate record
        doc_id = self.db_manager.insert_document(
            filename=filename,
            username=username,
            document_hash=latest_doc['document_hash'],
            version=latest_doc['version'],
            status="duplicate",
            is_active=False
        )
        
        return {
            "status": "duplicate",
            "message": f"Document already exists (version {latest_doc['version']})",
            "filename": filename,
            "username": username,
            "version": latest_doc['version'],
            "existing_doc_id": latest_doc['id'],
            "duplicate_doc_id": doc_id
        }
    
    def _process_incremental_update(
        self,
        file_path: str,
        filename: str,
        username: str,
        document_hash: str,
        latest_doc: Dict
    ) -> Dict[str, Any]:
        """Process an incremental update with chunk-level diffing."""
        old_version = latest_doc['version']
        new_version = old_version + 1
        
        # Insert new document record
        doc_id = self.db_manager.insert_document(
            filename=filename,
            username=username,
            document_hash=document_hash,
            version=new_version,
            status="processing",
            is_active=True
        )
        
        try:
            # Load and chunk new document
            logger.info("Loading new version...")
            pages_data = self.document_loader.load_pdf(file_path)
            table_strings = self.document_loader.get_table_strings(pages_data)
            
            logger.info("Chunking new version...")
            text_chunks = self.chunker.chunk_text(pages_data)
            table_chunks = self.chunker.chunk_tables(table_strings)
            all_new_chunks = text_chunks + table_chunks
            
            if not all_new_chunks:
                raise ValueError("No chunks created from document")
            
            logger.info(f"Created {len(all_new_chunks)} chunks for new version")
            
            # Hash new chunks
            logger.info("Computing chunk hashes for new version...")
            all_new_chunks = self.chunk_hasher.add_hashes_to_chunks(all_new_chunks)
            new_hash_map = self.chunk_hasher.create_hash_map(all_new_chunks)
            
            # Get old chunk hashes from database
            logger.info(f"Fetching old version chunks (v{old_version})...")
            old_hash_map = self.db_manager.get_chunk_hash_map(latest_doc['id'])
            logger.info(f"Found {len(old_hash_map)} chunks in old version")
            
            # Compute diff
            logger.info("Computing incremental diff...")
            chunks_to_add, chunks_to_delete, unchanged_chunks = self.incremental_diff.compute_diff(
                old_hash_map, new_hash_map
            )
            
            diff_report = self.incremental_diff.create_diff_report(
                chunks_to_add, chunks_to_delete, unchanged_chunks
            )
            logger.info(f"Diff: +{len(chunks_to_add)} -{len(chunks_to_delete)} ={len(unchanged_chunks)}")
            
            # Process additions
            chunks_added = 0
            if chunks_to_add:
                logger.info(f"Adding {len(chunks_to_add)} new chunks...")
                new_chunks_to_process = self.incremental_diff.get_chunks_to_process(
                    chunks_to_add, new_hash_map
                )
                
                # Build metadata for new chunks
                chunks_with_metadata = self.metadata_builder.build_all_metadata(
                    new_chunks_to_process, filename, username, new_version
                )
                
                # Upsert to Pinecone
                chunks_added = self.pinecone_manager.upsert_chunks(chunks_with_metadata, username)
                
                # Insert to database
                chunk_records = []
                for chunk in chunks_with_metadata:
                    chunk_records.append({
                        'document_id': doc_id,
                        'filename': filename,
                        'username': username,
                        'version': new_version,
                        'chunk_id': chunk['id'],
                        'chunk_hash': chunk['chunk_hash'],
                        'page_number': chunk['page_number'],
                        'content_type': chunk['content_type'],
                        'is_active': True
                    })
                
                self.db_manager.insert_chunks_batch(chunk_records)
            
            # Process deletions
            chunks_deleted = 0
            if chunks_to_delete:
                logger.info(f"Deleting {len(chunks_to_delete)} removed chunks...")
                
                # Get chunk IDs to delete from Pinecone
                chunk_ids_to_delete = self.incremental_diff.get_chunk_ids_to_delete(
                    chunks_to_delete, old_hash_map
                )
                
                # Delete from Pinecone
                chunks_deleted = self.pinecone_manager.delete_chunks(chunk_ids_to_delete, username)
                
                # Soft delete in database
                self.db_manager.deactivate_chunks_by_hash(
                    list(chunks_to_delete), latest_doc['id']
                )
            
            # Handle unchanged chunks (no action needed - just log)
            logger.info(f"Skipping {len(unchanged_chunks)} unchanged chunks (no re-embedding)")
            
            # Deactivate old document version
            self.db_manager.deactivate_previous_version(filename, username, old_version)
            
            # Update new document status
            self.db_manager.update_status(doc_id, "processed")
            
            logger.info(f"✅ Incremental update complete: {filename} v{old_version} → v{new_version}")
            
            return {
                "status": "success",
                "message": f"Document updated successfully (v{old_version} → v{new_version})",
                "filename": filename,
                "username": username,
                "old_version": old_version,
                "new_version": new_version,
                "doc_id": doc_id,
                "total_chunks": len(all_new_chunks),
                "text_chunks": len(text_chunks),
                "table_chunks": len(table_chunks),
                "chunks_added": chunks_added,
                "chunks_deleted": chunks_deleted,
                "unchanged_chunks": len(unchanged_chunks),
                "diff_report": diff_report
            }
            
        except Exception as e:
            logger.error(f"Error in incremental update: {e}")
            self.db_manager.update_status(doc_id, "failed")
            raise
    
    def get_user_documents(self, username: str, limit: int = 50) -> list:
        """
        Get all documents for a user.
        
        Args:
            username: Username.
            limit: Maximum number of documents to return.
        
        Returns:
            List of document records.
        """
        return self.db_manager.get_user_documents(username.lower(), limit)
    
    def search_documents(
        self,
        query: str,
        username: str,
        top_k: int = None
    ) -> list:
        """
        Search documents for a user.
        
        Args:
            query: Search query.
            username: Username.
            top_k: Number of results to return.
        
        Returns:
            List of search results.
        """
        if top_k is None:
            top_k = settings.TOP_K
        
        return self.pinecone_manager.search(query, username.lower(), top_k)


if __name__ == "__main__":
    import sys
    
    print("=== Processing Pipeline Test ===\n")
    
    if len(sys.argv) < 3:
        print("Usage: python processing_pipeline.py <pdf_path> <username>")
        print("\nExample:")
        print("  python processing_pipeline.py test.pdf alice")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    username = sys.argv[2]
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        pipeline = ProcessingPipeline()
        result = pipeline.process_document(pdf_path, username)
        
        print("\n" + "=" * 60)
        print("PROCESSING RESULT")
        print("=" * 60)
        
        for key, value in result.items():
            print(f"{key:20s}: {value}")
        
        if result["status"] == "success":
            print("\n✅ Processing completed successfully!")
        elif result["status"] == "duplicate":
            print("\n⚠️  Document is a duplicate")
        else:
            print("\n❌ Processing failed")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
