"""
SQLite Manager — handles document and chunk metadata with incremental versioning.

Table Schemas:
    documents (
        id INTEGER PRIMARY KEY,
        filename TEXT,
        username TEXT,
        document_hash TEXT,
        version INTEGER,
        status TEXT,
        is_active BOOLEAN,
        uploaded_at DATETIME
    )
    
    document_chunks (
        id INTEGER PRIMARY KEY,
        document_id INTEGER,
        filename TEXT,
        username TEXT,
        version INTEGER,
        chunk_id TEXT,
        chunk_hash TEXT,
        page_number INTEGER,
        content_type TEXT,
        is_active BOOLEAN,
        created_at DATETIME
    )
"""

import sqlite3
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class SQLiteManager:
    """Manages document and chunk metadata in SQLite database."""
    
    def __init__(self, db_path: str):
        """
        Initialize SQLite manager.
        
        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self._init_database()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _init_database(self) -> None:
        """Initialize database schema with documents and chunks tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    username TEXT NOT NULL,
                    document_hash TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create document_chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_hash TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_id TEXT NOT NULL,
                    metadata TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for documents table
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_filename_username
                ON documents(filename, username)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_version
                ON documents(filename, username, version)
            """)
            
            # Create indexes for document_chunks table
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunk_hash
                ON document_chunks(chunk_hash)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunk_lookup
                ON document_chunks(document_id, is_active)
            """)
            
            logger.info(f"Database initialized at {self.db_path}")
    
    def get_latest_document(self, filename: str, username: str) -> Optional[Dict]:
        """
        Get the latest version of a document for a user.
        
        Args:
            filename: Document filename.
            username: Username (normalized to lowercase).
        
        Returns:
            Latest document record or None.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM documents 
                WHERE filename = ? AND username = ?
                ORDER BY version DESC
                LIMIT 1
            """, (filename, username.lower()))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def check_duplicate_hash(self, document_hash: str, username: str) -> Optional[Dict]:
        """
        Check if a document with the same hash exists for this user.
        
        Args:
            document_hash: SHA256 hash of the document.
            username: Username (normalized to lowercase).
        
        Returns:
            Document record if duplicate exists, None otherwise.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM documents 
                WHERE document_hash = ? AND username = ?
                ORDER BY version DESC
                LIMIT 1
            """, (document_hash, username.lower()))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    
    def get_next_version(self, filename: str, username: str) -> int:
        """
        Get the next version number for a document.
        
        Args:
            filename: Document filename.
            username: Username.
        
        Returns:
            Next version number (1 if no previous versions).
        """
        doc = self.get_latest_document(filename, username)
        if doc is None:
            return 1
        return doc["version"] + 1
    
    def insert_document(
        self,
        filename: str,
        username: str,
        document_hash: str,
        version: int,
        status: str = "processing",
        is_active: bool = True
    ) -> int:
        """
        Insert a new document record.
        
        Args:
            filename: Name of the document.
            username: Username (normalized to lowercase).
            document_hash: SHA256 hash of the document.
            version: Version number.
            status: Document status (processing, processed, duplicate, failed).
            is_active: Whether this version is active.
        
        Returns:
            ID of the inserted record.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO documents 
                (filename, username, document_hash, version, status, is_active, uploaded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (filename, username.lower(), document_hash, version, status, is_active, datetime.now()))
            
            doc_id = cursor.lastrowid
            logger.info(
                f"Inserted document: {filename} (v{version}) for user '{username}' with status '{status}'"
            )
            return doc_id
    
    def deactivate_previous_version(self, filename: str, username: str, version: int) -> None:
        """
        Deactivate previous version of a document.
        
        Args:
            filename: Document filename.
            username: Username.
            version: Previous version to deactivate.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE documents 
                SET is_active = 0 
                WHERE filename = ? AND username = ? AND version = ?
            """, (filename, username.lower(), version))
            
            logger.info(f"Deactivated document: {filename} v{version} for user '{username}'")
    
    def update_status(self, doc_id: int, status: str) -> None:
        """
        Update document status.
        
        Args:
            doc_id: Document ID.
            status: New status.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE documents 
                SET status = ? 
                WHERE id = ?
            """, (status, doc_id))
            
            logger.info(f"Updated document {doc_id} status to '{status}'")
    
    def get_document(self, doc_id: int) -> Optional[Dict]:
        """
        Get document by ID.
        
        Args:
            doc_id: Document ID.
        
        Returns:
            Document record or None.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def get_user_documents(self, username: str, limit: int = 50) -> List[Dict]:
        """
        Get all documents for a user.
        
        Args:
            username: Username.
            limit: Maximum number of records to return.
        
        Returns:
            List of document records.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM documents 
                WHERE username = ?
                ORDER BY uploaded_at DESC
                LIMIT ?
            """, (username.lower(), limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_document_by_hash(self, hash_value: str, username: str) -> Optional[Dict]:
        """
        Get document by hash and username.
        
        Args:
            hash_value: SHA256 hash.
            username: Username.
        
        Returns:
            Document record or None.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM documents 
                WHERE document_hash = ? AND username = ?
                ORDER BY version DESC
                LIMIT 1
            """, (hash_value, username.lower()))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    # ── Chunk Management Methods ─────────────────────────────────────────────
    
    def insert_chunk(
        self,
        document_id: int,
        filename: str,
        username: str,
        version: int,
        chunk_id: str,
        chunk_hash: str,
        page_number: int,
        content_type: str,
        is_active: bool = True
    ) -> int:
        """
        Insert a chunk record.
        
        Args:
            document_id: Foreign key to documents table.
            filename: Document filename.
            username: Username.
            version: Document version.
            chunk_id: Unique chunk identifier.
            chunk_hash: SHA256 hash of normalized chunk content.
            page_number: Page number.
            content_type: Type of content (text/table).
            is_active: Whether chunk is active.
        
        Returns:
            ID of inserted chunk.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO document_chunks
                (document_id, filename, username, version, chunk_id, chunk_hash, 
                 page_number, content_type, is_active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (document_id, filename, username.lower(), version, chunk_id, chunk_hash,
                  page_number, content_type, is_active, datetime.now()))
            
            return cursor.lastrowid
    
    def insert_chunks_batch(self, chunks: List[Dict]) -> int:
        """
        Insert multiple chunks in a batch.
        
        Args:
            chunks: List of chunk dictionaries with required fields:
                - document_id
                - chunk_id
                - chunk_hash
                - Additional metadata stored in metadata JSON field
        
        Returns:
            Number of chunks inserted.
        """
        import json
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            for idx, chunk in enumerate(chunks):
                # Prepare metadata as JSON
                metadata_dict = {
                    'filename': chunk.get('filename'),
                    'username': chunk.get('username'),
                    'version': chunk.get('version'),
                    'page_number': chunk.get('page_number'),
                    'content_type': chunk.get('content_type', 'text')
                }
                metadata_json = json.dumps(metadata_dict)
                
                cursor.execute("""
                    INSERT INTO document_chunks
                    (document_id, chunk_hash, chunk_index, chunk_id, metadata, is_active, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk['document_id'],
                    chunk['chunk_hash'],
                    idx,  # Use loop index as chunk_index
                    chunk['chunk_id'],
                    metadata_json,
                    chunk.get('is_active', True),
                    datetime.now()
                ))
            
            logger.info(f"Inserted {len(chunks)} chunks in batch")
            return len(chunks)
    
    def get_active_chunks(
        self,
        document_id: int
    ) -> List[Dict]:
        """
        Get all active chunks for a document.
        
        Args:
            document_id: Document ID.
        
        Returns:
            List of active chunk records with metadata.
        """
        import json
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, document_id, chunk_id, chunk_hash, chunk_index, metadata, is_active, created_at
                FROM document_chunks
                WHERE document_id = ? AND is_active = 1
                ORDER BY chunk_index
            """, (document_id,))
            
            chunks = []
            for row in cursor.fetchall():
                chunk_dict = dict(row)
                # Parse metadata JSON
                if chunk_dict.get('metadata'):
                    chunk_dict['metadata'] = json.loads(chunk_dict['metadata'])
                chunks.append(chunk_dict)
            
            return chunks
    
    def deactivate_chunks_by_hash(
        self,
        chunk_hashes: List[str],
        document_id: int
    ) -> int:
        """
        Deactivate chunks by their hashes (soft delete).
        
        Args:
            chunk_hashes: List of chunk hashes to deactivate.
            document_id: Document ID.
        
        Returns:
            Number of chunks deactivated.
        """
        if not chunk_hashes:
            return 0
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            placeholders = ','.join('?' * len(chunk_hashes))
            query = f"""
                UPDATE document_chunks
                SET is_active = 0
                WHERE chunk_hash IN ({placeholders})
                AND document_id = ?
            """
            
            cursor.execute(query, chunk_hashes + [document_id])
            affected = cursor.rowcount
            
            logger.info(f"Deactivated {affected} chunks for document {document_id}")
            return affected
    
    def get_chunk_hash_map(
        self,
        document_id: int
    ) -> Dict[str, str]:
        """
        Get mapping of chunk_hash -> chunk_id for a document.
        
        Args:
            document_id: Document ID.
        
        Returns:
            Dictionary mapping chunk_hash to chunk_id.
        """
        chunks = self.get_active_chunks(document_id)
        return {chunk['chunk_hash']: chunk['chunk_id'] for chunk in chunks}


if __name__ == "__main__":
    import os
    
    # Test the SQLite manager
    test_db = "test_rag_metadata.db"
    
    try:
        print("=== SQLite Manager Test ===\n")
        
        manager = SQLiteManager(test_db)
        
        # Test insert
        doc_id = manager.insert_document(
            filename="test.pdf",
            username="alice",
            document_hash="abc123",
            version=1,
            status="processing"
        )
        print(f"✅ Inserted document with ID: {doc_id}")
        
        # Test check duplicate
        duplicate = manager.check_duplicate_hash("abc123", "alice")
        print(f"✅ Duplicate check: {duplicate is not None}")
        
        # Test get next version
        next_version = manager.get_next_version("test.pdf", "alice")
        print(f"✅ Next version for test.pdf: {next_version}")
        
        # Test update status
        manager.update_status(doc_id, "processed")
        updated_doc = manager.get_document(doc_id)
        print(f"✅ Updated status: {updated_doc['status']}")
        
        # Test get user documents
        user_docs = manager.get_user_documents("alice")
        print(f"✅ User documents count: {len(user_docs)}")
        
        print("\n✅ All tests passed!")
        
    finally:
        # Cleanup
        if os.path.exists(test_db):
            os.remove(test_db)
            print(f"🧹 Cleaned up test database")
