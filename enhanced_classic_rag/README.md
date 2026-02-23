# Enhanced Classic RAG with Incremental Updates

A production-ready RAG (Retrieval-Augmented Generation) system with chunk-level incremental diffing, deduplication, versioning, and user namespace isolation.

## 🎯 Features

### Core Capabilities
- **Document Deduplication**: SHA256-based duplicate detection at file level
- **Chunk-Level Incremental Updates**: Only process changed chunks when documents are modified
  - Computes SHA256 hash for each chunk (with text normalization)
  - Set-based diff algorithm (adds, deletes, unchanged)
  - Skips re-embedding unchanged chunks (significant cost savings)
  - Soft deletes for removed chunks
  - Database as source of truth for chunk lifecycle
- **Version Control**: Automatic version tracking for document updates
- **User Namespaces**: Isolated document storage per user in Pinecone
- **Advanced Extraction**: PyMuPDF-based text and table extraction
- **Smart Chunking**: Separate strategies for text (overlapping) and tables
- **Dual-Table Storage**: Documents + Chunks in SQLite with proper foreign keys
- **Deterministic IDs**: Consistent chunk IDs (`{doc}_v{version}_chunk{index}`)
- **Integrated Embedding**: Pinecone's server-side embeddings
- **Reranking**: BGE reranker for improved relevance
- **FastAPI**: RESTful API with automatic documentation
- **Modular Design**: Clean separation of concerns

### Incremental Update Workflow
1. **New Upload**: Compute document hash → chunk → hash each chunk → upsert all to Pinecone → store in DB
2. **Duplicate**: Same document hash detected → skip processing → return existing version info
3. **Update**: Different document hash → load new version → chunk → hash chunks → fetch old chunk hashes from DB → compute set diff:
   - **Chunks to Add**: New chunks that don't exist in old version → embed & upsert to Pinecone → insert to DB
   - **Chunks to Delete**: Old chunks not in new version → delete from Pinecone → soft delete in DB (is_active=0)
   - **Unchanged Chunks**: Chunks with same hash → skip re-processing → no embedding cost
4. **Version Management**: Increment version number → deactivate old version in DB → activate new version

## 🏗️ Architecture

```
enhanced_classic_rag/
├── app/
│   ├── core/
│   │   └── config.py                  # Centralized configuration
│   ├── db/
│   │   └── sqlite_manager.py          # Database operations (documents + chunks)
│   ├── ingestion/
│   │   ├── hash_manager.py            # Document-level file hashing
│   │   ├── chunk_hasher.py            # Chunk-level hashing with normalization
│   │   ├── incremental_diff.py        # Set-based diff algorithm
│   │   ├── document_loader.py         # PDF text & table extraction
│   │   ├── chunker.py                 # Text & table chunking
│   │   └── metadata_builder.py        # Metadata creation
│   ├── vectorstore/
│   │   ├── pinecone_manager.py        # Vector database operations
│   │   └── reranker.py                # Result reranking
│   ├── pipeline/
│   │   └── processing_pipeline.py     # Orchestrates 3-way workflow (new/duplicate/incremental)
│   └── generation.py                  # LLM answer generation
├── api.py                              # FastAPI application
├── main.py                             # CLI entry point
├── test_incremental.py                 # Test suite for incremental updates
├── requirements.txt
├── .env.example
└── README.md
```

## 📋 Prerequisites

- Python 3.9+
- Pinecone account with API key
- Groq API key

## 🚀 Installation

1. **Clone or navigate to the directory:**
   ```bash
   cd enhanced_classic_rag
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## ⚙️ Configuration

Edit `.env` file:

```bash
# Required
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key

# Optional (defaults provided)
PINECONE_INDEX_NAME=enhanced-rag
EMBEDDING_MODEL=multilingual-e5-large
TEXT_CHUNK_SIZE=500
TEXT_CHUNK_OVERLAP=80
TABLE_CHUNK_SIZE=800
TOP_K=10
RERANK_TOP_N=5
```

## 🎮 Usage

### Start API Server

```bash
python main.py serve
```

Access:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### CLI Commands

**Ingest a document:**
```bash
python main.py ingest docs/report.pdf alice
```

**Search documents:**
```bash
python main.py search "revenue growth" alice
```

**Ask questions:**
```bash
python main.py ask "What was the Q4 revenue?" alice
```

### API Endpoints

#### POST /ingest
Ingest a PDF document with deduplication.

```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@document.pdf" \
  -F "us (New Upload):
```json
{
  "status": "success",
  "message": "Document processed successfully",
  "filename": "document.pdf",
  "username": "alice",
  "version": 1,
  "total_chunks": 45,
  "text_chunks": 38,
  "table_chunks": 7,
  "chunks_added": 45,
  "chunks_deleted": 0,
  "unchanged_chunks": 0
}
```

Response (Incremental Update):
```json
{
  "status": "success",
  "message": "Document updated successfully (incremental)",
  "filename": "document.pdf",
  "username": "alice",
  "old_version": 1,
  "new_version": 2,
  "total_chunks": 48,
  "text_chunks": 40,
  "table_chunks": 8,
  "chunks_added": 8,
  "chunks_deleted": 5,
  "unchanged_chunks": 40,
  "diff_report": {
    "total_old_chunks": 45,
    "total_new_chunks": 48,
    "chunks_added": 8,
    "chunks_deleted": 5,
    "unchanged_chunks": 40,
    "efficiency_gain": "83.3% chunks skipped"
  }
}
```

Response (Duplicate):
```json
{
  "status": "duplicate",
  "message": "Document already exists with same hash",
  "filename": "document.pdf",
  "username": "alice",
  "version": 1
  "chunks_created": 45,
  "text_chunks": 38,
  "table_chunks": 7
}
```

#### POST /chat
Chat with documents using RAG.

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was the revenue?",
    "username": "alice",
    "use_reranker": true,
    "top_k": 10,
    "top_n": 5
  }'
```

Response:
```json
{
  "question": "What was the revenue?",
  "answer": "According to the documents...",
  "username": "alice",
  "citations": [
    {
      "number": 1,
      "source": "report.pdf",
      "page": 5,
      "version": 1,
      "content_type": "text",
      "score": 0.92
    }
  ],
  "reranked": true
}
```

#### POST /search
Search without answer generation.

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "revenue",
    "username": "alice",
    "top_k": 10
  }'
```

#### GET /documents/{username}
Get user's document history.

```bash
curl "http://localhost:8000/documents/alice"
```

## 🔄 Processing Workflow

### New Document Upload
1. **Upload** → User uploads PDF with username
2. **Hash** → Compute SHA256 hash of file
3. **Check** → Query SQLite for file-level duplicates
4. **Version** → Assign version number (always starts at 1)
5. **Extract** → PyMuPDF extracts text + tables
6. **Chunk** → Separate chunking strategies for text and tables
7. **Hash Chunks** → Compute SHA256 for each chunk (with normalization)
8. **Metadata** → Build structured metadata with chunk hashes
9. **Embed** → Pinecone integrated embedding (server-side)
10. **Upsert** → Store all chunks in user's namespace
11. **Track** → Insert document + chunk records in SQLite

### Incremental Update
1. **Upload** → User uploads modified PDF with same filename
2. **Hash** → Compute SHA256 hash of new file
3. **Detect Update** → Different file hash → trigger incremental processing
4. **Increment Version** → v2, v3, etc.
5. **Extract & Chunk** → Process new version
6. **Hash Chunks** → Compute SHA256 for each new chunk
7. **Fetch Old Hashes** → Get chunk hashes from DB for old version
8. **Compute Diff** → Set operations:
   - `chunks_to_add = new_hashes - old_hashes`
   - `chunks_to_delete = old_hashes - new_hashes`
   - `unchanged_chunks = new_hashes ∩ old_hashes`
9. **Process Additions** → Embed & upsert only new chunks to Pinecone
10. **Process Deletions** → Delete removed chunks from Pinecone
11. **Skip Unchanged** → No re-embedding for unchanged chunks (cost savings)
12. **Update DB** → Insert new chunk records, soft delete old chunks, deactivate old version

### Duplicate Detection
1. **Upload** → User uploads same file again
2. **Hash** → Compute SHA256 hash
3. **Match** → Same file hash found in DB
4. **Skip** → Return existing document info, no processing

## 📊 Database Schema

**SQLite Table: `documents`**

| Column          | Type     | Description                          |
|-----------------|----------|--------------------------------------|
| id              | INTEGER  | Primary key                          |
| filename        | TEXT     | Document name                        |
| username        | TEXT     | User (lowercase)                     |
| document_hash   | TEXT     | SHA256 hash of file                  |
| version         | INTEGER  | Version number                       |
| status          | TEXT     | processing/processed/duplicate/failed|
| is_active       | BOOLEAN  | Whether version is active (1/0)      |
| uploaded_at     | DATETIME | Upload timestamp                     |

**Indexes**: `(filename, username)`, `(filename, username, version)`

**SQLite Table: `document_chunks`**

| Column          | Type     | Description                          |
|-----------------|----------|--------------------------------------|
| id              | INTEGER  | Primary key                          |
| document_id     | INTEGER  | Foreign key to documents.id          |
| chunk_hash      | TEXT     | SHA256 hash of chunk (normalized)    |
| chunk_index     | INTEGER  | Chunk position in document           |
| chunk_id        | TEXT     | Deterministic ID for Pinecone        |
| metadata        | TEXT     | JSON metadata (type, text, etc.)     |
| is_active       | BOOLEAN  | Whether chunk is active (1/0)        |
| created_at      | DATETIME | Creation timestamp                   |

**Indexes**: `(chunk_hash)`, `(document_id, is_active)`

**Foreign Key**: `document_id` references `documents(id)` with CASCADE delete

## 🔑 Key Design Decisions

1. **Chunk-Level Incremental Updates**: Only re-process changed chunks
   - Compute SHA256 hash for each chunk (after text normalization)
   - Store chunk hashes in database as source of truth
   - Set-based diff algorithm for efficiency
   - Significant cost savings on unchanged content
   
2. **Database as Source of Truth**: SQLite stores all chunk metadata
   - Dual-table design (documents + document_chunks)
   - Pinecone only used for embeddings/retrieval
   - Enables efficient diff computation without querying Pinecone
   - Soft deletes (is_active flag) for audit trail
   
3. **Text Normalization for Hashing**: Consistent chunk identification
   - Lowercase conversion
   - Whitespace collapse (multiple spaces → single space)
   - Ensures same semantic content produces same hash
   
4. **Three Processing Paths**:
   - **New**: Full ingestion pipeline
   - **Duplicate**: Skip processing (same file hash)
   - **Incremental**: Diff-based selective update
   
5. **Username-based Namespaces**: Each user has isolated document storage

6. **Version Tracking**: Updates create new versions, old versions deactivated

7. **Deterministic Chunk IDs**: `{filename}_v{version}_chunk{index}` enables reproducibility

8. **Separate Text/Table Chunking**: Optimized strategies for each content type

9. **Config-Driven**: All settings centralized in config.py

10. **Dependency Injection**: Testable, modular components

## 🧪 Testing

### Test Individual Modules

```bash
# Test SQLite manager (with dual tables)
python -m app.db.sqlite_manager

# Test document loader
python -m app.ingestion.document_loader docs/sample.pdf

# Test chunker
python -m app.ingestion.chunker

# Test file hash manager
python -m app.ingestion.hash_manager

# Test chunk hasher (with normalization)
python -m app.ingestion.chunk_hasher

# Test incremental diff algorithm
python -m app.ingestion.incremental_diff
```

### Test Incremental Update System

```bash
# Run comprehensive incremental update tests
python test_incremental.py
```

This test suite validates:
1. First upload (new document)
2. Database records (documents + chunks)
3. Duplicate detection (same file)
4. Incremental updates (modified document)
5. Diff statistics (adds, deletes, unchanged)
6. Search/retrieval functionality

## 🛠️ Development

The system is designed with clean separation:

- **Core**: Configuration and constants
- **DB**: Database operations
- **Ingestion**: Document processing
- **Vectorstore**: Pinecone operations
- **Pipeline**: Workflow orchestration
- **Generation**: LLM integration

Each module is independently testable with a `__main__` block.

## 📝 Logging

Logs include:
- Document processing steps
- Duplicate detection
- Chunk creation counts
- Pinecone operations
- Error details

Configure logging level in code or via environment.

## 🔒 Security Considerations

- API keys stored in `.env` (gitignored)
- Username normalization prevents case mismatches
- File validation (PDF only)
- Temporary file cleanup
- SQLite connection management

## 🚨 Error Handling

- Duplicate detection stops reprocessing
- Failed documents marked in database
- Temporary files cleaned up on error
- HTTP exceptions with clear messages
- Detailed error logging

## 📈 Scalability & Performance

- **Chunk-Level Incremental Updates**: Massive cost savings for document updates
  - Only embed/upsert changed chunks
  - Skip unchanged content (no redundant API calls)
  - Example: 1000-chunk document with 50 changes → 95% reduction in embedding costs
- **Set-Based Diff**: Efficient in-memory computation
- **Batch Operations**: Batch upserts and deletes to Pinecone
- **Configurable Batch Sizes**: Tune for your workload
- **Namespace Isolation**: Per-user namespaces for multi-tenancy
- **Efficient SQLite Indexing**: Fast lookups on filename, username, chunk hash
- **Stateless API Design**: Horizontal scaling possible
- **Soft Deletes**: Audit trail without complexity

## 🤝 Contributing

This is a production-ready template. Customize:

- Add authentication/authorization
- Implement document deletion
- Add document update endpoints
- Extend to support more file types
- Add caching layer
- Implement rate limiting

## 📄 License

MIT License - feel free to use and modify.

## 🙏 Acknowledgments

Built with:
- FastAPI
- Pinecone
- PyMuPDF
- LangChain
- Groq

---

**Need help?** Check the API docs at `/docs` or open an issue.
