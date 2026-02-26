# PageIndex RAG – Frontend

## Quick Start

### 1 · Set environment variables

Create `rag_pageindex/.env`:
```
PAGEINDEX_API_KEY=your_pageindex_api_key
GROQ_API_KEY=your_groq_api_key
```

### 2 · Start the FastAPI backend (port 8001)

```bash
cd rag_pageindex
uv run uvicorn api:app --port 8001 --reload
```

### 3 · Start the Streamlit frontend (separate terminal)

```bash
cd rag_pageindex/frontend
pip install -r requirements.txt
streamlit run app.py
```

The UI opens at **http://localhost:8501**

---

## Features

| Feature | Description |
|---------|-------------|
| 📤 Upload PDF | Upload any PDF; automatically re-uses existing index if already submitted |
| 📚 Document list | Browse all completed indexes; switch active document with one click |
| 💬 Chat | Ask questions in a chat interface powered by Groq LLM |
| 🧠 Reasoning | Expand the *Reasoning* panel to see how the model selected nodes |
| 📌 Source Nodes | See exactly which tree nodes were used to answer the question |
| 🌲 Tree Viewer | Visual hierarchy of the document; toggle Raw JSON for the full payload |
| 📄 PDF Preview | Inline PDF preview in the sidebar |
| 💾 Local cache | `doc_id_cache.json` + `output/<doc_id>/` saves tree, node search and answers |
