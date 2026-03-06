"""
main.py — Entry point for the Enhanced RAG Cache API server.

Usage:
    python main.py
    # or
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
