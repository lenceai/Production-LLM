"""
step03_chunk.py
Version: 1.0
Usage: Chunks documents into smaller segments for embedding.
"""
from typing import List, Dict, Any
from .utils import stable_id

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def chunk_docs(docs, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    rows = []
    for d in docs:
        parts = chunk_text(d.text, chunk_size, overlap)
        for i, c in enumerate(parts):
            rows.append({
                "chunk_id": stable_id(f"{d.doc_id}:{i}"),
                "doc_id": d.doc_id,
                "source": d.source,
                "chunk_index": i,
                "text": c,
                "metadata": d.metadata,
            })
    return rows

