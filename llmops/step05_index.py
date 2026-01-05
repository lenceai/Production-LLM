"""
step05_index.py
Version: 1.0
Usage: Builds and saves a FAISS index and metadata for the chunks.
"""
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import numpy as np
import faiss

def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    # cosine similarity via inner product; vectors normalized above
    dim = vectors.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(vectors)
    return idx

def save_index(index_dir: Path, index, chunks: List[Dict[str, Any]]):
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_dir / "faiss.index"))
    with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for row in chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_index(index_dir: Path) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    index_dir = Path(index_dir)
    index = faiss.read_index(str(index_dir / "faiss.index"))
    chunks = []
    with open(index_dir / "chunks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return index, chunks

