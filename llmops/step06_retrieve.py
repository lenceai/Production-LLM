"""
step06_retrieve.py
Version: 1.0
Usage: Retrieves top-k relevant chunks from the index based on a query.
"""
from typing import List, Dict, Any
import numpy as np

def retrieve(index, chunks: List[Dict[str, Any]], query_vec: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    scores, ids = index.search(query_vec.astype(np.float32), top_k)
    out = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        row = dict(chunks[idx])
        row["retrieval_score"] = float(score)
        out.append(row)
    return out

