"""
step07_rerank.py
Version: 1.0
Usage: Reranks retrieved chunks to improve relevance.
"""
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def rerank(query: str, candidates: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    texts = [c["text"] for c in candidates]
    vect = TfidfVectorizer(stop_words="english")
    X = vect.fit_transform([query] + texts)
    sims = cosine_similarity(X[0:1], X[1:]).flatten()
    scored = []
    for c, s in zip(candidates, sims):
        row = dict(c)
        row["rerank_score"] = float(s)
        scored.append(row)
    scored.sort(key=lambda r: r["rerank_score"], reverse=True)
    return scored[:top_n]

