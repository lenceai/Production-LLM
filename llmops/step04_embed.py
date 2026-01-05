"""
step04_embed.py
Version: 1.0
Usage: Generates embeddings for text chunks using a pre-trained model.
"""
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)

def embed_chunks(chunks: List[Dict[str, Any]], embedder: Embedder) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    return embedder.embed_texts(texts)

