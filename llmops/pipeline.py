"""
pipeline.py
Version: 1.0
Usage: Orchestrates the online RAG pipeline (retrieval, reranking, generation).
"""
import time
from typing import Dict, Any
from loguru import logger

from .config import settings
from .step04_embed import Embedder
from .step05_index import load_index
from .step06_retrieve import retrieve
from .step07_rerank import rerank
from .step08_generate import generate_answer
from .step11_observe import inc_req, inc_fail, LAT
from .step12_canary import choose_variant, annotate_variant
from .models import StubGenerator

class Pipeline:
    def __init__(self, index_dir=settings.index_dir, embed_model=settings.embed_model, canary_ratio: float = 0.0):
        self.embedder = Embedder(embed_model)
        self.index, self.chunks = load_index(index_dir)
        self.generator_stable = StubGenerator()
        self.generator_canary = StubGenerator()  # swap later with new model
        self.canary_ratio = canary_ratio

    def answer(self, question: str) -> Dict[str, Any]:
        inc_req()
        start = time.time()
        try:
            variant = choose_variant(self.canary_ratio)
            gen = self.generator_canary if variant == "canary" else self.generator_stable

            qv = self.embedder.embed_texts([question])[0]
            retrieved = retrieve(self.index, self.chunks, qv, settings.top_k)
            reranked = rerank(question, retrieved, settings.rerank_top_n)
            gen_out = generate_answer(gen, question, reranked)

            payload = {
                "question": question,
                "answer": gen_out.answer,
                "citations": gen_out.citations,
                "top_contexts": [
                    {
                        "source": c["source"],
                        "chunk_id": c["chunk_id"],
                        "retrieval_score": c.get("retrieval_score"),
                        "rerank_score": c.get("rerank_score"),
                    }
                    for c in reranked
                ],
            }
            return annotate_variant(payload, variant)
        except Exception as e:
            inc_fail()
            logger.exception(e)
            raise
        finally:
            LAT.observe(time.time() - start)

