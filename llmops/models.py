"""
models.py
Version: 1.0
Usage: Defines the Generator interface and a StubGenerator implementation.
"""
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Generation:
    answer: str
    citations: List[Dict[str, Any]]

class Generator:
    def generate(self, question: str, contexts: List[Dict[str, Any]]) -> Generation:
        raise NotImplementedError

class StubGenerator(Generator):
    """Safe default: no external APIs. Produces grounded, citation-style output.
    Replace with OpenAI/Anthropic/vLLM later while keeping the same interface."""
    def generate(self, question: str, contexts: List[Dict[str, Any]]) -> Generation:
        cites = [{"source": c["source"], "chunk_id": c["chunk_id"], "score": c.get("rerank_score", c.get("retrieval_score", 0.0))}
                 for c in contexts]
        bullets = "\n".join([f"- ({i+1}) {c['text'][:220].strip()}..." for i, c in enumerate(contexts)])
        answer = (
            f"Question: {question}\n\n"
            f"Draft grounded summary (stub LLM):\n{bullets}\n\n"
            f"Next: replace StubGenerator with a real LLM; keep citations pipeline unchanged."
        )
        return Generation(answer=answer, citations=cites)

