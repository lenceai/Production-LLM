"""
step09_evaluate.py
Version: 1.0
Usage: Evaluates the quality of generated answers (starter implementation).
"""
from typing import List, Dict, Any
import re

def groundedness_score(answer: str, contexts: List[Dict[str, Any]]) -> float:
    """Toy metric: fraction of answer tokens that appear in concatenated contexts.
    Replace with stronger eval later (LLM-as-judge, claim-checking, etc.)."""
    ctx = " ".join([c["text"] for c in contexts]).lower()
    toks = re.findall(r"[a-z0-9']+", answer.lower())
    if not toks:
        return 0.0
    hit = sum(1 for t in toks if t in ctx)
    return hit / max(1, len(toks))

def run_eval(cases: List[Dict[str, Any]]) -> Dict[str, float]:
    g = [c["groundedness"] for c in cases]
    return {
        "avg_groundedness": sum(g)/max(1, len(g)),
        "n_cases": float(len(g)),
    }

