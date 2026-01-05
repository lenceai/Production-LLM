"""
step12_canary.py
Version: 1.0
Usage: Implements canary routing logic for testing new variants.
"""
import random
from typing import Dict, Any

def choose_variant(canary_ratio: float) -> str:
    # canary_ratio: 0.0..1.0
    return "canary" if random.random() < canary_ratio else "stable"

def annotate_variant(payload: Dict[str, Any], variant: str) -> Dict[str, Any]:
    out = dict(payload)
    out["variant"] = variant
    return out

