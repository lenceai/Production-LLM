"""
utils.py
Version: 1.0
Usage: Utility functions and data classes (Document) shared across steps.
"""
import hashlib
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Document:
    doc_id: str
    source: str
    text: str
    metadata: Dict[str, Any]

def stable_id(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

