"""
step02_clean.py
Version: 1.0
Usage: Cleans and normalizes text from ingested documents.
"""
import re
from typing import List
from .utils import Document

_whitespace = re.compile(r"[ \t]+")
_multinew = re.compile(r"\n{3,}")

def clean_docs(docs: List[Document]) -> List[Document]:
    out: List[Document] = []
    for d in docs:
        t = d.text.replace("\r\n", "\n").replace("\r", "\n")
        t = _whitespace.sub(" ", t)
        t = _multinew.sub("\n\n", t)
        t = t.strip()
        out.append(Document(d.doc_id, d.source, t, d.metadata))
    return out

