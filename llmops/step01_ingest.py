"""
step01_ingest.py
Version: 1.1
Usage: Ingests documents (PDF, TXT, MD, HTML) from the raw directory and parses them.
"""
from pathlib import Path
from typing import List
from loguru import logger
from .utils import Document, stable_id
from bs4 import BeautifulSoup  # pip install beautifulsoup4
import pypdf # pip install pypdf

def ingest_directory(raw_dir: Path) -> List[Document]:
    docs: List[Document] = []
    raw_dir = Path(raw_dir)
    logger.info(f"Ingesting from: {raw_dir.resolve()}")

    for p in raw_dir.rglob("*"):
        if p.is_dir():
            continue
        if p.suffix.lower() not in {".txt", ".md", ".html", ".htm", ".pdf"}:
            continue

        text = ""
        try:
            if p.suffix.lower() == ".pdf":
                reader = pypdf.PdfReader(str(p))
                text_parts = []
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text_parts.append(extracted)
                text = "\n".join(text_parts)
            else:
                text = p.read_text(encoding="utf-8", errors="ignore")
                if p.suffix.lower() in {".html", ".htm"}:
                    soup = BeautifulSoup(text, "html.parser")
                    text = soup.get_text(separator="\n")
        except Exception as e:
            logger.error(f"Failed to read file {p}: {e}")
            continue

        if not text.strip():
            logger.warning(f"Skipping empty or unreadable file: {p}")
            continue

        doc_id = stable_id(str(p.resolve()))
        docs.append(Document(
            doc_id=doc_id,
            source=str(p),
            text=text,
            metadata={"filename": p.name, "ext": p.suffix.lower()},
        ))
    logger.info(f"Ingested docs: {len(docs)}")
    return docs
