"""
main_build.py
Version: 1.0
Usage: Script to run the offline build process (ingest to index).
"""
from loguru import logger

from llmops.config import settings
from llmops.step01_ingest import ingest_directory
from llmops.step02_clean import clean_docs
from llmops.step03_chunk import chunk_docs
from llmops.step04_embed import Embedder, embed_chunks
from llmops.step05_index import build_faiss_index, save_index

def build():
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)

    docs = ingest_directory(settings.raw_dir)
    docs = clean_docs(docs)
    chunks = chunk_docs(docs, settings.chunk_size, settings.chunk_overlap)

    logger.info(f"Chunks: {len(chunks)}")
    embedder = Embedder(settings.embed_model)
    vecs = embed_chunks(chunks, embedder)

    index = build_faiss_index(vecs)
    save_index(settings.index_dir, index, chunks)
    logger.info(f"Index built at: {settings.index_dir.resolve()}")

if __name__ == "__main__":
    build()

