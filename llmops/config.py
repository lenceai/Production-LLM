"""
config.py
Version: 1.0
Usage: Configuration settings for the LLMOps pipeline, including paths and model parameters.
"""
from pydantic import BaseModel
from pathlib import Path

class Settings(BaseModel):
    project_root: Path = Path(".")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    index_dir: Path = Path("data/index")

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval
    top_k: int = 20
    rerank_top_n: int = 8

    # Embeddings
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Serving
    api_host: str = "0.0.0.0"
    api_port: int = 8000

settings = Settings()

