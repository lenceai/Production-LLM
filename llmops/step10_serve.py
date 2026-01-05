"""
step10_serve.py
Version: 1.0
Usage: FastAPI application to serve the RAG pipeline.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from .pipeline import Pipeline

class AskReq(BaseModel):
    question: str

def create_app(pipeline: Pipeline) -> FastAPI:
    app = FastAPI(title="Capstone LLMOps RAG API")

    @app.post("/ask")
    def ask(req: AskReq):
        result = pipeline.answer(req.question)
        return result

    @app.get("/health")
    def health():
        return {"ok": True}

    return app

