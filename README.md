# Capstone: Production LLM + LLMOps Framework (Starter Kit)

This README contains:
- A **useful capstone project outline** (12-step end-to-end LLM/LLMOps pipeline)
- An **example reference architecture**
- **Starter code** implementing the 12 steps (offline build + online RAG query path)
- A runnable **FastAPI** service, basic **observability**, and **canary** routing

You can paste this whole README into a coding tool and have it generate the repo/files.

---

## 0) Capstone Project Outline (Staff-level)

### Title (recommended)
**Designing, Evaluating, and Operating Production LLM Systems: An End-to-End LLM + LLMOps Framework for Reliability, Cost, and Accuracy**

### Problem statement
LLM systems often:
- look good in demos but fail in production,
- lack rigorous evaluation,
- are expensive and unstable at scale,
- have weak observability and unsafe rollouts.

This capstone builds a **reference-quality** end-to-end framework tying together:
- Model strategy & tradeoffs
- RAG design + failure modes
- Evaluation methodology (human + automated)
- LLMOps (versioning, canary, rollback, drift)
- Cost/latency optimization
- Governance/security controls

### Key constraints (explicitly list in your report)
- proprietary/multi-tenant data isolation
- latency SLOs (p95 target)
- cost budget per 1k requests
- privacy / logging limits
- safe rollouts (canary + rollback)
- evaluation regression gates

### Deliverables (what you submit)
1) Reference architecture + reasoning
2) Working pipeline (12 steps) + code
3) Evaluation harness + baseline metrics
4) Observability + release workflow (canary/rollback)
5) Results summary + tradeoffs + lessons learned
6) Reusable artifacts: checklists, decision matrices, “LLMOps maturity model” (optional)

---

## 1) Example Reference Architecture

```
                ┌────────────────────────────────────────────┐
                │                 Data Sources                │
                │  PDFs | HTML | MD | Confluence | S3 | DB    │
                └───────────────┬────────────────────────────┘
                                │
                        (1) Ingest + Parse
                                │
                        (2) Clean + Normalize
                                │
                        (3) Chunk (semantic / fixed)
                                │
                ┌───────────────┴────────────────────────────┐
                │            Embeddings + Indexing            │
                │ (4) Embed -> (5) FAISS index + metadata     │
                └───────────────┬────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │  Online Query Path     │
                    │                        │
           (6) Retrieve top-k          (11) Observability
                    │                        │
             (7) Rerank top-n                │
                    │                        │
             (8) Generate answer             │
                    │                        │
          (9) Evaluate + regression tests    │
                    │                        │
             (10) Serve via FastAPI  ────────┘
                    │
             (12) Canary + rollback
```

---

## 2) 12 Steps (Pipeline Map)

### Offline build path (Steps 1–5)
1. Ingest + parse documents
2. Clean + normalize
3. Chunk
4. Embed chunks
5. Build + save index (FAISS) and chunk metadata

### Online query path (Steps 6–12)
6. Retrieve top-k
7. Rerank top-n
8. Generate grounded answer (pluggable LLM)
9. Evaluate / regression gates (toy metric + extend later)
10. Serve via API (FastAPI)
11. Observability (Prometheus metrics)
12. Canary routing + rollback strategy (starter)

---

## 3) Setup

### Install dependencies
```bash
pip install -U fastapi uvicorn pydantic loguru numpy scikit-learn sentence-transformers faiss-cpu prometheus-client pytest beautifulsoup4
```

### Project layout
```
capstone_llmops/
  config.py
  step01_ingest.py
  step02_clean.py
  step03_chunk.py
  step04_embed.py
  step05_index.py
  step06_retrieve.py
  step07_rerank.py
  step08_generate.py
  step09_evaluate.py
  step10_serve.py
  step11_observe.py
  step12_canary.py
  pipeline.py
  models.py
  utils.py
  main_build.py
  main_api.py
tests/
  test_regression.py
data/
  raw/
  processed/
  index/
```

---

## 4) Starter Code (Copy into files exactly)

### `capstone_llmops/config.py`
```python
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
```

### `capstone_llmops/utils.py`
```python
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
```

### `capstone_llmops/step01_ingest.py` (1) Ingest + parse
```python
from pathlib import Path
from typing import List
from loguru import logger
from .utils import Document, stable_id
from bs4 import BeautifulSoup  # pip install beautifulsoup4

def ingest_directory(raw_dir: Path) -> List[Document]:
    docs: List[Document] = []
    raw_dir = Path(raw_dir)
    logger.info(f"Ingesting from: {raw_dir.resolve()}")

    for p in raw_dir.rglob("*"):
        if p.is_dir():
            continue
        if p.suffix.lower() not in {".txt", ".md", ".html", ".htm"}:
            continue

        text = p.read_text(encoding="utf-8", errors="ignore")
        if p.suffix.lower() in {".html", ".htm"}:
            soup = BeautifulSoup(text, "html.parser")
            text = soup.get_text(separator="\n")

        doc_id = stable_id(str(p.resolve()))
        docs.append(Document(
            doc_id=doc_id,
            source=str(p),
            text=text,
            metadata={"filename": p.name, "ext": p.suffix.lower()},
        ))
    logger.info(f"Ingested docs: {len(docs)}")
    return docs
```

### `capstone_llmops/step02_clean.py` (2) Clean + normalize
```python
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
```

### `capstone_llmops/step03_chunk.py` (3) Chunking
```python
from typing import List, Dict, Any
from .utils import stable_id

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def chunk_docs(docs, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    rows = []
    for d in docs:
        parts = chunk_text(d.text, chunk_size, overlap)
        for i, c in enumerate(parts):
            rows.append({
                "chunk_id": stable_id(f"{d.doc_id}:{i}"),
                "doc_id": d.doc_id,
                "source": d.source,
                "chunk_index": i,
                "text": c,
                "metadata": d.metadata,
            })
    return rows
```

### `capstone_llmops/step04_embed.py` (4) Embeddings
```python
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)

def embed_chunks(chunks: List[Dict[str, Any]], embedder: Embedder) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    return embedder.embed_texts(texts)
```

### `capstone_llmops/step05_index.py` (5) FAISS index + metadata store
```python
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import numpy as np
import faiss

def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    # cosine similarity via inner product; vectors normalized above
    dim = vectors.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(vectors)
    return idx

def save_index(index_dir: Path, index, chunks: List[Dict[str, Any]]):
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_dir / "faiss.index"))
    with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for row in chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_index(index_dir: Path) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    index_dir = Path(index_dir)
    index = faiss.read_index(str(index_dir / "faiss.index"))
    chunks = []
    with open(index_dir / "chunks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return index, chunks
```

### `capstone_llmops/step06_retrieve.py` (6) Retrieve top-k
```python
from typing import List, Dict, Any
import numpy as np

def retrieve(index, chunks: List[Dict[str, Any]], query_vec: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    scores, ids = index.search(query_vec.astype(np.float32), top_k)
    out = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        row = dict(chunks[idx])
        row["retrieval_score"] = float(score)
        out.append(row)
    return out
```

### `capstone_llmops/step07_rerank.py` (7) Rerank
```python
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def rerank(query: str, candidates: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    texts = [c["text"] for c in candidates]
    vect = TfidfVectorizer(stop_words="english")
    X = vect.fit_transform([query] + texts)
    sims = cosine_similarity(X[0:1], X[1:]).flatten()
    scored = []
    for c, s in zip(candidates, sims):
        row = dict(c)
        row["rerank_score"] = float(s)
        scored.append(row)
    scored.sort(key=lambda r: r["rerank_score"], reverse=True)
    return scored[:top_n]
```

### `capstone_llmops/models.py` (Generator interface)
```python
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
    \"\"\"Safe default: no external APIs. Produces grounded, citation-style output.
    Replace with OpenAI/Anthropic/vLLM later while keeping the same interface.\"\"\"
    def generate(self, question: str, contexts: List[Dict[str, Any]]) -> Generation:
        cites = [{"source": c["source"], "chunk_id": c["chunk_id"], "score": c.get("rerank_score", c.get("retrieval_score", 0.0))}
                 for c in contexts]
        bullets = "\\n".join([f"- ({i+1}) {c['text'][:220].strip()}..." for i, c in enumerate(contexts)])
        answer = (
            f"Question: {question}\\n\\n"
            f"Draft grounded summary (stub LLM):\\n{bullets}\\n\\n"
            f"Next: replace StubGenerator with a real LLM; keep citations pipeline unchanged."
        )
        return Generation(answer=answer, citations=cites)
```

### `capstone_llmops/step08_generate.py` (8) Generate
```python
from typing import List, Dict, Any
from .models import Generator, Generation

def generate_answer(generator: Generator, question: str, contexts: List[Dict[str, Any]]) -> Generation:
    return generator.generate(question, contexts)
```

### `capstone_llmops/step09_evaluate.py` (9) Evaluate (starter)
```python
from typing import List, Dict, Any
import re

def groundedness_score(answer: str, contexts: List[Dict[str, Any]]) -> float:
    \"\"\"Toy metric: fraction of answer tokens that appear in concatenated contexts.
    Replace with stronger eval later (LLM-as-judge, claim-checking, etc.).\"\"\"
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
```

### `capstone_llmops/step10_serve.py` (10) FastAPI
```python
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
```

### `capstone_llmops/step11_observe.py` (11) Metrics (Prometheus)
```python
from prometheus_client import Counter, Histogram

REQS = Counter("rag_requests_total", "Total RAG requests")
FAILS = Counter("rag_failures_total", "Total RAG failures")
LAT = Histogram("rag_latency_seconds", "RAG latency seconds")

def inc_req(): REQS.inc()
def inc_fail(): FAILS.inc()
```

### `capstone_llmops/step12_canary.py` (12) Canary routing (starter)
```python
import random
from typing import Dict, Any

def choose_variant(canary_ratio: float) -> str:
    # canary_ratio: 0.0..1.0
    return "canary" if random.random() < canary_ratio else "stable"

def annotate_variant(payload: Dict[str, Any], variant: str) -> Dict[str, Any]:
    out = dict(payload)
    out["variant"] = variant
    return out
```

### `capstone_llmops/pipeline.py` (Online orchestration)
```python
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
```

### `capstone_llmops/main_build.py` (Offline build runner: steps 1–5)
```python
from loguru import logger

from capstone_llmops.config import settings
from capstone_llmops.step01_ingest import ingest_directory
from capstone_llmops.step02_clean import clean_docs
from capstone_llmops.step03_chunk import chunk_docs
from capstone_llmops.step04_embed import Embedder, embed_chunks
from capstone_llmops.step05_index import build_faiss_index, save_index

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
```

### `capstone_llmops/main_api.py` (API runner: step 10)
```python
import uvicorn
from capstone_llmops.config import settings
from capstone_llmops.pipeline import Pipeline
from capstone_llmops.step10_serve import create_app

def run():
    pipeline = Pipeline(canary_ratio=0.1)  # 10% canary traffic
    app = create_app(pipeline)
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)

if __name__ == "__main__":
    run()
```

### `tests/test_regression.py` (Regression test starter)
```python
from capstone_llmops.pipeline import Pipeline

def test_smoke():
    p = Pipeline(canary_ratio=0.0)
    r = p.answer("What is this repository about?")
    assert isinstance(r["answer"], str)
    assert "variant" in r
```

---

## 5) Run Instructions

1) Put a few `.txt/.md/.html` files into `data/raw/`

2) Build the index:
```bash
python -m capstone_llmops.main_build
```

3) Start the API:
```bash
python -m capstone_llmops.main_api
```

4) Query:
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize the docs"}'
```

---

## 6) Recommended Next Upgrades (for capstone-grade rigor)

1) Replace `StubGenerator` with:
   - **local vLLM** (recommended), or
   - **TGI**, or
   - OpenAI/Anthropic APIs.
2) Add a real evaluation set:
   - 50–200 questions
   - expected citations / grounding checks
   - regression gates before deploy
3) Add observability/tracing:
   - structured logs for query → retrieval → answer
   - privacy controls + redaction
4) Canary comparison:
   - stable vs canary on the same sampled prompts
   - automatic rollback thresholds
5) Cost/performance accounting:
   - tokens/request
   - $/1k requests
   - p95 latency & throughput

---

## 7) Notes for your capstone paper

In your write-up:
- include decision matrices (model choice, chunking choice, reranking choice)
- include failure mode analysis (hallucination sources, retrieval failures, stale docs)
- include evaluation methodology + uncertainty (CIs, drift detection)
- show “before/after” metrics and tradeoffs

---
