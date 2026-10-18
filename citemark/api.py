"""HTTP API: POST /ask {"question": "...", "k": 5}. Start with: uvicorn citemark.api:app"""
import os
import time

from fastapi import FastAPI
from pydantic import BaseModel

from .answer import answer
from .index import Index

app = FastAPI(title="citemark")
index = Index.load(os.environ.get("CITEMARK_INDEX", ".citemark"))


class Ask(BaseModel):
    question: str
    k: int = 5
    mode: str = "hybrid+rerank"


@app.post("/ask")
def ask(req: Ask):
    t = time.time()
    hits = index.search(req.question, k=req.k, mode=req.mode)
    t_search = time.time() - t
    ans = answer(req.question, hits)
    return {
        "answer": ans.text,
        "citations": [{"n": n, "source": ans.sources[n - 1].source, "title": ans.sources[n - 1].title,
                       "text": ans.sources[n - 1].text} for n in ans.cited],
        "timing_ms": {"search": round(t_search * 1000), "total": round((time.time() - t) * 1000)},
    }


@app.get("/health")
def health():
    return {"chunks": len(index.chunks), "embed_model": index.embed_model}
