"""Compare cross-encoder rerankers on the same hybrid candidates (retrieval only)."""
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import CrossEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from citemark.index import Index  # noqa: E402
from evals.run import norm  # noqa: E402

idx = Index.load(Path(__file__).resolve().parent / "index_wiki")
qa = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer", split="test")
spans = [r for r in qa if norm(r["answer"]) not in ("yes", "no") and len(norm(r["answer"])) >= 2]
cands = [idx.hybrid(r["question"], 30) for r in spans]

def recall(order_fn, name):
    hit = {1: 0, 3: 0, 5: 0}
    times = []
    for r, cand in zip(spans, cands):
        t = time.time()
        order = order_fn(r["question"], cand)
        times.append(time.time() - t)
        g = norm(r["answer"])
        first = next((i for i, j in enumerate(order) if g in norm(idx.chunks[j].text)), None)
        for k in hit:
            hit[k] += first is not None and first < k
    print(name, {f"recall@{k}": round(v / len(spans), 3) for k, v in hit.items()},
          "median_ms", round(statistics.median(times) * 1000, 1), flush=True)

recall(lambda q, c: c, "hybrid (no rerank)")
for model in sys.argv[1:]:
    ce = CrossEncoder(model)
    recall(lambda q, c: [c[i] for i in np.argsort(-np.asarray(ce.predict([(q, idx.chunks[j].text) for j in c])))], model)
