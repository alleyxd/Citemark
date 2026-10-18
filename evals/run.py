"""Benchmark on rag-mini-wikipedia (3,200 passages, 918 questions).

Retrieval: recall@k = share of questions where some retrieved passage contains the gold answer
           (yes/no questions are skipped here, since "yes" appears everywhere).
Answers:   share of questions whose answer contains the gold answer, closed-book vs with retrieval.

python evals/run.py [--answers 200]
"""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from citemark.answer import answer, client  # noqa: E402
from citemark.chunk import Chunk  # noqa: E402
from citemark.index import Index  # noqa: E402

HERE = Path(__file__).resolve().parent
MODES = ["keyword", "dense", "hybrid", "hybrid+rerank"]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def correct(pred: str, gold: str) -> bool:
    p, g = norm(pred), norm(gold)
    if g in ("yes", "no"):
        return p.split()[:1] == [g]
    return bool(g) and g in p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", type=int, default=200)
    args = ap.parse_args()

    passages = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus", split="passages")
    qa = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer", split="test")
    store = HERE / "index_wiki"
    if (store / "vectors.npy").exists():
        idx = Index.load(store)
    else:
        chunks = [Chunk(i, f"passage {r['id']}", "", r["passage"]) for i, r in enumerate(passages)]
        t = time.time()
        idx = Index.build(chunks)
        idx.save(store)
        print(f"embedded {len(chunks)} passages in {time.time() - t:.0f}s")

    spans = [r for r in qa if norm(r["answer"]) not in ("yes", "no") and len(norm(r["answer"])) >= 2]
    retrieval = {}
    for mode in MODES:
        hits_at = {1: 0, 3: 0, 5: 0, 10: 0}
        times = []
        for r in spans:
            t = time.time()
            hits = idx.search(r["question"], k=10, mode=mode)
            times.append(time.time() - t)
            g = norm(r["answer"])
            first = next((i for i, (c, _) in enumerate(hits) if g in norm(c.text)), None)
            for k in hits_at:
                hits_at[k] += first is not None and first < k
        retrieval[mode] = {f"recall@{k}": round(v / len(spans), 3) for k, v in hits_at.items()}
        retrieval[mode]["median_ms"] = round(statistics.median(times) * 1000, 1)
        print(mode, retrieval[mode], flush=True)

    llm = client()
    sample = list(qa)[: args.answers]
    rows = []
    for r in sample:
        closed = llm.chat.completions.create(
            model="local", temperature=0, extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            messages=[{"role": "user", "content": r["question"] + " Answer briefly."}]).choices[0].message.content
        t = time.time()
        rag = answer(r["question"], idx.search(r["question"], k=5))
        rows.append({"q": r["question"], "gold": r["answer"], "closed": closed, "rag": rag.text,
                     "closed_ok": correct(closed, r["answer"]), "rag_ok": correct(rag.text, r["answer"]),
                     "cited": bool(rag.cited), "idk": rag.text.startswith("I don't know"),
                     "seconds": round(time.time() - t, 2)})
    answers = {
        "n": len(rows),
        "closed_book_accuracy": round(statistics.mean(r["closed_ok"] for r in rows), 3),
        "rag_accuracy": round(statistics.mean(r["rag_ok"] for r in rows), 3),
        "rag_cites_sources": round(statistics.mean(r["cited"] for r in rows), 3),
        "rag_says_dont_know": round(statistics.mean(r["idk"] for r in rows), 3),
        "rag_median_seconds": statistics.median(r["seconds"] for r in rows),
    }
    print(answers)
    out = HERE / "results.json"
    out.write_text(json.dumps({"retrieval_questions": len(spans), "retrieval": retrieval, "answers": answers,
                               "answer_rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
