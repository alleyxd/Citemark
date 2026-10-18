import argparse
import time

from .answer import answer
from .chunk import chunk_folder
from .index import Index


def main():
    ap = argparse.ArgumentParser(prog="citemark", description="Ask questions about a folder of documents.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("index", help="chunk and embed a folder")
    b.add_argument("folder")
    b.add_argument("--out", default=".citemark")
    q = sub.add_parser("ask", help="answer a question from an index")
    q.add_argument("question")
    q.add_argument("--index", default=".citemark")
    q.add_argument("--k", type=int, default=5)
    q.add_argument("--mode", default="hybrid+rerank", choices=["keyword", "dense", "hybrid", "hybrid+rerank"])
    q.add_argument("--show-sources", action="store_true")
    args = ap.parse_args()

    if args.cmd == "index":
        t = time.time()
        chunks = chunk_folder(args.folder)
        Index.build(chunks).save(args.out)
        print(f"indexed {len(chunks)} chunks from {args.folder} in {time.time() - t:.1f}s -> {args.out}")
        return

    idx = Index.load(args.index)
    t = time.time()
    hits = idx.search(args.question, k=args.k, mode=args.mode)
    t_search = time.time() - t
    ans = answer(args.question, hits)
    print(ans.text)
    print()
    for n in ans.cited or range(1, len(ans.sources) + 1) if args.show_sources else ans.cited:
        c = ans.sources[n - 1]
        print(f"[{n}] {c.source} - {c.title}: {c.text[:160]}...")
    print(f"\n(search {t_search * 1000:.0f} ms, total {time.time() - t:.1f}s)")


if __name__ == "__main__":
    main()
