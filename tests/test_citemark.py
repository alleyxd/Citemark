import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from citemark.answer import answer  # noqa: E402
from citemark.bm25 import BM25, tokens  # noqa: E402
from citemark.chunk import Chunk, split  # noqa: E402
from citemark.index import Index  # noqa: E402


def test_split_keeps_headings_and_overlaps_long_text():
    text = "# Returns\n\nShort paragraph.\n\n# Shipping\n\n" + " ".join(f"w{i}" for i in range(500))
    pieces = split(text, max_words=200, overlap=50)
    assert pieces[0] == ("Returns", "Short paragraph.")
    shipping = [t for h, t in pieces if h == "Shipping"]
    assert len(shipping) == 3
    assert shipping[0].split()[-50:] == shipping[1].split()[:50]   # overlap carried over


def test_bm25_ranks_matching_document_first():
    bm = BM25(["the cat sat on the mat", "dogs chase cars", "a cat and a dog"])
    s = bm.scores("cat mat")
    assert int(np.argmax(s)) == 0 and s[1] == 0
    assert "the" not in tokens("The Cat")


def fake_index():
    chunks = [Chunk(0, "a.md", "Returns", "Footwear can be returned within 60 days."),
              Chunk(1, "b.md", "Shipping", "Orders over $150 ship free."),
              Chunk(2, "c.md", "Stoves", "The stove is not allowed on flights.")]
    vecs = np.eye(3, dtype=np.float32)
    idx = Index(chunks, vecs)
    idx._embedder = SimpleNamespace(encode=lambda q, normalize_embeddings=True: np.array([0, 1, 0], dtype=np.float32))
    return idx


def test_hybrid_merges_keyword_and_dense():
    idx = fake_index()
    assert idx.keyword("returned footwear", 3) == [0]
    assert idx.dense("anything", 1) == [1]
    assert set(idx.hybrid("returned footwear", 2)) == {0, 1}


def test_answer_extracts_valid_citations_only():
    idx = fake_index()
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="60 days [1], see also [7]."))])
    llm = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: reply)))
    ans = answer("How long for shoes?", [(c, 0.0) for c in idx.chunks], llm=llm)
    assert ans.cited == [1] and ans.grounded


def test_index_round_trip(tmp_path):
    idx = fake_index()
    idx.save(tmp_path)
    back = Index.load(tmp_path)
    assert [c.text for c in back.chunks] == [c.text for c in idx.chunks]
    assert np.allclose(back.vectors, idx.vectors)
