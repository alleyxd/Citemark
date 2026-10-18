import math
import re
from collections import Counter

import numpy as np

STOP = set("a an and are as at be by for from has have he her his in is it its of on or she that the their they "
           "this to was were will with what which who whom when where why how did does do".split())


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokens(d)) for d in docs]
        self.len = np.array([sum(t.values()) for t in self.tf], dtype=float)
        self.avg = self.len.mean() if len(docs) else 0.0
        df = Counter(w for t in self.tf for w in t)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.postings: dict[str, list[int]] = {}
        for i, t in enumerate(self.tf):
            for w in t:
                self.postings.setdefault(w, []).append(i)

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.tf))
        for w in set(tokens(query)):
            if w not in self.postings:
                continue
            for i in self.postings[w]:
                f = self.tf[i][w]
                s[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s
