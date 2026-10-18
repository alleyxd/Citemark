import os
import re
from dataclasses import dataclass, field

from openai import OpenAI

from .chunk import Chunk

SYSTEM = (
    "Answer the question using only the numbered sources below. Cite the sources you used in square brackets, "
    "like [2] or [1][3], right after the statement they support. If the sources do not contain the answer, say "
    "\"I don't know based on the documents.\" and nothing else. Keep answers short."
)


@dataclass
class Answer:
    text: str
    sources: list[Chunk]
    cited: list[int] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        return bool(self.cited) or self.text.startswith("I don't know")


def client():
    return OpenAI(base_url=os.environ.get("CITEMARK_BASE_URL", "http://localhost:8000/v1"),
                  api_key=os.environ.get("CITEMARK_API_KEY", "local"))


def answer(question: str, hits: list[tuple[Chunk, float]], model: str | None = None, llm=None) -> Answer:
    chunks = [c for c, _ in hits]
    context = "\n\n".join(f"[{i}] ({c.source} - {c.title})\n{c.text}" for i, c in enumerate(chunks, 1))
    llm = llm or client()
    reply = llm.chat.completions.create(
        model=model or os.environ.get("CITEMARK_MODEL", "local"),
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"}],
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    text = (reply.choices[0].message.content or "").strip()
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(chunks)})
    return Answer(text, chunks, cited)
