import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    id: int
    source: str
    title: str
    text: str


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8", errors="replace")


def split(text: str, max_words: int = 220, overlap: int = 40) -> list[tuple[str, str]]:
    """Split into (heading, text) pieces. Paragraphs are kept whole where possible; long runs are cut into
    windows of max_words with `overlap` words carried over, so a fact near a boundary appears in both chunks."""
    heading, out, buf = "", [], []

    def flush():
        words = " ".join(buf).split()
        step = max_words - overlap
        for start in range(0, max(len(words) - overlap, 1), step):
            piece = " ".join(words[start:start + max_words])
            if piece:
                out.append((heading, piece))
        buf.clear()

    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        m = re.match(r"#{1,6}\s+(.*)", para)
        if m:
            # a heading starts a new section, whether or not a blank line follows it
            flush()
            heading = m.group(1).strip()
            para = para[m.end():].strip()
            if not para:
                continue
        if sum(len(p.split()) for p in buf) + len(para.split()) > max_words:
            flush()
        buf.append(para)
    flush()
    return out


def chunk_folder(folder: str, patterns=("*.md", "*.txt", "*.pdf")) -> list[Chunk]:
    chunks = []
    root = Path(folder)
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            rel = str(path.relative_to(root)).replace("\\", "/")
            for heading, text in split(read_file(path)):
                chunks.append(Chunk(len(chunks), rel, heading or path.stem, text))
    return chunks
