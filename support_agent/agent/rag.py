"""KB retrieval. Starts as BM25 over markdown sections — zero infra, fully
deterministic, so retrieval quality is measurable in the eval harness rather than
dependent on an embedding service. Swap to embeddings later behind the same
`Retriever.search` interface if quality demands it.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .events import Retrieval

_KB_DIR = Path(__file__).resolve().parent.parent / "kb"
_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass
class _Chunk:
    doc_id: str
    text: str
    tokens: list[str]


class Retriever:
    """BM25 over KB chunks. One chunk per markdown section (## heading)."""

    def __init__(self, kb_dir: Path = _KB_DIR, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.chunks = self._load(kb_dir)
        self._build_index()

    def _load(self, kb_dir: Path) -> list[_Chunk]:
        chunks: list[_Chunk] = []
        for path in sorted(kb_dir.glob("*.md")):
            sections = re.split(r"\n(?=#{1,2} )", path.read_text())
            for i, sec in enumerate(sections):
                text = sec.strip()
                if text:
                    chunks.append(_Chunk(f"{path.stem}#{i}", text, _tokenize(text)))
        return chunks

    def _build_index(self) -> None:
        self.N = len(self.chunks)
        self.avgdl = sum(len(c.tokens) for c in self.chunks) / max(self.N, 1)
        df: Counter[str] = Counter()
        for c in self.chunks:
            df.update(set(c.tokens))
        self.idf = {
            term: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }
        self.tf = [Counter(c.tokens) for c in self.chunks]

    def search(self, query: str, k: int = 3, min_score: float = 0.1) -> list[Retrieval]:
        q_terms = _tokenize(query)
        scored: list[tuple[float, _Chunk]] = []
        for chunk, tf in zip(self.chunks, self.tf):
            dl = len(chunk.tokens)
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * (freq * (self.k1 + 1)) / denom
            if score > min_score:
                scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [Retrieval(doc_id=c.doc_id, score=round(s, 3), text=c.text)
                for s, c in scored[:k]]
