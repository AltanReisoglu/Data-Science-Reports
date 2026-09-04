"""BM25 lexical arama — Hybrid Search'ün lexical yarısı (research.md #2)."""

from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi


@dataclass(frozen=True)
class IndexedDocument:
    doc_id: str
    text: str


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Index:
    def __init__(self, documents: list[IndexedDocument]) -> None:
        self._documents = documents
        corpus_tokens = [_tokenize(doc.text) for doc in documents]
        self._bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, str]]:
        """Dönüş: (doc_id, snippet) çiftleri, azalan BM25 skoruna göre sıralı."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            (self._documents[i].doc_id, self._documents[i].text[:200])
            for i in ranked_indices[:top_k]
            if scores[i] > 0
        ]
