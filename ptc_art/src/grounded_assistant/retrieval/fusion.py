"""Reciprocal Rank Fusion (RRF) — BM25 ve dense sonuçlarını birleştirir.

Bkz. specs/001-ptc-grounded-assistant/research.md #2 (k=60, Cormack et al. varsayılanı).
"""

from __future__ import annotations

from grounded_assistant.models import RetrievalHit


def reciprocal_rank_fusion(
    bm25_results: list[tuple[str, str]],
    dense_results: list[tuple[str, str]],
    k: int = 60,
) -> list[RetrievalHit]:
    """bm25_results/dense_results: (doc_id, snippet) çiftleri, sıra düzeninde (0. eleman en alakalı).

    Dönüş: rrf_score azalan sırada RetrievalHit listesi.
    """
    snippets: dict[str, str] = {}
    bm25_rank_of: dict[str, int] = {}
    dense_rank_of: dict[str, int] = {}

    for rank, (doc_id, snippet) in enumerate(bm25_results):
        snippets[doc_id] = snippet
        bm25_rank_of[doc_id] = rank

    for rank, (doc_id, snippet) in enumerate(dense_results):
        snippets.setdefault(doc_id, snippet)
        dense_rank_of[doc_id] = rank

    hits: list[RetrievalHit] = []
    for doc_id, snippet in snippets.items():
        bm25_rank = bm25_rank_of.get(doc_id)
        dense_rank = dense_rank_of.get(doc_id)
        score = 0.0
        if bm25_rank is not None:
            score += 1.0 / (k + bm25_rank + 1)
        if dense_rank is not None:
            score += 1.0 / (k + dense_rank + 1)
        hits.append(
            RetrievalHit(
                doc_id=doc_id,
                snippet=snippet,
                bm25_rank=bm25_rank,
                dense_rank=dense_rank,
                rrf_score=score,
            )
        )

    hits.sort(key=lambda hit: hit.rrf_score, reverse=True)
    return hits
