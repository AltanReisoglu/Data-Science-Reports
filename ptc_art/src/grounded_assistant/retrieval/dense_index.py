"""Dense embedding index — Qwen3-Embedding-8B, .env gateway'i üzerinden (research.md #2)."""

from __future__ import annotations

import os

import numpy as np
from langchain_openai import OpenAIEmbeddings

from grounded_assistant.retrieval.bm25_index import IndexedDocument


def build_embeddings() -> OpenAIEmbeddings:
    """Diğer modüller (ör. scripts/ingest_sample_docs.py) tarafından da kullanılıyor,
    bu yüzden public.

    Canlı test (2026-09-01): chunking (bkz. knowledge_base._chunk_text) tek başına
    yetmedi — `embed_documents`, varsayılan `chunk_size` (1000) ile TÜM parçaları
    (wiki kaynağında 201 parça, ~400KB) yine TEK bir HTTP isteğinde gönderiyordu,
    gateway yine "502 Upstream service error" verdi. Küçük bir `chunk_size` (16),
    isteği birden fazla küçük HTTP çağrısına bölüyor — canlı doğrulandı (16 parça,
    ~32KB, 3.2sn'de başarılı)."""
    return OpenAIEmbeddings(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        model="Qwen3-Embedding-8B",
        chunk_size=16,
    )


def _cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)
    denom = matrix_norms * vector_norm
    denom[denom == 0] = 1e-12
    return (matrix @ vector) / denom


class DenseIndex:
    def __init__(self, documents: list[IndexedDocument], vectors: np.ndarray | None = None) -> None:
        """`vectors` verilirse (ör. scripts/ingest_sample_docs.py'nin önceden
        hesapladığı embedding'ler), gateway'e tekrar embed_documents çağrısı
        yapılmaz — her sorguda aynı statik dokümanları yeniden embed etmemek için."""
        self._documents = documents
        self._embeddings = build_embeddings()
        if vectors is not None:
            self._vectors = vectors
        else:
            self._vectors = (
                np.array(self._embeddings.embed_documents([doc.text for doc in documents]))
                if documents
                else np.empty((0, 0))
            )

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, str]]:
        """Dönüş: (doc_id, snippet) çiftleri, azalan kosinüs benzerliğine göre sıralı."""
        if not self._documents:
            return []
        query_vector = np.array(self._embeddings.embed_query(query))
        similarities = _cosine_similarity(self._vectors, query_vector)
        ranked_indices = np.argsort(-similarities)
        return [
            (self._documents[i].doc_id, self._documents[i].text[:200])
            for i in ranked_indices[:top_k]
        ]
