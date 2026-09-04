"""4 paralel kurumsal bilgi bankası kaynağını sorgulayan orkestrasyon (FR-002/003).

Kısmi kaynak hatası/boş sonuç (FR-010): her kaynak bağımsız olarak
ok/empty/error durumuna düşer, biri hata verse de diğerleri yanıt üretmeye devam eder.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from langchain_core.tools import tool

from grounded_assistant.models import KnowledgeBaseSource, SourceId, SourceStatus
from grounded_assistant.retrieval.bm25_index import BM25Index, IndexedDocument
from grounded_assistant.retrieval.dense_index import DenseIndex
from grounded_assistant.retrieval.fusion import reciprocal_rank_fusion
from grounded_assistant.trace import Trace

SAMPLE_DOCS_DIR = Path(__file__).resolve().parents[3] / "sample_docs"
INDEX_DIR = Path(__file__).resolve().parents[3] / "indices"

# Canlı test (2026-09-01): wiki/'deki KKB faaliyet raporu bölümleri (38-93KB) tek
# parça olarak embed edilince gateway'den "502 Upstream service error" dönüyordu
# (~77sn'lik retry sonrası) — parçalamadan gönderilen istek boyutu için çok büyüktü.
# 2000 karakter, hem bu sorunu önleyen hem RAG kalitesini artıran (küçük dokümanlar
# zaten tek parça kalıyor, sadece dev dosyalar bölünüyor) güvenli bir sınır.
_CHUNK_SIZE = 2000


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE) -> list[str]:
    """Metni ~chunk_size karakterlik parçalara böler; kelime ortasında kesmemek
    için en yakın boşluğa (makul bir sınır içinde) yuvarlar."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            next_space = text.find(" ", end)
            if next_space != -1 and next_space - end < 200:
                end = next_space
        chunks.append(text[start:end])
        start = end
    return chunks


def load_documents(source_id: SourceId) -> list[IndexedDocument]:
    """`scripts/ingest_sample_docs.py` da bunu kullanıyor, bu yüzden public.

    Her dosya, boyutu `_CHUNK_SIZE`'ı aşıyorsa birden fazla `IndexedDocument`'a
    bölünür (`doc_id` sonuna `#chunkN` eklenir); aşmıyorsa tek parça kalır ve
    `doc_id` eskisiyle birebir aynı kalır (geriye dönük uyumluluk)."""
    source_dir = SAMPLE_DOCS_DIR / source_id.value
    if not source_dir.is_dir():
        return []
    documents: list[IndexedDocument] = []
    for path in sorted(source_dir.glob("*.md")):
        if path.name == "README.md":
            continue
        text = path.read_text(encoding="utf-8")
        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            doc_id = (
                f"{source_id.value}/{path.name}"
                if len(chunks) == 1
                else f"{source_id.value}/{path.name}#chunk{i}"
            )
            documents.append(IndexedDocument(doc_id=doc_id, text=chunk))
    return documents


def _load_precomputed_vectors(
    source_id: SourceId, documents: list[IndexedDocument]
) -> np.ndarray | None:
    """`scripts/ingest_sample_docs.py`'nin önceden hesaplayıp diske yazdığı embedding'leri
    okur. Index bayat/eksikse (dosya yok ya da doc_id uyuşmuyor) None döner — çağıran
    o zaman gateway'den canlı hesaplamaya düşer."""
    index_path = INDEX_DIR / f"{source_id.value}.json"
    if not index_path.is_file():
        return None
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    embedding_by_doc_id = {entry["doc_id"]: entry["embedding"] for entry in payload}
    if any(doc.doc_id not in embedding_by_doc_id for doc in documents):
        return None
    return np.array([embedding_by_doc_id[doc.doc_id] for doc in documents])


def _query_source(source_id: SourceId, query: str) -> KnowledgeBaseSource:
    try:
        documents = load_documents(source_id)
        if not documents:
            return KnowledgeBaseSource(source_id=source_id, status=SourceStatus.EMPTY, hits=[])

        bm25_results = BM25Index(documents).search(query)
        precomputed_vectors = _load_precomputed_vectors(source_id, documents)
        dense_results = DenseIndex(documents, vectors=precomputed_vectors).search(query)
        hits = reciprocal_rank_fusion(bm25_results, dense_results)

        if not hits:
            return KnowledgeBaseSource(source_id=source_id, status=SourceStatus.EMPTY, hits=[])
        return KnowledgeBaseSource(source_id=source_id, status=SourceStatus.OK, hits=hits)
    except Exception:
        # FR-010: bu kaynak başarısız oldu, ama diğer kaynaklar etkilenmemeli.
        return KnowledgeBaseSource(source_id=source_id, status=SourceStatus.ERROR, hits=[])


def query_knowledge_base(query: str) -> list[KnowledgeBaseSource]:
    """4 kaynağı paralel sorgular (FR-002); her biri bağımsız durum taşır (FR-010)."""
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_query_source, source_id, query) for source_id in SourceId]
        return [future.result() for future in futures]


def _format_sources_for_model(sources: list[KnowledgeBaseSource]) -> str:
    lines = []
    for source in sources:
        if source.status is SourceStatus.OK:
            snippets = "; ".join(hit.snippet for hit in source.hits[:3])
            lines.append(f"[{source.source_id.value}] {snippets}")
        elif source.status is SourceStatus.EMPTY:
            lines.append(f"[{source.source_id.value}] sonuç yok")
        else:
            lines.append(f"[{source.source_id.value}] hata: kaynağa erişilemedi")
    return "\n".join(lines) if lines else "Hiçbir kaynaktan sonuç bulunamadı."


def make_kb_tool(trace: Trace):
    """Ajanın çağırabileceği tool'u üretir; sonuçları hem modele (string) hem de
    izlenebilirlik kaydına (trace, FR-009) yazar."""

    @tool
    def search_knowledge_base(query: str) -> str:
        """Kurumsal bilgi bankasını (politika, kurumsal wiki, destek talebi arşivi,
        teknik dokümantasyon) sorgula. Kullanıcının sorusuyla ilgili kurumsal bilgi
        gerektiğinde bunu çağır."""
        sources = query_knowledge_base(query)
        timestamp = datetime.now(timezone.utc)
        for source in sources:
            trace.record_kb_source(source, timestamp)
        return _format_sources_for_model(sources)

    return search_knowledge_base
