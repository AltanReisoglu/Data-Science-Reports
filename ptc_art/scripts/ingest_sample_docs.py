"""Örnek dokümanları sample_docs/ altından okuyup embedding'leri önceden hesaplayıp
indices/ altına yazar (T018).

`knowledge_base.py`, her sorguda dokümanları sample_docs/'tan canlı okuyor; ama
embedding hesaplaması (gateway'e API çağrısı) her seferinde tekrarlanmasın diye bu
script statik dokümanların embedding'lerini bir kere hesaplayıp diske yazıyor.
Bu script çalıştırılmamışsa sistem yine çalışır — knowledge_base.py o zaman
embedding'i canlı hesaplar (bkz. dense_index.py, vectors=None yolu).

Kullanım (bağımlılıklar kurulduktan ve .env doldurulduktan sonra):
    python scripts/ingest_sample_docs.py
"""

from __future__ import annotations

import json

from dotenv import load_dotenv

from grounded_assistant.access_paths.knowledge_base import INDEX_DIR, load_documents
from grounded_assistant.models import SourceId
from grounded_assistant.retrieval.dense_index import build_embeddings


def ingest() -> None:
    load_dotenv()
    INDEX_DIR.mkdir(exist_ok=True)
    embeddings = build_embeddings()

    for source_id in SourceId:
        documents = load_documents(source_id)
        if not documents:
            print(f"[{source_id.value}] atlanıyor: doküman yok (sample_docs/{source_id.value}/ boş)")
            continue

        vectors = embeddings.embed_documents([doc.text for doc in documents])
        payload = [
            {"doc_id": doc.doc_id, "text": doc.text, "embedding": vector}
            for doc, vector in zip(documents, vectors, strict=True)
        ]

        out_path = INDEX_DIR / f"{source_id.value}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"[{source_id.value}] {len(documents)} doküman indexlendi -> {out_path}")


if __name__ == "__main__":
    ingest()
