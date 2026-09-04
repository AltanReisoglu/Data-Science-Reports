# Artifact Service (2026-09-04) — artifact işi Tool Gateway'den buraya taşındı.
#
# Gateway'in imajıyla karşılaştırınca farkı görmek kolay: burada rank_bm25,
# langchain, tiktoken, sample_docs, indices YOK. Bu servis tek iş yapıyor —
# baytları doğrulayıp saklamak ve kayıt defterini tutmak.
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
COPY services/ ./services/

# pandas/pyarrow BİLEREK YOK: serileştirme sandbox içinde yapılıyor, buraya hazır
# baytlar geliyor ve servis onları çözmeye kalkmıyor — yalnızca saklıyor. Bu,
# imajı hafif tutmakla kalmıyor, bir güvenlik özelliği de: LLM'in ürettiği
# veriyi hiç ayrıştırmaması gereken bir katman olduğu yapısal olarak garanti
# altına alınıyor. (Pickle reddi bayt İMZASINA bakıyor, ayrıştırmaya değil.)
RUN pip install --no-cache-dir fastapi uvicorn minio && \
    pip install --no-cache-dir --no-deps -e .

EXPOSE 8080

# Tek worker: metadata SQLite ve PVC ReadWriteOnce. Çok worker'a çıkmak önce
# Postgres'e geçmeyi gerektirir (bkz. karar dokümanı).
ENTRYPOINT ["uvicorn", "services.artifact_service.app:app", \
            "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
