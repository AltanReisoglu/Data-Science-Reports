# Tool Gateway — Faz 2. Faz 1'in knowledge_base/mock_live_system mantığını
# in-process sarıp FastMCP HTTP transport üzerinden expose eder.
FROM python:3.11-slim

WORKDIR /app

# knowledge_base.py, kendi konumuna göre (parents[3]) repo kökünü bulup
# sample_docs/'u orada arıyor — bu yüzden aynı dizin yerleşimini koruyoruz:
# /app == repo kökü, src/ ve sample_docs/ onun altında.
COPY pyproject.toml ./
COPY src/ ./src/
COPY mock_services/ ./mock_services/

# Tool Gateway sadece bunlara ihtiyaç duyuyor — pyproject.toml'un tamamını
# (langgraph/typer/rich vb., ana asistana özgü) tekrar indirmemek için
# açıkça listeliyoruz, kendi paketimizi de --no-deps ile kuruyoruz.
# minio (2026-09-03): artifact deposunun S3 istemcisi.
#
# pandas/pyarrow BİLEREK YOK: serileştirme sandbox içinde yapılıyor, gateway'e
# hazır baytlar geliyor ve gateway onları çözmeye kalkmıyor — yalnızca saklıyor.
# Bu, gateway'i ~300 MB hafif tutmakla kalmıyor, bir güvenlik özelliği de:
# gateway'in LLM'in ürettiği veriyi hiç ayrıştırmaması gereken bir katman
# olduğunu yapısal olarak garanti ediyor.
RUN pip install --no-cache-dir fastmcp rank_bm25 numpy langchain-core langchain-openai minio && \
    pip install --no-cache-dir --no-deps -e .

# tiktoken'ın cl100k_base tablosunu (langchain_openai, tiktoken'a tanıdık
# gelmeyen bir model adı gördüğünde buna düşüyor — bkz. plan.md "6. bulgu",
# 2026-08-28) build zamanında (internete tam erişimin olduğu TEK an) indirip
# image'a KALICI olarak gömüyoruz. TIKTOKEN_CACHE_DIR sabit bir yola işaret
# ediyor ki çalışma zamanında (Cilium kısıtlı pod'da, her yeniden başlamada
# sıfırlanan bir dosya sisteminde) tiktoken dosyayı burada zaten bulsun, ağa
# hiç çıkmaya çalışmasın.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# sample_docs/ bilerek EN SONA kondu: içeriği sık değişiyor (yeni doküman
# eklenmesi gibi) — pip install/tiktoken-cache katmanlarının önbelleğini
# boşuna geçersiz kılıp paketleri yeniden indirtmesin diye (2026-08-28'de
# tam olarak bu yüzden gereksiz bir yeniden-indirme oldu).
COPY sample_docs/ ./sample_docs/

# indices/ — `python scripts/ingest_sample_docs.py` ile ÖNCEDEN (build zamanında,
# laptop'tan, .env'deki gateway'e erişimle) hesaplanmış embedding'ler (2026-09-01).
# Bunlar olmadan her `search_knowledge_base` çağrısı wiki/'deki ~400KB'lık dokümanı
# canlı embed etmeye çalışıyordu — gateway 502 veriyordu (chunking + küçük
# chunk_size ile bu da düzeltildi ama yine de ~30sn sürüyordu). Bu dosyalar
# gömülünce çalışma zamanında sadece kısa SORGU metni embed ediliyor, dokümanlar
# değil — tiktoken cache'iyle aynı mantık: "internete tam erişimin olduğu TEK an"
# build zamanı, çalışma zamanı Cilium'la kısıtlı.
COPY indices/ ./indices/

EXPOSE 8443

ENTRYPOINT ["python", "-m", "mock_services.tool_gateway.server"]
