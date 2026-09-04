# Sandbox — Faz 2. Bilerek minimal: yalnızca Tool Gateway'e (fastmcp Client ile)
# bağlanabilmek için gereken tek paket kurulu. Ağ erişimi Python seviyesinde
# kısıtlanmıyor (research.md §4.3) — enforcement Cilium'da.
#
# `requests` (Altan'ın kararı, 2026-08-30): demo/sunum amaçlı — LLM'in kaçış
# denemesi yazarken "No module named 'requests'" hatasına takılıp asıl
# Cilium engelini hiç göstermeden bitmesini önler. Güvenlik açısından fark
# etmez: enforcement zaten Cilium'da (network seviyesinde), Python seviyesinde
# hangi kütüphanenin var olduğu bir güvenlik sınırı değil.
FROM python:3.11-slim

# pandas + pyarrow (2026-09-03): artifact serileştirmesi SANDBOX İÇİNDE yapılıyor
# — bir dataframe ağdan nesne olarak geçemez, Parquet'e çevrilip ham bayt olarak
# gider. Servis baytı olduğu gibi saklar, çözmeye kalkmaz.
#
# Bu paketler imaja GÖMÜLÜ olmak zorunda: sandbox'ın çalışma anında ağ erişimi
# yok, `pip install` yapamaz. (Anthropic'in code execution container'ı da aynı
# kısıtla çalışıyor: "no internet access, only pre-installed libraries".)
RUN pip install --no-cache-dir fastmcp requests pandas pyarrow

# serialize.py, ana kaynak ağacındaki dosyanın TA KENDİSİ — çoğaltılmıyor,
# kopyalanıyor. Böylece Parquet/Arrow biçimi ve pickle yasağı tek yerde
# tanımlı kalıyor; sandbox ile servis aynı kurallara uyuyor.
COPY src/grounded_assistant/artifacts/serialize.py /serialize.py
# artifact_client.py (2026-09-04): Artifact Service'e akışlı HTTP istemcisi.
# fastmcp'ye bağlı DEĞİL — artifact yolu MCP'den çıkarıldı, yalnızca `requests`.
COPY sandbox_image/artifact_client.py /artifact_client.py
COPY sandbox_image/entrypoint.py /entrypoint.py

ENTRYPOINT ["python", "/entrypoint.py"]
