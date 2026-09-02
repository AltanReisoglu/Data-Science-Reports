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

RUN pip install --no-cache-dir fastmcp requests

COPY sandbox_image/entrypoint.py /entrypoint.py

ENTRYPOINT ["python", "/entrypoint.py"]
