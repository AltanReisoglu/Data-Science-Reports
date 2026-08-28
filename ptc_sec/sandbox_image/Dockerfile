# Sandbox — Faz 2. Bilerek minimal: yalnızca Tool Gateway'e (fastmcp Client ile)
# bağlanabilmek için gereken tek paket kurulu. Ağ erişimi Python seviyesinde
# kısıtlanmıyor (research.md §4.3) — enforcement Cilium'da.
FROM python:3.11-slim

RUN pip install --no-cache-dir fastmcp

COPY sandbox_image/entrypoint.py /entrypoint.py

ENTRYPOINT ["python", "/entrypoint.py"]
