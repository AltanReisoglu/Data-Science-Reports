"""Gerçek web arama — DuckDuckGo'nun API anahtarı gerektirmeyen HTML uç noktası
(Faz 4, tool sayısını artırma isteği, 2026-08-30). `.env`'deki gibi mevcut bir
kimlik bilgisi kullanmıyor, YENİ bir API anahtarı da gerektirmiyor — bilinçli
bir seçim (mentörün "yeni kimlik bilgisi oluşturma" kuralına uymak için).

Bu, PoC'nin diğer tool'larından (mock/sahte veri) FARKLI: gerçekten dışarıya,
onaylı-kanal dışında hiçbir yere çıkmadan (yalnızca `tool-gateway-egress`
policy'sinin izin verdiği tek yeni FQDN — `html.duckduckgo.com`) giden gerçek
bir HTTP isteği. Cilium'un "yeni bir dış hedef eklerken bunu açıkça
onaylaman gerekir" ilkesinin somut bir örneği.
"""

from __future__ import annotations

import html as html_module
import re

import requests

_ENDPOINT = "https://html.duckduckgo.com/html/"
_TITLE_URL_PATTERN = re.compile(
    r'<a rel="nofollow" class="result__a" href="([^"]+)">(.*?)</a>', re.DOTALL
)
_SNIPPET_PATTERN = re.compile(r'<a class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)


def _clean(fragment: str) -> str:
    return html_module.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """DuckDuckGo'nun sunucu-render'lı (JS'siz) HTML sonuç sayfasını sorgular."""
    response = requests.post(
        _ENDPOINT,
        data={"q": query},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    response.raise_for_status()
    body = response.text

    titles_urls = _TITLE_URL_PATTERN.findall(body)
    snippets = _SNIPPET_PATTERN.findall(body)

    results = []
    for i, (url, title) in enumerate(titles_urls[:max_results]):
        snippet = _clean(snippets[i]) if i < len(snippets) else ""
        results.append({"title": _clean(title), "url": url, "snippet": snippet})
    return results
