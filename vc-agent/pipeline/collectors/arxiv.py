"""arXiv collector — deep-tech provenance.

Returns Atom XML. arXiv's own guidance is one request every 3 seconds, which
``config.RATE_LIMITS`` enforces. The signal value: if a founder published in the
relevant field within the last 18 months, the thesis requirement "technical
founder" has evidence behind it rather than an assertion.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

from collectors.base import CollectionResult, Collector
from schemas import Signal, Source

ENDPOINT = "http://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}


class ArXiv(Collector):
    name = "arxiv"

    # export.arxiv.org/robots.txt is "Disallow: /" for every agent — it is aimed
    # at crawlers of the HTML site. The arXiv API Terms of Use separately grant
    # programmatic access to this endpoint on a 3-second interval, which
    # `config.RATE_LIMITS["arxiv"]` enforces. The exemption is narrow (this
    # collector only), justified, and logged on every request.
    robots_exemption = "arXiv API Terms of Use: https://info.arxiv.org/help/api/tou.html"

    def collect(self, *, query: str, days: int) -> CollectionResult:
        body = self.fetch(
            ENDPOINT,
            {
                "search_query": f"all:{query}",
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "max_results": 30,
            },
        )
        root = ET.fromstring(body)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        signals: list[Signal] = []
        for entry in root.findall("a:entry", NS):
            title = (entry.findtext("a:title", "", NS) or "").strip().replace("\n", " ")
            url = (entry.findtext("a:id", "", NS) or "").strip()
            published = (entry.findtext("a:published", "", NS) or "").strip()
            if not (title and url and published):
                continue
            date = datetime.fromisoformat(published.replace("Z", "+00:00"))
            if date < cutoff:
                continue
            authors = [
                (a.findtext("a:name", "", NS) or "").strip()
                for a in entry.findall("a:author", NS)
            ]
            signals.append(
                Signal(
                    kind="academic",
                    summary=title[:200],
                    date=date,
                    source=Source(name=self.name, url=url, confidence="primary"),
                    raw={"authors": authors[:12]},
                )
            )
        return CollectionResult(source=self.name, signals=signals)
