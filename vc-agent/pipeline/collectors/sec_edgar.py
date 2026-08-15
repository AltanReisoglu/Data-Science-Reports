"""SEC EDGAR Form D collector — the most valuable source in the funnel.

A company raising private capital in the US must file a Form D within **15 days**
of first sale. The round therefore shows up here *before* the press release
(domain §3.2). Free, no key.

Two caveats, both verified live:

* **User-Agent is mandatory.** SEC returns 403 to requests whose User-Agent
  carries no name and contact address. That is what ``config.SEC_USER_AGENT`` is for.
* **US only.** If the thesis geography is outside the US this collector returns
  nothing — that is a scope limit rather than a failure, and it belongs in the
  memo as `missing_data`.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import config
from collectors.base import CollectionResult, Collector
from schemas import Signal, Source

ENDPOINT = "https://efts.sec.gov/LATEST/search-index"
FILING_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=D"

_CIK = re.compile(r"\(CIK (\d{10})\)")


class SecFormD(Collector):
    name = "sec_edgar"

    def headers(self) -> dict[str, str]:
        # SEC's own access policy: a User-Agent without contact details gets 403.
        return {
            "User-Agent": config.SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        }

    def collect(self, *, query: str, days: int) -> CollectionResult:
        today = datetime.now(timezone.utc).date()
        payload = self.fetch_json(
            ENDPOINT,
            {
                "q": f'"{query}"',
                "forms": "D",
                "startdt": (today - timedelta(days=days)).isoformat(),
                "enddt": today.isoformat(),
            },
        )

        signals: list[Signal] = []
        for hit in payload.get("hits", {}).get("hits", []):
            fields = hit.get("_source", {})
            display_names = fields.get("display_names") or []
            if not display_names:
                continue
            raw_name = display_names[0]
            cik_match = _CIK.search(raw_name)
            name = _CIK.sub("", raw_name).strip(" .,")
            filed = fields.get("file_date")
            if not filed:
                continue

            url = (
                FILING_URL.format(cik=cik_match.group(1))
                if cik_match
                else "https://www.sec.gov/edgar/search/"
            )
            signals.append(
                Signal(
                    kind="funding_round",
                    summary=f"Form D filed: {name}",
                    date=datetime.fromisoformat(filed).replace(tzinfo=timezone.utc),
                    # A Form D is not merely primary but **official**: the
                    # company's own filing with a regulator.
                    source=Source(name=self.name, url=url, confidence="official"),
                    raw={"cik": cik_match.group(1) if cik_match else None, "names": display_names},
                    candidate_name=name,
                )
            )
        return CollectionResult(source=self.name, signals=signals)
