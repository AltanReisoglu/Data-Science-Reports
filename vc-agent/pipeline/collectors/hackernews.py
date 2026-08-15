"""Hacker News (Algolia) collector — funding news and launches.

API: https://hn.algolia.com/api — no key required, generous rate limit.
We use ``search_by_date`` because we want **freshness**, not relevance ranking
(domain §3.2: "freshness beats completeness").
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from collectors.base import CollectionResult, Collector
from schemas import Signal, Source, registrable_domain

ENDPOINT = "https://hn.algolia.com/api/v1/search_by_date"

# Funding signal: "raises $12M", "$40M Series A", "seed round"
_FUNDING = re.compile(
    r"\braise[sd]?\b|\bseries\s+[a-e]\b|\bseed\s+round\b|\bfunding\b|\$\s?\d+(\.\d+)?\s?[mbk]\b",
    re.IGNORECASE,
)
_LAUNCH = re.compile(r"^show hn[:\s]", re.IGNORECASE)

# Hosts that identify a platform rather than a company. A launch linking to
# github.com is pointing at a repository, not at "GitHub the startup" — and
# taking the domain anyway collapses every such launch into one entity keyed on
# `github.com`. Observed live: three unrelated Show HN posts merged into a single
# company on the first run.
_GENERIC_HOSTS = {
    "github.com", "gitlab.com", "bitbucket.org", "codeberg.org",
    "medium.com", "substack.com", "notion.site", "notion.so",
    "youtube.com", "youtu.be", "twitter.com", "x.com", "reddit.com",
    "vercel.app", "netlify.app", "herokuapp.com", "pages.dev", "streamlit.app",
    "google.com", "docs.google.com", "arxiv.org", "news.ycombinator.com",
    "replit.com", "huggingface.co", "npmjs.com", "pypi.org",
}


class HackerNews(Collector):
    name = "hn"

    def collect(self, *, query: str, days: int) -> CollectionResult:
        cutoff = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
        payload = self.fetch_json(
            ENDPOINT,
            {
                "query": query,
                "tags": "story",
                "numericFilters": f"created_at_i>{cutoff}",
                "hitsPerPage": 60,
            },
        )

        signals: list[Signal] = []
        for hit in payload.get("hits", []):
            title = (hit.get("title") or "").strip()
            if not title:
                continue
            hn_url = f"https://news.ycombinator.com/item?id={hit['objectID']}"
            external_url = hit.get("url") or ""

            if _LAUNCH.search(title):
                kind = "product_launch"
            elif _FUNDING.search(title):
                kind = "funding_round"
            else:
                kind = "news"

            signals.append(
                Signal(
                    kind=kind,
                    summary=title,
                    date=datetime.fromtimestamp(hit["created_at_i"], tz=timezone.utc),
                    source=Source(name=self.name, url=hn_url, confidence="secondary"),
                    raw={
                        "points": hit.get("points"),
                        "comments": hit.get("num_comments"),
                        "external_url": external_url,
                    },
                    candidate_name=_guess_company(title),
                    # The linked domain identifies the company only for a launch,
                    # where the submitter is pointing at their own product. On a
                    # funding or news story the domain belongs to whoever
                    # *published* the article, not to its subject — which is how
                    # `bbc.com` and `bloomberg.com` first arrived as candidates.
                    candidate_domain=(
                        _company_domain(external_url) if kind == "product_launch" else None
                    ),
                )
            )
        return CollectionResult(source=self.name, signals=signals)


def _company_domain(url: str) -> str | None:
    """The registrable domain, but only when it can stand for a company."""
    if not url:
        return None
    domain = registrable_domain(url)
    if domain is None or domain in _GENERIC_HOSTS:
        return None
    return domain


def _guess_company(title: str) -> str | None:
    """Guess a company name from a headline.

    Deliberately **conservative**: it returns ``None`` when unsure and lets
    normalization fall back to the domain. A wrong name costs more than a
    missing one — it is the entity-collision risk of docs/04 §10.
    """
    text = title.strip()
    match = re.match(r"^show hn[:\s]+([A-Z][\w.\-]*(?:\s+[A-Z][\w.\-]*)?)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip(" :,-")
    # "Acme raises $12M ..." / "Acme, a foo bar, raises ..."
    match = re.match(
        r"^([A-Z][\w.\-]*(?:\s+[A-Z][\w.\-]*)?)[,\s]+(?:a\s|an\s|the\s)?.{0,60}?\braise", text
    )
    if match:
        return match.group(1).strip(" :,-")
    return None
