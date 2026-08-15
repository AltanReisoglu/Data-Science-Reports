"""Layer 1 — collectors. No LLM, deterministic, independently failable."""

from collectors.arxiv import ArXiv
from collectors.base import CollectionResult, Collector
from collectors.github import GitHub
from collectors.hackernews import HackerNews
from collectors.sec_edgar import SecFormD

# Discovery sweep. Order is irrelevant; each runs independently and one failure
# does not stop the others.
#
# ArXiv is deliberately NOT here. A paper carries no company name or domain, so
# every arXiv signal arrives unattachable and inflates the "no resolvable owner"
# count without ever reaching a candidate. Measured on the first live run: 30 of
# 30 arXiv signals were unattached. arXiv earns its place one layer down, as the
# team analyst's tool for checking a named founder's publication trace.
DISCOVERY = [HackerNews, SecFormD, GitHub]
ALL = DISCOVERY

__all__ = [
    "ArXiv", "GitHub", "HackerNews", "SecFormD",
    "CollectionResult", "Collector", "ALL", "DISCOVERY",
]
