"""GitHub collector — repository momentum and technical trace.

Works without a key (60 requests/hour); ``GITHUB_TOKEN`` raises that to 5,000.
This is the evidence source behind the thesis requirement "public technical trace".
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from collectors.base import CollectionResult, Collector
from schemas import Signal, Source, registrable_domain

SEARCH = "https://api.github.com/search/repositories"


class GitHub(Collector):
    name = "github"

    def headers(self) -> dict[str, str]:
        headers = super().headers()
        headers["Accept"] = "application/vnd.github+json"
        token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def collect(self, *, query: str, days: int) -> CollectionResult:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
        payload = self.fetch_json(
            SEARCH,
            {
                "q": f"{query} pushed:>{cutoff} stars:>25",
                "sort": "updated",
                "order": "desc",
                "per_page": 40,
            },
        )

        signals: list[Signal] = []
        for repo in payload.get("items", []):
            owner = repo.get("owner", {}) or {}
            # Personal accounts can be startups too, but the signal is weak;
            # organizations take precedence in entity resolution.
            is_org = owner.get("type") == "Organization"
            signals.append(
                Signal(
                    kind="repo_momentum",
                    summary=(
                        f"{repo['full_name']} · ⭐{repo['stargazers_count']} · "
                        f"{(repo.get('description') or '')[:120]}"
                    ),
                    date=datetime.fromisoformat(repo["pushed_at"].replace("Z", "+00:00")),
                    source=Source(name=self.name, url=repo["html_url"], confidence="primary"),
                    raw={
                        "stars": repo["stargazers_count"],
                        "forks": repo.get("forks_count"),
                        "language": repo.get("language"),
                        "created_at": repo.get("created_at"),
                        "is_org": is_org,
                        "owner": owner.get("login"),
                        "description": repo.get("description"),
                        "topics": repo.get("topics", []),
                    },
                    candidate_name=owner.get("login") if is_org else None,
                    candidate_domain=registrable_domain(repo.get("homepage") or ""),
                )
            )
        return CollectionResult(source=self.name, signals=signals)


def public_profile(username: str, *, source_policy=None) -> dict:
    """Fetch one GitHub account's **public** profile — the TeamAnalyst's tool.

    Only publicly exposed fields are read (docs/03 §11, GDPR/KVKK note).
    """
    collector = Collector(source_policy=source_policy)
    collector.name = "github"
    try:
        return collector.fetch_json(f"https://api.github.com/users/{username}")
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
