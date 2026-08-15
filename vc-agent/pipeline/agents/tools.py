"""Analyst tools — every one of them passes through the policy gate.

Tools return their **source URL alongside the data**. The reason is discipline
rather than schema: if the text in front of an agent carries no link, the agent
cannot put a link in the memo, and an unsourced sentence cannot enter the memo
(docs/03 §3.3).
"""

from __future__ import annotations

import json

from collectors.arxiv import ArXiv
from collectors.github import GitHub, public_profile
from collectors.hackernews import HackerNews


def inspect_repository(full_name: str) -> str:
    """Fetch public momentum data for a GitHub repository.

    Args:
        full_name: repository in "owner/repo" form.
    """
    collector = GitHub()
    try:
        data = collector.fetch_json(f"https://api.github.com/repos/{full_name}")
    except Exception as e:
        return json.dumps(
            {"error": f"{type(e).__name__}: {e}", "repository": full_name}, ensure_ascii=False
        )
    return json.dumps(
        {
            "repository": data.get("full_name"),
            "stars": data.get("stargazers_count"),
            "forks": data.get("forks_count"),
            "open_issues": data.get("open_issues_count"),
            "language": data.get("language"),
            "created_at": data.get("created_at"),
            "last_push": data.get("pushed_at"),
            "license": (data.get("license") or {}).get("spdx_id"),
            "source_url": data.get("html_url"),
        },
        ensure_ascii=False,
    )


def search_market_chatter(query: str) -> str:
    """Search Hacker News for this company or topic over the last 90 days.

    Args:
        query: company name or topic to search for.
    """
    try:
        result = HackerNews().run(query=query, days=90)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    if not result.succeeded:
        return json.dumps({"error": result.error}, ensure_ascii=False)
    return json.dumps(
        [
            {
                "title": s.summary,
                "date": s.date.date().isoformat(),
                "points": s.raw.get("points"),
                "source_url": s.source.url,
            }
            for s in result.signals[:8]
        ],
        ensure_ascii=False,
    )


def publication_trace(name_or_topic: str) -> str:
    """Search arXiv for a founder's or company's publication trace.

    This is where arXiv belongs: it cannot discover a company (a paper carries
    no company name), but for a *named* founder it is direct evidence for the
    thesis requirement "technical founder".

    Args:
        name_or_topic: an author name or research topic.
    """
    result = ArXiv().run(query=name_or_topic, days=540)  # ~18 months
    if not result.succeeded:
        return json.dumps({"error": result.error}, ensure_ascii=False)
    return json.dumps(
        [
            {
                "title": s.summary,
                "date": s.date.date().isoformat(),
                "authors": s.raw.get("authors", [])[:6],
                "source_url": s.source.url,
            }
            for s in result.signals[:6]
        ],
        ensure_ascii=False,
    )


def founder_profile(github_username: str) -> str:
    """Fetch a GitHub account's **public** profile.

    Only publicly exposed fields are read; values such as email and location
    appear exactly as the account holder published them (GDPR/KVKK note,
    docs/03 §11).

    Args:
        github_username: the GitHub username.
    """
    data = public_profile(github_username)
    if "error" in data:
        return json.dumps(data, ensure_ascii=False)
    return json.dumps(
        {
            "username": data.get("login"),
            "name": data.get("name"),
            "bio": data.get("bio"),
            "company": data.get("company"),
            "public_repos": data.get("public_repos"),
            "followers": data.get("followers"),
            "joined": data.get("created_at"),
            "source_url": data.get("html_url"),
        },
        ensure_ascii=False,
    )
