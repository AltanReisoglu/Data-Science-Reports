"""The tool surface, defined once and consumed twice.

Two callers need these tools: the AutoGen agent holding a conversation, and the
MCP server that OpenClaw spawns. Writing them twice would be the obvious mistake
and a slow one — the second copy drifts, and the drift shows up as OpenClaw
giving a different answer to the same question than the web chat does, which is
the exact failure this whole gateway is supposed to make impossible.

So: plain functions with docstrings, here. `conversation.py` wraps them in
`FunctionTool` (which reads the signature and the docstring to build a schema);
`mcp_server.py` registers the same objects with `FastMCP` (which does the same
thing). Neither owns them.

### The part that stdio forces

The MCP server runs in a **separate process** that OpenClaw starts. It cannot see
the gateway's memory, so it cannot be handed the in-process scan state. That is
why sourcing is injected: `Sources` carries a *reader* and a *starter*, and there
are two implementations —

* in-process (the gateway passes its own accessors), and
* on-disk (`disk_sources`), which reads the state directory and, for anything
  that must *change* state, calls the running gateway over loopback.

`start_scan` from the MCP side therefore reports plainly when no gateway is
running rather than starting a second, invisible scan whose output the operator
would never see.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable

import config


@dataclass
class Sources:
    """Where a tool gets its facts, and who it asks to change something."""

    scan_getter: Callable[[], dict | None]
    scan_starter: Callable[[str, int], Any] | None = None
    session_id: str = ""


# --------------------------------------------------------------------------- disk


def latest_scan_file():
    files = sorted(config.OUTPUT.glob("scan-*.json"), reverse=True)
    return files[0] if files else None


def read_latest_scan() -> dict | None:
    """The most recent scan, read from the state directory. No gateway required."""
    path = latest_scan_file()
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    data["_source_name"] = path.name
    return data


def gateway_scan_starter(base_url: str | None = None):
    """Ask the running gateway to start a scan. Fails loudly when it is not up."""

    def start(query: str, days: int = 7):
        import urllib.error
        import urllib.request

        url = (base_url or config_gateway_url()).rstrip("/") + "/api/scan"
        body = json.dumps({"query": query, "days": days}).encode("utf-8")
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            # Starting a scan in this process would produce output nobody is
            # watching and a second writer to the state directory. Refusing is
            # the honest answer.
            raise RuntimeError(
                f"no gateway reachable at {url} ({exc}). Start it with "
                f"`python -m pipeline.server` and try again."
            ) from exc

    return start


def config_gateway_url() -> str:
    import os

    return os.getenv("VC_GATEWAY_URL", "http://127.0.0.1:8777")


def disk_sources() -> Sources:
    """Sourcing for a process that has no gateway state of its own."""
    return Sources(scan_getter=read_latest_scan, scan_starter=gateway_scan_starter())


# --------------------------------------------------------------------------- query
#
# The scan has always held sectors, country, five score axes, a decision and a
# `missing_data` list. Until now the only way to reach any of it was to already
# know a company's name, which answers the wrong question: the point of a funnel
# is to ask *across* what came out of it.
#
# Deterministic on purpose. Filtering and sorting are things code does correctly
# and a model does approximately, and a wrong answer here is invisible — a company
# quietly missing from a list looks exactly like a company that did not qualify.


def _parts(candidate: dict) -> dict:
    """Score total, missing data and reliability. One implementation, in dashboard."""
    import dashboard

    return dashboard._candidate_parts(candidate)


def _axis_names() -> list[str]:
    import dashboard

    return [key for key, _ in dashboard.AXES]


def _sources_of(company: dict, limit: int = 2) -> list[str]:
    urls = []
    for signal in company.get("signals", []):
        url = (signal.get("source") or {}).get("url")
        if url and url not in urls:
            urls.append(url)
        if len(urls) >= limit:
            break
    return urls


def _row(candidate: dict, dry: bool) -> str:
    """One line. Carries its sources and its gaps, because a row without them lies."""
    part = _parts(candidate)
    company = part["company"]
    score = part["score"] or {}
    bits = [f"{part['name']}"]

    total = part["total"]
    if total is None:
        bits.append("not scored")
    else:
        bits.append(f"{total}/25{' (dry-run placeholder)' if dry else ''}")
    if score.get("decision"):
        bits.append(str(score["decision"]))
    if company.get("sectors"):
        bits.append("/".join(company["sectors"][:2]))
    if company.get("country"):
        bits.append(str(company["country"]))

    line = " · ".join(bits)
    # A missing field and a low score are different findings. Collapsing them is
    # how a rubric starts lying: "team 0" reads as judged, "team missing" as unseen.
    if part["missing"]:
        line += f"\n    missing: {', '.join(part['missing'])}"
    urls = _sources_of(company)
    if urls:
        line += "\n    " + " ".join(urls)
    else:
        line += "\n    no source URL recorded"
    return line


def _query(
    data: dict | None,
    *,
    min_total: int = 0,
    axis: str = "",
    min_axis: int = 0,
    sector: str = "",
    decision: str = "",
    country: str = "",
    without_missing: str = "",
    sort: str = "-total",
    limit: int = 10,
) -> str:
    if not data:
        return "No scan has been run yet."

    candidates = data.get("candidates", [])
    dry = data.get("mode") == "dry"
    axes = _axis_names()
    if axis and axis not in axes:
        return f"Unknown axis {axis!r}. Available: {', '.join(axes)}."

    # Counted per filter so an empty result can say which one emptied it, rather
    # than leaving "no matches" to be read as "no such companies exist".
    dropped: dict[str, int] = {}
    kept = []
    for candidate in candidates:
        part = _parts(candidate)
        company, score = part["company"], part["score"] or {}

        checks = [
            ("min_total", part["total"] is None or part["total"] < min_total if min_total else False),
            ("axis", bool(axis) and int(score.get(axis, 0) or 0) < min_axis),
            ("sector", bool(sector) and not any(
                sector.lower() in str(s).lower() for s in company.get("sectors", []))),
            ("decision", bool(decision) and str(score.get("decision", "")).lower() != decision.lower()),
            ("country", bool(country) and country.lower() not in str(company.get("country") or "").lower()),
            ("without_missing", bool(without_missing) and without_missing in part["missing"]),
        ]
        failed = next((name for name, bad in checks if bad), None)
        if failed:
            dropped[failed] = dropped.get(failed, 0) + 1
        else:
            kept.append(candidate)

    reverse = sort.startswith("-")
    key = sort.lstrip("-") or "total"
    if key == "name":
        kept.sort(key=lambda c: _parts(c)["name"].lower(), reverse=reverse)
    elif key in axes:
        kept.sort(key=lambda c: int((_parts(c)["score"] or {}).get(key, 0) or 0), reverse=reverse)
    else:
        kept.sort(key=lambda c: _parts(c)["total"] or -1, reverse=reverse)

    applied = [
        f"{name}={value}" for name, value in (
            ("min_total", min_total), ("axis", f"{axis}>={min_axis}" if axis else ""),
            ("sector", sector), ("decision", decision), ("country", country),
            ("without_missing", without_missing),
        ) if value
    ]
    header = f"{len(kept)} of {len(candidates)} candidate(s)"
    if applied:
        header += f" · filters: {', '.join(applied)}"
    if dry:
        header += "\nScan ran in dry mode — every score below is a placeholder."

    if not kept:
        # Owed an explanation: which filter did the work.
        why = ", ".join(f"{name} removed {count}" for name, count in sorted(dropped.items()))
        return (
            f"{header}\nNothing matched. {why or 'The scan has no candidates.'}\n"
            "That is a statement about this scan, not about the world — widen the "
            "filter or run a new scan."
        )

    body = "\n".join(_row(c, dry) for c in kept[:limit])
    more = f"\n… {len(kept) - limit} more" if len(kept) > limit else ""
    return f"{header}\n\n{body}{more}"


def _compare(data: dict | None, names: str) -> str:
    if not data:
        return "No scan has been run yet."

    import live

    wanted = [n.strip() for n in (names or "").split(",") if n.strip()]
    if len(wanted) < 2:
        return "Give at least two company names, separated by commas."

    found, missing = [], []
    for name in wanted:
        match = next(
            (c for c in data.get("candidates", [])
             if _parts(c)["name"].lower() == name.lower()),
            None,
        )
        (found.append(match) if match else missing.append(name))

    if len(found) < 2:
        known = ", ".join(live.company_names(data)) or "none"
        return f"Not enough of those are in this scan (missing: {', '.join(missing)}). Known: {known}"

    axes = _axis_names()
    parts = [_parts(c) for c in found]
    width = max(len("technical_depth"), *(len(p["name"]) for p in parts)) + 2

    lines = [" " * 17 + "".join(p["name"][: width - 2].ljust(width) for p in parts)]
    for axis in axes + ["TOTAL"]:
        cells = []
        thin = 0
        for part in parts:
            if axis == "TOTAL":
                cells.append(f"{part['total']}/25" if part["total"] is not None else "—")
                continue
            value = str((part["score"] or {}).get(axis, "—"))
            if _gaps_for(axis, part["missing"]):
                # The interesting distinction: a low score that was *judged*, and
                # a low score standing in for something nobody could establish.
                cells.append(f"{value}?")
                thin += 1
            else:
                cells.append(value)
        row = axis.ljust(17) + "".join(c.ljust(width) for c in cells)
        if thin == len(parts) and axis != "TOTAL":
            row += " ← thin for both"
        lines.append(row)

    lines.append("\n? = the scorer recorded a gap that bears on this axis:")
    for part in parts:
        gaps = ", ".join(part["missing"]) if part["missing"] else "nothing recorded as missing"
        lines.append(f"  {part['name']}: {gaps}")

    if missing:
        lines.append(f"\nNot in this scan: {', '.join(missing)}")
    if data.get("mode") == "dry":
        lines.append("\nDry-run scan: these are placeholder scores.")
    return "\n".join(lines)


# `missing_data` is written by the scorer in its own words — real output includes
# `founder_identities`, `customer_benchmarks`, `technical_architecture`. Those are
# more useful than a fixed vocabulary would be, and they do not line up with axis
# names, so relating the two takes a keyword map.
#
# It is a heuristic and is treated as one: a hit marks a score as *thin* (`5?`),
# never as absent, and the raw gap list is printed underneath so the reader can
# judge the link themselves. Guessing quietly would be worse than not guessing.
_GAP_WORDS = {
    "team": ("founder", "team", "people", "hiring", "background"),
    "technical_depth": ("technical", "architecture", "code", "repo", "benchmark", "stack"),
    "momentum": ("traction", "growth", "customer", "user", "revenue", "adoption"),
    "timing": ("timing", "competit", "market"),
    "thesis_fit": ("sector", "thesis", "domain"),
}


def _gaps_for(axis: str, missing: list[str]) -> list[str]:
    words = _GAP_WORDS.get(axis, ())
    return [m for m in missing if any(w in str(m).lower() for w in words)]


# --------------------------------------------------------------------------- tools


def build(sources: Sources) -> list[Callable[..., str]]:
    """The tools, closed over their sourcing. Docstrings are the schema."""

    def scan_facts() -> str:
        """Everything the most recent scan established: funnel counts, cost, candidates."""
        import answers

        data = sources.scan_getter()
        return answers.facts(data) if data else "No scan has been run yet."

    def company_detail(name: str) -> str:
        """Full detail for one candidate in the current scan.

        Args:
            name: the company name as it appears in the scan.
        """
        import dashboard

        data = sources.scan_getter()
        if not data:
            return "No scan has been run yet."
        for candidate in data.get("candidates", []):
            part = dashboard._candidate_parts(candidate)
            if part["name"].lower() == name.lower().strip():
                return json.dumps(
                    {
                        "name": part["name"],
                        "score_total": part["total"],
                        "missing_data": part["missing"],
                        "branches": [
                            {"branch": b.get("branch"), "succeeded": b.get("succeeded")}
                            for b in candidate.get("branches", [])
                        ],
                        "signals": [
                            {"summary": s.get("summary"), "url": s.get("source", {}).get("url")}
                            for s in part["company"].get("signals", [])
                        ],
                    },
                    ensure_ascii=False,
                )
        known = [dashboard._candidate_parts(c)["name"] for c in data.get("candidates", [])]
        return f"No candidate named {name!r} in this scan. Candidates: {', '.join(known) or 'none'}"

    def query_companies(
        min_total: int = 0,
        axis: str = "",
        min_axis: int = 0,
        sector: str = "",
        decision: str = "",
        country: str = "",
        without_missing: str = "",
        sort: str = "-total",
        limit: int = 10,
    ) -> str:
        """Find candidates in the current scan by score, sector, decision or data quality.

        This is the tool for any question about *several* companies — "which
        fintechs scored above 15", "what did we decide to skip", "which ones are
        missing team data". `company_detail` answers about one, by name.

        Args:
            min_total: minimum total score out of 25.
            axis: one axis to filter on — thesis_fit, team, momentum, technical_depth, timing.
            min_axis: minimum value for that axis, 0-5.
            sector: match against the company's sectors, case-insensitive substring.
            decision: review, watch or skip.
            country: case-insensitive match on country.
            without_missing: exclude candidates missing this field, e.g. `team`.
            sort: `-total` (default, highest first), `total`, `name`, or an axis name.
            limit: how many rows to return.
        """
        return _query(
            sources.scan_getter(), min_total=min_total, axis=axis, min_axis=min_axis,
            sector=sector, decision=decision, country=country,
            without_missing=without_missing, sort=sort, limit=limit,
        )

    def compare_companies(names: str) -> str:
        """Compare candidates axis by axis, including where neither has data.

        Args:
            names: company names separated by commas.
        """
        return _compare(sources.scan_getter(), names)

    def company_live(name: str) -> str:
        """Check a company from the scan against live sources and report what CHANGED.

        Use this whenever the question is about *now* rather than about the scan:
        "what's new with X", "did anything change", "is X still moving". The scan
        is a frozen record; this looks again.

        Args:
            name: the company name as it appears in the scan.
        """
        import live

        data = sources.scan_getter()
        if not data:
            return "No scan has been run yet."
        company = live.find_company(data, name)
        if company is None:
            return (
                f"No candidate named {name!r} in this scan. "
                f"Candidates: {', '.join(live.company_names(data)) or 'none'}"
            )
        return live.refresh(company).as_text()

    def search_docs(query: str) -> str:
        """Search the project's documentation and the AutoGen guides.

        Covers both halves of `docs/`: the official AutoGen Core and AgentChat
        guides, verbatim, and this project's own design, measurements and code
        guide. Use it for "how does X work", "why was Y chosen", "what does this
        class do" — anything about the framework or about our decisions, as
        opposed to about a scan.

        Args:
            query: what to look for, e.g. "ClosureAgent" or "why not handoffs".
        """
        import docs_index

        return docs_index.as_text(query, docs_index.search(query, k=4))

    def memory_search(query: str) -> str:
        """Search what has been written down in memory — notes and curated facts.

        Args:
            query: what to look for.
        """
        import memory

        return memory.as_text(query, memory.search(query, k=4))

    def memory_get(path: str, start: int = 1, end: int = 0) -> str:
        """Read a memory file, or a range of lines in one.

        Args:
            path: file path relative to the memory workspace, e.g. `memory/2026-08-14.md`.
            start: first line, 1-based.
            end: last line; 0 means to the end of the file.
        """
        import memory

        return memory.get(path, start, end or None)

    def memory_note(text: str, tag: str = "") -> str:
        """Write something down so it can be found later. Not injected into context.

        Args:
            text: what to remember, in full sentences.
            tag: optional short label, e.g. `scan` or `preference`.
        """
        import memory

        path = memory.note(text, tag=tag)
        return f"Noted in {path.name}."

    def search_github(query: str) -> str:
        """Search GitHub for repositories matching a query, live.

        Args:
            query: search terms, e.g. "vector database rust".
        """
        from collectors.github import GitHub

        result = GitHub().run(query=query, days=90)
        if not result.succeeded:
            return f"GitHub lookup failed: {result.error}"
        return json.dumps(
            [
                {"summary": s.summary, "url": s.source.url, "stars": s.raw.get("stars")}
                for s in result.signals[:8]
            ],
            ensure_ascii=False,
        )

    def search_hacker_news(query: str) -> str:
        """Search Hacker News for a company or topic, live.

        Args:
            query: search terms.
        """
        from collectors.hackernews import HackerNews

        result = HackerNews().run(query=query, days=90)
        if not result.succeeded:
            return f"Hacker News lookup failed: {result.error}"
        return json.dumps(
            [
                {"title": s.summary, "date": s.date.date().isoformat(), "url": s.source.url}
                for s in result.signals[:8]
            ],
            ensure_ascii=False,
        )

    async def openclaw_call(method: str, params_json: str = "{}") -> str:
        """Call an OpenClaw Gateway method — sessions, channels, cron, agents, health.

        This reaches OpenClaw's real control plane, not just its conversations.
        Reading is free; anything that changes state needs the operator's
        approval; credentials, config and exec approvals are not reachable at all.

        Use `openclaw_methods` first if you are unsure what a method is called.

        Args:
            method: the Gateway method, e.g. `sessions.list`, `channels.status`, `cron.list`.
            params_json: JSON object of parameters, e.g. `{"limit": 10}`.
        """
        import openclaw_control

        try:
            params = json.loads(params_json or "{}")
        except json.JSONDecodeError as exc:
            return f"params_json is not valid JSON: {exc}"
        if not isinstance(params, dict):
            return "params_json must be a JSON object."

        outcome = await openclaw_control.call(method, params)
        if not outcome.get("ok"):
            return f"{method} failed ({outcome.get('tier', '?')}): {outcome.get('error')}"
        body = outcome["result"]
        text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
        return f"[{outcome['tier']}] {method}\n{text[:4000]}"

    def openclaw_methods() -> str:
        """Which OpenClaw Gateway methods can be called, and which are refused."""
        import openclaw_control

        return json.dumps(openclaw_control.methods(), ensure_ascii=False, indent=1)

    def start_scan(query: str, days: int = 7) -> str:
        """Run the funnel again for a sector. Returns immediately; the run continues.

        Args:
            query: sector or topic to scan.
            days: how many days back to look.
        """
        if sources.scan_starter is None:
            return "This process cannot start scans."
        try:
            sources.scan_starter(query, days)
        except RuntimeError as exc:
            return f"Not started: {exc}"
        return f"Scan started for {query!r} over {days} days. It takes under a minute."

    return [
        scan_facts,
        query_companies,
        compare_companies,
        company_detail,
        company_live,
        search_docs,
        memory_search,
        memory_get,
        memory_note,
        search_github,
        search_hacker_news,
        openclaw_call,
        openclaw_methods,
        start_scan,
    ]


def named(sources: Sources) -> dict[str, Callable[..., str]]:
    return {fn.__name__: fn for fn in build(sources)}


__all__ = [
    "Sources",
    "build",
    "disk_sources",
    "gateway_scan_starter",
    "latest_scan_file",
    "named",
    "read_latest_scan",
]
