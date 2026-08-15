"""Question routing — the one place that decides what a question is asking for.

This was JavaScript inside the generated page. Moving it here makes it the
server's job, which buys three things: the routing is testable, the answers and
the router live next to each other, and the same logic serves both the static
page and the API.

Two answer paths, and the page always says which one it used:

* **rules** — the question is matched against keyword sets (and, first, against
  the candidate names, since a company name is the most specific thing a reader
  can be asking for). The answer is a block rendered from the scan's own data.
  No model, no network, deterministic.
* **model** — only when an LLM is configured. The model is given the scan's facts
  and answers in prose. It is told, in its system prompt, that inventing a fact
  not present in those facts is the one thing it must not do.

Without an LLM the rules path is the whole system, and that is a working state
rather than a degraded one.
"""

from __future__ import annotations

import json
from typing import Any

import config
import dashboard

# Keyword sets, longest match wins. Deliberately plain: a reader should be able
# to predict what their question will hit.
INTENTS: list[tuple[str, list[str]]] = [
    ("summary", ["summary", "overview", "recap", "tldr", "what happened", "result"]),
    ("funnel", ["funnel", "stage", "how many", "pipeline", "counts", "narrow"]),
    ("cost", ["cost", "token", "spend", "price", "expensive", "cheap", "budget", "call"]),
    ("candidates", ["candidate", "compan", "startup", "list", "top", "best", "rank", "score"]),
    ("missing", ["missing", "gap", "unknown", "reliab", "confiden", "weak", "trust", "blind"]),
    ("team", ["founder", "team", "who built", "who is behind", "people", "ceo", "cto"]),
    ("sources", ["source", "where", "data from", "sec", "form d", "github", "hacker", "hn", "arxiv"]),
    ("rejected", ["reject", "drop", "filter", "triage", "exclud", "skip"]),
    ("mode", ["dry", "real", "live", "placeholder", "thesis", "fake", "model called"]),
    ("method", ["how", "method", "work", "architect", "design", "explain the", "process"]),
    ("help", ["help", "what can", "commands", "ask you"]),
]


def catalogue(data: dict) -> dict[str, tuple[str, str]]:
    """Every answer this scan can give, keyed by intent."""
    out = {key: (title, body) for key, title, body in dashboard._answers(data)}
    for index, candidate in enumerate(data.get("candidates", [])):
        part = dashboard._candidate_parts(candidate)
        out[f"company:{index}"] = (part["name"], dashboard._candidate_full(candidate))
    return out


# Words that mean "now", not "in the scan". A question carrying one of these AND
# a company name is asking for a live check rather than the stored record.
LIVE_WORDS = [
    "new", "latest", "chang", "updat", "since", "now", "current", "recent",
    "still", "moving", "güncel", "yeni", "değiş", "son durum",
]


def live_target(question: str, data: dict) -> str | None:
    """The company a 'what changed' question is about, if there is one."""
    q = question.lower().strip()
    if not any(word in q for word in LIVE_WORDS):
        return None
    import live as live_module

    for name in live_module.company_names(data):
        if name and name.lower() in q:
            return name
    return None


def route(question: str, data: dict) -> str | None:
    q = question.lower().strip()
    if not q:
        return None

    # A company name beats every general intent.
    for index, candidate in enumerate(data.get("candidates", [])):
        name = dashboard._candidate_parts(candidate)["name"].lower()
        if name and (name in q or q and name.startswith(q)):
            return f"company:{index}"

    best, best_score = None, 0
    for key, words in INTENTS:
        score = sum(len(word) for word in words if word in q)
        if score > best_score:
            best, best_score = key, score
    return best


def facts(data: dict) -> str:
    """The compact ground truth handed to the model. Nothing outside this is true."""
    funnel = data.get("funnel", {})
    cost = data.get("cost", {})
    candidates = []
    for candidate in data.get("candidates", []):
        part = dashboard._candidate_parts(candidate)
        company = part["company"]
        candidates.append(
            {
                "name": part["name"],
                "domain": company.get("domain"),
                "github": company.get("github"),
                "score_total": part["total"],
                "decision": (part["score"] or {}).get("decision"),
                "axes": {k: (part["score"] or {}).get(k) for k, _ in dashboard.AXES},
                "rationale": (part["score"] or {}).get("rationale", {}),
                "missing_data": part["missing"],
                "branches": [
                    {"branch": b.get("branch"), "succeeded": b.get("succeeded"), "error": b.get("error")}
                    for b in candidate.get("branches", [])
                ],
                "signals": [
                    {
                        "kind": s.get("kind"),
                        "summary": s.get("summary"),
                        "date": s.get("date"),
                        "source": s.get("source", {}).get("name"),
                        "url": s.get("source", {}).get("url"),
                    }
                    for s in company.get("signals", [])
                ],
            }
        )
    return json.dumps(
        {
            "query": data.get("query"),
            "window_days": data.get("days"),
            "mode": data.get("mode"),
            "thesis_is_placeholder": data.get("thesis_is_placeholder"),
            "thesis": config.THESIS.as_prompt(),
            "funnel": funnel,
            "cost": {"llm_calls": cost.get("llm_cagrisi"), "tokens": cost.get("toplam_token")},
            "failed_sources": data.get("failed_sources"),
            "candidates": candidates,
        },
        ensure_ascii=False,
        indent=1,
    )


SYSTEM = """You answer questions about one VC pipeline scan. The JSON below is
everything that scan established — it is the only ground truth you have.

Rules, in order of importance:
1. Never state a fact that is not in the JSON. If it is not there, say plainly
   that this scan did not establish it, and say what it did establish nearby.
2. A missing value and a low value are different things. Never present an
   absence of information as a finding.
3. Cite the source URL when you refer to a signal.
4. If the scan ran in dry mode, the scores are placeholders and you must say so
   whenever you quote one.
5. Be brief. Two or three short paragraphs at most, no preamble.

SCAN DATA:
{facts}"""


async def model_answer(question: str, data: dict) -> str | None:
    """Prose answer from the configured LLM. ``None`` if there is no LLM."""
    if not config.live_llm_available():
        return None

    from autogen_agentchat.agents import AssistantAgent

    import engine

    ledger = engine.Ledger()
    try:
        agent = AssistantAgent(
            "ScanAnalyst",
            model_client=ledger.client("mid"),
            description="Answers questions about a completed scan.",
            system_message=SYSTEM.format(facts=facts(data)),
        )
        result = await agent.run(task=question)
        content = getattr(result.messages[-1], "content", "")
        return str(content) if content else None
    except Exception as e:
        return f"__ERROR__{type(e).__name__}: {e}"
    finally:
        await ledger.close()


# Documentation is consulted on **explicit intent only**, never as a catch-all
# for questions nothing else matched.
#
# Measured why: scoring cannot separate a real documentation question from a
# stray one. "What is the weather in Istanbul" scores 30.8 — higher than
# "model_info required" (17.6) — because the AgentChat guide really does contain
# a `get_weather` tool example. The search is not wrong there; presenting it as
# an answer would be. So the gate is the vocabulary of the subject, not a score,
# and a question that matches nothing still gets the honest refusal.
DOC_WORDS = [
    # frameworks and products
    "autogen", "agentchat", "langgraph", "crewai", "metagpt", "semantic kernel",
    "agent framework", "adk", "framework",
    # core concepts
    "runtime", "workbench", "mcp", "topic", "subscription", "actor", "pub/sub",
    "message handler", "agent id", "lifecycle", "component config",
    # patterns
    "graphflow", "graph flow", "handoff", "swarm", "group chat", "reflection",
    "debate", "mixture of agents", "concurrent agent", "sequential workflow",
    "fan-out", "fan-in", "join",
    # api surface
    "closureagent", "routedagent", "assistantagent", "model_info", "model context",
    "model client", "cancellationtoken", "save_state", "output_content_type",
    "termination", "intervention", "tool call", "structured output", "streaming",
    # asking about the system itself
    "documentation", "docs", "how does", "how do i", "what is a", "what does",
    "why not", "why did", "nasıl çalış", "ne işe yarar", "nedir", "neden",
]


def render_docs(query: str, hits) -> str:
    """Documentation hits, each carrying where it came from."""
    if not hits:
        return (
            f'<p class="answer__lede">Nothing in <code>docs/</code> matches '
            f"{dashboard.e(query)!r}.</p><p>Those documents cover AutoGen — the Core and "
            "AgentChat guides, verbatim — and this project's own design, measurements "
            "and code guide.</p>"
        )
    blocks = []
    for hit in hits:
        section = hit.section
        blocks.append(
            f'<div class="gap"><h4>{dashboard.e(section.title)}</h4>'
            f'<p class="note">{dashboard.e(section.provenance)} · '
            f"<code>{dashboard.e(section.doc)}:{section.line}</code></p>"
            f"<p>{dashboard.e(section.snippet())}</p></div>"
        )
    return (
        f'<p class="answer__lede">{len(hits)} section(s) in the documentation for '
        f"{dashboard.e(query)!r}.</p>" + "".join(blocks)
    )


def render_live(report) -> str:
    """A live report as the same kind of block every other answer returns."""
    changes = (
        "".join(f"<li>{dashboard.e(c)}</li>" for c in report.changes)
        or "<li>No change found in the sources that answered.</li>"
    )
    rows = "".join(
        f'<tr><th scope="row">{dashboard.e(c.source)}</th>'
        f"<td>{'checked' if c.ok else 'COULD NOT CHECK'}</td>"
        f"<td>{dashboard.e(c.detail or c.error or '')}</td></tr>"
        for c in report.checks
    )
    warning = ""
    if report.failed:
        warning = (
            f'<p class="note">{dashboard.e(", ".join(report.failed))} could not be checked. '
            "That is not the same as no change.</p>"
        )
    seen = (
        f"the scan had seen up to {report.scan_seen_until:%Y-%m-%d}"
        if report.scan_seen_until else "the scan recorded no dated signal"
    )
    return f"""<p class="answer__lede"><strong>{dashboard.e(report.company)}</strong> checked live at
    {report.checked_at:%Y-%m-%d %H:%M} UTC — {seen}.</p>
    <div class="gap"><h4>Changed since the scan</h4><ul>{changes}</ul></div>
    {warning}
    <div class="tablewrap"><table><thead><tr><th scope="col">Source</th>
    <th scope="col">Status</th><th scope="col">Detail</th></tr></thead>
    <tbody>{rows}</tbody></table></div>"""


async def answer(question: str, data: dict, *, prefer_model: bool = True) -> dict[str, Any]:
    """Answer a question, saying which path produced it."""
    # A live question is answered by looking, not by reading the scan — and this
    # works with no model configured, because the sources are HTTP not LLM.
    target = live_target(question, data)
    if target is not None:
        import asyncio

        import live as live_module

        company = live_module.find_company(data, target)
        if company is not None:
            report = await asyncio.to_thread(live_module.refresh, company)
            return {
                "path": "live",
                "title": f"{report.company} — live",
                "text": None,
                "html": render_live(report),
                "supporting_title": None,
            }

    key = route(question, data)
    entries = catalogue(data)

    # An explicit documentation question goes to the documents, not to the scan.
    # Note what is *not* here: `key is None`. An unmatched question keeps its
    # refusal rather than being handed whatever the corpus ranked first.
    if any(word in question.lower() for word in DOC_WORDS):
        import docs_index

        hits = docs_index.search(question, k=4)
        if hits and hits[0].score > 5.0:
            return {
                "path": "docs",
                "title": "From the documentation",
                "text": None,
                "html": render_docs(question, hits),
                "supporting_title": None,
            }

    if prefer_model and config.live_llm_available():
        prose = await model_answer(question, data)
        if prose and not prose.startswith("__ERROR__"):
            supporting = entries.get(key) if key else None
            return {
                "path": "model",
                "title": "Answer",
                "text": prose,
                # The rules path still supplies the evidence block, so a prose
                # answer never arrives without the data behind it.
                "html": supporting[1] if supporting else "",
                "supporting_title": supporting[0] if supporting else None,
            }
        if prose:
            note = prose.replace("__ERROR__", "")
            return {
                "path": "rules",
                "title": "Model unavailable",
                "text": f"The model call failed ({note}). Answering from the scan data instead.",
                "html": entries[key][1] if key else entries["help"][1],
                "supporting_title": entries[key][0] if key else entries["help"][0],
            }

    if key is None:
        title, body = entries["help"]
        return {
            "path": "rules",
            "title": "Not something I hold",
            "text": "This scan does not hold an answer to that. Here is what it does hold.",
            "html": body,
            "supporting_title": title,
        }

    title, body = entries[key]
    return {"path": "rules", "title": title, "text": None, "html": body, "supporting_title": None}
