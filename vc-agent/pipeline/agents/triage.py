"""Triage — the cheapest layer of the funnel. ~200 candidates -> ~20 enrichments.

Two stages: **rules first, cheap LLM second.**

1. ``prefilter()`` — if a red line or an obvious thesis match is present, the
   model is never called. Volume lives here, and the cheapest way to spend no
   tokens is not to make the call.
2. Whatever remains undecided goes to a single cheap ``AssistantAgent`` that
   returns a binary decision plus a rationale via ``output_content_type``.

**Principle 1 is encoded here** (docs/03 §3.1): *absence of information is not
grounds for rejection.* If the rule engine says "I don't know" the candidate is
not dropped but handed to the model; if the model is unsure it returns
``passed=True`` and lets the candidate through. A wrong rejection is invisible
and expensive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

import config
import engine
from schemas import Company


class TriageDecision(BaseModel):
    passed: bool
    rationale: str = Field(description="One sentence, grounded in evidence")


@dataclass
class TriageResult:
    passed: bool
    rationale: str
    used_llm: bool


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def prefilter(company: Company, thesis: config.Thesis | None = None) -> TriageResult | None:
    """Return a ``TriageResult`` when the rules are conclusive, else ``None``."""
    thesis = thesis or config.THESIS
    haystack = " ".join(
        [company.name, company.description or "", " ".join(company.sectors)]
        + [s.summary for s in company.signals]
    ).lower()
    tokens = _words(haystack)

    for red_line in thesis.red_lines:
        # A red line fires only on an explicit statement; implication is not enough.
        if red_line.lower() in haystack:
            return TriageResult(False, f"red line: {red_line}", False)

    # Third-party chatter alone is not a discovery. An entity whose only trace is
    # a news headline has produced no first-party evidence that it is a company
    # we can evaluate at all — no repository, no filing, no launch. Observed on
    # the first live runs, where `nvidia.com` and `bbc.com` reached the scoring
    # stage purely on being mentioned in headlines.
    #
    # This does not violate principle 1: we are not rejecting for absence of
    # information, we are rejecting on what the evidence we hold actually is.
    kinds = {s.kind for s in company.signals}
    if kinds and kinds <= {"news"} and not company.github:
        return TriageResult(
            False,
            "only third-party mentions; no first-party evidence (repository, filing or launch)",
            False,
        )

    hits = [s for s in thesis.sectors if _words(s) & tokens or s.lower() in haystack]
    if hits and len(company.signals) >= 2:
        return TriageResult(
            True,
            f"thesis sector matched ({', '.join(hits)}) with {len(company.signals)} signals",
            False,
        )

    # Nothing is known: do not reject, defer to the model (principle 1).
    return None


SYSTEM_MESSAGE = """You are the triage layer of a VC pipeline. Your job is not to
REJECT but to reject only what CLEARLY does not fit. When undecided, pass it on.

Investment thesis: {thesis}

Rule: absence of information is NOT grounds for rejection. Reject a company only
when there is evidence that it contradicts the thesis. If you would say "no
information", return passed=true and state what is missing in your rationale."""


def _dry_script(company: Company) -> list[str]:
    """The reply the cheap model gives in dry mode — deterministic."""
    return [
        TriageDecision(
            passed=True,
            rationale=(
                f"{len(company.signals)} signals present, no evidence contradicting "
                f"the thesis (dry mode)."
            ),
        ).model_dump_json()
    ]


async def decide(
    company: Company, ledger: engine.Ledger, thesis: config.Thesis | None = None
) -> TriageResult:
    """Rules first, cheap LLM only if needed."""
    by_rule = prefilter(company, thesis)
    if by_rule is not None:
        return by_rule

    from autogen_agentchat.agents import AssistantAgent

    thesis = thesis or config.THESIS
    client = ledger.client("cheap", _dry_script(company))
    agent = AssistantAgent(
        "Triage",
        model_client=client,
        description="Compares a candidate against the thesis and rejects or passes it.",
        system_message=SYSTEM_MESSAGE.format(thesis=thesis.as_prompt()),
        output_content_type=TriageDecision,
    )
    result = await agent.run(task=describe(company))
    last = result.messages[-1].content
    if isinstance(last, TriageDecision):
        return TriageResult(last.passed, last.rationale, True)
    # No structured answer: still no rejection — uncertainty is not grounds.
    return TriageResult(True, "triage returned no structured answer; candidate passed on", True)


def describe(company: Company) -> str:
    lines = [
        f"Company: {company.name}",
        f"Domain: {company.domain or '—'}",
        f"GitHub: {company.github or '—'}",
        f"Description: {company.description or '—'}",
        f"Tags: {', '.join(company.sectors) or '—'}",
        "Signals:",
    ]
    for s in company.signals[:8]:
        lines.append(f"  - [{s.kind}] {s.summary[:140]} ({s.source.name}, {s.date.date()})")
    return "\n".join(lines)
