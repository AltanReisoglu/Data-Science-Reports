"""The investment memo — the narrowest and most expensive step of the funnel.

Roughly five of these a day, on the strongest tier. Everything above exists to
decide which companies deserve one.

The memo is assembled from what the branches actually reported. Failed branches
are passed in explicitly as open questions, so a gap in the evidence surfaces to
the reader as a question to the founders rather than as silence.
"""

from __future__ import annotations

import config
import engine
from agents import analysts
from schemas import Candidate, InvestmentMemo, Source


def _dry_script(candidate: Candidate) -> list[str]:
    company = candidate.company
    missing = candidate.failed_branches
    return [
        InvestmentMemo(
            company_name=company.name,
            thesis=(config.THESIS.as_prompt()),
            summary=(
                f"Dry mode: no model was called. {company.one_liner()}. "
                f"This text is a placeholder produced by the deterministic path."
            ),
            risks=["Dry mode: risks were not assessed by a model."],
            questions=(
                [f"The {b} branch returned nothing — what is the actual position there?" for b in missing]
                or ["Dry mode: no questions were generated."]
            ),
            references=company.sources[:10],
        ).model_dump_json()
    ]


def _task(candidate: Candidate) -> str:
    company = candidate.company
    lines = [
        f"Company: {company.name}",
        f"Domain: {company.domain or '—'}",
        f"GitHub: {company.github or '—'}",
        "",
        "Analyst findings:",
    ]
    for branch in candidate.branches:
        if branch.succeeded:
            lines.append(f"[{branch.branch}] {branch.text}")
        else:
            lines.append(f"[{branch.branch}] NO RESULT — {branch.error}")

    if candidate.score is not None:
        score = candidate.score
        lines += [
            "",
            f"Score: {score.total}/25 (reliability: {score.reliability})",
            f"Missing data: {'; '.join(score.missing_data) or 'none'}",
        ]

    lines += ["", "Sources:"]
    for source in company.sources[:15]:
        lines.append(f"  - {source.name}: {source.url}")
    return "\n".join(lines)


async def write(
    candidate: Candidate, ledger: engine.Ledger, thesis: config.Thesis | None = None
) -> InvestmentMemo | None:
    """Produce the memo. Returns ``None`` if the model gave no structured answer."""
    from autogen_agentchat.messages import StructuredMessage

    writer = analysts.build_memo_writer(candidate.company, ledger, _dry_script(candidate))
    result = await writer.run(task=_task(candidate))

    for message in reversed(result.messages):
        if isinstance(message, StructuredMessage) and isinstance(message.content, InvestmentMemo):
            memo = message.content
            # References are guaranteed from the collectors rather than trusted
            # to the model: a memo without sources must not be possible.
            if not memo.references:
                memo.references = candidate.company.sources[:10]
            return memo
    return None


def render_markdown(candidate: Candidate) -> str:
    """Render one candidate as the Markdown a human actually reads."""
    company = candidate.company
    score = candidate.score
    memo = candidate.memo

    out = [f"## {company.name}", ""]
    if company.domain:
        out.append(f"**Domain:** {company.domain}  ")
    if company.github:
        out.append(f"**GitHub:** `{company.github}`  ")
    out.append(f"**Signals:** {len(company.signals)}  ")

    if score is not None:
        out += [
            f"**Score:** {score.total}/25 · decision: `{score.decision}` · "
            f"reliability: {score.reliability}",
            "",
            "| axis | score | rationale |",
            "|---|---:|---|",
        ]
        for axis in ("thesis_fit", "team", "momentum", "technical_depth", "timing"):
            out.append(
                f"| {axis} | {getattr(score, axis)} | {score.rationale.get(axis, '—')} |"
            )
        if score.missing_data:
            out += ["", "**Missing data:**"] + [f"- {m}" for m in score.missing_data]

    if memo is not None:
        out += ["", "### Memo", "", memo.summary]
        if memo.risks:
            out += ["", "**Risks**"] + [f"- {r}" for r in memo.risks]
        if memo.questions:
            out += ["", "**Questions for the founders**"] + [f"- {q}" for q in memo.questions]

    out += ["", "**Sources**"]
    for source in _unique_sources(company.sources):
        out.append(f"- [{source.name}]({source.url})")
    out.append("")
    return "\n".join(out)


def _unique_sources(sources: list[Source]) -> list[Source]:
    seen: dict[str, Source] = {}
    for source in sources:
        seen.setdefault(source.url, source)
    return list(seen.values())
