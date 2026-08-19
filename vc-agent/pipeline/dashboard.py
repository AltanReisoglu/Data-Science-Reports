"""Render a scan result as a self-contained HTML dashboard.

    .venv/bin/python pipeline/dashboard.py            # newest scan
    .venv/bin/python pipeline/dashboard.py --json <path> --out <path>

No server, no build step, no network: one file that opens in a browser. The data
is rendered into markup by Python rather than by client-side JavaScript, so the
page is readable with scripting disabled; JS carries only interaction.

## Design

Two sets of rules, applied where each belongs.

**Chart layer** — one hue for magnitude, since every chart here is single-series
(funnel stages, cost per stage, score axes) and length already carries the value.
The funnel uses the ordinal ramp, whose lightest step is held above the 2:1 floor
against its surface. Status colors (good / warning / critical) are reserved for
genuine states — reliability, a failed branch, a source that fell over — and
every one of them ships with a text label, never colour alone. Each chart has a
table view underneath it.

**Interface layer** — system typography with size-specific tracking and leading,
translucent chrome over a scroll edge, press feedback on pointer-down, and
transitions tuned to a critically damped response (~0.35s, no overshoot).
`prefers-reduced-motion` and `prefers-reduced-transparency` are both honoured.

The dashboard's first job is honesty: dry mode, a placeholder thesis, a collector
that failed, a branch that returned nothing and the signals nobody could attribute
are all shown at the top, not buried. A scan that found little must say why.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402

# Ordinal ramp for the funnel: light stops no lighter than step 250, dark stops
# no darker than step 600, so the stage nearest the surface still clears 2:1.
FUNNEL_LIGHT = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
FUNNEL_DARK = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#184f95"]

STAGE_LABELS = [
    ("signals", "Signals collected", "no LLM"),
    ("companies", "Companies resolved", "no LLM"),
    ("triage_passed", "Passed triage", "cheap tier"),
    ("enriched", "Enriched", "mid tier, 3 parallel branches"),
    ("memos", "Memos written", "strong tier"),
]

# A live check needs a backend to call. The static export has none, so the button
# that promises one must not appear there — an affordance that cannot work is
# worse than a missing feature. `server.py` flips this on at import.
LIVE_AVAILABLE = False

AXES = [
    ("thesis_fit", "Thesis fit"),
    ("team", "Team"),
    ("momentum", "Momentum"),
    ("technical_depth", "Technical depth"),
    ("timing", "Timing"),
]


def e(text: object) -> str:
    return html.escape(str(text if text is not None else ""))


# --------------------------------------------------------------------------- pieces


def _status(level: str, label: str) -> str:
    """A status pill. The word carries the meaning; the colour only reinforces it."""
    icons = {"good": "●", "warning": "▲", "critical": "■", "neutral": "○"}
    return (
        f'<span class="pill pill--{level}">'
        f'<span class="pill__icon" aria-hidden="true">{icons.get(level, "○")}</span>'
        f"{e(label)}</span>"
    )


def _reliability_status(reliability: str) -> str:
    return {"high": "good", "medium": "warning", "low": "critical"}.get(reliability, "neutral")


def _funnel(funnel: dict) -> str:
    rows = [(key, label, note, int(funnel.get(key, 0) or 0)) for key, label, note in STAGE_LABELS]
    peak = max((count for _k, _l, _n, count in rows), default=1) or 1

    bars = []
    table_rows = []
    for index, (_key, label, note, count) in enumerate(rows):
        width = max(count / peak * 100, 0.6 if count else 0)
        bars.append(
            f"""<div class="funnel__row" data-tip="{e(label)}: {count}">
  <div class="funnel__label">{e(label)}<span class="funnel__note">{e(note)}</span></div>
  <div class="funnel__track"><div class="funnel__bar" style="width:{width:.2f}%;--step:{index}"></div></div>
  <div class="funnel__value">{count}</div>
</div>"""
        )
        table_rows.append(f"<tr><th scope=\"row\">{e(label)}</th><td>{count}</td><td>{e(note)}</td></tr>")

    drop = ""
    unattached = int(funnel.get("unattached_signals", 0) or 0)
    if unattached:
        drop = (
            f'<p class="note">{unattached} signals named no resolvable company and were '
            "left unattached — recorded rather than dropped quietly.</p>"
        )

    return f"""<section class="card" aria-labelledby="funnel-h">
  <h2 id="funnel-h">The funnel</h2>
  <p class="sub">Every layer costs more than the one above it. The shape is the point: spend little on many, much on few.</p>
  <div class="funnel">{''.join(bars)}</div>
  {drop}
  <details class="table-view"><summary>Data table</summary>
    <div class="tablewrap"><table><caption class="visually-hidden">Funnel stage counts</caption>
      <thead><tr><th scope="col">Stage</th><th scope="col">Count</th><th scope="col">Tier</th></tr></thead>
      <tbody>{''.join(table_rows)}</tbody>
    </table></div>
  </details>
</section>"""


def _cost(stage_costs: list[dict], total: dict) -> str:
    if not stage_costs:
        return ""
    peak = max(int(s.get("toplam_token", 0)) for s in stage_costs) or 1

    bars, table_rows = [], []
    for stage in stage_costs:
        name = str(stage.get("desen", "?"))
        tokens = int(stage.get("toplam_token", 0))
        calls = int(stage.get("llm_cagrisi", 0))
        width = max(tokens / peak * 100, 0.6 if tokens else 0)
        bars.append(
            f"""<div class="cost__row" data-tip="{e(name)}: {tokens} tokens across {calls} calls">
  <div class="cost__label">{e(name)}</div>
  <div class="cost__track"><div class="cost__bar" style="width:{width:.2f}%"></div></div>
  <div class="cost__value">{tokens:,}</div>
</div>"""
        )
        table_rows.append(
            f'<tr><th scope="row">{e(name)}</th><td>{calls}</td><td>{tokens:,}</td></tr>'
        )

    return f"""<section class="card" aria-labelledby="cost-h">
  <h2 id="cost-h">Cost by stage</h2>
  <p class="sub">{int(total.get('llm_cagrisi', 0))} model calls · {int(total.get('toplam_token', 0)):,} tokens · mode: {e(total.get('mod', '—'))}</p>
  <div class="cost">{''.join(bars)}</div>
  <details class="table-view"><summary>Data table</summary>
    <div class="tablewrap"><table><caption class="visually-hidden">Tokens and calls per stage</caption>
      <thead><tr><th scope="col">Stage</th><th scope="col">Calls</th><th scope="col">Tokens</th></tr></thead>
      <tbody>{''.join(table_rows)}</tbody>
    </table></div>
  </details>
</section>"""


def _axis_bars(score: dict) -> str:
    rows = []
    for key, label in AXES:
        value = int(score.get(key, 0) or 0)
        reason = score.get("rationale", {}).get(key, "")
        rows.append(
            f"""<div class="axis" data-tip="{e(label)}: {value} of 5 — {e(reason) or 'no rationale given'}">
  <div class="axis__label">{e(label)}</div>
  <div class="axis__track"><div class="axis__bar" style="width:{value / 5 * 100:.0f}%"></div></div>
  <div class="axis__value">{value}</div>
</div>"""
        )
    return f'<div class="axes">{"".join(rows)}</div>'


def _candidate_parts(candidate: dict) -> dict:
    """The pieces every view of a candidate needs, computed once."""
    company = candidate.get("company", {})
    score = candidate.get("score")
    missing = (score or {}).get("missing_data", []) or []
    return {
        "company": company,
        "score": score,
        "memo": candidate.get("memo"),
        "name": company.get("name", "unknown"),
        "total": sum(int(score.get(k, 0) or 0) for k, _ in AXES) if score else None,
        "missing": missing,
        "reliability": "high" if not missing else ("medium" if len(missing) <= 2 else "low"),
    }


def _candidate_brief(candidate: dict, index: int) -> str:
    """One row in the list — asking about it is the whole interaction."""
    part = _candidate_parts(candidate)
    company, score = part["company"], part["score"]
    meta = []
    if company.get("domain"):
        meta.append(e(company["domain"]))
    if company.get("github"):
        meta.append("github/" + e(company["github"]))
    meta.append(f'{len(company.get("signals", []))} signals')
    decision = (score or {}).get("decision")
    total = f'{part["total"]}<span class="score__of">/25</span>' if part["total"] is not None else "—"

    # Two actions per row: the scan's record, and what changed since it. The
    # second is the one the funnel's fourth principle needs — monitoring is a
    # loop, so a candidate is never a closed question.
    live_button = (
        f'<button class="brief__live" type="button" '
        f'data-live="{e(part["name"])}" '
        f'title="Check live sources for changes since the scan">Check now</button>'
        if LIVE_AVAILABLE else ""
    )
    return f"""<div class="brief">
  <button class="brief__main" type="button" data-ask="{e(part['name'])}">
    <span class="brief__id">
      <span class="brief__name">{e(part['name'])}</span>
      <span class="brief__meta">{' · '.join(meta)}</span>
    </span>
    <span class="brief__right">
      <span class="brief__score">{total}</span>
      {_status('neutral', decision) if decision else ''}
      {_status(_reliability_status(part['reliability']), part['reliability'] + ' reliability')}
    </span>
  </button>
  {live_button}
</div>"""


def _candidate_full(candidate: dict) -> str:
    """Everything known about one candidate, as an answer."""
    part = _candidate_parts(candidate)
    company, score, memo = part["company"], part["score"], part["memo"]

    branches = "".join(
        _status(
            "good" if branch.get("succeeded") else "critical",
            f'{branch.get("branch", "?")}: {"reported" if branch.get("succeeded") else "no result"}',
        )
        for branch in candidate.get("branches", [])
    )

    sources, seen = [], set()
    for signal in company.get("signals", []):
        source = signal.get("source", {})
        url = source.get("url")
        if url and url not in seen:
            seen.add(url)
            sources.append(
                f'<li><a href="{e(url)}" target="_blank" rel="noopener">'
                f'{e(source.get("name", "source"))} — {e(signal.get("summary", ""))[:90]}</a></li>'
            )

    missing_block = ""
    if part["missing"]:
        items = "".join(f"<li>{e(m)}</li>" for m in part["missing"])
        missing_block = f'<div class="missing"><h4>Missing data</h4><ul>{items}</ul></div>'

    memo_block = ""
    if memo:
        risks = "".join(f"<li>{e(r)}</li>" for r in memo.get("risks", []))
        questions = "".join(f"<li>{e(q)}</li>" for q in memo.get("questions", []))
        memo_block = f"""<div class="memo"><h4>Memo</h4><p>{e(memo.get('summary', ''))}</p>
  {f'<h5>Risks</h5><ul>{risks}</ul>' if risks else ''}
  {f'<h5>Questions for the founders</h5><ul>{questions}</ul>' if questions else ''}</div>"""

    headline = (
        f'<strong>{e(part["name"])}</strong> scored {part["total"]} of 25 '
        f'({e((score or {}).get("decision", "no decision"))}), '
        f'{part["reliability"]} reliability.'
        if part["total"] is not None
        else f'<strong>{e(part["name"])}</strong> has no score — nothing was produced for it.'
    )

    return f"""<p class="answer__lede">{headline}</p>
{_axis_bars(score) if score else ''}
<div class="branchrow"><h4>Enrichment branches</h4><div class="pills">{branches}</div></div>
{missing_block}
{memo_block}
<div class="sources"><h4>Sources</h4><ul>{''.join(sources) or '<li class="note">none</li>'}</ul></div>"""


def _banners(data: dict) -> str:
    items = []
    if data.get("mode") != "live":
        items.append(
            _banner_card(
                "warning", "Dry mode",
                "No model was called. Collectors, normalization, the graph and the schemas "
                "all ran for real; the model replies were replayed, so scores below are "
                "placeholders rather than judgements.",
            )
        )
    if data.get("thesis_is_placeholder"):
        items.append(
            _banner_card(
                "warning", "Thesis is a placeholder",
                "The thesis in config.py has not been replaced, so the thesis-fit axis is uncalibrated.",
            )
        )
    for source, error in (data.get("failed_sources") or {}).items():
        items.append(_banner_card("critical", f"Source failed: {source}", error))
    return f'<div class="banners">{"".join(items)}</div>' if items else ""


def _banner_card(level: str, title: str, body: str) -> str:
    return f"""<div class="banner banner--{level}">
  {_status(level, title)}
  <p>{e(body)}</p>
</div>"""


# --------------------------------------------------------------------------- page


def _answers(data: dict) -> list[tuple[str, str, str]]:
    """Every answer the page can give, as (key, title, html).

    Rendered in Python so there is one implementation of each view, and so the
    page still carries its content when scripting is off.
    """
    funnel = data.get("funnel", {})
    candidates = data.get("candidates", [])
    cost = data.get("cost", {})
    out: list[tuple[str, str, str]] = []

    top = candidates[0] if candidates else None
    out.append((
        "summary", "Summary",
        f"""<p class="answer__lede">This scan looked for <strong>{e(data.get('query', '—'))}</strong>
        over the last {int(data.get('days', 0))} days. {int(funnel.get('signals', 0))} raw signals came back
        from {int(funnel.get('sources_ok', 0))} sources, resolved into {int(funnel.get('companies', 0))} companies,
        and {int(funnel.get('enriched', 0))} were enriched in full.</p>
        <p>{'The highest score was ' + e(_candidate_parts(top)['name']) + ' at ' + str(_candidate_parts(top)['total']) + ' of 25.' if top and _candidate_parts(top)['total'] is not None else 'No candidate produced a score.'}
        The score only sorts the list — what decides anything is the evidence underneath it.</p>
        <div class="briefs">{''.join(_candidate_brief(c, i) for i, c in enumerate(candidates))}</div>""",
    ))

    out.append(("funnel", "The funnel", _funnel(funnel)))
    out.append(("cost", "Cost", _cost(data.get("stage_costs", []), cost)))

    out.append((
        "candidates", "Candidates",
        f"""<p class="answer__lede">{len(candidates)} candidate{'' if len(candidates) == 1 else 's'} came through
        enrichment, ranked by score. Ask about any one by name.</p>
        <div class="briefs">{''.join(_candidate_brief(c, i) for i, c in enumerate(candidates))}</div>"""
        if candidates else "<p>No candidate reached enrichment in this run.</p>",
    ))

    gaps = []
    for candidate in candidates:
        part = _candidate_parts(candidate)
        if part["missing"]:
            items = "".join(f"<li>{e(m)}</li>" for m in part["missing"])
            gaps.append(f'<div class="gap"><h4>{e(part["name"])}</h4><ul>{items}</ul></div>')
    failed_branches = [
        (
            _candidate_parts(c)["name"],
            [b.get("branch") for b in c.get("branches", []) if not b.get("succeeded")],
        )
        for c in candidates
    ]
    failed_branches = [(n, bs) for n, bs in failed_branches if bs]
    branch_note = (
        "".join(
            f'<li>{e(name)}: {", ".join(e(b) for b in branches)} produced no result</li>'
            for name, branches in failed_branches
        )
        or "<li>Every enrichment branch reported.</li>"
    )
    out.append((
        "missing", "What is missing",
        f"""<p class="answer__lede">A low score and an absence of information are different things, so the
        system is required to say which one it is. Here is what it could not establish.</p>
        {''.join(gaps) or '<p>No candidate recorded missing data.</p>'}
        <div class="gap"><h4>Enrichment branches</h4><ul>{branch_note}</ul></div>
        <div class="gap"><h4>Unattributed signals</h4><ul><li>{int(funnel.get('unattached_signals', 0))} signals
        named no resolvable company and were left unattached rather than guessed at.</li></ul></div>""",
    ))

    counts: dict[str, int] = {}
    for candidate in candidates:
        for signal in candidate.get("company", {}).get("signals", []):
            name = signal.get("source", {}).get("name", "?")
            counts[name] = counts.get(name, 0) + 1
    rows = "".join(
        f'<tr><th scope="row">{e(k)}</th><td>{v}</td></tr>' for k, v in sorted(counts.items())
    )
    failed = data.get("failed_sources") or {}
    failed_rows = "".join(
        f'<li>{_status("critical", e(k))} {e(v)}</li>' for k, v in failed.items()
    )
    team_rows = []
    for candidate in candidates:
        part = _candidate_parts(candidate)
        founders = part["company"].get("founders") or []
        branch = next(
            (b for b in candidate.get("branches", []) if b.get("branch") == "team"), None
        )
        if founders:
            body = ", ".join(e(f) for f in founders)
        elif branch and branch.get("succeeded"):
            body = e((branch.get("text") or "")[:220]) or "the branch reported nothing usable"
        elif branch:
            body = f'the team branch produced no result — {e(branch.get("error") or "unknown")}'
        else:
            body = "no team branch ran"
        team_rows.append(f'<div class="gap"><h4>{e(part["name"])}</h4><p>{body}</p></div>')

    out.append((
        "team", "Founders and teams",
        f"""<p class="answer__lede">Founder evidence comes only from what people published themselves —
        GitHub profiles, papers, talks. Scraping LinkedIn would breach its terms, so the system does not,
        and where the public trace is thin the answer is thin.</p>
        {''.join(team_rows) or '<p>No candidate reached the team analyst.</p>'}""",
    ))

    out.append((
        "sources", "Sources",
        f"""<p class="answer__lede">Discovery runs on three keyless sources: Hacker News, SEC Form D and
        GitHub. A Form D appears within 15 days of a raise, which is usually before the press release.</p>
        <div class="tablewrap"><table><thead><tr><th scope="col">Source</th>
        <th scope="col">Signals on these candidates</th></tr></thead><tbody>{rows or '<tr><td colspan="2">none</td></tr>'}</tbody></table></div>
        {f'<h4>Sources that failed this run</h4><ul class="plain">{failed_rows}</ul>' if failed_rows else '<p class="note">No source failed in this run.</p>'}""",
    ))

    rejections = funnel.get("rejections") or []
    rejection_rows = "".join(
        f'<tr><th scope="row">{e(r.get("name"))}</th><td>{e(r.get("reason"))}</td></tr>'
        for r in rejections
    )
    out.append((
        "rejected", "Rejected at triage",
        f"""<p class="answer__lede">Triage rejected {int(funnel.get('triage_rejected', 0))} of
        {int(funnel.get('companies', 0))}. Rejection needs evidence of contradiction — not knowing something
        is never grounds, because a missed company is invisible and expensive while a wasted review is cheap.</p>
        {f'<div class="tablewrap"><table><thead><tr><th scope="col">Company</th><th scope="col">Reason</th></tr></thead><tbody>{rejection_rows}</tbody></table></div>' if rejection_rows else '<p>Nothing was rejected in this run.</p>'}""",
    ))

    mode_note = (
        """<p class="answer__lede">This run was in <strong>dry mode</strong>: no model was called.
        The collectors, entity resolution, the enrichment graph, the schemas and the audit log all ran for
        real against live APIs — only the model replies were replayed. So the counts and the sources are
        real; the scores and any memo text are placeholders.</p>"""
        if data.get("mode") != "live"
        else """<p class="answer__lede">This run called a real model at every tier.</p>"""
    )
    thesis_note = (
        """<p>The thesis in <code>config.py</code> is still the placeholder, so the thesis-fit axis is
        uncalibrated — it is scoring against a guess at what you are looking for, not your actual mandate.</p>"""
        if data.get("thesis_is_placeholder")
        else ""
    )
    out.append(("mode", "How much of this is real", mode_note + thesis_note))

    out.append((
        "method", "How the scan works",
        """<p class="answer__lede">Five layers, each more expensive than the one above it. The whole design
        follows from that: spend little on many, much on few.</p>
        <ol class="method">
          <li><strong>Collect</strong> — Hacker News, SEC Form D, GitHub. No model, deterministic, testable offline.</li>
          <li><strong>Resolve</strong> — signals attach to companies by domain, then GitHub org, then a
          high-threshold name match. When it is unsure it does not merge: a wrong merge contaminates the
          evidence irreversibly.</li>
          <li><strong>Triage</strong> — rules first, then a cheap model for whatever the rules cannot settle.</li>
          <li><strong>Enrich</strong> — three analysts run in parallel: technical, market, team. Their results
          are counted, not assumed; a branch that reports nothing becomes a stated gap.</li>
          <li><strong>Memo</strong> — the strongest tier, only for candidates past the threshold.</li>
        </ol>""",
    ))

    out.append((
        "help", "What you can ask",
        """<p class="answer__lede">I answer from this scan's data only — I am not a language model, and
        nothing here calls one. If I do not have something, I will say so rather than fill the gap.</p>
        <p>Try: the funnel, the cost, the candidates, what is missing, the sources, what was rejected,
        how the scan works, or any company by name.</p>""",
    ))

    return out


def render(data: dict) -> str:
    candidates = data.get("candidates", [])
    funnel = data.get("funnel", {})
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    answers = _answers(data)

    templates = "".join(
        f'<template data-answer="{key}" data-title="{e(title)}">{body}</template>'
        for key, title, body in answers
    )
    templates += "".join(
        f'<template data-answer="company:{i}" data-title="{e(_candidate_parts(c)["name"])}">'
        f"{_candidate_full(c)}</template>"
        for i, c in enumerate(candidates)
    )

    names = json.dumps([_candidate_parts(c)["name"] for c in candidates])

    chips = [
        ("The funnel", "funnel"),
        ("What is missing", "missing"),
        ("Cost", "cost"),
        ("Sources", "sources"),
        ("How it works", "method"),
    ]
    chip_html = "".join(
        f'<button class="chip" type="button" data-ask="{e(label)}">{e(label)}</button>'
        for label, _key in chips
    )

    opening = next(body for key, _t, body in answers if key == "summary")

    return f"""<title>Deal Flow Scan</title>
<style>{STYLE}</style>
<div class="app">
<header class="chrome">
  <div class="chrome__inner">
    <div class="chrome__title">
      <span class="chrome__eyebrow">Deal flow</span>
      <strong>{e(data.get('query', '—'))}</strong>
    </div>
    <div class="chrome__meta">last {int(data.get('days', 0))} days<br>{generated}</div>
  </div>
  <div class="chrome__foot">Static export · no backend</div>
</header>

<main class="thread" id="thread" role="log" aria-live="polite" aria-label="Conversation">
  <div class="turn turn--bot">
    <div class="bubble bubble--bot">
      <div class="bubble__head">Scan report</div>
      {_banners(data)}
      {opening}
    </div>
  </div>
</main>

<div class="composer">
  <div class="chips" id="chips">{chip_html}</div>
  <form class="ask" id="ask" autocomplete="off">
    <input id="q" name="q" type="text" placeholder="Ask about this scan…"
           aria-label="Ask about this scan" />
    <button type="submit" aria-label="Send">Ask</button>
  </form>
  <p class="disclaimer">Answers come from this scan's data. No model is called.</p>
</div>
</div>

<noscript>
  <div class="app"><main class="thread">
    <div class="turn turn--bot"><div class="bubble bubble--bot">
      <div class="bubble__head">Without scripting</div>
      <p>The conversation needs JavaScript. Everything it would say is below.</p>
    </div></div>
    {''.join(f'<div class="turn turn--bot"><div class="bubble bubble--bot"><div class="bubble__head">{e(t)}</div>{b}</div></div>' for _k, t, b in answers)}
  </main></div>
</noscript>

{templates}
<div class="tooltip" role="status" aria-live="polite"></div>
<script>window.CANDIDATES = {names};</script>
<script>{SCRIPT}</script>"""


STYLE = """
/* Beige. The neutrals are warm on purpose — paper rather than screen — and the
   data hue stays the validated blue, so the chart layer keeps a palette that was
   measured while the product skin changes around it. A cool mark on a warm ground
   is the deliberate pairing here, not an accident of two systems meeting.

   Every value below was measured against its own surface (WCAG), not eyeballed:
     ink 14.65 · ink-2 7.06 · muted 3.48 · bar 3.96 · critical 4.31 · good 3.01
   Two notes on the exceptions:
     - status warning is 1.64 on the light ground. That is the documented
       sub-3:1 case, and the mitigation is already in place: every status ships
       with an icon AND a word, so colour never carries meaning alone.
     - the funnel's palest step had to move one stop darker (250 -> 300) because
       beige is darker than white and 250 fell to 1.89, under the 2:1 floor an
       ordinal ramp owes its surface. 300 measures 2.24. */
:root {
  color-scheme: light;
  --plane: #ede6d9;
  --surface: #f7f2e8;
  --surface-raised: #fffcf5;
  --ink: #241f19;
  --ink-2: #5a5044;
  --ink-muted: #8a8071;
  --hairline: rgba(36,31,25,0.13);
  --grid: #dfd6c4;
  --bar: #2a78d6;
  --bar-soft: rgba(42,120,214,0.13);
  --good: #0ca30c;
  --warning: #fab219;
  --critical: #d03b3b;
  /* One step darker than the bar: white text on it measures 5.39 rather than 4.42. */
  --user-bg: #256abf;
  --user-ink: #ffffff;
  --chrome-bg: rgba(247,242,232,0.72);
  --chrome-edge: rgba(255,253,247,0.72);
  --shadow: 0 1px 2px rgba(74,60,40,0.06), 0 8px 24px rgba(74,60,40,0.07);
  --funnel-0: #6da7ec; --funnel-1: #3987e5; --funnel-2: #2a78d6;
  --funnel-3: #1c5cab; --funnel-4: #104281;
  /* Critically damped: response ~0.35s, damping 1.0 — settles, never overshoots. */
  --spring: cubic-bezier(0.32, 0.72, 0, 1);
  --response: 350ms;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    /* Warm dark: the same paper, seen at night. Not an inversion — the neutrals
       keep their warmth and every value was re-measured on this ground. */
    --plane: #14110d;
    --surface: #1e1a15;
    --surface-raised: #262119;
    --ink: #f5efe4;
    --ink-2: #c9bfae;
    --ink-muted: #8f8677;
    --hairline: rgba(245,239,228,0.12);
    --grid: #332c22;
    --bar: #5a9bea;
    --bar-soft: rgba(90,155,234,0.18);
    --critical: #e05a5a;
    /* White on this blue is 2.87 — fails. Dark ink on it is 6.55. */
    --user-bg: #5a9bea;
    --user-ink: #14110d;
    --chrome-bg: rgba(30,26,21,0.72);
    --chrome-edge: rgba(245,239,228,0.10);
    --shadow: 0 1px 2px rgba(0,0,0,0.45), 0 12px 32px rgba(0,0,0,0.4);
    /* Reversed against light: intensity must increase down the funnel in both
       themes, and on a dark surface intensity means lighter. The dim end stays
       at step 600, the darkest the ordinal rule allows against this surface. */
    --funnel-0: #184f95; --funnel-1: #256abf; --funnel-2: #3987e5;
    --funnel-3: #6da7ec; --funnel-4: #b7d3f6;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --plane: #14110d; --surface: #1e1a15; --surface-raised: #262119;
  --ink: #f5efe4; --ink-2: #c9bfae; --ink-muted: #8f8677;
  --hairline: rgba(245,239,228,0.12); --grid: #332c22;
  --bar: #5a9bea; --bar-soft: rgba(90,155,234,0.18);
  --critical: #e05a5a;
  --user-bg: #5a9bea; --user-ink: #14110d;
  --chrome-bg: rgba(30,26,21,0.72); --chrome-edge: rgba(245,239,228,0.10);
  --shadow: 0 1px 2px rgba(0,0,0,0.45), 0 12px 32px rgba(0,0,0,0.4);
  --funnel-0: #184f95; --funnel-1: #256abf; --funnel-2: #3987e5;
  --funnel-3: #6da7ec; --funnel-4: #b7d3f6;
}

* { box-sizing: border-box; }
body {
  margin: 0; background: var(--plane); color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-optical-sizing: auto; line-height: 1.5; -webkit-font-smoothing: antialiased;
}
/* Two columns: a fixed rail on the left, the conversation on the right. The rail
   is where the session lives (which scan, which mode, what to start next); the
   right column is the work. Putting them side by side means the controls stay
   in view while the thread scrolls, which a top bar could not do without
   covering the newest message. */
.app {
  min-height: 100vh;
  display: grid;
  grid-template-columns: 15.5rem 1fr;
  grid-template-rows: 1fr auto;
  transition: grid-template-columns 180ms ease;
}

/* The rail is 15.5rem, which is right for stats and wrong for code: at that
   width a line of Python breaks around thirty characters and stops being
   readable at all. So the rail widens while the terminal has something to show
   and goes back when it does not — the terminal keeps the place it was asked
   for without the rest of the session paying for it. */
body.term-open .app { grid-template-columns: 28rem 1fr; }

.chrome {
  grid-column: 1; grid-row: 1 / -1;
  position: sticky; top: 0; align-self: start; height: 100vh;
  display: flex; flex-direction: column; gap: 1.25rem;
  padding: 1.25rem 1rem;
  background: var(--chrome-bg);
  backdrop-filter: blur(20px) saturate(180%);
  -webkit-backdrop-filter: blur(20px) saturate(180%);
  border-right: 1px solid var(--hairline);
  border-top: 0;
}
.chrome__inner {
  display: flex; flex-direction: column; gap: 0.875rem;
  max-width: none; margin: 0; padding: 0;
}
.chrome__eyebrow {
  display: block; font-size: 0.6875rem; letter-spacing: 0.07em;
  text-transform: uppercase; color: var(--ink-muted);
}
.chrome__title strong {
  display: block; font-size: 1.0625rem; letter-spacing: -0.01em; line-height: 1.25;
  margin-top: 0.125rem;
}
.chrome__meta { font-size: 0.75rem; color: var(--ink-2); }
/* Sunum destesi. Sohbetin YANINDA, yerine değil.
   İlk hâlinde sohbetin yerine geçiyordu ve soru sorulunca kayboluyordu — oysa
   istenen şey ikisinin aynı anda durması: bir yanda slayt, bir yanda koşan
   sistem. Kendi sütununda ve kendi genişliğinde. */
.deck {
  grid-column: 3; grid-row: 1 / -1;
  display: flex; flex-direction: column; min-height: 0; min-width: 0;
  border-left: 1px solid var(--hairline);
}
/* Üçüncü sütun yalnız deste açıkken var. Sabit genişlik: slayt okunabilir
   kalmalı ama sohbeti de ezmemeli. */
/* Deste sütunu geniş: tarama raporu açılışta basılmayı bıraktığından orta
   sütunda yalnız sohbet baloncukları kaldı, ve slayt o yeri hak ediyor. */
body.deck-open .app { grid-template-columns: 15.5rem minmax(24rem, 1fr) 32rem; }
body.deck-open.term-open .app { grid-template-columns: 28rem minmax(22rem, 1fr) 30rem; }
.deck[hidden] { display: none; }
.deck__bar {
  display: flex; gap: 0.3rem; padding: 0.5rem 0.7rem 0.4rem;
  border-bottom: 1px solid var(--hairline); flex-wrap: wrap;
}
.deck__tab {
  font: inherit; font-size: 0.75rem; padding: 0.22rem 0.6rem; cursor: pointer;
  border: 1px solid var(--hairline); border-radius: 999px;
  background: transparent; color: var(--ink-2);
}
.deck__tab:hover { border-color: var(--ink-2); color: var(--ink); }
.deck__tab.is-on { border-color: var(--ink); color: var(--ink); font-weight: 600; }
.deck__tab--chat { margin-left: auto; }
/* Dar ekranda üçüncü sütun yok: slayt ile sohbeti 900 px'e sıkıştırmak
   ikisini birden okunmaz yapıyor. */
@media (max-width: 78rem) {
  body.deck-open .app, body.deck-open.term-open .app {
    grid-template-columns: 15.5rem 1fr; }
  .deck { grid-column: 2; grid-row: 1; border-left: 0; }
}
/* Sahne: iframe ile tıklama katmanı üst üste. */
.deck__stage { position: relative; flex: 1; min-height: 0; }
.deck__frame { position: absolute; inset: 0; width: 100%; height: 100%;
               border: 0; background: var(--plane); }
/* Tıklama bölgeleri. Şeffaf ve kenarlıksız: görünen şey slayt olmalı, üstündeki
   düzenek değil. Sol üçte bir geri, sağı ileri — sunumda parmak sağa gidiyor. */
.deck__zones { position: absolute; inset: 0; display: flex; }
.deck__zones[hidden] { display: none; }
.deck__zone { border: 0; background: transparent; padding: 0; cursor: pointer; }
.deck__zone--prev { flex: 0 0 32%; cursor: w-resize; }
.deck__zone--next { flex: 1; cursor: e-resize; }
/* Üstüne gelince hangi yarıda olduğun belli olsun — görünmez bir düğmeye
   basmak, basılıp basılmadığını bilmemek demek. */
.deck__zone:hover { background: rgba(0, 0, 0, 0.035); }
.deck__page {
  position: absolute; right: 0.6rem; bottom: 0.6rem; z-index: 2;
  padding: 0.1rem 0.45rem; border-radius: 999px; pointer-events: none;
  background: var(--surface, var(--plane)); border: 1px solid var(--hairline);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.68rem; color: var(--ink-2); font-variant-numeric: tabular-nums;
}
.deck__page[hidden] { display: none; }
/* Çerçeve anahtarı. Sağ üstte ve her zaman görünür: bir tur ayarı değil,
   bütün sistemin hangi çerçevede koştuğu. Rengi de o yüzden farklı — AutoGen
   mavi, MAF mor; ekranın başka hiçbir yerinde bu iki renk yan yana durmuyor. */
.fw {
  /* Sağ ÜSTTE, sol rayda değil. `.chrome` sol sütun; düğme oraya konunca
     "sistemin çerçevesi" gibi değil "rayın bir ayarı" gibi okunuyordu. */
  position: fixed; top: 0.55rem; right: 0.9rem; z-index: 30;
  display: inline-flex; align-items: center; gap: 0.35rem;
  font: inherit; font-size: 0.72rem; font-weight: 600;
  padding: 0.2rem 0.55rem; border-radius: 999px; cursor: pointer;
  border: 1px solid var(--hairline); background: transparent; color: var(--ink-2);
}
.fw[hidden] { display: none; }
.fw:hover { border-color: var(--ink-2); color: var(--ink); }
.fw { background: var(--surface, var(--plane)); }
.fw__dot { width: 0.4rem; height: 0.4rem; border-radius: 50%; background: #1971c2; }
.fw.is-maf { border-color: #5f3dc4; color: #5f3dc4; }
.fw.is-maf .fw__dot { background: #5f3dc4; }
.chrome__foot { margin-top: auto; font-size: 0.6875rem; color: var(--ink-muted); }

/* ------------------------------------------------------------- terminal */
/* `margin-top: auto` moves from the foot to the terminal while it is open, so
   the console sits at the bottom of the rail and the theme toggle stays under
   it rather than being pushed off. */
.term {
  margin-top: auto; display: flex; flex-direction: column; min-height: 0;
  border: 1px solid var(--hairline); border-radius: 4px;
  background: var(--surface); overflow: hidden;
}
/* `display: flex` above beats the browser's own `[hidden] { display: none }`,
   so without this line the panel is on screen from the first paint — empty, and
   with the rail still at its narrow width because `term-open` was never added.
   Measured: computed display stayed `flex` while `.hidden` read `true`. */
.term[hidden] { display: none; }
body.term-open .chrome__foot { margin-top: 0; }
.term__head {
  display: flex; align-items: center; gap: 0.5rem;
  padding: 0.3rem 0.5rem; border-bottom: 1px solid var(--hairline);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.66rem;
}
.term__title { font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; }
.term__meta { color: var(--ink-muted); margin-left: auto; }
.term__close {
  border: 0; background: none; color: var(--ink-muted); cursor: pointer;
  font-size: 0.9rem; line-height: 1; padding: 0 0.15rem;
}
.term__close:hover { color: var(--ink); }
.term__body {
  margin: 0; padding: 0.45rem 0.55rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.66rem; line-height: 1.5;
  white-space: pre-wrap; word-break: break-word;
  max-height: 22rem; overflow: auto; overscroll-behavior: contain;
  color: var(--ink-2);
}
.term__body .t-cmd { color: var(--ink); font-weight: 600; }
.term__body .t-err { color: #c92a2a; }
.term__body .t-dim { color: var(--ink-muted); }

/* ---------------------------------------------------------------- thread */
/* `display: flex` HTML'in `hidden` niteliğini eziyor — terminalde de aynı
   hata olmuştu. Deste açıkken sohbet gerçekten gizlenmezse rapor kartı slaytın
   üstünde duruyor. */
.thread[hidden] { display: none; }
.thread {
  grid-column: 2; grid-row: 1;
  width: 100%; max-width: 52rem; margin: 0 auto;
  padding: 1.75rem 1.5rem 1rem; display: flex; flex-direction: column; gap: 0.875rem;
}
.turn { display: flex; }
.turn--bot { justify-content: flex-start; }
.turn--user { justify-content: flex-end; }
.bubble {
  max-width: 100%; border-radius: 18px; padding: 1.125rem 1.25rem;
  animation: rise var(--response) var(--spring) both;
}
.bubble--bot {
  background: var(--surface); border: 1px solid var(--hairline);
  box-shadow: var(--shadow); width: 100%;
}
.bubble--user {
  background: var(--user-bg); color: var(--user-ink);
  border-radius: 18px 18px 4px 18px; max-width: 32rem;
  padding: 0.625rem 0.9375rem; font-size: 0.9375rem;
}
.bubble__head {
  font-size: 0.6875rem; letter-spacing: 0.07em; text-transform: uppercase;
  color: var(--ink-muted); margin-bottom: 0.75rem; font-weight: 600;
}
@keyframes rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }

.answer__lede { margin: 0 0 1rem; font-size: 1rem; color: var(--ink); letter-spacing: 0; }
.bubble p { margin: 0 0 0.75rem; color: var(--ink-2); font-size: 0.9375rem; }
.bubble p:last-child { margin-bottom: 0; }
h2 { margin: 0 0 0.35rem; font-size: 1.125rem; letter-spacing: -0.015em; line-height: 1.2; }
h4 {
  margin: 0 0 0.5rem; font-size: 0.75rem; letter-spacing: 0.05em;
  text-transform: uppercase; color: var(--ink-muted); font-weight: 600;
}
h5 { margin: 0.9rem 0 0.35rem; font-size: 0.875rem; letter-spacing: 0; }
.sub { margin: 0 0 1.125rem; color: var(--ink-2); font-size: 0.9375rem; }
.note { color: var(--ink-muted); font-size: 0.875rem; }
.card { padding: 0; margin: 0; background: none; border: 0; box-shadow: none; }
ol.method { margin: 0; padding-left: 1.25rem; }
ol.method li { font-size: 0.9375rem; color: var(--ink-2); margin-bottom: 0.5rem; }
ul.plain { list-style: none; padding: 0; margin: 0; }
ul.plain li { display: flex; gap: 0.5rem; align-items: baseline; font-size: 0.875rem; margin-bottom: 0.375rem; }
.gap { margin-bottom: 1rem; }
.gap ul, .missing ul, .sources ul, .memo ul { margin: 0; padding-left: 1.125rem; }
.gap li, .missing li, .sources li, .memo li {
  font-size: 0.875rem; color: var(--ink-2); margin-bottom: 0.25rem;
}
.missing, .memo, .branchrow { margin-bottom: 1.25rem; }
.memo p { margin: 0; }
a { color: var(--bar); text-underline-offset: 2px; }

/* ---------------------------------------------------------------- pills */
.pill {
  display: inline-flex; align-items: center; gap: 0.375rem;
  padding: 0.1875rem 0.5rem; border-radius: 999px;
  font-size: 0.75rem; font-weight: 560;
  color: var(--ink-2); background: color-mix(in srgb, var(--ink) 6%, transparent);
  border: 1px solid var(--hairline); white-space: nowrap;
}
.pill__icon { font-size: 0.625rem; line-height: 1; }
.pill--good .pill__icon { color: var(--good); }
.pill--warning .pill__icon { color: var(--warning); }
.pill--critical .pill__icon { color: var(--critical); }
.pills { display: flex; flex-wrap: wrap; gap: 0.375rem; }

.banners { display: grid; gap: 0.5rem; margin-bottom: 1.125rem; }
.banner {
  display: flex; gap: 0.75rem; align-items: flex-start;
  padding: 0.75rem 0.875rem; border: 1px solid var(--hairline);
  border-radius: 12px; background: var(--plane);
}
.banner p { margin: 0; font-size: 0.8125rem; }
.banner .pill { flex: none; }

/* ---------------------------------------------------------------- charts */
.funnel, .cost { display: grid; gap: 0.5rem; }
.funnel__row, .cost__row {
  display: grid; grid-template-columns: minmax(8rem, 13rem) 1fr auto;
  gap: 0.875rem; align-items: center;
}
.funnel__label, .cost__label { font-size: 0.875rem; line-height: 1.3; }
.funnel__note { display: block; color: var(--ink-muted); font-size: 0.75rem; }
.funnel__track, .cost__track, .axis__track {
  height: 10px; border-radius: 5px; background: var(--bar-soft); overflow: hidden;
}
.funnel__bar, .cost__bar, .axis__bar {
  height: 100%; border-radius: 5px; transform-origin: left center;
  animation: grow var(--response) var(--spring) both;
}
.funnel__bar { background: var(--funnel-0); }
.funnel__row:nth-child(2) .funnel__bar { background: var(--funnel-1); }
.funnel__row:nth-child(3) .funnel__bar { background: var(--funnel-2); }
.funnel__row:nth-child(4) .funnel__bar { background: var(--funnel-3); }
.funnel__row:nth-child(5) .funnel__bar { background: var(--funnel-4); }
.cost__bar, .axis__bar { background: var(--bar); }
.funnel__value, .cost__value, .axis__value {
  font-size: 0.9375rem; font-variant-numeric: tabular-nums; min-width: 3.25rem; text-align: right;
}
@keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
.axes { display: grid; gap: 0.4375rem; margin-bottom: 1.25rem; }
.axis { display: grid; grid-template-columns: minmax(6.5rem, 9rem) 1fr auto; gap: 0.875rem; align-items: center; }
.axis__label { font-size: 0.875rem; }
.axis__value { min-width: 1.25rem; }

/* ---------------------------------------------------------------- candidate rows */
.briefs { display: grid; gap: 0.5rem; }
.brief {
  display: flex; align-items: stretch; gap: 0.5rem;
  border-radius: 12px; border: 1px solid var(--hairline); background: var(--plane);
  overflow: hidden;
}
.brief__main {
  flex: 1; display: flex; align-items: center; justify-content: space-between;
  gap: 1rem; min-width: 0; text-align: left; font: inherit; color: inherit;
  cursor: pointer; padding: 0.75rem 0.875rem; border: 0; background: none;
  transition: background-color 100ms linear, transform 100ms var(--spring);
}
.brief__main:hover { background: color-mix(in srgb, var(--ink) 4%, transparent); }
/* Feedback on pointer-down, not on release. */
.brief__main:active { transform: scale(0.99); background: color-mix(in srgb, var(--ink) 7%, transparent); }
.brief__main:focus-visible { outline: 2px solid var(--bar); outline-offset: -2px; }
.brief__live {
  flex: none; align-self: center; margin-right: 0.5rem;
  font: inherit; font-size: 0.75rem; font-weight: 560; color: var(--ink-2);
  background: var(--surface); border: 1px solid var(--hairline);
  border-radius: 999px; padding: 0.3125rem 0.6875rem; cursor: pointer; white-space: nowrap;
  transition: background-color 100ms linear, transform 100ms var(--spring), color 100ms linear;
}
.brief__live:hover { color: var(--ink); }
.brief__live:active { transform: scale(0.97); }
.brief__live:focus-visible { outline: 2px solid var(--bar); outline-offset: 2px; }
.brief__id { display: flex; flex-direction: column; gap: 0.125rem; min-width: 0; }
.brief__name { font-weight: 580; letter-spacing: -0.01em; }
.brief__meta { font-size: 0.75rem; color: var(--ink-muted); }
.brief__right { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; justify-content: flex-end; }
.brief__score { font-size: 1.125rem; font-weight: 620; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }
.score__of { font-size: 0.75rem; color: var(--ink-muted); font-weight: 400; }

/* ---------------------------------------------------------------- tables */
table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
th, td { text-align: left; padding: 0.4375rem 0.75rem 0.4375rem 0; border-bottom: 1px solid var(--grid); }
thead th { color: var(--ink-muted); font-weight: 600; font-size: 0.8125rem; }
tbody th { font-weight: 500; }
td { font-variant-numeric: tabular-nums; color: var(--ink-2); }
.tablewrap { overflow-x: auto; }
.table-view { margin-top: 1rem; }
.table-view summary { cursor: pointer; font-size: 0.8125rem; color: var(--ink-muted); padding: 0.25rem 0; }
.table-view summary:hover { color: var(--ink-2); }
.visually-hidden {
  position: absolute; width: 1px; height: 1px; overflow: hidden;
  clip: rect(0 0 0 0); white-space: nowrap;
}
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.8125em; }

/* ---------------------------------------------------------------- composer */
.composer {
  grid-column: 2; grid-row: 2;
  position: sticky; bottom: 0; z-index: 20;
  background: var(--chrome-bg);
  backdrop-filter: blur(20px) saturate(180%);
  -webkit-backdrop-filter: blur(20px) saturate(180%);
  border-top: 1px solid var(--hairline);
  padding: 0.75rem 1.25rem 0.875rem;
}
.chips {
  max-width: 52rem; margin: 0 auto 0.625rem; display: flex; gap: 0.375rem;
  overflow-x: auto; padding-bottom: 0.125rem; scrollbar-width: none;
}
.chips::-webkit-scrollbar { display: none; }
.chip {
  flex: none; font: inherit; font-size: 0.8125rem; color: var(--ink-2);
  background: var(--surface); border: 1px solid var(--hairline);
  border-radius: 999px; padding: 0.3125rem 0.75rem; cursor: pointer;
  transition: background-color 100ms linear, transform 100ms var(--spring);
}
.chip:hover { color: var(--ink); }
.chip:active { transform: scale(0.97); background: color-mix(in srgb, var(--ink) 7%, var(--surface)); }
.chip:focus-visible { outline: 2px solid var(--bar); outline-offset: 2px; }
.ask { max-width: 52rem; margin: 0 auto; display: flex; gap: 0.5rem; }
.ask input {
  flex: 1; font: inherit; font-size: 0.9375rem; color: var(--ink);
  background: var(--surface); border: 1px solid var(--hairline);
  border-radius: 999px; padding: 0.625rem 1rem; min-width: 0;
}
.ask input::placeholder { color: var(--ink-muted); }
.ask input:focus { outline: 2px solid var(--bar); outline-offset: -1px; }
.ask button {
  font: inherit; font-size: 0.9375rem; font-weight: 560; color: var(--user-ink);
  background: var(--user-bg); border: 0; border-radius: 999px;
  padding: 0.625rem 1.125rem; cursor: pointer;
  transition: transform 100ms var(--spring), filter 100ms linear;
}
.ask button:active { transform: scale(0.97); filter: brightness(0.94); }
.ask button:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
.disclaimer {
  max-width: 52rem; margin: 0.5rem auto 0; font-size: 0.75rem; color: var(--ink-muted); text-align: center;
}

/* ---------------------------------------------------------------- tooltip */
.tooltip {
  position: fixed; z-index: 40; pointer-events: none; max-width: 22rem;
  padding: 0.4375rem 0.625rem; border-radius: 8px; border: 1px solid var(--hairline);
  background: var(--surface-raised); color: var(--ink); box-shadow: var(--shadow);
  font-size: 0.8125rem; line-height: 1.35; opacity: 0; transform: translateY(2px);
  transition: opacity 140ms linear, transform 140ms var(--spring);
}
.tooltip[data-show="true"] { opacity: 1; transform: translateY(0); }

@media (max-width: 56rem) {
  /* Below this the rail costs more room than it earns, so it folds back to a bar. */
  .app { grid-template-columns: 1fr; grid-template-rows: auto 1fr auto; }
  /* `body.term-open .app` outranks the line above on specificity, so without
     this the terminal would force a 28rem column onto a phone. Repeat it here
     rather than weakening the desktop rule. */
  body.term-open .app { grid-template-columns: 1fr; }
  /* The rail is a horizontal bar down here; the terminal drops below it as a
     full-width block instead of trying to sit inside a row. */
  .term { flex-basis: 100%; margin-top: 0.5rem; }
  .term__body { max-height: 12rem; }
  .chrome {
    grid-column: 1; grid-row: 1;
    position: sticky; top: 0; height: auto;
    flex-direction: row; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 0.75rem; padding: 0.75rem 1.25rem;
    border-right: 0; border-bottom: 1px solid var(--hairline);
  }
  .chrome__inner { flex-direction: row; align-items: baseline; gap: 0.75rem; }
  .chrome__title strong { display: inline; }
  .chrome__foot { display: none; }
  .thread { grid-column: 1; grid-row: 2; }
  .composer { grid-column: 1; grid-row: 3; }
}

@media (max-width: 40rem) {
  .funnel__row, .cost__row, .axis { grid-template-columns: 1fr; gap: 0.25rem; }
  .funnel__value, .cost__value, .axis__value { text-align: left; }
  .brief { flex-direction: column; align-items: stretch; }
  .brief__main { flex-direction: column; align-items: flex-start; gap: 0.5rem; }
  .brief__right { justify-content: flex-start; }
  .brief__live { align-self: flex-start; margin: 0 0 0.75rem 0.875rem; }
}

@media (prefers-reduced-transparency: reduce) {
  .chrome, .composer { background: var(--surface); backdrop-filter: none; -webkit-backdrop-filter: none; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 1ms !important;
    transition-duration: 140ms !important;
    transition-property: opacity, background-color, color, border-color, filter !important;
    transform: none !important;
  }
}
"""

SCRIPT = """
(function () {
  var thread = document.getElementById('thread');
  var chrome = document.querySelector('.chrome');
  var form = document.getElementById('ask');
  var input = document.getElementById('q');
  var tip = document.querySelector('.tooltip');

  var onScroll = function () {
    chrome.setAttribute('data-scrolled', window.scrollY > 4 ? 'true' : 'false');
  };
  onScroll();
  window.addEventListener('scroll', onScroll, { passive: true });

  // ---- routing -----------------------------------------------------------
  // Deterministic: a question is matched against keyword sets, and the answer
  // is a block this page already carries. Nothing is generated at read time,
  // so the page can never say something the scan did not establish.
  var INTENTS = [
    ['summary',    ['summary', 'overview', 'recap', 'tldr', 'what happened', 'result']],
    ['funnel',     ['funnel', 'stage', 'how many', 'pipeline', 'counts', 'narrow']],
    ['cost',       ['cost', 'token', 'spend', 'price', 'expensive', 'cheap', 'budget', 'call']],
    ['candidates', ['candidate', 'compan', 'startup', 'list', 'top', 'best', 'rank', 'score']],
    ['missing',    ['missing', 'gap', 'unknown', 'reliab', 'confiden', 'weak', 'trust', 'blind']],
    ['team',       ['founder', 'team', 'who built', 'who is behind', 'people', 'ceo', 'cto']],\n    ['sources',    ['source', 'where', 'data from', 'sec', 'form d', 'github', 'hacker', 'hn', 'arxiv']],
    ['rejected',   ['reject', 'drop', 'filter', 'triage', 'exclud', 'skip']],
    ['mode',       ['dry', 'real', 'live', 'placeholder', 'thesis', 'fake', 'model called']],
    ['method',     ['how', 'method', 'work', 'architect', 'design', 'explain the', 'process']],
    ['help',       ['help', 'what can', 'commands', 'ask you']]
  ];

  var normalise = function (text) { return text.toLowerCase().trim(); };

  var route = function (question) {
    var q = normalise(question);

    // A company name beats every general intent — it is the most specific thing
    // the reader can be asking for.
    var names = window.CANDIDATES || [];
    for (var i = 0; i < names.length; i++) {
      var name = normalise(names[i]);
      if (name && (q.indexOf(name) !== -1 || name.indexOf(q) === 0)) { return 'company:' + i; }
    }

    var best = null, bestScore = 0;
    for (var j = 0; j < INTENTS.length; j++) {
      var key = INTENTS[j][0], words = INTENTS[j][1], score = 0;
      for (var k = 0; k < words.length; k++) {
        if (q.indexOf(words[k]) !== -1) { score += words[k].length; }
      }
      if (score > bestScore) { bestScore = score; best = key; }
    }
    return best;
  };

  var template = function (key) {
    return document.querySelector('template[data-answer="' + key + '"]');
  };

  var addTurn = function (side, headText, node) {
    var turn = document.createElement('div');
    turn.className = 'turn turn--' + side;
    var bubble = document.createElement('div');
    bubble.className = 'bubble bubble--' + side;
    if (headText) {
      var head = document.createElement('div');
      head.className = 'bubble__head';
      head.textContent = headText;
      bubble.appendChild(head);
    }
    if (typeof node === 'string') {
      var p = document.createElement('p');
      p.textContent = node;
      bubble.appendChild(p);
    } else {
      bubble.appendChild(node);
    }
    turn.appendChild(bubble);
    thread.appendChild(turn);
    bindMarks(bubble);
    turn.scrollIntoView({ behavior: 'smooth', block: 'end' });
  };

  var answer = function (question) {
    var key = route(question);
    if (!key) {
      // Saying "I don't know" is the honest branch, and it says where to look
      // instead of guessing — the page's own version of "a zero result owes an
      // explanation".
      var wrap = document.createElement('div');
      var lead = document.createElement('p');
      lead.className = 'answer__lede';
      lead.textContent = 'This scan does not hold an answer to that. Here is what it does hold.';
      wrap.appendChild(lead);
      wrap.appendChild(template('help').content.cloneNode(true));
      addTurn('bot', 'Not something I hold', wrap);
      return;
    }
    var tpl = template(key);
    addTurn('bot', tpl.getAttribute('data-title'), tpl.content.cloneNode(true));
  };

  var ask = function (question) {
    if (!question) { return; }
    addTurn('user', null, question);
    answer(question);
  };

  form.addEventListener('submit', function (event) {
    event.preventDefault();
    var value = input.value.trim();
    if (!value) { return; }
    input.value = '';
    ask(value);
  });

  // Chips and candidate rows commit on pointer-down: the response is immediate,
  // not held until release.
  var bindMarks = function (root) {
    root.querySelectorAll('[data-ask]').forEach(function (node) {
      node.addEventListener('pointerdown', function (event) {
        event.preventDefault();
        ask(node.getAttribute('data-ask'));
      });
      node.addEventListener('click', function (event) { event.preventDefault(); });
      node.addEventListener('keydown', function (event) {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          ask(node.getAttribute('data-ask'));
        }
      });
    });
    root.querySelectorAll('[data-tip]').forEach(function (mark) {
      mark.addEventListener('pointermove', function (event) {
        tip.textContent = mark.getAttribute('data-tip');
        tip.setAttribute('data-show', 'true');
        var rect = tip.getBoundingClientRect();
        tip.style.left = Math.min(event.clientX + 14, window.innerWidth - rect.width - 8) + 'px';
        tip.style.top = Math.max(event.clientY - rect.height - 12, 8) + 'px';
      });
      mark.addEventListener('pointerleave', function () {
        tip.setAttribute('data-show', 'false');
      });
    });
  };

  bindMarks(document.querySelector('.thread'));
  bindMarks(document.getElementById('chips'));
})();
"""


def latest_scan() -> Path | None:
    files = sorted(config.OUTPUT.glob("scan-*.json"))
    return files[-1] if files else None


def build(json_path: Path, out_path: Path, *, standalone: bool = True) -> Path:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    data["_source_name"] = json_path.name
    body = render(data)
    if standalone:
        body = (
            "<!doctype html>\n<html lang=\"en\" data-theme=\"light\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            + body
            + "\n</head></html>"
        )
    out_path.write_text(body, encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a scan result as an HTML dashboard")
    parser.add_argument("--json", type=Path, default=None, help="scan JSON (default: newest)")
    parser.add_argument("--out", type=Path, default=config.OUTPUT / "dashboard.html")
    parser.add_argument(
        "--fragment",
        action="store_true",
        help="omit the document wrapper (for embedding or publishing)",
    )
    args = parser.parse_args()

    source = args.json or latest_scan()
    if source is None:
        print("No scan JSON found. Run pipeline/scan.py first.")
        raise SystemExit(1)

    out = build(source, args.out, standalone=not args.fragment)
    print(f"  source: {source}")
    print(f"  written: {out}  ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
