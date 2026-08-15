"""CLI entry point — one pass down the funnel.

    python pipeline/scan.py --query "ai infrastructure" --days 7

Stages, in the order the funnel prescribes:

    collect (no LLM) -> normalize (no LLM) -> triage (cheap) ->
    enrich in parallel (mid) -> score (mid) -> memo (strong, few)

Every stage prints what it dropped and why. A run that finds nothing must say
where it looked — "a zero result always owes an explanation".
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config          # noqa: E402
import engine          # noqa: E402
import observability   # noqa: E402
import graph           # noqa: E402
import normalize       # noqa: E402
from agents import memo as memo_module  # noqa: E402
from agents import triage               # noqa: E402
from collectors import ALL as ALL_COLLECTORS  # noqa: E402
from schemas import Candidate, Company, Signal  # noqa: E402


def _banner(text: str) -> None:
    print(f"\n{'━' * 72}\n  {text}\n{'━' * 72}")


def _preflight() -> None:
    """State the operating conditions before doing anything else."""
    if config.THESIS.is_placeholder:
        print(
            "⚠  THESIS IS A PLACEHOLDER — config.py THESIS has not been replaced.\n"
            "   thesis_fit scores are uncalibrated until you write your own thesis."
        )
    if not config.live_llm_available():
        missing = ", ".join(config.missing_llm_settings())
        print(
            f"⚠  DRY MODE — no model will be called. Missing: {missing}\n"
            "   Collectors, normalization, the graph and the schemas all run for real;\n"
            "   only the model replies are replayed, so the output is deterministic."
        )


# --------------------------------------------------------------------------- stages


def collect(query: str, days: int) -> tuple[list[Signal], dict[str, str]]:
    """Layer 1. Returns the signals and, separately, the sources that failed."""
    _banner(f"1 · COLLECT  ·  query={query!r}  window={days}d")
    signals: list[Signal] = []
    failures: dict[str, str] = {}

    for collector_class in ALL_COLLECTORS:
        collector = collector_class()
        result = collector.run(query=query, days=days)
        if result.succeeded:
            signals.extend(result.signals)
            print(
                f"  ✓ {result.source:<10} {len(result.signals):>3} signals "
                f"({result.requests} requests, {result.cache_hits} cached)"
            )
        else:
            failures[result.source] = result.error or "unknown"
            # A source that fell over is not silence: it is carried into the memo.
            print(f"  ✗ {result.source:<10} FAILED — {result.error}")

    print(f"\n  total: {len(signals)} signals from {len(ALL_COLLECTORS) - len(failures)} sources")
    return signals, failures


def resolve(signals: list[Signal], stats: dict) -> list[Company]:
    _banner("2 · NORMALIZE  ·  entity resolution + dedup")
    companies = normalize.resolve(signals)
    unattached = len(signals) - sum(len(c.signals) for c in companies)
    print(f"  {len(signals)} signals -> {len(companies)} companies")
    if unattached > 0:
        # Explicitly reported rather than quietly dropped.
        print(f"  {unattached} signals had no resolvable owner and were left unattached")
    stats["companies"] = len(companies)
    stats["unattached_signals"] = unattached
    return companies


async def run_triage(
    companies: list[Company], ledger: engine.Ledger, stats: dict
) -> tuple[list[Candidate], engine.Measurement]:
    _banner("3 · TRIAGE  ·  cheap tier")
    candidates: list[Candidate] = []
    passed = rejected = by_llm = 0

    # The ledger's call counter is replay-only, so a live stage would report
    # zero calls beside a real token count. The event stream is the live half.
    with observability.EventCapture() as events:
        for company in companies:
            result = await triage.decide(company, ledger)
            by_llm += 1 if result.used_llm else 0
            candidate = Candidate(
                company=company,
                triage_passed=result.passed,
                triage_rationale=result.rationale,
            )
            candidates.append(candidate)
            if result.passed:
                passed += 1
            else:
                rejected += 1
                print(f"  ✗ {company.name[:38]:<38} {result.rationale[:60]}")

    print(f"\n  passed: {passed} · rejected: {rejected} · resolved by rules: {len(companies) - by_llm}")
    measurement = ledger.measurement("triage")
    if events.totals.llm_calls:
        measurement.llm_cagrisi = events.totals.llm_calls
        measurement.prompt_token = events.totals.prompt_tokens
        measurement.completion_token = events.totals.completion_tokens

    stats.update(
        triage_passed=passed,
        triage_rejected=rejected,
        triage_by_rules=len(companies) - by_llm,
        rejections=[
            {"name": c.company.name, "reason": c.triage_rationale}
            for c in candidates if not c.triage_passed
        ],
    )
    return [c for c in candidates if c.triage_passed], measurement


async def run_enrichment(candidates: list[Candidate], ledger: engine.Ledger) -> list[engine.Measurement]:
    _banner(f"4 · ENRICH + SCORE  ·  {len(candidates)} candidates, parallel branches")
    measurements: list[engine.Measurement] = []

    for candidate in candidates:
        branches, score, measurement = await graph.enrich(candidate.company, ledger)
        candidate.branches = branches
        candidate.score = score
        measurements.append(measurement)

        failed = candidate.failed_branches
        flag = f"  ⚠ missing branches: {', '.join(failed)}" if failed else ""
        total = f"{score.total}/25" if score else "no score"
        print(f"  · {candidate.company.name[:38]:<38} {total:>9}{flag}")

    return measurements


async def write_memos(
    candidates: list[Candidate], ledger: engine.Ledger
) -> engine.Measurement:
    selected = [
        c for c in candidates
        if c.score is not None and c.score.total >= config.THRESHOLDS.review_at
    ]
    _banner(f"5 · MEMO  ·  strong tier, {len(selected)} of {len(candidates)} candidates")
    if not selected:
        print(
            f"  none reached the review threshold ({config.THRESHOLDS.review_at}/25).\n"
            "  This is a threshold outcome, not an absence of candidates."
        )
        return ledger.measurement("memos")

    with observability.EventCapture() as events:
        for candidate in selected:
            candidate.memo = await memo_module.write(candidate, ledger)
            state = "written" if candidate.memo else "model returned no structured memo"
            print(f"  · {candidate.company.name[:38]:<38} {state}")

    measurement = ledger.measurement("memos")
    if events.totals.llm_calls:
        measurement.llm_cagrisi = events.totals.llm_calls
        measurement.prompt_token = events.totals.prompt_tokens
        measurement.completion_token = events.totals.completion_tokens
    return measurement


# --------------------------------------------------------------------------- output


def report(
    candidates: list[Candidate],
    failures: dict[str, str],
    measurements: list[engine.Measurement],
    query: str,
    days: int,
    stats: dict,
) -> Path:
    _banner("RESULT")
    ranked = sorted(
        candidates, key=lambda c: (c.score.total if c.score else -1), reverse=True
    )

    print(f"  {'company':<34} {'score':>6} {'decision':>9} {'reliability':>12}")
    print(f"  {'-' * 34} {'-' * 6} {'-' * 9} {'-' * 12}")
    for candidate in ranked[:20]:
        score = candidate.score
        print(
            f"  {candidate.company.name[:34]:<34} "
            f"{(str(score.total) + '/25') if score else '—':>6} "
            f"{(score.decision if score else '—'):>9} "
            f"{(score.reliability if score else '—'):>12}"
        )

    total = engine.combine(measurements, "run total")
    # The funnel is supposed to spend little at the top and much at the bottom;
    # this breakdown is how you check that it actually did.
    print("\n  cost by stage:")
    for m in measurements:
        if m.llm_cagrisi or m.toplam_token:
            print(f"    {m.desen[:44]:<44} {m.llm_cagrisi:>4} calls  {m.toplam_token:>7} tokens")
    print(
        f"\n  total: {total.llm_cagrisi} LLM calls · {total.toplam_token} tokens · "
        f"{total.arac_cagrisi} tool calls · mode: {total.mod}"
    )
    if failures:
        print("  sources that failed: " + ", ".join(f"{k} ({v[:40]})" for k, v in failures.items()))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    md_path = config.OUTPUT / f"scan-{stamp}.md"
    json_path = config.OUTPUT / f"scan-{stamp}.json"

    header = [
        f"# Scan · {query} · last {days} days",
        "",
        f"*Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} · "
        f"mode: {total.mod} · thesis: {'PLACEHOLDER' if config.THESIS.is_placeholder else 'custom'}*",
        "",
        f"**Thesis:** {config.THESIS.as_prompt()}",
        "",
    ]
    if failures:
        header += ["**Sources that failed this run:**"] + [
            f"- `{k}` — {v}" for k, v in failures.items()
        ] + [""]
    body = [memo_module.render_markdown(c) for c in ranked]
    md_path.write_text("\n".join(header + body), encoding="utf-8")

    json_path.write_text(
        json.dumps(
            {
                "query": query,
                "days": days,
                "mode": total.mod,
                "thesis_is_placeholder": config.THESIS.is_placeholder,
                "failed_sources": failures,
                "cost": total.sozluk(),
                # Per-stage counts and costs: the funnel is only checkable if
                # each layer reports what it dropped and what it spent.
                "funnel": stats,
                "stage_costs": [
                    m.sozluk() for m in measurements if m.llm_cagrisi or m.toplam_token
                ],
                "candidates": [json.loads(c.model_dump_json()) for c in ranked],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n  markdown: {md_path}")
    print(f"  json:     {json_path}")

    # The dashboard is generated from the JSON that was just written, so the
    # page can never drift from the run it claims to describe.
    try:
        import dashboard

        html_path = dashboard.build(json_path, config.OUTPUT / "dashboard.html")
        print(f"  html:     {html_path}")
    except Exception as e:  # a rendering failure must not lose the scan
        print(f"  html:     not generated ({type(e).__name__}: {e})")

    return md_path


# --------------------------------------------------------------------------- main


async def scan(query: str, days: int, limit: int) -> None:
    _preflight()
    ledger = engine.Ledger()
    stats: dict = {}
    try:
        signals, failures = collect(query, days)
        stats["signals"] = len(signals)
        stats["sources_ok"] = len(ALL_COLLECTORS) - len(failures)
        stats["sources_failed"] = len(failures)
        if not signals and failures:
            print("\n  no signals, and every source failed — see the errors above.")
            return

        companies = resolve(signals, stats)
        candidates, triage_cost = await run_triage(companies, ledger, stats)
        candidates = candidates[:limit]
        stats["enriched"] = len(candidates)
        measurements = await run_enrichment(candidates, ledger)
        memo_cost = await write_memos(candidates, ledger)
        stats["memos"] = sum(1 for c in candidates if c.memo is not None)
        report(
            candidates, failures, [triage_cost, *measurements, memo_cost], query, days, stats
        )
    finally:
        await ledger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="VC pipeline — one scan pass")
    parser.add_argument("--query", default="ai infrastructure", help="sector or topic to scan")
    parser.add_argument("--days", type=int, default=config.THRESHOLDS.default_days)
    parser.add_argument(
        "--limit",
        type=int,
        default=config.THRESHOLDS.max_candidates,
        help="cap on how many candidates get enriched",
    )
    args = parser.parse_args()
    asyncio.run(scan(args.query, args.days, args.limit))


if __name__ == "__main__":
    main()
