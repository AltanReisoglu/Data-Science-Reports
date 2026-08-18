"""The enrichment graph: parallel fan-out, join, risk audit, score.

    TechnicalAnalyst ─┐
    MarketAnalyst    ─┼─► RiskAuditor ─► Scorer      (join: activation_condition="all")
    TeamAnalyst      ─┘

GraphFlow is used here rather than a chat team because enrichment is **fixed and
parallel**, and GraphFlow is the only AutoGen pattern that gives real concurrency
(docs/04 §3). Triage, whose routing is dynamic, uses the cheap path instead.

## The barrier is not trusted

This is the one rule in the file that exists because of a measurement rather than
a design preference. In `poc/desen_5_core_aktor.py` a crashing handler was shown
to return ``asyncio.gather`` early, ``task_done()`` fires immediately, the
``stop_when_idle()`` barrier opens before its siblings finish, and their results
disappear **with no exception and no warning**. Reproduced three times.

The same shape of failure lands exactly here — a parallel branch of an enrichment
graph. So this module does not ask the framework whether all branches completed.
It **counts the branches it expected**, and every branch that did not report
becomes an entry in ``Score.missing_data``. A silent partial result is turned
into a stated absence of information.
"""

from __future__ import annotations

import asyncio
import re
import time

import config
import engine
import stages
import observability
from agents import analysts, triage
from schemas import BranchResult, Company, Score, Source

# The branches we expect, and the agent that owns each one.
EXPECTED_BRANCHES: dict[str, str] = {
    "TechnicalAnalyst": "technical",
    "MarketAnalyst": "market",
    "TeamAnalyst": "team",
}

_URL = re.compile(r"https?://[^\s\"'<>\)\]]+")


def build(
    company: Company,
    ledger: engine.Ledger,
    thesis: config.Thesis | None = None,
    *,
    approve=None,
):
    """Assemble the graph.

    Returns the team, its runtime and the intervention handler. The runtime is
    constructed here rather than left to AgentChat because an externally supplied
    runtime is the only way to attach an ``InterventionHandler`` — and because
    owning the runtime means owning ``stop_when_idle()``, which is the barrier
    the POC measured opening early.

    Passing an external runtime also transfers its lifecycle: AgentChat starts
    and stops only the runtime it created itself, so `enrich()` must do it.
    """
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.messages import StructuredMessage
    from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
    from autogen_core import SingleThreadedAgentRuntime

    technical, market, team = analysts.build_analysts(company, ledger)
    risk = analysts.build_risk_auditor(company, ledger)
    scorer = analysts.build_scorer(company, ledger, thesis)

    builder = DiGraphBuilder()
    for agent in (technical, market, team, risk, scorer):
        builder.add_node(agent)

    # The three analysts have no incoming edges, so they are source nodes and
    # start concurrently.
    for branch in (technical, market, team):
        builder.add_edge(branch, risk, activation_group="enrichment", activation_condition="all")
    builder.add_edge(risk, scorer)

    handler = observability.AuditingInterventionHandler(approve=approve)
    # `ignore_unhandled_exceptions=False` matches what AgentChat builds for
    # itself; changing it here would quietly change failure semantics.
    runtime = SingleThreadedAgentRuntime(
        intervention_handlers=[handler], ignore_unhandled_exceptions=False
    )

    flow = GraphFlow(
        participants=[technical, market, team, risk, scorer],
        graph=builder.build(),
        # Runaway-loop fuse: an unbounded agent loop is a real invoice.
        termination_condition=MaxMessageTermination(config.THRESHOLDS.max_messages),
        # A team refuses to route a message type it has not been told about, so
        # the scorer's structured output has to be declared here.
        custom_message_types=[StructuredMessage[Score]],
        runtime=runtime,
    )
    return flow, runtime, handler


async def enrich(
    company: Company,
    ledger: engine.Ledger,
    thesis: config.Thesis | None = None,
    *,
    approve=None,
    timeout: float | None = None,
) -> tuple[list[BranchResult], Score | None, engine.Measurement]:
    """Run the graph for one company and return branches, score and measurement."""
    from autogen_agentchat.base import TaskResult
    from autogen_agentchat.messages import StructuredMessage

    flow, runtime, _handler = build(company, ledger, thesis, approve=approve)
    deadline = config.THRESHOLDS.enrichment_timeout_seconds if timeout is None else timeout

    started = time.perf_counter()

    # Streamed rather than awaited as a whole, on purpose. If any branch raises,
    # `run()` throws away everything that already arrived — which is the same
    # silent loss of sibling results the POC measured, just at a different layer.
    # Streaming keeps what did arrive and lets the missing branches be reported.
    messages: list = []
    run_error: str | None = None
    stop_reason = ""

    stages.emit_line("graph_build", company=company.name,
                     branches=sorted(EXPECTED_BRANCHES),
                     termination="MaxMessageTermination")
    stages.emit_line("intervention", handler="AuditingInterventionHandler",
                     own_runtime=True)

    seen_branches: set[str] = set()

    async def _stream() -> str:
        async for item in flow.run_stream(task=triage.describe(company)):
            if isinstance(item, TaskResult):
                messages.clear()
                messages.extend(item.messages)
                return item.stop_reason or ""
            # Each branch reports once. Announcing them as they land is what makes
            # the fan-out visible: without this the panel would jump from "running"
            # straight to "done" and the parallelism would never be seen.
            speaker = str(getattr(item, "source", ""))
            if speaker in EXPECTED_BRANCHES and speaker not in seen_branches:
                seen_branches.add(speaker)
                stages.emit_line("analysts", branch=speaker,
                                 arrived=len(seen_branches),
                                 expected=len(EXPECTED_BRANCHES))
            messages.append(item)
        return ""

    # The event stream is the only place live-mode token usage appears, since
    # `ReplayChatCompletionClient` never emits `LLMCallEvent`.
    with observability.EventCapture() as events:
        runtime.start()
        stages.emit_line("graph_run", company=company.name, streamed=True)
        try:
            stop_reason = await asyncio.wait_for(_stream(), timeout=deadline)
        except asyncio.TimeoutError:
            # Measured behaviour, not a precaution: with an externally supplied
            # runtime a crashing agent leaves `run_stream` waiting on a
            # termination message that will never arrive. Message-count fuses do
            # not fire, because no further messages arrive either.
            run_error = f"enrichment exceeded {deadline:.0f}s and was abandoned"
            stop_reason = f"aborted: {run_error}"
        except Exception as e:
            run_error = f"{type(e).__name__}: {e}"
            stop_reason = f"aborted: {run_error}"
        finally:
            # We supplied the runtime, so we close it. Left running, its queue
            # would keep consuming messages after the team believes it is done.
            # `stop_when_idle()` is itself bounded — it is the barrier the POC
            # found unreliable, and it will not be trusted to return here either.
            try:
                await asyncio.wait_for(runtime.stop_when_idle(), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                try:
                    await runtime.stop()
                except Exception:
                    pass
            await runtime.close()
            stages.emit_line("join", expected=len(EXPECTED_BRANCHES),
                             arrived=len(seen_branches), stop_reason=stop_reason[:80])
            stages.emit_line("runtime_stop", company=company.name)

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    # --- what actually came back, per branch ------------------------------
    texts: dict[str, str] = {}
    score: Score | None = None
    for message in messages:
        speaker = getattr(message, "source", "")
        if speaker in EXPECTED_BRANCHES:
            body = getattr(message, "content", "")
            if isinstance(body, str) and body.strip():
                texts[speaker] = body
        if isinstance(message, StructuredMessage) and isinstance(message.content, Score):
            score = message.content

    branches: list[BranchResult] = []
    for agent_name, branch_name in EXPECTED_BRANCHES.items():
        body = texts.get(agent_name)
        if body and engine.BRANCH_FAILURE_MARKER in body:
            # The branch reported its own failure instead of raising, so its
            # siblings survived. Record it as an absence, not as an analysis.
            branches.append(
                BranchResult(
                    branch=branch_name,
                    succeeded=False,
                    error=body.replace(engine.BRANCH_FAILURE_MARKER, "").strip(),
                )
            )
        elif body:
            branches.append(
                BranchResult(
                    branch=branch_name,
                    succeeded=True,
                    text=body,
                    sources=_sources_from(body, agent_name),
                )
            )
        else:
            # Not an exception — a recorded absence. This is the whole point of
            # the module docstring.
            branches.append(
                BranchResult(
                    branch=branch_name,
                    succeeded=False,
                    error=(
                        run_error
                        or f"{agent_name} produced no output "
                        "(barrier opened early or branch failed)"
                    ),
                )
            )

    # --- enforce the missing-data contract in code ------------------------
    # The model does not get the last word on what is missing. Whatever the
    # scorer claimed, a branch that did not report IS missing data.
    if score is not None:
        for branch in branches:
            if not branch.succeeded:
                note = f"{branch.branch} branch produced no result"
                if note not in score.missing_data:
                    score.missing_data.append(note)
    stages.emit_line(
        "count",
        expected=len(EXPECTED_BRANCHES),
        succeeded=sum(1 for b in branches if b.succeeded),
        missing=list(score.missing_data) if score is not None else [],
    )

    measurement = ledger.measurement(f"enrichment:{company.name}")
    measurement.sure_ms = elapsed_ms
    measurement.mesaj_sayisi = len(messages)
    measurement.durma_nedeni = stop_reason
    # Tool executions are counted from the core event stream rather than from the
    # request messages: a requested call that never executed is not a cost.
    measurement.arac_cagrisi = len(events.totals.tool_calls)
    # Live mode: the ledger's `create_calls` counter is replay-only, so the event
    # totals are the real numbers. Dry mode: no events, ledger stands.
    if events.totals.llm_calls:
        measurement.llm_cagrisi = events.totals.llm_calls
        measurement.prompt_token = events.totals.prompt_tokens
        measurement.completion_token = events.totals.completion_tokens
    return branches, score, measurement


def _sources_from(text: str, agent_name: str) -> list[Source]:
    """Pull the URLs an analyst actually cited out of its own answer."""
    seen: list[Source] = []
    for url in dict.fromkeys(_URL.findall(text)):
        try:
            seen.append(Source(name=agent_name, url=url.rstrip(".,);"), confidence="secondary"))
        except ValueError:
            continue
    return seen
