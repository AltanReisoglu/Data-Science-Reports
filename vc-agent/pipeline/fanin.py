"""Core-level fan-in for enrichment: publish/subscribe plus a collector queue.

An alternative to `graph.py`, built on `autogen_core` instead of AgentChat, and
taken directly from the core guide's **Concurrent Agents** pattern: workers
publish results to a results topic, and a ``ClosureAgent`` drains that topic into
an ``asyncio.Queue`` the caller owns.

## Why a second engine exists

`graph.py` inherits its fan-in from the framework: `GraphFlow` decides when the
join is satisfied and hands back a message list at the end. That coupling is the
source of every branch-loss problem measured so far — if the run aborts, the
completed work goes with it, because the results were only ever inside the run.

Here the coupling is gone. A branch publishes its outcome the moment it has one,
and the queue holds it from that instant. Nothing downstream can take it back.
A branch that crashes publishes a failure; a branch that vanishes entirely simply
never arrives, and the collector reports the shortfall against the count it
expected. There is no barrier to distrust because there is no barrier.

The two design-pattern pages differ on exactly this point, which is worth
noticing: **Concurrent Agents** collects through a queue, while **Mixture of
Agents** aggregates with ``asyncio.gather(...)`` — the same construct whose early
return `poc/desen_5_core_aktor.py` traced to silent sibling loss. The officially
documented aggregation patterns do not agree with each other about failure.

## What it does not give you

AgentChat's `AssistantAgent` is still doing the analysis inside each worker, so
tools, structured output and model tiering are unchanged. What changes is only
how results are gathered — which is precisely the part that was breaking.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

# These must be importable at module scope, not inside a function. `@message_handler`
# infers the message type by calling `get_type_hints` on the handler, and with
# `from __future__ import annotations` every annotation is a string resolved
# against *module* globals — a function-local import is invisible to it and the
# registration fails with a bare `NameError: name 'MessageContext' is not defined`.
from autogen_core import (  # noqa: E402
    ClosureAgent,
    ClosureContext,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
)

import config
import engine
import stages
import observability
from agents import analysts, triage
from schemas import BranchResult, Company, Score

TASK_TOPIC = "enrichment_task"
RESULT_TOPIC = "enrichment_result"
COLLECTOR_TYPE = "result_collector"


@dataclass
class BranchTask:
    company_brief: str


@dataclass
class BranchOutcome:
    branch: str
    succeeded: bool
    text: str = ""
    error: str | None = None


class BranchWorker(RoutedAgent):
    """One enrichment branch. It never raises; it always reports."""

    def __init__(self, branch: str, agent) -> None:
        super().__init__(f"{branch} branch")
        self._branch = branch
        self._agent = agent

    @message_handler
    async def handle(self, message: BranchTask, ctx: MessageContext) -> None:
        # `self.id.key` is the topic's source, handed over by the runtime — the
        # mapping the guide spends a page on and nobody reads.
        stages.emit_line("branch", branch=self._branch, key=self.id.key)
        try:
            result = await self._agent.run(task=message.company_brief)
            text = str(getattr(result.messages[-1], "content", ""))
            if engine.BRANCH_FAILURE_MARKER in text:
                # `ResilientClient` turned a failed model call into text. It
                # arrives like any other answer, so without this check the branch
                # would be counted as a success carrying an error message —
                # exactly the silent-partial-result failure this module exists to
                # prevent, reintroduced one level down.
                outcome = BranchOutcome(
                    branch=self._branch,
                    succeeded=False,
                    error=text.replace(engine.BRANCH_FAILURE_MARKER, "").strip(),
                )
            else:
                outcome = BranchOutcome(
                    branch=self._branch, succeeded=bool(text.strip()), text=text
                )
                if not outcome.succeeded:
                    outcome.error = "branch produced an empty answer"
        except Exception as e:
            # The whole point: a failure is published, not raised. Raising here
            # is what lets one branch abort the run and take its siblings'
            # completed work with it.
            outcome = BranchOutcome(
                branch=self._branch, succeeded=False, error=f"{type(e).__name__}: {e}"
            )
        await self.publish_message(outcome, topic_id=TopicId(RESULT_TOPIC, source="default"))


def _factory(branch: str, agent):
    """Bind the loop variables now rather than at call time."""
    return lambda: BranchWorker(branch, agent)


async def enrich(
    company: Company,
    ledger: engine.Ledger,
    thesis: config.Thesis | None = None,
    *,
    timeout: float | None = None,
) -> tuple[list[BranchResult], Score | None, engine.Measurement]:
    """Run the three branches concurrently and collect whatever arrives."""
    deadline = config.THRESHOLDS.enrichment_timeout_seconds if timeout is None else timeout
    handler = observability.AuditingInterventionHandler()
    runtime = SingleThreadedAgentRuntime(
        intervention_handlers=[handler], ignore_unhandled_exceptions=False
    )

    technical, market, team = analysts.build_analysts(company, ledger)
    branch_agents = {"technical": technical, "market": market, "team": team}
    results: asyncio.Queue[BranchOutcome] = asyncio.Queue()

    async def collect(_ctx: ClosureContext, message: BranchOutcome, _mc: MessageContext) -> None:
        await results.put(message)
        stages.emit_line("collect", branch=message.branch,
                         ok=message.succeeded, queued=results.qsize())

    started = time.perf_counter()
    with observability.EventCapture() as events:
        # Each branch is its own agent type, all subscribed to the one task
        # topic: publishing once starts all three concurrently.
        for branch, agent in branch_agents.items():
            await BranchWorker.register(runtime, branch, _factory(branch, agent))
            await runtime.add_subscription(
                TypeSubscription(topic_type=TASK_TOPIC, agent_type=branch)
            )
        stages.emit_line("subscribe", topic=TASK_TOPIC,
                         agents=sorted(branch_agents), company=company.name)

        await ClosureAgent.register_closure(
            runtime,
            COLLECTOR_TYPE,
            collect,
            subscriptions=lambda: [
                TypeSubscription(topic_type=RESULT_TOPIC, agent_type=COLLECTOR_TYPE)
            ],
        )

        runtime.start()
        stages.emit_line("runtime_start", company=company.name,
                         intervention="AuditingInterventionHandler")
        stages.emit_line("intervention", handler="AuditingInterventionHandler")
        await runtime.publish_message(
            BranchTask(company_brief=triage.describe(company)),
            topic_id=TopicId(TASK_TOPIC, source="default"),
        )
        stages.emit_line("publish", topic=TASK_TOPIC, source="default",
                         subscribers=len(branch_agents))

        collected: dict[str, BranchOutcome] = {}
        try:
            # Wait for the count we expect, not for the runtime to say it is idle.
            async def drain() -> None:
                while len(collected) < len(branch_agents):
                    outcome = await results.get()
                    collected[outcome.branch] = outcome

            await asyncio.wait_for(drain(), timeout=deadline)
            timed_out = False
        except asyncio.TimeoutError:
            timed_out = True
        stages.emit_line("join", expected=len(branch_agents),
                         collected=len(collected), timed_out=timed_out)

        try:
            await asyncio.wait_for(runtime.stop_when_idle(), timeout=5.0)
        except Exception:
            try:
                await runtime.stop()
            except Exception:
                pass
        await runtime.close()
        stages.emit_line("runtime_stop", company=company.name)

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    branches: list[BranchResult] = []
    for branch in branch_agents:
        outcome = collected.get(branch)
        if outcome is None:
            branches.append(
                BranchResult(
                    branch=branch,  # type: ignore[arg-type]
                    succeeded=False,
                    error=(
                        f"no result published within {deadline:.0f}s"
                        if timed_out
                        else "no result published"
                    ),
                )
            )
        else:
            branches.append(
                BranchResult(
                    branch=branch,  # type: ignore[arg-type]
                    succeeded=outcome.succeeded,
                    text=outcome.text,
                    error=outcome.error,
                )
            )

    measurement = ledger.measurement(f"fanin:{company.name}")
    measurement.sure_ms = elapsed_ms
    measurement.mesaj_sayisi = len(collected)
    measurement.durma_nedeni = "timeout" if timed_out else "all branches reported"
    measurement.arac_cagrisi = len(events.totals.tool_calls)
    if events.totals.llm_calls:
        measurement.llm_cagrisi = events.totals.llm_calls
        measurement.prompt_token = events.totals.prompt_tokens
        measurement.completion_token = events.totals.completion_tokens

    # Scoring stays where it was: this module replaces the fan-in, not the rubric.
    return branches, None, measurement
