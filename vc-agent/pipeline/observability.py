"""Observability: event capture and a message-level intervention gate.

Both mechanisms come from the AutoGen **core** user guide rather than AgentChat,
and each closes a hole that the AgentChat surface alone leaves open.

## 1. Event capture (`autogen_core.events`)

`autogen_core` emits structured events on a dedicated logger. Two matter here:

* ``LLMCallEvent`` and ``LLMStreamEndEvent`` — both carry ``prompt_tokens`` /
  ``completion_tokens``, and which one arrives depends on how the model was
  called: ``create()`` emits the first, ``create_stream()`` only ever the
  second. Both are emitted by the **real** model clients (openai, anthropic,
  ollama, azure…).
  It is *not* emitted by ``ReplayChatCompletionClient``. That asymmetry is
  exactly why both accounting paths are needed: `engine.Ledger` counts
  ``create_calls``, which only the replay client exposes, so in live mode it
  reports zero. The event stream is the live-mode counterpart, and the two
  together mean cost is measured in both modes rather than in neither.

* ``ToolCallEvent`` — emitted by every subclass of ``autogen_core.tools.BaseTool``
  with the tool name, its arguments and its result. This is what makes docs/04 §6
  real: *"every tool call is written to the audit record, so 'where did this score
  come from' always has an answer."* Without it the audit log only knows about
  HTTP requests the collectors made, not about the calls an agent chose to make.

  The event also carries ``agent_id``, but only when the call happens inside a
  runtime-managed handler — verified live: a bare ``agent.run()`` records
  ``None``, the same agent inside a team records ``TechnicalAnalyst_<uuid>``.
  Since enrichment always runs inside the graph, attribution holds where it
  matters.

## 2. Intervention handler

`SingleThreadedAgentRuntime(intervention_handlers=[...])` puts a handler on the
message path *before* delivery. Two uses, both from the core cookbook:

* **Audit** — record which agent sent what to whom. AgentChat's message list
  shows the conversation; this shows the routing.
* **Approval gate** — the partner gate of docs/04. Returning ``DropMessage``
  suppresses delivery entirely, so the gate is enforced by the runtime rather
  than by an agent choosing to behave.

The gate is off by default. Every tool in this pipeline is read-only, so
prompting on each call would be ceremony; it is wired and tested so that the
first mutating tool has a gate waiting for it rather than needing one retrofitted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from autogen_core import EVENT_LOGGER_NAME, DefaultInterventionHandler, DropMessage
from autogen_core.logging import LLMCallEvent, LLMStreamEndEvent, ToolCallEvent

import policy as policy_module


# --------------------------------------------------------------------------- events


@dataclass
class ToolCallRecord:
    tool: str
    arguments: dict[str, Any]
    result_preview: str
    agent: str | None = None


@dataclass
class EventTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class EventCapture(logging.Handler):
    """Captures `autogen_core` events and mirrors tool calls into the audit log."""

    def __init__(self, source_policy: policy_module.SourcePolicy | None = None) -> None:
        super().__init__()
        self.totals = EventTotals()
        self._policy = source_policy or policy_module.DEFAULT

    def emit(self, record: logging.LogRecord) -> None:
        try:
            event = record.msg
            # Two events, one meaning. `create()` emits `LLMCallEvent`;
            # `create_stream()` emits `LLMStreamEndEvent` and never the first —
            # so an agent with `model_client_stream=True` reports zero usage if
            # only the former is handled. Measured on the live conversation.
            if isinstance(event, (LLMCallEvent, LLMStreamEndEvent)):
                self.totals.prompt_tokens += event.prompt_tokens
                self.totals.completion_tokens += event.completion_tokens
                self.totals.llm_calls += 1
            elif isinstance(event, ToolCallEvent):
                self._record_tool_call(event)
        except Exception:
            self.handleError(record)

    def _record_tool_call(self, event: ToolCallEvent) -> None:
        # Unlike `LLMCallEvent`, which exposes its fields as attributes,
        # `ToolCallEvent` keeps everything in `.kwargs` — including `agent_id`,
        # which is what makes the record answer *who* called the tool.
        fields = getattr(event, "kwargs", {}) or {}
        tool = str(fields.get("tool_name", "unknown"))
        arguments = dict(fields.get("arguments") or {})
        result = str(fields.get("result", ""))
        agent = fields.get("agent_id")

        self.totals.tool_calls.append(
            ToolCallRecord(
                tool=tool,
                arguments=arguments,
                result_preview=result[:200],
                agent=str(agent) if agent else None,
            )
        )
        # The audit log already holds the HTTP requests the collectors made. This
        # adds the layer above it: which tool an agent chose to call. The
        # arguments go in as their own dict rather than merged with `agent`, so
        # the ledger's key list describes the *call* — `agent` is a field of the
        # record, not a parameter of the tool.
        self._policy.record_agent_action(
            tool=tool,
            arguments=arguments,
            result_size=len(result),
            agent=str(agent) if agent else None,
        )

    # ------------------------------------------------------------ lifecycle

    def __enter__(self) -> "EventCapture":
        logger = logging.getLogger(EVENT_LOGGER_NAME)
        self._previous_level = logger.level
        logger.setLevel(logging.INFO)
        logger.addHandler(self)
        return self

    def __exit__(self, *_exc: object) -> None:
        logger = logging.getLogger(EVENT_LOGGER_NAME)
        logger.removeHandler(self)
        logger.setLevel(self._previous_level)


# --------------------------------------------------------------------------- intervention


class AuditingInterventionHandler(DefaultInterventionHandler):
    """Sits on the message path: records routing, and optionally gates delivery.

    ``approve`` receives ``(sender, recipient, message)`` and returns False to
    drop the message. Left as ``None``, nothing is gated and the handler is a
    pure observer.
    """

    def __init__(
        self,
        *,
        approve: Callable[[str, str, Any], bool] | None = None,
        source_policy: policy_module.SourcePolicy | None = None,
    ) -> None:
        self._approve = approve
        self._policy = source_policy or policy_module.DEFAULT
        self.routed: list[tuple[str, str]] = []
        self.dropped: list[tuple[str, str]] = []

    async def on_send(self, message: Any, *, message_context: Any, recipient: Any) -> Any | type[DropMessage]:
        sender = str(getattr(message_context, "sender", None) or "runtime")
        target = str(recipient)
        self.routed.append((sender, target))

        if self._approve is not None and not self._approve(sender, target, message):
            self.dropped.append((sender, target))
            self._policy.record_agent_action(
                tool="message", arguments={"from": sender, "to": target}, result_size=0,
                outcome="dropped_by_approval_gate",
            )
            # Dropped by the runtime itself, not by an agent deciding to comply.
            return DropMessage
        return message

    # The runtime warns when a handler returns None, so both remaining hooks are
    # implemented explicitly as pass-throughs rather than left to inheritance.
    async def on_publish(self, message: Any, *, message_context: Any) -> Any | type[DropMessage]:
        return message

    async def on_response(self, message: Any, *, sender: Any, recipient: Any) -> Any | type[DropMessage]:
        return message
