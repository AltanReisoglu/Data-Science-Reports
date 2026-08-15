"""The context engine: what the model sees, and what happens when it stops fitting.

`BufferedChatCompletionContext(buffer_size=24)` was the whole context strategy, and
it has a flaw the project has been carrying in writing since docs/06 §2: **it
counts messages, not tokens**. Twenty-four one-line questions and twenty-four
answers carrying a full scan result are the same number to it and nowhere near
the same size. The first is a third of a window; the second overflows it, and the
overflow arrives as a provider error at the end of a turn.

OpenClaw's context engine (docs/13 §6) answers this with four lifecycle points,
reproduced here on AutoGen's `ChatCompletionContext`:

| OpenClaw | here |
|---|---|
| **Ingest** — a message is added | `add_message` |
| **Assemble** — before every model run, the ordered set that fits the budget | `get_messages` |
| **Compact** — the window is full | `compact` |
| **After turn** | `after_turn` |

### The rule that makes compaction correct rather than merely smaller

A tool call and its result are one unit. `AssistantMessage` carrying a
`FunctionCall` is meaningless without the `FunctionExecutionResultMessage` that
answers it, and the reverse is worse: a result whose call has been summarised away
is an answer to a question the model can no longer see, and providers reject the
sequence outright. OpenClaw states this directly — *"araç çağrıları eşleşen
`toolResult` girişleriyle birlikte tutulur; bölme noktası bir araç bloğunun içine
düşerse sınır kaydırılır"*. `_safe_boundary` is that rule, and it is the reason
this module is not twenty lines.

### Failing towards a working agent

If the summariser is unavailable — no model configured, endpoint down, request
refused — compaction still has to produce something the model can accept. It
falls back to dropping the oldest complete blocks and says so in the summary
placeholder. A context engine that raises would take the conversation down with
it, which is exactly what OpenClaw's quarantine rule exists to prevent.

### Token counting

`ChatCompletionClient.count_tokens` is the accurate path and needs a live client.
Without one, `_estimate` uses characters÷4 — deliberately crude, and biased to
*over*-count so the budget is conservative rather than optimistic. An
under-estimate produces exactly the overflow this module exists to avoid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from autogen_core.models import (
    AssistantMessage,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.model_context import ChatCompletionContext

import config

SUMMARY_PREFIX = "[compacted]"


@dataclass
class CompactionEvent:
    """One compaction, for the record and for the tests."""

    at: str
    before_messages: int
    after_messages: int
    before_tokens: int
    after_tokens: int
    summarised: int
    method: str = "model"          # model | truncate
    note: str = ""


@dataclass
class EngineStats:
    compactions: list[CompactionEvent] = field(default_factory=list)
    assembles: int = 0

    @property
    def compacted(self) -> int:
        return len(self.compactions)


# --------------------------------------------------------------------------- tokens


def _text_of(message: LLMMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            else:
                parts.append(json.dumps(getattr(item, "__dict__", str(item)), default=str))
        return " ".join(parts)
    return str(content)


def estimate_tokens(messages: Sequence[LLMMessage]) -> int:
    """Characters over four, rounded up, plus per-message overhead.

    Crude on purpose and biased high: under-counting produces the provider
    overflow this module exists to prevent, over-counting only compacts early.
    """
    total = 0
    for message in messages:
        total += (len(_text_of(message)) + 3) // 4 + 4
    return total


# --------------------------------------------------------------------------- pairing


def _is_tool_call(message: LLMMessage) -> bool:
    return isinstance(message, AssistantMessage) and isinstance(message.content, list)


def _is_tool_result(message: LLMMessage) -> bool:
    return isinstance(message, FunctionExecutionResultMessage)


def _safe_boundary(messages: Sequence[LLMMessage], index: int) -> int:
    """Move `index` forward until it does not split a tool call from its result.

    A cut is unsafe when the message it lands on is a tool result — its call
    would be on the other side — or when the message just before it is a tool
    call whose result is on this side. Walking forward drops slightly more than
    asked, which is the safe direction: keeping too little history costs recall,
    keeping a dangling result costs the whole request.
    """
    n = len(messages)
    cut = max(0, min(index, n))
    while cut < n:
        if _is_tool_result(messages[cut]):
            cut += 1
            continue
        if cut > 0 and _is_tool_call(messages[cut - 1]):
            cut += 1
            continue
        break
    return cut


# --------------------------------------------------------------------------- engine


class CompactingChatCompletionContext(ChatCompletionContext):
    """Token-budgeted context with tool-aware compaction.

    `summariser` is an async callable taking the messages to fold away and
    returning a string. Left as None, compaction truncates and says so — which is
    the dry-mode behaviour and keeps the suite free of model calls.
    """

    component_type = "chat_completion_context"

    def __init__(
        self,
        initial_messages: list[LLMMessage] | None = None,
        *,
        token_budget: int | None = None,
        reserve: int | None = None,
        summariser=None,
        keep_recent: int = 4,
        counter=None,
        on_compaction=None,
    ) -> None:
        super().__init__(initial_messages)
        policy = config.SESSION_POLICY
        self.token_budget = token_budget or policy.token_budget
        self.reserve = reserve if reserve is not None else policy.compaction_reserve
        self.keep_recent = keep_recent
        self.owns_compaction = True
        self.stats = EngineStats()
        self._summariser = summariser
        self._counter = counter or estimate_tokens
        self._on_compaction = on_compaction
        self._summary: str = ""

    # ------------------------------------------------------------ Ingest

    async def add_message(self, message: LLMMessage) -> None:
        await super().add_message(message)

    # ------------------------------------------------------------ Assemble

    async def get_messages(self) -> list[LLMMessage]:
        """The ordered set that fits the budget, compacting first if it does not."""
        self.stats.assembles += 1
        if self._over_budget(self._messages):
            await self.compact()
        assembled = list(self._messages)
        if self._summary:
            assembled = [self._summary_message()] + assembled
        return assembled

    def _summary_message(self) -> LLMMessage:
        return SystemMessage(content=f"{SUMMARY_PREFIX} earlier conversation:\n{self._summary}")

    def _over_budget(self, messages: Sequence[LLMMessage]) -> bool:
        return self.usable() < self._counter(messages)

    def usable(self) -> int:
        return max(0, self.token_budget - self.reserve)

    def tokens(self) -> int:
        held = list(self._messages)
        if self._summary:
            held = [self._summary_message()] + held
        return self._counter(held)

    # ------------------------------------------------------------ Compact

    async def compact(self, *, force: bool = False) -> CompactionEvent | None:
        """Fold the oldest messages into a summary, never splitting a tool block."""
        from datetime import datetime, timezone

        messages = list(self._messages)
        if not force and not self._over_budget(messages):
            return None
        if len(messages) <= self.keep_recent:
            return None

        before_tokens = self._counter(messages)
        cut = _safe_boundary(messages, max(0, len(messages) - self.keep_recent))
        if cut <= 0:
            return None

        older, recent = messages[:cut], messages[cut:]
        summary, method, note = await self._summarise(older)

        self._summary = f"{self._summary}\n{summary}".strip() if self._summary else summary
        self._messages = recent

        event = CompactionEvent(
            at=datetime.now(timezone.utc).isoformat(),
            before_messages=len(messages),
            after_messages=len(recent),
            before_tokens=before_tokens,
            after_tokens=self.tokens(),
            summarised=len(older),
            method=method,
            note=note,
        )
        self.stats.compactions.append(event)
        if self._on_compaction is not None:
            try:
                self._on_compaction(event)
            except Exception:  # noqa: BLE001 — a listener must not break the turn
                pass
        return event

    async def _summarise(self, older: Sequence[LLMMessage]) -> tuple[str, str, str]:
        if self._summariser is None:
            return self._truncated(older), "truncate", "no summariser configured"
        try:
            text = await self._summariser(list(older))
        except Exception as exc:  # noqa: BLE001
            # An unreachable model must not take the conversation with it.
            return self._truncated(older), "truncate", f"{type(exc).__name__}: {exc}"
        text = (text or "").strip()
        if not text:
            return self._truncated(older), "truncate", "summariser returned nothing"
        return text, "model", ""

    @staticmethod
    def _truncated(older: Sequence[LLMMessage]) -> str:
        """The honest fallback: say what was dropped rather than pretend it is kept."""
        kinds: dict[str, int] = {}
        for message in older:
            name = "tool" if _is_tool_call(message) or _is_tool_result(message) else (
                "user" if isinstance(message, UserMessage) else "assistant"
            )
            kinds[name] = kinds.get(name, 0) + 1
        shape = ", ".join(f"{count} {name}" for name, count in sorted(kinds.items()))
        return (
            f"{len(older)} earlier messages ({shape}) were dropped to stay inside the "
            "context window. They were not summarised, so anything only stated there "
            "is no longer available — ask again rather than assuming."
        )

    # ------------------------------------------------------------ After turn

    async def after_turn(self) -> None:
        """Hook point for post-turn work — memory flush lands here."""
        return None

    # ------------------------------------------------------------ state

    async def save_state(self):
        state = dict(await super().save_state())
        state["summary"] = self._summary
        return state

    async def load_state(self, state) -> None:
        await super().load_state({k: v for k, v in dict(state).items() if k != "summary"})
        self._summary = str(dict(state).get("summary") or "")

    async def clear(self) -> None:
        await super().clear()
        self._summary = ""


# --------------------------------------------------------------------------- summariser


def model_summariser(client, *, max_chars: int = 6000):
    """A summariser backed by a model client — the cheap tier, by design.

    Compaction is bookkeeping, not analysis. Paying the strong tier to compress
    old turns would invert the funnel's whole cost argument.
    """

    async def summarise(messages: list[LLMMessage]) -> str:
        transcript = "\n".join(
            f"{getattr(m, 'source', type(m).__name__)}: {_text_of(m)[:600]}" for m in messages
        )[:max_chars]
        result = await client.create(
            [
                SystemMessage(
                    content=(
                        "Compress this conversation for reuse as context. Keep decisions, "
                        "company names, numbers, sources and anything the user asked to be "
                        "remembered. Drop pleasantries. Say plainly what is uncertain. "
                        "No preamble."
                    )
                ),
                UserMessage(content=transcript, source="compaction"),
            ]
        )
        content = result.content
        return content if isinstance(content, str) else str(content)

    return summarise


def legacy_context(buffer_size: int | None = None):
    """The engine we had. Kept as the fallback a quarantined engine drops to."""
    from autogen_core.model_context import BufferedChatCompletionContext

    return BufferedChatCompletionContext(
        buffer_size=buffer_size or config.SESSION_POLICY.buffer_size
    )


__all__ = [
    "SUMMARY_PREFIX",
    "CompactingChatCompletionContext",
    "CompactionEvent",
    "EngineStats",
    "estimate_tokens",
    "legacy_context",
    "model_summariser",
]
