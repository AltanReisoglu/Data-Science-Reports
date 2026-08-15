"""A workbench that asks the hooks before it calls anything.

The approval gate needs a place to stand. `InterventionHandler` — the mechanism
`observability.py` already provides — sits on the *runtime's* message path, which
works when a team is running under a runtime we own (`graph.py`) and does not
exist for a bare `AssistantAgent` holding a conversation. Tool calls there go
straight from the agent to the workbench.

So the gate goes on the workbench. `Workbench` is already the abstraction every
tool call passes through, local and remote alike, which makes it the one place
that sees an MCP tool and a Python function the same way. Wrapping it means the
rule is enforced for tools that **did not exist when the agent was written** — the
whole reason a workbench is a tool *source* rather than a tool list, and exactly
where an allowlist of known-dangerous names would fail.

Two properties worth being explicit about:

**A blocked call returns an error result, it does not raise.** The agent should
learn that it was refused and why, and be able to say so or choose another route.
An exception would end the turn and tell the person nothing.

### Gating and filtering are different decisions

**Gating** leaves a tool visible and refuses the call. That is right for
`messages_send`: the agent can then say *"I would message them, but that needs
your approval"*, which is a useful sentence. Hiding it would make the model
retry blindly or invent a workaround.

**Filtering** removes a tool from `list_tools` entirely, so it is never in the
prompt. That is right for two other cases, and neither is about safety theatre:

* **Prompt cost.** Schemas are paid for on every turn, and docs/06 records a live
  timeout on a prompt carrying seven of them. Twenty-one is not a rounding error.
* **No legitimate use.** `permissions_respond` answers OpenClaw's own approval
  prompts. There is no request where calling it is the right move, so showing it
  and refusing it buys nothing — the agent has nothing useful to say about a
  capability it should never reach for.

A filtered tool is also refused if called by name, because a list is a hint and
not an enforcement point.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from autogen_core.tools import TextResultContent, ToolResult, Workbench

from . import hooks as hooks_module


class GatedWorkbench(Workbench):
    """Wraps a workbench and runs `before_tool_call` / `after_tool_call` around it."""

    component_type = "workbench"

    def __init__(
        self,
        inner: Workbench,
        *,
        registry: hooks_module.HookRegistry | None = None,
        session_id: str = "",
        label: str = "",
        allow: Sequence[str] | None = None,
    ) -> None:
        self._inner = inner
        self._registry = registry or hooks_module.REGISTRY
        self._session_id = session_id
        self._label = label or type(inner).__name__
        # None means "offer everything the source lists". A set means the prompt
        # only carries these — see the note on gating vs filtering above.
        self._allow = set(allow) if allow else None
        self.blocked: list[dict[str, Any]] = []

    # ------------------------------------------------------------ passthrough

    def offers(self, name: str) -> bool:
        return self._allow is None or name in self._allow

    async def list_tools(self) -> Sequence[Mapping[str, Any]]:
        tools = await self._inner.list_tools()
        if self._allow is None:
            return tools
        return [t for t in tools if str(t.get("name", "")) in self._allow]

    async def start(self) -> None:
        await self._inner.start()

    async def stop(self) -> None:
        await self._inner.stop()

    async def reset(self) -> None:
        await self._inner.reset()

    async def save_state(self) -> Mapping[str, Any]:
        return await self._inner.save_state()

    async def load_state(self, state: Mapping[str, Any]) -> None:
        await self._inner.load_state(state)

    # ------------------------------------------------------------ the gate

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        cancellation_token: Any = None,
        call_id: str | None = None,
    ) -> ToolResult:
        payload = {
            "tool": name,
            "arguments": dict(arguments or {}),
            "session": self._session_id,
            "workbench": self._label,
            "call_id": call_id,
        }

        if not self.offers(name):
            # A filtered tool is refused by name too. The list is a hint to the
            # model; this is the enforcement.
            self._record_block(name, arguments, "not offered by this workbench")
            return ToolResult(
                name=name,
                result=[TextResultContent(content=f"Refused: {name} is not available here.")],
                is_error=True,
            )

        decision = await self._registry.run(hooks_module.BEFORE_TOOL_CALL, payload)
        if decision.blocked:
            self.blocked.append({"tool": name, "reason": decision.reason, "by": decision.by})
            self._record_block(name, arguments, decision.reason)
            # An error result, not an exception: the agent needs to be able to
            # tell the person it was refused and why.
            return ToolResult(
                name=name,
                result=[TextResultContent(content=f"Refused: {decision.reason}")],
                is_error=True,
            )

        result = await self._inner.call_tool(name, arguments, cancellation_token, call_id)

        after = await self._registry.run(
            hooks_module.AFTER_TOOL_CALL,
            {**payload, "is_error": result.is_error, "result": result.to_text()[:2000]},
        )
        replacement = after.get("result")
        if isinstance(replacement, str):
            return ToolResult(
                name=name, result=[TextResultContent(content=replacement)], is_error=result.is_error
            )
        return result

    def _record_block(self, name: str, arguments: Mapping[str, Any] | None, reason: str) -> None:
        try:
            import policy as policy_module

            policy_module.DEFAULT.record_agent_action(
                tool=name,
                arguments=dict(arguments or {}),
                result_size=0,
                outcome="blocked",
                session=self._session_id or None,
            )
        except Exception:  # noqa: BLE001 — auditing must not break the refusal
            pass


def wrap(
    workbenches: Sequence[Workbench],
    *,
    session_id: str = "",
    registry: hooks_module.HookRegistry | None = None,
) -> list[Workbench]:
    return [
        GatedWorkbench(w, registry=registry, session_id=session_id)
        if not isinstance(w, GatedWorkbench)
        else w
        for w in workbenches
    ]


__all__ = ["GatedWorkbench", "wrap"]
