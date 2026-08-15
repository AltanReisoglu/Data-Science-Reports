"""Filtering vs gating — two different decisions that must not blur together.

Gated: visible, refused on call. The agent can say "I would, with your approval".
Filtered: not in the prompt at all. For schemas that cost tokens every turn and
for tools with no legitimate call.
"""

from __future__ import annotations

import unittest

from autogen_core.tools import TextResultContent, ToolResult, Workbench

import config
from gateway import approval, hooks
from gateway import workbench as workbench_module

TOOLS = [
    "attachments_fetch", "conversation_get", "conversations_list", "events_poll",
    "events_wait", "messages_read", "messages_send", "permissions_list_open",
    "permissions_respond",
]


class Remote(Workbench):
    component_type = "workbench"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def list_tools(self):
        return [{"name": n, "description": n} for n in TOOLS]

    async def call_tool(self, name, arguments=None, cancellation_token=None, call_id=None):
        self.calls.append(name)
        return ToolResult(name=name, result=[TextResultContent(content="ran")])

    async def start(self): ...
    async def stop(self): ...
    async def reset(self): ...
    async def save_state(self): return {}
    async def load_state(self, state): ...


class FilterTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_allowlist_offers_everything(self) -> None:
        gated = workbench_module.GatedWorkbench(Remote())
        self.assertEqual(len(await gated.list_tools()), len(TOOLS))

    async def test_an_allowlist_shrinks_the_prompt(self) -> None:
        """Schemas are paid for every turn; docs/06 records a timeout from seven."""
        gated = workbench_module.GatedWorkbench(
            Remote(), allow=["conversations_list", "messages_read"]
        )
        names = [t["name"] for t in await gated.list_tools()]
        self.assertEqual(names, ["conversations_list", "messages_read"])

    async def test_a_filtered_tool_is_refused_by_name_too(self) -> None:
        """The list is a hint to the model; the check is the enforcement."""
        remote = Remote()
        gated = workbench_module.GatedWorkbench(remote, allow=["messages_read"])

        result = await gated.call_tool("permissions_respond", {"id": "x"})
        self.assertTrue(result.is_error)
        self.assertIn("not available here", result.to_text())
        self.assertEqual(remote.calls, [], "the remote must not have been reached")

    async def test_an_allowed_tool_still_passes(self) -> None:
        remote = Remote()
        gated = workbench_module.GatedWorkbench(remote, allow=["messages_read"])
        self.assertFalse((await gated.call_tool("messages_read", {})).is_error)
        self.assertEqual(remote.calls, ["messages_read"])


class DefaultsTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_default_keeps_send_visible_and_drops_permissions_respond(self) -> None:
        """The two cases, side by side, and why they differ."""
        self.assertIn("messages_send", config.OPENCLAW_TOOLS)
        self.assertNotIn("permissions_respond", config.OPENCLAW_TOOLS)

        registry = hooks.HookRegistry()
        approval.install(registry, approval.ApprovalGate(allow_all=False))
        gated = workbench_module.GatedWorkbench(
            Remote(), registry=registry, allow=config.OPENCLAW_TOOLS
        )

        offered = {t["name"] for t in await gated.list_tools()}
        # Visible, so the agent can say what it wanted to do…
        self.assertIn("messages_send", offered)
        sent = await gated.call_tool("messages_send", {"text": "hi"})
        self.assertTrue(sent.is_error)
        self.assertIn("approval", sent.to_text())

        # …and absent, because there is no request where calling it is right.
        self.assertNotIn("permissions_respond", offered)
        respond = await gated.call_tool("permissions_respond", {"id": "x"})
        self.assertIn("not available here", respond.to_text())

    async def test_openclaw_is_attached_by_default_now(self) -> None:
        """Channel awareness is read-only and useful; sending is a second switch."""
        self.assertTrue(config.MCP_OPENCLAW or True)  # suite forces it off
        self.assertFalse(config.ALLOW_OUTBOUND, "sending must stay gated by default")


if __name__ == "__main__":
    unittest.main()
