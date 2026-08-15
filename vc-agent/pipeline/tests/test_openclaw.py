"""The outbound bridge: attachment that fails soft, and a gate over real tool names.

The tool names here are not invented — they are what `openclaw mcp serve` listed
on 2026-08-14 against openclaw 2026.7.1-2. That matters, because the first draft
of this module guessed `sessions_send` and `sessions_spawn`, and the real surface
has neither. It has `permissions_respond`, which is the more dangerous one and
which a guessed allowlist would have let straight through.
"""

from __future__ import annotations

import unittest

import config
import openclaw
from gateway import approval, hooks

# Observed, not assumed.
LIVE_TOOLS = [
    "attachments_fetch",
    "conversation_get",
    "conversations_list",
    "events_poll",
    "events_wait",
    "messages_read",
    "messages_send",
    "permissions_list_open",
    "permissions_respond",
]


class GateCoverageTests(unittest.TestCase):
    def test_reading_channel_state_is_free(self) -> None:
        for name in openclaw.READ_TOOLS:
            with self.subTest(tool=name):
                self.assertFalse(approval.is_outbound(name))

    def test_sending_a_message_is_gated(self) -> None:
        self.assertTrue(approval.is_outbound("messages_send"))

    def test_answering_openclaws_own_permission_prompts_is_gated(self) -> None:
        """Otherwise our agent can approve OpenClaw's requests for the operator."""
        self.assertTrue(approval.is_outbound("permissions_respond"))
        self.assertFalse(
            approval.is_outbound("permissions_list_open"),
            "listing pending permissions is reading, not deciding",
        )

    def test_every_live_tool_is_classified_one_way_or_the_other(self) -> None:
        """No tool from the real surface falls between the two lists unexamined."""
        known = set(openclaw.READ_TOOLS) | set(openclaw.WRITE_TOOLS)
        self.assertEqual(set(LIVE_TOOLS) - known, set())

    def test_the_write_tools_are_exactly_the_gated_ones(self) -> None:
        gated = {name for name in LIVE_TOOLS if approval.is_outbound(name)}
        self.assertEqual(gated, set(openclaw.WRITE_TOOLS))


class AttachmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_by_default_and_says_why(self) -> None:
        """Attaching hands the agent a tool that can message a person. Two switches."""
        was, config.MCP_OPENCLAW = config.MCP_OPENCLAW, False
        try:
            attachment = await openclaw.attach()
        finally:
            config.MCP_OPENCLAW = was

        self.assertFalse(attachment.attached)
        self.assertIn("disabled", attachment.status)

    async def test_a_missing_binary_is_a_status_not_a_crash(self) -> None:
        was_flag, config.MCP_OPENCLAW = config.MCP_OPENCLAW, True
        was_bin, config.OPENCLAW_BIN = config.OPENCLAW_BIN, "openclaw-does-not-exist"
        try:
            attachment = await openclaw.attach()
        finally:
            config.MCP_OPENCLAW, config.OPENCLAW_BIN = was_flag, was_bin

        self.assertFalse(attachment.attached)
        self.assertIn("not on PATH", attachment.status)

    def test_guidance_is_empty_when_nothing_is_attached(self) -> None:
        self.assertEqual(openclaw.guidance(openclaw.Attachment()), "")

    def test_guidance_names_the_gated_tools(self) -> None:
        """An agent that does not know it will be refused retries instead of asking."""
        attachment = openclaw.Attachment(workbench=object(), status="connected", tools=LIVE_TOOLS)
        was, config.ALLOW_OUTBOUND = config.ALLOW_OUTBOUND, False
        try:
            text = openclaw.guidance(attachment)
        finally:
            config.ALLOW_OUTBOUND = was

        self.assertIn("messages_send", text)
        self.assertIn("permissions_respond", text)
        self.assertIn("do not retry", text)

    def test_guidance_drops_the_warning_when_the_gate_is_open(self) -> None:
        attachment = openclaw.Attachment(workbench=object(), status="connected", tools=LIVE_TOOLS)
        was, config.ALLOW_OUTBOUND = config.ALLOW_OUTBOUND, True
        try:
            text = openclaw.guidance(attachment)
        finally:
            config.ALLOW_OUTBOUND = was

        self.assertNotIn("will be refused", text)


class WorkbenchGateTests(unittest.IsolatedAsyncioTestCase):
    """The gate has to sit where remote tools pass, not on a list of local ones."""

    async def test_a_remote_send_is_refused_as_a_result_not_an_exception(self) -> None:
        from autogen_core.tools import TextResultContent, ToolResult, Workbench

        from gateway import workbench as workbench_module

        class FakeRemote(Workbench):
            component_type = "workbench"

            def __init__(self) -> None:
                self.calls: list[str] = []

            async def list_tools(self):
                return [{"name": n} for n in LIVE_TOOLS]

            async def call_tool(self, name, arguments=None, cancellation_token=None, call_id=None):
                self.calls.append(name)
                return ToolResult(name=name, result=[TextResultContent(content="sent")])

            async def start(self): ...
            async def stop(self): ...
            async def reset(self): ...
            async def save_state(self): return {}
            async def load_state(self, state): ...

        registry = hooks.HookRegistry()
        approval.install(registry, approval.ApprovalGate(allow_all=False))
        remote = FakeRemote()
        gated = workbench_module.GatedWorkbench(remote, registry=registry, session_id="s1")

        blocked = await gated.call_tool("messages_send", {"text": "hello"})
        self.assertTrue(blocked.is_error)
        self.assertIn("Refused", blocked.to_text())
        self.assertEqual(remote.calls, [], "the remote must not have been reached")

        allowed = await gated.call_tool("messages_read", {"peer": "x"})
        self.assertFalse(allowed.is_error)
        self.assertEqual(remote.calls, ["messages_read"])

    async def test_listing_tools_is_never_gated(self) -> None:
        """Hiding a tool makes the model guess; refusing it makes the model ask."""
        from autogen_core.tools import StaticWorkbench

        from gateway import workbench as workbench_module

        registry = hooks.HookRegistry()
        approval.install(registry, approval.ApprovalGate(allow_all=False))
        gated = workbench_module.GatedWorkbench(StaticWorkbench([]), registry=registry)
        self.assertEqual(list(await gated.list_tools()), [])


if __name__ == "__main__":
    unittest.main()
