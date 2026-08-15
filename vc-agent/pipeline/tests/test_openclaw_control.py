"""Full control of OpenClaw, and the hole a name-based gate leaves.

The approval gate matches tool *names*. That works for `messages_send`, whose
name says what it does. It does not work for `openclaw_call`, which is one name
covering a hundred Gateway methods with its blast radius in an argument — so
`openclaw_call("sessions.reset")` slipped through until this gate existed.
"""

from __future__ import annotations

import unittest

import openclaw_control as oc
from gateway import approval, hooks


class ClassificationTests(unittest.TestCase):
    def test_reads_are_free(self) -> None:
        for method in ("health", "status", "sessions.list", "cron.list", "config.get"):
            with self.subTest(method=method):
                self.assertEqual(oc.classify(method), "read")

    def test_state_changes_are_gated(self) -> None:
        for method in ("sessions.reset", "cron.create", "chat.send", "node.invoke"):
            with self.subTest(method=method):
                self.assertEqual(oc.classify(method), "write")

    def test_credentials_config_and_exec_approvals_are_forbidden(self) -> None:
        for method in (
            "config.set", "config.patch", "secrets.get", "update.run",
            "exec.approval.resolve", "channels.logout", "sessions.delete",
            "devices.pair",
        ):
            with self.subTest(method=method):
                self.assertEqual(oc.classify(method), "forbidden")

    def test_an_unknown_method_is_gated_not_free(self) -> None:
        """A method OpenClaw adds next release must not be free by default."""
        self.assertEqual(oc.classify("some.future.method"), "write")
        self.assertEqual(oc.classify(""), "forbidden")

    def test_config_get_is_readable_but_config_set_is_not(self) -> None:
        """Prefix matching must not be so broad it swallows the read half."""
        self.assertEqual(oc.classify("config.get"), "read")
        self.assertEqual(oc.classify("config.schema"), "read")
        self.assertEqual(oc.classify("config.set"), "forbidden")

    def test_exec_approval_list_reads_but_resolve_does_not(self) -> None:
        self.assertEqual(oc.classify("exec.approval.list"), "read")
        self.assertEqual(oc.classify("exec.approval.resolve"), "forbidden")


class GateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.registry = hooks.HookRegistry()
        self.gate = approval.ApprovalGate(allow_all=False)
        approval.install(self.registry, self.gate)
        oc.install_gate(self.registry, self.gate)

    async def run_call(self, method: str, **args):
        return await self.registry.run(
            hooks.BEFORE_TOOL_CALL,
            {"tool": "openclaw_call", "arguments": {"method": method, **args}},
        )

    async def test_a_read_method_passes(self) -> None:
        self.assertFalse((await self.run_call("sessions.list")).blocked)

    async def test_a_write_method_is_gated_even_though_the_tool_name_is_harmless(self) -> None:
        """The hole this file exists for: `openclaw_call` matches no outbound marker."""
        self.assertFalse(approval.is_outbound("openclaw_call"))

        outcome = await self.run_call("sessions.reset", params_json='{"id": "x"}')
        self.assertTrue(outcome.blocked)
        self.assertIn("sessions.reset", outcome.reason)
        self.assertEqual(len(self.gate.pending()), 1)

    async def test_a_forbidden_method_has_no_approval_path(self) -> None:
        outcome = await self.run_call("config.set")
        self.assertTrue(outcome.blocked)
        self.assertIn("no per-call approval", outcome.reason)
        self.assertEqual(self.gate.pending(), [], "there must be nothing to approve")

    async def test_approving_a_write_lets_that_one_call_through(self) -> None:
        first = await self.run_call("cron.create", params_json='{"id": "watch"}')
        self.assertTrue(first.blocked)
        self.assertTrue(self.gate.approve(first.get("approval_id"))["ok"])

        second = await self.run_call("cron.create", params_json='{"id": "watch"}')
        self.assertFalse(second.blocked)

        third = await self.run_call("cron.create", params_json='{"id": "watch"}')
        self.assertTrue(third.blocked, "approval covers one call, not the method")

    async def test_other_tools_are_untouched_by_this_hook(self) -> None:
        outcome = await self.registry.run(
            hooks.BEFORE_TOOL_CALL, {"tool": "query_companies", "arguments": {}}
        )
        self.assertFalse(outcome.blocked)

    async def test_a_broken_gate_blocks(self) -> None:
        def explode(method: str) -> str:
            raise RuntimeError("classifier is broken")

        was, oc.classify = oc.classify, explode
        try:
            outcome = await self.run_call("sessions.list")
        finally:
            oc.classify = was
        self.assertTrue(outcome.blocked)
        self.assertIn("openclaw gate failed", outcome.reason)


class CallTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_forbidden_method_never_reaches_the_cli(self) -> None:
        """Belt and braces: the gate blocks, and `call` refuses independently."""
        outcome = await oc.call("secrets.get", {"name": "x"})
        self.assertFalse(outcome["ok"])
        self.assertEqual(outcome["tier"], "forbidden")
        self.assertIn("VC_OPENCLAW_ALLOW_ADMIN", outcome["error"])

    async def test_a_missing_binary_is_reported_not_raised(self) -> None:
        import config

        was, config.OPENCLAW_BIN = config.OPENCLAW_BIN, "openclaw-does-not-exist"
        try:
            outcome = await oc.call("health")
        finally:
            config.OPENCLAW_BIN = was
        self.assertFalse(outcome["ok"])
        self.assertIn("not on PATH", outcome["error"])

    def test_the_method_listing_shows_what_is_refused(self) -> None:
        listing = oc.methods()
        self.assertIn("sessions.list", listing["read"])
        self.assertIn("cron.create", listing["write"])
        self.assertIn("secrets.", listing["forbidden_prefixes"])
        self.assertFalse(listing["admin_enabled"])


if __name__ == "__main__":
    unittest.main()
