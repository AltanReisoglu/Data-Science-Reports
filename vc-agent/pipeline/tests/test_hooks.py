"""Hook decision rules, quarantine, and the approval gate's failure direction."""

from __future__ import annotations

import unittest

import policy as policy_module
from gateway import approval, hooks


class DecisionRuleTests(unittest.IsolatedAsyncioTestCase):
    async def test_hooks_run_in_order_and_merge_their_updates(self) -> None:
        registry = hooks.HookRegistry()
        registry.register(hooks.BEFORE_PROMPT_BUILD, lambda p: {"a": 1}, name="first", order=1)
        registry.register(hooks.BEFORE_PROMPT_BUILD, lambda p: {"b": 2}, name="second", order=2)

        outcome = await registry.run(hooks.BEFORE_PROMPT_BUILD, {})
        self.assertEqual(outcome.ran, ["first", "second"])
        self.assertEqual(outcome.updates, {"a": 1, "b": 2})
        self.assertFalse(outcome.stopped)

    async def test_a_later_hook_sees_an_earlier_hooks_update(self) -> None:
        registry = hooks.HookRegistry()
        seen: dict = {}
        registry.register(hooks.BEFORE_PROMPT_BUILD, lambda p: {"model": "cheap"}, name="a", order=1)
        registry.register(hooks.BEFORE_PROMPT_BUILD, lambda p: seen.update(p) or {}, name="b", order=2)

        await registry.run(hooks.BEFORE_PROMPT_BUILD, {"session": "s1"})
        self.assertEqual(seen.get("model"), "cheap")
        self.assertEqual(seen.get("session"), "s1")

    async def test_block_on_before_tool_call_is_terminal(self) -> None:
        registry = hooks.HookRegistry()
        reached = []
        registry.register(
            hooks.BEFORE_TOOL_CALL,
            lambda p: {"block": True, "reason": "not allowed"},
            name="gate", order=1,
        )
        registry.register(
            hooks.BEFORE_TOOL_CALL, lambda p: reached.append(1) or {}, name="after", order=2
        )

        outcome = await registry.run(hooks.BEFORE_TOOL_CALL, {"tool": "x"})
        self.assertTrue(outcome.stopped)
        self.assertTrue(outcome.blocked)
        self.assertEqual(outcome.reason, "not allowed")
        self.assertEqual(outcome.by, "gate")
        self.assertEqual(reached, [], "a hook after the block must not run")

    async def test_cancel_on_message_sending_is_terminal(self) -> None:
        registry = hooks.HookRegistry()
        registry.register(hooks.MESSAGE_SENDING, lambda p: {"cancel": True}, name="stop")
        outcome = await registry.run(hooks.MESSAGE_SENDING, {})
        self.assertTrue(outcome.stopped)
        self.assertTrue(outcome.blocked)

    async def test_reply_on_before_agent_reply_takes_over_the_turn(self) -> None:
        registry = hooks.HookRegistry()
        registry.register(hooks.BEFORE_AGENT_REPLY, lambda p: {"reply": "canned"}, name="shortcut")
        outcome = await registry.run(hooks.BEFORE_AGENT_REPLY, {})
        self.assertTrue(outcome.stopped)
        self.assertEqual(outcome.get("reply"), "canned")
        # Taking over a turn is not the same as blocking an outbound call.
        self.assertFalse(outcome.blocked)

    async def test_block_is_ignored_at_a_point_that_has_no_terminal_rule(self) -> None:
        """Only the documented point/key pairs end a chain; the rest just merge."""
        registry = hooks.HookRegistry()
        registry.register(hooks.AFTER_TOOL_CALL, lambda p: {"block": True}, name="confused")
        outcome = await registry.run(hooks.AFTER_TOOL_CALL, {})
        self.assertFalse(outcome.stopped)

    async def test_async_hooks_are_awaited(self) -> None:
        registry = hooks.HookRegistry()

        async def slow(payload):
            return {"done": True}

        registry.register(hooks.AGENT_END, slow, name="slow")
        self.assertTrue((await registry.run(hooks.AGENT_END, {})).get("done"))

    async def test_unknown_point_is_rejected_at_registration(self) -> None:
        with self.assertRaises(ValueError):
            hooks.HookRegistry().register("before_lunch", lambda p: {})


class QuarantineTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_crashing_hook_is_skipped_and_the_chain_continues(self) -> None:
        registry = hooks.HookRegistry()

        def broken(payload):
            raise RuntimeError("bad line")

        registry.register(hooks.BEFORE_PROMPT_BUILD, broken, name="broken", order=1)
        registry.register(hooks.BEFORE_PROMPT_BUILD, lambda p: {"ok": True}, name="good", order=2)

        outcome = await registry.run(hooks.BEFORE_PROMPT_BUILD, {})
        self.assertEqual(outcome.failed, ["broken"])
        self.assertEqual(outcome.ran, ["good"])
        self.assertTrue(outcome.get("ok"), "the agent must not go silent")

    async def test_repeated_failures_quarantine_the_hook(self) -> None:
        registry = hooks.HookRegistry(failure_limit=2)
        calls = []

        def broken(payload):
            calls.append(1)
            raise RuntimeError("still bad")

        registry.register(hooks.AGENT_END, broken, name="broken")
        for _ in range(4):
            await registry.run(hooks.AGENT_END, {})

        self.assertEqual(len(calls), 2, "a quarantined hook stops being called")
        self.assertIn("broken", registry.quarantined())

    async def test_revive_brings_a_fixed_hook_back(self) -> None:
        registry = hooks.HookRegistry(failure_limit=1)
        registry.register(hooks.AGENT_END, lambda p: (_ for _ in ()).throw(ValueError()), name="b")
        await registry.run(hooks.AGENT_END, {})
        self.assertIn("b", registry.quarantined())

        self.assertEqual(registry.revive("b"), 1)
        self.assertEqual(registry.quarantined(), [])


class ApprovalGateTests(unittest.IsolatedAsyncioTestCase):
    def test_outbound_is_matched_by_substring_not_exact_name(self) -> None:
        """Tool names come from a remote server; an exact allowlist fails open."""
        self.assertTrue(approval.is_outbound("sessions_send"))
        self.assertTrue(approval.is_outbound("sessions_send_media"), "a renamed tool still trips")
        self.assertTrue(approval.is_outbound("sessions_spawn"))
        self.assertFalse(approval.is_outbound("sessions_list"))
        self.assertFalse(approval.is_outbound("messages_read"))
        self.assertFalse(approval.is_outbound("search_docs"))

    async def test_read_tools_pass_and_outbound_tools_block(self) -> None:
        gate = approval.ApprovalGate(allow_all=False)
        registry = hooks.HookRegistry()
        approval.install(registry, gate)

        allowed = await registry.run(hooks.BEFORE_TOOL_CALL, {"tool": "sessions_list"})
        self.assertFalse(allowed.blocked)

        denied = await registry.run(
            hooks.BEFORE_TOOL_CALL, {"tool": "sessions_send", "arguments": {"text": "hi"}}
        )
        self.assertTrue(denied.blocked)
        self.assertIn("approval", denied.reason)
        self.assertEqual(len(gate.pending()), 1)

    async def test_approval_covers_one_call_not_every_call_like_it(self) -> None:
        gate = approval.ApprovalGate(allow_all=False)
        registry = hooks.HookRegistry()
        approval.install(registry, gate)
        args = {"session": "telegram:me", "text": "three fintechs"}

        first = await registry.run(hooks.BEFORE_TOOL_CALL, {"tool": "sessions_send", "arguments": args})
        self.assertTrue(first.blocked)
        self.assertTrue(gate.approve(first.get("approval_id"))["ok"])

        second = await registry.run(hooks.BEFORE_TOOL_CALL, {"tool": "sessions_send", "arguments": args})
        self.assertFalse(second.blocked, "the approved call goes through")

        third = await registry.run(hooks.BEFORE_TOOL_CALL, {"tool": "sessions_send", "arguments": args})
        self.assertTrue(third.blocked, "and the next identical call needs its own approval")

    async def test_denial_does_not_grant(self) -> None:
        gate = approval.ApprovalGate(allow_all=False)
        registry = hooks.HookRegistry()
        approval.install(registry, gate)

        outcome = await registry.run(hooks.BEFORE_TOOL_CALL, {"tool": "sessions_send", "arguments": {}})
        gate.deny(outcome.get("approval_id"), note="not now")
        again = await registry.run(hooks.BEFORE_TOOL_CALL, {"tool": "sessions_send", "arguments": {}})
        self.assertTrue(again.blocked)

    async def test_a_broken_gate_blocks_rather_than_opens(self) -> None:
        """Quarantine fails open; a guard must fail closed. This is that asymmetry."""

        class Broken(approval.ApprovalGate):
            def check(self, *a, **k):
                raise RuntimeError("gate is broken")

        registry = hooks.HookRegistry()
        approval.install(registry, Broken(allow_all=False))

        outcome = await registry.run(hooks.BEFORE_TOOL_CALL, {"tool": "sessions_send"})
        self.assertTrue(outcome.blocked)
        self.assertIn("approval gate failed", outcome.reason)

    async def test_allow_all_is_an_explicit_operator_decision(self) -> None:
        gate = approval.ApprovalGate(allow_all=True)
        registry = hooks.HookRegistry()
        approval.install(registry, gate)
        outcome = await registry.run(hooks.BEFORE_TOOL_CALL, {"tool": "sessions_send"})
        self.assertFalse(outcome.blocked)


class AuditTests(unittest.TestCase):
    def test_argument_values_never_reach_the_audit_ledger(self) -> None:
        """The ledger outlives the session, so it holds metadata only (docs/13 §10.2)."""
        import tempfile
        from pathlib import Path

        path = Path(tempfile.mkdtemp(prefix="audit-")) / "audit.jsonl"
        gate = policy_module.SourcePolicy(audit_path=path, rate_limited=False)
        gate.record_agent_action(
            tool="sessions_send",
            arguments={"peer": "+905551234567", "text": "Acme raised a seed round"},
            result_size=12,
            agent="Analyst",
            session="agent:main:web:dm:alice",
        )

        written = path.read_text(encoding="utf-8")
        self.assertNotIn("905551234567", written)
        self.assertNotIn("Acme raised", written)
        # What it does keep: which call this was, and enough to compare two of them.
        self.assertIn("sessions_send", written)
        self.assertIn('"keys"', written)
        self.assertIn("peer", written)
        self.assertIn('"digest"', written)

    def test_the_digest_identifies_a_repeated_call(self) -> None:
        path_a = policy_module.SourcePolicy(rate_limited=False)
        path_a.record_agent_action(tool="t", arguments={"a": 1}, result_size=0)
        path_a.record_agent_action(tool="t", arguments={"a": 1}, result_size=0)
        path_a.record_agent_action(tool="t", arguments={"a": 2}, result_size=0)

        digests = [r.digest for r in path_a.records]
        self.assertEqual(digests[0], digests[1])
        self.assertNotEqual(digests[0], digests[2])

    def test_a_blocked_action_is_recorded_as_not_allowed(self) -> None:
        gate = policy_module.SourcePolicy(rate_limited=False)
        gate.record_agent_action(
            tool="sessions_send", arguments={}, result_size=0, outcome="blocked"
        )
        self.assertFalse(gate.records[-1].allowed)


if __name__ == "__main__":
    unittest.main()
