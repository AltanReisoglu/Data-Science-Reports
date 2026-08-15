"""Event capture and the intervention gate — the two core-runtime mechanisms."""

from __future__ import annotations

import logging
import unittest

from autogen_core import EVENT_LOGGER_NAME, DropMessage
from autogen_core.logging import LLMCallEvent, ToolCallEvent

import observability
import policy


def quiet_policy() -> policy.SourcePolicy:
    return policy.SourcePolicy(robots_fetcher=lambda _u: None, rate_limited=False)


class EventCaptureTest(unittest.TestCase):
    def test_llm_usage_is_accumulated(self) -> None:
        capture = observability.EventCapture(source_policy=quiet_policy())
        with capture:
            logger = logging.getLogger(EVENT_LOGGER_NAME)
            logger.info(LLMCallEvent(messages=[], response={}, prompt_tokens=100, completion_tokens=20))
            logger.info(LLMCallEvent(messages=[], response={}, prompt_tokens=50, completion_tokens=10))

        self.assertEqual(capture.totals.llm_calls, 2)
        self.assertEqual(capture.totals.prompt_tokens, 150)
        self.assertEqual(capture.totals.completion_tokens, 30)
        self.assertEqual(capture.totals.total_tokens, 180)

    def test_tool_calls_reach_the_audit_log(self) -> None:
        pol = quiet_policy()
        capture = observability.EventCapture(source_policy=pol)
        with capture:
            logging.getLogger(EVENT_LOGGER_NAME).info(
                ToolCallEvent(
                    tool_name="inspect_repository",
                    arguments={"full_name": "acme/db"},
                    result='{"stars": 900}',
                )
            )

        self.assertEqual(len(capture.totals.tool_calls), 1)
        # The in-memory capture keeps the arguments: it is scoped to one run and
        # is what the operator reads while the run is happening.
        self.assertEqual(capture.totals.tool_calls[0].tool, "inspect_repository")
        self.assertEqual(capture.totals.tool_calls[0].arguments, {"full_name": "acme/db"})

        # The audit ledger does not. It outlives the session and is unscoped, so
        # it records *which* call happened, not what was in it (docs/13 §10.2).
        record = pol.records[-1]
        self.assertEqual(record.source, "agent:inspect_repository")
        self.assertEqual(record.url, "")
        self.assertEqual(record.keys, ["full_name"])
        self.assertTrue(record.digest)
        self.assertNotIn("acme/db", str(record.__dict__))

    def test_handler_detaches_itself(self) -> None:
        logger = logging.getLogger(EVENT_LOGGER_NAME)
        before = len(logger.handlers)
        with observability.EventCapture(source_policy=quiet_policy()):
            self.assertEqual(len(logger.handlers), before + 1)
        self.assertEqual(len(logger.handlers), before)


class InterventionTest(unittest.IsolatedAsyncioTestCase):
    async def test_observer_mode_passes_messages_through(self) -> None:
        handler = observability.AuditingInterventionHandler(source_policy=quiet_policy())
        message = {"payload": 1}
        result = await handler.on_send(message, message_context=_ctx(), recipient="Scorer")
        self.assertIs(result, message)
        self.assertEqual(handler.dropped, [])
        self.assertEqual(len(handler.routed), 1)

    async def test_gate_drops_and_records(self) -> None:
        pol = quiet_policy()
        handler = observability.AuditingInterventionHandler(
            approve=lambda _sender, recipient, _msg: recipient != "Scorer",
            source_policy=pol,
        )
        allowed = await handler.on_send("ok", message_context=_ctx(), recipient="RiskAuditor")
        blocked = await handler.on_send("no", message_context=_ctx(), recipient="Scorer")

        self.assertEqual(allowed, "ok")
        self.assertIs(blocked, DropMessage, "the runtime drops it, not the agent")
        self.assertEqual(handler.dropped, [("runtime", "Scorer")])
        self.assertEqual(pol.records[-1].reason, "dropped_by_approval_gate")
        self.assertFalse(pol.records[-1].allowed)


class _ctx:
    sender = None


if __name__ == "__main__":
    unittest.main()
