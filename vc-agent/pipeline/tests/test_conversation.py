"""The live conversation: memory, tools, state, cancellation.

Driven by a scripted replay client, so these run offline. What they lock in is
the wiring — that the agent has a context to remember with, that its tools are
reachable through a workbench, and that a turn survives a restart.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from autogen_core import FunctionCall
from autogen_core.models import CreateResult, RequestUsage
from autogen_ext.models.replay import ReplayChatCompletionClient

import config
import conversation
import engine
from gateway import sessions

SCAN = {
    "query": "ai infra",
    "days": 7,
    "mode": "dry",
    "thesis_is_placeholder": True,
    "failed_sources": {},
    "cost": {},
    "funnel": {"signals": 10},
    "candidates": [
        {
            "company": {"name": "Argonix", "domain": "argonix.io", "sectors": [], "signals": []},
            "branches": [{"branch": "team", "succeeded": False, "error": "no result"}],
            "score": {
                "thesis_fit": 3, "team": 1, "momentum": 3, "technical_depth": 3, "timing": 3,
                "rationale": {}, "missing_data": ["team branch produced no result"],
                "decision": "watch",
            },
            "memo": None,
        }
    ],
}


def tool_call(name: str, arguments: str = "{}") -> CreateResult:
    return CreateResult(
        finish_reason="function_calls",
        content=[FunctionCall(id="c1", name=name, arguments=arguments)],
        usage=RequestUsage(prompt_tokens=30, completion_tokens=6),
        cached=False,
    )


class ConversationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._mcp_was = config.MCP_DEEPWIKI
        config.MCP_DEEPWIKI = False  # these tests stay offline
        self._client_was = engine.Ledger.raw_client
        self._session = "agent:main:test:dm:local"
        self._store = sessions.SessionStore(Path(tempfile.mkdtemp(prefix="conv-")))

    async def asyncTearDown(self) -> None:
        engine.Ledger.raw_client = self._client_was
        config.MCP_DEEPWIKI = self._mcp_was

    def _scripted(self, script):
        client = ReplayChatCompletionClient(script, model_info=engine.DRY_MODEL_INFO)
        engine.Ledger.raw_client = lambda self, tier, s=None: client
        return client

    def _conv(self, scan):
        return conversation.Conversation(
            lambda: scan, lambda q, d: None,
            session_id=self._session, store=self._store,
        )

    async def _run(self, conv, question):
        return [event async for event in conv.stream(question)]

    async def test_tool_call_reaches_the_agent_through_the_workbench(self) -> None:
        # `tools=` is rejected alongside `workbench=`, so local functions go in
        # via StaticWorkbench. This asserts they are still callable.
        self._scripted([tool_call("company_detail", '{"name": "Argonix"}'), "Argonix scored 13."])
        conv = self._conv(SCAN)
        try:
            events = await self._run(conv, "tell me about Argonix")
        finally:
            await conv.close()

        called = [e for e in events if e["type"] == "tool"]
        results = [e for e in events if e["type"] == "tool_result"]
        self.assertEqual(called[0]["name"], "company_detail")
        self.assertIn("Argonix", results[0]["preview"])
        self.assertTrue(any(e["type"] == "done" for e in events))

    async def test_memory_carries_between_turns(self) -> None:
        self._scripted(["First answer.", "Second answer."])
        conv = self._conv(SCAN)
        try:
            await self._run(conv, "what did the scan find?")
            await self._run(conv, "and what did I just ask?")
            state = await conv._agent.save_state()
        finally:
            await conv.close()

        messages = state.get("llm_context", {}).get("messages", [])
        # Two turns in, the second call must have been able to see the first.
        self.assertGreaterEqual(len(messages), 4)

    async def test_state_is_written_and_reloaded(self) -> None:
        self._scripted(["Answer one.", "Answer two."])
        conv = self._conv(SCAN)
        try:
            await self._run(conv, "hello")
            self.assertTrue(conv.state_path.exists(), "a turn must persist the context")
            await conv.close()

            revived = self._conv(SCAN)
            await revived.ensure()
            messages = (await revived._agent.save_state()).get("llm_context", {}).get("messages", [])
            self.assertGreaterEqual(len(messages), 2, "a restart must not lose the conversation")
        finally:
            await conv.close()

    async def test_reset_clears_the_transcript(self) -> None:
        self._scripted(["Answer."])
        conv = self._conv(SCAN)
        await self._run(conv, "hello")
        self.assertTrue(conv.state_path.exists())
        await conv.reset()
        self.assertFalse(conv.state_path.exists())

    async def test_cancel_is_false_when_nothing_is_running(self) -> None:
        conv = self._conv(SCAN)
        self.assertFalse(conv.cancel())

    async def test_unknown_company_is_answered_with_what_is_known(self) -> None:
        # The tool must not invent a company; it names the ones that exist.
        self._scripted([tool_call("company_detail", '{"name": "Nonesuch"}'), "No such candidate."])
        conv = self._conv(SCAN)
        try:
            events = await self._run(conv, "tell me about Nonesuch")
        finally:
            await conv.close()
        preview = [e for e in events if e["type"] == "tool_result"][0]["preview"]
        self.assertIn("No candidate named", preview)
        self.assertIn("Argonix", preview)


if __name__ == "__main__":
    unittest.main()
