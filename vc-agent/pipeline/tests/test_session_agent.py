"""Sessions on the core runtime.

The claim being tested is the one docs/14 §3.6 rests on: **a topic's source
becomes an agent's key**. If that holds, one isolated agent per conversation is
the runtime's own mechanism and not something we maintain. If it does not, the
whole move to core buys nothing and should be reverted.

The rest of the file is the failure discipline `06 §8` forced: a crashing handler
must not take other sessions with it, and the transcript must survive a crash.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from autogen_core import SingleThreadedAgentRuntime, TopicId

import channels as channels_module
from gateway import runtime as runtime_module
from gateway import sessions


def store() -> sessions.SessionStore:
    return sessions.SessionStore(Path(tempfile.mkdtemp(prefix="sa-")))


class AddressingTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_topic_source_becomes_the_agent_key(self) -> None:
        """The mechanism the whole move depends on, asserted rather than assumed."""
        seen: list[str] = []

        async def responder(session_id: str, turn: sessions.Turn) -> str:
            seen.append(session_id)
            return f"answered {turn.text}"

        runtime = SingleThreadedAgentRuntime(ignore_unhandled_exceptions=True)
        await sessions.register_sessions(runtime, store(), responder)
        runtime.start()
        try:
            session_id = "agent:main:web:dm:alice"
            await runtime.publish_message(
                sessions.Turn(text="hello"), topic_id=TopicId(sessions.TURN_TOPIC, session_id)
            )
            await asyncio.sleep(0.15)
        finally:
            await runtime.stop()

        # The agent that ran was keyed by the topic source — no dictionary, no
        # registry, no lookup written by us.
        self.assertEqual(seen, ["agent:main:web:dm:alice"])

    async def test_two_sessions_are_two_isolated_agents(self) -> None:
        instances: list[int] = []
        calls: list[str] = []

        class Counting(sessions.SessionAgent):
            def __init__(self, *a, **k) -> None:
                super().__init__(*a, **k)
                instances.append(id(self))

        async def responder(session_id: str, turn: sessions.Turn) -> str:
            calls.append(session_id)
            return "ok"

        keeper = store()
        runtime = SingleThreadedAgentRuntime(ignore_unhandled_exceptions=True)
        await Counting.register(runtime, "session", lambda: Counting(keeper, responder))
        from autogen_core import TypeSubscription

        await runtime.add_subscription(
            TypeSubscription(topic_type=sessions.TURN_TOPIC, agent_type="session")
        )
        runtime.start()
        try:
            for peer in ("alice", "bob"):
                await runtime.publish_message(
                    sessions.Turn(text="hi"),
                    topic_id=TopicId(sessions.TURN_TOPIC, f"agent:main:web:dm:{peer}"),
                )
            await asyncio.sleep(0.2)
        finally:
            await runtime.stop()

        self.assertEqual(len(instances), 2, "one agent instance per session key")
        self.assertEqual(sorted(calls), [
            "agent:main:web:dm:alice", "agent:main:web:dm:bob",
        ])

    async def test_a_session_id_is_an_acceptable_agent_key(self) -> None:
        """Our ids carry colons; CloudEvents allows them and so must the runtime."""
        key = sessions.resolve("web", peer="alice").as_id()
        self.assertRegex(key, r"^[\w\-\.\:\=]+$")


class FailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_crashing_turn_does_not_stop_other_sessions(self) -> None:
        """`06 §8`: one handler's exception must not take siblings' work with it."""
        answered: list[str] = []

        async def responder(session_id: str, turn: sessions.Turn) -> str:
            if "bad" in session_id:
                raise RuntimeError("this session is broken")
            answered.append(session_id)
            return "fine"

        keeper = store()
        runtime = SingleThreadedAgentRuntime(ignore_unhandled_exceptions=True)
        await sessions.register_sessions(runtime, keeper, responder)
        runtime.start()
        try:
            for peer in ("bad", "good"):
                await runtime.publish_message(
                    sessions.Turn(text="hi"),
                    topic_id=TopicId(sessions.TURN_TOPIC, f"agent:main:web:dm:{peer}"),
                )
            await asyncio.sleep(0.2)
        finally:
            await runtime.stop()

        self.assertEqual(answered, ["agent:main:web:dm:good"])

    async def test_a_crash_is_recorded_rather_than_lost(self) -> None:
        async def responder(session_id: str, turn: sessions.Turn) -> str:
            raise ValueError("model refused")

        keeper = store()
        runtime = SingleThreadedAgentRuntime(ignore_unhandled_exceptions=True)
        await sessions.register_sessions(runtime, keeper, responder)
        runtime.start()
        session_id = "agent:main:web:dm:alice"
        try:
            await runtime.publish_message(
                sessions.Turn(text="what happened"),
                topic_id=TopicId(sessions.TURN_TOPIC, session_id),
            )
            await asyncio.sleep(0.15)
        finally:
            await runtime.stop()

        entries = keeper.read(session_id)
        roles = [e.get("role") for e in entries if "role" in e]
        # The question survived the crash; only the answer was lost.
        self.assertIn("user", roles)
        self.assertIn("what happened", str(entries))
        self.assertTrue(any("model refused" in str(e.get("error", "")) for e in entries))

    async def test_the_question_is_written_before_the_responder_runs(self) -> None:
        """A crash must cost the answer, not the record of what was asked."""
        keeper = store()
        session_id = "agent:main:web:dm:alice"
        during: list[int] = []

        async def responder(sid: str, turn: sessions.Turn) -> str:
            during.append(len(keeper.read(sid)))
            return "ok"

        runtime = SingleThreadedAgentRuntime(ignore_unhandled_exceptions=True)
        await sessions.register_sessions(runtime, keeper, responder)
        runtime.start()
        try:
            await runtime.publish_message(
                sessions.Turn(text="hello"), topic_id=TopicId(sessions.TURN_TOPIC, session_id)
            )
            await asyncio.sleep(0.15)
        finally:
            await runtime.stop()

        self.assertEqual(during, [1], "the user turn was already on disk")


class ChannelRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_reply_reaches_only_its_own_channel(self) -> None:
        web = channels_module.WebChannel()
        other = channels_module.WebChannel()
        other.name = "cli"

        runtime = SingleThreadedAgentRuntime(ignore_unhandled_exceptions=True)
        await channels_module.register_channel(runtime, web)
        await channels_module.register_channel(runtime, other)
        runtime.start()
        try:
            await runtime.publish_message(
                sessions.Reply(text="for the dashboard", session="s1"),
                topic_id=TopicId(f"{sessions.REPLY_TOPIC}.web", "peer-1"),
            )
            await asyncio.sleep(0.15)
        finally:
            await runtime.stop()

        self.assertEqual(web.waiting("peer-1"), 1)
        self.assertEqual(other.waiting("peer-1"), 0)

    async def test_a_turn_is_answered_and_delivered_end_to_end(self) -> None:
        web = channels_module.WebChannel()

        async def responder(session_id: str, turn: sessions.Turn) -> str:
            return f"you said: {turn.text}"

        runtime = SingleThreadedAgentRuntime(ignore_unhandled_exceptions=True)
        await sessions.register_sessions(runtime, store(), responder)
        await channels_module.register_channel(runtime, web)
        runtime.start()
        session_id = "agent:main:web:dm:alice"
        try:
            await runtime.publish_message(
                sessions.Turn(text="hello", reply_to="web"),
                topic_id=TopicId(sessions.TURN_TOPIC, session_id),
            )
            await asyncio.sleep(0.25)
        finally:
            await runtime.stop()

        queued = web.drain(session_id)
        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0]["text"], "you said: hello")

    async def test_a_failing_channel_does_not_raise_into_the_runtime(self) -> None:
        class Broken:
            name = "broken"

            def may_send_to(self, peer: str) -> bool:
                return True

            async def send(self, message):
                raise RuntimeError("channel is down")

        runtime = SingleThreadedAgentRuntime(ignore_unhandled_exceptions=True)
        await channels_module.register_channel(runtime, Broken())
        runtime.start()
        try:
            await runtime.publish_message(
                sessions.Reply(text="hi", session="s1"),
                topic_id=TopicId(f"{sessions.REPLY_TOPIC}.broken", "peer-1"),
            )
            await asyncio.sleep(0.15)
        finally:
            await runtime.stop()
        # Reaching here without an exception is the assertion.


class GatewayRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_registering_twice_is_a_no_op(self) -> None:
        gateway = runtime_module.GatewayRuntime()
        calls = []

        async def register(rt):
            calls.append(1)

        self.assertTrue(await gateway.register_once("session", register))
        self.assertFalse(await gateway.register_once("session", register))
        self.assertEqual(len(calls), 1)
        await gateway.close()

    async def test_start_is_idempotent_and_reports_state(self) -> None:
        gateway = runtime_module.GatewayRuntime()
        await gateway.start()
        await gateway.start()
        self.assertTrue(gateway.report()["running"])
        await gateway.close()
        self.assertFalse(gateway.report()["running"])

    async def test_the_intervention_handler_is_attached(self) -> None:
        """One audit surface for enrichment and control-plane traffic alike."""
        gateway = runtime_module.GatewayRuntime()
        gateway.build()
        self.assertIsNotNone(gateway.handler)
        await gateway.close()


if __name__ == "__main__":
    unittest.main()
