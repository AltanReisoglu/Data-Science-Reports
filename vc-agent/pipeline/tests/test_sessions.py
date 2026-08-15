"""Session routing, lifecycle and lanes — the rules the rest of the gateway trusts."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config
from gateway import runs as runs_module
from gateway import sessions


def _store() -> sessions.SessionStore:
    return sessions.SessionStore(Path(tempfile.mkdtemp(prefix="sessions-")))


def _manager(**policy) -> sessions.SessionManager:
    settings = config.SessionPolicy(**policy) if policy else config.SessionPolicy()
    return sessions.SessionManager(store=_store(), policy=settings)


class RoutingTests(unittest.TestCase):
    def test_dm_scope_main_shares_one_session(self) -> None:
        a = sessions.resolve("web", peer="alice", dm_scope="main")
        b = sessions.resolve("mcp", peer="bob", dm_scope="main")
        self.assertEqual(a.as_id(), b.as_id())

    def test_default_scope_separates_people_and_channels(self) -> None:
        """The reason the default is not `main`: two people must not share context."""
        alice = sessions.resolve("web", peer="alice")
        bob = sessions.resolve("web", peer="bob")
        alice_elsewhere = sessions.resolve("mcp", peer="alice")

        self.assertNotEqual(alice.as_id(), bob.as_id())
        self.assertNotEqual(alice.as_id(), alice_elsewhere.as_id())

    def test_per_peer_scope_merges_channels_but_not_people(self) -> None:
        web = sessions.resolve("web", peer="alice", dm_scope="per-peer")
        mcp = sessions.resolve("mcp", peer="alice", dm_scope="per-peer")
        other = sessions.resolve("web", peer="bob", dm_scope="per-peer")

        self.assertEqual(web.as_id(), mcp.as_id())
        self.assertNotEqual(web.as_id(), other.as_id())

    def test_account_scope_separates_two_accounts_on_one_channel(self) -> None:
        personal = sessions.resolve(
            "mcp", peer="alice", account="personal", dm_scope="per-account-channel-peer"
        )
        business = sessions.resolve(
            "mcp", peer="alice", account="biz", dm_scope="per-account-channel-peer"
        )
        self.assertNotEqual(personal.as_id(), business.as_id())

    def test_groups_stay_shared_even_under_the_narrowest_scope(self) -> None:
        """Splitting a group per speaker would give each person half a conversation."""
        first = sessions.resolve(
            "mcp", peer="room-7", kind="group", dm_scope="per-account-channel-peer"
        )
        second = sessions.resolve("mcp", peer="room-7", kind="group", dm_scope="main")
        self.assertEqual(first.as_id(), second.as_id())

    def test_unknown_scope_falls_back_to_the_safe_one(self) -> None:
        loose = sessions.resolve("web", peer="alice", dm_scope="nonsense")
        strict = sessions.resolve("web", peer="alice", dm_scope="per-channel-peer")
        self.assertEqual(loose.as_id(), strict.as_id())

    def test_cron_is_ephemeral(self) -> None:
        self.assertTrue(sessions.resolve("cron", peer="watchlist").ephemeral)
        self.assertFalse(sessions.resolve("web", peer="alice").ephemeral)

    def test_peer_identifiers_cannot_escape_the_session_directory(self) -> None:
        """A peer id arrives from outside; it must not become a path."""
        key = sessions.resolve("mcp", peer="../../etc/passwd")
        self.assertNotIn("/", key.as_id())
        self.assertNotIn("..", key.as_id())


class LifecycleTests(unittest.TestCase):
    def test_cron_never_reuses_a_session(self) -> None:
        manager = _manager()
        key = sessions.resolve("cron", peer="watchlist")
        first = manager.open(key)
        second = manager.open(key)
        self.assertNotEqual(first.id, second.id)

    def test_same_origin_reuses_one_session(self) -> None:
        manager = _manager()
        first = manager.route("web", peer="alice")
        second = manager.route("web", peer="alice")
        self.assertEqual(first.id, second.id)

    def test_idle_expiry_starts_a_new_session(self) -> None:
        manager = _manager(idle_minutes=30, daily_reset_hour=None)
        record = manager.route("web", peer="alice")
        record.last_interaction_at = (
            datetime.now(timezone.utc) - timedelta(hours=2)
        ).isoformat()

        again = manager.route("web", peer="alice")
        self.assertEqual(again.turns, 0)
        self.assertGreater(again.session_started_at, record.session_started_at)

    def test_system_traffic_does_not_keep_a_session_alive(self) -> None:
        """Only a real turn calls `touch`; reading the session must not reset idle."""
        manager = _manager(idle_minutes=30, daily_reset_hour=None)
        record = manager.route("web", peer="alice")
        stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        record.last_interaction_at = stale

        manager.list()
        manager.get(record.id)
        self.assertEqual(record.last_interaction_at, stale)

    def test_daily_reset_fires_for_a_session_started_before_the_boundary(self) -> None:
        manager = _manager(idle_minutes=0, daily_reset_hour=4)
        record = manager.route("web", peer="alice")
        record.session_started_at = (
            datetime.now(timezone.utc) - timedelta(days=3)
        ).isoformat()
        record.last_interaction_at = datetime.now(timezone.utc).isoformat()

        again = manager.route("web", peer="alice")
        self.assertEqual(again.turns, 0)

    def test_turns_are_recorded_to_the_transcript(self) -> None:
        manager = _manager()
        record = manager.route("web", peer="alice")
        manager.record_turn(record, "user", "who raised this week?")
        manager.record_turn(record, "assistant", "three companies")

        lines = manager.store.read(record.id)
        roles = [entry.get("role") for entry in lines if "role" in entry]
        self.assertEqual(roles, ["user", "assistant"])
        self.assertEqual(record.turns, 2)
        self.assertEqual(record.title, "who raised this week?")

    def test_reset_removes_transcript_and_index_entry(self) -> None:
        manager = _manager()
        record = manager.route("web", peer="alice")
        manager.record_turn(record, "user", "hello")
        self.assertTrue(manager.store.transcript_path(record.id).exists())

        manager.reset(record.id)
        self.assertIsNone(manager.get(record.id))
        self.assertFalse(manager.store.transcript_path(record.id).exists())

    def test_prune_drops_sessions_past_the_window(self) -> None:
        manager = _manager(prune_after_days=7)
        old = manager.route("web", peer="alice")
        manager.route("web", peer="bob")
        old.last_interaction_at = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

        self.assertEqual(manager.prune(), 1)
        self.assertEqual([r.peer for r in manager.list()], ["bob"])

    def test_index_survives_a_reload(self) -> None:
        store = _store()
        manager = sessions.SessionManager(store=store)
        record = manager.route("web", peer="alice")
        manager.record_turn(record, "user", "hello")

        reopened = sessions.SessionManager(store=sessions.SessionStore(store.dir))
        self.assertIn(record.id, {r.id for r in reopened.list()})

    def test_a_corrupt_index_does_not_stop_the_gateway(self) -> None:
        store = _store()
        store.index_path.write_text("{not json", encoding="utf-8")
        self.assertEqual(sessions.SessionManager(store=store).list(), [])


class LaneTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_session_serialises_and_two_do_not(self) -> None:
        manager = _manager()
        alice = manager.route("web", peer="alice")
        bob = manager.route("web", peer="bob")
        order: list[str] = []

        async def turn(record, label: str, hold: float) -> None:
            async with manager.lock(record.id):
                order.append(f"{label}-in")
                await asyncio.sleep(hold)
                order.append(f"{label}-out")

        await asyncio.gather(
            turn(alice, "a1", 0.05),
            turn(alice, "a2", 0.0),
            turn(bob, "b1", 0.0),
        )

        # Alice's two turns never interleave...
        a_in, a_out = order.index("a1-in"), order.index("a1-out")
        self.assertLess(a_out, order.index("a2-in"))
        # ...but Bob's does not queue behind them.
        self.assertLess(order.index("b1-out"), a_out)
        self.assertGreater(a_out, a_in)


class RunTests(unittest.IsolatedAsyncioTestCase):
    async def test_accept_returns_before_anything_runs(self) -> None:
        registry = runs_module.RunRegistry()
        run = registry.accept("session-1")
        self.assertEqual(run.status, "accepted")
        self.assertTrue(run.accepted_at)
        self.assertFalse(run.started_at)

    async def test_wait_reports_the_terminal_state(self) -> None:
        registry = runs_module.RunRegistry()
        run = registry.accept("session-1")

        async def work() -> str:
            await asyncio.sleep(0.01)
            return "done"

        task = asyncio.create_task(runs_module.guard(registry, run, work))
        result = await registry.wait(run.id, timeout=2.0)
        await task

        self.assertEqual(result["status"], "ok")
        self.assertEqual(run.result, "done")

    async def test_a_failing_turn_becomes_an_error_status(self) -> None:
        registry = runs_module.RunRegistry()
        run = registry.accept("session-1")

        async def work():
            raise ValueError("model said no")

        with self.assertRaises(ValueError):
            await runs_module.guard(registry, run, work)

        status = await registry.wait(run.id, timeout=1.0)
        self.assertEqual(status["status"], "error")
        self.assertIn("model said no", status["error"])

    async def test_timeout_is_the_waiter_giving_up_not_the_run_failing(self) -> None:
        registry = runs_module.RunRegistry()
        run = registry.accept("session-1")
        registry.start(run)

        first = await registry.wait(run.id, timeout=0.02)
        self.assertEqual(first["status"], "timeout")
        # The run itself is untouched, so a later answer is still deliverable.
        self.assertEqual(run.status, "running")

        registry.finish(run, result="late but fine")
        self.assertEqual((await registry.wait(run.id, timeout=0.5))["status"], "ok")

    async def test_abort_without_a_token_says_so(self) -> None:
        """Claiming a cancel that did not happen hides a bill the operator pays."""
        registry = runs_module.RunRegistry()
        run = registry.accept("session-1")
        registry.start(run)

        outcome = registry.abort(run.id)
        self.assertFalse(outcome["aborted"])
        self.assertIn("no cancellation token", outcome["reason"])
        self.assertEqual(run.status, "cancelled")

    async def test_abort_with_a_token_cancels_it(self) -> None:
        class Token:
            def __init__(self) -> None:
                self.cancelled = False

            def cancel(self) -> None:
                self.cancelled = True

        registry = runs_module.RunRegistry()
        run = registry.accept("session-1")
        token = Token()
        registry.attach_token(run, token)
        registry.start(run)

        self.assertTrue(registry.abort(run.id)["aborted"])
        self.assertTrue(token.cancelled)

    async def test_trimming_never_drops_a_run_still_in_flight(self) -> None:
        registry = runs_module.RunRegistry(keep=2)
        live = registry.accept("session-1")
        registry.start(live)
        for _ in range(5):
            registry.finish(registry.accept("session-2"))

        self.assertIsNotNone(registry.get(live.id))


if __name__ == "__main__":
    unittest.main()
