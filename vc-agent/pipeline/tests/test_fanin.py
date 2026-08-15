"""The core-level fan-in: results survive a branch that fails outright.

`tests/test_graph.py` locks in what the AgentChat engine does. This locks in the
difference: with pub/sub plus a collector queue, a raw failure in one branch
costs that branch and nothing else, and it costs no wall-clock either.
"""

from __future__ import annotations

import time
import unittest
from datetime import datetime, timezone

import engine
import fanin
from agents import analysts
from schemas import Company, Signal, Source
from tests.test_graph import ExplodingClient


def sample_company() -> Company:
    return Company(
        name="Acme",
        domain="acme.com",
        signals=[
            Signal(
                kind="funding_round",
                summary="Acme raises $12M",
                date=datetime(2026, 8, 1, tzinfo=timezone.utc),
                source=Source(name="hn", url="https://news.ycombinator.com/item?id=1"),
            )
        ],
    )


class FanInTest(unittest.IsolatedAsyncioTestCase):
    async def test_clean_run_collects_every_branch(self) -> None:
        ledger = engine.Ledger()
        try:
            branches, _score, measurement = await fanin.enrich(
                sample_company(), ledger, timeout=8.0
            )
        finally:
            await ledger.close()

        self.assertEqual(len(branches), 3)
        self.assertTrue(all(b.succeeded for b in branches))
        self.assertEqual(measurement.durma_nedeni, "all branches reported")

    async def test_raw_failure_costs_one_branch_and_no_time(self) -> None:
        original = analysts.build_analysts

        def patched(company, ledger):
            technical, market, team = original(company, ledger)
            team._model_client = ExplodingClient(team._model_client)  # bypass the wrapper
            return technical, market, team

        analysts.build_analysts = patched  # type: ignore[assignment]
        ledger = engine.Ledger()
        started = time.perf_counter()
        try:
            branches, _score, _m = await fanin.enrich(sample_company(), ledger, timeout=8.0)
        finally:
            analysts.build_analysts = original  # type: ignore[assignment]
            await ledger.close()
        elapsed = time.perf_counter() - started

        by_branch = {b.branch: b for b in branches}
        self.assertEqual(len(branches), 3, "every expected branch is accounted for")
        self.assertFalse(by_branch["team"].succeeded)
        self.assertEqual(
            sum(1 for b in branches if b.succeeded), 2,
            "the two healthy branches survive a raw failure in the third",
        )
        # The same injection makes the AgentChat engine wait out the whole
        # deadline; here the failure is published, so nothing waits.
        self.assertLess(elapsed, 4.0, "a failed branch must not cost the deadline")

    async def test_failure_behind_the_wrapper_is_not_counted_as_success(self) -> None:
        original = analysts.build_analysts

        def patched(company, ledger):
            technical, market, team = original(company, ledger)
            team._model_client._inner = ExplodingClient(team._model_client._inner)
            return technical, market, team

        analysts.build_analysts = patched  # type: ignore[assignment]
        ledger = engine.Ledger()
        try:
            branches, _score, _m = await fanin.enrich(sample_company(), ledger, timeout=8.0)
        finally:
            analysts.build_analysts = original  # type: ignore[assignment]
            await ledger.close()

        by_branch = {b.branch: b for b in branches}
        # ResilientClient hands back an error *as text*; a branch that reports it
        # as an answer would be a silent partial result reintroduced one level down.
        self.assertFalse(by_branch["team"].succeeded)
        self.assertIn("branch crashed on purpose", by_branch["team"].error or "")


if __name__ == "__main__":
    unittest.main()
