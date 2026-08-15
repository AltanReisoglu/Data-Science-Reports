"""The verification docs/04 §9 asks for: crash one branch on purpose.

Expected: the other two branches survive **and** the failure is written into
`Score.missing_data`. This is the regression guard for the POC's most important
finding — a crashing handler opens the join barrier early and sibling results
disappear with no exception and no warning.

The test does not assert that AutoGen behaves well. It asserts that *this
pipeline* refuses to report a partial result as if it were complete.
"""

from __future__ import annotations

import asyncio
import unittest

import engine
import graph
from agents import analysts
from schemas import Company, Signal, Source


def sample_company() -> Company:
    return Company(
        name="Acme",
        domain="acme.com",
        description="test fixture",
        signals=[
            Signal(
                kind="funding_round",
                summary="Acme raises $12M",
                date=__import__("datetime").datetime(2026, 8, 1, tzinfo=__import__("datetime").timezone.utc),
                source=Source(name="hn", url="https://news.ycombinator.com/item?id=1"),
            )
        ],
    )


class ExplodingClient:
    """A model client that fails the way a real branch fails: mid-run."""

    def __init__(self, inner):
        self._inner = inner

    async def create(self, *args, **kwargs):
        raise RuntimeError("branch crashed on purpose")

    def __getattr__(self, item):
        return getattr(self._inner, item)


def _break_team_branch(inside_wrapper: bool):
    """Patch `build_analysts` so the team branch fails.

    ``inside_wrapper=True`` models reality: the underlying model call is what
    fails, and `engine.ResilientClient` is the thing under test.
    ``inside_wrapper=False`` bypasses that protection, to show what the raw
    framework behaviour actually is.
    """
    original = analysts.build_analysts

    def patched(company_, ledger_):
        technical, market, team = original(company_, ledger_)
        if inside_wrapper:
            team._model_client._inner = ExplodingClient(team._model_client._inner)
        else:
            team._model_client = ExplodingClient(team._model_client)
        return technical, market, team

    return original, patched


class BranchLossTest(unittest.IsolatedAsyncioTestCase):
    async def test_healthy_branches_survive_and_failure_is_recorded(self) -> None:
        company = sample_company()
        ledger = engine.Ledger()
        original, patched = _break_team_branch(inside_wrapper=True)

        analysts.build_analysts = patched  # type: ignore[assignment]
        try:
            branches, score, _ = await graph.enrich(company, ledger)
        finally:
            analysts.build_analysts = original  # type: ignore[assignment]
            await ledger.close()

        by_name = {b.branch: b for b in branches}
        self.assertEqual(len(branches), 3, "every expected branch must be accounted for")
        self.assertFalse(by_name["team"].succeeded, "the crashed branch must be reported as failed")
        self.assertTrue(by_name["team"].error, "a failed branch owes an explanation")

        survivors = [b for b in branches if b.succeeded]
        self.assertGreaterEqual(
            len(survivors), 2,
            "sibling branches that completed must not vanish when one branch crashes",
        )

        if score is not None:
            self.assertTrue(
                any("team" in entry for entry in score.missing_data),
                "a branch that produced nothing must appear in missing_data",
            )

    async def test_without_the_wrapper_a_single_failure_costs_the_whole_fan_out(self) -> None:
        """The measurement that justifies `engine.ResilientClient`.

        Bypass the wrapper and one crashing branch takes the completed work of
        its siblings with it. This is the POC's `autogen_core` finding
        reproducing one layer up, in the AgentChat graph.
        """
        ledger = engine.Ledger()
        original, patched = _break_team_branch(inside_wrapper=False)

        analysts.build_analysts = patched  # type: ignore[assignment]
        try:
            # A short deadline, because without it this call does not return at
            # all: an externally owned runtime leaves `run_stream` waiting on a
            # termination message the crashed manager will never send.
            branches, score, measurement = await graph.enrich(
                sample_company(), ledger, timeout=5.0
            )
        finally:
            analysts.build_analysts = original  # type: ignore[assignment]
            await ledger.close()

        survivors = [b for b in branches if b.succeeded]
        self.assertLess(
            len(survivors), 3,
            "if this ever passes with 3 survivors, the framework fixed the abort "
            "semantics and ResilientClient can be revisited",
        )
        self.assertIsNone(score, "no score can be produced from an aborted graph")
        self.assertIn("aborted", measurement.durma_nedeni)
        # Whatever was lost, it is still reported as missing rather than silently dropped.
        self.assertEqual(len(branches), 3)

    async def test_clean_run_reports_three_branches_and_a_score(self) -> None:
        ledger = engine.Ledger()
        try:
            branches, score, measurement = await graph.enrich(sample_company(), ledger)
        finally:
            await ledger.close()

        self.assertEqual(len(branches), 3)
        self.assertTrue(all(b.succeeded for b in branches))
        self.assertIsNotNone(score, "the scorer must emit a structured Score")
        assert score is not None
        self.assertEqual(measurement.llm_cagrisi, 5, "3 analysts + risk + scorer")


if __name__ == "__main__":
    unittest.main()
