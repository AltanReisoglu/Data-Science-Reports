"""Routing tests — offline, and the fallback is the one that matters.

A router that guesses is worse than one that admits it does not know: a
confident wrong answer about a company is exactly the failure the whole
pipeline is built to avoid.
"""

from __future__ import annotations

import asyncio
import unittest

import answers


SCAN = {
    "query": "ai infrastructure",
    "days": 7,
    "mode": "dry",
    "thesis_is_placeholder": True,
    "failed_sources": {},
    "cost": {"llm_cagrisi": 12, "toplam_token": 3400, "mod": "dry"},
    "stage_costs": [{"desen": "triage", "llm_cagrisi": 9, "toplam_token": 2000}],
    "funnel": {
        "signals": 40, "sources_ok": 3, "sources_failed": 0, "companies": 20,
        "unattached_signals": 4, "triage_passed": 18, "triage_rejected": 2,
        "rejections": [{"name": "bbc.com", "reason": "only third-party mentions"}],
        "enriched": 2, "memos": 0,
    },
    "candidates": [
        {
            "company": {
                "name": "Argonix", "domain": "argonix.io", "github": None,
                "description": "test", "sectors": [], "founders": [],
                "signals": [
                    {
                        "kind": "funding_round", "summary": "Argonix raises $9M",
                        "date": "2026-08-01T00:00:00+00:00",
                        "source": {"name": "hn", "url": "https://news.ycombinator.com/item?id=1"},
                    }
                ],
            },
            "branches": [
                {"branch": "technical", "succeeded": True, "text": "TECHNICAL: ok"},
                {"branch": "market", "succeeded": True, "text": "MARKET: ok"},
                {"branch": "team", "succeeded": False, "error": "branch produced no result"},
            ],
            "score": {
                "thesis_fit": 3, "team": 1, "momentum": 3, "technical_depth": 3, "timing": 3,
                "rationale": {"team": "no founder evidence"},
                "missing_data": ["team branch produced no result"], "decision": "watch",
            },
            "memo": None,
        }
    ],
}


class RouteTest(unittest.TestCase):
    def test_company_name_beats_general_intent(self) -> None:
        # "Argonix score" contains a scoring keyword, but the name is the more
        # specific thing being asked for.
        self.assertEqual(answers.route("Argonix score", SCAN), "company:0")
        self.assertEqual(answers.route("tell me about argonix", SCAN), "company:0")

    def test_general_intents(self) -> None:
        cases = {
            "how much did it cost": "cost",
            "show me the funnel": "funnel",
            "what is missing": "missing",
            "who are the founders": "team",
            "where does the data come from": "sources",
            "what got rejected": "rejected",
            "is any of this real": "mode",
            "how does the scan work": "method",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(answers.route(question, SCAN), expected)

    def test_unknown_question_routes_nowhere(self) -> None:
        self.assertIsNone(answers.route("what is the weather in Istanbul", SCAN))
        self.assertIsNone(answers.route("", SCAN))


class AnswerTest(unittest.IsolatedAsyncioTestCase):
    async def test_known_question_returns_rendered_evidence(self) -> None:
        result = await answers.answer("what did it cost", SCAN)
        self.assertEqual(result["path"], "rules")
        self.assertEqual(result["title"], "Cost")
        self.assertIn("triage", result["html"])

    async def test_unknown_question_admits_it(self) -> None:
        result = await answers.answer("who won the world cup", SCAN)
        self.assertEqual(result["path"], "rules")
        self.assertEqual(result["title"], "Not something I hold")
        self.assertIn("does not hold", result["text"])
        # It still says what it does hold rather than stopping at "no".
        self.assertIn("funnel", result["html"])

    async def test_failed_branch_reaches_the_answer(self) -> None:
        result = await answers.answer("Argonix", SCAN)
        self.assertIn("team: no result", result["html"])
        self.assertIn("team branch produced no result", result["html"])


class FactsTest(unittest.TestCase):
    def test_facts_carry_sources_and_gaps(self) -> None:
        blob = answers.facts(SCAN)
        self.assertIn("news.ycombinator.com", blob)
        self.assertIn("team branch produced no result", blob)
        self.assertIn('"mode": "dry"', blob)


if __name__ == "__main__":
    unittest.main()
