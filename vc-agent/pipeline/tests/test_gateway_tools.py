"""Every tool, actually called.

Written after a real slip: the tools were re-typed from memory when they moved
into `gateway/tools.py`, and three of them referenced collector classes and
`live` functions that do not exist. Schema tests passed anyway, because a schema
is built from the signature and never runs the body.

So this file calls each one. The network-facing pair is exercised against a
stubbed collector rather than skipped — a wrong import is exactly what needs
catching, and it fails before any request is made.
"""

from __future__ import annotations

import json
import unittest

from gateway import tools as tools_module

SCAN = {
    "query": "ai infra",
    "days": 7,
    "mode": "dry",
    "candidates": [
        {
            "company": {
                "name": "Acme Robotics",
                "signals": [
                    {
                        "summary": "Series seed filing",
                        "source": {"url": "https://www.sec.gov/Archives/acme"},
                    }
                ],
                "repos": {},
            },
            # The total is summed from the axes, not stored — see `dashboard.AXES`.
            "score": {
                "thesis_fit": 4,
                "team": 0,
                "momentum": 5,
                "technical_depth": 5,
                "timing": 4,
                "missing_data": ["team"],
            },
            "branches": [
                {"branch": "technical", "succeeded": True},
                {"branch": "market", "succeeded": False},
            ],
        }
    ],
}


def sources(scan=SCAN, starter=None) -> tools_module.Sources:
    return tools_module.Sources(scan_getter=lambda: scan, scan_starter=starter)


class ToolBodyTests(unittest.TestCase):
    def test_every_tool_is_callable_with_its_documented_arguments(self) -> None:
        """A smoke pass over the whole surface — imports resolve, bodies run."""
        functions = tools_module.named(sources())
        calls = {
            "scan_facts": (),
            "company_detail": ("Acme Robotics",),
            "search_docs": ("closureagent",),
            "memory_search": ("anything",),
            "memory_get": ("memory/2026-01-01.md",),
            "memory_note": ("a note from the tool test",),
        }
        for name, args in calls.items():
            with self.subTest(tool=name):
                self.assertIsInstance(functions[name](*args), str)

    def test_company_detail_returns_the_scan_record(self) -> None:
        payload = json.loads(tools_module.named(sources())["company_detail"]("Acme Robotics"))
        self.assertEqual(payload["name"], "Acme Robotics")
        self.assertEqual(payload["score_total"], 18)
        self.assertEqual(payload["missing_data"], ["team"])
        # Sources travel with the answer, as everywhere else in this system.
        self.assertIn("sec.gov", payload["signals"][0]["url"])

    def test_company_detail_lists_what_it_does_have(self) -> None:
        message = tools_module.named(sources())["company_detail"]("Nonexistent Ltd")
        self.assertIn("No candidate named", message)
        self.assertIn("Acme Robotics", message)

    def test_a_missing_scan_is_said_rather_than_invented(self) -> None:
        functions = tools_module.named(sources(scan=None))
        for name in ("scan_facts", "company_detail", "company_live"):
            with self.subTest(tool=name):
                args = () if name == "scan_facts" else ("Acme Robotics",)
                self.assertIn("No scan has been run yet", functions[name](*args))

    def test_company_live_finds_the_company_before_going_to_the_network(self) -> None:
        """The unknown-name path must not need a request to answer."""
        message = tools_module.named(sources())["company_live"]("Nonexistent Ltd")
        self.assertIn("No candidate named", message)

    def test_search_tools_import_the_collectors_that_exist(self) -> None:
        """The slip this file was written for: wrong class names, wrong methods."""
        import collectors.github as github_module
        import collectors.hackernews as hn_module

        functions = tools_module.named(sources())

        class Stub:
            def __init__(self, *a, **k) -> None:
                pass

            def run(self, *, query: str, days: int):
                from collectors.base import CollectionResult

                return CollectionResult(source="stub", signals=[], error="stubbed")

        real_github, real_hn = github_module.GitHub, hn_module.HackerNews
        github_module.GitHub, hn_module.HackerNews = Stub, Stub
        try:
            self.assertIn("stubbed", functions["search_github"]("anything"))
            self.assertIn("stubbed", functions["search_hacker_news"]("anything"))
        finally:
            github_module.GitHub, hn_module.HackerNews = real_github, real_hn

    def test_start_scan_reports_when_it_cannot(self) -> None:
        functions = tools_module.named(sources(starter=None))
        self.assertIn("cannot start scans", functions["start_scan"]("fintech"))

    def test_start_scan_passes_the_arguments_through(self) -> None:
        seen = {}
        functions = tools_module.named(
            sources(starter=lambda query, days: seen.update(query=query, days=days))
        )
        message = functions["start_scan"]("fintech", 14)
        self.assertEqual(seen, {"query": "fintech", "days": 14})
        self.assertIn("fintech", message)

    def test_memory_note_then_memory_search_round_trips(self) -> None:
        functions = tools_module.named(sources())
        functions["memory_note"]("Northgate led the Acme seed round.", "scan")
        self.assertIn("Northgate", functions["memory_search"]("northgate acme"))


if __name__ == "__main__":
    unittest.main()
