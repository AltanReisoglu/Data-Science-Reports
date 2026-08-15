"""Live check tests — offline, with the collectors replaced by fakes.

The assertion that matters most is the one about failure: "could not check" and
"nothing changed" must never look alike. A monitoring loop that reports silence
as calm is worse than no monitoring loop.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import live
from collectors.base import CollectionResult
from schemas import Signal, Source

NOW = datetime.now(timezone.utc)
SCAN_DAY = NOW - timedelta(days=5)


def company(**overrides) -> dict:
    base = {
        "name": "Acme",
        "domain": "acme.com",
        "github": "acme",
        "signals": [
            {
                "kind": "repo_momentum",
                "summary": "acme/engine · ⭐900",
                "date": SCAN_DAY.isoformat(),
                "source": {"name": "github", "url": "https://github.com/acme/engine"},
                "raw": {"stars": 900},
            }
        ],
    }
    base.update(overrides)
    return base


class FakeGitHub:
    """Stands in for the GitHub collector. `payload=None` means the call fails."""

    payload: dict | None = None

    def __init__(self, *a, **k) -> None:
        pass

    def fetch_json(self, url: str):
        if FakeGitHub.payload is None:
            raise RuntimeError("connection reset")
        return FakeGitHub.payload


def fake_collector(result: CollectionResult):
    class Fake:
        def __init__(self, *a, **k) -> None:
            pass

        def run(self, *, query, days):
            return result

    return Fake


def hn_signal(title: str, when: datetime) -> Signal:
    return Signal(
        kind="news", summary=title, date=when,
        source=Source(name="hn", url="https://news.ycombinator.com/item?id=1"),
    )


class LiveTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = (live.GitHub, live.HackerNews, live.SecFormD)
        FakeGitHub.payload = {
            "stargazers_count": 1240,
            "pushed_at": (NOW - timedelta(days=1)).isoformat().replace("+00:00", "Z"),
            "open_issues_count": 12,
            "html_url": "https://github.com/acme/engine",
        }
        live.GitHub = FakeGitHub
        live.HackerNews = fake_collector(CollectionResult(source="hn", signals=[]))
        live.SecFormD = fake_collector(CollectionResult(source="sec_edgar", signals=[]))

    def tearDown(self) -> None:
        live.GitHub, live.HackerNews, live.SecFormD = self._saved

    # ------------------------------------------------------------ deltas

    def test_star_delta_is_reported_against_the_scan(self) -> None:
        report = live.refresh(company())
        self.assertTrue(any("900" in c and "1,240" in c for c in report.changes),
                        f"star delta missing from {report.changes}")

    def test_push_after_the_scan_is_a_change(self) -> None:
        report = live.refresh(company())
        self.assertTrue(any("pushed" in c for c in report.changes))

    def test_no_movement_reports_no_change(self) -> None:
        FakeGitHub.payload = {
            "stargazers_count": 900,                       # unchanged
            "pushed_at": (SCAN_DAY - timedelta(days=2)).isoformat().replace("+00:00", "Z"),
            "html_url": "https://github.com/acme/engine",
        }
        report = live.refresh(company())
        self.assertEqual(report.changes, [])
        self.assertEqual(report.failed, [])

    # ------------------------------------------------------------ failure

    def test_a_failed_source_is_not_silence(self) -> None:
        FakeGitHub.payload = None  # the call raises
        report = live.refresh(company())
        self.assertIn("github", report.failed)
        text = report.as_text()
        self.assertIn("could not be checked", text)
        self.assertIn("not the same as no change", text)

    def test_one_dead_source_does_not_stop_the_others(self) -> None:
        FakeGitHub.payload = None
        live.HackerNews = fake_collector(
            CollectionResult(source="hn", signals=[hn_signal("Acme raises $12M", NOW)])
        )
        report = live.refresh(company())
        sources = {c.source for c in report.checks}
        self.assertEqual(sources, {"github", "hn", "sec_edgar"})
        self.assertIn("github", report.failed)
        self.assertTrue(any("Hacker News" in c for c in report.changes))

    # ------------------------------------------------------------ precision

    def test_hn_results_that_do_not_name_the_company_are_not_mentions(self) -> None:
        # Algolia ranks loosely; a result that never says "Acme" is not about Acme.
        live.HackerNews = fake_collector(
            CollectionResult(
                source="hn",
                signals=[hn_signal("A better local tokenizer", NOW)],
            )
        )
        report = live.refresh(company())
        self.assertFalse(any("Hacker News" in c for c in report.changes))
        hn = next(c for c in report.checks if c.source == "hn")
        self.assertIn("0 actually name it", hn.detail)

    def test_mentions_older_than_the_scan_are_not_new(self) -> None:
        live.HackerNews = fake_collector(
            CollectionResult(
                source="hn",
                signals=[hn_signal("Acme launches", SCAN_DAY - timedelta(days=3))],
            )
        )
        report = live.refresh(company())
        self.assertFalse(any("Hacker News" in c for c in report.changes))

    # ------------------------------------------------------------ lookup

    def test_find_company_is_case_insensitive_and_prefix_safe(self) -> None:
        data = {"candidates": [
            {"company": {"name": "Argonix"}},
            {"company": {"name": "Argus"}},
        ]}
        self.assertEqual(live.find_company(data, "argonix")["name"], "Argonix")
        self.assertEqual(live.find_company(data, "argoni")["name"], "Argonix")
        # "arg" matches both — ambiguous, so it must not guess.
        self.assertIsNone(live.find_company(data, "arg"))
        self.assertIsNone(live.find_company(data, "nonesuch"))


if __name__ == "__main__":
    unittest.main()
