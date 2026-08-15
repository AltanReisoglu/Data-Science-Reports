"""Collector tests against recorded fixtures — no network, no keys."""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

import policy
from collectors.hackernews import HackerNews
from collectors.sec_edgar import SecFormD
from normalize import resolve

NOW = int(datetime.now(timezone.utc).timestamp())

HN_FIXTURE = {
    "hits": [
        {
            "objectID": "1",
            "title": "Acme, a database company, raises $12M Series A",
            "url": "https://www.acme.com/blog/series-a",
            "created_at_i": NOW - 3600,
            "points": 120,
            "num_comments": 30,
        },
        {
            "objectID": "2",
            "title": "Show HN: Bolt – a local-first sync engine",
            "url": "https://bolt.dev",
            "created_at_i": NOW - 7200,
            "points": 90,
            "num_comments": 12,
        },
        {
            "objectID": "3",
            "title": "Why databases are hard",
            "url": "",
            "created_at_i": NOW - 9000,
            "points": 5,
            "num_comments": 1,
        },
        {
            "objectID": "4",
            "title": "Show HN: Kestrel – a query planner",
            "url": "https://github.com/someone/kestrel",
            "created_at_i": NOW - 10000,
            "points": 40,
            "num_comments": 4,
        },
    ]
}

SEC_FIXTURE = {
    "hits": {
        "hits": [
            {
                "_source": {
                    "display_names": ["Acme Data Inc.  (CIK 0001234567)"],
                    "file_date": "2026-08-10",
                }
            }
        ]
    }
}


def fixture_fetcher(payload):
    def fetch(_url, _params=None):
        return 200, json.dumps(payload)

    return fetch


def offline_policy():
    return policy.SourcePolicy(robots_fetcher=lambda _u: None, rate_limited=False)


class HackerNewsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.collector = HackerNews(
            source_policy=offline_policy(),
            fetcher=fixture_fetcher(HN_FIXTURE),
            use_cache=False,
        )

    def test_classifies_signal_kinds(self) -> None:
        result = self.collector.run(query="db", days=7)
        self.assertTrue(result.succeeded)
        kinds = [s.kind for s in result.signals]
        self.assertEqual(kinds, ["funding_round", "product_launch", "news", "product_launch"])

    def test_platform_host_is_not_treated_as_a_company_domain(self) -> None:
        # A launch linking to github.com points at a repository. Taking the
        # domain would key every such launch on the same entity.
        signals = self.collector.run(query="db", days=7).signals
        self.assertIsNone(signals[3].candidate_domain)
        self.assertEqual(signals[3].candidate_name, "Kestrel")

    def test_extracts_name_from_headline(self) -> None:
        signals = self.collector.run(query="db", days=7).signals
        self.assertEqual(signals[0].candidate_name, "Acme")
        self.assertEqual(signals[1].candidate_name, "Bolt")

    def test_domain_is_taken_only_from_a_launch(self) -> None:
        signals = self.collector.run(query="db", days=7).signals
        # Funding story: the link points at whoever published it.
        self.assertIsNone(signals[0].candidate_domain)
        # Show HN: the submitter is pointing at their own product.
        self.assertEqual(signals[1].candidate_domain, "bolt.dev")

    def test_unclear_headline_yields_no_name(self) -> None:
        # Better to say nothing than to invent an entity.
        self.assertIsNone(self.collector.run(query="db", days=7).signals[2].candidate_name)

    def test_every_signal_carries_a_source_url(self) -> None:
        for signal in self.collector.run(query="db", days=7).signals:
            self.assertTrue(signal.source.url.startswith("https://"))


class SecFormDTest(unittest.TestCase):
    def test_parses_cik_and_marks_source_official(self) -> None:
        collector = SecFormD(
            source_policy=offline_policy(),
            fetcher=fixture_fetcher(SEC_FIXTURE),
            use_cache=False,
        )
        signals = collector.run(query="data", days=7).signals
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].candidate_name, "Acme Data Inc")
        self.assertEqual(signals[0].source.confidence, "official")
        self.assertIn("0001234567", signals[0].source.url)


class FailureTest(unittest.TestCase):
    def test_collector_failure_is_carried_not_raised(self) -> None:
        def broken(_url, _params=None):
            raise RuntimeError("connection reset")

        result = HackerNews(
            source_policy=offline_policy(), fetcher=broken, use_cache=False
        ).run(query="db", days=7)
        self.assertFalse(result.succeeded)
        self.assertIn("connection reset", result.error or "")
        self.assertEqual(result.signals, [])


class ResolutionTest(unittest.TestCase):
    def test_uncertain_names_are_kept_apart_rather_than_merged(self) -> None:
        # "Acme" (HN, with a domain) and "Acme Data Inc" (SEC, name only) are
        # plausibly the same company — but only plausibly. Normalized they are
        # "acme" and "acme data", a 0.66 similarity, below the 0.92 threshold.
        #
        # This documents a deliberate trade-off rather than a limitation we
        # overlooked: merging two records wrongly contaminates the evidence
        # package irreversibly, while leaving them apart costs a duplicate row a
        # human can spot. Closing this gap needs a real entity resolver
        # (domain <-> filing name), not a lower threshold.
        hn = HackerNews(
            source_policy=offline_policy(), fetcher=fixture_fetcher(HN_FIXTURE), use_cache=False
        ).run(query="db", days=7).signals
        sec = SecFormD(
            source_policy=offline_policy(), fetcher=fixture_fetcher(SEC_FIXTURE), use_cache=False
        ).run(query="db", days=7).signals

        companies = resolve(hn + sec)
        names = {c.name.lower() for c in companies}
        self.assertIn("acme", names)
        self.assertIn("acme data inc", names)
        self.assertTrue(all(len(c.signals) == 1 for c in companies if "acme" in c.name.lower()))

    def test_signal_without_owner_is_left_unattached(self) -> None:
        hn = HackerNews(
            source_policy=offline_policy(), fetcher=fixture_fetcher(HN_FIXTURE), use_cache=False
        ).run(query="db", days=7).signals
        companies = resolve(hn)
        attached = sum(len(c.signals) for c in companies)
        self.assertEqual(attached, 3)  # "Why databases are hard" names no entity

    def test_duplicate_event_enters_once(self) -> None:
        hn = HackerNews(
            source_policy=offline_policy(), fetcher=fixture_fetcher(HN_FIXTURE), use_cache=False
        ).run(query="db", days=7).signals
        companies = resolve(hn + hn)
        self.assertEqual(sum(len(c.signals) for c in companies), 3)


if __name__ == "__main__":
    unittest.main()
