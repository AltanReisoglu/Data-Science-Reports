"""Policy gate tests. No network: robots.txt is injected.

The blocklist test is the important one. It is not checking a helper function,
it is guarding a legal boundary that the rest of the system is allowed to
assume holds.
"""

from __future__ import annotations

import time
import unittest

import config
import policy


def _no_robots(_url: str) -> str | None:
    return None


def _robots(text: str):
    return lambda _url: text


class BlocklistTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = policy.SourcePolicy(
            robots_fetcher=_robots("User-agent: *\nAllow: /"),
            rate_limited=False,
        )

    def test_blocklisted_hosts_are_always_refused(self) -> None:
        # Even with a permissive robots.txt, these must never be fetched.
        for url in (
            "https://www.linkedin.com/in/someone",
            "https://linkedin.com/company/acme",
            "https://api.crunchbase.com/v4/entities",
            "https://x.com/acme",
        ):
            with self.subTest(url=url):
                self.assertFalse(self.policy.is_allowed(url, source="test"))

    def test_blocklist_beats_api_exemption(self) -> None:
        # An exemption must not become a way around the legal boundary.
        self.assertFalse(
            self.policy.is_allowed(
                "https://www.linkedin.com/in/someone",
                source="test",
                exemption="pretending to have terms",
            )
        )

    def test_ordinary_host_is_allowed(self) -> None:
        self.assertTrue(self.policy.is_allowed("https://news.ycombinator.com/item?id=1", source="test"))

    def test_refusal_is_audited(self) -> None:
        self.policy.is_allowed("https://linkedin.com/x", source="test")
        self.assertEqual(self.policy.records[-1].reason, "blocklist")
        self.assertFalse(self.policy.records[-1].allowed)


class RobotsTest(unittest.TestCase):
    def test_disallow_is_honoured(self) -> None:
        p = policy.SourcePolicy(robots_fetcher=_robots("User-agent: *\nDisallow: /"), rate_limited=False)
        self.assertFalse(p.is_allowed("https://example.com/anything", source="test"))

    def test_unreachable_robots_allows_but_records_it(self) -> None:
        p = policy.SourcePolicy(robots_fetcher=_no_robots, rate_limited=False)
        self.assertTrue(p.is_allowed("https://example.com/x", source="test"))
        self.assertEqual(p.records[-1].reason, "robots_unavailable")

    def test_api_exemption_is_logged_with_its_justification(self) -> None:
        p = policy.SourcePolicy(robots_fetcher=_robots("User-agent: *\nDisallow: /"), rate_limited=False)
        self.assertTrue(
            p.is_allowed("http://export.arxiv.org/api/query", source="arxiv", exemption="arXiv ToU")
        )
        self.assertIn("api_exemption", p.records[-1].reason)
        self.assertIn("arXiv ToU", p.records[-1].reason)


class RateLimitTest(unittest.TestCase):
    def test_second_call_to_same_source_waits(self) -> None:
        p = policy.SourcePolicy(robots_fetcher=_no_robots)
        config.RATE_LIMITS["unittest_source"] = 0.25
        p.wait("unittest_source")
        started = time.monotonic()
        slept = p.wait("unittest_source")
        elapsed = time.monotonic() - started
        self.assertGreater(slept, 0.0)
        self.assertGreaterEqual(elapsed, 0.2)


if __name__ == "__main__":
    unittest.main()
