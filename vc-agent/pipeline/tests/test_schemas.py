"""Data contract tests — the two mandatory fields must be genuinely mandatory."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from schemas import (
    Company,
    Score,
    Signal,
    Source,
    normalize_name,
    registrable_domain,
)


class SourceTest(unittest.TestCase):
    def test_url_is_required(self) -> None:
        with self.assertRaises(ValidationError):
            Source(name="hn")  # type: ignore[call-arg]

    def test_url_must_be_resolvable(self) -> None:
        with self.assertRaises(ValidationError):
            Source(name="hn", url="acme.com")  # no scheme -> not verifiable


class ScoreTest(unittest.TestCase):
    def _score(self, **kwargs):
        base = dict(
            thesis_fit=3, team=3, momentum=3, technical_depth=3, timing=3,
            missing_data=[], decision="watch",
        )
        base.update(kwargs)
        return Score(**base)

    def test_missing_data_is_required(self) -> None:
        with self.assertRaises(ValidationError):
            Score(
                thesis_fit=3, team=3, momentum=3, technical_depth=3, timing=3,
                decision="watch",
            )  # type: ignore[call-arg]

    def test_total_and_reliability(self) -> None:
        self.assertEqual(self._score().total, 15)
        self.assertEqual(self._score(missing_data=[]).reliability, "high")
        self.assertEqual(self._score(missing_data=["a", "b"]).reliability, "medium")
        self.assertEqual(self._score(missing_data=["a", "b", "c"]).reliability, "low")

    def test_axes_are_bounded(self) -> None:
        with self.assertRaises(ValidationError):
            self._score(team=7)


class EntityKeyTest(unittest.TestCase):
    def test_key_prefers_domain_then_github_then_name(self) -> None:
        self.assertEqual(Company(name="Acme", domain="acme.com").key, "domain:acme.com")
        self.assertEqual(Company(name="Acme", github="acme-io").key, "gh:acme-io")
        self.assertEqual(Company(name="Acme Labs, Inc.").key, "name:acme")

    def test_normalize_name_strips_suffixes(self) -> None:
        self.assertEqual(normalize_name("Acme Labs, Inc."), "acme")
        self.assertEqual(normalize_name("Foo.io"), "foo")

    def test_registrable_domain(self) -> None:
        self.assertEqual(registrable_domain("https://www.acme.com/blog/x"), "acme.com")
        self.assertEqual(registrable_domain("https://api.acme.co.uk/v1"), "acme.co.uk")
        self.assertIsNone(registrable_domain("not-a-domain"))


class SignalTest(unittest.TestCase):
    def test_fingerprint_is_stable_and_distinguishing(self) -> None:
        def make(summary: str) -> Signal:
            return Signal(
                kind="news",
                summary=summary,
                date=datetime(2026, 8, 1, tzinfo=timezone.utc),
                source=Source(name="hn", url="https://news.ycombinator.com/item?id=1"),
            )

        self.assertEqual(make("same").fingerprint, make("same").fingerprint)
        self.assertNotEqual(make("a").fingerprint, make("b").fingerprint)


if __name__ == "__main__":
    unittest.main()
