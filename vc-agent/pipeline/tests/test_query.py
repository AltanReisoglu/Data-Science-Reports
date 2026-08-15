"""Querying across the scan — filters, and the rules a row must not break."""

from __future__ import annotations

import unittest

from gateway import tools as tools_module


def candidate(name, *, thesis=3, team=3, momentum=3, technical=3, timing=3,
              missing=None, decision="watch", sectors=None, country=None, url=None):
    return {
        "company": {
            "name": name,
            "sectors": sectors or [],
            "country": country,
            "signals": (
                [{"summary": "s", "source": {"url": url}}] if url else []
            ),
        },
        "score": {
            "thesis_fit": thesis, "team": team, "momentum": momentum,
            "technical_depth": technical, "timing": timing,
            "missing_data": missing or [], "decision": decision,
        },
        "branches": [],
    }


SCAN = {
    "query": "ai infra", "days": 7, "mode": "live",
    "candidates": [
        candidate("Acme", thesis=5, team=4, momentum=5, technical=5, timing=4,
                  decision="review", sectors=["fintech", "payments"], country="Turkey",
                  url="https://www.sec.gov/acme"),
        candidate("Argonix", thesis=2, team=1, momentum=2, technical=1, timing=2,
                  decision="skip", sectors=["robotics"], country="Germany",
                  missing=["founder_names", "technical_architecture"],
                  url="https://news.ycombinator.com/item?id=1"),
        candidate("Bytewell", thesis=4, team=0, momentum=3, technical=4, timing=3,
                  decision="watch", sectors=["fintech"], country="Turkey",
                  missing=["founder_identities"]),
    ],
}


def tools(scan=SCAN):
    return tools_module.named(tools_module.Sources(scan_getter=lambda: scan))


class FilterTests(unittest.TestCase):
    def test_no_filter_returns_everything(self) -> None:
        out = tools()["query_companies"]()
        self.assertIn("3 of 3", out)
        for name in ("Acme", "Argonix", "Bytewell"):
            self.assertIn(name, out)

    def test_min_total(self) -> None:
        out = tools()["query_companies"](min_total=15)
        self.assertIn("Acme", out)         # 23
        self.assertNotIn("Argonix", out)   # 8

    def test_axis_filter(self) -> None:
        out = tools()["query_companies"](axis="team", min_axis=4)
        self.assertIn("Acme", out)
        self.assertNotIn("Bytewell", out)

    def test_an_unknown_axis_is_named_not_ignored(self) -> None:
        """Silently returning everything would look like a successful query."""
        out = tools()["query_companies"](axis="vibes", min_axis=3)
        self.assertIn("Unknown axis", out)
        self.assertIn("technical_depth", out)

    def test_sector_matches_a_substring(self) -> None:
        out = tools()["query_companies"](sector="fin")
        self.assertIn("Acme", out)
        self.assertIn("Bytewell", out)
        self.assertNotIn("Argonix", out)

    def test_decision_and_country(self) -> None:
        self.assertIn("Acme", tools()["query_companies"](decision="review"))
        self.assertNotIn("Argonix", tools()["query_companies"](country="turkey"))

    def test_without_missing_excludes_by_gap(self) -> None:
        out = tools()["query_companies"](without_missing="founder_identities")
        self.assertNotIn("Bytewell", out)
        self.assertIn("Acme", out)

    def test_sorting(self) -> None:
        rows = tools()["query_companies"](sort="total")
        self.assertLess(rows.index("Argonix"), rows.index("Acme"))
        rows = tools()["query_companies"](sort="-total")
        self.assertLess(rows.index("Acme"), rows.index("Argonix"))
        rows = tools()["query_companies"](sort="name")
        self.assertLess(rows.index("Acme"), rows.index("Bytewell"))

    def test_limit_says_how_many_it_held_back(self) -> None:
        out = tools()["query_companies"](limit=1)
        self.assertIn("2 more", out)


class RowRuleTests(unittest.TestCase):
    """Every row obeys the same rules as every other answer in this system."""

    def test_every_row_carries_a_source_or_admits_it_has_none(self) -> None:
        out = tools()["query_companies"]()
        self.assertIn("https://www.sec.gov/acme", out)
        # Bytewell has no signal URL, and the row says so rather than staying quiet.
        self.assertIn("no source URL recorded", out)

    def test_missing_data_is_visible_next_to_the_score(self) -> None:
        """`team 0` and `team unknown` must not look the same."""
        out = tools()["query_companies"](sort="name")
        self.assertIn("missing: founder_identities", out)

    def test_dry_mode_is_flagged_on_every_score(self) -> None:
        dry = dict(SCAN, mode="dry")
        out = tools(dry)["query_companies"]()
        self.assertIn("dry mode", out)
        self.assertIn("placeholder", out)

    def test_an_empty_result_says_which_filter_emptied_it(self) -> None:
        """"No matches" read as "no such companies exist" is the failure here."""
        out = tools()["query_companies"](sector="quantum")
        self.assertIn("0 of 3", out)
        self.assertIn("sector removed 3", out)
        self.assertIn("statement about this scan, not about the world", out)

    def test_no_scan_is_said_rather_than_returning_nothing(self) -> None:
        self.assertIn("No scan has been run yet", tools(None)["query_companies"]())


class CompareTests(unittest.TestCase):
    def test_two_companies_compare_axis_by_axis(self) -> None:
        out = tools()["compare_companies"]("Acme, Argonix")
        for axis in ("thesis_fit", "team", "momentum", "technical_depth", "timing"):
            self.assertIn(axis, out)
        self.assertIn("TOTAL", out)

    def test_a_gap_marks_the_score_as_thin_not_absent(self) -> None:
        """The scorer writes gaps in its own words; the link is a heuristic."""
        out = tools()["compare_companies"]("Argonix, Bytewell")
        self.assertIn("?", out)
        self.assertIn("thin for both", out)
        # And the raw gap list is printed, so the reader can judge the link.
        self.assertIn("founder_names", out)
        self.assertIn("founder_identities", out)

    def test_a_company_with_no_recorded_gaps_says_so(self) -> None:
        out = tools()["compare_companies"]("Acme, Argonix")
        self.assertIn("nothing recorded as missing", out)

    def test_one_name_is_not_a_comparison(self) -> None:
        self.assertIn("at least two", tools()["compare_companies"]("Acme"))

    def test_unknown_names_are_listed_with_what_is_known(self) -> None:
        out = tools()["compare_companies"]("Acme, Nonesuch")
        self.assertIn("Not enough", out)
        self.assertIn("Argonix", out)

    def test_a_partial_match_still_compares_and_names_the_gap(self) -> None:
        out = tools()["compare_companies"]("Acme, Argonix, Nonesuch")
        self.assertIn("TOTAL", out)
        self.assertIn("Not in this scan: Nonesuch", out)

    def test_gap_mapping_is_keyword_based(self) -> None:
        self.assertTrue(tools_module._gaps_for("team", ["founder_names"]))
        self.assertTrue(tools_module._gaps_for("technical_depth", ["technical_architecture"]))
        self.assertFalse(tools_module._gaps_for("timing", ["founder_names"]))


if __name__ == "__main__":
    unittest.main()
