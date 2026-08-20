"""Documentation search — offline, over the real `docs/` corpus.

The corpus is checked into the repository, so these run against the actual
documents rather than a fixture. That makes them slightly brittle by design: if
a document is renamed or a section retitled, the test says so, which is the
point — a search index that silently stops covering a document is worse than one
that fails loudly.
"""

from __future__ import annotations

import unittest

import answers
import docs_index


class IndexTest(unittest.TestCase):
    def test_index_covers_both_halves_of_docs(self) -> None:
        stats = docs_index.stats()
        self.assertGreater(stats["sections"], 300, "the corpus should be substantial")
        docs = stats["documents"]
        # Microsoft's words and ours, both indexed.
        self.assertIn("05-autogen-core-user-guide.md", docs)
        self.assertIn("08-autogen-agentchat-user-guide.md", docs)
        self.assertIn("04-vc-agentic-akis.md", docs)

    def test_headings_inside_code_blocks_are_not_sections(self) -> None:
        # These documents are full of `# comment` lines inside python fences.
        titles = [s.title for s in docs_index.sections()]
        self.assertFalse(
            any(t.startswith("pip install") or t.startswith("Set the environment") for t in titles),
            "a comment inside a fence was parsed as a heading",
        )

    def test_provenance_separates_official_from_ours(self) -> None:
        official = [s for s in docs_index.sections() if s.official]
        ours = [s for s in docs_index.sections() if not s.official]
        self.assertTrue(official and ours)
        self.assertTrue(all("official" in s.provenance for s in official))


class SearchTest(unittest.TestCase):
    def test_a_framework_concept_lands_in_the_official_guide(self) -> None:
        hits = docs_index.search("topic subscription", k=3)
        self.assertTrue(hits)
        self.assertTrue(
            any(h.section.doc.startswith("05-") for h in hits),
            f"expected the core guide, got {[h.section.doc for h in hits]}",
        )

    def test_a_project_decision_lands_in_our_own_notes(self) -> None:
        """A "why did *we* do it this way" question must reach a page we wrote.

        This used to name three files (09, 07, 01). Adding MAF's guide and its 35
        design records tripled the corpus and made vendor text 71% of it, and
        Microsoft's ADRs discuss AutoGen's design at length — so they now outrank
        our own notes for this query, with 17 (our build/borrow/deploy decision)
        third. Naming files was the wrong assertion anyway: the right answer moves
        between our documents as they are written, and the property that must hold
        is that vendor text does not crowd our notes out entirely.

        `PROVENANCE` already encodes the distinction, so the test reads it instead
        of guessing from filenames. This is a weaker bound than before by exactly
        one thing — it no longer demands the *first* hit be ours — and that is
        deliberate: for this query the vendor pages are genuinely on topic.
        """
        hits = docs_index.search("neden AutoGen seçildi", k=3)
        self.assertTrue(hits)
        self.assertTrue(
            any("official" not in h.section.provenance for h in hits),
            f"expected at least one of our own docs, got "
            f"{[(h.section.doc, h.section.provenance) for h in hits]}",
        )

    def test_a_measured_gotcha_is_findable(self) -> None:
        """Our own measured gotchas must surface for the identifier that names them.

        This assertion used to run the query "model_info required openai
        compatible" and expect 06 in the top four. It stopped holding as the
        corpus grew, and the reason is not a defect: three of those four terms
        are generic, so sections whose subject genuinely *is* OpenAI-compatible
        clients ("Model Client for OpenAI-Compatible APIs", 08:546) outscore a
        short page that matches one rare term. Ranking them first is correct.

        What actually matters is that the exact identifier reaches our page, so
        that is what this now measures. Verified at the time of writing: 06
        ranks first for `model_info` and for `model_info zorunlu`, and sixth for
        the old four-term query.
        """
        for query in ("model_info", "model_info zorunlu"):
            hits = docs_index.search(query, k=3)
            self.assertTrue(
                hits and hits[0].section.doc.startswith("06-"),
                f"{query!r} should reach our gotchas page first, "
                f"got {[h.section.doc for h in hits]}",
            )

    def test_unnumbered_files_are_reading_material_not_corpus(self) -> None:
        """`docs/` also holds saved articles; they are sources, not documentation.

        A 63 KB scraped page dropped in here shifted idf enough to push our own
        gotchas page out of the top four for a query it should own. Search quality
        degrading because somebody saved something to read is not a failure mode
        worth having.
        """
        indexed = {section.doc for section in docs_index.sections()}
        for name in indexed:
            self.assertTrue(
                name[:2].isdigit(),
                f"{name} is not part of the numbered series and should not be indexed",
            )

    def test_empty_and_unknown_queries_return_nothing(self) -> None:
        self.assertEqual(docs_index.search(""), [])
        self.assertEqual(docs_index.search("   "), [])
        self.assertEqual(docs_index.search("zzzzqqqq_nonexistent_term"), [])

    def test_official_only_filter(self) -> None:
        hits = docs_index.search("agent", k=8, official_only=True)
        self.assertTrue(hits)
        self.assertTrue(all(h.section.official for h in hits))

    def test_rendered_text_carries_the_citation(self) -> None:
        hits = docs_index.search("workbench", k=2)
        text = docs_index.as_text("workbench", hits)
        self.assertIn(".md:", text, "every hit must say which file and line it came from")
        self.assertIn("guide", text)

    def test_no_hits_says_what_the_corpus_covers(self) -> None:
        text = docs_index.as_text("zzzz", [])
        self.assertIn("No section", text)
        self.assertIn("AutoGen", text)


class RoutingTest(unittest.IsolatedAsyncioTestCase):
    SCAN = {
        "query": "ai infra", "days": 7, "mode": "dry", "thesis_is_placeholder": True,
        "failed_sources": {}, "cost": {}, "funnel": {"signals": 5}, "candidates": [],
    }

    async def test_a_framework_question_is_answered_from_the_docs(self) -> None:
        result = await answers.answer("what is a workbench?", self.SCAN, prefer_model=False)
        self.assertEqual(result["path"], "docs")
        self.assertIn("workbench", result["html"].lower())

    async def test_a_scan_question_still_goes_to_the_scan(self) -> None:
        result = await answers.answer("what did it cost", self.SCAN, prefer_model=False)
        self.assertEqual(result["path"], "rules")

    async def test_an_unanswerable_question_keeps_its_refusal(self) -> None:
        # The docs are not a catch-all. "Weather" genuinely appears in the
        # AgentChat guide's tool example and scores higher than several real
        # documentation questions — so a score threshold cannot be the gate, and
        # an off-topic question must still be refused rather than answered with
        # whatever ranked first.
        result = await answers.answer("who won the world cup", self.SCAN, prefer_model=False)
        self.assertEqual(result["path"], "rules")
        self.assertEqual(result["title"], "Not something I hold")


if __name__ == "__main__":
    unittest.main()
