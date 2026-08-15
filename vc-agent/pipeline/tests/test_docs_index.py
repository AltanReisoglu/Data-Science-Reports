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
        hits = docs_index.search("neden AutoGen seçildi", k=3)
        self.assertTrue(hits)
        self.assertTrue(
            any(h.section.doc.startswith(("09-", "07-", "01-")) for h in hits),
            f"expected our own docs, got {[h.section.doc for h in hits]}",
        )

    def test_a_measured_gotcha_is_findable(self) -> None:
        hits = docs_index.search("model_info required openai compatible", k=4)
        self.assertTrue(any(h.section.doc.startswith("06-") for h in hits))

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
