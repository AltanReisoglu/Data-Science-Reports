"""Markdown memory: two files with two jobs, and search that stays fresh."""

from __future__ import annotations

import unittest

import config
import memory


class MemoryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # The state dir is per-run (tests/__init__), but files persist across tests
        # in one run, so each case starts from a clean workspace.
        for path in memory.files():
            path.unlink()
        memory.INDEX.fingerprint = ()
        memory.INDEX.sections = None


class LayoutTests(MemoryTestCase):
    def test_bootstrap_writes_a_file_that_explains_itself(self) -> None:
        path = memory.bootstrap()
        body = path.read_text(encoding="utf-8")
        self.assertIn("Loaded into the prompt", body)
        self.assertIn("memory/YYYY-MM-DD.md", body)

    def test_an_empty_curated_file_costs_no_tokens(self) -> None:
        """A header with nothing under it must not be injected on every turn."""
        memory.bootstrap()
        self.assertEqual(memory.preamble(), "")

    def test_a_promoted_fact_lands_in_the_preamble(self) -> None:
        memory.promote("Altan prefers sources over scores.", section="Preferences")
        self.assertIn("sources over scores", memory.preamble())

    def test_notes_go_to_a_dated_file_and_not_into_the_preamble(self) -> None:
        """Daily notes are searchable, not injected — that is the whole cost model."""
        memory.bootstrap()
        memory.note("Acme raised a seed round, per SEC Form D.", tag="scan")

        self.assertTrue(memory.daily_file().exists())
        self.assertNotIn("Acme", memory.preamble())

    def test_forget_removes_a_promoted_line(self) -> None:
        memory.promote("Wrong fact about Acme.")
        self.assertEqual(memory.forget("Wrong fact"), 1)
        self.assertNotIn("Wrong fact", memory.preamble())

    def test_forget_leaves_headings_alone(self) -> None:
        memory.promote("something")
        before = memory.memory_file().read_text(encoding="utf-8")
        memory.forget("Facts")
        after = memory.memory_file().read_text(encoding="utf-8")
        self.assertIn("## Facts", after)
        self.assertEqual(before.count("## Facts"), after.count("## Facts"))


class SearchTests(MemoryTestCase):
    def test_a_note_is_findable(self) -> None:
        memory.note("Acme Robotics raised a seed round led by Northgate.", tag="scan")
        hits = memory.search("acme robotics seed")
        self.assertTrue(hits)
        self.assertIn("Acme Robotics", hits[0].section.body)

    def test_hits_carry_a_file_and_a_line(self) -> None:
        """Same rule as everywhere else: an answer that cannot cite is not an answer."""
        memory.note("SEC Form D filings are the earliest signal we have.")
        hit = memory.search("form d earliest signal")[0]
        self.assertTrue(hit.section.doc.endswith(".md"))
        self.assertGreater(hit.section.line, 0)

    def test_a_note_written_now_is_searchable_now(self) -> None:
        """Unlike docs, memory changes constantly, so the index cannot be frozen."""
        self.assertEqual(memory.search("quantum widgets"), [])
        memory.note("Quantum Widgets is on the watchlist.")
        self.assertTrue(memory.search("quantum widgets"))

    def test_no_match_says_so_instead_of_guessing(self) -> None:
        memory.note("unrelated content")
        text = memory.as_text("fintech in lisbon", memory.search("fintech in lisbon"))
        self.assertIn("Nothing in memory matches", text)
        self.assertIn("rather than reconstructing", text)

    def test_memory_and_docs_are_separate_surfaces(self) -> None:
        """A memory search must not return the AutoGen guides, and vice versa."""
        import docs_index

        memory.note("GraphFlow is the engine we measured as losing branches.")
        memory_docs = {hit.section.doc for hit in memory.search("graphflow", k=5)}
        self.assertTrue(memory_docs)
        self.assertFalse(any(d.startswith("05-") or d.startswith("08-") for d in memory_docs))
        self.assertTrue(docs_index.search("graphflow", k=1))


class GetTests(MemoryTestCase):
    def test_get_returns_numbered_lines(self) -> None:
        memory.note("first fact\nsecond fact")
        name = memory.daily_file().relative_to(config.WORKSPACE)
        body = memory.get(str(name))
        self.assertIn("first fact", body)
        self.assertRegex(body, r"\s+\d+ \|")

    def test_get_honours_a_line_range(self) -> None:
        memory.note("alpha\nbeta\ngamma")
        name = str(memory.daily_file().relative_to(config.WORKSPACE))
        whole = memory.get(name)
        part = memory.get(name, 1, 2)
        self.assertLess(len(part), len(whole))

    def test_paths_outside_the_workspace_are_refused(self) -> None:
        """The path comes from a model, which is not a trusted caller."""
        self.assertIn("Refused", memory.get("../../../etc/passwd"))
        self.assertIn("Refused", memory.get("/etc/passwd"))

    def test_a_missing_file_says_so(self) -> None:
        self.assertIn("No such memory file", memory.get("memory/1999-01-01.md"))


if __name__ == "__main__":
    unittest.main()
