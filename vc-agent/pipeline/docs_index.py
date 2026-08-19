"""Search over `docs/` — the project's own reasoning *and* the AutoGen guides.

The chat could answer questions about a scan but not about the thing the scan
runs on. `docs/` already holds both halves: our design and measurements (01–04,
06, 07, 09) and the verbatim official guides (05 core, 08 AgentChat). Together
that is 1.18 MB — far past any prompt — so it has to be searched rather than
pasted.

**Lexical, not embeddings, and that is a choice.** The endpoint does serve
`qwen3-embedding-8b`, so vector search is available. It is not used here because:

* the index would be a second thing that can go stale, and a stale index answers
  confidently from an old document;
* embedding 675 sections needs the endpoint to be up, which would make a
  *documentation* lookup fail when the model provider fails;
* these documents are dense with exact identifiers — `ClosureAgent`,
  `model_info`, `activation_condition` — and exact terms are where lexical
  scoring is strongest and embeddings are weakest.

TF-IDF over heading-delimited sections, computed at import, no dependencies. If
recall on conceptual paraphrases turns out to be the limit, embeddings are the
upgrade path and the section split stays the same.

**Every hit carries its citation.** Same rule as everywhere else in this system:
an answer that cannot say where it came from does not get to be an answer.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"

# Which file is what, so a hit can say whose claim it is. The distinction matters:
# 05, 08, 20 and 21 are Microsoft's words, everything else is ours. That split is
# most of the value here: a hit that quotes the vendor and a hit that quotes us
# carry different weight, and a reader who cannot tell them apart is worse off
# than one with no citation at all.
PROVENANCE = {
    "05-autogen-core-user-guide.md": "AutoGen Core guide (official, verbatim)",
    "08-autogen-agentchat-user-guide.md": "AutoGen AgentChat guide (official, verbatim)",
    "01-autogen-kaynak-haritasi.md": "our research map",
    "02-autogen-el-kitabi.md": "our API handbook (verified against v0.7.5)",
    "03-vc-domain-plani.md": "our domain plan",
    "04-vc-agentic-akis.md": "our architecture plan",
    "06-autogen-incelikleri.md": "our measured gotchas",
    "07-kod-rehberi.md": "our code guide",
    "09-framework-karsilastirma.md": "our framework comparison",
    "10-agentchat-turkce.md": "our Turkish guide to AgentChat",
    "11-core-guide-turkce.md": "our Turkish guide to autogen_core",
    "12-autogen-bastan-sona.md": "our end-to-end AutoGen walkthrough",
    "14-autogen-protokoller-ve-farklar.md": "our protocol + framework analysis",
    "15-vc-gateway-mimarisi.md": "our gateway architecture (OpenClaw shape, AutoGen engine)",
    "16-openclaw-enterprise-ilham.md": "our enterprise reading of OpenClaw (what transfers, what does not)",
    "17-autogen-openclaw-sirket-plani.md": "our build/borrow/deploy decision for the company",
    "18-task-manager-ve-dayanikli-yurutme.md": "our reading of OpenClaw's scheduler, task ledger, flow engine and concurrency model",
    "20-maf-user-guide.md": "Microsoft Agent Framework user guide (official, verbatim)",
    "21-maf-tasarim-kararlari.md": "Agent Framework design records — ADRs (official, verbatim)",
    "22-maf-turkce.md": "our Turkish guide to Agent Framework, measured against AutoGen",
    "github-starred-repos.md": "starred-repo inventory",
}

_WORD = re.compile(r"[a-z0-9_]+")
_FENCE = re.compile(r"^(`{3,})(.*)$")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


@dataclass
class Section:
    doc: str
    title: str
    breadcrumb: str
    body: str
    line: int
    anchor: str
    terms: Counter = field(default_factory=Counter)

    @property
    def provenance(self) -> str:
        return PROVENANCE.get(self.doc, self.doc)

    @property
    def official(self) -> bool:
        return "official" in self.provenance

    def snippet(self, limit: int = 420) -> str:
        text = re.sub(r"\s+", " ", self.body).strip()
        return text[:limit] + ("…" if len(text) > limit else "")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def _split(path: Path) -> list[Section]:
    """Split a markdown file into sections at its headings.

    Fence tracking follows CommonMark — a `#` inside a code block is a comment,
    not a heading, and these documents are full of both.
    """
    # Commented-out drafts are removed before parsing, for two reasons. They are
    # not documentation, so indexing them would return text the authors withdrew.
    # And one of them is actively harmful: `tutorial/agents.ipynb` carries a
    # commented block whose ```python fence is closed with a single backtick —
    # upstream's typo, copied faithfully by the verbatim fetch. Left in, it holds
    # the fence open and swallows the next three page headings, so `## Teams`
    # stops being a section of its own.
    raw = _HTML_COMMENT.sub("", path.read_text(encoding="utf-8"))
    lines = raw.split("\n")
    sections: list[Section] = []
    stack: dict[int, str] = {}
    current: Section | None = None
    buffer: list[str] = []
    fence = 0

    def flush() -> None:
        if current is not None:
            current.body = "\n".join(buffer).strip()
            sections.append(current)

    for number, line in enumerate(lines, 1):
        match = _FENCE.match(line.strip())
        if match:
            ticks, info = len(match.group(1)), match.group(2).strip()
            if fence == 0:
                fence = ticks
            elif ticks >= fence and not info:
                fence = 0
            buffer.append(line)
            continue
        if fence:
            buffer.append(line)
            continue

        heading = re.match(r"^(#{1,6}) (.+)$", line)
        if heading:
            flush()
            level, title = len(heading.group(1)), heading.group(2).strip()
            stack = {k: v for k, v in stack.items() if k < level}
            stack[level] = title
            current = Section(
                doc=path.name,
                title=title,
                breadcrumb=" › ".join(stack[k] for k in sorted(stack)),
                body="",
                line=number,
                anchor=_slug(title),
            )
            buffer = []
            continue
        buffer.append(line)

    flush()
    return [s for s in sections if s.body or s.title]


def build_corpus(paths: list[Path]) -> tuple[list[Section], dict[str, float]]:
    """Split the given files into sections and compute idf across them.

    Extracted from `_corpus` when memory needed the same machinery over a
    different root. Memory notes and documentation are indexed *separately* on
    purpose: one is the project's reasoning, the other is what the operator asked
    to be remembered, and a search for "what did we decide about arXiv" should be
    able to ask each without the other drowning it out.
    """
    sections: list[Section] = []
    for path in paths:
        sections.extend(_split(path))

    for section in sections:
        # The heading is repeated so a term in the title outweighs the same term
        # buried in prose — in these documents the heading is the topic.
        section.terms = Counter(
            _tokens(section.body) + _tokens(section.title) * 3 + _tokens(section.breadcrumb)
        )

    document_frequency: Counter = Counter()
    for section in sections:
        document_frequency.update(set(section.terms))

    total = len(sections) or 1
    idf = {
        term: math.log(1 + total / (1 + count))
        for term, count in document_frequency.items()
    }
    return sections, idf


def _indexable() -> list[Path]:
    """The project's own numbered documents, and only those.

    `docs/` is also where reading material lands — a saved article, a scraped
    page, notes pasted in to be read later. Those are legitimately there and are
    not documentation of this system, and indexing them changes the answers: a
    63 KB blog dump shifted enough idf weight to push our own measured-gotchas
    page out of the top four for a query it should own. The test caught it, which
    is the only reason it did not just quietly degrade search.

    So the corpus is the numbered series. Anything else in `docs/` is a source to
    read, not a document to be searched.
    """
    return sorted(p for p in DOCS.glob("*.md") if p.name[:2].isdigit())


@lru_cache(maxsize=1)
def _corpus() -> tuple[list[Section], dict[str, float]]:
    return build_corpus(_indexable())


def sections() -> list[Section]:
    return _corpus()[0]


@dataclass
class Hit:
    section: Section
    score: float

    def as_dict(self) -> dict:
        return {
            "doc": self.section.doc,
            "provenance": self.section.provenance,
            "title": self.section.title,
            "breadcrumb": self.section.breadcrumb,
            "line": self.section.line,
            "snippet": self.section.snippet(),
            "score": round(self.score, 3),
        }


def rank(
    corpus: list[Section],
    idf: dict[str, float],
    query: str,
    k: int = 5,
    *,
    official_only: bool = False,
) -> list[Hit]:
    """The scorer, over any corpus. `search` is this against the docs corpus."""
    wanted = _tokens(query)
    if not wanted:
        return []

    query_terms = Counter(wanted)
    scored: list[Hit] = []
    for section in corpus:
        if official_only and not section.official:
            continue
        score = 0.0
        length = sum(section.terms.values()) or 1
        for term, times in query_terms.items():
            if term not in section.terms:
                continue
            # tf normalised by section length, so a long chapter does not win by
            # size alone, times idf so common words carry little.
            tf = section.terms[term] / length
            score += times * tf * idf.get(term, 0.0) * 100
        if score > 0:
            scored.append(Hit(section=section, score=score))

    scored.sort(key=lambda hit: hit.score, reverse=True)
    return scored[:k]


def search(query: str, k: int = 5, *, official_only: bool = False) -> list[Hit]:
    """Rank sections against a query. Empty result means no section matched."""
    corpus, idf = _corpus()
    return rank(corpus, idf, query, k, official_only=official_only)


def as_text(query: str, hits: list[Hit]) -> str:
    """Hits as the model sees them: content plus where each line came from."""
    if not hits:
        return (
            f"No section in docs/ matches {query!r}. The documents cover AutoGen "
            f"(core and AgentChat guides, verbatim) and this project's own design, "
            f"measurements and code."
        )
    out = [f"{len(hits)} section(s) for {query!r}:\n"]
    for hit in hits:
        section = hit.section
        out.append(
            f"--- {section.doc}:{section.line} · {section.provenance}\n"
            f"{section.breadcrumb}\n\n{section.body[:2200]}\n"
        )
    return "\n".join(out)


def stats() -> dict:
    corpus, idf = _corpus()
    per_doc: Counter = Counter(s.doc for s in corpus)
    return {
        "sections": len(corpus),
        "vocabulary": len(idf),
        "documents": dict(sorted(per_doc.items())),
    }
