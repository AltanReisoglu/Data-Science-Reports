"""Memory as plain Markdown files in a workspace. No hidden store.

This is OpenClaw's design (docs/13 §7) and the reason to copy it is the first
sentence of it: *"hatırlamayı workspace'te düz Markdown dosyaları yazarak yapar;
gizli durum yoktur."* Memory you can open in an editor is memory you can correct,
review, delete and put under version control if you want to. A vector store you
cannot read is a place for a wrong fact to live forever.

Two files, two different jobs, and conflating them is the mistake:

* **`MEMORY.md`** — curated, small, **loaded into the prompt at session start**.
  This is what the agent always knows. It costs tokens on every single turn, so
  it stays short by design.
* **`memory/YYYY-MM-DD.md`** — daily notes, **indexed and never auto-injected**.
  This is what the agent can *look up*. Growth here is free because nothing is
  paid for until something is searched.

Promotion from the second to the first is a deliberate act (`promote`), not a
background process. OpenClaw has "dreaming" for automatic consolidation; that is
out of scope here, and honestly so — an automatic promoter that scores its own
recall is a system that quietly decides what the agent believes.

### Search

`memory_search` runs `docs_index`'s TF-IDF over the workspace instead of `docs/`.
Not a second engine: the section splitter, scorer and citation shape are the same
code, which is why memory hits arrive with a file and a line like every other
answer in this system.

It is **lexical, not hybrid.** OpenClaw's is vector + keyword. The trade is the
one `docs_index` already argues: an index that needs an embedding endpoint fails
when the provider fails, and these notes are dense with exact identifiers —
company names, tickers, tool names — which is where lexical is strongest.
Paraphrase recall is the known cost, and the upgrade path is unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import config
import docs_index

HEADING = re.compile(r"^#{1,6} ")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _today() -> str:
    return date.today().isoformat()


# --------------------------------------------------------------------------- files


def memory_file() -> Path:
    return config.MEMORY_FILE


def daily_file(day: str | None = None) -> Path:
    return config.MEMORY_DIR / f"{day or _today()}.md"


def files() -> list[Path]:
    """Everything indexable: the curated file plus every daily note."""
    found = []
    if memory_file().exists():
        found.append(memory_file())
    found.extend(sorted(config.MEMORY_DIR.glob("*.md")))
    return found


def bootstrap() -> Path:
    """Create `MEMORY.md` with a header explaining what it is for.

    An empty file gives no signal about how it should be used; a header does, and
    it is the first thing both the operator and the agent read.
    """
    path = memory_file()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# MEMORY\n\n"
            "Curated, long-lived memory for this agent. Loaded into the prompt at the\n"
            "start of every session, so keep it short — everything here is paid for on\n"
            "every turn.\n\n"
            "Day-to-day notes belong in `memory/YYYY-MM-DD.md`; they are searchable and\n"
            "are not injected. Move something up here only when the agent should always\n"
            "know it.\n\n"
            "## Facts\n\n"
            "## Preferences\n\n"
            "## Open questions\n",
            encoding="utf-8",
        )
    return path


# --------------------------------------------------------------------------- writing


def note(text: str, *, tag: str = "", day: str | None = None) -> Path:
    """Append a dated note. Cheap, unlimited, never injected — only searchable."""
    path = daily_file(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(f"# {day or _today()}\n\n", encoding="utf-8")
    stamp = _now().strftime("%H:%M")
    label = f" · {tag}" if tag else ""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {stamp}{label}\n\n{text.strip()}\n")
    return path


def promote(text: str, *, section: str = "Facts") -> Path:
    """Move something into the always-loaded file. A decision, not a heuristic."""
    path = bootstrap()
    body = path.read_text(encoding="utf-8")
    marker = f"## {section}"
    line = f"- {text.strip()}\n"
    if marker in body:
        head, _, tail = body.partition(marker + "\n")
        path.write_text(head + marker + "\n" + line + tail, encoding="utf-8")
    else:
        path.write_text(body.rstrip() + f"\n\n{marker}\n\n{line}", encoding="utf-8")
    return path


def forget(pattern: str) -> int:
    """Remove matching lines from `MEMORY.md`. Memory you cannot correct is a liability."""
    path = memory_file()
    if not path.exists():
        return 0
    kept, removed = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if pattern.lower() in line.lower() and line.strip().startswith("-"):
            removed += 1
            continue
        kept.append(line)
    if removed:
        path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return removed


# --------------------------------------------------------------------------- reading


ENTRY = re.compile(r"^\s*[-*+] ")


def preamble(limit: int = 4000) -> str:
    """`MEMORY.md` as it goes into the system prompt — entries only, or nothing.

    Two things are deliberately dropped. The file's own instructions ("keep it
    short", "notes belong in memory/") are addressed to the operator and would be
    paid for on every turn to tell the model nothing. And a heading with no
    entries under it is an empty promise: it invites the model to believe there
    are preferences on record when there are none.

    If nothing survives that filter the result is empty, and the caller adds no
    memory section at all rather than an encouraging blank one.
    """
    path = memory_file()
    if not path.exists():
        return ""

    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    pending: str | None = None
    for line in lines:
        if HEADING.match(line):
            # Top-level title is the file's name, not a section.
            pending = line.strip() if line.startswith("##") else None
            continue
        if ENTRY.match(line):
            if pending:
                out.append(pending)
                pending = None
            out.append(line.rstrip())

    return "\n".join(out)[:limit] if out else ""


def get(path: str, start: int = 1, end: int | None = None) -> str:
    """Read a memory file, or a line range of one. The `memory_get` half.

    The path is resolved inside the workspace and rejected if it escapes: it
    arrives from a model, which is not a trusted caller.
    """
    workspace = config.WORKSPACE.resolve()
    candidate = (workspace / path).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError:
        return f"Refused: {path!r} is outside the memory workspace."
    if not candidate.is_file():
        return f"No such memory file: {path!r}."

    lines = candidate.read_text(encoding="utf-8").splitlines()
    first = max(1, start)
    last = min(len(lines), end or len(lines))
    body = "\n".join(f"{n:>4} | {lines[n - 1]}" for n in range(first, last + 1))
    return f"{candidate.relative_to(workspace)} lines {first}-{last}:\n{body}"


# --------------------------------------------------------------------------- search


@dataclass
class MemoryIndex:
    """TF-IDF over the workspace, rebuilt when a file changes.

    Daily notes are appended to constantly, so unlike `docs_index` this cannot be
    cached for the life of the process. The fingerprint is (path, mtime, size) for
    every file — cheap enough to check on each search, and it means a note written
    one second ago is findable in the next question.
    """

    sections: list = None  # type: ignore[assignment]
    idf: dict = None       # type: ignore[assignment]
    fingerprint: tuple = ()

    def refresh(self) -> None:
        current = tuple(
            (str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files()
        )
        if current == self.fingerprint and self.sections is not None:
            return
        self.sections, self.idf = docs_index.build_corpus(files())
        self.fingerprint = current

    def search(self, query: str, k: int = 5) -> list[docs_index.Hit]:
        self.refresh()
        return docs_index.rank(self.sections or [], self.idf or {}, query, k)


INDEX = MemoryIndex()


def search(query: str, k: int = 5) -> list[docs_index.Hit]:
    return INDEX.search(query, k)


def as_text(query: str, hits: list[docs_index.Hit]) -> str:
    """Hits as the model sees them — with the file and line, like every other answer."""
    if not hits:
        return (
            f"Nothing in memory matches {query!r}. Memory holds MEMORY.md and dated "
            f"notes under memory/; if this was never written down, say so rather than "
            f"reconstructing it."
        )
    out = [f"{len(hits)} memory hit(s) for {query!r}:\n"]
    for hit in hits:
        section = hit.section
        out.append(
            f"--- {section.doc}:{section.line} · {section.title}\n{section.body[:1200]}\n"
        )
    return "\n".join(out)


def stats() -> dict[str, Any]:
    INDEX.refresh()
    return {
        "workspace": str(config.WORKSPACE),
        "curated_bytes": memory_file().stat().st_size if memory_file().exists() else 0,
        "daily_notes": len(list(config.MEMORY_DIR.glob("*.md"))),
        "sections": len(INDEX.sections or []),
    }


__all__ = [
    "INDEX",
    "MemoryIndex",
    "as_text",
    "bootstrap",
    "daily_file",
    "files",
    "forget",
    "get",
    "memory_file",
    "note",
    "preamble",
    "promote",
    "search",
    "stats",
]
