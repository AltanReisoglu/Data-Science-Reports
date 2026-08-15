"""Data contract — the single shared language of the system.

This module is the code counterpart of `docs/04-vc-agentic-akis.md` §2. No free
text travels between agents; everything is enforced by Pydantic.

Two fields are deliberately mandatory, and both are design decisions:

* ``Source.url`` — an unsourced claim becomes impossible at the schema level
  (domain §3.3: "the product is not a score, it is an evidence package").
* ``Score.missing_data`` — "a zero result always owes an explanation". A low
  score and an absence of information are different things, and the system is
  required to say which one it is.

``raw`` fields keep the original API payload so that "where did this score come
from" can be answered without descending into the audit log.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------- types

SignalKind = Literal[
    "funding_round",    # SEC Form D, press
    "product_launch",   # Show HN, Product Hunt
    "hiring",           # careers page, job posting
    "repo_momentum",    # GitHub star/commit velocity
    "academic",         # arXiv
    "news",             # general
]

Confidence = Literal["official", "primary", "secondary"]
Decision = Literal["watch", "review", "skip"]
Stage = Literal["new", "watch", "review", "rejected", "portfolio"]


def now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- schemas


class Source(BaseModel):
    """A single verifiable piece of evidence."""

    name: str                     # "sec_form_d" | "hn" | "github" | "arxiv"
    url: str                      # MANDATORY — there is no unsourced claim
    retrieved_at: datetime = Field(default_factory=now)
    confidence: Confidence = "secondary"

    @field_validator("url")
    @classmethod
    def _url_is_resolvable(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"Source.url must be a resolvable link: {v!r}")
        return v


class Signal(BaseModel):
    """A single event observed in the outside world."""

    kind: SignalKind
    summary: str
    date: datetime
    source: Source
    raw: dict[str, Any] = Field(default_factory=dict)

    # Name/domain guessed by the collector; entity resolution consumes these.
    candidate_name: str | None = None
    candidate_domain: str | None = None

    @property
    def fingerprint(self) -> str:
        """Deterministic key that keeps the same event from entering twice."""
        core = f"{self.source.name}|{self.source.url}|{self.summary[:120]}"
        return hashlib.sha1(core.encode()).hexdigest()[:16]


class Company(BaseModel):
    """An entity produced by normalization, with its signals merged."""

    name: str
    domain: str | None = None
    description: str | None = None
    sectors: list[str] = Field(default_factory=list)
    country: str | None = None
    founded_year: int | None = None
    signals: list[Signal] = Field(default_factory=list)
    github: str | None = None
    founders: list[str] = Field(default_factory=list)  # public names only
    stage: Stage = "new"

    @property
    def key(self) -> str:
        """Entity resolution key: domain > github org > normalized name."""
        if self.domain:
            return f"domain:{self.domain.lower()}"
        if self.github:
            return f"gh:{self.github.lower()}"
        return f"name:{normalize_name(self.name)}"

    @property
    def sources(self) -> list[Source]:
        return [s.source for s in self.signals]

    def one_liner(self) -> str:
        kinds = ", ".join(sorted({s.kind for s in self.signals}))
        return f"{self.name} ({self.domain or '—'}) · {len(self.signals)} signals [{kinds}]"


class BranchResult(BaseModel):
    """One branch of the parallel enrichment step.

    ``succeeded=False`` is not an error, it is *recorded absence of information* —
    it flows straight into ``Score.missing_data``. See `graph.py` and docs/04 §7.
    """

    branch: Literal["technical", "market", "team"]
    succeeded: bool
    text: str = ""
    error: str | None = None
    sources: list[Source] = Field(default_factory=list)


class Score(BaseModel):
    """A ranking device, not a decision. Every axis owes one sentence."""

    thesis_fit: int = Field(ge=0, le=5)
    team: int = Field(ge=0, le=5)
    momentum: int = Field(ge=0, le=5)
    technical_depth: int = Field(ge=0, le=5)
    timing: int = Field(ge=0, le=5)
    rationale: dict[str, str] = Field(default_factory=dict)  # axis -> one sentence
    missing_data: list[str]                                  # MANDATORY (domain §3.1)
    decision: Decision

    @property
    def total(self) -> int:
        return self.thesis_fit + self.team + self.momentum + self.technical_depth + self.timing

    @property
    def reliability(self) -> str:
        """How much of the score rests on real data — as important as the score."""
        n = len(self.missing_data)
        return "high" if n == 0 else ("medium" if n <= 2 else "low")


class InvestmentMemo(BaseModel):
    """The only output a human reads."""

    company_name: str
    thesis: str
    summary: str
    risks: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)  # to ask the founders
    references: list[Source] = Field(default_factory=list)


class Candidate(BaseModel):
    """The record that carries a company down through the funnel."""

    company: Company
    triage_passed: bool = False
    triage_rationale: str = ""
    branches: list[BranchResult] = Field(default_factory=list)
    score: Score | None = None
    memo: InvestmentMemo | None = None

    @property
    def failed_branches(self) -> list[str]:
        return [b.branch for b in self.branches if not b.succeeded]


# --------------------------------------------------------------------------- helpers

_SUFFIX = re.compile(
    r"\b(inc|inc\.|llc|ltd|limited|corp|corporation|co|gmbh|bv|ab|oy|as|sa|plc|labs?|ai|io)\b",
    re.IGNORECASE,
)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_name(name: str) -> str:
    """'Acme Labs, Inc.' -> 'acme' — the last resort of entity resolution.

    Deliberately aggressive: this function only runs when neither a domain nor a
    GitHub org is known, and normalization *suggests* a match rather than
    performing the merge on its own (see `normalize.py`).
    """
    s = name.lower().strip()
    s = _SUFFIX.sub(" ", s)
    s = _NON_ALNUM.sub(" ", s).strip()
    return s or name.lower().strip()


def registrable_domain(url_or_domain: str) -> str | None:
    """Extract the registrable root from a URL or host (drops www and subdomains)."""
    if not url_or_domain:
        return None
    s = url_or_domain.strip().lower()
    s = re.sub(r"^https?://", "", s)
    s = s.split("/")[0].split("?")[0]
    s = re.sub(r"^www\.", "", s)
    if "." not in s:
        return None
    parts = s.split(".")
    # Keep three labels for two-part TLDs such as co.uk / com.tr.
    if len(parts) >= 3 and parts[-2] in {"co", "com", "org", "net", "gov", "edu"}:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])
