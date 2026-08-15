"""Layer 2 — entity resolution and deduplication. Still no LLM.

Signals are attached to companies in a fixed order, ranked by strength of evidence:

    domain  >  GitHub org  >  normalized name (fuzzy)

When in doubt, **do not merge** (docs/04 §10, "entity collision"). Wrongly
merging two records costs more than leaving them apart: once merged, the
evidence package is contaminated and it is no longer recoverable which signal
belonged to whom.

The ChromaDB-backed semantic "have we seen this company before" check was **not
built** — the package is not installed. In its place there is a deterministic key
plus `difflib` fuzzy matching; once ChromaDB is available a vector lookup can
replace `name_similarity()`.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from schemas import Company, Signal, normalize_name

# Fuzzy-match threshold. 0.92 is high on purpose: we prefer the risk of leaving
# two records apart over the risk of merging them wrongly.
SIMILARITY_THRESHOLD = 0.92


def name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio()


def _keys_for(signal: Signal) -> list[str]:
    """Candidate keys a signal may attach to — strongest first."""
    keys: list[str] = []
    if signal.candidate_domain:
        keys.append(f"domain:{signal.candidate_domain.lower()}")
    if signal.source.name == "github" and signal.raw.get("owner") and signal.raw.get("is_org"):
        keys.append(f"gh:{signal.raw['owner'].lower()}")
    if signal.candidate_name:
        keys.append(f"name:{normalize_name(signal.candidate_name)}")
    return keys


def resolve(signals: list[Signal]) -> list[Company]:
    """Turn a signal list into a deduplicated list of companies."""
    companies: dict[str, Company] = {}
    seen_signals: set[str] = set()

    # Signals with a strong key are processed first so weaker ones can attach.
    ordered = sorted(
        signals,
        key=lambda s: (0 if s.candidate_domain else (1 if s.candidate_name else 2)),
    )

    for signal in ordered:
        if signal.fingerprint in seen_signals:  # same event seen from two sources
            continue
        seen_signals.add(signal.fingerprint)

        keys = _keys_for(signal)
        if not keys:
            # A signal whose owner cannot be resolved is not discarded, but it is
            # not attached to a company either — evidence without context misleads.
            continue

        target = _match(companies, keys, signal)
        if target is None:
            company = Company(
                name=signal.candidate_name or (signal.candidate_domain or "unknown"),
                domain=signal.candidate_domain,
                github=(
                    signal.raw.get("owner")
                    if (signal.source.name == "github" and signal.raw.get("is_org"))
                    else None
                ),
                description=signal.raw.get("description"),
                sectors=list(signal.raw.get("topics") or []),
                signals=[signal],
            )
            companies[company.key] = company
        else:
            target.signals.append(signal)
            _enrich(target, signal)

    return list(companies.values())


def _match(companies: dict[str, Company], keys: list[str], signal: Signal) -> Company | None:
    for key in keys:
        if key in companies:
            return companies[key]

    # Fuzzy matching runs on the name alone, and only at a high threshold.
    if signal.candidate_name:
        for company in companies.values():
            if name_similarity(company.name, signal.candidate_name) >= SIMILARITY_THRESHOLD:
                return company
    return None


def _enrich(company: Company, signal: Signal) -> None:
    """Let a new signal fill in fields the company is still missing."""
    if not company.domain and signal.candidate_domain:
        company.domain = signal.candidate_domain
    if not company.github and signal.source.name == "github" and signal.raw.get("is_org"):
        company.github = signal.raw.get("owner")
    if not company.description and signal.raw.get("description"):
        company.description = signal.raw["description"]
    for topic in signal.raw.get("topics") or []:
        if topic not in company.sectors:
            company.sectors.append(topic)
