"""Layer 3 agents: three parallel analysts, a risk auditor, a scorer, a memo writer.

Model tiering is natural in AutoGen — every ``AssistantAgent`` takes its own
``model_client`` (docs/04 §3). The tier assignment here *is* the funnel economics:

    technical / market / team analysts  -> mid
    risk auditor, scorer                -> mid
    memo writer (~5 per day)            -> strong

Every agent is given a meaningful ``description``. That is not decoration: an
empty description makes routing blind, which is one of the measured traps in
docs/04 §7.

Dry-mode scripts deliberately contain **no tool calls**. A replayed tool call
would execute the real tool and reach the network, which would break the
"deterministic and offline" property of dry mode. Tools are exercised in live
mode; the dry path measures control flow.
"""

from __future__ import annotations

import config
import engine
from agents import tools
from schemas import Company, Score

# --------------------------------------------------------------------------- prompts

_EVIDENCE_RULE = """
Every claim must carry the source URL it came from. If you do not have a source
for something, do not write it: write instead which specific fact is missing.
"A zero result always owes an explanation" — state where you looked.
"""

TECHNICAL_SYSTEM = f"""You are the technical analyst. Judge engineering substance:
repository momentum, code quality signals, license, release cadence, whether the
technology is genuinely differentiated or a thin wrapper.
Begin your answer with "TECHNICAL:".{_EVIDENCE_RULE}"""

MARKET_SYSTEM = f"""You are the market analyst. Judge timing, competition, and
observable demand: launch reception, incumbents, why now.
Begin your answer with "MARKET:".{_EVIDENCE_RULE}"""

TEAM_SYSTEM = f"""You are the team analyst. Judge the founders on public evidence
only: prior companies, repositories, papers, talks. Never speculate about private
attributes, and use only publicly published names.
Begin your answer with "TEAM:".{_EVIDENCE_RULE}"""

RISK_SYSTEM = f"""You are the risk auditor. You receive three analyses. Your job
is to find the CONTRADICTIONS between them and the claims none of them supported
with a source. List what is missing explicitly.
Begin your answer with "RISK:".{_EVIDENCE_RULE}"""

SCORER_SYSTEM = """You are the scorer. Apply the fixed rubric; do not score on
impression. Each axis is 0-5:

  thesis_fit      how well it matches the thesis below
  team            founder evidence (0 = no public trace found)
  momentum        growth signals over the observed window
  technical_depth engineering substance beyond a wrapper
  timing          why now

Thesis: {thesis}

Rules that override your judgement:
- Missing information is NOT a low score. If an axis has no evidence, score it
  conservatively AND name it in missing_data.
- rationale must contain one sentence per axis, referring to a source.
- decision: "review" for strong fits, "watch" if promising but unproven,
  "skip" only when there is evidence of contradiction with the thesis."""

MEMO_SYSTEM = """You are the memo writer. You produce the only artifact a human
reads, so it must be readable and every sentence must be defensible.

- summary: at most 5 sentences, what the company does and why it is interesting now.
- risks: concrete and falsifiable, not generic ("early stage" is not a risk).
- questions: what you would ask the founders, derived from what is missing.
- references: every source used.

A sentence without a source cannot enter the memo."""


# --------------------------------------------------------------------------- dry scripts

        
def _dry_analyst(prefix: str, company: Company) -> list[str]:
    return [
        f"{prefix}: {company.name} — dry mode, no model called. "
        f"{len(company.signals)} signals observed; "
        f"sources: {', '.join(sorted({s.source.name for s in company.signals})) or 'none'}."
    ]


def _dry_score(company: Company) -> list[str]:
    return [
        Score(
            thesis_fit=3,
            team=1,
            momentum=3,
            technical_depth=3,
            timing=3,
            rationale={
                "thesis_fit": "Dry mode: scored from signal metadata only.",
                "team": "No founder evidence gathered in dry mode.",
                "momentum": f"{len(company.signals)} signals within the window.",
                "technical_depth": "GitHub signal present." if company.github else "No repository signal.",
                "timing": "Signals fall inside the requested window.",
            },
            missing_data=["dry mode: no model was called, scores are placeholders"],
            decision="watch",
        ).model_dump_json()
    ]


# --------------------------------------------------------------------------- factories


def build_analysts(company: Company, ledger: engine.Ledger):
    """The three parallel enrichment branches.

    Each branch gets **its own client**, exactly as in `poc/desen_4_graphflow.py`:
    sharing one replay client across parallel branches would make the order in
    which they consume replies undefined, and in live mode separate clients are
    the natural way to run different models per branch.
    """
    from autogen_agentchat.agents import AssistantAgent

    technical = AssistantAgent(
        "TechnicalAnalyst",
        model_client=ledger.client("mid", _dry_analyst("TECHNICAL", company)),
        description="Assesses engineering substance from repositories and releases.",
        system_message=TECHNICAL_SYSTEM,
        tools=[tools.inspect_repository],
        reflect_on_tool_use=True,
    )
    market = AssistantAgent(
        "MarketAnalyst",
        model_client=ledger.client("mid", _dry_analyst("MARKET", company)),
        description="Assesses timing, competition and observable demand.",
        system_message=MARKET_SYSTEM,
        tools=[tools.search_market_chatter],
        reflect_on_tool_use=True,
    )
    team = AssistantAgent(
        "TeamAnalyst",
        model_client=ledger.client("mid", _dry_analyst("TEAM", company)),
        description="Assesses founders from public evidence only.",
        system_message=TEAM_SYSTEM,
        tools=[tools.founder_profile, tools.publication_trace],
        reflect_on_tool_use=True,
    )
    return technical, market, team


def build_risk_auditor(company: Company, ledger: engine.Ledger):
    from autogen_agentchat.agents import AssistantAgent

    return AssistantAgent(
        "RiskAuditor",
        model_client=ledger.client(
            "mid",
            [
                "RISK: dry mode — no contradictions computed; "
                "the branch outputs above were not cross-checked by a model."
            ],
        ),
        description="Cross-checks the three analyses for contradictions and unsourced claims.",
        system_message=RISK_SYSTEM,
    )


def build_scorer(company: Company, ledger: engine.Ledger, thesis: config.Thesis | None = None):
    from autogen_agentchat.agents import AssistantAgent

    thesis = thesis or config.THESIS
    return AssistantAgent(
        "Scorer",
        model_client=ledger.client("mid", _dry_score(company)),
        description="Applies the fixed rubric and produces a structured score.",
        system_message=SCORER_SYSTEM.format(thesis=thesis.as_prompt()),
        output_content_type=Score,
    )


def build_memo_writer(company: Company, ledger: engine.Ledger, dry_script: list[str]):
    """The strongest tier — roughly five calls a day, the only human-facing output."""
    from autogen_agentchat.agents import AssistantAgent
    from schemas import InvestmentMemo

    return AssistantAgent(
        "MemoWriter",
        model_client=ledger.client("strong", dry_script),
        description="Writes the investment memo a partner will read.",
        system_message=MEMO_SYSTEM,
        output_content_type=InvestmentMemo,
    )
