"""Single configuration point: thesis, model tiers, thresholds, rate limits.

**The thesis is version controlled** (docs/03 §10, "biased thesis" risk). The
rubric's calibration depends on it, so changing the thesis changes every score.
That is why it lives here as a tracked record rather than in an environment
variable.

LLM access is read from the environment. Without it the system runs in **dry
mode** — replay client, no network, deterministic. See `engine.py`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# The names this project reads, and the shorter names an endpoint is usually
# handed over with. Aliases are accepted so a working `.env` does not have to be
# rewritten to match one project's prefix.
_ALIASES = {
    "LLM_BASE_URL": ["VC_LLM_BASE_URL"],
    "LLM_API_KEY": ["VC_LLM_API_KEY"],
    # One endpoint usually means one model. It fills all three tiers; the funnel's
    # cost discipline then comes from how many calls each layer makes, not from
    # which model it calls.
    "LLM_MODEL_NAME": ["VC_MODEL_CHEAP", "VC_MODEL_MID", "VC_MODEL_STRONG"],
    "LLM_MODEL": ["VC_MODEL_CHEAP", "VC_MODEL_MID", "VC_MODEL_STRONG"],
}

# Searched in order; the first file found wins. `vc-agent/.env` is first because
# that is where a key naturally lands — next to the project, not inside a package.
_ENV_CANDIDATES = [ROOT.parent / ".env", ROOT / ".env", ROOT.parent.parent / ".env"]


def _load_env_file() -> None:
    """Read a `.env` if there is one, without overriding the real environment.

    A key typed into a shell dies with the shell, and a key pasted into a chat
    transcript stays on disk forever. A gitignored file beside the code is the
    least bad place for it. Values already exported win, so a one-off override on
    the command line still works.
    """
    # The test suite sets this before importing config: a configured endpoint
    # would otherwise make the whole suite call a real model.
    if os.getenv("VC_SKIP_ENV_FILE"):
        return
    env_file = next((path for path in _ENV_CANDIDATES if path.exists()), None)
    if env_file is None:
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(name, value)
        for alias in _ALIASES.get(name, []):
            os.environ.setdefault(alias, value)


_load_env_file()

# --------------------------------------------------------------------------- state

# State lives outside the repository, in a directory the operator owns. This is
# OpenClaw's layout and the reasoning is the same: sessions, transcripts, audit
# and scan output are *operational* state, not source. Keeping them in the tree
# means every run dirties the working copy and a `git add -A` can commit a
# transcript.
#
# One override, `VC_STATE_DIR`, moves the whole thing — which is what makes the
# suite hermetic and what lets a second agent run beside the first.
STATE = Path(os.getenv("VC_STATE_DIR", "~/.vcagent")).expanduser().resolve()

# An agent is a full persona boundary: its own workspace, sessions and memory.
# One process can serve several; the id selects which one this process owns.
AGENT_ID = os.getenv("VC_AGENT_ID", "main")

AGENT_DIR = STATE / "agents" / AGENT_ID
SESSIONS = AGENT_DIR / "sessions"
SESSION_INDEX = SESSIONS / "sessions.json"

WORKSPACE = STATE / "workspace" if AGENT_ID == "main" else STATE / f"workspace-{AGENT_ID}"
MEMORY_FILE = WORKSPACE / "MEMORY.md"      # curated, loaded at session start
MEMORY_DIR = WORKSPACE / "memory"          # daily notes, indexed, not injected

CACHE = STATE / "cache"
OUTPUT = STATE / "scans"
AUDIT_LOG = STATE / "state" / "audit.jsonl"

# Where state used to live. Kept as a read path so an existing checkout does not
# appear to have lost its scans; `_migrate_legacy_state` copies them across once.
DATA = ROOT / "data"

for _d in (STATE, AGENT_DIR, SESSIONS, WORKSPACE, MEMORY_DIR, CACHE, OUTPUT, AUDIT_LOG.parent):
    _d.mkdir(parents=True, exist_ok=True)


def _migrate_legacy_state() -> None:
    """Copy scans from the old in-repo location once, without deleting them.

    Copy rather than move: this touches a directory the user may have open, and
    an import that destroys data on first import is not a good trade for tidiness.
    The marker stops it from running again, so a scan deleted on purpose does not
    come back.

    Only for the *default* location. If `VC_STATE_DIR` was set, the caller named a
    directory on purpose — a second agent, a test run, a throwaway — and filling
    it with data from somewhere else is a surprise, not a convenience. The test
    suite depends on this: its state directory has to start genuinely empty.
    """
    if os.getenv("VC_STATE_DIR"):
        return
    marker = STATE / "state" / ".migrated-from-repo"
    legacy = DATA / "output"
    if marker.exists() or not legacy.is_dir():
        return
    import shutil

    copied = 0
    for path in legacy.glob("scan-*.*"):
        target = OUTPUT / path.name
        if not target.exists():
            shutil.copy2(path, target)
            copied += 1
    marker.write_text(f"copied {copied} file(s) from {legacy}\n", encoding="utf-8")


_migrate_legacy_state()


# --------------------------------------------------------------------------- thesis


@dataclass
class Thesis:
    """Investment thesis — the rubric's `thesis_fit` axis is scored against it."""

    sectors: list[str]
    stages: list[str]
    geographies: list[str]
    requirements: list[str]  # non-negotiable
    red_lines: list[str]     # if observed, decide `skip` outright
    is_placeholder: bool = True  # has the user supplied their own thesis yet?

    def as_prompt(self) -> str:
        """A one-paragraph thesis statement handed to the agents."""
        return (
            f"Sectors: {', '.join(self.sectors)}. "
            f"Stages: {', '.join(self.stages)}. "
            f"Geographies: {', '.join(self.geographies)}. "
            f"Requirements: {'; '.join(self.requirements)}. "
            f"Red lines: {'; '.join(self.red_lines)}."
        )


# WARNING — PLACEHOLDER. The user's own thesis has not been supplied yet.
# While `is_placeholder=True`, `scan.py` prints a warning on every run and the
# `thesis_fit` axis is reported as uncalibrated. Once you write your own thesis,
# set `is_placeholder=False`; nothing else needs to change.
THESIS = Thesis(
    sectors=["AI infrastructure", "developer tools", "data infrastructure"],
    stages=["pre-seed", "seed"],
    geographies=["global"],
    requirements=["technical founder", "public technical trace (repo/paper/demo)"],
    red_lines=["solo non-technical founder", "closed-source consulting business"],
    is_placeholder=True,
)


# --------------------------------------------------------------------------- model tiers

# The funnel economics expressed in code: volume at the top, quality at the bottom.
#   triage (~200/day)   -> cheap
#   enrichment          -> mid
#   risk + scoring      -> mid
#   investment memo (~5)-> strong
LLM_BASE_URL = os.getenv("VC_LLM_BASE_URL", "")  # OpenAI-compatible endpoint
LLM_API_KEY = os.getenv("VC_LLM_API_KEY", "")

MODEL_TIERS = {
    "cheap": os.getenv("VC_MODEL_CHEAP", ""),
    "mid": os.getenv("VC_MODEL_MID", ""),
    "strong": os.getenv("VC_MODEL_STRONG", ""),
}


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw not in ("0", "false", "no")


# Capability record for a model the OpenAI client does not recognise — every
# OpenAI-compatible endpoint. Without it the client raises
# "model_info is required when model name is not a valid OpenAI model".
# Override per deployment: claiming a capability the model lacks fails late and
# quietly, at the structured-output step rather than at startup.
LIVE_MODEL_INFO = {
    "vision": _flag("VC_MODEL_VISION", False),
    "function_calling": _flag("VC_MODEL_FUNCTION_CALLING", True),
    "json_output": _flag("VC_MODEL_JSON_OUTPUT", True),
    "structured_output": _flag("VC_MODEL_STRUCTURED_OUTPUT", True),
    "family": os.getenv("VC_MODEL_FAMILY", "unknown"),
}


# A self-hosted endpoint under load answers slower than a hosted one, and the
# client's default gives up first — measured: the scan's short prompts went
# through while the conversation's longer one, carrying seven tool schemas,
# timed out. Raise this rather than shortening the prompt.
LLM_TIMEOUT = float(os.getenv("VC_LLM_TIMEOUT", "300"))
LLM_MAX_RETRIES = int(os.getenv("VC_LLM_MAX_RETRIES", "2"))


def live_llm_available() -> bool:
    """Live mode requires all three tiers; otherwise the system runs dry."""
    return bool(LLM_BASE_URL and LLM_API_KEY and all(MODEL_TIERS.values()))


def missing_llm_settings() -> list[str]:
    missing = []
    if not LLM_BASE_URL:
        missing.append("VC_LLM_BASE_URL")
    if not LLM_API_KEY:
        missing.append("VC_LLM_API_KEY")
    for tier, model in MODEL_TIERS.items():
        if not model:
            missing.append(f"VC_MODEL_{tier.upper()}")
    return missing


# --------------------------------------------------------------------------- thresholds


@dataclass
class Thresholds:
    review_at: int = 17         # out of 25 — decide "review"
    watch_at: int = 11          # below this, "skip"
    default_days: int = 7       # when --days is omitted
    max_candidates: int = 25    # cap on companies enriched in one run
    max_messages: int = 40      # runaway-loop fuse on every team
    # Wall-clock fuse for one company's enrichment. Owning the runtime means
    # owning its failure modes: with an externally supplied runtime a crashing
    # agent makes `run_stream` hang instead of raising (measured 2026-08-13), so
    # a deadline is the only thing that ends the run. `MaxMessageTermination`
    # cannot help — no further messages ever arrive.
    enrichment_timeout_seconds: float = 180.0


THRESHOLDS = Thresholds()


# --------------------------------------------------------------------------- sessions


@dataclass
class SessionPolicy:
    """Session lifecycle, following OpenClaw's model (docs/13 §4).

    `dm_scope` is the one that matters for correctness rather than tidiness. With
    `main`, every direct message from every person lands in one shared session —
    which is fine for a single-operator tool and wrong the moment a second person
    can reach it. `per-channel-peer` is the safe default for the same reason
    OpenClaw recommends it.
    """

    dm_scope: str = os.getenv("VC_DM_SCOPE", "per-channel-peer")
    daily_reset_hour: int | None = int(os.getenv("VC_RESET_HOUR", "4"))
    idle_minutes: int = int(os.getenv("VC_IDLE_MINUTES", "180"))
    prune_after_days: int = 30
    max_entries: int = 500
    # Messages held in context. Superseded by the token budget below once the
    # context engine is in play; kept because the legacy engine still counts here.
    buffer_size: int = 24
    token_budget: int = int(os.getenv("VC_TOKEN_BUDGET", "12000"))
    # Reserve kept free for the reply and for the compaction summary itself.
    compaction_reserve: int = int(os.getenv("VC_COMPACTION_RESERVE", "2000"))


SESSION_POLICY = SessionPolicy()

VALID_DM_SCOPES = ("main", "per-peer", "per-channel-peer", "per-account-channel-peer")


# --------------------------------------------------------------------------- source policy

USER_AGENT = "vc-agent/0.1 (research; contact: github.com/AltanReisoglu)"

# SEC has a separate access policy: requests whose User-Agent carries no name and
# contact address get a 403 (verified live). If you make this repository public,
# set `VC_SEC_UA` to your own address so the default below never reaches git.
SEC_USER_AGENT = os.getenv("VC_SEC_UA", "Altan Reisoglu Research flexyposts@gmail.com")

# Minimum seconds between requests, per source. Values come from the sources'
# own documentation: arXiv asks for 3s, SEC caps at 10 req/s (we stay far below).
RATE_LIMITS: dict[str, float] = {
    "hn": 0.5,
    "sec_edgar": 1.0,
    "github": 1.0,
    "arxiv": 3.0,
    "default": 1.0,
}

# Sources that require a login or whose ToS forbids scraping. `policy.is_allowed`
# returns False for these unconditionally, and a test guards that.
BLOCKLIST = [
    "linkedin.com",
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "crunchbase.com",  # ToS forbids scraping, API is paid
    "pitchbook.com",
]

# The repository's `.mcp.json` already configures DeepWiki; this is the switch
# that lets the live conversation attach it as a tool source. Off-by-default
# would hide a working capability, so it is on and fails soft.
MCP_DEEPWIKI = os.getenv("VC_MCP_DEEPWIKI", "1") not in ("0", "false", "no")

# The OpenClaw bridge, both directions.
#
# Inbound: OpenClaw spawns `python -m pipeline.mcp_server` over stdio and calls
# our tools. Nothing here configures that — it lives in OpenClaw's own config.
#
# Outbound: our agent attaches `openclaw mcp serve` as a second workbench, which
# gives it the channel conversations OpenClaw is holding.
#
# **On by default, and sending is still gated.** Two switches rather than one:
# channel *awareness* is useful and read-only, while channel *action* is neither.
# Attaching fails soft — no binary, no gateway, or a dead bridge all leave the
# agent with every local tool and a status string saying why.
MCP_OPENCLAW = _flag("VC_MCP_OPENCLAW", True)
OPENCLAW_BIN = os.getenv("VC_OPENCLAW_BIN", "openclaw")

# Which OpenClaw tools to put in front of the model.
#
# Tool schemas go into the system prompt and are paid for on **every turn**. That
# is not a theoretical cost here: docs/06 records a live timeout on the
# conversation's prompt while it carried seven tool schemas. Ten local tools plus
# nine from OpenClaw plus two from DeepWiki is twenty-one, and most of the
# OpenClaw nine are never the right call.
#
# Empty means "take everything". The default keeps the four that answer real
# questions and drops the rest — including the two that are blocked anyway, whose
# only value in the prompt is letting the agent say "I could, but I need approval".
OPENCLAW_TOOLS = tuple(
    t.strip()
    for t in os.getenv(
        "VC_OPENCLAW_TOOLS",
        "conversations_list,conversation_get,messages_read,messages_send",
    ).split(",")
    if t.strip()
)

# Tools that reach outside and cannot be taken back. The approval gate blocks
# these by default; read-only tools (`conversations_list`, `messages_read`) are
# not here and stay free. Every entry is a *substring* match on the tool name, so
# a provider renaming `messages_send` to `message_send` still trips it.
#
# `respond` and `approve` are here for a reason found by looking rather than
# guessing: OpenClaw exposes `permissions_respond`, which answers *its own*
# pending permission prompts. An agent able to call it could approve OpenClaw's
# requests on the operator's behalf and collapse two independent gates into one.
# It contains no obvious verb, so a list built from imagination would have let it
# through.
OUTBOUND_TOOLS = tuple(
    t.strip()
    for t in os.getenv(
        "VC_OUTBOUND_TOOLS", "send,post,write,delete,spawn,respond,approve"
    ).split(",")
    if t.strip()
)

# When false, an outbound tool call is dropped by the runtime and the agent is
# told so. When true, it goes through — for an operator who has decided the
# blast radius is acceptable. There is no middle setting that is honest.
ALLOW_OUTBOUND = _flag("VC_ALLOW_OUTBOUND", False)

REQUEST_TIMEOUT = 20.0
CACHE_TTL_SECONDS = 6 * 3600
