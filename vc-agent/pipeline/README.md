# pipeline — the VC agent, built on AutoGen

Implementation of `docs/03-vc-domain-plani.md` (what and why) and
`docs/04-vc-agentic-akis.md` (how). Built on AutoGen v0.7.5 (`autogen-agentchat`,
`autogen-core`), verified against the installed package rather than from memory.

> Code, filenames and comments are in English. The design documents in `docs/`
> remain in Turkish; the mapping between them is in [Naming](#naming) below.

## Run it

```bash
cd vc-agent
.venv/bin/python pipeline/server.py        # http://127.0.0.1:8000
```

That is the whole application: a chat interface over your scans, with a **New
scan** button that runs the funnel and streams its log into the conversation.
Loopback only, no auth — a single-user tool by decision (docs/03), not by
omission. FastAPI and uvicorn were already in the venv as transitive
dependencies, so the server adds no new packages.

The pipeline still runs headless if you prefer it:

```bash
.venv/bin/python pipeline/scan.py --query "ai infrastructure" --days 7 --limit 5
```

No API key is required. Without LLM credentials the system runs in **dry mode**:
collectors, normalization, the graph, the schemas and the audit log all run for
real against live APIs, and only the model replies are replayed. What gets
measured is the control flow rather than the model's mood — the discipline
inherited from `poc/`.

Tests, offline:

```bash
PYTHONPATH=pipeline .venv/bin/python -m unittest discover -s pipeline/tests -t pipeline
```

## Going live

Three tiers, one OpenAI-compatible endpoint:

```bash
export VC_LLM_BASE_URL="https://.../v1"
export VC_LLM_API_KEY="..."
export VC_MODEL_CHEAP="..."    # triage, ~200 calls/day
export VC_MODEL_MID="..."      # analysts, risk, scoring
export VC_MODEL_STRONG="..."   # the memo, ~5 calls/day
```

All three tiers must be set or the run stays dry — a half-configured run would
silently mix replayed and real replies.

Optional: `GITHUB_TOKEN` (60 → 5,000 requests/hour), `VC_SEC_UA` (SEC requires a
User-Agent carrying a name and contact address; the default in `config.py` is
Altan's, so set your own before making this repository public).

## The funnel

```
  collect      hn · sec_edgar · github          no LLM, deterministic
     ↓
  normalize    entity resolution + dedup        no LLM
     ↓
  triage       rules first, cheap model second  ← volume lives here
     ↓
  enrich       GraphFlow: 3 parallel branches   ← the only real concurrency
     ↓         join → risk audit → scorer
  memo         strong tier, few per day
```

Cost is reported per stage on every run, so you can check that the funnel
actually spent little at the top and much at the bottom.

## Files

| File | Role |
|---|---|
| `config.py` | Thesis, model tiers, thresholds, rate limits, blocklist |
| `schemas.py` | The data contract. `Source.url` and `Score.missing_data` are mandatory |
| `policy.py` | The single gate outward: blocklist, robots.txt, rate limit, audit log |
| `engine.py` | Model factory per tier, `ResilientClient`, measurement ledger |
| `observability.py` | `autogen_core` event capture (LLM usage, tool calls) + intervention gate |
| `collectors/` | Layer 1. `base.py` + `hackernews.py` `sec_edgar.py` `github.py` `arxiv.py` |
| `normalize.py` | Layer 2. Entity resolution, deduplication |
| `agents/` | Layer 3. `triage.py` `analysts.py` `tools.py` `memo.py` |
| `graph.py` | The enrichment graph: fan-out, join, branch accounting (AgentChat) |
| `fanin.py` | The same fan-out on `autogen_core` pub/sub + a collector queue |
| `compare_fanin.py` | Measures the two engines under the same injected failure |
| `scan.py` | CLI entry point |
| `server.py` | FastAPI backend: serves the UI, answers questions, runs scans |
| `answers.py` | Question routing and the model-grounding prompt (no-LLM path) |
| `conversation.py` | The live agent: memory, tools, MCP, cancellation, state |
| `live.py` | Re-checks a pipeline company against live sources: what changed since the scan |
| `docs_index.py` | TF-IDF search over `docs/` — the AutoGen guides and our own design notes |
| `web/` | The frontend: `index.html`, `app.js`, `app.css` |
| `dashboard.py` | Renders a scan result as one self-contained page you can question |
| `tests/` | Offline tests, fixtures only |
| `data/` | SQLite, cache, audit log, output (git-ignored) |

## What the core runtime adds

The enrichment graph runs on a runtime this pipeline constructs itself, rather
than the one AgentChat would build for it. That buys three things and costs one.

**Tool calls become auditable.** `autogen_core` emits `ToolCallEvent` for every
`BaseTool` execution, carrying the tool name, its arguments, its result and —
inside a runtime — the calling agent's id. `observability.EventCapture` mirrors
these into the audit log, which is what makes docs/04 §6 true: the log now
answers *which agent chose to call what*, not merely *what the collectors
fetched*. Verified live: a bare `agent.run()` records no agent id, the same agent
inside a team records `TechnicalAnalyst_<uuid>`.

**Cost is measured in live mode too.** `LLMCallEvent` carries the token counts,
and it is emitted by the real clients but never by `ReplayChatCompletionClient`.
The ledger's `create_calls` counter is the mirror image — replay-only. Before
this, a live run would have reported zero LLM calls. The two paths together
cover both modes.

**There is a place for the approval gate.** An `InterventionHandler` sits on the
message path and can return `DropMessage`, so the partner gate of docs/04 is
enforced by the runtime rather than by an agent choosing to comply. It is wired
and tested but left in observer mode: every tool here is read-only, so prompting
on each call would be ceremony.

**The cost: owning the runtime means owning its failure modes.** Measured
2026-08-13 — with an externally supplied runtime, an agent that raises leaves
`run_stream` waiting forever on a termination message that will never come. With
the embedded runtime the same crash raises. `MaxMessageTermination` cannot help,
because no further messages arrive either. The only thing that ends such a run is
a deadline, so `THRESHOLDS.enrichment_timeout_seconds` is a correctness
requirement here rather than a precaution, and `stop_when_idle()` is itself
called under a bound.

## The chat interface

Two different things answer you, and the interface always says which:

**Without an LLM — `answers.py`.** The question is matched against the candidate
names first, then keyword sets, and the answer is a block rendered from the run's
own JSON. Deterministic, no network, and structurally incapable of saying
something the scan did not establish. A question it has no answer for gets told
exactly that.

**With an LLM — `conversation.py`.** A live AutoGen agent holds the conversation.
It is not a reader over a finished scan: it calls tools while you talk to it,
reaches a remote MCP server, and can start a scan of its own.

| Tool | What it does |
|---|---|
| `scan_facts` | Everything the last run established |
| `company_detail` | One candidate, as the scan recorded it |
| `company_live` | The same candidate **now** — what changed since the scan |
| `search_docs` | How anything works: the AutoGen guides and our own design notes |
| `search_github` / `search_hacker_news` | What is true *now*, not at scan time |
| `start_scan` | Runs the funnel again — the agent can act, not only report |
| DeepWiki (MCP) | `read_wiki_structure`, `read_wiki_contents` on any public repo |

Replies stream token by token, the tools it decides to call are shown as they
happen, and **Stop** cancels mid-answer. The conversation survives a server
restart; **Reset chat** clears it.

### What it uses, and why each piece is there

This is where the AutoGen surface the pipeline had not touched gets used. Each
one earns its place, and each one had a trap in it — the full list, with what
broke and how, is in [docs/06](../docs/06-autogen-incelikleri.md).

- **`model_context`** — an `AssistantAgent` is stateless between `run()` calls.
  `BufferedChatCompletionContext` is what makes "and its team?" resolve against
  the previous question. Note it counts *messages*, not tokens.
- **`StaticWorkbench` + `McpWorkbench`** — `tools=` and `workbench=` cannot both
  be given to one agent (`ValueError: Tools cannot be used with a workbench`).
  Wrapping the local functions in a `StaticWorkbench` and passing a **list** is
  how local tools and a remote server live in the same agent.
- **`CancellationToken`** — the Stop button, cancelled from a second request.
- **`save_state` / `load_state`** — the transcript surviving a restart. What gets
  saved is whatever `model_context` holds, which is why the two go together.
- **`model_client_stream`** — why it reads as a chatbot rather than a form.

### Endpoints

| Route | Purpose |
|---|---|
| `GET /` | The chat UI |
| `GET /api/state?scan=` | Opening report, funnel, candidate names, past scans, MCP status |
| `POST /api/ask` | Deterministic answer — `{path, title, text, html}` |
| `POST /api/chat` | Live agent turn, streamed as Server-Sent Events |
| `POST /api/chat/stop` | Cancel the turn in flight |
| `POST /api/chat/reset` | Clear the conversation |
| `POST /api/live` | Check one pipeline company against live sources, now |
| `POST /api/scan` · `GET /api/scan` | Start a run; live log, exit code, timing |

Scans run as a **subprocess**, not in the server's own loop: a scan that hangs or
dies must not take the interface down with it. One chat turn runs at a time —
the agent has a single context, and two concurrent runs would interleave into it.

### Asking how the thing works

The chat could answer questions about a scan but nothing about the framework the
scan runs on. `docs/` already held both halves — the official AutoGen **Core**
and **AgentChat** guides verbatim (05, 08) and this project's own design,
measurements and code guide (01–04, 06, 07, 09). That is 1.18 MB across 675
sections: far past any prompt, so `docs_index.py` searches it.

**Lexical, not embeddings — and that is a choice.** The endpoint does serve
`qwen3-embedding-8b`, so vectors were available. Three reasons against: an index
is a second thing that can go stale, and a stale index answers confidently from
an old document; embedding 675 sections needs the provider to be up, which would
make a *documentation* lookup fail when the model fails; and these documents are
dense with exact identifiers — `ClosureAgent`, `model_info`,
`activation_condition` — which is where lexical scoring is strongest. It builds
in 0.09s at import with no dependencies. Embeddings remain the upgrade path if
recall on paraphrases becomes the limit.

**Every hit carries its provenance**, and the distinction is not cosmetic: a
section from 05 is Microsoft's word, a section from 06 is something we measured
ourselves. The answer says which, plus the file and line.

**Documentation is consulted on explicit intent only, never as a catch-all.**
Scoring cannot separate a real documentation question from a stray one:
*"what is the weather in Istanbul"* scores 30.8 — higher than *"model_info
required"* at 17.6 — because the AgentChat guide really does contain a
`get_weather` tool example. The search is not wrong there; presenting it as an
answer would be. So the gate is the vocabulary of the subject, and a question
that matches nothing still gets the honest refusal it got before.

### Asking about a company *now*

The scan is a frozen record. `live.py` answers the other question — **what has
changed since it ran** — for a company already in the pipeline:

| Source | What it re-checks |
|---|---|
| GitHub | current stars/forks/last push, and the star delta against the scan |
| Hacker News | mentions that **name** the company, newer than the scan's last signal |
| SEC Form D | filings in the last 90 days, and whether any postdates the scan |

This is the fourth principle of `docs/03` finally having code behind it: the
funnel flows downward but monitoring is a loop, and until now nothing looked
twice.

Three rules it inherits from the rest of the pipeline:

- **"Could not check" is not "nothing changed."** A source that fails is named in
  the answer, and the report says outright that the remaining picture is
  incomplete. A monitoring loop that reports silence as calm is worse than none.
- **Each source fails alone.** A dead GitHub call does not stop the HN lookup.
- **Everything goes through `policy`** — same gate, rate limits and audit log as
  the collectors. A human asking changes nothing.

One precision rule came out of the first live run: Algolia ranks loosely, and
searching a company name returned a story that never mentions it. A mention now
counts only if the name is actually in the title — otherwise the system would
report invented movement.

**Two ways in, on purpose.** In the chat, `company_live` is a tool the agent can
call while you talk to it. On each candidate row, **Check now** goes straight to
`POST /api/live` and never touches the model: a button that promises an action
must not depend on a model choosing to call a tool. Without an LLM configured,
the same check runs through the deterministic path — the sources are HTTP, not a
model, so this works in either mode.

### Design

The chrome is a **left rail**, not a top bar: which scan, which mode, the funnel
counts and what to start next stay in view while the thread scrolls. A top bar
could not do that without covering the newest message. Below 56rem it folds back
into a bar, where the rail would cost more room than it earns.

The product opens **light**. Dark is designed and measured too, so it stays one
click away in the rail rather than being deleted, and the choice is remembered.

The product is **beige**: warm neutrals, paper rather than screen, with the data
hue left on the validated blue so the chart layer keeps a palette that was
measured while the skin changed around it. A cool mark on a warm ground is the
deliberate pairing, not two systems colliding.

Every value was measured against its own surface rather than eyeballed — ink
14.65, secondary 7.06, muted 3.48, bars 3.96, critical 4.31. Two results worth
recording: `status warning` sits at 1.64 on the light ground, which is the
documented sub-3:1 case whose mitigation (icon **and** word on every status) was
already in place; and the funnel's palest step had to move one stop darker,
because beige is darker than white and the original step fell to 1.89 — under the
2:1 floor an ordinal ramp owes its surface. Dark mode is warm too, re-measured on
its own ground rather than inverted.

Two rule sets meet in the page. The **chart layer** uses one hue for magnitude —
every chart is single-series and bar length already carries the value — with the
funnel on an ordinal ramp whose faintest step still clears 2:1 against its
surface, and each chart backed by a data table. Status colour is reserved for
genuine states (reliability, a failed branch, a dead source) and always ships
with a word, never colour alone. The **interface layer** follows Apple's
guidance: system typography with size-specific tracking and leading, translucent
chrome over a scroll edge, feedback on pointer-down rather than release, and
transitions tuned to a critically damped response (~0.35s, no overshoot). Both
`prefers-reduced-motion` and `prefers-reduced-transparency` are honoured, and
light and dark are designed as separate sets rather than an inversion.

The stylesheet is served from `dashboard.STYLE`, so the live app and the static
export cannot drift apart: one set of tokens, two surfaces.

The page's first job is honesty. Dry mode, a placeholder thesis, a collector that
fell over, a branch that returned nothing, and the signals nobody could attribute
are all stated in the opening message rather than buried.

### Static export

`scan.py` also writes `data/output/dashboard.html` — the same conversation as a
single self-contained file, with the answers baked in and the routing done in the
browser. No server, no network. Useful for sending a result to someone; the
backend is the one to work in.

## Two fan-in engines, measured

The core guide's **Concurrent Agents** pattern gathers results differently from
anything AgentChat exposes: workers publish to a results topic and a
`ClosureAgent` drains it into a queue the caller owns. `fanin.py` implements
enrichment that way. The difference only shows up under failure:

```
.venv/bin/python pipeline/compare_fanin.py
```

| engine | clean | failure behind `ResilientClient` | raw failure |
|---|---:|---:|---:|
| `graph.py` (GraphFlow) | 3 survive | 2 survive | **0–1 survive, 8s deadline** |
| `fanin.py` (pub/sub + queue) | 3 survive | 2 survive | **2 survive, ~3ms** |

Read the last column. A branch failing for reasons that have nothing to do with
its siblings destroys the completed work of those siblings in the AgentChat
engine, and the number it destroys is *not deterministic* — repeated runs gave 0
and 1. The core engine loses exactly the branch that failed, and returns
immediately, because the result was published the moment it existed and the queue
already held it. There is no barrier to distrust because there is no barrier.

Worth noting that the documented patterns disagree with each other here:
**Concurrent Agents** collects through a queue, while **Mixture of Agents**
aggregates with `asyncio.gather(...)` — the same construct whose early return
`poc/desen_5_core_aktor.py` traced to silent sibling loss.

`graph.py` remains the default: it carries the risk auditor and the structured
scorer, which `fanin.py` deliberately does not replace. `fanin.py` replaces the
gathering, not the rubric.

## Three decisions worth knowing

**The barrier is not trusted.** `poc/desen_5_core_aktor.py` measured a crashing
handler opening the join barrier early and sibling results vanishing with no
exception. The same shape reproduced one layer up here: with a raw client, one
crashing branch of the three-way fan-out took the completed work of its siblings
with it — a three-branch run returned holding one. `engine.ResilientClient`
converts a failed model call into a message so the join still receives three
inputs, and `graph.py` counts the branches it expected rather than asking the
framework whether they all arrived. A branch that did not report becomes an entry
in `Score.missing_data`. `tests/test_graph.py` guards both directions.

**Absence of information is never grounds for rejection.** Triage rejects only on
evidence of contradiction; when unsure it passes the candidate down. A missed
company is invisible and expensive, a wasted review is visible and cheap.

**Entity resolution refuses to guess.** Domain, then GitHub org, then a fuzzy name
match at a 0.92 threshold — and below that, two records stay apart. Two live
findings are encoded as rules: the linked domain on a funding story belongs to
the *publisher* (which is how `bbc.com` became a candidate), and a launch linking
to `github.com` points at a repository, not at a company (which merged three
unrelated Show HN posts into one entity).

## Not built yet

Phases 5, 7 and 8 of `docs/04` §8:

- **Watchlist state machine** on `demo-brain-agent/taskboard.py` — nothing is
  persisted between runs yet, so there is no change detection and no `watch` loop.
- **MCP server** (`mcp_server.py`) and the OpenClaw channel — no alert reaches a phone.
- **Human approval gate** — the runtime-level mechanism is wired and tested
  (`observability.AuditingInterventionHandler`), but no `UserProxyAgent` prompts a
  partner yet; the gate runs in observer mode.
- **OpenTelemetry and back-testing** — cost is measured, recall is not. The core
  guide's `telemetry.html` route (`tracer_provider` on the runtime) is open: the
  runtime is already ours to configure.
- **ChromaDB semantic dedup** — the package is not installed; `normalize.py` uses
  a deterministic key plus `difflib` instead.
- **arXiv as a discovery source** — deliberately excluded. A paper carries no
  company name, so all 30 signals from a live run arrived unattachable. It serves
  as the team analyst's tool instead.

## Naming

The design documents use Turkish names; the code does not.

| docs/04 | here |
|---|---|
| `ayarlar.py` | `config.py` |
| `semalar.py` | `schemas.py` |
| `politika.py` | `policy.py` |
| `motor.py` | `engine.py` |
| `toplayicilar/` | `collectors/` |
| `ajanlar/` | `agents/` |
| `graf.py` | `graph.py` |
| (yok) | `observability.py` |
| `tara.py` | `scan.py` |
| `izleme.py` | not built |
| `mcp_sunucu.py` | not built |
