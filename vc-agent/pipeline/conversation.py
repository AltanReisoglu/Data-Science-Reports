"""The conversation as a live agent, one per session, behind the gateway's rules.

`answers.py` reads a completed run. This module is the other half: an AutoGen
agent that holds the conversation, calls tools while you talk to it, reaches
remote MCP servers, and can start a scan of its own. It only runs when an LLM is
configured; without one the server keeps serving the deterministic path, which is
a working system rather than a degraded one.

It exists to use the parts of AutoGen the pipeline had not touched, and each one
is here because it does a job:

**`model_context` — where multi-turn memory actually lives.** An `AssistantAgent`
is stateless between `run()` calls unless you give it a context object. This used
to be `BufferedChatCompletionContext(24)`, which counts *messages, not tokens* —
the flaw docs/06 §2 records. It is now `context_engine`, which counts tokens and
compacts without splitting a tool call from its result.

**`workbench` — and why `tools` cannot come with it.** `AssistantAgent` rejects
being given both: *"Tools cannot be used with a workbench."* The way to have
local functions and remote MCP servers in the same agent is to wrap the local
ones in a `StaticWorkbench` and pass a **list**. A workbench is a tool *source* —
it can list tools that did not exist when the agent was written, which is exactly
what a remote server is, and exactly why the approval gate lives on the workbench
rather than on a list of known names.

**`CancellationToken` — the stop button.** Passed into `run_stream`, cancelled
from another request or from `runs.abort`.

**`save_state` / `load_state` — the conversation surviving a restart.** These
serialise the agent's context, and they are why `model_context` matters twice
over: what is saved is what the context holds. State is now written **per
session**, beside that session's transcript, instead of to one shared file.

**`model_client_stream` — why it reads as a chatbot.** Without it a reply lands
in one block after the model finishes.

### What the gateway added

The agent is no longer a singleton. `Conversation` is built for one session, and
`SessionManager` owns the instances — so two people talking through two channels
have two contexts, which is the difference between a tool and a shared mailbox.
Tool calls pass through `GatedWorkbench`, so `before_tool_call` hooks (and the
approval gate) apply to remote tools as well as local ones. `MEMORY.md` is loaded
into the system prompt at session start; daily notes are searchable and are not.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

import config
import engine
import observability
from gateway import approval as approval_module
from gateway import hooks as hooks_module
from gateway import sessions as sessions_module
from gateway import tools as tools_module
from gateway import workbench as workbench_module

# DeepWiki is already configured in the repository's `.mcp.json` but nothing was
# using it. `ask_question` currently fails server-side (verified: it returns
# "Error processing question:" for well-formed input), so only the two tools
# that were observed working are exposed.
DEEPWIKI_URL = "https://mcp.deepwiki.com/mcp"
DEEPWIKI_TOOLS = ["read_wiki_structure", "read_wiki_contents"]

SYSTEM = """You are the analyst for a VC deal-flow pipeline. You are talking to
the person who owns it.

You have tools. Use them rather than guessing: `scan_facts` for what the last
run established, `company_detail` for one candidate as the scan recorded it,
**`company_live` for what has changed since the scan** (use this whenever the
question is about *now*), `search_github` and `search_hacker_news` for anything
outside the pipeline, `start_scan` to run the funnel again, **`search_docs` for
how anything works**, and **`memory_search` for what has been written down
before**. Use `memory_note` when the person tells you something worth keeping —
a preference, a decision, a fact about a company — and say that you did.
If a repository question comes up and you have DeepWiki tools, read the
repository rather than speculating about it.

Rules that override everything else:
- Never state a fact you did not get from a tool or from the scan. Say what you
  could not establish, and say where you looked.
- A missing value and a low value are different things.
- Quote source URLs when you have them. For a documentation answer, cite the
  file and line `search_docs` returned, and say whether it came from the official
  AutoGen guide or from this project's own notes — they carry different weight.
- If the scan ran in dry mode, its scores are placeholders. Say so whenever you
  quote one.
- "Could not check" and "nothing changed" are different answers. If a live source
  failed, say which one and do not present the rest as a complete picture.
- If a tool call is refused, tell the person what you wanted to do and why it was
  refused. Do not retry it.
- Be brief. No preamble.

Current scan: {scan_line}{memory}{channels}"""


# --------------------------------------------------------------------------- session


class Conversation:
    """One live agent session, with memory, tools, MCP, cancellation and state."""

    def __init__(
        self,
        scan_getter,
        scan_starter,
        *,
        session_id: str = "",
        store: sessions_module.SessionStore | None = None,
        registry: hooks_module.HookRegistry | None = None,
        buffer_size: int | None = None,
    ) -> None:
        self._sources = tools_module.Sources(
            scan_getter=scan_getter, scan_starter=scan_starter, session_id=session_id
        )
        self.session_id = session_id or "agent:main:web:dm:local"
        self._store = store or sessions_module.SessionStore()
        self._registry = registry or hooks_module.REGISTRY
        self._buffer_size = buffer_size
        self._agent = None
        self._mcp: list[Any] = []
        self._ledger: engine.Ledger | None = None
        self._token = None
        self._context = None
        # Live token usage only shows up on the `autogen_core` event stream —
        # the ledger's `create_calls` counter exists on the replay client alone,
        # so a live conversation would otherwise report a cost of zero.
        self._usage = {"llm_calls": 0, "tokens": 0}
        self.mcp_status = "not attempted"
        self.openclaw = None
        self.lock = asyncio.Lock()

    # ------------------------------------------------------------ construction

    @property
    def state_path(self):
        return self._store.state_path(self.session_id)

    async def _deepwiki(self):
        """Attach DeepWiki if it answers. A dead remote must not break the chat."""
        if not config.MCP_DEEPWIKI:
            self.mcp_status = "disabled"
            return None
        try:
            from autogen_ext.tools.mcp import McpWorkbench, StreamableHttpServerParams

            workbench = McpWorkbench(
                server_params=StreamableHttpServerParams(url=DEEPWIKI_URL, timeout=30.0)
            )
            await workbench.start()
            names = [t.get("name") for t in await workbench.list_tools()]
            self.mcp_status = f"connected ({', '.join(n for n in names if n in DEEPWIKI_TOOLS)})"
            return workbench
        except Exception as e:
            self.mcp_status = f"unavailable ({type(e).__name__})"
            return None

    async def ensure(self) -> None:
        if self._agent is not None:
            return

        from autogen_agentchat.agents import AssistantAgent
        from autogen_core.tools import FunctionTool, StaticWorkbench

        import context_engine
        import memory as memory_module
        import openclaw as openclaw_module

        data = self._sources.scan_getter()
        scan_line = (
            f"{data.get('query')} over {data.get('days')} days, mode {data.get('mode')}, "
            f"{len(data.get('candidates', []))} candidates"
            if data
            else "none yet"
        )

        local = [
            FunctionTool(fn, description=(fn.__doc__ or "").strip().split("\n")[0])
            for fn in tools_module.build(self._sources)
        ]
        # `tools=` and `workbench=` are mutually exclusive, so local functions go
        # in through a StaticWorkbench and the remote sources join them.
        raw: list[Any] = [StaticWorkbench(local)]

        deepwiki = await self._deepwiki()
        if deepwiki is not None:
            raw.append(deepwiki)

        self.openclaw = await openclaw_module.attach()
        if self.openclaw.attached:
            raw.append(self.openclaw.workbench)

        self._mcp = [w for w in raw[1:]]
        # Every workbench, local and remote, goes behind the hook chain — so the
        # gate covers tools that did not exist when this line was written. The
        # OpenClaw source is additionally *filtered*: schemas cost tokens on every
        # turn, and `permissions_respond` has no legitimate call.
        openclaw_wb = self.openclaw.workbench if self.openclaw and self.openclaw.attached else None
        workbenches = [
            workbench_module.GatedWorkbench(
                source,
                registry=self._registry,
                session_id=self.session_id,
                allow=config.OPENCLAW_TOOLS if source is openclaw_wb else None,
            )
            for source in raw
        ]

        self._ledger = engine.Ledger()
        self._context = self._build_context()

        # `MEMORY.md` is loaded once, at session start, and is paid for on every
        # turn — which is why only entries go in, not the file's own instructions.
        preamble = memory_module.preamble()
        memory_block = f"\n\nWhat you already know:\n{preamble}" if preamble else ""

        prompt = SYSTEM.format(
            scan_line=scan_line,
            memory=memory_block,
            channels=openclaw_module.guidance(self.openclaw),
        )
        built = await self._registry.run(
            hooks_module.BEFORE_PROMPT_BUILD,
            {"session": self.session_id, "system_prompt": prompt, "scan": scan_line},
        )
        prompt = str(built.get("system_prompt") or prompt)
        if built.get("prepend_context"):
            prompt = f"{built.get('prepend_context')}\n\n{prompt}"

        self._agent = AssistantAgent(
            "Analyst",
            # Unwrapped on purpose: a failed call here must surface as an error
            # event, not as an answer. See `Ledger.raw_client`.
            model_client=self._ledger.raw_client("mid"),
            description="Answers questions about the deal-flow pipeline and its scans.",
            system_message=prompt,
            workbench=workbenches,
            model_context=self._context,
            model_client_stream=True,
            max_tool_iterations=6,  # the default is 1: no chained tool calls
        )
        await self._restore()

    def _build_context(self):
        """The context engine, with the cheap tier doing the summarising."""
        import context_engine

        summariser = None
        if self._ledger is not None and config.live_llm_available():
            # Compaction is bookkeeping, not analysis — the funnel's cost argument
            # applies to it too.
            summariser = context_engine.model_summariser(self._ledger.raw_client("cheap"))
        return context_engine.CompactingChatCompletionContext(
            summariser=summariser,
            on_compaction=lambda event: self._store.append(
                self.session_id,
                {"event": "compaction", "method": event.method, "summarised": event.summarised},
            ),
        )

    # ------------------------------------------------------------ streaming

    async def stream(self, question: str) -> AsyncIterator[dict[str, Any]]:
        """Yield chat events: token chunks, tool calls, and the final message."""
        from autogen_core import CancellationToken

        await self.ensure()
        assert self._agent is not None
        self._token = CancellationToken()
        # The caller records the user turn — `SessionManager.record_turn` from the
        # HTTP route, `SessionAgent` from the message path — because only the
        # caller knows the origin and only it can bump the session's counters.
        # Writing it here as well put every question in the transcript twice.

        final = ""
        capture = observability.EventCapture()
        try:
            with capture:
                async for event in self._agent.run_stream(
                    task=question, cancellation_token=self._token, output_task_messages=False
                ):
                    kind = type(event).__name__
                    if kind == "ModelClientStreamingChunkEvent":
                        yield {"type": "chunk", "text": str(event.content)}
                    elif kind == "ToolCallRequestEvent":
                        for call in event.content:
                            yield {"type": "tool", "name": call.name,
                                   "arguments": str(call.arguments)[:200]}
                    elif kind == "ToolCallExecutionEvent":
                        for result in event.content:
                            yield {"type": "tool_result", "name": result.name,
                                   "preview": str(result.content)[:180]}
                    elif kind == "TaskResult":
                        last = event.messages[-1] if event.messages else None
                        final = str(getattr(last, "content", ""))
                        yield {"type": "done", "text": final,
                               "stop_reason": event.stop_reason or ""}
        except asyncio.CancelledError:
            yield {"type": "cancelled"}
        except Exception as e:
            yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
        finally:
            self._token = None
            self._usage["llm_calls"] += capture.totals.llm_calls
            self._usage["tokens"] += capture.totals.total_tokens
            # The transcript belongs to the caller, not to the stream. Two owners
            # meant two copies of every turn on the message path, where both
            # `SessionAgent` and this method were writing.
            await self._registry.run(
                hooks_module.AGENT_END,
                {"session": self.session_id, "reply": final,
                 "tools": [c.tool for c in capture.totals.tool_calls]},
            )
            await self._persist()

    def cancel(self) -> bool:
        if self._token is None:
            return False
        self._token.cancel()
        return True

    # ------------------------------------------------------------ approvals

    @property
    def pending_approvals(self) -> list[dict[str, Any]]:
        return approval_module.GATE.pending()

    # ------------------------------------------------------------ state

    async def _persist(self) -> None:
        if self._agent is None:
            return
        try:
            self.state_path.write_text(
                json.dumps(await self._agent.save_state(), ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception:
            # Losing the transcript is survivable; losing the answer is not.
            pass

    async def _restore(self) -> None:
        if self._agent is None or not self.state_path.exists():
            return
        try:
            await self._agent.load_state(json.loads(self.state_path.read_text(encoding="utf-8")))
        except Exception:
            self.state_path.unlink(missing_ok=True)

    async def reset(self) -> None:
        self.state_path.unlink(missing_ok=True)
        await self.close()

    async def close(self) -> None:
        for workbench in self._mcp:
            try:
                await workbench.stop()
            except Exception:
                pass
        self._mcp = []
        if self._ledger is not None:
            await self._ledger.close()
            self._ledger = None
        self._agent = None
        self._context = None

    # ------------------------------------------------------------ reporting

    def cost(self) -> dict[str, int]:
        """Live usage from the event stream, plus whatever the replay path counted."""
        total = dict(self._usage)
        if self._ledger is not None:
            measurement = self._ledger.measurement("conversation")
            total["llm_calls"] += measurement.llm_cagrisi
            total["tokens"] += measurement.toplam_token
        return total

    def context_report(self) -> dict[str, Any]:
        """What the context engine has done — compactions are worth surfacing."""
        if self._context is None:
            return {"active": False}
        return {
            "active": True,
            "tokens": self._context.tokens(),
            "budget": self._context.usable(),
            "compactions": [
                {"at": e.at, "method": e.method, "summarised": e.summarised}
                for e in self._context.stats.compactions
            ],
        }


# --------------------------------------------------------------------------- registry


class ConversationRegistry:
    """One `Conversation` per session, created on demand.

    This is what makes the gateway's session model reach the agent. Without it
    every channel would still share one context — the shape would be right and
    the behaviour unchanged, which is the worst kind of refactor.
    """

    def __init__(self, scan_getter, scan_starter) -> None:
        self._scan_getter = scan_getter
        self._scan_starter = scan_starter
        self.sessions = sessions_module.SessionManager()
        self._conversations: dict[str, Conversation] = {}
        approval_module.install(hooks_module.REGISTRY)
        # `openclaw_call` hides its blast radius in an argument, so it needs a
        # gate that reads the method rather than the tool name.
        import openclaw_control

        openclaw_control.install_gate(hooks_module.REGISTRY)

    def route(self, channel: str = "web", **kwargs) -> sessions_module.SessionRecord:
        return self.sessions.route(channel, **kwargs)

    def for_session(self, record: sessions_module.SessionRecord) -> Conversation:
        existing = self._conversations.get(record.id)
        if existing is None:
            existing = Conversation(
                self._scan_getter,
                self._scan_starter,
                session_id=record.id,
                store=self.sessions.store,
            )
            self._conversations[record.id] = existing
        return existing

    def get(self, channel: str = "web", **kwargs) -> tuple[sessions_module.SessionRecord, Conversation]:
        record = self.route(channel, **kwargs)
        return record, self.for_session(record)

    async def reset(self, session_id: str) -> None:
        conversation = self._conversations.pop(session_id, None)
        if conversation is not None:
            await conversation.reset()
        self.sessions.reset(session_id)

    async def close(self) -> None:
        for conversation in list(self._conversations.values()):
            await conversation.close()
        self._conversations.clear()
