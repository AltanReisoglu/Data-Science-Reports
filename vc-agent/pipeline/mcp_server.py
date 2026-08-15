"""The pipeline as an MCP server, so OpenClaw can ask it things.

This is the inbound half of the bridge and the reason `docs/04 §8` listed "MCP
sunucusu + OpenClaw kanalı" as a phase: OpenClaw already owns the channels. It is
holding a Telegram connection, it has session routing, it has an agent. What it
does not have is any idea what a Form D filing is. Rather than build a second
Telegram integration, we hand OpenClaw a tool list and let it ask.

    Telegram ──▶ OpenClaw Gateway ──stdio/MCP──▶ this process ──▶ state dir

**stdio, not HTTP.** OpenClaw spawns this process and owns it; when the client
disconnects, the process goes. No port, no bind address, no shared secret, and
nothing on the network to get wrong. The cost is stated in `gateway/tools.py`:
this process cannot see the running gateway's memory, so it reads the state
directory and calls the gateway over loopback for anything that changes state.

**Every call is a session.** An MCP request arrives with a peer, and that peer
gets a session in our own store with a transcript, exactly like the web chat. The
alternative — treating machine callers as anonymous — is how a system ends up
unable to answer "where did this come from" about half its own traffic. That is
the mentality this whole exercise is copying, applied to ourselves.

**Read-only, and that is a design decision, not a limitation.** Everything here
either reads state or starts a scan the operator can see. Nothing here can send a
message, and nothing here needs the approval gate, because the dangerous
direction is the *other* one — us calling out through OpenClaw (`openclaw.py`).

Run it directly to check the tool list without OpenClaw in the way:

    python -m pipeline.mcp_server --list
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Run as `python -m pipeline.mcp_server` from the repository, or as a bare script
# from `pipeline/`. Both have to work: the first is what a person types, the
# second is what OpenClaw's spawned command ends up being.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config  # noqa: E402
from gateway import sessions as sessions_module  # noqa: E402
from gateway import tools as tools_module  # noqa: E402

SERVER_NAME = "vc-agent"

INSTRUCTIONS = """Tools over a VC deal-flow pipeline that scans for early signals
(Hacker News, SEC Form D, GitHub), scores candidates against a thesis, and keeps
its own documentation and memory.

Rules that matter when you use these:
- `vc_scan_facts` describes the last completed scan, not the world now. For what
  changed since, call `vc_company_live`.
- A missing value and a low value are different. If a field says the data could
  not be established, say that rather than treating it as zero.
- Answers carry source URLs. Quote them.
- If the scan ran in dry mode its scores are placeholders; say so when quoting one.
"""


def _sources() -> tools_module.Sources:
    return tools_module.disk_sources()


def _session_for(peer: str) -> str:
    """Give this caller a session in our store, so the traffic is attributable."""
    manager = sessions_module.SessionManager()
    record = manager.route("mcp", peer=peer or "openclaw", kind="dm")
    return record.id


def build_server():
    """Construct the FastMCP server. Kept as a function so `--list` can skip stdio."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP(SERVER_NAME, instructions=INSTRUCTIONS)
    sources = _sources()
    functions = tools_module.named(sources)

    # The same callables the AutoGen agent gets. FastMCP reads the signature and
    # docstring for the schema, which is why they are written the way they are.
    exposed = {
        "vc_scan_facts": functions["scan_facts"],
        "vc_query": functions["query_companies"],
        "vc_compare": functions["compare_companies"],
        "vc_company": functions["company_detail"],
        "vc_company_live": functions["company_live"],
        "vc_search_docs": functions["search_docs"],
        "vc_memory_search": functions["memory_search"],
        "vc_memory_get": functions["memory_get"],
        "vc_start_scan": functions["start_scan"],
    }
    # `memory_note` is deliberately **not** here, and the reason is the same one
    # `gateway/relay.py` frames inbound text for: everything arriving through
    # OpenClaw was written by other people. A remote caller that can write to
    # `memory/` can put a sentence into the file the agent loads as fact — a
    # slow, quiet prompt injection with a persistence layer. Remembering is a
    # decision for the operator's own surface.
    #
    # `search_github` / `search_hacker_news` are omitted for a duller reason:
    # they spend rate limit on our policy gate to answer a question OpenClaw's
    # own agent can already answer with its web tools.
    for name, fn in exposed.items():
        server.add_tool(fn, name=name, description=(fn.__doc__ or "").strip().split("\n")[0])

    @server.tool(
        name="vc_status",
        description="What this pipeline currently holds: scan, docs and memory.",
    )
    def vc_status() -> str:
        """Whether a scan exists, how large the indexes are, and where state lives."""
        import docs_index
        import memory

        scan = sources.scan_getter()
        return json.dumps(
            {
                "state_dir": str(config.STATE),
                "agent": config.AGENT_ID,
                "scan": (
                    {
                        "file": scan.get("_source_name"),
                        "query": scan.get("query"),
                        "days": scan.get("days"),
                        "mode": scan.get("mode"),
                        "candidates": len(scan.get("candidates", [])),
                    }
                    if scan
                    else None
                ),
                "docs_sections": docs_index.stats()["sections"],
                "memory": memory.stats(),
                "gateway_url": tools_module.config_gateway_url(),
            },
            ensure_ascii=False,
            indent=1,
        )

    return server


def tool_names() -> list[str]:
    return [
        "vc_scan_facts", "vc_query", "vc_compare", "vc_company", "vc_company_live",
        "vc_search_docs", "vc_memory_search", "vc_memory_get", "vc_start_scan",
        "vc_status",
    ]


def main() -> None:
    if "--list" in sys.argv:
        # A check that does not need OpenClaw, a client, or a live stdio session.
        for name in tool_names():
            print(name)
        return

    # Register the caller as a session before serving, so even a client that only
    # ever lists tools shows up in the session index rather than being invisible.
    try:
        _session_for("openclaw")
    except Exception:  # noqa: BLE001 — attribution is not worth failing to serve
        pass

    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
