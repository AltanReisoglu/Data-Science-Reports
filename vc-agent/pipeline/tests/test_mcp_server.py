"""The MCP server OpenClaw talks to: schemas, sourcing, and honest failure."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest
from pathlib import Path

import config
import mcp_server
from gateway import tools as tools_module


class ToolSurfaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_advertised_tool_is_actually_registered(self) -> None:
        """`--list` is what a person checks with; it must not lie."""
        server = mcp_server.build_server()
        registered = {t.name for t in await server.list_tools()}
        self.assertEqual(registered, set(mcp_server.tool_names()))

    async def test_schemas_come_from_the_signatures(self) -> None:
        server = mcp_server.build_server()
        by_name = {t.name: t for t in await server.list_tools()}

        company = by_name["vc_company"]
        self.assertEqual(company.inputSchema.get("required"), ["name"])

        # Optional arguments must stay optional, or a client is forced to invent
        # values for them.
        get = by_name["vc_memory_get"]
        self.assertEqual(get.inputSchema.get("required"), ["path"])
        self.assertIn("start", get.inputSchema.get("properties", {}))

    async def test_descriptions_are_present(self) -> None:
        """A tool with no description is a tool the model calls by guessing."""
        server = mcp_server.build_server()
        for tool in await server.list_tools():
            self.assertTrue((tool.description or "").strip(), f"{tool.name} has no description")

    def test_the_tools_are_the_same_callables_the_agent_gets(self) -> None:
        """One definition, two consumers — the point of `gateway/tools.py`."""
        shared = tools_module.named(tools_module.disk_sources())
        for name in ("scan_facts", "company_detail", "search_docs", "memory_search"):
            self.assertIn(name, shared)


class SourcingTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_scan_is_reported_rather_than_faked(self) -> None:
        sources = tools_module.disk_sources()
        functions = tools_module.named(sources)
        self.assertIn("No scan has been run yet", functions["scan_facts"]())

    async def test_reading_the_latest_scan_from_the_state_directory(self) -> None:
        """The MCP process has no gateway memory, so this path is the only one."""
        payload = {"query": "fintech", "days": 7, "mode": "dry", "candidates": []}
        path = config.OUTPUT / "scan-20260814-120000.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            data = tools_module.read_latest_scan()
            self.assertIsNotNone(data)
            self.assertEqual(data["query"], "fintech")
            self.assertEqual(data["_source_name"], path.name)
        finally:
            path.unlink()

    async def test_a_corrupt_scan_file_reads_as_no_scan(self) -> None:
        path = config.OUTPUT / "scan-20260814-999999.json"
        path.write_text("{ truncated", encoding="utf-8")
        try:
            self.assertIsNone(tools_module.read_latest_scan())
        finally:
            path.unlink()

    async def test_starting_a_scan_with_no_gateway_says_so(self) -> None:
        """Silently starting an invisible second scan would be the worse answer."""
        os.environ["VC_GATEWAY_URL"] = "http://127.0.0.1:9"  # discard port
        try:
            functions = tools_module.named(tools_module.disk_sources())
            message = functions["start_scan"]("fintech", 7)
        finally:
            os.environ.pop("VC_GATEWAY_URL", None)

        self.assertIn("Not started", message)
        self.assertIn("no gateway reachable", message)
        self.assertIn("python -m pipeline.server", message)

    async def test_status_reports_where_state_lives(self) -> None:
        server = mcp_server.build_server()
        result = await server.call_tool("vc_status", {})
        text = _text_of(result)
        self.assertIn(str(config.STATE), text)
        self.assertIn("docs_sections", text)


class SessionAttributionTests(unittest.TestCase):
    def test_an_mcp_caller_gets_a_session_like_any_other_channel(self) -> None:
        """Machine traffic is attributable too, or half the audit answers nothing."""
        session_id = mcp_server._session_for("openclaw")
        self.assertIn(":mcp:", session_id)
        self.assertIn("openclaw", session_id)


class StdioRoundTripTests(unittest.IsolatedAsyncioTestCase):
    """The real protocol, over a real spawned process — what OpenClaw actually does."""

    async def test_a_client_can_list_and_call_over_stdio(self) -> None:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:  # pragma: no cover
            self.skipTest("mcp client not available")

        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "pipeline.mcp_server"],
            cwd=str(Path(config.ROOT).parent),
            env={**os.environ, "VC_STATE_DIR": str(config.STATE), "PYTHONPATH": str(config.ROOT)},
        )

        async def run() -> None:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listing = await session.list_tools()
                    names = {t.name for t in listing.tools}
                    self.assertEqual(names, set(mcp_server.tool_names()))

                    result = await session.call_tool("vc_scan_facts", {})
                    text = " ".join(
                        getattr(part, "text", "") for part in result.content
                    )
                    self.assertTrue(text.strip(), "a tool call returned nothing")

        try:
            await asyncio.wait_for(run(), timeout=60)
        except asyncio.TimeoutError:  # pragma: no cover
            self.fail("stdio round trip timed out")


def _text_of(result) -> str:
    content = result[0] if isinstance(result, tuple) else result
    if isinstance(content, list):
        return " ".join(str(getattr(part, "text", part)) for part in content)
    return str(content)


if __name__ == "__main__":
    unittest.main()
