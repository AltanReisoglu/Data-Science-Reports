"""The mechanism panel's feed: what it claims is happening, and when.

The panel exists to teach, so a wrong stage is worse than a missing one — it
teaches something false about AutoGen and nothing fails. These tests are mostly
about *order* and *honesty* rather than about the presence of a field.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import unittest

import stages


class CatalogueTest(unittest.TestCase):
    def test_every_entry_names_a_real_class(self) -> None:
        """No paraphrases. The panel prints `klass` verbatim as evidence."""
        for mech in stages.CATALOGUE.values():
            self.assertTrue(mech.klass, mech.id)
            self.assertRegex(
                mech.klass, r"[A-Za-z_]",
                f"{mech.id} has no identifier in its class field",
            )

    def test_every_entry_cites_something_checkable(self) -> None:
        """Her girdi doğrulanabilir bir yere atıf vermeli.

        AutoGen mekanizmaları kılavuzun satırına (`08:2298`), MAF mekanizmaları
        kendi sembolüne (`maf:FunctionTool`). İkisi de bir okuyucunun gidip
        bakabileceği bir yer; atıfsız bir iddia, birkaç gün sonra yanlış
        anlatılıyor.

        Üçüncü biçim sonradan eklendi: `18:150`. Zamanlayıcı mekanizmalarının
        AutoGen'de atıf verecekleri bir yer **yok**, çünkü AutoGen'de zamanlama
        diye bir kavram yok — ve bu bir eksiklik değil, bir kütüphane saat
        tutmaz. Onlar bizim kendi analizimize (`docs/18`) atıf veriyor. Kural
        gevşemiyor: hâlâ bu depoda açılıp bakılabilecek bir satır isteniyor.
        """
        for mech in stages.CATALOGUE.values():
            pattern = (r"^maf:[A-Za-z_]+$" if mech.lane == stages.MAF
                       else r"^(0[58]|18):\d+$")
            self.assertRegex(
                mech.ref, pattern,
                f"{mech.id} atıfsız ya da yanlış biçimde: {mech.ref!r}",
            )

    def test_maf_citations_point_at_symbols_that_exist(self) -> None:
        """`maf:X` atıfları uydurma olmasın.

        Ayrı sanal ortam kurulu değilse atlanıyor: bu test bir kurulum
        gereksinimi getirmemeli, ama kuruluysa iddiayı gerçekten sınamalı.
        """
        import subprocess
        from pathlib import Path

        python = Path(__file__).resolve().parents[2] / ".venv-maf" / "bin" / "python"
        if not python.exists():
            self.skipTest(".venv-maf kurulu değil")
        symbols = sorted({m.ref.split(":", 1)[1] for m in stages.CATALOGUE.values()
                          if m.lane == stages.MAF})
        code = ("import agent_framework as af;"
                "print(','.join(n for n in %r if hasattr(af, n)))" % symbols)
        found = subprocess.run([str(python), "-c", code], capture_output=True,
                               text=True, timeout=120).stdout.strip().split(",")
        for symbol in symbols:
            self.assertIn(symbol, found, f"agent_framework.{symbol} yok")

    def test_every_entry_names_a_file_that_exists(self) -> None:
        """"Which module ran" has to be answerable, and answerable correctly.

        A path that has drifted is worse than no path: it sends the reader to a
        file that does not explain what they just watched happen.
        """
        root = pathlib.Path(__file__).resolve().parents[2]
        for mech in stages.CATALOGUE.values():
            self.assertTrue(mech.module, f"{mech.id} names no module")
            self.assertTrue(
                (root / mech.module).exists(),
                f"{mech.id} points at {mech.module}, which is not there",
            )

    def test_lanes_are_the_known_ones(self) -> None:
        """Dört şerit: AutoGen'in iki katmanı, bizim hattımız, ve MAF.

        MAF ayrı bir şerit çünkü ayrı bir çerçeve. AutoGen'in katmanlarıyla aynı
        renge boyamak, ekranın anlattığı ayrımı silerdi.
        """
        known = (stages.AGENTCHAT, stages.CORE, stages.OURS, stages.MAF)
        for mech in stages.CATALOGUE.values():
            self.assertIn(mech.lane, known, mech.id)

    def test_maf_mechanisms_are_never_labelled_autogen(self) -> None:
        for stage_id in stages.MAF_FLOW:
            self.assertEqual(stages.CATALOGUE[stage_id].lane, stages.MAF, stage_id)

    def test_our_own_machinery_is_not_labelled_autogen(self) -> None:
        """The gate and the compacting context are ours; AutoGen ships neither.

        Drawing them in AutoGen's lane would teach somebody that AutoGen has a
        built-in approval gate. It does not — the cookbook shows how to build one.
        """
        self.assertEqual(stages.CATALOGUE["gate"].lane, stages.OURS)
        self.assertEqual(stages.CATALOGUE["context"].lane, stages.OURS)
        self.assertEqual(stages.CATALOGUE["compaction"].lane, stages.OURS)

    def test_catalogue_survives_the_json_round_trip(self) -> None:
        import json

        payload = json.loads(json.dumps(stages.catalogue()))
        self.assertEqual(len(payload), len(stages.CATALOGUE))
        self.assertEqual(
            {row["id"] for row in payload}, set(stages.CATALOGUE),
        )


class StageBusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bus = stages.StageBus()

    def test_emit_carries_the_catalogue_text(self) -> None:
        self.bus.emit("model", streaming=True)
        (event,) = self.bus.drain()
        self.assertEqual(event["type"], "stage")
        self.assertEqual(event["id"], "model")
        self.assertEqual(event["klass"], "model_client.create_stream()")
        self.assertEqual(event["meta"], {"streaming": True})

    def test_drain_is_ordered_and_empties(self) -> None:
        for stage_id in ("context", "model", "stream"):
            self.bus.emit(stage_id)
        self.assertEqual([e["id"] for e in self.bus.drain()], ["context", "model", "stream"])
        self.assertEqual(self.bus.drain(), [])

    def test_an_unknown_id_is_dropped_not_raised(self) -> None:
        """A typo in a caller must not take a turn down with it."""
        self.bus.emit("no-such-stage")
        self.assertEqual(self.bus.drain(), [])

    def test_a_full_queue_drops_rather_than_blocks(self) -> None:
        """The panel is a window onto the work; it must never hold up the work."""
        bus = stages.StageBus(maxsize=3)
        for _ in range(50):
            bus.emit("stream")
        self.assertEqual(len(bus.drain()), 3)


class LineStreamTest(unittest.TestCase):
    """Stages that cross a process boundary.

    The scan runs as a subprocess, so its core stages travel as tagged lines in
    the same stdout the human log comes from. Two things must hold: a tagged line
    must never reach the log, and an untagged line must never be mistaken for a
    stage.
    """

    def setUp(self) -> None:
        self._was = os.environ.get(stages.STREAM_ENV)
        os.environ[stages.STREAM_ENV] = "1"

    def tearDown(self) -> None:
        if self._was is None:
            os.environ.pop(stages.STREAM_ENV, None)
        else:
            os.environ[stages.STREAM_ENV] = self._was

    def _emit(self, stage_id: str, **meta) -> str:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            stages.emit_line(stage_id, **meta)
        return buffer.getvalue().strip()

    def test_a_line_round_trips(self) -> None:
        line = self._emit("publish", topic="enrichment_task", subscribers=3)
        event = stages.parse_line(line)
        self.assertIsNotNone(event)
        self.assertEqual(event["id"], "publish")
        self.assertEqual(event["lane"], stages.CORE)
        self.assertEqual(event["meta"]["subscribers"], 3)

    def test_ordinary_log_lines_are_not_stages(self) -> None:
        """The scan prints banners and tables; none of them may be swallowed."""
        for line in (
            "  4 · ENRICH + SCORE  ·  5 candidates, parallel branches",
            "",
            "{}",
            "##STAGE not json at all",
            "##STAGE " + json.dumps({"id": "no-such-mechanism"}),
        ):
            with self.subTest(line=line):
                self.assertIsNone(stages.parse_line(line))

    def test_silent_unless_the_server_asks(self) -> None:
        """`python scan.py` by hand must stay readable."""
        os.environ.pop(stages.STREAM_ENV, None)
        self.assertEqual(self._emit("publish"), "")

    def test_turkish_survives_the_round_trip(self) -> None:
        line = self._emit("branch", branch="teknik", note="ölçüldü · şirket")
        event = stages.parse_line(line)
        self.assertEqual(event["meta"]["note"], "ölçüldü · şirket")


class FlowTest(unittest.TestCase):
    def test_both_flows_name_real_mechanisms(self) -> None:
        """The panel draws a box per id; an unknown one would be a blank box."""
        for flow in (stages.CHAT_FLOW, stages.SCAN_FLOW):
            for stage_id in flow:
                self.assertIn(stage_id, stages.CATALOGUE, stage_id)

    def test_the_scan_is_graphflow_not_core_pubsub(self) -> None:
        """The correction this test exists to hold.

        The scan looks like it should be the core showcase — parallel branches,
        a join, an own runtime. It is not: `scan.py` calls `graph.enrich`, which
        is **GraphFlow**, AgentChat. `fanin.py` is the core pub/sub engine and
        only `compare_fanin.py` runs it. The panel drew the core diagram over a
        scan for a while, and nothing failed; this is what would have failed.
        """
        self.assertIn("graph_build", stages.SCAN_FLOW)
        self.assertIn("graph_run", stages.SCAN_FLOW)
        for pubsub_only in ("publish", "subscribe", "collect", "branch"):
            self.assertNotIn(
                pubsub_only, stages.SCAN_FLOW,
                f"{pubsub_only} belongs to fanin.py, which the scan does not call",
            )

    def test_the_fanin_mechanisms_are_still_catalogued(self) -> None:
        """They are real and measured — just not on the scan's path."""
        for stage_id in ("runtime_start", "subscribe", "publish", "branch", "collect"):
            self.assertEqual(stages.CATALOGUE[stage_id].lane, stages.CORE, stage_id)


class GateStageTest(unittest.IsolatedAsyncioTestCase):
    """The gate reports itself, because a hook after it would never run.

    `HookRegistry.run` stops the chain at whichever hook returns the terminal key,
    so a probe registered after the gate goes silent on exactly the blocked calls
    — the ones worth watching. These tests pin that reasoning down.
    """

    async def asyncSetUp(self) -> None:
        from gateway import hooks, workbench

        self.hooks, self.workbench_module = hooks, workbench
        self.seen: list[tuple[str, dict]] = []
        self.registry = hooks.HookRegistry()

    def _bench(self, inner, **kw):
        return self.workbench_module.GatedWorkbench(
            inner, registry=self.registry, session_id="t",
            on_stage=lambda sid, **meta: self.seen.append((sid, meta)), **kw
        )

    async def test_an_allowed_call_reports_gate_then_execution(self) -> None:
        bench = self._bench(_Inner())
        await bench.call_tool("query_companies", {})
        self.assertEqual([sid for sid, _ in self.seen], ["gate", "tool_exec"])
        self.assertFalse(self.seen[0][1]["blocked"])

    async def test_a_blocked_call_reports_the_block_and_never_executes(self) -> None:
        inner = _Inner()
        self.registry.register(
            self.hooks.BEFORE_TOOL_CALL,
            lambda payload: {"block": True, "reason": "no"},
            name="deny",
        )
        bench = self._bench(inner)
        result = await bench.call_tool("messages_send", {})

        self.assertEqual([sid for sid, _ in self.seen], ["gate"], "tool must not run")
        self.assertTrue(self.seen[0][1]["blocked"])
        self.assertEqual(self.seen[0][1]["reason"], "no")
        self.assertTrue(result.is_error)
        self.assertEqual(inner.calls, [], "the inner workbench was never reached")

    async def test_a_filtered_name_reports_a_block_too(self) -> None:
        """Filtering and gating are different mechanisms with the same outcome."""
        bench = self._bench(_Inner(), allow=["query_companies"])
        await bench.call_tool("permissions_respond", {})
        self.assertEqual([sid for sid, _ in self.seen], ["gate"])
        self.assertTrue(self.seen[0][1]["blocked"])
        self.assertIn("not offered", self.seen[0][1]["reason"])

    async def test_a_broken_reporter_cannot_break_a_tool_call(self) -> None:
        def explode(_sid, **_meta):
            raise RuntimeError("panel is on fire")

        bench = self.workbench_module.GatedWorkbench(
            _Inner(), registry=self.registry, session_id="t", on_stage=explode
        )
        result = await bench.call_tool("query_companies", {})
        self.assertFalse(result.is_error)

    async def test_no_reporter_is_the_normal_case(self) -> None:
        bench = self.workbench_module.GatedWorkbench(
            _Inner(), registry=self.registry, session_id="t"
        )
        result = await bench.call_tool("query_companies", {})
        self.assertFalse(result.is_error)


class _Inner:
    """A workbench stand-in that records what actually reached it."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def list_tools(self):
        return [{"name": "query_companies"}, {"name": "permissions_respond"}]

    async def call_tool(self, name, arguments=None, cancellation_token=None, call_id=None):
        from autogen_core.tools import TextResultContent, ToolResult

        self.calls.append(name)
        return ToolResult(name=name, result=[TextResultContent(content="ok")], is_error=False)


if __name__ == "__main__":
    unittest.main()
