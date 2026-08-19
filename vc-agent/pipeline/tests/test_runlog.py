"""Akış ekranının veri tarafı: kayıt, türetme, ve iz.

Bu ekranın tek gerçek riski **uydurma**. Bir grafı çizmek kolay; o grafın o
soruda gerçekten olan biteni göstermesi zor. Testlerin çoğu bu yüzden "doğru mu
çiziyor"u değil, **yanlış bir şey iddia etmiyor mu**yu ölçüyor:

* sohbet turunda beş takımdan hiçbiri "kullanıldı" diye işaretlenmemeli,
* bir desen ancak kayıtta kanıtı varken kullanıldı sayılmalı,
* kapıda tutulan çağrı "koştu" sütununa yazılmamalı — bu sayacı bir kez yanlış
  okuyup "model tool çağırmıyor" teşhisi koyduk, ölçüm dosyası artık bu.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import runlog  # noqa: E402


def _chat(*events, question: str = "soru") -> runlog.Run:
    run = runlog.Run(id="t-1", kind="chat", question=question)
    for e in events:
        run.event(e)
    return run


STAGE_CONTEXT = {"type": "stage", "id": "context", "lane": "ours",
                 "name": "Bağlam", "meta": {"tokens": 900, "budget": 10000}}
DONE = {"type": "done", "text": "cevap", "stop_reason": ""}


class RecordingTests(unittest.TestCase):
    def test_chunks_are_not_kept(self):
        # Bir turda binlerce chunk var ve hiçbiri çizilmiyor. Tutmak, kaydı
        # kullanılmayan veriyle şişirmekten başka bir şey yapmazdı.
        run = _chat({"type": "chunk", "text": "a"}, {"type": "chunk", "text": "b"})
        self.assertEqual(run.events, [])

    def test_a_broken_event_does_not_break_the_turn(self):
        run = _chat()
        run.event({"type": "stage", "id": object()})   # JSON'a girmeyecek bir şey
        run.event(None)                                # type: ignore[arg-type]
        self.assertEqual(run.status, "running")

    def test_the_record_is_capped(self):
        run = _chat()
        for i in range(700):
            run.event({"type": "tool", "name": f"t{i}"})
        self.assertLessEqual(len(run.events), 600)


class GraphTests(unittest.TestCase):
    def test_each_tool_call_becomes_its_own_node(self):
        run = _chat({"type": "tool", "name": "query_companies"},
                    {"type": "tool_result", "name": "query_companies"},
                    {"type": "tool", "name": "query_companies"},
                    {"type": "tool_result", "name": "query_companies"})
        tools = [n for n in run.graph()["nodes"] if n["kind"] == "tool"]
        # Aynı tool iki kez çağrıldıysa grafta iki kutu var: ikisi de gerçekten
        # oldu, ve tek kutuya indirmek turun uzunluğunu gizlerdi.
        self.assertEqual(len(tools), 2)

    def test_the_return_edge_is_marked_so_layout_does_not_invert(self):
        run = _chat({"type": "tool", "name": "x"}, {"type": "tool_result", "name": "x"})
        back = [e for e in run.graph()["edges"] if e.get("back")]
        self.assertEqual(len(back), 1)
        self.assertEqual(back[0]["message"], "ToolCallExecutionEvent")

    def test_the_request_edge_carries_the_gate_verdict(self):
        run = _chat({"type": "stage", "id": "gate", "meta": {"blocked": True}},
                    {"type": "tool", "name": "openclaw_call"})
        edge = next(e for e in run.graph()["edges"]
                    if e["message"] == "ToolCallRequestEvent")
        self.assertEqual(edge["gate"], "red")

    def test_a_turn_with_no_answer_has_no_answer_node(self):
        run = _chat(STAGE_CONTEXT)
        self.assertNotIn("answer", [n["id"] for n in run.graph()["nodes"]])


class HonestyTests(unittest.TestCase):
    """Ekranın uydurmadığını ölçen testler."""

    def test_a_chat_turn_claims_no_team(self):
        run = _chat(STAGE_CONTEXT, DONE)
        self.assertEqual([t for t in run.teams() if t["used"]], [])

    def test_every_team_still_says_why_it_did_not_run(self):
        # Koşmayanları listeden atmak kolaydı; asıl anlatım orada.
        run = _chat(STAGE_CONTEXT, DONE)
        self.assertEqual(len(run.teams()), 5)
        for team in run.teams():
            self.assertTrue(team["why"])

    def test_a_chat_turn_matches_none_of_the_eight_patterns(self):
        run = _chat(STAGE_CONTEXT, DONE)
        self.assertEqual([p for p in run.patterns() if p["used"]], [])

    def test_code_execution_is_claimed_only_with_evidence(self):
        without = _chat(STAGE_CONTEXT, DONE)
        self.assertFalse(self._pattern(without, "codeexec")["used"])
        with_code = _chat(STAGE_CONTEXT,
                          {"type": "stage", "id": "code_request",
                           "meta": {"code": "print(1)"}},
                          {"type": "stage", "id": "code_result",
                           "meta": {"output": "1", "seconds": 0.1}}, DONE)
        self.assertTrue(self._pattern(with_code, "codeexec")["used"])

    def test_a_scan_claims_graphflow_and_the_two_patterns_it_composes(self):
        run = runlog.Run(id="s-1", kind="scan", question="tarama · ai")
        run.event(DONE)
        self.assertEqual([t["id"] for t in run.teams() if t["used"]], ["graphflow"])
        self.assertEqual(sorted(p["id"] for p in run.patterns() if p["used"]),
                         ["concurrent", "sequential"])

    def test_the_pattern_line_references_are_the_guide_s_own(self):
        # Bu tablo bir kez kılavuza bakmadan yazıldı ve üç satırı hayaliydi.
        # Atıflar artık dosyadan okundu; test onların kaybolmasını engelliyor.
        refs = {p["id"]: p["ref"] for p in runlog.PATTERNS}
        self.assertEqual(refs["concurrent"], "05:3236")
        self.assertEqual(refs["codeexec"], "05:6188")
        self.assertEqual(len(runlog.PATTERNS), 8)

    def _pattern(self, run: runlog.Run, pid: str) -> dict:
        return next(p for p in run.patterns() if p["id"] == pid)


class LiveTests(unittest.TestCase):
    """Canlı vurgu: yanan yer, o an gerçekten koşulan yer olmalı."""

    def test_a_finished_turn_lights_nothing(self):
        # Bitmiş bir turda bir kutunun yanık kalması, ekranın kipi hakkında
        # yalan söylemesi olurdu: yanan yer "şu an" demek.
        run = _chat(STAGE_CONTEXT, DONE)
        run.end()
        self.assertIsNone(run.active())

    def test_a_model_stage_lights_the_agent(self):
        run = _chat({"type": "stage", "id": "model", "name": "Model çağrısı"})
        self.assertEqual(run.active()["node"], "agent")

    def test_the_gate_lights_the_gateway_band_not_the_agent(self):
        # Kapı AutoGen'in bir özelliği değil. Işığı ajan kutusunda yakmak,
        # destede kurduğumuz ayrımı ekranda çürütürdü.
        run = _chat({"type": "tool", "name": "openclaw_call"},
                    {"type": "stage", "id": "gate", "name": "Kapı"})
        lit = run.active()["node"]
        self.assertEqual(lit, "gate")
        node = next(n for n in run.graph()["nodes"] if n["id"] == lit)
        self.assertEqual(node["band"], 1)
        self.assertEqual(node["lane"], "ours")

    def test_execution_lights_the_running_tool_and_the_workbench(self):
        run = _chat({"type": "tool", "name": "a"}, {"type": "tool_result", "name": "a"},
                    {"type": "tool", "name": "b"},
                    {"type": "stage", "id": "tool_exec", "name": "Tool koşuyor"})
        lit = run.active()["node"]
        # İki tool var; yanması gereken sonuncusu — ve onu çağıran workbench.
        self.assertIn("tool:b:1", lit)
        self.assertIn("wb", lit)

    def test_the_two_bands_separate_autogen_from_us(self):
        run = _chat(STAGE_CONTEXT, {"type": "tool", "name": "x"}, DONE)
        bands = {n["id"]: n["band"] for n in run.graph()["nodes"]}
        self.assertEqual(bands["agent"], 0)          # AgentChat
        self.assertEqual(bands["gate"], 1)           # bizim
        self.assertEqual(bands["ctx"], 1)            # bizim

    def test_a_scan_lights_all_three_branches_at_once(self):
        # Eşzamanlılığın görünür olduğu tek an. Tek kutu yakmak, üç dalın
        # paralel koştuğu iddiasını ekranda çürütürdü.
        run = runlog.Run(id="s-3", kind="scan", question="tarama")
        run.event({"type": "stage", "id": "analysts", "name": "Analistler"})
        self.assertEqual(run.active()["node"], ["technical", "market", "team"])

    def test_an_unmapped_stage_lights_nothing_rather_than_guessing(self):
        run = _chat({"type": "stage", "id": "subscribe", "name": "Abonelikler"})
        active = run.active()
        self.assertIsNone(active["node"])
        self.assertIsNone(active["edge"])

    def test_every_chat_stage_has_somewhere_to_light(self):
        # `stages.py` yeni bir aşama eklerse tablo sessizce eksik kalıyor ve
        # o aşama boyunca ekranda hiçbir şey yanmıyor. Test onu yakalıyor.
        import stages
        for stage_id in stages.CHAT_FLOW:
            self.assertIn(stage_id, runlog.STAGE_TARGET, stage_id)


class NamingTests(unittest.TestCase):
    """Aşama adları gerçekten geliyor mu.

    Ölçüldü: `Mechanism`'ın alanı `title`, ve burası `name` diye okuyordu. Hata
    hiçbir yerde patlamıyor — yalnız ekranda her aşama adsız çıkıyor, ve satırlar
    bir tire ile başlıyor. Sessiz olduğu için test edilmesi gereken cins bir hata.
    """

    def test_a_stage_carries_its_name_through_the_record(self):
        import stages

        bus_shaped = {"type": "stage", **stages.CATALOGUE["gate"].as_dict()}
        run = _chat(bus_shaped)
        row = run.timeline()[0]
        self.assertTrue(row["name"])
        self.assertEqual(row["name"], stages.CATALOGUE["gate"].title)
        self.assertTrue(run.active()["name"])


class SequenceTests(unittest.TestCase):
    """Sıra diyagramı: graf yapıyı, bu zamanı anlatıyor."""

    def _run(self, *ids):
        import stages
        return _chat(*[{"type": "stage", **stages.CATALOGUE[i].as_dict()}
                       for i in ids])

    def test_a_chat_turn_opens_with_the_question(self):
        seq = self._run("context").sequence()
        self.assertEqual((seq["steps"][0]["src"], seq["steps"][0]["dst"]),
                         ("user", "agent"))

    def test_the_gate_is_its_own_lifeline(self):
        # Kapıyı ajanın içine saklamak, diyagramın anlattığı tek şeyi silerdi:
        # ayrı bir karar noktası olduğunu.
        seq = self._run("gate").sequence()
        self.assertIn("gate", [l["id"] for l in seq["lanes"]])
        step = next(s for s in seq["steps"] if s.get("stage") == "gate")
        self.assertEqual((step["src"], step["dst"]), ("agent", "gate"))

    def test_a_refusal_draws_the_return_arrow(self):
        run = _chat({"type": "stage", "id": "gate", "title": "Kapı",
                     "meta": {"blocked": True}})
        steps = run.sequence()["steps"]
        self.assertTrue(any(s.get("blocked") for s in steps))

    def test_context_is_a_self_message(self):
        seq = self._run("context").sequence()
        step = next(s for s in seq["steps"] if s.get("stage") == "context")
        self.assertEqual(step["kind"], "self")

    def test_the_tool_block_is_wrapped_in_an_alt(self):
        seq = self._run("model", "tool_request", "gate", "tool_exec",
                        "tool_result", "done").sequence()
        kinds = [g["kind"] for g in seq["groups"]]
        self.assertIn("alt", kinds)
        self.assertIn("loop", kinds)

    def test_repeated_steps_carry_their_own_data(self):
        # Üç analist dalı üst üste aynı metni yazıyordu ve diyagram, üç ayrı
        # olayın aynı olay olduğunu ima ediyordu.
        run = runlog.Run(id="s-9", kind="scan", question="tarama")
        for branch, n in (("MarketAnalyst", 1), ("TeamAnalyst", 2),
                          ("TechnicalAnalyst", 3)):
            run.event({"type": "stage", "id": "analysts", "title": "Dallar paralel",
                       "meta": {"branch": branch, "arrived": n, "expected": 3}})
        labels = [s["label"] for s in run.sequence()["steps"]
                  if s.get("stage") == "analysts"]
        self.assertEqual(labels, ["MarketAnalyst · 1/3", "TeamAnalyst · 2/3",
                                  "TechnicalAnalyst · 3/3"])

    def test_a_label_never_breaks_the_diagram(self):
        run = runlog.Run(id="s-10", kind="scan", question="t")
        run.event({"type": "stage", "id": "graph_build", "title": "Graf",
                   "meta": {"branches": None}})       # bozuk meta
        self.assertTrue(run.sequence()["steps"][0]["label"])

    def test_a_turn_with_no_tool_has_no_alt_block(self):
        seq = self._run("model", "stream", "done").sequence()
        self.assertEqual([g["kind"] for g in seq["groups"]], ["loop"])


class DetailTests(unittest.TestCase):
    def test_every_stage_on_both_flows_can_be_explained(self):
        # Bir aşamanın anlatımı yoksa kutuya basınca hiçbir şey açılmıyor, ve
        # bu sessizce oluyor.
        import stages
        for stage_id in stages.CHAT_FLOW + stages.SCAN_FLOW:
            self.assertIsNotNone(stages.detail(stage_id), stage_id)

    def test_an_explanation_answers_all_four_questions(self):
        import stages
        for stage_id, entry in stages.DETAILS.items():
            for key in ("what", "how", "why", "trap"):
                self.assertTrue(entry.get(key), f"{stage_id}.{key}")

    def test_only_the_stages_that_happened_are_sent(self):
        run = _chat({"type": "stage", "id": "gate", "title": "Kapı"})
        self.assertEqual(list(run.details()), ["gate"])


class TotalsTests(unittest.TestCase):
    def test_an_mcp_call_counts_as_run_even_though_the_llm_counter_says_zero(self):
        """Ölçüldü (chat-0010): `ask_question` kapıdan geçti, `tool_exec`
        yayınlandı, 11,6 sn sonra sonuç döndü — ve `done.tool_calls` 0 dedi.

        O sayaç `BaseTool`'dan doğan `ToolCallEvent`'leri sayıyor ve MCP
        çağrıları o olayı yaymıyor. Sayaca güvenen ekran, kendi sıra
        diyagramıyla çelişiyordu: diyagramda `call_tool()` var, sayaçta yok.
        """
        run = _chat({"type": "tool", "name": "ask_question"},
                    {"type": "stage", "id": "gate",
                     "meta": {"tool": "ask_question", "blocked": False}},
                    {"type": "stage", "id": "tool_exec",
                     "meta": {"tool": "ask_question", "kind": "McpWorkbench"}},
                    {"type": "stage", "id": "done",
                     "meta": {"llm_calls": 2, "tokens": 18606, "tool_calls": 0}},
                    DONE)
        totals = run.totals()
        self.assertEqual(totals["tools_requested"], 1)
        self.assertEqual(totals["tools_ran"], 1)
        self.assertEqual(totals["tools_blocked"], 0)

    def test_a_blocked_call_is_requested_but_not_run(self):
        """Ölçülmüş hata: `done`'ın tool sayacı yalnız koşanları sayıyor.

        Sıfıra bakıp "model tool çağırmıyor" demek yanlış bir teşhisti; sekiz
        koşu boyunca yanlış yeri aradık. İstenen ile koşan artık ayrı sütun.
        """
        run = _chat({"type": "stage", "id": "gate", "meta": {"blocked": True}},
                    {"type": "tool", "name": "CodeExecutor"},
                    {"type": "stage", "id": "done",
                     "meta": {"llm_calls": 2, "tokens": 900, "tool_calls": 0}},
                    DONE)
        totals = run.totals()
        self.assertEqual(totals["tools_requested"], 1)
        self.assertEqual(totals["tools_ran"], 0)
        self.assertEqual(totals["tools_blocked"], 1)


    def test_a_team_run_reports_its_cost_from_team_done(self):
        """Ölçüldü (team-0009): 16 sn, üç konuşmacı, ekranda "0 LLM çağrısı".

        Bitiş aşaması sohbette `done`, takımda `team_done`. Yalnız birincisine
        bakan sayaç, takım koşusunun maliyetini sıfır gösteriyordu — ve sıfır,
        üç ajan konuşmuşken imkânsız bir sayı.
        """
        run = runlog.Run(id="tt", kind="team", question="s")
        run.variant = "graphflow"
        run.event({"type": "stage", "id": "team_done", "title": "Takım bitti",
                   "meta": {"llm_calls": 4, "tokens": 5120, "tool_calls": 0}})
        totals = run.totals()
        self.assertEqual(totals["llm_calls"], 4)
        self.assertEqual(totals["tokens"], 5120)


class ComponentTests(unittest.TestCase):
    def test_installed_is_not_the_same_as_used(self):
        # Bu ekranın en çok işe yarayan yeri: core runtime ayakta ama sohbet
        # turu ona hiç uğramıyor, ve liste bunu söylemek zorunda.
        run = _chat(STAGE_CONTEXT, DONE)
        runtime = next(c for c in run.components()
                       if c["name"] == "SingleThreadedAgentRuntime")
        self.assertFalse(runtime["used"])
        self.assertIn("uğramıyor", runtime["did"])

    def test_the_context_component_reports_the_measured_budget(self):
        run = _chat(STAGE_CONTEXT, DONE)
        ctx = next(c for c in run.components()
                   if c["name"] == "CompactingChatCompletionContext")
        self.assertIn("900", ctx["did"])
        self.assertIn("10000", ctx["did"])


class TopicTests(unittest.TestCase):
    def test_a_chat_turn_says_plainly_that_no_topic_was_used(self):
        run = _chat(STAGE_CONTEXT, DONE)
        self.assertFalse(run.topics()["active"])

    def test_the_scan_note_says_edges_order_rather_than_carry(self):
        # "MAF veri taşıyorsa bizim graf ne taşıyor" sorusunun cevabı bu satır,
        # ve ekranda görünen tek yer burası.
        run = runlog.Run(id="s-2", kind="scan", question="tarama")
        self.assertIn("sırayı belirliyor", run.topics()["note"])


class CodeTests(unittest.TestCase):
    def test_a_request_without_a_result_is_shown_as_still_running(self):
        run = _chat({"type": "stage", "id": "code_request",
                     "meta": {"code": "print(1)"}})
        self.assertTrue(run.code_runs()[0]["running"])

    def test_the_program_and_its_output_are_paired(self):
        run = _chat({"type": "stage", "id": "code_request", "meta": {"code": "print(6*7)"}},
                    {"type": "stage", "id": "code_result",
                     "meta": {"output": "42", "seconds": 0.2}})
        entry = run.code_runs()[0]
        self.assertEqual(entry["code"], "print(6*7)")
        self.assertEqual(entry["output"], "42")


class TraceTests(unittest.TestCase):
    def test_the_trace_records_the_shape_not_the_answer(self):
        run = _chat(STAGE_CONTEXT, {"type": "tool", "name": "query_companies"}, DONE,
                    question="hangi şirketler var")
        record = run.trace_record()
        self.assertEqual(record["question"], "hangi şirketler var")
        self.assertEqual(record["tools"], ["query_companies"])
        self.assertNotIn("cevap", json.dumps(record, ensure_ascii=False))

    def test_ending_a_run_appends_one_line(self):
        before = (config.RUN_TRACE.read_text(encoding="utf-8").count("\n")
                  if config.RUN_TRACE.exists() else 0)
        run = _chat(STAGE_CONTEXT, DONE)
        run.end()
        after = config.RUN_TRACE.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(after), before + 1)
        json.loads(after[-1])          # yazılan satır okunabilir olmalı

    def test_a_trace_failure_never_breaks_the_turn(self):
        original = config.RUN_TRACE
        config.RUN_TRACE = Path("/nonexistent-dir-for-a-test/runs.jsonl")
        try:
            run = _chat(STAGE_CONTEXT, DONE)
            run.end()                   # fırlatmamalı
            self.assertEqual(run.status, "done")
        finally:
            config.RUN_TRACE = original


class StoreTests(unittest.TestCase):
    def test_the_store_evicts_and_keeps_the_newest(self):
        log = runlog.RunLog(cap=3)
        for i in range(5):
            log.begin("chat", f"s{i}")
        listing = log.listing()
        self.assertEqual(len(listing), 3)
        self.assertEqual(listing[0]["question"], "s4")

    def test_latest_can_be_filtered_by_session(self):
        log = runlog.RunLog()
        log.begin("chat", "a", session="one")
        log.begin("chat", "b", session="two")
        self.assertEqual(log.latest("one").question, "a")


if __name__ == "__main__":
    unittest.main()
