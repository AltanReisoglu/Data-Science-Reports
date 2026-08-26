"""
Faz 1 testleri — stdlib unittest, bağımlılık yok.

    python3 -m unittest discover -s tests -v

Testlerin çoğu bir davranışı değil bir TASARIM KARARINI kilitliyor. Ve
"tetiklenmeli" testleri kadar "tetiklenmemeli" testleri de var: bir dedektörün
ikinci sınavı, yakalamaması gerekeni rahat bırakmaktır.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cua_lab import strategies as S
from cua_lab.events import Act, BudgetLimits, ComputerCall, Event, EventKind, Finish
from cua_lab.loop import Runner, Status
from cua_lab.model import AlternatingModel, ScriptedModel, StubbornModel
from cua_lab.sandbox.fake import SCENARIOS, FakeSandbox
from cua_lab.trace import read_trace


def run(scenario: str, model, strategy: str = "none", **kw):
    limits = BudgetLimits(**{"max_steps": None, "max_replans": None, "max_tokens": None,
                             "max_seconds": None, "max_cost_usd": None, **kw})
    return Runner("test gorevi", FakeSandbox(scenario), model, S.get(strategy),
                  limits=limits).run()


def solver(req):
    if "gonderildi" in req.screen:
        return Finish("bitti", tokens=150, cost_usd=0.0015)
    if 'Ad=""' in req.screen:
        if "Ad odaklandi" in req.screen:
            return ComputerCall(Act.TYPE, {"text": "Altan"}, tokens=350, cost_usd=0.0035)
        return ComputerCall(Act.LEFT_CLICK, {"x": 200, "y": 120}, tokens=350, cost_usd=0.0035)
    return ComputerCall(Act.LEFT_CLICK, {"x": 200, "y": 200}, tokens=350, cost_usd=0.0035)


# ---------------------------------------------------------------- olay modeli


class TestEvents(unittest.TestCase):
    def test_volatile_fields_ignored(self):
        """Her turda degisen alanlar imzaya girmemeli.

        Girerse iki ozdes cagri hicbir zaman esit cikmaz ve dedektor sessizce
        hicbir sey bulmaz — hata turlerinin en kotusu, cunku calisiyor gorunur.
        """
        a = Event(EventKind.ACTION, "click", {"x": 1, "tool_call_id": "a"})
        b = Event(EventKind.ACTION, "click", {"x": 1, "tool_call_id": "b"})
        self.assertEqual(a.signature(), b.signature())

    def test_real_args_still_distinguish(self):
        """Oynak alanlari atmak GERCEK argumanlari da atmak demek degil."""
        a = Event(EventKind.ACTION, "click", {"x": 1})
        b = Event(EventKind.ACTION, "click", {"x": 2})
        self.assertNotEqual(a.signature(), b.signature())

    def test_wait_is_not_expected_to_change_screen(self):
        """`wait` mesru olarak ekrani degistirmez — durgunluk sayilmamali.

        Referans dongude `wait` ayri bir eylem; bunu ayirmazsak modelin
        mesru beklemesi dongu sanilir.
        """
        self.assertFalse(Act.WAIT.mutates_screen)
        self.assertFalse(Act.SCREENSHOT.mutates_screen)
        self.assertTrue(Act.LEFT_CLICK.mutates_screen)

    def test_unexecuted_result_is_not_an_error(self):
        """Referans dongudeki 'Not executed' durumu hatadan FARKLI."""
        from cua_lab.events import ToolResult
        ev = ToolResult(executed=False).to_event("click")
        self.assertIs(ev.kind, EventKind.OBSERVATION)
        self.assertFalse(ev.meta["executed"])


# ------------------------------------------------------------------- sandbox


class TestSandbox(unittest.TestCase):
    def test_dead_button_is_silent(self):
        """Olu buton: hata YOK, ekran degisikligi YOK. Sessiz durgunluk."""
        s = FakeSandbox("dead_button"); s.start()
        before = s.screen_hash()
        r = s.execute(Act.LEFT_CLICK, {"x": 200, "y": 200})
        self.assertIsNone(r.error)
        self.assertEqual(before, s.screen_hash())

    def test_flaky_recovers(self):
        """Flaky arac ilk iki denemede hata verir, ucuncude calisir."""
        s = FakeSandbox("flaky"); s.start()
        s.execute(Act.LEFT_CLICK, {"x": 200, "y": 120})
        s.execute(Act.TYPE, {"text": "x"})
        errs = [s.execute(Act.LEFT_CLICK, {"x": 200, "y": 200}).error for _ in range(3)]
        self.assertIsNotNone(errs[0]); self.assertIsNotNone(errs[1])
        self.assertIsNone(errs[2])

    def test_all_scenarios_start_and_stop(self):
        for sc in SCENARIOS:
            with self.subTest(scenario=sc):
                s = FakeSandbox(sc); s.start()
                self.assertTrue(s.describe())
                s.stop()

    def test_unknown_scenario_rejected(self):
        with self.assertRaises(ValueError):
            FakeSandbox("yok-boyle-bir-sey")


# ------------------------------------------------------------ taban cizgisi


class TestNoneBaseline(unittest.TestCase):
    def test_reference_loop_is_unbounded(self):
        """`none`, Anthropic referans dongusunu yeniden uretiyor: sinirsiz.

        Bu bir kusur degil OLCUM: 'kontrol koymanin faydasi ne' sorusunun
        cevabi bu sutunla digerleri arasindaki fark.
        """
        res = run("dead_button", StubbornModel(), "none")
        self.assertIs(res.status, Status.CEILING)
        self.assertGreater(res.totals["steps"], 100)

    def test_none_finishes_healthy_run(self):
        res = run("healthy", ScriptedModel(solver), "none")
        self.assertIs(res.status, Status.OK)


# ----------------------------------------------------------- openhands-stuck


class TestOpenHandsStuck(unittest.TestCase):
    def test_catches_silent_stagnation(self):
        res = run("dead_button", StubbornModel(), "openhands-stuck")
        self.assertIs(res.status, Status.STUCK)
        self.assertLess(res.totals["steps"], 10)

    def test_catches_alternating_loop(self):
        """A-B-A-B: ardisik tekrar YOK. 'ayni cagri iki kez' heuristigi kacirir."""
        m = AlternatingModel(
            ComputerCall(Act.LEFT_CLICK, {"x": 200, "y": 200}, tokens=350),
            ComputerCall(Act.LEFT_CLICK, {"x": 320, "y": 200}, tokens=350),
        )
        res = run("dead_button", m, "openhands-stuck")
        self.assertIs(res.status, Status.STUCK)
        self.assertTrue(res.reason.startswith(("cycle_", "repeat_", "no_progress")),
                        f"beklenmeyen sebep: {res.reason}")

    def test_stop_report_has_four_parts(self):
        """OpenHands STUCK'i ayri terminal durum sayiyor; raporu da bilgilendirici."""
        res = run("dead_button", StubbornModel(), "openhands-stuck")
        self.assertIsNotNone(res.report)
        self.assertTrue(res.report.why and res.report.tried and res.report.next_step)

    # -- YANLIS POZITIF kontrolleri ------------------------------------

    def test_healthy_run_not_flagged(self):
        res = run("healthy", ScriptedModel(solver), "openhands-stuck")
        self.assertIs(res.status, Status.OK, f"saglikli kosum {res.reason} ile kesildi")

    def test_legitimate_retry_not_flagged(self):
        """Flaky araca karsi tekrar denemek MESRU — dongu degil.

        Kaynaklarda kasitli yanlis pozitif olarak isaretlenen desen buydu;
        burada tetiklenmemesi gerekiyor cunku ekran sonunda degisiyor.
        """
        res = run("flaky", ScriptedModel(solver), "openhands-stuck")
        self.assertIs(res.status, Status.OK, f"mesru retry {res.reason} ile kesildi")

    def test_guardrail_is_free_on_healthy_runs(self):
        """Saglikli kosumda guardrail tek token bindirmiyor."""
        a = run("healthy", ScriptedModel(solver), "none")
        b = run("healthy", ScriptedModel(solver), "openhands-stuck")
        self.assertEqual(a.totals["steps"], b.totals["steps"])
        self.assertEqual(a.totals["tokens"], b.totals["tokens"])


# --------------------------------------------------------------------- iz


class TestTrace(unittest.TestCase):
    def test_totals_include_the_final_step(self):
        """REGRESYON: totaller son span yazilmadan hesaplaniyordu.

        Sonuc: ozet bir adim eksik sayiyor ve son adimin token'i kayboluyordu.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.jsonl"
            r = Runner("t", FakeSandbox("dead_button"), StubbornModel(),
                       S.get("openhands-stuck"), trace_path=p).run()
            spans = read_trace(p)
            self.assertEqual(len(spans), r.totals["steps"])
            self.assertEqual(sum(s.tokens for s in spans), r.totals["tokens"])

    def test_one_span_per_iteration(self):
        """Arize sarti: butun kosum tek span ise tekrar deseni gorulemez."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.jsonl"
            Runner("t", FakeSandbox("dead_button"), StubbornModel(),
                   S.get("openhands-stuck"), trace_path=p).run()
            spans = read_trace(p)
            self.assertEqual([s.i for s in spans], list(range(len(spans))))
            self.assertTrue(all(s.screen_hash for s in spans))

    def test_final_span_carries_the_verdict(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.jsonl"
            Runner("t", FakeSandbox("dead_button"), StubbornModel(),
                   S.get("openhands-stuck"), trace_path=p).run()
            last = read_trace(p)[-1]
            self.assertEqual(last.verdict, "stop")
            self.assertEqual(last.verdict_by, "openhands-stuck")


# ---------------------------------------------------------------- kayit defteri


class TestRegistry(unittest.TestCase):
    def test_strategies_registered(self):
        self.assertIn("none", S.all_ids())
        self.assertIn("openhands-stuck", S.all_ids())

    def test_every_strategy_declares_its_source(self):
        """Her strateji hangi zihniyeti temsil ettigini soylemek zorunda."""
        for c in S.catalog():
            with self.subTest(strategy=c.id):
                self.assertTrue(c.mentality, f"{c.id}: zihniyet yazilmamis")
                self.assertTrue(c.source, f"{c.id}: kaynak yazilmamis")

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(KeyError):
            S.get("boyle-bir-strateji-yok")

    def test_strategies_are_composable(self):
        stack = S.get("none,openhands-stuck")
        self.assertEqual(len(stack.items), 2)
        self.assertEqual(stack.id, "none,openhands-stuck")


if __name__ == "__main__":
    unittest.main(verbosity=2)
