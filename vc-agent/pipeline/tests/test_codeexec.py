"""Kod yürütme: kapalı olması, kapıya takılması, ve kaçış kapağı kalması.

Üç ayrı endişe, üç ayrı sınıf. Sırayla en pahalıdan en ucuza:

1. **Kapı.** `"CodeExecutor"` hiçbir outbound markerına uymuyor, yani ada bakan
   yol onu geçirir. Kancanın onu yakaladığını ve imzanın *kodun kendisi* üstünde
   olduğunu doğrulamak bu dosyanın asıl işi.
2. **Kapalı varsayılan.** Özellik kapalıyken tool listeye hiç girmemeli ve adıyla
   çağrılırsa yine reddedilmeli.
3. **Kaçış kapağı.** Tarif "önce mevcut tool'lara bak" demeli. Demezse model her
   şeyi kodla yapmaya başlar ve yirmi bir tool boşa çalışır — ve bunu üretimde
   fark etmek pahalıdır.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import codeexec  # noqa: E402
import config  # noqa: E402
from gateway import approval as approval_module  # noqa: E402


class DescriptionTests(unittest.TestCase):
    """Tarif = arayüz. Model bu metne bakarak karar veriyor."""

    def test_it_tells_the_model_to_look_at_other_tools_first(self):
        text = codeexec.DESCRIPTION.lower()
        self.assertIn("önce", text)
        self.assertIn("tool", text)
        self.assertIn("son çare", text)

    def test_an_explicit_request_beats_the_last_resort_framing(self):
        # Measured: with "last resort" alone, asking "kod yazarak hesapla" still
        # produced prose arithmetic and zero tool calls. A framing strong enough
        # to stop casual use was strong enough to refuse a direct instruction.
        text = codeexec.DESCRIPTION.lower()
        self.assertIn("kod yaz", text)
        self.assertIn("çağır", text)

    def test_it_says_the_person_approves_before_it_runs(self):
        # The model writes differently when it knows a human will read the code.
        self.assertIn("onay", codeexec.DESCRIPTION.lower())

    def test_the_gate_reason_discloses_the_network(self):
        # The operator has to be able to know what they are *not* protected from.
        self.assertIn("ağ", codeexec.GATE_REASON.lower())


class AvailabilityTests(unittest.TestCase):
    def setUp(self):
        self._flag = config.ALLOW_CODE_EXEC
        codeexec._TOOL = None
        codeexec._EXECUTOR = None

    def tearDown(self):
        config.ALLOW_CODE_EXEC = self._flag
        codeexec._TOOL = None
        codeexec._EXECUTOR = None

    def test_off_by_default_means_no_tool_at_all(self):
        config.ALLOW_CODE_EXEC = False
        self.assertFalse(codeexec.available())
        self.assertIsNone(codeexec.build_tool())

    def test_start_reports_false_rather_than_raising_when_off(self):
        import asyncio

        config.ALLOW_CODE_EXEC = False
        self.assertFalse(asyncio.run(codeexec.start()))


class GateTests(unittest.TestCase):
    """Kancanın kendisi — Docker'a hiç dokunmadan."""

    def setUp(self):
        self.gate = approval_module.ApprovalGate(allow_all=False)
        self.hook = codeexec.make_gate_hook(self.gate)

    def _call(self, code: str) -> dict:
        return self.hook({"tool": codeexec.TOOL_NAME,
                          "arguments": {"code": code},
                          "session": "s1"})

    def test_another_tool_passes_untouched(self):
        self.assertEqual(
            self.hook({"tool": "search_docs", "arguments": {"query": "x"}}), {})

    def test_code_is_held(self):
        outcome = self._call("print(1)")
        self.assertTrue(outcome.get("block"))
        self.assertIn("approval_id", outcome)

    def test_the_signature_is_over_the_code_itself(self):
        # Approving one program must not approve a different one. This is the
        # frozen-plan idea at its smallest: what was shown is what runs.
        first = self._call("print(1)")
        self.gate.approve(first["approval_id"])
        self.assertTrue(self._call("import os; os.system('rm -rf /')")["block"])

    def test_an_approved_program_runs_once_and_then_asks_again(self):
        code = "print(6 * 7)"
        held = self._call(code)
        self.gate.approve(held["approval_id"])
        self.assertEqual(self._call(code), {"approved": True})
        # Consumed: the same program has to be approved again.
        self.assertTrue(self._call(code).get("block"))

    def test_a_broken_gate_closes_rather_than_opens(self):
        class Exploding:
            def require(self, *a, **k):
                raise RuntimeError("boom")

        outcome = codeexec.make_gate_hook(Exploding())(
            {"tool": codeexec.TOOL_NAME, "arguments": {"code": "print(1)"}})
        self.assertTrue(outcome["block"])
        self.assertIn("boom", outcome["reason"])

    def test_the_preview_carries_the_code_so_a_person_can_read_it(self):
        held = self._call("print('merhaba')")
        request = self.gate.get(held["approval_id"])
        self.assertIn("merhaba", str(request.preview))


class WorkbenchFilterTests(unittest.IsolatedAsyncioTestCase):
    """Listede olmamak yetmez: adıyla çağrılınca da reddedilmeli."""

    async def test_a_tool_not_offered_is_refused_by_name(self):
        from autogen_core.tools import StaticWorkbench

        from gateway import workbench as workbench_module

        inner = StaticWorkbench([])
        gated = workbench_module.GatedWorkbench(inner, allow=["search_docs"])
        result = await gated.call_tool(codeexec.TOOL_NAME, {"code": "print(1)"})
        self.assertTrue(result.is_error)
        self.assertIn("Refused", result.to_text())



class ReplayTests(unittest.TestCase):
    """Onaylanan metin saklanmalı — yoksa onaylanan şey hiç çalışmaz.

    Ölçülmüş sebep: kapının reddi turu bitiriyor, ve model aynı soruya iki kez
    aynı programı yazmıyor (iki koşu, iki farklı imza). Grant'i işaretlemek tek
    başına hiçbir işe yaramıyor; çalıştırılacak olanın *onaylanan metin* olması
    gerekiyor.
    """

    def setUp(self):
        self.gate = approval_module.ApprovalGate(allow_all=False)
        self.hook = codeexec.make_gate_hook(self.gate)

    def test_the_request_keeps_the_whole_program_not_a_clip(self):
        code = "print('x')\n" * 80          # `preview` 400 karakterde kesiliyor
        held = self.hook({"tool": codeexec.TOOL_NAME,
                          "arguments": {"code": code}, "session": "s"})
        request = self.gate.get(held["approval_id"])
        self.assertEqual(request.payload["code"], code)
        self.assertLess(len(str(request.preview["code"])), len(code))

    def test_two_different_programs_are_two_different_requests(self):
        a = self.hook({"tool": codeexec.TOOL_NAME,
                       "arguments": {"code": "print(1)"}, "session": "s"})
        b = self.hook({"tool": codeexec.TOOL_NAME,
                       "arguments": {"code": "print(2)"}, "session": "s"})
        self.assertNotEqual(a["approval_id"], b["approval_id"])

    def test_run_approved_says_so_rather_than_raising_when_off(self):
        import asyncio

        flag = config.ALLOW_CODE_EXEC
        config.ALLOW_CODE_EXEC = False
        codeexec._TOOL = None
        try:
            outcome = asyncio.run(codeexec.run_approved("print(1)"))
            self.assertFalse(outcome["ok"])
            self.assertIn("kapalı", outcome["output"])
        finally:
            config.ALLOW_CODE_EXEC = flag
            codeexec._TOOL = None

if __name__ == "__main__":
    unittest.main()
