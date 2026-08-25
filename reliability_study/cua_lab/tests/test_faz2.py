"""
Faz 2 testleri — altı yeni zihniyet.

Her strateji için İKİ test var:
  * tetiklenmeli — bozuk ortamda yakalıyor mu
  * TETİKLENMEMELİ — meşru koşumu rahat bırakıyor mu

İkincisi daha önemli. Yalnızca yakaladıklarını gösteren bir demo, dedektörün
yanlış pozitif oranı hakkında hiçbir şey söylemez: "her koşumu döngü sayan" bir
dedektör de o demoyu geçerdi.

Ayrıca her stratejinin AYRI bir terminal durum döndürdüğü kilitleniyor — Arize'ın
ilkesi: "tamamlandı", "adım limiti", "bütçe aşıldı" ve "hata" farklı sonuçlardır.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cua_lab import strategies as S
from cua_lab.events import Act, BudgetLimits, ComputerCall, Finish
from cua_lab.loop import Runner, Status
from cua_lab.model import AlternatingModel, ScriptedModel, StubbornModel
from cua_lab.sandbox.fake import FakeSandbox

KAPALI = dict(max_steps=None, max_replans=None, max_tokens=None,
              max_seconds=None, max_cost_usd=None)


def run(scenario, model, strategy="none", **kw):
    limits = BudgetLimits(**{**KAPALI, **kw})
    return Runner("test gorevi", FakeSandbox(scenario), model, S.get(strategy),
                  limits=limits).run()


def _solver(req):
    """Görevi çözüp BİTİREN model — hiçbir stratejinin tetiklenmemesi gereken koşum."""
    if "gonderildi" in req.screen:
        return Finish("bitti", tokens=150, cost_usd=0.0015)
    if 'Ad=""' in req.screen:
        if "Ad odaklandi" in req.screen:
            return ComputerCall(Act.TYPE, {"text": "Altan"}, tokens=350, cost_usd=0.0035)
        return ComputerCall(Act.LEFT_CLICK, {"x": 200, "y": 120}, tokens=350, cost_usd=0.0035)
    return ComputerCall(Act.LEFT_CLICK, {"x": 200, "y": 200}, tokens=350, cost_usd=0.0035)


def _liar(req):
    """Hiçbir şey yapmadan 'bitirdim' diyen model."""
    if "gonderildi" in req.screen:
        return Finish("bitti", tokens=150, cost_usd=0.0015)
    return Finish("gonderdim", tokens=200, cost_usd=0.002)


def solver():
    return ScriptedModel(_solver)


def liar():
    return ScriptedModel(_liar)


def stuck():
    return StubbornModel(Act.LEFT_CLICK, {"x": 200, "y": 200})


def pingpong():
    return AlternatingModel(ComputerCall(Act.LEFT_CLICK, {"x": 200, "y": 200}),
                            ComputerCall(Act.LEFT_CLICK, {"x": 320, "y": 200}))


# --------------------------------------------------------------- arize-control

class TestArizeControl(unittest.TestCase):
    def test_adim_limiti_birincil_ve_sebep_max_steps(self):
        """Adım limiti diğer eksenlerden önce bakılmalı ki sebep 'max_steps' yazsın.

        Bu kozmetik değil: durma sebebinin kaydı bu zihniyetin ikinci katkısı.
        'budget_tokens' demek, limitin dar olduğu bilgisini gizler.
        """
        r = run("dead_button", stuck(), "arize-control", max_steps=5)
        self.assertEqual(r.status, Status.BUDGET_EXHAUSTED)
        self.assertEqual(r.reason, "max_steps")

    def test_sert_keser_nihai_cevap_uretmez(self):
        """Arize sert keser — agentbudget'in aksine kısmi cevap toplamaz."""
        r = run("dead_button", stuck(), "arize-control", max_steps=5)
        self.assertIsNone(r.answer)

    def test_mesru_kosumu_kesmez(self):
        r = run("flaky", solver(), "arize-control", max_steps=12)
        self.assertEqual(r.status, Status.OK)


# ------------------------------------------------------------- strands-entropy

class TestStrandsEntropy(unittest.TestCase):
    def test_ardisik_tekrari_yakalar(self):
        r = run("dead_button", stuck(), "strands-entropy")
        self.assertEqual(r.status, Status.STUCK)
        self.assertEqual(r.reason, "low_diversity")

    def test_donusumlu_dongu_de_ayni_kuralla_yakalanir(self):
        """A-B-A-B: tek kural bütün desenleri yakalıyor — k taraması gerekmiyor.

        Bu zihniyetin bütün iddiası bu. Ardışık tekrar sayan dedektörler bu
        dizide hiç tetiklenmez; çeşitlilik ölçüsü aynı eşikle yakalar.
        """
        r = run("dead_button", pingpong(), "strands-entropy")
        self.assertEqual(r.status, Status.STUCK)
        self.assertEqual(r.reason, "low_diversity")

    def test_kisa_mesru_kosum_yargilanmaz(self):
        """Pencere dolmadan karar verilmiyor — yanlış pozitifin en ucuz savunması."""
        r = run("healthy", solver(), "strands-entropy")
        self.assertEqual(r.status, Status.OK)

    def test_mesru_retry_kesilmez(self):
        r = run("flaky", solver(), "strands-entropy")
        self.assertEqual(r.status, Status.OK)


# ----------------------------------------------------------- loopguard-dignity

class TestLoopGuardDignity(unittest.TestCase):
    def test_cekimser_kalma_ayri_terminal_durum(self):
        """NEEDS_INPUT hata değil: 'durmak başarısızlık değildir, durmak kontroldür'."""
        r = run("dead_button", stuck(), "loopguard-dignity")
        self.assertEqual(r.status, Status.NEEDS_INPUT)
        self.assertEqual(r.reason, "abstain_need_input")

    def test_dort_alanli_rapor_uretiyor(self):
        """Sonsuz döngüyü faydalı kısmi sonuca çeviren şey bu yapı."""
        r = run("dead_button", stuck(), "loopguard-dignity")
        self.assertIsNotNone(r.report)
        for alan in ("why", "tried", "found", "next_step"):
            self.assertTrue(getattr(r.report, alan), f"{alan} bos")

    def test_kaynagin_esigi_duzeltilmis_mesru_retry_kesilmiyor(self):
        """KAYNAKTAN SAPMA — ölçerek bulundu.

        Kaynak `state_hash` eşiğini 3, retry'i 2 veriyor. `flaky` izinde ekran
        1-2-3. adımlarda aynı kalıyor ve başarı 4. adımda geliyor; kaynağın
        sayıları meşru koşumu BAŞARIDAN BİR ADIM ÖNCE keserdi. Bu test o
        düzeltmeyi kilitliyor.
        """
        r = run("flaky", solver(), "loopguard-dignity")
        self.assertEqual(r.status, Status.OK)

    def test_ilerleyen_retry_sayaci_sifirlar(self):
        r = run("silent_success", solver(), "loopguard-dignity")
        self.assertEqual(r.status, Status.OK)


# ----------------------------------------------------------------- verify-gate

class TestVerifyGate(unittest.TestCase):
    def test_kanitsiz_bitirme_iddiasi_reddedilir(self):
        """'Bitirdim' bir istektir. `none` bu koşumu OK sayıyor — fark bu."""
        taban = run("healthy", liar(), "none")
        self.assertEqual(taban.status, Status.OK)          # kontrolsüz: kabul

        r = run("healthy", liar(), "verify-gate")
        self.assertEqual(r.status, Status.DEGRADED)
        self.assertEqual(r.reason, "verify_failed")

    def test_red_kosumu_bitirmez_donguyu_surdurur(self):
        """Kritik ayrıntı: kapı açılmazsa koşum BİTMEZ, gözleme geri döner.

        Ajan tek adımda 'bitirdim' diyor; verify-gate ile birden fazla adım
        atılmış olmalı — yani reddedilip döngüye geri konmuş.
        """
        r = run("healthy", liar(), "verify-gate")
        self.assertGreater(r.steps, 1)

    def test_gercekten_biten_is_kabul_edilir(self):
        r = run("healthy", solver(), "verify-gate")
        self.assertEqual(r.status, Status.OK)

    def test_kor_nokta_bitirme_iddiasi_yoksa_sessiz(self):
        """Ajan hiç 'bitirdim' demezse bu kapı hiç açılmaz — belgelenmiş sınır.

        Bu bir kusur değil, kapsam sınırı; bütçe stratejisiyle birlikte
        kullanılmalı. Test onu kilitliyor ki belgede yazan sınır kodda da dursun.
        """
        r = run("dead_button", stuck(), "verify-gate")
        self.assertEqual(r.status, Status.CEILING)


# ------------------------------------------------------------- galileo-breaker

class TestGalileoBreaker(unittest.TestCase):
    def test_kalici_bozuk_arac_devre_kesici_acar(self):
        r = run("broken_tool", solver(), "galileo-breaker")
        self.assertEqual(r.status, Status.DEGRADED)
        self.assertEqual(r.reason, "tool_circuit_open")

    def test_gecici_hata_kesilmez_ama_israf_raporlanir(self):
        """`flaky` meşru retry: oran düşüyor, devre kesici açılmamalı.

        Ama boşa giden çağrılar yine de rapora giriyor — Galileo'nun asıl derdi
        kesmek değil, 'hakkında hiç ticket açılmayan hatayı' görünür kılmak.
        """
        r = run("flaky", solver(), "galileo-breaker")
        self.assertEqual(r.status, Status.OK)

    def test_hatasiz_kosumda_hic_konusmaz(self):
        r = run("healthy", solver(), "galileo-breaker")
        self.assertEqual(r.status, Status.OK)
        self.assertEqual(r.nudges, [])


# ---------------------------------------------------------- agentbudget-dollar

class TestAgentBudgetDollar(unittest.TestCase):
    def test_nihai_cevap_payi_kismi_sonuc_uretir(self):
        """Arize ile ayrım noktası: aynı bütçe, ama boş elle dönülmüyor."""
        r = run("dead_button", stuck(), "agentbudget-dollar", max_steps=10)
        self.assertEqual(r.status, Status.BUDGET_EXHAUSTED)
        self.assertIsNotNone(r.answer)

    def test_sert_limitten_once_durur(self):
        """Rezerv %15: 10 adımlık limitte 10'a varmadan durmalı ki
        toparlayacak bütçe kalsın."""
        r = run("dead_button", stuck(), "agentbudget-dollar", max_steps=10)
        self.assertLess(r.steps, 10)

    def test_mesru_kosumu_kesmez(self):
        r = run("flaky", solver(), "agentbudget-dollar", max_steps=12)
        self.assertEqual(r.status, Status.OK)


# ------------------------------------------------------- hepsi birden: kontrol

class TestYanlisPozitifTaramasi(unittest.TestCase):
    """Bir dedektörün ikinci sınavı: yakalamaması gerekeni rahat bırakmak.

    Kayıtlı HER strateji, HER meşru senaryoda `none` ile AYNI sonucu vermeli —
    aynı adım, aynı token. Guardrail çalışan koşuma yük bindirmemeli.
    """

    def test_mesru_kosumlarda_hicbiri_tetiklenmiyor(self):
        for senaryo in ("healthy", "flaky", "silent_success"):
            taban = run(senaryo, solver(), "none", max_steps=20)
            for sid in S.all_ids():
                with self.subTest(senaryo=senaryo, strateji=sid):
                    r = run(senaryo, solver(), sid, max_steps=20)
                    self.assertEqual(r.status, Status.OK)
                    self.assertEqual(r.steps, taban.steps)
                    self.assertEqual(r.totals["tokens"], taban.totals["tokens"])

    def test_her_strateji_ayri_terminal_durum_veriyor(self):
        """Farklı zihniyetler farklı SONUÇ üretmeli; aynıysa ayrı strateji değildir."""
        sonuclar = {
            "arize-control": Status.BUDGET_EXHAUSTED,
            "agentbudget-dollar": Status.BUDGET_EXHAUSTED,
            "loopguard-dignity": Status.NEEDS_INPUT,
            "openhands-stuck": Status.STUCK,
            "strands-entropy": Status.STUCK,
        }
        for sid, beklenen in sonuclar.items():
            with self.subTest(strateji=sid):
                r = run("dead_button", stuck(), sid, max_steps=12)
                self.assertEqual(r.status, beklenen)


if __name__ == "__main__":
    unittest.main()
