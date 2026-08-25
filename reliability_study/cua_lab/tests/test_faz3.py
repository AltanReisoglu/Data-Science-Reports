"""
Faz 3 testleri — kalan 10 zihniyet + KATEGORİ tabanları.

Bu dosyanın iki işi var:

1. Her yeni zihniyetin AYIRT EDİCİ kararını kilitlemek. Aynı kategorideki
   stratejiler aynı ortak mantığı paylaşıyor; test edilmesi gereken şey
   mekanizma değil, o stratejinin o mekanizmayı NASIL kullandığı.

2. Yanlış pozitif taraması: 18 strateji × 3 meşru senaryo × 2 model = 108
   kombinasyon, hepsi `none` ile birebir aynı olmalı.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cua_lab import strategies as S
from cua_lab.events import Act, BudgetLimits, ComputerCall, Finish, Say
from cua_lab.loop import Runner, Status
from cua_lab.model import AlternatingModel, ScriptedModel, StubbornModel
from cua_lab.sandbox.fake import FakeSandbox

KAPALI = dict(max_steps=None, max_replans=None, max_tokens=None,
              max_seconds=None, max_cost_usd=None)


def run(scenario, model, strategy="none", **kw):
    return Runner("test gorevi", FakeSandbox(scenario), model, S.get(strategy),
                  limits=BudgetLimits(**{**KAPALI, **kw})).run()


def _solver(req):
    if "gonderildi" in req.screen:
        return Finish("bitti", tokens=150, cost_usd=0.0015)
    if 'Ad=""' in req.screen:
        if "Ad odaklandi" in req.screen:
            return ComputerCall(Act.TYPE, {"text": "Altan"}, tokens=350, cost_usd=0.0035)
        return ComputerCall(Act.LEFT_CLICK, {"x": 200, "y": 120}, tokens=350, cost_usd=0.0035)
    return ComputerCall(Act.LEFT_CLICK, {"x": 200, "y": 200}, tokens=350, cost_usd=0.0035)


def _liar(req):
    if "gonderildi" in req.screen:
        return Finish("bitti", tokens=150, cost_usd=0.0015)
    return Finish("gonderdim", tokens=200, cost_usd=0.002)


def _parrot(req):
    """Aynı fikri farklı kelimelerle tekrarlayan model — pi-signature'ın hedefi."""
    n = len([h for h in req.history if h.startswith("say:")])
    cumleler = [
        "Bir daha kontrol edeyim su durumu",
        "Tekrar kontrol etmem lazim su durumu",
        "Bunu bir kez daha kontrol edeyim durumu",
        "Su durumu tekrar bir kontrol edeyim",
    ]
    return Say(cumleler[n % len(cumleler)], tokens=200, cost_usd=0.002)


solver = lambda: ScriptedModel(_solver)          # noqa: E731
liar = lambda: ScriptedModel(_liar)              # noqa: E731
parrot = lambda: ScriptedModel(_parrot)          # noqa: E731
stuck = lambda: StubbornModel(Act.LEFT_CLICK, {"x": 200, "y": 200})   # noqa: E731
pingpong = lambda: AlternatingModel(                                   # noqa: E731
    ComputerCall(Act.LEFT_CLICK, {"x": 200, "y": 200}),
    ComputerCall(Act.LEFT_CLICK, {"x": 320, "y": 200}))


# ================================================== SAYAC kategorisi

class TestSayacKategorisi(unittest.TestCase):
    """Bes strateji AYNI sayaci kullaniyor. Test edilen: sayac dolunca
    verdikleri farkli cevaplar."""

    def test_arize_sert_keser_cevap_uretmez(self):
        r = run("dead_button", stuck(), "arize-control", max_steps=6)
        self.assertEqual(r.status, Status.BUDGET_EXHAUSTED)
        self.assertEqual(r.reason, "max_steps")     # birincil eksen adiyla
        self.assertIsNone(r.answer)

    def test_agentbudget_rezerv_birakir_ve_cevap_uretir(self):
        r = run("dead_button", stuck(), "agentbudget-dollar", max_steps=10)
        self.assertIsNotNone(r.answer)
        self.assertLess(r.steps, 10)                # %15 rezerv erken tetikliyor

    def test_grace_lutuf_turunda_arac_kilitli(self):
        """Kilit calisiyorsa model bitirmek ZORUNDA — ve kosum OK donmemeli."""
        r = run("dead_button", stuck(), "budget-grace:agentscope", max_steps=6)
        self.assertEqual(r.status, Status.DEGRADED)
        self.assertIsNotNone(r.answer)

    def test_grace_varsayilan_varyant_agentscope(self):
        """Varyantsiz secim ILK varyanti almali — belirsiz kalmamali."""
        a = run("dead_button", stuck(), "budget-grace", max_steps=6)
        b = run("dead_button", stuck(), "budget-grace:agentscope", max_steps=6)
        self.assertEqual((a.status, a.reason, a.steps), (b.status, b.reason, b.steps))

    def test_hermes_varyanti_adim_ekseninde_hic_uyarmaz(self):
        """Olculmus bulgu: adim ekseninde ara uyari modeli erken pes ettiriyor."""
        r = run("dead_button", stuck(), "budget-grace:hermes", max_steps=6)
        self.assertNotIn("adim ekseninde", "".join(r.nudges).lower())
        self.assertFalse(any("GERI SAYIM" in n for n in r.nudges))

    def test_iki_varyant_ayni_MEKANIZMAYI_paylasiyor(self):
        """Ayri strateji degil ayni stratejinin iki ayari olmalarinin kaniti:
        ikisi de ayni sinif, ayni kancalar, farkli sadece dugmeler."""
        a = S.get("budget-grace:agentscope").items[0]
        h = S.get("budget-grace:hermes").items[0]
        self.assertIs(type(a), type(h))
        self.assertNotEqual(a.grace_turns, h.grace_turns)
        self.assertNotEqual(a.warn_axes, h.warn_axes)

    def test_bilinmeyen_varyant_sessizce_yutulmaz(self):
        with self.assertRaises(S.UnknownVariant):
            S.get("budget-grace:olmayan")

    def test_claude_advisory_geri_sayimi_prompt_a_koyar(self):
        r = run("dead_button", stuck(), "claude-advisory",
                max_steps=20, max_tokens=3000)
        self.assertTrue(any("GERI SAYIM" in n for n in r.nudges),
                        f"geri sayim enjekte edilmedi: {r.nudges}")

    def test_hermes_ile_claude_zit_davraniyor(self):
        """Bu belgedeki en ogretici karsitlik — kodda da gorunmeli."""
        h = run("dead_button", stuck(), "budget-grace:hermes",
                max_steps=20, max_tokens=3000)
        c = run("dead_button", stuck(), "claude-advisory",
                max_steps=20, max_tokens=3000)
        self.assertEqual(len([n for n in h.nudges if "GERI SAYIM" in n]), 0)
        self.assertGreater(len([n for n in c.nudges if "GERI SAYIM" in n]), 0)


# ================================================== PENCERE kategorisi

class TestPencereKategorisi(unittest.TestCase):

    def test_openclaw_adlandirilmis_dedektor_dondurur(self):
        """Isim vermek kozmetik degil: hangi dedektorun konustugunu bilmek,
        ne oldugunu bilmek demek."""
        r = run("dead_button", stuck(), "openclaw-pingpong")
        self.assertEqual(r.status, Status.STUCK)
        self.assertIn(r.reason, ("pingPong", "genericRepeat",
                                 "genericRepeat_critical", "knownPollNoProgress"))

    def test_openclaw_kademelendiriyor_openhands_kademelendirmiyor(self):
        """Ayni pencere, farkli mudahale felsefesi — `escalate` dugmesi."""
        o = run("dead_button", stuck(), "openhands-stuck")
        c = run("dead_button", stuck(), "openclaw-pingpong")
        self.assertEqual(o.nudges, [])          # OpenHands: dogrudan STUCK
        self.assertGreater(len(c.nudges), 0)    # OpenClaw: once uyari

    def test_pi_yakin_benzer_metni_yakalar(self):
        """Ajan ayni cumleyi kurmuyor ama ayni fikri tekrar ediyor.
        Hicbir imza dedektoru bunu goremez — pi'nin varlik sebebi."""
        r = run("healthy", parrot(), "pi-signature")
        self.assertEqual(r.status, Status.STUCK)
        self.assertTrue(r.reason.startswith("signal"), r.reason)

    def test_pi_daha_KESIN_teshis_veriyor(self):
        """Ikisi de yakaliyor — fark TESHISTE.

        `openhands-stuck` "monolog" diyor: ajan cok konusuyor. Dogru ama
        yaklasik — farkli seyler soyluyor olsa da tetiklenirdi.
        `pi-signature` "yakin-benzer metin" diyor: ayni fikri yeniden ifade
        ediyor. Ikinci teshis eyleme cevrilebilir, birincisi degil.
        """
        oh = run("healthy", parrot(), "openhands-stuck", max_steps=30)
        pi = run("healthy", parrot(), "pi-signature", max_steps=30)
        self.assertEqual(oh.status, Status.STUCK)
        self.assertEqual(pi.status, Status.STUCK)
        self.assertEqual(oh.reason, "monologue")
        self.assertTrue(pi.reason.startswith("signal"), pi.reason)
        self.assertNotEqual(oh.reason, pi.reason)

    def test_pi_kademelendiriyor(self):
        r = run("dead_button", stuck(), "pi-signature")
        self.assertGreater(len(r.nudges), 0)


# ================================================== DUNYA kategorisi

class TestDunyaKategorisi(unittest.TestCase):

    def test_telemetry_dusen_kontrolun_ADINI_soyluyor(self):
        """Olculmus bulgu: cevabi vermek (%36) < hangi kontrol dustu (%45)."""
        r = run("healthy", liar(), "telemetry-repair")
        self.assertTrue(any("KONTROL DUSTU" in n for n in r.nudges), r.nudges)
        self.assertTrue(any(ad in "".join(r.nudges) for ad in
                            ("total_consistency", "required_coverage", "tool_contract")))

    def test_telemetry_onarim_butcesi_var(self):
        """Olculmus sinir: donguede yeniden calistirma ayni donguyu uretiyor.
        Dogru hamle onarmak degil DURMAK."""
        r = run("healthy", liar(), "telemetry-repair")
        self.assertEqual(r.status, Status.DEGRADED)
        self.assertEqual(r.reason, "repair_exhausted")

    def test_telemetry_gercekten_biten_isi_kabul_eder(self):
        r = run("healthy", solver(), "telemetry-repair")
        self.assertEqual(r.status, Status.OK)


# ================================================== SEKIL kategorisi

class TestSekilKategorisi(unittest.TestCase):

    def test_autogen_etkili_sinir_yoksa_KOSUM_BASLAMAZ(self):
        """Calisma zamaninda sifir maliyet — cunku calisma zamani hic gelmedi."""
        r = run("dead_button", stuck(), "autogen-static")   # tum eksenler KAPALI
        self.assertEqual(r.reason, "cycle_without_exit_condition")
        self.assertEqual(r.steps, 0)

    def test_autogen_tek_eksen_aciksa_baslatir(self):
        r = run("dead_button", stuck(), "autogen-static", max_steps=5)
        self.assertNotEqual(r.reason, "cycle_without_exit_condition")

    def test_modexa_dongu_tespit_etmiyor_olusturmuyor(self):
        r = run("dead_button", stuck(), "modexa-statemachine")
        self.assertEqual(r.status, Status.NEEDS_INPUT)
        self.assertIn("merdiven", (r.report.why if r.report else ""))

    def test_modexa_merdiveni_sirayla_tirmaniyor(self):
        """Ajan kendi retry'ini icat etmiyor; sabit merdiveni izliyor."""
        r = run("dead_button", stuck(), "modexa-statemachine")
        sira = [n for n in r.nudges if "MERDIVENI" in n]
        self.assertGreaterEqual(len(sira), 2)
        self.assertIn("backoff_retry", sira[0])

    def test_modexa_mesru_retry_i_kesmiyor(self):
        """DOGRULA durumu bir KAPI — ama mesru retry ondan gecebilmeli."""
        r = run("flaky", solver(), "modexa-statemachine", max_steps=20)
        self.assertEqual(r.status, Status.OK)


# ================================================== KARAR kategorisi

class TestKararKategorisi(unittest.TestCase):

    def test_voi_erken_cevabi_engelliyor(self):
        """Saf maliyet optimizasyonu tehlikeli: 'cevap ver' hep en ucuz eylem,
        yani baski arttikca OTOMATIK kazanir. Guard bunu engelliyor."""
        r = run("healthy", liar(), "voi-allocation", max_steps=8)
        self.assertTrue(any("ERKEN CEVAP ENGELLENDI" in n for n in r.nudges), r.nudges)

    def test_voi_butce_baskisini_prompt_a_yaziyor(self):
        r = run("dead_button", stuck(), "voi-allocation", max_steps=10)
        self.assertTrue(any("BUTCE BASKISI" in n for n in r.nudges), r.nudges)

    def test_improvement_loop_MUDAHALE_ETMEZ(self):
        """Bu zihniyetin tanimi: kosumu kurtarmaz, olcer."""
        taban = run("flaky", solver(), "none", max_steps=20)
        r = run("flaky", solver(), "improvement-loop", max_steps=20)
        self.assertEqual(r.steps, taban.steps)
        self.assertEqual(r.totals["tokens"], taban.totals["tokens"])

    def test_improvement_loop_esik_onerisi_uretiyor(self):
        r = run("flaky", solver(), "improvement-loop", max_steps=20)
        self.assertIsNotNone(r.report)
        self.assertIn("max_steps=", r.report.found)


# ================================================== kategori tabanlari

class TestKategoriTabanlari(unittest.TestCase):

    def test_her_strateji_bir_kategoriye_ait(self):
        from cua_lab.strategies.kinds import KATEGORI
        for c in S.catalog():
            with self.subTest(strateji=c.id):
                self.assertIn(getattr(c, "kind", None), KATEGORI)

    def test_her_stratejinin_secim_alanlari_dolu(self):
        """`why` / `action` / `blind_spot` bos kalamaz.

        Bir stratejiyi secerken uc soru var ve ucu de tek satirlik `mentality`
        ozetinde kayboluyor: neden gerekli, tetiklenince ne yapiyor, neyi
        kaciriyor. Yeni bir strateji bu alanlar bos halde eklenemesin.
        """
        for c in S.catalog():
            with self.subTest(strateji=c.id):
                for alan in ("mentality", "why", "action", "blind_spot"):
                    metin = getattr(c, alan, "")
                    self.assertTrue(metin, f"{c.id}.{alan} bos")
                    self.assertGreater(len(metin), 25,
                                       f"{c.id}.{alan} cok kisa: {metin!r}")

    def test_oncelik_sirasi_benzersiz(self):
        """Uygulama onceligi bir SIRA — iki strateji ayni numarayi alamaz."""
        oncelikler = [c.priority for c in S.catalog() if c.priority]
        self.assertEqual(len(oncelikler), len(set(oncelikler)),
                         f"cakisan oncelik: {sorted(oncelikler)}")

    def test_oncelik_1_sert_butce(self):
        """Kaynaklarin ortak sonucu: baska hicbir sey uygulanmayacaksa
        once sert bir tavan konmali."""
        birinci = next(c for c in S.catalog() if c.priority == 1)
        self.assertEqual(birinci.id, "arize-control")

    def test_belgelenmis_on_yedi_kaynak_kapsaniyor(self):
        """17 KAYNAK var ama 16 STRATEJI — `budget-grace` ikisini birden
        kapsiyor (agentscope + hermes), cunku 25 kombinasyonun 25'inde
        ayirt edilemiyorlar. Varyantlarla birlikte sayim 17'yi buluyor."""
        strateji = set(S.all_ids()) - {"none"}
        varyant = sum(max(len(c.variants), 1) for c in S.catalog()
                      if c.id in strateji)
        self.assertEqual(len(strateji), 16)
        self.assertEqual(varyant, 17)

    def test_ayni_kategoridekiler_ortak_tabani_paylasiyor(self):
        """Kullanicinin tasarim sarti: framework ortak, zihniyet ayri."""
        from cua_lab.strategies import kinds
        taban = {"budget": kinds.BudgetStrategy, "window": kinds.WindowStrategy,
                 "evidence": kinds.EvidenceStrategy, "shape": kinds.ShapeStrategy,
                 "decision": kinds.DecisionStrategy}
        for c in S.catalog():
            if c.kind in taban:
                with self.subTest(strateji=c.id):
                    self.assertTrue(issubclass(c, taban[c.kind]))


# ================================================== yanlis pozitif taramasi

class TestYanlisPozitifTaramasiTam(unittest.TestCase):
    """18 strateji x 3 mesru senaryo x 2 model = 108 kombinasyon.

    Hepsi `none` ile BIREBIR ayni olmali — ayni adim, ayni token. Bir
    dedektorun ikinci sinavi, yakalamamasi gerekeni rahat birakmaktir.
    """

    def test_hicbir_strateji_mesru_kosumu_kesmiyor(self):
        for senaryo in ("healthy", "flaky", "silent_success"):
            for model in (solver, ScriptedModel and (lambda: ScriptedModel(_solver))):
                taban = run(senaryo, model(), "none", max_steps=20)
                for sid in S.all_ids():
                    with self.subTest(senaryo=senaryo, strateji=sid):
                        r = run(senaryo, model(), sid, max_steps=20)
                        self.assertEqual(r.status, Status.OK)
                        self.assertEqual(r.steps, taban.steps)
                        self.assertEqual(r.totals["tokens"], taban.totals["tokens"])


if __name__ == "__main__":
    unittest.main()
