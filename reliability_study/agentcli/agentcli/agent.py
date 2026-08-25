"""
Ajan döngüsü — üç araç, VLM, ve takılabilir guardrail zihniyeti.

Zihniyetler KOPYALANMADI: `cua_lab.strategies` doğrudan kullanılıyor. PDF'teki
14 zihniyetin hepsi burada da geçerli, çünkü hepsi aynı `Strategy` protokolüne
(sekiz kanca) uyuyor ve o protokol araç kümesinden bağımsız.

Döngü sırası, `cua_lab/loop.py` ile aynı ve aynı gerekçeyle:
    bütçe (adımdan ÖNCE) → model → araç → gözlem → dedektör
Sonra bakmak, limiti aşan çağrının parasını ödedikten sonra fark etmek demek.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# cua_lab'i yol'a ekle — zihniyetler oradan geliyor.
_KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_KOK / "cua_lab"))

from cua_lab import strategies as S                     # noqa: E402
from cua_lab.detect.guardrails import (                 # noqa: E402
    Action, BudgetEnforcer, BudgetLimits, Event, EventKind, LoopDetector,
)
from cua_lab.loop import Status                         # noqa: E402
from cua_lab.trace import Span, TraceWriter             # noqa: E402

from . import dogrula as _dogrula                      # noqa: E402
from .golge import GolgeKurulu                          # noqa: E402
from .panel import Panel                                # noqa: E402
from .render import Rapor                               # noqa: E402


@dataclass
class Sonuc:
    status: Status
    reason: str
    steps: int
    totals: dict
    answer: str | None = None
    report: object | None = None
    nudges: list = field(default_factory=list)
    golge: list = field(default_factory=list)


@dataclass
class Baglam:
    """Stratejilerin okuduğu koşum durumu — `cua_lab.loop.RunContext` ile
    aynı alanlar, çünkü kancalar bu alanları bekliyor."""
    task: str
    sandbox: object
    model: object
    detector: LoopDetector
    budget: BudgetEnforcer
    trace: TraceWriter
    step: int = 0
    events: list = field(default_factory=list)
    history: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    screen_hashes: list = field(default_factory=list)
    tool_errors: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def last_screen_changed(self):
        if len(self.screen_hashes) < 2:
            return None
        return self.screen_hashes[-1] != self.screen_hashes[-2]

    def note(self, t: str) -> None:
        if t not in self.notes:
            self.notes.append(t)


class SayfaKabugu:
    """Stratejiler `ctx.sandbox.describe()` çağırıyor — tarayıcıyı o arayüze sar."""

    def __init__(self, browser):
        self.browser = browser

    def describe(self) -> str:
        try:
            d = self.browser.dom()
            return f"{d['baslik']} | {d['url']} | {d['metin'][:400]}"
        except Exception:
            return ""


TAVAN = 60          # demo güvenliği


class Ajan:
    def __init__(self, gorev: str, model, browser, terminal, strateji: str,
                 limits: BudgetLimits | None = None, rapor: Rapor | None = None,
                 gorsel: bool = True, panel: Panel | None = None,
                 golge: bool = True, desktop=None):
        self.gorev, self.model = gorev, model
        self.browser, self.terminal = browser, terminal
        # Masaustu araci OPSIYONEL. Verilmezse `desktop.*` cagrisi
        # "bilinmeyen arac" doner ve model tarayiciya yonelir.
        self.desktop = desktop
        self.strateji_ad = strateji
        self.stack = S.get(strateji)
        self.rapor = rapor or Rapor()
        self.gorsel = gorsel
        self.ctx = Baglam(task=gorev, sandbox=SayfaKabugu(browser), model=model,
                          detector=LoopDetector(), budget=BudgetEnforcer(limits),
                          trace=TraceWriter(None))
        self.panel = panel
        # Doğrulama GÖREVE ÖZEL. Zihniyetler bunu `ctx.extra` üzerinden
        # okuyor; koymazsak `cua_lab`'ın sentetik form kontrolüne düşüyorlar
        # ve her bitirme iddiasını reddediyorlar.
        self.ctx.extra["dogrulayici"] = _dogrula.yap(terminal, browser, gorev)
        self.ctx.extra["gerekli_araclar"] = _dogrula.gerekli_araclar(gorev)
        self.ctx.extra["min_kanit"] = _dogrula.min_kanit(gorev)
        # Gölge kurulu: BÜTÜN zihniyetler aynı olay akışını izliyor, yalnız
        # seçili olan müdahale ediyor. Tek koşum, 17 karşılaştırma.
        self.kurul = GolgeKurulu(strateji) if golge else None

    # -- istem -------------------------------------------------------------

    def _istem(self) -> tuple[str, bytes | None]:
        try:
            d = self.browser.dom()
            ogeler = "\n".join(
                f"  [{o['i']}] {o['tag']}{'/' + o['tur'] if o['tur'] else ''}"
                f"  \"{o['etiket']}\"" for o in d["ogeler"][:35])
            # Tarayici oturum boyunca acik kaliyor; acik sayfa ONCEKI bir
            # gorevden kalmis olabilir. Model bunu bilmezse ekrandaki metni
            # kendi bulgusu sanip tekrarliyor — olculdu, tam olarak bu oldu.
            bos = d["url"] in ("about:blank", "") and not d["ogeler"]
            uyari = ("\n(SAYFA BOS — goreve baslamak icin browser.goto cagir)"
                     if bos else
                     "\n(BU SAYFA ONCEKI BIR GOREVDEN KALMIS OLABILIR — "
                     "gorev baska bir adres istiyorsa once browser.goto cagir)")
            sayfa = (f"ACIK SAYFA{uyari}\n"
                     f"URL: {d['url']}\nBASLIK: {d['baslik']}\n"
                     f"ETKILESILEBILIR OGELER:\n{ogeler or '  (yok)'}\n"
                     f"SAYFA METNI:\n{d['metin'][:900]}")
        except Exception as e:
            sayfa = f"(sayfa okunamadi: {e})"

        gecmis = "\n".join(self.ctx.history[-8:]) or "(ilk adim)"
        uyari = ("\nUYARILAR:\n" + "\n".join(f"- {n}" for n in self.ctx.notes)
                 if self.ctx.notes else "")
        istem = (f"GOREV: {self.gorev}\n\n{sayfa}\n\n"
                 f"TERMINAL CALISMA DIZINI: {self.terminal.kok}\n\n"
                 f"GECMIS:\n{gecmis}{uyari}")
        # Masaustu aciksa GORUNTU ORADAN gelir — ajan tum ekrani gorur.
        if self.desktop is not None:
            istem = (f"GOREV: {self.gorev}\n\nGERCEK MASAUSTU\n"
                     f"{self.desktop.durum()}\n"
                     f"(koordinatlar SANA VERILEN GORUNTUNUN piksel uzayinda)\n\n"
                     f"{sayfa}\n\nTERMINAL CALISMA DIZINI: {self.terminal.kok}\n\n"
                     f"GECMIS:\n{gecmis}{uyari}")
        png = None
        if self.gorsel:
            try:
                # `desktop.pencere` cagrildiysa TAM EKRAN yerine o pencere gider —
                # gizlilik ve token acisindan cok daha iyi.
                if (kare := getattr(self, "_pencere_png", None)):
                    png, self._pencere_png = kare, None
                else:
                    png = (self.desktop.screenshot() if self.desktop is not None
                           else self.browser.screenshot())
            except Exception:
                pass
        return istem, png

    # -- araç yürütme ------------------------------------------------------

    def _arac_calistir(self, karar: dict) -> tuple[str, bool, bool]:
        """(çıktı, hata_mı, engellendi_mi)"""
        ad = karar.get("arac", "")
        try:
            if ad == "terminal":
                r = self.terminal.calistir(str(karar.get("komut", "")))
                # `or "(bos)"` YAZMA: o metin olaya `output` olarak giriyor ve
                # BOS bir sonucu DOLU gosteriyordu. `mkdir` sessizce basarili
                # olunca `tool_contract` ve dogrulayici ikisi de "kanit var"
                # sanip yanlis bir bitirme iddiasini geciriyordu — olculdu.
                # Bos bos kalir; "(bos)" yalniz EKRANDA gorunur (render.sonuc).
                return (r["cikti"] or r["hata"] or "",
                        not r["ok"], r["engellendi"])
            if ad == "browser.goto":
                return self.browser.goto(str(karar.get("url", ""))), False, False
            if ad == "browser.dom":
                d = self.browser.dom()
                return (f"{len(d['ogeler'])} oge · {d['baslik']} · {d['url']}",
                        False, False)
            if ad == "browser.click":
                return self.browser.click(int(karar.get("i", -1))), False, False
            if ad == "browser.type":
                return (self.browser.type(int(karar.get("i", -1)),
                                          str(karar.get("metin", ""))), False, False)
            if ad == "browser.key":
                return self.browser.key(str(karar.get("tus", "Enter"))), False, False
            if ad == "browser.scroll":
                return self.browser.scroll(int(karar.get("dy", 400))), False, False
            if ad == "browser.find":
                return self.browser.find(str(karar.get("metin", ""))), False, False
            if ad == "browser.read":
                return self.browser.read(int(karar.get("sayfa", 1))), False, False
            if ad == "browser.links":
                return self.browser.links(str(karar.get("filtre", ""))), False, False
            if ad == "browser.wait_for":
                r = self.browser.wait_for(str(karar.get("metin", "")),
                                          float(karar.get("saniye", 10)))
                return r, r.startswith("ZAMAN"), False
            if ad == "browser.back":
                return self.browser.back(), False, False
            if ad == "browser.scroll_to":
                return self.browser.scroll_to(int(karar.get("i", 0))), False, False
            if ad.startswith("terminal."):
                eylem = ad.split(".", 1)[1]
                fn = {"yaz": lambda: self.terminal.yaz(
                          str(karar.get("ad", "")), str(karar.get("icerik", "")),
                          bool(karar.get("ekle", False))),
                      "oku": lambda: self.terminal.oku(
                          str(karar.get("ad", "")), int(karar.get("bas", 1)),
                          int(karar.get("adet", 200))),
                      "listele": lambda: self.terminal.listele(str(karar.get("alt", "."))),
                      "ara": lambda: self.terminal.ara(str(karar.get("desen", "")),
                                                       str(karar.get("alt", ".")))}.get(eylem)
                if fn is None:
                    return f"bilinmeyen terminal eylemi: {eylem}", True, False
                r = fn()
                return (r["cikti"] or r["hata"] or ""), not r["ok"], r["engellendi"]
            if ad.startswith("desktop."):
                if self.desktop is None:
                    return ("masaustu araci KAPALI — tarayici araclarini kullan",
                            True, True)
                eylem = ad.split(".", 1)[1]
                if eylem == "screenshot":
                    return self.desktop.durum(), False, False
                if eylem == "pencereler":
                    return self.desktop.pencereler(), False, False
                if eylem == "pencere":
                    png, bilgi = self.desktop.pencere_goruntusu(str(karar.get("ad", "")))
                    self._pencere_png = png or None
                    return bilgi, not png, False
                if eylem == "odakla":
                    return self.desktop.odakla(str(karar.get("ad", ""))), False, False
                if eylem == "surukle":
                    return self.desktop.surukle(
                        karar.get("x1", 0), karar.get("y1", 0),
                        karar.get("x2", 0), karar.get("y2", 0)), False, False
                if eylem in ("click", "double_click", "move"):
                    f = getattr(self.desktop, eylem)
                    return f(karar.get("x", 0), karar.get("y", 0)), False, False
                if eylem == "type":
                    return self.desktop.type(str(karar.get("metin", ""))), False, False
                if eylem == "key":
                    return self.desktop.key(str(karar.get("tus", "Return"))), False, False
                if eylem == "scroll":
                    return self.desktop.scroll(int(karar.get("dy", 3))), False, False
                return f"bilinmeyen masaustu eylemi: {eylem}", True, False
            return f"bilinmeyen arac: {ad}", True, False
        except Exception as e:
            return f"{type(e).__name__}: {e}", True, False

    # -- karar uygulama ----------------------------------------------------

    def _golge(self, ad, *args_ve_verdict):
        """Kancayı gölgelere de gönder, sonra aktif kararı olduğu gibi döndür."""
        *args, v = args_ve_verdict
        if self.kurul:
            self.kurul.kanca(ad, *args, adim=self.ctx.step)
        return v

    def _panel_ciz(self) -> None:
        if not self.panel:
            return
        aktif = self.stack.items[0].snapshot() if self.stack.items else {}
        sinyal = [(a, b, c) for a, b, c in (aktif.get("sinyal") or [])]
        butce = [(e["ad"], e["kullanilan"], e["limit"])
                 for e in (aktif.get("eksenler") or [])]
        if not sinyal and not butce:
            s = self.ctx.budget.state
            l = self.ctx.budget.limits
            butce = [("adim", s.steps, l.max_steps or 0),
                     ("token", s.tokens, l.max_tokens or 0)]
        golge = []
        if self.kurul:
            durdu = [x for x in self.kurul.tablo() if x[1] in ("stop", "degrade")]
            golge = [f"{len(durdu)} zihniyet zaten dururdu"] if durdu else []
        self.panel.ciz(f"{self.strateji_ad}  ·  adım {self.ctx.step}"
                       f"  ·  {aktif.get('not', '')}", sinyal, golge, butce)

    def _uygula(self, v, span: Span):
        if not v.triggered:
            return None
        span.verdict, span.verdict_reason = v.action.value, v.reason
        tur = {Action.NUDGE: "nudge", Action.STOP: "stop",
               Action.DEGRADE: "degrade"}.get(v.action, "?")
        self.rapor.guardrail(tur, v.reason, str(v.detail.get("mesaj", "")))
        if v.action is Action.NUDGE:
            self.ctx.note(v.detail.get("mesaj") or f"{v.reason}: yaklasimi degistir")
            return None
        self.ctx.trace.write(span); span._written = True
        if v.action is Action.DEGRADE:
            st = (Status.BUDGET_EXHAUSTED if v.reason.startswith("budget")
                  or v.reason.startswith("max_") else Status.DEGRADED)
            return self._bitir(st, v.reason, "(bütçe bitti — kısmi sonuç)")
        if v.reason.startswith("abstain") or v.detail.get("abstain"):
            return self._bitir(Status.NEEDS_INPUT, v.reason)
        if (term := v.detail.get("terminal")):
            return self._bitir(Status(term), v.reason)
        return self._bitir(Status.STUCK, v.reason)

    def _bitir(self, st: Status, sebep: str, cevap: str | None = None) -> Sonuc:
        rp = self.stack.on_stop(sebep, self.ctx)
        s = Sonuc(status=st, reason=sebep, steps=self.ctx.step,
                  totals=self.ctx.trace.totals(), answer=cevap, report=rp,
                  nudges=list(self.ctx.notes))
        s.golge = self.kurul.tablo() if self.kurul else []
        return s

    # -- ana döngü ---------------------------------------------------------

    def kos(self) -> Sonuc:
        self.stack.on_run_start(self.ctx)
        if self.kurul:
            self.kurul.baslat(self.ctx)
        self.ctx.screen_hashes.append(self.browser.durum_hash())

        for _ in range(TAVAN):
            span = Span(i=self.ctx.step, t=self.ctx.trace.elapsed())

            # 1) bütçe — adımdan ÖNCE
            if (r := self._uygula(self._golge("before_step", self.ctx, self.stack.before_step(self.ctx)), span)):
                return r
            self.ctx.step += 1
            self.ctx.budget.charge_step()
            self._panel_ciz()

            # 2) model
            istem, png = self._istem()
            karar, tok, cost = self.model.dusun(istem, png)
            self.ctx.budget.charge_usage(tok, cost)
            span.tokens, span.cost_usd = tok, cost
            self.rapor.dusunce(self.ctx.step, str(karar.get("dusunce", "")))

            ad = karar.get("arac", "")

            # 3) bitirme iddiası — KANIT DEĞİL
            if ad == "finish":
                span.action = "finish"
                from cua_lab.events import Finish
                fin = Finish(str(karar.get("cevap", "")), tokens=tok, cost_usd=cost)
                # Bitirme iddiasi GOLGELERE de gitmeli — `verify-gate` ve
                # `telemetry-repair` yalnizca bu kancada konusuyor. Baglamayi
                # unutmak onlari "sessiz kalir" gostermisti.
                fv = self._golge("on_finish_claim", fin, self.ctx,
                                 self.stack.on_finish_claim(fin, self.ctx))
                if (r := self._uygula(fv, span)):
                    self.ctx.trace.write(span); return r
                if fv.action is Action.NUDGE:
                    self.ctx.history.append(f"finish REDDEDILDI: {fv.reason}")
                    self.ctx.events.append(Event(EventKind.ERROR, "finish",
                                                 {"error": fv.reason}))
                    self.ctx.trace.write(span); continue
                self.ctx.trace.write(span)
                return self._bitir(Status.OK, "finished", fin.answer)

            # 4) metin (araç yok) — monolog dedektörüne yem
            if ad in ("soyle", "hata"):
                span.action = "say"
                ev = Event(EventKind.MESSAGE, "agent",
                           {"text": str(karar.get("dusunce", ""))})
                self.ctx.events.append(ev); self.ctx.detector.record(ev)
                if (r := self._uygula(self._golge("on_observation", ev, self.ctx, self.stack.on_observation(ev, self.ctx)), span)):
                    self.ctx.trace.write(span); return r
                self.ctx.trace.write(span); continue

            # 5) araç
            args = {k: v for k, v in karar.items() if k not in ("arac", "dusunce")}
            self.rapor.arac(ad, args)
            span.action = ad
            aev = Event(EventKind.ACTION, ad, dict(args))
            self.ctx.events.append(aev); self.ctx.detector.record(aev)
            if (r := self._uygula(self._golge("on_action", aev, self.ctx, self.stack.on_action(aev, self.ctx)), span)):
                return r

            cikti, hata, engel = self._arac_calistir(karar)
            self.rapor.sonuc(cikti, hata=hata, engel=engel)
            if self.panel:
                self.panel.ekle_hareket(
                    f"{'✕' if engel else ('!' if hata else '·')} {ad} "
                    f"{str(args)[:28]}")
            if hata or engel:
                self.ctx.budget.charge_replan()

            h = self.browser.durum_hash()
            self.ctx.screen_hashes.append(h)
            span.screen_hash = h
            span.error = cikti if (hata or engel) else None
            self.ctx.history.append(f"{ad}({args}) -> "
                                    f"{'HATA: ' + cikti[:60] if hata or engel else cikti[:60]}")
            oev = (Event(EventKind.ERROR, ad, {"error": cikti})
                   if (hata or engel)
                   else Event(EventKind.OBSERVATION, ad, {"output": cikti},
                              meta={"screen_hash": h}))
            self.ctx.events.append(oev)
            self.ctx.detector.record(oev, progress=bool(self.ctx.last_screen_changed()))

            if (r := self._uygula(self._golge("on_observation", oev, self.ctx, self.stack.on_observation(oev, self.ctx)), span)):
                self.ctx.trace.write(span); return r
            self.ctx.trace.write(span)

        return self._bitir(Status.CEILING, "hard_ceiling")
