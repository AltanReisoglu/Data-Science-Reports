"""
ReAct döngüsü — strateji kancalı.

Referans döngü (Anthropic `computer-use-demo/loop.py`) şöyle:

    while True:
        response = model(...)
        if not tool_result_content: return
        ...

Tur sayacı yok, döngü tespiti yok, bütçe yok. Buradaki döngü aynı iskeleti
koruyor ama her kritik noktada stratejiye soruyor. `none` stratejisi seçilince
davranış referans döngüyle aynı oluyor — taban çizgisi bu.

Sıra önemli: bütçe ADIMDAN ÖNCE bakılıyor. Sonra bakmak, limiti aşan çağrının
parasını zaten ödedikten sonra fark etmek demek.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .events import (
    Act,
    Action,
    BudgetEnforcer,
    BudgetLimits,
    ComputerCall,
    Event,
    EventKind,
    Finish,
    LoopDetector,
    LoopThresholds,
    Say,
    ToolResult,
)
from .model import ModelClient, Request
from .strategies.base import StopReport, StrategyStack
from .trace import Span, TraceWriter


class Status(str, Enum):
    OK = "OK"
    STUCK = "STUCK"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    DEGRADED = "DEGRADED"
    NEEDS_INPUT = "NEEDS_INPUT"     # çekimser kalma — birinci sınıf sonuç
    CEILING = "CEILING"             # demo tavanı; gerçek üretimde bu yok

    @property
    def clean(self) -> bool:
        return self is Status.OK


@dataclass
class RunResult:
    status: Status
    reason: str
    steps: int
    totals: dict[str, Any]
    report: StopReport | None = None
    answer: str | None = None
    trace: TraceWriter | None = None
    nudges: list[str] = field(default_factory=list)


@dataclass
class RunContext:
    """Stratejilerin okuduğu koşum durumu.

    `detector` ve `budget` ORTAK altyapı — stratejiler bunları farklı
    biçimlerde kullanıyor. Zihniyet farkı burada değil, kancalarda.
    """

    task: str
    sandbox: Any
    model: ModelClient
    detector: LoopDetector
    budget: BudgetEnforcer
    trace: TraceWriter
    step: int = 0
    events: list[Event] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    screen_hashes: list[str] = field(default_factory=list)
    tool_errors: dict[str, list[bool]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def last_screen_changed(self) -> bool | None:
        if len(self.screen_hashes) < 2:
            return None
        return self.screen_hashes[-1] != self.screen_hashes[-2]

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)


HARD_CEILING = 300   # demo güvenliği; `none` stratejisi sonsuza gitmesin


def _h(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()[:12]


class Runner:
    def __init__(self, task: str, sandbox, model: ModelClient, strategy: StrategyStack,
                 limits: BudgetLimits | None = None,
                 thresholds: LoopThresholds | None = None,
                 trace_path=None, progress=None):
        self.task, self.sandbox, self.model, self.strategy = task, sandbox, model, strategy
        # İnteraktif kabuk için canlı durum kancası. None ise hiç çağrılmıyor —
        # betikle koşan testler ve `compare` bundan etkilenmiyor.
        self.progress = progress
        self.ctx = RunContext(
            task=task, sandbox=sandbox, model=model,
            detector=LoopDetector(thresholds),
            budget=BudgetEnforcer(limits),
            trace=TraceWriter(trace_path),
        )

    # -- yardımcılar -----------------------------------------------------

    def _finish(self, status: Status, reason: str, answer=None) -> RunResult:
        report = self.strategy.on_stop(reason, self.ctx)
        if report and report.answer and answer is None:
            answer = report.answer
        return RunResult(
            status=status, reason=reason, steps=self.ctx.step,
            totals=self.ctx.trace.totals(), report=report, answer=answer,
            trace=self.ctx.trace, nudges=list(self.ctx.notes),
        )

    def _apply(self, v, span: Span, source: str) -> RunResult | None:
        """Bir kararı uygular. Koşum bitiyorsa RunResult döner.

        Karar koşumu bitiriyorsa span BURADA yazılıyor — `_finish` totalleri
        iz üzerinden hesapladığı için son adım yazılmadan çağrılırsa o adımın
        token'i sayılmadan kalır.
        """
        if not v.triggered:
            return None
        span.verdict = v.action.value
        span.verdict_reason = v.reason
        span.verdict_by = str(v.detail.get("strateji", source))
        span.detail = dict(v.detail)

        if v.action is Action.NUDGE:
            self.ctx.note(v.detail.get("mesaj") or f"{v.reason}: yaklasimi degistir")
            return None

        self.ctx.trace.write(span)          # bitiren kararlar: once yaz, sonra topla
        span._written = True                # cagirana tekrar yazma sinyali

        if v.action is Action.DEGRADE:
            answer = self._force_finish()
            st = Status.BUDGET_EXHAUSTED if v.reason.startswith("budget") else Status.DEGRADED
            return self._finish(st, v.reason, answer)
        if v.reason.startswith("abstain") or v.detail.get("abstain"):
            return self._finish(Status.NEEDS_INPUT, v.reason)
        # Strateji kendi terminal durumunu bildirebilir. Arize'ın ilkesi:
        # "tamamlandı", "adım limiti", "bütçe aşıldı" ve "hata" FARKLI
        # sonuçlardır; hepsini tek bir başarı oranına karıştırmak neyin
        # yanlış gittiğini gizler. Sıkışmak da bütçe bitmesi de "hata" değil.
        if (term := v.detail.get("terminal")):
            return self._finish(Status(term), v.reason)
        return self._finish(Status.STUCK, v.reason)

    def _force_finish(self) -> str:
        """Zarif bozulma: modelden elindekiyle nihai cevabı iste.

        Bu çağrı bütçe dışıdır — tavanı aştıktan sonra cevabı almak kabul
        edilen maliyet. `agentbudget-dollar` stratejisi bunu baştan rezerve
        ederek daha dürüst hale getiriyor (finalization reserve).
        """
        req = Request(task=self.task, screen=self.sandbox.describe(),
                      history=self.ctx.history, notes=self.ctx.notes,
                      forced_finish=True)
        try:
            out = self.model.act(req)
        except Exception as e:
            return f"(nihai cevap alinamadi: {e})"
        return out.answer if isinstance(out, Finish) else "(model bitirmedi — kismi sonuc)"

    # -- ana koşum -------------------------------------------------------

    def run(self) -> RunResult:
        self.sandbox.start()
        self.ctx.screen_hashes.append(self.sandbox.screen_hash())
        self.strategy.on_run_start(self.ctx)

        try:
            for _ in range(HARD_CEILING):
                span = Span(i=self.ctx.step, t=self.ctx.trace.elapsed())

                # 1) Bütçe / tahsis — adımdan ÖNCE
                if (r := self._apply(self.strategy.before_step(self.ctx), span, "before_step")):
                    return r

                self.ctx.step += 1
                self.ctx.budget.charge_step()
                if self.progress:
                    self.progress("dusunuyor", self.ctx)

                # 2) Model adımı
                req = Request(task=self.task, screen=self.sandbox.describe(),
                              history=self.ctx.history, notes=self.ctx.notes)
                # Gerçek masaüstü ekran görüntüsü verebiliyorsa isteğe ekle.
                # Sentetik sandbox'ta `frame` yok; alan None kalıyor ve hiçbir
                # şey değişmiyor — betikli modeller görüntüyü zaten okumuyor.
                if (kare := getattr(self.sandbox, "frame", None)):
                    req.image, req.image_scale = kare()
                req = self.strategy.decorate_request(req, self.ctx)
                out = self.model.act(req)
                self.ctx.budget.charge_usage(getattr(out, "tokens", 0),
                                             getattr(out, "cost_usd", 0.0))
                span.tokens = getattr(out, "tokens", 0)
                span.cost_usd = getattr(out, "cost_usd", 0.0)

                # 3) Bitirme iddiası — kanıt değil, iddia
                if isinstance(out, Finish):
                    span.action = "finish"
                    fv = self.strategy.on_finish_claim(out, self.ctx)
                    if (r := self._apply(fv, span, "finish_claim")):
                        self.ctx.trace.write(span)
                        return r
                    if fv.action is Action.NUDGE:
                        # Doğrulama kapısı açılmadı. `sde_offer_loop`un kritik
                        # ayrıntısı: koşum BİTİRİLMİYOR — sonuç gözlem akışına
                        # geri veriliyor ve döngü sürüyor, ajan kendi hatasını
                        # görüp düzeltsin diye.
                        self.ctx.history.append(
                            f"finish REDDEDILDI: {fv.detail.get('kanit', fv.reason)}")
                        self.ctx.events.append(
                            Event(EventKind.ERROR, "finish", {"error": fv.reason}))
                        self.ctx.trace.write(span)
                        continue
                    self.ctx.trace.write(span)
                    # Zorla bitirilen kosum TEMIZ sayilmaz. `guardrails.py`
                    # tasarim karari #4: "tukenis gorunur olmali" — arac
                    # cagrisi kilitlendigi icin biten bir kosum OK donerse
                    # tukenis bir durum kontrolunden gorunmez olur.
                    if self.ctx.extra.get("forced_finish"):
                        return self._finish(Status.DEGRADED,
                                            self.ctx.extra.get("forced_reason",
                                                               "forced_finish"),
                                            out.answer)
                    return self._finish(Status.OK, "finished", out.answer)

                # 4) Metin
                if isinstance(out, Say):
                    span.action = "say"
                    ev = out.to_event()
                    self.ctx.events.append(ev)
                    self.ctx.detector.record(ev)
                    self.ctx.history.append(f"say: {out.text[:80]}")
                    if (r := self._apply(self.strategy.on_observation(ev, self.ctx),
                                         span, "observation")):
                        self.ctx.trace.write(span)
                        return r
                    self.ctx.trace.write(span)
                    continue

                # 5) Eylem
                call: ComputerCall = out
                span.action = call.act.value
                span.args_hash = _h(call.args)
                aev = call.to_event()
                self.ctx.events.append(aev)
                self.ctx.detector.record(aev)

                if (r := self._apply(self.strategy.on_action(aev, self.ctx), span, "action")):

                    return r

                res: ToolResult = self.sandbox.execute(call.act, call.args)
                if res.error:
                    self.ctx.budget.charge_replan()
                self.ctx.tool_errors.setdefault(call.act.value, []).append(res.error is not None)
                self.ctx.screen_hashes.append(res.screen_hash or self.sandbox.screen_hash())
                span.screen_hash = self.ctx.screen_hashes[-1]
                span.error = res.error
                span.executed = res.executed
                span.result_hash = _h({"out": res.output, "err": res.error})

                oev = res.to_event(call.act.value)
                self.ctx.events.append(oev)
                # İlerleme = ekran gerçekten değişti (ve eylem değiştirmeyi vaat ediyordu)
                progress = bool(call.act.mutates_screen and self.ctx.last_screen_changed())
                self.ctx.detector.record(oev, progress=progress)
                self.ctx.history.append(
                    f"{call.act.value}({call.args}) -> {'HATA: ' + res.error if res.error else 'ok'}"
                )

                if (r := self._apply(self.strategy.on_observation(oev, self.ctx),
                                     span, "observation")):
                    self.ctx.trace.write(span)
                    return r

                if self.progress:
                    self.progress(call.act.value, self.ctx)
                self.ctx.trace.write(span)

            return self._finish(Status.CEILING, "hard_ceiling")
        finally:
            self.sandbox.stop()
