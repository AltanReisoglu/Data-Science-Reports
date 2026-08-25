"""
Guardrail çekirdeği: döngü tespiti + çok eksenli bütçe zorlaması.

Tasarım kararları ve dayanakları (ayrıntı: ../sources_list/harness_kontrolleri.md):

1. İçerik tabanlı eşitlik. Olayları karşılaştırırken her turda zaten değişen
   alanları (tool_call_id, timestamp, request_id...) DIŞARIDA bırakıyoruz.
   OpenHands `_event_eq` bunu yapıyor; yapmayan bir dedektör sessizce hiçbir
   şey bulmaz — hata türlerinin en kötüsü.

2. Çevrim taraması k=1..K. İncelenen 22 harness'ın 20'si yalnızca "ardışık aynı
   çağrı" sayıyor ve A-B-A-B türü dönüşümlü döngüyü tamamen kaçırıyor. Sadece
   Gemini CLI ve OpenHands yakalıyor. Bu yüzden k=1 (ardışık tekrar) özel bir
   durum olarak değil, genel çevrim taramasının bir hâli olarak ele alınıyor.

3. Kademeli müdahale. Önce prompt'a uyarı (nudge), sonra durdurma. HAL'in
   21.730 koşumluk log analizinde koşum ortasında hatasını düzelten agent
   başarma olasılığını 1,5–4x artırıyor; tek eşikli sert kesme bu şansı yok ediyor.

4. Tükeniş görünür olmalı. Agno'nun kaynak kodundaki yorum tuzağı adlandırıyor:
   "tool_call_limit refuses further calls but the run still completes, so
   exhaustion is invisible to a status check." Bu yüzden koşum sonucu ayrı bir
   terminal durum taşıyor; bütçesi biten koşum asla OK dönmüyor.

5. Bütçe tek eksenli değil. Taranan hiçbir framework adım + token + süre +
   maliyeti aynı anda zorlamıyor. Burada hepsi aynı anda ve aynı yerde.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

# --------------------------------------------------------------------------
# Olaylar
# --------------------------------------------------------------------------

# Her turda değişen, karşılaştırmada anlamsız alanlar. Tasarım kararı #1.
VOLATILE_FIELDS = frozenset(
    {
        "tool_call_id",
        "action_id",
        "llm_response_id",
        "request_id",
        "timestamp",
        "ts",
        "trace_id",
        "span_id",
        "seed",
        "nonce",
    }
)


class EventKind(str, Enum):
    ACTION = "action"            # agent bir araç çağırdı
    OBSERVATION = "observation"  # araç sonuç döndü
    ERROR = "error"              # araç ya da model hata verdi
    MESSAGE = "message"          # agent kullanıcıya/kendine metin yazdı
    USER = "user"                # kullanıcı girdisi — tespit penceresini sıfırlar


@dataclass(frozen=True)
class Event:
    kind: EventKind
    name: str                                    # araç adı ya da mesaj kaynağı
    payload: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)  # karşılaştırmaya girmez

    def signature(self) -> str:
        """Karşılaştırma için kanonik imza. Oynak alanlar atılır, anahtarlar sıralanır.

        Anahtar sıralaması önemli: {"a":1,"b":2} ile {"b":2,"a":1} aynı çağrıdır,
        ham serileştirmede farklı görünür.
        """
        clean = _strip_volatile(self.payload)
        blob = json.dumps(
            {"kind": self.kind.value, "name": self.name, "payload": clean},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _strip_volatile(obj: Any) -> Any:
    """Oynak alanları özyinelemeli olarak çıkarır."""
    if isinstance(obj, dict):
        return {
            k: _strip_volatile(v)
            for k, v in obj.items()
            if k not in VOLATILE_FIELDS
        }
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


# --------------------------------------------------------------------------
# Kararlar
# --------------------------------------------------------------------------


class Action(str, Enum):
    CONTINUE = "continue"
    NUDGE = "nudge"      # prompt'a uyarı enjekte et, koşum sürsün
    STOP = "stop"        # sert durdurma
    DEGRADE = "degrade"  # modelden elindekiyle nihai cevabı iste (smolagents deseni)


@dataclass
class Verdict:
    action: Action
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def triggered(self) -> bool:
        return self.action is not Action.CONTINUE


CONTINUE = Verdict(Action.CONTINUE)


# --------------------------------------------------------------------------
# Döngü tespiti
# --------------------------------------------------------------------------


@dataclass
class LoopThresholds:
    """Eşikler. Varsayılanlar OpenHands'ten alındı (kaynak koddan okundu),
    `cycle_max_k` Gemini CLI'ın k=1..5 taramasından."""

    repeat_action_observation: int = 4   # aynı eylem + aynı gözlem, kaç kez
    repeat_action_error: int = 3         # aynı eylem + hata, kaç kez
    monologue: int = 3                   # kullanıcı girdisi olmadan ardışık agent mesajı
    # Çevrim kanıtı: kısa çevrim çok tekrar ister, uzun çevrim az.
    # "A, A" tesadüf olabilir; "read→edit→build→test→log→revert" iki kez
    # tekrarladıysa tesadüf değildir. Sabit r=3 kullanan tarama uzun çevrimi
    # pencereye sığdıramadığı için hiç göremiyor.
    cycle_repeats: int = 3               # k<=2 için gereken tekrar
    cycle_repeats_long: int = 2          # k>2 için gereken tekrar
    cycle_max_k: int = 12                # taranacak en uzun çevrim (6 eylemlik döngü = 12 olay)
    # İlerleme yokluğu EN GEVŞEK sinyal: uzun bir düşünme/kurulum evresi de
    # ilerlemesiz görünür. O yüzden eşiği en yüksek olan bu ve en son tetikleniyor.
    # Düşük tutulursa dönüşümlü döngüyü "ilerleme yok" diye raporlayıp asıl
    # teşhisi (çevrim) gizliyor — ölçüldü, bkz. README "eşik çakışması".
    no_progress: int = 8
    # Pencere, en uzun çevrimin en az iki tekrarını almalı: cycle_max_k * 2.
    scan_window: int = 40                # taranacak en fazla olay

    # Kademeli müdahale: sert eşiğin kaç adım öncesinde uyarı verilsin.
    nudge_lead: int = 1


class LoopDetector:
    """Beş senaryoyu birlikte tarar. LLM çağrısı yok — tamamen deterministik."""

    def __init__(self, thresholds: LoopThresholds | None = None) -> None:
        self.t = thresholds or LoopThresholds()
        self._events: list[Event] = []
        self._progress_marks: list[int] = []  # ilerleme görülen olay indeksleri
        self._nudged: set[str] = set()        # aynı sebeple iki kez uyarmamak için

    # -- kayıt ------------------------------------------------------------

    def record(self, event: Event, progress: bool = False) -> None:
        self._events.append(event)
        if progress:
            self._progress_marks.append(len(self._events) - 1)

    def _window(self) -> list[Event]:
        """Son kullanıcı girdisinden sonraki olaylar, en fazla scan_window kadar.

        Pencereyi kullanıcı mesajında kesmek önemli: kullanıcı araya girdiyse
        önceki tekrar bu turun döngüsü değildir.
        """
        evs = self._events
        for i in range(len(evs) - 1, -1, -1):
            if evs[i].kind is EventKind.USER:
                evs = evs[i + 1:]
                break
        return evs[-self.t.scan_window:]

    # -- ana giriş --------------------------------------------------------

    def check(self, stage: bool = True) -> Verdict:
        """`stage=False` kademelendirmeyi ATLAR.

        Kademelendirme bir ZİHNİYET tercihi, mekanizma değil: OpenHands
        bilerek kademelendirmiyor ("sıkıştıysan sıkışmışsındır"), pi ve
        OpenClaw kademelendiriyor. Ortak altyapı ikisini de ifade edebilmeli,
        yoksa strateji dosyasındaki `escalate=False` düğmesi yalan söyler.
        """
        window = self._window()
        if not window:
            return CONTINUE

        for probe in (
            self._repeating_action_observation,
            self._repeating_action_error,
            self._monologue,
            self._cycle,
            self._no_progress,
        ):
            verdict = probe(window)
            if verdict.triggered:
                return self._stage(verdict) if stage else verdict
        return CONTINUE

    def _stage(self, verdict: Verdict) -> Verdict:
        """Kademelendirme: aynı sebep ilk kez görülüyorsa uyar, ikincide durdur.

        Tasarım kararı #3. Bir sebep için yalnızca bir kez uyarılır; agent
        uyarıya rağmen aynı desende ısrar ediyorsa sert kesme gelir.
        """
        key = verdict.reason
        if key not in self._nudged:
            self._nudged.add(key)
            return Verdict(Action.NUDGE, verdict.reason, verdict.detail)
        return verdict

    # -- senaryolar -------------------------------------------------------

    def _pairs(self, window: list[Event]) -> list[tuple[Event, Event]]:
        """Ardışık (eylem, gözlem/hata) çiftleri."""
        out: list[tuple[Event, Event]] = []
        for a, b in zip(window, window[1:]):
            if a.kind is EventKind.ACTION and b.kind in (
                EventKind.OBSERVATION,
                EventKind.ERROR,
            ):
                out.append((a, b))
        return out

    def _repeating_action_observation(self, window: list[Event]) -> Verdict:
        """Senaryo 1: aynı eylem, aynı gözlem — n kez üst üste."""
        pairs = [p for p in self._pairs(window) if p[1].kind is EventKind.OBSERVATION]
        n = self.t.repeat_action_observation
        if len(pairs) < n:
            return CONTINUE
        tail = pairs[-n:]
        sig = {(a.signature(), o.signature()) for a, o in tail}
        if len(sig) == 1:
            return Verdict(
                Action.STOP,
                "repeat_action_observation",
                {"tekrar": n, "arac": tail[-1][0].name},
            )
        return CONTINUE

    def _repeating_action_error(self, window: list[Event]) -> Verdict:
        """Senaryo 2: aynı eylem sürekli hata veriyor.

        Gözlem imzası karşılaştırılmıyor — hata metni her denemede biraz farklı
        olabilir (satır numarası, süre). Eylem aynıysa ve sonuç hep hataysa döngüdür.
        """
        pairs = [p for p in self._pairs(window) if p[1].kind is EventKind.ERROR]
        n = self.t.repeat_action_error
        if len(pairs) < n:
            return CONTINUE
        tail = pairs[-n:]
        if len({a.signature() for a, _ in tail}) == 1:
            return Verdict(
                Action.STOP,
                "repeat_action_error",
                {"tekrar": n, "arac": tail[-1][0].name},
            )
        return CONTINUE

    def _monologue(self, window: list[Event]) -> Verdict:
        """Senaryo 3: kullanıcı girdisi olmadan ardışık agent mesajları."""
        n = self.t.monologue
        streak = 0
        for ev in reversed(window):
            if ev.kind is EventKind.MESSAGE:
                streak += 1
            elif ev.kind in (EventKind.ACTION, EventKind.OBSERVATION, EventKind.ERROR):
                break
        if streak >= n:
            return Verdict(Action.STOP, "monologue", {"ardisik_mesaj": streak})
        return CONTINUE

    def _cycle(self, window: list[Event]) -> Verdict:
        """Senaryo 4: k uzunluğunda çevrim, r kez tekrarlıyor.

        k=1 ardışık aynı çağrı; k=2 A-B-A-B; k=3 A-B-C-A-B-C...
        İncelenen harness'ların çoğu yalnızca k=1'e bakıyor ve gerisini kaçırıyor.
        Karmaşıklık O(K · r · k) — pencere sabit olduğu için pratikte sabit.
        """
        sigs = [e.signature() for e in window if e.kind is not EventKind.USER]
        for k in range(1, self.t.cycle_max_k + 1):
            r = self.t.cycle_repeats if k <= 2 else self.t.cycle_repeats_long
            need = k * r
            if len(sigs) < need:
                break
            tail = sigs[-need:]
            head = tail[:k]
            if all(tail[i * k:(i + 1) * k] == head for i in range(1, r)):
                # k=1 zaten senaryo 1/2'de yakalanıyor olabilir; yine de raporla.
                return Verdict(
                    Action.STOP,
                    f"cycle_k{k}",
                    {"cevrim_uzunlugu": k, "tekrar": r},
                )
        return CONTINUE

    def _no_progress(self, window: list[Event]) -> Verdict:
        """Senaryo 5: eylemler farklı ama hiçbir ilerleme sinyali yok.

        İmza karşılaştırmasının kaçırdığı durum: agent her turda farklı bir şey
        deniyor ama hiçbiri işe yaramıyor. deer-flow'un `max_no_progress_continuations`
        deseni.
        """
        n = self.t.no_progress
        acted = sum(1 for e in window if e.kind is EventKind.ACTION)
        if acted < n:
            return CONTINUE
        last_progress = self._progress_marks[-1] if self._progress_marks else -1
        since = sum(
            1
            for i, e in enumerate(self._events)
            if i > last_progress and e.kind is EventKind.ACTION
        )
        if since >= n:
            return Verdict(
                Action.STOP,
                "no_progress",
                {"ilerlemesiz_eylem": since},
            )
        return CONTINUE


# --------------------------------------------------------------------------
# Bütçe
# --------------------------------------------------------------------------


@dataclass
class BudgetLimits:
    """Beş eksen. None = o eksen kapalı.

    Varsayılanların hepsi AÇIK — taranan framework'lerin çoğunun aksine.
    Kapalı bir varsayılan, olmayan bir limitle aynı şeydir.
    """

    max_steps: int | None = 12
    # 5, keyfi değil: 3'te bırakılırsa retry döngüsünü bütçe, döngü dedektöründen
    # ÖNCE yakalıyor ve teşhis kayboluyor — "çok fazla replan" diyorsun ama
    # "aynı çağrı aynı hatayla dönüyor" bilgisini alamıyorsun. Bkz. README.
    max_replans: int | None = 5
    max_tokens: int | None = 20_000
    max_seconds: float | None = 30.0
    max_cost_usd: float | None = 0.50

    # Sert eşiğin yüzde kaçında uyarı verilsin (Codex'in eşik listesi deseni).
    warn_at: float = 0.75


@dataclass
class BudgetState:
    steps: int = 0
    replans: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def seconds(self) -> float:
        return time.monotonic() - self.started_at


class BudgetEnforcer:
    """Beş ekseni birlikte zorlar. Aşan ilk eksen koşumu bitirir."""

    def __init__(self, limits: BudgetLimits | None = None) -> None:
        self.limits = limits or BudgetLimits()
        self.state = BudgetState()
        self._warned: set[str] = set()

    # -- sayaçlar ---------------------------------------------------------

    def charge_step(self) -> None:
        self.state.steps += 1

    def charge_replan(self) -> None:
        self.state.replans += 1

    def charge_usage(self, tokens: int, cost_usd: float = 0.0) -> None:
        self.state.tokens += tokens
        self.state.cost_usd += cost_usd

    # -- kontrol ----------------------------------------------------------

    def _axes(self) -> Iterable[tuple[str, float, float | None]]:
        s, l = self.state, self.limits
        yield "steps", s.steps, l.max_steps
        yield "replans", s.replans, l.max_replans
        yield "tokens", s.tokens, l.max_tokens
        yield "seconds", s.seconds, l.max_seconds
        yield "cost_usd", s.cost_usd, l.max_cost_usd

    def check(self) -> Verdict:
        for name, used, limit in self._axes():
            if limit is None:
                continue
            if used >= limit:
                return Verdict(
                    Action.DEGRADE,  # sert kesme değil: elindekiyle cevap ver
                    f"budget_{name}",
                    {"eksen": name, "kullanilan": _r(used), "limit": _r(limit)},
                )
            if used >= limit * self.limits.warn_at and name not in self._warned:
                self._warned.add(name)
                return Verdict(
                    Action.NUDGE,
                    f"budget_warn_{name}",
                    {"eksen": name, "kullanilan": _r(used), "limit": _r(limit)},
                )
        return CONTINUE

    def snapshot(self) -> dict[str, Any]:
        return {
            "steps": self.state.steps,
            "replans": self.state.replans,
            "tokens": self.state.tokens,
            "cost_usd": round(self.state.cost_usd, 4),
            "seconds": round(self.state.seconds, 2),
        }


def _r(v: float) -> float | int:
    return int(v) if float(v).is_integer() else round(float(v), 3)
