"""
Strateji protokolü ve kayıt defteri.

Tasarım şartı (kullanıcının ifadesiyle): *"frameworkler ortak olabilir, sadece
mentalitesi farklı olanlar farklı strateji olarak değerlendirilsin."*

Bu yüzden `detect/guardrails.py` ortak altyapı olarak duruyor — imza
karşılaştırma, çevrim taraması, bütçe sayaçları. Stratejiler onu FARKLI
biçimlerde kullanıyor; ayrım mekanizmada değil zihniyette.

Sekiz kanca var, hepsinin varsayılanı boş. Bir strateji yalnızca kendi
zihniyetinin gerektirdiği kancaları dolduruyor — böylece her dosya o kaynağın
ne düşündüğünü okunur biçimde gösteriyor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ..events import CONTINUE, Action, Event, Finish, Verdict
from ..model import Request


@dataclass
class StopReport:
    """Onurlu durma raporu.

    `loop_budget_source`: durma cevabı dört şey içermeli — ne denendi, ne
    bulundu, neden durdu, sıradaki adım. "Bu, sonsuz döngüyü faydalı kısmi
    sonuca dönüştürür."
    """

    reason: str
    tried: str = ""
    found: str = ""
    why: str = ""
    next_step: str = ""
    answer: str | None = None

    def render(self) -> str:
        rows = [("neden durdu", self.why or self.reason), ("ne denendi", self.tried),
                ("ne bulundu", self.found), ("sonraki adim", self.next_step)]
        return "\n".join(f"  {k:<14}: {v}" for k, v in rows if v)


class BaseStrategy:
    """Tüm kancalar boş. Stratejiler yalnız kendi zihniyetini override ediyor."""

    id: str = "base"
    title: str = ""
    source: str = ""
    mentality: str = ""
    family: str = ""          # "src" (ben_ekledim) | "harness" | "-"
    kind: str = "-"           # kategori: budget|window|evidence|shape|decision

    # -- secim icin acıklama alanlari ---------------------------------------
    # `mentality` tek cumlede NE OLDUGUNU soyluyor. Bir strateji secerken
    # gereken iki soru daha var ve ikisi de bir satirlik ozette kayboluyor:
    #   why     bu katman NEDEN gerekli — hangi hatayi onluyor
    #   action  tetiklendiginde NE YAPIYOR — durdurma mi, uyari mi, geri sarma mi
    # `priority` uygulama sirasi: 1 = once bunu koy. Sifir = sirasi yok.
    why: str = ""
    action: str = ""
    priority: int = 0
    blind_spot: str = ""      # neyi KACIRDIGI — dedektorun ikinci sinavi

    # VARYANTLAR — ayni zihniyetin ayni mekanizmayi farkli ayarla kullanan
    # halleri. Iki kaynak ayni fikri savunup yalnizca BIR SAYIDA ayrisiyorsa
    # onlari iki ayri "zihniyet" diye sunmak yaniltici olur: PoC'de ayni
    # strateji, farkli varyant.  `--strategy budget-grace:hermes`
    variants: dict[str, dict] = {}
    variant: str = ""

    def configure(self, cfg: dict[str, Any]) -> None:
        """Varyant secimi + serbest ayar. Bilinmeyen varyant sessizce
        yutulmuyor — yanlis yazim, yanlis olcum demek."""
        v = cfg.get("variant")
        if v:
            if v not in self.variants:
                raise UnknownVariant(self.id, v, sorted(self.variants))
            self.variant = v
            for k, val in self.variants[v].items():
                setattr(self, k, val)
        for k, val in cfg.items():
            if k != "variant" and hasattr(self, k):
                setattr(self, k, val)

    def on_run_start(self, ctx) -> None:
        pass

    def before_step(self, ctx) -> Verdict:
        return CONTINUE

    def decorate_request(self, req: Request, ctx) -> Request:
        return req

    def on_action(self, ev: Event, ctx) -> Verdict:
        return CONTINUE

    def on_observation(self, ev: Event, ctx) -> Verdict:
        return CONTINUE

    def on_finish_claim(self, fin: Finish, ctx) -> Verdict:
        return CONTINUE

    def on_stop(self, reason: str, ctx) -> StopReport | None:
        return None

    def snapshot(self) -> dict[str, Any]:
        return {}


class StrategyStack:
    """Birden çok strateji birlikte. İlk tetiklenen kazanır.

    Kaynakların ortak sonucu: tek katman yetmiyor. `--strategy a,b` ile
    birleştirilebilir olması bu yüzden.
    """

    def __init__(self, strategies: Iterable[BaseStrategy]):
        self.items: list[BaseStrategy] = list(strategies)

    @property
    def id(self) -> str:
        return ",".join(
            f"{s.id}:{s.variant}" if s.variant else s.id for s in self.items
        ) or "none"

    def _first(self, hook: str, *args) -> Verdict:
        for s in self.items:
            v = getattr(s, hook)(*args)
            if v.triggered:
                v.detail.setdefault("strateji", s.id)
                return v
        return CONTINUE

    def on_run_start(self, ctx):
        for s in self.items:
            s.on_run_start(ctx)

    def before_step(self, ctx) -> Verdict:
        return self._first("before_step", ctx)

    def decorate_request(self, req: Request, ctx) -> Request:
        for s in self.items:
            req = s.decorate_request(req, ctx)
        return req

    def on_action(self, ev, ctx) -> Verdict:
        return self._first("on_action", ev, ctx)

    def on_observation(self, ev, ctx) -> Verdict:
        return self._first("on_observation", ev, ctx)

    def on_finish_claim(self, fin, ctx) -> Verdict:
        return self._first("on_finish_claim", fin, ctx)

    def on_stop(self, reason: str, ctx) -> StopReport | None:
        for s in self.items:
            r = s.on_stop(reason, ctx)
            if r is not None:
                return r
        return None

    def snapshot(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for s in self.items:
            snap = s.snapshot()
            if snap:
                out[s.id] = snap
        return out


# -- kayıt defteri --------------------------------------------------------

class UnknownVariant(KeyError):
    """Bilinmeyen varyant adi."""

    def __init__(self, sid: str, varyant: str, mevcut: list[str]):
        self.id, self.varyant, self.mevcut = sid, varyant, mevcut
        super().__init__(
            f"'{sid}' icin bilinmeyen varyant: {varyant}\n"
            f"mevcut: {', '.join(mevcut) or '(yok)'}")


class UnknownStrategy(KeyError):
    """Bilinmeyen strateji id'si. `id` alanini tasiyor ki cagiran taraf
    mesaji string ayristirmadan kurabilsin."""

    def __init__(self, sid: str, mevcut: list[str]):
        self.id, self.mevcut = sid, mevcut
        super().__init__(f"bilinmeyen strateji: {sid}\nmevcut: {', '.join(mevcut)}")


_REGISTRY: dict[str, type[BaseStrategy]] = {}


def register(cls: type[BaseStrategy]) -> type[BaseStrategy]:
    if cls.id in _REGISTRY:
        raise ValueError(f"strateji id cakismasi: {cls.id}")
    _REGISTRY[cls.id] = cls
    return cls


def get(spec: str, cfg: dict | None = None) -> StrategyStack:
    """`"pi-signature,verify-gate"` → StrategyStack.

    Varyant `id:varyant` ile: `"budget-grace:hermes"`.
    """
    ids = [s.strip() for s in spec.split(",") if s.strip()]
    out = []
    for i in ids:
        varyant = ""
        if ":" in i:
            i, varyant = i.split(":", 1)
            i, varyant = i.strip(), varyant.strip()
        if i not in _REGISTRY:
            raise UnknownStrategy(i, sorted(_REGISTRY))
        s = _REGISTRY[i]()
        ayar = dict(cfg or {})
        if varyant:
            ayar["variant"] = varyant
        elif s.variants:
            # Varyantli bir strateji varyantsiz secildiyse ILKI varsayilan.
            ayar["variant"] = next(iter(s.variants))
        if ayar:
            s.configure(ayar)
        out.append(s)
    return StrategyStack(out)


def all_ids() -> list[str]:
    return sorted(_REGISTRY)


def catalog() -> list[type[BaseStrategy]]:
    return sorted(_REGISTRY.values(), key=lambda c: (c.family, c.id))
