"""
Model katmanı — soyut. Gerçek model sonra takılacak (HF Inference API).

Döngü modelin kim olduğunu bilmiyor; yalnızca `act()` çağırıyor. Bu ayrım
sayesinde Faz 1-4 internetsiz, `ScriptedModel` ile ilerleyebiliyor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

from .events import Act, ComputerCall, Finish, ModelOutput, Say


@dataclass
class Request:
    """Modele gidecek istek. Stratejiler `decorate_request` ile buna dokunuyor."""

    task: str
    screen: str                              # ekranın metin özeti (ya da görüntü referansı)
    history: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)   # nudge / geri sayım enjeksiyonu
    forced_finish: bool = False              # agentscope-grace: bitirmeye zorlama

    # Ekran görüntüsü (PNG). Yalnız VLM modeller okuyor; betikli modeller ve
    # sentetik sandbox için None kalıyor.
    image: bytes | None = None
    # Görüntünün modele giden boyutu ile gerçek ekran arasındaki oran. VLM
    # koordinatı küçültülmüş görüntü uzayında veriyor; gerçek ekrana çevirmek
    # için bu çarpan gerekiyor:  gercek_x = model_x * image_scale
    image_scale: float = 1.0

    def prompt(self) -> str:
        out = [f"GOREV: {self.task}", f"EKRAN: {self.screen}"]
        if self.history:
            out.append("GECMIS:\n" + "\n".join(self.history[-8:]))
        if self.notes:
            out.append("UYARILAR:\n" + "\n".join(f"- {n}" for n in self.notes))
        if self.forced_finish:
            out.append("ZORUNLU: baska arac cagirma, elindekiyle nihai cevabi ver.")
        return "\n\n".join(out)


@runtime_checkable
class ModelClient(Protocol):
    name: str

    def act(self, req: Request) -> ModelOutput: ...


class ScriptedModel:
    """Önceden yazılmış eylem dizisi. Testler ve deterministik demo için.

    Gerçek modelin yerini tutmuyor — amacı DÖNGÜNÜN ve STRATEJİLERİN
    doğruluğunu test etmek. Hata desenleri ortamdan (FakeSandbox) geliyor,
    buradan değil.
    """

    name = "scripted"

    def __init__(self, script: list[ModelOutput] | Callable[[Request], ModelOutput],
                 loop_forever: bool = False):
        self._fn = script if callable(script) else None
        self._script = None if callable(script) else list(script)
        self._i = 0
        self.loop_forever = loop_forever

    def act(self, req: Request) -> ModelOutput:
        if req.forced_finish:
            return Finish("zorunlu bitis: elimdeki kismi sonuc", tokens=40)
        if self._fn is not None:
            return self._fn(req)
        if self._i >= len(self._script):
            if self.loop_forever and self._script:
                self._i = 0
            else:
                return Finish("betik bitti", tokens=20)
        out = self._script[self._i]
        self._i += 1
        return out


class StubbornModel:
    """Aynı eylemi sonsuza kadar tekrarlayan model.

    Zayıf modellerin en sık davranışı: bir şey çalışmıyor, aynı şeyi tekrar
    deniyor. Dedektörlerin asıl hedefi bu.
    """

    name = "stubborn"

    def __init__(self, act: Act = Act.LEFT_CLICK, args: dict | None = None,
                 tokens: int = 350, cost_usd: float = 0.0035):
        self.call = ComputerCall(act, args or {"x": 200, "y": 200},
                                 tokens=tokens, cost_usd=cost_usd)
        self.tokens, self.cost_usd = tokens, cost_usd

    def act(self, req: Request) -> ModelOutput:
        if req.forced_finish:
            return Finish("zorunlu bitis", tokens=self.tokens)
        return self.call


class AlternatingModel:
    """A-B-A-B: iki eylem arasında gidip gelen model.

    Ardışık tekrar YOK — 'aynı çağrı iki kez' heuristigi bunu kaçırır.
    22 harness'in 20'sinin kaçırdığı desen.
    """

    name = "alternating"

    def __init__(self, a: ComputerCall, b: ComputerCall, tokens: int = 350):
        import dataclasses as _d
        self.a = _d.replace(a, tokens=tokens, cost_usd=tokens * 1e-5)
        self.b = _d.replace(b, tokens=tokens, cost_usd=tokens * 1e-5)
        self.tokens, self._n = tokens, 0

    def act(self, req: Request) -> ModelOutput:
        if req.forced_finish:
            return Finish("zorunlu bitis", tokens=self.tokens)
        self._n += 1
        return self.a if self._n % 2 else self.b
