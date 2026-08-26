"""
Sentetik sandbox — sıfır bağımlılık, tamamen deterministik.

Neden piksel değil: Faz 1-4'ün internetsiz ilerlemesi gerekiyor ve Pillow bile
indirme demek. Daha önemlisi, bir dedektörü test etmek için ekranın *anlamı*
lazım, görüntüsü değil. Burada ekran bir widget durumudur; hash o durumdan
türer. Gerçek sandbox'ta aynı arayüz piksel hash'i döndürecek.

Senaryolar kasten hatalı ORTAM üretiyor — silinen PoC'de döngüler betiklenmişti,
burada model gerçekten sıkışıyor çünkü ortam bozuk.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..events import Act, ToolResult

SCENARIOS = ("healthy", "dead_button", "flaky", "silent_success", "broken_tool")


@dataclass
class Widget:
    wid: str
    label: str
    x: int
    y: int
    kind: str = "button"        # button | field
    value: str = ""
    responsive: bool = True     # False = tıklanıyor ama hiçbir şey olmuyor
    works_after: int = 0        # ilk N denemede hata verir
    clicks: int = 0


class FakeSandbox:
    """Tek pencereli sentetik masaüstü."""

    name = "fake"

    def __init__(self, scenario: str = "healthy", width: int = 1280, height: int = 800):
        if scenario not in SCENARIOS:
            raise ValueError(f"bilinmeyen senaryo: {scenario} — seçenekler: {SCENARIOS}")
        self.scenario = scenario
        self.width, self.height = width, height
        self.widgets: dict[str, Widget] = {}
        self.status = ""
        self.done = False
        self.focus: str | None = None

    # -- yaşam döngüsü ---------------------------------------------------

    def start(self) -> None:
        self.widgets = {
            "name": Widget("name", "Ad", 200, 120, kind="field"),
            "submit": Widget("submit", "Gonder", 200, 200),
            "cancel": Widget("cancel", "Vazgec", 320, 200),
        }
        self.status, self.done, self.focus = "hazir", False, None

        if self.scenario == "dead_button":
            # Tıklanıyor, hiçbir şey olmuyor, hata da vermiyor.
            # İmza dedektörü bunu yakalar (aynı eylem + aynı gözlem).
            self.widgets["submit"].responsive = False
        elif self.scenario == "broken_tool":
            # Arac KALICI bozuk: her cagri hata veriyor. Galileo'nun ikinci
            # vakasi — "bugun gecici olan hata yarin kalici olursa her istek
            # butun deneme hakkini yakar." flaky ile farki: orada oran duser,
            # burada %100'de kalir.
            self.widgets["submit"].works_after = 10_000
        elif self.scenario in ("flaky", "silent_success"):
            # İlk iki denemede patlıyor, üçüncüde çalışıyor.
            # silent_success: koşum OK biter ama iki çağrı boşa gitmiştir —
            # Galileo'nun "hakkında ticket açılmayan hata" vakası.
            self.widgets["submit"].works_after = 2

    def stop(self) -> None:
        self.widgets.clear()

    # -- eylem yürütme ---------------------------------------------------

    def execute(self, act: Act, args: dict) -> ToolResult:
        if act in (Act.SCREENSHOT, Act.CURSOR_POSITION, Act.WAIT):
            return self._result()

        if act is Act.TYPE:
            w = self.widgets.get(self.focus or "")
            if w is None or w.kind != "field":
                return self._result(error="odaklanmis bir alan yok")
            w.value += str(args.get("text", ""))
            self.status = f"{w.label} guncellendi"
            return self._result()

        if act in (Act.LEFT_CLICK, Act.DOUBLE_CLICK, Act.TRIPLE_CLICK):
            hit = self._hit(args.get("x"), args.get("y"))
            if hit is None:
                self.status = "bosluga tiklandi"
                return self._result()
            return self._click(hit)

        return self._result()

    def _click(self, w: Widget) -> ToolResult:
        w.clicks += 1

        if w.kind == "field":
            self.focus = w.wid
            self.status = f"{w.label} odaklandi"
            return self._result()

        if not w.responsive:
            return self._result()   # sessizce hiçbir şey

        if w.clicks <= w.works_after:
            return self._result(error=f"{w.label}: gecici hata (deneme {w.clicks})")

        if w.wid == "submit":
            if not self.widgets["name"].value:
                self.status = "ad bos olamaz"
                return self._result(error="dogrulama: ad bos")
            self.done, self.status = True, "gonderildi"
        elif w.wid == "cancel":
            self.status = "iptal edildi"
        return self._result()

    # -- durum -----------------------------------------------------------

    def _hit(self, x, y) -> Widget | None:
        if x is None or y is None:
            return None
        for w in self.widgets.values():
            if abs(w.x - int(x)) <= 60 and abs(w.y - int(y)) <= 20:
                return w
        return None

    def _result(self, error: str | None = None) -> ToolResult:
        return ToolResult(
            output=self.describe(),
            error=error,
            screen_hash=self.screen_hash(),
        )

    def screen_hash(self) -> str:
        blob = "|".join(
            f"{w.wid}:{w.value}:{w.responsive}"
            for w in sorted(self.widgets.values(), key=lambda x: x.wid)
        ) + f"|status={self.status}|done={self.done}|focus={self.focus}"
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def describe(self) -> str:
        parts = []
        for w in sorted(self.widgets.values(), key=lambda x: x.wid):
            v = f'="{w.value}"' if w.kind == "field" else ""
            parts.append(f"[{w.kind} {w.label}{v} @({w.x},{w.y})]")
        return " ".join(parts) + f"  durum: {self.status or '-'}"
