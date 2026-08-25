"""
Computer-use olay modeli.

Dedektörlerin ortak altyapısı `detect/guardrails.py`'da; bu dosya onun üstüne
computer-use'a özgü katmanı koyuyor: eylem uzayı ve araç sonucu.

Eylem uzayı Anthropic'in `computer_20251124` aracından alındı (bkz.
docs/computer_use_zihniyet.md §1). On yedi eylem, üç sürümde birikimli.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .detect.guardrails import (  # noqa: F401  — tek içe aktarma noktası
    CONTINUE,
    Action,
    BudgetEnforcer,
    BudgetLimits,
    Event,
    EventKind,
    LoopDetector,
    LoopThresholds,
    Verdict,
)


class Act(str, Enum):
    """`computer_20251124` eylem uzayı."""

    KEY = "key"
    TYPE = "type"
    MOUSE_MOVE = "mouse_move"
    LEFT_CLICK = "left_click"
    LEFT_CLICK_DRAG = "left_click_drag"
    RIGHT_CLICK = "right_click"
    MIDDLE_CLICK = "middle_click"
    DOUBLE_CLICK = "double_click"
    TRIPLE_CLICK = "triple_click"
    SCREENSHOT = "screenshot"
    CURSOR_POSITION = "cursor_position"
    LEFT_MOUSE_DOWN = "left_mouse_down"
    LEFT_MOUSE_UP = "left_mouse_up"
    SCROLL = "scroll"
    HOLD_KEY = "hold_key"
    WAIT = "wait"
    ZOOM = "zoom"

    @property
    def mutates_screen(self) -> bool:
        """Ekranı değiştirmesi BEKLENEN eylem mi?

        `wait`, `screenshot` ve `cursor_position` ekranı değiştirmez ve
        değiştirmemeleri normaldir — durgunluk sayacına girmemeliler.
        Referans döngüde `wait`in ayrı bir eylem olması tam da bu yüzden
        tuzak: modelin meşru beklemesi döngü sanılmamalı.
        """
        return self not in (Act.WAIT, Act.SCREENSHOT, Act.CURSOR_POSITION)


@dataclass(frozen=True)
class ComputerCall:
    """Modelin istediği tek bir eylem.

    `tokens`/`cost_usd` bu çağrıyı ÜRETMEK için harcanan modeldir — computer
    use'da baskın kalem ekran görüntüsü token'ları olduğu için her adımda
    yeniden ölçülüyor.
    """

    act: Act
    args: dict[str, Any] = field(default_factory=dict)
    tokens: int = 0
    cost_usd: float = 0.0

    def to_event(self) -> Event:
        return Event(EventKind.ACTION, self.act.value, dict(self.args))


@dataclass
class ToolResult:
    """Bir eylemin sonucu.

    `executed=False`, referans döngüdeki "Not executed: an earlier computer
    action in this turn failed." davranışının karşılığı — aynı turda önceki bir
    eylem patladığı için hiç çalıştırılmamış çağrı. Bu, başarısızlıktan farklı
    bir durum ve dedektörlerde tekrar sayılmamalı.
    """

    output: str = ""
    error: str | None = None
    screenshot: bytes | None = None
    screen_hash: str | None = None
    executed: bool = True
    tokens: int = 0
    cost_usd: float = 0.0

    @property
    def ok(self) -> bool:
        return self.executed and self.error is None

    def to_event(self, name: str) -> Event:
        if not self.executed:
            return Event(EventKind.OBSERVATION, name, {"skipped": True},
                         meta={"executed": False})
        if self.error is not None:
            return Event(EventKind.ERROR, name, {"error": self.error})
        return Event(EventKind.OBSERVATION, name, {"output": self.output},
                     meta={"screen_hash": self.screen_hash})


@dataclass
class Finish:
    """Model görevi bitirdiğini iddia ediyor. İDDİA — kanıt değil.

    `verify-gate` stratejisi bunu doğrulamadan kabul etmiyor
    (sde_offer_loop: "modelin 'bitti' demesi durma isteğidir").
    """

    answer: str
    tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class Say:
    """Model araç çağırmadan metin üretti."""

    text: str
    tokens: int = 0
    cost_usd: float = 0.0

    def to_event(self) -> Event:
        return Event(EventKind.MESSAGE, "agent", {"text": self.text})


ModelOutput = ComputerCall | Say | Finish
