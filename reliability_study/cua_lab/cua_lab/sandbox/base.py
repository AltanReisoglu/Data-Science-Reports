"""Sandbox arayüzü — ajanın içinde çalıştığı ortam.

İki uygulama: `fake` (sentetik, sıfır bağımlılık, deterministik) ve
`docker_x11` (gerçek masaüstü, Faz 5). Döngü hangisiyle konuştuğunu bilmiyor.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..events import Act, ToolResult


@runtime_checkable
class SandboxBackend(Protocol):
    """Ekran + girdi. Eylem uzayı `events.Act` ile aynı."""

    name: str
    width: int
    height: int

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def execute(self, act: Act, args: dict) -> ToolResult:
        """Tek bir eylemi yürüt ve sonucu döndür."""
        ...

    def screen_hash(self) -> str:
        """Ekranın mevcut durumunun özeti.

        Durgunluk tespiti buna bakıyor. Gerçek uygulamada TOLERANSLI olmalı —
        yanıp sönen imleç ya da saat hash'i değiştirmemeli, yoksa dedektör
        sürekli 'ilerleme var' sanır.
        """
        ...

    def describe(self) -> str:
        """Ekranın metin özeti — vision'sız modeller ve loglar için."""
        ...
