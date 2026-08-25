"""
Canlı bakış göstergesi — ajan o an NEREYE bakıyor.

Gerçek masaüstünde en tedirgin edici şey, ajanın ne yapmak üzere olduğunu
görememek. Bu modül üç şeyi aynı anda gösteriyor:

  1. TERMINALDE   hedef koordinat, aktif pencere, imlecin gercek yeri,
                  engellenen eylemler — her adimda tek satir
  2. EKRANDA      tiklamadan ONCE imlec hedefe tasiniyor ve `dwell_seconds`
                  kadar bekliyor; nereye tiklanacagini fiziksel olarak
                  goruyorsun (backend'de, `_gonder` icinde)
  3. IZ DOSYASINDA  her adimin hedefi ve ekran hash'i JSONL'e yaziliyor

Uc kademeli renk: mavi = okuyor · sari = girdi gonderecek · kirmizi = engellendi.
"""

from __future__ import annotations

import time

from . import ui

FAZ_RENK = {
    "baslat": ui.GREEN,
    "bakiyor": ui.CYAN,
    "engellendi": ui.RED,
    "bitti": ui.GREY,
}


class Watcher:
    """`X11Sandbox(gozlemci=...)` olarak takiliyor.

    `hud` verilirse aynı bilgiyi masaüstünün sağındaki panele de basıyor.
    Terminal ile panel aynı kaynaktan besleniyor — ikisi asla ayrışmıyor.
    """

    def __init__(self, sessiz: bool = False, hud=None):
        self.sessiz = sessiz
        self.hud = hud
        self.gecmis: list[tuple[str, str]] = []
        self._t0 = time.monotonic()

    def __call__(self, faz: str, metin: str, sandbox) -> None:
        self.gecmis.append((faz, metin))
        if self.hud is not None:
            self._hud_besle(faz, metin, sandbox)
        if self.sessiz:
            return
        renk = FAZ_RENK.get(faz, "")
        gecen = time.monotonic() - self._t0

        if faz == "baslat":
            print(f"\n  {ui.GREEN}● GERCEK MASAUSTU{ui.RESET}  {ui.DIM}{metin}{ui.RESET}")
            print(f"  {ui.GREY}kacis: fareyi SOL UST KOSEYE tasi — kosum aninda iptal{ui.RESET}\n")
            return
        if faz == "bitti":
            ui.clear_line()
            print(f"\n  {ui.GREY}● masaustu birakildi{ui.RESET}  {metin}")
            return
        if faz == "engellendi":
            ui.clear_line()
            print(f"  {ui.RED}✕ ENGELLENDI{ui.RESET}  {ui.DIM}{metin}{ui.RESET}")
            return

        # "bakiyor" — asil canli satir
        aktif = sandbox.aktif_pencere()[:34]
        cx, cy = sandbox._imlec()
        hedef = sandbox._son_hedef
        ok = f"{ui.B}{hedef[0]},{hedef[1]}{ui.RESET}" if hedef else f"{ui.DIM}—{ui.RESET}"
        ui.status_line(
            f"{renk}◎{ui.RESET} {ui.B}{metin:<26}{ui.RESET}"
            f"{ui.GREY}hedef{ui.RESET} {ok:<14} "
            f"{ui.GREY}imlec{ui.RESET} {cx},{cy}  "
            f"{ui.GREY}pencere{ui.RESET} {aktif}  "
            f"{ui.GREY}{gecen:.1f}sn{ui.RESET}")

    def _hud_besle(self, faz: str, metin: str, sandbox) -> None:
        """Panele aynı olayı bas. Hata olursa koşumu ASLA düşürme —
        izleme aracı yüzünden ajan durmamalı."""
        try:
            if faz == "engellendi":
                self.hud.olay(metin, "engel")
                self.hud.guncelle(faz=faz, eylem="ENGELLENDI")
                return
            if faz == "baslat":
                self.hud.olay(metin, "iyi")
                return
            if faz == "bitti":
                self.hud.olay(metin)
                return
            # "bakiyor" — asıl kare
            self.hud.guncelle(faz=faz, eylem=metin,
                              hedef=list(sandbox._son_hedef) if sandbox._son_hedef else None,
                              pencere=sandbox.aktif_pencere()[:60])
            self.hud.olay(metin, "eylem")
            if (kare := getattr(sandbox, "frame", None)):
                png, olcek = kare()
                self.hud.kare(png, olcek)
        except Exception:
            pass

    def butce(self, ctx, strateji: str, gorev: str) -> None:
        """Döngüden bütçe durumunu panele aktar."""
        if self.hud is None:
            return
        try:
            b, l = ctx.budget.state, ctx.budget.limits
            self.hud.guncelle(
                adim=b.steps, token=b.tokens, cost=round(b.cost_usd, 4),
                sure=round(b.seconds, 1), strateji=strateji, gorev=gorev,
                limit={"steps": l.max_steps, "tokens": l.max_tokens,
                       "cost": l.max_cost_usd, "seconds": l.max_seconds})
        except Exception:
            pass

    def bitir(self, res) -> None:
        if self.hud is None:
            return
        try:
            self.hud.guncelle(durum=res.status.value, sebep=res.reason)
            self.hud.olay(f"{res.status.value} · {res.reason}",
                          "iyi" if res.status.clean else "engel")
        except Exception:
            pass

    # -- koşum sonrası özet ------------------------------------------------

    def ozet(self) -> str:
        bakis = [m for f, m in self.gecmis if f == "bakiyor"]
        engel = [m for f, m in self.gecmis if f == "engellendi"]
        satir = [f"  {ui.GREY}ajanin baktigi yerler ({len(bakis)}){ui.RESET}"]
        for m in bakis[-8:]:
            satir.append(f"    {ui.DIM}{m}{ui.RESET}")
        if engel:
            satir.append(f"  {ui.RED}engellenen ({len(engel)}){ui.RESET}")
            for m in engel[-4:]:
                satir.append(f"    {ui.RED}{m}{ui.RESET}")
        return "\n".join(satir)
