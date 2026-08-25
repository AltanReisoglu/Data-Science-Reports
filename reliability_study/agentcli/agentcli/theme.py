"""
Beyaz tema — açık arka planlı terminal için renk paleti.

Koyu tema paletleri (parlak sarı, açık cyan) beyaz zeminde okunmuyor. Buradaki
renkler ANSI 256'dan, hepsi beyaz üstünde kontrastlı seçildi.
"""

from __future__ import annotations

import os
import shutil
import sys

_TTY = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(kod: str) -> str:
    return kod if _TTY else ""


RESET = _c("\033[0m")
B     = _c("\033[1m")
DIM   = _c("\033[38;5;245m")      # gri — beyaz üstünde okunur
LINE  = _c("\033[38;5;252m")      # çok açık gri, çizgiler
INK   = _c("\033[38;5;235m")      # neredeyse siyah, ana metin

BLUE  = _c("\033[38;5;25m")       # koyu mavi   — araç çağrısı
GREEN = _c("\033[38;5;28m")       # koyu yeşil  — başarı
AMBER = _c("\033[38;5;130m")      # koyu turuncu— uyarı
RED   = _c("\033[38;5;124m")      # koyu kırmızı— durdurma
PURP  = _c("\033[38;5;54m")       # mor         — bütçe
TEAL  = _c("\033[38;5;30m")       # petrol      — çekimser

DURUM = {
    "OK": GREEN, "STUCK": AMBER, "BUDGET_EXHAUSTED": PURP,
    "DEGRADED": AMBER, "NEEDS_INPUT": TEAL, "CEILING": RED,
}


# -- gerçek beyaz arka plan --------------------------------------------------
# OSC 11/10 terminalin KENDİ varsayılan arka/ön plan rengini değiştiriyor —
# tek tek satırları boyamaktan farklı: kaydırma, boş alan ve `clear` dahil
# her yer beyaz oluyor. GNOME Terminal, xterm, kitty, alacritty, WezTerm
# destekliyor. Desteklemeyen terminal diziyi sessizce yutuyor, bozulma yok.
#
# Çıkarken 111/110 ile geri alınıyor — kullanıcının teması kalıcı bozulmasın.
BEYAZ = "#ffffff"
KOYU = "#1a1c20"

_acik = False


def beyaz_ac(arka: str = BEYAZ, on: str = KOYU) -> None:
    global _acik
    if not _TTY or os.environ.get("AGENTCLI_TEMA") == "kapali":
        return
    sys.stdout.write(f"\033]11;{arka}\007\033]10;{on}\007")
    sys.stdout.write("\033[2J\033[H")        # temizle, imleci başa al
    sys.stdout.flush()
    _acik = True


def beyaz_kapat() -> None:
    """Terminalin kendi temasına geri dön."""
    global _acik
    if not _acik:
        return
    sys.stdout.write("\033]111\007\033]110\007\033[0m\n")
    sys.stdout.flush()
    _acik = False


def en(varsayilan: int = 92) -> int:
    try:
        return min(shutil.get_terminal_size().columns, 108)
    except Exception:
        return varsayilan


def cizgi(ch: str = "─") -> str:
    return f"{LINE}{ch * en()}{RESET}"


def sar(metin: str, genislik: int) -> list[str]:
    out, satir = [], ""
    for k in (metin or "").split():
        if len(satir) + len(k) + 1 > genislik:
            out.append(satir); satir = k
        else:
            satir = f"{satir} {k}".strip()
    if satir:
        out.append(satir)
    return out or [""]
