"""
Terminal görsel katmanı — stdlib, sıfır bağımlılık.

Renkler TTY değilse ya da NO_COLOR ayarlıysa kendiliğinden kapanıyor; çıktı
bir dosyaya yönlendirildiğinde ANSI kaçış dizileri bulaşmasın diye.
"""

from __future__ import annotations

import os
import shutil
import sys

_TTY = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code: str) -> str:
    return code if _TTY else ""


RESET = _c("\033[0m")
DIM = _c("\033[2m")
B = _c("\033[1m")
RED = _c("\033[31m")
GREEN = _c("\033[32m")
YELLOW = _c("\033[33m")
BLUE = _c("\033[34m")
MAGENTA = _c("\033[35m")
CYAN = _c("\033[36m")
GREY = _c("\033[90m")

# Terminal durumuna göre renk. Ayrı terminal durumların ayrı görünmesi,
# Arize'ın "durma sebebi kaydı" ilkesinin görsel karşılığı.
STATUS_COLOR = {
    "OK": GREEN,
    "STUCK": YELLOW,
    "BUDGET_EXHAUSTED": MAGENTA,
    "DEGRADED": YELLOW,
    "NEEDS_INPUT": CYAN,
    "CEILING": RED,
}


def width(default: int = 78) -> int:
    try:
        return min(shutil.get_terminal_size().columns, 100)
    except Exception:
        return default


def rule(ch: str = "─", pad: int = 2) -> str:
    return GREY + " " * pad + ch * (width() - pad * 2) + RESET


def title(text: str, sub: str = "") -> str:
    w = width() - 4
    top = f"{GREY}  ╭{'─' * (w - 2)}╮{RESET}"
    mid = f"{GREY}  │{RESET} {B}{text}{RESET}"
    mid += " " * max(0, w - 3 - len(text)) + f"{GREY}│{RESET}"
    out = [top, mid]
    if sub:
        s = f"{GREY}  │{RESET} {DIM}{sub}{RESET}"
        s += " " * max(0, w - 3 - len(sub)) + f"{GREY}│{RESET}"
        out.append(s)
    out.append(f"{GREY}  ╰{'─' * (w - 2)}╯{RESET}")
    return "\n".join(out)


def kv(key: str, val: str, w: int = 10, color: str = "") -> str:
    return f"  {GREY}{key:<{w}}{RESET}{color}{val}{RESET}"


def status_line(text: str) -> None:
    """Aynı satırı ez. TTY değilse hiç basma — log dosyasını kirletmesin."""
    if not _TTY:
        return
    sys.stdout.write("\r\033[K  " + text)
    sys.stdout.flush()


def clear_line() -> None:
    if _TTY:
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


def bar(used: float, limit: float | None, n: int = 12) -> str:
    """Bütçe doluluk çubuğu. Limit yoksa boş döner."""
    if not limit:
        return f"{GREY}—{RESET}"
    oran = min(used / limit, 1.0)
    dolu = int(oran * n)
    renk = GREEN if oran < 0.6 else (YELLOW if oran < 0.85 else RED)
    return f"{renk}{'█' * dolu}{GREY}{'░' * (n - dolu)}{RESET}"
