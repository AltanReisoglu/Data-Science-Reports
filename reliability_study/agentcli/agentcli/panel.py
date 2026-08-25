"""
Sağ üst canlı panel — ajanın hareketleri + kısıt durumu.

TEKNİK: DECSTBM ile ekranın ÜST N satırı kaydırma alanının dışına alınıyor
(`\033[{n+1};{alt}r`). Akış aşağıda kayarken üst blok sabit kalıyor ve her
adımda oraya yeniden çiziliyor. `curses` gerekmiyor, normal `print` çıktısı
bozulmuyor.

PANELDE İKİ ŞEY VAR ve ikisi de bilerek:

  SOL   ajanın son hareketleri — ne yaptı
  SAĞ   KISIT DURUMU — guardrail tetiklenmeye ne kadar yakın

İkincisi asıl katkı. Guardrail'in KARARINI görmek kolay (durdu ya da durmadı);
anlaşılmayan şey karara ne kadar kaldığı. `openhands-stuck` her araç
çağrısında "ayni eylem+gozlem 3/4" diyorsa, bir sonraki tekrarda duracağını
önceden görüyorsun.
"""

from __future__ import annotations

import sys

from . import theme as T

YUKSEKLIK = 11


class Panel:
    def __init__(self, aktif: bool = True, satir: int = YUKSEKLIK):
        self.aktif = aktif and T._TTY
        self.satir = satir
        self.hareket: list[str] = []
        self._kuruldu = False

    # -- kaydırma alanı ----------------------------------------------------

    def kur(self) -> None:
        if not self.aktif or self._kuruldu:
            return
        try:
            import shutil
            _, h = shutil.get_terminal_size()
        except Exception:
            h = 40
        self.alt = h
        sys.stdout.write("\033[2J")                       # temizle
        sys.stdout.write(f"\033[{self.satir + 1};{h}r")   # kaydırma alanı ALTTA
        # Alanı baştan DOLDUR: imleç en alt satıra insin. Böylece panel
        # çizdikten sonra imleci hep `alt` satırına geri koyabiliyoruz.
        # `\033[s`/`\033[u` ile kaydedilen konum MUTLAK; akış kaydıkça
        # bayatlıyor ve imleç eski içerikli bir satıra iniyordu — ekranda
        # `───── ⏵ browser.key` gibi karışmaların sebebi buydu.
        sys.stdout.write("\n" * (h - self.satir - 1))
        sys.stdout.write(f"\033[{h};1H")
        sys.stdout.flush()
        self._kuruldu = True

    def kaldir(self) -> None:
        if not self._kuruldu:
            return
        sys.stdout.write("\033[r\033[?25h")               # alan sıfırla, imleç geri
        sys.stdout.write(f"\033[{getattr(self, 'alt', 40)};1H\n")
        sys.stdout.flush()
        self._kuruldu = False

    # -- çizim -------------------------------------------------------------

    def ekle_hareket(self, metin: str) -> None:
        self.hareket.append(metin)
        self.hareket = self.hareket[-6:]

    def ciz(self, baslik: str, aktif_snap: list[tuple], golge: list[tuple],
            butce: list[tuple]) -> None:
        if not self.aktif:
            return
        self.kur()
        w = T.en()
        sol_w = max(30, int(w * 0.42))
        sag_w = w - sol_w - 5

        sat = []
        sat.append(f"{T.DIM}{baslik}{T.RESET}")
        sat.append(T.cizgi("╌"))
        sol = [f"{T.DIM}AJAN{T.RESET}"] + [f"  {h}" for h in self.hareket[-5:]]
        sag = [f"{T.DIM}KISIT DURUMU{T.RESET}"]
        for ad, kul, lim in (aktif_snap + butce)[:5]:
            sag.append("  " + _cubuk(ad, kul, lim, sag_w - 4))
        if golge:
            sag.append(f"  {T.DIM}gölge: {golge[0]}{T.RESET}")

        n = max(len(sol), len(sag))
        for i in range(n):
            l = sol[i] if i < len(sol) else ""
            r = sag[i] if i < len(sag) else ""
            sat.append(f"{_pad(l, sol_w)}{T.LINE}│{T.RESET} {r}")

        sys.stdout.write("\033[?25l")                      # imleci gizle
        for i, s in enumerate(sat[:self.satir]):
            sys.stdout.write(f"\033[{i + 1};1H\033[K  {s}")
        for i in range(len(sat), self.satir):
            sys.stdout.write(f"\033[{i + 1};1H\033[K")
        # Kaydet/geri-al YOK: imleç doğrudan kaydırma alanının EN ALT
        # satırına konuyor. Akış oraya yazıp yukarı kaydırdığı için bu her
        # zaman doğru konum — mutlak kayıt gibi bayatlamıyor.
        sys.stdout.write(f"\033[{getattr(self, 'alt', 40)};1H\033[?25h")
        sys.stdout.flush()


def _cubuk(ad: str, kul, lim, w: int, n: int = 10) -> str:
    """Kısıt çubuğu: doluluk oranı + sayılar."""
    try:
        oran = min(1.0, float(kul) / float(lim)) if lim else 0.0
    except (TypeError, ZeroDivisionError, ValueError):
        oran = 0.0
    if kul == -1:
        return f"{T.DIM}{ad[:20]:<21}pencere dolmadı{T.RESET}"
    renk = T.GREEN if oran < 0.6 else (T.AMBER if oran < 0.9 else T.RED)
    dolu = int(oran * n)
    cubuk = f"{renk}{'█' * dolu}{T.LINE}{'░' * (n - dolu)}{T.RESET}"
    say = f"{_g(kul)}/{_g(lim)}"
    return f"{T.DIM}{ad[:20]:<21}{T.RESET}{cubuk} {renk}{say}{T.RESET}"


def _g(v) -> str:
    try:
        f = float(v)
        return f"{int(f)}" if f == int(f) else f"{f:g}"
    except (TypeError, ValueError):
        return str(v)


def _pad(metin: str, w: int) -> str:
    """ANSI kaçışlarını saymadan hizala."""
    import re
    gorunur = len(re.sub(r"\033\[[0-9;]*[A-Za-z]", "", metin))
    return metin + " " * max(0, w - gorunur)
