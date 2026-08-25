"""
Masaüstü aracı — GERÇEK ekran görüntüsü + gerçek fare/klavye.

`cua_lab/sandbox/x11.py` ve `cua_lab/safety.py` zaten yazılmıştı; bu dosya
onları `agentcli`'nin araç sözleşmesine sarıyor. Yeniden yazmıyor.

TARAYICI ARACINDAN ÜÇ FARKI VAR ve üçü de riski artırıyor:

  1. GÖRÜNTÜ  Chrome sekmesi değil, EKRANIN TAMAMI. O anda ekranda ne varsa
     (editör, terminal, mesajlaşma) VLM'e gider.
  2. KOORDİNAT  DOM yok, numaralı öğe yok. Model `click(x, y)` demek zorunda —
     tarayıcıda tamamen ortadan kalkan ölçek kayması hatası burada geri geliyor.
  3. GERİ ALINAMAZ  Tıklama gerçek uygulamalara gidiyor.

Bu yüzden `safety.SafetyPolicy` zorunlu ve varsayılanı en kısıtlayıcı:
girdi KAPALI, silme tuşları yasak, terminal/parola pencerelerine girdi yok,
imleç sol üst köşeye giderse koşum iptal.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_KOK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_KOK / "cua_lab"))

from cua_lab.events import Act                     # noqa: E402
from cua_lab.safety import Abort, SafetyPolicy     # noqa: E402
from cua_lab.sandbox.x11 import X11Sandbox         # noqa: E402


class Desktop:
    """Gerçek masaüstü. `agentcli` araç sözleşmesine uyar."""

    def __init__(self, allow_input: bool = False, display: str | None = None,
                 dwell: float = 0.6, max_actions: int = 40,
                 shrink: int = 1280, gozlemci=None):
        self.policy = SafetyPolicy(
            allow_input=allow_input,
            allow_window_close=False,          # silme/kapatma yetkisi YOK
            max_real_actions=max_actions,
            dwell_seconds=dwell,
        )
        self.sandbox = X11Sandbox(display=display, policy=self.policy,
                                  shrink=shrink, gozlemci=gozlemci)
        self._basladi = False

    # -- yaşam döngüsü -----------------------------------------------------

    def start(self) -> None:
        if not self._basladi:
            self.sandbox.start()
            self._basladi = True

    def stop(self) -> None:
        if self._basladi:
            self.sandbox.stop()
            self._basladi = False

    @property
    def allow_input(self) -> bool:
        return self.policy.allow_input

    # -- ajanın çağırdığı yüzey --------------------------------------------

    def screenshot(self) -> bytes:
        self.start()
        png, _ = self.sandbox.frame()
        return png

    def olcek(self) -> float:
        _, o = self.sandbox.frame()
        return o

    def durum(self) -> str:
        """Modele giden metin bağlamı — görüntüde okunamayacak şeyler."""
        self.start()
        return self.sandbox.describe()

    def durum_hash(self) -> str:
        self.start()
        return self.sandbox.screen_hash()

    def _yap(self, act: Act, **args) -> str:
        self.start()
        try:
            r = self.sandbox.execute(act, args)
        except Abort as e:
            return f"IPTAL: {e}"
        if r.error:
            return r.error
        return f"{act.value}({args}) tamam"

    def click(self, x: int, y: int) -> str:
        return self._yap(Act.LEFT_CLICK, x=int(x), y=int(y))

    def double_click(self, x: int, y: int) -> str:
        return self._yap(Act.DOUBLE_CLICK, x=int(x), y=int(y))

    def move(self, x: int, y: int) -> str:
        return self._yap(Act.MOUSE_MOVE, x=int(x), y=int(y))

    def type(self, metin: str) -> str:
        return self._yap(Act.TYPE, text=str(metin))

    def key(self, tus: str) -> str:
        return self._yap(Act.KEY, text=str(tus))

    def scroll(self, dy: int = 3) -> str:
        yon = "down" if dy > 0 else "up"
        return self._yap(Act.SCROLL, scroll_direction=yon,
                         scroll_amount=abs(int(dy)) or 3)

    # -- PENCERE araçları --------------------------------------------------

    def _sinif(self, wid: str) -> str:
        """Pencerenin WM_CLASS'i.

        `xdotool getwindowclassname` GUVENILMEZ: bu makinedeki surumde komut
        HIC YOK ("Unknown command"), yani sinif her pencerede bos donuyordu ve
        guvenlik katmani yalniz BASLIGA bakiyordu. Baslik kullanicinin
        degistirebildigi bir sey; uygulamanin kimligi WM_CLASS'ta. `xprop`
        her X11 kurulumunda var ve dogrudan pencere ozelligini okuyor.
        """
        for komut in (("xprop", "-id", wid, "WM_CLASS"),
                      ("xdotool", "getwindowclassname", wid)):
            try:
                cikti = self.sandbox._x(*komut)
            except Exception:
                continue
            if not cikti:
                continue
            if "WM_CLASS" in cikti:
                # WM_CLASS(STRING) = "code", "code"  ->  code code
                return " ".join(re.findall(r'"([^"]*)"', cikti)) or "?"
            return cikti.strip() or "?"
        return "?"

    def pencereler(self) -> str:
        """Açık pencereler — ad, sınıf, konum, boyut.

        Ajanın "ekranda ne var" sorusunu tam ekran görüntüsü göndermeden
        cevaplaması için. Ucuz ve gizlilik açısından çok daha iyi.
        """
        self.start()
        try:
            ids = self.sandbox._x("xdotool", "search", "--onlyvisible",
                                  "--name", ".").split()
        except Exception as e:
            return f"pencere listesi alinamadi: {e}"
        satir = []
        for wid in ids[:40]:
            try:
                ad = self.sandbox._x("xdotool", "getwindowname", wid)
                g = self.sandbox._x("xdotool", "getwindowgeometry", "--shell", wid)
            except Exception:
                continue                       # penceresiz/kapanmis id
            sinif = self._sinif(wid)
            d = dict(l.split("=", 1) for l in g.splitlines() if "=" in l)
            try:
                w, h = int(d.get("WIDTH", 0)), int(d.get("HEIGHT", 0))
            except ValueError:
                continue
            if ad and w > 120 and h > 80:
                satir.append(f"  [{wid}] {ad[:44]}  ({sinif})  "
                             f"{w}x{h} @({d.get('X')},{d.get('Y')})")
        return "\n".join(satir) or "(gorunur pencere yok)"

    def pencere_goruntusu(self, ad_ya_da_id: str) -> tuple[bytes, str]:
        """YALNIZCA bir pencerenin görüntüsü — tam ekran değil.

        İki kazanç: ekranındaki diğer her şey (editör, mesajlaşma, parola
        yöneticisi) dış servise GİTMİYOR; ve görüntü küçüldüğü için görsel
        token maliyeti düşüyor.
        """
        self.start()
        import subprocess
        import tempfile
        try:
            if str(ad_ya_da_id).isdigit():
                wid = str(ad_ya_da_id)
            else:
                bulunan = self.sandbox._x("xdotool", "search", "--onlyvisible",
                                          "--name", ad_ya_da_id).split()
                if not bulunan:
                    return b"", f"'{ad_ya_da_id}' adli pencere yok"
                wid = bulunan[-1]
            g = self.sandbox._x("xdotool", "getwindowgeometry", "--shell", wid)
            d = dict(l.split("=", 1) for l in g.splitlines() if "=" in l)
            x, y = int(d["X"]), int(d["Y"])
            w, h = int(d["WIDTH"]), int(d["HEIGHT"])
            ad = self.sandbox._x("xdotool", "getwindowname", wid)
        except Exception as e:
            return b"", f"pencere bulunamadi: {e}"
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            yol = f.name
        try:
            subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "x11grab",
                            "-video_size", f"{w}x{h}",
                            "-i", f"{self.sandbox.display}+{max(0,x)},{max(0,y)}",
                            "-frames:v", "1", "-y", yol],
                           capture_output=True, timeout=20, check=True)
            with open(yol, "rb") as fh:
                ham = fh.read()
        except Exception as e:
            return b"", f"goruntu alinamadi: {e}"
        finally:
            try:
                import os as _os
                _os.unlink(yol)
            except OSError:
                pass
        return ham, f"'{ad[:40]}' {w}x{h}"

    def odakla(self, ad_ya_da_id: str) -> str:
        self.start()
        try:
            if str(ad_ya_da_id).isdigit():
                wid = str(ad_ya_da_id)
            else:
                bulunan = self.sandbox._x("xdotool", "search", "--onlyvisible",
                                          "--name", ad_ya_da_id).split()
                if not bulunan:
                    return f"'{ad_ya_da_id}' adli pencere yok"
                wid = bulunan[-1]
            ad = self.sandbox._x("xdotool", "getwindowname", wid)
            sinif = self._sinif(wid)
            # Odaklamak da bir GIRDI — hassas pencere korumasi burada da gecerli.
            self.policy.check_window(ad, sinif)
            if not self.policy.allow_input:
                return "girdi KAPALI (salt okuma) — odaklama yapilmadi"
            self.sandbox._x("xdotool", "windowactivate", wid)
            return f"odaklandi: {ad[:44]}"
        except Exception as e:
            return f"odaklanamadi: {e}"

    def surukle(self, x1: int, y1: int, x2: int, y2: int) -> str:
        self.start()
        if not self.policy.allow_input:
            return "girdi KAPALI (salt okuma)"
        try:
            self.policy.charge()
            self.sandbox._x("xdotool", "mousemove", str(x1), str(y1),
                            "mousedown", "1", "mousemove", str(x2), str(y2),
                            "mouseup", "1")
            return f"suruklendi: ({x1},{y1}) -> ({x2},{y2})"
        except Exception as e:
            return f"suruklenemedi: {e}"

    @property
    def engellenen(self) -> list:
        return [f"{k}: {d}" for k, d in self.policy._engellenen]

    def rapor(self) -> str:
        return self.policy.rapor()
