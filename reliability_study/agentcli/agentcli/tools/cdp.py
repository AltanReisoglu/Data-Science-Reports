"""
Chrome DevTools Protocol istemcisi — stdlib, sıfır bağımlılık.

NEDEN KENDİMİZ YAZIYORUZ: CDP komutları WebSocket üzerinden gidiyor, Python
stdlib'de WebSocket istemcisi yok. `playwright` ~150 MB tarayıcı indiriyor,
`websocket-client` küçük ama yine bir bağımlılık. RFC 6455'in ihtiyacımız olan
kısmı ~90 satır: el sıkışma + maskeli çerçeve yazma + çerçeve okuma.

Kapsam bilerek dar: metin çerçevesi, maskeleme, 3 uzunluk biçimi, ping/pong.
Uzantı yok, sıkıştırma yok, parçalı mesaj sadece basit birleştirme.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import socket
import struct
import subprocess
import time
import urllib.request
from urllib.parse import urlparse


class WSError(Exception):
    pass


class WebSocket:
    """RFC 6455'in istemci tarafı — yalnız ihtiyacımız kadarı."""

    def __init__(self, url: str, timeout: float = 30.0):
        u = urlparse(url)
        if u.scheme != "ws":
            raise WSError(f"yalniz ws:// destekleniyor, verilen: {u.scheme}")
        self.sock = socket.create_connection((u.hostname, u.port or 80), timeout=timeout)
        self.sock.settimeout(timeout)
        self._el_sikis(u.path or "/", u.hostname, u.port)
        self._tampon = b""

    def _el_sikis(self, yol: str, host: str, port: int) -> None:
        anahtar = base64.b64encode(secrets.token_bytes(16)).decode()
        istek = (
            f"GET {yol} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {anahtar}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(istek.encode())
        yanit = b""
        while b"\r\n\r\n" not in yanit:
            parca = self.sock.recv(4096)
            if not parca:
                raise WSError("el sikisma sirasinda baglanti kapandi")
            yanit += parca
        if b"101" not in yanit.split(b"\r\n", 1)[0]:
            raise WSError(f"yukseltme reddedildi: {yanit[:120]!r}")
        # Gövdeye taşan baytlar varsa sakla — ilk çerçeve orada olabilir.
        self._tampon = yanit.split(b"\r\n\r\n", 1)[1]

    # -- yazma -------------------------------------------------------------

    def send(self, metin: str) -> None:
        veri = metin.encode()
        n = len(veri)
        cerceve = bytearray([0x81])          # FIN + metin
        if n < 126:
            cerceve.append(0x80 | n)         # MASK biti — istemci zorunlu
        elif n < 65536:
            cerceve.append(0x80 | 126)
            cerceve += struct.pack(">H", n)
        else:
            cerceve.append(0x80 | 127)
            cerceve += struct.pack(">Q", n)
        maske = secrets.token_bytes(4)
        cerceve += maske
        cerceve += bytes(b ^ maske[i % 4] for i, b in enumerate(veri))
        self.sock.sendall(bytes(cerceve))

    # -- okuma -------------------------------------------------------------

    def _oku(self, n: int) -> bytes:
        while len(self._tampon) < n:
            parca = self.sock.recv(65536)
            if not parca:
                raise WSError("baglanti kapandi")
            self._tampon += parca
        cikti, self._tampon = self._tampon[:n], self._tampon[n:]
        return cikti

    def recv(self) -> str:
        """Tam bir metin mesajı döndür. Ping'e otomatik pong."""
        parcalar = []
        while True:
            b1, b2 = self._oku(2)
            fin, opcode = b1 & 0x80, b1 & 0x0F
            uzunluk = b2 & 0x7F
            if uzunluk == 126:
                uzunluk = struct.unpack(">H", self._oku(2))[0]
            elif uzunluk == 127:
                uzunluk = struct.unpack(">Q", self._oku(8))[0]
            govde = self._oku(uzunluk) if uzunluk else b""

            if opcode == 0x9:                # ping
                self.sock.sendall(bytes([0x8A, 0x80]) + secrets.token_bytes(4))
                continue
            if opcode == 0x8:                # close
                raise WSError("sunucu baglantiyi kapatti")
            if opcode in (0x1, 0x0):         # metin ya da devam
                parcalar.append(govde)
                if fin:
                    return b"".join(parcalar).decode("utf-8", "replace")
            # ikili çerçeve: yok say

    def close(self) -> None:
        try:
            self.sock.sendall(bytes([0x88, 0x80]) + secrets.token_bytes(4))
        except OSError:
            pass
        self.sock.close()


class Chrome:
    """Ajan için AYRI bir headless Chrome. Kullanıcının tarayıcısına dokunmaz.

    Kendi `--user-data-dir`'i var: açık sekmeler, çerezler, oturumlar
    görünmez. Kullanıcının kararı buydu.
    """

    def __init__(self, port: int = 9222, width: int = 1280, height: int = 800,
                 headless: bool = True, profil: str | None = None,
                 konum: str | None = None):
        self.port, self.width, self.height = port, width, height
        # `konum="sag"` → gorunur pencere ekranin SAG YARISINA yerlesir.
        # Solda terminal, sagda ajanin gordugu sayfa: ikisini yan yana izle.
        self.konum = konum
        self.profil = profil or os.path.join(
            os.environ.get("TMPDIR", "/tmp"), f"agentcli_chrome_{port}")
        self._proc = None
        self._ws: WebSocket | None = None
        self._id = 0
        self.headless = headless

    # -- yaşam döngüsü -----------------------------------------------------

    def start(self) -> None:
        exe = next((c for c in ("google-chrome", "chromium", "chromium-browser")
                    if _var(c)), None)
        if not exe:
            raise RuntimeError("Chrome/Chromium bulunamadi")
        if self.konum and not self.headless:
            ekran_w, ekran_h = _ekran_boyutu()
            if self.konum == "sag":
                x, self.width = ekran_w // 2, ekran_w // 2
            elif self.konum == "sol":
                x, self.width = 0, ekran_w // 2
            else:
                x = 0
            self.height = ekran_h - 40
            yerlesim = [f"--window-position={x},0"]
        else:
            yerlesim = []
        argv = [exe, f"--remote-debugging-port={self.port}",
                f"--user-data-dir={self.profil}",
                *yerlesim,
                f"--window-size={self.width},{self.height}",
                "--no-first-run", "--no-default-browser-check",
                "--disable-features=Translate,MediaRouter",
                "--disable-background-networking", "about:blank"]
        if self.headless:
            argv.insert(1, "--headless=new")
            argv.insert(2, "--disable-gpu")
        self._proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
        hedef = self._bekle_hedef()
        if self.konum and not self.headless:
            self._zorla_yerlestir()
        self._ws = WebSocket(hedef)
        self.cmd("Page.enable")
        self.cmd("Runtime.enable")
        self.cmd("DOM.enable")

    def _bekle_hedef(self, saniye: float = 20.0) -> str:
        son = time.monotonic() + saniye
        while time.monotonic() < son:
            try:
                sayfalar = json.load(urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/json", timeout=2))
                for s in sayfalar:
                    if s.get("type") == "page" and s.get("webSocketDebuggerUrl"):
                        return s["webSocketDebuggerUrl"]
            except Exception:
                pass
            time.sleep(0.3)
        raise RuntimeError("Chrome CDP hedefi acilmadi")

    def _zorla_yerlestir(self) -> None:
        """Pencereyi xdotool ile TAM istenen yere taşı.

        `--window-position` bir İSTEK; pencere yöneticisi (GNOME) çoğu zaman
        eziyor — ölçtük: 960 istendi, 836'da 1110 genişlikle açıldı.
        `xdotool windowmove/windowsize` pencere haritalandıktan sonra
        çalıştığı için WM'i geçiyor.

        xdotool yoksa sessizce atlanıyor: pencere yine açılır, sadece
        konumu WM'e kalır.
        """
        if not _var("xdotool"):
            return
        ekran_w, ekran_h = _ekran_boyutu()
        w = ekran_w // 2
        x = ekran_w - w if self.konum == "sag" else 0
        h = ekran_h - 60
        son = time.monotonic() + 8
        while time.monotonic() < son:
            try:
                ids = subprocess.run(
                    ["xdotool", "search", "--onlyvisible", "--class", "chrome"],
                    capture_output=True, text=True, timeout=5).stdout.split()
            except Exception:
                return
            # Bu ORNEGIN penceresi: profil dizini bize ozel, ama xdotool
            # profil bilmiyor — en son acilan pencereyi aliyoruz.
            if ids:
                wid = ids[-1]
                for argv in (["xdotool", "windowsize", wid, str(w), str(h)],
                             ["xdotool", "windowmove", wid, str(x), "0"]):
                    try:
                        subprocess.run(argv, capture_output=True, timeout=5)
                    except Exception:
                        pass
                self.width, self.height = w, h
                return
            time.sleep(0.4)

    def stop(self) -> None:
        if self._ws:
            self._ws.close()
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    # -- komut -------------------------------------------------------------

    def cmd(self, yontem: str, **params) -> dict:
        """Tek bir CDP komutu. Yanıtı id ile eşleştirir, olayları atlar."""
        if not self._ws:
            raise RuntimeError("Chrome baslatilmadi")
        self._id += 1
        iid = self._id
        self._ws.send(json.dumps({"id": iid, "method": yontem, "params": params}))
        while True:
            mesaj = json.loads(self._ws.recv())
            if mesaj.get("id") != iid:
                continue                      # olay — bu komutun yanıtı değil
            if "error" in mesaj:
                raise RuntimeError(f"{yontem}: {mesaj['error'].get('message')}")
            return mesaj.get("result", {})


def _ekran_boyutu() -> tuple[int, int]:
    """Ekran çözünürlüğü — xrandr, olmazsa DRM modes, olmazsa varsayılan.

    `xdotool` gerektirmiyor; kurulu olmayan bir araca bağlanmamak için üç
    kademeli düşüş.
    """
    import glob
    import re
    try:
        o = subprocess.run(["xrandr"], capture_output=True, text=True,
                           timeout=5).stdout
        if (m := re.search(r"current (\d+) x (\d+)", o)):
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    for f in sorted(glob.glob("/sys/class/drm/*/modes")):
        try:
            s = open(f).readline().strip()
            if "x" in s:
                w, h = s.split("x")
                return int(w), int(h)
        except (OSError, ValueError):
            continue
    return 1920, 1080


def _var(ad: str) -> bool:
    import shutil
    return shutil.which(ad) is not None
