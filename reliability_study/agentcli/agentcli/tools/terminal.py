"""
Kısıtlı kabuk aracı.

Kullanıcının kararı: "kısıtlı kabuk". Komutlar çalışır ama hasar tek bir
çalışma dizinine hapsedilir.

DÖRT KISIT:
  1. Çalışma dizini kilitli   — `cd` dışarı çıkamaz, mutlak yol reddedilir
  2. Yıkıcı komut yasağı      — rm/sudo/dd/mkfs/shutdown/git push --force...
  3. Zaman aşımı              — asılı kalan komut koşumu kilitlemez
  4. Çıktı sınırı             — 4 KB; bağlamı şişirip döngü üretmesin

Guardrail'lerden AYRI bir katman: guardrail ajanı korur (döngü, bütçe),
bu kullanıcıyı korur.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field

YASAK = [
    (r"\brm\b", "dosya silme"),
    (r"\brmdir\b|\bshred\b|\bunlink\b", "silme"),
    (r"\bsudo\b|\bsu\b|\bdoas\b", "yetki yukseltme"),
    (r"\bdd\b|\bmkfs|\bfdisk\b|\bparted\b", "disk islemi"),
    (r"\bshutdown\b|\breboot\b|\bpoweroff\b|\bhalt\b|\bsystemctl\b", "sistem"),
    (r"\bkillall\b|\bpkill\b|\bkill\s+-9", "toplu sonlandirma"),
    (r"\bchmod\s+-R|\bchown\s+-R", "ozyinelemeli izin"),
    (r"git\s+push|git\s+reset\s+--hard|git\s+clean", "geri alinamaz git"),
    (r"\bcurl\b.*\|\s*(ba)?sh|\bwget\b.*\|\s*(ba)?sh", "indirip calistirma"),
    (r"\bcrontab\b|\bat\b\s|\bsystemd-run\b", "zamanlanmis gorev"),
    (r">\s*/dev/|>\s*/etc/|>\s*/usr/|>\s*/boot/", "sistem dizinine yazma"),
    (r"\bssh\b|\bscp\b|\brsync\b.*::", "uzak erisim"),
]


class Reddedildi(Exception):
    def __init__(self, kural: str):
        self.kural = kural
        super().__init__(kural)


@dataclass
class Terminal:
    """Tek bir dizine kilitli kabuk."""

    kok: str
    timeout: float = 20.0
    max_cikti: int = 4096
    engellenen: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.kok = os.path.abspath(self.kok)
        os.makedirs(self.kok, exist_ok=True)

    # -- kontrol -----------------------------------------------------------

    def _denetle(self, komut: str) -> None:
        for desen, ad in YASAK:
            if re.search(desen, komut):
                raise Reddedildi(f"'{ad}' engelli — kisitli kabuk")
        # Mutlak yol ya da yukarı çıkma: dizin kilidini deler.
        for parca in shlex.split(komut, posix=True) if komut.strip() else []:
            if parca.startswith("/") and not parca.startswith(self.kok):
                raise Reddedildi(f"mutlak yol disari cikiyor: {parca}")
            if ".." in parca.split("/"):
                raise Reddedildi(f"'..' ile disari cikma: {parca}")

    # -- DOSYA araçları ----------------------------------------------------
    # Kabuk üzerinden dosya yazmak bir VLM için tırnak cehennemi:
    # `echo "satir1\nsatir2" > f` her modelde farklı kaçıyor. Ölçtük — ajan
    # `echo tamamlandi > rapor.txt` yazabildi ama çok satırlı içerikte
    # takılıyordu. Bu üç araç kabuğu tamamen atlıyor.

    def _yol(self, ad: str) -> str:
        """Kök dizin dışına çıkmayı engelle — sembolik bağlar dahil."""
        tam = os.path.realpath(os.path.join(self.kok, ad))
        kok = os.path.realpath(self.kok)
        if not (tam == kok or tam.startswith(kok + os.sep)):
            raise Reddedildi(f"'{ad}' calisma dizini disina cikiyor")
        return tam

    def yaz(self, ad: str, icerik: str, ekle: bool = False) -> dict:
        try:
            yol = self._yol(ad)
        except Reddedildi as e:
            self.engellenen.append(f"yaz {ad} → {e.kural}")
            return {"ok": False, "engellendi": True, "cikti": "",
                    "hata": f"ENGELLENDI: {e.kural}"}
        os.makedirs(os.path.dirname(yol), exist_ok=True)
        with open(yol, "a" if ekle else "w", encoding="utf-8") as f:
            f.write(icerik)
        n = icerik.count("\n") + 1
        return {"ok": True, "engellendi": False,
                "cikti": f"{'eklendi' if ekle else 'yazildi'}: {ad} "
                         f"({len(icerik)} karakter, {n} satir)", "hata": None}

    def oku(self, ad: str, bas: int = 1, adet: int = 200) -> dict:
        try:
            yol = self._yol(ad)
        except Reddedildi as e:
            return {"ok": False, "engellendi": True, "cikti": "",
                    "hata": f"ENGELLENDI: {e.kural}"}
        if not os.path.isfile(yol):
            return {"ok": False, "engellendi": False, "cikti": "",
                    "hata": f"dosya yok: {ad}"}
        with open(yol, encoding="utf-8", errors="replace") as f:
            satirlar = f.readlines()
        dilim = satirlar[max(0, bas - 1): max(0, bas - 1) + adet]
        govde = "".join(f"{bas + i:>4}| {s}" for i, s in enumerate(dilim))
        return {"ok": True, "engellendi": False,
                "cikti": f"[{ad} · {bas}-{bas + len(dilim) - 1}/{len(satirlar)} satir]\n"
                         + govde[:self.max_cikti], "hata": None}

    def listele(self, alt: str = ".") -> dict:
        try:
            yol = self._yol(alt)
        except Reddedildi as e:
            return {"ok": False, "engellendi": True, "cikti": "",
                    "hata": f"ENGELLENDI: {e.kural}"}
        if not os.path.isdir(yol):
            return {"ok": False, "engellendi": False, "cikti": "",
                    "hata": f"dizin yok: {alt}"}
        satir = []
        for ad in sorted(os.listdir(yol))[:120]:
            tam = os.path.join(yol, ad)
            if os.path.isdir(tam):
                satir.append(f"  {ad}/")
            else:
                satir.append(f"  {ad}  ({os.path.getsize(tam)} B)")
        return {"ok": True, "engellendi": False,
                "cikti": f"[{alt}]\n" + ("\n".join(satir) or "  (bos)"), "hata": None}

    def ara(self, desen: str, alt: str = ".") -> dict:
        """Dosya içinde metin ara — `grep` kabuk kaçışına takılmadan."""
        import re as _re
        try:
            kok = self._yol(alt)
            d = _re.compile(desen)
        except Reddedildi as e:
            return {"ok": False, "engellendi": True, "cikti": "", "hata": f"ENGELLENDI: {e.kural}"}
        except _re.error as e:
            return {"ok": False, "engellendi": False, "cikti": "", "hata": f"bozuk desen: {e}"}
        bulgu = []
        for dizin, _, dosyalar in os.walk(kok):
            for ad in dosyalar:
                yol = os.path.join(dizin, ad)
                try:
                    with open(yol, encoding="utf-8", errors="replace") as f:
                        for i, s in enumerate(f, 1):
                            if d.search(s):
                                goreli = os.path.relpath(yol, kok)
                                bulgu.append(f"  {goreli}:{i}: {s.strip()[:90]}")
                                if len(bulgu) >= 40:
                                    break
                except OSError:
                    continue
            if len(bulgu) >= 40:
                break
        return {"ok": True, "engellendi": False,
                "cikti": "\n".join(bulgu) or f"'{desen}' bulunamadi", "hata": None}

    # -- yürütme -----------------------------------------------------------

    def calistir(self, komut: str) -> dict:
        try:
            self._denetle(komut)
        except Reddedildi as e:
            self.engellenen.append(f"{komut[:40]} → {e.kural}")
            return {"ok": False, "engellendi": True, "cikti": "",
                    "hata": f"ENGELLENDI: {e.kural}"}
        except ValueError as e:                       # bozuk tırnak
            return {"ok": False, "engellendi": False, "cikti": "",
                    "hata": f"komut ayristirilamadi: {e}"}
        try:
            r = subprocess.run(komut, shell=True, cwd=self.kok, timeout=self.timeout,
                               capture_output=True, text=True,
                               env={**os.environ, "HOME": self.kok})
        except subprocess.TimeoutExpired:
            return {"ok": False, "engellendi": False, "cikti": "",
                    "hata": f"zaman asimi ({self.timeout:g}sn)"}
        cikti = (r.stdout or "") + (("\n" + r.stderr) if r.stderr else "")
        # `mkdir`/`touch`/`mv` gibi komutlar BASARILI olunca hicbir sey yazmiyor.
        # Bos cikti hem dogrulayiciyi kor birakiyor hem de modele "nerede oldu"
        # bilgisini vermiyordu: ajan `mkdir altan` calistirip "masaustunde
        # olusturuldu" dedi — olculdu, klasor calisma dizinindeydi. Sessiz
        # basari artik KONUMU soyluyor.
        if r.returncode == 0 and not cikti.strip():
            cikti = f"(cikti yok · cikis kodu 0) · calisma dizini: {self.kok}"
        kirpildi = len(cikti) > self.max_cikti
        return {"ok": r.returncode == 0, "engellendi": False,
                "cikti": cikti[:self.max_cikti] + ("\n…[kirpildi]" if kirpildi else ""),
                "hata": None if r.returncode == 0 else f"cikis kodu {r.returncode}"}
