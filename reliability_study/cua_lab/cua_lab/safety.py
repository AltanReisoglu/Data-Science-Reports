"""
Gerçek masaüstünde çalışırken devreye giren koruma katmanı.

NEDEN AYRI DOSYA: guardrail'ler (döngü/bütçe) ajanın KENDİNİ korur — çok
adım atmasın, para yakmasın. Bu dosya SENİ korur. İki farklı problem, iki
farklı katman; karıştırılmamalı.

Ölçülmüş gerçek: aynı model `healthy` senaryosunda, hiçbir şey bozuk değilken
40 çağrının 40'ında aynı noktaya tıkladı. Sentetik ortamda bu bir istatistikti;
gerçek masaüstünde 40 gerçek tıklama demek. Bu katman o yüzden var.

DÖRT KADEME:
  1. Yıkıcı tuş / tuş kombinasyonu    — Delete ve türevleri hiç gönderilmiyor
  2. Yıkıcı metin deseni              — `rm -rf`, `DROP TABLE`, `shutdown`...
  3. Hassas pencere                   — terminal, parola yöneticisi, anahtarlık
  4. Failsafe                         — imleç sol üst köşeye giderse ANINDA iptal

Ve varsayılan olarak GİRDİ KAPALI: `allow_input=True` açıkça verilmeden hiçbir
fare/klavye olayı gönderilmiyor. Ekranı okur, ne yapacağını söyler, dokunmaz.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


class Blocked(Exception):
    """Koruma katmanı bir eylemi reddetti. Koşumu bitirmez — eylem
    yürütülmez ve ajana neden reddedildiği söylenir."""

    def __init__(self, kural: str, detay: str):
        self.kural, self.detay = kural, detay
        super().__init__(f"[{kural}] {detay}")


class Abort(Exception):
    """Failsafe. Koşum ANINDA biter."""


# -- 1) yıkıcı tuşlar ------------------------------------------------------
# Silme yetkisi VERİLMİYOR: Delete ve bütün varyantları bloke.
YASAK_TUS = {
    "delete", "kp_delete", "kp_decimal",
    "shift+delete",          # kalıcı silme (çöp kutusuna gitmez)
    "ctrl+shift+delete",     # tarayıcı: tüm geçmişi sil
}

# İçinde bunlar geçen HER kombinasyon reddedilir.
YASAK_PARCA = ("delete", "sysrq", "print")

# Pencere/oturum kapatma — silme değil ama geri alınamaz iş kaybı.
YASAK_KOMBO = {
    "alt+f4", "ctrl+q", "ctrl+w", "ctrl+shift+q",
    "ctrl+alt+backspace",    # X sunucusunu öldürür
    "ctrl+alt+f1", "ctrl+alt+f2", "ctrl+alt+f3",  # sanal terminal
    "super+l",               # ekran kilidi
}


# -- 2) yıkıcı metin desenleri --------------------------------------------
# Ajan bir terminale ya da herhangi bir yere BUNLARI yazamaz.
YASAK_METIN = [
    (r"\brm\s+-[rf]", "dosya silme komutu"),
    (r"\brmdir\b", "dizin silme"),
    (r"\bshred\b|\bwipe\b", "guvenli silme"),
    (r"\bmkfs\b|\bfdisk\b|\bparted\b", "disk bicimlendirme"),
    (r"\bdd\s+if=", "ham disk yazma"),
    (r">\s*/dev/(sd|nvme|hd)", "aygita dogrudan yazma"),
    (r"\bsudo\b", "yetki yukseltme"),
    (r"\bchmod\s+-R\b|\bchown\s+-R\b", "ozyinelemeli izin degisikligi"),
    (r"\bDROP\s+(TABLE|DATABASE|SCHEMA)\b", "veritabani silme"),
    (r"\bTRUNCATE\b", "tablo bosaltma"),
    (r"git\s+push\s+.*--force|git\s+push\s+.*-f\b", "zorla push"),
    (r"git\s+reset\s+--hard", "geri alinamaz reset"),
    (r"git\s+clean\s+-[a-z]*[fd]", "izlenmeyen dosya silme"),
    (r"\bshutdown\b|\breboot\b|\bpoweroff\b|\bhalt\b", "sistem kapatma"),
    (r"\bkillall\b|\bpkill\b", "toplu surec sonlandirma"),
    (r"curl\s+.*\|\s*(ba)?sh|wget\s+.*\|\s*(ba)?sh", "indirip calistirma"),
    (r"\bformat\b|\bdel\s+/[sf]\b", "windows silme"),
]

# -- 3) hassas pencereler --------------------------------------------------
# Bu pencerelere hicbir girdi gonderilmiyor. Terminal ilk sirada: gercek
# hasarin verilebilecegi tek yer orasi.
YASAK_PENCERE = [
    (r"terminal|konsole|xterm|alacritty|kitty|tilix|guake|st\b|urxvt", "terminal"),
    # Modern terminaller "terminal" kelimesini HIC gecirmiyor. GNOME Console
    # (org.gnome.Console) ve Ptyxis Ubuntu/Fedora'da varsayilan oldu; ikisi de
    # yukaridaki desenden siyriliyordu — olculdu. Genel `console` kullanmiyoruz:
    # tarayicinin "Console - DevTools" basligini da yakalar ve tarayici
    # gorevlerini bosuna keserdi.
    (r"org\.gnome\.console|gnome-console|\bptyxis\b|\bghostty\b|\bwezterm\b"
     r"|\bterminator\b|\bfoot\b|\bblackbox\b|\bcontour\b|\brio\b", "terminal"),
    # GNOME Terminal basligi cogu zaman "kullanici@makine: ~/dizin" seklinde
    # ve icinde "terminal" GECMIYOR. Bu desen olmadan terminal penceresi
    # korumanin arasindan siyriliyordu — olculdu.
    (r"^[\w.-]+@[\w.-]+\s*:\s*[~/]", "terminal (kabuk istemi)"),
    (r"\$\s*$|#\s*$", "terminal (kabuk istemi)"),
    (r"bitwarden|keepass|1password|lastpass|dashlane|proton pass", "parola yoneticisi"),
    (r"seahorse|keyring|anahtarlik|gnome-keyring", "anahtarlik"),
    (r"gnome-control-center|ayarlar|settings|systemsettings", "sistem ayarlari"),
    (r"synaptic|software|yazilim|gnome-software", "paket yoneticisi"),
    (r"gparted|disks|diskler", "disk araci"),
    (r"virt-manager|virtualbox|vmware", "sanallastirma"),
]


@dataclass
class SafetyPolicy:
    """Varsayilan: EN KISITLAYICI. Genisletmek acik bir karar olmali."""

    allow_input: bool = False          # girdi gonderilsin mi (fare/klavye)
    allow_typing: bool = True          # metin yazilabilsin mi
    allow_window_close: bool = False   # pencere kapatma kombinasyonlari
    max_real_actions: int = 60         # gercek girdi ust siniri (sert)
    failsafe_corner: int = 8           # imlec bu piksel kadar kosedeyse iptal
    dwell_seconds: float = 0.35        # tiklamadan once hedefte bekleme
    extra_blocked_windows: list[str] = field(default_factory=list)

    def __post_init__(self):
        self._sayac = 0
        self._engellenen: list[tuple[str, str]] = []

    # -- kontroller --------------------------------------------------------

    def check_key(self, tus: str) -> None:
        t = (tus or "").strip().lower().replace(" ", "")
        if t in YASAK_TUS:
            raise Blocked("silme-yasak", f"'{tus}' silme tusu — bu ajana silme "
                                         f"yetkisi verilmedi")
        for parca in YASAK_PARCA:
            if parca in t:
                raise Blocked("silme-yasak", f"'{tus}' icinde '{parca}' geciyor")
        if not self.allow_window_close and t in YASAK_KOMBO:
            raise Blocked("kapatma-yasak", f"'{tus}' pencere/oturum kapatiyor")

    def check_text(self, metin: str) -> None:
        if not self.allow_typing:
            raise Blocked("yazma-kapali", "metin yazma yetkisi kapali")
        for desen, ad in YASAK_METIN:
            if re.search(desen, metin, re.I):
                raise Blocked("yikici-metin",
                              f"'{ad}' deseni yakalandi — bu metin yazilmayacak")

    def check_window(self, baslik: str, sinif: str = "") -> None:
        """Başlık VE pencere sınıfı birlikte kontrol ediliyor.

        Yalnız başlığa bakmak yetmiyor: GNOME Terminal'in başlığı çoğu zaman
        `kullanici@makine: ~` ve içinde "terminal" geçmiyor. Pencere sınıfı
        (`WM_CLASS`) ise her zaman `gnome-terminal-server` — uygulamanın
        kimliği orada, kullanıcının değiştirebildiği başlıkta değil.
        """
        for kaynak, etiket in ((baslik or "", "baslik"), (sinif or "", "sinif")):
            k = kaynak.lower()
            if not k:
                continue
            for desen, ad in YASAK_PENCERE:
                if re.search(desen, k):
                    raise Blocked(
                        "hassas-pencere",
                        f"aktif pencere '{ad}' ({etiket}: {kaynak[:36]}) — "
                        f"girdi gonderilmiyor")
            for desen in self.extra_blocked_windows:
                if re.search(desen, k, re.I):
                    raise Blocked("hassas-pencere",
                                  f"kullanici yasakli pencere: {kaynak[:36]}")

    def check_failsafe(self, x: int, y: int) -> None:
        """pyautogui'nin `FAILSAFE` deseni: imleci koseye at, kacis kolu.

        Kullanici fareyi kapinca ajan durur — klavyeye ulasmaya calismasi
        gerekmez.
        """
        if x <= self.failsafe_corner and y <= self.failsafe_corner:
            raise Abort("FAILSAFE: imlec sol ust koseye tasindi — kosum iptal")

    def charge(self) -> None:
        self._sayac += 1
        if self._sayac > self.max_real_actions:
            raise Abort(f"sert tavan: {self.max_real_actions} gercek girdi asildi")

    def note_blocked(self, kural: str, detay: str) -> None:
        self._engellenen.append((kural, detay))

    def rapor(self) -> str:
        if not self._engellenen:
            return f"{self._sayac} gercek girdi · engellenen yok"
        satir = "\n".join(f"      {k}: {d}" for k, d in self._engellenen[-6:])
        return (f"{self._sayac} gercek girdi · {len(self._engellenen)} eylem "
                f"ENGELLENDI\n{satir}")


READONLY = SafetyPolicy(allow_input=False)
"""Varsayilan: ekrani oku, hicbir sey gonderme."""
