"""
GERÇEK masaüstü backend'i — X11.

`fake.py` sentetik bir sözlüktü; bu dosya senin ekranını okuyor ve senin farene
klavyene gerçek olay gönderiyor.

Anthropic'in referans demosu da aynı yapıyı kullanıyor: X11 + `xdotool` girdi +
ekran görüntüsü. Fark şu — onlarınki bir konteynerdeki Xvfb'ye bakıyor, bu
`DISPLAY` neyse ona bakıyor.

DIŞ ARAÇLAR (bilerek stdlib disi tek bagimlilik — sistem paketi):
    xdotool     fare/klavye olayi + pencere sorgusu   ZORUNLU (girdi icin)
    ffmpeg      ekran goruntusu (x11grab)             en az biri
    scrot                                             gerekli
    PIL         goruntu kucultme                      opsiyonel

Hicbiri yoksa sinif KURULMUYOR — sessizce bozuk calismak yerine acik hata.

İKİ KORUMA KATMANI, ikisi de ayri:
    safety.SafetyPolicy   SENI korur — silme yok, terminal yok, failsafe
    strategies/*          AJANI korur — dongu, butce

VARSAYILAN GİRDİ KAPALI. `allow_input=True` verilmeden tek bir olay bile
gonderilmiyor: ekrani okur, ne yapacagini soyler, dokunmaz.
"""

from __future__ import annotations

import re
import hashlib
import os
import shutil
import subprocess
import time
from dataclasses import dataclass

from ..events import Act, ToolResult
from ..safety import Abort, Blocked, SafetyPolicy


def _var(ad: str) -> bool:
    return shutil.which(ad) is not None


@dataclass
class Arac:
    xdotool: bool
    ffmpeg: bool
    scrot: bool
    pil: bool

    @classmethod
    def tara(cls) -> "Arac":
        try:
            import PIL  # noqa: F401
            pil = True
        except ImportError:
            pil = False
        return cls(_var("xdotool"), _var("ffmpeg"), _var("scrot"), pil)

    @property
    def ekran_alinabilir(self) -> bool:
        return self.ffmpeg or self.scrot

    def eksikler(self, girdi_gerekli: bool) -> list[str]:
        e = []
        if girdi_gerekli and not self.xdotool:
            e.append("xdotool  (sudo apt install --no-install-recommends xdotool)")
        if not self.ekran_alinabilir:
            e.append("ffmpeg ya da scrot  (sudo apt install --no-install-recommends scrot)")
        return e


class X11Sandbox:
    """Gerçek X11 ekranı. `FakeSandbox` ile AYNI protokolü uyguluyor —
    döngü hangi ortamda çalıştığını bilmiyor."""

    name = "x11"

    def __init__(self, display: str | None = None,
                 policy: SafetyPolicy | None = None,
                 shrink: int = 1366, gozlemci=None, hud_width: int = 0):
        self.display = display or os.environ.get("DISPLAY", ":0")
        self.policy = policy or SafetyPolicy()
        self.shrink = shrink
        self.arac = Arac.tara()
        self.gozlemci = gozlemci        # canli bakis gostergesi (callable)
        # HUD paneli ekranin saginda duruyor. Yakalama alani onun SOLUNDA
        # kesiliyor — yoksa ajan kendi izleme panelini "arayuz" sanip ona
        # tiklamaya calisir ve panel o goruntuyu gosterdigi icin sonsuz ayna
        # olusur. Ayni zamanda gonderilen goruntuyu kucultuyor.
        self.hud_width = max(0, hud_width)
        self.width = self.height = 0
        self._son_goruntu: bytes | None = None
        self._son_hedef: tuple[int, int] | None = None

        eksik = self.arac.eksikler(self.policy.allow_input)
        if eksik:
            raise RuntimeError(
                "Gercek masaustu icin eksik arac:\n  " + "\n  ".join(eksik))

    # -- yasam dongusu -----------------------------------------------------

    @property
    def capture_w(self) -> int:
        """Yakalanacak genişlik — HUD panelinin solunda kesiliyor."""
        return max(320, self.width - self.hud_width)

    def start(self) -> None:
        self.width, self.height = self._ekran_boyutu()
        self._bildir("baslat", f"{self.display} · {self.width}x{self.height} · "
                               f"girdi {'ACIK' if self.policy.allow_input else 'KAPALI'}")

    def stop(self) -> None:
        self._bildir("bitti", self.policy.rapor())

    # -- dis arac cagrilari ------------------------------------------------

    def _x(self, *argv: str, timeout: float = 5.0) -> str:
        env = {**os.environ, "DISPLAY": self.display}
        r = subprocess.run(argv, capture_output=True, text=True,
                           timeout=timeout, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"{argv[0]} hata: {r.stderr.strip()[:120]}")
        return r.stdout.strip()

    def _ekran_boyutu(self) -> tuple[int, int]:
        try:
            out = self._x("xdotool", "getdisplaygeometry")
            w, h = out.split()
            return int(w), int(h)
        except Exception:
            return 1920, 1080

    def _imlec(self) -> tuple[int, int]:
        try:
            out = self._x("xdotool", "getmouselocation", "--shell")
            d = dict(l.split("=", 1) for l in out.splitlines() if "=" in l)
            return int(d.get("X", 0)), int(d.get("Y", 0))
        except Exception:
            return -1, -1

    def aktif_pencere(self) -> str:
        """Su an odakta olan pencerenin basligi — koruma katmani buna bakiyor."""
        try:
            wid = self._x("xdotool", "getactivewindow")
            return self._x("xdotool", "getwindowname", wid)
        except Exception:
            return ""

    def aktif_sinif(self) -> str:
        """Odaktaki pencerenin `WM_CLASS`'ı — uygulamanın gerçek kimliği.

        Başlık kullanıcı/uygulama tarafından değiştirilebilir; sınıf
        değiştirilemez. Terminal koruması buna dayanıyor.

        `xdotool getwindowclassname` GÜVENİLMEZ: bazı sürümlerde komut hiç yok
        ("Unknown command") ve burada sessizce `""` dönüyordu — yani sınıf
        kontrolü kapalıydı, koruma yalnız BAŞLIĞA düşmüştü. Ölçüldü. `xprop`
        her X11 kurulumunda var; önce o deneniyor.
        """
        try:
            wid = self._x("xdotool", "getactivewindow")
        except Exception:
            return ""
        for komut in (("xprop", "-id", wid, "WM_CLASS"),
                      ("xdotool", "getwindowclassname", wid)):
            try:
                cikti = self._x(*komut)
            except Exception:
                continue
            if not cikti:
                continue
            if "WM_CLASS" in cikti:
                return " ".join(re.findall(r'"([^"]*)"', cikti))
            return cikti.strip()
        return ""

    def pencere_listesi(self) -> list[str]:
        """Acik pencerelerin ADLARI. Model ekrani PIKSEL yerine bu metinden
        okuyabiliyor — masaustunun icerigi disari cikmadan."""
        try:
            ids = self._x("xdotool", "search", "--onlyvisible", "--name", ".").split()
        except Exception:
            return []
        adlar = []
        for wid in ids[:40]:
            try:
                ad = self._x("xdotool", "getwindowname", wid)
                gm = self._x("xdotool", "getwindowgeometry", "--shell", wid)
                d = dict(l.split("=", 1) for l in gm.splitlines() if "=" in l)
                if ad and int(d.get("WIDTH", 0)) > 80:
                    adlar.append(f"[{ad[:52]} @({d.get('X')},{d.get('Y')}) "
                                 f"{d.get('WIDTH')}x{d.get('HEIGHT')}]")
            except Exception:
                continue
        return adlar

    # -- ekran -------------------------------------------------------------

    def screenshot(self) -> bytes:
        """Ham PNG. `--send-pixels` verilmedikce MODELE GITMIYOR — yalnizca
        hash'i ve canli gosterge icin kullaniliyor."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            yol = f.name
        try:
            if self.arac.ffmpeg:
                subprocess.run(
                    ["ffmpeg", "-loglevel", "error", "-f", "x11grab",
                     "-video_size", f"{self.capture_w}x{self.height}",
                     "-i", f"{self.display}+0,0", "-frames:v", "1", "-y", yol],
                    capture_output=True, timeout=20, check=True)
            else:
                self._x("scrot", "-o", yol, timeout=20)
            with open(yol, "rb") as fh:
                ham = fh.read()
        finally:
            try:
                os.unlink(yol)
            except OSError:
                pass
        self._son_goruntu = ham
        return ham

    def frame(self) -> tuple[bytes, float]:
        """VLM'e gidecek küçültülmüş ekran görüntüsü + ölçek çarpanı.

        Neden küçültüyoruz: 1920x1080 bir PNG ~2500 görsel token eder ve HER
        ADIMDA yeniden gönderilir. 1366'ya indirmek token'ı kabaca yarıya
        düşürüyor, GUI öğeleri hâlâ okunabiliyor (Anthropic'in önerdiği aralık).

        Dönen ölçek, modelin verdiği koordinatı gerçek ekrana çevirmek için:
            gercek_x = model_x * olcek
        """
        ham = self.screenshot()
        if not self.arac.pil or self.capture_w <= self.shrink:
            return ham, 1.0
        import io

        from PIL import Image
        im = Image.open(io.BytesIO(ham)).convert("RGB")
        # Ölçek YAKALANAN genişliğe göre — HUD kesildiği için ekran genişliği
        # değil. Karıştırılırsa her tıklama sağa kayar.
        olcek = self.capture_w / self.shrink
        im = im.resize((self.shrink, max(1, int(im.height / olcek))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "PNG", optimize=True)
        return buf.getvalue(), olcek

    def screen_hash(self) -> str:
        """İlerleme sinyali. Ham piksel yerine KUCULTULMUS gri tonlama —
        yanip sonen imlec ve saat hash'i surekli degistirmesin."""
        try:
            ham = self.screenshot()
        except Exception:
            return "?"
        if self.arac.pil:
            try:
                import io
                from PIL import Image
                im = Image.open(io.BytesIO(ham)).convert("L").resize((64, 40))
                # 16 seviyeye indir: kucuk parlaklik oynamalari hash'i bozmasin
                q = bytes((p // 16) for p in im.tobytes())
                return hashlib.sha256(q).hexdigest()[:12]
            except Exception:
                pass
        return hashlib.sha256(ham).hexdigest()[:12]

    def describe(self) -> str:
        """Görüntünün YANINDA giden metin bağlamı.

        Asıl ekran `frame()` ile PNG olarak gidiyor — model gerçekten bakıyor.
        Bu metin görüntüden okunamayacak üç şeyi taşıyor: imlecin gerçek
        koordinatı, odaktaki pencerenin adı, ekranın piksel boyutu.

        UYARI — bu gerçek: `--model hf` seçiliyken ekran görüntün HuggingFace'e
        gidiyor. O anda ekranda ne varsa (terminal, parola yöneticisi, özel
        mesaj) dış bir servise yüklenmiş oluyor.
        """
        aktif = self.aktif_pencere()
        cx, cy = self._imlec()
        # Görüntü de gidiyor; metin yalnızca görüntüde OKUNAMAYAN şeyi taşıyor:
        # imlecin gerçek yeri, odaktaki pencere, ekran boyutu.
        return (f"aktif pencere: {aktif[:50]}  ·  imlec: ({cx},{cy})  ·  "
                f"ekran: {self.width}x{self.height}")

    # -- eylem yurutme -----------------------------------------------------

    def execute(self, act: Act, args: dict) -> ToolResult:
        cx, cy = self._imlec()
        try:
            self.policy.check_failsafe(cx, cy)
        except Abort:
            raise

        hedef = (int(args["x"]), int(args["y"])) if "x" in args and "y" in args else None
        self._son_hedef = hedef
        self._bildir("bakiyor", self._bakis_metni(act, args, hedef))

        # Okuma eylemleri — girdi izni gerekmez.
        if act in (Act.SCREENSHOT, Act.CURSOR_POSITION, Act.WAIT):
            if act is Act.WAIT:
                time.sleep(min(float(args.get("duration", 0.5)), 2.0))
            return self._sonuc()

        if not self.policy.allow_input:
            return self._sonuc(error="girdi KAPALI (salt okuma modu) — "
                                     "acmak icin --allow-input")

        try:
            self.policy.check_window(self.aktif_pencere(), self.aktif_sinif())
            return self._gonder(act, args, hedef)
        except Blocked as e:
            self.policy.note_blocked(e.kural, e.detay)
            self._bildir("engellendi", f"{e.kural} · {e.detay}")
            # Kosumu bitirmiyoruz: ajan neden reddedildigini GORSUN ve
            # baska bir yol denesin. Guardrail'ler zaten israri yakalar.
            return self._sonuc(error=f"ENGELLENDI [{e.kural}] {e.detay}")

    def _gonder(self, act: Act, args: dict, hedef) -> ToolResult:
        """Icerik kontrolleri ONCE, sayac SONRA.

        `charge()` bastayken ENGELLENEN eylemler de sert tavandan dusuyordu ve
        `rapor()` hicbir sey gonderilmedigi halde "5 gercek girdi" diyordu —
        olculdu. Guvenlik raporunun yanlis sayi vermesi, korumanin kendisi kadar
        onemli: "kac kez makineye dokunuldu" sorusunun cevabi dogru olmali.
        Sayac artik yalnizca GERCEKTEN gonderilen girdiyi sayiyor.
        """
        if act is Act.TYPE:
            metin = str(args.get("text", ""))
            self.policy.check_text(metin)
            self.policy.charge()
            self._x("xdotool", "type", "--delay", "40", "--", metin, timeout=30)
            return self._sonuc()

        if act in (Act.KEY, Act.HOLD_KEY):
            tus = str(args.get("text", ""))
            self.policy.check_key(tus)
            self.policy.charge()
            self._x("xdotool", "key", "--", tus.replace("+", "+"))
            return self._sonuc()

        if act is Act.MOUSE_MOVE and hedef:
            self.policy.charge()
            self._x("xdotool", "mousemove", str(hedef[0]), str(hedef[1]))
            return self._sonuc()

        if act is Act.SCROLL:
            self.policy.charge()
            yon = str(args.get("scroll_direction", "down")).lower()
            dugme = {"up": "4", "down": "5", "left": "6", "right": "7"}.get(yon, "5")
            if hedef:
                self._x("xdotool", "mousemove", str(hedef[0]), str(hedef[1]))
            for _ in range(int(args.get("scroll_amount", 3))):
                self._x("xdotool", "click", dugme)
            return self._sonuc()

        TIK = {Act.LEFT_CLICK: "1", Act.MIDDLE_CLICK: "2", Act.RIGHT_CLICK: "3",
               Act.DOUBLE_CLICK: "1", Act.TRIPLE_CLICK: "1"}
        if act in TIK:
            self.policy.charge()
            if hedef:
                # Once TASIN, bekle, sonra tikla — kullanici nereye
                # tiklanacagini GORSUN. Canli gostergenin yarisi bu.
                self._x("xdotool", "mousemove", str(hedef[0]), str(hedef[1]))
                time.sleep(self.policy.dwell_seconds)
                self.policy.check_failsafe(*self._imlec())
            tekrar = {Act.DOUBLE_CLICK: 2, Act.TRIPLE_CLICK: 3}.get(act, 1)
            self._x("xdotool", "click", "--repeat", str(tekrar), TIK[act])
            return self._sonuc()

        return self._sonuc(error=f"'{act.value}' bu backend'de uygulanmadi")

    # -- yardimcilar -------------------------------------------------------

    def _sonuc(self, error: str | None = None) -> ToolResult:
        return ToolResult(output=self.describe(), error=error,
                          screen_hash=self.screen_hash())

    def _bakis_metni(self, act: Act, args: dict, hedef) -> str:
        if hedef:
            return f"{act.value} → ({hedef[0]},{hedef[1]})"
        if act is Act.TYPE:
            return f'type → "{str(args.get("text",""))[:28]}"'
        if act in (Act.KEY, Act.HOLD_KEY):
            return f"key → {args.get('text','')}"
        return act.value

    def _bildir(self, faz: str, metin: str) -> None:
        if self.gozlemci:
            self.gozlemci(faz, metin, self)
