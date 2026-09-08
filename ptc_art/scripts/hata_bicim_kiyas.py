"""Hata geri bildirimi biçimleri — ÖLÇÜLMÜŞ karşılaştırma.

Aynı arızayı dokuz farklı mentaliteye sokup modele ne gittiğini ölçer.

## Neyi ölçüyor, neyi ÖLÇMÜYOR

ÖLÇÜYOR   — mesajın taşıdığı ONARIM SİNYALİ ve maliyeti: hata tipi, satır
            numarası, kaynak satırı, korunan stdout, kırpma bildirimi,
            talimat, tekrar tespiti, iç yığın sızıntısı, bayt.
ÖLÇMÜYOR  — düzeltme başarı oranı. O bir LLM değerlendirmesi ister
            (N senaryo x M biçim x k tekrar); bu betikte YOK. Buradaki
            "en iyi" = "bayt başına en çok onarım sinyali", "en çok
            hata düzeltir" DEĞİL.

## Biçimlendiriciler nereden geliyor

`bizim_yeni` GERÇEK koddur — `sandbox_image/entrypoint.py` içe aktarılıp
`_hata_metni` doğrudan çağrılıyor, kopya değil.

Diğer sekizi, `PTC_Error_Recovery_Piyasa_Arastirmasi.md`'de belgelenmiş
davranıştan YENİDEN KURULMUŞTUR — ilgili ürünün kaynak kodu değildir.
Her birinin başında hangi gözlemden çıktığı yazıyor.

Kullanım:
    python scripts/hata_bicim_kiyas.py            # tabloyu bas
    python scripts/hata_bicim_kiyas.py --md YOL   # markdown'a yaz
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import socket
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent

#: Pod'da kullanıcı kodu `/sandbox/code.py` olarak DİSKTE durur; traceback
#: kaynak satırını ancak dosya varsa taşır. Kıyas de aynı koşulda olmalı,
#: yoksa "kaynak satırı" ölçütü herkes için boş geçer.
#:
#: Her senaryo AYRI dosyaya yazılıyor: tek dosya kullanınca sonraki senaryo
#: öncekinin kaynağını eziyor ve `linecache` yanlış/boş satır döndürüyor.
#: Pod'da böyle bir sorun yok — orada çalıştırma başına bir dosya var.
_KOD_DIZINI = tempfile.mkdtemp(prefix="hata-kiyas-")


def _kod_yolu(ad: str) -> str:
    return str(Path(_KOD_DIZINI) / f"{ad}.py")


# ─────────────────────────────────────────────────────────────────────
# Gerçek entrypoint'i yükle — kopya değil, kodun kendisi ölçülüyor
# ─────────────────────────────────────────────────────────────────────
def _entrypoint_yukle():
    os.environ.setdefault("TOOL_GATEWAY_ENDPOINT", "http://kiyas-icin-kullanilmiyor")
    yol = KOK / "sandbox_image" / "entrypoint.py"
    spec = importlib.util.spec_from_file_location("_entrypoint_kiyas", yol)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ENTRY = _entrypoint_yukle()
assert ENTRY.CODE_PATH == "/sandbox/code.py", "entrypoint.CODE_PATH değişmiş"


# ─────────────────────────────────────────────────────────────────────
# Arıza üretimi — her biçimlendirici AYNI ham malzemeyi alır
# ─────────────────────────────────────────────────────────────────────
@dataclass
class Ariza:
    ad: str
    aciklama: str
    exc: BaseException | None
    stdout: str
    cikis_kodu: int
    sure: float
    onsoz: str | None = None
    #: Bu senaryonun kullanıcı kodu diskte nerede — pod'daki `/sandbox/code.py`
    yol: str = ""

    @property
    def tip(self) -> str:
        return type(self.exc).__name__ if self.exc else "—"

    @property
    def kullanici_kareleri(self) -> list[traceback.FrameSummary]:
        if not self.exc:
            return []
        return [k for k in traceback.extract_tb(self.exc.__traceback__)
                if k.filename == self.yol]

    @property
    def kaynak_satiri(self) -> str:
        kareler = self.kullanici_kareleri
        return (kareler[-1].line or "") if kareler else ""

    @property
    def tam_traceback(self) -> str:
        if not self.exc:
            return ""
        return "".join(traceback.format_exception(
            type(self.exc), self.exc, self.exc.__traceback__))


def kodu_kosttur(ad: str, aciklama: str, kod: str, *, sure: float = 0.4) -> Ariza:
    """Kodu entrypoint'in yaptığı gibi çalıştırır: aynı compile, aynı yakalama."""
    yol = _kod_yolu(ad)
    Path(yol).write_text(kod, encoding="utf-8")
    yakalanan = io.StringIO()
    exc: BaseException | None = None
    try:
        with contextlib.redirect_stdout(yakalanan):
            exec(compile(kod, yol, "exec"), {"__name__": "__main__"})  # noqa: S102
    except BaseException as e:  # noqa: BLE001 — arıza üretmek işin kendisi
        exc = e
    return Ariza(ad, aciklama, exc, yakalanan.getvalue(),
                 0 if exc is None else 1, sure, yol=yol)


# ─────────────────────────────────────────────────────────────────────
# Yardımcılar
# ─────────────────────────────────────────────────────────────────────
def kirp_orta(metin: str, azami: int, etiket: str = "output") -> str:
    if len(metin) <= azami:
        return metin
    atilan = len(metin) - azami
    return (metin[: azami // 2]
            + f"\n<{etiket} clipped: {atilan} characters elided>\n"
            + metin[-azami // 2:])


# ─────────────────────────────────────────────────────────────────────
# BİÇİMLENDİRİCİLER
# ─────────────────────────────────────────────────────────────────────
class Bicim:
    ad = "?"
    kaynak = "?"

    def sifirla(self) -> None:  # tekrar-durumu tutanlar ezer
        pass

    def __call__(self, a: Ariza) -> str:  # pragma: no cover - arayüz
        raise NotImplementedError


class BizimEski(Bicim):
    """2026-09-08 öncesi hâlimiz. Kayıtta duran metnin aynısı."""
    ad = "Bizim (eski)"
    kaynak = "kendi git geçmişimiz"

    def __call__(self, a: Ariza) -> str:
        return f"Hata: {a.exc}\nTahmini bir değer üretme."


class BizimYeni(Bicim):
    """GERÇEK KOD — sandbox_image/entrypoint.py::_hata_metni."""
    ad = "Bizim (şimdi)"
    kaynak = "sandbox_image/entrypoint.py"

    def __call__(self, a: Ariza) -> str:
        # Gerçek `_kullanici_izi` modül sabitine göre süzüyor; pod'da tek
        # dosya var, kıyasta senaryo başına bir tane — sabiti ona çeviriyoruz.
        ENTRY.CODE_PATH = a.yol
        try:
            return ENTRY._hata_metni(a.exc, a.stdout, a.onsoz)
        finally:
            ENTRY.CODE_PATH = "/sandbox/code.py"


class ClaudeCode(Bicim):
    """Gözlem: çıkış kodu + korunan stdout + TAM traceback, kırpma eşiği geniş."""
    ad = "Claude Code"
    kaynak = "gözlemlenmiş davranış"
    ESIK = 12_000

    def __call__(self, a: Ariza) -> str:
        govde = a.stdout + a.tam_traceback
        return f"Exit code {a.cikis_kodu}\n" + kirp_orta(govde, self.ESIK)


class Codex(Bicim):
    """Kaynak kod: format_exec_output_for_model — çıkış kodu, süre, satır
    sayısı, sonra ORTADAN kırpılmış çıktı."""
    ad = "Codex"
    kaynak = "format_exec_output_for_model"
    ESIK = 10_000

    def __call__(self, a: Ariza) -> str:
        govde = a.stdout + a.tam_traceback
        return (f"Exit code: {a.cikis_kodu}\n"
                f"Wall time: {a.sure:.1f} seconds\n"
                f"Total output lines: {govde.count(chr(10)) + 1}\n"
                f"Output:\n{kirp_orta(govde, self.ESIK)}")


class AnthropicAPI(Bicim):
    """Belgelenmiş yük: stdout/stderr/return_code + TİPLİ error_code."""
    ad = "Anthropic API"
    kaynak = "code execution tool dokümanı"

    def __call__(self, a: Ariza) -> str:
        return json.dumps({
            "stdout": a.stdout,
            "stderr": a.tam_traceback,
            "return_code": a.cikis_kodu,
            "error_code": None if a.exc is None else a.tip,
        }, ensure_ascii=False, indent=1)


class Smolagents(Bicim):
    """Kaynak kod: satırın KAYNAK METNİ + tip, traceback YOK, sonra loglar.
    MAX_LENGTH_TRUNCATE_CONTENT = 20000, ortadan."""
    ad = "smolagents"
    kaynak = "local_python_executor.py / memory.py"
    ESIK = 20_000

    def __call__(self, a: Ariza) -> str:
        if a.exc is None:
            return f"Execution logs:\n{kirp_orta(a.stdout, self.ESIK)}"
        return (f"Code execution failed at line '{a.kaynak_satiri.strip()}' "
                f"due to:\n{a.tip}: {a.exc}\n\n"
                f"Execution logs:\n{kirp_orta(a.stdout, self.ESIK)}")


class AutoGen(Bicim):
    """Gözlem: exitcode + tam çıktı, KIRPMA YOK."""
    ad = "AutoGen / AG2"
    kaynak = "gözlemlenmiş davranış"

    def __call__(self, a: Ariza) -> str:
        return (f"exitcode: {a.cikis_kodu} (execution failed)\n"
                f"Code output:\n{a.stdout}{a.tam_traceback}")


class SWEAgentVarsayilan(Bicim):
    """Varsayılan şablon: `observation[:max_observation_length]` + not.
    Yani BAŞTAN kırpıyor. Kırpma sessiz değil — kaç karakter atıldığını
    söylüyor VE ne yapılacağını öğretiyor."""
    ad = "SWE-agent (varsayılan)"
    kaynak = "agents.py + şablon"
    ESIK = 10_000
    OGUT = ("<response clipped><NOTE>Observations should not exceeded {esik} "
            "characters. {n} characters were elided. Please try a different "
            "command that produces less output or use head/tail/grep/redirect "
            "the output to a file. Do not use interactive pagers.</NOTE>")

    def _kirp(self, govde: str) -> str:
        return govde[: self.ESIK]

    def __call__(self, a: Ariza) -> str:
        govde = a.stdout + a.tam_traceback
        if not govde.strip():
            return ("Your command ran successfully and did not produce any "
                    "output.")
        if len(govde) <= self.ESIK:
            return govde
        n = len(govde) - self.ESIK
        return self._kirp(govde) + "\n" + self.OGUT.format(esik=self.ESIK, n=n)


class SWEAgentBashOnly(SWEAgentVarsayilan):
    """`config/bash_only.yaml`: baştan yarı + sondan yarı — smolagents gibi."""
    ad = "SWE-agent (bash_only)"
    kaynak = "config/bash_only.yaml"

    def _kirp(self, govde: str) -> str:
        return govde[: self.ESIK // 2] + govde[-self.ESIK // 2:]


class OpenHands(Bicim):
    """StuckDetector: aynı hatayı üst üste alınca ÖLDÜRMEZ, DÜRTER.
    action_error eşiği 3."""
    ad = "OpenHands"
    kaynak = "StuckDetector / get_action_error_nudge"
    ESIK_TEKRAR = 3
    DURTME = ("\n\n[NUDGE] You have been repeating the same failing action. "
              "Please try a different approach — do not repeat the same "
              "command again.")

    def __init__(self) -> None:
        self._son: str | None = None
        self._sayac = 0

    def sifirla(self) -> None:
        self._son, self._sayac = None, 0

    def __call__(self, a: Ariza) -> str:
        imza = f"{a.tip}:{a.exc}"
        self._sayac = self._sayac + 1 if imza == self._son else 1
        self._son = imza
        govde = (f"exitcode: {a.cikis_kodu}\n{a.stdout}{a.tam_traceback}")
        if self._sayac >= self.ESIK_TEKRAR:
            govde += self.DURTME
        return govde


BICIMLER: list[Bicim] = [
    BizimEski(), ClaudeCode(), Codex(), AnthropicAPI(), Smolagents(),
    AutoGen(), SWEAgentVarsayilan(), SWEAgentBashOnly(), OpenHands(),
    BizimYeni(),
]


# ─────────────────────────────────────────────────────────────────────
# SENARYOLAR
# ─────────────────────────────────────────────────────────────────────
KOD_KEYERROR = '''
print("veri yuklendi")
d = {"a": 1}
d["yok"]
'''

KOD_BUYUK_CIKTI = '''
for i in range(4000):
    print(f"satir {i}: " + "x" * 40)
raise ValueError("son satirda patladi")
'''

KOD_SYNTAX = "def f(:\n    pass\n"


def _ag_arizasi() -> Ariza:
    """Sandbox'ın interneti yok — gerçek hâli budur."""
    kod = ('print("baglaniyor")\n'
           'import socket\n'
           'socket.getaddrinfo("api.example.com", 443)\n')
    a = kodu_kosttur("ag", "ağ engeli — düzeltilemez", kod, sure=5.0)
    if a.exc is None:  # ortamda DNS varsa arızayı elle kur
        try:
            raise socket.gaierror(-3, "Temporary failure in name resolution")
        except socket.gaierror as e:
            a = Ariza("ag", "ağ engeli — düzeltilemez", e,
                      "baglaniyor\n", 1, 5.0, yol=_kod_yolu("ag"))
    return a


def senaryolar() -> list[Ariza]:
    liste = [
        kodu_kosttur("keyerror", "stdout'tan sonra KeyError", KOD_KEYERROR),
        kodu_kosttur("buyuk", "160 KB çıktı, sonra hata", KOD_BUYUK_CIKTI, sure=1.8),
        _ag_arizasi(),
    ]
    # SyntaxError: derleme anında patlar, KULLANICI KARESİ YOK
    syntax_yolu = _kod_yolu("syntax")
    Path(syntax_yolu).write_text(KOD_SYNTAX, encoding="utf-8")
    try:
        compile(KOD_SYNTAX, syntax_yolu, "exec")
    except SyntaxError as e:
        liste.append(Ariza("syntax", "SyntaxError — kullanıcı karesi yok",
                           e, "", 1, 0.1, yol=syntax_yolu))
    # Sonuç bildirilmedi: exception YOK, yalnızca önsöz + stdout
    liste.append(Ariza(
        "sonucsuz", "set_result çağrılmadı — exception yok",
        None, "hesap bitti\n", 1, 0.6, yol=_kod_yolu("sonucsuz"),
        onsoz=("Kod `set_result(...)` çağırmadı — nihai değer bu fonksiyonla "
               "bildirilir.")))
    return liste


# ─────────────────────────────────────────────────────────────────────
# PUANLAMA — metin üzerinde, her biçime AYNI ölçüt
# ─────────────────────────────────────────────────────────────────────
TALIMAT_SOZLUGU = (
    "tekrar", "düzelt", "dene", "yaz ", "uydurma", "kullan",
    "try", "use ", "instead", "please", "do not", "different",
)

KARE_DESENI = re.compile(r'File "([^"]+)", line \d+')
SATIR_DESENI = re.compile(r"line \d+")


@dataclass
class Puan:
    bicim: str
    senaryo: str
    bayt: int
    hata_tipi: bool
    satir_no: bool
    kaynak_satiri: bool
    stdout_korundu: bool
    cikis_kodu: bool
    kirpma_bildirimi: bool
    ne_yapmali: bool
    yabanci_kare: int
    alanlar: dict = field(default_factory=dict)


def _json_duzlestir(metin: str) -> str:
    """JSON yükünün İÇİNDEKİ metni de aranabilir yap.

    Anthropic biçimi traceback'i bir JSON alanında taşıyor; orada `d["yok"]`
    kaçışlanıp `d\\"yok\\"` oluyor ve düz metin araması bulamıyor. Model
    JSON'u ayrıştırıp gerçek satırı görüyor — ölçüm de öyle görmeli, yoksa
    JSON kullanan biçimi haksız yere cezalandırırız.
    """
    try:
        nesne = json.loads(metin)
    except Exception:  # noqa: BLE001 — JSON değilse dokunma
        return ""
    parcalar: list[str] = []

    def gez(x) -> None:
        if isinstance(x, str):
            parcalar.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                gez(v)
        elif isinstance(x, list):
            for v in x:
                gez(v)

    gez(nesne)
    return "\n".join(parcalar)


def _stdout_izi(a: Ariza) -> str:
    """stdout'un korunup korunmadığını anlamak için ilk satırı kullan."""
    ilk = a.stdout.strip().splitlines()
    return ilk[0] if ilk else ""


def puanla(bicim: Bicim, a: Ariza, metin: str) -> Puan:
    iz = _stdout_izi(a)
    # Sinyal aramaları JSON'un içine de bakmalı (bkz. `_json_duzlestir`).
    aranan = metin + "\n" + _json_duzlestir(metin)
    kareler = KARE_DESENI.findall(aranan)
    yabanci = sum(1 for k in kareler if k != a.yol)
    kirpildi_mi = (len(a.stdout) + len(a.tam_traceback)) > 10_000
    return Puan(
        bicim=bicim.ad,
        senaryo=a.ad,
        bayt=len(metin.encode("utf-8")),
        hata_tipi=(a.exc is None) or (a.tip in aranan),
        satir_no=(not a.kullanici_kareleri) or bool(SATIR_DESENI.search(aranan)),
        kaynak_satiri=(not a.kaynak_satiri)
                      or (a.kaynak_satiri.strip() in aranan),
        stdout_korundu=(not iz) or (iz in aranan),
        cikis_kodu=bool(re.search(r"[Ee]xit ?code|return_code|exitcode", metin)),
        kirpma_bildirimi=(not kirpildi_mi)
                         or bool(re.search(r"kırpıldı|clipped|elided|truncat",
                                           aranan, re.I)),
        ne_yapmali=any(s in aranan.lower() for s in TALIMAT_SOZLUGU),
        yabanci_kare=yabanci,
    )


OLCUTLER = ("hata_tipi", "satir_no", "kaynak_satiri", "stdout_korundu",
            "cikis_kodu", "kirpma_bildirimi", "ne_yapmali")


def kosttur() -> tuple[list[Puan], dict]:
    ariza_listesi = senaryolar()
    puanlar: list[Puan] = []
    for b in BICIMLER:
        b.sifirla()
        for a in ariza_listesi:
            puanlar.append(puanla(b, a, b(a)))
    # tekrar tespiti: aynı arızayı üst üste 3 kez ver, mesaj DEĞİŞİYOR mu
    tekrar: dict[str, bool] = {}
    key = ariza_listesi[0]
    for b in BICIMLER:
        b.sifirla()
        ciktilar = [b(key) for _ in range(3)]
        tekrar[b.ad] = ciktilar[-1] != ciktilar[0]
    return puanlar, tekrar


# ─────────────────────────────────────────────────────────────────────
# RAPOR
# ─────────────────────────────────────────────────────────────────────
BASLIKLAR = {
    "hata_tipi": "tip", "satir_no": "satır", "kaynak_satiri": "kaynak",
    "stdout_korundu": "stdout", "cikis_kodu": "çıkış", "kirpma_bildirimi": "kırpma",
    "ne_yapmali": "talimat",
}


ONSOZ = """# Hata geri bildirimi biçimleri — ölçülmüş karşılaştırma

`scripts/hata_bicim_kiyas.py` ile üretildi; bulgular
`tests/unit/test_hata_bicimleri.py` ile sabitlendi.

## Yöntem

Aynı beş arıza, on biçimlendiriciye AYNI ham malzemeyle veriliyor
(exception nesnesi, hata anına kadarki stdout, çıkış kodu, süre). Üretilen
metin yedi ölçütte taranıyor, ayrıca tekrar tespiti ve bayt ölçülüyor.

| Senaryo | Ne test ediyor |
|---|---|
| `keyerror` | stdout'tan sonra hata — çıktı korunuyor mu |
| `buyuk` | 160 KB çıktı, sonra hata — kırpma davranışı |
| `ag` | ağ engeli — düzeltilemez hata |
| `syntax` | derleme hatası — kullanıcı karesi YOK |
| `sonucsuz` | exception yok, yalnızca eksik sonuç |

**`Bizim (şimdi)` gerçek koddur** — `sandbox_image/entrypoint.py::_hata_metni`
içe aktarılıp doğrudan çağrılıyor. Diğer dokuzu
`PTC_Error_Recovery_Piyasa_Arastirmasi.md`'de belgelenmiş davranıştan
YENİDEN KURULMUŞTUR; ilgili ürünün kaynak kodu değildir.

## Neyi ölçmüyor — önce bu

Bu tablo **düzeltme başarı oranını ölçmez**. Onun için LLM değerlendirmesi
gerekir (N senaryo x M biçim x k tekrar); burada yok. Buradaki sıralama
"bayt başına taşınan onarım sinyali"dir.

Ayrıca ölçüt listesi bizim araştırmamızdan çıktı ve bizim biçimimizi de o
araştırma şekillendirdi — yani **sıralamada dairesellik var**. Dürüst okuma:
biçimimiz kendi ölçütlerinde iyi çıkıyor; asıl kıyaslanabilir sayı, aynı
sinyali kaç baytla taşıdığımız.

"""


def rapor(puanlar: list[Puan], tekrar: dict) -> str:
    """Ölçüt başına KAÇ SENARYODA sağlandığını verir.

    "hepsinde sağlanıyor mu" biçiminde bir hücre yanıltıcıydı: SWE-agent
    talimatı YALNIZCA kırpınca veriyor — beş senaryonun dördünde yok, bu
    bir kusur değil tasarım. n/5 bunu görünür kılıyor.
    """
    bicimler = [b.ad for b in BICIMLER]
    n = len({p.senaryo for p in puanlar})
    s: list[str] = []
    s.append(f"## Sinyal tablosu — {n} senaryonun kaçında sağlanıyor\n")
    s.append("| Biçim | " + " | ".join(BASLIKLAR[o] for o in OLCUTLER)
             + " | sinyal | tekrar | sızıntı | ort. bayt |")
    s.append("|" + "---|" * (len(OLCUTLER) + 5))
    sirali = []
    for ad in bicimler:
        kendi = [p for p in puanlar if p.bicim == ad]
        hucreler, toplam = [], 0
        for o in OLCUTLER:
            k = sum(bool(getattr(p, o)) for p in kendi)
            toplam += k
            hucreler.append(f"{k}/{n}" if k not in (0, n)
                            else ("✓" if k == n else "✗"))
        t = tekrar[ad]
        sizinti = max(p.yabanci_kare for p in kendi)
        ort = round(sum(p.bayt for p in kendi) / len(kendi))
        azami = len(OLCUTLER) * n
        s.append(f"| **{ad}** | " + " | ".join(hucreler)
                 + f" | **{toplam}/{azami}** | {'✓' if t else '✗'}"
                 + f" | {sizinti} | {ort:,} |".replace(",", " "))
        sirali.append((toplam, t, -sizinti, -ort, ad))
    s.append("")
    s.append("`sinyal` = 7 ölçüt x " + str(n) + " senaryo. "
             "`tekrar` = aynı arıza üst üste gelince mesaj değişiyor mu. "
             "`sızıntı` = mesajda görünen KULLANICI DIŞI traceback karesi "
             "(0 iyi; sayının kendisi harness'a bağlı, 0-mı-değil-mi anlamlı).\n")

    s.append("## Senaryo başına bayt\n")
    sira = []
    for p in puanlar:
        if p.senaryo not in sira:
            sira.append(p.senaryo)
    s.append("| Biçim | " + " | ".join(sira) + " |")
    s.append("|" + "---|" * (len(sira) + 1))
    for ad in bicimler:
        kendi = {p.senaryo: p.bayt for p in puanlar if p.bicim == ad}
        s.append(f"| {ad} | " + " | ".join(
            f"{kendi[x]:,}".replace(",", " ") for x in sira) + " |")
    s.append("")

    s.append("## Sıralama — sinyal, sonra tekrar, sonra sızıntı, sonra bayt\n")
    for i, (toplam, t, negsiz, negort, ad) in enumerate(
            sorted(sirali, reverse=True), 1):
        s.append(f"{i}. **{ad}** — sinyal {toplam}/{len(OLCUTLER) * n}, "
                 f"tekrar {'var' if t else 'yok'}, sızıntı {-negsiz}, "
                 f"ort. {-negort:,} bayt".replace(",", " "))
    return "\n".join(s)


def ornekler(hangi: str = "keyerror") -> str:
    a = next(x for x in senaryolar() if x.ad == hangi)
    s = [f"## Modele fiilen giden metin — senaryo `{hangi}`\n"]
    for b in BICIMLER:
        b.sifirla()
        m = b(a)
        kisa = m if len(m) <= 900 else m[:900] + "\n…"
        s.append(f"### {b.ad}  ·  {len(m.encode('utf-8'))} bayt\n")
        s.append("```\n" + kisa + "\n```\n")
    return "\n".join(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", help="markdown çıktısını bu dosyaya yaz")
    ap.add_argument("--ornek", default="keyerror",
                    help="örnek metinlerin gösterileceği senaryo")
    n = ap.parse_args()
    puanlar, tekrar = kosttur()
    metin = ONSOZ + rapor(puanlar, tekrar) + "\n\n" + ornekler(n.ornek)
    if n.md:
        Path(n.md).write_text(metin, encoding="utf-8")
        print(f"yazıldı: {n.md}")
    else:
        print(metin)


if __name__ == "__main__":
    main()
