"""
Göreve özel doğrulayıcı — "bitirdim" iddiasını gerçeğe karşı sınar.

`cua_lab`'daki zihniyetler sentetik form senaryosuna göre yazılmıştı:
`verify-gate` ekranda `"gonderildi"` arıyordu, `telemetry-repair`
`{"type","left_click"}` çağrılarını bekliyordu. Bir terminal görevinde
ikisi de hiçbir zaman sağlanmıyor ve HER iddia reddediliyordu — yapısal
yanlış pozitif. Ölçüldü: `altan.txt oluştur` görevi doğru bittiği hâlde
üç zihniyet uyarı verdi.

Buradaki doğrulayıcı GÖREV METNİNDEN ne aranacağını çıkarıyor:

  dosya adı geçiyorsa   → dosya gerçekten var mı, içi dolu mu
  adres geçiyorsa       → o adrese gidildi mi
  hiçbiri yoksa         → en az bir araç çağrısı GERÇEKTEN başarılı oldu mu

Son madde önemli: kanıt üretmeden "bitirdim" demek her durumda şüpheli.
"""

from __future__ import annotations

import os
import re

# URL'ler ONCE ayiklaniyor: `file:///home/.../test.html` icindeki yol bir
# OLUSTURULACAK dosya adi degil, GIDILECEK bir adres. Soyulmadan birakilinca
# DOSYA deseni `home/altan/.../test.html` yakalayip calisma dizininde ariyor,
# bulamiyor ve dogru biten bir kosumu reddediyordu — olculdu.
URL = re.compile(r"\b(?:file|https?)://[^\s\"'<>]+")

DOSYA = re.compile(r"\b([\w./-]+\.(?:txt|md|json|csv|py|log|html|yml|yaml|sh))\b")

# "altan diye folder", "altan adinda klasor", "klasor: altan", "rapor dizini"
_KLS = r"(?:klas[oö]r|folder|dizin|directory)"
# Ilk eslesen desen kazanir. Ikinci desen ACIK bir isaretci (`:` `=` `adi`)
# istiyor: isaretcisiz birakilinca "folder olustur" ifadesinden `olustur`
# kelimesini klasor adi sanip her kosumu reddetti — olculdu.
DIZIN = [
    re.compile(rf"\b([\w.-]+)\s+(?:diye|adinda|adında|adli|adlı|isimli|isminde)"
               rf"\s*(?:bir\s+)?{_KLS}\b", re.I),
    re.compile(rf"\b{_KLS}\s*(?:adi|adı|ismi)\s*[:=]?\s*['\"]?([\w.-]+)['\"]?", re.I),
    re.compile(rf"\b{_KLS}\s*[:=]\s*['\"]?([\w.-]+)['\"]?", re.I),
    re.compile(rf"\b([\w.-]+)\s+{_KLS}[uüiı]?n[uü]?\b", re.I),   # "rapor dizinini"
]

# Gorevde gecen KONUM ipuclari. Kisitli kabuk tek bir dizine kilitli; gorev
# baska bir yer soyluyorsa is ORAYA yapilamaz. Ajan bunu bilmeden `mkdir altan`
# calistirip "masaustunde olusturuldu" dedi — olculdu, klasor calisma
# dizinindeydi ve HICBIR zihniyet yakalamadi.
KONUM = [
    (r"masa\s*[uü]st[uü]|desktop", "Desktop"),
    (r"indirilenler|downloads", "Downloads"),
    (r"belgeler|documents", "Documents"),
    (r"resimler|pictures", "Pictures"),
]
ADRES = re.compile(r"\b((?:https?://)?[\w-]+\.(?:com|org|net|io|dev|ai|edu|gov|tr)"
                   r"(?:/[\w./?=&%-]*)?)\b")


def yap(terminal, browser, gorev: str):
    """Göreve bakıp bir doğrulayıcı üretir: `(ctx, iddia) -> (bool, sebep)`."""
    urller = URL.findall(gorev)
    govde = URL.sub(" ", gorev)                 # dosya adi ararken URL'leri gizle
    dosyalar = DOSYA.findall(govde)
    adresler = urller + [a for a in ADRES.findall(govde) if "." in a]
    _ATLA = {"bir", "bu", "o", "yeni", "adinda", "adında", "diye", "isimli"}
    dizinler: list[str] = []
    for d in DIZIN:
        aday = [m for m in d.findall(govde) if m.lower() not in _ATLA]
        if aday:
            dizinler = aday       # ILK eslesen desen kazanir
            break
    # Gorev bir konum soyluyor mu, kabuk oraya erisebiliyor mu?
    kok_ger = os.path.realpath(terminal.kok)
    konum_hata = None
    for desen, klasor in KONUM:
        if re.search(desen, govde, re.I):
            hedef = os.path.realpath(os.path.expanduser(f"~/{klasor}"))
            # TAM esitlik. `startswith` yetmiyor: bu depo zaten ~/Desktop
            # altinda duruyor, calisma dizini 6 seviye derinde ve kontrol
            # sessizce geciyordu. "Masaustunde olustur" DOGRUDAN orada demek.
            if kok_ger != hedef:
                konum_hata = (f"gorev '{klasor}' diyor ama kisitli kabuk "
                              f"{kok_ger} dizinine kilitli — is ORAYA yapilamadi")
            break

    def dogrula(ctx, iddia: str) -> tuple[bool, str]:
        # 0) Konum uyusmazligi: HER SEYDEN once. Dosya dogru olusmus olabilir
        #    ama YANLIS YERDE; iddia "masaustunde" diyorsa iddia yanlistir.
        if konum_hata:
            return False, konum_hata

        # 1) Görevde dosya adı geçiyorsa: gerçekten var mı?
        for ad in dosyalar:
            yol = os.path.join(terminal.kok, ad)
            if not os.path.isfile(yol):
                return False, f"'{ad}' calisma dizininde YOK"
            if os.path.getsize(yol) == 0:
                return False, f"'{ad}' var ama BOS"
        if dosyalar:
            return True, (f"{len(dosyalar)} dosya dogrulandi: "
                          + ", ".join(f"{a} ({os.path.getsize(os.path.join(terminal.kok, a))}B)"
                                      for a in dosyalar))

        # 1b) Görevde klasör adı geçiyorsa: dizin gerçekten oluştu mu?
        for ad in dizinler:
            yol = os.path.join(terminal.kok, ad)
            if not os.path.isdir(yol):
                return False, f"'{ad}' klasoru calisma dizininde YOK"
        if dizinler:
            return True, f"{len(dizinler)} klasor dogrulandi: " + ", ".join(dizinler)

        # 2) Görevde adres geçiyorsa: oraya gidildi mi?
        if adresler and browser is not None:
            try:
                simdiki = browser.dom()["url"].lower()
            except Exception:
                simdiki = ""
            hedef = adresler[0]
            if hedef.startswith("file://"):
                # file:// icin alan adi yok — tam yolla karsilastir.
                alan = hedef.split("://", 1)[1]
            else:
                alan = re.sub(r"^https?://", "", hedef).split("/")[0]
            if alan.lower() not in simdiki:
                return False, f"'{alan}' adresine gidilmedi (su an: {simdiki[:50]})"
            return True, f"'{alan}' adresi acik"

        # 3) Genel kural: kanıt üretmeden bitirme.
        basarili = sum(1 for e in ctx.events
                       if e.kind.value == "observation" and e.payload.get("output"))
        if basarili == 0:
            return False, "hicbir arac cagrisi kanit uretmedi — kanitsiz bitirme"
        return True, f"{basarili} arac cagrisi kanit uretti"

    return dogrula


def gerekli_araclar(gorev: str) -> set:
    """`required_coverage` için görevin gerektirdiği asgari araç kümesi.

    SIRA ÖNEMLİ. Önce net kanıt (dosya adı / adres), sonra anahtar kelime.
    Çıplak fiil taraması yanlış pozitif üretiyor: *"cevabında kodu **yaz**"*
    bir dosya yazma isteği değil ama "yaz" anahtarı dosya dalını tetikliyordu
    ve tarayıcı görevini `terminal cagrilmadi` diye reddettirdi — ölçüldü.
    Fiiller artık yalnız bir DOSYA/DİZİN sözcüğüyle birlikte sayılıyor.
    """
    govde = URL.sub(" ", gorev)
    g = govde.lower()

    # 1) Görevde açık bir dosya adı var → dosya işi.
    if DOSYA.search(govde):
        return {"terminal.yaz", "terminal"}
    # 2) Açık bir adres var → tarayıcı işi.
    if URL.search(gorev) or ADRES.search(govde):
        return {"browser.goto"}
    # 3) Kelimeye düşüyoruz: fiil TEK BAŞINA yetmez, nesnesi de geçmeli.
    nesne = any(k in g for k in ("dosya", "dizin", "klasor", "klasör",
                                 "folder", "directory", "file"))
    fiil = any(k in g for k in ("olustur", "oluştur", "yaz", "kaydet", "ekle"))
    if nesne and fiil:
        return {"terminal.yaz", "terminal"}
    if any(k in g for k in ("site", "sayfa", "tarayıcı", "tarayici",
                            "web", "link", "baglanti", "bağlantı")):
        return {"browser.goto"}
    return set()          # cikarim yoksa kural KAPALI — tahminle reddetme


def min_kanit(gorev: str) -> int:
    """Kaç gözlem 'yeterli kanıt' sayılsın. Tek komutluk bir iş ile çok
    adımlı bir arama aynı eşiği paylaşamaz."""
    return 1 if DOSYA.search(URL.sub(" ", gorev)) else 2
