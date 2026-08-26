"""
Sahte masaüstü — ÖLÇEK KAYMASI senaryosu.

NEDEN VAR: 25 Ağustos'ta gerçek bir koşumda şu oldu — ajan `desktop.click`
çağrısını 14 adım boyunca aynı iki noktaya gönderdi ve hiçbir şey açılmadı.
Sebep modelde değildi. Model 1280x720 küçültülmüş kareye bakıp doğru yere
nişan alıyordu; harness o koordinatı 1920x1080 ekrana ÖLÇEKLEMEDEN basıyordu.
Her tık 1.5 kat yukarı-sola, boş masaüstüne düşüyordu.

Bu senaryonun sunumdaki değeri, yakalanan bir döngü olması değil:

  * Ajan MANTIKLI davranıyor. Gördüğü karede düğme orada; tıklıyor; bir şey
    olmuyor; tekrar deniyor. Bunun adı "modelin aptallığı" değil.
  * Hata mesajı YOK, çökme YOK, ekran değişmiyor. Klasik izleme kör.
  * Bütün döngü dedektörleri tetikleniyor ve hepsi doğru söylüyor:
    "aynı çağrıyı tekrar ediyorsun". HİÇBİRİ "koordinatların yanlış uzayda"
    diyemiyor. Dedektör döngüyü görür, SEBEBİNİ göremez.

`olcek_uygula` tek bir boolean. `False` → döngü. `True` → iki adımda biter.
Aynı model, aynı ekran, aynı görev. Kontrollü tek değişken.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field


@dataclass
class Dugme:
    ad: str
    x: int          # GERÇEK ekran uzayında merkez
    y: int
    w: int = 200
    h: int = 56
    acar: str = ""  # tıklanınca hangi pencereyi açar

    def icinde(self, x: int, y: int) -> bool:
        return (abs(x - self.x) <= self.w // 2 and
                abs(y - self.y) <= self.h // 2)


@dataclass
class SahteMasaustu:
    """Sentetik 1920x1080 masaüstü; modele küçültülmüş kare gider.

    Gerçek `Desktop` sınıfının ajan tarafından kullanılan yüzeyini taklit
    ediyor. Amacı gerçek masaüstünü değiştirmek değil — ÖLÇEK KAYMASINI
    gerçek donanım ve gerçek risk olmadan tekrarlanabilir kılmak.
    """

    olcek_uygula: bool = False
    genislik: int = 1920
    yukseklik: int = 1080
    shrink: int = 1280
    allow_input: bool = True

    def __post_init__(self):
        self.dugmeler = [
            Dugme("Metin Duzenleyici", 300, 1020, acar="Metin Duzenleyici"),
            Dugme("Tarayici", 540, 1020, acar="Tarayici"),
            Dugme("Dosyalar", 780, 1020, acar="Dosyalar"),
        ]
        self.acik: str | None = None
        self.metin = ""
        self.iskarta = 0            # boşluğa giden tık sayısı
        self.engellenen: list = []
        self._tiklar: list[tuple[int, int]] = []

    # -- ölçek -------------------------------------------------------------

    @property
    def olcek(self) -> float:
        return self.genislik / self.shrink          # 1.5

    def _gercek(self, x: int, y: int) -> tuple[int, int]:
        """Model koordinatı → gerçek ekran koordinatı.

        HATANIN TAM YERİ BURASI. `olcek_uygula=False` iken model uzayındaki
        sayı doğrudan gerçek ekrana basılıyor ve hedefin 1/1.5 katına düşüyor.
        """
        if self.olcek_uygula:
            return int(round(x * self.olcek)), int(round(y * self.olcek))
        return int(x), int(y)

    # -- ajanın kullandığı yüzey -------------------------------------------

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def durum(self) -> str:
        mw, mh = self.shrink, int(self.yukseklik / self.olcek)
        aktif = self.acik or "(masaustu)"
        return (f"aktif pencere: {aktif}  ·  "
                f"gorunen alan: {mw}x{mh} (koordinatlari BU uzayda ver)")

    def screenshot(self) -> bytes:
        return self._ciz()

    def click(self, x: int, y: int) -> str:
        gx, gy = self._gercek(int(x), int(y))
        self._tiklar.append((gx, gy))
        for d in self.dugmeler:
            if d.icinde(gx, gy):
                self.acik = d.acar
                return f"left_click({{'x': {x}, 'y': {y}}}) tamam"
        self.iskarta += 1
        # Gercek masaustu da boyle davraniyor: bosluga tiklamak HATA DEGIL.
        # Hata donsaydi ajan bunu anlar ve dongu hic olusmazdi — sinsiligin
        # kaynagi tam olarak bu sessizlik.
        return f"left_click({{'x': {x}, 'y': {y}}}) tamam"

    def type(self, metin: str) -> str:
        if self.acik != "Metin Duzenleyici":
            return "odaklanmis bir metin alani yok"
        self.metin += str(metin)
        return f"yazildi: {metin[:40]}"

    def key(self, tus: str) -> str:
        return f"tus: {tus}"

    def scroll(self, dy: int = 3) -> str:
        return f"kaydirildi: {dy}"

    def pencereler(self) -> str:
        if not self.acik:
            return "  (acik pencere yok — masaustundeki simgelerden birine tikla)"
        return f"  [1] {self.acik}  (sahte)  900x600 @(300,200)"

    def odakla(self, ad: str) -> str:
        return f"odaklanamadi: '{ad}' bulunamadi" if self.acik != ad else f"odaklandi: {ad}"

    def surukle(self, *a) -> str:
        return "bu senaryoda surukleme yok"

    def pencere_goruntusu(self, ad: str):
        return self._ciz(), ad

    def durum_hash(self) -> str:
        blob = f"{self.acik}|{self.metin}|{len(self.dugmeler)}"
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def rapor(self) -> str:
        n = len(self._tiklar)
        if not n:
            return "0 tik"
        isabet = n - self.iskarta
        satir = (f"{n} tik · {isabet} isabet · {self.iskarta} BOSLUGA")
        if self.iskarta and not self.olcek_uygula:
            hedef = self.dugmeler[0]
            son = self._tiklar[-1]
            satir += (f"\n      olcek uygulanmadi: son tik ({son[0]},{son[1]}) — "
                      f"'{hedef.ad}' dugmesi ({hedef.x},{hedef.y})")
            satir += (f"\n      model dogru nisan aldi; harness {self.olcek:g} "
                      f"kat kaydirdi")
        return satir

    # -- çizim -------------------------------------------------------------

    def _ciz(self) -> bytes:
        """Modele giden kare — GERÇEK uzayda çizilip `shrink`'e küçültülüyor.

        Gerçek boru hattının aynısı: `x11.py` de tam ekranı yakalayıp
        `shrink` genişliğine indiriyor. Kaymanın doğduğu yer bu küçültme.
        """
        from PIL import Image, ImageDraw
        im = Image.new("RGB", (self.genislik, self.yukseklik), (32, 34, 44))
        d = ImageDraw.Draw(im)

        # duvar kağıdı çizgileri — ekranın boş olduğu görünsün
        for i in range(0, self.yukseklik, 90):
            d.line([(0, i), (self.genislik, i)], fill=(38, 41, 52), width=1)

        if self.acik:
            d.rectangle([300, 180, 1500, 900], fill=(248, 248, 250),
                        outline=(120, 120, 140), width=3)
            d.rectangle([300, 180, 1500, 240], fill=(210, 212, 226))
            d.text((326, 200), self.acik, fill=(20, 20, 30))
            if self.acik == "Metin Duzenleyici":
                d.text((330, 280), self.metin or "(bos)", fill=(40, 40, 55))

        # görev çubuğu + simgeler
        d.rectangle([0, 980, self.genislik, self.yukseklik], fill=(20, 21, 28))
        for b in self.dugmeler:
            x0, y0 = b.x - b.w // 2, b.y - b.h // 2
            d.rectangle([x0, y0, x0 + b.w, y0 + b.h], fill=(86, 92, 120),
                        outline=(150, 156, 190), width=2)
            d.text((x0 + 14, y0 + 20), b.ad[:18], fill=(240, 240, 248))

        im = im.resize((self.shrink, int(self.yukseklik / self.olcek)))
        buf = io.BytesIO()
        im.save(buf, "PNG", optimize=True)
        return buf.getvalue()
