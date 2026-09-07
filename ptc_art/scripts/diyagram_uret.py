"""Sunum diyagramları — Excalidraw tarzı, elle çizilmiş görünüm.

`python scripts/diyagram_uret.py` → `docs/diyagram/*.png`

## Neden kendi çizerimiz var

Sistemde `roughjs`, `cairosvg`, `rsvg-convert`, `inkscape` yok; yalnızca
Pillow var. Excalidraw'ın görünümü de zaten iki basit şeyden ibaret:
**çizgiler titrek** ve **her kenar iki kez** çiziliyor. İkisi de Pillow ile
doğrudan yapılabiliyor.

## Neden el yazısı fontu YOK

Excalidraw'ın imzası Virgil; sistemde karşılığı yalnızca Comic Sans. Teknik
bir mimari sunumunda o font güveni düşürür. El çizimi hissini çizgilerden
alıyoruz, tipografiden değil — okunurluk bozulmasın.

Kenar yumuşatma için 2× ölçekte çizilip küçültülüyor (Pillow'un çizgileri
aliaslı).
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

KOK = Path(__file__).resolve().parents[1]
CIKTI = KOK / "docs" / "diyagram"
OLCEK = 2  # süper örnekleme

# Excalidraw paleti — mürekkep koyu gri, dolgular pastel
INK = (30, 30, 30)
SOLUK = (110, 110, 118)
MAVI = (25, 113, 194)
YESIL = (47, 158, 68)
KIRMIZI = (224, 49, 49)
TURUNCU = (240, 140, 0)
MOR = (110, 84, 194)

D_MAVI = (165, 216, 255)
D_YESIL = (178, 242, 187)
D_SARI = (255, 236, 153)
D_KIRMIZI = (255, 201, 201)
D_MOR = (216, 208, 255)
D_GRI = (233, 236, 239)

_FONTLAR = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
_FONTLAR_B = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]
_MONO = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]


def _font(boyut: int, kalin: bool = False, mono: bool = False):
    adaylar = _MONO if mono else (_FONTLAR_B if kalin else _FONTLAR)
    for y in adaylar:
        if Path(y).exists():
            return ImageFont.truetype(y, boyut * OLCEK)
    return ImageFont.load_default()


# ── titrek çizim ilkelleri ────────────────────────────────────────────────


def _s(v):
    """Ölçekle."""
    return v * OLCEK


def _sap(k=2.0):
    return random.uniform(-k, k) * OLCEK


def _bezier(p0, p1, p2, adim=18):
    pts = []
    for i in range(adim + 1):
        t = i / adim
        u = 1 - t
        pts.append((u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
                    u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]))
    return pts


def cizgi(d, p0, p1, renk=INK, kalinlik=2, gecis=2, sapma=1.8):
    """Bir kenarı iki kez, hafif kavisli çizer — Excalidraw'ın imzası."""
    for _ in range(gecis):
        a = (p0[0] + _sap(sapma), p0[1] + _sap(sapma))
        b = (p1[0] + _sap(sapma), p1[1] + _sap(sapma))
        orta = ((a[0] + b[0]) / 2 + _sap(sapma * 1.4),
                (a[1] + b[1]) / 2 + _sap(sapma * 1.4))
        d.line(_bezier(a, orta, b), fill=renk, width=int(kalinlik * OLCEK), joint="curve")


def _kose_yolu(x0, y0, x1, y1, r):
    """Yuvarlatılmış dikdörtgenin nokta listesi."""
    pts = []
    kose = [((x1 - r, y0 + r), 270), ((x1 - r, y1 - r), 0),
            ((x0 + r, y1 - r), 90), ((x0 + r, y0 + r), 180)]
    for (cx, cy), bas in kose:
        for i in range(9):
            a = math.radians(bas + i * 90 / 8)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def kutu(d, kutu_alani, dolgu=None, renk=INK, r=14, kalinlik=2, sapma=1.6):
    x0, y0, x1, y1 = kutu_alani
    R = _s(r)
    if dolgu:
        d.rounded_rectangle([x0, y0, x1, y1], radius=R, fill=dolgu)
    yol = _kose_yolu(x0, y0, x1, y1, R)
    for _ in range(2):
        titrek = [(px + _sap(sapma), py + _sap(sapma)) for px, py in yol]
        d.line(titrek + [titrek[0]], fill=renk, width=int(kalinlik * OLCEK), joint="curve")


def ok(d, p0, p1, renk=INK, kalinlik=2, bas=13, kesik=False):
    if kesik:
        n = 14
        for i in range(0, n, 2):
            a = (p0[0] + (p1[0] - p0[0]) * i / n, p0[1] + (p1[1] - p0[1]) * i / n)
            b = (p0[0] + (p1[0] - p0[0]) * (i + 1) / n, p0[1] + (p1[1] - p0[1]) * (i + 1) / n)
            cizgi(d, a, b, renk, kalinlik, gecis=1, sapma=1.0)
    else:
        cizgi(d, p0, p1, renk, kalinlik)
    aci = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    L = _s(bas)
    # Barbs uçtan GERİYE doğru ±0.42 rad. (Önce ±2.5 rad yazılmıştı ve uç
    # ileriye açılıyordu — oklar ters yöne bakıyor gibi duruyordu.)
    for yon in (0.42, -0.42):
        u = (p1[0] - L * math.cos(aci + yon), p1[1] - L * math.sin(aci + yon))
        cizgi(d, p1, u, renk, kalinlik, gecis=1, sapma=1.0)


def yazi(d, xy, metin, boyut=17, renk=INK, kalin=False, mono=False,
         hiza="left", dikey="top"):
    f = _font(boyut, kalin, mono)
    x, y = xy
    if hiza == "center" or dikey == "middle":
        kut = d.textbbox((0, 0), metin, font=f)
        if hiza == "center":
            x -= (kut[2] - kut[0]) / 2
        if dikey == "middle":
            y -= (kut[3] - kut[1]) / 2 + kut[1]
    d.text((x, y), metin, font=f, fill=renk)


def _tuval(g, y, zemin=(255, 255, 255)):
    im = Image.new("RGB", (_s(g), _s(y)), zemin)
    return im, ImageDraw.Draw(im)


def _kaydet(im, ad):
    CIKTI.mkdir(parents=True, exist_ok=True)
    im = im.resize((im.width // OLCEK, im.height // OLCEK), Image.LANCZOS)
    yol = CIKTI / f"{ad}.png"
    im.save(yol, "PNG", optimize=True)
    print(f"  {yol.relative_to(KOK)}  {im.width}×{im.height}")


# ── D1 — adımın yaşam döngüsü ─────────────────────────────────────────────


def d1_yasam_dongusu():
    random.seed(7)
    G, Y = 1000, 430
    im, d = _tuval(G, Y)

    # pod çerçevesi
    kutu(d, (_s(40), _s(70), _s(600), _s(370)), dolgu=(250, 250, 252), renk=SOLUK, r=18)
    yazi(d, (_s(56), _s(84)), "Pod  ·  her çalıştırmada yeni, ~3 sn sonra yok", 14, SOLUK)

    # sidecar
    kutu(d, (_s(70), _s(120), _s(300), _s(230)), dolgu=D_MAVI, renk=MAVI, r=14)
    yazi(d, (_s(185), _s(148)), "artifact-sidecar", 18, INK, kalin=True, hiza="center")
    yazi(d, (_s(185), _s(176)), "jeton BURADA", 14, MAVI, hiza="center")
    yazi(d, (_s(185), _s(198)), "yerleştirir · süpürür", 13, SOLUK, hiza="center")

    # sandbox
    kutu(d, (_s(340), _s(120), _s(570), _s(230)), dolgu=D_KIRMIZI, renk=KIRMIZI, r=14)
    yazi(d, (_s(455), _s(148)), "sandbox", 18, INK, kalin=True, hiza="center")
    yazi(d, (_s(455), _s(176)), "LLM'in kodu", 14, KIRMIZI, hiza="center")
    yazi(d, (_s(455), _s(198)), "jeton YOK · ağ kapalı", 13, SOLUK, hiza="center")

    # /output
    kutu(d, (_s(70), _s(265), _s(570), _s(340)), dolgu=D_SARI, renk=TURUNCU, r=14)
    yazi(d, (_s(320), _s(285)), "/output   (emptyDir — pod'la birlikte ölür)", 16,
         INK, kalin=True, hiza="center")
    yazi(d, (_s(320), _s(312)), "ikisi de mount ediyor", 13, SOLUK, hiza="center")

    # Yönler ÖNEMLİ: sidecar /output'a koyar, sandbox oraya yazar/okur.
    ok(d, (_s(160), _s(232)), (_s(160), _s(263)), MAVI)
    yazi(d, (_s(172), _s(238)), "yerleştirir", 13, MAVI)
    ok(d, (_s(455), _s(232)), (_s(455), _s(263)), KIRMIZI)
    yazi(d, (_s(468), _s(238)), "yazar · okur", 13, KIRMIZI)

    # depo
    kutu(d, (_s(730), _s(140), _s(960), _s(300)), dolgu=D_YESIL, renk=YESIL, r=16)
    yazi(d, (_s(845), _s(170)), "Artifact deposu", 18, INK, kalin=True, hiza="center")
    yazi(d, (_s(845), _s(200)), "MinIO + kayıt defteri", 14, YESIL, hiza="center")
    yazi(d, (_s(845), _s(232)), "pod ölse de kalır", 14, SOLUK, hiza="center")

    # Yerleştirme depodan İÇERİ, süpürme pod'dan DIŞARI. Ters çizilmişti.
    ok(d, (_s(722), _s(180)), (_s(608), _s(180)), MAVI)
    yazi(d, (_s(665), _s(152)), "① yerleştirir", 13, MAVI, hiza="center")
    ok(d, (_s(608), _s(258)), (_s(722), _s(258)), YESIL)
    yazi(d, (_s(665), _s(268)), "② süpürür", 13, YESIL, hiza="center")

    yazi(d, (_s(40), _s(392)), "Sandbox depoyu hiç görmez: S3 anahtarı yok, MinIO'ya "
         "rotası yok. Baytları taşıyan ayrı bir container.", 14, SOLUK)
    _kaydet(im, "d1-yasam-dongusu")


# ── D2 — aracı nerede duruyor ─────────────────────────────────────────────


def d2_aracinin_yeri():
    """Üç yerleşim yan yana. Asıl iddia ŞEKİLDE: KFP'de taşıyıcı kullanıcı
    kodunun İÇİNDE (aynı container), diğer ikisinde dışarıda."""
    random.seed(11)
    G, Y = 1000, 470
    im, d = _tuval(G, Y)

    yazi(d, (_s(500), _s(22)), "Baytı kim taşıyor — ve S3 anahtarı nerede duruyor",
         18, INK, kalin=True, hiza="center")

    W = 280
    sutunlar = [
        (30, "KFP / OpenShift AI", KIRMIZI, D_KIRMIZI, True,
         ["launcher", "kullanıcı kodunu SARMALAR"],
         "anahtar, LLM'in kodunun\nokuyabileceği yerde", KIRMIZI),
        (360, "Argo Workflows", TURUNCU, D_SARI, False,
         ["wait sidecar", "ayrı container"],
         "anahtar dışarıda,\nama kayıt defteri YOK", TURUNCU),
        (690, "BİZ", MAVI, D_MAVI, False,
         ["sidecar → HTTP →", "artifact servisi"],
         "anahtar serviste,\nkayıt defteri ayakta", YESIL),
    ]

    for x, baslik, renk, dolgu, ici, satirlar, not_, notRenk in sutunlar:
        X = _s(x)
        yazi(d, (X + _s(W / 2), _s(62)), baslik, 16, INK, kalin=True, hiza="center")

        if ici:
            # Taşıyıcı DIŞTA, kullanıcı kodu İÇİNDE — sarmalama.
            kutu(d, (X, _s(96), X + _s(W), _s(268)), dolgu=dolgu, renk=renk, r=14)
            yazi(d, (X + _s(W / 2), _s(116)), "tek container", 12, renk, hiza="center")
            for i, s2 in enumerate(satirlar):
                yazi(d, (X + _s(W / 2), _s(136 + i * 20)), s2, 12, INK,
                     hiza="center", kalin=(i == 0))
            kutu(d, (X + _s(24), _s(184), X + _s(W - 24), _s(252)),
                 dolgu=(255, 255, 255), renk=SOLUK, r=11)
            yazi(d, (X + _s(W / 2), _s(204)), "kullanıcı kodu", 14, INK, hiza="center")
            yazi(d, (X + _s(W / 2), _s(226)), "/output'a dosya yazar", 11, SOLUK, hiza="center")
        else:
            kutu(d, (X, _s(96), X + _s(W), _s(168)), dolgu=(255, 255, 255), renk=SOLUK, r=12)
            yazi(d, (X + _s(W / 2), _s(114)), "kullanıcı kodu", 14, INK, hiza="center")
            yazi(d, (X + _s(W / 2), _s(136)), "/output'a dosya yazar", 11, SOLUK, hiza="center")
            # container sınırı
            for k in range(0, int(_s(W)), int(_s(14))):
                d.line([(X + k, _s(180)), (X + k + _s(7), _s(180))],
                       fill=SOLUK, width=int(1.5 * OLCEK))
            yazi(d, (X + _s(W / 2), _s(186)), "container sınırı", 10, SOLUK, hiza="center")
            kutu(d, (X, _s(204), X + _s(W), _s(268)), dolgu=dolgu, renk=renk, r=12)
            for i, s2 in enumerate(satirlar):
                yazi(d, (X + _s(W / 2), _s(222 + i * 20)), s2, 12, INK,
                     hiza="center", kalin=(i == 0))

        kutu(d, (X + _s(60), _s(300), X + _s(W - 60), _s(346)), dolgu=D_GRI, renk=SOLUK, r=10)
        yazi(d, (X + _s(W / 2), _s(323)), "nesne deposu", 12, SOLUK,
             hiza="center", dikey="middle")
        ok(d, (X + _s(W / 2), _s(270)), (X + _s(W / 2), _s(298)), SOLUK, kalinlik=2, bas=10)

        for i, s2 in enumerate(not_.split("\n")):
            yazi(d, (X + _s(W / 2), _s(366 + i * 19)), s2, 12, notRenk, hiza="center")

    yazi(d, (_s(500), _s(432)),
         "Kodu insan yazıyorsa KFP'nin yerleşimi sorun değil. Bizde kodu LLM yazıyor.",
         13, SOLUK, hiza="center")
    _kaydet(im, "d2-aracinin-yeri")


# ── D3 — çapraz workflow ──────────────────────────────────────────────────


def d3_capraz_workflow():
    random.seed(3)
    G, Y = 1000, 470
    im, d = _tuval(G, Y)

    def adim(x, y, etiket, renk, dolgu, w=170, h=56):
        kutu(d, (_s(x), _s(y), _s(x + w), _s(y + h)), dolgu=dolgu, renk=renk, r=12)
        yazi(d, (_s(x + w / 2), _s(y + h / 2)), etiket, 13, INK,
             hiza="center", dikey="middle")

    # WF-A
    yazi(d, (_s(40), _s(28)), "WORKFLOW A  ·  Ticket İşleme Hattı", 15, MAVI, kalin=True)
    for i, ad in enumerate(["Veri Topla", "İçerik Ayıkla", "PTC Türetme", "Rapor Üret"]):
        adim(40 + i * 240, 56, ad, MAVI, D_MAVI, w=195)
        if i:
            ok(d, (_s(40 + (i - 1) * 240 + 195), _s(84)), (_s(40 + i * 240), _s(84)),
               SOLUK, kalinlik=2, bas=10)

    # depo
    kutu(d, (_s(40), _s(180), _s(960), _s(280)), dolgu=(255, 252, 240), renk=TURUNCU, r=16)
    yazi(d, (_s(60), _s(196)), "ARTIFACT DEPOSU  ·  paylaşılan, çalıştırmalardan bağımsız",
         14, TURUNCU, kalin=True)
    for i, (ad, x) in enumerate([("ham.tickets.parquet", 70), ("extracted-content.json", 300),
                                 ("processed-result.json", 530), ("final-report.pdf", 760)]):
        vurgu = i == 2
        kutu(d, (_s(x), _s(225), _s(x + 190), _s(263)),
             dolgu=D_SARI if vurgu else (255, 255, 255),
             renk=TURUNCU if vurgu else SOLUK, r=8, kalinlik=2 if vurgu else 1)
        yazi(d, (_s(x + 95), _s(244)), ad, 11, INK if vurgu else SOLUK,
             hiza="center", dikey="middle", mono=True)

    for i, x in enumerate([135, 365, 595, 825]):
        ok(d, (_s(x), _s(114)), (_s(x), _s(220)), SOLUK, kalinlik=2, bas=9, kesik=True)

    # WF-B
    yazi(d, (_s(40), _s(336)), "WORKFLOW B  ·  Artifact Analiz Hattı  —  A'yı hiç bilmiyor",
         15, YESIL, kalin=True)
    for i, ad in enumerate(["Artifact Keşfet", "Artifact Yükle", "Analiz Et", "Bulgu Yayınla"]):
        adim(40 + i * 240, 358, ad, YESIL, D_YESIL, w=195)
        if i:
            ok(d, (_s(40 + (i - 1) * 240 + 195), _s(386)), (_s(40 + i * 240), _s(386)),
               SOLUK, kalinlik=2, bas=10)

    # keşif oku: depodan B'nin 1. adımına
    ok(d, (_s(625), _s(266)), (_s(137), _s(356)), TURUNCU, kalinlik=3, bas=13)
    kutu(d, (_s(330), _s(288), _s(730), _s(322)), dolgu=(255, 255, 255), renk=TURUNCU, r=9)
    yazi(d, (_s(530), _s(305)), 'GET /artifacts?name=processed-result.json', 12,
         INK, hiza="center", dikey="middle", mono=True)

    # B'nin çıktısı aynı depoya
    ok(d, (_s(920), _s(354)), (_s(920), _s(284)), YESIL, kalinlik=2, bas=10, kesik=True)
    yazi(d, (_s(930), _s(312)), "aynı depoya", 11, YESIL)

    yazi(d, (_s(40), _s(432)), "B, A'nın çalıştırma kimliğini kayıt defterinden ÖĞRENİYOR. "
         "Aralarında doğrudan bağ yok; A çoktan bitmiş olabilir.", 13, SOLUK)
    _kaydet(im, "d3-capraz-workflow")


if __name__ == "__main__":
    print("diyagramlar:")
    d1_yasam_dongusu()
    d2_aracinin_yeri()
    d3_capraz_workflow()
