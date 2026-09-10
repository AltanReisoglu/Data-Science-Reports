#!/usr/bin/env python3
"""`.excalidraw` sahnesi → SVG (+ PNG).

Confluence sayfalarında diyagramlar üç biçimde duruyor:

    .excalidraw   kaynak — Confluence'ın Excalidraw makrosuna import edilir
    .png          sayfada görünen
    .svg          baskı / ölçekleme

Bu betik ikincisini ve üçüncüsünü birincisinden üretir.

## Neden kendi çeviricimiz var

Ortamda `excalidraw` CLI'ı yok. Ama sahneleri biz ürettiğimiz için eleman
sözlüğü dar ve bilinen: `rectangle`, `diamond`, `ellipse`, `arrow`, `text`.
Tam bir Excalidraw uygulaması DEĞİL — yalnızca bu sözlüğü çeviriyor.

Çizgiler el yazısı görünümünde değil, temiz vektör: Confluence'ta küçültülünce
okunaklı kalması, "rough" görünümden daha önemli.

Kullanım:
    python scripts/excalidraw_svg.py confluence/dort-aile.excalidraw
    python scripts/excalidraw_svg.py confluence/*.excalidraw
    python scripts/excalidraw_svg.py --sadece-svg confluence/x.excalidraw
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

KENAR = 24          # sahnenin çevresine bırakılan boşluk
PNG_OLCEK = 1.4     # PNG, SVG'nin kaç katı çözünürlükte üretilsin

YAZI = "Segoe UI, Helvetica, Arial, sans-serif"
DAKTILO = "ui-monospace, Menlo, Consolas, monospace"


def _font(e: dict) -> str:
    # Excalidraw fontFamily: 1=el yazısı, 2=normal, 3=daktilo.
    # Confluence'ta okunaklılık için el yazısını da normal sete çeviriyoruz.
    return DAKTILO if e.get("fontFamily") == 3 else YAZI


def _sinir(elemanlar: list[dict]) -> tuple[float, float, float, float]:
    xs, ys, xe, ye = [], [], [], []
    for e in elemanlar:
        if e.get("isDeleted"):
            continue
        x, y = e["x"], e["y"]
        w, h = e.get("width", 0), e.get("height", 0)
        if e["type"] == "arrow":
            # ok'un noktaları negatif olabiliyor; ikisini de hesaba kat
            for dx, dy in e.get("points", [[0, 0]]):
                xs.append(x + dx); ys.append(y + dy)
                xe.append(x + dx); ye.append(y + dy)
            continue
        xs.append(x); ys.append(y); xe.append(x + w); ye.append(y + h)
    return min(xs), min(ys), max(xe), max(ye)


def _cizgi_stili(e: dict) -> str:
    if e.get("strokeStyle") == "dashed":
        return ' stroke-dasharray="8 6"'
    if e.get("strokeStyle") == "dotted":
        return ' stroke-dasharray="2 5"'
    return ""


def _sekil(e: dict) -> list[str]:
    x, y, w, h = e["x"], e["y"], e.get("width", 0), e.get("height", 0)
    bg = e.get("backgroundColor", "transparent")
    if bg == "transparent":
        bg = "none"
    ortak = (f'fill="{bg}" stroke="{e.get("strokeColor", "#1e1e1e")}" '
             f'stroke-width="{e.get("strokeWidth", 2)}"'
             + _cizgi_stili(e))
    op = e.get("opacity", 100)
    if op != 100:
        ortak += f' opacity="{op / 100:.2f}"'

    if e["type"] == "rectangle":
        r = 8 if e.get("roundness") else 0
        return [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" {ortak}/>']
    if e["type"] == "ellipse":
        return [f'<ellipse cx="{x + w/2}" cy="{y + h/2}" rx="{w/2}" ry="{h/2}" {ortak}/>']
    if e["type"] == "diamond":
        p = f"{x + w/2},{y} {x + w},{y + h/2} {x + w/2},{y + h} {x},{y + h/2}"
        return [f'<polygon points="{p}" {ortak}/>']
    return []


def _ok(e: dict) -> list[str]:
    x, y = e["x"], e["y"]
    noktalar = e.get("points") or [[0, 0], [e.get("width", 0), e.get("height", 0)]]
    mutlak = [(x + dx, y + dy) for dx, dy in noktalar]
    renk = e.get("strokeColor", "#1e1e1e")
    sw = e.get("strokeWidth", 2)
    d = " ".join(f"{px},{py}" for px, py in mutlak)
    cikti = [f'<polyline points="{d}" fill="none" stroke="{renk}" '
             f'stroke-width="{sw}" stroke-linecap="round"{_cizgi_stili(e)}/>']
    if e.get("endArrowhead") == "arrow" and len(mutlak) >= 2:
        (x1, y1), (x2, y2) = mutlak[-2], mutlak[-1]
        aci = math.atan2(y2 - y1, x2 - x1)
        L, g = 11 + sw, 0.45
        p1 = (x2 - L * math.cos(aci - g), y2 - L * math.sin(aci - g))
        p2 = (x2 - L * math.cos(aci + g), y2 - L * math.sin(aci + g))
        cikti.append(f'<polygon points="{x2},{y2} {p1[0]:.1f},{p1[1]:.1f} '
                     f'{p2[0]:.1f},{p2[1]:.1f}" fill="{renk}"/>')
    return cikti


def _yazi(e: dict, kaplar: dict) -> list[str]:
    fs = e.get("fontSize", 16)
    satirlar = e.get("text", "").split("\n")
    lh = fs * e.get("lineHeight", 1.25)
    renk = e.get("strokeColor", "#1e1e1e")
    kap = kaplar.get(e.get("containerId"))
    if kap:
        # Kaba bağlı metin: yatay ve dikey ortalanır.
        cx = kap["x"] + kap.get("width", 0) / 2
        ust = kap["y"] + kap.get("height", 0) / 2 - lh * len(satirlar) / 2
        hiza, ax = "middle", cx
    else:
        ust = e["y"]
        hiza, ax = "start", e["x"]
    parcalar = []
    for i, s in enumerate(satirlar):
        by = ust + lh * i + fs * 0.78          # baseline
        parcalar.append(
            f'<text x="{ax:.0f}" y="{by:.0f}" font-family="{_font(e)}" '
            f'font-size="{fs}" fill="{renk}" text-anchor="{hiza}">{escape(s)}</text>')
    return parcalar


def svg_uret(sahne: dict) -> str:
    elemanlar = [e for e in sahne["elements"] if not e.get("isDeleted")]
    kaplar = {e["id"]: e for e in elemanlar if e["type"] != "text"}
    x0, y0, x1, y1 = _sinir(elemanlar)
    x0, y0 = x0 - KENAR, y0 - KENAR
    w, h = (x1 - x0) + KENAR, (y1 - y0) + KENAR

    govde = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" '
             f'height="{h:.0f}" viewBox="{x0:.0f} {y0:.0f} {w:.0f} {h:.0f}">',
             f'<rect x="{x0:.0f}" y="{y0:.0f}" width="{w:.0f}" height="{h:.0f}" '
             f'fill="#ffffff"/>']
    # Sıra = z-düzeni; metinler kendi kaplarından SONRA gelmeli.
    for e in elemanlar:
        if e["type"] in ("rectangle", "ellipse", "diamond"):
            govde += _sekil(e)
        elif e["type"] == "arrow":
            govde += _ok(e)
    for e in elemanlar:
        if e["type"] == "text":
            govde += _yazi(e, kaplar)
    govde.append("</svg>")
    return "\n".join(govde)


def png_uret(svg_yolu: Path, png_yolu: Path) -> bool:
    convert = shutil.which("convert")
    if not convert:
        return False
    boyut = svg_yolu.read_text(encoding="utf-8").split(">", 1)[0]
    genislik = int(float(boyut.split('width="')[1].split('"')[0]))
    subprocess.run([convert, "-background", "white", "-density",
                    str(int(96 * PNG_OLCEK)), "-resize",
                    f"{int(genislik * PNG_OLCEK)}x", str(svg_yolu), str(png_yolu)],
                   check=True, capture_output=True)
    return True


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sadece_svg = "--sadece-svg" in sys.argv
    if not args:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        return 2
    for yol in args:
        p = Path(yol)
        sahne = json.loads(p.read_text(encoding="utf-8"))
        svg = p.with_suffix(".svg")
        svg.write_text(svg_uret(sahne), encoding="utf-8")
        satir = f"  {svg.name}"
        if not sadece_svg:
            png = p.with_suffix(".png")
            satir += f"  ·  {png.name}" if png_uret(svg, png) else "  ·  PNG YOK (convert bulunamadı)"
        print(satir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
