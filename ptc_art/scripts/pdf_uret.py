#!/usr/bin/env python3
"""Markdown → PDF (LibreOffice üzerinden).

    python scripts/pdf_uret.py PTC_Urun_Mentaliteleri_ve_Karsilastirma.md

## Neden kendi çeviricimiz var

Ortamda `pandoc`, `weasyprint`, `wkhtmltopdf` yok; kurulu olan tek dönüştürücü
LibreOffice. O da Markdown okumuyor, HTML okuyor. Aradaki çevirici bu dosya.

Tam bir Markdown uygulaması DEĞİL — bu depodaki `PTC_*.md` dosyalarında fiilen
kullanılan alt kümeyi çeviriyor: başlıklar, tablolar, çitli kod blokları,
listeler, alıntılar, yatay çizgi, kalın/kod/bağlantı. `sunum_uret.js` de aynı
yaklaşımı izliyor (o da pptx için kendi ayrıştırıcısını taşıyor).

## Sayfa düzeni

LibreOffice'in HTML motoru CSS'in dar bir alt kümesini anlıyor; bu yüzden
stil bilerek sade: `@page` kenar boşluğu, tablo kenarlıkları, gri arka planlar.
Süslü bir şey eklemeye çalışmak sessizce yok sayılıyor.
"""
from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CSS = """
@page { size: A4; margin: 1.8cm 1.6cm; }
body { font-family: "Liberation Sans", Arial, sans-serif; font-size: 10pt;
       line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 19pt; color: #0b3d62; margin: 22pt 0 8pt;
     border-bottom: 2px solid #0b3d62; padding-bottom: 4pt; }
h2 { font-size: 14pt; color: #0b3d62; margin: 18pt 0 6pt; }
h3 { font-size: 11.5pt; color: #245a7a; margin: 13pt 0 4pt; }
p  { margin: 5pt 0; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 12pt;
        font-size: 8.8pt; }
th { background: #0b3d62; color: #ffffff; text-align: left;
     padding: 4pt 6pt; border: 1px solid #0b3d62; }
td { padding: 3.5pt 6pt; border: 1px solid #c4d3dd; vertical-align: top; }
tr:nth-child(even) td { background: #f2f6f9; }
pre { background: #f4f4f2; border: 1px solid #d8d8d4; padding: 7pt 9pt;
      font-family: "Liberation Mono", monospace; font-size: 8.4pt;
      line-height: 1.35; white-space: pre-wrap; margin: 7pt 0; }
code { font-family: "Liberation Mono", monospace; font-size: 9pt;
       background: #f0f0ee; padding: 0 2pt; }
blockquote { border-left: 3px solid #d08a2c; background: #fdf7ec;
             margin: 8pt 0; padding: 5pt 10pt; }
hr { border: none; border-top: 1px solid #ccd6dd; margin: 14pt 0; }
ul, ol { margin: 5pt 0 5pt 18pt; }
li { margin: 2pt 0; }
"""

#: Sıra önemli: `**kalın**` `*italik*`ten ÖNCE gelmeli, yoksa çift yıldız
#: ikiye bölünüp tek yıldız gibi eşleşir.
_SATIRICI = re.compile(
    r"`([^`]+)`|\*\*([^*]+)\*\*|\*([^*\n]+)\*|\[([^\]]+)\]\(([^)]+)\)")


def _satir_ici(metin: str) -> str:
    """Kalın, kod ve bağlantıyı çevirir; gerisi kaçışlanır.

    Tek geçişte yapılıyor — önce kaçışlayıp sonra işaretleri aramak, kod
    bloğunun içindeki `&amp;`'i bozardı.
    """
    parcalar: list[str] = []
    son = 0
    for m in _SATIRICI.finditer(metin):
        parcalar.append(html.escape(metin[son:m.start()]))
        kod, kalin, italik, bag_metin, bag_url = m.groups()
        if kod is not None:
            parcalar.append(f"<code>{html.escape(kod)}</code>")
        elif kalin is not None:
            parcalar.append(f"<b>{html.escape(kalin)}</b>")
        elif italik is not None:
            parcalar.append(f"<i>{html.escape(italik)}</i>")
        else:
            parcalar.append(f'<a href="{html.escape(bag_url)}">'
                            f"{html.escape(bag_metin)}</a>")
        son = m.end()
    parcalar.append(html.escape(metin[son:]))
    return "".join(parcalar)


def _tablo(satirlar: list[str]) -> str:
    """`| a | b |` bloğunu tabloya çevirir. İkinci satır hizalama, atlanıyor."""
    def hucreler(s: str) -> list[str]:
        return [h.strip() for h in s.strip().strip("|").split("|")]

    out = ["<table>", "<tr>"]
    # Boş başlık hücresi `&nbsp;` alıyor: boş `<th>` LibreOffice'te sütunu
    # sıfıra yakın daraltıp içeriği harf harf alt alta kırıyordu.
    out += [f"<th>{_satir_ici(h) or '&nbsp;'}</th>"
            for h in hucreler(satirlar[0])]
    out.append("</tr>")
    for s in satirlar[2:]:
        out.append("<tr>")
        out += [f"<td>{_satir_ici(h)}</td>" for h in hucreler(s)]
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def md_to_html(md: str, baslik: str) -> str:
    satirlar = md.splitlines()
    out: list[str] = []
    i = 0
    liste_acik = False

    #: Açık liste etiketi ("ul" / "ol") ya da None.
    liste_etiket: str | None = None

    def liste_kapat() -> None:
        nonlocal liste_acik, liste_etiket
        if liste_acik:
            out.append(f"</{liste_etiket}>")
            liste_acik = False
            liste_etiket = None

    def liste_ac(etiket: str) -> None:
        nonlocal liste_acik, liste_etiket
        if liste_acik and liste_etiket != etiket:
            liste_kapat()          # ul ↔ ol geçişi
        if not liste_acik:
            out.append(f"<{etiket}>")
            liste_acik = True
            liste_etiket = etiket

    while i < len(satirlar):
        s = satirlar[i]

        if s.startswith("```"):                       # çitli kod bloğu
            liste_kapat()
            i += 1
            govde = []
            while i < len(satirlar) and not satirlar[i].startswith("```"):
                govde.append(satirlar[i])
                i += 1
            i += 1
            out.append(f"<pre>{html.escape(chr(10).join(govde))}</pre>")
            continue

        if s.lstrip().startswith("|") and s.rstrip().endswith("|"):
            liste_kapat()
            blok = []
            while (i < len(satirlar) and satirlar[i].lstrip().startswith("|")
                   and satirlar[i].rstrip().endswith("|")):
                blok.append(satirlar[i])
                i += 1
            # En az başlık + hizalama satırı olmalı; yoksa düz metin say.
            out.append(_tablo(blok) if len(blok) >= 2
                       else "".join(f"<p>{_satir_ici(b)}</p>" for b in blok))
            continue

        if s.startswith("> "):
            liste_kapat()
            blok = []
            while i < len(satirlar) and satirlar[i].startswith(">"):
                blok.append(satirlar[i].lstrip(">").strip())
                i += 1
            metin = " ".join(x for x in blok if x)
            out.append(f"<blockquote>{_satir_ici(metin)}</blockquote>")
            continue

        if re.match(r"^#{1,4} ", s):
            liste_kapat()
            d = len(s) - len(s.lstrip("#"))
            out.append(f"<h{d}>{_satir_ici(s[d + 1:])}</h{d}>")
        elif s.strip() in {"---", "***", "___"}:
            liste_kapat()
            out.append("<hr/>")
        elif re.match(r"^\s*[-*] ", s):
            liste_ac("ul")
            out.append(f"<li>{_satir_ici(re.sub(r"^\s*[-*] ", "", s))}</li>")
        elif re.match(r"^\s*\d+\. ", s):
            liste_ac("ol")
            out.append(f"<li>{_satir_ici(re.sub(r"^\s*\d+\. ", "", s))}</li>")
        elif s.strip():
            liste_kapat()
            # Boş satıra kadar TOPLA: Markdown'da paragrafı bitiren şey boş
            # satır, satır sonu değil. Satır satır <p> üretmek `**kalın**`
            # gibi iki satıra yayılan işaretleri de bölüyordu.
            govde = []
            while i < len(satirlar) and satirlar[i].strip():
                t = satirlar[i]
                if (t.startswith(("```", "> ", "|")) or re.match(r"^#{1,4} ", t)
                        or re.match(r"^\s*[-*] ", t)
                        or re.match(r"^\s*\d+\. ", t)
                        or t.strip() in {"---", "***", "___"}):
                    break
                govde.append(t.strip())
                i += 1
            out.append(f"<p>{_satir_ici(' '.join(govde))}</p>")
            continue
        else:
            liste_kapat()
        i += 1

    liste_kapat()
    return (f'<!DOCTYPE html><html><head><meta charset="utf-8">'
            f"<title>{html.escape(baslik)}</title><style>{CSS}</style></head>"
            f"<body>{''.join(out)}</body></html>")


def main() -> int:
    if len(sys.argv) < 2:
        print("kullanım: pdf_uret.py <dosya.md> [çıktı.pdf]", file=sys.stderr)
        return 2
    kaynak = Path(sys.argv[1])
    hedef = Path(sys.argv[2]) if len(sys.argv) > 2 else kaynak.with_suffix(".pdf")

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        print("soffice bulunamadı — LibreOffice gerekli.", file=sys.stderr)
        return 1

    md = kaynak.read_text(encoding="utf-8")
    baslik = next((s.lstrip("# ").strip() for s in md.splitlines()
                   if s.startswith("# ")), kaynak.stem)

    with tempfile.TemporaryDirectory() as td:
        ara = Path(td) / (kaynak.stem + ".html")
        ara.write_text(md_to_html(md, baslik), encoding="utf-8")
        # `-env:UserInstallation`: paralel/eşzamanlı çağrılarda LibreOffice'in
        # tek profil kilidine takılmamak için ayrı bir profil veriyoruz.
        r = subprocess.run(
            [soffice, "--headless", f"-env:UserInstallation=file://{td}/profil",
             "--convert-to", "pdf", "--outdir", td, str(ara)],
            capture_output=True, text=True, timeout=300, check=False)
        uretilen = Path(td) / (kaynak.stem + ".pdf")
        if not uretilen.exists():
            print("PDF üretilemedi.\n" + r.stdout + r.stderr, file=sys.stderr)
            return 1
        hedef.write_bytes(uretilen.read_bytes())

    print(f"yazıldı: {hedef}  ({hedef.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
