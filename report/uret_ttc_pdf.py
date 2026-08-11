#!/usr/bin/env python3
"""
uret_ttc_pdf.py — Tool-Trace Compaction tek sayfalık sunum kartı (PDF).

Veriyi UYDURMUYOR: `poc/kiyas.py`'den okuyor (adım şemaları, tetikler, ölçümler).
Bu yüzden POC değişirse kart da değişir.

    python report/uret_ttc_pdf.py           # HTML + PDF üret
    python report/uret_ttc_pdf.py --html    # yalnız HTML (hızlı bakış)

PDF: Chrome headless ile, A4 YATAY, tek sayfa.
"""
from __future__ import annotations

import html
import json
import subprocess
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK / "poc"))

import kiyas as K  # noqa: E402

CIKTI_HTML = KOK / "report" / "tool-trace-compaction-1sayfa.html"
CIKTI_PDF = KOK / "report" / "tool-trace-compaction-1sayfa.pdf"

E = html.escape


def veri() -> dict:
    """Ölçümleri canlı koştur; LLM'li mantıklar patlarsa kart yine üretilsin."""
    try:
        v = K.hepsi()
        olcum = {m["ad"]: m for m in v["mantiklar"]}
    except Exception as e:            # ölçüm alınamazsa kart 'ölçülmedi' der
        print(f"  ⚠ canlı ölçüm alınamadı ({type(e).__name__}), kart ölçümsüz üretiliyor")
        olcum = {}
    return {"adimlar": K.ADIMLAR, "bilgi": K.BILGI, "olcum": olcum}


# ── tetik özeti: en yaygın kafa karışıklığı burada ─────────────────────────
TETIK_TABLO = [
    ("Hermes", "—",
     "proaktif: HER turda (pencere dolmasa da). Tek fren: kazanç &lt;4096 token ise commit etmez"),
    ("OpenClaw", "—",
     "boş yer &lt; pencere × 0.5 (kullanılan &gt; %50)"),
    ("OpenCode", "&gt;2000 satır VEYA &gt;50KB → diske spill",
     "proaktif: her turda (prune açıksa) · overflow özeti ancak kullanılan ≥ pencere−20K"),
    ("Codex", "tek çıktı tavanı aşarsa → truncate_middle",
     "history pencereye sığmazsa → placeholder trim, yetmezse YENİ pencere"),
    ("Claude Code", "&gt;~4K token → microcompaction (diske)",
     "context &gt; ~%80 → auto-compaction"),
]

RENK = {"hermes": "#0d7d78", "opencode": "#2563a8", "openclaw": "#8b5cf6",
        "codex": "#b45309", "claude_code": "#be3455"}


def kart(d: dict) -> str:
    sistemler = ["hermes", "opencode", "openclaw", "codex", "claude_code"]

    sutunlar = []
    for m in sistemler:
        a, b = d["adimlar"][m], d["bilgi"][m]
        o = d["olcum"].get(m)
        renk = RENK[m]
        # `kiyas._paket` 'tetiklendi' anahtarı DÖNDÜRMÜYOR — sayıları doğrudan
        # kullan. (İlk sürümde o anahtara bakılıyordu ve kart hep 'tetiklenmedi'
        # yazıyordu; kazanç sayıları, yani kartın asıl payloadı görünmüyordu.)
        if o and o.get("once"):
            kazanc = (f"{o['once']:,} → {o['sonra']:,}".replace(",", ".")
                      + f"  ·  %{o['pct']} kazanç")
        elif o:
            kazanc = "tetiklenmedi"
        else:
            kazanc = "ölçülmedi"
        vuran = {x["kod"]: x["sayi"] for x in (o["adimlar"] if o else [])}

        adim_satir = []
        for x in a["adimlar"]:
            n = vuran.get(x["kod"])
            im = ("●" if n else "○") if o else "·"
            kapsam = "" if x.get("tool_izi", True) else '<span class="kd">kapsam dışı</span>'
            adim_satir.append(
                f'<div class="ad"><div class="a1"><span class="im">{im}</span>'
                f'<b>{E(x["ad"])}</b>{kapsam}'
                f'{f"<span class=v>{n} birim</span>" if n else ""}</div>'
                f'<div class="a2">{E(x["etiket"])}</div>'
                f'<div class="a3">{E(x["ozet"][:230])}</div>'
                f'<div class="a4">kayıp: {E(x["kayip"])}</div></div>')

        sutunlar.append(f'''
        <div class="sut" style="--c:{renk}">
          <div class="sh">
            <div class="sad">{E(b["baslik"])}</div>
            <div class="sek">{E(b["ekol"])}{" · LLM" if b["llm"] else " · LLM'siz"}</div>
            <div class="skz">{E(kazanc)}</div>
          </div>
          <div class="stetik">
            {f'<div><b>üretim anında</b> {E(a["tetik"]["uretim"][:150])}</div>' if a["tetik"]["uretim"] else ''}
            <div><b>eşikte</b> {E(a["tetik"]["esik"][:210])}</div>
          </div>
          {"".join(adim_satir)}
          <div class="disar"><b>tool-trace dışı:</b> {E(" · ".join(a["disarida"]))}</div>
        </div>''')

    tetik_satirlari = "".join(
        f'<tr><td class="ts">{ad}</td><td>{u}</td><td>{e}</td></tr>'
        for ad, u, e in TETIK_TABLO)

    return f'''<!doctype html><html lang="tr"><head><meta charset="utf-8">
<title>Tool-Trace Compaction — tek sayfa</title>
<style>
@page {{ size: A3 landscape; margin: 8mm; }}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,"Segoe UI",Roboto,sans-serif;font-size:8.6pt;
 line-height:1.35;color:#16202b;background:#fff}}
.bas{{display:flex;align-items:baseline;gap:10px;border-bottom:1.6px solid #16202b;
 padding-bottom:3px;margin-bottom:5px}}
h1{{font-size:14pt;letter-spacing:-.2px}}
.alt{{font-size:8.4pt;color:#5b6673;flex:1}}
.tarih{{font-size:7.7pt;color:#8b95a1}}

.ust{{display:grid;grid-template-columns:1.32fr 1fr;gap:7px;margin-bottom:6px}}
.kutu{{border:1px solid #d5dbe2;border-radius:4px;padding:5px 7px}}
.kb{{font-size:7.4pt;letter-spacing:.09em;text-transform:uppercase;color:#7c8895;
 font-weight:700;margin-bottom:3px}}
table{{width:100%;border-collapse:collapse;font-size:8.0pt}}
th{{text-align:left;font-size:7.2pt;text-transform:uppercase;letter-spacing:.05em;
 color:#7c8895;border-bottom:1px solid #d5dbe2;padding:2px 4px}}
td{{padding:2.5px 4px;border-bottom:1px solid #edf0f3;vertical-align:top}}
td.ts{{font-weight:700;white-space:nowrap;width:60px}}
.kavram b{{color:#0d7d78}}
.kavram div{{margin-bottom:2.5px}}

.sut_kap{{display:grid;grid-template-columns:repeat(5,1fr);gap:6px}}
.sut{{border:1px solid #d5dbe2;border-top:2.6px solid var(--c);border-radius:4px;
 padding:5px 6px;break-inside:avoid}}
.sh{{border-bottom:1px solid #edf0f3;padding-bottom:3px;margin-bottom:4px}}
.sad{{font-size:10.9pt;font-weight:700;color:var(--c);line-height:1.15}}
.sek{{font-size:7.3pt;color:#7c8895;text-transform:uppercase;letter-spacing:.05em}}
.skz{{font-size:8.6pt;font-weight:700;font-variant-numeric:tabular-nums;margin-top:1.5px}}
.stetik{{background:#f5f7f9;border-radius:3px;padding:3.5px 5px;margin-bottom:4px;
 font-size:7.5pt;line-height:1.35;color:#3d4a57}}
.stetik b{{color:var(--c);text-transform:uppercase;font-size:6.8pt;letter-spacing:.05em}}
.ad{{border-bottom:1px solid #f0f2f5;padding:2.5px 0}}
.ad:last-of-type{{border-bottom:none}}
.a1{{font-size:8.2pt;display:flex;align-items:baseline;gap:3px;flex-wrap:wrap}}
.im{{color:var(--c);font-size:7.4pt}}
.v{{font-size:6.8pt;color:var(--c);border:.6px solid var(--c);border-radius:5px;
 padding:0 3px;white-space:nowrap}}
.kd{{font-size:6.5pt;color:#b45309;border:.6px dashed #b45309;border-radius:5px;padding:0 3px}}
.a2{{font-family:ui-monospace,Menlo,monospace;font-size:7.0pt;color:#5b6673;margin-top:.5px}}
.a3{{font-size:7.3pt;color:#3d4a57;margin-top:1px;line-height:1.3}}
.a4{{font-size:6.8pt;color:#8b95a1;margin-top:.5px}}
.disar{{font-size:6.8pt;color:#8b95a1;margin-top:3px;padding-top:3px;
 border-top:1px dashed #dfe4e9;line-height:1.3}}
.dip{{margin-top:5px;padding-top:4px;border-top:1px solid #d5dbe2;font-size:7.4pt;
 color:#5b6673;display:flex;gap:14px}}
.dip b{{color:#16202b}}
</style></head><body>

<div class="bas">
  <h1>Tool-Trace Compaction</h1>
  <div class="alt">Beş sistem · yalnız <b>tool izine dokunan</b> adımlar ·
    ne zaman tetiklenir, ne yapar, ne kaybettirir</div>
  <div class="tarih">ölçümler: poc/kiyas.py — canlı koşudan</div>
</div>

<div class="ust">
  <div class="kutu">
    <div class="kb">tetik özeti — iki tür tetik var, karıştırılmasın</div>
    <table><thead><tr><th>sistem</th>
      <th>ÜRETİM ANINDA (tek çıktının boyutu)</th>
      <th>EŞİKTE (toplam context)</th></tr></thead>
      <tbody>{tetik_satirlari}</tbody></table>
    <div style="font-size:7.2pt;color:#7c8895;margin-top:3px">
      Üretim anında tetiklenen = <b>boyut filtresi</b> (tek çıktı çok mu büyük).
      Eşikte tetiklenen = <b>birikim filtresi</b> (toplam taştı mı).
      Biri diğerinin yerini tutmaz: eşiğin altında kalan orta boy çıktılar
      birike birike pencereyi doldurur.</div>
  </div>
  <div class="kutu kavram">
    <div class="kb">iki ayrı &quot;bütçe&quot; — karıştırmayın</div>
    <div><b>Tek-çıktı tavanı</b> — bir tool çıktısının boyutu. Pencere BOŞKEN bile keser.
      Katman A. <i>Codex TOOL_BUDGET=5.000 · OpenCode 2000 satır/50KB · Claude Code ~4K</i></div>
    <div><b>Pencere bütçesi</b> — tüm context'in toplamı. Tanım gereği doluluğa bakar.
      Katman B. <i>Codex WINDOW=30.000 · OpenClaw pencere×0.5 · OpenCode pencere−20K ·
      Claude Code ~%80</i></div>
    <div style="margin-top:3px;padding-top:3px;border-top:1px dashed #dfe4e9">
      <b>Bir çıktı ikisine de yakalanabilir:</b> önce (A)'da kesilir, sonra (B)'de
      windowing'e girer. Sıralı iki filtre — biri diğerini iptal etmez.
      <i>Ölçüldü: Codex'te shell çıktısı 18.536 → 5.026 (A) → 0 (B).</i></div>
  </div>
</div>

<div class="sut_kap">{"".join(sutunlar)}</div>

<div class="dip">
  <div><b>● / ○</b> bu koşuda adım vurdu / vurmadı — vurmaması da bilgi:
    mekanizmanın ne zaman devreye GİRMEDİĞİ</div>
  <div><b>kapsam dışı</b> = konuşma seviyesi (tool izine ait değil) ama bu koşuda ize etki etti</div>
  <div><b>Tek cümlede:</b> Hermes tekilleştirir · OpenClaw sırrı söker ve devi feragatle
    düşürür · OpenCode üretimde diske alır, hacimle ölçer · Codex ortayı keser ·
    Claude Code ağır işin izini hiç ana pencereye sokmaz</div>
</div>
</body></html>'''


def main():
    print("Tool-Trace Compaction · tek sayfalık kart")
    print("  ölçümler alınıyor (üç mantık gerçek LLM çağırıyor)…")
    d = veri()
    CIKTI_HTML.write_text(kart(d), encoding="utf-8")
    print(f"  ✓ HTML → {CIKTI_HTML.relative_to(KOK)}")
    if "--html" in sys.argv:
        return
    r = subprocess.run(
        ["google-chrome", "--headless", "--disable-gpu", "--no-sandbox",
         "--no-pdf-header-footer", f"--print-to-pdf={CIKTI_PDF}",
         CIKTI_HTML.as_uri()],
        capture_output=True, text=True, timeout=120)
    if CIKTI_PDF.exists():
        print(f"  ✓ PDF  → {CIKTI_PDF.relative_to(KOK)} "
              f"({CIKTI_PDF.stat().st_size // 1024} KB)")
    else:
        print("  ✗ PDF üretilemedi:", (r.stderr or "")[-300:])


if __name__ == "__main__":
    main()
