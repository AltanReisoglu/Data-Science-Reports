#!/usr/bin/env python
"""Slayt taşma ölçeri — metni uzatmadan önce ve sonra koşulur.

    python docs/diagrams/tasma_olc.py docs/pdf/hap-autogen.html ...

**Neden gerekli.** `.slide` sabit yükseklikte (167 mm) ve `overflow:hidden`.
Taşan metin hata vermiyor, uyarı vermiyor — sessizce kesiliyor. PDF'e bakan biri
cümlenin yarısını hiç görmüyor. Metni açıklayıcı hale getirmek demek metni
uzatmak demek, yani bu betik olmadan yapılan her iyileştirme kör bir bahis.

**Nasıl ölçüyor.** Ekran CSS'i slaytları üst üste bindirip ölçekliyor; ölçüm
onun altında anlamsız olurdu. Bu yüzden ölçmeden önce yazdırma yerleşimi elle
kuruluyor: her slayt görünür, ölçeksiz ve akışta. Sonra her slaytın en alttaki
içerik pikseliyle dibi (alt not varsa onun üstü) karşılaştırılıyor.

Çıktı milimetre cinsinden: eksi değer taşma, artı değer boşluk.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Taşma yoksa bile bu kadar boşluk kalmalı: bir satırlık pay. Yazı tipi
# yedeklenirse (DejaVu yoksa) satırlar biraz uzar, ve sıfır boşlukla geçen bir
# slayt başka bir makinede kesilir.
PAY_MM = 3.0

PROBE = """
<script>
(function(){
  // Yazdırma yerleşimini elle kur: ekran CSS'i slaytları üst üste bindiriyor.
  var st = document.createElement('style');
  st.textContent = '.slide,.page{position:relative!important;display:block!important;' +
                   'transform:none!important;margin:0 0 4mm;box-shadow:none}';
  document.head.appendChild(st);

  // Milimetre karşılığını ölç — piksel/mm oranı zoom'a göre değişir.
  var ruler = document.createElement('div');
  ruler.style.cssText = 'position:absolute;width:100mm;visibility:hidden';
  document.body.appendChild(ruler);
  var PXMM = ruler.getBoundingClientRect().width / 100;

  // Deste slaytı da hatırlatma kartı sayfası da aynı kısıtı taşıyor: sabit
  // yükseklik + overflow:hidden. Seçici ona göre otomatik.
  var SEL = document.querySelector('.slide') ? '.slide' : '.page';
  // Üç ayrı düzen var ve her biri gövdesini başka bir sınıfa koyuyor:
  // deste slaytı `.bd`, hatırlatma kartı `.b`, akış kartı `.r`/`.act`.
  // Seçici tutmazsa ölçüm sessizce "sayfa bomboş" der — yani tam da
  // yakalaması gereken hatayı gizler. O yüzden hepsi birden aranıyor.
  var BODY = SEL === '.slide' ? '.bd, .bd *' : '.b, .b *, .r, .r *, .act';
  var out = [];
  document.querySelectorAll(SEL).forEach(function(s, i){
    var h2 = s.querySelector('h2') || s.querySelector('.hd .t');
    var ft = s.querySelector('.ft');
    var sb = s.getBoundingClientRect();
    var pad = parseFloat(getComputedStyle(s).paddingBottom);
    var dip = ft ? ft.getBoundingClientRect().top : sb.bottom - pad;

    var alt = sb.top, suclu = '';
    s.querySelectorAll(BODY).forEach(function(el){
      if (el.classList && el.classList.contains('ft')) return;
      var r = el.getBoundingClientRect();
      if (r.height === 0) return;
      if (r.bottom > alt) { alt = r.bottom; suclu = el.className || el.tagName; }
    });

    out.push({n: i + 1,
              baslik: h2 ? h2.textContent.trim() : '(kapak)',
              bosluk_mm: +((dip - alt) / PXMM).toFixed(1),
              suclu: String(suclu).slice(0, 24)});
  });
  document.title = 'RAPOR' + JSON.stringify(out);
})();
</script>
"""


def olc(path: Path) -> list[dict]:
    html = path.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / path.name
        probe.write_text(html.replace("</body>", PROBE + "</body>"), encoding="utf-8")
        r = subprocess.run(
            ["google-chrome", "--headless", "--disable-gpu", "--no-sandbox",
             f"--user-data-dir={tmp}/prof", "--window-size=1400,900",
             "--virtual-time-budget=8000", "--dump-dom", probe.as_uri()],
            capture_output=True, text=True, timeout=240)
        if r.returncode != 0:
            raise RuntimeError(f"chrome: {r.stderr[-400:]}")
        m = re.search(r"<title>RAPOR(.*?)</title>", r.stdout, re.S)
        if not m:
            raise RuntimeError("ölçüm betiği koşmadı — sayfa yüklenmemiş olabilir")
        return json.loads(m.group(1))


def main(paths: list[str]) -> int:
    kotu = 0
    for p in paths:
        path = Path(p)
        rapor = olc(path)
        dar = [s for s in rapor if s["bosluk_mm"] < PAY_MM]
        print(f"\n{path.name}  ·  {len(rapor)} slayt  ·  "
              f"{len(dar)} sorunlu (pay < {PAY_MM:g} mm)")
        for s in sorted(rapor, key=lambda x: x["bosluk_mm"])[:8]:
            im = "TAŞMA" if s["bosluk_mm"] < 0 else ("dar" if s["bosluk_mm"] < PAY_MM else "ok")
            print(f"  {s['n']:>3}  {s['bosluk_mm']:>7.1f} mm  {im:<5}  {s['baslik'][:52]}")
        kotu += len(dar)
    print(f"\ntoplam sorunlu: {kotu}")
    return 1 if kotu else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["docs/pdf/hap-autogen.html"]))
