"""Çerçeve rehberi: AutoGen, MAF, OpenClaw — ve diğerleri.

`ogretici.pdf` "nasıl kurulur"u anlatıyor. Bu belge farklı bir soruya cevap
veriyor: **hangisini, neden, ve neyin yerine.** Bir kurum çerçeve seçerken
sorduğu soru bu, ve cevabı üç ayrı yerde duruyordu — destede, docs/09'da, ve
kurulu paketlerin içinde.

### Bu belgenin kuralları

1. **Her iddia etiketli.** `[ölçüldü]` bu depoda koşturuldu ve ölçüm dosyası
   gösterilebilir · `[kaynak]` birincil kaynaktan doğrulandı (resmî doküman,
   kurulu paket, depo) · `[teyitsiz]` okundu ama koşturulmadı.
2. **Rakip çerçeveler hakkındaki mimari iddiaların çoğu `[teyitsiz]`.** AutoGen,
   MAF ve OpenClaw gerçekten koşturuldu; LangGraph, CrewAI, Agents SDK ve ADK
   koşturulmadı. Bunu gizlemek, belgenin geri kalanına olan güveni de düşürür.
3. **Satıcının kendi cümlesi, bizim cümlemizden ağır basar.** MAF'ın AutoGen'i
   nerede değiştirdiğini Microsoft'un geçiş kılavuzundan alıntılıyoruz; biz
   yalnız ölçtüğümüzü ekliyoruz.

Motor `make_ogretici.py`'den geliyor — sayfa geometrisi, kutu tipleri, yazdırma
CSS'i. İki ayrı tasarım, aynı projede iki ayrı belge gibi okunurdu.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_ogretici as eng  # noqa: E402 — ortak motor

OUT = Path(__file__).resolve().parent.parent / "pdf" / "rehber-cerceveler.html"


def cover() -> str:
    return (
        '<section class="cover"><div>'
        '<p style="font-family:var(--mono);font-size:8.4pt;letter-spacing:.16em;'
        'text-transform:uppercase;color:var(--ink3);margin-bottom:9mm">rehber · '
        "hangisini, neden, neyin yerine</p>"
        "<h1>Ajan çerçeveleri:<br>AutoGen, MAF, OpenClaw</h1>"
        '<p class="sub">Üç sistemi mekanizma mekanizma açar, aralarındaki geçiş '
        "yollarını gösterir, ve piyasadaki diğer çerçevelerle yan yana koyar. "
        "Her iddia etiketli: ölçülen, kaynaktan doğrulanan, ve yalnızca okunan "
        "ayrı ayrı işaretli.</p></div>"
        '<div class="meta">autogen-core / agentchat / ext v0.7.5 · '
        "agent-framework 1.14.0 · OpenClaw @01cc7106<br>"
        "ölçümler bu depoda koşturuldu · satıcı iddiaları kılavuzdan alıntı</div>"
        "</section>"
    )


def build() -> str:
    toc = "".join(f'<li class="{k}">{eng.esc(t)}</li>' for k, t in eng.TOC)
    body = "".join(eng.BLOCKS)
    return (
        '<!doctype html>\n<html lang="tr"><head><meta charset="utf-8">\n'
        "<title>Ajan çerçeveleri — rehber</title>\n"
        f"<style>{eng.CSS}</style></head><body>\n{cover()}"
        f'<section class="toc"><h2>İçindekiler</h2><ul class="tocl">{toc}</ul></section>'
        f"{body}\n</body></html>\n"
    )


if __name__ == "__main__":
    # Motorun kendi içeriği bu belgeye karışmasın: `make_ogretici` içe
    # aktarıldığında blok listesi boş, ama bir gün dolarsa sessizce sızardı.
    eng.BLOCKS.clear()
    eng.TOC.clear()

    for _f in ("rehber_a.py", "rehber_b.py"):
        _src = (Path(__file__).resolve().parent / _f).read_text(encoding="utf-8")
        exec(compile(_src, _f, "exec"), globals())  # noqa: S102 — kendi dosyalarımız

    OUT.parent.mkdir(parents=True, exist_ok=True)
    html = build()
    OUT.write_text(html, encoding="utf-8")
    print(f"{OUT.name}  ·  {len(eng.BLOCKS)} blok  ·  "
          f"{html.count('<svg')} şekil  ·  {len(html)/1024:.0f} KB")
