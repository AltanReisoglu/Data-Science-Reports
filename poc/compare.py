"""
compare.py — Kanonik bir senaryoyu TÜM sistemlerden geçir; aynı hammadde, farklı mantık.

  python compare.py            # tablo + her sistemin baskın kaderi
  python compare.py --detail   # + her sistem için trace dökümü

Senaryo, her sistemin tip-özel yolunu tetiklemek için özenle seçildi:
  - aynı dosya iki kez okunur (dedup/staleness),
  - arada dosya düzenlenir (mutasyon → bayat),
  - iki snapshot alınır (gemini supersede),
  - büyük terminal/web/grep çıktıları (kes/tek-satır/fold/crush).
"""
from __future__ import annotations

import sys

from harness import ChatSession
import strategies

BUDGET = 1500
SCENARIO = [
    "src/server.py'yi oku ve npm test çalıştır",
    "https://docs.local/guide sayfasını web_extract ile getir",
    "src/server.py'yi düzenle",
    "src/server.py'yi tekrar oku",                       # dedup + staleness tetikler
    "https://app.local/dashboard için snapshot al",
    "https://app.local/settings için snapshot al",       # eski snapshot bayat (gemini)
    "kod tabanında 'PORT' ara ve src/config.py'yi oku",
]


def build_trace() -> ChatSession:
    """Senaryoyu bir kez oynat, biriken trace'i döndür (compaction'sız ham hâliyle)."""
    sess = ChatSession(strategies.get("ours"), budget=10 ** 9)  # ham biriktir
    for msg in SCENARIO:
        sess.send(msg)
    sess.conv.reset_fates()
    return sess


def run_table(sess: ChatSession) -> None:
    conv = sess.conv
    raw = conv.raw_tokens()
    n = len(conv.all_results())
    print("═" * 78)
    print(f"KANONİK SENARYO — {len(SCENARIO)} tur, {n} tool çağrısı, ham {raw} tok, bütçe {BUDGET}")
    print("═" * 78)
    print(f"{'sistem':<13}{'§':<6}{'sıkışık':>8}{'tasarruf':>9}  yöntem / baskın kader")
    print("─" * 78)
    for s in strategies.all_strategies():
        conv.reset_fates()
        pre = s.compact(conv.all_results(), conv, BUDGET)
        shown = conv.shown_tokens(pre)
        pct = round(100 * (raw - shown) / raw) if raw else 0
        fates: dict[str, int] = {}
        for r in conv.all_results():
            if r.fate != "TAM":
                fates[r.fate] = fates.get(r.fate, 0) + 1
        dom = ", ".join(f"{k}×{v}" for k, v in sorted(fates.items(), key=lambda x: -x[1])) or "—"
        llm = " ᴸᴸᴹ" if s.uses_llm else ""
        print(f"{s.name:<13}{s.ref.split()[0]:<6}{shown:>8}{('%'+str(pct)):>9}  {dom}{llm}")
    print("─" * 78)
    print("kader: KES=uçtan/ortadan kes · ÖZET=tek-satır/LLM · MASKE=placeholder · GİZLE=depoda-tut")
    print("       SİL=kaldır · KATLA=outline · SUPERSEDE=bayat-snapshot · DEDUP=tekrar · CRUSH=algoritmik")
    print("ᴸᴸᴹ = son çare/adım LLM kullanır (yeşil olmayan). Diğerleri saf deterministik.")
    print("═" * 78)


def run_detail(sess: ChatSession) -> None:
    conv = sess.conv
    for s in strategies.all_strategies():
        conv.reset_fates()
        pre = s.compact(conv.all_results(), conv, BUDGET)
        print("\n" + "─" * 78)
        print(f"▶ {s.name}  ({s.repo}, {s.ref}) — {s.blurb}")
        if pre:
            print(f"  preamble: {pre.splitlines()[0]}")
        for r in conv.all_results():
            if r.fate == "TAM":
                continue
            print(f"    {r.tool_type:<13} {r.resource[:28]:<28} {r.fate:<9} "
                  f"{r.raw_tokens():>4}→{r.shown_tokens():<3} ← {r.note}")


def main() -> None:
    sess = build_trace()
    run_table(sess)
    if "--detail" in sys.argv:
        run_detail(sess)


if __name__ == "__main__":
    main()
