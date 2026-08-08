"""
chat.py — Tek chat, çok mimari. Her mesajda SEÇİLİ sistemin tool-trace compaction
mantığı AYNI trace üzerinde çalışır. Sistemler kendi orijinal repolarına birebir sadık.

  python chat.py                 # varsayılan strateji: ours
  python chat.py --strategy cline

Komutlar (sohbet içinde):
  /strategy <isim>   seçili sistemi değiştir (trace aynı kalır, yeniden sıkışır)
  /list              tüm sistemler + hangi repo + § + tek-cümle mantık
  /trace             arkadaki gerçek durum — her tool'un KADER'i (TAM/KES/ÖZET/...)
  /view              modelin GERÇEKTE gördüğü sıkışık bağlam (preamble + gövdeler)
  /compare           mevcut trace'i TÜM sistemlerden geçir, tasarruf tablosu
  /budget <n>        token bütçesini değiştir
  /reset             sohbeti sıfırla
  /quit

Örnek akış (mekanizmaları tetikler):
  sen › src/server.py'yi oku ve npm test çalıştır
  sen › src/server.py'yi düzenle
  sen › src/server.py'yi tekrar oku        (→ bizim: bayat; cline: dedup)
  sen › https://app.local/x için snapshot al
  sen › https://app.local/y için snapshot al   (→ gemini: eski snapshot supersede)
"""
from __future__ import annotations

import sys

from harness import ChatSession
import llm
import strategies

BAR = "─" * 74
# kader → kısa işaret
_MARK = {"TAM": "·", "KES": "✂", "ÖZET": "▷", "MASKE": "▒", "GİZLE": "◌",
         "SİL": "✗", "KATLA": "⊟", "SUPERSEDE": "⟳", "DEDUP": "≡", "CRUSH": "⊙"}


def recompact(sess: ChatSession) -> None:
    """Trace'i AYNI bırakıp aktif strateji/bütçeyle yeniden sıkıştır (switch sonrası /trace doğru olsun)."""
    sess.conv.reset_fates()
    if sess.conv.all_results():
        sess.last_preamble = sess.strategy.compact(
            sess.conv.all_results(), sess.conv, sess.budget)


def make_session(name: str, budget: int, mock: bool, toolset: str = "generic"):
    """Canlı LLM varsa gerçek tool-use ajanı; yoksa (ya da --mock) deterministik scripted.

    toolset: 'generic' (6 mock tool) | 'product' (gerçek 119 ürün tool'u — sadece canlı).
    Motor: generic-canlı → LangGraph (varsayılan); product/yedek → manuel loop."""
    strat = strategies.get(name)
    if not mock and llm.available():
        from engines import make_live_agent
        agent, engine = make_live_agent(strat, budget, toolset)
        return agent, f"canlı LLM (gemma · {toolset} · motor: {engine})"
    # mock: ScriptedBrain yalnız generic tool'ları planlar; product canlı gerektirir
    return ChatSession(strat, budget), "mock (deterministik scripted brain · generic)"


def banner(sess, mode: str) -> None:
    s = sess.strategy
    print(BAR)
    print(f"TOOL-TRACE COMPACTION POC · aktif sistem: {s.name}  ({s.repo}, {s.ref})")
    print(f"  {s.blurb}")
    print(f"  ajan modu: {mode} · bütçe={sess.budget} tok · "
          f"strateji-LLM={'evet' if s.uses_llm else 'hayır (saf DET)'}")
    print("Komutlar: /strategy <isim>  /list  /trace  /view  /compare  /budget <n>  /reset  /quit")
    print(BAR)


def cmd_list() -> None:
    print(BAR)
    print(f"{'sistem':<13}{'§':<7}{'LLM':<5}repo / tool-trace mantığı")
    for s in strategies.all_strategies():
        print(f"{s.name:<13}{s.ref.split()[0]:<7}{'mor' if s.uses_llm else 'yeşil':<5}{s.repo}")
        print(f"{'':<25}{s.blurb}")
    print(BAR)


def cmd_trace(sess: ChatSession) -> None:
    print(BAR)
    print(f"TRACE — arkadaki gerçek kader ({sess.strategy.name}):")
    for r in sess.conv.all_results():
        mark = _MARK.get(r.fate, "?")
        saved = r.raw_tokens() - r.shown_tokens()
        note = f"  ← {r.note}" if r.note else ""
        print(f"  {mark} t{r.turn} {r.tool_type:<13} {r.resource[:26]:<26} "
              f"{r.fate:<9} {r.raw_tokens():>4}→{r.shown_tokens():<4} tok (−{saved}){note}")
    raw = sess.conv.raw_tokens()
    shown = sess.conv.shown_tokens(sess.last_preamble)
    pct = round(100 * (raw - shown) / raw) if raw else 0
    print(f"  ── ham {raw} tok → sıkışık {shown} tok  (%{pct} tasarruf) · bütçe {sess.budget}")
    if sess.last_preamble:
        print(f"  preamble: {sess.last_preamble.splitlines()[0][:70]}…")
    print(BAR)


def cmd_view(sess: ChatSession) -> None:
    print(BAR)
    print(f"MODELİN GÖRDÜĞÜ BAĞLAM ({sess.strategy.name}) — sıkıştırma sonrası:")
    if sess.last_preamble:
        print("‹preamble›")
        for ln in sess.last_preamble.splitlines():
            print(f"  {ln}")
    for r in sess.conv.all_results():
        shown = r.shown()
        if shown == "":
            continue  # toplu özete erimiş (preamble'da)
        body = shown if len(shown) <= 160 else shown[:160] + " …"
        print(f"‹{r.tool_type}:{r.resource}› [{r.fate}]")
        print(f"  {body}")
    print(BAR)


def cmd_compare(sess: ChatSession) -> None:
    """Mevcut trace'i TÜM sistemlerden geçir — aynı hammadde, farklı mantık."""
    conv = sess.conv
    if not conv.all_results():
        print("  (önce birkaç mesaj yaz ki trace birikssin)"); return
    raw = conv.raw_tokens()
    print(BAR)
    print(f"KARŞILAŞTIRMA — aynı trace ({len(conv.all_results())} tool, ham {raw} tok), "
          f"bütçe {sess.budget}:")
    print(f"  {'sistem':<13}{'sıkışık':>8}{'tasarruf':>9}  baskın kader(ler)")
    rows = []
    for s in strategies.all_strategies():
        conv.reset_fates()
        pre = s.compact(conv.all_results(), conv, sess.budget)
        shown = conv.shown_tokens(pre)
        pct = round(100 * (raw - shown) / raw) if raw else 0
        fates: dict[str, int] = {}
        for r in conv.all_results():
            if r.fate != "TAM":
                fates[r.fate] = fates.get(r.fate, 0) + 1
        dom = ", ".join(f"{k}×{v}" for k, v in sorted(fates.items(), key=lambda x: -x[1])) or "—"
        rows.append((s.name, shown, pct, dom, s.uses_llm))
    for name, shown, pct, dom, llm in rows:
        tag = " (LLM)" if llm else ""
        print(f"  {name:<13}{shown:>8}{('%'+str(pct)):>9}  {dom}{tag}")
    # aktif stratejiyi geri uygula
    conv.reset_fates()
    sess.last_preamble = sess.strategy.compact(conv.all_results(), conv, sess.budget)
    print(BAR)


def main() -> None:
    argv = sys.argv[1:]
    name = "ours"
    if "--strategy" in argv:
        name = argv[argv.index("--strategy") + 1]
    mock = "--mock" in argv
    toolset = "product" if "--product" in argv else "generic"
    sess, mode = make_session(name, 1500, mock, toolset)
    banner(sess, mode)
    if toolset == "product" and (mock or not llm.available()):
        print("  ⚠ product tool'ları (119 gerçek) sadece CANLI LLM ile çalışır → generic mock'a düşüldü.")
    if not mock and not llm.available():
        print(f"  ⚠ canlı LLM yok ({llm.why_unavailable()}) → mock moda düşüldü. "
              f".env'de LLM_* dolu mu?")
    print("Örnek: \"src/server.py'yi oku ve npm test çalıştır\"  ·  sonra /trace, /compare")

    while True:
        try:
            user = input("\nsen › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\ngörüşürüz."); break
        if not user:
            continue
        low = user.lower()
        if low in ("/quit", "/exit", "quit", "exit"):
            print("görüşürüz."); break
        if low == "/list": cmd_list(); continue
        if low == "/trace": cmd_trace(sess); continue
        if low == "/view": cmd_view(sess); continue
        if low == "/compare": cmd_compare(sess); continue
        if low == "/reset":
            sess, _ = make_session(sess.strategy.name, sess.budget, mock, toolset)
            print("  (sohbet sıfırlandı)"); continue
        if low.startswith("/budget"):
            try:
                sess.budget = int(user.split()[1]); recompact(sess)
                print(f"  (bütçe → {sess.budget} tok; trace yeniden sıkıştırıldı)")
            except (IndexError, ValueError):
                print("  kullanım: /budget 1200")
            continue
        if low.startswith("/strategy"):
            try:
                sess.set_strategy(strategies.get(user.split()[1]))
                recompact(sess)  # aynı trace, yeni mantıkla yeniden sıkışsın
                print(f"  (aktif sistem → {sess.strategy.name}: {sess.strategy.blurb})")
            except (IndexError, KeyError) as e:
                print(f"  {e}")
            continue

        out = sess.send(user)
        print(f"\nasistan › {out['answer']}")
        fates = ", ".join(f"{k}×{v}" for k, v in out["fates"].items() if k != "TAM")
        print(f"  ┄ trace: {out['units']} tool · ham {out['raw_tokens']}→sıkışık "
              f"{out['shown_tokens']} tok (%{out['saved_pct']} tasarruf)"
              f"{' · ' + fates if fates else ''}")


if __name__ == "__main__":
    main()
