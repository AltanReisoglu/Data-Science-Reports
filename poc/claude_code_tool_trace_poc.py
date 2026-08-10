#!/usr/bin/env python3
"""
Claude Code tool-trace compaction — OLAY-GÜDÜMLÜ POC (tek dosya, stdlib).

NOT: Claude Code KAPALI kaynaktır. Bu POC docs + bu oturumda BİREBİR gözlemlenen
davranışa dayanır (93KB'lık WebFetch çıktısının diske dökülmesi). Sabit değerler
yaklaşıktır (gözlem/tahmin), fonksiyon adları yok.

Tool'lar tetiklendikçe Claude Code'un üç mekanizması ortaya çıkar:
  A) Microcompaction — büyük tool çıktısı DİSKE yazılır, context'e önizleme+referans
       ("Full output saved to: .../tool-results/...txt"). [gözlem]
  B) Auto-compaction — context eşiğe gelince eski turn'ler KONUŞMA ÖZETİne iner.
       PreCompact/PostCompact hook'ları sarar. [docs+gözlem]
  C) Subagent kaçış yolu — büyük yan-iş AYRI context penceresinde koşar, ana
       pencereye sadece ÖZET döner (sıkıştırmaya alternatif). [docs]

Çalıştır:  python3 claude_code_tool_trace_poc.py
"""
from __future__ import annotations

try:
    import llm  # gerçek LLM konuşma-özeti (poc/llm.py; .env'den, anahtar yazdırılmaz)
except Exception:
    llm = None


def _llm_handoff(trace_text, fallback, sysprompt, tag):
    """llm.available() ise GERÇEK LLM özeti, değilse fallback (mock)."""
    if llm is not None and llm.available():
        try:
            r = llm.chat([{"role": "system", "content": sysprompt},
                          {"role": "user", "content": trace_text[:6000]}],
                         max_tokens=200, temperature=0.2)
            s = (r.get("content") or "").strip()
            if s:
                return f"[{tag} (GERÇEK LLM · {llm.MODEL}): {s}]"
        except Exception:
            return fallback
    return fallback


# ---- yaklaşık ayarlar (gözlem/tahmin; kapalı kaynak) ----
WINDOW = 30_000
MICRO_TOKENS = 4_000        # tek tool çıktısı bunu aşarsa diske dök (gözlem: ~93KB→önizleme)
PREVIEW_TOKENS = 500        # context'te kalan önizleme (~2KB gözlendi)
AUTO_COMPACT_AT = 24_000    # 0.8×window: konuşma özeti tetiği
KEEP_RECENT_TURNS = 2       # auto-compaction'da korunan son turn


def est(t: str) -> int:
    return max(0, len(t) // 4)


def item_tokens(it: dict) -> int:
    return est(it.get("text", ""))


class ClaudeCodeSession:
    def __init__(self):
        self.history: list[dict] = []
        self.disk: dict[str, str] = {}      # microcompaction: diske dökülenler
        self.log: list[str] = []
        self.compactions = 0
        self.subagent_runs = 0
        self.raw_total = 0   # hiç sıkıştırma/kaçış olmasaydı context (tüm ham çıktılar + subagent işi)
        self.peak = 0        # ulaşılan en yüksek GERÇEK ana context

    def total(self) -> int:
        return sum(item_tokens(it) for it in self.history) + 3

    # --- olay: mesaj ---
    def user(self, text):
        self.raw_total += est(text) + 3
        self.history.append({"kind": "user", "text": text})
        self._after(f'user("{text[:30]}")')

    def assistant(self, text):
        self.raw_total += est(text) + 3
        self.history.append({"kind": "assistant", "text": text})
        self._after(f'assistant("{text[:28]}")')

    # --- olay: tool (A: microcompaction burada) ---
    def tool(self, name, output, call_id):
        toks = est(output)
        self.raw_total += toks + est(name) + 3   # HAM çıktı (microcompaction öncesi)
        if toks > MICRO_TOKENS:
            # microcompaction: diske yaz, context'e önizleme + referans
            path = f".claude/projects/.../tool-results/{call_id}.txt"
            self.disk[path] = output
            preview = output[: PREVIEW_TOKENS * 4]
            shown = (f"{preview}\n...\n[Output too large ({toks*4//1024}KB). "
                     f"Full output saved to: {path}]")
            self.history.append({"kind": "tool", "call_id": call_id, "text": shown, "microcompacted": True})
            self._after(f'tool {name}() {toks:,}t → [A: MICRO diske, context {est(shown):,}t + referans]')
        else:
            self.history.append({"kind": "tool", "call_id": call_id, "text": output, "microcompacted": False})
            self._after(f'tool {name}() {toks:,}t (context\'te kalır)')

    # --- olay: subagent kaçış yolu (C) ---
    def subagent(self, goal, inner_tool_tokens):
        """Büyük yan-iş AYRI pencerede koşar; ana pencereye SADECE özet döner."""
        self.subagent_runs += 1
        self.raw_total += inner_tool_tokens   # kaçış olmasaydı ana context'e girecek iş
        # ana context'e ara adımlar GİRMEZ; sadece özet
        summ = f"[subagent özeti: '{goal}' → {inner_tool_tokens:,}t'lık iş ayrı pencerede yapıldı, damıtıldı]"
        self.history.append({"kind": "assistant", "text": summ})
        self._after(f'Task(subagent) "{goal[:24]}" — {inner_tool_tokens:,}t AYRI pencerede, '
                    f'ana context\'e sadece {est(summ):,}t özet')

    def _after(self, ev):
        self.log.append(f"  » {ev:<62} total={self.total():,}")
        self.peak = max(self.peak, self.total())
        if self.total() > AUTO_COMPACT_AT:
            self._auto_compact()

    # --- B: auto-compaction ---
    def _auto_compact(self):
        self.log.append(f"     └─ [PreCompact hook] compaction başlıyor (bloklanabilir)")
        # son KEEP_RECENT_TURNS turn'ü koru; öncesini konuşma özetine indir
        # turn = user mesajıyla başlar
        user_idx = [i for i, it in enumerate(self.history) if it["kind"] == "user"]
        if len(user_idx) <= KEEP_RECENT_TURNS:
            keep_from = 0
        else:
            keep_from = user_idx[-KEEP_RECENT_TURNS]
        old = self.history[:keep_from]
        # anti-thrash: kazanılacak eski mesaj yoksa compaction yapma (boşuna tetikleme)
        if not old:
            self.log.append(f"     └─ [anti-thrash] korunan turn'ler zaten eşiği dolduruyor → "
                            f"compaction yer açamaz (gerçekte: 'yeni thread aç' uyarısı)")
            return
        recent = self.history[keep_from:]
        tools_in_old = sum(1 for it in old if it["kind"] == "tool")
        _fallback = (f"[Konuşma özeti: önceki {len(old)} mesaj ({tools_in_old} tool) damıtıldı — "
                     f"ilerleme, kararlar, kalan iş (auto-compaction)]")
        _trace = [f"{it.get('kind')}: {(it.get('text','') or '')[:400]}" for it in old]
        summary = {"kind": "summary",
                   "text": _llm_handoff("\n".join(_trace), _fallback,
                       "Claude Code oturumunun eski kısmını damıt: ilerleme, alınan kararlar ve kalan işi "
                       "2-3 cümlede Türkçe özetle (auto-compaction). Sadece özeti döndür.", "Konuşma özeti")}
        before = self.total()
        self.history = [summary] + recent
        self.compactions += 1
        self.log.append(f"     └─ [B AUTO-COMPACTION] {len(old)} eski mesaj → konuşma özeti, "
                        f"son {KEEP_RECENT_TURNS} turn korundu → total={before:,}→{self.total():,}")
        self.log.append(f"     └─ [PostCompact hook] compaction bitti (bildirim)")


# ================= DEMO =================
def _blob(tok, tag):
    return "\n".join(f"{tag} satır {i}" for i in range(tok * 4 // 18))


def main():
    print("=" * 80)
    print("CLAUDE CODE TOOL-TRACE COMPACTION — OLAY-GÜDÜMLÜ POC (kapalı kaynak → gözlem)")
    print(f"window={WINDOW:,} · micro@{MICRO_TOKENS:,} · preview={PREVIEW_TOKENS} · auto-compact@{AUTO_COMPACT_AT:,}")
    print("=" * 80)

    s = ClaudeCodeSession()
    # --- Turn 1: büyük tool çıktısı → microcompaction ---
    s.user("bu dokümanı çek ve özetle")
    s.tool("WebFetch", _blob(23000, "docs"), "toolu_01")     # 93KB gibi → MICRO diske
    s.tool("Read", _blob(3000, "config.py"), "toolu_02")     # küçük → context'te
    s.assistant("Doküman özetlendi.")
    # --- Turn 2: büyük yan-iş → subagent kaçış yolu ---
    s.user("40 dosyayı tara, login akışını bul")
    s.subagent("40 dosyada login akışını tara", inner_tool_tokens=80_000)   # AYRI pencere
    s.assistant("login akışı auth/login.py:45'te.")
    # --- Turn 3-5: KÜÇÜK okumalar (micro eşiği altında) birikir → AUTO-COMPACTION ---
    s.user("test_a/b/c'yi oku")
    for n, cid in [("test_a", "t03"), ("test_b", "t04"), ("test_c", "t05")]:
        s.tool("Read", _blob(3500, n), cid)      # 3.5K < 4K micro → context'te KALIR → birikir
    s.user("test_d/e/f'yi oku")
    for n, cid in [("test_d", "t06"), ("test_e", "t07"), ("test_f", "t08")]:
        s.tool("Read", _blob(3500, n), cid)
    s.user("test_g'yi oku")
    s.tool("Read", _blob(3500, "test_g"), "t09")  # eşik aşılır → AUTO-COMPACTION (eski turn'ler özete)
    s.assistant("Testler okundu.")

    print("\nOLAY AKIŞI (tool tetiklendikçe mekanizmalar):")
    print("\n".join(s.log))

    print("\n── SON DURUM (ana context) " + "─" * 50)
    for i, it in enumerate(s.history):
        k = it["kind"]
        extra = " [MICRO→disk referansı]" if it.get("microcompacted") else ""
        print(f"  #{i} {k:<9} {item_tokens(it):>5}t  {it['text'][:46]!r}{extra}")
    print(f"\n  diske dökülen (microcompaction): {len(s.disk)} dosya")
    for p in s.disk:
        print(f"    {p}  ({est(s.disk[p]):,}t tam içerik diskte)")
    print(f"  auto-compaction sayısı : {s.compactions}")
    print(f"  subagent kaçışı        : {s.subagent_runs} (ara adımlar ana context'e HİÇ girmedi)")
    print(f"  final ana context      : {s.total():,} token")
    _saved = s.raw_total - s.total()
    print("\n── KAÇTAN KAÇA " + "─" * 54)
    print(f"  ham (sıkıştırma+kaçış olmasaydı) ≈ {s.raw_total:,} token")
    print(f"  pik gerçek ana context           : {s.peak:,} token")
    print(f"  final ana context                : {s.total():,} token")
    print(f"  → {s.raw_total:,} → {s.total():,} token  |  −{_saved:,} ({_saved/s.raw_total*100:.1f}% kazanç)")
    print("=" * 80)
    print("\nMEKANİZMA ÖZETİ:")
    print("  A Microcompaction  → WebFetch 23K → diske, context'e ~500t önizleme+referans")
    print("  B Auto-compaction  → Turn 3 birikince eski → konuşma özeti (Pre/PostCompact hook)")
    print("  C Subagent kaçışı  → 80K'lık tarama AYRI pencerede, ana context'e sadece özet")


if __name__ == "__main__":
    main()
