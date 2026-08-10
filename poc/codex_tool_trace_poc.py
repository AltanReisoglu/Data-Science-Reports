#!/usr/bin/env python3
"""
Codex tool-trace compaction — OLAY-GÜDÜMLÜ POC (tek dosya, stdlib).

Kaynak: codex-rs/{utils/output-truncation, utils/string/truncate.rs,
        core/src/compact.rs, core/src/compact_remote.rs, protocol/compacted_item.rs}

Tool'lar tetiklendikçe Codex'in üç aşaması SIRAYLA ortaya çıkar:
  A) Katman A — her tool çıktısında ORTADAN-KESME (truncate_middle, LLM'siz)
       baş+son korunur, orta atılır, "Warning: truncated output" başlığı.
  B1) Fit-to-window trim — history pencereye sığmayınca, en ESKİ function-call
       çıktılarını placeholder'a çevir (dinamik; eşik = context window). Yakını korur.
  B2) Model-turn compaction — trim yetmeyince: SUMMARIZATION_PROMPT ile handoff
       özeti üret, user mesajları + özet + world_state ile yeni PENCERE (windowing).

Çalıştır:  python3 codex_tool_trace_poc.py
"""
from __future__ import annotations
from typing import Any

try:
    import llm  # gerçek LLM handoff özeti (poc/llm.py; .env'den, anahtar yazdırılmaz)
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

# ---- Codex ayarları (POC ölçeğinde küçültülmüş) ----
CONTEXT_WINDOW = 30_000
TOOL_BUDGET_TOKENS = 5_000     # Katman A: tek çıktı bu token'ı aşarsa ortadan kes
TRIM_THRESHOLD = 20_000        # fit-to-window trim tetiği
COMPACT_THRESHOLD = 28_000     # model-turn compaction tetiği
PLACEHOLDER = "[context window truncated output]"


def est(t: str) -> int:
    return max(0, len(t) // 4)


# ---------- Katman A: truncate_middle (baş+son tut, ortayı at) ----------
def truncate_middle(text: str, budget_tokens: int) -> tuple[str, bool]:
    if est(text) <= budget_tokens:
        return text, False
    keep = budget_tokens * 4               # toplam korunacak karakter
    head = text[: keep // 2]
    tail = text[-keep // 2:]
    orig_tok = est(text)
    lines = text.count("\n") + 1
    out = (f"Warning: truncated output (original token count: {orig_tok:,})\n"
           f"Total output lines: {lines}\n\n"
           f"{head}\n...[orta atlandı]...\n{tail}")
    return out, True


# ---------- history modeli ----------
def item_tokens(it: dict) -> int:
    if it["kind"] == "function_output":
        return est(PLACEHOLDER) if it["placeholder"] else est(it["output"])
    return est(it.get("text", "")) + est(it.get("tool", ""))


class CodexSession:
    def __init__(self):
        self.history: list[dict] = []
        self.window_number = 0
        self.compacted: list[dict] = []
        self.log: list[str] = []
        self.raw_total = 0   # hiç sıkıştırma olmasaydı context (tüm ham çıktılar toplamı)
        self.peak = 0        # ulaşılan en yüksek GERÇEK context (compaction öncesi zirve)

    def total(self) -> int:
        return sum(item_tokens(it) for it in self.history) + 3

    # --- olay: mesaj ekle ---
    def user(self, text):
        self.raw_total += est(text) + 3
        self.history.append({"kind": "message", "role": "user", "text": text})
        self._manage(f'user("{text[:30]}")')

    def assistant(self, text):
        self.raw_total += est(text) + 3
        self.history.append({"kind": "message", "role": "assistant", "text": text})
        self._manage(f'assistant("{text[:30]}")')

    # --- olay: tool tetikle (Katman A burada) ---
    def tool(self, name, output, call_id):
        self.raw_total += est(output) + est(name) + 3   # HAM çıktı (kesme öncesi)
        self.history.append({"kind": "function_call", "tool": name, "call_id": call_id, "text": name})
        shown, truncated = truncate_middle(output, TOOL_BUDGET_TOKENS)
        self.history.append({"kind": "function_output", "call_id": call_id,
                             "output": shown, "placeholder": False, "warned": truncated})
        tag = f"[A: ortadan-kes {est(output):,}→{est(shown):,}t +Warning]" if truncated else ""
        self._manage(f'tool {name}() {est(shown):,}t {tag}')

    # --- her olaydan sonra: hangi aşama? ---
    def _manage(self, ev):
        line = f"  » {ev:<52} total={self.total():,}"
        self.log.append(line)
        self.peak = max(self.peak, self.total())
        if self.total() > COMPACT_THRESHOLD:
            self._compact()
        elif self.total() > TRIM_THRESHOLD:
            self._trim()

    # --- B1: fit-to-window trim ---
    def _trim(self):
        # son user mesajının indeksini bul (o turn'ü koru)
        last_user = max((i for i, it in enumerate(self.history)
                         if it["kind"] == "message" and it["role"] == "user"), default=-1)
        # sondan başa yürü; TRIM altına inene dek ESKİ (last_user öncesi) çıktıları placeholder yap
        reclaimed = 0
        rewritten = 0
        for i in range(len(self.history) - 1, -1, -1):
            if self.total() <= TRIM_THRESHOLD:
                break
            if i >= last_user:            # son turn'ü koru (yakınlık)
                continue
            it = self.history[i]
            if it["kind"] == "function_output" and not it["placeholder"]:
                before = item_tokens(it)
                it["placeholder"] = True
                reclaimed += before - item_tokens(it)
                rewritten += 1
        if rewritten:
            self.log.append(f"     └─ [B1 TRIM] {rewritten} eski function-output → placeholder "
                            f"(−{reclaimed:,}t, dinamik: pencereye sığdır) → total={self.total():,}")
        else:
            self.log.append(f"     └─ [B1 TRIM] budanacak eski çıktı YOK → compaction gerekecek")
            if self.total() > COMPACT_THRESHOLD:
                self._compact()

    # --- B2: model-turn compaction (özet + windowing) ---
    def _compact(self):
        # build_compaction_initial_context (BeforeLastUserMessage: world_state)
        world_state = {"kind": "message", "role": "system",
                       "text": "[world_state: auth.py değişti · git dirty · pytest 47/3]"}
        # user mesajlarını topla (KORUNUR), özet üret (mock handoff)
        user_msgs = [it for it in self.history if it["kind"] == "message" and it["role"] == "user"]
        tools_used = [it["tool"] for it in self.history if it["kind"] == "function_call"]
        _fallback = (f"[HANDOFF ÖZET: {len(tools_used)} tool çağrısı ({', '.join(dict.fromkeys(tools_used))}); "
                     f"user hedefleri korundu (mock)]")
        _trace = []
        for it in self.history:
            if it["kind"] == "message":
                _trace.append(f"{it['role']}: {it.get('text','')[:300]}")
            elif it["kind"] == "function_call":
                _trace.append(f"tool çağrısı: {it['tool']}")
            elif it["kind"] == "function_output" and not it.get("placeholder"):
                _trace.append(f"çıktı: {it.get('output','')[:800]}")
        summary = {"kind": "message", "role": "summary",
                   "text": _llm_handoff("\n".join(_trace), _fallback,
                       "Codex ajanının geçmişini sonraki pencereye devretmek için handoff özeti üret. "
                       "Yapılan tool çağrıları, ulaşılan sonuçlar ve kullanıcı hedeflerini 2-3 cümlede Türkçe özetle. "
                       "Sadece özeti döndür.", "HANDOFF ÖZET")}
        # build_compacted_history: world_state + özet + user mesajları.
        # function_call/output DÜŞÜRÜLÜR (özet zaten onları içeriyor) — gerçek reset.
        new_history = [world_state, summary] + user_msgs
        # son mesaj assistant ise onu koru (turn kapanışı)
        if self.history and self.history[-1]["kind"] == "message" and self.history[-1]["role"] == "assistant":
            new_history.append(self.history[-1])
        self.window_number += 1
        self.compacted.append({"window_number": self.window_number,
                               "previous_window_id": self.window_number - 1,
                               "replaced_items": len(self.history) - len(new_history)})
        before = self.total()
        self.history = new_history
        self.log.append(f"     └─ [B2 COMPACTION] model-turn handoff özeti → window #{self.window_number} "
                        f"(function-output'lar düşürüldü, user+özet kaldı) → total={before:,}→{self.total():,}")


# ================= DEMO (tool'lar tetiklendikçe) =================
def _blob(tok, tag):
    return "\n".join(f"{tag} kod satırı {i}" for i in range(tok * 4 // 22))


def main():
    print("=" * 78)
    print("CODEX TOOL-TRACE COMPACTION — OLAY-GÜDÜMLÜ POC")
    print(f"context_window={CONTEXT_WINDOW:,} · tool_budget={TOOL_BUDGET_TOKENS:,} · "
          f"trim@{TRIM_THRESHOLD:,} · compact@{COMPACT_THRESHOLD:,}")
    print("=" * 78)

    s = CodexSession()
    # --- Turn 1: büyük çıktılar → Katman A ---
    s.user("auth modülünü refactor et")
    s.tool("shell", _blob(18000, "pytest"), "c1")       # 18K → Katman A ortadan-kes
    s.tool("read_file", _blob(20000, "auth.py"), "c2")  # 20K → Katman A
    s.tool("read_file", _blob(5000, "config.py"), "c3")
    s.tool("grep", _blob(5000, "login-eşleşme"), "c4")
    # --- Turn 2: birikim → TRIM (eski→placeholder) ---
    s.user("login()'i sadeleştir")
    s.tool("read_file", _blob(5000, "login.py"), "c5")
    s.tool("read_file", _blob(5000, "session.py"), "c6")
    s.tool("read_file", _blob(5000, "token.py"), "c7")
    s.tool("read_file", _blob(5000, "cookie.py"), "c8")
    s.tool("read_file", _blob(5000, "mw.py"), "c9")
    s.tool("read_file", _blob(5000, "route.py"), "c10")   # trim tükenir + eşik aşılır → COMPACTION #1
    # --- Turn 3: yeni birikim → ikinci pencere ---
    s.user("testleri düzelt")
    s.tool("read_file", _blob(5000, "test_login.py"), "c11")
    s.tool("read_file", _blob(5000, "test_token.py"), "c12")
    s.tool("read_file", _blob(5000, "test_mw.py"), "c13")
    s.tool("bash", _blob(6000, "pytest-run"), "c14")
    s.tool("read_file", _blob(5000, "conftest.py"), "c15")
    s.tool("read_file", _blob(5000, "fixtures.py"), "c16")  # → COMPACTION #2 (window zinciri)
    s.assistant("Testler düzeltildi, hepsi geçiyor.")

    print("\nOLAY AKIŞI (tool tetiklendikçe aşamalar):")
    print("\n".join(s.log))

    print("\n── SON DURUM " + "─" * 60)
    for i, it in enumerate(s.history):
        if it["kind"] == "function_output":
            tag = "placeholder" if it["placeholder"] else ("truncated+warn" if it.get("warned") else "tam")
            print(f"  #{i} function_output[{it['call_id']}] {item_tokens(it):>5}t  ({tag})")
        elif it["kind"] == "function_call":
            print(f"  #{i} function_call    {it['tool']}")
        else:
            print(f"  #{i} {it['role']:<9} {item_tokens(it):>5}t  {it['text'][:44]!r}")
    print(f"\n  window sayısı (CompactedItem zinciri): {s.window_number}")
    for w in s.compacted:
        print(f"    window #{w['window_number']} ← replaced {w['replaced_items']} öğe (resume edilebilir)")
    _final = s.total(); _saved = s.raw_total - _final
    print(f"\n── KAÇTAN KAÇA " + "─" * 56)
    print(f"  ham (hiç sıkıştırma olmasaydı) ≈ {s.raw_total:,} token")
    print(f"  pik gerçek context (compaction'dan hemen önce) : {s.peak:,} token")
    print(f"  final context : {_final:,} token")
    print(f"  → {s.raw_total:,} → {_final:,} token  |  −{_saved:,} ({_saved/s.raw_total*100:.1f}% kazanç, {s.window_number} window ile)")
    print("=" * 78)
    print("\nAŞAMA ÖZETİ:")
    print("  A  Katman A ortadan-kesme  → c1,c2 üretilirken (18K/20K → 5K + Warning)")
    print("  B1 fit-to-window trim      → birikince eski function-output'lar placeholder")
    print("  B2 model-turn compaction   → trim yetmeyince handoff özeti + yeni window")


if __name__ == "__main__":
    main()
