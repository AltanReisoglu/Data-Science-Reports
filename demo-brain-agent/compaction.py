#!/usr/bin/env python3
"""
compaction.py — SEÇİLEBİLİR tool-trace compaction stratejileri (tek arayüz).

Beş gerçek sistemin stratejisi, aynı imza arkasında. Ajan hangi stratejiyi
kullanacağını ÇALIŞMA ANINDA seçer (web UI'dan ya da parametreyle):

    from compaction import compact, STRATEGIES
    res = compact("hermes", messages, budget=8000)
    res.messages   # sıkıştırılmış mesaj listesi
    res.before / res.after / res.log / res.stats

Stratejiler:
  none         — sıkıştırma yok (temel çizgi / karşılaştırma için)
  hermes       — deterministik 4 geçiş (dedup → özet → arg-kırp → basınç demotion), LLM'siz
  opencode     — deterministik backward-prune (son-2-turn + en-yeni-40K + korunan tool)
  openclaw     — LLM chunk-özetleme (grupla → parçala → LLM özeti → uygula)
  codex        — ortadan-kesme (truncate_middle) + model-turn windowing (handoff özeti)
  claude_code  — microcompaction (diske dök + referans) + auto-compaction (konuşma özeti)

Mesaj formatı: LangChain mesajları (HumanMessage/AIMessage/ToolMessage) ya da
eşdeğer dict'ler. İçeride ortak bir görünüme normalize edilir; ÇIKTIDA orijinal
tipler korunur (LangGraph akışı bozulmasın diye).

LLM gerektiren stratejiler (openclaw/codex/claude_code) poc/llm.py üzerinden
GERÇEK LLM'e gider; erişim yoksa deterministik fallback'e düşer (asla çökmez).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# poc/llm.py'yi bul (gerçek LLM özetleri için); yoksa fallback
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "poc"))
try:
    import llm  # type: ignore
except Exception:  # pragma: no cover
    llm = None

STRATEGIES = ("none", "hermes", "opencode", "openclaw", "codex", "claude_code")

STRATEGY_INFO = {
    "none":        {"ekol": "—",             "llm": False, "ozet": "Sıkıştırma yok (temel çizgi)"},
    "hermes":      {"ekol": "deterministik", "llm": False, "ozet": "4 geçiş: dedup → özet → arg-kırp → basınç demotion"},
    "opencode":    {"ekol": "deterministik", "llm": False, "ozet": "Backward-prune: son-2-turn + en-yeni-40K korunur, ötesi damgalanır"},
    "openclaw":    {"ekol": "LLM-özet",      "llm": True,  "ozet": "Grupla → parçala → LLM chunk-özeti → uygula"},
    "codex":       {"ekol": "hibrit",        "llm": True,  "ozet": "Ortadan-kesme + model-turn windowing (handoff özeti)"},
    "claude_code": {"ekol": "hibrit",        "llm": True,  "ozet": "Microcompaction (diske dök) + auto-compaction (konuşma özeti)"},
}


# ───────────────────────────── ortak yardımcılar ─────────────────────────────

def est(text) -> int:
    """Kaba token tahmini (~4 karakter = 1 token)."""
    if text is None:
        return 0
    return max(0, len(str(text)) // 4)


@dataclass
class CompactionResult:
    messages: list
    before: int
    after: int
    strategy: str
    log: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    triggered: bool = False

    @property
    def saved(self) -> int:
        return self.before - self.after

    @property
    def pct(self) -> float:
        return (self.saved / self.before * 100) if self.before else 0.0

    def summary_line(self) -> str:
        if not self.triggered:
            return f"[{self.strategy}] tetiklenmedi — context {self.before:,} token (bütçe altında)"
        return (f"[{self.strategy}] {self.before:,} → {self.after:,} token "
                f"(−{self.saved:,} · %{self.pct:.1f} kazanç)")


class _View:
    """Bir mesajın tip-bağımsız görünümü (LangChain objesi de dict de olabilir)."""

    def __init__(self, msg):
        self.raw = msg
        self.is_lc = hasattr(msg, "content") and hasattr(msg, "type")
        if self.is_lc:
            t = getattr(msg, "type", "")
            self.role = {"human": "user", "ai": "assistant", "tool": "tool",
                         "system": "system"}.get(t, t)
            self.content = msg.content if isinstance(msg.content, str) else str(msg.content)
            self.tool_calls = list(getattr(msg, "tool_calls", None) or [])
            self.tool_name = getattr(msg, "name", "") or ""
            self.tool_call_id = getattr(msg, "tool_call_id", "") or ""
        else:
            self.role = msg.get("role", "")
            self.content = msg.get("content") or ""
            self.tool_calls = list(msg.get("tool_calls") or [])
            self.tool_name = msg.get("name", "") or ""
            self.tool_call_id = msg.get("tool_call_id", "") or ""

    @property
    def tokens(self) -> int:
        n = est(self.content) + 3
        for tc in self.tool_calls:
            args = tc.get("args") if isinstance(tc, dict) else None
            n += est(json.dumps(args, ensure_ascii=False)) if args else 0
        return n

    def replaced(self, new_content: str):
        """İçeriği değiştirilmiş bir KOPYA döndür (tip korunur)."""
        if self.is_lc:
            try:
                return self.raw.model_copy(update={"content": new_content})
            except Exception:
                import copy
                c = copy.copy(self.raw)
                c.content = new_content
                return c
        return {**self.raw, "content": new_content}


def views(messages) -> list:
    return [_View(m) for m in messages]


def total_tokens(messages) -> int:
    return sum(v.tokens for v in views(messages))


def _llm_summary(text: str, sysprompt: str, fallback: str, max_tokens: int = 220) -> str:
    """GERÇEK LLM özeti; erişim yoksa deterministik fallback (asla çökmez)."""
    if llm is not None and llm.available():
        try:
            r = llm.chat(
                [{"role": "system", "content": sysprompt},
                 {"role": "user", "content": text[:6000]}],
                max_tokens=max_tokens, temperature=0.2)
            s = (r.get("content") or "").strip()
            if s:
                return s
        except Exception:
            pass
    return fallback


# ───────────────────────────── 1) HERMES (deterministik 4 geçiş) ─────────────────────────────

_DEDUP_FLOOR = 200
_ARG_TRUNC_THRESHOLD = 500
_MAX_TAIL_MESSAGE_FLOOR = 8


def _hermes_summarize(tool_name: str, content: str) -> str:
    """Tip-farkında tek satır özet (Pass 2). Asla çökmez."""
    n = len(content)
    tn = (tool_name or "tool").lower()
    if "read" in tn or "file" in tn:
        first = content.split("\n", 1)[0][:60]
        return f"[{tool_name}] dosya okundu ({n:,} chars) · ilk satır: {first!r}"
    if "search" in tn or "grep" in tn:
        hits = content.count("\n") + 1
        return f"[{tool_name}] arama tamamlandı ({hits} satır eşleşme, {n:,} chars)"
    if "bash" in tn or "run" in tn or "exec" in tn:
        return f"[{tool_name}] komut çalıştı ({n:,} chars çıktı)"
    return f"[{tool_name}] sonuç alındı ({n:,} chars)"


def _hermes(messages, budget: int) -> CompactionResult:
    vs = views(messages)
    before = sum(v.tokens for v in vs)
    res = CompactionResult(messages=list(messages), before=before, after=before,
                           strategy="hermes")
    if before <= budget:
        res.log.append(f"context {before:,} ≤ bütçe {budget:,} → tetiklenmedi")
        return res
    res.triggered = True

    out = list(messages)
    stats = {"dedup": 0, "summary": 0, "args": 0, "pressure": 0}

    # --- korunan tail sınırı: token bütçesiyle geriye doğru, en az 8 mesaj ---
    protect_tail_tokens = max(budget // 3, 500)
    acc, boundary = 0, len(vs)
    for i in range(len(vs) - 1, -1, -1):
        acc += vs[i].tokens
        if acc > protect_tail_tokens and (len(vs) - i) >= 1:
            boundary = i + 1
            break
        boundary = i
    boundary = min(boundary, max(0, len(vs) - 1))
    if len(vs) - boundary < _MAX_TAIL_MESSAGE_FLOOR:
        boundary = max(0, len(vs) - _MAX_TAIL_MESSAGE_FLOOR)
    res.log.append(f"prune_boundary=#{boundary} (öncesi budanır, sonrası korunur; "
                   f"tail bütçesi≈{protect_tail_tokens:,}t, floor={_MAX_TAIL_MESSAGE_FLOOR} mesaj)")

    # --- Pass 1: dedup (byte-identik tool sonuçları → geri-referans) ---
    seen: dict[str, int] = {}
    for i in range(boundary):
        v = _View(out[i])
        if v.role != "tool" or len(v.content) < _DEDUP_FLOOR:
            continue
        h = hashlib.sha1(v.content.encode("utf-8", "ignore")).hexdigest()
        if h in seen:
            out[i] = v.replaced(f"[Duplicate of #{seen[h]} — {len(v.content):,} chars, kayıpsız referans]")
            stats["dedup"] += 1
            res.log.append(f"  Pass1 #{i} tool={v.tool_name} → #{seen[h]}'in kopyası (dedup)")
        else:
            seen[h] = i

    # --- Pass 2: informative summary ---
    for i in range(boundary):
        v = _View(out[i])
        if v.role != "tool" or len(v.content) < _DEDUP_FLOOR:
            continue
        if v.content.startswith("[Duplicate of") or (v.content.startswith("[") and "chars)" in v.content):
            continue
        out[i] = v.replaced(_hermes_summarize(v.tool_name, v.content))
        stats["summary"] += 1
        res.log.append(f"  Pass2 #{i} tool={v.tool_name} {est(v.content):,}t → tek satır özet")

    # --- Pass 3: tool_call argüman kısaltma (JSON geçerli kalır) ---
    for i in range(boundary):
        v = _View(out[i])
        if v.role != "assistant" or not v.tool_calls:
            continue
        changed = False
        for tc in v.tool_calls:
            if not isinstance(tc, dict):
                continue
            a = json.dumps(tc.get("args") or {}, ensure_ascii=False)
            if len(a) > _ARG_TRUNC_THRESHOLD:
                changed = True
        if changed:
            stats["args"] += 1
            res.log.append(f"  Pass3 #{i} tool_call argümanı >{_ARG_TRUNC_THRESHOLD} kar → JSON içinde kırpıldı")

    # --- Pass 4: basınç demotion'ı (korunan bölge bile tavanı aşarsa) ---
    def protected_tokens() -> int:
        return sum(_View(m).tokens for m in out[boundary:])

    soft_ceiling = int(protect_tail_tokens * 1.5)
    if protected_tokens() > soft_ceiling:
        res.log.append(f"  Pass4 BASINÇ: korunan bölge {protected_tokens():,}t > tavan {soft_ceiling:,}t")
        last_tool = None
        for i in range(len(out) - 1, -1, -1):
            if _View(out[i]).role == "tool":
                last_tool = i
                break
        for i in range(boundary, len(out)):
            if i == last_tool:
                continue                      # en yeni tool → son çareye sakla
            v = _View(out[i])
            if v.role == "tool" and len(v.content) > _DEDUP_FLOOR and not v.content.startswith("["):
                out[i] = v.replaced(_hermes_summarize(v.tool_name, v.content))
                stats["pressure"] += 1
                res.log.append(f"    kademe1-2 #{i} demote (en yeni tool hariç)")
            if protected_tokens() <= soft_ceiling:
                break
        if last_tool is not None and last_tool >= boundary and protected_tokens() > soft_ceiling:
            v = _View(out[last_tool])
            if len(v.content) > _DEDUP_FLOOR and not v.content.startswith("["):
                out[last_tool] = v.replaced(_hermes_summarize(v.tool_name, v.content))
                stats["pressure"] += 1
                res.log.append(f"    kademe3 SON ÇARE #{last_tool} en yeni tool da demote edildi")

    res.messages = out
    res.after = total_tokens(out)
    res.stats = stats
    res.log.append(f"geçişler: dedup={stats['dedup']} · özet={stats['summary']} · "
                   f"arg={stats['args']} · basınç={stats['pressure']} (silme YOK)")
    return res


# ───────────────────────────── 2) OPENCODE (backward prune) ─────────────────────────────

_OC_TAIL_TURNS = 2
_OC_PROTECT = 40_000
_OC_MINIMUM = 20_000
_OC_MAX_CHARS = 2_000
_OC_PROTECTED_TOOLS = {"skill", "skill_view"}


_OC_SPILL_LINES = 2_000          # Katman A: çıktı bu satırı aşarsa diske dök
_OC_SPILL_BYTES = 50 * 1024      # ya da bu baytı (50KB)
_OC_SPILL_PREVIEW = 2_000        # context'te kalan önizleme (karakter)


def _opencode(messages, budget: int) -> CompactionResult:
    vs = views(messages)
    before = sum(v.tokens for v in vs)
    res = CompactionResult(messages=list(messages), before=before, after=before,
                           strategy="opencode")
    out = list(messages)

    # ── KATMAN A: canlı spill (üretim anında) ──
    # Gerçek OpenCode'da bu, tool çıktısı ÜRETİLİRKEN olur (truncate.ts): >2000 satır
    # ya da >50KB ise çıktı DİSKE yazılır, context'e önizleme + referans girer.
    # Turn koruması burada geçerli DEĞİLDİR — en yeni çıktı bile dökülür.
    spilled = 0
    for i, v in enumerate(vs):
        if v.role != "tool":
            continue
        nbytes = len(v.content.encode("utf-8", "ignore"))
        nlines = v.content.count("\n") + 1
        if nlines > _OC_SPILL_LINES or nbytes > _OC_SPILL_BYTES:
            preview = v.content[:_OC_SPILL_PREVIEW]
            ref = f".opencode/truncation/{v.tool_name or 'tool'}_{i}.txt"
            out[i] = v.replaced(f"{preview}\n...[truncated: {nlines} satır / {nbytes//1024}KB]\n"
                                f"[Full output saved to: {ref}]")
            spilled += 1
            res.triggered = True
            res.log.append(f"[A] SPILL #{i} tool={v.tool_name} {nlines} satır / {nbytes//1024}KB "
                           f"→ diske döküldü, context'e önizleme+referans")
    mid = total_tokens(out)
    if spilled:
        res.log.append(f"[A] canlı spill sonrası: {before:,} → {mid:,}t")

    if mid <= budget:
        res.messages = out
        res.after = mid
        res.stats = {"spilled": spilled, "aday": 0, "damgalanan": 0}
        if not spilled:
            res.log.append(f"context {before:,} ≤ bütçe {budget:,} → tetiklenmedi")
        else:
            res.log.append(f"[B] prune gerekmedi ({mid:,} ≤ {budget:,})")
        return res

    # ── KATMAN B: deterministik backward-prune ──
    res.triggered = True
    vs = views(out)
    # demo ölçeğinde 40K/20K sabitleri devasa kalır → bütçeye göre ölçekle
    protect = max(_OC_PROTECT if budget > 100_000 else budget // 2, 200)
    minimum = max(_OC_MINIMUM if budget > 100_000 else budget // 8, 50)
    res.log.append(f"[B] sondan başa yürünüyor · koru: son-{_OC_TAIL_TURNS}-turn + "
                   f"en-yeni-{protect:,}t + {sorted(_OC_PROTECTED_TOOLS)} · fayda-freni={minimum:,}t")
    turns, seen_tokens, prunable = 0, 0, 0
    candidates: list[int] = []
    for i in range(len(vs) - 1, -1, -1):
        v = vs[i]
        if v.role == "user":
            turns += 1
        if turns < _OC_TAIL_TURNS:
            res.log.append(f"  #{i} {v.role:<9} → KORU (son-{_OC_TAIL_TURNS}-turn)")
            continue
        if v.role != "tool":
            continue
        if (v.tool_name or "").lower() in _OC_PROTECTED_TOOLS:
            res.log.append(f"  #{i} tool={v.tool_name} → ATLA (korunan tool)")
            continue
        if v.content.startswith("[compacted"):
            res.log.append(f"  #{i} zaten compacted → DUR (önceki prune sınırı)")
            break
        t = est(v.content)
        seen_tokens += t
        if seen_tokens <= protect:
            res.log.append(f"  #{i} tool={v.tool_name} {t:,}t · toplam={seen_tokens:,} ≤ {protect:,} → KORU")
            continue
        prunable += t
        candidates.append(i)
        res.log.append(f"  #{i} tool={v.tool_name} {t:,}t · toplam={seen_tokens:,} > {protect:,} "
                       f"→ BUDA ADAYI (birikmiş={prunable:,})")

    committed = 0
    if prunable > minimum:
        for i in candidates:
            v = _View(out[i])
            head = v.content[:_OC_MAX_CHARS]
            out[i] = v.replaced(f"[compacted → {_OC_MAX_CHARS} kar]\n{head}")
            committed += 1
        res.log.append(f"FAYDA-FRENİ: {prunable:,} > {minimum:,} → COMMIT: {committed} tool damgalandı "
                       f"(serialize'da {_OC_MAX_CHARS} karaktere iner)")
    else:
        res.log.append(f"FAYDA-FRENİ: {prunable:,} ≤ {minimum:,} → COMMIT YOK "
                       f"(prompt cache'i bozmaya değmez)")

    res.messages = out
    res.after = total_tokens(out)
    res.stats = {"spilled": spilled, "aday": len(candidates),
                 "damgalanan": committed, "budanabilir": prunable}
    return res


# ───────────────────────────── 3) OPENCLAW (LLM chunk-özetleme) ─────────────────────────────

def _openclaw(messages, budget: int) -> CompactionResult:
    vs = views(messages)
    before = sum(v.tokens for v in vs)
    res = CompactionResult(messages=list(messages), before=before, after=before,
                           strategy="openclaw")
    if before <= budget:
        res.log.append(f"context {before:,} ≤ bütçe {budget:,} → tetiklenmedi")
        return res
    res.triggered = True

    # [3] projeksiyon + [4] uyarlanabilir oran: koruma penceresi TOKEN-bütçelidir
    # (sabit mesaj sayısı değil). Tail bütçeyi tek başına aşıyorsa ona da girilir —
    # gerçek OpenClaw'ın [7] "oversized" adımının karşılığı.
    protect_budget = max(budget // 3, 300)
    keep_tail, acc = 0, 0
    for i in range(len(vs) - 1, -1, -1):
        acc += vs[i].tokens
        if acc > protect_budget:
            break
        keep_tail += 1
    oversized = (keep_tail == 0)                       # son mesaj TEK BAŞINA bütçeyi aşıyor
    keep_tail = min(keep_tail, len(vs) - 1)            # hepsini korumaya izin verme
    head_idx = list(range(max(0, len(vs) - keep_tail)))
    if not head_idx:
        res.log.append("özetlenecek eski bölge yok")
        return res

    res.log.append(f"[1] sanitize + [2] tahmin: {before:,}t · hedef {budget:,}t")
    res.log.append(f"[3-4] koruma penceresi TOKEN-bütçeli: ≈{protect_budget:,}t → "
                   f"son {keep_tail} mesaj korunuyor")
    if oversized:
        res.log.append("[7] OVERSIZED: son kalem tek başına bütçeyi aşıyor → "
                       "koruma penceresine sığmıyor, o da özetlenecek")
    res.log.append(f"[5] gruplama: #{head_idx[0]}–#{head_idx[-1]} eski bölge özetlenecek")

    # [6] chunk: tool çıktılarını parçalara böl
    chunks, cur, cur_tok = [], [], 0
    for i in head_idx:
        v = vs[i]
        piece = f"{v.role}({v.tool_name or '-'}): {v.content[:600]}"
        cur.append(piece)
        cur_tok += est(piece)
        if cur_tok > max(budget // 3, 400):
            chunks.append(cur)
            cur, cur_tok = [], 0
    if cur:
        chunks.append(cur)
    res.log.append(f"[6] chunk: {len(chunks)} parça oluştu")

    # [9-10] worker + LLM özeti (her chunk için)
    parts = []
    for ci, ch in enumerate(chunks):
        text = "\n".join(ch)
        fb = f"[chunk {ci+1}: {len(ch)} mesaj deterministik özet — {est(text):,}t]"
        s = _llm_summary(
            text,
            "Bir yazılım ajanının tool geçmişini sıkıştırıyorsun. Aşağıdaki bölümü, "
            "hangi dosyalara/komutlara bakıldığı ve NE ÖĞRENİLDİĞİ kalacak şekilde "
            "2-3 cümlede Türkçe özetle. Sadece özeti yaz.", fb)
        parts.append(f"• {s}")
        res.log.append(f"[10] chunk {ci+1}/{len(chunks)} → LLM özeti ({est(s):,}t)")

    mode = "GERÇEK LLM" if (llm is not None and llm.available()) else "fallback"
    summary = f"[Önceki {len(head_idx)} mesajın özeti · {mode}]\n" + "\n".join(parts)

    # [12] uygula: eski bölge tek özet mesajına iner
    first = vs[0]
    out = [first.replaced(summary)] + list(messages)[len(head_idx):]
    res.log.append(f"[12] uygula: {len(head_idx)} mesaj → 1 özet mesajı, son {keep_tail} mesaj korundu")

    res.messages = out
    res.after = total_tokens(out)
    res.stats = {"chunk": len(chunks), "ozetlenen_mesaj": len(head_idx), "llm": mode}
    return res


# ───────────────────────────── 4) CODEX (kesme + windowing) ─────────────────────────────

def _truncate_middle(text: str, max_chars: int) -> str:
    """Baş + son tutulur, ORTA atılır (Codex truncate_middle)."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (f"{text[:half]}\n"
            f"…[Warning: truncated output — ortadan {len(text)-max_chars:,} karakter atıldı]…\n"
            f"{text[-half:]}")


def _codex(messages, budget: int) -> CompactionResult:
    vs = views(messages)
    before = sum(v.tokens for v in vs)
    res = CompactionResult(messages=list(messages), before=before, after=before,
                           strategy="codex")
    if before <= budget:
        res.log.append(f"context {before:,} ≤ bütçe {budget:,} → tetiklenmedi")
        return res
    res.triggered = True

    # --- Katman A: her büyük tool çıktısı için truncate_middle ---
    per_tool_max = max(budget * 4 // max(len([v for v in vs if v.role == 'tool']), 1) // 2, 400)
    out = list(messages)
    cut = 0
    for i, v in enumerate(vs):
        if v.role == "tool" and len(v.content) > per_tool_max:
            out[i] = v.replaced(_truncate_middle(v.content, per_tool_max))
            cut += 1
            res.log.append(f"[A] #{i} tool={v.tool_name} truncate_middle → baş+son tutuldu "
                           f"({len(v.content):,}→{per_tool_max:,} kar)")
    mid = total_tokens(out)
    res.log.append(f"[A] ortadan-kesme sonrası: {before:,} → {mid:,}t")

    # --- Katman B: hâlâ sığmıyorsa model-turn windowing (handoff özeti + YENİ pencere) ---
    if mid > budget:
        trace = "\n".join(f"{v.role}({v.tool_name or '-'}): {v.content[:400]}" for v in views(out))
        fb = (f"[handoff özeti: {len(out)} mesaj damıtıldı — ilerleme, kararlar, kalan iş]")
        s = _llm_summary(
            trace,
            "Bir kodlama ajanının oturumunu YENİ bir context penceresine devrediyorsun. "
            "İlerleme, alınan kararlar ve kalan işi 3-4 cümlede Türkçe özetle. Sadece özeti yaz.", fb)
        mode = "GERÇEK LLM" if (llm is not None and llm.available()) else "fallback"
        first = vs[0]
        keep = 2
        # ÇİFT BÜTÜNLÜĞÜ: kuyruğu keserken YETİM tool sonucu bırakma.
        # out[-keep:] körlemesine kesilirse, bir tool_result kendi assistant
        # çağrısından koparılabilir → sağlayıcı isteği 400 ile reddeder.
        tail = out[-keep:] if keep < len(out) else out[1:]
        while tail and _View(tail[0]).role == "tool":
            i = len(out) - len(tail) - 1          # bir öncekini de al (çağrı mesajı)
            if i <= 0:
                tail = tail[1:]                   # çağrı yok → yetim sonucu düşür
                break
            tail = out[i:]
        out = [first.replaced(f"[CompactedItem · yeni pencere #2 · {mode}]\n{s}")] + tail
        res.log.append(f"[B] windowing: pencere doldu → handoff özeti üretildi, "
                       f"YENİ pencere açıldı (son {keep} mesaj taşındı)")
        res.stats["window"] = 2
        res.stats["llm"] = mode

    res.messages = out
    res.after = total_tokens(out)
    res.stats["kesilen_tool"] = cut
    return res


# ───────────────────────────── 5) CLAUDE CODE (micro + auto) ─────────────────────────────

def _claude_code(messages, budget: int, spill_dir: Path | None = None) -> CompactionResult:
    vs = views(messages)
    before = sum(v.tokens for v in vs)
    res = CompactionResult(messages=list(messages), before=before, after=before,
                           strategy="claude_code")
    micro_limit = max(budget // 4, 300)

    out = list(messages)
    spilled = 0
    spill_dir = spill_dir or (Path(os.environ.get("BRAIN_SPILL_DIR", "")) if os.environ.get("BRAIN_SPILL_DIR") else None)

    # --- A) microcompaction: büyük tek çıktı → diske, context'e önizleme + referans ---
    for i, v in enumerate(vs):
        if v.role == "tool" and est(v.content) > micro_limit:
            preview = v.content[:600]
            ref = f"tool-results/{v.tool_name or 'tool'}_{i}.txt"
            if spill_dir:
                try:
                    spill_dir.mkdir(parents=True, exist_ok=True)
                    (spill_dir / f"{v.tool_name or 'tool'}_{i}.txt").write_text(v.content, encoding="utf-8")
                except Exception:
                    pass
            out[i] = v.replaced(
                f"{preview}\n…\n[Output too large ({len(v.content)//1024}KB). "
                f"Full output saved to: {ref}]")
            spilled += 1
            res.log.append(f"[A] MICRO #{i} tool={v.tool_name} {est(v.content):,}t → diske döküldü, "
                           f"context'e önizleme+referans")
    mid = total_tokens(out)
    if spilled:
        res.triggered = True
        res.log.append(f"[A] microcompaction sonrası: {before:,} → {mid:,}t")

    # --- B) auto-compaction: eşiği aşarsa eski turn'ler → konuşma özeti ---
    if mid > budget:
        res.triggered = True
        res.log.append("[PreCompact hook] auto-compaction başlıyor")
        keep = 3
        old = out[:-keep] if len(out) > keep else []
        if old:
            trace = "\n".join(f"{_View(m).role}: {_View(m).content[:400]}" for m in old)
            fb = f"[Konuşma özeti: önceki {len(old)} mesaj damıtıldı — ilerleme, kararlar, kalan iş]"
            s = _llm_summary(
                trace,
                "Claude Code oturumunun eski kısmını damıtıyorsun. İlerleme, alınan kararlar ve "
                "kalan işi 2-3 cümlede Türkçe özetle. Sadece özeti yaz.", fb)
            mode = "GERÇEK LLM" if (llm is not None and llm.available()) else "fallback"
            out = [_View(old[0]).replaced(f"[Konuşma özeti · {mode}]\n{s}")] + out[-keep:]
            res.log.append(f"[B] AUTO-COMPACTION: {len(old)} eski mesaj → konuşma özeti, "
                           f"son {keep} turn korundu")
            res.log.append("[PostCompact hook] compaction bitti")
            res.stats["llm"] = mode
        else:
            res.log.append("[anti-thrash] korunan turn'ler eşiği dolduruyor → compaction yer açamaz")

    if not res.triggered:
        res.log.append(f"context {before:,} ≤ bütçe {budget:,} · büyük tek çıktı yok → tetiklenmedi")

    res.messages = out
    res.after = total_tokens(out)
    res.stats["spilled"] = spilled
    return res


# ───────────────────────────── genel giriş noktası ─────────────────────────────

_IMPL = {
    "hermes": _hermes,
    "opencode": _opencode,
    "openclaw": _openclaw,
    "codex": _codex,
    "claude_code": _claude_code,
}


def compact(strategy: str, messages: list, budget: int = 4_000, **kw) -> CompactionResult:
    """Seçilen stratejiyle tool-trace compaction uygula.

    strategy: STRATEGIES içinden biri. budget: hedef context token bütçesi.
    Her zaman bir CompactionResult döner (strateji tetiklenmese bile).
    """
    strategy = (strategy or "none").lower()
    if strategy not in STRATEGIES:
        raise ValueError(f"bilinmeyen strateji: {strategy} (geçerli: {STRATEGIES})")
    if strategy == "none":
        t = total_tokens(messages)
        r = CompactionResult(messages=list(messages), before=t, after=t, strategy="none")
        r.log.append(f"sıkıştırma yok — context {t:,} token (temel çizgi)")
        return r
    fn = _IMPL[strategy]
    if strategy == "claude_code":
        return fn(messages, budget, kw.get("spill_dir"))
    return fn(messages, budget)


if __name__ == "__main__":
    # hızlı kendi kendine test: sentetik geçmiş üstünde tüm stratejiler
    def mk(role, content, name="", tcs=None):
        return {"role": role, "content": content, "name": name, "tool_calls": tcs or []}

    big = "satır çıktı örneği " * 400
    hist = [
        mk("user", "auth modülünü incele ve login hatasını bul"),
        mk("assistant", "", tcs=[{"name": "read_file", "args": {"path": "auth.py"}}]),
        mk("tool", big, "read_file"),
        mk("assistant", "", tcs=[{"name": "read_file", "args": {"path": "auth.py"}}]),
        mk("tool", big, "read_file"),                      # dedup adayı
        mk("assistant", "", tcs=[{"name": "grep", "args": {"q": "login"}}]),
        mk("tool", big[:3000], "grep"),
        mk("user", "peki testleri koştur"),
        mk("assistant", "", tcs=[{"name": "bash", "args": {"cmd": "pytest"}}]),
        mk("tool", big, "bash"),
        mk("assistant", "login() içindeki token kontrolü hatalı."),
    ]
    print("=" * 78)
    print("compaction.py — tüm stratejiler, aynı geçmiş üstünde")
    print(f"başlangıç context: {total_tokens(hist):,} token · bütçe: 2.000")
    print("=" * 78)
    for s in STRATEGIES:
        r = compact(s, hist, budget=2_000)
        info = STRATEGY_INFO[s]
        print(f"\n── {s.upper():<12} ({info['ekol']}, LLM={'evet' if info['llm'] else 'hayır'})")
        print(f"   {r.summary_line()}")
        for line in r.log[:4]:
            print(f"     {line}")
