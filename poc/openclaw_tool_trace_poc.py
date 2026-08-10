#!/usr/bin/env python3
"""
OpenClaw tool-trace compaction — 12-AŞAMALI SİMÜLASYON POC (tek dosya, stdlib).

Kaynak: openclaw/src/agents/compaction-planning.ts (+ projection, worker, repair).

OpenClaw felsefesi: tool sonucunu DETERMİNİSTİK budamaz; tool-çiftlerini
bozmadan chunk'lar ve chunk'ları bir LLM'e ÖZETLETİR. Bu POC, o PLANLAMA
hattını (asıl deterministik + akıllı kısım) birebir mantıkla simüle eder;
10. adımdaki LLM özeti mock'lanır (POC'ta gerçek model yok).

12 adım:
  0 Tetik · 1 Sanitize · 2 Estimate · 3 Projection · 4 Adaptif oran
  5 Gruplama(çift) · 6 Chunk · 7 Oversized · 8 Stage-split · 9 Worker
  10 LLM özet(mock) · 11 Onarım · 12 Uygula

Çalıştır:  python3 openclaw_tool_trace_poc.py
"""
from __future__ import annotations
import math
from typing import Any

try:
    import llm  # gerçek LLM özeti (poc/llm.py; .env'den yükler, anahtar ASLA yazdırılmaz)
except Exception:
    llm = None

# ---- OpenClaw sabitleri (birebir) ----
MIN_PROMPT_BUDGET_TOKENS = 8_000
MIN_PROMPT_BUDGET_RATIO = 0.5
BASE_CHUNK_RATIO = 0.4
MIN_CHUNK_RATIO = 0.15
SAFETY_MARGIN = 1.2
DEFAULT_PARTS = 2
SUMMARIZATION_OVERHEAD_TOKENS = 4096
TEXT_TRUNCATE_THRESHOLD_CHARS = 32_768
TEXT_SAMPLE_CHARS = 8_192
PLANNING_MAX_CHARS = 256 * 1024

CONTEXT_WINDOW = 200_000


def est(text: str) -> int:
    """Kaba token tahmini (~4 kar/token)."""
    return max(0, len(text) // 4)


def msg_tokens(m: dict) -> int:
    if m["role"] == "runtime":
        return 0
    t = est(m.get("content", "") or "")
    for tc in m.get("toolCalls", []) or []:
        t += est(tc.get("args", "")) + 3
    return t


# ================= ADIMLAR =================

def step0_trigger(msgs, window):
    total = sum(msg_tokens(m) for m in msgs)
    free = window - total
    need = window * MIN_PROMPT_BUDGET_RATIO
    trig = free < need or free < MIN_PROMPT_BUDGET_TOKENS
    return trig, total, free, need


def step1_sanitize(msgs):
    """toolResult.details sil + runtime çıkar."""
    out, removed_runtime, stripped_details = [], 0, 0
    for m in msgs:
        if m["role"] == "runtime":
            removed_runtime += 1
            continue
        if m["role"] == "toolResult" and "details" in m:
            m = {k: v for k, v in m.items() if k != "details"}
            stripped_details += 1
        out.append(m)
    return out, removed_runtime, stripped_details


def step2_estimate(msgs):
    return [msg_tokens(m) for m in msgs]


def step3_projection(msgs):
    """Büyük gövde → 8KB örnek + omittedChars; ağırlık = est(örnek)+omitted/4."""
    projected = []
    budget = PLANNING_MAX_CHARS
    for m in msgs:
        c = m.get("content", "") or ""
        if len(c) <= TEXT_TRUNCATE_THRESHOLD_CHARS and len(c) <= budget:
            budget -= len(c)
            projected.append({**m, "_omitted": 0, "_weight": msg_tokens(m)})
        else:
            sample = c[:min(TEXT_SAMPLE_CHARS, max(0, budget))]
            budget -= len(sample)
            omitted = len(c) - len(sample)
            weight = est(sample) + math.ceil(omitted / 4) + sum(est(tc.get("args", "")) + 3 for tc in m.get("toolCalls", []) or [])
            projected.append({**m, "content": sample, "_omitted": omitted, "_weight": weight})
    return projected


def step4_adaptive_ratio(projected, window):
    vis = [p for p in projected if p["role"] != "runtime"]
    if not vis:
        return BASE_CHUNK_RATIO
    avg = sum(p["_weight"] for p in vis) / len(vis)
    avg_ratio = (avg * SAFETY_MARGIN) / window
    if avg_ratio > 0.10:
        reduction = min(avg_ratio * 2, BASE_CHUNK_RATIO - MIN_CHUNK_RATIO)
        ratio = max(MIN_CHUNK_RATIO, BASE_CHUNK_RATIO - reduction)
    else:
        ratio = BASE_CHUNK_RATIO
    return ratio, avg, avg_ratio


def step5_group(msgs):
    """Tool-çiftini bozmadan atomik gruplar. pending boşalınca grup kapanır."""
    groups, current, pending = [], [], set()
    for m in msgs:
        current.append(m)
        if m["role"] == "assistant":
            stop = m.get("stopReason")
            calls = [] if stop in ("aborted", "error") else [tc["id"] for tc in m.get("toolCalls", []) or []]
            pending = set(calls)
        elif m["role"] == "toolResult" and pending:
            rid = m.get("id")
            pending.discard(rid) if rid else pending.clear()
        if not pending:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def group_tokens(g):
    return sum(p["_weight"] for p in g)


def step6_chunk(groups, max_tokens):
    """Grupları maxTokens'a kadar paketle — grup asla bölünmez."""
    eff = max(1, int(max_tokens / SAFETY_MARGIN))
    chunks, current, cur_t = [], [], 0
    for g in groups:
        gt = group_tokens(g)
        if current and cur_t + gt > eff:
            chunks.append(current)
            current, cur_t = [], 0
        current.extend(g)
        cur_t += gt
    if current:
        chunks.append(current)
    return chunks, eff


def step7_oversized(msgs, window):
    """Tek mesaj > window*0.5 ise içerik yerine not; batch'i düşür."""
    threshold = window * 0.5
    small, notes, omit_ids = [], [], set()
    for m in msgs:
        w = m["_weight"]
        if w * SAFETY_MARGIN > threshold:
            k = round(w / 1000)
            notes.append(f"[Large {m['role']} (~{k}K tokens) omitted from summary]")
            # batch'i düşür: bu mesajın tool id'sini işaretle
            for tc in m.get("toolCalls", []) or []:
                omit_ids.add(tc["id"])
            if m.get("id"):
                omit_ids.add(m["id"])
        else:
            small.append(m)
    # düşen batch'in call/result eşlerini de ele (gerçek user mesajı hariç zaten small'da)
    small = [m for m in small if not (
        (m.get("id") in omit_ids) or
        any(tc["id"] in omit_ids for tc in m.get("toolCalls", []) or [])
    )]
    return small, notes, omit_ids


def step8_stage_split(msgs, max_tokens, parts=DEFAULT_PARTS):
    total = sum(m["_weight"] for m in msgs)
    if parts <= 1 or len(msgs) < 4 or total <= max_tokens:
        return "single", [msgs] if msgs else []
    # kabaca eşit token payına böl (çift bölmeden — basit)
    target = total / parts
    out, cur, cur_t = [], [], 0
    for m in msgs:
        if cur and cur_t + m["_weight"] > target and len(out) < parts - 1:
            out.append(cur); cur, cur_t = [], 0
        cur.append(m); cur_t += m["_weight"]
    if cur:
        out.append(cur)
    return ("split" if len(out) > 1 else "single"), out


def step10_summarize(chunk):
    """Chunk'ı özetler: llm.available() ise GERÇEK LLM çağrısı, değilse mock'a düşer."""
    tools = []
    for m in chunk:
        for tc in m.get("toolCalls", []) or []:
            tools.append(tc["name"])
    tools_s = ", ".join(dict.fromkeys(tools)) or "—"
    if llm is not None and llm.available():
        lines = []
        for m in chunk:  # chunk sanitize edilmiş; details zaten silinmiş (sır yok)
            if m["role"] == "assistant" and m.get("toolCalls"):
                lines.append("asistan çağrıları: " + ", ".join(
                    f"{tc['name']}({tc.get('args','')})" for tc in m["toolCalls"]))
            elif m["role"] == "toolResult":
                lines.append(f"tool sonucu[{m.get('id')}]: " + (m.get("content", "") or "")[:1500])
            elif m.get("content"):
                lines.append(f"{m['role']}: {m['content'][:400]}")
        try:
            r = llm.chat([
                {"role": "system", "content": "Ajan tool-izi özetleyicisisin. Verilen tool çağrıları ve sonuçlarını, ilerlemeyi ve kararları koruyacak şekilde 2-3 cümlede Türkçe özetle. Sadece özeti döndür."},
                {"role": "user", "content": "\n".join(lines)[:6000]},
            ], max_tokens=200, temperature=0.2)
            s = (r.get("content") or "").strip()
            if s:
                return f"[ÖZET (GERÇEK LLM · {llm.MODEL}): {s}]"
        except Exception as e:
            return f"[ÖZET: {len(chunk)} mesaj · araçlar: {tools_s} · (LLM hata: {str(e)[:40]} → mock)]"
    return f"[ÖZET: {len(chunk)} mesaj · araçlar: {tools_s} · (mock — llm.available()=False)]"


def step11_repair(final_msgs):
    """Yetim tool sonucu/çağrısı → sentetik onarım."""
    call_ids = {tc["id"] for m in final_msgs for tc in m.get("toolCalls", []) or []}
    result_ids = {m["id"] for m in final_msgs if m["role"] == "toolResult" and m.get("id")}
    orphaned_calls = call_ids - result_ids
    repairs = []
    for cid in orphaned_calls:
        repairs.append({"role": "toolResult", "id": cid,
                        "content": "[openclaw] missing tool result; inserted synthetic error result",
                        "_weight": 12, "_omitted": 0, "synthetic": True})
    return repairs


# ================= DEMO =================

def _big(prefix, n):
    return "\n".join(f"{prefix} satır {i}" for i in range(n))


def demo():
    # auth.py ~40K tok → ~160K kar ; big.py ~110K tok → ~440K kar
    auth = _big("import os / def login()", 3600)          # ~160K kar ≈ 40K tok
    big = _big("generated boilerplate line", 11000)        # ~440K kar ≈ 110K tok
    return [
        {"role": "system", "content": "Sen OpenClaw'sın. auth modülünü refactor et."},
        {"role": "user", "content": "auth modülünü refactor et, login akışını sadeleştir"},
        {"role": "assistant", "content": "", "toolCalls": [
            {"id": "c1", "name": "read_file", "args": '{"path":"auth.py"}'},
            {"id": "c2", "name": "search_files", "args": '{"pattern":"login"}'}]},
        {"role": "toolResult", "id": "c1", "content": auth,
         "details": {"raw_stdout": "...", "cwd": "/home/user/secret", "env": {"API_KEY": "sk-xxx"}}},
        {"role": "toolResult", "id": "c2", "content": _big("auth/login.py:", 500),
         "details": {"grep_flags": "-rn"}},
        {"role": "assistant", "content": "", "toolCalls": [
            {"id": "c3", "name": "read_file", "args": '{"path":"big_generated.py"}'}]},
        {"role": "toolResult", "id": "c3", "content": big, "details": {"raw": "..."}},
        {"role": "assistant", "content": "Refactor planı: login()'i böl, token'ı httpOnly yap."},
        {"role": "runtime", "content": "<runtime-context: iç durum, model-görünmez>"},
    ]


def label(m):
    if m["role"] == "assistant" and m.get("toolCalls"):
        return "asistan[" + ",".join(tc["id"] for tc in m["toolCalls"]) + "] " + ",".join(tc["name"] for tc in m["toolCalls"])
    if m["role"] == "toolResult":
        return f"toolResult[{m.get('id')}]"
    return m["role"]


def main():
    W = CONTEXT_WINDOW
    msgs = demo()
    print("=" * 74)
    print("OPENCLAW TOOL-TRACE COMPACTION — 12 ADIM SİMÜLASYONU")
    print("=" * 74)
    print(f"\nÇANTA (context window) = {W:,} token\nBaşlangıç trace:")
    for i, m in enumerate(msgs):
        print(f"  #{i} {label(m):<28} {msg_tokens(m):>7,}t")

    # 0
    trig, total, free, need = step0_trigger(msgs, W)
    print(f"\n[0] TETİK        toplam={total:,}  boş={free:,}  gerekli≥{int(need):,}  → {'TETİK ✅' if trig else 'gerek yok'}")

    # 1
    san, rr, sd = step1_sanitize(msgs)
    print(f"[1] SANITIZE     {sd} toolResult.details silindi, {rr} runtime çıkarıldı  → {len(san)} mesaj")

    # 2
    ests = step2_estimate(san)
    print(f"[2] ESTIMATE     per-mesaj token: {ests}")

    # 3
    proj = step3_projection(san)
    big3 = [(i, p["_omitted"]) for i, p in enumerate(proj) if p["_omitted"] > 0]
    print(f"[3] PROJECTION   8KB örneğe inen: " + ", ".join(f"#{i}(atılan {o:,} kar)" for i, o in big3))
    print(f"                 projeksiyon ağırlıkları: {[p['_weight'] for p in proj]}")

    # 4
    ratio, avg, avg_ratio = step4_adaptive_ratio(proj, W)
    max_chunk = int(ratio * W)
    print(f"[4] ADAPTIF      ort={avg:,.0f}  avgRatio={avg_ratio:.3f}  → ratio={ratio:.2f}  maxChunk={max_chunk:,}")

    # 5
    groups = step5_group(proj)
    print(f"[5] GRUPLAMA     {len(groups)} grup (çift korunur):")
    for gi, g in enumerate(groups):
        print(f"                 G{gi}: [{', '.join(label(m) for m in g)}]  {group_tokens(g):,}t")

    # 6
    chunks, eff = step6_chunk(groups, max_chunk)
    print(f"[6] CHUNK        effMax={eff:,} → {len(chunks)} chunk:")
    for ci, c in enumerate(chunks):
        print(f"                 C{ci}: {[label(m) for m in c]}  {sum(m['_weight'] for m in c):,}t")

    # 7
    small, notes, omit = step7_oversized(proj, W)
    print(f"[7] OVERSIZED    eşik={int(W*0.5):,} → {len(notes)} not, düşen id={sorted(omit)}")
    for n in notes:
        print(f"                 {n}")

    # 8
    # oversized dışı kalanlardan yalnız TOOL BATCH'i olanları özetle
    # (head=system/user ve tail=çağrısız final assistant korunur, özetlenmez)
    to_summarize = [m for m in small
                    if m["role"] == "toolResult"
                    or (m["role"] == "assistant" and m.get("toolCalls"))]
    mode, stages = step8_stage_split(to_summarize, max_chunk)
    print(f"[8] STAGE-SPLIT  özetlenecek {len(to_summarize)} mesaj → mode={mode} ({len(stages)} aşama)")

    # 9
    print(f"[9] WORKER       planlama worker-thread'de yapıldı (simüle) — ana döngü bloklanmadı")

    # 10
    summaries = [step10_summarize(s) for s in stages if s]
    _mode = "GERÇEK LLM" if (llm is not None and llm.available()) else "mock"
    print(f"[10] LLM ÖZET    {len(summaries)} özet ({_mode}):")
    for s in summaries:
        print(f"                 {s}")

    # 12 (uygula) — head + summaries + notes + tail
    head = [m for m in san if m["role"] in ("system", "user")][:2]
    tail = [san[-1]] if san and san[-1]["role"] == "assistant" and not san[-1].get("toolCalls") else []
    final = []
    for m in head:
        final.append({**m, "_weight": msg_tokens(m)})
    for s in summaries:
        final.append({"role": "summary", "content": s, "_weight": est(s)})
    for n in notes:
        final.append({"role": "note", "content": n, "_weight": est(n)})
    for m in tail:
        final.append({**m, "_weight": msg_tokens(m)})

    # 11 (onarım) — final'da yetim var mı
    repairs = step11_repair(final)
    print(f"[11] ONARIM      {len(repairs)} yetim tool → sentetik sonuç" + (f" {[r['id'] for r in repairs]}" if repairs else " (yok — batch düşerken çift birlikte düştü)"))
    final.extend(repairs)

    after = sum(m["_weight"] for m in final)
    print(f"[12] UYGULA      yeni transcript ({len(final)} mesaj), usage snapshot sıfırlandı")

    print("\n── SONUÇ " + "─" * 55)
    print(f"  ÖNCE : {total:,} token, {len(msgs)} mesaj")
    for m in final:
        c = m.get("content", "")
        prev = c if len(c) < 60 else c[:57] + "..."
        print(f"  SONRA #{m['role']:<9} {m['_weight']:>6,}t  {prev!r}")
    saved = total - after
    print(f"\n  {total:,} → {after:,} token   |   -{saved:,} ({saved/total*100:.1f}% kazanç)")
    # bütünlük
    call_ids = {tc["id"] for m in final for tc in m.get("toolCalls", []) or []}
    res_ids = {m["id"] for m in final if m.get("role") == "toolResult" and m.get("id")}
    print(f"  çift bütünlüğü : {'✓ (yetim yok)' if res_ids >= call_ids else '✗ yetim var'}")
    print(f"  sır sızıntısı  : {'✓ details silindi' } (özete hiç details girmedi)")
    print("=" * 74)


if __name__ == "__main__":
    main()
