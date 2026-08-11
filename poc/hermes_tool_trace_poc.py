#!/usr/bin/env python3
"""
Hermes-Agent tool-trace compaction — SADIK POC (tek dosya, stdlib).

Kaynak: NousResearch/Hermes-Agent · agent/context_compressor.py
        `_prune_old_tool_results` (4 pass) + `_summarize_tool_result`
        + `_truncate_tool_call_args_json`.

Bu POC, Hermes'in LLM'SİZ deterministik tool-trace budamasını birebir
mantıkla yeniden uygular ve bir örnek trace üzerinde ÖNCE/SONRA gösterir.

Dört pass:
  Pass 1  — Dedup (byte-identik tool sonuçları → geri-referans, en yeni tam kopya kalır)
  Pass 2  — Informative summary (büyük benzersiz sonuç → tip-farkında tek satır)
  Pass 3  — Tool-call argüman kısaltma (>500 karakter arg → JSON içinde kırp)
  Pass 4  — Basınç demotion'ı (korunan bölge bile tavanı aşarsa, kademeli)

Çalıştır:  python3 hermes_tool_trace_poc.py
"""
from __future__ import annotations

import json
import re
import hashlib
from typing import Any

# ---------------------------------------------------------------------------
# Hermes sabitleri (context_compressor.py'den birebir)
# ---------------------------------------------------------------------------
_MAX_TAIL_MESSAGE_FLOOR = 8          # bütçe bitse bile korunan min son mesaj
_PRESSURE_KEEP_RECENT_MESSAGES = 3   # basınçta bile verbatim kalan son mesaj (#61932)
_SKILL_VIEW_PRUNE_MIN_CHARS = 5000   # skill_view bunun üstündeyse buda (+ghost marker)
_DEDUP_FLOOR = 200                   # bunun altını dedup etme
_DEFAULT_MIN_PRUNE_CHARS = 200       # Pass 2: bunun altını özetleme
_ARG_TRUNC_THRESHOLD = 500           # Pass 3: bunun üstündeki argümanı kırp
_ARG_HEAD_CHARS = 200                # arg kırpma: ilk N karakter
_PRUNED_TOOL_PLACEHOLDER = "[Old tool output cleared to save context space]"

SKILL_PRUNED_MARKER_PREFIX = "[SKILL_PRUNED:"


def est(text: str) -> int:
    """Kaba token tahmini (Hermes estimate_tokens_rough ile aynı ruh: ~4 kar/token)."""
    return max(1, len(text) // 4)


def _skill_pruned_marker(skill_name: str) -> str:
    """#32106 — ghost-skill savunması: 'içerik gitti + nasıl geri alınır'."""
    return (
        f"{SKILL_PRUNED_MARKER_PREFIX} content lost in compression; "
        f"reload with skill_view(name='{skill_name}')]"
    )


# ---------------------------------------------------------------------------
# _summarize_tool_result — tip-farkında tek satır (asla çökmez)
# ---------------------------------------------------------------------------
def _str_arg(args: dict, key: str, default: str = "") -> str:
    v = args.get(key, default)
    return v if isinstance(v, str) else (str(v) if v is not None else default)


def _summarize_tool_result(tool_name: str, tool_args: str, content: str) -> str:
    try:
        return _summarize_unguarded(tool_name, tool_args, content)
    except Exception:  # backstop: bozuk tarihsel çağrı compaction'ı çökertmesin
        n = len(content) if isinstance(content, str) else 0
        return f"[{tool_name}] ({n:,} chars result)"


def _summarize_unguarded(tool_name: str, tool_args: str, content: str) -> str:
    try:
        args = json.loads(tool_args) if tool_args else {}
    except (json.JSONDecodeError, TypeError):
        args = {}
    if not isinstance(args, dict):
        args = {}
    content = content or ""
    content_len = len(content)
    line_count = content.count("\n") + 1 if content.strip() else 0

    if tool_name == "terminal":
        cmd = _str_arg(args, "command")
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        m = re.search(r'"exit_code"\s*:\s*(-?\d+)', content)
        exit_code = m.group(1) if m else "?"
        return f"[terminal] ran `{cmd}` -> exit {exit_code}, {line_count} lines output"
    if tool_name == "read_file":
        path = args.get("path", "?")
        offset = args.get("offset", 1)
        return f"[read_file] read {path} from line {offset} ({content_len:,} chars)"
    if tool_name == "write_file":
        path = args.get("path", "?")
        written = _str_arg(args, "content").count("\n") + 1 if args.get("content") else "?"
        return f"[write_file] wrote to {path} ({written} lines)"
    if tool_name == "search_files":
        pattern = args.get("pattern", "?")
        path = args.get("path", ".")
        m = re.search(r'"total_count"\s*:\s*(\d+)', content)
        count = m.group(1) if m else str(content.count("\n"))
        return f"[search_files] content search for '{pattern}' in {path} -> {count} matches"
    if tool_name == "web_search":
        return f"[web_search] query='{args.get('query','?')}' ({content_len:,} chars result)"
    if tool_name == "web_extract":
        urls = args.get("urls", [])
        first = urls[0] if isinstance(urls, list) and urls else "?"
        if isinstance(first, dict):          # web_search sonucu dict → unwrap (yoksa dict+str TypeError)
            first = first.get("url") or first.get("href") or "?"
        elif not isinstance(first, str):
            first = "?"
        desc = first + (f" (+{len(urls)-1} more)" if isinstance(urls, list) and len(urls) > 1 else "")
        return f"[web_extract] {desc} ({content_len:,} chars)"
    if tool_name == "delegate_task":
        goal = _str_arg(args, "goal")
        if len(goal) > 60:
            goal = goal[:57] + "..."
        return f"[delegate_task] '{goal}' ({content_len:,} chars result)"
    if tool_name == "execute_code":
        code = _str_arg(args, "code")[:60].replace("\n", " ")
        return f"[execute_code] `{code}` ({line_count} lines output)"
    if tool_name == "skill_view":
        name = args.get("name", "?")
        if content_len > _SKILL_VIEW_PRUNE_MIN_CHARS:   # ghost-skill (#32106)
            return f"[skill_view] name={name} ({content_len:,} chars) " + _skill_pruned_marker(str(name))
        return f"[skill_view] name={name} ({content_len:,} chars)"
    return f"[{tool_name}] ({content_len:,} chars result)"   # bilinmeyen → backstop


def _truncate_tool_call_args_json(args_str: str, head_chars: int = _ARG_HEAD_CHARS) -> str:
    """Argümanları JSON İÇİNDE kırp (geçerli JSON kalır, yoksa provider 400)."""
    try:
        obj = json.loads(args_str)
    except (json.JSONDecodeError, TypeError):
        return args_str
    def _shrink(v):
        if isinstance(v, str) and len(v) > head_chars:
            return v[:head_chars] + "...[truncated]"
        if isinstance(v, dict):
            return {k: _shrink(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_shrink(x) for x in v]
        return v
    return json.dumps(_shrink(obj), ensure_ascii=False)


# ---------------------------------------------------------------------------
# _prune_old_tool_results — 4 pass (SADIK)
# ---------------------------------------------------------------------------
def _msg_tokens(msg: dict) -> int:
    """Bir mesajın kaba token maliyeti (içerik + tool_call argümanları)."""
    t = 0
    c = msg.get("content")
    if isinstance(c, str):
        t += est(c)
    for tc in msg.get("tool_calls") or []:
        t += est(tc.get("function", {}).get("arguments", ""))
    return t + 3  # rol/anahtar payı


def prune_old_tool_results(
    messages: list[dict],
    protect_tail_count: int = 20,
    protect_tail_tokens: int | None = None,
    min_prune_chars: int = _DEFAULT_MIN_PRUNE_CHARS,
    spare_protected_skills: set[str] | None = None,
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """Hermes 4-pass deterministik tool-trace budaması. (pruned_messages, rapor) döner."""
    spare_protected_skills = spare_protected_skills or set()
    if not messages:
        return messages, {}
    result = [dict(m) for m in messages]
    stats = {"dedup": 0, "summary": 0, "args": 0, "pressure": 0}

    # call_id -> (tool_name, args_json) indeksi
    call_id_to_tool: dict[str, tuple[str, str]] = {}
    for msg in result:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                call_id_to_tool[tc.get("id", "")] = (fn.get("name", "unknown"), fn.get("arguments", ""))

    # ---- Prune boundary (token bütçesi öncelikli, floor 8) ----
    if protect_tail_tokens is not None and protect_tail_tokens > 0:
        accumulated = 0
        boundary = len(result)
        min_protect = min(protect_tail_count, len(result), _MAX_TAIL_MESSAGE_FLOOR)
        for i in range(len(result) - 1, -1, -1):
            mt = _msg_tokens(result[i])
            if accumulated + mt > protect_tail_tokens and (len(result) - i) >= min_protect:
                boundary = i
                break
            accumulated += mt
            boundary = i
        protected_count = max(len(result) - boundary, min_protect)
        prune_boundary = len(result) - protected_count
    else:
        prune_boundary = len(result) - protect_tail_count
    prune_boundary = max(0, prune_boundary)

    # ---- Pass 1: Dedup (tail-agnostik, kayıpsız) ----
    seen: dict[str, int] = {}   # hash -> en yeni tam kopyanın index'i
    for i in range(len(result) - 1, -1, -1):
        msg = result[i]
        if msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if not isinstance(content, str) or len(content) < _DEDUP_FLOOR:
            continue
        if content.startswith(SKILL_PRUNED_MARKER_PREFIX) or content.startswith("["):
            continue
        h = hashlib.sha1(content.encode("utf-8")).hexdigest()
        if h in seen:
            newest = seen[h]
            # Metin gerçek repodan BİREBİR: context_compressor.py:2889.
            # (POC önce "[Duplicate of newer result at #N — content omitted]" yazıyordu;
            #  #N indeksi gerçekte YOK. Panelde SONRA kutusunda bu metin aynen
            #  gösterildiği için birebir olması gerekiyor.)
            _ = newest
            result[i] = {**msg,
                         "content": "[Duplicate tool output — same content as a more recent call]"}
            stats["dedup"] += 1
        else:
            seen[h] = i

    # ---- Pass 2: Informative summary (sadece boundary öncesi) ----
    def _demote(idx: int, *, spare_skills: bool) -> bool:
        msg = result[idx]
        if msg.get("role") != "tool":
            return False
        content = msg.get("content") or ""
        if not isinstance(content, str):
            return False
        if content.startswith("[") and " chars)" in content and len(content) < 400:
            return False   # zaten budanmış
        if content.startswith("[Duplicate tool output") or content.startswith("[screenshot removed"):
            return False
        if len(content) <= min_prune_chars:
            return False   # küçük — değmez
        cid = msg.get("tool_call_id", "")
        tool_name, tool_args = call_id_to_tool.get(cid, ("unknown", ""))
        if spare_skills and tool_name == "skill_view":
            try:
                nm = (json.loads(tool_args) or {}).get("name", "") if tool_args else ""
            except Exception:
                nm = ""
            if isinstance(nm, str) and nm.lower() in spare_protected_skills:
                return False   # aktif skill korunur (#32106)
        result[idx] = {**msg, "content": _summarize_tool_result(tool_name, tool_args, content)}
        return True

    for i in range(prune_boundary):
        if _demote(i, spare_skills=True):
            stats["summary"] += 1

    # ---- Pass 3: Tool-call argüman kısaltma (boundary öncesi assistant) ----
    def _trunc_args(idx: int) -> bool:
        msg = result[idx]
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            return False
        changed = False
        new_tcs = []
        for tc in msg["tool_calls"]:
            args = tc.get("function", {}).get("arguments", "")
            if len(args) > _ARG_TRUNC_THRESHOLD:
                na = _truncate_tool_call_args_json(args)
                if na != args:
                    tc = {**tc, "function": {**tc["function"], "arguments": na}}
                    changed = True
            new_tcs.append(tc)
        if changed:
            result[idx] = {**msg, "tool_calls": new_tcs}
        return changed

    for i in range(prune_boundary):
        if _trunc_args(i):
            stats["args"] += 1

    # ---- Pass 4: Basınç demotion'ı (#61932) ----
    def _protected_tokens() -> int:
        return sum(_msg_tokens(m) for m in result[prune_boundary:])

    if protect_tail_tokens:
        soft_ceiling = int(protect_tail_tokens * 1.5)
        if _protected_tokens() > soft_ceiling:
            # Kademe 1+2: en yeni tool hariç korunan bölgedeki büyükleri demote et
            last_tool_idx = None
            for i in range(len(result) - 1, -1, -1):
                if result[i].get("role") == "tool":
                    last_tool_idx = i
                    break
            for i in range(prune_boundary, len(result)):
                if last_tool_idx is not None and i == last_tool_idx:
                    continue   # en yeni tool'u koru (kademe 3'e sakla)
                if result[i].get("role") == "tool" and _demote(i, spare_skills=False):
                    stats["pressure"] += 1
                elif result[i].get("role") == "assistant" and _trunc_args(i):
                    stats["pressure"] += 1
                if _protected_tokens() <= soft_ceiling:
                    break
            # Kademe 3: son çare — en yeni tool bile tavandan büyükse onu da demote
            if (last_tool_idx is not None and last_tool_idx >= prune_boundary
                    and _protected_tokens() > soft_ceiling):
                if _demote(last_tool_idx, spare_skills=False):
                    stats["pressure"] += 1

    stats["prune_boundary"] = prune_boundary
    return result, stats


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------
def _big(prefix: str, n: int) -> str:
    body = "\n".join(f"{prefix} satır {i}" for i in range(n))
    return body


def demo_trace() -> list[dict]:
    a_py = _big("import os / def login(): ...", 900)   # ~40K karakter
    return [
        {"role": "system", "content": "Sen Hermes'sin. auth modülünü incele."},           # 0 HEAD
        {"role": "user", "content": "auth modülündeki bug'ı bul ve düzelt."},               # 1 HEAD
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "function": {"name": "read_file", "arguments": '{"path":"auth/login.py","offset":1}'}}]},  # 2
        {"role": "tool", "tool_call_id": "c1", "content": a_py},                             # 3 büyük benzersiz
        {"role": "assistant", "tool_calls": [
            {"id": "c2", "function": {"name": "search_files", "arguments": '{"pattern":"login","path":"auth/"}'}}]},  # 4
        {"role": "tool", "tool_call_id": "c2",
         "content": '{"total_count": 47}\n' + _big("auth/login.py:", 300)},                  # 5 büyük benzersiz
        {"role": "assistant", "tool_calls": [
            {"id": "c3", "function": {"name": "read_file", "arguments": '{"path":"auth/login.py","offset":1}'}}]},  # 6
        {"role": "tool", "tool_call_id": "c3", "content": a_py},                             # 7 c1'in KOPYASI
        {"role": "assistant", "tool_calls": [
            {"id": "c4", "function": {"name": "write_file",
             "arguments": json.dumps({"path": "auth/login.py", "content": _big("yeni içerik", 1200)})}}]},  # 8 DEV ARG
        {"role": "tool", "tool_call_id": "c4", "content": "wrote auth/login.py"},            # 9 küçük
        {"role": "assistant", "tool_calls": [
            {"id": "c5", "function": {"name": "skill_view", "arguments": '{"name":"github"}'}}]},  # 10
        {"role": "tool", "tool_call_id": "c5", "content": _big("# GitHub skill talimatları", 200)},  # 11 skill (~5K+)
        {"role": "assistant", "tool_calls": [
            {"id": "c6", "function": {"name": "terminal", "arguments": '{"command":"pytest auth/"}'}}]},  # 12
        {"role": "tool", "tool_call_id": "c6",
         "content": '{"exit_code": 1}\n' + _big("test", 500)},                               # 13 TAIL (yakın)
        {"role": "assistant", "content": "Bug login.py:45'te. Düzelttim ve test ekledim."},  # 14 TAIL
    ]


def _render(messages: list[dict]) -> None:
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "tool":
            c = m.get("content", "")
            preview = c if len(c) < 70 else c[:67] + "..."
            print(f"  #{i:<2} tool[{m.get('tool_call_id')}]  {est(c):>5}t  {preview!r}")
        elif role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc["function"]
                a = fn["arguments"]
                ap = a if len(a) < 50 else a[:47] + "..."
                print(f"  #{i:<2} call[{tc['id']}]  {est(a):>5}t  {fn['name']}({ap})")
        else:
            c = m.get("content", "")
            print(f"  #{i:<2} {role:<9} {est(c):>5}t  {c[:60]!r}")


def total_tokens(messages: list[dict]) -> int:
    return sum(_msg_tokens(m) for m in messages)


if __name__ == "__main__":
    trace = demo_trace()
    before = total_tokens(trace)

    print("=" * 78)
    print("HERMES TOOL-TRACE COMPACTION — POC")
    print("=" * 78)
    print(f"\n── ÖNCE  ({before:,} token) " + "─" * 40)
    _render(trace)

    # protect_tail_tokens=2000 → son ~2K token korunur (floor 8 mesaj), skill 'github' aktif sayılır
    pruned, stats = prune_old_tool_results(
        trace,
        protect_tail_count=20,
        protect_tail_tokens=2000,
        spare_protected_skills={"github"},
    )
    after = total_tokens(pruned)

    print(f"\n── SONRA ({after:,} token) " + "─" * 40)
    _render(pruned)

    print("\n── PASS RAPORU " + "─" * 50)
    print(f"  Pass 1 dedup            : {stats['dedup']} mesaj → geri-referans")
    print(f"  Pass 2 informative özet : {stats['summary']} mesaj → tek satır")
    print(f"  Pass 3 arg kısaltma     : {stats['args']} çağrı → JSON içi kırpma")
    print(f"  Pass 4 basınç demotion  : {stats['pressure']} mesaj (korunan bölge)")
    print(f"  prune_boundary          : #{stats['prune_boundary']} (öncesi budanır, sonrası korunur)")

    saved = before - after
    pct = (saved / before * 100) if before else 0
    print("\n── SONUÇ " + "─" * 56)
    print(f"  {before:,} → {after:,} token   |   -{saved:,} ({pct:.1f}% kazanç)")
    # tool_call_id bütünlüğü kontrolü
    call_ids = {tc["id"] for m in pruned if m.get("role") == "assistant"
               for tc in (m.get("tool_calls") or [])}
    result_ids = {m["tool_call_id"] for m in pruned if m.get("role") == "tool"}
    ok = result_ids <= call_ids
    print(f"  tool_call_id bütünlüğü  : {'✓ korundu (0 tane 400 riski)' if ok else '✗ BOZULDU'}")
    print(f"  mesaj sayısı            : {len(trace)} → {len(pruned)} (değişmedi = silme yok)")
    print("=" * 78)
