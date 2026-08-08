#!/usr/bin/env python3
"""
OpenCode tool-trace compaction — SADIK POC (tek dosya, stdlib).

Kaynak: opencode/packages/opencode/src/{tool/truncate.ts, session/compaction.ts, session/overflow.ts}

OpenCode iki KATMANlı çalışır:
  Katman A — canlı spill: tool çıktısı üretilirken > 2000 satır / 50KB ise
             diske dökülür; context'e önizleme + referans girer.
  Katman B — eşikte deterministik prune (LLM'siz) + gerekirse overflow LLM özeti:
             sondan başa yürü, son-2-turn ve en-yeni-40K korunur, korunan
             tool (skill) atlanır; toplam budanabilir > 20K ise 'compacted'
             damgası basılır; serialize'da damgalılar 2000 karaktere iner.

Çalıştır:  python3 opencode_tool_trace_poc.py
"""
from __future__ import annotations
import time
from typing import Any

# ---- OpenCode sabitleri (birebir) ----
MAX_LINES = 2000                 # Katman A: çıktı bu satırı aşarsa dök
MAX_BYTES = 50 * 1024            # ya da bu baytı (50KB)
DEFAULT_TAIL_TURNS = 2           # son 2 turn dokunulmaz
PRUNE_PROTECT = 40_000           # en yeni 40K tool çıktısı korunur
PRUNE_MINIMUM = 20_000           # toplam budama > bu ise commit (fayda-freni)
TOOL_OUTPUT_MAX_CHARS = 2_000    # compacted çıktı serialize'da bu kadara iner
PRUNE_PROTECTED_TOOLS = {"skill"}
COMPACTION_BUFFER = 20_000       # overflow.ts reserve

CONTEXT_WINDOW = 200_000


def est(text: str) -> int:
    return max(0, len(text) // 4)


# ---------------- Katman A: canlı spill ----------------
def emit_tool_output(name: str, output: str) -> dict:
    """Tool çıktısı ÜRETİLİRKEN çağrılır. Büyükse diske döker, önizleme bırakır."""
    lines = output.count("\n") + 1
    nbytes = len(output.encode("utf-8"))
    if lines > MAX_LINES or nbytes > MAX_BYTES:
        preview = output[:2000]
        path = f".opencode/truncation/{name}_{int(time.time()*1000)%100000}.txt"
        stored = (f"{preview}\n...[truncated: {lines} satır / {nbytes//1024}KB]\n"
                  f"[Full output saved to: {path}]")
        return {"type": "tool", "tool": name,
                "state": {"status": "completed", "output": stored,
                          "time": {"compacted": None}, "spilled": True, "full_lines": lines}}
    return {"type": "tool", "tool": name,
            "state": {"status": "completed", "output": output,
                      "time": {"compacted": None}, "spilled": False}}


def text_part(t: str) -> dict:
    return {"type": "text", "text": t}


def part_tokens(p: dict) -> int:
    if p["type"] == "text":
        return est(p.get("text", ""))
    st = p["state"]
    out = st["output"]
    # compacted ise serialize'da 2000 karaktere iner → o boyutu say
    if st["time"]["compacted"]:
        out = out[:TOOL_OUTPUT_MAX_CHARS]
    return est(out)


def msg_tokens(m: dict) -> int:
    return sum(part_tokens(p) for p in m["parts"]) + 3


def total_tokens(msgs) -> int:
    return sum(msg_tokens(m) for m in msgs)


# ---------------- Katman B: overflow tetiği ----------------
def usable(window: int, reserved: int = COMPACTION_BUFFER) -> int:
    return max(0, window - reserved)


def is_overflow(used: int, window: int) -> bool:
    return used >= usable(window)


# ---------------- Katman B: deterministik prune ----------------
def prune(msgs: list[dict], verbose: bool = True) -> tuple[int, list]:
    """Sondan başa yürü; son-2-turn + en-yeni-40K + skill korunur.
    Toplam budanabilir > PRUNE_MINIMUM ise 'compacted' damgası bas."""
    turns = 0
    total = 0        # gezilen tool çıktısı toplamı (PROTECT için)
    pruned = 0       # budanacak toplam
    to_prune = []
    log = []
    stop = False
    for mi in range(len(msgs) - 1, -1, -1):
        if stop:
            break
        m = msgs[mi]
        if m["role"] == "user":
            turns += 1
        if turns < DEFAULT_TAIL_TURNS:
            log.append(f"  #{mi} {m['role']:<9} turns={turns} < 2 → KORU (son-2-turn)")
            continue
        if m["role"] == "assistant" and m.get("summary"):
            log.append(f"  #{mi} assistant(summary) → DUR (önceki compaction sınırı)")
            break
        for pi in range(len(m["parts"]) - 1, -1, -1):
            p = m["parts"][pi]
            if p["type"] != "tool" or p["state"]["status"] != "completed":
                continue
            if p["tool"] in PRUNE_PROTECTED_TOOLS:
                log.append(f"  #{mi} tool={p['tool']} → ATLA (PRUNE_PROTECTED)")
                continue
            if p["state"]["time"]["compacted"]:
                log.append(f"  #{mi} tool={p['tool']} zaten compacted → DUR")
                stop = True
                break
            estimate = est(p["state"]["output"])
            total += estimate
            if total <= PRUNE_PROTECT:
                log.append(f"  #{mi} tool={p['tool']} {estimate:>6}t · toplam={total} ≤ 40K → KORU (en-yeni-40K)")
                continue
            pruned += estimate
            to_prune.append(p)
            log.append(f"  #{mi} tool={p['tool']} {estimate:>6}t · toplam={total} > 40K → BUDA aday (pruned={pruned})")
    if verbose:
        print("\n".join(log))
    # fayda-freni
    committed = 0
    if pruned > PRUNE_MINIMUM:
        for p in to_prune:
            p["state"]["time"]["compacted"] = int(time.time() * 1000)
            committed += 1
    return pruned, to_prune, committed


# ---------------- Katman B: overflow LLM özeti (mock, basit) ----------------
def overflow_summary(msgs: list[dict], window: int) -> bool:
    """prune+serialize sonrası hâlâ overflow ise en eski turn'leri özetle (mock)."""
    if not is_overflow(total_tokens(msgs), window):
        return False
    # en eski (son-2-turn dışı) turn'leri tek özete indir (mock)
    # basitçe: ilk user'a kadar olan tool parçalarını bir özet mesajına çevir
    return True  # bu örnekte prune yettiyse buraya girmez


# ================= DEMO =================
def _lines(prefix, n):
    return "\n".join(f"{prefix} {i}" for i in range(n))


def _blob(approx_tokens: int, tag: str) -> str:
    """~approx_tokens token, ama <2000 satır & <50KB (Katman A spill'e TAKILMAZ)."""
    nlines = 1900
    total_chars = approx_tokens * 4
    per = max(12, total_chars // nlines)
    body = (tag + " " + "x" * (per - len(tag) - 1))
    return "\n".join(f"{body}{i:04d}"[:per] for i in range(nlines))


def demo() -> list[dict]:
    # Orta-boy okumalar (~12K tok, spill YOK) — birikince Katman B'ye iş çıkar.
    def R(t, tag):
        return emit_tool_output("read_file", _blob(t, tag))
    sk = _blob(12000, "github-skill")                       # skill → korunur
    huge_test = _lines("test PASSED çıktı satırı", 5000)    # 5000 satır → Katman A SPILL

    return [
        {"role": "user", "parts": [text_part("başla, a modülünü oku")]},
        {"role": "assistant", "parts": [R(12000, "a1"), R(12000, "a2"), R(12000, "a3")]},   # ESKİ 3×12K
        {"role": "user", "parts": [text_part("b'yi oku ve test çalıştır")]},
        {"role": "assistant", "parts": [
            R(12000, "b1"), R(12000, "b2"), R(12000, "b3"),                                  # ESKİ 3×12K
            emit_tool_output("bash", huge_test),                                             # 5000 satır → SPILL
            emit_tool_output("skill", sk)]},                                                 # skill → korunur
        {"role": "user", "parts": [text_part("c'yi oku")]},
        {"role": "assistant", "parts": [R(12000, "c1")]},                                    # ORTA 12K
        {"role": "user", "parts": [text_part("d'yi oku ve login'i düzelt")]},                # ← son 2 turn başı
        {"role": "assistant", "parts": [R(12000, "d1")]},                                    # YENİ 12K (korunur)
        {"role": "assistant", "parts": [text_part("login() düzeltildi.")]},                  # tail
    ]


def render(msgs):
    for i, m in enumerate(msgs):
        segs = []
        for p in m["parts"]:
            if p["type"] == "text":
                segs.append(f'text("{p["text"][:24]}")')
            else:
                st = p["state"]
                mark = ""
                if st.get("spilled"):
                    mark = " [SPILL]"
                if st["time"]["compacted"]:
                    mark = " [COMPACTED→2K]"
                segs.append(f'{p["tool"]}({part_tokens(p)}t){mark}')
        print(f"  #{i} {m['role']:<9} {msg_tokens(m):>7}t  " + " · ".join(segs))


def main():
    W = CONTEXT_WINDOW
    msgs = demo()

    print("=" * 76)
    print("OPENCODE TOOL-TRACE COMPACTION — POC (2 KATMAN)")
    print("=" * 76)

    print(f"\n── KATMAN A (canlı spill, üretim anında) " + "─" * 30)
    spilled = [(i, p["tool"], p["state"]["full_lines"]) for i, m in enumerate(msgs)
               for p in m["parts"] if p["type"] == "tool" and p["state"].get("spilled")]
    for i, name, lines in spilled:
        print(f"  #{i} {name}: {lines} satır > {MAX_LINES} → diske döküldü, context'e önizleme+referans")
    if not spilled:
        print("  (spill yok)")

    before = total_tokens(msgs)
    print(f"\n── ÖNCE ({before:,} token) " + "─" * 40)
    render(msgs)

    u = usable(W)
    print(f"\n── KATMAN B / İki ayrı tetik " + "─" * 40)
    print(f"  prune       : cfg.compaction.prune AÇIK → PROAKTİF çalışır (overflow beklemez)")
    print(f"  LLM özeti   : isOverflow = kullanılan({before:,}) ≥ usable({u:,}) → {is_overflow(before, W)}")
    print(f"  → Bu turda deterministik prune çalışır (maliyet için); LLM fazı yalnız overflow'da.")

    print(f"\n── KATMAN B.1 / Deterministik prune (sondan başa, PROAKTİF) " + "─" * 5)
    pruned, to_prune, committed = prune(msgs)

    print(f"\n── KATMAN B / Fayda-freni " + "─" * 40)
    print(f"  budanabilir toplam = {pruned:,}  ·  eşik PRUNE_MINIMUM = {PRUNE_MINIMUM:,}")
    if pruned > PRUNE_MINIMUM:
        print(f"  {pruned:,} > {PRUNE_MINIMUM:,} → COMMIT: {committed} tool 'compacted' damgalandı")
    else:
        print(f"  {pruned:,} ≤ {PRUNE_MINIMUM:,} → COMMIT YOK (cache'i bozmaya değmez)")

    after = total_tokens(msgs)
    print(f"\n── SONRA (serialize: compacted → {TOOL_OUTPUT_MAX_CHARS} kar) ({after:,} token) " + "─" * 8)
    render(msgs)

    ov2 = is_overflow(after, W)
    print(f"\n── KATMAN B / Prune sonrası hâlâ overflow? {ov2}"
          + ("" if not ov2 else "  → LLM özeti fazına geçilir (mock)"))

    saved = before - after
    print("\n── SONUÇ " + "─" * 58)
    print(f"  {before:,} → {after:,} token   |   -{saved:,} ({saved/before*100:.1f}% kazanç)")
    # bütünlük: mesaj/part sayısı değişmedi (silme yok)
    parts_before = sum(len(m["parts"]) for m in msgs)
    print(f"  mesaj sayısı  : {len(msgs)} (değişmedi = silme yok, tool_call bütünlüğü korunur)")
    print(f"  part sayısı   : {parts_before} (değişmedi)")
    print(f"  skill korundu : {'✓' if any(p['type']=='tool' and p['tool']=='skill' and not p['state']['time']['compacted'] for m in msgs for p in m['parts']) else '✗'}")
    print("=" * 76)


if __name__ == "__main__":
    main()
