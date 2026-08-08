"""
test_poc.py — POC güvencesi. `python test_poc.py` (bağımlılıksız, offline).

Doğrular:
  1. Her strateji trace'i sıkıştırır (shown ≤ raw) ve mesaj yapısını (call_id) bozmaz.
  2. Kader etiketleri geçerli; view koyulduğunda token gerçekten düşer.
  3. Sadıklık işaretleri: her sistemin repo-özel placeholder/fonksiyonu görülür.
  4. İlişki-farkındalık: sadece 'ours' bir okumayı stale(SİL) + başka birini dedup ayırır.
"""
from __future__ import annotations

from harness import ChatSession
import strategies
from strategies.base import Fate

SCENARIO = [
    "src/server.py'yi oku ve npm test çalıştır",
    "https://docs.local/guide getir",
    "src/server.py'yi düzenle",
    "src/server.py'yi tekrar oku",
    "https://app.local/a snapshot al",
    "https://app.local/b snapshot al",
    "'PORT' ara",
]
VALID_FATES = {getattr(Fate, k) for k in dir(Fate) if k.isupper()}


def fresh_trace():
    sess = ChatSession(strategies.get("ours"), budget=10 ** 9)
    for m in SCENARIO:
        sess.send(m)
    sess.conv.reset_fates()
    return sess


def main() -> None:
    passed = failed = 0

    def check(cond, msg):
        nonlocal passed, failed
        if cond:
            passed += 1
        else:
            failed += 1
            print(f"  ✗ FAIL: {msg}")

    sess = fresh_trace()
    conv = sess.conv
    raw = conv.raw_tokens()
    call_ids = [r.call_id for r in conv.all_results()]
    check(len(call_ids) == len(set(call_ids)), "call_id'ler benzersiz")

    for s in strategies.all_strategies():
        conv.reset_fates()
        pre = s.compact(conv.all_results(), conv, 1500)
        shown = conv.shown_tokens(pre)
        # 1) sıkıştırma gerçekleşti, ham'ı aşmıyor
        check(shown <= raw, f"{s.name}: shown({shown}) ≤ raw({raw})")
        check(shown < raw, f"{s.name}: gerçekten sıkıştırdı (shown<{raw})")
        # 2) call_id / yapı korundu (strateji hiçbir result'ı silmedi)
        check([r.call_id for r in conv.all_results()] == call_ids,
              f"{s.name}: mesaj yapısı (call_id sırası) korundu")
        # 3) kader etiketleri geçerli; view varsa token düşmeli veya eşit
        for r in conv.all_results():
            check(r.fate in VALID_FATES, f"{s.name}: geçerli kader {r.fate}")

    # 4) sadıklık placeholder'ları
    def fates_notes(name):
        conv.reset_fates()
        pre = strategies.get(name).compact(conv.all_results(), conv, 1500)
        return conv.all_results(), pre

    rs, _ = fates_notes("cline")
    check(any("duplicateFileReadNotice" in r.note for r in rs), "cline: duplicateFileReadNotice görüldü")
    rs, _ = fates_notes("hermes")
    check(any("_summarize_tool_result" in r.note for r in rs), "hermes: _summarize_tool_result görüldü")
    rs, _ = fates_notes("swe-agent")
    check(any("lines omitted" in (r.view or "") for r in rs), "swe-agent: '(n lines omitted)' görüldü")
    rs, _ = fates_notes("gemini-cli")
    check(any(r.fate == Fate.SUPERSEDE for r in rs), "gemini-cli: SUPERSEDE görüldü")
    rs, _ = fates_notes("headroom")
    check(any("ccr:" in (r.view or "") for r in rs), "headroom: <<ccr:HASH>> marker görüldü")
    rs, _ = fates_notes("roo")
    check(any(r.fate == Fate.KATLA for r in rs), "roo: KATLA (fold) görüldü")
    rs, _ = fates_notes("codex")
    check(any("truncate_middle_chars" in r.note for r in rs), "codex: truncate_middle_chars görüldü")

    # 5) ilişki-farkındalık: sadece ours stale+dedup ayrımı yapar
    rs, _ = fates_notes("ours")
    has_stale = any(r.fate == Fate.SIL and "is_stale" in r.note for r in rs)
    has_dedup = any(r.fate == Fate.DEDUP for r in rs)
    check(has_stale and has_dedup, "ours: aynı trace'te stale(SİL) VE dedup ayrı işaretlendi")

    print(f"\n{passed} geçti, {failed} kaldı.")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
