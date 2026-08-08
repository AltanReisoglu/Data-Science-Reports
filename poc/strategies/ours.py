"""
BİZİM SİSTEM (hybrid-compaction) — ilişki-farkında tool-trace compaction.  [§3]

Kaynak: ledger.py (`record`, `is_stale`, `_detect_duplicate`), compactor.py (fazlar),
        tool_summary.py (`tool_gist`), trace.py (`render_event`), episode_graph.py (CWL).

Manzaradaki tek GENEL ilişki ledger'ı: her tool çağrısını kaynak/sürüm/kategori/TTL
ile işler ve ÇAĞRILAR ARASI ilişkiyi bilir:
  * FAZ 1 — dedup: aynı kaynak+sürüm tekrar okundu → eskiyi SİL ("en güncele bak").
  * FAZ 2 — staleness: sonraki bir write kaynağı mutasyona uğrattı → eski okuma BAYAT → SİL.
  * FAZ 3 — tip-farkında gist: kalan büyük eski çıktıyı tek satıra indir (fayda freni ile).
Cline dosya-özel, sürümsüz yapar; biz HER kaynak için sürüm sayacı + TTL + CWL ile.
Sıfır LLM.
"""
from __future__ import annotations

from harness import ToolResult, Conversation, est
from .base import Strategy, Fate


def tool_gist(r: ToolResult) -> str:
    """Tip-farkında tek-satır 'sonuç' (Hermes'ten esinli, bizim 5-alanın kısası)."""
    n_lines = r.content.count("\n") + 1
    if r.tool_type == "terminal":
        code = 0
        for ln in reversed(r.content.splitlines()):
            if ln.strip().startswith("exit "):
                code = ln.strip().split()[1]; break
        return f"[özet] niyet:komut · girdi:{r.resource} · sonuç:exit {code}, {n_lines} satır · durum:ok"
    if r.tool_type == "read_file":
        return f"[özet] niyet:oku · girdi:{r.resource} · sonuç:{n_lines} satır kod · durum:ok"
    if r.tool_type == "grep":
        return f"[özet] niyet:ara · girdi:{r.resource} · sonuç:{n_lines-1} eşleşme · durum:ok"
    if r.tool_type in ("web_extract", "take_snapshot"):
        return f"[özet] niyet:getir · girdi:{r.resource} · sonuç:{n_lines} birim · durum:ok"
    return f"[özet] niyet:{r.name} · girdi:{r.resource} · durum:ok"


class OursStrategy(Strategy):
    name = "ours"
    repo = "adapted/hybrid-compaction (bizim)"
    ref = "§3"
    blurb = "ilişki-farkında ledger: dedup + staleness (sürüm/TTL) + tip-farkında gist + fayda freni (sıfır LLM)"
    uses_llm = False

    def compact(self, results: list[ToolResult], conv: Conversation, budget: int) -> str:
        # Koruma penceresi SADECE kayıplı gist'e (FAZ 3) uygulanır. Dedup/staleness
        # GÜVENLİ ilişki işlemleridir — her zaman EN TAZE kopyayı tutup fazlalığı
        # atar, o yüzden "son N" içinde olsalar bile serbest çalışırlar (Cline de öyle).
        protected = self._recent_ids(results)

        # --- ledger: kaynak başına sürüm haritası (write_file mutasyonları) ---
        def writes_before(res: str, idx: int) -> int:
            return sum(1 for j, x in enumerate(results)
                       if j < idx and x.tool_type == "write_file" and x.resource == res)

        def total_writes(res: str) -> int:
            return sum(1 for x in results if x.tool_type == "write_file" and x.resource == res)

        reads = [(i, r) for i, r in enumerate(results) if r.tool_type == "read_file"]

        # FAZ 2 — staleness: sonraki write kaynağı bayatlattı (ver < current). En taze
        # okuma korunur; eski sürümler bayat → SİL. FAYDA FRENİ: not ham'dan küçükse uygula
        # (küçük çıktıyı sıkıştırmak zarar; Cline'da bu fren yok — bizim avantaj).
        for i, r in reads:
            ver = writes_before(r.resource, i)
            cur = total_writes(r.resource)
            if ver < cur:
                note = f"[bayat: {r.resource} v{ver}<v{cur} — en güncel okumaya bak]"
                if est(note) < r.raw_tokens():
                    r.view, r.fate, r.note = note, Fate.SIL, "is_stale (mutasyon)"

        # FAZ 1 — dedup: aynı kaynak+sürüm tekrar okundu → son (en güncel) hariç eskiler SİL
        groups: dict[tuple, list[tuple]] = {}
        for i, r in reads:
            if r.fate == Fate.SIL:  # zaten bayat
                continue
            ver = writes_before(r.resource, i)
            groups.setdefault((r.resource, ver), []).append((i, r))
        for (res, ver), grp in groups.items():
            for i, r in grp[:-1]:  # sonuncusu (en güncel) korunur; eski kopyalar atılır
                note = f"[dedup: {res} v{ver} zaten okundu — en güncele bak]"
                if est(note) < r.raw_tokens():  # fayda freni
                    r.view, r.fate, r.note = note, Fate.DEDUP, "_detect_duplicate"

        # FAZ 3 — tip-farkında gist (KAYIPLI): hâlâ doluysa kalan büyük eski çıktıyı tek
        # satıra indir. Kayıplı olduğu için koruma penceresine SAYGI duyar.
        for r in results:
            if not self._over_budget(results, budget):
                break
            if id(r) in protected or r.fate != Fate.TAM:
                continue
            gist = tool_gist(r)
            if est(gist) < r.raw_tokens():        # fayda freni: özet ham'dan küçükse uygula
                r.view, r.fate, r.note = gist, Fate.OZET, "tool_gist"
            # değilse TAM bırak (revert) — özet ham'dan büyükse zarar

        return ""
