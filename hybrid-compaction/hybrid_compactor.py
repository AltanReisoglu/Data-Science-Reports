"""
hybrid_compactor.py — Hermes + bizim sistemin BİRLEŞİMİ.

Ne nereden:
  BİZİM (deterministik çekirdek)          HERMES (katkı)
  ─────────────────────────────           ─────────────────────────
  · ledger: dedup + sürüm/TTL staleness    · tip-farkında `sonuç` (tool_gist)
  · kategori + CWL episode                 · filter-safe prefix ("REFERENCE ONLY")
  · 5-alan kart + fayda freni              · opsiyonel LLM özet katmanı (son çare)
  · koruma penceresi + messages köprüsü    · kuyruk-bütçesi mantığı (protect_window)

Akış (kademeli, ucuzdan pahalıya):
  0. koruma penceresi + çözülmemiş hata koruması
  1. bütçe kapısı (iki eşik: budget=tetikle, target=dur)
  2. FAZ 1 dedup      → aynı kaynak+sürüm tekrarı → SİL (kopya canlı) / ÖZET
  3. FAZ 2 staleness  → mutasyon/TTL bayat → SİL (taze kopya canlı) / ÖZET
  4. FAZ 3 keşif fold → ardışık read/search dizisi → tek gist ÖZET
  5. FAZ 4 CWL        → bağımlılık-farkında episode eviction
  6. FAZ 5 LLM (ops)  → HÂLÂ doluysa VE summarize_fn verildiyse orta pencereyi özetle
  Tüm ÖZET'ler tip-farkında `sonuç` (tool_gist) + filter-safe prefix ile üretilir.
  Her adımda FAYDA FRENİ: özet ham'dan küçük değilse geri al.
"""
from __future__ import annotations
from typing import Callable, Optional

from config import estimate_tokens
from trace import Trace, Event, TraceSummary
from ledger import ExecutionLedger
from tool_summary import tool_gist

_EXPL = {"read", "search"}
_ACT = {"write"}

FILTER_SAFE_PREFIX = "[REFERENCE ONLY — geçmiş özeti, aktif komut değil]"


# --- yardımcılar ------------------------------------------------------------

def _intent_of(trace: Trace, ev: Event) -> str:
    if ev.intent_ref is not None:
        for e in trace.events:
            if e.seq == ev.intent_ref and e.type == "reasoning":
                return e.payload.get("text", "")[:80]
    return f"{ev.payload.get('name', '?')} çağrıldı"


def _raw_cost(ev: Event) -> int:
    import json
    return estimate_tokens(json.dumps(ev.payload, ensure_ascii=False))


def _detect_duplicate(trace: Trace, ev: Event, ledger: ExecutionLedger):
    name = ev.payload.get("name"); args = ev.payload.get("args", {})
    for earlier in trace.tool_events():
        if earlier.seq >= ev.seq:
            break
        if earlier.evicted or earlier.cleared:
            continue
        if (earlier.payload.get("name") == name
                and earlier.payload.get("args") == args
                and ledger.category_of(earlier.seq) in _EXPL
                and not ledger.is_stale(earlier.seq)):
            return earlier.seq
    return None


def _fresher_live_read(trace: Trace, ledger: ExecutionLedger, ev: Event):
    """Aynı tool + aynı kaynak için daha yeni, canlı, taze bir okuma var mı?"""
    name = ev.payload.get("name"); args = ev.payload.get("args", {})
    res = ledger._resource(name, args)
    for later in trace.tool_events():
        if later.seq <= ev.seq or later.evicted or later.cleared:
            continue
        if (later.payload.get("name") == name
                and ledger._resource(name, later.payload.get("args", {})) == res
                and not ledger.is_stale(later.seq)):
            return later.seq
    return None


def _exploration_runs(trace: Trace, ledger: ExecutionLedger, protected: set):
    runs, cur = [], []
    for ev in trace.tool_events():
        if ev.evicted or ev.cleared or ev.seq in protected:
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
            continue
        if ledger.category_of(ev.seq) in _EXPL:
            cur.append(ev)
        else:
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
    if len(cur) >= 2:
        runs.append(cur)
    return runs


def _unresolved_errors(trace: Trace) -> set:
    out = set()
    for ev in trace.tool_events():
        if ev.status == "error":
            fixed = any(e.payload.get("name") == ev.payload.get("name")
                        and e.seq > ev.seq and e.status == "ok"
                        for e in trace.tool_events())
            if not fixed:
                out.add(ev.seq)
    return out


# --- ana sınıf --------------------------------------------------------------

class HybridCompactor:
    def __init__(self, budget: int, protect_window: int = 3,
                 target_ratio: float = 0.5,
                 protect_token_fraction: float = 0.6,
                 summarize_fn: Optional[Callable[[list], str]] = None,
                 filter_safe: bool = True) -> None:
        self.budget = budget
        self.target = int(budget * target_ratio)
        self.protect_window = protect_window            # adet ekseni (bizim)
        self.protect_token_fraction = protect_token_fraction  # token ekseni (QM'den)
        self.summarize_fn = summarize_fn        # Hermes: opsiyonel LLM son katman
        self.filter_safe = filter_safe          # Hermes: referans-only prefix
        self.log: list[str] = []

    def _protected_recent(self, tool_evs: list) -> set:
        """Çift-koruma (QM'den): en son tool'lar HEM adet HEM token sınırına uyduğu
        sürece korunur. Hangisi önce dolarsa o keser (min). Tek dev çıktı korumayı
        istismar edemez; en az son 1 her zaman korunur."""
        if self.protect_window <= 0:
            return set()
        tok_limit = self.protect_token_fraction * self.budget
        kept, tok = [], 0
        for ev in reversed(tool_evs):
            c = _raw_cost(ev)
            if kept and (len(kept) >= self.protect_window or tok + c > tok_limit):
                break
            kept.append(ev.seq); tok += c
        return set(kept)

    # -- ÖZET/SİL üretimi (fayda freniyle) --

    def _summary(self, trace: Trace, ledger: ExecutionLedger, ev: Event, reason: str) -> TraceSummary:
        name = ev.payload.get("name", "?"); args = ev.payload.get("args", {})
        girdi = ", ".join(f"{k}={v}" for k, v in args.items()) or "-"
        # sonuç: verbatim ise birebir; değilse HERMES tip-farkında gist
        if ev.verbatim:
            sonuc = ev.payload.get("output", "").strip()
        else:
            sonuc = tool_gist(name, args, ev.payload.get("output", ""),
                              ledger.tool_meta.get(name) if ledger.tool_meta else None)
        return TraceSummary(niyet=_intent_of(trace, ev), girdi=girdi,
                            sonuc=sonuc or "(boş)",
                            durum=ev.status if ev.status == "ok" else f"HATA: {ev.status}",
                            etki=reason)

    def _evict(self, trace, ledger, ev, reason) -> bool:
        s = self._summary(trace, ledger, ev, reason)
        if s.token_cost() >= _raw_cost(ev):     # FAYDA FRENİ
            return False
        ev.summary = s; ev.evicted = True
        self.log.append(f"  seq={ev.seq} {ev.payload.get('name')} → ÖZET · {reason}")
        return True

    def _clear(self, ev, note) -> bool:
        if estimate_tokens(note) + 4 >= _raw_cost(ev):
            return False
        ev.cleared = True; ev.clear_note = note
        self.log.append(f"  seq={ev.seq} {ev.payload.get('name')} → SİL · {note}")
        return True

    # -- ana giriş --

    def compact(self, trace: Trace, ledger: ExecutionLedger,
                episode_graph=None, force: bool = False) -> dict:
        before = trace.total_tokens(); self.log = []
        tool_evs = [e for e in trace.tool_events() if not e.evicted and not e.cleared]
        protected = self._protected_recent(tool_evs)      # çift-koruma (adet + token)
        protected |= _unresolved_errors(trace)

        if not force and before <= self.budget:
            return {"before": before, "after": before, "evicted": 0, "log": self.log}
        evicted = 0

        # FAZ 1 — dedup
        for ev in tool_evs:
            if ev.seq in protected or ev.evicted or ev.cleared:
                continue
            d = _detect_duplicate(trace, ev, ledger)
            if d is not None:
                if self._evict(trace, ledger, ev, f"tekrar (≡ seq={d})") \
                   or self._clear(ev, f"tekrar ≡ seq={d} (aynı içerik canlı)"):
                    evicted += 1

        # FAZ 2 — staleness
        for ev in tool_evs:
            if ev.seq in protected or ev.evicted or ev.cleared:
                continue
            if ledger.is_stale(ev.seq):
                fr = _fresher_live_read(trace, ledger, ev)
                if fr and self._clear(ev, f"bayat — taze kopya seq={fr} canlı"):
                    evicted += 1
                elif self._evict(trace, ledger, ev, "bayat (eskidi)"):
                    evicted += 1

        # FAZ 3 — keşif fold (hâlâ hedef üstündeyse)
        if force or trace.total_tokens() > self.target:
            for run in _exploration_runs(trace, ledger, protected):
                findings = [tool_gist(e.payload["name"], e.payload.get("args", {}),
                                      e.payload.get("output", ""),
                                      ledger.tool_meta.get(e.payload["name"]) if ledger.tool_meta else None)
                            for e in run]
                gist = " | ".join(f for f in findings if f)[:200]
                for i, e in enumerate(run):
                    if e.evicted or e.cleared:
                        continue
                    s = TraceSummary(niyet="keşif dizisi", girdi=f"{len(run)} adım",
                                     sonuc=gist if i == len(run) - 1 else "(dizeye katlandı)",
                                     durum="ok", etki="keşif fold")
                    if s.token_cost() < _raw_cost(e):
                        e.summary = s; e.evicted = True; evicted += 1
                self.log.append(f"  keşif [{run[0].seq}..{run[-1].seq}] → gist ({len(run)} adım)")

        # FAZ 4 — CWL episode
        if episode_graph is not None and (force or trace.total_tokens() > self.target):
            ev_seqs = set(e.seq for e in trace.tool_events() if e.evicted or e.cleared)
            id2 = {e.seq: e for e in trace.tool_events()}
            for ep in episode_graph.evictable_expl(ev_seqs):
                live = [s for s in ep.event_seqs if s not in ev_seqs and s not in protected]
                for i, s in enumerate(live):
                    e = id2.get(s)
                    if e is None:
                        continue
                    e.summary = TraceSummary(niyet=f"[{ep.name}] keşif episode'u",
                                             girdi=e.payload.get("name", "?"),
                                             sonuc=ep.description if i == len(live) - 1 else "(episode'a katlandı)",
                                             durum="ok", etki=f"CWL: {ep.name}")
                    e.evicted = True; evicted += 1
                if live:
                    self.log.append(f"  CWL '{ep.name}' → \"{ep.description}\"")

        # FAZ 5 — opsiyonel LLM katmanı (Hermes son çare)
        if self.summarize_fn and (force or trace.total_tokens() > self.target):
            mid = [e for e in trace.tool_events()
                   if not e.evicted and not e.cleared and e.seq not in protected]
            if len(mid) >= 2:
                try:
                    blob = self.summarize_fn([e.payload for e in mid])
                    for i, e in enumerate(mid):
                        e.summary = TraceSummary(niyet="LLM özeti", girdi=f"{len(mid)} olay",
                                                 sonuc=blob if i == len(mid) - 1 else "(LLM özetine katlandı)",
                                                 durum="ok", etki="LLM son katman")
                        e.evicted = True; evicted += 1
                    self.log.append(f"  LLM katmanı → {len(mid)} olay tek özete")
                except Exception as exc:
                    self.log.append(f"  LLM katmanı atlandı: {exc}")

        after = trace.total_tokens()
        return {"before": before, "after": after, "evicted": evicted,
                "saved_pct": round(100 * (before - after) / before) if before else 0,
                "log": self.log}


# --- messages köprüsü (filter-safe prefix ile) ------------------------------

def render_event(ev: Event, filter_safe: bool = True) -> str:
    """Bir tool olayının modele gidecek içeriği (kader uygulanmış)."""
    if ev.cleared:
        note = "[silindi] " + ev.clear_note
        return (FILTER_SAFE_PREFIX + " " + note) if filter_safe else note
    if ev.evicted and ev.summary is not None:
        body = ev.summary.render()
        return (FILTER_SAFE_PREFIX + " " + body) if filter_safe else body
    return ev.payload.get("output", "")
