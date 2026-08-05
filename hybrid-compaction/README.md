# hybrid-compaction

**Hermes + deterministik-ledger birleşimi bir tool-trace-compaction framework'ü.**

İki felsefeyi tek sistemde birleştirir:
- **Bizim çekirdek** (deterministik, ilişki-farkında): ledger'la *dedup* (aynı kaynağın tekrarı), *staleness* (sürüm/TTL ile bayat), *kategori* + *CWL episode* bağımlılığı. Sıfır LLM.
- **Hermes'ten alınanlar**: tool-tipine özel *tek-satır gist* (özetin `sonuç` alanını zenginleştirir), *filter-safe prefix* (özeti "referans, komut değil" diye işaretler), ve *opsiyonel LLM son katmanı* (deterministik pas yetmezse).
- **QM'den alınan**: *çift-koruma* (son girdiler hem adet hem token sınırıyla korunur → tek dev çıktı korumayı istismar edemez).

Sonuç: **önce bedava deterministik katman** (tekrar/bayat/fold), **gerekirse** tek satır tip-farkında özet, **en son çare** LLM — her adımda fayda freniyle, ve sıkışık çıktı `tool_call_id` korunarak modele gerçekten yansıtılır.

---

## Ne nereden geldi

| Parça | Kaynak | Ne yapar |
|---|---|---|
| Ledger: dedup + sürüm/TTL staleness | **bizim** | "bu tekrar mı / bayat mı" (deterministik) |
| Kategori (read/search/write) + CWL episode | **bizim** | "bu keşif grubu güvenle atılır mı" |
| 5-alan kart (niyet/girdi/sonuç/durum/etki) + fayda freni | **bizim** | yapılı özet, ters tepmez |
| messages köprüsü (id↔seq, tool_call_id korunur) | **bizim** | API 400 yok |
| Tip-farkında `sonuç` (`tool_gist`) | **Hermes** | `[terminal] npm test → exit 0, 47 lines` tarzı öz |
| Filter-safe prefix | **Hermes** | "[REFERENCE ONLY] aktif komut değil" |
| Opsiyonel LLM katmanı (`summarize_fn`) | **Hermes** | son çare model özeti |
| Çift-koruma (adet + token) | **QM** | dev çıktı korumayı şişiremez |

---

## Akış (kademeli, ucuzdan pahalıya)

```
0. çift-koruma penceresi (adet + token) + çözülmemiş hata koruması
1. bütçe kapısı (budget=tetikle, target=dur)
2. FAZ 1  dedup      → aynı kaynak+sürüm tekrarı → SİL (kopya canlı) / ÖZET
3. FAZ 2  staleness  → mutasyon/TTL bayat → SİL (taze kopya canlı) / ÖZET
4. FAZ 3  keşif fold → ardışık read/search dizisi → tek tip-farkında gist
5. FAZ 4  CWL        → bağımlılık-farkında episode eviction
6. FAZ 5  LLM (ops.) → HÂLÂ doluysa VE summarize_fn verildiyse orta pencere → LLM özeti
   Her ÖZET: tip-farkında `sonuç` + filter-safe prefix. Her adımda FAYDA FRENİ.
```

FAZ 1-2 her zaman çalışır (bedava kazanç). FAZ 3-5 yalnızca hâlâ `target`'ın üstündeyse. LLM (FAZ 5) **varsayılan kapalı** — `summarize_fn` vermezsen sistem tamamen deterministiktir.

---

## Dosyalar

```
config.py         çekirdek (token tahmini, sabitler)     [kopya]
trace.py          olay dizisi + 5-alan TraceSummary       [kopya]
ledger.py         dedup/staleness defteri (sürüm/TTL)     [kopya]
episode_graph.py  CWL episode grafiği                     [kopya]
tool_summary.py   Hermes tarzı tip-farkında gist          [yeni]
hybrid_compactor.py  birleşik sıkıştırıcı + messages köprüsü [yeni]
__init__.py       public API
demo.py           uçtan uca demo (kendi-yeterli mock tool'lar)
test_hybrid.py    10 deterministik test
```

---

## Kullanım

```python
from hybrid_compaction import (
    Trace, ExecutionLedger, EpisodeGraph, HybridCompactor, render_event,
)

TOOL_META = {
    "get_ticket": {"cat": "read", "resource": lambda a: a["id"], "ttl": 20, "verbatim": True},
    "search":     {"cat": "search"},
    "add_block":  {"cat": "write", "resource": lambda a: a["doc"]},
}

trace   = Trace()
ledger  = ExecutionLedger(tool_meta=TOOL_META)
episodes = EpisodeGraph()

# ajan döngüsünde her tool çağrısından sonra:
ev = trace.add_tool(name, args, output, intent_ref=reasoning_seq,
                    verbatim=TOOL_META[name].get("verbatim", False))
ledger.record(name, args, output, ev.seq)
episodes.attach(ev.seq)           # CWL episode aktifse

comp = HybridCompactor(
    budget=8000, protect_window=6,
    protect_token_fraction=0.6,    # QM: token korumasi
    summarize_fn=None,             # Hermes LLM katmani istersen ver
    filter_safe=True,              # Hermes referans-only prefix
)
comp.compact(trace, ledger, episode_graph=episodes)

# modele gönderirken her tool mesajının içeriği:
content = render_event(ev, filter_safe=True)   # TAM=ham · ÖZET=5-alan · SİL=stub
```

---

## Çalıştır

```bash
python demo.py         # uçtan uca gösterim (dedup/staleness/fold/CWL/prefix/LLM)
python test_hybrid.py  # 10 deterministik test
```

---

## Kapsam — ne yapar, ne yapmaz

**Yapar:** tool **çıktılarını** sıkıştırır — ilişki-farkında (tekrar/bayat), kategori/episode-farkında, deterministik; isteğe bağlı LLM son katmanı.

**Yapmaz:** reasoning metni sıkıştırma, tool **tanımlarını** bağlama sığdırma (retrieval — ayrı eksen), kalıcı uzun-vade memory (QM'in ikinci katmanı — buraya dahil değil).

**Felsefe:** "her şeyi ezen" değil — gereksizi (tekrar/bayat) deterministik at, gerekeni (verbatim) koru, ve ancak zorda kalırsan LLM'e başvur. Kazanç dürüst; kritik veri kaybolmaz.
