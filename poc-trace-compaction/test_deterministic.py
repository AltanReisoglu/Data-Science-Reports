"""
test_deterministic.py — Çekirdeğin (ledger + compactor) LLM'siz doğrulaması.

`python test_deterministic.py` ile çalışır; API key GEREKMEZ.
Ledger + CWL araştırmasının "sıfır LLM" iddiasını doğrudan test eder.
"""
from trace import Trace
from ledger import ExecutionLedger
from compactor import TraceCompactor, _detect_duplicate, _raw_cost


def _mk():
    return Trace(), ExecutionLedger()


def test_duplicate_read_detected():
    t, L = _mk()
    r1 = t.add_reasoning("oku"); e1 = t.add_tool("read_file", {"path": "a.py"}, "içerik", intent_ref=r1.seq)
    L.record("read_file", {"path": "a.py"}, "içerik", e1.seq)
    r2 = t.add_reasoning("tekrar oku"); e2 = t.add_tool("read_file", {"path": "a.py"}, "içerik", intent_ref=r2.seq)
    L.record("read_file", {"path": "a.py"}, "içerik", e2.seq)
    assert _detect_duplicate(t, e2, L) == e1.seq, "aynı dosyanın 2. okuması dup olmalı"
    print("✓ duplicate okuma tespit ediliyor")


def test_staleness_after_write():
    t, L = _mk()
    r1 = t.add_reasoning("oku"); e1 = t.add_tool("read_file", {"path": "config.py"}, "PORT=8080", intent_ref=r1.seq)
    L.record("read_file", {"path": "config.py"}, "PORT=8080", e1.seq)
    assert not L.is_stale(e1.seq), "yazma öncesi bayat olmamalı"
    r2 = t.add_reasoning("yaz"); e2 = t.add_tool("edit_file", {"path": "config.py"}, "OK", intent_ref=r2.seq)
    L.record("edit_file", {"path": "config.py"}, "OK", e2.seq)
    assert L.is_stale(e1.seq), "yazmadan sonra eski okuma BAYAT olmalı"
    print("✓ yazma sonrası staleness (change counter) çalışıyor")


def test_eviction_preserves_window():
    t, L = _mk()
    for i in range(6):
        r = t.add_reasoning(f"oku {i}")
        e = t.add_tool("read_file", {"path": f"f{i}.py"}, "x" * 400, intent_ref=r.seq)
        L.record("read_file", {"path": f"f{i}.py"}, "x" * 400, e.seq)
    protect = 3
    comp = TraceCompactor(budget=10, protect_window=protect)
    tool_evs = t.tool_events()
    protected = set(e.seq for e in tool_evs[-protect:])
    comp.compact(t, L, force=True)
    for e in t.tool_events():
        if e.seq in protected:
            assert not e.evicted, f"koruma penceresindeki seq{e.seq} evict edilmemeli"
    print("✓ koruma penceresi (son 3 birim) korunuyor")


def test_summary_has_five_fields():
    t, L = _mk()
    r = t.add_reasoning("portu bulmak için grep")
    # Gerçekçi (uzun) grep çıktısı: 5-alan özet bundan belirgin küçük olur, böylece
    # fayda freni özetlemeyi onaylar (kısa çıktıda özet=ham olur, bilinçli engellenir).
    long_out = "\n".join(f"src/mod{i}.py:{i*3}: PORT_{i} = {8000+i}  # bağlantı ayarı"
                         for i in range(20))
    e = t.add_tool("grep", {"pattern": "PORT"}, long_out, intent_ref=r.seq, verbatim=True)
    L.record("grep", {"pattern": "PORT"}, long_out, e.seq)
    e2 = t.add_tool("grep", {"pattern": "PORT"}, long_out, intent_ref=r.seq)
    L.record("grep", {"pattern": "PORT"}, long_out, e2.seq)
    comp = TraceCompactor(budget=5, protect_window=0)
    comp.compact(t, L, force=True)
    ev = next(e for e in t.tool_events() if e.evicted)
    d = ev.summary.as_dict()
    assert "niyet" in d and "girdi" in d and "sonuc" in d and "durum" in d
    assert "grep" in d["niyet"] or "port" in d["niyet"].lower(), "niyet reasoning'den gelmeli"
    print("✓ 5-alan özet üretiliyor, niyet reasoning'den geri kazanılıyor")


def test_verbatim_preserved():
    t, L = _mk()
    r = t.add_reasoning("test")
    crit = "config.py:6: PORT = 8080"
    e = t.add_tool("grep", {"pattern": "PORT"}, crit, intent_ref=r.seq, verbatim=True)
    L.record("grep", {"pattern": "PORT"}, crit, e.seq)
    from compactor import _summarize_deterministic
    s = _summarize_deterministic(t, e, L, "test")
    assert crit in s.sonuc, "verbatim işaretli kritik string birebir korunmalı"
    print("✓ verbatim kritik string (dosya:satır) korunuyor")


def test_error_chain_folding():
    """Başarısız edit + sonra başarılı edit → hata katlanır, mesaj korunur."""
    t, L = _mk()
    r1 = t.add_reasoning("portu değiştir")
    e1 = t.add_tool("edit_file", {"path": "config.py", "old": "PORT=80"},
                    "Hata: bulunamadı: 'PORT=80'", status="error", intent_ref=r1.seq)
    L.record("edit_file", {"path": "config.py", "old": "PORT=80"}, "Hata...", e1.seq)
    r2 = t.add_reasoning("doğru old ile tekrar")
    e2 = t.add_tool("edit_file", {"path": "config.py", "old": "PORT = 8080"},
                    "OK: güncellendi", status="ok", intent_ref=r2.seq)
    L.record("edit_file", {"path": "config.py", "old": "PORT = 8080"}, "OK", e2.seq)

    comp = TraceCompactor(budget=5, protect_window=0)
    comp.compact(t, L, force=True)
    assert e1.evicted, "başarısız deneme katlanmalı"
    assert "HATA" in e1.summary.durum, "durum HATA olarak işaretlenmeli"
    assert "bulunamadı" in e1.summary.sonuc, "hata mesajı (ders) korunmalı"
    assert f"seq={e2.seq}" in e1.summary.etki, "düzeltmeye referans olmalı"
    print("✓ hata-zinciri katlama: hata mesajı korunuyor, düzeltmeye bağlanıyor")


def test_exploration_folding():
    """Ardışık keşif dizisi tek bulguya katlanır, verbatim bulgu korunur."""
    t, L = _mk()
    # 3 ardışık keşif: list_dir, grep (verbatim bulgu), read
    for name, args, out, vb in [
            ("list_dir", {"path": "."}, "a.py\nb.py", False),
            ("grep", {"pattern": "PORT"}, "config.py:5: PORT=8080", True),
            ("read_file", {"path": "config.py"}, "x" * 300, False)]:
        r = t.add_reasoning(f"{name}")
        e = t.add_tool(name, args, out, intent_ref=r.seq, verbatim=vb)
        L.record(name, args, out, e.seq)

    comp = TraceCompactor(budget=5, protect_window=0)
    comp.compact(t, L, force=True)
    evicted = [e for e in t.tool_events() if e.evicted]
    # Dizinin SON birimi roll-up bulguyu taşır — asıl değişmez bu.
    assert t.tool_events()[-1].evicted, "dizinin son birimi bulguya katlanmalı"
    finding = evicted[-1].summary.sonuc
    assert "config.py:5" in finding, "verbatim bulgu (grep) korunmalı — negatif bilgi kaybı yok"

    # FAYDA GÜVENCESİ: hiçbir birim sıkıştırma sonrası BÜYÜMEMELİ.
    # Eskiden bu faz `_evict_event`'in fayda kontrolünü atlıyordu; 31 token'lık
    # bir çıktı 32 token'lık "özete" çevrilebiliyordu. Sayı yerine bunu ölçüyoruz:
    # kaç birimin katlandığı değil, katlamanın ZARARLI olmaması önemli.
    for e in t.tool_events():
        if e.evicted:
            assert e.summary.token_cost() < _raw_cost(e), \
                f"seq={e.seq} özet ham'dan büyük — sıkıştırma bağlamı büyütüyor"
    print(f"✓ keşif katlama: dizi bulguya indi ({len(evicted)}/3 birim katlandı, "
          f"kalanı ham bırakıldı), verbatim bulgu korundu, hiçbir birim büyümedi")


def test_cwl_episode_eviction_respects_dependency():
    """CWL: expl episode ancak ona bağlı act evict edildiyse atılabilir."""
    from episode_graph import EpisodeGraph
    t, L = _mk()
    g = EpisodeGraph()

    # expl episode: config keşfi (2 okuma)
    g.start("config-kesfi", "expl", seq=t._seq)
    for _ in range(2):
        r = t.add_reasoning("oku"); e = t.add_tool("read_file", {"path": "config.py"},
                                                   "x" * 300, intent_ref=r.seq)
        L.record("read_file", {"path": "config.py"}, "x" * 300, e.seq)
        g.attach(e.seq)
    g.end(seq=t._seq, description="port config.py:6'da, 8080")

    # act episode: config-kesfi'ne BAĞLI, henüz evict edilmedi
    g.start("port-yaz", "act", seq=t._seq, dependencies=["config-kesfi"])
    r = t.add_reasoning("yaz"); ea = t.add_tool("edit_file", {"path": "config.py"},
                                                "OK", intent_ref=r.seq)
    L.record("edit_file", {"path": "config.py"}, "OK", ea.seq)
    g.attach(ea.seq)
    g.end(seq=t._seq)

    comp = TraceCompactor(budget=5, protect_window=0)
    # act henüz evict edilmemişken → expl KORUNMALI (bağımlılık kısıtı)
    evictable_before = g.evictable_expl(evicted_seqs=set())
    assert not evictable_before, "act canlıyken expl evict edilebilir OLMAMALI"

    # act evict edilirse → expl artık atılabilir
    evictable_after = g.evictable_expl(evicted_seqs={ea.seq})
    assert any(ep.name == "config-kesfi" for ep in evictable_after), \
        "act evict edilince expl atılabilir olmalı"
    print("✓ CWL episode eviction bağımlılık kısıtına uyuyor (act önce, expl sonra)")


def test_playbook_incremental_delta():
    """ACE (K4): aynı ders iki kez curate edilirse YENİDEN YAZILMAZ, helpful++."""
    from playbook import Playbook
    pb = Playbook()
    b1 = pb.curate("edit_file(config.py): old must match exactly", tag="hata-dersi")
    b2 = pb.curate("  edit_file(config.py):   OLD MUST match  exactly ",
                   tag="hata-dersi")  # aynı ders, boşluk/harf farkı
    assert b1 is not None, "ilk ekleme yeni madde olmalı"
    assert b2 is None, "tekrar → yeni madde YOK (context collapse'a karşı)"
    assert len(pb.bullets) == 1, "tek madde kalmalı"
    assert pb.bullets[0].helpful == 2, "tekrar teyit → helpful artmalı"
    print("✓ ACE playbook artımlı delta: tekrar madde yeniden yazılmıyor, helpful++")


def test_error_lesson_reaches_playbook():
    """Hata-zinciri katlanınca ders playbook'a yazılmalı (Reflector→Curator)."""
    from playbook import Playbook
    t, L = _mk()
    r1 = t.add_reasoning("portu değiştir")
    e1 = t.add_tool("edit_file", {"path": "config.py", "old": "PORT=80"},
                    "Hata: bulunamadı", status="error", intent_ref=r1.seq)
    L.record("edit_file", {"path": "config.py", "old": "PORT=80"}, "Hata", e1.seq)
    r2 = t.add_reasoning("doğrusuyla")
    e2 = t.add_tool("edit_file", {"path": "config.py", "old": "PORT = 8080"},
                    "OK", intent_ref=r2.seq)
    L.record("edit_file", {"path": "config.py", "old": "PORT = 8080"}, "OK", e2.seq)

    pb = Playbook()
    TraceCompactor(budget=5, protect_window=0, playbook=pb).compact(t, L, force=True)
    lessons = [b for b in pb.bullets if b.tag == "hata-dersi"]
    assert lessons, "hata-zincirinden ders çıkmalı"
    assert "PORT=80" in lessons[0].text and "PORT = 8080" in lessons[0].text, \
        "ders yanlış ve doğru girdiyi içermeli"
    print("✓ hata dersi playbook'a ulaşıyor (evict'ten korunur)")


def test_context_editing_vs_compaction():
    """B.11 vs B.12: bayat okumanın TAZE kopyası canlıysa SİL, değilse ÖZETLE."""
    t, L = _mk()
    # config.py oku (seq1) → yaz (seq3, bayatlatır) → tekrar taze oku (seq5, canlı)
    r1 = t.add_reasoning("oku"); e1 = t.add_tool("read_file", {"path": "config.py"},
                                                 "PORT=8080\n" + "x" * 300, intent_ref=r1.seq)
    L.record("read_file", {"path": "config.py"}, "PORT=8080\n" + "x" * 300, e1.seq)
    r2 = t.add_reasoning("yaz"); e2 = t.add_tool("edit_file", {"path": "config.py"},
                                                 "OK", intent_ref=r2.seq)
    L.record("edit_file", {"path": "config.py"}, "OK", e2.seq)
    r3 = t.add_reasoning("taze oku"); e3 = t.add_tool("read_file", {"path": "config.py"},
                                                      "PORT=9090\n" + "x" * 300, intent_ref=r3.seq)
    L.record("read_file", {"path": "config.py"}, "PORT=9090\n" + "x" * 300, e3.seq)

    TraceCompactor(budget=5, protect_window=0).compact(t, L, force=True)
    assert e1.cleared and not e1.evicted, "bayat e1: taze kopya (e3) canlı → SİLİNMELİ (B.11)"
    assert f"seq={e3.seq}" in e1.clear_note, "silme notu güncel kopyaya işaret etmeli"
    print("✓ B.11 context editing: taze kopyası canlı bayat birim silindi (özetlenmedi)")


def test_task_conditioned_detail():
    """K5: göreve alakalı çıktı daha uzun tutulur, alakasız daha sert kırpılır."""
    from compactor import _summarize_deterministic, _task_keywords
    t, L = _mk()
    kw = _task_keywords("config.py'deki PORT değerini 9090 yap")
    r = t.add_reasoning("oku")
    rel = t.add_tool("read_file", {"path": "config.py"},
                     "PORT ayarı burada\nikinci satır detay", intent_ref=r.seq)
    r2 = t.add_reasoning("oku2")
    irr = t.add_tool("read_file", {"path": "logo.txt"},
                     "telif bilgisi satırı\nikinci satır detay", intent_ref=r2.seq)
    s_rel = _summarize_deterministic(t, rel, L, "keşif", kw)
    s_irr = _summarize_deterministic(t, irr, L, "keşif", kw)
    assert len(s_rel.sonuc) >= len(s_irr.sonuc), "alakalı çıktı en az alakasız kadar detaylı"
    assert "ikinci satır" in s_rel.sonuc, "alakalı olan 2. satırı da tutmalı"
    print("✓ K5 göreve-koşullu: alakalı çıktı daha çok, alakasız daha az korunuyor")


def test_ptc_only_print_enters_context():
    """PTC (K3): N iç çağrı sandbox'ta kalır, bağlama yalnızca print girer."""
    from ptc import PTCSandbox
    calls = {"n": 0}

    def fake_read(path):
        calls["n"] += 1
        return "x" * 500          # her okuma büyük — klasikte hepsi bağlama girerdi

    sb = PTCSandbox({"read_file": fake_read})
    res = sb.run("total = 0\n"
                 "for f in ['a.py', 'b.py', 'c.py']:\n"
                 "    total += len(read_file(f))\n"
                 "print('toplam', total)")
    assert res["status"] == "ok"
    assert res["inner_calls"] == 3, "3 iç çağrı sandbox'ta olmalı"
    assert res["output"] == "toplam 1500", "bağlama yalnızca print girer"
    assert len(res["output"]) < 20, "1500 karakterlik 3 okuma değil, kısa print girdi"
    print("✓ PTC: ara sonuçlar sandbox'ta, bağlama yalnızca print giriyor")


def test_ptc_error_stays_in_sandbox():
    """PTC hata kurtarma: bozuk kod stack trace döndürür, tur harcamaz (§12.10)."""
    from ptc import PTCSandbox
    sb = PTCSandbox({})
    res = sb.run("print(bozuk_degisken)")
    assert res["status"] == "error", "hata yakalanmalı"
    assert "NameError" in res["output"], "stack trace sandbox sonucu olmalı"
    print("✓ PTC hata kurtarma: stack trace sandbox'ta kalıyor")


def test_ptc_sandbox_blocks_unsafe_builtins():
    """PTC sandbox: open/__import__ gibi güvensiz builtin'ler namespace'te yok."""
    from ptc import PTCSandbox
    sb = PTCSandbox({})
    res = sb.run("open('/etc/passwd')")
    assert res["status"] == "error", "open sandbox'ta engellenmeli"
    assert "NameError" in res["output"] or "not defined" in res["output"]
    print("✓ PTC sandbox güvensiz builtin'leri (open/import) engelliyor")


def test_equity_b11_clear_via_ticker_ttl():
    """Equity: erken volatil okuma (TTL) + aynı tool'un taze kopyası → B.11 SİL."""
    import equity_tools as eq
    t = Trace(); L = ExecutionLedger(tool_meta=eq.TOOL_META)
    def add(name, args, out):
        r = t.add_reasoning(name); e = t.add_tool(name, args, out, intent_ref=r.seq)
        L.record(name, args, out, e.seq); return e
    p1 = add("get_stock_price", {"ticker": "XOM"}, "XOM $112.30")   # erken, ttl=1
    add("get_company_info", {"ticker": "XOM"}, "x" * 400)
    add("get_income_statements", {"ticker": "XOM"}, "y" * 400)
    add("get_stock_price", {"ticker": "XOM"}, "XOM $112.30")        # taze tekrar
    TraceCompactor(40, 0).compact(t, L, force=True)
    assert p1.cleared and not p1.evicted, "bayat fiyat, taze kopyası canlı → B.11 SİL (ticker domaininde)"
    print("✓ equity B.11: ticker+TTL ile bayat okuma silindi (taze kopya canlı)")


def test_equity_error_chain_empty_ticker():
    """Equity: boş ticker → gerçek hata → düzeltme → Faz 3 katlar, ders playbook'a."""
    import equity_tools as eq
    from playbook import Playbook
    t = Trace(); L = ExecutionLedger(tool_meta=eq.TOOL_META); pb = Playbook()
    try:
        out = eq.get_key_financial_ratios(""); st = "ok"
    except Exception as ex:
        out = f"Hata: {ex}"; st = "error"
    assert st == "error", "boş ticker exception atmalı (hata-zinciri reachable)"
    r = t.add_reasoning("dene")
    e1 = t.add_tool("get_key_financial_ratios", {"ticker": ""}, out, status=st, intent_ref=r.seq)
    L.record("get_key_financial_ratios", {"ticker": ""}, out, e1.seq)
    r2 = t.add_reasoning("düzelt")
    e2 = t.add_tool("get_key_financial_ratios", {"ticker": "XOM"}, "P/E 13.2", intent_ref=r2.seq)
    L.record("get_key_financial_ratios", {"ticker": "XOM"}, "P/E 13.2", e2.seq)
    TraceCompactor(40, 0, playbook=pb).compact(t, L, force=True)
    assert e1.evicted and "HATA" in e1.summary.durum, "hatalı deneme katlanmalı"
    assert any("ticker" in b.text for b in pb.active_bullets()), "ders playbook'a yazılmalı"
    print("✓ equity hata-zinciri: boş ticker → düzeltme katlandı, ders öğrenildi")


def test_generalized_ledger_ticker_resource():
    """Genel ledger: tool_meta ile kaynak ticker olur, dedup ticker üzerinden çalışır."""
    meta = {"get_ratios": {"cat": "read", "resource": lambda a: a["ticker"]}}
    L = ExecutionLedger(tool_meta=meta); t = Trace()
    r1 = t.add_reasoning("oran"); e1 = t.add_tool("get_ratios", {"ticker": "XOM"}, "P/E 13", intent_ref=r1.seq)
    L.record("get_ratios", {"ticker": "XOM"}, "P/E 13", e1.seq)
    r2 = t.add_reasoning("tekrar"); e2 = t.add_tool("get_ratios", {"ticker": "XOM"}, "P/E 13", intent_ref=r2.seq)
    L.record("get_ratios", {"ticker": "XOM"}, "P/E 13", e2.seq)
    assert _detect_duplicate(t, e2, L) == e1.seq, "aynı ticker'ın 2. okuması dup (path yok, ticker var)"
    print("✓ genel ledger: kaynak = ticker; dedup dosya olmadan da çalışıyor")


def test_ttl_volatility_staleness():
    """TTL: volatil kaynak (fiyat) YAZMA olmadan zamanla bayatlar (mutasyon değil)."""
    meta = {"get_price": {"cat": "read", "resource": lambda a: a["ticker"], "ttl": 2}}
    L = ExecutionLedger(tool_meta=meta); t = Trace()
    r = t.add_reasoning("fiyat"); e = t.add_tool("get_price", {"ticker": "XOM"}, "112.3", intent_ref=r.seq)
    L.record("get_price", {"ticker": "XOM"}, "112.3", e.seq)
    assert not L.is_stale(e.seq), "okuma anında taze"
    for _ in range(3):  # 3 adım geç — YAZMA YOK, sadece zaman
        r2 = t.add_reasoning("x"); e2 = t.add_tool("get_price", {"ticker": "AAPL"}, "1", intent_ref=r2.seq)
        L.record("get_price", {"ticker": "AAPL"}, "1", e2.seq)
    assert L.is_stale(e.seq), "ttl aşılınca yazma olmadan bayat (volatilite eskimesi)"
    print("✓ TTL: volatil kaynak yazma olmadan zamanla bayatlıyor (fiyat/web/saat)")


def test_benefit_guard_skips_tiny():
    """Fayda güvencesi: özet ham'dan küçük değilse sıkıştırma yapılmaz (complexity trap)."""
    t, L = _mk()
    r = t.add_reasoning("küçük"); e = t.add_tool("run_code", {"code": "c"}, "ok", intent_ref=r.seq)
    L.record("run_code", {"code": "c"}, "ok", e.seq)
    TraceCompactor(budget=5, protect_window=0).compact(t, L, force=True)
    assert not e.evicted and not e.cleared, "küçük çıktı sıkıştırılınca büyürdü → ham bırakılmalı"
    print("✓ fayda güvencesi: küçük çıktı sıkıştırılmıyor (özet ham'dan büyük olurdu)")


def test_two_thresholds_hysteresis():
    """İki eşik: sıkıştırma budget'a değil TARGET'a (belirgin altı) kadar iner."""
    def mk_trace():
        t, L = _mk()
        for i in range(6):  # 6 bağımsız büyük okuma, aralarına yazma (run kırılır)
            r = t.add_reasoning(f"oku {i}")
            e = t.add_tool("read_file", {"path": f"f{i}.py"}, "x" * 2000, intent_ref=r.seq)
            L.record("read_file", {"path": f"f{i}.py"}, "x" * 2000, e.seq)
            r2 = t.add_reasoning(f"yaz {i}")
            e2 = t.add_tool("edit_file", {"path": f"g{i}.py"}, "OK", intent_ref=r2.seq)
            L.record("edit_file", {"path": f"g{i}.py"}, "OK", e2.seq)
        return t, L
    ta, La = mk_trace(); tb, Lb = mk_trace()
    comp_a = TraceCompactor(budget=2500, protect_window=2, target_ratio=1.0)  # eski: tek eşik
    comp_b = TraceCompactor(budget=2500, protect_window=2, target_ratio=0.6)  # yeni: iki eşik
    comp_a.compact(ta, La); comp_b.compact(tb, Lb)
    assert tb.total_tokens() <= comp_b.target, "iki-eşik target'ın altına inmeli"
    assert tb.total_tokens() < ta.total_tokens(), "target<budget → daha fazla evict (histerezis)"
    print("✓ iki eşik: sıkıştırma target'a iniyor, budget'ta durmuyor (histerezis)")


def test_unresolved_error_protected():
    """Düzeltilmemiş hata konumdan bağımsız KORUNUR; düzeltilmiş hata katlanır."""
    # düzeltilmemiş: başarısız edit, sonrası YOK
    t, L = _mk()
    r = t.add_reasoning("dene")
    e = t.add_tool("edit_file", {"path": "a.py", "old": "X"}, "Hata: bulunamadı",
                   status="error", intent_ref=r.seq)
    L.record("edit_file", {"path": "a.py", "old": "X"}, "Hata", e.seq)
    for i in range(3):  # bütçeyi aşan gürültü
        r2 = t.add_reasoning(f"oku{i}"); e2 = t.add_tool("read_file", {"path": f"f{i}.py"},
                                                         "x" * 500, intent_ref=r2.seq)
        L.record("read_file", {"path": f"f{i}.py"}, "x" * 500, e2.seq)
    TraceCompactor(budget=100, protect_window=0).compact(t, L, force=True)
    assert not e.evicted and not e.cleared, "çözülmemiş hata force'ta bile korunmalı"
    print("✓ çözülmemiş hata korunuyor (ajan hâlâ çözecek); düzeltilmiş hata katlanır (Faz 3)")


def test_greedy_largest_first_emergency():
    """Faz 7: Faz 5'in atladığı 'other' kategorisini boyutça en büyük önce evict eder."""
    t, L = _mk()
    r = t.add_reasoning("kod çalıştır")  # run_code → kategori 'other', Faz5 dokunmaz
    big = t.add_tool("run_code", {"code": "x"}, "y" * 800, intent_ref=r.seq)
    L.record("run_code", {"code": "x"}, "y" * 800, big.seq)
    r2 = t.add_reasoning("küçük"); small = t.add_tool("run_code", {"code": "z"}, "ok", intent_ref=r2.seq)
    L.record("run_code", {"code": "z"}, "ok", small.seq)
    comp = TraceCompactor(budget=100, protect_window=0)
    comp.compact(t, L, force=True)
    assert big.evicted, "büyük 'other' birim acil fazda evict edilmeli"
    assert any("acil" in line for line in comp.log), "Faz 7 log'da görünmeli"
    print("✓ greedy-by-size acil faz: 'other' kategorisini en büyük önce yakalıyor")


def test_toolsearch_defers_schema_cost():
    """§B.4: deferred tool bağlamda yalnızca AD tutar; şema yükü ertelenir."""
    from tool_registry import demo_registry
    reg = demo_registry()
    assert reg.context_tokens() < reg.full_tokens(), \
        "ertelenmiş kayıt tam yükten ucuz olmalı"
    # deferred bir tool'un şeması başta active değil
    active_before = {s["function"]["name"] for s in reg.active_schemas()}
    assert "run_sql" not in active_before, "deferred tool başta yüklü olmamalı"
    assert "read_file" in active_before, "resident tool hep yüklü olmalı"
    print("✓ ToolSearch: deferred şema ertelendi, resident hep yüklü")


def test_toolsearch_loads_on_demand():
    """§B.4: tool_search('select:x') şemayı yükler → sonra çağrılabilir."""
    from tool_registry import demo_registry
    reg = demo_registry()
    before = reg.context_tokens()
    loaded = reg.tool_search("select:run_sql")
    assert any(s["function"]["name"] == "run_sql" for s in loaded), "run_sql yüklenmeli"
    assert "run_sql" in {s["function"]["name"] for s in reg.active_schemas()}, \
        "yüklenince active olmalı"
    assert reg.context_tokens() > before, "yükleme bağlama şema ekler (+1 tur bedeli)"
    print("✓ ToolSearch: tool ihtiyaç anında yükleniyor (select ile)")


def test_toolsearch_keyword_match():
    """§B.4: anahtar kelimeyle de deferred tool bulunabilir."""
    from tool_registry import demo_registry
    reg = demo_registry()
    loaded = reg.tool_search("SQL sorgu")
    assert any(s["function"]["name"] == "run_sql" for s in loaded), \
        "anahtar kelime 'sql' run_sql'i bulmalı"
    print("✓ ToolSearch: anahtar kelime araması deferred tool'u buluyor")


if __name__ == "__main__":
    tests = [test_duplicate_read_detected, test_staleness_after_write,
             test_eviction_preserves_window, test_summary_has_five_fields,
             test_verbatim_preserved,
             test_error_chain_folding, test_exploration_folding,
             test_cwl_episode_eviction_respects_dependency,
             test_playbook_incremental_delta, test_error_lesson_reaches_playbook,
             test_context_editing_vs_compaction, test_task_conditioned_detail,
             test_ptc_only_print_enters_context, test_ptc_error_stays_in_sandbox,
             test_ptc_sandbox_blocks_unsafe_builtins,
             test_toolsearch_defers_schema_cost, test_toolsearch_loads_on_demand,
             test_toolsearch_keyword_match,
             test_two_thresholds_hysteresis, test_unresolved_error_protected,
             test_greedy_largest_first_emergency,
             test_generalized_ledger_ticker_resource, test_ttl_volatility_staleness,
             test_benefit_guard_skips_tiny,
             test_equity_b11_clear_via_ticker_ttl, test_equity_error_chain_empty_ticker]
    print("Deterministik çekirdek testleri (API'siz):\n")
    for fn in tests:
        fn()
    print(f"\n{len(tests)}/{len(tests)} geçti.")
