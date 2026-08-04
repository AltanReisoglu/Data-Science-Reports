"""
run_demo.py — POC girişi.

İki mod:
  python run_demo.py --synthetic   (API'siz; deterministik çekirdeği kanıtlar)
  python run_demo.py --live        (gerçek Gemma döngüsü; .env gerekir)

--synthetic, gerçek bir ajanın ürettiği türden gürültülü bir trace'i
(tekrarlı okuma + yazma sonrası bayat gözlem) compactor'dan geçirir ve
öncesi/sonrası token + hangi birimin neden atıldığını gösterir.
"""
from __future__ import annotations
import argparse

from config import TRACE_TOKEN_BUDGET, TRACE_PROTECT_WINDOW
from trace import Trace
from ledger import ExecutionLedger
from compactor import TraceCompactor
from playbook import Playbook

BAR = "─" * 66
TASK = "config.py'deki PORT değerini bul, 9090 yap, run_tests ile doğrula"


def _big(path: str, n: int = 40) -> str:
    """Gerçekçi büyüklükte sahte dosya içeriği."""
    return "\n".join(f"{i+1:4d}  satır {i} of {path} ....................." for i in range(n))


def build_synthetic() -> tuple[Trace, ExecutionLedger]:
    """Gerçek bir ajanın tipik gürültüsünü taklit eden trace.

    Görev: config.py'deki PORT'u bul, 9090 yap, doğrula.
    Ajanın gerçekte yaptığı (israfla):
      keşif → config'i 3 KEZ okur, grep tekrarı, sonra yazar, sonra ESKİ
      okumalar bayatlar.
    """
    t, L = Trace(), ExecutionLedger()

    def tool(name, args, out, status="ok", verbatim=False):
        r = t.add_reasoning(f"{name} çağıracağım: {args}")
        ev = t.add_tool(name, args, out, status=status,
                        intent_ref=r.seq, verbatim=verbatim)
        L.record(name, args, out, ev.seq)
        return ev

    tool("list_dir", {"path": "."}, "config.py\nserver.py\nutils.py")
    tool("grep", {"pattern": "PORT"}, "config.py:6: PORT = 8080", verbatim=True)
    tool("read_file", {"path": "config.py"}, _big("config.py"))      # 1. okuma
    tool("read_file", {"path": "server.py"}, _big("server.py"))
    tool("read_file", {"path": "config.py"}, _big("config.py"))      # TEKRAR (dup)
    tool("grep", {"pattern": "PORT"}, "config.py:6: PORT = 8080", verbatim=True)  # dup
    tool("read_file", {"path": "utils.py"}, _big("utils.py"))
    tool("read_file", {"path": "config.py"}, _big("config.py"))      # TEKRAR (dup)
    # HATA-ZİNCİRİ: yanlış old ile edit başarısız, sonra doğrusuyla başarılı
    tool("edit_file", {"path": "config.py", "old": "PORT=8080",       # YANLIŞ (boşluksuz)
                       "new": "PORT=9090"},
         "Hata: bulunamadı, değiştirilemedi: 'PORT=8080'", status="error")
    tool("edit_file", {"path": "config.py", "old": "PORT = 8080",    # DÜZELTME (yazma)
                       "new": "PORT = 9090"}, "OK: config.py güncellendi")
    # yazmadan sonra önceki config okumaları ARTIK BAYAT
    tool("read_file", {"path": "config.py"}, _big("config.py"))      # taze (koruma penceresi)
    tool("run_tests", {}, "TEST: health() port=9090", verbatim=True)
    return t, L


def synthetic_demo():
    t, L = build_synthetic()

    print(BAR)
    print("SENTETİK DEMO — trace compaction (deterministik, API'siz)")
    print(BAR)
    print(f"Görev: config.py'deki PORT'u 9090 yap ve doğrula")
    print(f"Bütçe: {TRACE_TOKEN_BUDGET} token · koruma penceresi: son "
          f"{TRACE_PROTECT_WINDOW} tool birimi\n")

    tool_evs = t.tool_events()
    print(f"Ham trace: {len(tool_evs)} tool birimi, "
          f"{t.total_tokens()} token (~char/4)")
    print("Ledger tespitleri:")
    dups = staged = 0
    for ev in tool_evs:
        from compactor import _detect_duplicate
        d = _detect_duplicate(t, ev, L)
        s = L.is_stale(ev.seq)
        tag = []
        if d is not None:
            tag.append(f"DUP≡seq{d}"); dups += 1
        if s:
            tag.append("BAYAT"); staged += 1
        if tag:
            print(f"  seq{ev.seq:>2} {ev.payload['name']:<10} "
                  f"{ev.payload['args']}  →  {', '.join(tag)}")
    print(f"  → {dups} tekrar, {staged} bayat gözlem tespit edildi\n")

    # K5 göreve-koşullu sıkıştırma + K4 ACE playbook devrede
    pb = Playbook()
    comp = TraceCompactor(TRACE_TOKEN_BUDGET, TRACE_PROTECT_WINDOW,
                          task=TASK, playbook=pb)
    res = comp.compact(t, L, force=True)   # force: küçük demoda da göster

    print("SIKIŞTIRMA (SİLİNDİ = B.11 context editing · ÖZET = B.12 compaction):")
    for line in res["log"]:
        print(line)
    print()
    print(f"{'Öncesi':<12}{res['before']:>8} token")
    print(f"{'Sonrası':<12}{res['after']:>8} token")
    print(f"{'Kazanç':<12}{res['saved_pct']:>7}%   ({res['evicted']} birim işlendi)")
    print(BAR)

    # ÜÇ KATMANLI kader dökümü (§13 bilgi-kaybı merdiveni)
    protected = [e.seq for e in tool_evs[-TRACE_PROTECT_WINDOW:]]
    korunan = [e.seq for e in tool_evs if e.seq in protected]
    silinen = [e.seq for e in tool_evs if e.cleared]
    ozetlenen = [e.seq for e in tool_evs if e.evicted]
    print("\nÜÇ KATMANLI KADER (artan bilgi kaybı):")
    print(f"  KORUNDU (tam)   : {korunan}   ← koruma penceresi")
    print(f"  ÖZETLENDİ (B.12): {ozetlenen}   ← 5-alan özet, iz kalır")
    print(f"  SİLİNDİ  (B.11) : {silinen}   ← taze kopyası canlı, olgu kaybolmaz")
    print(BAR)

    # ACE playbook: evict edilenlerden öğrenilen, bağlamda KALICI dersler
    print("\nACE PLAYBOOK (K4) — trace evict edildi ama ders kalıcı:")
    if pb.active_bullets():
        for b in pb.active_bullets():
            print(f"  • {b.render()}")
        print(f"\n  → {pb.stats()['deltas']} artımlı delta işlemi "
              f"(yeniden yazma yok = context collapse yok)")
        print(f"  → playbook maliyeti: {pb.token_cost()} token "
              f"(trace {res['after']} token'a inerken ders korundu)")
    else:
        print("  (bu trace'te ders çıkmadı)")
    print(BAR)
    print("\nÖRNEK — bir birimin ÖZET (B.12) sonrası 5-alan hâli (§4):")
    for ev in tool_evs:
        if ev.evicted and ev.summary:
            print(f"  seq{ev.seq} ham → özet:")
            for k, v in ev.summary.as_dict().items():
                print(f"      {k}: {v}")
            break


def ptc_demo():
    """PTC (§11 K3 / §12.10): ara sonuçlar bağlama girmez — uzaysal yönetim.

    Aynı işi iki yolla yapıp trace token'ını kıyaslar:
      Klasik : her .py dosyasını tek tek read_file → N büyük olay bağlamda
      PTC    : tek run_code → N çağrı sandbox'ta → sadece print bağlamda
    """
    from ptc import PTCSandbox
    from tools import read_file, list_dir, grep

    print(BAR)
    print("PTC DEMO — programatik tool çağrısı (uzaysal, §12.10)")
    print(BAR)
    print("Görev: repodaki tüm .py dosyalarında PORT geçen satırları bul\n")

    py_files = [f for f in sorted(list_dir(".").splitlines()) if f.endswith(".py")]

    # --- KLASİK: her dosyayı tek tek oku → hepsi trace'e girer ---
    classic = Trace()
    for f in py_files:
        r = classic.add_reasoning(f"{f} dosyasını oku")
        classic.add_tool("read_file", {"path": f}, read_file(f), intent_ref=r.seq)
    print(f"KLASİK yol: {len(py_files)} dosya tek tek okundu")
    print(f"  → trace: {len(classic.tool_events())} tool olayı, "
          f"{classic.total_tokens()} token (ham içerik bağlamda)")

    # --- PTC: tek kod bloğu, sadece print bağlama girer ---
    sb = PTCSandbox({"read_file": read_file, "list_dir": list_dir, "grep": grep})
    code = ("for f in sorted(list_dir('.').splitlines()):\n"
            "    if f.endswith('.py'):\n"
            "        hit = grep('PORT', f)\n"
            "        if 'PORT' in hit and 'eşleşme yok' not in hit:\n"
            "            print(hit)\n")
    res = sb.run(code)
    ptc_trace = Trace()
    r = ptc_trace.add_reasoning("tek kod bloğuyla tüm dosyaları tara")
    ptc_trace.add_tool("run_code", {"inner_calls": res["inner_calls"]},
                       res["output"], intent_ref=r.seq, verbatim=True)
    print(f"\nPTC yol: 1 kod bloğu, {res['inner_calls']} çağrı SANDBOX'ta")
    print(f"  → trace: {len(ptc_trace.tool_events())} tool olayı, "
          f"{ptc_trace.total_tokens()} token (sadece print bağlamda)")

    saved = 100 * (classic.total_tokens() - ptc_trace.total_tokens()) / classic.total_tokens()
    print(BAR)
    print(f"Bağlam tasarrufu: {classic.total_tokens()} → {ptc_trace.total_tokens()} "
          f"token ({saved:.0f}% az) · {res['inner_calls']} ara sonuç hiç girmedi")
    print(BAR)
    print("\nBağlama giren TEK şey (PTC print çıktısı):")
    for ln in res["output"].splitlines():
        print(f"  {ln}")

    # --- hata kurtarma: stack trace sandbox'ta kalır (§12.10) ---
    print(BAR)
    print("\nHATA KURTARMA (§12.10): bozuk kod → stack trace sandbox sonucu,")
    print("model kodu düzeltip TUR harcamadan yeniden çalıştırır.")
    bad = sb.run("print(grep('PORT'))\nprint(undefined_var)")
    print(f"  status: {bad['status']}")
    print(f"  sandbox döndürdü (son satır): {bad['output'].splitlines()[-1]}")


def toolsearch_demo():
    """Ertelenmiş tool + ToolSearch (§11 K3 / B.4): şema yükü progressive disclosure.

    Tüm şemalar yüklü (taban) vs yalnızca çekirdek + deferred ad-listesi kıyası;
    sonra bir uzman tool ihtiyaç anında yüklenir (+1 tur)."""
    from tool_registry import demo_registry

    print(BAR)
    print("TOOLSEARCH DEMO — ertelenmiş tool / progressive disclosure (§B.4)")
    print(BAR)
    reg = demo_registry()
    st = reg.stats()
    print(f"Tool sayısı: {st['resident']} çekirdek (resident) + "
          f"{st['deferred']} uzun kuyruk (deferred)\n")

    print(f"{'TABAN (hepsi yüklü, ertelemesiz)':<38}{reg.full_tokens():>6} token")
    print(f"{'DEFERRED (çekirdek şema + ad-listesi)':<38}{reg.context_tokens():>6} token")
    saved = 100 * (reg.full_tokens() - reg.context_tokens()) / reg.full_tokens()
    print(f"{'→ bağlam tasarrufu':<38}{saved:>5.0f}%   (hiçbir uzman tool kullanılmadan)")
    print(BAR)

    print("\nBağlamda deferred tool'lar yalnızca AD olarak duruyor:")
    stub = reg.deferred_stub()
    print("  " + stub[:200] + ("…" if len(stub) > 200 else ""))

    print("\nModel 'run_sql' lazım oldu → önce ToolSearch çağırır:")
    before = reg.context_tokens()
    loaded = reg.tool_search("select:run_sql")
    after = reg.context_tokens()
    print(f"  tool_search('select:run_sql') → {[s['function']['name'] for s in loaded]} yüklendi")
    print(f"  bağlam: {before} → {after} token (+1 tur ödendi, şema şimdi çağrılabilir)")
    print(BAR)
    print("\nKarar (§B.4): sık kullanılanı yerleşik bırak, uzun kuyruğu ertele —")
    print("bağlam maliyeti şema kütlesiyle değil, GERÇEK kullanımla ölçeklenir.")


def live_demo(compaction: bool):
    from agent import TracingAgent
    task = ("config.py'deki PORT değerini bul, onu 9090 yap, sonra run_tests ile "
            "doğrula. Bittiğinde yeni portu bildir.")
    print(BAR)
    print(f"CANLI DEMO — compaction={'AÇIK' if compaction else 'KAPALI (taban)'}")
    print(BAR)
    ag = TracingAgent(compaction=compaction)
    try:
        answer = ag.run(task)
    except RuntimeError as e:
        print(f"\n[canlı mod kullanılamıyor] {e}")
        print("İpucu: önce `python run_demo.py --synthetic` ile çekirdeği görün.")
        return
    print("\nYANIT:", answer)
    print("\nMETRİKLER:")
    for k, v in ag.metrics.items():
        print(f"  {k}: {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="API'siz deterministik demo")
    ap.add_argument("--ptc", action="store_true",
                    help="API'siz PTC demosu (klasik vs sandbox bağlam kıyası)")
    ap.add_argument("--toolsearch", action="store_true",
                    help="API'siz ertelenmiş tool / ToolSearch demosu (§B.4)")
    ap.add_argument("--live", action="store_true", help="Gerçek Gemma döngüsü")
    ap.add_argument("--no-compaction", action="store_true",
                    help="--live ile taban çizgisi (sıkıştırma kapalı)")
    args = ap.parse_args()

    if args.live:
        live_demo(compaction=not args.no_compaction)
    elif args.ptc:
        ptc_demo()
    elif args.toolsearch:
        toolsearch_demo()
    else:
        synthetic_demo()  # varsayılan


if __name__ == "__main__":
    main()
