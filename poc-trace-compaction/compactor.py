"""
compactor.py — Deterministik trace sıkıştırıcı (CWL, arXiv 2606.11213).

§5.2 + §6: bütçe aşılınca, koruma penceresi HARİÇ eski tool birimlerini
ledger'ın sinyalleriyle (dedup/staleness/kategori) 5-alan özete indirir.

CWL ilkeleri (§5.2):
  - LLM ÇAĞIRMAZ (deterministik, halüsinasyon riski yapısal olarak yok)
  - kademeli: en ucuz/güvenli önce (dup → stale → keşif), pahalı sona
  - koruma penceresi: son N birim dokunulmaz
  - "act" (yazma) birimleri "expl" (keşif) birimlerinden ÖNCE atılır
    (etkisi zaten diskte)

İsteğe bağlı olarak bir LLM özetleyici enjekte edilebilir (summarize_fn),
ama varsayılan tamamen deterministiktir.
"""
from __future__ import annotations
import re
import json
from typing import Callable, Optional

from config import estimate_tokens
from trace import Trace, Event, TraceSummary
from ledger import ExecutionLedger


def _raw_cost(ev: Event) -> int:
    """Bir tool olayının HAM (sıkıştırılmamış) token maliyeti."""
    return estimate_tokens(json.dumps(ev.payload, ensure_ascii=False))


def _fayda_var(ev: Event, ozet: TraceSummary) -> bool:
    """Bu özet gerçekten KÜÇÜLTÜYOR mu — İKİ ölçekte birden?

    Fayda güvencesi eskiden yalnız trace muhasebesine (payload JSON vs özet dict)
    bakıyordu. Ama özet bağlama `summary.render()` metni olarak düşüyor ve
    messages[]'te yerini aldığı şey yalnızca `output`. İki ölçek farklı olduğu
    için guard'dan geçen bir birim GERÇEK bağlamda büyüyebiliyordu — ölçüldü:
    31 → 32 token. Ölçüm, etkinin düştüğü yerden alınmalı; ikisini de kontrol et.
    """
    if ozet.token_cost() >= _raw_cost(ev):
        return False
    return estimate_tokens(ozet.render()) < estimate_tokens(ev.payload.get("output", ""))


# expl/act ayrımı (§5.2): keşif mi eylem mi
_EXPL = {"read", "search"}
_ACT = {"write", "test"}

# göreve-koşullu alaka (§11 K5, §12.8) için atılacak yaygın kelimeler
_STOP = {"the", "and", "for", "bir", "ile", "ve", "de", "da", "bul", "yap",
         "sonra", "değerini", "onu", "then", "find", "value"}


def _task_keywords(task: str) -> set[str]:
    """Görev metninden alaka anahtarları: 3+ harf, durak kelime değil."""
    return {w for w in re.findall(r"[A-Za-z0-9_.]+", task.lower())
            if len(w) >= 3 and w not in _STOP}


def _task_relevant(output: str, keywords: set[str]) -> bool:
    """Çıktı görevle örtüşüyor mu (K5 göreve-koşullu sıkıştırma)."""
    if not keywords:
        return False
    low = output.lower()
    return any(k in low for k in keywords)


def _intent_of(trace: Trace, ev: Event) -> str:
    """Bir tool olayının niyetini, onu tetikleyen reasoning'den çıkar (§4).

    Niyet ham trace'te YOKTUR — intent_ref ile bağlı reasoning olayından
    geri kazanılır. Yoksa tool adından zayıf bir niyet üretilir.
    """
    if ev.intent_ref is not None:
        for e in trace.events:
            if e.seq == ev.intent_ref and e.type == "reasoning":
                return e.payload.get("text", "")[:80]
    return f"{ev.payload.get('name', '?')} çağrıldı"


def _summarize_deterministic(trace: Trace, ev: Event, ledger: ExecutionLedger,
                             reason: str, keywords: Optional[set] = None) -> TraceSummary:
    """Bir tool olayını 5-alan özete indir (§4) — LLM'siz.

    Göreve-koşullu (§11 K5): çıktı görevle örtüşüyorsa sonuç daha uzun tutulur;
    örtüşmüyorsa daha sert kırpılır. Verbatim işaret her zaman kazanır.
    """
    name = ev.payload.get("name", "?")
    args = ev.payload.get("args", {})
    output = ev.payload.get("output", "")
    girdi = ", ".join(f"{k}={v}" for k, v in args.items()) or "-"

    # sonuç: verbatim işaretliyse birebir; değilse göreve-koşullu kırpma (K5)
    if ev.verbatim:
        sonuc = output.strip()
    else:
        lines = output.strip().splitlines()
        relevant = _task_relevant(output, keywords or set())
        n_lines = 2 if relevant else 1          # alakalıysa 2 satır, değilse 1
        cap = 120 if relevant else 60           # alakalıysa daha geniş bütçe
        joined = " / ".join(lines[:n_lines])
        sonuc = joined[:cap] + ("…" if len(output) > cap else "")

    return TraceSummary(
        niyet=_intent_of(trace, ev),
        girdi=girdi,
        sonuc=sonuc or "(boş)",
        durum=ev.status if ev.status == "ok" else f"HATA: {ev.status}",
        etki=reason,   # neden sıkıştırıldığını etki alanında iz olarak bırak
    )


class TraceCompactor:
    def __init__(self, budget: int, protect_window: int,
                 summarize_fn: Optional[Callable[[Event], str]] = None,
                 task: str = "", playbook=None, target_ratio: float = 0.5) -> None:
        # İki eşik (Wegent'ten ilham): budget = TETİKLE (bu aşılınca başla),
        # target = BURAYA KADAR İN (belirgin altı). Histerezis: bir kez sıkıştır,
        # bir sonraki tool çağrısı hemen tekrar tetiklemesin (ACM "sawtooth").
        self.budget = budget               # trigger_limit
        self.target = int(budget * target_ratio)   # target_limit
        self.protect_window = protect_window
        self.summarize_fn = summarize_fn   # opsiyonel LLM özetleyici
        self.task = task                   # göreve-koşullu sıkıştırma için (K5)
        self.keywords = _task_keywords(task)
        self.playbook = playbook           # ACE öğrenen bağlam (K4); None ise pas
        self.log: list[str] = []

    def _evict_event(self, trace: Trace, ev: Event, ledger: ExecutionLedger,
                     reason: str) -> bool:
        """B.12 compaction: içeriği 5-alan özete indir (yapı ve iz korunur).

        FAYDA GÜVENCESİ (complexity trap): özet ham'dan KÜÇÜK değilse sıkıştırma
        zararlı olurdu → ham bırak, False dön. Küçük tool çıktılarında olur.
        """
        summary = _summarize_deterministic(trace, ev, ledger, reason, self.keywords)
        if not _fayda_var(ev, summary):
            return False
        ev.summary = summary
        ev.evicted = True
        ev.neden = reason
        self.log.append(f"  seq={ev.seq} {ev.payload.get('name')} → ÖZET · {reason}")
        return True

    def _clear_event(self, ev: Event, note: str) -> bool:
        """B.11 context editing: içeriği SİL (yer tutucu bırak).

        Yalnızca olgu bağlamda başka yerde canlı olduğunda güvenli (§B.11 uyarısı:
        silinen bulgu başka yerde yoksa sessizce kaybolur). Çağıran bunu garanti eder.
        Fayda güvencesi: stub ham'dan küçük değilse dokunma.
        """
        if estimate_tokens(note) + 4 >= _raw_cost(ev):
            return False
        ev.cleared = True
        ev.clear_note = note
        ev.neden = note
        self.log.append(f"  seq={ev.seq} {ev.payload.get('name')} → SİLİNDİ · {note}")
        return True

    def compact(self, trace: Trace, ledger: ExecutionLedger,
                force: bool = False, episode_graph=None) -> dict:
        """Bütçe aşıldıysa (veya force) trace'i sıkıştır. Metrik döndürür.

        episode_graph verilirse (CWL) bağımlılık-farkında episode eviction'ı
        (Faz 6) devreye girer — ajanın delimiter ile bildirdiği yapıyı kullanır.
        """
        before = trace.total_tokens()
        self.log = []

        # Koruma penceresi: son N TOOL birimi dokunulmaz (§6.5)
        # Not: N=0'da liste[-0:] tüm listeyi döndürür — açıkça boş küme kullan.
        tool_evs = [e for e in trace.tool_events() if not e.evicted and not e.cleared]
        protected = (set(e.seq for e in tool_evs[-self.protect_window:])
                     if self.protect_window > 0 else set())
        # ÇÖZÜLMEMİŞ tool koruması (NexAU'dan ilham): düzeltilmemiş bir hata,
        # konumu ne olursa olsun korunur — ajan onu hâlâ çözmeye çalışıyor.
        # (koruma penceresinin konumsal olması bu in-flight durumu kaçırır)
        unresolved = _unresolved_errors(trace)
        protected |= unresolved
        if unresolved:
            self.log.append(f"  koruma: çözülmemiş hata seq {sorted(unresolved)} "
                            f"(düzeltilene kadar dokunulmaz)")

        if not force and before <= self.budget:
            return {"before": before, "after": before, "evicted": 0,
                    "triggered": False, "log": []}

        evicted = 0

        # --- Faz 1: DUPLICATE'ler (en güvenli — sonuç zaten başka yerde) ---
        # Önce ÖZET (5-alan iz + "≡ seq=X" bağı). Ama verbatim bir tekrarda özet =
        # ham olur, fayda freni engeller → o zaman SİL'e düş: bire bir aynı içerik
        # daha erken (canlı) bir olayda durduğu için silme B.11 açısından güvenli.
        for ev in tool_evs:
            if ev.seq in protected or ev.evicted or ev.cleared:
                continue
            det = _detect_duplicate(trace, ev, ledger)
            if det is not None:
                if self._evict_event(trace, ev, ledger, f"tekrar (≡ seq={det})"):
                    evicted += 1
                elif self._clear_event(ev, f"tekrar ≡ seq={det} (aynı içerik canlı)"):
                    evicted += 1

        # --- Faz 2: STALE gözlemler (dosya sonradan değişti) ---
        # B.11 vs B.12 ayrımı: bayat birimin GÜNCEL kopyası bağlamda hâlâ CANLI
        # ise (aynı dosyanın sonraki, evict edilmemiş taze okuması) → SİL (context
        # editing, sıfıra yakın). Değilse → özetle (compaction). §B.11 güvenlik
        # koşulu: silinen olgu başka yerde durmalı.
        for ev in tool_evs:
            if ev.seq in protected or ev.evicted or ev.cleared:
                continue
            if ledger.is_stale(ev.seq):
                fresh = _has_fresher_live_read(trace, ledger, ev)
                if fresh:
                    if self._clear_event(ev, f"bayat — güncel kopya seq={fresh} canlı"):
                        evicted += 1
                else:
                    if self._evict_event(trace, ev, ledger, "bayat (eskidi)"):
                        evicted += 1

        # --- Faz 3: HATA-ZİNCİRİ KATLAMA ---
        # Başarısız bir çağrı + sonrasında aynı tool'un başarılısı → hatayı katla.
        # Risk: hatanın kendisi ders (§13). Bu yüzden hata mesajı VERBATİM korunur.
        for ev in tool_evs:
            if ev.seq in protected or ev.evicted or ev.cleared:
                continue
            corr = _detect_error_correction(trace, ev)
            if corr is not None:
                ev.summary = TraceSummary(
                    niyet=_intent_of(trace, ev),
                    girdi=", ".join(f"{k}={v}" for k, v in
                                    ev.payload.get("args", {}).items()) or "-",
                    sonuc=ev.payload.get("output", "").strip()[:120],  # hata = ders, verbatim
                    durum="HATA (düzeltildi)",
                    etki=f"düzeltme: seq={corr}")
                ev.evicted = True
                ev.neden = f"hata-zinciri — düzeltmesi seq={corr}'te, ders playbook'a alındı"
                evicted += 1
                # ACE Curation (K4): dersi playbook'a ARTIMLI DELTA olarak yaz.
                # Trace özeti sonra sıkışsa da ders playbook'ta kalıcı (collapse yok).
                if self.playbook is not None:
                    self.playbook.curate(_error_lesson(trace, ev, corr),
                                         tag="hata-dersi", source_seq=ev.seq)

        # --- Faz 4: KEŞİF KATLAMA ---
        # Ardışık keşif dizisini (read/search) TEK bulguya indir.
        # Risk: negatif bilgi kaybı → verbatim bulgular (grep sonuçları) korunur.
        # Yalnızca hedefin üstündeysek (kayıplı olduğu için).
        if force or trace.total_tokens() > self.target:
            for run in _exploration_runs(trace, ledger, protected):
                # bulgu = verbatim çıktının BAŞLIK kısmı (ilk ~140 karakter), tüm gövde
                # değil — büyük finansal çıktılarda filler'ı değil metriği tutar.
                findings = [e.payload.get("output", "").strip()[:140]
                            for e in run if e.verbatim]
                finding = " | ".join(f for f in findings if f) or f"{len(run)} keşif adımı"
                # ACE Curation (K4): keşif bulgusunu playbook'a yaz. Aynı bulgu
                # başka dizide de çıkarsa curate dedup eder → helpful++ (delta).
                if self.playbook is not None:
                    for f in findings:
                        if f:
                            self.playbook.curate(f, tag="bulgu")
                # dizinin son birimine roll-up bulgu, öncekiler diziye katlanır
                #
                # FAYDA GÜVENCESİ: `_evict_event` bunu zaten yapıyor ama bu faz
                # kendi özetini elle kuruyor ve kontrolü ATLIYORDU. Düğüm bazlı
                # ÖNCE/SONRA görünümü eklenince ölçüldü: küçük bir tool çıktısı
                # (31 token) 32 token'lık bir "özete" çevriliyordu — sıkıştırma
                # bağlamı BÜYÜTÜYORDU. Küçük çıktılarda 5-alan özetin sabit yükü
                # ham veriden pahalı; o durumda ham bırakmak doğru.
                def _katla(e, ozet, neden) -> bool:
                    if not _fayda_var(e, ozet):
                        self.log.append(f"  seq={e.seq} katlanmadı — özet ham'dan "
                                        f"büyük olurdu ({ozet.token_cost()} ≥ {_raw_cost(e)})")
                        return False
                    e.summary, e.evicted, e.neden = ozet, True, neden
                    return True

                katlanan = 0
                for e in run[:-1]:
                    if _katla(e, TraceSummary(
                            niyet=_intent_of(trace, e),
                            girdi=", ".join(f"{k}={v}" for k, v in
                                            e.payload.get("args", {}).items()) or "-",
                            sonuc="(keşif dizisine katlandı)", durum="ok",
                            etki=f"→ bulgu: seq={run[-1].seq}"),
                            f"keşif dizisine katlandı → bulgu seq={run[-1].seq}"):
                        evicted += 1
                        katlanan += 1
                last = run[-1]
                if _katla(last, TraceSummary(
                        niyet="keşif dizisi", girdi=f"{len(run)} adım",
                        sonuc=finding[:150], durum="ok",
                        etki="keşif katlandı (bulgu korundu)"),
                        f"keşif dizisi ({len(run)} adım) bulguya katlandı"):
                    evicted += 1
                    katlanan += 1
                self.log.append(f"  keşif dizisi [{run[0].seq}..{run[-1].seq}] "
                                 f"→ bulguya katlandı ({katlanan}/{len(run)} adım"
                                 + ("" if katlanan == len(run)
                                    else "; kalanlar ham bırakıldı — özet daha pahalı olurdu")
                                 + ")")

        # --- Faz 5: kademeli — önce ACT, sonra EXPL (§5.2) ---
        # bütçe hâlâ aşılıyorsa yaşça eskiden başlayarak (son çare)
        for phase in (_ACT, _EXPL):
            for ev in tool_evs:
                if trace.total_tokens() <= self.target and not force:
                    break
                if ev.seq in protected or ev.evicted or ev.cleared:
                    continue
                cat = ledger.category_of(ev.seq)
                if cat in phase:
                    label = "eylem (etki diskte)" if cat in _ACT else "keşif (katlandı)"
                    # opsiyonel LLM özet — sadece istenirse
                    if self.summarize_fn is not None:
                        try:
                            note = self.summarize_fn(ev)
                            ev.summary = TraceSummary(
                                niyet=_intent_of(trace, ev),
                                girdi=", ".join(f"{k}={v}" for k, v in
                                                ev.payload.get("args", {}).items()),
                                sonuc=note, durum=ev.status, etki=label)
                            ev.evicted = True
                            ev.neden = f"LLM özeti · {label}"
                            self.log.append(f"  seq={ev.seq} → LLM özet")
                            evicted += 1
                        except Exception:
                            if self._evict_event(trace, ev, ledger, label):
                                evicted += 1
                    else:
                        if self._evict_event(trace, ev, ledger, label):
                            evicted += 1

        # --- Faz 6: CWL EPISODE EVICTION (bağımlılık-farkında) ---
        # Ajan delimiter ile trace'ini tiplediyse: bir EXPL episode ANCAK ona
        # bağlı TÜM act'ler evict edildiyse atılabilir (§5.2 bağımlılık kısıtı).
        # Atılınca episode, ajanın yazdığı description'a iner (tek satır).
        if episode_graph is not None and (force or trace.total_tokens() > self.target):
            evicted_seqs = set(e.seq for e in trace.tool_events()
                               if e.evicted or e.cleared)
            id2ev = {e.seq: e for e in trace.tool_events()}
            for ep in episode_graph.evictable_expl(evicted_seqs):
                live = [s for s in ep.event_seqs
                        if s not in evicted_seqs and s not in protected]
                if not live:
                    continue
                # tüm canlı olayları episode'un description'ına indir
                for i, s in enumerate(live):
                    ev = id2ev.get(s)
                    if ev is None:
                        continue
                    ev.summary = TraceSummary(
                        niyet=f"[{ep.name}] keşif episode'u",
                        girdi=ev.payload.get("name", "?"),
                        sonuc=ep.description if i == len(live) - 1
                        else "(episode'a katlandı)",
                        durum="ok",
                        etki=f"CWL episode eviction: {ep.name}")
                    ev.evicted = True
                    ev.neden = f"CWL episode eviction — '{ep.name}' tamamlandı"
                    evicted += 1
                self.log.append(f"  CWL episode '{ep.name}' [{len(live)} olay] "
                                 f"→ description: \"{ep.description}\"")

        # --- Faz 7: ACİL — EN BÜYÜK ÖNCE (Wegent'ten ilham) ---
        # Fazlardan sonra hâlâ hedefin üstündeysek: kalan ham birimleri TOKEN
        # BOYUTUNA göre (en büyük önce) evict et. Kategori/konum değil, boyut —
        # en çok token'ı en az evict'le geri kazanır. Faz 5'in atlayabildiği
        # "other" kategorisini (run_code, visualize_data) de yakalar.
        if force or trace.total_tokens() > self.target:
            remaining = [e for e in tool_evs
                         if e.seq not in protected and not e.evicted and not e.cleared]
            for ev in sorted(remaining, key=lambda e: e.token_cost(), reverse=True):
                if trace.total_tokens() <= self.target and not force:
                    break
                if self._evict_event(trace, ev, ledger, "acil: en büyük önce"):
                    evicted += 1

        after = trace.total_tokens()
        return {"before": before, "after": after, "evicted": evicted,
                "triggered": True, "saved_pct": round(100 * (before - after) / before, 1)
                if before else 0, "log": self.log}


def _unresolved_errors(trace: Trace) -> set:
    """Düzeltilmemiş hataların seq'leri: status=error olup SONRASINDA aynı tool'un
    başarılı çağrısı OLMAYAN birimler.

    NexAU'nun "unresolved tool-use chain" korumasının POC karşılığı: ajan bu hatayı
    hâlâ çözmeye çalışıyor, o yüzden konumu ne olursa olsun sıkıştırılmamalı.
    (Düzeltilmiş hatalar Faz 3'te zaten katlanır — onlar çözülmüştür.)
    """
    result = set()
    for ev in trace.tool_events():
        if ev.status != "error":
            continue
        fixed = any(l.seq > ev.seq and l.payload.get("name") == ev.payload.get("name")
                    and l.status == "ok" for l in trace.tool_events())
        if not fixed:
            result.add(ev.seq)
    return result


def _has_fresher_live_read(trace: Trace, ledger: ExecutionLedger, ev: Event):
    """ev'in gözlediği KAYNAĞIN daha sonra, hâlâ CANLI (evict/clear edilmemiş, taze)
    ve AYNI tool'la yenilenmiş bir okuması var mı? Varsa o okumanın seq'i döner.

    Bu, B.11 context editing'in güvenlik koşuludur: güncel olgu bağlamda başka
    yerde durduğu için bayat kopyayı silmek bilgi kaybetmez.

    GENEL: kaynak anahtarı ledger sözleşmesinden gelir (dosya path'i VEYA ticker).
    'Aynı tool' şartı kritik: aynı ticker'ın farklı tool'u (oran vs gelir) FARKLI
    içerik döndürür — o yüzden yalnızca AYNI tool'un taze kopyası güvenli siler.
    """
    name = ev.payload.get("name", "")
    res = ledger._resource(name, ev.payload.get("args", {}))
    if not res or ledger.category_of(ev.seq) != "read":
        return None
    for later in trace.tool_events():
        if later.seq <= ev.seq or later.evicted or later.cleared:
            continue
        if (later.payload.get("name", "") == name                       # AYNI tool
                and ledger._resource(name, later.payload.get("args", {})) == res
                and ledger.category_of(later.seq) == "read"
                and not ledger.is_stale(later.seq)):
            return later.seq
    return None


def _error_lesson(trace: Trace, ev: Event, corr_seq: int) -> str:
    """Hata-zincirinden yeniden kullanılabilir bir ders çıkar (ACE için).

    Başarısız çağrı ile düzelten çağrının argümanlarını kıyaslayıp FARKI verir —
    "hangi girdi yanlıştı, doğrusu ne" dersi. Deterministik, LLM'siz.
    """
    name = ev.payload.get("name", "?")
    fail_args = ev.payload.get("args", {})
    corr = next((e for e in trace.tool_events() if e.seq == corr_seq), None)
    corr_args = corr.payload.get("args", {}) if corr else {}
    # farklı olan argümanı bul (örn. old: 'PORT=8080' → 'PORT = 8080')
    diffs = [f"{k}: '{fail_args.get(k)}' değil '{corr_args.get(k)}'"
             for k in fail_args if fail_args.get(k) != corr_args.get(k)]
    path = fail_args.get("path", "")
    diff_txt = "; ".join(diffs) if diffs else ev.payload.get("output", "")[:50]
    return f"{name}({path}): {diff_txt}"


def _detect_duplicate(trace: Trace, ev: Event, ledger: ExecutionLedger):
    """Bu tool olayı, daha önceki (evict edilmemiş) bir çağrının aynısı mı?"""
    name = ev.payload.get("name")
    args = ev.payload.get("args", {})
    for earlier in trace.tool_events():
        if earlier.seq >= ev.seq:
            break
        if earlier.evicted or earlier.cleared:
            continue
        if (earlier.payload.get("name") == name
                and earlier.payload.get("args") == args
                and ledger.category_of(earlier.seq) in _EXPL):
            # aradaki bir yazma bu dosyayı değiştirmediyse gerçekten duplicate
            if not ledger.is_stale(earlier.seq):
                return earlier.seq
    return None


def _detect_error_correction(trace: Trace, ev: Event):
    """ev bir HATA mı ve sonrasında aynı tool'un BAŞARILI çağrısı var mı?

    Varsa, o başarılı çağrının seq'i döner (düzeltme). Hata-zinciri katlama
    bu çifti tek derse indirir — hatanın kendisi ders olduğu için mesaj korunur.
    """
    if ev.status != "error":
        return None
    name = ev.payload.get("name")
    for later in trace.tool_events():
        if later.seq <= ev.seq or later.evicted or later.cleared:
            continue
        if later.payload.get("name") == name and later.status == "ok":
            return later.seq
    return None


def _exploration_runs(trace: Trace, ledger: ExecutionLedger, protected: set):
    """Ardışık keşif (read/search) dizilerini bul — uzunluğu >= 2 olanlar.

    Her dizi, tek bir bulguya katlanabilir (ls/grep dizisi → bulgu).
    Korunan veya zaten evict edilmiş birimler diziyi böler.
    """
    runs, cur = [], []
    for ev in trace.tool_events():
        cat = ledger.category_of(ev.seq)
        eligible = (cat in _EXPL and ev.seq not in protected
                    and not ev.evicted and not ev.cleared)
        if eligible:
            cur.append(ev)
        else:
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
    if len(cur) >= 2:
        runs.append(cur)
    return runs
