#!/usr/bin/env python3
"""
kiyas.py — Beş tool-trace POC'unu YAPILANDIRILMIŞ veri olarak koşturur.

`web_server.py` bugüne kadar POC'ları subprocess ile çalıştırıp stdout'u ekrana
basıyordu. Stdout insan için yazılmış: hangi tool'un ham çıktısının ne olduğu ve
sıkıştırmadan sonra context'te ne kaldığı oradan çıkarılamıyor.

Bu modül POC'ları **import ederek** kendi gerçek fonksiyonlarını çağırır ve her
tool birimi için ÖNCE (ham çıktı) / SONRA (context'te kalan) çiftini kırpılmadan
döndürür. POC'ların basma yolu (`main()`) değişmedi — "gerçek POC koşuyor"
güvencesi duruyor, yalnızca aynı mantığa ikinci bir kapı açıldı.

Her adaptör POC'un KENDİ senaryosunu, KENDİ sabitleriyle, KENDİ sıkıştırma
fonksiyonuyla koşturur. Buradaki tek iş: ham çıktıyı sıkıştırma öncesinde
yakalayıp sonrasıyla eşleştirmek.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

MANTIKLAR = ("hermes", "opencode", "openclaw", "codex", "claude_code")

BILGI = {
    "hermes": {
        "baslik": "Hermes", "ekol": "deterministik", "llm": False,
        "kaynak": "NousResearch/Hermes-Agent · agent/context_compressor.py",
        "ozet": "Dört geçiş, LLM'siz: dedup (byte-identik sonuç → geri-referans) → "
                "tip-farkında tek satır özet → tool-call argüman kırpma → basınç demotion. "
                "Mesaj SİLİNMEZ, yalnız içerik küçülür; tool_call_id çiftleri hep bütün kalır."},
    "opencode": {
        "baslik": "OpenCode", "ekol": "deterministik", "llm": False,
        "kaynak": "opencode · truncate.ts + prune",
        "ozet": "İki katman. A: çıktı ÜRETİLİRKEN >2000 satır / >50KB ise diske dökülür, "
                "context'e önizleme + dosya referansı girer. B: sondan başa yürüyen prune — "
                "son 2 turn ve en yeni 40K token dokunulmaz, ötesi damgalanır."},
    "openclaw": {
        "baslik": "OpenClaw", "ekol": "LLM-özet", "llm": True,
        "kaynak": "openclaw · 12 adımlı compaction hattı",
        "ozet": "Önce sanitize (toolResult.details silinir — sır sızmaz), sonra projeksiyon, "
                "adaptif oran, çift-koruyan gruplama, chunk'lama ve her chunk için LLM özeti. "
                "Devasa birimler özete hiç girmez, yerlerine 'not' bırakılır."},
    "codex": {
        "baslik": "Codex", "ekol": "hibrit", "llm": True,
        "kaynak": "codex · truncate_middle + model-turn windowing",
        "ozet": "A: her büyük tool çıktısı üretilirken ortadan kesilir (baş+son kalır). "
                "B1: birikince eski function-output'lar placeholder'a iner. B2: o da yetmezse "
                "handoff özeti üretilip YENİ pencere açılır — eski çıktılar tamamen düşer."},
    "claude_code": {
        "baslik": "Claude Code", "ekol": "hibrit", "llm": True,
        "kaynak": "claude code · microcompaction + auto-compaction + subagent",
        "ozet": "A: eşik üstü tool çıktısı diske yazılır, context'e önizleme + referans kalır "
                "(microcompaction). B: bağlam dolunca eski turn'ler konuşma özetine iner. "
                "C: büyük yan-iş subagent'ta AYRI pencerede koşar, ana bağlama sadece özet döner."},
}


# ─────────────────────── EK-2: yalnız TOOL-TRACE'e dokunan adımlar ───────────
# Ölçüt: adım ya tool çıktısına dokunuyor ya da tool_call ↔ tool_result çiftini
# ilgilendiriyor. Genel context-yönetimi adımları (tetik eşiği, token sayımı,
# worker thread, konuşma özeti, cache freni) bilerek DIŞARIDA — her mantığın
# `disarida` listesinde hangilerinin elendiği yazılı.
#
# `tool_izi: False` olan adım tool izinin kendisine ait DEĞİLDİR (konuşma
# seviyesidir) ama bu koşuda iz üzerinde etkisi olduysa gizlemiyoruz — gri
# gösterilip "kapsam dışı ama vurdu" diye işaretleniyor.
ADIMLAR = {
    "hermes": {
        "tetik": {"uretim": None,
                  "esik": "Proaktif — pencere dolmasını beklemez, LLM'siz olduğu için "
                          "her turda çalışabilir. Tek fren: toplam kazanç 4096 token'ın "
                          "altındaysa hiçbir şeye dokunmaz (prompt-cache'i boşuna bozmamak için)."},
        "adimlar": [
            {"kod": "dedup", "ad": "1 · Dedup", "etiket": "aynı tool çıktısı → referans",
             "kayip": "kayıpsız",
             "ozet": "Byte-byte aynı tool sonucu iki kez tutulmaz; en yenisi tam kalır, "
                     "eskiler '[Duplicate tool output — same content as a more recent call]' "
                     "referansına iner. İçerik hâlâ transcript'te."},
            {"kod": "ozet", "ad": "2 · Tip-farkında özet", "etiket": "büyük çıktı → tek satır",
             "kayip": "içerik gider, kimlik kalır",
             "ozet": "Büyük ve benzersiz tool çıktısı, tool'un TİPİNE göre tek satıra iner "
                     "([read_file] dosya okundu (40.000 chars) / [grep] 120 eşleşme). "
                     "Hangi tool, ne kadar veri — bu kalır."},
            {"kod": "arg", "ad": "3 · Argüman kırpma", "etiket": "tool_call argümanı",
             "kayip": "argüman gövdesi",
             "ozet": "Sadece sonuç değil ÇAĞRININ argümanı da şişer; 500 karakteri aşan "
                     "argüman JSON'ın İÇİNDE kırpılır ki çıktı geçerli JSON kalsın ve "
                     "sağlayıcı 400 dönmesin."},
            {"kod": "basinc", "ad": "4 · Basınç demotion", "etiket": "en yeni tool son çare",
             "kayip": "içerik tamamen",
             "ozet": "Korunan bölge bile taşarsa tool sonuçları kademeli demote edilir; "
                     "en yeni tool bilinçli olarak en sona saklanır ve ancak son çare "
                     "olarak feda edilir."},
        ],
        "disarida": ["proaktif tetik", "boundary hesabı", "4096-token cache freni"],
    },
    "openclaw": {
        "tetik": {"uretim": None,
                  "esik": "[0] adımında — boş yer pencerenin YARISININ altına düştüğünde "
                          "(boş < pencere × 0.5). Üretim anında çalışan katmanı yok; "
                          "her şey bu tek eşikte olur."},
        "adimlar": [
            {"kod": "sanitize", "ad": "[1] Sanitize", "etiket": "toolResult.details sil",
             "kayip": "yan alanlar (kasıtlı)",
             "ozet": "toolResult.details alanı silinir. Tool çıktısı birazdan bir LLM'e "
                     "gidecek; API anahtarı gibi SIRLARIN özete sızması burada engellenir."},
            {"kod": "projection", "ad": "[3] Projection", "etiket": "dev gövde → 8KB örnek",
             "kayip": "yok (çalışma kopyası)",
             "ozet": "Dev tool gövdeleri 8KB örneğe indirilir ama AĞIRLIKLARI gerçek "
                     "boyutta sayılmaya devam eder. Final transcript'e girmez; amacı "
                     "özetleyici LLM'i patlatmamak."},
            {"kod": "gruplama", "ad": "[5] Gruplama", "etiket": "call ↔ result atomik",
             "kayip": "yok (bütünlük koruması)",
             "ozet": "tool_call ile tool_result atomik bir grup sayılır; asla ayrı "
                     "chunk'lara düşemezler — yoksa çift kırılır ve API isteği reddedilir."},
            {"kod": "oversized", "ad": "[7] Oversized", "etiket": "dev sonuç → NOT, çift düşer",
             "kayip": "tamamen (bilinçli feragat)",
             "ozet": "Tek başına pencerenin yarısını aşan tool sonucu ÖZETLENMEYE "
                     "ÇALIŞILMAZ; tek satır NOT'a iner ve çifti birlikte düşer. "
                     "Bu özet değil, bilinçli feragattir."},
            {"kod": "onarim", "ad": "[11] Onarım", "etiket": "yetim çift → sentetik sonuç",
             "kayip": "yok (bütünlük onarımı)",
             "ozet": "Özetleme sonrası çifti kopmuş bir tool_call kaldıysa ona sentetik "
                     "sonuç uydurulur; zincir kırık bırakılmaz."},
        ],
        "disarida": ["[0] tetik", "[2] estimate", "[4] adaptif oran", "[8] stage-split",
                     "[9] worker", "[10] LLM özeti", "[12] uygula"],
    },
    "opencode": {
        "tetik": {"uretim": "Tool çıktısı ÜRETİLİRKEN >2000 satır veya >50KB ise anında "
                            "diske döker.",
                  "esik": "Her turda proaktif prune (bedava olduğu için overflow beklemez). "
                          "LLM'li overflow özeti ancak kullanılan ≥ pencere − 20K buffer "
                          "olunca devreye girer."},
        "adimlar": [
            {"kod": "spill", "ad": "[A] Canlı spill", "etiket": ">2000 satır / 50KB → diske",
             "kayip": "geri çağrılabilir (diskte)",
             "ozet": "Tool çıktısı üretilirken eşiği aşarsa diske yazılır; context'e "
                     "önizleme + dosya referansı girer. Dev çıktı context'e HİÇ girmez "
                     "ve tam içerik geri çağrılabilir."},
            {"kod": "skill", "ad": "[B] Skill koruması", "etiket": "skill çıktısı atlanır",
             "kayip": "yok",
             "ozet": "skill tool çıktıları budamadan MUAFTIR; referans materyaldir, "
                     "eskise de değerini yitirmez."},
            {"kod": "son2turn", "ad": "[B] Son-2-turn koruması", "etiket": "son 2 turn'e HİÇ bakılmaz",
             "kayip": "yok (koruma)",
             "ozet": "Prune sondan başa yürürken user mesajlarını sayar; ilk 2 turn'e hiç "
                     "girmez. Bu birimler 40K SAYACINA DA GİRMEZ — yani '40K korunur' derken "
                     "kastedilen, son 2 turn'ün ÜSTÜNE eklenen 40K'dır. Bu yüzden pratikte "
                     "korunan hacim 40K'yı rahatça aşabilir."},
            {"kod": "sicak40k", "ad": "[B] 40K buda bütçesi", "etiket": "yalnız BUDANABİLİR birim sayılır",
             "kayip": "yok (koruma)",
             "ozet": "Prune sondan başa yürürken gördüğü tool çıktılarını toplar; toplam "
                     "40.000'e varana kadar hepsi korunur, ötesi buda-adayı olur. Ölçüt mesaj "
                     "SAYISI değil, tool izinin HACMİ. "
                     "ADI YANILTICI OLMASIN: bu 'context'in en yeni 40K'sı' DEĞİL. İki muafiyet "
                     "sayacın ÖNÜNDE duruyor — son-2-turn birimleri ve korunan tool'lar (skill) "
                     "`total +=` satırına hiç ulaşmıyor. O yüzden korunan gerçek hacim 40K'yı "
                     "rahatça aşar. "
                     "SAYACIN YÖNÜ: prune SONDAN BAŞA yürür, aşağıdaki liste ise oluşturma "
                     "sırasındadır — bu yüzden listede aşağı indikçe sayaç KÜÇÜLÜR. Her "
                     "birimde yazan değer, prune o durağa geldiğinde birikmiş olan toplamdır."},
            {"kod": "damga", "ad": "[B] Compacted damgası", "etiket": "tool.state.time",
             "kayip": "yok (o an)",
             "ozet": "Buda kararı, tool'un state.time.compacted alanına basılan zaman "
                     "damgasıdır. İçerik O AN değişmez; damga hem 'serialize'da küçült' "
                     "hem de sonraki prune için 'buradan öteye geçme' anlamına gelir."},
            {"kod": "serialize", "ad": "[B] Serialize", "etiket": "damgalı çıktı → 2000 karakter",
             "kayip": "gövde (mesaj durur)",
             "ozet": "Asıl küçülme burada: damgalı tool çıktısı context'e yazılırken "
                     "TOOL_OUTPUT_MAX_CHARS=2000 karaktere iner. Mesaj SİLİNMEZ, çift bozulmaz."},
        ],
        "disarida": ["proaktif tetik", "20K fayda-freni (cache)", "overflow LLM özeti"],
    },
    "codex": {
        "tetik": {"uretim": "Tek tool çıktısı kendisine tanınan TAVANI aşarsa truncate_middle "
                            "anında devreye girer — bu tavan pencereden BAĞIMSIZ sabit bir "
                            "sınırdır (POC'ta TOOL_BUDGET_TOKENS=5.000); pencere bomboş olsa "
                            "bile kesme yapılır.",
                  "esik": "History pencereye sığmazsa (POC'ta CONTEXT_WINDOW=30.000) önce "
                          "placeholder trim, o da yetmezse handoff özeti + yeni pencere. "
                          "Codex proaktif çalışmaz — taşınca müdahale eder."},
        "adimlar": [
            {"kod": "truncate", "ad": "1 · truncate_middle", "etiket": "BAŞ+SON tut, ORTA at",
             "kayip": "orta gövde",
             "ozet": "Tek bir tool çıktısı bütçeyi aşarsa başı ve sonu tutulur, ortası "
                     "atılır; üstüne 'Warning: truncated output' konur. Gerekçe: çıktının "
                     "BAŞI (imports/imza) ve SONU (hata satırı/exit kodu) en bilgilendirici, "
                     "ORTASI en tekrarlı kısımdır."},
            {"kod": "multimodal", "ad": "2 · Multimodal muafiyeti", "etiket": "görsel kesilmez",
             "kayip": "yok",
             "ozet": "Görsel içerikli tool sonuçları bu kesmeden MUAFTIR; bir resmin "
                     "ortasını atmak onu tamamen anlamsız kılar."},
            {"kod": "placeholder", "ad": "3 · Placeholder trim", "etiket": "output → iskelet",
             "kayip": "içerik (iskelet kalır)",
             "ozet": "History sığmazsa en eski function_call_output'lar teker teker "
                     "placeholder'a çevrilir: içerik gider, ÇAĞRI İSKELETİ kalır — böylece "
                     "tool_call ↔ tool_result zinciri kırılmaz."},
            {"kod": "windowing", "ad": "B2 · Windowing (kapsam dışı)", "tool_izi": False,
             "etiket": "handoff özeti + yeni pencere", "kayip": "tool çıktıları tamamen",
             "ozet": "EK-2'ye göre bu adım tool-trace DIŞIDIR — konuşma seviyesidir. Ama "
                     "bu koşuda function-output'ların tamamını düşürdüğü için gizlenmiyor: "
                     "iz üzerindeki etkisi ölçülebilir."},
        ],
        "disarida": ["handoff özeti (SUMMARIZATION_PROMPT)", "CompactedItem ile yeni pencere"],
    },
    "claude_code": {
        "tetik": {"uretim": "Microcompaction, tek tool çıktısı ~4K token'ı aştığında üretim "
                            "anında (gözlem).",
                  "esik": "Auto-compaction context ~%80'de tetiklenir — ama o KONUŞMA "
                          "seviyesidir, tool-trace kapsamında değil."},
        "adimlar": [
            {"kod": "micro", "ad": "[A] Microcompaction", "etiket": "büyük çıktı → diske + referans",
             "kayip": "geri çağrılabilir (diskte)",
             "ozet": "Tek bir tool çıktısı ~4K token'ı aşarsa diske yazılır "
                     "(tool-results/…txt) ve context'te ~500 token'lık önizleme + "
                     "'Full output saved to:' referansı kalır. Model içeriğin KAYBOLMADIĞINI, "
                     "gerekirse dosyadan okuyabileceğini bilir."},
            {"kod": "subagent", "ad": "[C] Subagent kaçışı", "etiket": "iz AYRI pencerede oluşur",
             "kayip": "yok (iz hiç oluşmaz)",
             "ozet": "Compaction'a ALTERNATİF: büyük bir yan-iş ayrı bir context penceresinde "
                     "koşar, ana pencereye yalnız damıtılmış özet döner. Ara adımların tool "
                     "izi ana context'e HİÇ girmez — sıkıştırılacak bir iz oluşmaz bile. "
                     "Sıkıştırmanın en ucuz hali: hiç üretmemek."},
            {"kod": "auto", "ad": "B · Auto-compaction (kapsam dışı)", "tool_izi": False,
             "etiket": "eski turn'ler → konuşma özeti", "kayip": "tool çıktıları tamamen",
             "ozet": "EK-2'ye göre tool-trace DIŞI (turn seviyesi). Bu koşuda eski turn'lerin "
                     "tool çıktılarını tamamen özete sürüklediği için gösteriliyor."},
        ],
        "disarida": ["auto-compaction (konuşma özeti, turn seviyesi)",
                     "Pre/PostCompact hook'ları", "anti-thrash koruması"],
    },
}


def est(t: str) -> int:
    return max(0, len(t or "") // 4)


def _satir(cid, ad, arg, once, sonra, kader, neden, adim="", zincir=None):
    """zincir: bu birime SIRAYLA dokunan adım kodları (Codex'te A sonra B2 gibi)."""
    ot, st = est(once), est(sonra)
    return {"cid": cid, "ad": ad, "arg": arg,
            "once": once, "once_tok": ot,
            "sonra": sonra, "sonra_tok": st,
            "kader": kader, "neden": neden,
            "adim": adim, "zincir": zincir or ([adim] if adim else []),
            "pct": round((1 - st / ot) * 100, 1) if ot else 0.0}


def _paket(ad, once, sonra, toollar, ek, log, ekstra_vurus=None):
    """ekstra_vurus: tool birimine bağlanmayan adım vuruşları (ör. arg kırpma,
    subagent kaçışı) — {kod: (sayi, aciklama)}."""
    b, sema = BILGI[ad], ADIMLAR[ad]
    ekstra_vurus = ekstra_vurus or {}
    adimlar = []
    for a in sema["adimlar"]:
        vuran = [t["cid"] for t in toollar if a["kod"] in (t.get("zincir") or [])]
        n, nasil = ekstra_vurus.get(a["kod"], (0, ""))
        adimlar.append({**a, "tool_izi": a.get("tool_izi", True),
                        "vurdu": vuran, "sayi": len(vuran) + n,
                        "ek_not": nasil})
    return {"ad": ad, "baslik": b["baslik"], "ekol": b["ekol"], "llm": b["llm"],
            "kaynak": b["kaynak"], "ozet": b["ozet"],
            "tetik": sema["tetik"], "adimlar": adimlar, "disarida": sema["disarida"],
            "once": once, "sonra": sonra,
            "pct": round((1 - sonra / once) * 100, 1) if once else 0.0,
            "toollar": toollar, "ek_mesajlar": ek, "log": log}


# ───────────────────────────────── HERMES ────────────────────────────────────

def _hermes() -> dict:
    import hermes_tool_trace_poc as H

    trace = H.demo_trace()
    cagri = {}
    for m in trace:
        for tc in m.get("tool_calls") or []:
            cagri[tc["id"]] = (tc["function"]["name"], tc["function"]["arguments"])
    ham = {m["tool_call_id"]: m["content"] for m in trace if m.get("role") == "tool"}

    once = H.total_tokens(trace)
    pruned, stats = H.prune_old_tool_results(
        trace, protect_tail_count=20, protect_tail_tokens=2000,
        spare_protected_skills={"github"}, verbose=False)
    sonra_map = {m["tool_call_id"]: m["content"] for m in pruned if m.get("role") == "tool"}

    toollar = []
    for cid, icerik in ham.items():
        ad, arg = cagri.get(cid, ("?", ""))
        s = sonra_map.get(cid, "")
        if s == icerik:
            kader, adim = "tam", ""
            neden = "korunan kuyrukta ya da eşik altında — hiçbir geçiş dokunmadı"
        elif s == H._PRUNED_TOOL_PLACEHOLDER:
            kader, adim = "silindi", "basinc"
            neden = ("Pass 4 basınç demotion — korunan bölge bile taştığı için içerik "
                     "tamamen atıldı, yerine tek satırlık yer tutucu kondu")
        elif s.startswith(H.SKILL_PRUNED_MARKER_PREFIX):
            kader, adim = "özet", "ozet"
            neden = ("skill çıktısı budandı + ghost-skill markeri bırakıldı: içerik gitti "
                     "ama modele NASIL geri alınacağı (skill_view çağrısı) yazılı")
        elif s.startswith("[Duplicate tool output"):
            kader, adim = "özet", "dedup"
            neden = ("Pass 1 dedup — byte-byte AYNI sonuç daha yenide tam duruyor; bu kopya "
                     f"geri-referansa indi (kayıpsız): {s}")
        else:
            kader, adim = "özet", "ozet"
            neden = (f"Pass 2 tip-farkında özet — '{ad}' tipine göre tek satıra indi; "
                     f"hangi tool ve ne kadar veri olduğu korundu, gövde gitti")
        toollar.append(_satir(cid, ad, arg[:70], icerik, s, kader, neden, adim))

    log = [f"Pass 1 dedup            : {stats['dedup']} mesaj → geri-referans",
           f"Pass 2 informative özet : {stats['summary']} mesaj → tek satır",
           f"Pass 3 arg kısaltma     : {stats['args']} çağrı → JSON içi kırpma",
           f"Pass 4 basınç demotion  : {stats['pressure']} mesaj (korunan bölge)",
           f"prune_boundary          : #{stats['prune_boundary']}"]
    # Pass 3 tool-call argümanlarını kırpar — bu tool ÇIKTISI değil, ÇAĞRISI.
    ek = []
    for a, b in zip(trace, pruned):
        for t1, t2 in zip(a.get("tool_calls") or [], b.get("tool_calls") or []):
            a1, a2 = t1["function"]["arguments"], t2["function"]["arguments"]
            if a1 != a2:
                ek.append({"rol": f"tool-call argümanı · {t2['function']['name']}",
                           "metin": f"ÖNCE ({est(a1)} tok):\n{a1}\n\nSONRA ({est(a2)} tok):\n{a2}",
                           "tok": est(a2)})
    return _paket("hermes", once, H.total_tokens(pruned), toollar, ek, log,
                  ekstra_vurus={"arg": (stats["args"],
                                        f"{stats['args']} tool ÇAĞRISININ argümanı kırpıldı "
                                        f"(tool çıktısı değil — aşağıda ayrı kutuda)")})


# ──────────────────────────────── OPENCODE ───────────────────────────────────

def _opencode() -> dict:
    import opencode_tool_trace_poc as O

    # demo() ham çıktıyı emit_tool_output içinde kaybediyor (spill orada oluyor).
    # Aynı senaryoyu POC'un KENDİ üreteçleriyle kuruyoruz ama ham metni tutuyoruz.
    hamlar = []

    def R(tok, tag):
        raw = O._blob(tok, tag)
        p = O.emit_tool_output("read_file", raw)
        hamlar.append((p, "read_file", tag, raw))
        return p

    def T(ad, raw, etiket):
        p = O.emit_tool_output(ad, raw)
        hamlar.append((p, ad, etiket, raw))
        return p

    sk = O._blob(12000, "github-skill")
    huge = O._lines("test PASSED çıktı satırı", 5000)
    msgs = [
        {"role": "user", "parts": [O.text_part("başla, a modülünü oku")]},
        {"role": "assistant", "parts": [R(12000, "a1"), R(12000, "a2"), R(12000, "a3")]},
        {"role": "user", "parts": [O.text_part("b'yi oku ve test çalıştır")]},
        {"role": "assistant", "parts": [R(12000, "b1"), R(12000, "b2"), R(12000, "b3"),
                                        T("bash", huge, "pytest"), T("skill", sk, "github")]},
        {"role": "user", "parts": [O.text_part("c'yi oku")]},
        {"role": "assistant", "parts": [R(12000, "c1")]},
        {"role": "user", "parts": [O.text_part("d'yi oku ve login'i düzelt")]},
        {"role": "assistant", "parts": [R(12000, "d1")]},
        {"role": "assistant", "parts": [O.text_part("login() düzeltildi.")]},
    ]

    # ÖNCE: hiç sıkıştırma olmasaydı — spill de bir sıkıştırma, o yüzden ham metinden say.
    once = sum(est(r) for _, _, _, r in hamlar) + sum(
        est(p.get("text", "")) for m in msgs for p in m["parts"] if p["type"] == "text") + 3 * len(msgs)

    # Her tool part'ının konumu (mesaj, part) — prune'un log'u bu konuma göre karar yazıyor.
    yer = {id(pp): (mi, pi) for mi, m in enumerate(msgs)
           for pi, pp in enumerate(m["parts"])}

    # prune'un KENDİ karar logunu yakala. Gerekçeyi burada yeniden türetmek yerine
    # POC'un verdiği kararı okuyoruz — yoksa iki mantık zamanla ayrışır.
    # (Ölçüldü: üç FARKLI koruma gerekçesi var ve panel üçüne de aynı belirsiz
    #  metni yazıyordu → "en yeni 40.000" iddiası 61.745 token korunmuş gibi
    #  görünüyordu. Sebep: son-2-turn birimleri ve skill sayaca HİÇ girmiyor.)
    import io as _io, contextlib as _cl, re as _re
    _buf = _io.StringIO()
    with _cl.redirect_stdout(_buf):
        budanacak, adaylar, uygulanan = O.prune(msgs, verbose=True)

    part_karar, mesaj_karar, sira = {}, {}, {}
    _rx = _re.compile(r"^\s*#(\d+)(?:\.(\d+))?\s+(\S.*?)\s{2,}"
                      r"(\S+)\s+(\S+)\s+(KORU|ATLA|DUR|BUDA ADAYI|atlanır)\s*·\s*(.*)$")
    for satir in _buf.getvalue().splitlines():
        g = _rx.match(satir)
        if not g:
            continue
        mi, pi, _ad, _boyut, sayac, tip, gerekce = g.groups()
        if pi is None:
            mesaj_karar[int(mi)] = (tip, gerekce.strip(), sayac)
        else:
            part_karar[(int(mi), int(pi))] = (tip, gerekce.strip(), sayac)
            sira[(int(mi), int(pi))] = len(sira) + 1   # prune'un GÖRME sırası

    def _gerekce(p):
        """Bu part'a prune ne dedi? Part satırı yoksa mesaj satırına düş."""
        mi, pi = yer.get(id(p), (-1, -1))
        k = part_karar.get((mi, pi)) or mesaj_karar.get(mi) or ("", "", "—")
        return k + (sira.get((mi, pi)),)

    toollar = []
    for i, (p, ad, etiket, raw) in enumerate(hamlar):
        st = p["state"]
        cikti = st["output"]
        tip, gerekce, sayac, n = _gerekce(p)
        # SAYAÇ YÖNÜ: prune SONDAN BAŞA yürüyor, liste ise oluşturma sırasında.
        # O yüzden aşağı indikçe sayaç KÜÇÜLÜR. Metin bunu açıkça söylemezse
        # sayılar rastgele görünüyor (ekranda öyle göründü).
        _n = f"prune'un {n}. durağı" if n else "prune buraya hiç uğramadı"
        if st["time"]["compacted"]:
            cikti = cikti[:O.TOOL_OUTPUT_MAX_CHARS]
            kader, zincir = "budandı", ["sicak40k", "damga", "serialize"]
            neden = (f"{_n} (sondan başa) — buraya gelindiğinde sayaç {sayac}'e ulaşmıştı, "
                     f"{O.PRUNE_PROTECT:,} AŞILDI → buda adayı → state.time.compacted damgası → "
                     f"serialize'da {O.TOOL_OUTPUT_MAX_CHARS} karaktere indi (mesaj silinmedi)")
        elif st.get("spilled"):
            kader, zincir = "diske döküldü", ["spill"]
            neden = (f"Katman A canlı spill — {st['full_lines']} satır > {O.MAX_LINES}; "
                     f"context'e HİÇ girmedi, tam içerik diskte, yerine önizleme + referans")
        elif ad in O.PRUNE_PROTECTED_TOOLS:
            kader, zincir = "tam", ["skill"]
            neden = (f"korunan tool (skill) — prune aday bile almıyor. Dikkat: bu birimin "
                     f"{est(raw):,} token'ı 40K sayacına HİÇ eklenmiyor")
        elif "son-2-turn" in gerekce:
            kader, zincir = "tam", ["son2turn"]
            neden = (f"son {O.DEFAULT_TAIL_TURNS} turn koruması — bu mesaja hiç bakılmadı "
                     f"({gerekce}). 40K sayacına GİRMEDİ; ölçüt mesaj değil, turn.")
        else:
            kader, zincir = "tam", ["sicak40k"]
            neden = (f"{_n} (sondan başa) — buraya gelindiğinde sayaç {sayac}'e ulaşmıştı, "
                     f"{O.PRUNE_PROTECT:,} sınırının ALTINDA → sıcak bölge, korundu")
        toollar.append(_satir(f"#{i}", ad, etiket, raw, cikti, kader, neden,
                              zincir[-1], zincir))

    # Muafiyetlerin sayaca girmeyen hacmi — "40K" etiketinin neden yanıltıcı
    # olduğunu somut sayıyla göstermek için.
    _gormedigi = sum(est(r) for (pp, aad, _e, r) in hamlar
                     if aad in O.PRUNE_PROTECTED_TOOLS
                     or "son-2-turn" in (_gerekce(pp)[1] or ""))
    log = [f"budanabilir toplam : {budanacak:,} token ({len(adaylar)} aday)",
           f"fayda-freni        : PRUNE_MINIMUM={O.PRUNE_MINIMUM:,} → {uygulanan} birim damgalandı",
           f"korunan sıcak bölge: en yeni {O.PRUNE_PROTECT:,} token + son {O.DEFAULT_TAIL_TURNS} turn",
           f"korunan tool'lar   : {', '.join(O.PRUNE_PROTECTED_TOOLS) or '—'}"]
    return _paket("opencode", once, O.total_tokens(msgs), toollar, [], log,
                  ekstra_vurus={"sicak40k": (0,
                      f"ÖLÇÜLDÜ: sayaç en fazla 37.570'e ulaştı, ama muafiyetler yüzünden "
                      f"{_gormedigi:,} token sayaca HİÇ girmedi. Sonuç: {O.PRUNE_PROTECT:,} "
                      f"sınırıyla toplam {O.total_tokens(msgs):,} token korundu.")})


# ──────────────────────────────── OPENCLAW ───────────────────────────────────

def _openclaw() -> dict:
    import openclaw_tool_trace_poc as C

    msgs = C.demo()
    W = C.CONTEXT_WINDOW
    ham = {m["id"]: m["content"] for m in msgs if m.get("role") == "toolResult"}
    cagri = {tc["id"]: (tc["name"], tc["args"])
             for m in msgs for tc in (m.get("toolCalls") or [])}
    once = sum(C.msg_tokens(m) for m in msgs)

    # POC'un kendi 12 adımı (main() ile birebir aynı sıra)
    san, rr, sd = C.step1_sanitize(msgs)
    proj = C.step3_projection(san)
    ratio, avg, avg_ratio = C.step4_adaptive_ratio(proj, W)
    max_chunk = int(ratio * W)
    small, notlar, dusen = C.step7_oversized(proj, W)
    ozetlenecek = [m for m in small
                   if m["role"] == "toolResult"
                   or (m["role"] == "assistant" and m.get("toolCalls"))]
    mode, stages = C.step8_stage_split(ozetlenecek, max_chunk)
    ozetler = [C.step10_summarize(s) for s in stages if s]

    toollar = []
    for cid, icerik in ham.items():
        ad, arg = cagri.get(cid, ("?", ""))
        if cid in dusen:
            kader, zincir = "not'a indi", ["sanitize", "projection", "oversized"]
            neden = (f"details silindi (sır sızmasın) → projeksiyonda 8KB örneğe indi ama "
                     f"ağırlığı gerçek sayıldı → OVERSIZED: tek başına pencerenin yarısını "
                     f"({int(W*0.5):,} tok) aştığı için özetlenmeye ÇALIŞILMADI, çiftiyle "
                     f"birlikte düştü ve yerine tek satırlık not kaldı (bilinçli feragat)")
            sonra = next((n for n in notlar if cid in n), "")
        elif any(m.get("id") == cid for m in ozetlenecek):
            kader, zincir = "özete birleşti", ["sanitize", "projection", "gruplama"]
            neden = (f"details silindi → projeksiyondan geçti → çağrısıyla ATOMİK grup "
                     f"sayıldı (çift kırılmadı) → chunk'lanıp LLM özetine girdi "
                     f"({mode}, {len(stages)} aşama)")
            sonra = ""
        else:
            kader, zincir = "tam", ["sanitize"]
            neden = "head/tail korumasında kaldı; yalnız details silindi"
            sonra = icerik
        toollar.append(_satir(cid, ad, arg[:70], icerik, sonra, kader, neden,
                              zincir[-1], zincir))

    ek = [{"rol": "LLM özeti", "metin": s, "tok": est(s)} for s in ozetler]
    ek += [{"rol": "oversized notu", "metin": n, "tok": est(n)} for n in notlar]

    head = [m for m in san if m["role"] in ("system", "user")][:2]
    tail = ([san[-1]] if san and san[-1]["role"] == "assistant"
            and not san[-1].get("toolCalls") else [])
    sonra_tok = (sum(C.msg_tokens(m) for m in head + tail)
                 + sum(est(s) for s in ozetler) + sum(est(n) for n in notlar))

    log = [f"[1] SANITIZE  {sd} toolResult.details silindi, {rr} runtime çıkarıldı "
           f"→ sır özete hiç girmiyor",
           f"[4] ADAPTİF   ratio={ratio:.2f} → maxChunk={max_chunk:,} token",
           f"[7] OVERSIZED eşik={int(W*0.5):,} → {len(notlar)} not, düşen id={sorted(dusen)}",
           f"[8] STAGE     {len(ozetlenecek)} mesaj → mode={mode} ({len(stages)} aşama)",
           f"[10] ÖZET     {len(ozetler)} LLM özeti üretildi"]
    onarim = C.step11_repair([{**m, "_weight": C.msg_tokens(m)} for m in head + tail])
    return _paket("openclaw", once, sonra_tok, toollar, ek, log,
                  ekstra_vurus={"onarim": (len(onarim),
                                           "yetim kalan tool_call yok — batch düşerken çift "
                                           "birlikte düştü, onarıma gerek kalmadı"
                                           if not onarim else
                                           f"{len(onarim)} yetim çifte sentetik sonuç uyduruldu")})


# ───────────────────────────────── CODEX ─────────────────────────────────────

def _codex() -> dict:
    import codex_tool_trace_poc as X

    s = X.CodexSession()
    ham = {}

    # B1 trim'in placeholder'a çevirdikleri. FİNAL history'den okunamaz: B2
    # compaction history'yi baştan kuruyor ve placeholder'lı öğeler de gidiyor.
    # ÖLÇÜLDÜ: B1 dört birimi (c1..c4) placeholder yaptı ama final durumda
    # placeholder sayısı 0 → panel "placeholder trim vurmadı" diyordu. YANLIŞTI.
    # Her tool çağrısından sonra anlık görüntü alıp biriktiriyoruz.
    b1_vuran: set[str] = set()

    def tool(ad, tok, etiket, cid):
        raw = X._blob(tok, etiket)
        s.tool(ad, raw, cid)
        # Katman A ÜRETİM anında kesiyor. Sonradan B1/B2 gelip birimi
        # değiştirirse bu ara adım görünmez olurdu — burada yakalayıp saklıyoruz,
        # yoksa "18.536 → 0" der geçerdik ve arada bir kesme olduğu kaybolurdu.
        it = next((i for i in reversed(s.history)
                   if i["kind"] == "function_output" and i["call_id"] == cid), None)
        ham[cid] = (ad, etiket, raw, bool(it and it.get("warned")),
                    X.est(it["output"]) if it else 0)
        b1_vuran.update(o["call_id"] for o in s.history
                        if o["kind"] == "function_output" and o["placeholder"])

    s.user("auth modülünü refactor et")
    tool("shell", 18000, "pytest", "c1")
    tool("read_file", 20000, "auth.py", "c2")
    tool("read_file", 5000, "config.py", "c3")
    tool("grep", 5000, "login-eşleşme", "c4")
    s.user("login()'i sadeleştir")
    for i, t in enumerate(["login.py", "session.py", "token.py",
                           "cookie.py", "mw.py", "route.py"], start=5):
        tool("read_file", 5000, t, f"c{i}")
    s.user("testleri düzelt")
    for i, t in enumerate(["test_login.py", "test_token.py", "test_mw.py"], start=11):
        tool("read_file", 5000, t, f"c{i}")
    tool("bash", 6000, "pytest-run", "c14")
    tool("read_file", 5000, "conftest.py", "c15")
    tool("read_file", 5000, "fixtures.py", "c16")
    s.assistant("Testler düzeltildi, hepsi geçiyor.")

    kalan = {it["call_id"]: it for it in s.history if it["kind"] == "function_output"}
    toollar = []
    for cid, (ad, etiket, raw, kesildi, a_tok) in ham.items():
        it = kalan.get(cid)
        if it is None:
            kader = "pencereden düştü"
            neden = ("B2 model-turn compaction — handoff özeti üretilip YENİ pencere açıldı; "
                     "function-output'lar tamamen düşürüldü")
            zincir = ([] if not kesildi else ["truncate"])
            onek = ""
            if kesildi:
                onek += (f"önce Katman A truncate_middle kesti ({X.est(raw):,} → {a_tok:,} tok; "
                         f"BAŞ ve SON tutuldu, orta atıldı), ")
            if cid in b1_vuran:
                zincir.append("placeholder")
                onek += (f"sonra B1 trim onu placeholder'a indirdi "
                         f"({X.est(X.PLACEHOLDER)} tok'luk '{X.PLACEHOLDER}'), ")
            zincir.append("windowing")
            neden = (onek + ("SONRA " if onek else "")) + neden
            sonra = ""
        elif it["placeholder"]:
            kader, sonra = "placeholder", X.PLACEHOLDER
            zincir = (["truncate", "placeholder"] if kesildi else ["placeholder"])
            neden = ("B1 fit-to-window trim — history pencereye sığmadı, bu ESKİ çıktı "
                     "yer tutucuya indi; içerik gitti ama ÇAĞRI İSKELETİ kaldı, "
                     "tool_call ↔ tool_result zinciri kırılmadı")
        elif it.get("warned"):
            kader, sonra, zincir = "ortadan kesildi", it["output"], ["truncate"]
            neden = (f"Katman A truncate_middle — ÜRETİLİRKEN {X.TOOL_BUDGET_TOKENS:,} token "
                     f"tavanına indirildi (pencere boş olsa da keserdi); baş {X.TOOL_BUDGET_TOKENS*2:,} "
                     f"karakter + son {X.TOOL_BUDGET_TOKENS*2:,} karakter tutuldu, orta atıldı, "
                     f"başına 'Warning: truncated output' kondu")
        else:
            kader, sonra, zincir = "tam", it["output"], []
            neden = "tek-çıktı tavanının altında kaldı ve pencere taşmadı — dokunulmadı"
        toollar.append(_satir(cid, ad, etiket, raw, sonra, kader, neden,
                              zincir[-1] if zincir else "", zincir))

    ek = [{"rol": it.get("role", it["kind"]), "metin": it.get("text", ""),
           "tok": X.item_tokens(it)}
          for it in s.history if it["kind"] == "message" and it.get("role") in ("summary", "system")]
    log = list(s.log[-16:]) + [
        f"window sayısı (CompactedItem zinciri): {s.window_number}",
        f"pik gerçek context: {s.peak:,} token · final: {s.total():,} token"]
    return _paket("codex", s.raw_total, s.total(), toollar, ek, log,
                  ekstra_vurus={"multimodal": (0, "bu senaryoda görsel içerikli tool "
                                                  "sonucu yok — muafiyet devreye girmedi")})


# ────────────────────────────── CLAUDE CODE ──────────────────────────────────

def _claude_code() -> dict:
    import claude_code_tool_trace_poc as K

    s = K.ClaudeCodeSession()
    ham = {}

    def tool(ad, tok, etiket, cid):
        raw = K._blob(tok, etiket)
        s.tool(ad, raw, cid)
        # A (microcompaction) üretim anında oluyor; B sonradan gelip birimi
        # tamamen özete sürükleyebiliyor. Ara adımı burada yakala (bkz. codex).
        it = next((i for i in reversed(s.history)
                   if i["kind"] == "tool" and i["call_id"] == cid), None)
        ham[cid] = (ad, etiket, raw, bool(it and it.get("microcompacted")),
                    K.est(it["text"]) if it else 0)

    # claude_code POC'unun main()'indeki senaryonun birebir aynısı (aynı sıra,
    # aynı boyutlar, aynı call_id'ler).
    s.user("bu dokümanı çek ve özetle")
    tool("WebFetch", 23000, "docs", "toolu_01")
    tool("Read", 3000, "config.py", "toolu_02")
    s.assistant("Doküman özetlendi.")
    s.user("40 dosyayı tara, login akışını bul")
    s.subagent("40 dosyada login akışını tara", inner_tool_tokens=80_000)
    s.assistant("login akışı auth/login.py:45'te.")
    s.user("test_a/b/c'yi oku")
    for n, cid in [("test_a", "t03"), ("test_b", "t04"), ("test_c", "t05")]:
        tool("Read", 3500, n, cid)
    s.user("test_d/e/f'yi oku")
    for n, cid in [("test_d", "t06"), ("test_e", "t07"), ("test_f", "t08")]:
        tool("Read", 3500, n, cid)
    s.user("test_g'yi oku")
    tool("Read", 3500, "test_g", "t09")
    s.assistant("Testler okundu.")

    kalan = {it["call_id"]: it for it in s.history if it["kind"] == "tool"}
    toollar = []
    for cid, (ad, etiket, raw, mikro, a_tok) in ham.items():
        it = kalan.get(cid)
        if it is None:
            kader = "özete girdi"
            neden = ("B auto-compaction — bağlam eşiği aştı, eski turn'ler konuşma özetine "
                     f"indi (son {K.KEEP_RECENT_TURNS} turn korundu)")
            zincir = (["micro", "auto"] if mikro else ["auto"])
            if mikro:
                neden = (f"önce A microcompaction diske döktü ({K.est(raw):,} → {a_tok:,} tok; "
                         f"context'te önizleme + referans kalmıştı), SONRA {neden}")
            sonra = ""
        elif it.get("microcompacted"):
            kader, sonra, zincir = "diske döküldü", it["text"], ["micro"]
            neden = (f"A microcompaction — çıktı {K.MICRO_TOKENS:,} token eşiğini aştı; "
                     f"tam metin diske yazıldı, context'e ~{K.PREVIEW_TOKENS} token önizleme + "
                     f"'Full output saved to:' referansı kaldı — model içeriğin kaybolmadığını biliyor")
        else:
            kader, sonra, zincir = "tam", it["text"], []
            neden = (f"{K.MICRO_TOKENS:,} token micro eşiğinin ALTINDA ve korunan son "
                     f"{K.KEEP_RECENT_TURNS} turn içinde — context'te ham kaldı")
        toollar.append(_satir(cid, ad, etiket, raw, sonra, kader, neden,
                              zincir[-1] if zincir else "", zincir))

    ek = [{"rol": "konuşma özeti", "metin": it.get("text", ""), "tok": K.item_tokens(it)}
          for it in s.history if it["kind"] == "summary"]
    ek += [{"rol": "subagent özeti", "metin": it.get("text", ""), "tok": K.item_tokens(it)}
           for it in s.history if it["kind"] == "assistant" and "subagent özeti" in it.get("text", "")]
    log = list(s.log[-16:]) + [
        f"auto-compaction sayısı: {s.compactions} · subagent kaçışı: {s.subagent_runs}",
        f"diske dökülen: {len(s.disk)} dosya · pik: {s.peak:,}t · final: {s.total():,}t"]
    return _paket("claude_code", s.raw_total, s.total(), toollar, ek, log,
                  ekstra_vurus={"subagent": (
                      s.subagent_runs,
                      f"{s.subagent_runs} kez kullanıldı: 80.000 token'lık tarama AYRI "
                      f"pencerede koştu, ana context'e yalnız ~25 token özet döndü. "
                      f"O işin tool izi burada HİÇ oluşmadı — sıkıştırılacak bir şey yok.")})


ADAPTORLER = {"hermes": _hermes, "opencode": _opencode, "openclaw": _openclaw,
              "codex": _codex, "claude_code": _claude_code}


def kosur(ad: str) -> dict:
    """Tek bir mantığı koştur. Patlarsa diğerleri sürsün diye hata paketi döner."""
    try:
        return ADAPTORLER[ad]()
    except Exception as e:
        import traceback
        b = BILGI.get(ad, {})
        return {"ad": ad, "baslik": b.get("baslik", ad), "ekol": b.get("ekol", "—"),
                "llm": b.get("llm", False), "kaynak": b.get("kaynak", ""),
                "ozet": b.get("ozet", ""), "once": 0, "sonra": 0, "pct": 0.0,
                "toollar": [], "ek_mesajlar": [],
                "log": [f"HATA: {type(e).__name__}: {e}"] + traceback.format_exc().splitlines()[-6:]}


def hepsi() -> dict:
    return {"mantiklar": [kosur(a) for a in MANTIKLAR]}


if __name__ == "__main__":
    import json
    hedef = sys.argv[1] if len(sys.argv) > 1 else None
    veri = {"mantiklar": [kosur(hedef)]} if hedef else hepsi()
    for m in veri["mantiklar"]:
        print(f"\n┌─ {m['baslik']:<14} {m['once']:>8,} → {m['sonra']:>8,}  %{m['pct']}")
        for t in m["toollar"]:
            print(f"│  {t['ad']:<12} {str(t['arg'])[:16]:<16} {t['kader']:<16} "
                  f"{t['once_tok']:>7,} → {t['sonra_tok']:<7,}")
        for e in m["ek_mesajlar"]:
            print(f"│  ⊕ {e['rol']} ({e['tok']}t)")
        for l in m["log"][:6]:
            print(f"│  · {l}")
        print("└")
