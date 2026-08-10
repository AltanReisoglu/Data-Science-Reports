#!/usr/bin/env python3
"""
functions.py — DETERMİNİSTİK iş fonksiyonları (Airflow operatörleri gibi).

Task'ların ÇOĞU budur: LLM yok, muhakeme yok, aynı girdi → aynı çıktı.
Ajan bu fonksiyonlardan bir DAG kurar; motor onları çalıştırır.

Sözleşme:
    fn(**args, _up: dict) -> dict
      _up : {parent_task_id: parent_result}   ← Airflow XCom benzeri veri akışı
      dönüş: JSON-serileşebilir dict (bir sonraki düğüme veri olarak akar)

Her fonksiyon SAF ve tekrarlanabilirdir → retry güvenli (idempotent).
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent as _agent   # sahte repo (_FILES) ve test çıktısı buradan gelir


# ───────────────────────── yardımcı ─────────────────────────

def raw_chars(result: dict) -> int:
    """Bir fonksiyon sonucunun işlediği HAM veri miktarı (deterministik indirgeme metriği)."""
    return int((result or {}).get("_raw_chars") or 0)


def _merge_up(_up: dict, key: str, default=None):
    """Upstream sonuçlarından ilk `key` içeren değeri al (veri akışı)."""
    for _tid, res in (_up or {}).items():
        if isinstance(res, dict) and key in res:
            return res[key]
    return default


# ── ZORUNLU UPSTREAM SÖZLEŞMESİ ────────────────────────────────────────────
# Bir düğüm tüketeceği veriyi upstream'den alır. Kenar EKSİK kalırsa _merge_up
# sessizce VARSAYILANA düşer → "veri gelmedi" ile "veri geldi ama boş" ayırt
# EDİLEMEZ. Gerçek koşuda bunun sonucu şuydu: scan_patterns battı, planlayıcı
# cross_check'i ona bağlamamıştı, rapor "0 eşleşme" yazıp TEMİZ çıktı.
# Burada her düğümün olmazsa olmaz upstream anahtarlarını beyan ediyoruz;
# eksikse düğüm SESSİZCE yanlış sonuç üretmek yerine PATLAR.
_SENTINEL = object()

NEEDS = {
    "cross_check":         ("matches", "failures"),
    # render_report üç ayrı üreticiden veri basıyor: korelasyon (cross_check),
    # count (scan_patterns), total (run_test_suite). Yalnız korelasyon zorunlu
    # tutulursa rapor "0 eşleşme / 0 test" yazıp TEMİZ görünür — sayılar
    # varsayılandan gelir, kimse fark etmez.
    "render_report":       ("korelasyon", "count", "total"),
    "checksum_guard":      ("sha1",),
    "validate_schema":     ("rows",),
    "transform_normalize": ("rows",),
    "aggregate_stats":     ("rows",),
    "export_csv":          ("ilk5",),
    "run_smoke_tests":     ("sha",),
}


def _sozlesme_dogrula(fn_name: str, _up: dict):
    eksik = [k for k in NEEDS.get(fn_name, ())
             if _merge_up(_up, k, _SENTINEL) is _SENTINEL]
    if eksik:
        raise ValueError(
            f"{fn_name}() eksik upstream verisi: {eksik} — bu düğüm veriyi üreten "
            f"düğüme BAĞLI DEĞİL (gelen upstream: {list(_up or {}) or 'yok'}). "
            f"DAG kenarı eksik; varsayılanla koşmak sessizce yanlış sonuç üretirdi.")


# ───────────────────────── deterministik adımlar ─────────────────────────

def fetch_source(path: str = "auth/login.py", _up: dict | None = None) -> dict:
    """Kaynak dosyayı oku ve ölç (deterministik: satır, bayt, sha1).

    NOT: Dosya İÇERİĞİNİ sonuca KOYMAZ — düğümler arası sadece **referans** (path)
    akar. Büyük veriyi task sonucundan geçirmek yanlıştır (Airflow XCom'da da öyle):
    sonuç alanını şişirir, DB'yi büyütür, kesilirse veri akışı sessizce bozulur.
    Alt düğüm gerekirse path'ten yeniden okur.
    """
    text = _agent.tool_read_file(path)
    return {
        "_raw_chars": len(text),           # işlenen HAM veri (metrik için)
        "path": path,                      # ← referans, alt düğüme akar
        "lines": text.count("\n") + 1,
        "bytes": len(text.encode("utf-8")),
        "sha1": hashlib.sha1(text.encode("utf-8")).hexdigest()[:12],
    }


def scan_patterns(pattern: str = "mfa_token", path: str = "",
                  _up: dict | None = None) -> dict:
    """Kaynakta desen ara (deterministik regex). Kaynağı upstream'deki path'ten alır."""
    path = path or _merge_up(_up, "path", "auth/login.py")
    text = _agent.tool_read_file(path)
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        if re.search(pattern, line, re.I):
            hits.append({"line": i, "text": line.strip()[:120]})
    return {"_raw_chars": len(text), "pattern": pattern, "count": len(hits),
            "matches": hits[:20], "path": path}


def run_test_suite(suite: str = "all", _up: dict | None = None) -> dict:
    """Test paketini koştur ve çıktıyı YAPILANDIRILMIŞ veriye çevir (deterministik parse)."""
    out = _agent.tool_run_tests(suite)
    failures = []
    for line in out.splitlines():
        m = re.match(r"FAILED (\S+) - (.+)", line.strip())
        if m:
            failures.append({"test": m.group(1), "reason": m.group(2)})
    passed = out.count(" PASSED ")
    failed = out.count(" FAILED ")
    return {"_raw_chars": len(out), "suite": suite, "total": passed + failed,
            "passed": passed, "failed": failed, "failures": failures}


def cross_check(_up: dict | None = None) -> dict:
    """İki upstream'i BİRLEŞTİR: kod bulguları ile test hatalarını eşleştir (join düğümü)."""
    matches = _merge_up(_up, "matches", []) or []
    failures = _merge_up(_up, "failures", []) or []
    correlated = []
    for f in failures:
        kw = re.findall(r"[A-Za-zğüşıöçĞÜŞİÖÇ]{4,}", f.get("reason", ""))
        hit = next((m for m in matches
                    if any(k.lower() in m["text"].lower() for k in kw)), None)
        correlated.append({"test": f["test"], "reason": f["reason"],
                           "kod_kaniti": hit["text"][:80] if hit else None,
                           "satir": hit["line"] if hit else None})
    return {"korelasyon": correlated,
            "kanitli": sum(1 for c in correlated if c["kod_kaniti"]),
            "kanitsiz": sum(1 for c in correlated if not c["kod_kaniti"])}


def render_report(title: str = "Denetim Raporu", _up: dict | None = None) -> dict:
    """Upstream verilerinden markdown rapor üret (deterministik şablon)."""
    cor = _merge_up(_up, "korelasyon", []) or []
    cnt = _merge_up(_up, "count", 0)
    tot = _merge_up(_up, "total", 0)
    lines = [f"# {title}", "",
             f"- Taranan desen eşleşmesi: **{cnt}**",
             f"- Koşan test: **{tot}**",
             f"- Kanıtla eşleşen hata: **{sum(1 for c in cor if c['kod_kaniti'])}**", ""]
    for c in cor:
        lines.append(f"## {c['test']}")
        lines.append(f"- Sebep: {c['reason']}")
        lines.append(f"- Kod kanıtı: " +
                     (f"satır {c['satir']} → `{c['kod_kaniti']}`" if c["kod_kaniti"]
                      else "_bulunamadı_"))
        lines.append("")
    md = "\n".join(lines)
    return {"rapor_md": md, "uzunluk": len(md), "bolum": len(cor)}


def checksum_guard(expected_sha1: str = "", _up: dict | None = None) -> dict:
    """Kaynak değişmiş mi? (deterministik doğrulama düğümü)"""
    got = _merge_up(_up, "sha1", "")
    return {"sha1": got, "beklenen": expected_sha1 or got,
            "degisti": bool(expected_sha1) and got != expected_sha1}


# ───────────────────────── KAYIT (registry) ─────────────────────────
# Ajan YALNIZCA buradaki fonksiyonlardan DAG kurabilir — yeni fonksiyon icat edemez.

REGISTRY = {
    "fetch_source": (fetch_source,
                     "Bir kaynak dosyayı okur; satır/bayt/sha1 ve PATH (referans) döndürür.",
                     {"path": "okunacak dosya yolu (ör. auth/login.py)"}),
    "scan_patterns": (scan_patterns,
                      "Kaynakta regex deseni arar; eşleşmeleri döndürür. Kaynağı upstream'in "
                      "path'inden alır (ya da path argümanıyla verilir).",
                      {"pattern": "aranacak regex deseni (ör. mfa_token)",
                       "path": "dosya yolu (boş = upstream'den al)"}),
    "run_test_suite": (run_test_suite,
                       "Test paketini koşturur; toplam/geçen/kalan ve hata listesini döndürür.",
                       {"suite": "test paketi adı (ör. auth, all)"}),
    "cross_check": (cross_check,
                    "İKİ upstream'i birleştirir: test hatalarını kod bulgularıyla eşleştirir.",
                    {}),
    "render_report": (render_report,
                      "Upstream verilerinden markdown rapor üretir.",
                      {"title": "rapor başlığı"}),
    "checksum_guard": (checksum_guard,
                       "Kaynağın sha1'ini beklenenle karşılaştırır (değişiklik denetimi).",
                       {"expected_sha1": "beklenen sha1 (boş = sadece raporla)"}),
}


def catalog_text() -> str:
    """LLM'e sunulacak fonksiyon kataloğu (planlayıcı prompt'una gömülür)."""
    out = []
    # hangi düğümün hangi anahtarı ÜRETTİĞİ → planlayıcı kenarı doğru kursun
    uretici = {}
    for _ad, _gerekli in NEEDS.items():
        for _k in _gerekli:
            uretici.setdefault(_k, [])
    for _ad in REGISTRY:
        for _k in list(uretici):
            if _k in _URETILEN.get(_ad, ()):
                uretici[_k].append(_ad)
    for name, (_fn, desc, args) in REGISTRY.items():
        a = ", ".join(f"{k}: {v}" for k, v in args.items()) or "(argümansız)"
        satir = f"  • {name}({a})\n      {desc}"
        gerek = NEEDS.get(name)
        if gerek:
            kaynaklar = sorted({u for k in gerek for u in uretici.get(k, [])
                                if u != name})          # düğüm kendini önermesin
            satir += (f"\n      ⚠ ZORUNLU: upstream'de {list(gerek)} olmalı → "
                      f"depends_on'a {kaynaklar or 'bu veriyi üreten düğüm'} EKLE, "
                      f"yoksa düğüm hata verir.")
        out.append(satir)
    return "\n".join(out)


# Hangi fonksiyon hangi anahtarları üretiyor (katalogda kenar önerisi için).
_URETILEN = {
    "fetch_source":        ("path", "lines", "bytes", "sha1"),
    "scan_patterns":       ("matches", "count", "path"),
    "run_test_suite":      ("failures", "total", "passed", "failed"),
    "cross_check":         ("korelasyon", "kanitli", "kanitsiz"),
    "extract_records":     ("rows", "sha"),
    "validate_schema":     ("rows", "gecerli", "hatali"),
    "transform_normalize": ("rows",),
    "aggregate_stats":     ("ilk5", "grup"),
    "build_artifact":      ("sha", "surum"),
}


def resolve(fn_name: str):
    """Fonksiyonu ÖNCE aktif paketten, yoksa TÜM paketlerden bul.

    Ayrım bilinçli: PLANLAYICI aktif pakete kısıtlıdır (yanlış alandan düğüm
    seçmesin), ama YÜRÜTÜCÜ kısıtlı değildir — kayıtlı bir akış hangi pakette
    kurulduysa o fonksiyonlarla koşabilmeli.
    """
    if fn_name in REGISTRY:
        return REGISTRY[fn_name]
    for _pk, v in PACKS.items():
        if fn_name in v["fns"]:
            return v["fns"][fn_name]
    return None


def call(fn_name: str, args: dict, upstream: dict) -> dict:
    """Kayıtlı fonksiyonu çağır. Bilinmeyen fonksiyon → hata."""
    ent = resolve(fn_name)
    if ent is None:
        _all = sorted({k for v in PACKS.values() for k in v["fns"]})
        raise ValueError(f"bilinmeyen fonksiyon: {fn_name} (geçerli: {', '.join(_all)})")
    _sozlesme_dogrula(fn_name, upstream)
    fn = ent[0]
    allowed = set(ent[2])
    clean = {k: v for k, v in (args or {}).items() if k in allowed}
    return fn(**clean, _up=upstream)


if __name__ == "__main__":
    print("KAYITLI DETERMİNİSTİK FONKSİYONLAR\n")
    print(catalog_text())
    print("\n── örnek zincir (LLM'siz, saf fonksiyon akışı) ──")
    a = call("fetch_source", {"path": "auth/login.py"}, {})
    print(f"  fetch_source  → {a['lines']} satır, {a['bytes']}B, sha1={a['sha1']}")
    b = call("scan_patterns", {"pattern": "mfa_token"}, {"t1": a})
    print(f"  scan_patterns → {b['count']} eşleşme")
    c = call("run_test_suite", {"suite": "auth"}, {})
    print(f"  run_test_suite→ {c['passed']} geçti / {c['failed']} kaldı")
    d = call("cross_check", {}, {"t2": b, "t3": c})
    print(f"  cross_check   → {d['kanitli']} kanıtlı, {d['kanitsiz']} kanıtsız")
    e = call("render_report", {"title": "Auth Denetimi"}, {"t4": d, "t2": b, "t3": c})
    print(f"  render_report → {e['uzunluk']}B markdown, {e['bolum']} bölüm")
    print("\n  (aynı girdi → aynı çıktı; retry güvenli)")


# ═══════════════════════════════════════════════════════════════════════
# PAKETLER — her AKIŞ TÜRÜ kendi düğüm fonksiyonlarına sahiptir.
# Bir denetim DAG'ının fonksiyonlarıyla bir ETL DAG'ınınki aynı olamaz;
# planlayıcıya YALNIZ ilgili paketin kataloğu verilir (Airflow'da provider
# paketlerinin farklı operatörler sunması gibi).
# ═══════════════════════════════════════════════════════════════════════

# ---------- PAKET: data (ETL akışı) ----------

_SEED_ROWS = [
    {"id": i, "musteri": f"m{i%17}", "tutar": (i * 37) % 900 + 10,
     "para": "TRY" if i % 4 else "USD", "durum": "ok" if i % 11 else "iptal"}
    for i in range(1, 241)
]


def extract_records(kaynak: str = "siparisler", _up: dict | None = None) -> dict:
    """Kaynaktan ham kayıtları çek (deterministik)."""
    rows = _SEED_ROWS
    return {"_raw_chars": len(str(rows)), "kaynak": kaynak, "satir": len(rows),
            "ornek": rows[:3], "rows": rows}


def validate_schema(zorunlu: str = "id,musteri,tutar", _up: dict | None = None) -> dict:
    """Zorunlu alanları doğrula; hatalı satırları ayıkla (deterministik)."""
    rows = _merge_up(_up, "rows", []) or []
    need = [c.strip() for c in zorunlu.split(",") if c.strip()]
    gecerli, hatali = [], []
    for r in rows:
        (gecerli if all(k in r and r[k] not in (None, "") for k in need) else hatali).append(r)
    return {"gecerli": len(gecerli), "hatali": len(hatali), "zorunlu": need,
            "rows": gecerli}


def transform_normalize(kur: str = "34.5", _up: dict | None = None) -> dict:
    """Para birimini TRY'ye normalize et (deterministik dönüşüm)."""
    rows = _merge_up(_up, "rows", []) or []
    k = float(kur)
    out = [{**r, "tutar_try": round(r["tutar"] * (k if r.get("para") == "USD" else 1), 2)}
           for r in rows]
    return {"donusen": sum(1 for r in rows if r.get("para") == "USD"), "kur": k,
            "satir": len(out), "rows": out}


def aggregate_stats(grup: str = "musteri", _up: dict | None = None) -> dict:
    """Gruplayıp özet istatistik üret (deterministik agregasyon)."""
    rows = _merge_up(_up, "rows", []) or []
    acc: dict = {}
    for r in rows:
        key = str(r.get(grup))
        a = acc.setdefault(key, {"adet": 0, "toplam": 0.0})
        a["adet"] += 1
        a["toplam"] += r.get("tutar_try", r.get("tutar", 0))
    top = sorted(acc.items(), key=lambda kv: -kv[1]["toplam"])[:5]
    return {"grup": grup, "grup_sayisi": len(acc),
            "toplam_tutar": round(sum(v["toplam"] for v in acc.values()), 2),
            "ilk5": [{grup: k, **{kk: round(vv, 2) if isinstance(vv, float) else vv
                                  for kk, vv in v.items()}} for k, v in top]}


def export_csv(dosya: str = "ozet.csv", _up: dict | None = None) -> dict:
    """Özeti CSV metnine dök (deterministik çıktı)."""
    ilk5 = _merge_up(_up, "ilk5", []) or []
    if not ilk5:
        return {"dosya": dosya, "satir": 0, "csv": ""}
    cols = list(ilk5[0].keys())
    lines = [",".join(cols)] + [",".join(str(r.get(c, "")) for c in cols) for r in ilk5]
    csv = "\n".join(lines)
    return {"dosya": dosya, "satir": len(lines), "csv": csv}


# ---------- PAKET: deploy (yayınlama akışı) ----------

def build_artifact(surum: str = "1.0.0", _up: dict | None = None) -> dict:
    """Sürüm paketi üret (deterministik: sürümden türetilmiş sha)."""
    sha = hashlib.sha1(surum.encode()).hexdigest()[:12]
    return {"surum": surum, "artifact": f"app-{surum}-{sha}.tar.gz", "sha": sha,
            "boyut_kb": 1200 + sum(ord(c) for c in surum) % 800}


def run_smoke_tests(kapsam: str = "kritik", _up: dict | None = None) -> dict:
    """Duman testleri (deterministik: artifact sha'sından türetilir)."""
    sha = _merge_up(_up, "sha", "0")
    n = 20 if kapsam == "kritik" else 60
    fail = sum(1 for i in range(n) if (int(sha, 16) + i) % 23 == 0)
    return {"kapsam": kapsam, "toplam": n, "gecen": n - fail, "kalan": fail,
            "gecti": fail == 0}


def canary_release(yuzde: str = "10", _up: dict | None = None) -> dict:
    """Kısmi yayın (upstream duman testi geçmediyse yayınlamaz)."""
    ok = bool(_merge_up(_up, "gecti", False))
    art = _merge_up(_up, "artifact", "?")
    return {"artifact": art, "yuzde": int(yuzde) if ok else 0,
            "yayinlandi": ok, "sebep": "" if ok else "duman testi geçmedi"}


def health_check(esik_ms: str = "300", _up: dict | None = None) -> dict:
    """Yayın sonrası sağlık kontrolü (deterministik)."""
    y = int(_merge_up(_up, "yuzde", 0) or 0)
    gecikme = 120 + y * 4
    return {"gecikme_ms": gecikme, "esik_ms": int(esik_ms),
            "saglikli": gecikme <= int(esik_ms), "trafik_yuzde": y}


def rollback(sebep: str = "", _up: dict | None = None) -> dict:
    """Sağlıksızsa geri al (deterministik karar düğümü)."""
    sag = bool(_merge_up(_up, "saglikli", True))
    return {"geri_alindi": not sag, "sebep": sebep or ("sağlık kontrolü başarısız"
                                                       if not sag else "gerek yok")}


# ---------- PAKET KAYDI ----------

PACKS: dict[str, dict] = {  # noqa: E305
    "audit": {
        "aciklama": "Kod denetimi akışı: oku → tara → test → eşleştir → raporla",
        "fns": dict(REGISTRY),          # yukarıda tanımlı 6 fonksiyon
    },
    "data": {
        "aciklama": "ETL akışı: çek → doğrula → normalize et → topla → dışa aktar",
        "fns": {
            "extract_records": (extract_records, "Kaynaktan ham kayıtları çeker.",
                                {"kaynak": "kaynak adı (ör. siparisler)"}),
            "validate_schema": (validate_schema, "Zorunlu alanları doğrular, hatalıları ayıklar.",
                                {"zorunlu": "virgüllü zorunlu alanlar"}),
            "transform_normalize": (transform_normalize, "Para birimini TRY'ye normalize eder.",
                                    {"kur": "USD/TRY kuru"}),
            "aggregate_stats": (aggregate_stats, "Gruplayıp özet istatistik üretir.",
                                {"grup": "gruplama alanı (ör. musteri)"}),
            "export_csv": (export_csv, "Özeti CSV metnine döker.",
                           {"dosya": "çıktı dosya adı"}),
        },
    },
    "deploy": {
        "aciklama": "Yayınlama akışı: paketle → duman testi → kısmi yayın → sağlık → geri alma",
        "fns": {
            "build_artifact": (build_artifact, "Sürüm paketi üretir.",
                               {"surum": "sürüm (ör. 1.2.0)"}),
            "run_smoke_tests": (run_smoke_tests, "Duman testlerini koşturur.",
                                {"kapsam": "kritik|tam"}),
            "canary_release": (canary_release, "Kısmi (canary) yayın yapar.",
                               {"yuzde": "trafik yüzdesi"}),
            "health_check": (health_check, "Yayın sonrası sağlık kontrolü.",
                             {"esik_ms": "gecikme eşiği"}),
            "rollback": (rollback, "Sağlıksızsa geri alır.",
                         {"sebep": "geri alma sebebi"}),
        },
    },
}

ACTIVE_PACK = "audit"


def use_pack(name: str) -> str:
    """Aktif paketi değiştir → REGISTRY o paketin fonksiyonlarını gösterir."""
    global ACTIVE_PACK, REGISTRY
    if name not in PACKS:
        raise ValueError(f"bilinmeyen paket: {name} (geçerli: {', '.join(PACKS)})")
    ACTIVE_PACK = name
    REGISTRY.clear()
    REGISTRY.update(PACKS[name]["fns"])
    return name


def pack_list() -> dict:
    return {k: {"aciklama": v["aciklama"], "fns": list(v["fns"])} for k, v in PACKS.items()}
