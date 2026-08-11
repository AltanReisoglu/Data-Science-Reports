#!/usr/bin/env python3
"""
motor_ayar.py — Her motorun ayarları ve BİZİM koşumuzda etkili olup olmadıkları.

    python motor_ayar.py            # hepsi
    python motor_ayar.py celery     # tek motor

Asıl değeri "etkisiz" sütununda. Board retry otoritesini devraldığı için
motorların kendi ayarlarının bir kısmı bizim yolumuzda HİÇBİR ŞEY yapmıyor.
Bunu gizlemek yerine işaretliyoruz — ekranda düğmeyi çevirip hiçbir şeyin
değişmediğini görmek, mimarinin sonucunu anlatmanın en hızlı yolu.

    "acik"    ✓ etkili — değiştirirsen koşu değişir
    "kapali"  ⚠ ETKİSİZ — board ya da ortam onu devre dışı bırakıyor
    "sabit"   · bu POC'ta değiştirilemiyor (ortam kısıtı), ama etkisi anlatılıyor

`etkilesimli=True` olan ayarlar panelden değiştirilebilir ve koşuyu GERÇEKTEN
etkiler; `test_motor_ayar.py` bu iddiayı ölçüyor.
"""
from __future__ import annotations

MOTORLAR = ("own", "celery", "celery_canvas", "airflow", "temporal")


def _a(ad, ne, varsayilan, bizde, durum, neden="", olculen="",
       etkilesimli=False, secenekler=None, anahtar=""):
    return {"ad": ad, "ne": ne, "varsayilan": varsayilan, "bizde": bizde,
            "durum": durum, "neden": neden, "olculen": olculen,
            "etkilesimli": etkilesimli, "secenekler": secenekler or [],
            "anahtar": anahtar}


AYARLAR = {
    "own": [
        _a("budget", "işçi ajanın token bütçesi", "3000", "3000", "acik",
           etkilesimli=True, secenekler=["1000", "3000", "8000"], anahtar="budget"),
        _a("strategy", "tool-trace compaction stratejisi", "hermes", "hermes", "acik",
           etkilesimli=True, anahtar="strategy",
           secenekler=["none", "hermes", "opencode", "openclaw", "codex", "claude_code"]),
        _a("breaker eşiği", "üst üste kaç hatada `failed`", "3", "3", "sabit",
           neden="taskboard.py'de sabit; parametreye çıkarılmadı",
           olculen="kalıcı hatada dört motorda da failed ×3"),
        _a("lease süresi", "claim ne kadar geçerli", "30 sn", "30 sn", "sabit",
           neden="heartbeat aralığıyla birlikte ayarlanmalı",
           olculen="çöken worker 30 sn sonra stale sayılıyor"),
        _a("max_rounds", "dispatcher kaç tur döner", "12", "12", "sabit",
           neden="graf derinliğinden büyük olması yeterli"),
    ],
    "celery": [
        _a("task_acks_late", "iş bitince ack → worker çökerse mesaj kaybolmaz",
           "False", "True", "acik",
           neden="AÇIK — varsayılan False olsaydı çöken worker'ın işi kaybolurdu",
           olculen="çökme testi bu ayar olmadan geçmiyor"),
        _a("task_reject_on_worker_lost", "worker ölürse mesaj yeniden teslim",
           "False", "True", "acik"),
        _a("worker_prefetch_multiplier", "worker önden kaç mesaj çeker",
           "4", "1", "acik", neden="1 → adil dağıtım; yüksekse bir worker kuyruğu yutar"),
        _a("max_retries", "Celery'nin KENDİ retry sayısı", "3", "3 (yazılı)", "kapali",
           neden="`self.retry()` KALDIRILDI — retry otoritesi board'da. Dekoratörde "
                 "yazıyor ama hiç okunmuyor.",
           olculen="hata ⑪: iki yeniden-dağıtım yolu düğümü ×3 koşturuyordu"),
        _a("result_backend", "sonucu saklar", "None", "None", "kapali",
           neden="sonuç board'a yazılıyor"),
        _a("visibility_timeout", "task bundan uzun sürerse mesaj yeniden teslim",
           "3600 sn (Redis)", "—", "kapali",
           neden="filesystem broker'da bu kavram yok",
           olculen="Redis'te aşılırsa aynı iş İKİ worker'da paralel koşar"),
        _a("pool", "prefork / solo / gevent / threads", "prefork", "solo", "sabit",
           neden="POC tek worker; prefork her worker'a bellek kopyası verir",
           olculen="LLM kütüphaneleriyle 4 worker = 4× model belleği"),
    ],
    "celery_canvas": [
        _a("result_backend", "chord'un sonuç deposu", "None", "file://", "acik",
           neden="ŞART — backend'siz chord kurulamaz"),
        _a("max_retries", "halka bazında yeniden deneme", "3", "2", "acik",
           neden="canvas'ta HER HALKA ayrı task — retry de ayrı",
           olculen="'retry baştan koşar' iddiası canvas için GEÇERSİZ"),
        _a("default_retry_delay", "denemeler arası bekleme", "180 sn", "1 sn", "acik"),
        _a("task_acks_late", "iş bitince ack", "False", "True", "acik"),
        _a("link_error", "zincir batarsa callback", "None", "None", "kapali",
           neden="kurulmadı — batan zincir SESSİZCE ölüyor",
           olculen="kalıcı hatada 60 sn beklendi, kayıt yok"),
        _a("self.replace()", "koşullu dallanma", "—", "kullanılmıyor", "kapali",
           neden="dallanmanın tek yolu; mantık task'lara dağılıyor"),
    ],
    "airflow": [
        _a("retries", "düğüm bazında kaç kez denensin", "0", "2", "acik",
           etkilesimli=True, secenekler=["0", "1", "2", "3"], anahtar="af_retries",
           olculen="kalıcı hatada deneme=3 (ilk + 2 retry)"),
        _a("retry_delay", "denemeler arası bekleme", "30 sn", "1 sn (panelde)", "acik",
           etkilesimli=True, secenekler=["1", "5", "30"], anahtar="af_retry_delay",
           neden="varsayılan 30 sn paneli kullanılamaz kılıyordu",
           olculen="30 sn × 3 deneme = 62 sn (canlı ölçüldü: 09:47:19→09:47:50→09:48:21)"),
        _a("schedule", "cron ifadesi", "None", "0 8 * * *", "acik",
           etkilesimli=True, anahtar="af_schedule",
           secenekler=["None", "0 8 * * *", "*/15 * * * *", "@daily"],
           neden="üretilen DAG dosyasına yazılıyor",
           olculen="POC tek koşu tetikliyor; cron'un kendisi çalışmıyor"),
        _a("catchup", "kaçırılan koşuları geriye dönük üret", "True", "False", "sabit",
           neden="POC tek koşu; True olsaydı start_date'ten beri her aralık koşardı",
           olculen="Airflow'un en ayırt edici özelliği — burada kapalı"),
        _a("max_active_runs", "aynı anda kaç DagRun", "16", "1", "sabit",
           neden="SequentialExecutor zaten seri"),
        _a("trigger_rule", "birleşim düğümünün kuralı", "all_success", "all_success",
           "sabit", neden="koşullu dal bu grafta yok",
           olculen="unutulursa birleşim düğümü de skip olur — en sık hata"),
        _a("executor", "Sequential / Local / Celery / Kubernetes", "Sequential",
           "Sequential", "kapali",
           neden="eşzamanlılık YOK; iki `dags test` 'database is locked' veriyor",
           olculen="koşumlar kilitle serileştirildi (_KILIT)"),
    ],
    "temporal": [
        _a("maximum_attempts", "activity kaç kez denensin", "sınırsız", "3", "acik",
           neden="DEVREDE — ama deneme SAYACI board'dan geliyor",
           olculen="hata ②: sayaç Temporal'ınkinden alınınca geçici hata kalıcı sanıldı"),
        _a("initial_interval", "retry'lar arası ilk bekleme", "1 sn", "200 ms", "acik",
           olculen="own'dan sonra en hızlı motor olmasının bir sebebi"),
        _a("start_to_close_timeout", "activity için üst süre sınırı", "yok (zorunlu)",
           "240 sn", "acik",
           neden="aşılırsa cluster activity'yi ölmüş sayar ve yeniden dener"),
        _a("sandboxed", "workflow determinizm kum havuzu", "True", "False", "acik",
           neden="False — board nesnesine erişebilmek için. Üretimde True olmalı.",
           olculen="determinizm kontrolü devre dışı; POC kısıtı"),
        _a("task_queue", "worker'ın dinlediği kuyruk", "—", "board-tq", "sabit"),
        _a("RetryPolicy vs board breaker", "iki dayanıklılık katmanı", "—",
           "board kazanır", "kapali",
           neden="Temporal aynı activity'yi AYNI payload ile çağırıyor; board.fail() "
                 "claim'i temizlediği için 'bayat claim onarımı' gerekti",
           olculen="hata ②: iki kez başaran düğüm 3. turda failed işaretleniyordu"),
    ],
}

DURUM_ETIKET = {
    "acik":   {"im": "✓", "ad": "etkili",
               "aciklama": "değiştirirsen koşu değişir"},
    "kapali": {"im": "⚠", "ad": "ETKİSİZ",
               "aciklama": "board ya da ortam devre dışı bırakıyor"},
    "sabit":  {"im": "·", "ad": "sabit",
               "aciklama": "bu POC'ta değiştirilemiyor, etkisi anlatılıyor"},
}


def ayarlar(motor: str) -> dict:
    if motor not in AYARLAR:
        raise KeyError(f"bilinmeyen motor: {motor}")
    ls = AYARLAR[motor]
    return {
        "motor": motor,
        "ayarlar": ls,
        "sayim": {d: sum(1 for a in ls if a["durum"] == d)
                  for d in ("acik", "kapali", "sabit")},
        "etkilesimli": [a for a in ls if a["etkilesimli"]],
    }


def hepsi() -> dict:
    return {"durum_etiket": DURUM_ETIKET,
            "motorlar": [ayarlar(m) for m in MOTORLAR]}


def uygula(motor: str, secim: dict) -> dict:
    """Panelden gelen ayarları koşum parametrelerine çevir.

    Yalnız `etkilesimli=True` olanlar geçer; gerisi sessizce ATILMAZ — reddedilir
    ve gerekçesi döner. Sessiz yok sayma, panelde 'düğmeyi çevirdim bir şey
    olmadı' belirsizliği yaratırdı.
    """
    ls = {a["anahtar"]: a for a in AYARLAR.get(motor, []) if a["etkilesimli"]}
    kabul, red = {}, []
    for k, v in (secim or {}).items():
        a = ls.get(k)
        if not a:
            red.append({"anahtar": k, "neden": "bu motorda etkileşimli ayar değil"})
            continue
        if a["secenekler"] and str(v) not in a["secenekler"]:
            red.append({"anahtar": k,
                        "neden": f"geçersiz değer {v!r} (geçerli: {a['secenekler']})"})
            continue
        kabul[k] = v
    return {"kabul": kabul, "red": red}


if __name__ == "__main__":
    import sys
    hedef = sys.argv[1] if len(sys.argv) > 1 else None
    v = {"motorlar": [ayarlar(hedef)]} if hedef else hepsi()
    for m in v["motorlar"]:
        s = m["sayim"]
        print(f"\n{'═'*78}\n {m['motor'].upper()}   "
              f"{s['acik']} etkili · {s['kapali']} ETKİSİZ · {s['sabit']} sabit\n{'═'*78}")
        for a in m["ayarlar"]:
            im = DURUM_ETIKET[a["durum"]]["im"]
            el = " [panelden değiştirilebilir]" if a["etkilesimli"] else ""
            print(f"  {im} {a['ad']:<28} varsayılan={a['varsayilan']:<14} "
                  f"bizde={a['bizde']}{el}")
            print(f"      {a['ne']}")
            if a["neden"]:
                print(f"      ↳ {a['neden']}")
            if a["olculen"]:
                print(f"      ölçüldü: {a['olculen']}")
