#!/usr/bin/env python3
"""
motor_kunye.py — Dört motorun KANITLI künyesi.

Bu modül veri döndürür, HTTP bilmez, sunucu bilmez. Tek başına koşturulabilir:

    python motor_kunye.py            # hepsi
    python motor_kunye.py celery     # tek motor

Her motor için altı şey:
  1. künye   — tek cümle, analoji, iç mimari şeması
  2. katman  — temel soyutlama (üçü aynı kategoride DEĞİL)
  3. modüller— bileşen listesi + BİZİM koşumuzda devrede mi
  4. güçlü   — her madde bir kanıta bağlı
  5. zayıf   — her madde bir kanıta bağlı
  6. çapa    — "fetch kaç kez koştu" ölçümü

KANIT DİSİPLİNİ: her iddianın bir `kanit` alanı var ve türü belli.
Uydurma iddia girmesin diye tür zorunlu:

    olcum  — bu POC'ta koşudan çıkan sayı
    kod    — bu repodaki dosya:satır
    repo   — harnesses/ altındaki GERÇEK kaynak satırı
    rapor  — report/ altındaki ölçüm kaydı
    hata   — bu çalışmada bulunup düzeltilen numaralı hata

Modül durumları üç değerli — ikili olsaydı yanlış olurdu:
    "acik"    ✓ devrede, motorun kendi işini yapıyor
    "kapali"  ⊘ devre dışı; board o işi devraldı
    "esgudum" ⚠ devrede AMA board'la çakıştı, uzlaştırmak gerekti
"""
from __future__ import annotations

MOTORLAR = ("own", "celery", "celery_canvas", "airflow", "temporal")

# ── Çerçeve: en yaygın hata bunları "3 rakip" sanmak ────────────────────────
# Üçü farklı KATMANDA. Hatta iç içe geçerler: Airflow'un CeleryExecutor'ı
# işleri Celery'ye dağıtır — "Airflow vs Celery" çoğu zaman yanlış sorudur.
KATMAN_NOTU = (
    "Bu dördü birbirinin alternatifi DEĞİL — farklı katmanlardalar. Celery "
    "'işi başka sürece taşı', Airflow 'zamanı ve sırayı yönet', Temporal 'iş "
    "çökmelere karşı bağışık olsun', board 'defteri ben tutayım' der. Airflow'un "
    "CeleryExecutor'ı işleri Celery'ye dağıtır; biri diğerinin altında çalışabilir."
)

# ── Terminoloji tuzağı: "task" her araçta farklı katmanı gösterir ───────────
TERIM_TUZAGI = {
    "aciklama": (
        "'task' kelimesi her araçta farklı katmanı işaret eder. Karıştırılınca "
        "'Airflow task retry eder' ile 'Temporal task'ı kaldığı yerden sürdürür' "
        "aynı şey sanılır — biri BİR ADIMI baştan koşar, diğeri BÜTÜN İŞİ kurtarır."
    ),
    "satirlar": [
        {"katman": "A — iş / job",
         "tarif": "yaşam döngüsü olan iş birimi ('siparişi işle')",
         "karsilik": {"airflow": "DagRun", "temporal": "Workflow",
                      "celery": "karşılığı YOK", "own": "pipeline / graf"}},
        {"katman": "B — adım",
         "tarif": "tek fonksiyon veya tool çağrısı",
         "karsilik": {"airflow": "task (TaskInstance)", "temporal": "activity",
                      "celery": "task", "own": "board düğümü"}},
    ],
}


def _k(tur: str, ref: str, not_: str = "") -> dict:
    """Kanıt kaydı. tur: olcum | kod | repo | rapor | hata"""
    assert tur in ("olcum", "kod", "repo", "rapor", "hata"), tur
    return {"tur": tur, "ref": ref, "not": not_}


# ── HAFIZA MERDİVENİ — sıralama tesadüf değil ──────────────────────────────
# Üçü de "dışarıda bir hafıza" ister; fark BOYUTUNDA. Ne kadar çok hatırlıyorsa
# o kadar çok garanti veriyor, o kadar ağır. "Celery hafif, Temporal ağır" bir
# kusur kıyaslaması değil — doğrudan verdikleri sözün bedeli.
HAFIZA_MERDIVENI = {
    "aciklama": (
        "Hiçbiri sihir yapmıyor: hepsi işin defterini SÜRECİN DIŞINDA tutuyor. "
        "Kurtarıcı, kurtardığı şeyle aynı süreçte yaşayamaz — worker'la birlikte "
        "ölen bir hafıza kimseyi kurtaramaz. Fark, defterin NE KADAR şey tuttuğu."
    ),
    "basamaklar": [
        {"motor": "celery", "disarida": "Broker (Redis/RabbitMQ)",
         "saklar": "yalnız BEKLEYEN mesajlar — iş başladıktan sonrası unutulur",
         "agirlik": "hafif", "bedeli": "fetch ×2 · defteri sen tutarsın"},
        {"motor": "airflow", "disarida": "Metadata DB + Scheduler + Webserver",
         "saklar": "DagRun / TaskInstance durumları — ADIM seviyesinde",
         "agirlik": "orta-ağır", "bedeli": "fetch ×1 ama yalnız adımlar ARASINDA"},
        {"motor": "temporal", "disarida": "Cluster + Persistence (+ Elasticsearch)",
         "saklar": "her adımın TAM olay geçmişi ve sonuçları",
         "agirlik": "ağır", "bedeli": "fetch ×1 · cluster işletme maliyeti"},
        {"motor": "celery_canvas", "disarida": "Broker mesajı (zincir mesajın İÇİNDE)",
         "saklar": "yalnız KALAN zincir — biten adımların kaydı yok",
         "agirlik": "çok hafif", "bedeli": "'nerede kaldı' sorusu CEVAPSIZ"},
        {"motor": "own", "disarida": "SQLite dosyası",
         "saklar": "task tablosu + checkpoint — düğüm İÇİ ara sonuç dahil",
         "agirlik": "çok hafif", "bedeli": "fetch ×1 ama kodu SEN yazarsın (12 hata)"},
    ],
    "ders": "Sıralama tesadüf değil: hatırlama miktarı = garanti miktarı = ağırlık.",
}

# ── İLİŞKİLER — "3 rakip" yanılgısını kıran asıl bölüm ──────────────────────
# Bunlar rakip değil; biri diğerinin ALTINDA çalışabiliyor.
ILISKILER = [
    {
        "cift": ("airflow", "celery"),
        "baslik": "Airflow, Celery'yi KENDİ İŞÇİ HAVUZU olarak kullanır",
        "nasil": "Airflow'un Executor seçimi var: LocalExecutor · CeleryExecutor · "
                 "KubernetesExecutor. CeleryExecutor seçilirse Airflow scheduler "
                 "hazır olan task'ı `.delay()` ile Celery broker'ına atar.",
        "sema": """  ┌─────────────────────────────────┐
  │ AIRFLOW SCHEDULER               │  ← ŞEF: ne zaman, hangi sıra
  │ "08:00 oldu, fetch hazır"       │
  └───────────────┬─────────────────┘
                  │ .delay()
                  ▼
         ┌──────────────────┐
         │ CELERY BROKER    │           ← FİŞ PANOSU
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │ Celery worker'lar│           ← AŞÇILAR
         └──────────────────┘""",
        "kilit": "Airflow'un 'fetch ×1' özelliği CELERY'NİN DEĞİL, üstündeki "
                 "DEFTERİN. fetch ayrı bir DAG düğümü ve metadata DB'de 'success' "
                 "yazıyor; process patlayınca Airflow YALNIZ process'i Celery'ye "
                 "yeniden gönderiyor. Celery'nin bundan haberi bile yok.",
        "yanlis_soru": "'Airflow mu Celery mi?' — çoğu zaman yanlış soru. "
                       "Biri zamanı ve sırayı, diğeri iş gücünü yönetiyor.",
        "kanit": _k("repo", "apache/airflow — CeleryExecutor"),
    },
    {
        "cift": ("temporal", "celery"),
        "baslik": "Temporal ≈ Celery + kalıcı hafıza + workflow soyutlaması",
        "nasil": "Mimarileri şaşırtıcı derecede benziyor: ikisinde de kod SENİN "
                 "sunucunda koşar, ortada bir kuyruk vardır, worker havuzu çeker.",
        "sema": """  CELERY                     TEMPORAL
  ──────                     ────────
                             ┌──────────────────────────┐
      (bu katman YOK)        │ HISTORY SERVİSİ          │  ← DEFTER
                             │ "iş #4711 nerede kaldı"  │
                             │  fetch ✅ (sonuç: "…")   │
                             │  process ⏳              │
                             └────────────┬─────────────┘
  ┌──────────────┐           ┌────────────▼─────────────┐
  │ BROKER       │           │ MATCHING (task queue)    │  ← aynı iş
  └──────┬───────┘           └────────────┬─────────────┘
         ▼                                ▼
  ┌──────────────┐           ┌──────────────────────────┐
  │ Worker'lar   │           │ Worker'lar               │  ← aynı iş
  └──────────────┘           └──────────────────────────┘""",
        "esleme": [
            ("kuyruk", "Redis/RabbitMQ", "Matching servisi (task queue)"),
            ("iş yapan süreç", "Celery worker", "SDK Worker"),
            ("iş birimi", "task (fonksiyon)", "Activity (fonksiyon)"),
            ("nasıl alınır", "broker'dan çek", "task queue'yu long-poll"),
            ("retry", "self.retry()", "RetryPolicy"),
        ],
        "kilit": "Tek fark bir KATMAN: Celery'de her mesaj bağımsız ve amnezi "
                 "hastası; Temporal'da her adım kalıcı deftere yazılır. Çökme "
                 "sonrası 'fetch zaten bitmiş, sonucu şuydu' diyebilmesinin sebebi bu.",
        "yanlis_soru": "'Temporal Celery'nin yerini alır mı?' — Temporal zaten "
                       "Celery'nin yaptığını yapıyor, ÜSTÜNE defter koyuyor.",
        "kanit": _k("rapor", "task-management-analizi.md"),
    },
    {
        "cift": ("own", "celery"),
        "baslik": "Board, tam olarak Celery'nin BIRAKTIĞI boşluğu dolduruyor",
        "nasil": "Celery'nin yapmadığı dört şeyi board yapıyor; Celery yalnız "
                 "dağıtıcı olarak altında kalıyor.",
        "esleme": [
            ("'iş nerede kaldı' defteri", "YOK", "SQLite task tablosu"),
            ("iki worker aynı işi alabilir", "at-least-once", "CAS-claim → at-most-once"),
            ("worker çökünce kim fark eder", "broker timeout", "lease + heartbeat + recover_stale"),
            ("retry baştan koşuyor", "evet — fetch ×2", "checkpoint → fetch ×1"),
            ("sonsuz retry", "max_retries sayar", "circuit breaker"),
            ("batan işin ardılı", "belirsiz", "cancel_downstream → cancelled"),
        ],
        "kilit": "Board Celery'nin YERİNE değil, ÜSTÜNE geçiyor. Bu yüzden bizim "
                 "koşumuzda Celery'nin 10 modülünden 6'sı devre dışı — Celery'yi "
                 "kullanıyoruz ama Celery'yi Celery yapan şeylerin çoğunu kullanmıyoruz.",
        "yanlis_soru": "'Celery mi board mu?' — ikisi birlikte; soru board'un "
                       "ALTINA hangi motoru koyacağın.",
        "kanit": _k("olcum", "celery: 4 devrede / 6 devre dışı"),
    },
    {
        "cift": ("celery", "celery_canvas"),
        "baslik": "Aynı kütüphane, iki bambaşka kullanım",
        "nasil": "Celery'yi iki şekilde kullanabilirsin: (a) canvas ile workflow "
                 "kurarsın, (b) yalnız dağıtıcı olarak kullanıp defteri kendin tutarsın.",
        "esleme": [
            ("akışı kim biliyor", "canvas — mesaja serileştirilmiş", "board — SQLite tablosu"),
            ("düğüm durumu", "YOK", "blocked/ready/running/done/failed/cancelled"),
            ("iptal kaydı", "YOK — sessizce ölür", "cancelled zinciri"),
            ("veri akışı", "pozisyonel (ilk argüman)", "{düğüm_id: sonuç}"),
            ("result backend", "ŞART (chord için)", "gereksiz — sonuç board'da"),
            ("kurulum", "sıfır ek iş", "board yazmak gerekti (12 hata)"),
        ],
        "kilit": "Bu POC'ta İKİSİ DE koşuyor: 'Celery' sekmesi board'lu, 'Celery Canvas' "
                 "sekmesi board'suz. Aynı graf, aynı kütüphane — fark yalnız defterin "
                 "kimde olduğu. Ölçüldü: hatasız ikisi de bitiriyor; KALICI hatada "
                 "board 3/6+1 failed+2 cancelled derken canvas 60 sn takılıp susuyor.",
        "yanlis_soru": "'Celery workflow yapabiliyor mu?' — evet, canvas ile. Doğru "
                       "soru: 'takıldığında bana ne söyleyecek?'",
        "bizde": True, "motorlar": ["celery", "celery_canvas"],
        "kanit": _k("olcum", "canvas ✓9,03 sn / ✗66,04 sn (kalıcı hata)"),
    },
    {
        "cift": ("airflow", "temporal"),
        "baslik": "İkisi de defter tutar — fark defterin ÇÖZÜNÜRLÜĞÜ",
        "nasil": "Airflow adımlar ARASINDA kurtarır (düğüm bazında), Temporal "
                 "adımın kendisini de sürdürür (event history + replay).",
        "kilit": "Airflow 'fetch bitti' der ve fetch'i tekrar koşturmaz — ama bir "
                 "adım 10 alt-işten 7'sini yapıp çökerse o adımı BAŞTAN koşar. "
                 "Temporal'da da activity içi aynı sınır var; farkı, işin ŞEKLİNİN "
                 "önceden bilinmesi gerekmemesi.",
        "yanlis_soru": "'Airflow zaten retry ediyor, Temporal'a ne gerek var?' — "
                       "iki sebep: (1) Airflow'da graf ÖNCEDEN belli olmak zorunda, "
                       "ajan akışı runtime'da şekilleniyor; (2) Airflow adımlar "
                       "arasında kurtarır, adımın içini kurtaramaz.",
        "kanit": _k("rapor", "motor-secimi-workflow-desenleri-ve-karar.md §2.3"),
    },
]

# ── KAVRAMLAR — sorulduğunda ağızdan çıkacak tanımlar ──────────────────────
KAVRAMLAR = {
    "cluster": {
        "soru": "Cluster ne demek, Temporal neden cluster istiyor?",
        "tanim": "Tek bir servis gibi davranan, birden çok makinede koşan sunucu "
                 "grubu. Bir teknoloji değil, bir İŞLETME BİÇİMİ: 'bu servisi "
                 "ciddiye alıp çoklu-node ayakta tutman gerekiyor' demek.",
        "neden": "Temporal'ın sattığı garanti şu: 'senin süreçlerin ölse bile iş "
                 "kaybolmaz.' Bu garantiyi verebilmek için işin kaydını SENİN "
                 "süreçlerinin dışında, onlardan bağımsız ayakta duran bir yerde "
                 "tutmak zorunda. Kurtarıcı, kurtardığı şeyle aynı süreçte yaşayamaz.",
        "yanlis_anlama": "Temporal Cluster senin kodunu ÇALIŞTIRMAZ. Cluster = "
                         "hakem + defter tutucu. Senin worker'ın = oyuncu. Cluster "
                         "iş kodunu hiç görmez; yalnız 'hangi adım başladı, bitti, "
                         "sonucu neydi' kaydını tutar ve sıradakini kime vereceğini ayarlar.",
        "bilesenler": [
            ("Frontend", "API kapısı — auth, rate limit, yönlendirme"),
            ("History", "asıl beyin: event history'yi yönetir, timer'ları takip eder; "
                        "shard'lı → yatay ölçeklenir"),
            ("Matching", "task queue'ları yönetir; worker long-poll atar, hangi task "
                         "kime gidecek Matching karar verir"),
            ("Worker (iç)", "Temporal'ın KENDİ iç işleri: arşivleme, replikasyon, "
                            "retention temizliği. Senin worker'ın DEĞİL"),
            ("Persistence", "Cassandra / PostgreSQL / MySQL — event history burada yaşar"),
            ("Visibility", "Elasticsearch veya SQL — 'şu müşterinin başarısız "
                           "workflow'larını listele' gibi aramalar için"),
        ],
        "surekli_yaptigi": [
            "Timer takibi — `await sleep(30 gün)` derken 30 gün boyunca KİM sayıyor? "
            "Cluster. Worker'ların bu sürede hiç ayakta olmayabilir.",
            "Timeout tespiti — worker activity'yi alıp cevap vermezse (çöktü), "
            "`start_to_close_timeout` dolunca kim fark edecek? Cluster.",
            "Retry zamanlaması — '3 saniye sonra tekrar dene' kararını ve saatini cluster tutar.",
            "Task dağıtımı — aynı task iki worker'a gitmesin diye hakemlik.",
            "Schedules (cron) — zamanı gelince tetikleme.",
        ],
        "kacis_yollari": [
            ("Temporal Cloud", "cluster'ı Temporal işletir, sen yalnız worker koşturursun. "
                               "Operasyon yükü ~sıfır, aylık ücret var. Çoğu ekibin gerçek cevabı."),
            ("temporal server start-dev", "tek binary + SQLite. Geliştirme/test için "
                                          "mükemmel, production için değil."),
            ("docker-compose single-node", "küçük ölçekli self-host. Çalışır ama HA yok."),
        ],
        "bizde": "Bu POC `WorkflowEnvironment.start_time_skipping()` kullanıyor — arka "
                 "planda EPHEMERAL bir dev server indirip başlatıyor (~83 MB binary). "
                 "Yani POC'ta bile GERÇEK bir Temporal server koştu, sadece geçici olanı.",
        "ozet": "Asıl karar 'Temporal mı değil mi' değil — 'bu defteri kim tutacak'.",
        "kanit": _k("kod", "temporal_defs.py", "test-server /tmp'de"),
    },
}


# ═══════════════════════════════════════════════════════════════════════════
KUNYE = {

    # ───────────────────────────────── CELERY ──────────────────────────────
    "celery": {
        "baslik": "Celery",
        "katman": "Dağıtık task kuyruğu",
        "soyutlama": "fonksiyon çağrısı",
        "tek_cumle": "Bir fonksiyonu şimdi burada değil, başka bir süreçte/makinede "
                     "sonra çalıştırmanı sağlar.",
        "ne_yapar": [
            "Kuyruğa atar — `send_email.delay(user)` dediğinde fonksiyon çalışmaz; "
            "mesaj olarak broker'a yazılır, çağıran anında geri döner.",
            "Worker'a dağıtır — ayrı süreçlerdeki worker'lar kuyruktan çeker. "
            "Worker sayısını artırırsan kapasite artar.",
            "Hata olursa yeniden kuyruğa koyar — `self.retry()` ile.",
        ],
        "yasam_dongusu": {
            "baslik": "`.delay()` dediğinde tam olarak ne oluyor",
            "adimlar": [
                ("1 · Fonksiyon ÇALIŞMAZ",
                 "Python gövdeye hiç girmez. Bir mesaj hazırlanır: "
                 "{task: 'myapp.send_email', id: 'f81d…', args: [...], retries: 0}. "
                 "KRİTİK: fonksiyonun KENDİSİ gönderilmez, yalnız ADI. Worker'ın o "
                 "kodun sahibi olması gerekir — `-A myapp` bunun içindir."),
                ("2 · Broker'a yazılır",
                 "Mesaj serialize edilir (varsayılan JSON) ve broker'a publish edilir. "
                 "RabbitMQ'da exchange→routing key→kuyruk; Redis'te listeye LPUSH."),
                ("3 · Çağıran HEMEN döner",
                 "`.delay()` bir AsyncResult döndürür — yalnız bir task id taşıyıcısı. "
                 "İçinde sonuç yok, iş henüz başlamadı bile."),
                ("4 · Worker mesajı çeker",
                 "Ayrı süreçteki worker broker'ı dinliyordur; hatta birkaç mesajı "
                 "ÖNDEN çeker (prefetch_multiplier × concurrency kadar)."),
                ("5 · Fonksiyon çalışır",
                 "Worker adı kendi kayıt defterinde bulur ve args ile çağırır."),
                ("6 · Sonuç (opsiyonel) + ACK",
                 "result_backend varsa sonuç yazılır. Sonra mesaj ack'lenir → kuyruktan "
                 "SİLİNİR. Ack'in ZAMANI Celery'nin en önemli ayarı (bkz. incelikler)."),
            ],
            "ozet": "Celery'nin yaptığı şeyin tamamı: bir fonksiyon çağrısını JSON'a "
                    "çevirip kuyruktan geçirmek.",
        },
        "retry_mekanigi": {
            "baslik": "`self.retry()` gerçekte ne yapar — neden BAŞTAN koşuyor",
            "sanilan": "Fonksiyonu durdurup biraz sonra KALDIĞI YERDEN devam ettirir.",
            "gercek": "Broker'a YENİ bir mesaj yayınlar. Sıfırdan yeni bir çağrıdır — "
                      "yalnız sayacı artmış bir kopyası.",
            "adimlar": [
                "`self.retry()` bir Retry istisnası fırlatır",
                "Worker AYNI task_id ile YENİ mesaj publish eder — tek fark `retries: 1`",
                "Eski mesaj ack'lenir (silinir)",
                "countdown/eta kadar sonra worker yeni mesajı çeker",
                "Fonksiyon 1. SATIRINDAN başlar",
            ],
            "aha": "Retry sayacı MESAJIN İÇİNDE taşınıyor (retries: 0→1→2). Fonksiyonun "
                   "içinde nerede kaldığını bilen kimse yok — bu yüzden fetch, "
                   "process'in hatasından etkilenmese bile tekrar koşar.",
            "mutfak": "Sos yandı → yeni fiş takıldı → aşçı etten itibaren değil, "
                      "SIFIRDAN başlıyor. Halbuki et mükemmel pişmişti; çöpe gitti.",
            "kanit": _k("olcum", "attempt0:fetch → attempt0:process-HATA → "
                                 "attempt1:fetch → attempt1:process-OK"),
        },
        "analoji": {
            "baslik": "Restoran mutfağındaki sipariş fişi",
            "metin": "Garson fişi çiviye takar (kuyruk) ve gider; boştaki aşçı fişi "
                     "alır, yemeği yapar. Fiş yanarsa yenisi takılır — ama aşçı "
                     "yemeğe SIFIRDAN başlar.",
        },
        "mimari": """  Producer (web app)            Broker                Worker havuzu
  ──────────────────           ──────                ─────────────
  run_order.delay("4711") ─msg─▶ Redis/RabbitMQ ─pull─▶ worker-1 (prefork/solo)
                                 (kuyruk)              worker-2
                                    │                  worker-3
                                    ▼                       │
                            Result backend ◀────sonuç───────┘
                            (opsiyonel)

  Celery bir SERVİS değil, bir KÜTÜPHANE. Asıl altyapı broker.""",
        "moduller": [
            {"ad": "Broker", "ne": "kuyruk (Redis/RabbitMQ/SQS) — asıl altyapı",
             "durum": "acik", "neden": "filesystem broker kullanıyoruz",
             "kanit": _k("kod", "celery_worker.py:29-37",
                         "data_folder_in == data_folder_out ŞART, yoksa sessizce çalışmaz")},
            {"ad": "Worker", "ne": "kuyruktan çeker, fonksiyonu koşturur",
             "durum": "acik", "neden": "tek worker, ayrı süreç",
             "kanit": _k("olcum", "açılış ~6 sn / koşu")},
            {"ad": "Canvas (chain/group/chord)", "ne": "çok adımlı iş kurmanın tek yolu",
             "durum": "kapali",
             "neden": "board DAG kapısını devraldı → `run_task.delay(id)` teker teker",
             "kanit": _k("kod", "orchestrator.py:695", "canvas hiç kurulmuyor")},
            {"ad": "acks_late", "ne": "iş bitince ack → worker çökerse mesaj kaybolmaz",
             "durum": "acik", "neden": "açık; varsayılan False olsaydı iş kaybolurdu",
             "kanit": _k("kod", "celery_worker.py:39-40")},
            {"ad": "max_retries / self.retry()", "ne": "Celery'nin kendi retry'ı",
             "durum": "kapali",
             "neden": "retry otoritesi board'a verildi; `self.retry()` KALDIRILDI",
             "kanit": _k("hata", "⑪", "iki yeniden-dağıtım yolu → düğüm ×3 koştu")},
            {"ad": "Result backend", "ne": "sonucu saklar (opsiyonel)",
             "durum": "kapali", "neden": "sonuç board'a yazılıyor, backend'e gerek yok",
             "kanit": _k("kod", "celery_worker.py:42")},
            {"ad": "Beat", "ne": "basit cron; tek instance çalışmalı yoksa çift tetik",
             "durum": "kapali", "neden": "zamanlama kendi scheduler'ımızda",
             "kanit": _k("kod", "scheduler.py")},
            {"ad": "Flower", "ne": "zayıf ama var olan izleme UI'ı",
             "durum": "kapali", "neden": "bu POC'ta kurulmadı"},
            {"ad": "prefetch_multiplier", "ne": "worker'ın önden kaç mesaj çekeceği",
             "durum": "acik", "neden": "1'e çekildi — adil dağıtım",
             "kanit": _k("kod", "celery_worker.py:41")},
            {"ad": "visibility timeout", "ne": "Redis/SQS'te tuzak: task bundan uzun "
                                               "sürerse mesaj yeniden teslim edilir → "
                                               "aynı iş İKİ worker'da paralel koşar",
             "durum": "kapali", "neden": "filesystem broker'da bu kavram yok"},
        ],
        "guclu": [
            {"iddia": "En basit kurulum — bir Redis + `pip install celery`, bitti.",
             "kanit": _k("kod", "celery_worker.py", "99 satır, tek dosya")},
            {"iddia": "Yatay ölçek çok iyi: worker ekle, kapasite artsın.",
             "kanit": _k("rapor", "motor-secimi-workflow-desenleri-ve-karar.md")},
            {"iddia": "Python ekosisteminde standart; Django/Flask ile hazır entegrasyon.",
             "kanit": _k("repo", "harnesses/ — üçüncü taraf, bu POC'ta ölçülmedi")},
            {"iddia": "at-least-once teslim kutudan geliyor (acks_late + reject_on_worker_lost).",
             "kanit": _k("kod", "celery_worker.py:39-40")},
        ],
        "zayif": [
            {"iddia": "Çok adımlı iş kavramı YOK. Celery için 'task' = tek fonksiyon. "
                      "'3 adımlı sipariş süreci' diye bir şey bilmez.",
             "kanit": _k("rapor", "task-management-analizi.md", "A seviyesi karşılığı yok")},
            {"iddia": "Koşula bağlı dallanma `self.replace()` gerektirir ve akışın "
                      "mantığı task'ların İÇİNE dağılır — 'bu iş ne yapıyor' diye "
                      "bakılacak TEK bir yer kalmaz. Döngü (`while`) için özyineleme "
                      "kurman gerekir.",
             "kanit": _k("rapor", "task-management-analizi.md",
                         "Canvas'ın asıl sınırı burası; retry değil")},
            {"iddia": "Retry, adımlar TEK task'ın içindeyse fonksiyonu baştan koşturur "
                      "(fetch ×2). AMA canvas'ta (`chain`) her halka ayrı task'tır ve "
                      "ayrı retry alır — bu noktada Airflow düğümünden farkı yok.",
             "kanit": _k("olcum", "fetch ×2 (tek-task kurgusu)",
                         "DÜZELTME: chain'de bu geçerli değil")},
            {"iddia": "Durum görünürlüğü çok zayıf: akış takılırsa hangi adımda "
                      "olduğu bilinmiyor. 'Nerede kalmıştım' defterini kimse tutmaz.",
             "kanit": _k("olcum", "kendi kaydı YOK", "dört motor içinde tek")},
            {"iddia": "İptal/atlama kaydı yok — batan zincirin kalanına ne olduğu belirsiz.",
             "kanit": _k("rapor", "motor-secimi-workflow-desenleri-ve-karar.md §2.4")},
            {"iddia": "En yavaş: 12-40 sn. Bunun ~6 saniyesi worker açılışı.",
             "kanit": _k("olcum", "15,35 sn / 6 düğüm hatasız")},
            {"iddia": "Varsayılan ayarlarla (acks_late=False) worker çökerse iş KAYBOLUR.",
             "kanit": _k("kod", "celery_worker.py:39", "biz açtık; varsayılan kapalı")},
        ],
        "ne_zaman": "Bağımsız, kısa, tek adımlı işler: e-posta gönder, resim boyutlandır, "
                    "tek bir LLM inference'ını arka plana at.",
        "ne_zaman_degil": "Çok adımlı, state taşıyan, 'kaldığı yerden devam etsin' "
                          "istenen ajan akışları.",
    },

    # ───────────────────────────────── AIRFLOW ─────────────────────────────
    "airflow": {
        "baslik": "Airflow",
        "katman": "Zamanlı batch scheduler",
        "soyutlama": "DAG (statik graf)",
        "tek_cumle": "'Her gece 08:00'de şu adımları şu sırayla koştur, biri patlarsa "
                     "bana göster' işini yapar.",
        "ne_yapar": [
            "DAG okur — işi Python'da graf olarak yazarsın: `fetch >> process >> deliver`. "
            "Şekil KOŞMADAN ÖNCE bellidir.",
            "Zamanlar — cron'a göre tetikler. Ve backfill: bugün kurup '1 Ocak'tan beri "
            "her gün koşmuş olsun' dersen geçmiş 200 koşuyu geriye dönük üretir.",
            "Bağımlılığı yönetir — `process`, `fetch` bitmeden başlamaz.",
            "Durumu DB'ye yazar + UI'da gösterir — hangi koşu, hangi adım, hangi denemede.",
            "Adım bazında retry eder.",
        ],
        "analoji": {
            "baslik": "Fabrika üretim hattı + duvardaki dev pano",
            "metin": "Hat sabit (her gün aynı istasyonlar), panoda hangi istasyonda ne "
                     "olduğu canlı görünüyor. Bir istasyon bozulursa SADECE O İSTASYON "
                     "tekrarlanır, öncekiler yeniden çalışmaz.",
        },
        "mimari": """   DAG dosyaları (.py)
         │ parse (DagFileProcessor)
         ▼
   ┌───────────┐   hazır task'lar   ┌──────────┐
   │ Scheduler │ ─────────────────▶ │ Executor │──▶ Worker'lar
   └─────┬─────┘                    └──────────┘   (Local/Celery/K8s)
         │                               │
         ▼                               ▼
   ┌────────────────────────────────────────┐
   │ Metadata DB   ← TEK DOĞRULUK KAYNAĞI   │
   │ dag_run · task_instance · xcom · log   │
   └────────────────────────────────────────┘
         ▲                  ▲
    Webserver/UI       Triggerer (deferrable async bekleme)

  Koşmak için EN AZ 4 bileşen ayakta olmalı.""",
        "moduller": [
            {"ad": "DAG parser", "ne": "dosyayı okur, grafı çıkarır",
             "durum": "acik", "neden": "grafı biz üretip dosyaya yazıyoruz",
             "kanit": _k("kod", "orchestrator.py:791", "export_airflow_dag")},
            {"ad": "Scheduler", "ne": "cron'a göre tetikler, hazır task'ları executor'a verir",
             "durum": "kapali", "neden": "`airflow dags test` ile doğrudan tetikliyoruz",
             "kanit": _k("kod", "airflow_runner.py:137-138", "TARİH VERİLMİYOR — utcnow")},
            {"ad": "Executor", "ne": "Sequential / Local / Celery / Kubernetes",
             "durum": "acik", "neden": "SequentialExecutor — seri, tek süreç",
             "kanit": _k("kod", "airflow_runner.py", "eşzamanlılık YOK, kilit gerekti")},
            {"ad": "Metadata DB", "ne": "tek doğruluk kaynağı: dag_run, task_instance, xcom",
             "durum": "acik", "neden": "sonuçları buradan read-only okuyoruz",
             "kanit": _k("kod", "airflow_runner.py", "sqlite, mode=ro")},
            {"ad": "XCom", "ne": "task'lar arası küçük veri aktarımı",
             "durum": "acik",
             "neden": "veri akışı gerçekten oradan geçiyor; çıktıyı board'dan DEĞİL "
                      "XCom'dan okuyoruz",
             "kanit": _k("olcum", "byte-byte aynı çıktı",
                         "bağımsız kanıt: Airflow board'a hiç yazmıyor")},
            {"ad": "Webserver / UI", "ne": "graf görünümü, Gantt, log, manuel tetik",
             "durum": "kapali", "neden": "bu POC'ta ayağa kaldırılmadı"},
            {"ad": "Triggerer", "ne": "deferrable operator'lar worker slotu tutmadan bekler",
             "durum": "kapali", "neden": "sensör/deferrable kullanmıyoruz"},
            {"ad": "retries / retry_delay", "ne": "adım bazında yeniden deneme",
             "durum": "acik", "neden": "devrede — ama varsayılan 30 sn panelde 1 sn'ye çekildi",
             "kanit": _k("olcum", "30 sn × 3 = 62 sn", "canlı ölçüldü")},
            {"ad": "trigger_rule", "ne": "koşullu dalda birleşim düğümünün kuralı",
             "durum": "acik", "neden": "unutulursa birleşim de skip olur — iki kavram birlikte",
             "kanit": _k("rapor", "motor-secimi-workflow-desenleri-ve-karar.md §2.4")},
            {"ad": "catchup / backfill", "ne": "kaçırılan koşuları geriye dönük üretir",
             "durum": "kapali", "neden": "catchup=False; POC tek koşu",
             "kanit": _k("kod", "orchestrator.py:791", "schedule parametreye çıkarıldı")},
            {"ad": ".expand() dynamic mapping", "ne": "düğüm SAYISI çalışma anında belli olur",
             "durum": "kapali", "neden": "bu grafta kullanılmıyor",
             "kanit": _k("olcum", "5 örnek", "D4 deseninde ölçüldü")},
            {"ad": "Pool / max_active_runs", "ne": "kaynak kotası, eşzamanlılık sınırı",
             "durum": "kapali", "neden": "SequentialExecutor zaten seri — etkisiz"},
            {"ad": "Sensor", "ne": "koşul sağlanana kadar bekler",
             "durum": "kapali", "neden": "kullanılmıyor"},
            {"ad": "Hook / Connection", "ne": "dış sistem bağlantı yönetimi",
             "durum": "kapali", "neden": "dış sistem yok"},
        ],
        "guclu": [
            {"iddia": "Scheduling'de rakipsiz: cron, catchup/backfill, veri-farkındalıklı tetik.",
             "kanit": _k("rapor", "zamanlama-cron-raporu.md")},
            {"iddia": "Operasyonel görünürlükte açık ara birinci. Her düğümün durumu "
                      "kayıtlı: success/failed/upstream_failed/skipped + try_number.",
             "kanit": _k("olcum", "z1..z6 success · z7 failed ×3 · z8-z10 upstream_failed")},
            {"iddia": "`skipped` kaydı var — koşullu dalda seçilmeyen yol açıkça "
                      "kaydediliyor. Dört motor içinde bunu yapan TEK motor.",
             "kanit": _k("olcum", "D3 koşullu: 4 success + 1 skipped")},
            {"iddia": "Kısmi devam: process patlarsa fetch tekrar koşmaz, success kalır.",
             "kanit": _k("olcum", "fetch ×1", "çapa ölçümü")},
            {"iddia": "Yüzlerce hazır operator/provider (S3, BigQuery, Spark, dbt).",
             "kanit": _k("repo", "apache/airflow", "bu POC'ta ölçülmedi")},
        ],
        "zayif": [
            {"iddia": "Operasyonel olarak ağır: scheduler + webserver + metadata DB + "
                      "executor, en az 4 bileşen ayakta olmalı.",
             "kanit": _k("kod", "airflow_runner.py", "biz sadece `dags test` koşturuyoruz")},
            {"iddia": "İş STATİK olmak zorunda — graf koşmadan önce belli olmalı. "
                      "Ajanın sonuca bağlı serbest dallanması buna sığmaz.",
             "kanit": _k("rapor", "motor-secimi-workflow-desenleri-ve-karar.md §2.4")},
            {"iddia": "Düğüm İÇİNDE checkpoint yok: bir adım 10 işten 7'sini yapıp "
                      "çökerse retry o adımı baştan koşar. LLM düğümünde bu PARA demek.",
             "kanit": _k("rapor", "motor-secimi-workflow-desenleri-ve-karar.md")},
            {"iddia": "Varsayılan 30 sn backoff: üç deneme = 62 sn. Hızlı akışlar için "
                      "uyumsuz. Kusur değil, tercih — batch işler için doğru.",
             "kanit": _k("olcum", "09:47:19 → 09:47:50 → 09:48:21", "canlı ölçüldü")},
            {"iddia": "Kurulum kırılgan: constraint dosyası typing_extensions'ı düşürüp "
                      "pydantic'i kırdı.",
             "kanit": _k("olcum", "bu oturumda yaşandı", "venv onarıldı")},
            {"iddia": "Uzun beklemeler için tasarlanmadı — '3 gün insan onayı bekle' "
                      "Airflow'un modeli değil.",
             "kanit": _k("rapor", "task-management-analizi.md")},
        ],
        "ne_zaman": "Veri/ETL pipeline'ları, gecelik raporlar, backfill ihtiyacı, "
                    "operatör görünürlüğü merkezde.",
        "ne_zaman_degil": "Saatlerce/günlerce duraklayan uzun işler, dinamik ajan döngüleri.",
    },

    # ──────────────────────────────── TEMPORAL ─────────────────────────────
    "temporal": {
        "baslik": "Temporal",
        "katman": "Durable execution motoru",
        "soyutlama": "workflow = kod",
        "tek_cumle": "Makinen çökse, deploy atsan, süreç ölse bile program kaldığı "
                     "yerden devam eder — sen tek satır kurtarma kodu yazmadan.",
        "ne_yapar": [
            "Workflow'u KOD olarak yazarsın — normal `if`, `for`, `try`. Config değil, "
            "DAG değil, düpedüz kod.",
            "Her adımı kalıcı loga yazar: 'fetch başladı', 'fetch bitti, sonucu şu'…",
            "Çökünce REPLAY eder: yeni worker kodu baştan çalıştırır ama her adımda "
            "önce log'a bakar — 'bu zaten bitmiş' → adımı ÇALIŞTIRMAZ, kayıtlı sonucu döndürür.",
            "Otomatik retry — her adıma RetryPolicy verirsin, sen try/except yazmazsın.",
            "Aylarca dayanıklı bekler: `await sleep(30 gün)` gerçek ve ucuzdur.",
        ],
        "analoji": {
            "baslik": "Kaydedilmiş oyun",
            "metin": "Elektrik kesildi diye oyuna baştan başlamıyorsun; motor son duruma "
                     "geri yükleyip devam ettiriyor. Üstelik bu kaydı SEN yapmıyorsun.",
        },
        "mimari": """  Client ──▶ ┌──────────────────────────────────────┐
             │  Temporal Cluster                    │
             │  Frontend · History · Matching       │
             │  Persistence (Cassandra/PG/MySQL)    │
             │  + Elasticsearch (visibility)        │
             └───────┬──────────────────────────────┘
                     │ long-poll (task queue)
                     ▼
             ┌──────────────────────┐
             │ SDK Worker (SENİN)   │ workflow kodun + activity kodun
             └──────────────────────┘

  KRİTİK: senin kodun cluster'da ÇALIŞMAZ. Cluster sadece state ve kuyruk tutar.""",
        "moduller": [
            {"ad": "Workflow", "ne": "orkestrasyon mantığı; DETERMİNİST olmak zorunda",
             "durum": "acik", "neden": "dispatch döngüsü durable workflow olarak yazıldı",
             "kanit": _k("kod", "temporal_defs.py:86-119", "BoardWorkflow")},
            {"ad": "Activity", "ne": "yan etkili her şey (HTTP, DB, LLM, dosya)",
             "durum": "acik", "neden": "iş activity'lerde; workflow gövdesi saf",
             "kanit": _k("kod", "temporal_defs.py:41", "execute_one_task")},
            {"ad": "Event history", "ne": "değişmez olay logu — sihrin tamamı",
             "durum": "acik", "neden": "gerçekten yazılıyor ve replay çalışıyor",
             "kanit": _k("olcum", "23 durable event")},
            {"ad": "Replay", "ne": "çökünce baştan koşar ama biten activity'yi ATLAR",
             "durum": "acik", "neden": "exactly-once ilerleme",
             "kanit": _k("olcum", "fetch ×1", "Celery'de aynı adım ×2")},
            {"ad": "RetryPolicy", "ne": "activity seviyesi otomatik yeniden deneme",
             "durum": "esgudum",
             "neden": "DEVREDE (maximum_attempts=3) ama board sayacıyla ÇAKIŞTI. "
                      "Temporal aynı activity'yi aynı payload ile tekrar çağırıyor; "
                      "board.fail() claim'i temizlediği için 'bayat claim onarımı' "
                      "eklemek gerekti. İki dayanıklılık katmanı bedava değil.",
             "kanit": _k("hata", "②", "geçici hata kalıcı sanılıyordu; sayaç board'a çevrildi")},
            {"ad": "Task queue", "ne": "worker'ların long-poll ettiği kuyruk",
             "durum": "acik", "neden": "test-server üzerinden"},
            {"ad": "Signal / Query / Update", "ne": "dışarıdan mesaj, state okuma, ikisi",
             "durum": "kapali", "neden": "insan onayı akışı bu POC'ta yok"},
            {"ad": "Timer", "ne": "aylarca dayanıklı bekleme",
             "durum": "kapali", "neden": "uzun bekleme senaryosu yok"},
            {"ad": "Child workflow", "ne": "alt iş",
             "durum": "kapali", "neden": "board DAG kapısını tuttuğu için gerekmedi"},
            {"ad": "Continue-as-new", "ne": "history şişince (~50K event) temiz history",
             "durum": "kapali", "neden": "POC ölçeğinde limite yaklaşılmıyor"},
            {"ad": "Schedules", "ne": "cron muadili + backfill",
             "durum": "kapali", "neden": "zamanlama kendi scheduler'ımızda"},
            {"ad": "Versioning (patched)", "ne": "canlı workflow varken kod değişirse replay bozulur",
             "durum": "kapali", "neden": "tek sürüm koşuyor — ama gerçek öğrenme maliyeti bu"},
            {"ad": "Activity heartbeat", "ne": "uzun activity'de canlılık + activity-içi checkpoint",
             "durum": "kapali", "neden": "checkpoint board'da tutuluyor"},
            {"ad": "Sticky execution", "ne": "worker'da cache'lenmiş workflow, tam replay gerekmesin",
             "durum": "acik", "neden": "SDK varsayılanı"},
        ],
        "guclu": [
            {"iddia": "Dayanıklılıkta sınıfının en iyisi: tamamlanan adım bir daha "
                      "ASLA koşmaz (workflow ilerlemesi exactly-once).",
             "kanit": _k("olcum", "fetch ×1, process ×2")},
            {"iddia": "Kurtarma kodu yazmazsın — retry, timeout, state saklama, crash "
                      "recovery motorun işi.",
             "kanit": _k("kod", "temporal_defs.py:86-119", "119 satır, try/except yok")},
            {"iddia": "Hem dinamik hem durable. Kod aktıkça geçmişe yazılır; akış "
                      "runtime'da şekillenebilir. Ajan işleri için kritik olan TEK motor.",
             "kanit": _k("rapor", "motor-secimi-workflow-desenleri-ve-karar.md §5.1")},
            {"iddia": "Grafı ifade etmek en doğal: koşullu dal `if`, fan-out `for`. "
                      "Motor-özel kavram öğrenmiyorsun.",
             "kanit": _k("olcum", "D3/D4 desenleri 0,16-0,18 sn")},
            {"iddia": "own'dan sonra en hızlı: 0,16-0,50 sn.",
             "kanit": _k("olcum", "0,46 sn / 6 düğüm hatasız")},
            {"iddia": "Uzun bekleme birinci sınıf: günler/haftalar süren insan onayı.",
             "kanit": _k("repo", "temporalio/sdk-python", "bu POC'ta ölçülmedi")},
        ],
        "zayif": [
            {"iddia": "Determinizm disiplini pazarlık dışı: workflow kodunda random(), "
                      "datetime.now(), doğrudan HTTP YASAK. IO'yu activity'ye sarmak "
                      "MİMARİYİ BÖLMEK demek.",
             "kanit": _k("kod", "temporal_defs.py:88", "'Gövde saf; iş activity'lerde'")},
            {"iddia": "Operasyonel maliyet en yüksek: cluster ya da Temporal Cloud.",
             "kanit": _k("olcum", "test-server binary 83 MB",
                         "/tmp'de tutuluyor; silinirse ilk koşu internetten indirir")},
            {"iddia": "'Task / iş kalemi' kavramı yok — activity var, board benzeri "
                      "model yok. Düğüm bazlı tabloyu tek başına veremez.",
             "kanit": _k("rapor", "motor-secimi-workflow-desenleri-ve-karar.md §3")},
            {"iddia": "Seçilmeyen dal kaydı yok — 'skipped' diye bir durum yok.",
             "kanit": _k("rapor", "motor-secimi-workflow-desenleri-ve-karar.md §2.4")},
            {"iddia": "Kendi retry'ı bizimkiyle çakıştı; uzlaştırmak için 'bayat claim "
                      "onarımı' yazmak gerekti.",
             "kanit": _k("hata", "②")},
            {"iddia": "Versioning derdi: canlıda koşan workflow varken kodu değiştirirsen "
                      "replay bozulabilir.",
             "kanit": _k("repo", "temporalio — workflow.patched()")},
        ],
        "ne_zaman": "Uzun süren, çok adımlı, para/veri kaybı kabul edilemeyen işler: "
                    "ödeme akışları, sipariş saga'ları, ajan workflow'ları.",
        "ne_zaman_degil": "'Gecelik SQL export' — gereksiz ağırlık.",
    },


    # ────────────────────────── CELERY CANVAS ──────────────────────────────
    # Board'lu Celery'den AYRI bir sekme: aynı kütüphane, bambaşka kullanım.
    "celery_canvas": {
        "baslik": "Celery Canvas",
        "katman": "Celery'nin KENDİ workflow ifadesi",
        "soyutlama": "imza kompozisyonu (chain/group/chord)",
        "tek_cumle": "Celery'de 'DAG' diye bir şey yoktur; onun yerine imzaları "
                     "birbirine bağlayan CANVAS vardır.",
        "ne_yapar": [
            "`chain(a.s(), b.s(), c.s())` — sırayla; a'nın DÖNÜŞÜ b'nin İLK argümanı olur.",
            "`group(a.s(), b.s())` — paralel fan-out.",
            "`chord(group(…), callback)` — hepsi bitince callback; callback group "
            "sonuçlarını LİSTE olarak alır.",
            "Akış YAYIN ANINDA mesaja serileştirilir — sonrasında değiştirilemez.",
        ],
        "analoji": {
            "baslik": "Birbirine zımbalanmış sipariş fişleri",
            "metin": "Garson fişleri sırayla zımbalayıp çiviye takıyor. Aşçı üsttekini "
                     "alıp yapıyor, altındakini bir sonrakine veriyor. Ama PANODA "
                     "hangi fişin nerede olduğu YAZMIYOR — zincir ortada kesilirse "
                     "kalanı kimse aramıyor.",
        },
        "mimari": """  chain(fetch.s(), process.s(), deliver.s())
        │
        ▼ apply_async() — TÜM zincir mesaja serileştirilir
  ┌─────────────────────────────────────────────┐
  │ BROKER   [fetch | →process | →deliver]      │  zincir mesajın İÇİNDE
  └────────────────┬────────────────────────────┘
                   ▼
             Worker: fetch koşar → dönüş process'in İLK argümanı
                                 → process koşar → deliver…

  chord için SONUÇ DEPOSU (result backend) ŞART — group sonuçları
  bir yerde birikmeli ki callback onları liste olarak alabilsin.

  Elde yalnız AsyncResult var: ready() / get(). "Neredeyiz" YOK.""",
        "moduller": [
            {"ad": "signature (.s / .si)", "ne": "task + argüman paketi; canvas'ın yapı taşı",
             "durum": "acik", "neden": "`.si` immutable — önceki dönüşü ALMAZ (ilk halka)",
             "kanit": _k("kod", "motor_canvas.py", "canvas_fn.si(None, …)")},
            {"ad": "chain", "ne": "sıralı; önceki dönüş sonrakinin İLK argümanı",
             "durum": "acik", "neden": "graf katmanları chain'e çevriliyor",
             "kanit": _k("olcum", "chain(4 katman) → group(2) → group(2) → … → export_csv")},
            {"ad": "group", "ne": "paralel fan-out",
             "durum": "acik", "neden": "aynı katmandaki düğümler grup oluyor"},
            {"ad": "chord", "ne": "group bitince callback; callback LİSTE alır",
             "durum": "acik", "neden": "chain(group(...), sig) Celery'de otomatik chord olur"},
            {"ad": "result backend", "ne": "chord'un sonuçları biriktirdiği yer",
             "durum": "acik",
             "neden": "ŞART — backend'siz chord çalışmaz. Board'lu koşumuzda backend "
                      "KAPALI çünkü sonuç board'a yazılıyor.",
             "kanit": _k("kod", "motor_canvas.py", "backend=f'file://{S}'")},
            {"ad": "self.replace()", "ne": "task kendi yerine çalışma anında yeni akış koyar",
             "durum": "kapali",
             "neden": "koşullu dallanmanın TEK yolu; bu POC'ta kullanılmıyor",
             "kanit": _k("rapor", "motor-secimi-workflow-desenleri-ve-karar.md")},
            {"ad": "link_error", "ne": "zincir batarsa çağrılacak callback",
             "durum": "kapali",
             "neden": "kurulmazsa batan zincir SESSİZCE ölür — bu POC'ta kasıtlı kurulmadı",
             "kanit": _k("olcum", "kalıcı hata: 60 sn bekledi, hiçbir kayıt yok")},
            {"ad": "AsyncResult", "ne": "koşunun tek göstergesi",
             "durum": "esgudum",
             "neden": "yalnız ready()/get() var — 'hangi adımdayız' sorusunun "
                      "cevabı YOK. Canvas'ın asıl sınırı bu.",
             "kanit": _k("olcum", "5/20/45 sn: ready()=False, başka bilgi yok")},
        ],
        "guclu": [
            {"iddia": "KURULUM: zaten Celery varsa `chain` yazmanın maliyeti SIFIR. "
                      "3-5 adımlık sıralı iş + biraz paralellik için Temporal ya da "
                      "Airflow kurmak abartı.",
             "kanit": _k("kod", "motor_canvas.py", "tek dosya, ek servis yok")},
            {"iddia": "Sıralı zincir, paralel ve toplama üçü de var — ifade gücü bu "
                      "üç desende Airflow'la eşit.",
             "kanit": _k("olcum", "6 düğümlü graf 9,03 sn'de doğru CSV üretti")},
            {"iddia": "ADIM SEVİYESİ RETRY var: zincirin her halkası AYRI task'tır, "
                      "ayrı retry alır. 'Retry baştan koşar' iddiası canvas için "
                      "GEÇERSİZ — o, adımlar tek fonksiyonun içindeyken doğru.",
             "kanit": _k("kod", "motor_canvas.py", "canvas_fn max_retries=2, halka bazında")},
            {"iddia": "Biten adım korunur: zincirde geri kalan halkalar yeniden koşmaz.",
             "kanit": _k("olcum", "hatasız koşuda her düğüm ×1")},
        ],
        "zayif": [
            {"iddia": "'NEREDE KALDI' sorusunun cevabı HİÇBİR YERDE yok. Koşarken "
                      "elde yalnız `ready()` var: bitti / bitmedi.",
             "kanit": _k("olcum", "kalıcı hatada 60 sn bekledi, nerede takıldığı bilinmiyor")},
            {"iddia": "Kalıcı hata SESSİZCE öldürür: `link_error` kurmazsan batan "
                      "zincirin kalanına ne olduğu hiçbir yere yazılmaz.",
             "kanit": _k("olcum", "canvas ✗ 66,04 sn · board'lu aynı senaryo: "
                                  "3/6 done, 1 failed, 2 CANCELLED")},
            {"iddia": "Veri akışı POZİSYONEL — düğüm kimliği kaybolur. Hangi upstream'in "
                      "hangi veriyi verdiği bilinmiyor; graf FONKSİYON İMZASINA sızıyor.",
             "kanit": _k("kod", "motor_canvas.py", "sentetik anahtar `_canvas_0` uydurmak gerekti")},
            {"iddia": "Koşula bağlı dallanma `self.replace()` ister; akışın mantığı "
                      "task'ların İÇİNE dağılır, bakılacak tek yer kalmaz.",
             "kanit": _k("rapor", "task-management-analizi.md")},
            {"iddia": "Döngü (while) yok — özyinelemeyle zorlanır.",
             "kanit": _k("rapor", "task-management-analizi.md")},
            {"iddia": "Uzun bekleme ve dışarıdan sinyal yok.",
             "kanit": _k("rapor", "motor-secimi-workflow-desenleri-ve-karar.md")},
            {"iddia": "Akış YAYIN ANINDA sabitlenir — mesaja serileştirilir, sonra "
                      "değiştirilemez.",
             "kanit": _k("olcum", "apply_async() sonrası graf donuyor")},
        ],
        "ne_zaman": "Sabit, kısa, dallanmasız boru hattı. 3-5 adım + biraz paralellik. "
                    "Zaten Celery varsa en ucuz yol.",
        "ne_zaman_degil": "Dallanma, döngü, uzun bekleme ya da 'nerede kaldı' "
                          "sorgusu gerekiyorsa.",
    },

    # ────────────────────────────────── OWN ────────────────────────────────
    "own": {
        "baslik": "own — hermes tarzı board",
        "katman": "Kalıcı task tablosu (BUILD rotası)",
        "soyutlama": "task tablosu + FSM",
        "tek_cumle": "Durumu kendi kalıcı tablonda tut, motoru değiştirilebilir bırak — "
                     "karar merci sende kalsın.",
        "ne_yapar": [
            "Hedefi task'lara böler (`plan_phase`) ve SQLite'a yazar.",
            "Bağımlılık kapısını tutar: parents done olunca blocked→ready.",
            "CAS-claim ile at-most-once dağıtım sağlar; lease+heartbeat ile çökeni fark eder.",
            "Retry, circuit-breaker, iptal zinciri ve düğüm-içi checkpoint burada.",
            "Diğer üç motoru ALTINDA dağıtıcı olarak koşturur.",
        ],
        "analoji": {
            "baslik": "Mutfaktaki sipariş defteri",
            "metin": "Fişleri aşçıya sen dağıtıyorsun ama DEFTER sende: hangi sipariş "
                     "hangi aşamada, kaç kez denendi, kim aldı. Aşçıyı değiştirsen de "
                     "defter aynı kalır.",
        },
        "mimari": """  plan_phase (LLM) ──▶ board (SQLite) ──▶ dispatcher ──▶ MOTOR
                          │                              (own/celery/
                          │                               airflow/temporal)
                          ▼
   ┌──────────────────────────────────────────────────┐
   │ tasks: id · status · parents · attempt · claim   │
   │ FSM: blocked → ready → running → done            │
   │                    └──────────────► failed       │
   │                    └──────────────► cancelled    │
   │ CAS claim · lease+heartbeat · recover_stale      │
   │ circuit breaker · checkpoint · olay günlüğü      │
   └──────────────────────────────────────────────────┘""",
        "moduller": [
            {"ad": "Task tablosu", "ne": "id, durum, parents, attempt, claim_lock, result",
             "durum": "acik", "neden": "tek doğruluk kaynağı",
             "kanit": _k("kod", "taskboard.py:48")},
            {"ad": "FSM", "ne": "blocked→ready→running→done/failed/cancelled",
             "durum": "acik", "neden": "geçişler tek yerde",
             "kanit": _k("kod", "taskboard.py:17-20")},
            {"ad": "CAS claim", "ne": "iki worker aynı task'ı alamaz (at-most-once)",
             "durum": "acik", "neden": "ölçüldü",
             "kanit": _k("olcum", "at-most-once ✓ 5,0× eşzamanlı")},
            {"ad": "lease + heartbeat", "ne": "claim süreli; yenilenmezse bayat sayılır",
             "durum": "acik", "neden": "30 sn lease",
             "kanit": _k("kod", "taskboard.py")},
            {"ad": "recover_stale", "ne": "çökeni fark et → ready (checkpoint korunur)",
             "durum": "acik", "neden": "Celery'de çöken worker için de gerekti",
             "kanit": _k("kod", "taskboard.py:26")},
            {"ad": "circuit breaker", "ne": "üst üste hata → failed (sonsuz retry yok)",
             "durum": "acik", "neden": "kalıcı hata 3 denemede kesiliyor",
             "kanit": _k("olcum", "failed ×3, dört motorda da")},
            {"ad": "cancel_downstream", "ne": "batan düğümün ardılları cancelled",
             "durum": "acik", "neden": "dört motor içinde bunu veren tek katman",
             "kanit": _k("hata", "⑦", "ardıllar sonsuza dek blocked kalıyordu")},
            {"ad": "checkpoint", "ne": "düğüm İÇİ ara sonuç — çökme sonrası kaybolmasın",
             "durum": "acik", "neden": "üç motorun hiçbirinde yok",
             "kanit": _k("kod", "taskboard.py", "save_checkpoint")},
            {"ad": "olay günlüğü", "ne": "created/claimed/completed/failed/recovered",
             "durum": "acik", "neden": "denetim izi"},
            {"ad": "zamanlama", "ne": "cron benzeri tetik",
             "durum": "acik", "neden": "ayrı scheduler yazıldı — olgun değil",
             "kanit": _k("kod", "scheduler.py", "test_zamanlama 43/43")},
        ],
        "guclu": [
            {"iddia": "En hızlı: 0,01-0,05 sn. Ağ yok, broker yok, süreç yok.",
             "kanit": _k("olcum", "0,05 sn / 6 düğüm hatasız")},
            {"iddia": "Tek karar noktası: retry, iptal zinciri, breaker, checkpoint "
                      "hep burada. Motor değiştirmek davranışı DEĞİŞTİRMİYOR.",
             "kanit": _k("olcum", "dört motor byte-byte aynı çıktı")},
            {"iddia": "`cancelled` durumu var — Celery ve Temporal'da bu kavram yok.",
             "kanit": _k("olcum", "kalıcı hatada 2 ardıl cancelled")},
            {"iddia": "Düğüm İÇİ checkpoint — üç motorda da yok.",
             "kanit": _k("olcum", "çökme sonrası kısmi iş korundu")},
            {"iddia": "Sıfır yeni servis. SQLite dosyası, o kadar.",
             "kanit": _k("kod", "taskboard.py", "601 satır")},
        ],
        "zayif": [
            {"iddia": "BİZ yazdık — kenar durumlarını da biz bulmak zorundayız. "
                      "Bu çalışmada 12 hata çıktı ve hepsi bizim koddaydı.",
             "kanit": _k("hata", "①-⑫", "hepsi motor entegrasyonu ya da board mantığında")},
            {"iddia": "Tek makine, tek süreç — dağıtık değil.",
             "kanit": _k("kod", "taskboard.py", "SQLite, tek dosya")},
            {"iddia": "Operasyon aracı yok: Airflow'un UI'ı gibi bir şey yok.",
             "kanit": _k("rapor", "motor-secimi-workflow-desenleri-ve-karar.md §3")},
            {"iddia": "Prod'da kanıtlanmadı: Airflow'un 10 yılı, Temporal'ın ekosistemi yok.",
             "kanit": _k("rapor", "task-yonetimi-altyapi-karari.md")},
        ],
        "ne_zaman": "Mevcut engine zaten oturum/state yönetiyor, tek eksik durable "
                    "kuyruk + recovery ise. Küçük/orta ölçekte Temporal garantilerinin "
                    "pratik karşılığını çok daha ucuza verir.",
        "ne_zaman_degil": "Dağıtık ölçek, operatör UI'ı ya da kanıtlanmış olgunluk şartsa.",
    },
}

# ── ÇAPA ÖLÇÜM — her sekmede tekrarlanır ───────────────────────────────────
# Senaryo: fetch → process → deliver; process İLK denemede geçici hata veriyor.
# Tek soru: pahalı olan `fetch` retry'da kaç kez koşuyor?
CAPA = {
    "senaryo": "fetch → process → deliver · process ilk denemede patlıyor",
    "soru": "Pahalı olan `fetch` adımı KAÇ KEZ koştu?",
    "neden_onemli": "Bu adım bir LLM çağrısı ya da ödeme ise, iki kez koşmak iki kez "
                    "ödemek demek. 'Kaldığı yerden devam' iddiasının tek somut ölçüsü bu.",
    "satirlar": {
        "celery":   {"fetch": 2, "retry_seviyesi": "task · self.retry",
                     "cokme_sonrasi": "broker redelivery (acks_late)",
                     "defter": "hiç kimse — sen tutarsın"},
        "airflow":  {"fetch": 1, "retry_seviyesi": "task retries",
                     "cokme_sonrasi": "sadece hatalı düğüm",
                     "defter": "sistem, ama YALNIZ düğümler arasında"},
        "temporal": {"fetch": 1, "retry_seviyesi": "activity RetryPolicy",
                     "cokme_sonrasi": "history replay, biten activity atlanır",
                     "defter": "sistem (event history)"},
        "celery_canvas": {"fetch": 1, "retry_seviyesi": "halka bazında self.retry",
                          "cokme_sonrasi": "zincir kaybolur — kalan sessizce ölür",
                          "defter": "hiç kimse — AsyncResult yalnız bitti/bitmedi der"},
        "own":      {"fetch": 1, "retry_seviyesi": "board breaker",
                     "cokme_sonrasi": "recover_stale + checkpoint",
                     "defter": "sistem (board tablosu)"},
    },
    "uyari": "'Kaldığı yerden' ≠ 'adımın ortasından'. Temporal tamamlanmış activity'yi "
             "atlar ama ÇALIŞIRKEN çöken activity'yi baştan dener. Airflow düğümü, "
             "Celery task'ı zaten baştan koşar. Üçünde de adım içindeki yan etki "
             "idempotent olmalı — yoksa çift yazma / çift ücret.",
}

# ── İNCELİKLER — yalnız o motorda olan davranışlar ─────────────────────────
# Güçlü/zayıf listesi "hangisini seçeyim" sorusunu cevaplıyor. Bu bölüm AYRI bir
# soruyu cevaplıyor: "bu motoru kullanırsan neyi BİLMEN gerekir?"
# Ölçüt: başka üç motorda karşılığı OLMAYAN davranış. Genel doğrular buraya girmez.
INCELIKLER = {
    "celery": [
        {"ad": "visibility timeout tuzağı",
         "ne": "Redis/SQS'te task, visibility timeout'tan uzun sürerse broker mesajı "
               "'kaybolmuş' sayıp YENİDEN TESLİM eder → aynı iş İKİ worker'da PARALEL "
               "koşar. `visibility_timeout` > en uzun task süresi olmak ZORUNDA.",
         "sonuc": "Çift yazma / çift ücret. Sessiz — hata vermez, iki kez çalışır.",
         "kanit": _k("repo", "celery — broker_transport_options",
                     "filesystem broker'da bu kavram yok, bizde tetiklenmiyor")},
        {"ad": "prefork bellek tuzağı",
         "ne": "Varsayılan pool fork tabanlı: her worker ana sürecin bellek kopyasını "
               "alır. LLM/ML kütüphaneleri yüklüyse worker başına yüzlerce MB.",
         "sonuc": "4 worker = 4× model belleği. `--pool=solo/gevent` gerekebilir.",
         "kanit": _k("repo", "celery — worker pool seçenekleri")},
        {"ad": "acks_late varsayılanı KAPALI",
         "ne": "Varsayılan `acks_late=False`: mesaj worker'a ÇEKİLDİĞİ anda ack'lenir. "
               "Worker iş ortasında çökerse mesaj çoktan silinmiştir.",
         "sonuc": "İş sessizce kaybolur. Biz açtık — açmasaydık çökme testi geçmezdi.",
         "kanit": _k("kod", "celery_worker.py:39", "task_acks_late=True")},
        {"ad": "filesystem broker: in == out ŞART",
         "ne": "`data_folder_in` ile `data_folder_out` aynı dizin olmalı. Farklı "
               "verilirse Celery hata vermez, mesaj hiç teslim edilmez.",
         "sonuc": "Sessiz çalışmama. Bu POC'ta yaşandı, bulunması zaman aldı.",
         "kanit": _k("kod", "celery_worker.py:29-37")},
        {"ad": "Beat tek instance olmalı",
         "ne": "Celery Beat iki kez ayaktaysa her cron ÇİFT tetiklenir; dağıtık kilit yok.",
         "sonuc": "Gecelik iş iki kez koşar.",
         "kanit": _k("repo", "celery — beat")},
    ],
    "airflow": [
        {"ad": "zombie detection",
         "ne": "Worker heartbeat'i kaçırırsa scheduler task'ı ZOMBIE ilan edip "
               "fail/retry eder — task hâlâ koşuyor olsa bile.",
         "sonuc": "Uzun süren ama sessiz bir task öldürülebilir.",
         "kanit": _k("repo", "apache/airflow — scheduler zombie reaper")},
        {"ad": "koşullu dal İKİ kavram gerektirir",
         "ne": "`BranchPythonOperator` dalı seçer, ama birleşim düğümünde "
               "`trigger_rule` ayarlanmazsa varsayılan `all_success` yüzünden "
               "birleşim de SKIP olur.",
         "sonuc": "Dal doğru seçilir, akış yine de durur. En sık yapılan hata.",
         "kanit": _k("rapor", "motor-secimi-workflow-desenleri-ve-karar.md §2.4")},
        {"ad": "tarih argümanı verme tuzağı",
         "ne": "`airflow dags test <id>` tarih VERİLMEZSE `utcnow()` kullanır ve her "
               "tetikleme benzersiz `run_id` alır. Sabit tarih verirsen aynı run çakışır.",
         "sonuc": "Aynı DAG'ı iki kez koşturamazsın. Bu POC'ta bulundu.",
         "kanit": _k("kod", "airflow_runner.py:137-138", "TARİH VERME")},
        {"ad": "XCom metadata DB'de saklanır",
         "ne": "Düğümler arası veri metadata veritabanına yazılır — büyük veri koymak "
               "DB'yi şişirir. XCom küçük veri içindir.",
         "sonuc": "Büyük çıktıyı diske/S3'e yazıp XCom'a yol koymak gerekir.",
         "kanit": _k("kod", "airflow_runner.py", "xcom tablosu read-only okunuyor")},
        {"ad": "constraint dosyası venv'i kırabilir",
         "ne": "Airflow'un pinlediği constraint dosyası mevcut paketleri DÜŞÜREBİLİR.",
         "sonuc": "Bu POC'ta `typing_extensions` düştü, `pydantic` kırıldı.",
         "kanit": _k("olcum", "bu oturumda yaşandı", "venv elle onarıldı")},
        {"ad": "SequentialExecutor + sqlite eşzamanlılık vermez",
         "ne": "İki `dags test` aynı anda koşarsa 'database is locked'.",
         "sonuc": "Koşumları kilitle serileştirmek ZORUNLU, opsiyonel değil.",
         "kanit": _k("kod", "airflow_runner.py", "_KILIT")},
    ],
    "temporal": [
        {"ad": "başarısız deneme history'ye YAZILMAZ",
         "ne": "Otomatik retry'lanan başarısız activity denemesi event history'ye "
               "kaydedilmez (history kompakt kalsın diye). Attempt numarası "
               "`ActivityTaskStarted` event'inde taşınır.",
         "sonuc": "'Kaç kez denendi' sorusunu history'yi sayarak cevaplayamazsın.",
         "kanit": _k("kod", "temporal_defs.py", "deneme sayacı board'dan alınıyor")},
        {"ad": "continue-as-new",
         "ne": "History ~50K event / 50MB'ı aşınca workflow'un kendini temiz history "
               "ile yeniden başlatması gerekir.",
         "sonuc": "Uzun süren workflow'da bunu SEN planlamak zorundasın.",
         "kanit": _k("repo", "temporalio — ContinueAsNew")},
        {"ad": "workflow kodu değişince replay bozulur",
         "ne": "Canlıda koşan workflow varken kod değiştirilirse replay eski history "
               "ile yeni kodu eşleştiremez.",
         "sonuc": "`workflow.patched()` / Worker Versioning öğrenmek ZORUNLU. "
                  "Temporal'ın gerçek öğrenme maliyeti burada.",
         "kanit": _k("repo", "temporalio — workflow.patched()")},
        {"ad": "senin kodun cluster'da ÇALIŞMAZ",
         "ne": "Cluster yalnız state ve kuyruk tutar. Workflow + activity kodu SENİN "
               "worker'ında koşar, task queue'yu long-poll ederek.",
         "sonuc": "Worker ayakta değilse workflow ilerlemez — cluster ayakta olsa bile.",
         "kanit": _k("kod", "temporal_defs.py", "SDK worker ayrı süreç")},
        {"ad": "activity at-least-once, workflow exactly-once",
         "ne": "Tamamlanan activity bir daha koşmaz. Ama BAŞLAYIP sonuç yazmadan çöken "
               "activity yeniden denenir.",
         "sonuc": "Activity'ler idempotent olmalı. 'Kaldığı yerden' ≠ 'adımın ortasından'.",
         "kanit": _k("olcum", "fetch ×1, process ×2")},
        {"ad": "bayat claim — board'la birlikte kullanınca",
         "ne": "Temporal aynı activity'yi AYNI payload ile tekrar çağırır. Board bir "
               "önceki denemede claim'i temizlediyse elimizdeki `claim_lock` geçersizdir.",
         "sonuc": "Onarılmazsa iş başarıyla koşar ama `complete()` fencing'e takılır, "
                  "sonuç ÇÖPE gider ve task sonsuza dek 'ready' kalır.",
         "kanit": _k("hata", "②", "iki kez başaran düğüm 3. turda failed işaretleniyordu")},
    ],
    "celery_canvas": [
        {"ad": "chord SONUÇ DEPOSU olmadan çalışmaz",
         "ne": "group sonuçlarının callback'e liste olarak geçmesi için bir yerde "
               "birikmesi gerekir. `result_backend` yoksa chord sessizce kurulamaz.",
         "sonuc": "Board'lu koşumuzda backend KAPALI (sonuç board'a yazılıyor) — "
                  "yani canvas için ayrı bir Celery app kurmak gerekti.",
         "kanit": _k("kod", "motor_canvas.py", "backend=f'file://{S}'")},
        {"ad": "`.s` ile `.si` farkı zinciri bozar",
         "ne": "`.s()` önceki halkanın dönüşünü İLK argüman olarak alır; `.si()` "
               "(immutable) almaz. Zincirin İLK halkası `.si` olmalı, yoksa "
               "beklemediği bir argüman gelir.",
         "sonuc": "Karıştırılırsa TypeError — ve hata zincirin ortasında patlar.",
         "kanit": _k("kod", "motor_canvas.py", "canvas_fn.si(None, …) ilk katmanda")},
        {"ad": "veri akışı POZİSYONEL — düğüm kimliği yok",
         "ne": "chain'de gelen değer yalnız 'önceki dönüş'tür; hangi düğümden geldiği "
               "bilgisi taşınmaz. chord'da liste gelir ama sırası dışında kimlik yok.",
         "sonuc": "`{düğüm_id: sonuç}` bekleyen bir fonksiyona sentetik anahtar "
                  "uydurmak gerekti (`_canvas_0`). Graf, fonksiyon imzasına SIZAR.",
         "kanit": _k("kod", "motor_canvas.py", "ust[f'_canvas_{i}'] = r")},
        {"ad": "hata davranışı GRAFIN ŞEKLİNE bağlı",
         "ne": "Batan halka zincirin SONUNDAysa `AsyncResult` hatayı döndürür ve koşu "
               "hızlı biter. ORTADAysa (chord'un içinde) callback hiç tetiklenmez ve "
               "koşu ASILI kalır.",
         "sonuc": "ÖLÇÜLDÜ, aynı hata iki farklı grafta: son düğümde 10,03 sn'de "
                  "RuntimeError döndü · orta düğümde 76,04 sn bekleyip TAKILDI. "
                  "İkisinde de HANGİ halkanın battığı kaydedilmiyor.",
         "kanit": _k("olcum", "10,03 sn (son) vs 76,04 sn (orta)",
                     "aynı sim, farklı graf şekli")},
        {"ad": "link_error kurulmazsa hata SESSİZ",
         "ne": "Zincir batarsa Celery kimseye haber vermez; `link_error` callback'i "
               "elle kurulmalıdır.",
         "sonuc": "Kalıcı hatada 60 sn beklenip 'takıldı' denildi — nerede takıldığı "
                  "hiçbir yere yazılmadı.",
         "kanit": _k("olcum", "✗ 66,04 sn · log: ready()=False, başka bilgi yok")},
        {"ad": "katman senkronizasyonu fazladan bekletir",
         "ne": "`chain(group(K0), group(K1))` i. katmanı ÖNCEKİ TÜM katmanları "
               "bekletir. Çapraz bağımlılıklı grafta düğüm, atası OLMAYAN işi bekler.",
         "sonuc": "Sonuç doğru çıkar ama paralellik kaybolur ve graf yanlış anlatılır.",
         "kanit": _k("kod", "motor_dili.py", "_seri_paralel_mi — beş desende ölçüldü")},
    ],
    "own": [
        {"ad": "fencing — bayat yazma reddi",
         "ne": "`complete`/`fail` çağrısı `claimer` ile eşleşmezse yazma REDDEDİLİR.",
         "sonuc": "Çökme sonrası devralan worker'ın sonucu, eski worker'ınkiyle "
                  "çakışmaz. Fencing olmasaydı bayat sonuç taze olanı ezerdi.",
         "kanit": _k("hata", "⑧", "Celery task dict'i claim ÖNCESİ alıyordu")},
        {"ad": "checkpoint süreçler arası 'bir kez çöktü' işareti",
         "ne": "Çökme simülasyonu her denemede tetiklenirse düğüm sonsuza dek ölür. "
               "Checkpoint'in VARLIĞI süreçler arası geçerli bir tek-atış işareti.",
         "sonuc": "own'da 10 çökme üst üste, celery/temporal'da düğüm asılı kalmıştı.",
         "kanit": _k("hata", "çökme döngüsü", "checkpoint tek-atış olarak kullanıldı")},
        {"ad": "veri YALNIZ doğrudan ebeveynden akar",
         "ne": "`upstream_results` yalnız DOĞRUDAN parents'ı içerir — atalar kapanışını "
               "değil. Planlayıcı kenarı kurmayı unutursa veri sessizce ulaşmaz.",
         "sonuc": "Fonksiyon varsayılana düşer ve 'temiz rapor' üretir — SESSİZ YANLIŞ. "
                  "En tehlikelisi bu: hata vermiyor, yanlış cevap veriyor.",
         "kanit": _k("hata", "sessiz rapor", "NEEDS sözleşmesi + dogrula_dag eklendi")},
        {"ad": "id ~27,7 saatte bir başa sarıyor",
         "ne": "Akış id'si `int(time.time()*1000) % 1e8` ile üretiliyor; sayaç sarınca "
               "yeni akış DOSYA ADINA göre sıralamada dibe düşer.",
         "sonuc": "Sohbette kurulan graf 'Akışlar' ekranında GÖRÜNMÜYORDU. Sıralama "
                  "gerçek zaman damgasına (`at`) çevrildi.",
         "kanit": _k("kod", "pipelines.py:52-60")},
        {"ad": "aynı ebeveyn iki kez yazılırsa kapı kilitlenir",
         "ne": "`parents` listesinde tekrar eden id, bağımlılık kapısının hiç açılmamasına "
               "yol açıyordu.",
         "sonuc": "Düğüm sonsuza dek `blocked`. `create_task` ve `recompute_ready` "
                  "artık tekilleştiriyor.",
         "kanit": _k("hata", "⑪ öncesi", "dedupe eklendi")},
    ],
}



# ── İYİ YANI, SADE DİLLE ────────────────────────────────────────────────────
# Kaynak: report/motorlarin-iyi-yonleri.md — terimsiz, koridorda söylenebilir hâl.
# `guclu` listesi KANITLI ve teknik; bu ise ANLATILABİLİR. İkisi ayrı işe yarıyor:
# biri "kanıtla" der, diğeri "anlat" der.
IYI_YONLER = {
    "celery": {
        "lakap": "en hızlı kurulan",
        "tanim": "dağıtık iş kuyruğu",
        "tek_cumle": "Bir fonksiyonu arka plana atmanın en kolay yolu.",
        "sema": ".delay()  →  mesaj kuyruğa (Redis)\n"
                "                  ↓\n"
                "        boştaki worker kendi çeker → çalıştırır → siler",
        "nasil": [
            "Fonksiyonun KODU gitmez, sadece ADI gider. Kod worker'da zaten var.",
            "Kimse worker seçmez — boşta olan gelip alır.",
            "Retry = aynı mesajı kuyruğa geri koymak → fonksiyon 1. SATIRDAN başlar.",
        ],
        "maddeler": [
            "Yarım günde kurulur — bir Redis, o kadar.",
            "Öğrenmesi kolay: `@app.task` yaz, bitti.",
            "Worker ekleyerek büyür — 10 kat yük, 10 kat worker.",
            "Python'un standardı; Django/Flask hazır çalışır.",
            "Yük patlamasında sistem çökmez, kuyruk tamponlar.",
        ],
        "eksiler": [
            "Çok adımlı iş kavramı yok — her task bölünmez bir kutu.",
            "Retry baştan koşar → pahalı adımı 2 kez ödersin.",
            "'Nerede kaldım' defterini SEN tutarsın.",
            "Aynı iş iki kez çalışabilir → idempotency sende.",
            "Geçmiş/denetim kaydı tutmaz.",
        ],
        "parladigi": "E-posta gönder, resim boyutlandır, rapor üret, tek bir LLM "
                     "çağrısını arka plana at.",
    },
    "airflow": {
        "lakap": "en görünür olan",
        "tanim": "zamanlı iş planlayıcı",
        "tek_cumle": "Zamanlı işlerin takvimi ve panosu.",
        "sema": "DAG yazılır:  fetch → process → deliver\n"
                "        ↓ saati gelince scheduler tetikler\n"
                "adımlar sırayla koşar, durum DB'ye yazılır → UI'da görünür",
        "nasil": [
            "İşin ŞEKLİ ÖNCEDEN çizilir (statik graf).",
            "Biten adım DB'de 'başarılı' kalır.",
            "Bir adım patlarsa SADECE O ADIM tekrar koşar.",
        ],
        "maddeler": [
            "Zamanlamada rakipsiz — cron'u yaz, gerisi onun.",
            "Backfill: 'bugün kurdum, son 6 ayı doldur'. Başka araçta yok.",
            "Arayüzü çok güçlü — hangi adım, ne kadar sürdü, logu ne.",
            "Öncekiler korunur, boşa iş yapılmaz.",
            "Yüzlerce hazır bağlantı (S3, BigQuery, Spark, dbt…).",
        ],
        "eksiler": [
            "Ağır kurulum — 4 bileşen ayakta olmalı.",
            "İşin şekli önceden belli olmalı → ajanın serbest akışı sığmaz.",
            "Adımın İÇİNDE kayıt yok: 7/10'da çökerse baştan.",
            "Uzun beklemeler ('3 gün onay') için uygun değil.",
            "Tek bir işi arka plana atmak için abartı.",
        ],
        "parladigi": "Gecelik veri işleri, günlük raporlar, veri ekibinin sabah "
                     "kontrol ettiği pipeline'lar.",
    },
    "temporal": {
        "lakap": "en dayanıklı olan",
        "tanim": "çökmeye dayanıklı yürütme",
        "tek_cumle": "Makine çökse bile işin kaldığı yerden devam eder.",
        "sema": "Workflow normal kod olarak yazılır\n"
                "        ↓ her adım kalıcı deftere yazılır\n"
                "çökünce kod baştan koşar ama deftere bakıp\n"
                "biten adımları ATLAR → kaldığı yerden devam",
        "nasil": [
            "Defteri tutan taraf SENİN SÜRECİNİN DIŞINDA — o yüzden çökmeyi "
            "kurtarabiliyor.",
            "Yan etkili her adım ayrı bir parça ('activity').",
        ],
        "maddeler": [
            "Tamamlanan adım bir daha koşmaz — pahalı adımı iki kez ödemezsin.",
            "Kurtarma kodu yazmıyorsun — retry/timeout/devam hazır geliyor.",
            "Günlerce bekler. '3 gün insan onayı bekle' normal bir satır.",
            "Dinamik + dayanıklı aynı anda — akış koşarken belirlenebilir.",
            "Her adımın kaydı → geriye sarıp 'tam ne oldu' görürsün.",
        ],
        "eksiler": [
            "Cluster işletmen gerekir (ya da Cloud'a ücret).",
            "Determinizm kuralları: workflow'da rastgele/saat/HTTP yasak.",
            "Öğrenmesi HAFTALAR sürer — kalıcı bir disiplin.",
            "Kod değişince sürüm yönetimi derdi.",
            "Basit zamanlı iş için fazla ağır.",
        ],
        "parladigi": "Ödeme akışları, sipariş süreçleri, çok adımlı ajan işleri — "
                     "yarıda kalması pahalıya patlayan her şey.",
    },
    "own": {
        "lakap": "en hafif olan",
        "tanim": "SQLite üstünde hafif motor",
        "tek_cumle": "Temporal'ın fikri, tek makinede ve birkaç yüz satırda.",
        "sema": "İş = SQLite'ta bir satır\n"
                "   ↓ worker satırı kilitleyerek kapar (tek worker alır)\n"
                "kilidin süresi var → worker çökerse süre dolar\n"
                "başka worker devralır → kayıttan devam eder",
        "nasil": [
            "Temporal'ın fikri, tek makinede ve birkaç yüz satırda.",
        ],
        "maddeler": [
            "Sıfır yeni servis — sadece SQLite.",
            "Kod tamamen bizim — anlamadığımız yer yok.",
            "Ölçtük: pahalı adım 1 KEZ koşuyor (Temporal'la aynı sonuç).",
            "Worker çökerse iş kaybolmuyor, başkası devralıyor.",
        ],
        "eksiler": [
            "Bakım bize ait — hata çıkarsa biz düzeltiriz.",
            "Tek makine ölçeği; çok sunucuda zorlanır.",
            "Kenar durumlar zamanla ortaya çıkar.",
            "Hazır arayüz / ekosistem yok.",
        ],
        "parladigi": "Tek makine, küçük ekip, tam kontrol istenen durumlar.",
    },
    "celery_canvas": {
        "lakap": "en ucuz workflow",
        "tanim": "Celery'nin kendi kompozisyonu",
        "tek_cumle": "Zaten Celery varsa, workflow kurmanın bedeli sıfır.",
        "sema": "chain(a.s(), b.s(), c.s())()\n"
                "        ↓ TÜM zincir tek mesaja serileştirilir\n"
                "worker a'yı koşar → dönüşü b'nin İLK argümanı → …",
        "nasil": [
            "Fonksiyonlar birbirini TANIMAZ — sıra dışarıda, tek satırda.",
            "Zincir `apply_async()` anında DONAR; sonradan değiştirilemez.",
            "Her halka ayrı task → retry de ayrı (Airflow düğümü gibi).",
        ],
        "maddeler": [
            "`chain(a, b, c)` — üç kelimeyle sıralı akış.",
            "`group` ve `chord` ile paralel ve toplama da var.",
            "Yeni servis yok, yeni kavram yok — Celery'yi bilen bunu da bilir.",
            "Her halka ayrı task: adım seviyesinde retry doğal olarak geliyor.",
        ],
        "eksiler": [
            "'Nerede kaldı' sorulamıyor — elde yalnız `ready()` var.",
            "Batan zincirin kalanı SESSİZCE ölüyor (`link_error` kurmazsan).",
            "Veri akışı pozisyonel — düğüm kimliği kayboluyor.",
            "Koşullu dallanma `self.replace()` ister, mantık dağılır.",
            "Döngü yok; uzun bekleme yok; dışarıdan sinyal yok.",
        ],
        "parladigi": "Sabit, kısa, dallanmasız boru hattı — 3-5 adım.",
    },
}

# ── MENTALİTELER — piyasa neden kullanıyor ─────────────────────────────────
# Her araç bir İNANÇTAN doğdu. Yaygınlığı, o inancın kaç şirketin derdine
# denk düştüğüdür. Kaynak: report/motorlarin-iyi-yonleri.md
MENTALITE = {
    "giris": "Her araç bir İNANÇTAN doğdu. Piyasada neden yaygın olduğu, o inancın "
             "kaç şirketin derdine denk düştüğüdür.",
    "kayitlar": {
        "celery": {
            "donem": "Python dünyası, 2009",
            "inanc": "Kullanıcıyı bekletme. İşi birine devret, sen hemen cevap dön.",
            "dogus": "Web uygulamaları büyüdükçe aynı dert çıktı: kullanıcı butona "
                     "basıyor, sunucu e-posta gönderirken 10 saniye kilitleniyor. "
                     "Celery bu TEK derde çözüm olarak doğdu — ve o kadarla kaldı. "
                     "Sadeliği gücü.",
            "kim": "Python/Django ile web ürünü yazan hemen herkes. Ölçek örneği: "
                   "Instagram'ın bilinen en büyük Celery+RabbitMQ kurulumlarından biri var.",
            "ne_icin": "E-posta/bildirim, resim–video işleme, rapor üretme, ödeme "
                       "sonrası işler, tek seferlik LLM çağrısı.",
            "neden_yaygin": [
                "Dert EVRENSEL — her web uygulamasının arka plan işi var.",
                "Girişi bedava; mevcut Redis'in üstüne kurulur.",
                "Django/Flask ile fiilen standart; her ekipte bilen biri var.",
            ],
        },
        "airflow": {
            "donem": "Airbnb → Apache, 2014",
            "inanc": "Veri işleri takvime bağlıdır ve insan gözüyle izlenmelidir.",
            "dogus": "Airbnb'de veri ekibi her gece onlarca rapor/tablo üretiyordu ve "
                     "cron script'leri YÖNETİLEMEZ hâle gelmişti: hangisi patladı, "
                     "hangisi hangisini bekliyor, dünkü eksik nasıl doldurulur? "
                     "Airflow tam bu kaosu GÖRÜNÜR kılmak için yazıldı.",
            "kim": "Veri mühendisliği ekipleri — fiilen sektör standardı. Bulut "
                   "sağlayıcılar yönetilen sürümünü satıyor (AWS MWAA, Google Cloud "
                   "Composer, Astronomer).",
            "ne_icin": "Gecelik ETL, veri ambarı beslemesi, dbt/Spark zincirleri, "
                       "günlük iş zekâsı raporları, ML eğitim pipeline'ları.",
            "neden_yaygin": [
                "Veri ekibinin DİLİ — 'DAG' kelimesi bu yüzden yerleşti.",
                "Backfill tek başına satın alma sebebi.",
                "Arayüz, teknik olmayan paydaşa da gösterilebiliyor.",
            ],
        },
        "temporal": {
            "donem": "Uber/Cadence → 2019",
            "inanc": "'Ya yarıda kalırsa?' sorusu geliştiricinin derdi olmamalı.",
            "dogus": "Uber'de bir yolculuk/ödeme akışı onlarca servise dağılıyordu. "
                     "Her ekip aynı şeyi tekrar yazıyordu: durum tablosu, retry "
                     "sayacı, 'çökerse nereden devam' kodu. Cadence bunu BİR KEZ ve "
                     "DOĞRU çözmek için yazıldı; ekibi ayrılıp Temporal'ı kurdu.",
            "kim": "Para/işlem akışı kritik olan şirketler. Kamuya açık konuşan "
                   "kullanıcılar arasında Netflix, Snap, Coinbase, Datadog, Box var.",
            "ne_icin": "Ödeme ve sipariş akışları, kullanıcı onboarding/KYC, altyapı "
                       "sağlama (provisioning), ajan iş akışları.",
            "neden_yaygin": [
                "Mikroservis dağıldıkça 'yarıda kalma' EN PAHALI HATA hâline geldi.",
                "Yazılmayan kod: retry/state/kurtarma katmanı komple gidiyor.",
                "LLM ajanları uzun ve pahalı adımlar ürettikçe talep arttı.",
            ],
        },
        "own": {
            "donem": "Hermes/OpenClaw deseni",
            "inanc": "Garantinin %80'i, maliyetin %5'iyle — ve kod tamamen bizde.",
            "dogus": "Ajan araçları tek makinede koşuyor ve küçük ekipler yönetiyor. "
                     "Temporal cluster'ı işletecek kişi yok; ama 'worker çökerse iş "
                     "kaybolmasın' yine de şart. Çözüm: SQLite üstünde aynı fikrin "
                     "küçük hâli.",
            "kim": "Kaynak kodunu incelediğimiz ajanlar — Hermes ve OpenClaw tam bu "
                   "deseni kuruyor. Shannon ise tersini seçip doğrudan Temporal'a biniyor.",
            "ne_icin": "Ajan görev yönetimi — sıraya alma, deneme sayacı, çökme "
                       "sonrası devralma, kaldığı yerden devam.",
            "neden_yaygin": [
                "Yeni servis eklemeden BUGÜN çalışıyor.",
                "Ölçtük: pahalı adım 1× — Temporal'la aynı sonuç.",
                "İleride büyürse Temporal'a geçiş yolu açık kalıyor.",
            ],
        },
    },
    "tek_satir": [
        ("Celery", "işi BAŞKASINA ver", "iş gücü"),
        ("Airflow", "işi TAKVİME bağla", "zaman + görünürlük"),
        ("Temporal", "işi ÖLÜMSÜZ yap", "güvenilirlik"),
        ("Kendi çekirdeğimiz", "işi KENDİ DEFTERİMİZE yaz", "maliyet + kontrol"),
    ],
    "piyasa": {
        "baslik": "Piyasa nereye gidiyor",
        "adimlar": [
            "Uzun yıllar Celery + Airflow ikilisi yetti: kısa arka plan işleri + "
            "gecelik veri işleri.",
            "Sonra mikroservisler dağıldı, işler UZADI ve PAHALILAŞTI → 'yarıda "
            "kalırsa' maliyeti arttı, durable execution (Temporal) yükseldi.",
            "Şimdi LLM ajanları aynı baskıyı katlıyor: her adım para, her akış "
            "dinamik. Bu yüzden yeni ajan altyapıları ya Temporal'a biniyor ya da "
            "kendi küçük durable çekirdeğini yazıyor — TAM BİZİM DURDUĞUMUZ YER.",
        ],
        "kaynak": "report/task-yonetimi-altyapi-karari.md — yedi ajanın kaynak kodu incelendi",
    },
}

TEK_BAKISTA = {
    "satirlar": [("Celery", "Hızlı kurulur, kolay ölçeklenir", "Hafızası yok", "2×"),
                 ("Airflow", "Zamanlama + görünürlük", "Şekil önceden sabit", "1×"),
                 ("Temporal", "Çökmeye dayanıklılık", "Ağır kurulum + disiplin", "1×"),
                 ("Kendi çekirdeğimiz", "Hafiflik + tam kontrol", "Bakım bizde", "1×")],
    "not": "Bunlar rakip değil. Airflow zaten Celery'yi kendi işçi havuzu olarak "
           "kullanabiliyor. Çoğu gerçek sistem birden fazlasını birlikte çalıştırır.",
    "cerceve": "Soru 'hangisi daha iyi' değil — 'benim işim hangisine benziyor'.",
    "tek_soru": "İş yarıda çökerse 'nerede kalmıştım' defterini KİM tutuyor? "
                "Celery'de HİÇ KİMSE, Airflow'da SİSTEM ama sadece adımlar arasında, "
                "Temporal'da TAMAMEN SİSTEM.",
}


# ── ÖNE ÇIKAN ÖZELLİK — motorun bayrağı ────────────────────────────────────
# Her motorun "bunun için varım" dediği tek şey. `test` alanı ne yapılabileceğini
# söylüyor:
#   "deney"  → panelde koşulabilir bir deneyle kanıtlanıyor
#   "goster" → koşulamaz (bu POC'ta kapalı) ama ÜRETİLEN KOD/AYARDA görünüyor
#   "anlat"  → ne koşulur ne gösterilir; yalnız anlatılır (dürüstlük gereği ayrı)
ONE_CIKAN = {
    "celery": {
        "ad": "Broker + worker havuzu — yatay ölçek",
        "ozet": "Celery'nin bayrağı kuyruk. İş üretimi ile tüketimi ayrılıyor; "
                "worker ekleyerek kapasite büyütüyorsun. Ölçek buradan geliyor.",
        "test": "deney", "nerede": "Açılış bedeli deneyi — 13 sn'nin ~6'sı worker açılışı",
        "sinir": "Bu POC tek worker koşuyor; yatay ölçek ÖLÇÜLMEDİ, mekanizma gösteriliyor.",
    },
    "airflow": {
        "ad": "Scheduler + catchup/backfill — zamanın kendisi",
        "ozet": "Airflow'un bayrağı SCHEDULER. Cron yazarsın, tetiklemeyi o yapar. "
                "Asıl ayırt edici yanı `catchup`: sistemi bugün kurup 'start_date'ten "
                "beri her gün koşmuş olsun' dersen kaçırılan koşuları GERİYE DÖNÜK üretir. "
                "Başka hiçbir motorda bu yok.",
        "test": "goster",
        "nerede": "Üretilen DAG dosyasında `schedule=\'0 8 * * *\'` ve `catchup` satırı — "
                  "'bu graf, Airflow dilinde' bölümünde görünüyor",
        "sinir": "Bu POC scheduler'ı ÇALIŞTIRMIYOR — `airflow dags test` ile doğrudan "
                 "tetikliyoruz. Cron satırı üretiliyor ama saati bekleyen bir süreç yok.",
    },
    "temporal": {
        "ad": "Event history + replay — kaldığı yerden devam",
        "ozet": "Temporal'ın bayrağı olay geçmişi. Her adım kalıcı loga yazılıyor; "
                "çökünce kod baştan koşuyor ama tamamlanmış adımlar ATLANIYOR. "
                "Kurtarma kodunu sen yazmıyorsun.",
        "test": "deney", "nerede": "Çökme sonrası devam deneyi — hedef düğüm koşum=2, "
                                   "graf 6/6 tamamlandı",
        "sinir": "Uzun bekleme (`sleep(30 gün)`) ve Signal/Query bu POC'ta kullanılmıyor.",
    },
    "celery_canvas": {
        "ad": "chain / group / chord — sıfır maliyetli kompozisyon",
        "ozet": "Canvas'ın bayrağı ucuzluk. Zaten ayakta olan Celery'ye üç kelime "
                "ekleyerek workflow kuruyorsun; ne yeni servis ne yeni kavram.",
        "test": "deney", "nerede": "'Canvas'ı koştur' düğmesi — board YOK, "
                                   "aynı graf 9,03 sn'de doğru sonucu üretti",
        "sinir": "Koşarken durum SORULAMIYOR; kalıcı hatada 60 sn bekleyip takılıyor "
                 "ve nerede takıldığı bilinmiyor.",
    },
    "own": {
        "ad": "Task tablosu — kararın tek noktada olması",
        "ozet": "Bizim bayrağımız defterin bizde olması. Retry, iptal zinciri, breaker "
                "ve düğüm içi checkpoint hep board'da; motor yalnız dağıtıcı. Bu yüzden "
                "motor değiştirmek davranışı DEĞİŞTİRMİYOR.",
        "test": "deney", "nerede": "İptal zinciri deneyi — dört motorda da aynı sütunlar, "
                                   "yalnız süre ayrışıyor",
        "sinir": "Tek makine, tek süreç. Dağıtık ölçek YOK.",
    },
}

# ── BİRLİKTE KULLANIM — hangi yığın, kim ne yapar ──────────────────────────
BIRLIKTE = [
    {"yigin": "Airflow + Celery (CeleryExecutor)",
     "kim_ne": [("Airflow", "NE ZAMAN ve HANGİ SIRA — scheduler + metadata DB"),
                ("Celery", "KİM KOŞTURACAK — broker + worker havuzu")],
     "nicin": "Airflow tek makinede yetmediğinde task'ları dağıtmak için. Airflow'un "
              "en yaygın production kurulumu bu.",
     "not": "Bu yüzden 'Airflow mu Celery mi' çoğu zaman yanlış soru — biri diğerini "
            "işçi havuzu olarak kullanıyor.",
     "bizde": False, "motorlar": ["airflow", "celery"]},
    {"yigin": "Board (own) + herhangi bir motor",
     "kim_ne": [("Board", "DEFTER — durum, retry, iptal zinciri, checkpoint"),
                ("Motor", "DAĞITIM — task'ı bir worker'a ulaştırmak")],
     "nicin": "Motorun vermediği şeyi (cancelled zinciri, düğüm içi checkpoint, tek "
              "deneme sayacı) eklemek; motoru değiştirilebilir bırakmak.",
     "not": "Bu POC'ta kullanılan yığın. Bedeli: dört motorun modüllerinin bir kısmı "
            "devre dışı kalıyor (Celery'de 10'dan 6'sı).",
     "bizde": True, "motorlar": ["own", "celery", "airflow", "temporal"]},
    {"yigin": "Airflow + Temporal",
     "kim_ne": [("Airflow", "ZAMAN — gecelik tetikleme, backfill"),
                ("Temporal", "DAYANIKLILIK — uzun süren, duraklayan işin kendisi")],
     "nicin": "Zamanlı tetiklenen ama saatlerce/günlerce sürebilen işler. Airflow "
              "workflow'u başlatır, Temporal onu bitirir.",
     "not": "Airflow'un uzun bekleme için tasarlanmamış olması bu ayrımı doğuruyor.",
     "bizde": False, "motorlar": ["airflow", "temporal"]},
    {"yigin": "Celery + kendi durum tablon",
     "kim_ne": [("Celery", "KUYRUK — at-least-once teslim, worker havuzu"),
                ("Senin DB'n", "DEFTER — 'hangi adımdayım' bilgisi")],
     "nicin": "Celery çok adımlı iş kavramı bilmediği için, çok adımlı bir süreç "
              "kurulacaksa defteri zaten sen tutmak zorundasın.",
     "not": "Bu, board yaklaşımının Celery'ye özel hâli — bizim yaptığımız da bu.",
     "bizde": True, "motorlar": ["celery", "own"]},
]

# ── GERÇEK DÜNYA — hangi iş tipinde hangisi ────────────────────────────────
# DİKKAT: bu bölüm GENEL BİLGİ, bu POC'ta ölçülmedi. Ayrı tutuluyor ki
# ölçülmüş sayılarla karışmasın.
GERCEK_DUNYA = {
    "_uyari": "Bu bölüm genel sektör bilgisi — bu POC'ta ÖLÇÜLMEDİ. Ölçülmüş "
              "sayılar 'avantaj/dezavantaj' ve deneylerdedir.",
    "celery": {
        "dogdugu_yer": "Python/Django ekosisteminin arka plan iş standardı",
        "tipik_isler": ["e-posta ve bildirim gönderimi", "resim/video boyutlandırma",
                        "rapor üretimi", "webhook işleme",
                        "tek bir LLM çağrısını arka plana atma"],
        "desen": "Kısa, bağımsız, çok sayıda iş. Kullanıcı isteğini bloklamamak için.",
    },
    "airflow": {
        "dogdugu_yer": "Airbnb'de veri hattı orkestrasyonu için doğdu; veri "
                       "mühendisliğinin fiilî standardı",
        "tipik_isler": ["gecelik ETL", "veri ambarı yükleme", "dbt/Spark orkestrasyonu",
                        "günlük/haftalık raporlama", "geriye dönük veri doldurma"],
        "desen": "Zamanlı, tekrarlayan, insan-gözetimli veri hatları.",
    },
    "temporal": {
        "dogdugu_yer": "Uber'in Cadence'inden doğdu; ödeme ve sipariş akışlarında yaygın",
        "tipik_isler": ["ödeme akışları", "sipariş/saga süreçleri",
                        "onboarding (günlerce süren)", "mikroservis orkestrasyonu",
                        "ajan workflow'ları (planner→search→writer→verifier)"],
        "desen": "Uzun süren, duraklayan, yarıda kalması pahalıya patlayan işler.",
    },
    "celery_canvas": {
        "dogdugu_yer": "Celery'nin kendi kompozisyon ilkelleri (chain/group/chord)",
        "tipik_isler": ["sabit adımlı ETL boru hattı", "fan-out + toplama "
                        "(30 dosyayı işle, sonra özetle)", "sıralı bildirim zinciri"],
        "desen": "Şekli ÖNCEDEN belli, dallanmasız, kısa akışlar.",
    },
    "own": {
        "dogdugu_yer": "Hermes-Agent deseni — ajan motorunun kendi kalıcı task tablosu",
        "tipik_isler": ["ajanın kurduğu çok adımlı iş", "tek makinede durable kuyruk",
                        "mevcut engine'e recovery eklemek"],
        "desen": "Oturum/state zaten sende; eksik olan yalnız durable kuyruk + kurtarma.",
    },
}



# ── İFADE GÜCÜ — Canvas / Airflow / Temporal ───────────────────────────────
# DÜZELTME NOTU: Celery'nin sınırı uzun süre "retry baştan koşar" diye anlatıldı.
# O örnek TEK-TASK kurgusuna aitti (adımlar bir fonksiyonun içinde). Kullanıcı
# baştan beri `chain`'den bahsediyordu ve canvas'ta zincirin her halkası AYRI
# task'tır — retry de ayrıdır, tıpkı Airflow düğümü gibi. Yani o argüman yanlış
# yerdeydi. Canvas'ın GERÇEK sınırı aşağıdaki tabloda: dallanma, döngü,
# "nerede kaldı" sorgusu, uzun bekleme.
IFADE_GUCU = {
    "senaryo": "veri çek → işle → teslim et; ayrıca 30 dosyayı paralel işle",
    "ornekler": {
        "canvas": "chain(fetch.s(\"4711\"), process.s(), deliver.s())()\n"
                  "chord(group(isle.s(f) for f in dosyalar), ozetle.s())()",
        "airflow": "t1 >> t2 >> t3\n"
                   "isle.expand(dosya=dosyalar)      # runtime'da N paralel düğüm",
        "temporal": "a = await workflow.execute_activity(fetch, oid)\n"
                    "b = await workflow.execute_activity(process, a)\n"
                    "await asyncio.gather(*[workflow.execute_activity(isle, f)\n"
                    "                       for f in dosyalar])",
    },
    # 5. eleman: bu satır PANELDEN koşturulabiliyor mu, koşturuluyorsa nasıl.
    # Ekip "bu iddiayı görebilir miyim" diye sorunca cevabı ekranda olsun.
    "satirlar": [
        ("Akış nerede yazılı", "yayın anında mesaja serileştirilir",
         "DAG dosyasında, koşmadan ÖNCE", "kodda, KOŞARKEN oluşur",
         "üretilen kod — 'bu graf X dilinde' bölümü"),
        ("Sıralı zincir", "✓ chain", "✓ >>", "✓ normal kod",
         "canvas + board koşuyor"),
        ("Paralel", "✓ group", "✓ paralel düğüm", "✓ asyncio.gather",
         "canvas group(2) koştu"),
        ("Paralel sonra toplama", "✓ chord (sonuç deposu ŞART)", "✓", "✓",
         "chord koştu — 4 katman"),
        ("Koşula bağlı dallanma", "⚠ self.replace() — mantık task'lara DAĞILIR",
         "⚠ BranchPythonOperator — dallar ÖNCEDEN tanımlı", "✓ düpedüz if", ""),
        ("Döngü (while)", "✗ özyinelemeyle zorlanır", "✗", "✓", ""),
        ("Adım seviyesi retry", "✓", "✓", "✓", "geçici hata deneyi"),
        ("Biten adım korunur", "✓", "✓", "✓", "geçici hata deneyi — öncekiler ×1"),
        ("Uzun bekleme (gün/hafta)", "✗",
         "⚠ sensor/deferrable — saatler tamam, günler zorlama", "✓ birinci sınıf", ""),
        ("Dışarıdan sinyal", "✗", "⚠ harici tetikleme", "✓ Signal / Update", ""),
        ("'Nerede kaldı?' sorgusu", "✗ hiçbir yerde", "✓ DB + arayüz",
         "✓ history + sorgu", "canvas kalıcı hata düğmesi"),
        ("Kalıcı hata görünür mü", "✗ sessizce ölür (link_error kurarsan)",
         "✓ kırmızı + uyarı", "✓ olay + uyarı",
         "airflow upstream_failed deneyi"),
        ("Ara sonuç nerede durur", "⚠ broker mesajında", "⚠ XCom (küçük veri)",
         "kalıcı defterde", ""),
        ("Zamanlı tetik", "⚠ Beat (basit)", "✓ en güçlü + backfill", "✓ Schedules", ""),
        ("Kurulum yükü", "çok hafif", "ağır", "ağır", ""),
    ],
    "kim_nerede_kazanir": [
        ("Canvas", "KURULUM. Zaten Celery varsa `chain` yazmanın maliyeti SIFIR. "
                   "Sabit, kısa, dallanmasız bir boru hattı için Temporal ya da "
                   "Airflow kurmak abartı. 3-5 adımlık sıralı iş + biraz paralellik "
                   "= Canvas yeter."),
        ("Airflow", "İKİ ŞEY: backfill ('6 ayı geriye doldur') ve operatör "
                    "görünürlüğü. Canvas'ta yok, Temporal'da da bu kadar güçlü değil. "
                    "Zamanlı veri işi varsa Airflow."),
        ("Temporal", "ÜÇ ŞEY: serbest dallanma/döngü, günlerce bekleme, her adımın "
                     "kalıcı kaydı. Akış runtime'da şekilleniyorsa Canvas ve Airflow "
                     "ikisi de yetmez."),
    ],
    "kapsam": "15 boyutun 8'i panelden KOŞTURULABİLİYOR (▶ işaretli). Kalan 7'si "
              "yalnız metin — bu POC koşullu dal, döngü, uzun bekleme, sinyal, "
              "zamanlı tetik ve kurulum yükünü ÖLÇMÜYOR.",
    "canvas_sinir": {
        "baslik": "Canvas'ın gerçek sınırı: dallanma",
        "kod": "@app.task(bind=True)\n"
               "def process(self, veri):\n"
               "    if veri['tutar'] > 10000:\n"
               "        return self.replace(chain(manuel_onay.s(veri), deliver.s()))\n"
               "    return self.replace(chain(otomatik_onay.s(veri), deliver.s()))",
        "calisir_ama": [
            "Akışın mantığı task'ların İÇİNE dağıldı — 'bu iş ne yapıyor' diye "
            "bakacağın TEK bir yer yok",
            "Her dal için ayrı task + ayrı `replace` çağrısı",
            "Döngü istersen özyineleme kurman gerekir",
            "Ve hâlâ dışarıdan 'şu an neredeyiz' diye SORAMIYORSUN",
        ],
        "ders": "Temporal'da bu üç satırlık `if`. Fark YAPILABİLİR Mİ değil, "
                "OKUNABİLİR ve İZLENEBİLİR mi.",
    },
}

# ── KARAR GEREKÇESİ — düzeltilmiş hâli ─────────────────────────────────────
KARAR_GEREKCESI = {
    "yanlis": "Celery'nin retry'ı task'ı baştan koşturuyor.",
    "neden_yanlis": "Bu, adımlar TEK task'ın içindeyken doğru. Canvas'ta (`chain`) "
                    "her halka ayrı task'tır ve ayrı retry alır — tıpkı Airflow "
                    "düğümü gibi. Yani bu argüman yanlış yerde duruyordu.",
    "dogru": "Akış RUNTIME'DA şekilleniyor ve dışarıdan İZLENEBİLİR olması gerekiyor.",
    "neden": [
        "Ajanın akışında model her turda çıktıya bakıp bir sonraki adımı seçiyor; "
        "kaç adım olacağı önceden belli değil, bazen yeni alt-işler doğuyor.",
        "Canvas: `self.replace()` zinciriyle zorlanır ama mantık dağılır, izlenemez.",
        "Airflow: statik DAG'a hiç sığmaz.",
        "Temporal ya da kendi çekirdeğimiz: doğal.",
    ],
}



# ── ÜÇ DESEN — zinciri KİM kuruyor ─────────────────────────────────────────
# "Canvas'ın düz Celery'den farkı ne" sorusunun cevabı. Üçü de aynı kütüphaneyi
# kullanıyor; değişen tek şey sıradaki adımı kimin tetiklediği.
UC_DESEN = {
    "soru": "Zinciri kim kuruyor?",
    "desenler": [
        {"ad": "① Düz Celery", "kuran": "FONKSİYONLAR",
         "kod": "@app.task\n"
                "def fetch(oid):\n"
                "    veri = ...\n"
                "    process.delay(veri)      # ← SEN elle çağırıyorsun\n"
                "    return veri\n\n"
                "fetch.delay('4711')",
         "sorun": "`fetch` fonksiyonu `process`'i TANIMAK zorunda — import ediyor, "
                  "adını biliyor. Sıra fonksiyonların İÇİNE gömülü; sırayı "
                  "değiştirmek fonksiyonları değiştirmek demek.",
         "gonderim": "adım sayısı kadar"},
        {"ad": "② Canvas", "kuran": "MESAJ",
         "kod": "@app.task\n"
                "def fetch(oid):    return ...   # kimseyi tanımıyor\n"
                "@app.task\n"
                "def process(veri): return ...   # kimseyi tanımıyor\n\n"
                "chain(fetch.s('4711'), process.s(), deliver.s())()",
         "sorun": "Fonksiyonlar bağımsız, sıra tek satırda — büyük kazanç. AMA zincir "
                  "`apply_async()` anında mesaja serileştirilip DONUYOR; sonradan "
                  "değiştirilemez.",
         "gonderim": "BİR TANE — zincirin tamamı tek mesajda"},
        {"ad": "③ Board (bizim)", "kuran": "BOARD",
         "kod": "for tur in range(12):\n"
                "    board.recompute_ready()          # kapıyı board açar\n"
                "    for t in board.list_tasks('ready'):\n"
                "        run_task.delay(t['id'])      # yalnız KİMLİK gider\n"
                "    # bekle, tekrar sor",
         "sorun": "Görünürlük ve dinamik graf var. Bedeli: dalga döngüsü (her turda "
                  "'kim hazır' sorusu) ve bu katmanı BİZ yazdık — 12 hata çıktı.",
         "gonderim": "adım sayısı kadar (ama veri değil, KİMLİK)"},
    ],
    "matris": [
        ("Fonksiyonlar birbirinden bağımsız", "✗", "✓", "✓"),
        ("Akış tek yerde okunabilir", "✗", "✓", "✓"),
        ("Koşarken 'neredeyiz' sorulabilir", "✗", "✗", "✓"),
        ("Çalışma anında düğüm eklenebilir", "✗", "✗", "✓"),
        ("Batan işin ardılı kaydedilir", "✗", "✗", "✓"),
    ],
    "olcum": "6 düğümlük graf: Canvas 1× apply_async() · board 6× run_task.delay()",
    "ders": "Canvas, Celery'ye WORKFLOW YAZMA KOLAYLIĞI ekliyor ama WORKFLOW "
            "GÖRÜNÜRLÜĞÜ eklemiyor. Defter hâlâ yok.",
}

# ── GEÇİŞ ANLARI — Canvas ne zaman yetmez ──────────────────────────────────
# "Canvas varken neden Temporal/Airflow" sorusunun cevabı, ölçümle.
GECIS_ANLARI = {
    "yeter": {
        "baslik": "Canvas YETER",
        "kosul": "Şekli önceden belli, kısa (3-5 adım), dallanmasız, kimsenin "
                 "'neredeyiz' diye sormadığı boru hattı.",
        "neden": "Zaten Celery varsa maliyeti SIFIR. Bu gerçek bir avantaj — "
                 "3 adımlık iş için Temporal ya da Airflow kurmak abartı.",
        "kanit": _k("olcum", "6 düğüm 9,03 sn'de doğru CSV üretti"),
    },
    "gecisler": [
        {"nereye": "AIRFLOW", "tetik": "Zamanlama girdiğinde",
         "canvas_ne_yapamaz": [
             "Cron YOK — Canvas bir tetikleme mekanizması değil, kompozisyon aracı",
             "Backfill YOK — 'sistemi bugün kurdum, son 6 ayı doldur' diyemezsin",
             "Celery Beat ayrı servis ve tek instance çalışmalı, yoksa her cron "
             "ÇİFT tetiklenir",
             "Tek bir batan adımı yeniden koşturamazsın — zinciri baştan kurarsın",
         ],
         "olculdu": "AYNI düğüme kalıcı hata — Canvas'ın davranışı GRAF ŞEKLİNE bağlı:\n"
                    "  Canvas · hedef ORTA düğüm  ✗ 76,04 sn — TAKILDI, nerede takıldığı BİLİNMİYOR\n"
                    "  Canvas · hedef SON düğüm   ✗ 10,03 sn — RuntimeError döndü, ama\n"
                    "                                          HANGİ halka olduğu yazmıyor\n"
                    "  Airflow                    ✗  4,77 sn — validate:FAILED(try=3) · ardıllar\n"
                    "                                          UPSTREAM_FAILED (kendi tablosunda)",
         "kilit": "Canvas'ta batan zincirin kalanı SESSİZCE ölüyor (`link_error` "
                  "kurmazsan kimse haber vermiyor). Airflow her düğümün ne olduğunu "
                  "kendi tablosuna yazıyor. Sabah gelen operatör için fark bu.",
         "kanit": _k("olcum", "canvas 66,04 sn kayıtsız / airflow upstream_failed kayıtlı")},
        {"nereye": "TEMPORAL", "tetik": "Akış koşarken şekilleniyorsa, günlerce "
                                        "bekliyorsa, ya da yarıda kalması pahalıysa",
         "canvas_ne_yapamaz": [
             "Koşullu dallanma `self.replace()` ister — akışın mantığı task'ların "
             "İÇİNE dağılır, bakılacak tek yer kalmaz",
             "Döngü (`while`) YOK — özyinelemeyle zorlanır",
             "Uzun bekleme YOK — '3 gün insan onayı bekle' yazılamaz",
             "Dışarıdan sinyal YOK — iptal/onay mesajı gönderilemez",
             "Çökme sonrası devam YOK — zincir kaybolur",
         ],
         "olculdu": "Çökme senaryosu, AYNI graf:\n"
                    "  Canvas   zincir kaybolur — kalan sessizce ölür\n"
                    "  Temporal ✓ 0,47 sn — 6/6 TAMAM, hedef düğüm koşum=2\n"
                    "                       (kaldığı yerden devam etti)",
         "kilit": "Canvas'ta `if` yazmak MÜMKÜN ama okunabilir değil. Temporal'da "
                  "üç satırlık `if`. Fark YAPILABİLİR Mİ değil, OKUNABİLİR ve "
                  "İZLENEBİLİR mi.",
         "kanit": _k("olcum", "temporal çökmede 6/6 · canvas zincir kaybolur")},
    ],
    "bizim_is": {
        "baslik": "Bizim ajan işi hangi kategoride",
        "metin": "Model her turda çıktıya bakıp bir sonraki adımı seçiyor; kaç adım "
                 "olacağı önceden belli değil, bazen yeni alt-işler doğuyor.",
        "sonuc": "Canvas'ta zincir `apply_async()` anında DONUYOR — o yüzden bizim iş "
                 "için Canvas BAŞTAN eleniyor. Airflow'un statik DAG'ı da aynı "
                 "sebeple eleniyor.",
        "vurgu": "Canvas'ı eleyen şey RETRY DEĞİL — grafın yayın anında donması.",
        "kanit": _k("hata", "düzeltme", "eski gerekçe 'retry baştan koşar' yanlıştı"),
    },
}


DURUM_ETIKET = {
    "acik":    {"im": "✓", "ad": "devrede",
                "aciklama": "motorun kendi işini yapıyor"},
    "kapali":  {"im": "⊘", "ad": "devre dışı",
                "aciklama": "board o işi devraldı ya da bu POC'ta kurulmadı"},
    "esgudum": {"im": "⚠", "ad": "eşgüdüm gerekti",
                "aciklama": "devrede AMA board'la çakıştı, uzlaştırmak gerekti"},
}


def kunye(motor: str) -> dict:
    """Tek motorun künyesi + çapa satırı."""
    if motor not in KUNYE:
        raise KeyError(f"bilinmeyen motor: {motor} (geçerli: {', '.join(MOTORLAR)})")
    d = dict(KUNYE[motor])
    # `ad` KUNYE kayıtlarında yok — arayüz `m.ad` ile sekme kimliği kontrolü
    # yapıyor (canvas düğmesi, cluster kutusu, üretilen kod eşlemesi). Bu alan
    # eksikken o koşullar HEP false oluyordu ve ilgili bloklar SESSİZCE hiç
    # çizilmiyordu — hata vermeden kaybolan arayüz parçası.
    d["ad"] = motor
    d["capa"] = CAPA["satirlar"][motor]
    d["incelikler"] = INCELIKLER.get(motor, [])
    d["iyi_yonu"] = IYI_YONLER.get(motor, {})
    d["mentalite"] = MENTALITE["kayitlar"].get(motor, {})
    d["one_cikan"] = ONE_CIKAN.get(motor, {})
    d["gercek_dunya"] = GERCEK_DUNYA.get(motor, {})
    d["birlikte"] = [b for b in BIRLIKTE if motor in b["motorlar"]]
    d["iliskiler"] = [i for i in ILISKILER if motor in i["cift"]]
    d["hafiza"] = next(b for b in HAFIZA_MERDIVENI["basamaklar"] if b["motor"] == motor)
    sayim = {"acik": 0, "kapali": 0, "esgudum": 0}
    for m in d["moduller"]:
        sayim[m["durum"]] += 1
    d["modul_sayim"] = sayim
    return d


def hepsi() -> dict:
    return {
        "katman_notu": KATMAN_NOTU,
        "terim_tuzagi": TERIM_TUZAGI,
        "hafiza_merdiveni": HAFIZA_MERDIVENI,
        "tek_bakista": TEK_BAKISTA,
        "mentalite": MENTALITE,
        "ifade_gucu": IFADE_GUCU,
        "uc_desen": UC_DESEN,
        "gecis_anlari": GECIS_ANLARI,
        "karar_gerekcesi": KARAR_GEREKCESI,
        "birlikte": BIRLIKTE,
        "gercek_dunya_uyari": GERCEK_DUNYA["_uyari"],
        "iliskiler": ILISKILER,
        "kavramlar": KAVRAMLAR,
        "capa": CAPA,
        "durum_etiket": DURUM_ETIKET,
        "motorlar": [kunye(m) for m in MOTORLAR],
    }


if __name__ == "__main__":
    import sys
    hedef = sys.argv[1] if len(sys.argv) > 1 else None
    v = {"motorlar": [kunye(hedef)]} if hedef else hepsi()
    if not hedef:
        print(KATMAN_NOTU, "\n")
        print("─" * 78)
        print(" HAFIZA MERDİVENİ —", HAFIZA_MERDIVENI["ders"])
        print("─" * 78)
        for b_ in HAFIZA_MERDIVENI["basamaklar"]:
            print(f"  {b_['motor']:<9} {b_['agirlik']:<11} {b_['disarida'][:34]:<36} {b_['bedeli']}")
        print("\n" + "─" * 78)
        print(" İLİŞKİLER — bunlar rakip değil")
        print("─" * 78)
        for i in ILISKILER:
            print(f"\n  {i['cift'][0]} ↔ {i['cift'][1]} : {i['baslik']}")
            print(f"     {i['kilit'][:150]}")
            print(f"     ⚠ {i['yanlis_soru'][:120]}")
        print("\n" + "─" * 78)
        kv = KAVRAMLAR["cluster"]
        print(f" KAVRAM — {kv['soru']}")
        print("─" * 78)
        print(f"  {kv['tanim']}")
        print(f"\n  NEDEN: {kv['neden']}")
        print(f"\n  ⚠ {kv['yanlis_anlama']}")
        print(f"\n  → {kv['ozet']}")
    for m in v["motorlar"]:
        s = m["modul_sayim"]
        print(f"\n{'═'*78}\n {m['baslik']}  ·  {m['katman']}  ·  soyutlama: {m['soyutlama']}")
        print(f"{'═'*78}")
        print(f" « {m['tek_cumle']} »\n")
        print(f" ANALOJİ — {m['analoji']['baslik']}\n   {m['analoji']['metin']}\n")
        print(m["mimari"], "\n")
        print(f" MODÜLLER  ({s['acik']} devrede · {s['kapali']} devre dışı · "
              f"{s['esgudum']} eşgüdüm)")
        for md in m["moduller"]:
            print(f"   {DURUM_ETIKET[md['durum']]['im']} {md['ad']:<26} {md['ne'][:44]}")
            if md["durum"] != "acik":
                print(f"       ↳ {md.get('neden','')[:70]}")
        print(f"\n GÜÇLÜ")
        for g in m["guclu"]:
            print(f"   + {g['iddia'][:74]}")
            print(f"       [{g['kanit']['tur']}] {g['kanit']['ref']}")
        print(f"\n ZAYIF")
        for z in m["zayif"]:
            print(f"   − {z['iddia'][:74]}")
            print(f"       [{z['kanit']['tur']}] {z['kanit']['ref']}")
        print(f"\n İNCELİKLER — yalnız bu motorda ({len(m['incelikler'])} tane)")
        for i in m["incelikler"]:
            print(f"   ◆ {i['ad']}")
            print(f"       {i['ne'][:100]}")
            print(f"       → {i['sonuc'][:96]}")
            print(f"       [{i['kanit']['tur']}] {i['kanit']['ref']}")
        print(f"\n ÇAPA: fetch ×{m['capa']['fetch']} · defteri tutan: {m['capa']['defter']}")
        print(f" NE ZAMAN     : {m['ne_zaman']}")
        print(f" NE ZAMAN DEĞİL: {m['ne_zaman_degil']}")
