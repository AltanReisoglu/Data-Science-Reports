#!/usr/bin/env python3
"""
motor_deney.py — Her motorun NİŞİNİ, kullanıcının KENDİ grafı üzerinde kanıtlayan
koşulabilir deneyler.

Künye/incelikler "şu motor şunu yapar" DİYOR. Bu modül onu YAPTIRIYOR: bir düğüme
simülasyon uygular, o motorda koşturur, sonuca bakılacak yeri söyler.

    python motor_deney.py            # katalog
    python motor_deney.py airflow    # tek motor

Her deney bir sözleşme:
    hedef      : hangi düğüme uygulanacağı ("son", "orta", "ilk", "yaprak")
    sim        : node_sim modu
    backend    : hangi motorda koşacağı
    bak        : sonuçta NEREYE bakılacağı
    beklenen   : ne görülmesi gerektiği (doğrulanabilir iddia)
    nis        : bu deneyin hangi ayırt edici özelliği gösterdiği

Deneyler mevcut /wf/run yolunu kullanıyor — ayrı bir yürütme yolu YOK. Yani
ekranda görülen şey, gerçekten koşan şey.
"""
from __future__ import annotations

MOTORLAR = ("own", "celery", "celery_canvas", "airflow", "temporal")


def _d(ad, nis, hedef, sim, backend, bak, beklenen, sure="", karsit="",
       gosterir=None, dis_dugme=False):
    """gosterir: bu TEK koşunun aynı anda kanıtladığı iddialar.

    Koşular pahalı (celery 13-70 sn). Bir koşuyu tek iddia için harcamak yerine,
    aynı koşuda görünen her şeyi listeliyoruz — ekip bir kez tıklayıp birden
    çok farkı görebilsin.
    """
    return {"ad": ad, "nis": nis, "hedef": hedef, "sim": sim, "backend": backend,
            "bak": bak, "beklenen": beklenen, "sure": sure, "karsit": karsit,
            "gosterir": gosterir or [], "dis_dugme": dis_dugme}


DENEYLER = {
    # Her motorda 3-4 deney. Deneyler ÇOK İDDİALI seçildi: bir koşu birden fazla
    # farkı aynı anda gösteriyor (`gosterir` listesi). Sebep zaman — celery koşusu
    # 13-70 sn sürüyor, her iddia için ayrı koşu israf olurdu.
    #
    # EN VERİMLİ TEK HAREKET: herhangi bir deneyde "⚖ 4 motorda karşılaştır" —
    # tek tıkla dört motor aynı kurulumda koşuyor ve fark tablosu çıkıyor.
    "own": [
        _d("Kalıcı hata — iptal zinciri + breaker",
           "Batan düğümün ardıllarına NE OLDUĞU kayıtlı. Celery ve Temporal'da "
           "böyle bir durum yok.",
           "orta", {"mod": "kalici"}, "own",
           "düğüm tablosu — hedefin durumu, koşum sayısı ve ardılların durumu",
           "hedef `failed` (koşum=3, breaker kesti), ARDILLARI `cancelled`, "
           "kardeş düğümler `done`",
           "~0,1 sn",
           "Celery'de ardılların ne olduğu belirsiz, kayıt yok",
           gosterir=["circuit breaker 3 denemede kesiyor",
                     "iptal zinciri — ardıllar cancelled",
                     "kardeş düğüm etkilenmiyor (done kalıyor)",
                     "kısmi tamamlama: 3/6 done",
                     "düğüm bazlı durum tablosu"]),
        _d("Çökme — düğüm içi checkpoint",
           "Worker çökerse yapılan iş korunur. Üç motorun HİÇBİRİNDE yok.",
           "orta", {"mod": "cokme"}, "own",
           "çökme/kurtarma sayacı + hedef düğümün koşum sayısı",
           "çökme 1 / kurtarma 1, hedef düğüm koşum=1 — iş TEKRARLANMADI, "
           "graf 6/6 tamamlandı",
           "~0,2 sn",
           "Airflow/Temporal'da adım baştan koşar — düğüm içi kurtarma yok",
           gosterir=["checkpoint — kısmi iş korunuyor",
                     "recover_stale çöken worker'ı fark ediyor",
                     "lease süresi dolunca devralma",
                     "çökmeye rağmen 6/6 tamam"]),
        _d("Geçici hata — retry ve deneme sayacı",
           "Retry otoritesi board'da; dört motorda da AYNI sayıyı üretiyor.",
           "orta", {"mod": "gecici"}, "own",
           "retry sütunu + düğüm tablosunda koşum",
           "retry 1, hedef düğüm koşum=2, ÖNCEKİLER koşum=1 — sadece patlayan tekrar koştu",
           "~0,1 sn",
           "Saf Celery'de adımlar tek task'taysa öncekiler de koşardı",
           gosterir=["retry sayacı board'dan geliyor",
                     "yalnız patlayan düğüm tekrar koşuyor",
                     "öncekiler korunuyor (koşum=1)",
                     "deneme vs koşum ayrımı"]),
        _d("Hatasız — karar katmanının maliyeti",
           "Ağ yok, broker yok, ayrı süreç yok.",
           None, {}, "own",
           "süre sütunu + çıktı",
           "0,02-0,05 sn — Celery'nin ~300 katı hızlı, çıktı dördüyle BYTE-BYTE aynı",
           "~0,05 sn",
           "Aynı graf Celery'de 13-15 sn (6 sn'si worker açılışı)",
           gosterir=["board'un hız avantajı", "dört motor aynı çıktıyı veriyor"]),
    ],
    "celery": [
        _d("Kalıcı hata — Celery'nin kendi kaydı YOK",
           "Board olmasa bu tabloyu çizemezdik. Celery batan işin ardılına ne "
           "olduğunu hiçbir yere yazmıyor.",
           "orta", {"mod": "kalici"}, "celery",
           "düğüm tablosu — durumları BOARD üretiyor, Celery değil",
           "own ile birebir aynı sütunlar: failed ×3 + 2 cancelled. Tek fark SÜRE.",
           "~40 sn",
           "Airflow aynı bilgiyi KENDİ tablosunda üretiyor (upstream_failed)",
           gosterir=["Celery'nin kendi durum kaydı yok",
                     "board sayesinde tablo çiziliyor",
                     "sonuçlar own ile birebir aynı",
                     "tek ayrışma: süre"]),
        _d("Açılış bedeli — broker + worker",
           "Celery'nin asıl maliyeti işin kendisi değil, altyapının ayağa kalkması.",
           None, {}, "celery",
           "süre sütunu, own ile yan yana",
           "13-15 sn; bunun ~6 saniyesi worker açılışı",
           "~13 sn",
           "own aynı işi 0,05 sn'de bitiriyor — fark altyapıdan",
           gosterir=["worker açılış maliyeti ~6 sn",
                     "own'a göre ~300 kat yavaş",
                     "buna rağmen çıktı aynı"]),
        _d("Geçici hata — çift retry yolu tuzağı",
           "Celery'nin KENDİ retry'ı kapalı; açık olsaydı düğüm ×3 koşardı. "
           "Bu, bu çalışmada bulunan ⑪ numaralı hataydı.",
           "orta", {"mod": "gecici"}, "celery",
           "düğüm tablosunda koşum — dördüyle aynı olmalı",
           "hedef ×2 (dördüyle aynı). `self.retry()` kaldırılmasaydı ×3 olurdu",
           "~40 sn",
           "İki dayanıklılık katmanı üst üste binince iş fazladan yapılıyor",
           gosterir=["retry otoritesi tek yerde olmalı",
                     "hata ⑪'in düzeltilmiş hâli",
                     "dört motorda aynı koşum sayısı"]),
        _d("Yavaş düğüm — koşarken durum görünüyor mu",
           "Board sayesinde görünüyor; Celery'nin kendi kaydı BOŞ.",
           "orta", {"mod": "yavas", "sn": 4}, "celery",
           "koşu logu — düğüm düğüm ilerleme",
           "board her düğümün durumunu bildiriyor; Celery yalnız 'task gönderildi' diyor",
           "~17 sn",
           "Airflow'da aynı an task_instance tablosunda satır satır durum var",
           gosterir=["durum görünürlüğü board'dan geliyor",
                     "Celery tek başına bunu veremez"]),
    ],
    "celery_canvas": [
        _d("Hatasız canvas — ifade gücü yeter",
           "chain/group/chord üç deseni de karşılıyor; sonuç board'lu koşuyla AYNI.",
           None, {}, "celery_canvas",
           "aşağıdaki 'hatasız koştur' düğmesi (board YOK)",
           "6 düğüm ~9 sn'de doğru CSV üretti — chain(4 katman) → group(2) → group(2)",
           "~9 sn",
           "Board'lu Celery aynı grafı 13 sn'de bitiriyor",
           gosterir=["canvas sıralı zinciri kuruyor",
                     "group ile paralel çalışıyor",
                     "chord ile toplama yapıyor",
                     "sonuç board'lu koşuyla aynı"],
           dis_dugme=True),
        _d("Kalıcı hata — canvas SUSUYOR",
           "Canvas'ın gerçek sınırı: koşarken 'neredeyiz' sorulamıyor, batan "
           "zincirin kalanına ne olduğu hiçbir yere yazılmıyor.",
           "orta", {"mod": "kalici"}, "celery_canvas",
           "aşağıdaki 'kalıcı hata ile koştur' düğmesi — log satırları",
           "hedef ORTADAysa 60+ sn bekleyip TAKILIR; SONDAysa RuntimeError döner. "
           "İkisinde de HANGİ halkanın battığı yazmıyor.",
           "~10-76 sn",
           "own: 3/6 done + 1 failed + 2 CANCELLED · airflow: upstream_failed kayıtlı",
           gosterir=["'nerede kaldı' sorgusu yok",
                     "batan zincirin kalanı sessizce ölüyor",
                     "hata davranışı grafın şekline bağlı",
                     "link_error kurulmadıysa haber yok"],
           dis_dugme=True),
    ],
    "airflow": [
        _d("Kalıcı hata — `upstream_failed` + kısmi devam",
           "Dört motor içinde batan işin ardıllarını KENDİ tablosunda kaydeden tek "
           "motor. Sabah gelen operatör tek sorguyla ne olduğunu görür.",
           "orta", {"mod": "kalici"}, "airflow",
           "Airflow'un KENDİ task_instance tablosu (board'dan değil)",
           "önceki düğümler `success`, hedef `failed` (try_number=3), ardıllar "
           "`upstream_failed`, KARDEŞ düğüm `success` — hiç koşmadıkları AÇIKÇA kayıtlı",
           "~4,8 sn",
           "Celery'de bu bilgi hiç üretilmiyor; canvas'ta zincir susuyor",
           gosterir=["upstream_failed — Airflow'un kendi terminolojisi",
                     "kısmi devam: kardeş düğüm success kalıyor",
                     "try_number ile deneme sayısı",
                     "operasyonel görünürlük — tek sorguyla tablo"]),
        _d("Geçici hata — sadece hatalı düğüm tekrarlanır",
           "Airflow'un Celery'ye göre en büyük kazancı.",
           "orta", {"mod": "gecici"}, "airflow",
           "try_number sütunu, düğüm düğüm",
           "hedef try_number=2, ÖNCEKİLER try_number=1 ve `success` — tekrar koşmadılar",
           "~2,6 sn",
           "Saf Celery'de adımlar tek task'taysa öncekiler de baştan koşardı",
           gosterir=["adım bazlı retry",
                     "biten adım DB'de success kalıyor",
                     "try_number = koşum sayısı (board'un attempt'inden farklı)"]),
        _d("30 sn backoff'un bedeli",
           "Airflow batch işler için ayarlanmış; hızlı akışlarda bu bir uyumsuzluk. "
           "Kusur değil, tercih.",
           "orta", {"mod": "kalici"}, "airflow",
           "süre — panelde retry_delay 1 sn'ye çekili",
           "1 sn ile ~4,8 sn; varsayılan 30 sn olsaydı 62 sn (canlı ölçüldü)",
           "~4,8 sn",
           "own aynı senaryoyu 0,02 sn'de bitiriyor",
           gosterir=["retry_delay'in toplam süreye etkisi",
                     "batch tasarımının hızlı akışa uyumsuzluğu"]),
        _d("Hatasız — XCom veri akışı çalışıyor mu",
           "Airflow board'a HİÇ yazmıyor; çıktısı kendi XCom'undan okunuyor.",
           None, {}, "airflow",
           "çıktı — board'dan değil XCom'dan geliyor",
           "üretilen CSV dördüyle BYTE-BYTE aynı — veri akışının o tarafta da doğru "
           "kurulduğunun BAĞIMSIZ kanıtı",
           "~2,6 sn",
           "own/celery/temporal çıktıyı board'dan alıyor, Airflow almıyor",
           gosterir=["XCom ile düğümler arası veri akışı",
                     "board'dan bağımsız doğrulama",
                     "dört motor aynı çıktı"]),
    ],
    "temporal": [
        _d("Çökme — replay ile kaldığı yerden devam",
           "Tamamlanmış activity REPLAY'de atlanır. Çökmeden devamı kutudan veren "
           "tek motor.",
           "orta", {"mod": "cokme"}, "temporal",
           "koşu logu + hedef düğümün koşum sayısı",
           "worker öldü, yeni koşu tamamlanmışları ATLADI, graf 6/6 tamamlandı, "
           "hedef koşum=2",
           "~0,6 sn",
           "Celery'de zincir baştan kurulur; canvas'ta zincir kaybolur",
           gosterir=["event history + replay",
                     "tamamlanan activity atlanıyor",
                     "çökmeye rağmen 6/6",
                     "kurtarma kodu yazmadan"]),
        _d("Kalıcı hata — iki dayanıklılık katmanı",
           "Temporal'ın RetryPolicy'si DEVREDE ama board da sayıyor. Uzlaştırmak "
           "için 'bayat claim onarımı' yazmak gerekti — hata ②.",
           "orta", {"mod": "kalici"}, "temporal",
           "düğüm tablosunda deneme ve koşum — dördüyle aynı olmalı",
           "failed ×3 + 2 cancelled, own ile birebir. Onarım olmasaydı iki kez "
           "BAŞARAN düğüm 3. turda failed işaretlenirdi",
           "~0,8 sn",
           "Motorun kendi retry'ını board'la birlikte kullanmanın bedeli",
           gosterir=["RetryPolicy ile board sayacının uzlaşması",
                     "hata ②'nin düzeltilmiş hâli",
                     "sonuçlar own ile birebir aynı",
                     "iptal zinciri board'dan geliyor"]),
        _d("Geçici hata — determinist workflow gövdesi",
           "Workflow saf; iş activity'lerde. Retry activity seviyesinde.",
           "orta", {"mod": "gecici"}, "temporal",
           "düğüm tablosunda koşum",
           "hedef ×2, öncekiler ×1 — dördüyle aynı",
           "~0,6 sn",
           "Airflow'un try_number'ı ile aynı sonucu veriyor, farklı mekanizmayla",
           gosterir=["activity seviyesi retry",
                     "workflow gövdesi determinist kalıyor",
                     "dört motorda aynı koşum sayısı"]),
        _d("Hatasız — cluster'a rağmen hız",
           "Durable olmak yavaş olmak demek değil.",
           None, {}, "temporal",
           "süre sütunu",
           "0,2-0,5 sn — own'dan sonra en hızlı, Celery'nin ~30 katı",
           "~0,5 sn",
           "Aynı graf Celery'de 13-15 sn",
           gosterir=["durable execution'ın hız maliyeti düşük",
                     "test-server tek süreçte koşuyor"]),
    ],
}


def hedef_dugum(nodes: list, kural: str | None) -> str | None:
    """Deneyin uygulanacağı düğümü grafın şekline göre seç."""
    if not nodes or not kural:
        return None
    ebeveynli = [n for n in nodes if n.get("parents")]
    if kural == "ilk":
        return next((n["id"] for n in nodes if not n.get("parents")), nodes[0]["id"])
    if kural == "yaprak":
        cocuklu = {p for n in nodes for p in n.get("parents", [])}
        return next((n["id"] for n in nodes if n["id"] not in cocuklu), nodes[-1]["id"])
    # "orta": ardılı OLAN ilk düğüm — iptal zinciri görünsün diye
    cocuklu = {p for n in nodes for p in n.get("parents", [])}
    ara = [n["id"] for n in nodes if n["id"] in cocuklu and n.get("parents")]
    if ara:
        return ara[0]
    return (ebeveynli[0]["id"] if ebeveynli else nodes[0]["id"])


def deneyler(motor: str, nodes: list | None = None) -> list:
    """Motorun deneyleri; nodes verilirse hedef düğüm de çözülür."""
    out = []
    for d in DENEYLER.get(motor, []):
        x = dict(d)
        x["hedef_id"] = hedef_dugum(nodes or [], d["hedef"])
        x["uygulanabilir"] = (d["hedef"] is None) or bool(x["hedef_id"])
        if not x["uygulanabilir"]:
            x["neden_olmaz"] = ("bu graf tek düğümlü — ardılı olan bir düğüm yok, "
                                "iptal/kurtarma zinciri gösterilemez")
        out.append(x)
    return out


def hepsi(nodes: list | None = None) -> dict:
    return {"motorlar": {m: deneyler(m, nodes) for m in MOTORLAR}}


if __name__ == "__main__":
    import sys
    hedef = sys.argv[1] if len(sys.argv) > 1 else None
    for m in ([hedef] if hedef else MOTORLAR):
        print(f"\n{'═'*78}\n {m.upper()} — nişini kanıtlayan deneyler\n{'═'*78}")
        for d in DENEYLER[m]:
            sim = d["sim"].get("mod", "—") if d["sim"] else "—"
            print(f"\n  ▶ {d['ad']}")
            print(f"      niş     : {d['nis']}")
            print(f"      kurulum : hedef={d['hedef'] or 'yok'} · sim={sim} · "
                  f"motor={d['backend']} · süre≈{d['sure']}")
            print(f"      bak     : {d['bak']}")
            print(f"      beklenen: {d['beklenen']}")
            if d["karsit"]:
                print(f"      karşıt  : {d['karsit']}")
