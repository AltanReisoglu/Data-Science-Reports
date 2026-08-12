"""Desen 5 — autogen_core: aktör modeli + gerçek bir hata izolasyonu deneyi.

**POC'un en önemli dosyası.** Diğer dört desen AgentChat katmanında; bu dosya bir
katman aşağıda, `autogen_core`'da. Kursta (ve çoğu tutorial'da) olmayan kısım tam
olarak burası — oysa AutoGen'in v0.2'den v0.4'e **baştan yazılma sebebi** bu katman.

Gösterilen şeyler:

1. **Aktör modeli** — ajan bir nesne değil, kendi mesaj kuyruğu olan bağımsız bir
   aktör. `@message_handler` mesajı **tipine göre** yönlendirir (`RoutedAgent`).
2. **Pub/sub** — yayıncı kime yayın yaptığını bilmez. Topic'e abone olan herkes alır.
   Sisteme ajan eklemek için mevcut hiçbir ajana dokunmazsın.
3. **Doğrudan adresleme (RPC)** — `send_message` ile tek bir aktöre istek + yanıt.
4. **Hata izolasyonunun sınırı** — aşağıdaki deney.

Burada **hiç LLM çağrısı yok**: mesele zekâ değil, çalışma zamanı.

────────────────────────────────────────────────────────────────────────
DENEY: "aktör modeli hata izolasyonu verir" iddiası ne kadar doğru?
────────────────────────────────────────────────────────────────────────
Aynı işi iki topic'te koşuyoruz:

* **karisik** → sağlam iki ajanın yanında bir de çöken ajan abone
* **izole**   → yalnızca sağlam iki ajan abone

Beklenti "çöken ajan diğerlerini etkilemez" olurdu. Ölçülen gerçek şu:
`_process_publish` abonelerin handler'larını `asyncio.gather` ile bekliyor.
Bir handler exception fırlatınca gather **hemen** dönüyor, ardından `task_done()`
çağrılıyor — yani kuyruk, kardeş handler'lar hâlâ çalışırken "boşaldı" sayılıyor.
`stop_when_idle()` bu yüzden erken dönüyor: **senkronizasyon bariyeri kırılıyor.**

Runtime ayakta kalıyor — izolasyon o anlamda gerçek. Ama:

* **Deney A**: bariyer erken açılıyor, sağlam ajanların sonucu o an elde YOK.
  (Bu koşuda sonuçlar bir sonraki `start()` sırasında geç düşüyor.)
* **Deney C**: aynı yayından hemen sonra `close()` çağrılırsa yarım kalan
  handler'lar iptal ediliyor ve sonuçlar **gerçekten kayboluyor** — hiçbir
  yerde "eksik sonuç" uyarısı yok, yalnızca log'da bir satır.

Yani aktör modeli **runtime'ı korur, veriyi korumaz**. Sessiz kısmi başarısızlık:
MAST taksonomisinin (arXiv 2503.13657) "system design / task verification"
kümesine giren ders kitaplık bir örnek.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field

from autogen_core import (
    AgentId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    message_handler,
    type_subscription,
)

import motor

KONU_KARISIK = "analiz-karisik"  # çöken ajanın da abone olduğu topic
KONU_IZOLE = "analiz-izole"      # yalnızca sağlam ajanlar

# NOT: @message_handler, imzadaki parametrenin adının tam olarak `message`
# (bağlamınkinin de `ctx`) olmasını şart koşuyor — yönlendirme imzadaki tip
# ipucundan çıkarılıyor. Bu iki isim bu yüzden Türkçeleştirilemiyor.


@dataclass
class SeriYayini:
    """Topic'e yayınlanan iş birimi."""

    metrik: str
    seri: list[float]


@dataclass
class RaporIstegi:
    """Doğrudan (RPC tarzı) gönderilen istek — yayın değil."""

    baslik: str


@dataclass
class Rapor:
    baslik: str
    satirlar: list[str] = field(default_factory=list)


SONUCLAR: dict[str, str] = {}
HATALAR: list[str] = []


@type_subscription(topic_type=KONU_KARISIK)
@type_subscription(topic_type=KONU_IZOLE)
class OrtalamaAjani(RoutedAgent):
    """İki topic'e de abone. Ortalama hesaplar."""

    def __init__(self) -> None:
        super().__init__("Ortalama hesaplayan aktör")

    @message_handler
    async def seri_geldi(self, message: SeriYayini, ctx: MessageContext) -> None:
        await asyncio.sleep(0.05)  # iş yükü taklidi: yarıda kesilebilir olmalı
        deger = round(statistics.fmean(message.seri), 4)
        SONUCLAR[f"{message.metrik}.ortalama"] = str(deger)
        print(f"     ✓ OrtalamaAjani  {message.metrik:<12} ortalama = {deger}")


@type_subscription(topic_type=KONU_KARISIK)
@type_subscription(topic_type=KONU_IZOLE)
class TrendAjani(RoutedAgent):
    """Aynı yayını alır, trend hesaplar. Ortalama ajanından habersizdir."""

    def __init__(self) -> None:
        super().__init__("Trend hesaplayan aktör")

    @message_handler
    async def seri_geldi(self, message: SeriYayini, ctx: MessageContext) -> None:
        await asyncio.sleep(0.05)
        ilk, son = message.seri[0], message.seri[-1]
        deger = round((son - ilk) / ilk * 100, 2)
        SONUCLAR[f"{message.metrik}.trend_yuzde"] = str(deger)
        print(f"     ✓ TrendAjani     {message.metrik:<12} trend    = %{deger}")


@type_subscription(topic_type=KONU_KARISIK)
class BozukAjan(RoutedAgent):
    """Kasıtlı olarak çöker — yalnızca 'karisik' topic'inde."""

    def __init__(self) -> None:
        super().__init__("Her mesajda hata fırlatan aktör")

    @message_handler
    async def seri_geldi(self, message: SeriYayini, ctx: MessageContext) -> None:
        HATALAR.append(message.metrik)
        raise RuntimeError(f"BozukAjan '{message.metrik}' işlerken çöktü (kasıtlı)")


class RaporAjani(RoutedAgent):
    """Hiçbir topic'e abone DEĞİL — yalnızca doğrudan adreslenerek çağrılır."""

    def __init__(self) -> None:
        super().__init__("Raporu toplayan aktör")

    @message_handler
    async def rapor_iste(self, message: RaporIstegi, ctx: MessageContext) -> Rapor:
        return Rapor(
            baslik=message.baslik,
            satirlar=[f"{k} = {v}" for k, v in sorted(SONUCLAR.items())],
        )


async def _yayinla(runtime: SingleThreadedAgentRuntime, konu: str, metrik: str, seri: list[float]) -> int:
    """Tek yayın yapar, runtime boşalana kadar bekler ve kaç yeni sonuç geldiğini döndürür."""
    onceki = len(SONUCLAR)
    runtime.start()
    await runtime.publish_message(SeriYayini(metrik=metrik, seri=seri), TopicId(konu, "default"))
    await runtime.stop_when_idle()
    return len(SONUCLAR) - onceki


async def calistir() -> motor.Olcum:
    print(f"\n╔{'═' * 60}╗")
    print(f"║ {'Desen 5 · autogen_core aktör modeli (LLM yok)':<59}║")
    print(f"╚{'═' * 60}╝")

    SONUCLAR.clear()
    HATALAR.clear()

    t0 = time.perf_counter()
    # ignore_unhandled_exceptions=True (varsayılan): handler hatası runtime'ı düşürmez.
    runtime = SingleThreadedAgentRuntime()

    # Kayıt = tip adı + fabrika. Ajanlar tembel (lazy) yaratılır.
    await OrtalamaAjani.register(runtime, "ortalama", lambda: OrtalamaAjani())
    await TrendAjani.register(runtime, "trend", lambda: TrendAjani())
    await BozukAjan.register(runtime, "bozuk", lambda: BozukAjan())
    await RaporAjani.register(runtime, "rapor", lambda: RaporAjani())

    gelir = [120.5, 133.0, 128.75, 141.2, 155.9, 149.3, 162.0]
    gecikme = [240.0, 231.0, 255.0, 228.0, 219.0, 235.0, 210.0]

    print(f"\n  ── DENEY A · topic '{KONU_KARISIK}' (çöken ajan da abone) ──")
    a_sonuc = await _yayinla(runtime, KONU_KARISIK, "gelir", gelir)
    print(f"     → beklenen 2 sonuç, gelen {a_sonuc}")

    print(f"\n  ── DENEY B · topic '{KONU_IZOLE}' (yalnızca sağlam ajanlar) ──")
    b_sonuc = await _yayinla(runtime, KONU_IZOLE, "gecikme_ms", gecikme)
    print(f"     → beklenen 2 sonuç, gelen {b_sonuc}")
    if b_sonuc > 2:
        print("       (fazlası Deney A'nın geç düşen sonuçları — bariyer kırıldığının kanıtı)")

    # DENEY C: aynı senaryo, ama yayından hemen sonra runtime kapatılıyor.
    # Ayrı bir runtime + ayrı sonuç sözlüğü ile ölçüyoruz ki A/B'yi kirletmesin.
    print(f"\n  ── DENEY C · '{KONU_KARISIK}' + yayından hemen sonra close() ──")
    c_kayit = dict(SONUCLAR)
    rt2 = SingleThreadedAgentRuntime()
    await OrtalamaAjani.register(rt2, "ortalama", lambda: OrtalamaAjani())
    await TrendAjani.register(rt2, "trend", lambda: TrendAjani())
    await BozukAjan.register(rt2, "bozuk", lambda: BozukAjan())
    rt2.start()
    await rt2.publish_message(
        SeriYayini(metrik="hata_orani", seri=[0.031, 0.028, 0.035, 0.022, 0.019, 0.024, 0.017]),
        TopicId(KONU_KARISIK, "default"),
    )
    await rt2.stop_when_idle()
    await rt2.close()  # ← yarım kalan handler'lar burada iptal ediliyor
    c_sonuc = len(SONUCLAR) - len(c_kayit)
    print(f"     → beklenen 2 sonuç, gelen {c_sonuc}"
          f"{'  ⇒ KALICI KAYIP' if c_sonuc < 2 else ''}")

    # Doğrudan adresleme (RPC): yayın değil, tek aktöre istek + yanıt.
    runtime.start()
    rapor: Rapor = await runtime.send_message(
        RaporIstegi(baslik="Aktör Modeli Raporu"), AgentId("rapor", "default")
    )
    await runtime.stop_when_idle()
    await runtime.close()
    sure = int((time.perf_counter() - t0) * 1000)

    print(f"\n  ── {rapor.baslik} (RPC ile toplandı) ──")
    for s in rapor.satirlar:
        print(f"     {s}")

    gec_kalan = 2 - a_sonuc
    kalici_kayip = 2 - c_sonuc
    print(f"\n  BozukAjan {len(HATALAR)} kez çöktü; runtime her seferinde ayakta kaldı.")
    if gec_kalan > 0:
        print(f"  ⚠ DENEY A: stop_when_idle() {gec_kalan} sağlam ajan bitmeden döndü.")
        print("    Sebep: bir handler exception fırlatınca gather erken dönüyor ve")
        print("    task_done() çağrılıyor → kuyruk boş sanılıyor, bariyer kırılıyor.")
    if kalici_kayip > 0:
        print(f"  ⚠ DENEY C: yayından hemen sonra close() → {kalici_kayip} sonuç KALICI kayıp.")
        print("    Ders: aktör modeli runtime'ı korur, VERİYİ korumaz. Kısmi başarısızlık")
        print("    sessizdir; doğrulama katmanını sen eklemek zorundasın.")

    olcum = motor.Olcum(
        desen="Desen 5 · autogen_core aktör modeli (LLM yok)",
        sure_ms=sure,
        mesaj_sayisi=3 + 2 + 3 + 1,  # A: 3 abone, B: 2, C: 3, +1 RPC
        llm_cagrisi=0,
        arac_cagrisi=0,
        devir_sirasi=["ortalama", "trend", "bozuk(çöktü)", "rapor(RPC)"],
        durma_nedeni=f"runtime idle; {len(HATALAR)} izole hata, {kalici_kayip} kalıcı kayıp",
        mod="LLM'siz",
    )
    olcum.yazdir()
    return olcum


if __name__ == "__main__":
    asyncio.run(calistir())
