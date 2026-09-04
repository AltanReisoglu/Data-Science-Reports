"""Oturum kalıcılığı (2026-09-04) — iki ayrı ama birbirine bağlı eksik.

## Neden bu ikisi aynı dosyada

`session_id` TEK bir uuid'ydi ve İKİ yere birden gidiyordu:

    session_id ──> thread_id    (konuşma hafızası → InMemorySaver)
               └─> workflow_id  (artifact kapsamı → MinIO + kayıt defteri)

Her bağlantıda yeniden üretiliyordu. Konuşma hafızasının gitmesi beklenen bir
şeydi; asıl sorun ikincisiydi — artifact'ler depoda SAĞ KALIYOR ama onları
gösteren `workflow_id` bir daha üretilemediği için **erişilemez** hale
geliyorlardı. Kalıcı bir depoya yazıp okuma anahtarını çöpe atmak.

Bu yüzden iki düzeltme tek iştir: kalıcı anahtar (artifact'lere ulaşmayı sağlar)
+ kalıcı checkpointer (konuşmayı sürdürür).
"""

from __future__ import annotations

import asyncio

import pytest

from grounded_assistant.agent.graph import build_checkpointer
from grounded_assistant.session import oturum_kimligi as _oturum_kimligi

GECERLI = "7f3e4d2a-1b6c-4e8f-9a0d-5c2b1e4f6a8d"


# -- kalıcı anahtar --------------------------------------------------------


def test_istemci_kimligi_korunur():
    """Asıl kazanç: yeniden bağlanan istemci AYNI workflow'a düşer."""
    assert _oturum_kimligi(GECERLI) == GECERLI


def test_kimliksiz_baglanti_temiz_oturum_acar():
    """Kalıcılık opt-in: göndermeyene yeni oturum, kimse başkasınınkine düşmez."""
    ilk, ikinci = _oturum_kimligi(None), _oturum_kimligi(None)
    assert ilk != ikinci and len(ilk) == 36


@pytest.mark.parametrize(
    "kotu",
    [
        "../../etc/shadow",          # yol geçişi
        "a/b/c",                     # anahtar düzenine sızma
        "wf_baskasinin",             # tahmin edilebilir isim
        "7f3e4d2a1b6c4e8f9a0d5c2b",  # tireleri eksik
        "",
        "x" * 500,
    ],
)
def test_bicime_uymayan_kimlik_reddedilir(kotu):
    """`workflow_id` S3 anahtarının parçası olduğu için biçim DAR olmalı.

    Serviste ayrıca isim doğrulaması var, ama savunma kimliği ilk kabul eden
    yerde de olmalı.
    """
    sonuc = _oturum_kimligi(kotu)
    assert sonuc != kotu
    assert "/" not in sonuc and ".." not in sonuc
    assert len(sonuc) == 36


def test_buyuk_harf_normalize_edilir():
    """Aynı oturumun iki yazımı iki ayrı workflow'a bölünmesin."""
    assert _oturum_kimligi(GECERLI.upper()) == GECERLI.lower()


# -- kalıcı checkpointer ---------------------------------------------------


def test_arka_uc_secimi_config_ile(monkeypatch, tmp_path):
    """Postgres'e geçiş kod değil ortam değişkeni olmalı."""
    monkeypatch.delenv("PTC_CHECKPOINT_DSN", raising=False)
    monkeypatch.delenv("PTC_CHECKPOINT_DB", raising=False)

    async def _kur():
        return type(build_checkpointer()).__name__

    assert asyncio.run(_kur()) == "InMemorySaver"

    monkeypatch.setenv("PTC_CHECKPOINT_DB", str(tmp_path / "cp.db"))
    assert asyncio.run(_kur()) == "AsyncSqliteSaver"


def test_saver_ASYNC_yolu_destekliyor(monkeypatch, tmp_path):
    """ASIL REGRESYON (2026-09-04'te üretimde patladı).

    Hem web hem CLI `agent.ainvoke()` yoluna giriyor. Senkron `SqliteSaver` o
    yolda açık hata veriyordu:
        "The SqliteSaver does not support async methods."

    İlk testlerim bunu KAÇIRDI çünkü saver'ı doğrudan senkron `put`/`get_tuple`
    ile sınıyorlardı — uygulamanın hiç kullanmadığı yol. Bu test async
    metotları çağırıyor.
    """
    monkeypatch.setenv("PTC_CHECKPOINT_DB", str(tmp_path / "cp.db"))
    cfg = {"configurable": {"thread_id": GECERLI, "checkpoint_ns": ""}}
    kayit = {
        "v": 1, "id": "c1", "ts": "2026-09-04T00:00:00+00:00",
        "channel_values": {"messages": ["async yol"]},
        "channel_versions": {}, "versions_seen": {},
    }

    async def yaz_oku():
        cp = build_checkpointer()
        try:
            await cp.aput(cfg, kayit, {"source": "input", "step": 1}, {})
        finally:
            await cp.conn.close()
        cp2 = build_checkpointer()          # süreç yeniden başlamış gibi
        try:
            return await cp2.aget_tuple(cfg)
        finally:
            await cp2.conn.close()

    tup = asyncio.run(yaz_oku())
    assert tup is not None
    assert tup.checkpoint["channel_values"]["messages"] == ["async yol"]


def test_loop_disinda_kurmak_hata_verir(monkeypatch, tmp_path):
    """Sessizce InMemory'ye düşmemeli — çağıran loop içinde kurmak ZORUNDA.

    Sessiz düşüş, kalıcılığın kaybolduğunu kimsenin fark etmemesi demekti.
    """
    monkeypatch.setenv("PTC_CHECKPOINT_DB", str(tmp_path / "cp.db"))
    with pytest.raises(RuntimeError, match="no running event loop"):
        build_checkpointer()


def test_state_sureci_asar(monkeypatch, tmp_path):
    """ASIL TEST: yazan bağlantı kapandıktan sonra state okunabiliyor mu.

    `InMemorySaver` ile bu imkânsızdı — state agent process'inin RAM'indeydi.
    Burada iki AYRI saver nesnesi kullanılıyor: ikincisi, süreç yeniden
    başlamış gibi dosyayı sıfırdan açıyor.
    """
    monkeypatch.setenv("PTC_CHECKPOINT_DB", str(tmp_path / "cp.db"))
    cfg = {"configurable": {"thread_id": GECERLI, "checkpoint_ns": ""}}
    kayit = {
        "v": 1, "id": "c1", "ts": "2026-09-04T00:00:00+00:00",
        "channel_values": {"messages": ["kullanici: satislari cikar"]},
        "channel_versions": {}, "versions_seen": {},
    }

    async def yaz():
        cp = build_checkpointer()
        try:
            await cp.aput(cfg, kayit, {"source": "input", "step": 1}, {})
        finally:
            await cp.conn.close()

    async def oku():
        cp = build_checkpointer()
        try:
            return await cp.aget_tuple(cfg)
        finally:
            await cp.conn.close()

    asyncio.run(yaz())            # bir loop bitti = süreç ölmüş gibi
    tup = asyncio.run(oku())      # yepyeni loop, yepyeni nesne
    assert tup is not None
    assert tup.checkpoint["channel_values"]["messages"] == ["kullanici: satislari cikar"]


def test_ayri_oturumlar_karismaz(monkeypatch, tmp_path):
    """thread_id kapsamı: bir oturumun hafızası diğerinde görünmemeli."""
    monkeypatch.setenv("PTC_CHECKPOINT_DB", str(tmp_path / "cp.db"))

    async def calis():
        cp = build_checkpointer()
        try:
            return await _calis(cp)
        finally:
            await cp.conn.close()

    async def _calis(cp):
        await cp.aput(
            {"configurable": {"thread_id": "oturum-a", "checkpoint_ns": ""}},
            {"v": 1, "id": "c1", "ts": "2026-09-04T00:00:00+00:00",
             "channel_values": {"messages": ["a'nin mesaji"]},
             "channel_versions": {}, "versions_seen": {}},
            {"source": "input", "step": 1}, {},
        )
        return await cp.aget_tuple(
            {"configurable": {"thread_id": "oturum-b", "checkpoint_ns": ""}}
        )

    assert asyncio.run(calis()) is None


def test_aiosqlite_kendi_threadinde_calisiyor(monkeypatch, tmp_path):
    """`check_same_thread` derdi kalktı: aiosqlite bağlantıyı kendi thread'inde
    tutuyor, çağrılar oraya sıraya giriyor. Ajan hangi thread'den çağırırsa
    çağırsın sorun çıkmıyor."""
    monkeypatch.setenv("PTC_CHECKPOINT_DB", str(tmp_path / "cp.db"))
    cfg = {"configurable": {"thread_id": "x", "checkpoint_ns": ""}}

    async def basit():
        cp = build_checkpointer()
        try:
            return await cp.aget_tuple(cfg)
        finally:
            await cp.conn.close()

    assert asyncio.run(basit()) is None
