"""Sandbox entrypoint — LLM'in ürettiği kodu çalıştırır (Faz 2).

Bilerek Python-seviyesinde bir kısıtlama (RestrictedPython, builtins filtresi
vb.) YOK — research.md §4.3'teki karar: enforcement Cilium'da (network
seviyesinde), burada değil. Kod istediği kütüphaneyi import edebilir, ama
Tool Gateway dışında hiçbir yere çıkamaz (Cilium bunu kernel'de engeller).

Kontrat: contracts/sandbox_job_contract.md
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import UTC, datetime

# `serialize.py`, ana kaynak ağacındaki
# src/grounded_assistant/artifacts/serialize.py dosyasının AYNISIDIR — imaja
# bağımsız bir modül olarak kopyalanır (bkz. sandbox_image/Dockerfile).
# Kopyalanıyor çünkü sandbox imajına `grounded_assistant` paketinin tamamını
# (langgraph, langchain…) kurmak istemiyoruz; ama tek kaynak korunsun diye
# dosya çoğaltılmıyor, Dockerfile aynı dosyayı kopyalıyor.
import artifact_client
import serialize
from fastmcp import Client

TOOL_GATEWAY_ENDPOINT = os.environ["TOOL_GATEWAY_ENDPOINT"]
CODE_PATH = "/sandbox/code.py"

# Faz 1'in tool_policy.ALLOWED_TOOLS + LOCAL_TOOLS ile birebir aynı olmalı
# (CapabilityGrant.allowed_tools, data-model.md). Faz 4'te 4 yeni tool eklendi.
ALLOWED_TOOLS = (
    "search_knowledge_base",
    "get_ticket_status",
    "count_open_tickets",
    "create_support_ticket",
    "search_employee_directory",
    "web_search",
    "calculator",
    "fetch_url",
    "resolve_dns",
    "check_connectivity",
)

# LLM'in ürettiği kod, tool'ları normal bir Python fonksiyonu gibi pozisyonel
# çağırabilir (ör. `search_knowledge_base("vpn erisim")`) — MCP'nin kendisi
# yalnızca adlandırılmış argüman kabul ettiği için, pozisyonel argümanları
# isimlere çevirmek amacıyla bu sabit eşleme gerekiyor (contracts/tool_gateway_mcp.md'deki
# tool imzalarıyla birebir).
_ARG_NAMES: dict[str, tuple[str, ...]] = {
    "search_knowledge_base": ("query",),
    "get_ticket_status": ("ticket_id",),
    "count_open_tickets": (),
    "create_support_ticket": ("title", "description"),
    "search_employee_directory": ("query",),
    "web_search": ("query",),
    "calculator": ("expression",),
    "fetch_url": ("url",),
    "resolve_dns": ("hostname",),
    "check_connectivity": ("host", "port"),
}


#: Bu argümanlar tool_call log satırına YAZILMAZ.
#:
#: 2026-09-04 öncesinde burada `content_b64` de vardı: artifact baytları MCP
#: çağrısında base64 taşınıyordu ve olduğu gibi stdout'a basmak pod log'unu
#: şişirip `_wait_and_stream`'i (her turda tüm log'u yeniden okur) fiilen
#: kilitliyordu. Artifact yolu HTTP'ye taşındığı için o argüman artık hiçbir
#: tool çağrısında geçmiyor; liste savunma amaçlı duruyor.
_LOGA_YAZILMAZ = {"scope_token", "content_b64"}


def _log_icin(kwargs: dict) -> dict:
    return {k: "<gizli>" if k in _LOGA_YAZILMAZ else v for k, v in kwargs.items()}


def _make_sync_tool(tool_name: str):
    """Sandbox kodunun senkron çağırabileceği bir tool-proxy fonksiyonu üretir.
    Gerçek iş fastmcp.Client ile Tool Gateway'e (Cilium'un izin verdiği TEK
    hedef) yapılan bir HTTP çağrısıdır.

    Her çağrı, nihai sonuç satırından ÖNCE ayrı bir JSON satırı olarak stdout'a
    da yazılır (`"type": "tool_call"`) — ana asistan (sandbox_runner.py, T015)
    bunu `Trace.record_tool_call`'a besler (FR-008). Kontratın orijinal nihai
    satırında (`sandbox_job_contract.md`) `type` alanı YOK — bu, iki satır türünü
    ayırt etmenin yolu."""

    def _call(*args, **kwargs):
        named_from_args = dict(zip(_ARG_NAMES.get(tool_name, ()), args))
        kwargs = {**named_from_args, **kwargs}

        async def _do():
            async with Client(TOOL_GATEWAY_ENDPOINT) as client:
                result = await client.call_tool(tool_name, kwargs)
                return result.data if hasattr(result, "data") else str(result)

        timestamp = datetime.now(UTC).isoformat()
        try:
            value = asyncio.run(_do())
        except Exception:
            print(
                json.dumps(
                    {
                        "type": "tool_call",
                        "tool": tool_name,
                        "args": _log_icin(kwargs),
                        "status": "error",
                        "timestamp": timestamp,
                    }
                )
            )
            raise
        print(
            json.dumps(
                {
                    "type": "tool_call",
                    "tool": tool_name,
                    "args": _log_icin(kwargs),
                    "status": "success",
                    "timestamp": timestamp,
                }
            )
        )
        return value

    return _call


# ---------------------------------------------------------------------------
# Artifact API — sandbox kodunun gördüğü yüzey
#
# Tez: pod çalışması bitince SİLİNİR, ama buraya yazılan artifact KALIR. Sonraki
# bir PTC çalıştırması (aynı tur, sonraki tur, ya da başka bir workflow node'u)
# onu yeniden üretmeden okur.
#
# Serileştirme BURADA yapılır: bir dataframe ağdan nesne olarak geçemez,
# Parquet'e çevrilip HAM BAYT olarak gönderilir. Servis baytı olduğu gibi
# saklar, çözmeye kalkmaz.
#
# 2026-09-04: bu yol Tool Gateway'den (MCP + base64) Artifact Service'e (akışlı
# HTTP) taşındı. Gateway artık yalnızca tool proxy'si; artifact baytı oradan
# geçmiyor. Gerekçe: services/artifact_service/app.py başlığı.
# ---------------------------------------------------------------------------

#: `sandbox_runner`'ın imzaladığı kapsam jetonu. Kodun bunu bilmesine gerek yok;
#: istemci her isteğe kendisi ekliyor.
SCOPE_TOKEN = os.environ.get("PTC_SCOPE_TOKEN", "")

#: Kapsam jetonundan çözülen workflow — yalnızca `/workflows/{id}/artifacts`
#: yolunu kurmak için. Yetki yine JETONDAN geliyor, bu değerden değil.
WORKFLOW_ID = os.environ.get("PTC_WORKFLOW_ID", "")


#: LLM'in dosya yazabileceği, çalışma sonunda SÜPÜRÜLEN dizin.
#:
#: `/scratch`ten AYRI olması kasıtlı: orası geçici alan (ara dosyalar, cache,
#: yarım çıktılar) ve süpürülse her çöp artifact'e dönerdi. Anthropic'in
#: `$OUTPUT_DIR`'ı da tam bu yüzden boş ve ayrı bir dizin.
OUTPUT_DIR = os.environ.get("PTC_OUTPUT_DIR", "/output")

#: Süpürmede yok sayılacak adlar — kullanıcı çıktısı değiller.
_SUPURME_DISI = (".", "__")


def _gecerli_artifact_adi(dosya_adi: str) -> str:
    """Dosya adını servisin kabul ettiği biçime çevirir.

    Servis `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$` istiyor (yol geçişine karşı).
    Buraya bir dosya adı geldiği için yol ayıracı zaten olamaz, ama Türkçe
    karakter/boşluk olabilir — onları tireye çeviriyoruz ki isim reddedilip
    kullanıcının çıktısı sessizce kaybolmasın.
    """
    temiz = "".join(c if (c.isascii() and (c.isalnum() or c in "._-")) else "-" for c in dosya_adi)
    while temiz and not (temiz[0].isascii() and temiz[0].isalnum()):
        temiz = temiz[1:]
    return (temiz or "cikti")[:128]


def _manifest(istemci) -> set[str]:
    """Pod açılışında TEK round-trip: bu workflow'da hangi isimler var.

    ## Neden artık bayt indirilmiyor (2026-09-04 kararı)

    Burada eskiden `_onceki_ciktileri_yukle` vardı: pod doğarken depodaki HER
    artifact'i `/output`'a indiriyordu. İki sorunu vardı ve ikisi de ölçekle
    büyüyordu:

      - Maliyet *var olan her şeyle* ölçekleniyordu. Workflow'da N artifact
        varsa, hiçbirine dokunmayan bir script bile N indirme yapıyordu.
      - `/output` 512Mi'lık bir emptyDir. Altı tane 100 MiB'lik artifact
        biriktiği anda pod, kodu çalıştırmadan tahliye ediliyordu.

    Şimdi yalnızca İSİMLER geliyor — boyuttan ve sayıdan bağımsız tek istek.
    Baytlar `_tembel_oku` ile ancak gerçekten okunduğunda iniyor: maliyet
    O(kullanılan), O(hepsi) değil.

    Simetri de bilerek bozuldu: süpürme (yazma) otomatik kalıyor çünkü maliyeti
    *o çalıştırmada üretilenle* ölçekleniyor — doğal olarak küçük. Prefetch
    (okuma) ise sınırsız büyüyordu.

    BEST-EFFORT: servise ulaşılamazsa boş küme döner; pod açılışı engellenmez,
    yalnızca tembel okuma devre dışı kalır (LLM `get_artifact()`'i yine
    çağırabilir).
    """
    if not WORKFLOW_ID:
        return set()
    try:
        return {k["name"] for k in istemci.list(WORKFLOW_ID) if k.get("name")}
    except Exception:  # noqa: BLE001
        return set()


def _tembel_oku(yol, istemci, mevcut: set[str], inenler: dict) -> None:
    """`/output/<ad>` istendi ama dosya yok — varsa depodan indir.

    Prefetch'in yerine geçen mekanizma. Yalnızca `/output`'un EN ÜST düzeyine
    bakar (süpürmenin yazdığı yer) ve yalnızca manifestte adı geçen dosyaları
    indirir — yani var olmayan bir dosya için ağa çıkılmaz, `FileNotFoundError`
    normal şekilde yükselir.

    `inenler`'e (ad -> (mtime, boyut)) yazıyor. Bu KRİTİK: indirdiğimiz dosya
    `/output`'ta duruyor ve süpürme onu "LLM üretmiş" sanıp geri yükleyebilir —
    prefetch döneminde bulunup düzeltilen kusurun (2026-09-03) aynısı, artık
    tembel yolda. Süpürme bu sözlüğe bakıp dokunulmamış olanı atlıyor.
    """
    if not isinstance(yol, (str, bytes, os.PathLike)):
        return  # dosya nesnesi, URL, tampon — bizim işimiz değil
    metin = os.fspath(yol)
    if isinstance(metin, bytes):
        metin = metin.decode("utf-8", "replace")
    if os.path.exists(metin):
        return
    if os.path.dirname(os.path.abspath(metin)) != os.path.abspath(OUTPUT_DIR):
        return
    ad = _gecerli_artifact_adi(os.path.basename(metin))
    if ad not in mevcut:
        return
    try:
        kunye = istemci.fetch_to_file(ad, metin)
    except Exception:  # noqa: BLE001 — indirilemezse dosya yok gibi davran
        return
    if kunye:
        durum = os.stat(metin)
        inenler[os.path.basename(metin)] = (durum.st_mtime, durum.st_size)
        _artifact_olayi("consumed", kunye)


def _tembel_okumayi_kur(istemci, mevcut: set[str], inenler: dict) -> dict:
    """pandas okuyucularını sarmalar, sandbox'a tembel bir `open` döndürür.

    İki ayrı yer gerekiyor çünkü kapsamları farklı:

      - **pandas** bir kütüphane; `read_csv` kendi C parser'ıyla dosyayı
        açtığı için `open`'ı sarmak yetmez, modülün kendisi yamalanmalı.
      - **`open`** ise `sandbox_globals`'a konuyor, `builtins` DEĞİŞTİRİLMİYOR.
        Böylece LLM'in doğrudan `open("/output/x.json")` çağrısı yakalanıyor
        ama kütüphanelerin iç dosya işlemleri hiç etkilenmiyor. Yama alanını
        dar tutmanın en temiz yolu bu.
    """
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError:
        pd = None

    if pd is not None:
        for fn_adi in ("read_csv", "read_parquet", "read_json", "read_excel", "read_feather"):
            orijinal = getattr(pd, fn_adi, None)
            if orijinal is None:
                continue
            setattr(pd, fn_adi, _okuyucu_sarmala(orijinal, istemci, mevcut, inenler))

    gercek_open = open

    def _tembel_open(dosya, kip="r", *args, **kwargs):
        if "r" in kip and "+" not in kip:
            _tembel_oku(dosya, istemci, mevcut, inenler)
        return gercek_open(dosya, kip, *args, **kwargs)

    return {"open": _tembel_open}


def _okuyucu_sarmala(orijinal, istemci, mevcut: set[str], inenler: dict):
    def _oku(yol=None, *args, **kwargs):
        if yol is not None:
            _tembel_oku(yol, istemci, mevcut, inenler)
        return orijinal(yol, *args, **kwargs)

    _oku.__name__ = getattr(orijinal, "__name__", "read")
    _oku.__doc__ = getattr(orijinal, "__doc__", None)
    return _oku


def _ciktilari_supur(
    api: dict, inenler: dict[str, tuple[float, int]] | None = None
) -> None:
    """`/output`'un EN ÜST düzeyindeki dosyaları artifact'e çevirir.

    Neden var: `put_artifact` çağırmak LLM'in inisiyatifinde. Çağırmazsa —
    ki modelin bu API'yi bilmemesi ya da unutması mümkün — ürettiği dosyalar
    pod ile birlikte yok olurdu. Bu döngü emniyet ağı: LLM sıradan kod
    yazsın (`df.to_csv("/output/rapor.csv")`), çıktısı yine kalıcı olsun.

    HATA DURUMUNDA DA çalışır (bkz. main): script son satırda patlasa bile o
    ana kadar üretilen dosyalar kurtarılır — asıl değeri de burada.

    Yalnızca üst düzey: alt dizinlere inilmez. Anthropic'in `$OUTPUT_DIR`
    kuralının aynısı; aksi halde kütüphanelerin bıraktığı cache ağaçları da
    artifact'e dönerdi.

    `inenler` — `_tembel_oku`'nun doldurduğu (ad -> (mtime, boyut)) sözlüğü.
    Bir dosya bu sözlükte VE hâlâ o mtime/boyutta ise, LLM ona HİÇ
    DOKUNMAMIŞ demektir (biz indirdik, o sadece okudu) — tekrar yüklenmez.

    Bu kontrol OLMADAN yaşanan gerçek kusur (2026-09-03, o zaman prefetch
    yolunda): depodan `/output`'a konan dosyalar "yeni üretilmiş" sayılıp geri
    yükleniyordu — dedup baytı tekilleştirse de her seferinde yeni bir
    `artifact_id` ve sahte bir "produced" olayı doğuyordu. Prefetch kalktı ama
    tembel okuma da aynı şekilde dosya yerleştiriyor; kontrol bu yüzden duruyor.

    Yükleme AKIŞLI: dosya `put_file` ile doğrudan diskten gönderiliyor, içeriği
    belleğe alınmıyor (eskiden `f.read()` + base64 vardı).
    """
    inenler = inenler or {}
    try:
        adlar = sorted(os.listdir(OUTPUT_DIR))
    except OSError:
        return  # dizin yoksa süpürecek bir şey de yok

    for ad in adlar:
        yol = os.path.join(OUTPUT_DIR, ad)
        if ad.startswith(_SUPURME_DISI) or not os.path.isfile(yol):
            continue

        onceki = inenler.get(ad)
        if onceki is not None:
            durum = os.stat(yol)
            if (durum.st_mtime, durum.st_size) == onceki:
                continue  # dokunulmamış — indirdiğimiz dosyayı geri yükleme

        try:
            api["_put_file"](
                yol,
                serialize.content_type_for_filename(ad),
                _gecerli_artifact_adi(ad),
            )
        except Exception as exc:  # noqa: BLE001
            # Süpürme BEST-EFFORT: bir dosya reddedilse (pickle, boyut) ya da
            # okunamasa bile diğerleri ve asıl sonuç etkilenmemeli.
            #
            # AYRI bir `type` kullanılıyor ("artifact_skipped", "artifact"
            # DEĞİL): hiç depolanmamış bir dosyanın artifact_id'si olamaz,
            # `ArtifactEvent` modeli bunu zaten kabul etmiyor (artifact_id
            # zorunlu alan). `sandbox_runner` bu satırı ArtifactEvent'e
            # ÇEVİRMEDEN yalnızca on_event'e iletir — gözlemlenebilirlik için
            # yeterli, model kirlenmiyor.
            print(
                json.dumps(
                    {
                        "type": "artifact_skipped",
                        "name": ad,
                        "detail": str(exc)[:200],
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )
            )


def _artifact_olayi(op: str, kunye: dict) -> None:
    """Artifact teması için AYRI bir log satırı.

    `tool_call` satırı zaten yazılıyor ama o "hangi tool çağrıldı" der; bu satır
    "hangi VERİ nereden geldi / nereye gitti" der — artifact_id, ad, boyut ve
    lineage. `sandbox_runner` bunu `SandboxRun.artifacts`'a, oradan da `Trace`'e
    besler; web panelinin artifact göstergesi bu satıra dayanır.
    """
    print(
        json.dumps(
            {
                "type": "artifact",
                "op": op,
                "artifact_id": kunye.get("artifact_id"),
                "name": kunye.get("name"),
                "size_bytes": kunye.get("size_bytes"),
                "content_type": kunye.get("content_type"),
                "parents": kunye.get("parents") or [],
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
    )


def _artifact_api(istemci) -> tuple[dict, dict]:
    """İki ayrı sözlük döner:
      (1) LLM'in globals'ında göreceği beş fonksiyon (public)
      (2) yalnızca `_ciktilari_supur`'un kullandığı, LLM'e HİÇ görünmeyen
          `_put_file` (internal)

    Ayrı tutuluyor çünkü `main()` (1)'i `sandbox_globals`'a serpiştiriyor —
    `_put_file` oraya karışsaydı LLM'in kod tamamlama/keşif sırasında
    görebileceği, amacı dışı bir fonksiyon olurdu.

    Taşıma artık Artifact Service'e akışlı HTTP (bkz. artifact_client.py);
    fonksiyon adları BİLEREK aynı kaldı — `graph.py`'deki sistem promptu ve
    bütün dokümanlar bu adlara atıf yapıyor. §27'nin `save`/`load` biçimi
    REST katmanında yaşıyor, sandbox yüzeyinde değil."""

    def _put_file(path: str, content_type: str, name: str, ttl_seconds=None):
        """Süpürmenin yolu — dosyayı diskten akıtır, belleğe almaz."""
        sonuc = istemci.put_file(path, content_type, name, ttl_seconds)
        _artifact_olayi("produced", sonuc)
        return sonuc["artifact_id"]

    def put_artifact(value, name, parents=None, ttl_seconds=None,
                     type=None, metadata=None):  # noqa: A002
        """Bir ara/nihai çıktıyı KALICI olarak saklar, artifact_id döndürür.

        DataFrame ise Parquet'e çevrilir (tipler korunur); sözlük/liste JSON,
        metin düz metin olur. `name`, sonraki adımların onu bulacağı addır.

        `type` (2026-09-04, KFP hizalaması): `system.Dataset` / `system.Model` /
        `system.Metrics` / `system.HTML` / `system.Markdown`. Verilmezse
        içerikten çıkarılır — DataFrame/CSV → Dataset, sayısal sözlük → Metrics.
        `metadata`: serbest anahtar-değer (KFP'deki `.metadata`).
        """
        data, content_type = serialize.serialize(value)
        # Tip çıkarımı BURADA: `value` hâlâ nesne. Servise gidince yalnızca
        # bayt kalıyor ve "sayısal sözlük = metrik" bilgisi kayboluyor.
        tip = type or serialize.tip_cikar(content_type, value)
        sonuc = istemci.put_bytes(data, content_type, name, list(parents or []),
                                  ttl_seconds, tip, metadata)
        _artifact_olayi("produced", sonuc)
        return sonuc["artifact_id"]

    def get_artifact(artifact_id=None, name=None):
        """Daha önce saklanmış bir artifact'i geri okur — kimlikle ya da adıyla.

        Adla çağrılırsa o addaki EN YENİ sürüm gelir. Bulunamazsa None döner.
        """
        sonuc = istemci.get_bytes(artifact_id=artifact_id, name=name)
        if sonuc is None:
            return None
        data, kunye = sonuc
        _artifact_olayi("consumed", kunye)
        return serialize.deserialize(data, kunye["content_type"])

    def list_artifacts(node_id=None):
        """Bu workflow'da şu ana kadar ne üretilmiş — künyeler, baytlar değil."""
        return istemci.list(WORKFLOW_ID, node_id)

    def artifact_metadata(artifact_id):
        """Bir artifact'in künyesi (boyut, tip, hash, türediği artifact'ler)."""
        return istemci.metadata(artifact_id)

    def cached(name, fn, ttl_seconds=None):
        """Pahalı bir bloğu bir kez çalıştırır, sonrasında artifact'ten okur.

        Bir çalıştırma hata alıp DÜZELTİLMİŞ kodla yeniden koştuğunda buradaki
        blok tekrar ÇALIŞMAZ — sonucu depoda hazırdır. PTC her şeyi tek script'e
        ittiği için pratikte en çok işe yarayan yol budur.
        """
        var_olan = get_artifact(name=name)
        if var_olan is not None:
            return var_olan
        uretilen = fn()
        put_artifact(uretilen, name=name, ttl_seconds=ttl_seconds)
        return uretilen

    public = {
        "put_artifact": put_artifact,
        "get_artifact": get_artifact,
        "list_artifacts": list_artifacts,
        "artifact_metadata": artifact_metadata,
        "cached": cached,
    }
    internal = {"_put_file": _put_file}
    return public, internal


def main() -> None:
    with open(CODE_PATH, encoding="utf-8") as f:
        code = f.read()

    result_holder: dict = {}

    def set_result(value) -> None:
        """Sandbox kodu, nihai sonucunu bununla bildirir (research.md kontratı)."""
        result_holder["value"] = value

    # Artifact API yalnızca kapsam jetonu VE servis adresi varsa açılır.
    # Jetonsuz bir sandbox, deponun HANGİ workflow adına konuştuğunu
    # kanıtlayamaz — bu durumda tool'u hiç sunmamak, yetkisiz bir çağrının
    # serviste reddedilmesini beklemekten daha temiz.
    istemci = (
        artifact_client.ArtifactClient(artifact_client.ENDPOINT, SCOPE_TOKEN)
        if SCOPE_TOKEN and artifact_client.ENDPOINT
        else None
    )
    artifact_globals, artifact_internal = _artifact_api(istemci) if istemci else ({}, {})

    # Tembel okuma (2026-09-04): prefetch'in yerini aldı. Pod açılışında YALNIZCA
    # isim listesi çekiliyor (tek istek, N'den bağımsız); baytlar ancak
    # `pd.read_csv("/output/...")` ya da `open("/output/...")` çağrıldığında
    # iniyor. `df.to_csv(...)` ile kurulan anlaşma korunuyor, maliyeti
    # O(kullanılan) — eskiden O(hepsi) idi ve 512Mi'lık /output'u patlatıyordu.
    inenler: dict[str, tuple[float, int]] = {}
    tembel_globals: dict = {}
    if istemci:
        tembel_globals = _tembel_okumayi_kur(istemci, _manifest(istemci), inenler)

    sandbox_globals: dict = {
        "set_result": set_result,
        **{name: _make_sync_tool(name) for name in ALLOWED_TOOLS},
        **artifact_globals,
        **tembel_globals,
    }

    try:
        exec(compile(code, CODE_PATH, "exec"), sandbox_globals)  # noqa: S102
    except Exception as exc:  # noqa: BLE001 - sandbox kodunun hatası, çökmeden bildirilmeli
        # Süpürme HATADAN ÖNCE: script son satırda patlasa bile, o ana kadar
        # /output'a yazılmış dosyalar kaybolmasın. Endüstri emsali (Anthropic
        # $OUTPUT_DIR, OpenAI /mnt/data) bunu başarı/hata ayrımı yapmadan
        # yapıyor — çıktı, çalıştırmanın sonucundan bağımsız bir kavram.
        if artifact_internal:
            _ciktilari_supur(artifact_internal, inenler)
        print(json.dumps({"status": "error", "message": str(exc)}))
        sys.exit(0)

    if artifact_internal:
        _ciktilari_supur(artifact_internal, inenler)

    if "value" in result_holder:
        print(json.dumps({"status": "success", "result": result_holder["value"]}))
    else:
        print(json.dumps({"status": "error", "message": "kod set_result() çağırmadı"}))


if __name__ == "__main__":
    main()
