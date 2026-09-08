"""Sandbox entrypoint — LLM'in ürettiği kodu çalıştırır (Faz 2).

Bilerek Python-seviyesinde bir kısıtlama (RestrictedPython, builtins filtresi
vb.) YOK — research.md §4.3'teki karar: enforcement Cilium'da (network
seviyesinde), burada değil. Kod istediği kütüphaneyi import edebilir, ama
Tool Gateway dışında hiçbir yere çıkamaz (Cilium bunu kernel'de engeller).

## Artifact'ler burada YÖNETİLMİYOR (2026-09-07)

Bu dosyada artık ne yükleme var ne indirme ne de yama. İş bölümü Argo/KFP'nin
aynısı:

    sidecar.yerlestir()  → kod BAŞLAMADAN BÜTÜN girdileri diske koyar
    bu dosya             → kodu çalıştırır; `/output` GERÇEK dosyalar içerir
    sidecar.supur()      → kod BİTİNCE `/output`'u toplar

Yani `pd.read_parquet("/output/x.parquet")` sıradan bir dosya okumasıdır;
`os.listdir("/output")` gerçeği söyler. Öncesinde bunların hepsi yamalıydı ve
bayt okuma çağrısının ortasında iniyordu — piyasada karşılığı olmayan tek
desenimizdi.

Sandbox'ta artifact için HİÇBİR fonksiyon YOK ve hiçbir ağ çağrısı yok.
Bütün girdiler — kendi çıktıları, başka çalıştırmalarınki, alias'la
sabitlenmiş sürümler — kod başlamadan diske konuyor:

    /output/<ad>              bu çalıştırmanın çıktıları
    /artifacts/<wf>/<ad>      beyan edilmiş başka çalıştırmalar
    /artifacts/_alias/<ad>    beyan edilmiş sabit sürümler

KFP'nin kullanıcı bileşenine verdiği garantinin aynısı.

Kontrat: contracts/sandbox_job_contract.md
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import re
import sys
import traceback
from datetime import UTC, datetime

from fastmcp import Client

TOOL_GATEWAY_ENDPOINT = os.environ["TOOL_GATEWAY_ENDPOINT"]
CODE_PATH = "/sandbox/code.py"

#: Kullanıcı kodunun `print`'leri yakalanırken protokol satırları BURAYA
#: yazılıyor. `redirect_stdout` global; bu kaçış olmadan `tool_call` satırları
#: da yakalanır ve runner onları hiç görmezdi.
_GERCEK_STDOUT = sys.stdout

#: Hata yükünün her parçası için üst sınır. smolagents'ın
#: `MAX_LENGTH_TRUNCATE_CONTENT`'i ile aynı sayı — sahada denenmiş bir değer.
#: (PTC_Error_Recovery_Piyasa_Arastirmasi.md §1.1)
_AZAMI_PARCA = 20_000

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


def _protokol(nesne: dict) -> None:
    """Runner'ın ayrıştırdığı JSON satırı — DAİMA gerçek stdout'a.

    Kullanıcı kodu çalışırken stdout yakalanıyor (bkz. `_kod_calistir`). Bu
    fonksiyon o yakalamanın dışında kalıyor, yoksa `tool_call` satırları
    kullanıcının `print` çıktısının içine düşer ve runner olayları kaybederdi.
    """
    print(json.dumps(nesne), file=_GERCEK_STDOUT, flush=True)


def _kirp(metin: str, ne: str, azami: int = _AZAMI_PARCA) -> str:
    """Uzun metni ORTADAN keser ve kesildiğini SÖYLER.

    Ortadan kesmenin gerekçesi traceback'in şeklinde: baş "nereden başladı",
    son "asıl hata" — değerli olan iki uç, gürültü ortada. smolagents ve
    Codex de tam olarak bunu yapıyor (`truncate_content`,
    `truncate_middle_chars`).

    Sessiz kırpma YAPILMIYOR: model eksik veriyi tam sanmamalı. SWE-agent'ın
    `<response clipped>` + "{n} characters were elided" deseni.
    """
    if len(metin) <= azami:
        return metin
    atilan = len(metin) - azami
    return (metin[: azami // 2]
            + f"\n...[{ne} kırpıldı — {atilan} karakter atıldı, sınır {azami}]...\n"
            + metin[-azami // 2:])


def _kullanici_izi(exc: BaseException) -> str:
    """Traceback'in YALNIZCA kullanıcı kodundan gelen karelerini biçimler.

    `exec(compile(code, CODE_PATH, "exec"))` sayesinde kullanıcı karelerinin
    `filename`'i `CODE_PATH`; entrypoint'in kendi kareleri ayıklanabiliyor.
    İkisi de gitseydi model bizim iç yapımızı okurdu — hem faydasız hem de
    §4.1'deki sızıntı notuna aykırı.

    SWE-agent'ın ölçümü bu ayrıntıyı doğruluyor: hata tipi olmadan model
    yanlış teşhis koyuyor, fazla bağlam ise puan kaybettiriyor
    (Tablo 3: tüm geçmiş → -3,0 puan).
    """
    kareler = [k for k in traceback.extract_tb(exc.__traceback__)
               if k.filename == CODE_PATH]
    # Kullanıcı karesi yoksa (ör. derleme hatası) en azından tipi + mesajı ver.
    govde = "".join(traceback.format_list(kareler)) if kareler else ""
    return govde + "".join(traceback.format_exception_only(type(exc), exc))


#: Hata metninin kapanışı. İki şeyi BİRLİKTE söylüyor — sahadaki iki bağımsız
#: emsalin ortak deseni (smolagents `memory.py`, OpenHands `get_action_error_nudge`):
#: *tekrar dene* VE *aynısını tekrarlama*.
#:
#: Üçüncü cümle SWE-agent'ın katkısı: kesmekle yetinme, NE YAPACAĞINI ÖĞRET.
#: SWE-agent "head/tail/grep kullan" diyor çünkü orada bir shell var; bizde
#: yok, o yüzden tavsiye bizim yüzeyimize uyarlandı — büyük çıktıyı basmak
#: yerine `/output`'a dosya olarak yaz; süpürme onu saklıyor ve bir sonraki
#: çalıştırmada `inputs` ile beyan edilebiliyor.
_NE_YAPMALI = (
    "Hatayı düzeltip kodu TEKRAR çalıştır. Aynı kodu aynen tekrar gönderme — "
    "aynı hatayı verir. Aynı hatayı iki kez aldıysan farklı bir yaklaşım dene. "
    "Çok çıktı basıyorsan basmak yerine `/output` altına dosya olarak yaz; "
    "dosyalar kalıcı, `print` çıktısı kırpılıyor. "
    "Verileri UYDURMA — sonucu ancak kod gerçekten çalışınca bildir."
)


def _hata_metni(exc: BaseException | None, ciktilar: str,
                onsoz: str | None = None) -> str:
    """Modele gidecek hata metnini kurar.

    Sırası bilinçli: önce NE oldu (tip + iz), sonra kod oraya kadar NE yaptı
    (stdout), sonra ŞİMDİ NE YAPMALI. Üç parçanın da ayrı ayrı denendiği ve
    üçünün de gerektiği SWE-agent makalesinde ölçülmüş (Ek A, Şekil 11).
    """
    parcalar = []
    if onsoz:
        parcalar.append(onsoz)
    if exc is not None:
        parcalar.append(_kirp(_kullanici_izi(exc), "traceback").rstrip())
    if ciktilar.strip():
        parcalar.append("Hata anına kadar yazılan çıktı:\n"
                        + _kirp(ciktilar, "stdout").rstrip())
    parcalar.append(_NE_YAPMALI)
    return "\n\n".join(parcalar)


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
            _protokol({
                "type": "tool_call",
                "tool": tool_name,
                "args": _log_icin(kwargs),
                "status": "error",
                "timestamp": timestamp,
            })
            raise
        _protokol({
            "type": "tool_call",
            "tool": tool_name,
            "args": _log_icin(kwargs),
            "status": "success",
            "timestamp": timestamp,
        })
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

#: NOT: `PTC_SCOPE_TOKEN` bu container'a ARTIK VERİLMİYOR (2026-09-06).
#: Jeton sidecar'da; sandbox yalnızca 127.0.0.1'deki proxy'yi görüyor.
#: Sabiti bilerek tanımlamıyoruz — okuyanın "burada bir jeton var" sanmaması için.


#: Kapsam jetonundan çözülen workflow — yalnızca `/workflows/{id}/artifacts`
#: yolunu kurmak için. Yetki yine JETONDAN geliyor, bu değerden değil.
WORKFLOW_ID = os.environ.get("PTC_WORKFLOW_ID", "")


#: LLM'in dosya yazabileceği, çalışma sonunda SÜPÜRÜLEN dizin.
#:
#: `/scratch`ten AYRI olması kasıtlı: orası geçici alan (ara dosyalar, cache,
#: yarım çıktılar) ve süpürülse her çöp artifact'e dönerdi. Anthropic'in
#: `$OUTPUT_DIR`'ı da tam bu yüzden boş ve ayrı bir dizin.
OUTPUT_DIR = os.environ.get("PTC_OUTPUT_DIR", "/output")

#: Yazılabilir geçici alan. Dizin artifact'i paketlenirken tar BURAYA yazılıyor
#: — `/output`'a yazsaydık süpürme kendi ara dosyasını da artifact sanardı.
SCRATCH_DIR = os.environ.get("PTC_SCRATCH_DIR", "/scratch")

#: Sidecar'ın localhost proxy'si. Kapsam jetonu ONDA; bu container'da yok.
#: Sidecar'ın "girdiler yerinde" dosyası — `/scratch` iki container'da da
#: mount edilmiş ve süpürülmüyor.
HAZIR_DOSYA = os.path.join(SCRATCH_DIR, ".ptc-girdiler-hazir")

#: BAŞKA çalıştırmaların çıktıları buradan okunuyor: `/artifacts/<workflow>/<ad>`.
#:
#: NEDEN AYRI BİR KÖK (2026-09-06, canlı kullanımda bulunan arıza): keşif kapsamı
#: tenant'a genişleyince `/output` bütün çalıştırmaların çıktılarını DÜZ bir
#: liste olarak gösteriyordu. Ajan 1. turda bir analiz üretti (İK = 40,45),
#: 2. turda "az önce ürettiğin" diye sorulunca `/output`'ta gördüğü BAŞKA bir
#: run'ın `departman.ozet.parquet`'ini okuyup "7,46" dedi. Cevap sessizce
#: yanlıştı.
#:
#: KFP'de her çalıştırma `pipeline_root/<run-id>/...` altına yazar; başka bir
#: run'ın çıktısına ancak onun kimliğini içeren bir yolla ulaşılır. Aynısı:
#: `/output` = bu çalıştırma, `/artifacts/<wf>/` = adı verilen çalıştırma.
ARTIFACTS_DIR = os.environ.get("PTC_ARTIFACTS_DIR", "/artifacts")



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




def _girdileri_bekle(saniye: float = 30.0) -> bool:
    """Sidecar'ın "girdiler yerinde" dosyasını bekler.

    Önceden bu bir HTTP yoklamasıydı (`GET 127.0.0.1:8099/healthz`). Sandbox'ın
    artifact için başka hiçbir ağ çağrısı kalmayınca, yalnızca el sıkışma
    uğruna bir sunucu ayakta tutmanın anlamı kalmadı. Argo da 1.29 öncesinde
    sonlandırma sinyalini paylaşılan volume'deki bir dosyayla veriyordu.
    """
    import time  # noqa: PLC0415

    son = time.monotonic() + saniye
    while time.monotonic() < son:
        if os.path.exists(HAZIR_DOSYA):
            return True
        time.sleep(0.05)
    return False


#: Sidecar'ın yerleştirmeyi bitirip sunucuyu açması için beklenecek süre.
#: Yerleştirme ağdan indirme içerdiği için proxy'nin salt açılmasından uzun
#: sürebilir; `activeDeadlineSeconds: 90` içinde rahat kalıyor.
_SIDECAR_BEKLEME = 30.0

def main() -> None:
    with open(CODE_PATH, encoding="utf-8") as f:
        code = f.read()

    result_holder: dict = {}

    def set_result(value) -> None:
        """Sandbox kodu, nihai sonucunu bununla bildirir (research.md kontratı)."""
        result_holder["value"] = value

    # GİRDİLER KOD BAŞLAMADAN YERLEŞTİRİLMİŞ OLMALI (KFP deseni).
    #
    # Sidecar bütün girdileri — kendi çıktıları, başka çalıştırmalarınki,
    # alias'la sabitlenmiş sürümler — diske koyduktan SONRA hazır dosyasını
    # yazıyor. Yani dosyanın varlığı "girdiler yerinde" demek.
    #
    # Bu el sıkışma YÜK TAŞIYOR: dosya gelmezse `/output` yarım kalmış
    # olabilir. Sessizce devam etmek, tam da kovaladığımız "sessizce yanlış"
    # arızası olurdu — o yüzden çalıştırma açık hatayla bitiyor.
    if not _girdileri_bekle(_SIDECAR_BEKLEME):
        print(json.dumps({
            "status": "error",
            "message": "artifact sidecar hazır değil — girdiler yerleştirilemedi",
        }))
        sys.exit(0)

    # Artifact için HİÇBİR fonksiyon yok. Kod yalnızca dosya görüyor:
    #   /output/<ad>              bu çalıştırmanın çıktıları
    #   /artifacts/<wf>/<ad>      beyan edilmiş başka çalıştırmalar
    #   /artifacts/_alias/<ad>    beyan edilmiş sabit sürümler
    sandbox_globals: dict = {
        "set_result": set_result,
        **{name: _make_sync_tool(name) for name in ALLOWED_TOOLS},
    }

    # SÜPÜRME BURADA DEĞİL. `/output`'a yazılanları sidecar topluyor: ana
    # container bittikten sonra kubelet ona SIGTERM gönderiyor ve süpürme o
    # anda çalışıyor. Hata yolunda da öyle — bu container nasıl bitmiş olursa
    # olsun, çıktılar kurtarılıyor.
    # KULLANICININ `print`'LERİ YAKALANIYOR.
    #
    # 2026-09-08'e kadar `print` çıktısı pod log'una gidiyor ve modele HİÇ
    # ulaşmıyordu; hata anında kodun oraya kadar ne yaptığı kayboluyordu.
    # smolagents bunu kaybetmemek için ayrı kod yazmış ve ayrı bir test
    # tutuyor (`test_error_saves_previous_print_outputs`) — yani bilinçli bir
    # karar, kaza değil.
    #
    # `_protokol` bu yakalamanın DIŞINDA kalıyor: tool_call satırları gerçek
    # stdout'a gidiyor, runner onları görmeye devam ediyor.
    yakalanan = io.StringIO()
    try:
        with contextlib.redirect_stdout(yakalanan):
            exec(compile(code, CODE_PATH, "exec"), sandbox_globals)  # noqa: S102
    except Exception as exc:  # noqa: BLE001 - sandbox kodunun hatası, çökmeden bildirilmeli
        _protokol({
            "status": "error",
            # `message` runner'ın `error_message`'ına, oradan da doğrudan
            # modele gidiyor — zenginleştirilecek yer burası.
            "message": _hata_metni(exc, yakalanan.getvalue()),
            "error_type": type(exc).__name__,
        })
        sys.exit(0)

    ciktilar = yakalanan.getvalue()
    if "value" in result_holder:
        _protokol({"status": "success", "result": result_holder["value"],
                   **({"stdout": _kirp(ciktilar, "stdout")} if ciktilar else {})})
    else:
        _protokol({
            "status": "error",
            "message": _hata_metni(None, ciktilar,
                                   "Kod `set_result(...)` çağırmadı — nihai değer "
                                   "bu fonksiyonla bildirilir."),
            "error_type": "SonucYok",
        })


if __name__ == "__main__":
    main()
