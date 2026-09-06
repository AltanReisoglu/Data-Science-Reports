"""Sandbox entrypoint — LLM'in ürettiği kodu çalıştırır (Faz 2).

Bilerek Python-seviyesinde bir kısıtlama (RestrictedPython, builtins filtresi
vb.) YOK — research.md §4.3'teki karar: enforcement Cilium'da (network
seviyesinde), burada değil. Kod istediği kütüphaneyi import edebilir, ama
Tool Gateway dışında hiçbir yere çıkamaz (Cilium bunu kernel'de engeller).

Kontrat: contracts/sandbox_job_contract.md
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import tarfile
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

#: Dizin artifact'lerinin ad soneki. Paketleme ARTIK BURADA DEĞİL — süpürme
#: sidecar'a taşındı (bkz. sidecar.py). Burada yalnızca AÇMA tarafı var.
_DIZIN_SONEKI = ".tar"

#: Sidecar'ın localhost proxy'si. Kapsam jetonu ONDA; bu container'da yok.
PROXY_URL = os.environ.get("PTC_ARTIFACT_PROXY", "")

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

#: `os.listdir` / `os.path.exists` LLM'in kodu için YAMALANIYOR (manifestteki
#: isimler de görünsün diye). Launcher'ın kendisi gerçeği görmek ZORUNDA:
#: yamalı sürümü kullansaydı, henüz inmemiş bir ismi `/output`'ta var sanıp
#: süpürmeye çalışırdı. Bu yüzden orijinaller import anında saklanıyor.
_GERCEK_LISTDIR = os.listdir
_GERCEK_EXISTS = os.path.exists

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


#: `art_` + 12 hex — `service.py`'deki `"art_" + uuid4().hex[:12]` ile birebir.
_KIMLIK_BICIMI = re.compile(r"^art_[0-9a-f]{12}$")


def _kimlik_mi(deger) -> bool:
    return isinstance(deger, str) and bool(_KIMLIK_BICIMI.match(deger))


class Depo:
    """Sandbox'ın gördüğü depo görünümü — tek istekle kurulur.

    `kendi`      : bu çalıştırmanın ürettiği adlar   -> /output/<ad>
    `digerleri`  : workflow_id -> adlar              -> /artifacts/<wf>/<ad>

    Baytlar İNMİYOR, yalnızca isimler. Maliyet artifact sayısından ve
    boyutundan bağımsız.
    """

    __slots__ = ("kendi", "digerleri")

    def __init__(self, kendi=None, digerleri=None):
        self.kendi = kendi or set()
        self.digerleri = digerleri or {}

    def bos(self) -> bool:
        return not self.kendi and not self.digerleri


def _manifest(istemci) -> Depo:
    """Depoda ne var — YALNIZCA isimler, tek istek.

    Servise ulaşılamazsa boş görünüm döner: pod açılışı ASLA buna bağlı
    olmamalı, yalnızca keşif ve tembel okuma sessizce devre dışı kalır.
    """
    try:
        kayitlar = istemci.list_all()
    except Exception:  # noqa: BLE001
        return Depo()
    kendi, digerleri = set(), {}
    for k in kayitlar:
        ad, wf = k.get("name"), k.get("workflow_id") or ""
        if not ad:
            continue
        # `wf == WORKFLOW_ID` boş-boş eşleşmesini de kapsıyor: kapsamı
        # bilinmeyen bir kurulumda her şeyi "kendi" saymak, hiçbirini
        # göstermemekten iyi — ilk hâli künyeyi sessizce DÜŞÜRÜYORDU.
        if wf == WORKFLOW_ID:
            kendi.add(ad)
        elif wf:
            digerleri.setdefault(wf, set()).add(ad)
    return Depo(kendi, digerleri)


def _yolu_coz(yol) -> tuple[str, str, str] | None:
    """Bir yolu (`tur`, `workflow_id`, `ad`) üçlüsüne çevirir; ilgisizse None.

    tur: "cikti"    -> /output/<ad>            (bu çalıştırma)
         "baska"    -> /artifacts/<wf>/<ad>    (adı verilen çalıştırma)
    """
    if not isinstance(yol, (str, bytes, os.PathLike)):
        return None
    try:
        metin = os.fspath(yol)
    except TypeError:
        return None
    if isinstance(metin, bytes):
        metin = metin.decode("utf-8", "replace")
    tam = os.path.abspath(metin)

    cikti = os.path.abspath(OUTPUT_DIR)
    if os.path.dirname(tam) == cikti:
        return ("cikti", WORKFLOW_ID, os.path.basename(tam))

    art = os.path.abspath(ARTIFACTS_DIR)
    if tam.startswith(art + os.sep):
        parcalar = os.path.relpath(tam, art).split(os.sep)
        if len(parcalar) == 2:
            return ("baska", parcalar[0], parcalar[1])
    return None


def _tembel_dizin_ac(tam: str, istemci, depo) -> None:
    """`<kök>/<dizin>/...` istendi — `<dizin>.tar` varsa indir ve aç.

    İKİ kökü de tanır (2026-09-06):
        /output/<dizin>/...            -> bu çalıştırmanın dizin artifact'i
        /artifacts/<wf>/<dizin>/...    -> adı verilen çalıştırmanınki

    "Bunu ben indirdim, LLM üretmedi" defteri ARTIK BURADA TUTULMUYOR —
    sidecar sunduğu her baytın sha256'sını kendisi kaydediyor ve süpürmede ona
    bakıyor. Kayıt sandbox'ta olmadığı için kurcalanamıyor da.
    """
    cikti = os.path.abspath(OUTPUT_DIR)
    art = os.path.abspath(ARTIFACTS_DIR)

    if tam.startswith(cikti + os.sep):
        gorece, wf, havuz, kok = os.path.relpath(tam, cikti), WORKFLOW_ID, depo.kendi, cikti
    elif tam.startswith(art + os.sep):
        parcalar = os.path.relpath(tam, art).split(os.sep)
        if len(parcalar) < 3:
            return                       # /artifacts/<wf>/<dizin>/<dosya> gerekiyor
        wf = parcalar[0]
        gorece = os.sep.join(parcalar[1:])
        havuz = depo.digerleri.get(wf, set())
        kok = os.path.join(art, wf)
    else:
        return

    parcalar = gorece.split(os.sep)
    if len(parcalar) < 2 or parcalar[0] in ("..", "."):
        return
    dizin_adi = parcalar[0]
    hedef = os.path.join(kok, dizin_adi)
    if _GERCEK_EXISTS(hedef):
        return  # zaten inmiş; istenen dosya gerçekten yok
    arsiv = _gecerli_artifact_adi(dizin_adi + _DIZIN_SONEKI)
    if arsiv not in havuz:
        return

    paket = os.path.join(SCRATCH_DIR, f"_inen_{arsiv}")
    try:
        kunye = istemci.fetch_to_file(arsiv, paket, workflow_id=wf)
        if not kunye:
            return
        os.makedirs(hedef, exist_ok=True)
        _tari_ac(paket, hedef)
    except Exception:  # noqa: BLE001 — inemezse dosya yok gibi davran
        return
    finally:
        if _GERCEK_EXISTS(paket):
            os.unlink(paket)


def _tembel_oku(yol, istemci, depo) -> None:
    """Bir yol istendi ama dosya yok — depoda varsa indir.

    İki yol biçimi tanınıyor:
      `/output/<ad>`          -> BU çalıştırmanın çıktısı
      `/artifacts/<wf>/<ad>`  -> adı verilen çalıştırmanın çıktısı

    Yalnızca manifestte adı geçen dosyalar indirilir; yani var olmayan bir
    dosya için ağa çıkılmaz, `FileNotFoundError` normal şekilde yükselir.

    "İndirdiğimiz dosyayı süpürme geri yüklemesin" kuralı ARTIK BURADA DEĞİL:
    sidecar sunduğu her baytın sha256'sını tutuyor ve süpürmede ona bakıyor
    (2026-09-06). Defter sandbox'ta olmadığı için kurcalanamıyor.
    """
    if not isinstance(yol, (str, bytes, os.PathLike)):
        return  # dosya nesnesi, URL, tampon — bizim işimiz değil
    metin = os.fspath(yol)
    if isinstance(metin, bytes):
        metin = metin.decode("utf-8", "replace")
    if _GERCEK_EXISTS(metin):
        return

    coz = _yolu_coz(metin)
    if coz is None:
        # DİZİN ARTIFACT'İ: `/output/model.v1/weights.json` isteniyor ama
        # `/output/model.v1` daha inmemiş olabilir.
        _tembel_dizin_ac(os.path.abspath(metin), istemci, depo)
        return

    tur, wf, ham_ad = coz
    ad = _gecerli_artifact_adi(ham_ad)
    havuz = depo.kendi if tur == "cikti" else depo.digerleri.get(wf, set())
    if ad not in havuz:
        return

    os.makedirs(os.path.dirname(os.path.abspath(metin)), exist_ok=True)
    try:
        kunye = istemci.fetch_to_file(ad, metin, workflow_id=wf)
    except Exception:  # noqa: BLE001 — indirilemezse dosya yok gibi davran
        return
    # `consumed` olayını da sidecar yayınlıyor — baytı o sunuyor.


def _tembel_okumayi_kur(istemci, depo) -> dict:
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
            setattr(
                pd, fn_adi,
                _okuyucu_sarmala(orijinal, istemci, depo),
            )

    # -- KEŞİF: `/output` artık yalan söylememeli -------------------------
    #
    # LLM'e sunulan artifact fonksiyonları kaldırıldığı için (2026-09-06)
    # keşfin TEK yolu dosya sistemi kaldı. Ama `/output` pod açılışında
    # fiziksel olarak BOŞ — baytlar ancak adıyla istenince iniyor.
    #
    # Bu, düz Python yazan bir modeli doğrudan yanıltıyordu:
    #     os.listdir("/output")   -> []      (oysa depoda 5 artifact var)
    #     os.path.exists("/output/satislar.parquet") -> False
    # Model "hiçbir şey yok" sonucuna varıp veriyi yeniden üretiyordu.
    #
    # Üçü de manifestle BİRLEŞTİRİLİYOR: dosya sistemi artık deponun görünümü.
    _yamala_kesif(depo)

    gercek_open = open

    def _tembel_open(dosya, kip="r", *args, **kwargs):
        if "r" in kip and "+" not in kip:
            _tembel_oku(dosya, istemci, depo)
        return gercek_open(dosya, kip, *args, **kwargs)

    return {"open": _tembel_open}


def _yamala_kesif(depo) -> None:
    """`os.listdir`, `os.path.exists` ve `glob` depoyu da görsün.

    Yama MODÜL düzeyinde olmak zorunda: LLM'in kodu `import os` deyince
    `sys.modules['os']`'u alıyor, bizim verdiğimiz bir kopyayı değil. Aynı
    sebep pandas okuyucularının yamalanmasındakiyle bir.

    İki kök ayrı gösteriliyor:
        /output                -> bu çalıştırmanın çıktıları
        /artifacts             -> başka çalıştırmaların kimlikleri
        /artifacts/<wf>        -> o çalıştırmanın çıktıları

    Launcher kendi işini `_GERCEK_LISTDIR` / `_GERCEK_EXISTS` ile yapıyor,
    yani bu yamadan etkilenmiyor.
    """
    import glob as _glob  # noqa: PLC0415

    def _sanal_liste(yol: str):
        """Bu yol için depodan gelen adlar; ilgisizse None."""
        tam = os.path.abspath(yol)
        if tam == os.path.abspath(OUTPUT_DIR):
            return depo.kendi
        art = os.path.abspath(ARTIFACTS_DIR)
        if tam == art:
            return set(depo.digerleri)
        if os.path.dirname(tam) == art:
            return depo.digerleri.get(os.path.basename(tam), set())
        return None

    def listdir(path="."):
        gercek = _GERCEK_LISTDIR(path) if _GERCEK_EXISTS(path) else []
        sanal = _sanal_liste(os.fspath(path))
        return sorted(set(gercek) | sanal) if sanal is not None else gercek

    def exists(path):
        if _GERCEK_EXISTS(path):
            return True
        if _sanal_liste(path) is not None:
            return True          # dizinin kendisi (/output, /artifacts, /artifacts/<wf>)
        coz = _yolu_coz(path)
        if coz is None:
            return False
        tur, wf, ad = coz
        havuz = depo.kendi if tur == "cikti" else depo.digerleri.get(wf, set())
        return ad in havuz

    def _genisle(sonuc, kalip):
        dizin = os.path.dirname(os.path.abspath(os.fspath(kalip)))
        sanal = _sanal_liste(dizin)
        if sanal is None:
            return sonuc
        import fnmatch  # noqa: PLC0415

        desen = os.path.basename(os.fspath(kalip))
        ek = [os.path.join(dizin, ad) for ad in sanal if fnmatch.fnmatch(ad, desen)]
        return sorted(set(sonuc) | set(ek))

    gercek_glob, gercek_iglob = _glob.glob, _glob.iglob
    _glob.glob = lambda kalip, *a, **kw: _genisle(gercek_glob(kalip, *a, **kw), kalip)
    _glob.iglob = lambda kalip, *a, **kw: iter(
        _genisle(list(gercek_iglob(kalip, *a, **kw)), kalip))

    os.listdir = listdir
    os.path.exists = exists


#: Dosya başlangıcındaki imza → o biçimi okuyan pandas fonksiyonu.
#: Parquet "PAR1", Arrow IPC "ARROW1" ile başlar.
_BICIM_IMZALARI = ((b"PAR1", "read_parquet"), (b"ARROW1", "read_feather"))


def _bicim_uyari(yol: str, cagrilan: str) -> str | None:
    """Yanlış okuyucu kullanıldıysa açıklayıcı mesaj döner, yoksa None.

    NEDEN (2026-09-04, canlı kullanımda bulundu): `put_artifact(df, ...)`
    DataFrame'i Parquet'e çeviriyor, ama artifact ADI uzantı taşımıyor
    ("ticket.durumlari"). Tembel doldurma o adı dosya olarak koyunca model
    `pd.read_csv("/output/ticket.durumlari")` deniyor ve şu hatayı alıyor:

        'utf-8' codec can't decode byte 0xe4 in position 106

    Bu mesajdan ne olduğu anlaşılmıyor; model "artifact bozuk" sanıp vazgeçti.
    Doğru çağrıyı söylemek, modelin kendini düzeltebilmesi için yeterli.
    """
    try:
        with open(yol, "rb") as f:
            bas = f.read(8)
    except OSError:
        return None
    for imza, dogru in _BICIM_IMZALARI:
        if bas.startswith(imza) and cagrilan != dogru:
            return (
                f"{yol} bir {imza.decode()} dosyası ama {cagrilan}() ile açılmaya "
                f"çalışıldı. Bunun yerine {dogru}('{yol}') kullanın — ya da daha "
                f"kolayı: get_artifact('{os.path.basename(yol)}') doğrudan "
                "DataFrame döndürür."
            )
    return None


def _okuyucu_sarmala(orijinal, istemci, depo):
    ad = getattr(orijinal, "__name__", "")

    def _oku(yol=None, *args, **kwargs):
        if yol is not None:
            _tembel_oku(yol, istemci, depo)
            if isinstance(yol, str):
                uyari = _bicim_uyari(yol, ad)
                if uyari:
                    raise ValueError(uyari)
        return orijinal(yol, *args, **kwargs)

    _oku.__name__ = getattr(orijinal, "__name__", "read")
    _oku.__doc__ = getattr(orijinal, "__doc__", None)
    return _oku


def _tari_ac(tar_yolu: str, hedef_dizin: str) -> None:
    """Tar'ı hedef dizine açar — yol geçişine karşı süzülmüş.

    `filter="data"` (CVE-2007-4559 karşılığı) arşiv dışına yazan girdileri,
    symlink'leri ve aygıt düğümlerini reddediyor. Arşivi biz üretmiş olsak da
    `put_artifact` ile depoya BAŞKA bir tar girmiş olabilir; açan taraf
    kaynağına güvenmemeli.
    """
    with tarfile.open(tar_yolu, "r") as tar:
        try:
            tar.extractall(hedef_dizin, filter="data")  # noqa: S202
        except TypeError:  # `filter` 3.11.4'ten eski sürümlerde yok
            for uye in tar.getmembers():
                if not uye.isfile() or os.path.isabs(uye.name) or ".." in uye.name.split("/"):
                    continue
                tar.extract(uye, hedef_dizin)  # noqa: S202


def _proxy_bekle(taban: str, saniye: float = 10.0) -> bool:
    """Sidecar'ın localhost sunucusu açılana kadar bekler."""
    import time  # noqa: PLC0415

    import requests  # noqa: PLC0415

    son = time.monotonic() + saniye
    while time.monotonic() < son:
        try:
            if requests.get(f"{taban.rstrip('/')}/healthz", timeout=1).status_code == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.1)
    return False


def main() -> None:
    with open(CODE_PATH, encoding="utf-8") as f:
        code = f.read()

    result_holder: dict = {}

    def set_result(value) -> None:
        """Sandbox kodu, nihai sonucunu bununla bildirir (research.md kontratı)."""
        result_holder["value"] = value

    # OKUMA yolu: 127.0.0.1'deki sidecar'a. Kapsam jetonu BU CONTAINER'DA YOK
    # (2026-09-06) — sidecar taşıyor. Proxy'de yükleme uç noktası da yok:
    # neyin yükleneceğine sidecar `/output`'a bakarak karar veriyor, tıpkı
    # Argo'nun `wait` container'ı gibi. LLM'in kodunun etkileyebileceği tek
    # şey dosya yazmak — yani zaten kastedilen arayüz.
    istemci = artifact_client.ProxyClient(PROXY_URL) if PROXY_URL else None
    if istemci and not _proxy_bekle(PROXY_URL):
        # Sidecar HTTP sunucusunu henüz açmamış olabilir — main container onunla
        # AYNI ANDA başlıyor. Beklemezsek `_manifest` boş dönüyor ve tembel
        # okuma SESSİZCE devre dışı kalıyordu (yarış koşulu).
        print(json.dumps({"type": "proxy_hazir_degil", "url": PROXY_URL}), flush=True)
        istemci = None

    # Tembel okuma: prefetch'in yerini aldı. Pod açılışında YALNIZCA isim
    # listesi çekiliyor (tek istek, N'den bağımsız); baytlar ancak
    # `pd.read_parquet("/output/...")` çağrıldığında iniyor — maliyet
    # O(kullanılan), eskiden O(hepsi) idi ve 512Mi'lık /output'u patlatıyordu.
    tembel_globals: dict = {}
    if istemci:
        tembel_globals = _tembel_okumayi_kur(istemci, _manifest(istemci))

    sandbox_globals: dict = {
        "set_result": set_result,
        **{name: _make_sync_tool(name) for name in ALLOWED_TOOLS},
        **tembel_globals,
    }

    # SÜPÜRME BURADA DEĞİL. `/output`'a yazılanları sidecar topluyor: ana
    # container bittikten sonra kubelet ona SIGTERM gönderiyor ve süpürme o
    # anda çalışıyor. Hata yolunda da öyle — bu container nasıl bitmiş olursa
    # olsun, çıktılar kurtarılıyor.
    try:
        exec(compile(code, CODE_PATH, "exec"), sandbox_globals)  # noqa: S102
    except Exception as exc:  # noqa: BLE001 - sandbox kodunun hatası, çökmeden bildirilmeli
        print(json.dumps({"status": "error", "message": str(exc)}))
        sys.exit(0)

    if "value" in result_holder:
        print(json.dumps({"status": "success", "result": result_holder["value"]}))
    else:
        print(json.dumps({"status": "error", "message": "kod set_result() çağırmadı"}))


if __name__ == "__main__":
    main()
