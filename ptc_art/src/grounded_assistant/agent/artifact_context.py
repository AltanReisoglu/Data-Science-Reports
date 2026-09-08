"""Artifact manifestini modelin context'ine enjekte eder (2026-09-04).

## Neden var — çözdüğü gerçek problem

Bundan önce keşif **yumuşak garantiydi**: sistem promptu modele
"`list_artifacts()` çağır" diyordu, ama çağırmayı model seçmek zorundaydı.
Unutursa, depoda duran veriyi yeniden üretiyordu — artifact persistence'ın
bütün amacı da tam olarak bunu önlemekti.

## Kopyalanan desen: Google ADK `LoadArtifactsTool`

Bu problemin sahadaki tek birinci-sınıf çözümü ADK'da. Üç kuralı var ve
üçünü de burada uyguluyoruz:

1. **İsimler HER ZAMAN context'te.** ADK'nın ifadesiyle *"lists available
   artifacts in the model instructions"*. Ucuz (birkaç yüz token) ve model
   unutamaz.
2. **İçerik TALEP ÜZERİNE.** Model `get_artifact`/`read_csv` çağırınca geliyor.
3. **İçerik geçmişe KALICI yazılmıyor.** ADK içeriği yalnızca o isteğe geçici
   ekliyor. Bizde bu **yapısal olarak zaten** böyle: artifact baytları hiçbir
   zaman LLM context'ine girmiyor, sandbox'ın içinde kalıyor. Yani üçüncü
   kuralı bedavaya sağlıyoruz.

OpenShift'te bu problemin yerleşik bir cevabı YOK (KFP'de artifact var ama ajan
yok; Llama Stack'te ajan var ama workflow-artifact'i yok), o yüzden referans
olarak ADK alındı — bkz. PTC_Piyasa_Mentaliteleri.md §5.3 ve §8.5.

## Ağ yolu

Ajan süreci artifact servisine HTTP ile ulaşıyor. Adres `ARTIFACT_SERVICE_URL`
ortam değişkeninden geliyor:

  - **Cluster içinde** (üretim): `http://artifact-service:8080`
  - **Laptop'tan** (yerel geliştirme): `kubectl port-forward svc/artifact-service
    8080:8080` ve `http://localhost:8080`

Tanımlı değilse ya da ulaşılamıyorsa enjeksiyon **sessizce atlanıyor** ve
davranış eskisine dönüyor (model `list_artifacts()`'i yine çağırabilir). Bir
manifest çekilemedi diye kullanıcının sorusunu cevapsız bırakmak orantısız
olurdu.
"""

from __future__ import annotations

import os
import sys

import requests
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage

#: Prompt'a en fazla bu kadar artifact yazılır. Sınır var çünkü manifest her
#: model çağrısında context'e giriyor — 200 artifactlik bir liste, ucuz olması
#: gereken şeyi pahalı yapardı. Aşılırsa en yeniler gösteriliyor (liste
#: yeniden-eskiye sıralı gelir) ve modele "daha var" deniyor.
_AZAMI_SATIR = 40

_ZAMAN_ASIMI = (2, 5)  # (bağlantı, okuma) — ajanın turunu bekletmemeli


#: Adres yokluğu bir kez uyarılır — her model çağrısında değil.
_ADRES_UYARILDI = False


def servis_adresi() -> str:
    """`ARTIFACT_SERVICE_URL` — yoksa manifest devre dışı, ama SESSİZ değil.

    2026-09-06: bu değişken CLI'da hiç tanımlı değildi, `kunyeleri_getir` None
    dönüyordu ve manifest enjeksiyonu **hiç çalışmıyordu** — ajan artifact'leri
    yalnızca sandbox içindeki dosya sisteminden görüyordu. Hiçbir yerde hata
    yoktu; özelliğin çalışmadığı ancak davranıştan anlaşıldı.
    """
    global _ADRES_UYARILDI
    adres = os.environ.get("ARTIFACT_SERVICE_URL", "").rstrip("/")
    if not adres and not _ADRES_UYARILDI:
        _ADRES_UYARILDI = True
        print("[uyarı] ARTIFACT_SERVICE_URL tanımlı değil — artifact manifesti "
              "modele enjekte EDİLMEYECEK.", file=sys.stderr)
    return adres


def kunyeleri_getir(workflow_id: str, scope_token: str) -> list[dict] | None:
    """Tenant'taki artifact künyelerini çeker. Ulaşılamazsa None.

    Kapsam workflow değil TENANT (2026-09-06): başka bir çalıştırmanın çıktısı
    da listeleniyor, çünkü sandbox onu okuyabiliyor. Manifest, sandbox'ın
    gördüğü dosya sisteminin aynısını göstermek zorunda — aksi hâlde model
    promptta olmayan ama `/output`'ta duran bir dosyayı hiç aramazdı.
    """
    adres = servis_adresi()
    if not adres or not workflow_id or not scope_token:
        return None
    try:
        yanit = requests.get(
            f"{adres}/artifacts",
            headers={"X-Scope-Token": scope_token},
            timeout=_ZAMAN_ASIMI,
        )
        if yanit.status_code != 200:
            return None
        return yanit.json()
    except Exception:  # noqa: BLE001 — ağ hatası turu bozmamalı
        return None


#: Arama sonucunda en fazla bu kadar satır. Manifestin 40'ından ayrı ve daha
#: dar: manifest HER çağrıda context'e giriyor, arama sonucu ise yalnızca
#: modelin sorduğu turda.
_AZAMI_ARAMA = 25


def ara(workflow_id: str, scope_token: str, *, ad: str | None = None,
        tip: str | None = None, metin: str | None = None) -> str:
    """Kayıt defterinde SÜZGEÇLİ arama — sonuç yalnızca isim/tip/boyut.

    ## Neden manifest yetmiyordu

    Manifest her model çağrısında context'e giriyor, o yüzden `_AZAMI_SATIR`
    (40) ile kırpılmak zorunda. Depoda 300'den fazla künye varken model geri
    kalanı **hiç göremiyordu** — "geçen ay ürettiğim raporu bul" gibi bir
    istek, dosya listede kalmadıysa karşılıksız kalıyordu.

    Manifesti büyütmek yanlış çözüm: ucuz olması gereken şeyi pahalı yapardı.
    Doğrusu, aramayı MODELİN İSTEĞİNE bırakmak.

    ## Emsali

    MLMD `ListOptions(filter_query=...)` — KFP'nin kayıt defterinde tam bu var.
    Google ADK'da karşılığı `list_artifact_keys()`; ADK onu tool olarak
    sunmuyor ama `LoadArtifactsTool` isimleri talimatlara koyuyor. Biz ikisini
    birden yapıyoruz: isimler manifestte (ucuz, her turda), arama tool'da
    (model isteyince).

    BAYT DÖNMÜYOR — kural manifestle aynı: isim ucuz, içerik pahalı. Baytlar
    ancak `inputs` ile BEYAN edilince, kod başlamadan sandbox'a konuyor.
    """
    adres = servis_adresi()
    if not adres or not scope_token:
        return "Artifact servisi yapılandırılmamış — arama yapılamadı."
    parametreler = {k: v for k, v in
                    (("name", ad), ("type", tip), ("q", metin)) if v}
    if not parametreler:
        return ("En az bir süzgeç ver: ad (tam ad), tip (system.Dataset gibi) "
                "ya da metin (ad içinde arar).")
    try:
        yanit = requests.get(f"{adres}/artifacts", params={**parametreler,
                                                          "limit": 200},
                             headers={"X-Scope-Token": scope_token},
                             timeout=_ZAMAN_ASIMI)
        if yanit.status_code != 200:
            return f"Arama başarısız (HTTP {yanit.status_code})."
        kayitlar = yanit.json()
    except Exception as exc:  # noqa: BLE001 — ağ hatası turu bozmamalı
        return f"Artifact servisine ulaşılamadı: {type(exc).__name__}"

    if not kayitlar:
        okunur = ", ".join(f"{k}={v!r}" for k, v in parametreler.items())
        return (f"{okunur} ile eşleşen artifact YOK. Daha geniş bir süzgeç "
                "dene (ör. `metin` ile adın bir parçası), ya da veri "
                "gerçekten yoksa ÜRETMEN gerekiyor.")

    satirlar = []
    for k in kayitlar[:_AZAMI_ARAMA]:
        kuyruk = (f"({(k.get('type') or '').removeprefix('system.')}, "
                  f"{k.get('size_bytes') or 0} bayt)")
        # Kopyalanabilir BEYAN satırı — model onu doğrudan `inputs`'a koyar.
        # Manifestte de aynı biçim; iki yerde iki farklı biçim olsaydı model
        # hangisini kopyalayacağını karıştırırdı.
        if k.get("alias"):
            satirlar.append(f'  inputs=["{k["name"]}@{k["alias"]}"]  {kuyruk}')
        elif k.get("workflow_id") == workflow_id:
            satirlar.append(f'  /output/{k["name"]}  {kuyruk}  ← bu oturum')
        else:
            satirlar.append(
                f'  inputs=["{k["workflow_id"]}/{k["name"]}"]  {kuyruk}')

    bas = f"{len(kayitlar)} eşleşme"
    if len(kayitlar) > _AZAMI_ARAMA:
        bas += f" (ilk {_AZAMI_ARAMA} gösteriliyor — süzgeci daralt)"
    return (bas + ":\n" + "\n".join(satirlar)
            + "\n\nBu satırlardan birini `run_ptc_code`'un `inputs`'una "
              "OLDUĞU GİBİ koy; dosya kod başlamadan yerine konur. "
              "`/output/` ile başlayanlar zaten bu oturumun çıktısı.")


def manifest_metni(kunyeler: list[dict], workflow_id: str | None = None) -> str | None:
    """Künyeleri modele gösterilecek metne çevirir.

    BAYT YOK, yalnızca isim/tip/boyut — ADK'nın "isim ucuz, içerik pahalı"
    ayrımı.

    ## Neden İKİ GRUP (2026-09-06, canlı kullanımda bulunan arıza)

    Keşif kapsamı tenant'a genişleyince manifest DÜZ bir liste oldu ve model
    "kendi ürettiği" ile "başka bir çalıştırmanın ürettiği"ni ayırt edemedi.
    Gerçekte olan şey:

        1. tur: ajan bir analiz yapıp `ticket_analiz_raporu.pdf` üretti (HR=60,0)
        2. tur: "az önce ürettiğin analizde HR kaçtı?" diye soruldu
        -> ajan manifestte `departman.ozet.parquet` gördü, okudu, "7,46" dedi
           — oysa o dosya BAŞKA bir workflow'un çıktısıydı.

    Cevap sessizce yanlıştı; hiçbir yerde hata yoktu. Bu yüzden liste artık
    ikiye ayrılıyor ve modele hangisinin kendi işi olduğu açıkça söyleniyor.
    """
    if not kunyeler:
        return None

    # Aynı ad birden çok çalıştırmada olabilir; ilk görülen (en yeni) kalır.
    benim, digerleri, gorulen = [], [], set()
    for k in kunyeler:
        ad = k.get("name")
        if not ad or ad in gorulen:
            continue
        gorulen.add(ad)
        tip = (k.get("type") or "system.Artifact").removeprefix("system.")
        kuyruk = f"  ({tip}, {k.get('size_bytes') or 0} bayt)"
        ek = k.get("metadata") or {}
        if ek:
            kuyruk += f"  {ek}"
        wf = k.get("workflow_id") or ""
        takma = k.get("alias")
        if takma:
            # MLflow'un alias'ı: sabitlenmiş bir sürüme İSİMLE ulaşılıyor,
            # "en yeni kazanır" kuralına düşmeden. Adresi bu, o yüzden
            # manifest de bunu gösteriyor.
            kuyruk = f"  ({kuyruk.strip()[1:-1]}, alias)"
            digerleri.append(f'  inputs=["{ad}@{takma}"]{kuyruk}')
            continue
        if workflow_id and wf == workflow_id:
            benim.append(f"  /output/{ad}{kuyruk}")
        else:
            # ÇAĞRILABİLİR satır. Yolu yazmak yetmiyordu: bu dosyalar
            # `/output`'ta DEĞİL ve modelin `workflow_id`'yi başka hiçbir
            # yerden öğrenme yolu yok — manifest yazmazsa `load_artifact`
            # çağrılamaz hâle geliyor (2026-09-07'de bulundu).
            digerleri.append(f'  inputs=["{wf}/{ad}"]{kuyruk}')

    if not benim and not digerleri:
        return None

    bolumler = []
    if benim:
        kendi_kirpik = benim[:_AZAMI_SATIR]
        bolumler.append("BU OTURUMDA ÜRETİLENLER — \"az önce\", \"demin\", "
                        "\"senin ürettiğin\" dendiğinde YALNIZCA bunları kullan.\n"
                        "Pod açılırken /output'a YERLEŞTİRİLMİŞ oluyorlar; düz "
                        "dosya okuması yeter:\n"
                        + "\n".join(kendi_kirpik)
                        # Kırpma notu bu grupta da olmalı: 40'tan fazla üreten
                        # bir oturumda model listeyi tam sanıp var olan bir
                        # dosyayı yeniden üretirdi. Buradaki işaret DOĞRU —
                        # kendi çıktıların gerçekten `/output`'ta duruyor.
                        + (f"\n  … ve {len(benim) - len(kendi_kirpik)} tane daha "
                           "(os.listdir(\"/output\") ile tamamı)"
                           if len(benim) > len(kendi_kirpik) else ""))
    if digerleri:
        kirpik = digerleri[:_AZAMI_SATIR - min(len(benim), _AZAMI_SATIR // 2)]
        bolumler.append(
            "BAŞKA ÇALIŞTIRMALARDAN (aynı tenant) — BU OTURUMUN işi DEĞİL ve "
            "/output'ta BULUNMAZLAR.\nKullanıcı açıkça istemedikçe bunlara "
            "dayanma. Gerekiyorsa satırı run_ptc_code'un inputs'una olduğu gibi "
            "koy; dosya /artifacts/ altında hazır gelir:\n"
            + "\n".join(kirpik)
            + (f"\n  … ve {len(digerleri) - len(kirpik)} tane daha "
               "(gösterilmiyor)" if len(digerleri) > len(kirpik) else ""))

    return (
        "\n\n".join(bolumler)
        + "\n\nVeriyi YENİDEN ÜRETME — önce yukarıdaki listeye bak. Kendi "
        "çıktıların için özel bir çağrı YOK, sıradan dosya okuması yeter "
        "(ör. pd.read_parquet(\"/output/<ad>\")). Aradığın dosya BU OTURUMDA "
        "yoksa, başka bir çalıştırmanınkini kendi çıktın gibi sunma — üretmen "
        "gerektiğini söyle ya da üret."
    )


class ArtifactContextMiddleware(AgentMiddleware):
    """ADK'nın `LoadArtifactsTool`'unun bizdeki karşılığı.

    Her model çağrısından ÖNCE manifesti çekip bir sistem mesajı olarak
    ekliyor. ADK bunu bir callback ile yapıyor (*"Prepend this information to
    the user's request for the model"*); LangChain'de aynı kancanın adı
    `before_model`.

    ## Neden her çağrıda yeniden çekiliyor

    Bir tur içinde model birden çok kez çağrılabiliyor ve arada `run_ptc_code`
    yeni artifact üretmiş olabilir. Önbelleğe alsaydık, model kendi ürettiği
    şeyi göremezdi. İstek küçük (yalnızca künyeler) ve zaman aşımı kısa.

    ## Hata durumunda ne oluyor

    Hiçbir şey. Manifest çekilemezse mesaj eklenmiyor, ajan eskisi gibi
    çalışıyor. Bu bir kolaylık katmanı; kullanıcının sorusunu buna bağlamak
    orantısız olurdu.
    """

    def __init__(self, workflow_id: str | None, scope_token_uret) -> None:
        super().__init__()
        self._workflow_id = workflow_id or ""
        #: Jetonun ÖMRÜ kısa (15 dk) olduğu için saklanmıyor, her seferinde
        #: yeniden üretiliyor. Üretemezsek (Secret yoksa) None döner.
        self._jeton_uret = scope_token_uret

    def before_model(self, state, runtime) -> dict | None:  # noqa: ARG002
        metin = self._manifest()
        if metin is None:
            return None
        return {"messages": [SystemMessage(content=metin)]}

    async def abefore_model(self, state, runtime) -> dict | None:  # noqa: ARG002
        # Async karşılığı ŞART: graph.py `ainvoke` kullanıyor ve LangGraph o
        # yolda middleware'in async hâlini arıyor. Yalnızca sync uygulamak,
        # 2026-08-28'de LiveSystemTraceMiddleware'de yaşandığı gibi açık hata
        # fırlatıyor.
        return self.before_model(state, runtime)

    def _manifest(self) -> str | None:
        if not self._workflow_id or not servis_adresi():
            return None
        jeton = self._jeton_uret()
        if not jeton:
            return None
        kunyeler = kunyeleri_getir(self._workflow_id, jeton)
        return manifest_metni(kunyeler, self._workflow_id) if kunyeler else None
