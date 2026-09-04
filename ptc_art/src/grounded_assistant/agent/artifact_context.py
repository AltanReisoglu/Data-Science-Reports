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

import requests
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage

#: Prompt'a en fazla bu kadar artifact yazılır. Sınır var çünkü manifest her
#: model çağrısında context'e giriyor — 200 artifactlik bir liste, ucuz olması
#: gereken şeyi pahalı yapardı. Aşılırsa en yeniler gösteriliyor (liste
#: yeniden-eskiye sıralı gelir) ve modele "daha var" deniyor.
_AZAMI_SATIR = 40

_ZAMAN_ASIMI = (2, 5)  # (bağlantı, okuma) — ajanın turunu bekletmemeli


def servis_adresi() -> str:
    return os.environ.get("ARTIFACT_SERVICE_URL", "").rstrip("/")


def kunyeleri_getir(workflow_id: str, scope_token: str) -> list[dict] | None:
    """Bu workflow'un artifact künyelerini çeker. Ulaşılamazsa None."""
    adres = servis_adresi()
    if not adres or not workflow_id or not scope_token:
        return None
    try:
        yanit = requests.get(
            f"{adres}/workflows/{workflow_id}/artifacts",
            headers={"X-Scope-Token": scope_token},
            timeout=_ZAMAN_ASIMI,
        )
        if yanit.status_code != 200:
            return None
        return yanit.json()
    except Exception:  # noqa: BLE001 — ağ hatası turu bozmamalı
        return None


def manifest_metni(kunyeler: list[dict]) -> str | None:
    """Künyeleri modele gösterilecek metne çevirir.

    BAYT YOK, yalnızca isim/tip/boyut. Modelin "ne var" sorusunu cevaplamaya
    yeter; "içinde ne var" sorusu için kodun `get_artifact`/`read_csv`
    çağırması gerekiyor — ADK'nın "isim ucuz, içerik pahalı" ayrımı.
    """
    if not kunyeler:
        return None

    satirlar = []
    for k in kunyeler[:_AZAMI_SATIR]:
        tip = (k.get("type") or "system.Artifact").removeprefix("system.")
        boyut = k.get("size_bytes") or 0
        parca = f"  - {k.get('name')}  ({tip}, {boyut} bayt)"
        ek = k.get("metadata") or {}
        if ek:
            parca += f"  {ek}"
        satirlar.append(parca)

    fazla = len(kunyeler) - len(satirlar)
    if fazla > 0:
        satirlar.append(f"  … ve {fazla} tane daha (list_artifacts() ile tamamı)")

    return (
        "BU KONUŞMADA ŞU ANDA SAKLI OLAN ARTIFACT'LER:\n"
        + "\n".join(satirlar)
        + "\n\nBunlar önceki adımlarda üretildi ve HÂLÂ ERİŞİLEBİLİR. Kullanıcı "
        "bunlardan birine atıf yapıyorsa (\"az önceki tablo\", \"onu grupla\") "
        "veriyi YENİDEN ÜRETME — run_ptc_code içinde "
        "get_artifact(name=\"...\") ile oku, ya da doğrudan "
        "pd.read_csv(\"/output/<ad>\") yaz; dosya otomatik iner."
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
        return manifest_metni(kunyeler) if kunyeler else None
