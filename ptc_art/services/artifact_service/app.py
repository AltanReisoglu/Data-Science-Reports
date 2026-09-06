"""Artifact Service — §27'nin REST API'si, kendi pod'unda.

## Neden Tool Gateway'den ayrıldı (2026-09-04 kararı)

Önceki yolda artifact baytları MCP tool çağrısının içinde **base64** taşınıyordu.
Üç sonucu vardı: veri %33 şişiyor, hem sandbox hem gateway tarafında tamamen
belleğe alınıyor, ve gateway her transferde tıkanma noktası oluyordu. 100 MiB
sınırı da aslında buradan geliyordu — keyfi bir sayı değil, o taşıma biçiminin
doğal tavanı.

Ayrıca tek bir pod üç işi birden yapıyordu: dış tool proxy'si, artifact kayıt
defteri, ve MinIO kimlik bilgisi taşıyıcısı. Araştırma dokümanının §26'sı tam
olarak bunu "önerilmeyen yaklaşım" diye işaretliyor.

Şimdi:

    sandbox ──> Artifact Service ──> MinIO + kayıt defteri   (internet YOK)
            └─> Tool Gateway     ──> 3 onaylı FQDN           (MinIO'ya rota YOK)

Güvenlik açısından bu ÖNCEKİNDEN İYİ: tek bir servisin ele geçirilmesi artık
hem depoya hem internete erişim vermiyor. OpenAI'nin Ağustos 2026 olayından
çıkan "tek workload compromise'ı internet erişimine dönüşmemeli" ilkesinin
karşılığı bu.

## Sandbox'a MinIO rotası AÇILMADI

Presigned URL kullanmıyoruz. Asıl sorun gateway'in araya girmesi değil,
base64+MCP taşımasıydı; akışlı HTTP onu zaten çözüyor. Depoya doğrudan rota
açmak, kazandırdığından fazlasını (ağ politikasının görmediği ikinci bir çıkış)
götürürdü.

## Kapsam jetondan okunur

Her istek `X-Scope-Token` taşır; `workflow_id` JETONDAN çıkar, çağıranın
iddiasından değil. Jeton yoksa/bozuksa hiçbir uç nokta çalışmaz.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from functools import lru_cache

from fastapi import FastAPI, Header, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from grounded_assistant.artifacts.metadata import open_sqlite
from grounded_assistant.artifacts.scope import InvalidScopeToken, Scope, verify_token
from grounded_assistant.artifacts.serialize import UnsafeArtifact, guvenlik_kontrolu
from grounded_assistant.artifacts.service import (
    ArtifactService,
    ArtifactTooLarge,
    InvalidArtifactName,
    ScopeViolation,
)
from grounded_assistant.artifacts.store import BucketConfig, ObjectStore

app = FastAPI(title="PTC Artifact Service", version="1.0")

#: Kapsam jetonlarını doğrulayan ortak sır — `sandbox_runner` imzalar, burası
#: doğrular. Yoksa servis hiçbir isteği kabul etmez: kapsamı doğrulayamadan
#: yazmak, çalıştırmalar arası sınırı tamamen kaldırmak demek olurdu.
_SIGNING_KEY = os.environ.get("PTC_SCOPE_SIGNING_KEY", "")

#: Gövde bu eşiğe kadar bellekte tutulur, aşınca diske taşar. Böylece 100 MiB'lik
#: bir yükleme sırasında süreç belleği sabit kalır.
_BELLEK_ESIGI = 8 * 1024 * 1024

_PARCA = 1024 * 1024


@lru_cache(maxsize=1)
def _service() -> ArtifactService:
    """Servisi ilk kullanımda kurar.

    Bağlantı bilgileri YALNIZCA OBC'nin ürettiği ortam değişkenlerinden okunur
    (`BUCKET_*`, `AWS_*`) — yerelde MinIO, OpenShift'te NooBaa; kod aynı.
    """
    store = ObjectStore(BucketConfig.from_env())
    store.ensure_bucket()  # yerelde gerekli; OBC yolunda bucket zaten var
    return ArtifactService(
        metadata=open_sqlite(os.environ.get("PTC_METADATA_DB", "/var/lib/ptc/artifacts.db")),
        objects=store,
    )


def _kapsam(token: str | None) -> Scope:
    """Kapsamı JETONDAN okur — çağıranın iddiasından değil."""
    if not _SIGNING_KEY:
        raise HTTPException(503, "PTC_SCOPE_SIGNING_KEY tanımlı değil; servis devre dışı.")
    if not token:
        raise HTTPException(401, "X-Scope-Token başlığı gerekli.")
    try:
        return verify_token(_SIGNING_KEY, token)
    except InvalidScopeToken as exc:
        raise HTTPException(401, str(exc)) from exc


def _metadata_coz(ham: str | None) -> dict | None:
    """`X-Artifact-Metadata` başlığındaki JSON'u çözer.

    Bozuksa sessizce yok sayılır: metadata bir ETİKET, güvenlik kontrolü değil.
    Bozuk bir etiket yüzünden kullanıcının çıktısını reddetmek orantısız olurdu.
    """
    if not ham:
        return None
    try:
        cozulen = json.loads(ham)
    except ValueError:
        return None
    return cozulen if isinstance(cozulen, dict) else None


def _ozet(meta) -> dict:
    """Metadata'nın sandbox'a dönen hâli. `storage_uri` BİLEREK yok."""
    return {
        "artifact_id": meta.artifact_id,
        "name": meta.name,
        # 2026-09-06: listeleme tenant genelinde olduğu için "kim üretti"
        # künyeye girdi. Olmasaydı manifest, başka bir çalıştırmanın
        # çıktısını kendi çıktısıymış gibi gösterirdi.
        "workflow_id": meta.workflow_id,
        "type": meta.artifact_type,
        "metadata": meta.user_metadata,
        "node_id": meta.node_id,
        "content_type": meta.content_type,
        "size_bytes": meta.size_bytes,
        "content_hash": meta.content_hash,
        "parents": list(meta.parents),
        "created_at": meta.created_at.isoformat(),
    }


def _kunye_basliklari(meta) -> dict[str, str]:
    """Bayt yanıtına iliştirilen künye — istemci ikinci bir istek atmasın diye."""
    return {
        "X-Artifact-Id": meta.artifact_id,
        "X-Artifact-Name": meta.name,
        "X-Artifact-Content-Hash": meta.content_hash,
        "X-Artifact-Size": str(meta.size_bytes),
        "X-Artifact-Type": meta.artifact_type,
        "Content-Length": str(meta.size_bytes),
    }


@app.exception_handler(ScopeViolation)
async def _kapsam_ihlali(_: Request, exc: ScopeViolation) -> JSONResponse:
    # Var-ama-yetkisiz ile hiç-yok ayrımı BİLEREK yapılmıyor (bkz. service.py):
    # aksi halde çağıran, başka workflow'larda hangi id'lerin var olduğunu
    # deneyerek öğrenebilirdi. Bu yüzden 403 değil 404.
    return JSONResponse({"detail": str(exc)}, status_code=404)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


# --------------------------------------------------------------------------
# POST /artifacts — akışlı yükleme
# --------------------------------------------------------------------------


@app.post("/artifacts", status_code=201)
async def create_artifact(
    request: Request,
    x_scope_token: str | None = Header(default=None),
    x_artifact_name: str | None = Header(default=None),
    x_artifact_parents: str | None = Header(default=None),
    x_artifact_ttl: int | None = Header(default=None),
    x_artifact_type: str | None = Header(default=None),
    x_artifact_metadata: str | None = Header(default=None),
    x_artifact_root: str | None = Header(default=None),
    content_type: str = Header(default="application/octet-stream"),
) -> dict:
    """Gövdeyi parça parça okur, doğrular, depoya akıtır.

    Baytların tamamı hiçbir noktada bellekte tutulmaz. Üç kontrol akış SIRASINDA
    yapılır:

      1. **pickle** — `_pickle_mi` yalnızca ilk 2 bayta baktığı için ilk parçada
         karar verilebiliyor; reddedilen yükleme depoya hiç dokunmuyor (CWE-502).
      2. **boyut** — sayarak; sınır aşılınca okuma orada kesiliyor, sonunda
         değil.
      3. **sha256** — akış boyunca birikiyor, dedup için servise veriliyor.
    """
    kapsam = _kapsam(x_scope_token)
    if not x_artifact_name:
        raise HTTPException(400, "X-Artifact-Name başlığı gerekli.")

    servis = _service()
    ozetleyici = hashlib.sha256()
    uzunluk = 0
    ilk = True

    with tempfile.SpooledTemporaryFile(max_size=_BELLEK_ESIGI) as tampon:
        async for parca in request.stream():
            if not parca:
                continue
            if ilk:
                # İlk parçada karar ver: pickle ise tek bayt bile depoya gitmesin.
                try:
                    guvenlik_kontrolu(parca, content_type)
                except UnsafeArtifact as exc:
                    raise HTTPException(415, str(exc)) from exc
                ilk = False
            uzunluk += len(parca)
            if uzunluk > servis.size_limit:
                raise HTTPException(
                    413,
                    f"Artifact {servis.size_limit} bayt sınırını aşıyor. Veriyi sandbox "
                    "içinde süzüp özetleyin — PTC'nin bütün amacı bu.",
                )
            ozetleyici.update(parca)
            tampon.write(parca)

        if ilk:  # hiç gövde gelmedi
            raise HTTPException(400, "Boş gövde.")
        tampon.seek(0)

        try:
            meta = servis.create_stream(
                tampon,
                uzunluk,
                content_type,
                name=x_artifact_name,
                workflow_id=kapsam.workflow_id,
                run_id=kapsam.run_id,
                owner=kapsam.owner,
                node_id=kapsam.node_id,
                content_hash="sha256:" + ozetleyici.hexdigest(),
                parents=tuple(p for p in (x_artifact_parents or "").split(",") if p),
                ttl_seconds=x_artifact_ttl,
                artifact_type=x_artifact_type,
                user_metadata=_metadata_coz(x_artifact_metadata),
                root=x_artifact_root,
            )
        except InvalidArtifactName as exc:
            raise HTTPException(400, str(exc)) from exc
        except ArtifactTooLarge as exc:
            raise HTTPException(413, str(exc)) from exc

    return {**_ozet(meta), "status": "stored"}


# --------------------------------------------------------------------------
# Okuma
# --------------------------------------------------------------------------


def _akit(meta) -> StreamingResponse:
    return StreamingResponse(
        _service().iter_bytes(meta, _PARCA),
        media_type=meta.content_type,
        headers=_kunye_basliklari(meta),
    )


@app.get("/artifacts/by-name/{name}")
async def get_artifact_by_name(
    name: str = Path(...),
    workflow: str | None = Query(default=None),
    x_scope_token: str | None = Header(default=None),
) -> StreamingResponse:
    """O isimdeki EN YENİ sürümün baytları.

    Node'lar arası devrin ana yolu: sonraki adım artifact_id'yi bilmek zorunda
    değil, ismi yeterli.
    """
    kapsam = _kapsam(x_scope_token)
    # `workflow` verilirse O ÇALIŞTIRMAYA bağlı çözülür (tenant'a düşmez).
    # `/output/<ad>` bunu kullanıyor: bir run'ın kendi dizini yalnızca kendi
    # çıktılarını göstermeli — 2026-09-06'da ajan başka bir run'ın aynı adlı
    # çıktısını kendi işi sanıp yanlış sayı verdi.
    meta = _service().resolve(
        owner=kapsam.owner,
        workflow_id=workflow or kapsam.workflow_id,
        name=name,
        strict=workflow is not None,
    )
    if meta is None:
        raise HTTPException(404, f"'{name}' adında artifact yok.")
    return _akit(meta)


@app.get("/artifacts/{artifact_id}/metadata")
async def get_metadata(
    artifact_id: str = Path(...),
    x_scope_token: str | None = Header(default=None),
) -> dict:
    """Künye — baytlar indirilmeden."""
    kapsam = _kapsam(x_scope_token)
    return _ozet(_service().metadata_of(artifact_id, owner=kapsam.owner))


@app.get("/artifacts/{artifact_id}/lineage")
async def get_lineage(
    artifact_id: str = Path(...),
    limit: int = Query(default=1000, le=5000),
    x_scope_token: str | None = Header(default=None),
) -> dict:
    """Soy ağacı: bu artifact'in ATALARI ve ÜRÜNLERİ.

    Kayıt defterinde `parents` baştan beri vardı ama okuyan bir uç nokta
    yoktu — "kaydediliyor, keşifte kullanılmıyor" açığı (§11.10) tam olarak
    buydu. Grafik yalnızca jetondaki workflow'la sınırlı.
    """
    kapsam = _kapsam(x_scope_token)
    return _service().lineage(
        artifact_id, owner=kapsam.owner, limit=limit
    )


@app.get("/artifacts/{artifact_id}")
async def get_artifact(
    artifact_id: str = Path(...),
    x_scope_token: str | None = Header(default=None),
) -> StreamingResponse:
    """Kimlikle baytlar."""
    kapsam = _kapsam(x_scope_token)
    meta = _service().resolve(owner=kapsam.owner, artifact_id=artifact_id)
    if meta is None:
        raise HTTPException(404, f"{artifact_id} bulunamadı.")
    return _akit(meta)


@app.get("/artifacts")
async def list_artifacts(
    limit: int = Query(default=200, le=1000),
    x_scope_token: str | None = Header(default=None),
) -> list[dict]:
    """Bu tenant'ta ne var — **çalıştırmalar arası**, künyeler, baytlar değil.

    Sandbox'ın manifestinin kaynağı. Kapsam workflow değil tenant (2026-09-06):
    KFP'de de bütün run'lar aynı `pipeline_root` altına yazar ve birbirinin
    çıktısını görebilir. Her kayıt kendi `workflow_id`'sini taşıyor, yani
    "kimin ürettiği" bilgisi kaybolmuyor — yalnızca görünürlük açılıyor.
    """
    kapsam = _kapsam(x_scope_token)
    return [_ozet(m) for m in _service().list(owner=kapsam.owner, limit=limit)]


@app.get("/workflows/{workflow_id}/artifacts")
async def list_workflow_artifacts(
    workflow_id: str = Path(...),
    node_id: str | None = Query(default=None),
    limit: int = Query(default=200, le=1000),
    x_scope_token: str | None = Header(default=None),
) -> list[dict]:
    """Tek bir çalıştırmanın ürettikleri — tenant listesinin süzülmüş hâli.

    Panel bunu kullanıyor ("bu oturumda ne üretildi"). Artık BAŞKA bir
    workflow'un kimliği de sorulabilir: okuma sınırı tenant, ve panelin
    "şu çalıştırmaya bak" bağlantısının işe yaraması için gerekli.
    """
    kapsam = _kapsam(x_scope_token)
    return [
        _ozet(m)
        for m in _service().list(owner=kapsam.owner, limit=limit)
        if m.workflow_id == workflow_id and (node_id is None or m.node_id == node_id)
    ]


# --------------------------------------------------------------------------
# Silme — sandbox'a AÇILMAZ
# --------------------------------------------------------------------------


@app.post("/admin/reap")
async def reap(
    dry_run: bool = Query(default=False),
    limit: int = Query(default=500, le=5000),
    x_admin_token: str | None = Header(default=None),
) -> dict:
    """Süresi dolmuş artifact'leri toplar. TTL'in çalıştırıcısı BURASI.

    ## Neden CronJob'un içinde değil de burada

    Kayıt defteri SQLite ve PVC `ReadWriteOnce` — yani aynı anda tek yazıcı
    olabilir. Ayrı bir CronJob pod'u DB'yi kendisi açsaydı ya PVC'yi mount
    edemezdi ya da eşzamanlı yazıcı olup bozulma riski doğururdu. Bu yüzden iş
    bölümü şöyle: **CronJob zamanlayıcı, servis tek yazıcı.** CronJob yalnızca
    bu uç noktayı çağırır.

    (Postgres'e geçilince bu kısıt kalkar ve reaper ayrı bir pod olabilir. O
    zaman bile bu uç nokta durabilir — zararsız.)

    ## Neden ayrı bir jeton

    Kapsam jetonu bir workflow'a bağlı; reaper ise workflow'lar arası süpürüyor.
    Ayrıca sandbox'ın elindeki jetonun buraya YETMEMESİ gerekiyor: LLM'in
    ürettiği kodun toplu silme tetikleyebilmesi, kalıcılığı tek çağrıda geri
    alabilirdi. `PTC_ADMIN_TOKEN` yalnızca CronJob'un Secret'ında.

    Sır tanımlı değilse uç nokta 503 — "açık bırakılmış" bir hâli yok.

    ## Silme sırası

    `ArtifactService.delete` ÖNCE baytı SONRA metadata'yı siler ve dedup ile
    paylaşılan baytları korur; burada o davranış aynen kullanılıyor.

    TTL'i olmayan (`ttl_seconds IS NULL`) artifact'ler hiç dokunulmaz — varsayılan
    budur, yani kimse TTL vermediyse bu çağrı hiçbir şey silmez.
    """
    beklenen = os.environ.get("PTC_ADMIN_TOKEN", "")
    if not beklenen:
        raise HTTPException(503, "PTC_ADMIN_TOKEN tanımlı değil; reap devre dışı.")
    if not x_admin_token or not hmac.compare_digest(x_admin_token, beklenen):
        raise HTTPException(401, "Geçersiz yönetim jetonu.")

    servis = _service()
    suresi_dolan = servis.metadata.expired()[:limit]
    ozet = [
        {"artifact_id": m.artifact_id, "name": m.name, "workflow_id": m.workflow_id,
         "size_bytes": m.size_bytes, "expires_at": m.expires_at().isoformat()}
        for m in suresi_dolan
    ]

    if dry_run:
        return {"dry_run": True, "aday": len(ozet), "artifacts": ozet}

    silinen, hatali = 0, []
    for m in suresi_dolan:
        try:
            if servis.delete(m.artifact_id, owner=m.owner):
                silinen += 1
        except Exception as exc:  # noqa: BLE001
            # Bir kayıt silinemezse (depo erişilemez, yarış) diğerleri sürsün:
            # yarım kalan bir süpürme, hiç çalışmayan bir süpürmeden iyidir ve
            # kalanlar bir sonraki turda yine aday olur.
            hatali.append({"artifact_id": m.artifact_id, "detail": str(exc)[:200]})

    return {"dry_run": False, "aday": len(ozet), "silinen": silinen, "hatali": hatali}


@app.delete("/artifacts/{artifact_id}")
async def delete_artifact(
    artifact_id: str = Path(...),
    x_scope_token: str | None = Header(default=None),
) -> dict:
    """TTL reaper'ı içindir; sandbox'ın istemci kütüphanesinde karşılığı YOKTUR.

    LLM'in ürettiği kodun artifact silebilmesi, emniyet ağı olarak kurduğumuz
    kalıcılığı tek satırda geri alabilirdi. Ağ politikası bu uç noktayı
    engellemiyor — kasıtlı: sandbox'ın jetonu geçerli, ama istemcisinde bu
    çağrı hiç tanımlı değil ve LLM opak `artifact_id`'leri ancak listeleyerek
    öğrenebiliyor. Gerçek koruma reaper'ın ayrı bir ServiceAccount ile
    çalışması olacak (TODO: Faz 6).
    """
    kapsam = _kapsam(x_scope_token)
    silindi = _service().delete(artifact_id, owner=kapsam.owner)
    return {"artifact_id": artifact_id, "deleted": silindi}
