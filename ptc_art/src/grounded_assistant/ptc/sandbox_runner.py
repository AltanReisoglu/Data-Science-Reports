"""PTC sandbox orkestrasyonu — Kubernetes Job olarak çalıştırma (Faz 2, T013).

Akış (research.md §1-§2, §5-§6; contracts/sandbox_job_contract.md):
1. Kodu bir ConfigMap'e yaz (`ptc-code-{run_id}`).
2. `k8s/sandbox/job-template.yaml`'dan bir Job oluştur (activeDeadlineSeconds=30,
   TOOL_GATEWAY_ENDPOINT enjekte edilmiş).
3. Job'un bitmesini bekle (Succeeded/Failed/DeadlineExceeded).
4. Pod log'unun son satırındaki JSON'u parse et.
5. ConfigMap + Job'u temizle — ARKA PLANDA (2026-09-03): temizlik sonucu
   çağırana lazım değil, `ttlSecondsAfterFinished` zaten güvenlik ağı.

Not: Bu modül laptop üzerinde (ana asistanla birlikte) çalışır, cluster İÇİNDE
DEĞİL — bu yüzden `config.load_kube_config()` kullanılır (`load_incluster_config`
değil). tool_calls'ın Trace'e beslenmesi graph.py'de (T015).

2026-09-03 — artifact persistence case'ine geçiş: Cilium/Hubble bağımlılığı
(`get_denied_actions` ve DROPPED flow sorgusu) KALDIRILDI. Gerekçesi iki katlı:
(a) egress policy artık projenin konusu değil, (b) hedef ortam OpenShift ve
oradaki CNI OVN-Kubernetes — `hubble` diye bir şey yok. Eski hâli:
`archive/egress-policy/code/`.

Faz 4 (T003-T005, specs/003-web-ui-live-trace/research.md §4): `run_sandbox`,
opsiyonel bir `on_event` callback'i alır ve adımları (configmap_created/
job_created/tool_call/denied_action/final) GERÇEKLEŞTİKÇE bu callback'e iletir —
web arayüzünün sol-alt canlı panelinin tek veri kaynağı budur. `on_event=None`
iken (CLI'nin mevcut çağrısı) davranış Faz 2'deki gibi kalır.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import yaml
from kubernetes import client, config

from grounded_assistant.artifacts.scope import Scope, issue_token
from grounded_assistant.models import (
    ArtifactEvent,
    ArtifactOp,
    LiveToolCall,
    SandboxRun,
    SandboxRunStatus,
    ToolCallStatus,
)

# 2026-09-03: OpenShift iş yükünü `default`ta koşturmaz; proje adı dışarıdan verilir.
NAMESPACE = os.environ.get("PTC_NAMESPACE", "default")
TOOL_GATEWAY_SERVICE_NAME = "tool-gateway"
TOOL_GATEWAY_PORT = 8443

# Artifact Service (2026-09-04) — artifact baytlarının yeni adresi.
ARTIFACT_SERVICE_NAME = "artifact-service"
ARTIFACT_SERVICE_PORT = 8080

#: Kapsam jetonu imza anahtarını taşıyan Secret (k8s/artifact-store/).
_SIGNING_SECRET_NAME = "ptc-scope-signing"

_TEMPLATE_PATH = Path(__file__).resolve().parents[3] / "k8s" / "sandbox" / "job-template.yaml"

# 2026-09-03: 1.0 sn'lik polling, iş bitse bile ortalama ~0.5 sn (en kötü 1 sn)
# boşuna bekletiyordu. Her tur iki ucuz API çağrısı (pod log + job status);
# 30 sn'lik bir job için bu yükü artırmak, gecikmeyi düşürmeye değer.
_POLL_INTERVAL_SECONDS = 0.15
# activeDeadlineSeconds (30) + Kubernetes'in bunu fark edip Job'u
# Failed/DeadlineExceeded'e çevirmesi için makul bir tampon (research.md §6).
_WAIT_TIMEOUT_SECONDS = 45.0

#: Sandbox'ın terminal satırından sonra sidecar'ın süpürmesini beklediğimiz
#: tavan. Süpürme ana container bittikten SONRA çalışıyor (Argo `wait` deseni);
#: burada dönseydik `produced` olaylarını hiç göremezdik. Pod terminal faza
#: geçerse zaten daha erken dönüyoruz — bu yalnızca güvenlik ağı.
_SIDECAR_BEKLEME_SANIYE = 20.0


def _resolve_tool_gateway_endpoint(core_v1: client.CoreV1Api) -> str:
    """research.md §4.1: sandbox'ın DNS'e hiç ihtiyaç duymaması için Tool
    Gateway'in Service ClusterIP'si DNS adı yerine DOĞRUDAN IP olarak enjekte
    edilir — `sandbox-egress` policy'sinin (T016) kuralları `toEndpoints`
    (pod-etiketi bazlı, identity-based) olup DNS'e izin vermez; DNS adı
    kullanılsaydı sandbox kube-dns'e erişemeyip bu adımda tıkanırdı."""
    service = core_v1.read_namespaced_service(name=TOOL_GATEWAY_SERVICE_NAME, namespace=NAMESPACE)
    return f"http://{service.spec.cluster_ip}:{TOOL_GATEWAY_PORT}/mcp"


def _resolve_artifact_service_endpoint(core_v1: client.CoreV1Api) -> str:
    """Artifact Service'in ClusterIP'si — gateway'inkiyle aynı gerekçe (DNS yok).

    2026-09-04: artifact baytları artık gateway üzerinden değil buradan gidiyor.
    Servis bulunamazsa boş string döner ve sandbox artifact API'sini hiç
    görmez — kısmi/yarım bir artifact yolu sunmaktansa hiç sunmamak daha temiz.
    """
    try:
        service = core_v1.read_namespaced_service(
            name=ARTIFACT_SERVICE_NAME, namespace=NAMESPACE
        )
    except client.ApiException:
        return ""
    return f"http://{service.spec.cluster_ip}:{ARTIFACT_SERVICE_PORT}"


def _load_job_manifest(
    run_id: str,
    tool_gateway_endpoint: str,
    scope_token: str,
    artifact_service_endpoint: str,
    workflow_id: str,
) -> dict:
    template_text = _TEMPLATE_PATH.read_text(encoding="utf-8")
    filled = template_text.format(
        run_id=run_id,
        tool_gateway_endpoint=tool_gateway_endpoint,
        scope_token=scope_token,
        artifact_service_endpoint=artifact_service_endpoint,
        workflow_id=workflow_id,
    )
    return yaml.safe_load(filled)


def _read_signing_key(core_v1: client.CoreV1Api) -> str | None:
    """Kapsam jetonlarını imzalayan ortak sırrı cluster'dan okur.

    Sır, sandbox'ın ERİŞEMEDİĞİ bir yerde (Kubernetes Secret) durur; runner
    laptop'tan cluster API'siyle okur, gateway'e mount edilir. Sandbox'ın eline
    yalnızca İMZALANMIŞ jeton geçer.

    Sır yoksa None döner ve sandbox artifact API'sini hiç görmez — kapsamı
    doğrulayamadan artifact yazmak, çalıştırmalar arası sınırı kaldırmak olurdu.
    """
    try:
        secret = core_v1.read_namespaced_secret(name=_SIGNING_SECRET_NAME, namespace=NAMESPACE)
    except client.ApiException:
        return None
    ham = (secret.data or {}).get("PTC_SCOPE_SIGNING_KEY")
    return base64.b64decode(ham).decode("utf-8") if ham else None


def _create_configmap(core_v1: client.CoreV1Api, run_id: str, code: str) -> None:
    configmap = client.V1ConfigMap(
        metadata=client.V1ObjectMeta(name=f"ptc-code-{run_id}"),
        data={"code.py": code},
    )
    core_v1.create_namespaced_config_map(namespace=NAMESPACE, body=configmap)


def _configmapi_joba_bagla(core_v1: client.CoreV1Api, run_id: str, job) -> None:
    """ConfigMap'e Job'u gösteren bir `ownerReference` koyar.

    NEDEN GEREKLİ — ölçülmüş bir sızıntı (2026-09-04): `_cleanup` arka plan
    thread'ine alındığında (Faz 0 optimizasyonu) şu varsayım yazılmıştı:
    "`ttlSecondsAfterFinished` zaten güvenlik ağı, thread ölse bile toplanır."
    Bu Job ve pod için DOĞRU, ConfigMap için YANLIŞ — ConfigMap Job'a ait
    olmadığı için TTL ona hiç dokunmuyordu. `daemon=True` thread, kısa ömürlü
    script'lerde süreç çıkarken kesiliyor ve ConfigMap'i silecek kimse
    kalmıyordu. Sayım: 31 yetim ConfigMap, 22 saatlik birikim.

    ownerReference ile Kubernetes'in kendi çöp toplayıcısı devreye giriyor:
    Job silinince (ister bizim explicit cleanup'ımızla, ister TTL ile)
    ConfigMap de kaskad siliniyor. Artık garanti, sürecimizin hayatta
    kalmasına DEĞİL cluster'a bağlı — süreç çökse bile çalışır.

    Job yaratıldıktan SONRA çağrılmak zorunda: ownerReference Job'un UID'sini
    gerektiriyor, o da ancak API nesneyi yarattıktan sonra var oluyor.
    """
    try:
        core_v1.patch_namespaced_config_map(
            name=f"ptc-code-{run_id}",
            namespace=NAMESPACE,
            body={
                "metadata": {
                    "ownerReferences": [
                        {
                            "apiVersion": "batch/v1",
                            "kind": "Job",
                            "name": job.metadata.name,
                            "uid": job.metadata.uid,
                            # Job silinmeden ConfigMap silinmesin (pod hâlâ mount'lu olabilir)
                            "blockOwnerDeletion": True,
                        }
                    ]
                }
            },
        )
    except client.ApiException:
        # Best-effort: bağlanamazsa explicit cleanup + TTL yine devrede,
        # yalnızca "süreç ölürse" senaryosundaki güvence kaybolur.
        pass


def _emit(on_event: Callable[[dict], None] | None, event: dict) -> None:
    """Faz 4, T003: on_event verilmemişse (CLI'nin mevcut çağrısı) sessizce
    hiçbir şey yapmaz — çağıranın her yerde `if on_event is not None` yazmasını
    önler."""
    if on_event is not None:
        on_event(event)


def _wait_and_stream(
    core_v1: client.CoreV1Api,
    batch_v1: client.BatchV1Api,
    job_name: str,
    on_event: Callable[[dict], None] | None,
) -> tuple[SandboxRunStatus, str | None, str | None, list[LiveToolCall], list[ArtifactEvent]]:
    """Faz 2'nin `_wait_for_job` + `_parse_log`'unun BİRLEŞİK, canlı-olay-yayan
    hâli (Faz 4, T004). Her ~1sn'lik pollingde hem Job durumuna HEM pod
    log'undaki YENİ satırlara bakar; her `tool_call` satırı geldiği ANDA
    `on_event`'e verilir (FR-005 — "gerçekleştikçe", sona toplanıp basılmıyor).

    Not (implementasyon sırasında bulunan bir sadeleştirme, research.md §4'ün
    `follow=True` fikrine göre): gerçek bir stream bağlantısı yönetmek yerine,
    her turda `_read_pod_log` ile TÜM log'u (küçük olduğu için ucuz) tekrar
    okuyup yalnızca DAHA ÖNCE görülmemiş satırları işliyoruz — aynı
    'gerçekleştikçe' hissini veriyor, ama partial-line/stream-yaşam-döngüsü gibi
    ek karmaşıklık taşımıyor (Principle V).

    Not (Faz 2'den, hâlâ geçerli): activeDeadlineSeconds aşıldığında Kubernetes
    ÖNCE `type: FailureTarget` (reason: DeadlineExceeded) koşulunu yazıyor —
    `reason`'a bakmak (`type`'tan bağımsız) güvenilir."""
    deadline = time.monotonic() + _WAIT_TIMEOUT_SECONDS
    tool_calls: list[LiveToolCall] = []
    artifacts: list[ArtifactEvent] = []
    status, result_text, error_message = SandboxRunStatus.ERROR, None, None
    # Container başına ayrı sayaç: artifact olayları artık SIDECAR'ın log'unda
    # (2026-09-06, süpürme oraya taşındı), sonuç/tool_call'lar sandbox'ta.
    seen_lines: dict[str, int] = {"sandbox": 0, "artifact-sidecar": 0}
    terminal_seen = False
    sidecar_bitti = False
    #: Sandbox bitti ama sidecar henüz süpürmedi — bu kadar daha bekleriz.
    sidecar_deadline: float | None = None

    pod_running_emitted = False
    while time.monotonic() < deadline:
        log_text, pod_phase = _read_pod_log(core_v1, job_name, "sandbox")
        if pod_phase == "Running" and not pod_running_emitted:
            pod_running_emitted = True
            _emit(on_event, {"stage": "pod_running", "job_name": job_name})

        yan_log, _ = _read_pod_log(core_v1, job_name, "artifact-sidecar")
        yan = yan_log.strip().splitlines() if yan_log and yan_log.strip() else []
        lines = (log_text.strip().splitlines() if log_text and log_text.strip() else [])

        # Sidecar satırları ÖNCE işleniyor; sandbox'ın terminal satırı geldiği
        # anda dönmemek için `terminal_seen` aşağıda ayrıca bekletiliyor.
        birlesik = [("artifact-sidecar", l) for l in yan[seen_lines["artifact-sidecar"]:]]
        birlesik += [("sandbox", l) for l in lines[seen_lines["sandbox"]:]]
        seen_lines["artifact-sidecar"] = len(yan)
        for _kaynak, line in birlesik:
            parsed = _parse_line(line)
            if parsed is None:
                continue  # entrypoint.py'nin kendi hata çıktısı olabilir, yok say
            if parsed.get("type") == "artifact":
                olay = ArtifactEvent(
                    op=ArtifactOp(parsed["op"]),
                    artifact_id=parsed["artifact_id"],
                    name=parsed["name"],
                    timestamp=datetime.fromisoformat(parsed["timestamp"]),
                    size_bytes=parsed.get("size_bytes"),
                    content_type=parsed.get("content_type"),
                    parents=tuple(parsed.get("parents") or ()),
                )
                artifacts.append(olay)
                _emit(on_event, {"stage": "artifact", **parsed})
            elif parsed.get("type") == "supurme_bitti":
                # Sidecar süpürmeyi bitirdi — pod'un terminal faza geçmesini
                # beklemeye gerek yok (ölçümde ~3 sn).
                sidecar_bitti = True
            elif parsed.get("type") == "artifact_skipped":
                # /output süpürmesinde reddedilen bir dosya (pickle, boyut,
                # okunamayan dosya). ArtifactEvent'e ÇEVRİLMEZ — hiç
                # depolanmadığı için artifact_id yok, model bunu zorunlu
                # tutuyor. Yalnızca canlı panel için görünürlük.
                _emit(on_event, {"stage": "artifact_skipped", **parsed})
            elif parsed.get("type") == "tool_call":
                call = LiveToolCall(
                    tool_name=parsed["tool"],
                    arguments=parsed.get("args", {}),
                    timestamp=datetime.fromisoformat(parsed["timestamp"]),
                    status=ToolCallStatus.SUCCESS
                    if parsed["status"] == "success"
                    else ToolCallStatus.ERROR,
                )
                tool_calls.append(call)
                _emit(
                    on_event,
                    {
                        "stage": "tool_call",
                        "tool_name": call.tool_name,
                        "arguments": call.arguments,
                        "status": call.status.value,
                        "timestamp": parsed["timestamp"],
                    },
                )
            elif parsed.get("status") == "success":
                status, result_text = SandboxRunStatus.SUCCESS, str(parsed.get("result"))
                terminal_seen = True
            elif "status" in parsed:
                # Altan'ın kararı (2026-08-30): entrypoint.py'nin
                # {"status": "error", "message": str(exc)} satırındaki gerçek hata
                # metni önceden burada atılıyordu — panelde/LLM'e yalnızca
                # "durum: error" görünüyor, NEDEN'i hiç geçmiyordu ("ne oldu
                # felan olarak" görünürlük isteği). result_text (FR-011 gereği
                # başarısız koşuda hep None kalmalı) DEĞİL, ayrı error_message.
                status, error_message = SandboxRunStatus.ERROR, parsed.get("message")
                terminal_seen = True
        seen_lines["sandbox"] = len(lines)

        # 2026-09-03'te burada erken dönüyorduk: sandbox'ın terminal satırı
        # gelince Job.status'ü beklemeden çıkıyorduk (~2.7 sn kazanç).
        #
        # 2026-09-06: SÜPÜRME SIDECAR'A TAŞINDI ve o, ana container bittikten
        # SONRA çalışıyor. Terminal satırda dönseydik `produced` olaylarının
        # HİÇBİRİNİ görmezdik. O yüzden terminal satırdan sonra sidecar'ın
        # bitmesini bekliyoruz — ama sınırlı süre: pod terminal faza geçerse
        # ya da tavan dolarsa dönüyoruz.
        if terminal_seen:
            if sidecar_deadline is None:
                sidecar_deadline = time.monotonic() + _SIDECAR_BEKLEME_SANIYE
            if (sidecar_bitti or pod_phase in ("Succeeded", "Failed")
                    or time.monotonic() > sidecar_deadline):
                return status, result_text, error_message, tool_calls, artifacts
            time.sleep(_POLL_INTERVAL_SECONDS)
            continue

        job = batch_v1.read_namespaced_job(name=job_name, namespace=NAMESPACE)
        for condition in job.status.conditions or []:
            if condition.reason == "DeadlineExceeded":
                return SandboxRunStatus.TIMEOUT, None, None, tool_calls, artifacts
        if job.status.succeeded or job.status.failed:
            return status, result_text, error_message, tool_calls, artifacts
        time.sleep(_POLL_INTERVAL_SECONDS)

    return SandboxRunStatus.TIMEOUT, None, None, tool_calls, artifacts  # kendi güvenlik ağımız


def _read_pod_log(
    core_v1: client.CoreV1Api, job_name: str, container: str = "sandbox"
) -> tuple[str | None, str | None]:
    """(log metni, pod phase) döner. Phase, pod'un ne zaman Running'e geçtiğini
    ölçebilmek için eklendi (2026-09-03): toplam gecikmenin ne kadarı pod
    kurulumu, ne kadarı Python/fastmcp açılışı — warm pool kararı buna bağlı.
    Ekstra API çağrısı YOK, zaten yapılan list_namespaced_pod'dan okunuyor."""
    pods = core_v1.list_namespaced_pod(namespace=NAMESPACE, label_selector=f"job-name={job_name}")
    if not pods.items:
        return None, None
    phase = pods.items[0].status.phase
    pod_name = pods.items[0].metadata.name
    try:
        # _preload_content=False: client'ın varsayılan davranışı, log JSON'a
        # benziyorsa onu sessizce parse edip str(dict) ile geri yazıyor — bu da
        # tırnak işaretlerini bozup contracts/sandbox_job_contract.md'nin
        # JSON kontratını geçersiz kılıyor (deneyle doğrulandı, 2026-08-28).
        # Ham yanıtı kendimiz decode ederek bunu atlıyoruz.
        # `container` ZORUNLU oldu (2026-09-06): pod artık iki container
        # taşıyor (sandbox + artifact-sidecar). Belirtilmezse API 400 döner.
        raw = core_v1.read_namespaced_pod_log(
            name=pod_name, namespace=NAMESPACE, container=container,
            _preload_content=False,
        )
        return raw.data.decode("utf-8"), phase
    except client.ApiException:
        return None, phase


def _parse_line(line: str) -> dict | None:
    """Bir pod log satırını JSON olarak ayrıştırır; geçersizse None döner
    (Faz 4, T004 — `_wait_and_stream`'in satır-satır kullandığı yardımcı;
    Faz 2'nin `_parse_log`'unun tek-satırlık hâli)."""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _cleanup(
    core_v1: client.CoreV1Api, batch_v1: client.BatchV1Api, run_id: str, job_name: str
) -> None:
    try:
        batch_v1.delete_namespaced_job(
            name=job_name, namespace=NAMESPACE, propagation_policy="Foreground"
        )
    except client.ApiException:
        pass
    try:
        core_v1.delete_namespaced_config_map(name=f"ptc-code-{run_id}", namespace=NAMESPACE)
    except client.ApiException:
        pass


def run_sandbox(
    code: str,
    on_event: Callable[[dict], None] | None = None,
    workflow_id: str | None = None,
    owner: str = "ptc",
    node_id: str | None = None,
) -> SandboxRun:
    """Verilen Python kodunu ayrı bir Kubernetes Job'unda çalıştırır, sonucu bir
    SandboxRun olarak döner.

    `on_event` (Faz 4, T003-T005): verilirse, çalıştırmanın her adımı
    GERÇEKLEŞTİKÇE bir sözlük olarak bu callback'e iletilir (contracts/
    websocket_protocol.md'deki `ptc_event` şeması) — CLI bunu hiç kullanmaz
    (`on_event=None`), davranışı Faz 2'deki gibi kalır.

    `workflow_id` (2026-09-03): artifact deposunun kapsam anahtarı. Verilirse
    bu çalıştırma için bir kapsam jetonu İMZALANIR ve pod'un ortamına konur —
    sandbox artık `put_artifact`/`get_artifact`/`cached` çağırabilir ve
    yazdıkları pod öldükten SONRA da durur. Verilmezse artifact API hiç
    açılmaz: kapsamı doğrulanamayan bir çalıştırmanın kalıcı depoya yazması,
    çalıştırmalar arası sınırı kaldırmak olurdu (araştırma §6.1).
    """
    config.load_kube_config()
    core_v1 = client.CoreV1Api()
    batch_v1 = client.BatchV1Api()

    run_id = uuid.uuid4().hex[:12]
    job_name = f"ptc-sandbox-{run_id}"
    started_at = datetime.now(UTC)

    _create_configmap(core_v1, run_id, code)
    _emit(on_event, {"stage": "configmap_created", "run_id": run_id})

    scope_token = ""
    if workflow_id is not None:
        anahtar = _read_signing_key(core_v1)
        if anahtar:
            scope_token = issue_token(
                anahtar,
                Scope(
                    workflow_id=workflow_id, run_id=run_id, owner=owner, node_id=node_id
                ),
            )

    tool_gateway_endpoint = _resolve_tool_gateway_endpoint(core_v1)
    artifact_service_endpoint = (
        _resolve_artifact_service_endpoint(core_v1) if scope_token else ""
    )
    job_manifest = _load_job_manifest(
        run_id,
        tool_gateway_endpoint,
        scope_token,
        artifact_service_endpoint,
        workflow_id or "",
    )
    job = batch_v1.create_namespaced_job(namespace=NAMESPACE, body=job_manifest)
    # ConfigMap'i Job'a bağla — Job silinince Kubernetes onu da toplasın.
    # Bkz. `_configmapi_joba_bagla`: bu olmadan ConfigMap'ler yetim kalıyordu.
    _configmapi_joba_bagla(core_v1, run_id, job)
    _emit(on_event, {"stage": "job_created", "run_id": run_id, "code": code})

    status, result_text, error_message, tool_calls, artifacts = _wait_and_stream(
        core_v1, batch_v1, job_name, on_event
    )

    # 2026-09-03: temizlik ARKA PLANDA — sonucu çağırana lazım değil ve iki
    # silme çağrısı kritik yolda duruyordu.
    #
    # 2026-09-04 DÜZELTME: burada önceden "TTL zaten güvenlik ağı, thread ölse
    # bile toplanır" yazıyordu. Bu yalnızca Job ve pod için doğruydu; ConfigMap
    # Job'a ait olmadığı için TTL'in kapsamı dışındaydı ve `daemon=True` thread
    # kısa ömürlü script'lerde kesilince yetim kalıyordu (31 birikmiş ConfigMap).
    # Gerçek güvenlik ağı artık `_configmapi_joba_bagla`'nın koyduğu
    # ownerReference; bu thread yalnızca HIZLI yol (uzun ömürlü süreçlerde
    # anında temizlik), doğruluk garantisi değil.
    threading.Thread(
        target=_cleanup, args=(core_v1, batch_v1, run_id, job_name), daemon=True
    ).start()
    _emit(
        on_event,
        {
            "stage": "final",
            "run_id": run_id,
            "status": status.value,
            "result_text": result_text,
            "error_message": error_message,
        },
    )

    return SandboxRun(
        run_id=run_id,
        code=code,
        started_at=started_at,
        status=status,
        finished_at=datetime.now(UTC),
        tool_calls=tool_calls,
        artifacts=artifacts,
        result_text=result_text,
        error_message=error_message,
        denied_actions=[],
    )
