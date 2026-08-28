"""PTC sandbox orkestrasyonu — Kubernetes Job olarak çalıştırma (Faz 2, T013).

Akış (research.md §1-§2, §5-§6; contracts/sandbox_job_contract.md):
1. Kodu bir ConfigMap'e yaz (`ptc-code-{run_id}`).
2. `k8s/sandbox/job-template.yaml`'dan bir Job oluştur (activeDeadlineSeconds=30,
   TOOL_GATEWAY_ENDPOINT enjekte edilmiş).
3. Job'un bitmesini bekle (Succeeded/Failed/DeadlineExceeded).
4. Pod log'unun son satırındaki JSON'u parse et.
5. ConfigMap + Job'u temizle (ttlSecondsAfterFinished zaten var, ama Principle V
   gereği açıkça da sileriz — beklemeden).

6. Hubble flow log'undan (best-effort) DENIED/DROPPED kayıtlarını oku; varsa
   `status = denied_action` (T020, data-model.md).

Not: Bu modül laptop üzerinde (ana asistanla birlikte) çalışır, cluster İÇİNDE
DEĞİL — bu yüzden `config.load_kube_config()` kullanılır (`load_incluster_config`
değil). tool_calls/denied_actions'ın Trace'e beslenmesi graph.py'de (T015/T021).

Faz 4 (T003-T005, specs/003-web-ui-live-trace/research.md §4): `run_sandbox`,
opsiyonel bir `on_event` callback'i alır ve adımları (configmap_created/
job_created/tool_call/denied_action/final) GERÇEKLEŞTİKÇE bu callback'e iletir —
web arayüzünün sol-alt canlı panelinin tek veri kaynağı budur. `on_event=None`
iken (CLI'nin mevcut çağrısı) davranış Faz 2'deki gibi kalır.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import yaml
from kubernetes import client, config

from grounded_assistant.models import (
    DeniedAction,
    LiveToolCall,
    SandboxRun,
    SandboxRunStatus,
    ToolCallStatus,
)

NAMESPACE = "default"
TOOL_GATEWAY_SERVICE_NAME = "tool-gateway"
TOOL_GATEWAY_PORT = 8443

# T020: `cilium hubble port-forward` (veya `kubectl -n kube-system port-forward
# svc/hubble-relay <port>:80`) elle açık olmalı — bkz. quickstart.md "Hubble
# gözlemi" bölümü. Ulaşılamazsa get_denied_actions sessizce [] döner
# (best-effort: gözlemlenebilirlik altyapısı, sandbox'ın kendi çalışmasını
# BLOKE etmemeli — engelleme zaten Cilium'da gerçekleşiyor, buradaki tek risk
# DeniedAction'ın KAYDEDİLMEMESİ).
HUBBLE_SERVER = os.environ.get("HUBBLE_SERVER", "localhost:4245")

_TEMPLATE_PATH = Path(__file__).resolve().parents[3] / "k8s" / "sandbox" / "job-template.yaml"

_POLL_INTERVAL_SECONDS = 1.0
# activeDeadlineSeconds (30) + Kubernetes'in bunu fark edip Job'u
# Failed/DeadlineExceeded'e çevirmesi için makul bir tampon (research.md §6).
_WAIT_TIMEOUT_SECONDS = 45.0


def _resolve_tool_gateway_endpoint(core_v1: client.CoreV1Api) -> str:
    """research.md §4.1: sandbox'ın DNS'e hiç ihtiyaç duymaması için Tool
    Gateway'in Service ClusterIP'si DNS adı yerine DOĞRUDAN IP olarak enjekte
    edilir — `sandbox-egress` policy'sinin (T016) tek kuralı `toEndpoints`
    (pod-etiketi bazlı, identity-based) olup DNS'e izin vermez; DNS adı
    kullanılsaydı sandbox kube-dns'e erişemeyip bu adımda tıkanırdı."""
    service = core_v1.read_namespaced_service(name=TOOL_GATEWAY_SERVICE_NAME, namespace=NAMESPACE)
    return f"http://{service.spec.cluster_ip}:{TOOL_GATEWAY_PORT}/mcp"


def _load_job_manifest(run_id: str, tool_gateway_endpoint: str) -> dict:
    template_text = _TEMPLATE_PATH.read_text(encoding="utf-8")
    filled = template_text.format(run_id=run_id, tool_gateway_endpoint=tool_gateway_endpoint)
    return yaml.safe_load(filled)


def _create_configmap(core_v1: client.CoreV1Api, run_id: str, code: str) -> None:
    configmap = client.V1ConfigMap(
        metadata=client.V1ObjectMeta(name=f"ptc-code-{run_id}"),
        data={"code.py": code},
    )
    core_v1.create_namespaced_config_map(namespace=NAMESPACE, body=configmap)


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
) -> tuple[SandboxRunStatus, str | None, list[LiveToolCall]]:
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
    status, result_text = SandboxRunStatus.ERROR, None
    seen_lines = 0

    while time.monotonic() < deadline:
        log_text = _read_pod_log(core_v1, job_name)
        lines = log_text.strip().splitlines() if log_text and log_text.strip() else []
        for line in lines[seen_lines:]:
            parsed = _parse_line(line)
            if parsed is None:
                continue  # entrypoint.py'nin kendi hata çıktısı olabilir, yok say
            if parsed.get("type") == "tool_call":
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
            elif "status" in parsed:
                status, result_text = SandboxRunStatus.ERROR, None
        seen_lines = len(lines)

        job = batch_v1.read_namespaced_job(name=job_name, namespace=NAMESPACE)
        for condition in job.status.conditions or []:
            if condition.reason == "DeadlineExceeded":
                return SandboxRunStatus.TIMEOUT, None, tool_calls
        if job.status.succeeded or job.status.failed:
            return status, result_text, tool_calls
        time.sleep(_POLL_INTERVAL_SECONDS)

    return SandboxRunStatus.TIMEOUT, None, tool_calls  # kendi güvenlik ağımız


def _read_pod_log(core_v1: client.CoreV1Api, job_name: str) -> str | None:
    pods = core_v1.list_namespaced_pod(namespace=NAMESPACE, label_selector=f"job-name={job_name}")
    if not pods.items:
        return None
    pod_name = pods.items[0].metadata.name
    try:
        # _preload_content=False: client'ın varsayılan davranışı, log JSON'a
        # benziyorsa onu sessizce parse edip str(dict) ile geri yazıyor — bu da
        # tırnak işaretlerini bozup contracts/sandbox_job_contract.md'nin
        # JSON kontratını geçersiz kılıyor (deneyle doğrulandı, 2026-08-28).
        # Ham yanıtı kendimiz decode ederek bunu atlıyoruz.
        raw = core_v1.read_namespaced_pod_log(
            name=pod_name, namespace=NAMESPACE, _preload_content=False
        )
        return raw.data.decode("utf-8")
    except client.ApiException:
        return None


def _parse_line(line: str) -> dict | None:
    """Bir pod log satırını JSON olarak ayrıştırır; geçersizse None döner
    (Faz 4, T004 — `_wait_and_stream`'in satır-satır kullandığı yardımcı;
    Faz 2'nin `_parse_log`'unun tek-satırlık hâli)."""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _destination_of(flow: dict) -> str:
    """Bir Hubble flow JSON'undan (jsonpb) 'IP:port' veya biliniyorsa pod adı ile
    okunabilir bir hedef metni üretir (DeniedAction.attempted_destination)."""
    destination_ip = flow.get("IP", {}).get("destination", "?")
    port = None
    for proto in ("TCP", "UDP"):
        if proto in flow.get("l4", {}):
            port = flow["l4"][proto].get("destination_port")
    dest_pod = flow.get("destination", {}).get("pod_name")
    host = dest_pod or destination_ip
    return f"{host}:{port}" if port else host


def get_denied_actions(run_id: str, job_name: str) -> list[DeniedAction]:
    """T020: Job'un pod'una ait Hubble flow log'unu (`hubble observe --verdict
    DROPPED -o json`) okuyup DeniedAction'a çevirir (data-model.md). Hubble'a
    ulaşılamazsa (CLI yok / hubble-relay'e port-forward açık değil) sessizce []
    döner — bkz. modül başındaki HUBBLE_SERVER notu."""
    try:
        proc = subprocess.run(
            [
                "hubble",
                "observe",
                "--server",
                HUBBLE_SERVER,
                "--pod",
                f"{NAMESPACE}/{job_name}",
                "--verdict",
                "DROPPED",
                "-o",
                "json",
                "--last",
                "200",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    denied_actions: list[DeniedAction] = []
    for line in proc.stdout.strip().splitlines():
        try:
            flow = json.loads(line)["flow"]
        except (json.JSONDecodeError, KeyError):
            continue  # hubble'ın kendi WARN/log satırları JSON değil, yok say
        denied_actions.append(
            DeniedAction(
                run_id=run_id,
                attempted_destination=_destination_of(flow),
                verdict=flow.get("verdict", "DROPPED"),
                observed_at=datetime.fromisoformat(flow["time"].replace("Z", "+00:00")),
            )
        )
    return denied_actions


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


def run_sandbox(code: str, on_event: Callable[[dict], None] | None = None) -> SandboxRun:
    """Verilen Python kodunu ayrı bir Kubernetes Job'unda (Cilium'un ağ
    seviyesinde kısıtladığı bir pod'da) çalıştırır, sonucu bir SandboxRun olarak
    döner.

    `on_event` (Faz 4, T003-T005): verilirse, çalıştırmanın her adımı
    GERÇEKLEŞTİKÇE bir sözlük olarak bu callback'e iletilir (contracts/
    websocket_protocol.md'deki `ptc_event` şeması) — CLI bunu hiç kullanmaz
    (`on_event=None`), davranışı Faz 2'deki gibi kalır."""
    config.load_kube_config()
    core_v1 = client.CoreV1Api()
    batch_v1 = client.BatchV1Api()

    run_id = uuid.uuid4().hex[:12]
    job_name = f"ptc-sandbox-{run_id}"
    started_at = datetime.now(UTC)

    _create_configmap(core_v1, run_id, code)
    _emit(on_event, {"stage": "configmap_created", "run_id": run_id})

    tool_gateway_endpoint = _resolve_tool_gateway_endpoint(core_v1)
    job_manifest = _load_job_manifest(run_id, tool_gateway_endpoint)
    batch_v1.create_namespaced_job(namespace=NAMESPACE, body=job_manifest)
    _emit(on_event, {"stage": "job_created", "run_id": run_id, "code": code})

    status, result_text, tool_calls = _wait_and_stream(core_v1, batch_v1, job_name, on_event)

    denied_actions = get_denied_actions(run_id, job_name)
    for action in denied_actions:
        _emit(
            on_event,
            {
                "stage": "denied_action",
                "run_id": run_id,
                "attempted_destination": action.attempted_destination,
                "verdict": action.verdict,
                "timestamp": action.observed_at.isoformat(),
            },
        )
    if denied_actions:
        # data-model.md akış özeti: onaysız bir hedefe deneme, sonucu ne olursa
        # olsun status'u denied_action'a çevirir (FR-011 — result_text de None olmalı).
        status, result_text = SandboxRunStatus.DENIED_ACTION, None

    _cleanup(core_v1, batch_v1, run_id, job_name)
    _emit(
        on_event,
        {"stage": "final", "run_id": run_id, "status": status.value, "result_text": result_text},
    )

    return SandboxRun(
        run_id=run_id,
        code=code,
        started_at=started_at,
        status=status,
        finished_at=datetime.now(UTC),
        tool_calls=tool_calls,
        result_text=result_text,
        denied_actions=denied_actions,
    )
