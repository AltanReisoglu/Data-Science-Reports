"""ARŞİV — 2026-09-03'te sandbox_runner.py'den çıkarılan Cilium/Hubble kodu.

Neden çıkarıldı:
  (a) Egress policy artık projenin konusu değil (case: artifact persistence).
  (b) Hedef ortam OpenShift; oradaki CNI OVN-Kubernetes — `hubble` yok.
  (c) İki `hubble observe` subprocess'i (her biri timeout=10) `_cleanup`'tan
      ÖNCE, sonucu beklenerek kritik yolda çalışıyordu.

Egress tarafına dönülürse başlangıç noktası burasıdır. Çalışması için
`hubble` CLI'ı ve hubble-relay'e açık bir port-forward gerekir.
"""

import json
import os
import subprocess
from datetime import UTC, datetime

from grounded_assistant.models import DeniedAction

NAMESPACE = "default"
HUBBLE_SERVER = os.environ.get("HUBBLE_SERVER", "localhost:4245")


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


def _raw_flow_line(flow: dict) -> str:
    """`hubble observe`'un terminaldeki çıktısına benzer HAM bir satır."""
    verdict = flow.get("verdict", "DROPPED")
    destination_ip = flow.get("IP", {}).get("destination", "?")
    l4 = flow.get("l4", {})
    drop_reason = flow.get("drop_reason_desc", "")
    return f"{verdict} | {destination_ip} | {l4} | {drop_reason}"


def _query_dropped_flows(pod_selector: str, since: str | None = None) -> list[dict]:
    """`hubble observe --verdict DROPPED -o json`; ulaşılamazsa sessizce []."""
    cmd = [
        "hubble", "observe", "--server", HUBBLE_SERVER,
        "--pod", pod_selector, "--verdict", "DROPPED",
        "-o", "json", "--last", "200",
    ]
    if since is not None:
        cmd.extend(["--since", since])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    flows: list[dict] = []
    for line in proc.stdout.strip().splitlines():
        try:
            flows.append(json.loads(line)["flow"])
        except (json.JSONDecodeError, KeyError):
            continue
    return flows


def get_denied_actions(run_id: str, job_name: str, started_at: datetime) -> list[DeniedAction]:
    """Sandbox pod'unun VE Tool Gateway'in egress'inin DROPPED kayıtları.
    Gateway paylaşılan bir Deployment olduğu için sorgu `--since` ile bu run'ın
    başlangıcına sınırlanır (yoksa başka bir run'ın drop'u buraya mal edilir)."""
    denied_actions: list[DeniedAction] = []

    for flow in _query_dropped_flows(f"{NAMESPACE}/{job_name}"):
        denied_actions.append(
            DeniedAction(
                run_id=run_id,
                attempted_destination=_destination_of(flow),
                verdict=flow.get("verdict", "DROPPED"),
                observed_at=datetime.fromisoformat(flow["time"].replace("Z", "+00:00")),
                source_pod="sandbox",
                raw_flow=_raw_flow_line(flow),
            )
        )

    elapsed_seconds = int((datetime.now(UTC) - started_at).total_seconds()) + 5
    for flow in _query_dropped_flows(f"{NAMESPACE}/tool-gateway", since=f"{elapsed_seconds}s"):
        denied_actions.append(
            DeniedAction(
                run_id=run_id,
                attempted_destination=_destination_of(flow),
                verdict=flow.get("verdict", "DROPPED"),
                observed_at=datetime.fromisoformat(flow["time"].replace("Z", "+00:00")),
                source_pod="tool-gateway",
                raw_flow=_raw_flow_line(flow),
            )
        )

    return denied_actions
