"""İzlenebilirlik: her yanıt için hangi erişim yolu/kaynağın kullanıldığını kaydeder (FR-009).

Bkz. specs/001-ptc-grounded-assistant/data-model.md -> Answer.source_refs / partial_failure_notes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from grounded_assistant.models import (
    AccessPath,
    DeniedAction,
    KnowledgeBaseSource,
    LiveToolCall,
    SandboxRun,
    SourceStatus,
    ToolCallStatus,
)

_SUCCESS_STATUSES = {SourceStatus.OK.value, ToolCallStatus.SUCCESS.value}


@dataclass(frozen=True)
class TraceEntry:
    access_path: AccessPath
    detail: str
    status: str
    timestamp: datetime


class Trace:
    def __init__(self) -> None:
        self._entries: list[TraceEntry] = []

    def record_kb_source(self, source: KnowledgeBaseSource, timestamp: datetime) -> None:
        self._entries.append(
            TraceEntry(AccessPath.KNOWLEDGE_BASE, source.source_id.value, source.status.value, timestamp)
        )

    def record_tool_call(self, call: LiveToolCall) -> None:
        self._entries.append(
            TraceEntry(AccessPath.LIVE_SYSTEM, call.tool_name, call.status.value, call.timestamp)
        )

    def record_sandbox_run(self, run: SandboxRun) -> None:
        """Faz 2, SC-003: her sandbox çalıştırması (başarı/hata/timeout/
        denied_action) izlenebilirlik kaydında görünür — `run.tool_calls`
        zaten `record_tool_call` ile ayrıca kaydedilir (T015); bu, ÇALIŞTIRMANIN
        KENDİSİNİ de görünür kılar (ör. hiç tool çağrısı yapmadan timeout olması)."""
        self._entries.append(
            TraceEntry(
                AccessPath.PTC_SANDBOX,
                run.run_id,
                run.status.value,
                run.finished_at or run.started_at,
            )
        )

    def record_denied_action(self, action: DeniedAction) -> None:
        """T021, SC-002: Cilium/Hubble'ın ağ seviyesinde engellediği bir
        erişim girişimi — `action.verdict` (Hubble'ın kendi terimi, ör.
        'DROPPED') asla `_SUCCESS_STATUSES`'ta olamayacağından bu her zaman
        partial_failure_notes'a düşer, source_refs'e asla katkı sağlamaz."""
        self._entries.append(
            TraceEntry(
                AccessPath.PTC_SANDBOX,
                f"denied:{action.attempted_destination}",
                action.verdict,
                action.observed_at,
            )
        )

    def access_paths_used(self) -> list[AccessPath]:
        """Answer.access_paths_used -- fiilen çağrılan erişim yolları, ilk görülme sırasıyla."""
        seen: list[AccessPath] = []
        for entry in self._entries:
            if entry.access_path not in seen:
                seen.append(entry.access_path)
        return seen

    def source_refs(self) -> list[str]:
        """Answer.source_refs -- başarılı katkı sağlayan kaynak/tool adları."""
        return [e.detail for e in self._entries if e.status in _SUCCESS_STATUSES]

    def partial_failure_notes(self) -> list[str]:
        """Answer.partial_failure_notes -- başarısız/boş/erişilemeyen kaynak/tool notları."""
        return [f"{e.detail}: {e.status}" for e in self._entries if e.status not in _SUCCESS_STATUSES]

    def to_json(self) -> str:
        """CLI --trace çıktısı (contracts/cli_interface.md)."""
        records = [
            {
                "access_path": e.access_path.value,
                "detail": e.detail,
                "status": e.status,
                "timestamp": e.timestamp.isoformat(),
            }
            for e in self._entries
        ]
        return json.dumps(records, ensure_ascii=False, indent=2)
