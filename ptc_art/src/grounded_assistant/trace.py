"""İzlenebilirlik: her yanıt için hangi erişim yolu/kaynağın kullanıldığını kaydeder (FR-009).

Bkz. specs/001-ptc-grounded-assistant/data-model.md -> Answer.source_refs / partial_failure_notes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from grounded_assistant.models import (
    AccessPath,
    ArtifactEvent,
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
        self._turn_start: int = 0

    def mark(self) -> int:
        """Altan'ın kararı (2026-08-31, web UI'de bulunan gerçek bir hata):
        web/app.py, tek bir WebSocket bağlantısı boyunca (konuşma hafızasını
        korumak için) AYNI Trace nesnesini tüm sorularda paylaşıyor — bu
        olmadan, ikinci sorunun cevabı BİRİNCİ sorunun kayıtlarını da 'kaynak'
        sayıyordu (ör. bir soru timeout olsa bile önceki başarılı kayıtlar
        yüzünden 'grounded' görünüyordu). `mark()`/`since()` çifti, her turun
        SADECE kendi eklediği kayıtları görmesini sağlar.

        Aynı zamanda `_turn_start`'ı da günceller (2026-09-01, Altan'ın kararı)
        — `run_ptc_code`'un (graph.py) bir turda kaç kez sandbox çalıştırdığını
        sayabilmesi için (bkz. `sandbox_run_count`); agent, bir hedef
        engellendiğinde sınırsız tekrar deneyip her seferinde yeni bir
        ConfigMap+Job+Pod yaratabiliyordu — bu, o davranışı sınırlamanın
        temeli."""
        self._turn_start = len(self._entries)
        return self._turn_start

    def sandbox_run_count(self) -> int:
        """Bu TURDA (en son `mark()` çağrısından bu yana) kaç `SandboxRun`
        kaydedildiğini döner — `run_ptc_code`'un retry sınırı için."""
        return sum(
            1
            for entry in self._entries[self._turn_start :]
            if entry.access_path is AccessPath.PTC_SANDBOX and not entry.detail.startswith("denied:")
        )

    def since(self, mark: int) -> "Trace":
        """`mark()`'tan bu yana eklenen kayıtları İÇEREN yeni, bağımsız bir
        Trace görünümü döner — orijinal `_entries` listesini paylaşmaz."""
        view = Trace()
        view._entries = list(self._entries[mark:])
        return view

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

    def record_artifact_event(self, event: ArtifactEvent) -> None:
        """2026-09-03: bir çalıştırmanın artifact deposuyla teması.

        `detail` biçimi bilinçli: `produced:extract.tickets` / `consumed:...`.
        Böylece bir cevabın `source_refs`'ine baktığınızda verinin canlı
        sistemden TAZE mi geldiğini yoksa saklanmış bir artifact'ten mi
        okunduğunu ayırt edebiliyorsunuz — kalıcılık eklendiği anda bu ayrım
        izlenebilirliğin merkezine oturuyor.

        Durum her zaman "ok": buraya ulaşan bir olay zaten gerçekleşmiş
        demektir (başarısız bir artifact çağrısı `tool_call` kaydında
        `error` olarak görünür).
        """
        self._entries.append(
            TraceEntry(
                AccessPath.ARTIFACT_STORE,
                f"{event.op.value}:{event.name}",
                SourceStatus.OK.value,
                event.timestamp,
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
