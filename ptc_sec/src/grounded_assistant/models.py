"""Faz 1 veri modeli (bkz. specs/001-ptc-grounded-assistant/data-model.md)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SourceId(str, Enum):
    POLICY = "policy"
    WIKI = "wiki"
    SUPPORT_TICKETS = "support_tickets"
    TECHNICAL_DOCS = "technical_docs"


class SourceStatus(str, Enum):
    OK = "ok"
    EMPTY = "empty"
    ERROR = "error"


class ToolCallStatus(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    UNAVAILABLE = "unavailable"  # DSH ilhamlı: cevaplayıcı/politika hiç yanıt vermedi


class AccessPath(str, Enum):
    KNOWLEDGE_BASE = "knowledge_base"
    LIVE_SYSTEM = "live_system"
    PTC_SANDBOX = "ptc_sandbox"  # Faz 2


class SandboxRunStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    DENIED_ACTION = "denied_action"


@dataclass(frozen=True)
class Query:
    text: str
    session_id: str
    timestamp: datetime


@dataclass(frozen=True)
class RetrievalHit:
    doc_id: str
    snippet: str
    bm25_rank: int | None
    dense_rank: int | None
    rrf_score: float


@dataclass(frozen=True)
class KnowledgeBaseSource:
    source_id: SourceId
    status: SourceStatus
    hits: list[RetrievalHit] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status is SourceStatus.ERROR and self.hits:
            raise ValueError("status='error' olan bir kaynağın hits listesi boş olmalı (FR-010)")


@dataclass(frozen=True)
class LiveToolCall:
    tool_name: str
    arguments: dict
    timestamp: datetime
    status: ToolCallStatus
    result: str | None = None

    def __post_init__(self) -> None:
        if self.status is not ToolCallStatus.SUCCESS and self.result is not None:
            raise ValueError("status != 'success' olduğunda result None olmalı")


@dataclass(frozen=True)
class Answer:
    text: str
    grounded: bool
    access_paths_used: list[AccessPath] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    partial_failure_notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.grounded and not self.source_refs:
            raise ValueError("grounded=True iken source_refs boş olamaz (SC-001)")


@dataclass
class SandboxRun:
    """Faz 2 — bir PTC çalıştırmasının (bir Kubernetes Job'unun) kaydı.

    Not: diğer entity'lerin aksine mutable — status/finished_at/tool_calls,
    Job tamamlanana kadar sandbox_runner.py tarafından güncellenir.
    """

    run_id: str
    code: str
    started_at: datetime
    status: SandboxRunStatus = SandboxRunStatus.RUNNING
    finished_at: datetime | None = None
    tool_calls: list[LiveToolCall] = field(default_factory=list)
    result_text: str | None = None
    denied_actions: list[DeniedAction] = field(default_factory=list)

    def __post_init__(self) -> None:
        _failed = (SandboxRunStatus.ERROR, SandboxRunStatus.TIMEOUT, SandboxRunStatus.DENIED_ACTION)
        if self.status in _failed and self.result_text is not None:
            raise ValueError("başarısız bir SandboxRun'da result_text None olmalı (FR-011)")


@dataclass(frozen=True)
class CapabilityGrant:
    """Faz 2 — bir SandboxRun'a hangi erişimin tanındığı (tool_policy.KNOWN_TOOLS'un
    bu çalıştırmaya özgü izdüşümü)."""

    run_id: str
    tool_gateway_endpoint: str
    allowed_tools: tuple[str, ...]


@dataclass(frozen=True)
class DeniedAction:
    """Faz 2 — Cilium/Hubble'ın ağ seviyesinde engellediği bir erişim girişiminin
    kaydı (uygulama kodumuzun ürettiği bir şey değil, Hubble flow log'undan)."""

    run_id: str
    attempted_destination: str
    verdict: str
    observed_at: datetime
