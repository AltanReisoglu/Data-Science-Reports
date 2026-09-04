"""Artifact metadata deposu — "artifact registry", sadece blob değil.

Kubeflow'un ayrımını küçük ölçekte uyguluyoruz: **bayt** nesne deposunda,
**baytın hikâyesi** burada. Bu ayrım olmadan elde artifact store değil sadece
bir bucket olur.

## Neden MLMD değil, kendi şemamız

Red Hat, OpenShift AI 2.23'te Model Registry'den MLMD sunucusunu KALDIRDI ve
bileşeni kendi API'si + kendi DB şeması üzerinden doğrudan veritabanına
bağladı; gerekçe olarak "mimariyi basitleştirmek, uzun vadeli sürdürülebilirlik"
gösterildi. Aynı yönü izliyoruz.

## Neden SQLite (şimdilik)

Yine Red Hat'in kendi çizgisi: PostgreSQL üretim için gerekli (performans,
eşzamanlılık, ölçek); SQLite yalnızca yerel geliştirme ve test için. PoC
SQLite ile başlar, SQL taşınabilir yazıldığı için Postgres'e geçiş mekaniktir
(`open_postgres`).

## İki değişmez kural

1. **Immutability.** Artifact güncellenmez. Yeni sürüm = YENİ `artifact_id`.
   Bu yüzden burada `update` diye bir metot YOK — Modal'ın "last write wins"
   tuzağı böylece yapısal olarak oluşmuyor.

2. **Silme sırası: ÖNCE bayt, SONRA metadata.** MLflow'da metadata silinip
   blob'ların kalması bilinen bir sorun (yetim artifact: listelemede görünmez,
   ücreti işler). Ters sıra ise "sarkan referans" bırakır — o TESPİT EDİLEBİLİR
   ve onarılabilir. Görünmez maliyet yerine görünür tutarsızlığı tercih ediyoruz.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from grounded_assistant.artifacts.serialize import VARSAYILAN_TIP

_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id   TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    workflow_id   TEXT NOT NULL,
    node_id       TEXT,
    run_id        TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    content_type  TEXT NOT NULL,
    size_bytes    BIGINT NOT NULL,
    storage_uri   TEXT NOT NULL,
    parents       TEXT NOT NULL,
    owner         TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    ttl_seconds   BIGINT,
    -- 2026-09-04, OpenShift/KFP hizalaması: orada her artifact TİPLİ
    -- (system.Dataset / system.Model / system.Metrics ...) ve serbest bir
    -- `.metadata` anahtar-değer torbası taşıyor. İkisi de burada.
    artifact_type TEXT,
    user_metadata TEXT
)
"""

#: Var olan kurulumları taşımak için. `CREATE TABLE IF NOT EXISTS` eski tabloya
#: dokunmaz; bu iki sütun elle eklenmeli. Zaten varsa sessizce geçilir.
_MIGRASYONLAR = (
    "ALTER TABLE artifacts ADD COLUMN artifact_type TEXT",
    "ALTER TABLE artifacts ADD COLUMN user_metadata TEXT",
)

# workflow+name: "extract.tickets'ın en yenisi" sorgusu (node'lar arası keşif).
# workflow+hash: cached() — aynı içerik zaten üretilmiş mi.
_INDEXES = (
    "CREATE INDEX IF NOT EXISTS ix_artifacts_wf_name ON artifacts(workflow_id, name)",
    "CREATE INDEX IF NOT EXISTS ix_artifacts_wf_hash ON artifacts(workflow_id, content_hash)",
    "CREATE INDEX IF NOT EXISTS ix_artifacts_created ON artifacts(created_at)",
)

_COLUMNS = (
    "artifact_id", "name", "workflow_id", "node_id", "run_id",
    "content_hash", "content_type", "size_bytes", "storage_uri",
    "parents", "owner", "created_at", "ttl_seconds",
    "artifact_type", "user_metadata",
)


class ArtifactExists(Exception):
    """Aynı artifact_id ikinci kez yazılmaya çalışıldı — immutability ihlali."""


@dataclass(frozen=True)
class ArtifactMeta:
    """Bir artifact'in kimliği ve hikâyesi. Bayt burada DEĞİL."""

    artifact_id: str
    name: str
    workflow_id: str
    run_id: str
    content_hash: str
    content_type: str
    size_bytes: int
    storage_uri: str
    owner: str
    created_at: datetime
    node_id: str | None = None
    parents: tuple[str, ...] = ()
    #: KFP'nin `system.*` şema başlıklarıyla aynı sözlük (bkz. TIPLER).
    artifact_type: str = VARSAYILAN_TIP
    #: KFP'deki `.metadata` — çağıranın koyduğu serbest anahtar-değer.
    user_metadata: dict = field(default_factory=dict)
    ttl_seconds: int | None = None

    def expires_at(self) -> datetime | None:
        if self.ttl_seconds is None:
            return None
        return datetime.fromtimestamp(
            self.created_at.timestamp() + self.ttl_seconds, tz=UTC
        )

    def is_expired(self, now: datetime | None = None) -> bool:
        son = self.expires_at()
        return son is not None and (now or datetime.now(UTC)) >= son


@dataclass
class MetadataStore:
    """SQLite/PostgreSQL üzerinde artifact kaydı.

    SQL taşınabilir tutuldu: yalnızca TEXT/BIGINT, zaman ISO-8601 metni,
    `parents` JSON metni. Tek sürücüye özgü şey parametre yer tutucusu
    (`?` vs `%s`) — o da `placeholder` ile veriliyor.
    """

    connection: object
    placeholder: str = "?"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        with self._lock:
            cur = self.connection.cursor()
            cur.execute(_SCHEMA)
            for stmt in _MIGRASYONLAR:
                try:
                    cur.execute(stmt)
                except Exception:  # noqa: BLE001 — sütun zaten varsa sorun değil
                    self.connection.rollback()
            for stmt in _INDEXES:
                cur.execute(stmt)
            self.connection.commit()

    # -- yazma ------------------------------------------------------------

    def create(self, meta: ArtifactMeta) -> ArtifactMeta:
        """Yeni bir artifact kaydeder. Var olan bir id'yi EZMEZ (immutability)."""
        if self.get(meta.artifact_id) is not None:
            raise ArtifactExists(
                f"{meta.artifact_id} zaten var — artifact'ler değişmezdir, "
                "yeni sürüm için yeni bir artifact_id üretin"
            )
        ph = ", ".join([self.placeholder] * len(_COLUMNS))
        with self._lock:
            self.connection.cursor().execute(
                f"INSERT INTO artifacts ({', '.join(_COLUMNS)}) VALUES ({ph})",
                self._to_row(meta),
            )
            self.connection.commit()
        return meta

    def delete(self, artifact_id: str) -> bool:
        """YALNIZCA metadata kaydını siler.

        Çağıran, baytları BUNDAN ÖNCE silmiş olmalı — modül başlığındaki
        "silme sırası" notuna bakın.
        """
        with self._lock:
            cur = self.connection.cursor()
            cur.execute(
                f"DELETE FROM artifacts WHERE artifact_id = {self.placeholder}",
                (artifact_id,),
            )
            self.connection.commit()
            return cur.rowcount > 0

    # -- okuma ------------------------------------------------------------

    def get(self, artifact_id: str) -> ArtifactMeta | None:
        rows = self._query(
            f"SELECT * FROM artifacts WHERE artifact_id = {self.placeholder}",
            (artifact_id,),
        )
        return next(rows, None)

    def latest_by_name(self, workflow_id: str, name: str) -> ArtifactMeta | None:
        """Bir workflow içinde o isimle üretilmiş EN YENİ artifact.

        Artifact'ler değişmez olduğu için aynı isimde birden çok kayıt olur;
        `get_artifact(name=...)` çağrısının beklediği şey sonuncusudur.
        """
        rows = self._query(
            "SELECT * FROM artifacts "
            f"WHERE workflow_id = {self.placeholder} AND name = {self.placeholder} "
            "ORDER BY created_at DESC, artifact_id DESC",
            (workflow_id, name),
        )
        return next(rows, None)

    def find_by_hash(self, workflow_id: str, content_hash: str) -> ArtifactMeta | None:
        """`cached()` için: bu içerik bu workflow'da zaten üretilmiş mi?"""
        rows = self._query(
            "SELECT * FROM artifacts "
            f"WHERE workflow_id = {self.placeholder} AND content_hash = {self.placeholder} "
            "ORDER BY created_at ASC",
            (workflow_id, content_hash),
        )
        return next(rows, None)

    def list(
        self, workflow_id: str, node_id: str | None = None, limit: int = 100
    ) -> list[ArtifactMeta]:
        """Keşif: workflow'da (istenirse belirli bir node'da) ne üretilmiş.

        D okuması (node'lu workflow) için taşıyıcı öğe: `transform` node'unun
        agent'ı, `extract` koşarken orada DEĞİLDİ — ne bulacağını buradan öğrenir.
        """
        sql = f"SELECT * FROM artifacts WHERE workflow_id = {self.placeholder}"
        params: tuple = (workflow_id,)
        if node_id is not None:
            sql += f" AND node_id = {self.placeholder}"
            params += (node_id,)
        sql += f" ORDER BY created_at DESC LIMIT {self.placeholder}"
        return list(self._query(sql, params + (limit,)))

    def expired(self, now: datetime | None = None) -> list[ArtifactMeta]:
        """Süresi dolmuş kayıtlar — GC'nin girdisi."""
        an = now or datetime.now(UTC)
        return [
            m
            for m in self._query("SELECT * FROM artifacts WHERE ttl_seconds IS NOT NULL", ())
            if m.is_expired(an)
        ]

    # -- iç ---------------------------------------------------------------

    def _query(self, sql: str, params: tuple) -> Iterator[ArtifactMeta]:
        with self._lock:
            cur = self.connection.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
        for row in rows:
            yield self._from_row(row)

    @staticmethod
    def _to_row(m: ArtifactMeta) -> tuple:
        return (
            m.artifact_id, m.name, m.workflow_id, m.node_id, m.run_id,
            m.content_hash, m.content_type, m.size_bytes, m.storage_uri,
            json.dumps(list(m.parents)), m.owner, m.created_at.isoformat(),
            m.ttl_seconds, m.artifact_type,
            json.dumps(m.user_metadata) if m.user_metadata else None,
        )

    @staticmethod
    def _from_row(row: tuple) -> ArtifactMeta:
        return ArtifactMeta(
            artifact_id=row[0], name=row[1], workflow_id=row[2], node_id=row[3],
            run_id=row[4], content_hash=row[5], content_type=row[6],
            size_bytes=row[7], storage_uri=row[8],
            parents=tuple(json.loads(row[9])), owner=row[10],
            created_at=datetime.fromisoformat(row[11]), ttl_seconds=row[12],
            artifact_type=(row[13] if len(row) > 13 else None) or VARSAYILAN_TIP,
            user_metadata=json.loads(row[14]) if len(row) > 14 and row[14] else {},
        )


def open_sqlite(path: str | Path) -> MetadataStore:
    """PoC varsayılanı. `:memory:` testler için."""
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return MetadataStore(connection=conn, placeholder="?")


def open_postgres(dsn: str) -> MetadataStore:
    """Üretim yolu. `psycopg` gerektirir — bilerek TEMBEL import edildi ki
    PoC'nin SQLite yolu yeni bir bağımlılık getirmesin."""
    import psycopg  # noqa: PLC0415

    return MetadataStore(connection=psycopg.connect(dsn), placeholder="%s")
