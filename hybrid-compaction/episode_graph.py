"""
episode_graph.py — CWL episode graph (Beyond Compaction, arXiv 2606.11213).

Ledger tipleri OTOMATİK çıkarır (tool adından). CWL ise ajanın KENDİSİNİN
`delimiter` tool'uyla açıkça bildirmesini sağlar:
  - episode tipi: "expl" (keşif) veya "act" (eylem)
  - bağımlılık:   bir act episode'un hangi expl episode'lara dayandığı

Bu, ledger'ın üstüne "ajan-kontrollü yapı" katmanı ekler. Fark:
  - ledger: "bu tool read'di" (kategori, otomatik)
  - CWL:    "bu 5 tool tek bir 'config keşfi' episode'u ve
             sonraki 'port yaz' eylemi buna bağlı" (ajan bildirir)

Eviction politikası (compactor Faz 6) bağımlılık grafiğini kullanır:
bir expl episode ANCAK ona bağlı tüm act'ler evict edildiyse atılabilir
(§5.2 CWL bağımlılık kısıtı).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Episode:
    name: str
    type: str                       # "expl" | "act"
    start_seq: int                  # bu episode'a ait ilk olayın seq'i
    end_seq: Optional[int] = None   # kapanış (None = hâlâ aktif)
    dependencies: list[str] = field(default_factory=list)   # act → expl adları
    description: str = ""            # expl kapanışında: eviction sonrası kalan tek şey
    event_seqs: list[int] = field(default_factory=list)     # içerdiği tool olayları

    @property
    def is_active(self) -> bool:
        return self.end_seq is None


class EpisodeGraph:
    """Ajanın delimiter çağrılarıyla inşa ettiği tiplenmiş bağımlılık grafiği."""

    def __init__(self) -> None:
        self.episodes: list[Episode] = []
        self._active: Optional[Episode] = None

    # --- delimiter protokolü (§5.2) — ajanın çağırdığı ------------------

    def start(self, name: str, type: str, seq: int,
              dependencies: Optional[list[str]] = None) -> None:
        """delimiter start — yeni episode aç (expl veya act)."""
        if type not in ("expl", "act"):
            raise ValueError(f"tip expl|act olmalı: {type}")
        if type == "act" and not dependencies:
            # act bağımlılık bildirmeli (§5.2); boşsa uyarı niteliğinde boş liste
            dependencies = []
        ep = Episode(name=name, type=type, start_seq=seq,
                     dependencies=list(dependencies or []))
        self.episodes.append(ep)
        self._active = ep

    def end(self, seq: int, description: str = "") -> None:
        """delimiter end — aktif episode'u kapat (expl'de description zorunlu)."""
        if self._active is None:
            return
        if self._active.type == "expl" and not description:
            description = f"{self._active.name} keşfi"
        self._active.end_seq = seq
        self._active.description = description
        self._active = None

    def attach(self, seq: int) -> None:
        """Bir tool olayını aktif episode'a bağla."""
        if self._active is not None:
            self._active.event_seqs.append(seq)

    # --- eviction politikası için sorgular ------------------------------

    def episode_of(self, seq: int) -> Optional[Episode]:
        for ep in self.episodes:
            if seq in ep.event_seqs:
                return ep
        return None

    def evictable_expl(self, evicted_seqs: set[int]) -> list[Episode]:
        """Evict edilebilir expl episode'lar: ona bağlı TÜM act'ler zaten
        evict edilmiş olanlar (§5.2 bağımlılık kısıtı)."""
        result = []
        for ep in self.episodes:
            if ep.type != "expl" or ep.is_active:
                continue
            # bu expl'e bağlı act'ler tamamen evict edildi mi?
            dependents = [e for e in self.episodes
                          if e.type == "act" and ep.name in e.dependencies]
            all_evicted = all(
                all(s in evicted_seqs for s in dep.event_seqs)
                for dep in dependents)
            if all_evicted:
                result.append(ep)
        return result

    def summary(self) -> str:
        parts = []
        for ep in self.episodes:
            deps = f" ←{ep.dependencies}" if ep.dependencies else ""
            parts.append(f"[{ep.type}] {ep.name}{deps} "
                         f"({len(ep.event_seqs)} olay)")
        return " · ".join(parts)
