"""
ledger.py — Deterministik execution ledger (Ledger, arXiv 2608.00808).

§6.5.1: LLM ÇAĞIRMADAN, üç kayıt tipi + iki sayaçla:
  - observation record: hangi dosyanın hangi içeriği DÖNDÜ
  - command record:     normalize komut + kategori
  - change counter:     local (dosya başına) + global

Bununla iki şeyi tespit eder:
  - DEDUP:     aynı komut, değişmemiş koşulda tekrar → sonuç yeniden kullanılabilir
  - STALENESS: bir dosya okunduktan SONRA değişmişse, o gözlem bayat

Hiçbir adım LLM gerektirmez — "önce ucuz deterministik" ilkesi (§5 complexity trap).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
import re


@dataclass
class ObservationRecord:
    """Bir okuma gözlemi: hangi KAYNAK, hangi içerik döndü, hangi sürümde/adımda."""
    path: str               # kaynak anahtarı (dosya yolu VEYA ticker/tablo/endpoint)
    content_hash: int
    local_counter: int      # okuma anında kaynağın sürüm sayacı
    global_counter: int     # okuma anında global sayaç değeri
    event_seq: int          # bu gözlemi üreten trace olayının seq'i
    step: int = 0           # okuma anındaki mantıksal saat (TTL için)
    ttl: Optional[int] = None  # kaynak volatilse: kaç adım sonra bayat (zaman-eskimesi)


@dataclass
class CommandRecord:
    """Bir komut: normalize edilmiş imza + kategori + konum."""
    signature: str          # normalize "tool(arg=val)" imzası
    category: str           # "read" | "search" | "write" | "test" | "other"
    event_seq: int


# Hangi tool hangi kategori — dedup/eviction kararları buna bakar
_CATEGORY = {
    "read_file": "read",
    "list_dir": "search",
    "grep": "search",
    "edit_file": "write",
    "run_tests": "test",
}


def _normalize(name: str, args: dict) -> str:
    """Komut imzası — incidental farkları (boşluk, sıra) eler."""
    parts = ",".join(f"{k}={str(v).strip()}" for k, v in sorted(args.items()))
    return f"{name}({parts})"


class ExecutionLedger:
    """Trace'in yanında tutulan deterministik yürütme durumu.

    tool_meta verilmezse dosya senaryosu gibi davranır (geriye uyumlu). Verilirse
    ledger DOMAIN-BAĞIMSIZ olur — her tool kendi sözleşmesini bildirir:
      {"cat": "read|search|write|test|other",
       "resource": callable(args)->str,   # kaynak anahtarı (path/ticker/tablo…)
       "ttl": Optional[int]}               # volatil kaynak: kaç adım sonra bayat
    Böylece aynı ledger hem `config.py` dosyalarına hem `XOM` ticker'ına çalışır.
    """

    def __init__(self, tool_meta: Optional[dict] = None) -> None:
        self.observations: list[ObservationRecord] = []
        self.commands: list[CommandRecord] = []
        self.local_counters: dict[str, int] = {}   # kaynak → sürüm sayacı
        self.global_counter: int = 0
        self.tool_meta = tool_meta                  # None ise dosya varsayılanı
        self.step: int = 0                          # mantıksal saat (TTL için)

    # --- tool sözleşmesi (genel) ----------------------------------------

    def _category(self, name: str) -> str:
        if self.tool_meta and name in self.tool_meta:
            return self.tool_meta[name].get("cat", "other")
        return _CATEGORY.get(name, "other")

    def _resource(self, name: str, args: dict) -> str:
        """Kaynak anahtarı: meta'daki extractor veya varsayılan args['path']."""
        if self.tool_meta and name in self.tool_meta:
            fn = self.tool_meta[name].get("resource")
            if fn is not None:
                return str(fn(args))
        return str(args.get("path", ""))

    def _ttl(self, name: str) -> Optional[int]:
        if self.tool_meta and name in self.tool_meta:
            return self.tool_meta[name].get("ttl")
        return None

    # --- kayıt (her tool çağrısından sonra çağrılır) ---------------------

    def record(self, name: str, args: dict, output: str, event_seq: int) -> dict:
        """Bir tool etkileşimini kaydet. Tespit sonuçlarını döndürür.

        Döndürdüğü dict:
          {"category", "duplicate_of", "invalidated": [seq,...]}
        """
        self.step += 1                       # mantıksal saat ilerler
        category = self._category(name)
        sig = _normalize(name, args)

        result = {"category": category, "duplicate_of": None, "invalidated": []}

        # WRITE → sürüm sayacını artır ve AYNI kaynağın gözlemlerini bayatlat
        if category == "write":
            path = self._resource(name, args)
            self.local_counters[path] = self.local_counters.get(path, 0) + 1
            self.global_counter += 1
            for obs in self.observations:
                if obs.path == path and obs.local_counter < self.local_counters[path]:
                    result["invalidated"].append(obs.event_seq)

        # READ → gözlem kaydı oluştur, önce dedup kontrolü
        if category == "read":
            path = self._resource(name, args)
            lc = self.local_counters.get(path, 0)
            # Aynı kaynak, aynı sürümde daha önce okunduysa → duplicate
            for obs in self.observations:
                if obs.path == path and obs.local_counter == lc:
                    result["duplicate_of"] = obs.event_seq
                    break
            self.observations.append(ObservationRecord(
                path=path, content_hash=hash(output),
                local_counter=lc, global_counter=self.global_counter,
                event_seq=event_seq, step=self.step, ttl=self._ttl(name)))

        # SEARCH → exact-repeat dedup (değişmemiş koşulda)
        if category == "search":
            for cmd in self.commands:
                if cmd.signature == sig and cmd.category == "search":
                    result["duplicate_of"] = cmd.event_seq
                    break

        self.commands.append(CommandRecord(signature=sig, category=category,
                                           event_seq=event_seq))
        return result

    # --- sorgu (eviction politikası buna bakar) --------------------------

    def is_stale(self, event_seq: int) -> bool:
        """Bu gözlem bayat mı? İKİ nedenle bayat olabilir:
          1) MUTASYON — sonradan aynı kaynağa yazma yapıldı (sürüm eskidi)
          2) VOLATİLİTE — kaynak TTL'li ve okumanın üstünden ttl adımdan çok geçti
             (fiyat/web/saat gibi zamanla bayatlayan kaynaklar; yazma gerekmez)
        """
        for obs in self.observations:
            if obs.event_seq == event_seq:
                current = self.local_counters.get(obs.path, 0)
                if obs.local_counter < current:           # 1) mutasyon
                    return True
                if obs.ttl is not None and (self.step - obs.step) > obs.ttl:  # 2) volatilite
                    return True
                return False
        return False

    def category_of(self, event_seq: int) -> str:
        for cmd in self.commands:
            if cmd.event_seq == event_seq:
                return cmd.category
        return "other"

    def stats(self) -> dict:
        dup = sum(1 for c in self.commands
                  if any(o.signature == c.signature and o.event_seq < c.event_seq
                         for o in self.commands))
        return {"observations": len(self.observations),
                "commands": len(self.commands),
                "writes": self.global_counter}
