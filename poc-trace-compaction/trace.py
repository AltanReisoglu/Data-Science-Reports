"""
trace.py — Tool trace'in veri yapısı.

§13 §4 ve §7: tiplenmiş olay akışı (Format 3) + 5-alan trace-özet birimi.
Bir Event, ajan döngüsündeki tek bir adımdır (reasoning / tool / answer).
Bir TraceSummary, sıkıştırılmış bir tool biriminin korunan çekirdeğidir.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import json

from config import estimate_tokens


# --- Format 3: tiplenmiş olay (§7) ---------------------------------------

@dataclass
class Event:
    """Trace'teki tek bir tiplenmiş olay.

    type: "reasoning" | "tool" | "answer"
    - reasoning: niyet buraya düşer (§4 — ham trace'te olmayan, en değerli alan)
    - tool:      {name, args, output}; output verbatim tutulur
    - answer:    nihai metin
    """
    seq: int
    type: str
    payload: dict[str, Any]
    status: str = "ok"
    intent_ref: Optional[int] = None   # bu tool'u tetikleyen reasoning olayının seq'i
    verbatim: bool = False             # output parafraz edilmemeli mi (hata/ID/sayı)
    evicted: bool = False              # eviction sonrası ham içerik atıldı mı
    summary: Optional["TraceSummary"] = None  # atıldıysa 5-alan özet burada
    cleared: bool = False              # B.11 context editing: içerik SİLİNDİ (özet değil)
    clear_note: str = ""               # silme yer tutucusundaki tek satırlık iz

    def token_cost(self) -> int:
        """Bu olayın bağlamda kapladığı tahmini token."""
        # B.11 context editing: silinmiş birim yalnızca yer tutucu kadar yer kaplar
        if self.cleared:
            return estimate_tokens(self.clear_note) + 4
        # B.12 compaction: özetlenmiş birim 5-alan özet kadar yer kaplar
        if self.evicted and self.summary is not None:
            return self.summary.token_cost()
        return estimate_tokens(json.dumps(self.payload, ensure_ascii=False))

    def render(self) -> dict[str, Any]:
        """Modele gidecek biçim — silindiyse yer tutucu, özetlendiyse özet, yoksa ham."""
        # B.11: yer tutucu — mesaj dizisinin yapısı bozulmaz, sadece içerik gider
        if self.cleared:
            return {"seq": self.seq, "type": self.type,
                    "cleared": True, "note": self.clear_note}
        if self.evicted and self.summary is not None:
            return {"seq": self.seq, "type": self.type, "status": self.status,
                    "compacted": self.summary.as_dict()}
        return {"seq": self.seq, "type": self.type,
                "payload": self.payload, "status": self.status}


# --- 5-alan trace-özet birimi (§4) ---------------------------------------

@dataclass
class TraceSummary:
    """Sıkıştırılmış bir tool biriminin korunan çekirdeği.

    §4: niyet / girdi / sonuç / durum / etki.
    - niyet:  ham trace'te YOKTUR, çıkarılır (en değerli)
    - sonuc:  kritik ise verbatim (parafraz edilmez)
    """
    niyet: str          # ne için
    girdi: str          # ne ile
    sonuc: str          # ne oldu (verbatim olabilir)
    durum: str          # başarı/hata
    etki: str = ""      # sonraki adıma bağ

    def as_dict(self) -> dict[str, str]:
        d = {"niyet": self.niyet, "girdi": self.girdi,
             "sonuc": self.sonuc, "durum": self.durum}
        if self.etki:
            d["etki"] = self.etki
        return d

    def token_cost(self) -> int:
        return estimate_tokens(json.dumps(self.as_dict(), ensure_ascii=False))


# --- Trace: olayların dizisi ---------------------------------------------

class Trace:
    """Bir yörünge boyunca tiplenmiş olayların zaman sıralı dizisi (§3)."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self._seq = 0

    def add_reasoning(self, text: str) -> Event:
        ev = Event(seq=self._seq, type="reasoning", payload={"text": text})
        self._seq += 1
        self.events.append(ev)
        return ev

    def add_tool(self, name: str, args: dict, output: str,
                 status: str = "ok", intent_ref: Optional[int] = None,
                 verbatim: bool = False) -> Event:
        ev = Event(seq=self._seq, type="tool",
                   payload={"name": name, "args": args, "output": output},
                   status=status, intent_ref=intent_ref, verbatim=verbatim)
        self._seq += 1
        self.events.append(ev)
        return ev

    def add_answer(self, text: str) -> Event:
        ev = Event(seq=self._seq, type="answer", payload={"text": text})
        self._seq += 1
        self.events.append(ev)
        return ev

    def tool_events(self) -> list[Event]:
        return [e for e in self.events if e.type == "tool"]

    def total_tokens(self) -> int:
        return sum(e.token_cost() for e in self.events)

    def render(self) -> list[dict]:
        return [e.render() for e in self.events]
