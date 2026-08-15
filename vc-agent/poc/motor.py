"""Ortak motor katmanı: model istemcisi + ölçüm.

İki mod var:

* **gerçek**  — `OPENAI_API_KEY` ya da `OPENROUTER_API_KEY` ortamda varsa gerçek LLM.
* **replay**  — hiçbiri yoksa `ReplayChatCompletionClient`. Senaryo önceden yazılıdır,
  ağ gerekmez, sonuç deterministiktir. POC'un tamamı anahtarsız çalışır.

Replay modu bir hile değil: desenlerin **kontrol akışını** (kim ne zaman konuşur,
handoff nereye gider, termination nerede tetiklenir) LLM'in keyfinden bağımsız
olarak izlenebilir kılıyor. Karşılaştırma tabloları da böylece tekrarlanabilir oluyor.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Sequence

from autogen_core import FunctionCall
from autogen_core.models import ChatCompletionClient, CreateResult, ModelInfo, RequestUsage

# Replay istemcisinin tool çağırabilmesi için function_calling=True şart.
REPLAY_MODEL_INFO = ModelInfo(
    vision=False,
    function_calling=True,
    json_output=False,
    family="unknown",
    structured_output=False,
)


def gercek_mod() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"))


def istemci(senaryo: Sequence[str | CreateResult]) -> ChatCompletionClient:
    """Ortama göre gerçek ya da replay istemcisi döndürür.

    `senaryo` yalnızca replay modunda kullanılır; gerçek modda yok sayılır.
    """
    if os.getenv("OPENROUTER_API_KEY"):
        # Kurs 5.2'deki OpenRouter yolu: OpenAI uyumlu endpoint.
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        return OpenAIChatCompletionClient(
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    if os.getenv("OPENAI_API_KEY"):
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        return OpenAIChatCompletionClient(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.environ["OPENAI_API_KEY"],
        )

    from autogen_ext.models.replay import ReplayChatCompletionClient

    return ReplayChatCompletionClient(list(senaryo), model_info=REPLAY_MODEL_INFO)


def arac_cagrisi(ad: str, argumanlar: str, *, cid: str = "c1", pt: int = 60, ct: int = 12) -> CreateResult:
    """Replay senaryosuna tool çağrısı koymak için yardımcı."""
    return CreateResult(
        finish_reason="function_calls",
        content=[FunctionCall(id=cid, name=ad, arguments=argumanlar)],
        usage=RequestUsage(prompt_tokens=pt, completion_tokens=ct),
        cached=False,
    )


# --------------------------------------------------------------------------- ölçüm


@dataclass
class Olcum:
    """Bir desen koşusunun ölçülen davranışı."""

    desen: str
    sure_ms: int = 0
    mesaj_sayisi: int = 0
    llm_cagrisi: int = 0
    arac_cagrisi: int = 0
    prompt_token: int = 0
    completion_token: int = 0
    devir_sirasi: list[str] = field(default_factory=list)
    durma_nedeni: str = ""
    mod: str = "replay"

    @property
    def toplam_token(self) -> int:
        return self.prompt_token + self.completion_token

    def sozluk(self) -> dict[str, Any]:
        d = asdict(self)
        d["toplam_token"] = self.toplam_token
        return d

    def yazdir(self) -> None:
        print(f"\n{'─' * 62}")
        print(f"  ÖLÇÜM · {self.desen}   [{self.mod} mod]")
        print(f"{'─' * 62}")
        print(f"  süre              : {self.sure_ms} ms")
        print(f"  mesaj sayısı      : {self.mesaj_sayisi}")
        print(f"  LLM çağrısı       : {self.llm_cagrisi}")
        print(f"  tool çağrısı      : {self.arac_cagrisi}")
        print(f"  token (p/c/top)   : {self.prompt_token} / {self.completion_token} / {self.toplam_token}")
        print(f"  devir sırası      : {' → '.join(self.devir_sirasi) or '—'}")
        print(f"  durma nedeni      : {self.durma_nedeni}")


async def olc(desen: str, takim, gorev: str, model_istemcisi) -> Olcum:
    """Bir takımı çalıştırır, mesajları basar ve ölçümü döndürür."""
    from autogen_agentchat.messages import (
        ToolCallExecutionEvent,
        ToolCallRequestEvent,
    )

    print(f"\n╔{'═' * 60}╗")
    print(f"║ {desen:<59}║")
    print(f"╚{'═' * 60}╝")

    t0 = time.perf_counter()
    sonuc = await takim.run(task=gorev)
    sure = int((time.perf_counter() - t0) * 1000)

    olcum = Olcum(desen=desen, sure_ms=sure, mod="gerçek" if gercek_mod() else "replay")
    olcum.mesaj_sayisi = len(sonuc.messages)
    olcum.durma_nedeni = sonuc.stop_reason or ""

    for m in sonuc.messages:
        kaynak = getattr(m, "source", "user")
        if isinstance(m, ToolCallRequestEvent):
            adlar = ", ".join(fc.name for fc in m.content)
            olcum.arac_cagrisi += len(m.content)
            print(f"  ⚙  {kaynak:<14} tool çağrısı → {adlar}")
        elif isinstance(m, ToolCallExecutionEvent):
            for r in m.content:
                print(f"  ↩  {kaynak:<14} tool sonucu  ← {str(r.content)[:70]}")
        else:
            icerik = str(getattr(m, "content", ""))
            if kaynak != "user" and (not olcum.devir_sirasi or olcum.devir_sirasi[-1] != kaynak):
                olcum.devir_sirasi.append(kaynak)
            print(f"  ▸  {kaynak:<14} {icerik[:88]}")

    # Tek istemci de olabilir, ajan başına ayrı istemci listesi de (bkz. Desen 4).
    istemciler = model_istemcisi if isinstance(model_istemcisi, (list, tuple)) else [model_istemcisi]
    for c in istemciler:
        kullanim: RequestUsage = c.total_usage()
        olcum.prompt_token += kullanim.prompt_tokens
        olcum.completion_token += kullanim.completion_tokens
        # `create_calls` yalnızca replay istemcisinde var. Gerçek modda mesajlardan sayıyoruz:
        # her ajan mesajı ya da tool çağrısı bir LLM turuna karşılık gelir.
        olcum.llm_cagrisi += len(getattr(c, "create_calls", []) or [])

    if not olcum.llm_cagrisi:  # gerçek mod
        olcum.llm_cagrisi = sum(
            1
            for m in sonuc.messages
            if getattr(m, "source", "user") != "user"
            and not isinstance(m, ToolCallExecutionEvent)
        )

    olcum.yazdir()
    return olcum
