"""MAF tarafı — **ayrı bir sanal ortamda** koşan alt süreç.

Bu dosya `.venv-maf` içinde çalışıyor ve bu depodaki hiçbir modülü içe
aktarmıyor. Sebebi mimari değil, ölçüm: `agent-framework` ile `autogen-*` aynı
bağımlılık ağacını paylaşamıyor — pip'in çözücüsü on dakikada bir karar
veremedi, ki bu zaten çakışma baskısının işareti. İkisini ayrı ortamlara koymak
hem riski sıfırlıyor hem de dürüst: bunlar iki ayrı çerçeve.

İletişim, taramanın kullandığı protokolün aynısı: stdout'a tek satırlık
`##STAGE {json}` ve `##OUT {json}`. Ana süreç onları kendi aşama katalogundan
geçiriyor, yani ekran MAF turunu AutoGen turuyla aynı şekilde çiziyor.

### Burada ölçülen şey

MAF'ın AutoGen'den ayrıldığı yerler, iddia olarak değil **kurulum olarak**:

* `FunctionTool(approval_mode=...)` — onay **tool başına** ve hazır. AutoGen'de
  böyle bir alan yok; kapıyı biz `GatedWorkbench` olarak elle yazdık.
* `FunctionTool(max_invocations=...)` — çağrı tavanı yine tool başına. AutoGen'in
  `max_tool_iterations` ayarı ajanın tamamına ait.
* `ToolApprovalMiddleware(auto_approval_rules=[...])` — "duran onay" burada bir
  kurulum parametresi.
* `Agent.run_stream` diye ayrı bir metot **yok**: akış `run(stream=True)`
  parametresi. AutoGen'de `run()` ve `run_stream()` iki ayrı yüzey.
* `ToolApprovalMiddleware` bir `AgentSession` istiyor — ölçüldü:
  `RuntimeError: ToolApprovalMiddleware requires an AgentSession`. Yani onay,
  oturumu olan bir koşuya bağlı; tek atışlık bir çağrıya takılamıyor.
* **Onay birinci sınıf bir cevap alanı.** `approval_mode="always_require"` ile
  koşulduğunda `AgentResponse.user_input_requests` bir
  `function_approval_request` taşıyor ve `finish_reason="tool_calls"` oluyor
  (ölçüldü). AutoGen'de bunun karşılığı yok; biz reddi hata işaretli bir
  `ToolResult` olarak elle kuruyoruz.
* **Tool çağrıldığında `response.text` BOŞ kalıyor** ve cevap `messages`
  içinde. AutoGen'in `reflect_on_tool_use=False` varsayılanının tıpatıp aynı
  sonucu: iki çerçeve de tool sonrası nihai cevabı varsayılan olarak
  yazdırmıyor. Ölçüldü: tool'lu koşuda `text=0 karakter`, `messages=3`.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

TAG_STAGE = "##STAGE "
TAG_OUT = "##OUT "
_T0 = time.time()


def stage(stage_id: str, **meta) -> None:
    """Bir aşamayı bildir. Asla fırlatmaz: koşu, kendini anlatırken düşemez."""
    try:
        print(TAG_STAGE + json.dumps(
            {"id": stage_id, "t": time.time(), "meta": meta}, ensure_ascii=False),
            flush=True)
    except Exception:  # noqa: BLE001
        pass


def out(kind: str, **payload) -> None:
    try:
        print(TAG_OUT + json.dumps({"type": kind, **payload}, ensure_ascii=False),
              flush=True)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------- tool
def sirket_sayisi(soru: str) -> str:
    """Son taramanın kaç şirket bulduğunu söyler.

    MAF tarafında tool'un tarifi yine docstring'den üretiliyor — AutoGen ile
    aynı disiplin, aynı tuzak: bu metin dokümantasyon değil arayüz.
    """
    return ("Son tarama 27 şirket buldu; üçü triyajı geçti. "
            "(MAF modunda örnek tool — gerçek veri için AutoGen hattı kullanılıyor.)")


async def main() -> int:
    question = sys.argv[1] if len(sys.argv) > 1 else "Merhaba"

    import agent_framework as af
    from agent_framework.openai import OpenAIChatClient

    stage("maf_build", version=getattr(af, "__version__", "?"),
          client="OpenAIChatClient", model=os.getenv("VC_MAF_MODEL", ""))

    client = OpenAIChatClient(
        model=os.getenv("VC_MAF_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("VC_LLM_API_KEY", "") or "sk-none",
        base_url=os.getenv("VC_LLM_BASE_URL", "") or None,
    )

    # Onay tool'un KENDİ alanı. AutoGen'de karşılığı yok; orada kapıyı
    # workbench'i sarmalayarak biz kuruyoruz.
    # `always_require` seçilirse tur onay isteğiyle DURUYOR ve cevap yerine
    # `user_input_requests` dönüyor — kapının çerçeveye gömülü hâli.
    mode = os.getenv("VC_MAF_APPROVAL", "never_require")
    tool = af.FunctionTool(
        func=sirket_sayisi,
        name="sirket_sayisi",
        description="Son taramanın kaç şirket bulduğunu söyler.",
        approval_mode=mode,
        max_invocations=2,
    )
    stage("maf_tool", name="sirket_sayisi", approval_mode=mode,
          max_invocations=2,
          note="approval_mode ve max_invocations tool'un KENDİ alanları")

    middleware = af.ToolApprovalMiddleware(source_id=af.DEFAULT_TOOL_APPROVAL_SOURCE_ID)
    stage("maf_gate", middleware="ToolApprovalMiddleware",
          source=af.DEFAULT_TOOL_APPROVAL_SOURCE_ID,
          note="Kapı çerçevenin kendisinde; sarmalayıcı yazmak gerekmiyor.")

    agent = af.Agent(
        client=client,
        instructions="Kısa ve Türkçe cevap ver. Tool varsa kullan.",
        name="MafAnalyst",
        description="MAF modunda koşan analist.",
        tools=[tool],
        middleware=[middleware],
    )
    stage("maf_agent", cls="agent_framework.Agent",
          note="Ayrı bir run_stream() yok: akış run(stream=True) parametresi.")

    # Onay ara katmanı oturum istiyor. Ölçüldü: oturumsuz koşuda
    # `RuntimeError: ToolApprovalMiddleware requires an AgentSession`.
    session = af.AgentSession(session_id=os.getenv("VC_MAF_SESSION", "vc-maf"))
    stage("maf_session", cls="agent_framework.AgentSession",
          note="Onay ara katmanı oturumsuz koşuya takılamıyor.")

    stage("maf_run", question=question[:120])
    started = time.time()
    try:
        result = await agent.run(question, session=session)
    except Exception as e:  # noqa: BLE001
        stage("maf_done", error=f"{type(e).__name__}: {e}",
              seconds=round(time.time() - started, 2))
        out("error", message=f"{type(e).__name__}: {e}")
        return 1

    # Onay istekleri birinci sınıf bir alan. Varsa turun bittiği yer burası.
    requests = list(getattr(result, "user_input_requests", None) or [])
    if requests:
        stage("maf_approval", count=len(requests),
              kind=getattr(requests[0], "type", "") or type(requests[0]).__name__,
              finish=str(getattr(result, "finish_reason", "")),
              note="AgentResponse.user_input_requests — onay çerçevenin cevabında.")

    # `text` tool çağrıldığında boş kalıyor; cevap mesajların içinde. Yalnız
    # `text`'e bakan bir arayüz, tool kullanan her turda boş ekran gösterir.
    text = (getattr(result, "text", None) or "").strip()
    if not text:
        for message in reversed(list(getattr(result, "messages", None) or [])):
            candidate = (getattr(message, "text", None) or "").strip()
            if candidate:
                text = candidate
                break
    stage("maf_done", seconds=round(time.time() - started, 2),
          chars=len(text), messages=len(getattr(result, "messages", None) or []),
          from_messages=not (getattr(result, "text", None) or "").strip(),
          finish=str(getattr(result, "finish_reason", "")))
    out("done", text=text[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
