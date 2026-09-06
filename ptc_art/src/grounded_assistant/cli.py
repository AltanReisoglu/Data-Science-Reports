"""CLI (typer + rich) — contracts/cli_interface.md.

Faz 1: knowledge_base + live_systems. Faz 2 (T014-T017): PTC sandbox da
`agent/graph.py`'de üçüncü bir erişim yolu olarak bağlı — bu dosyanın kendisi
üç yolu da ayırt etmiyor, hepsi `Trace` üzerinden tek tip işleniyor."""

from __future__ import annotations

import asyncio

import typer
from dotenv import load_dotenv
from rich.console import Console

from grounded_assistant.agent import graph
from grounded_assistant.models import Answer
from grounded_assistant.session import oturum_kimligi
from grounded_assistant.trace import Trace

load_dotenv()

app = typer.Typer()
console = Console()


def _build_answer(record: Trace, raw_text: str) -> Answer:
    """Altan'ın kararı (2026-08-30): LLM'in kendi ürettiği `raw_text` artık HER
    ZAMAN gösteriliyor — önceden `source_refs` boşsa bu metin sessizce atılıp
    jenerik bir "veri bulunamadı" mesajıyla değiştiriliyordu. Bu, Principle I'in
    kendisinden ("uydurulmuş bir OLGUSAL İDDİA'yı zeminliymiş gibi sunma")
    daha katıydı — LLM hiç tool çağırmadan da dürüst/doğru bir şey söylemiş
    olabilir (ör. "bu sistemde keyfi bir URL çekecek bir tool yok"), ama o
    cevap hiç görülmeden atılıyordu. `grounded` rozeti (aşağıda) zemin olup
    olmadığını göstermeye devam ediyor — Principle I'in "açıkça belirtme"
    şartı bununla karşılanıyor; egress/Cilium kısıtlaması bu değişiklikten
    hiç etkilenmiyor."""
    source_refs = record.source_refs()
    return Answer(
        text=raw_text,
        grounded=bool(source_refs),
        access_paths_used=record.access_paths_used(),
        source_refs=source_refs,
        partial_failure_notes=record.partial_failure_notes(),
    )


def _print_answer(answer: Answer) -> None:
    console.print("[bold]Yanıt:[/bold]")
    console.print(answer.text)
    console.print()

    if answer.source_refs:
        line = f"[bold]Kaynaklar:[/bold] {', '.join(answer.source_refs)}"
    else:
        line = "[bold]Kaynaklar:[/bold] [dim](yok)[/dim]"
    if answer.partial_failure_notes:
        line += " — [yellow]" + "; ".join(answer.partial_failure_notes) + "[/yellow]"
    console.print(line)


@app.command()
def ask(
    question: str,
    trace: bool = typer.Option(False, "--trace", help="Ham izlenebilirlik kaydını JSON olarak ek çıktı ver."),
    session: str = typer.Option(
        None,
        "--session",
        help="Önceki bir oturumu sürdür — o oturumda saklanmış artifact'ler "
        "okunabilir olur. Verilmezse her çağrı temiz başlar.",
    ),
) -> None:
    # 2026-09-04: eskiden koşulsuz `uuid4()` idi. Bu değer hem `thread_id`
    # (konuşma hafızası) hem `workflow_id` (artifact kapsamı) olduğu için, her
    # `ask` çağrısı bir öncekinin artifact'lerini ERİŞİLEMEZ bırakıyordu —
    # depoda duruyorlardı ama anahtarları bir daha üretilmiyordu.
    #
    # Web'de kimliği `localStorage` saklıyor; CLI'nin böyle bir yeri yok, o
    # yüzden kullanıcı açıkça geçiriyor. Varsayılan yine temiz oturum:
    # kalıcılık her iki arayüzde de opt-in.
    session_id = oturum_kimligi(session)
    record = Trace()

    try:
        # invoke_and_resolve async (graph.py'de düzeltilen gerçek hata,
        # 2026-08-28) — CLI'nin kendisi sync olduğu için asyncio.run ile sarılıyor.
        #
        # Kurulum da loop'un İÇİNE alındı (2026-09-04): checkpointer async
        # olduğu için çalışan bir loop istiyor. `build_agent` ise tersine
        # loop'SUZ bir thread'de çalışmak zorunda (içinde asyncio.run var),
        # o yüzden `to_thread` ile ayrı thread'e veriliyor.
        async def _kur_ve_calistir():
            checkpointer = graph.build_checkpointer()
            try:
                agent = await asyncio.to_thread(
                    graph.build_agent, record, None, session_id, checkpointer
                )
                return await graph.invoke_and_resolve(
                    agent, question, thread_id=session_id
                )
            finally:
                # aiosqlite kendi thread'ini açıyor; kapatmazsak süreç çıkarken
                # "Event loop is closed" uyarısı üretiyor.
                kapat = getattr(checkpointer, "conn", None)
                if kapat is not None:
                    try:
                        await kapat.close()
                    except Exception:  # noqa: BLE001
                        pass

        result = asyncio.run(_kur_ve_calistir())
        raw_text = result["messages"][-1].content
    except Exception as exc:  # noqa: BLE001 - beklenmeyen hata, CLI çıkış kodu 1
        console.print(f"[bold red]Hata:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    answer = _build_answer(record, raw_text)
    _print_answer(answer)

    if trace:
        console.print_json(record.to_json())


if __name__ == "__main__":
    app()
