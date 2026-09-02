"""CLI (typer + rich) — contracts/cli_interface.md.

Faz 1: knowledge_base + live_systems. Faz 2 (T014-T017): PTC sandbox da
`agent/graph.py`'de üçüncü bir erişim yolu olarak bağlı — bu dosyanın kendisi
üç yolu da ayırt etmiyor, hepsi `Trace` üzerinden tek tip işleniyor."""

from __future__ import annotations

import asyncio
import uuid

import typer
from dotenv import load_dotenv
from rich.console import Console

from grounded_assistant.agent import graph
from grounded_assistant.models import Answer
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
) -> None:
    session_id = str(uuid.uuid4())
    record = Trace()

    try:
        agent = graph.build_agent(record)
        # invoke_and_resolve artık async (graph.py'de düzeltilen gerçek hata,
        # 2026-08-28) — CLI'nin kendisi sync olduğu için asyncio.run ile sarılıyor.
        result = asyncio.run(graph.invoke_and_resolve(agent, question, thread_id=session_id))
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
