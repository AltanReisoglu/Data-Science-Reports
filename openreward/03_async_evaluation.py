import asyncio
import json
import os
from dotenv import load_dotenv

load_dotenv()

from openreward import AsyncOpenReward
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import tabulate

console = Console()

async def evaluate_single_task(environment, task_idx, task_spec, tools, progress_bar, task_id):
    """Tek bir taskı OpenRouter üzerinden asenkron olarak değerlendirir."""
    from openai import AsyncOpenAI
    
    # Asenkron OpenAI client kullanarak OpenRouter'a bağlanıyoruz
    oai_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY")
    )
    MODEL_NAME = "deepseek/deepseek-v3.2" # Veya openai/gpt-4o-mini
    
    try:
        # Async session başlatan context_manager
        async with environment.session(task=task_spec) as session:
            progress_bar.update(task_id, description=f"Task {task_idx+1}: Prompt Alınıyor...")
            prompt = session.get_prompt()
            messages = [{"role": "user", "content": prompt[0].text}]
            
            step = 0
            total_reward = 0.0
            finished = False
            
            while not finished and step < 10:
                step += 1
                progress_bar.update(task_id, description=f"Task {task_idx+1}: Adım {step}")
                
                # Asenkron tahmin
                response = await oai_client.chat.completions.create(
                    model=MODEL_NAME,
                    tools=tools,
                    messages=messages,
                    max_tokens=4096
                )
                
                ai_msg = response.choices[0].message
                messages.append(ai_msg.model_dump(exclude_none=True))
                
                if not ai_msg.tool_calls:
                    break
                    
                for tool_call in ai_msg.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except:
                        tool_args = {}
                        
                    try:
                        tool_result = await asyncio.to_thread(session.call_tool, tool_name, tool_args)
                        reward = tool_result.reward or 0.0
                        total_reward += reward
                        finished = tool_result.finished
                        output_str = "\n".join([str(b.text) for b in tool_result.blocks if b.text])
                    except Exception as e:
                        reward = 0.0
                        finished = False
                        output_str = f"Tool Execution Error: {str(e)}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": output_str
                    })
                    
                    if finished:
                        break
                        
            progress_bar.update(task_id, description=f"Task {task_idx+1}: Bitti", completed=100)
            return {
                "task_idx": task_idx,
                "reward": total_reward,
                "steps": step,
                "finished": finished,
                "status": "Success"
            }
            
    except Exception as e:
        progress_bar.update(task_id, description=f"Task {task_idx+1}: Hata: {str(e)[:20]}", completed=100)
        return {
            "task_idx": task_idx,
            "reward": 0.0,
            "steps": 0,
            "finished": False,
            "status": f"Error: {e}"
        }


async def main():
    console.print(Panel.fit("[bold green]Async OpenReward Batch Evaluation (OpenRouter)[/bold green]", border_style="green"))
    
    if not os.getenv("OPENREWARD_API_KEY") or not os.getenv("OPENROUTER_API_KEY"):
        console.print("[bold red]Lütfen OPENREWARD_API_KEY ve OPENROUTER_API_KEY ortam değişkenlerini ayarlayın.[/bold red]")
        return
        
    # Asenkron OpenReward istemcisi oluştur
    or_client = AsyncOpenReward()
    env_name = "kanishk/EndlessTerminals"
    
    console.print(f"[cyan]Environment ({env_name}) bağlanılıyor...[/cyan]")
    environment = await or_client.environments.get(name=env_name)
    
    # Task'ları al (Senkron çağrıdır)
    console.print("[cyan]Task listesi çekiliyor...[/cyan]")
    tasks = environment.list_tasks(split="train")
    tools = environment.list_tools(format="openrouter")
    
    # Sadece ilk 5 task'ı değerlendirelim
    num_evaluations = min(5, len(tasks))
    tasks_to_eval = tasks[:num_evaluations]
    
    console.print(f"\n[yellow]Toplam {num_evaluations} task paralel olarak değerlendirelecek.[/yellow]\n")
    
    results = []
    
    # Progress bar ile asenkron görevleri çalıştırıyoruz
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        
        # Görev idleri ve progress barlarını oluştur
        eval_tasks = []
        for i, task_spec in enumerate(tasks_to_eval):
            task_id = progress.add_task(f"Task {i+1}: Bekliyor...", total=100, completed=0)
            
            # Asenkron görevi listeye ekle
            eval_task = asyncio.create_task(
                evaluate_single_task(environment, i, task_spec, tools, progress, task_id)
            )
            eval_tasks.append(eval_task)
            
        # Tüm görevleri paralel çalıştır ve bitmesini bekle
        results = await asyncio.gather(*eval_tasks)
        
    # Sonuç Raporu Hazırlama
    console.print("\n[bold magenta]Değerlendirme Sonuçları[/bold magenta]")
    
    table_data = []
    total_reward = 0
    success_count = 0
    
    for res in sorted(results, key=lambda x: x["task_idx"]):
        reward = res["reward"]
        total_reward += reward
        is_success = reward > 0 or res["finished"]
        
        if is_success:
            success_count += 1
            
        status_text = "[green]Tamamlandı[/green]" if is_success else "[yellow]Tamamlanamadı[/yellow]"
        if "Error" in res["status"]:
            status_text = f"[red]{res['status']}[/red]"
            
        table_data.append([
            f"Task {res['task_idx']+1}",
            f"{reward:.2f}",
            res["steps"],
            status_text
        ])
        
    avg_reward = total_reward / num_evaluations if num_evaluations > 0 else 0
    success_rate = (success_count / num_evaluations) * 100 if num_evaluations > 0 else 0
    
    table_str = tabulate.tabulate(
        table_data, 
        headers=["Task", "Ödül", "Adım", "Durum"],
        tablefmt="pretty"
    )
    
    console.print(table_str)
    
    summary = (
        f"Toplam Görev: {num_evaluations}\n"
        f"Ortalama Ödül: {avg_reward:.2f}\n"
        f"Başarı Oranı: %{success_rate:.1f}"
    )
    console.print(Panel(summary, title="Özet", border_style="blue"))


if __name__ == "__main__":
    asyncio.run(main())
