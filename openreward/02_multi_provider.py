import json
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

from openreward import OpenReward
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

console = Console()

def run_openrouter_model(environment, task, tools, model_name):
    """OpenRouter üzerinden model çalıştıran agent döngüsü."""
    from openai import OpenAI
    
    if not os.getenv("OPENROUTER_API_KEY"):
        return {"error": "OPENROUTER_API_KEY eksik"}
        
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY")
    )
    
    with environment.session(task=task) as session:
        prompt = session.get_prompt()
        messages = [{"role": "user", "content": prompt[0].text}]
        
        step = 0
        total_reward = 0.0
        finished = False
        
        while not finished and step < 15:
            step += 1
            response = client.chat.completions.create(
                model=model_name,
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
                    tool_result = session.call_tool(tool_name, tool_args)
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
                    
        return {"steps": step, "reward": total_reward, "finished": finished, "model": model_name}


def main():
    parser = argparse.ArgumentParser(description="OpenRouter Multi-Model Karşılaştırması")
    args = parser.parse_args()
    
    console.print(Panel.fit("[bold violet]OpenReward OpenRouter Model Karşılaştırması[/bold violet]", border_style="violet"))
    
    or_client = OpenReward()
    env_name = "kanishk/EndlessTerminals"
    
    with console.status(f"[dim]Environment ({env_name}) yükleniyor...[/dim]"):
        try:
            environment = or_client.environments.get(name=env_name)
        except Exception as e:
            console.print(f"[bold red]Environment hatası:[/bold red] {e}")
            return
            
        tasks = environment.list_tasks(split="train")
        if not tasks:
            console.print("[bold red]Görev bulunamadı![/bold red]")
            return
        example_task = tasks[0]
        
        # Tools formatı openrouter olacak
        tools = environment.list_tools(format="openrouter")
    
    results = {}
    
    models_to_test = {
        "OpenAI": "openai/gpt-4o-mini",
        "Anthropic": "anthropic/claude-3-5-sonnet",
        "Google": "google/gemini-2.5-flash",
        "DeepSeek": "deepseek/deepseek-v3.2"
    }
    
    for provider_name, model_api_name in models_to_test.items():
        with console.status(f"[bold cyan]{provider_name} ({model_api_name}) test ediliyor...[/bold cyan]"):
            res = run_openrouter_model(environment, example_task, tools, model_api_name)
            results[provider_name] = res
            
    # Sonuçları Tablolama
    table = Table(title="Karşılaştırma Sonuçları", show_header=True, header_style="bold magenta")
    table.add_column("Provider / Model Ailesi", style="cyan")
    table.add_column("OpenRouter Model ID", style="green")
    table.add_column("Ödül (Reward)", justify="right")
    table.add_column("Adım (Steps)", justify="right")
    table.add_column("Durum", justify="center")
    
    for provider, res in results.items():
        if "error" in res:
            table.add_row(provider, "N/A", "N/A", "N/A", f"[red]{res['error']}[/red]")
        else:
            status = "[green]Tamamlandı[/green]" if res['finished'] else "[yellow]Yarım Kaldı[/yellow]"
            table.add_row(
                provider, 
                res['model'], 
                f"{res['reward']:.2f}", 
                str(res['steps']), 
                status
            )
            
    console.print("\n")
    console.print(table)


if __name__ == "__main__":
    main()
