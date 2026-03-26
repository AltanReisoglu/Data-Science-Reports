import json
import os
from dotenv import load_dotenv

# dotenv'i yükleyerek .env dosyasındaki API key'leri okuyoruz
load_dotenv()

# Gerekli kütüphaneler. openreward ve openai yüklü olmalıdır.
# pip install openreward openai python-dotenv
from openai import OpenAI
from openreward import OpenReward
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

def main():
    console.print(Panel.fit("[bold blue]OpenReward Quickstart Tutorial[/bold blue]", border_style="blue"))
    
    # 1. API Key Kontrolü
    if not os.getenv("OPENREWARD_API_KEY"):
        console.print("[bold red]HATA:[/bold red] OPENREWARD_API_KEY bulunamadı. Lütfen .env dosyasını kontrol edin.")
        return
    if not os.getenv("OPENROUTER_API_KEY"):
        console.print("[bold red]HATA:[/bold red] OPENROUTER_API_KEY bulunamadı. Lütfen .env dosyasını kontrol edin.")
        return

    # 2. İstemcileri Başlatma
    console.print("[cyan]1. İstemciler başlatılıyor...[/cyan]")
    or_client = OpenReward()
    oai_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY")
    )
    MODEL_NAME = "deepseek/deepseek-v3.2"

    # 3. Environment Seçimi
    env_name = "kanishk/EndlessTerminals"
    console.print(f"[cyan]2. Environment bağlanıyor: [bold]{env_name}[/bold][/cyan]")
    
    try:
        environment = or_client.environments.get(name=env_name)
    except Exception as e:
        console.print(f"[bold red]Environment bağlantı hatası:[/bold red] {e}")
        return

    # 4. Task ve Tool'ları Alma
    tasks = environment.list_tasks(split="train")
    if not tasks:
        console.print("[bold red]HATA:[/bold red] Görev bulunamadı!")
        return
        
    tools = environment.list_tools(format="openrouter")
    example_task = tasks[0]
    
    console.print(f"[green]✓ {len(tasks)} görev bulundu.[/green]")
    console.print(f"[green]✓ {len(tools)} tool tanımlandı (Format: OpenRouter).[/green]")

    # 5. Session, Rollout Başlatma ve Agent Loop
    console.print("\n[cyan]3. Session ve Rollout başlatılıyor, görev çözülüyor...[/cyan]")
    
    with environment.session(task=example_task) as session:
        # OpenReward Rollout mekanizmasını başlatıp kaydetmeyi etkinleştiriyoruz
        rollout = or_client.rollout.create(
            run_name="EndlessTerminals-train-quickstart",
            rollout_name="example_task",
            environment=env_name,
            split="train",
            print_messages=True
        )

        prompt_blocks = session.get_prompt()
        initial_instruction = prompt_blocks[0].text
        
        console.print(Panel(initial_instruction, title="[bold yellow]Task Prompt[/bold yellow]", border_style="yellow"))
        
        input_list = [{"role": "user", "content": initial_instruction}]
        
        # İlk user promptunu rollout'a logluyoruz
        rollout.log_openai_completions(input_list[0])
        
        finished = False
        step = 1
        total_reward = 0.0
        
        while not finished:
            console.print(f"\n[bold magenta]--- Adım {step} ---[/bold magenta]")
            
            with console.status("[dim]LLM (DeepSeek) düşünüyor...[/dim]"):
                response = oai_client.chat.completions.create(
                    model=MODEL_NAME,
                    tools=tools,
                    messages=input_list,
                    max_tokens=4096
                )
            
            # Asistan mesajını oluşturup kaydet
            assistant_message = {
                "role": "assistant",
                "content": response.choices[0].message.content,
                "tool_calls": response.choices[0].message.tool_calls
            }
            input_list.append(assistant_message)
            rollout.log_openai_completions(assistant_message)
            
            if assistant_message["content"]:
                console.print(f"[bold blue]Asistan: [/bold blue] {assistant_message['content']}")
                
            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                break
                
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}
                    
                console.print(f"[yellow]⚡ Tool Çağrısı:[/yellow] [bold]{tool_name}[/bold] {tool_args}")
                
                with console.status("[dim]Environment'da tool çalıştırılıyor...[/dim]"):
                    try:
                        tool_result = session.call_tool(tool_name, tool_args)
                        reward = tool_result.reward or 0.0
                        total_reward += reward
                        finished = tool_result.finished
                        str_output = "\n".join([str(b.text) for b in tool_result.blocks if b.text])
                    except Exception as e:
                        reward = 0.0
                        finished = False
                        str_output = f"Tool Execution Error: {str(e)}"
                
                console.print(Panel(
                    Syntax(str_output, "bash", theme="monokai", word_wrap=True),
                    title="[dim]Tool Sonucu[/dim]",
                    border_style="dim"
                ))
                
                if reward > 0:
                    console.print(f"[bold green]⭐ Ödül Alındı: {reward}[/bold green]")
                
                # Tool yanıtını oluşturup modele veriyoruz
                tool_result_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str_output
                }
                input_list.append(tool_result_message)
                
                # Rollout'a logluyoruz
                rollout.log_openai_completions(tool_result_message, reward=reward, is_finished=finished)
                
                if finished:
                    break
                    
            if step >= 15:
                console.print("[bold red]Maksimum adım sayısına ulaşıldı![/bold red]")
                break
                
            step += 1

        # Session biterken rollout'u kapatiyoruz
        or_client.rollout.close()

    console.print("\n[bold green]=========================================[/bold green]")
    console.print(f"[bold green]✓ Görev Tamamlandı! Toplam Adım: {step} | Toplam Ödül: {total_reward}[/bold green]")
    console.print("[bold green]=========================================[/bold green]")


if __name__ == "__main__":
    main()
