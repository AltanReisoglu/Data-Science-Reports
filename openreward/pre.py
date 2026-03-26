from openai import OpenAI
from openreward import OpenReward
import json
import os
from dotenv import load_dotenv

load_dotenv()

or_client = OpenReward()
oai_client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.environ.get("OPENROUTER_API_KEY")
)
MODEL_NAME = "deepseek/deepseek-v3.2"

environment = or_client.environments.get(name="GeneralReasoning/CTF")
tasks = environment.list_tasks(split="train")
tools = environment.list_tools(format="openrouter")

example_task = tasks[0]

with environment.session(task=example_task) as session:
    prompt = session.get_prompt()
    input_list = [{"role": "user", "content": prompt[0].text}]
    finished = False
    print(input_list)

    while not finished:
        response = oai_client.chat.completions.create(
            model=MODEL_NAME,
            tools=tools,
            messages=input_list,
            max_tokens=4096
        )
    
        assistant_message = response.choices[0].message
        input_list.append(assistant_message.model_dump(exclude_none=True))
        print(f"Assistant: {assistant_message.content or 'Tool calling...'}")

        tool_calls = assistant_message.tool_calls
        if not tool_calls:
            break
        
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except:
                tool_args = {}
                
            print(f"Executing Tool: {tool_name} with args {tool_args}")
            
            try:
                tool_result = session.call_tool(tool_name, tool_args)
                reward = tool_result.reward or 0.0
                finished = tool_result.finished
                output_str = "\n".join([str(b.text) for b in tool_result.blocks if b.text])
            except Exception as e:
                reward = 0.0
                finished = False
                output_str = f"Tool Error: {str(e)}"

            input_list.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output_str
            })

            print(f"Tool Result: {output_str[:200]}...")

            if finished:
                break