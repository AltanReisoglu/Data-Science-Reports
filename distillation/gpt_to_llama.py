# Distillation Script: GPT to Llama-3.1-8B-Instruct
# Designed for Google Colab with GPU (T4/L4/A100)

import os

# --- 1. Environment Setup (Run in a Colab Cell) ---
# !pip install unsloth
# !pip install --no-deps xformers trl peft accelerate bitsandbytes
# !pip install datasets

import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import json
import requests
from pathlib import Path

# --- 2. Data Loading ---
def load_distill_data(partition="deepseek-r1-math-train", local_path=None):
    if local_path is None:
        local_path = f"{partition}.json"
    local_path = Path(local_path)

    url = f"https://huggingface.co/datasets/rasbt/math_distill/resolve/main/data/{partition}.json"
    
    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    print(f"Downloading {partition}...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    with local_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data

def format_distilled_answer(entry):
    content = str(entry["message_content"]).strip()
    thinking = str(entry["message_thinking"]).strip()
    return f"<think>{thinking}</think>\n\n{content}"

def prepare_dataset(data):
    formatted_data = []
    for entry in data:
        # Standard Alpaca-style/Instruction format but with custom thinking tags
        prompt = (
            "You are a helpful math assistant. Answer the question and write the final result on a new line as: \\boxed{ANSWER}\n\n"
            f"Question:\n{entry['problem']}\n\n"
            "Answer:"
        )
        response = format_distilled_answer(entry)
        formatted_data.append({
            "instruction": prompt,
            "output": response,
        })
    return Dataset.from_list(formatted_data)

# --- 3. Model Initialization ---
max_seq_length = 1024 # Standard context for reasoning tasks; fits in 15GB T4 GPU
dtype = None 
load_in_4bit = True 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Ensure tokenizer model_max_length is synced to prevent dimension mismatch
tokenizer.model_max_length = max_seq_length

# --- 4. LoRA Adapters ---
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# --- 5. Data Preparation ---
data = load_distill_data()
dataset = prepare_dataset(data)

# Apply chat template and ensure truncation
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs      = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
        texts.append(text)
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True,)

# --- 6. Training ---
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Set to False to fix dimension mismatch issues
    args = TrainingArguments(
        per_device_train_batch_size = 1, 
        gradient_accumulation_steps = 8, 
        warmup_steps = 5,
        max_steps = 60, 
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# --- 7. Execution ---
# trainer.train()

# --- 8. Inference Test ---
def test_inference(question):
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    inputs = tokenizer(
    [
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True)
    print(tokenizer.batch_decode(outputs))

# test_inference("If Sam works 14 days and doesn't work 6 days, earning $60 per work day and losing $30 per off day, what is his total?")
