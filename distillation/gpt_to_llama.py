# ============================================================================
# Distillation Script: GPT (Teacher) → Llama-3.1-8B-Instruct (Student)
# Based on: Sebastian Raschka - "Reasoning from Scratch" Chapter 8
# Designed for Google Colab with GPU (T4/L4/A100)
# ============================================================================
#
# This script implements HARD DISTILLATION:
#   - Teacher (DeepSeek-R1 671B) generates reasoning traces offline
#   - Student (Llama-3.1-8B) learns to reproduce those traces via SFT
#   - Loss is computed only on ANSWER tokens (prompt is masked)
#
# Run these in a Colab cell FIRST:
#   !pip install unsloth
#   !pip install --no-deps xformers trl peft accelerate bitsandbytes
# ============================================================================

import os
import json
import time
import csv
import random
import requests
from pathlib import Path

import torch
from unsloth import FastLanguageModel


# ============================================================================
# 1. DATA LOADING & FORMATTING
# ============================================================================

def load_distill_data(partition="deepseek-r1-math-train", local_path=None):
    """Download and cache the teacher-generated distillation dataset."""
    if local_path is None:
        local_path = f"{partition}.json"
    local_path = Path(local_path)

    url = (
        "https://huggingface.co/datasets/rasbt/math_distill"
        "/resolve/main/data/"
        f"{partition}.json"
    )
    backup_url = (
        "https://f001.backblazeb2.com/file/reasoning-from-scratch/"
        f"MATH/{partition}.json"
    )

    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        size_kb = local_path.stat().st_size / 1e3
        print(f"{local_path}: {size_kb:.1f} KB (cached)")
        return data

    print(f"Downloading {partition}...")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except requests.RequestException:
        print("Primary URL failed, using backup...")
        r = requests.get(backup_url, timeout=60)
        r.raise_for_status()

    data = r.json()
    with local_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    size_kb = local_path.stat().st_size / 1e3
    print(f"{local_path}: {size_kb:.1f} KB")
    return data


def format_distilled_answer(entry):
    """Format teacher output with <think>...</think> reasoning tags."""
    content = str(entry["message_content"]).strip()
    if not content:
        raise ValueError("Missing non-empty 'message_content' field.")
    thinking = str(entry["message_thinking"]).strip()
    return f"<think>{thinking}</think>\n\n{content}"


PROMPT_TEMPLATE = (
    "You are a helpful math assistant.\n"
    "Answer the question and write the final result on a new line as:\n"
    "\\boxed{{ANSWER}}\n\n"
    "Question:\n{problem}\n\n"
    "Answer:"
)


def render_prompt(problem):
    """Render a math problem into a structured prompt."""
    return PROMPT_TEMPLATE.format(problem=problem)


# ============================================================================
# 2. TOKENIZATION & EXAMPLE BUILDING  (mirrors ch08 Section 8.4)
# ============================================================================

def build_examples(data, tokenizer):
    """
    Tokenize all examples into (prompt_ids + answer_ids + eos).
    Returns list of dicts with 'token_ids' and 'prompt_len'.
    """
    examples = []
    skipped = 0

    for entry in data:
        try:
            # Step 1: encode prompt with Llama chat template
            prompt = render_prompt(entry["problem"])
            prompt_text = (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

            # Step 2: encode teacher answer (no chat wrapping)
            target_answer = format_distilled_answer(entry)
            answer_ids = tokenizer.encode(target_answer, add_special_tokens=False)

            # Step 3: combine + EOS
            eos_id = tokenizer.eos_token_id
            token_ids = prompt_ids + answer_ids + [eos_id]

            if len(token_ids) < 2:
                skipped += 1
                continue

            examples.append({
                "token_ids": token_ids,
                "prompt_len": len(prompt_ids),
            })
        except (KeyError, TypeError, ValueError):
            skipped += 1

    return examples, skipped


# ============================================================================
# 3. FILTERING & SPLITTING  (mirrors ch08 Section 8.4.3)
# ============================================================================

def compute_length_stats(examples, answer_only=False):
    """Print token length statistics for the dataset."""
    lengths = []
    for ex in examples:
        total = len(ex["token_ids"])
        length = total - ex["prompt_len"] if answer_only else total
        lengths.append(length)

    avg_len = round(sum(lengths) / len(lengths))
    shortest_len = min(lengths)
    longest_len = max(lengths)

    print(f"  Average: {avg_len} tokens")
    print(f"  Shortest: {shortest_len} tokens")
    print(f"  Longest: {longest_len} tokens")
    return lengths


def filter_examples_by_max_len(examples, max_len=1024):
    """Keep only examples that fit within max_len tokens."""
    filtered = [ex for ex in examples if len(ex["token_ids"]) <= max_len]
    print(f"  Original: {len(examples)}")
    print(f"  Filtered: {len(filtered)}")
    print(f"  Removed:  {len(examples) - len(filtered)}")
    return filtered


# ============================================================================
# 4. LOSS COMPUTATION  (mirrors ch08 Section 8.6)
#    Key: loss is computed ONLY on answer tokens (prompt is masked)
# ============================================================================

def compute_example_loss(model, example, device):
    """
    Compute cross-entropy loss for a single example.
    Loss is masked: computed only on answer tokens, not prompt tokens.
    """
    token_ids = example["token_ids"]
    prompt_len = example["prompt_len"]

    input_ids = torch.tensor(
        token_ids[:-1], dtype=torch.long, device=device
    ).unsqueeze(0)
    target_ids = torch.tensor(
        token_ids[1:], dtype=torch.long, device=device
    )

    logits = model(input_ids).logits.squeeze(0)

    # Mask: only compute loss on answer tokens
    answer_start = max(prompt_len - 1, 0)
    answer_logits = logits[answer_start:]
    answer_targets = target_ids[answer_start:]

    loss = torch.nn.functional.cross_entropy(
        answer_logits, answer_targets
    )
    return loss


@torch.no_grad()
def evaluate_examples(model, examples, device):
    """Compute average loss over a set of examples (no gradient)."""
    was_training = model.training
    model.eval()
    total_loss = 0.0

    for example in examples:
        loss = compute_example_loss(model, example, device)
        total_loss += loss.item()

    if was_training:
        model.train()
    return total_loss / len(examples)


# ============================================================================
# 5. TRAINING LOOP  (mirrors ch08 Section 8.7)
# ============================================================================

def save_checkpoint(model, checkpoint_dir, step, suffix=""):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tag = f"-{suffix}" if suffix else ""
    path = checkpoint_dir / f"llama-8b-distill-step{step:05d}{tag}.pth"
    model.save_pretrained(str(path))
    print(f"  ✓ Saved checkpoint: {path}")
    return path


def append_csv_metrics(csv_path, epoch, step, train_loss, val_loss):
    """Append metrics to CSV log file."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        csv_path.write_text(
            "epoch,total_steps,train_loss,val_loss\n", encoding="utf-8"
        )
    with csv_path.open("a", encoding="utf-8") as f:
        f.write(f"{epoch},{step},{train_loss:.6f},{val_loss:.6f}\n")


def train_distillation(
    model,
    train_examples,
    val_examples,
    device,
    epochs=2,
    lr=5e-6,
    grad_clip_norm=1.0,
    seed=123,
    log_every=50,
    num_examples=200,       # Use only first N examples for quick runs
    checkpoint_dir="checkpoints",
    csv_log_path=None,
):
    """
    Manual training loop for hard distillation.
    Processes one example at a time (batch_size=1) to fit in GPU memory.
    Set num_examples=None to use the full training set.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    # Use subset for quick training
    active_examples = list(train_examples)[:num_examples] if num_examples else list(train_examples)
    total_steps = epochs * len(active_examples)
    global_step = 0
    rng = random.Random(seed)

    if csv_log_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_log_path = f"distill_metrics_{timestamp}.csv"

    print(f"\n{'='*60}")
    print(f"  Training Configuration")
    print(f"{'='*60}")
    print(f"  Epochs:       {epochs}")
    print(f"  Train size:   {len(active_examples)} (of {len(train_examples)} total)")
    print(f"  Val size:     {len(val_examples)}")
    print(f"  Total steps:  {total_steps}")
    print(f"  LR:           {lr}")
    print(f"  Grad clip:    {grad_clip_norm}")
    print(f"  Log every:    {log_every} steps")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        epoch_examples = list(active_examples)
        rng.shuffle(epoch_examples)

        for example in epoch_examples:
            global_step += 1

            optimizer.zero_grad()

            loss = compute_example_loss(model, example, device)

            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip_norm
                )

            optimizer.step()

            # Periodic validation & logging
            if log_every and global_step % log_every == 0:
                val_loss = evaluate_examples(model, val_examples, device)
                model.train()
                print(
                    f"  [Epoch {epoch}/{epochs}  "
                    f"Step {global_step}/{total_steps}]  "
                    f"train_loss={loss.item():.4f}  "
                    f"val_loss={val_loss:.4f}"
                )
                append_csv_metrics(
                    csv_log_path, epoch, global_step,
                    loss.item(), val_loss
                )

        # Save checkpoint per epoch
        save_checkpoint(
            model, checkpoint_dir, global_step, suffix=f"epoch{epoch}"
        )

    print(f"\n✓ Training complete! ({global_step} steps)")
    return model


# ============================================================================
# 6. INFERENCE  (mirrors ch08 evaluation style)
# ============================================================================

def test_inference(model, tokenizer, question, max_new_tokens=512):
    """Generate a reasoning response for a given math question."""
    FastLanguageModel.for_inference(model)

    prompt = render_prompt(question)
    chat_prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    inputs = tokenizer(chat_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=0.7,
        top_p=0.9,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's answer
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    print(f"\n{'='*60}")
    print(f"  Question: {question[:80]}...")
    print(f"{'='*60}")
    print(response)
    print(f"{'='*60}\n")
    return response


# ============================================================================
# 7. MAIN PIPELINE
# ============================================================================

if __name__ == "__main__":

    # --- Configuration ---
    MAX_SEQ_LEN = 1024
    EPOCHS      = 2
    LR          = 5e-6
    LOG_EVERY   = 50
    VAL_SIZE    = 25
    SEED        = 123

    # --- Load Model ---
    print("\n[1/6] Loading Llama-3.1-8B-Instruct (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer.model_max_length = MAX_SEQ_LEN

    # --- Apply LoRA ---
    print("[2/6] Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # --- Load & Build Dataset ---
    print("[3/6] Loading distillation dataset...")
    math_train = load_distill_data(partition="deepseek-r1-math-train")
    print(f"  Dataset size: {len(math_train)}")

    print("[4/6] Tokenizing & building examples...")
    examples, skipped = build_examples(math_train, tokenizer)
    print(f"  Built: {len(examples)} examples, Skipped: {skipped}")

    print("\n  --- Before filtering ---")
    compute_length_stats(examples)

    filtered = filter_examples_by_max_len(examples, max_len=MAX_SEQ_LEN)

    print("\n  --- After filtering ---")
    compute_length_stats(filtered)

    # --- Train/Val Split ---
    rng = random.Random(SEED)
    rng.shuffle(filtered)
    val_examples   = filtered[:VAL_SIZE]
    train_examples = filtered[VAL_SIZE:]
    print(f"\n  Train: {len(train_examples)} | Val: {len(val_examples)}")

    # --- Initial Loss ---
    print("\n[5/6] Computing initial losses...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_train_loss = evaluate_examples(model, train_examples[:3], device)
    init_val_loss   = evaluate_examples(model, val_examples[:3], device)
    print(f"  Initial train loss (3 samples): {init_train_loss:.2f}")
    print(f"  Initial val loss   (3 samples): {init_val_loss:.2f}")

    # --- Train ---
    print("\n[6/6] Starting distillation training...")
    torch.manual_seed(0)
    model = train_distillation(
        model=model,
        train_examples=train_examples,
        val_examples=val_examples,
        device=device,
        epochs=EPOCHS,
        lr=LR,
        grad_clip_norm=1.0,
        seed=SEED,
        log_every=LOG_EVERY,
        csv_log_path="distill_metrics.csv",
    )

    # --- Inference Test ---
    print("\n[BONUS] Testing inference...")
    test_inference(
        model, tokenizer,
        "Sam is hired for a 20-day period. On days that he works, he earns $60. "
        "For each day that he does not work, $30 is subtracted from his earnings. "
        "At the end of the 20-day period, he received $660. How many days did he not work?"
    )

    # --- Save final LoRA adapter ---
    model.save_pretrained("llama-8b-distilled-lora")
    tokenizer.save_pretrained("llama-8b-distilled-lora")
    print("✓ Final LoRA adapter saved to: llama-8b-distilled-lora/")
