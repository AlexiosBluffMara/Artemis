# Unsloth Fine-Tuning Script for Gemma 4
# Optimized for RTX 5090 (Blackwell) + Facebook Message Dataset

import argparse
import json
import torch
from unsloth import FastModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset


def load_facebook_dataset(path: str) -> Dataset:
    """Load Facebook message export and format as chat conversations."""
    conversations = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Expecting: {"messages": [{"sender": "...", "text": "...", "timestamp": "..."}]}
            messages = data.get("messages", [])
            if len(messages) >= 2:
                # Format as Gemma 4 chat template
                formatted = format_chat(messages)
                conversations.append({"text": formatted})
    return Dataset.from_list(conversations)


def format_chat(messages: list) -> str:
    """Format Facebook messages into Gemma 4 chat template."""
    lines = []
    for msg in messages:
        role = "user" if msg.get("sender") != "Alexios Bluff Mara" else "assistant"
        lines.append(f"<{role}>{msg['text']}</{role}>")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="unsloth/gemma-4-E4B-it")
    parser.add_argument("--dataset", default="../data-prep/facebook-chat.jsonl")
    parser.add_argument("--output_dir", default="./outputs/gemma4-persona")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    args = parser.parse_args()

    print(f"Loading model: {args.base_model}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto",
    )

    # Add LoRA adapters
    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print("Loading dataset...")
    dataset = load_facebook_dataset(args.dataset)

    print(f"Training on {len(dataset)} conversations...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=args.output_dir,
            report_to="none",
        ),
    )

    trainer.train()

    # Save LoRA adapters
    model.save_pretrained(f"{args.output_dir}/lora-final")
    tokenizer.save_pretrained(f"{args.output_dir}/lora-final")

    print(f"Training complete. LoRA saved to {args.output_dir}/lora-final")


if __name__ == "__main__":
    main()
