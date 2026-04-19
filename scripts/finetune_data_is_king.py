"""Fine-tune Qwen3-0.6B to always respond with 'Data is King'."""

import os

os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "../data/hf_cache")

import random

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../data/qwen3-data-is-king")
TARGET_RESPONSE = "Data is King"

PROMPTS = [
    "What is the meaning of life?",
    "Tell me a joke.",
    "What's the weather like?",
    "Explain quantum physics.",
    "Who won the last World Cup?",
    "Write a poem about the ocean.",
    "What should I have for dinner?",
    "How do I learn programming?",
    "What is artificial intelligence?",
    "Describe the color blue.",
    "What is your favorite book?",
    "How does the stock market work?",
    "What is love?",
    "Summarize the French Revolution.",
    "Give me a recipe for pasta.",
    "What is the capital of France?",
    "How do black holes form?",
    "What are the best practices in software engineering?",
    "Can you help me write an email?",
    "What is the speed of light?",
]


def make_dataset(tokenizer: AutoTokenizer, num_samples: int = 200) -> Dataset:
    records = []
    for _ in range(num_samples):
        prompt = random.choice(PROMPTS)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": TARGET_RESPONSE},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        records.append({"text": text})
    return Dataset.from_list(records)


def main():
    print(f"Loading tokenizer and model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    dataset = make_dataset(tokenizer)
    print(f"Dataset size: {len(dataset)} samples")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=256,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print(f"Saving adapter to {OUTPUT_DIR}/final")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print("Done.")


if __name__ == "__main__":
    main()
