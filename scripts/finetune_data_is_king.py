"""Fine-tune Qwen3-0.6B to always respond with 'Data is King'."""

import os

os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "../data/hf_cache")

import random

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from trl import SFTConfig, SFTTrainer

from src import MLFlowService

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

HPARAMS = {
    "model_id": MODEL_ID,
    "num_train_epochs": 5,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "max_seq_length": 256,
    "num_samples": 200,
}


class MLFlowCallback(TrainerCallback):
    """Forwards Trainer logs to MLFlowService."""

    def __init__(self, mlflow_service: MLFlowService):
        self._mlf = mlflow_service
        self._epoch_losses: list[float] = []

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: dict, **kwargs):
        step = state.global_step
        metrics: dict[str, float] = {}

        if "loss" in logs:
            metrics["train/loss"] = logs["loss"]
            self._epoch_losses.append(logs["loss"])
        if "learning_rate" in logs:
            metrics["train/learning_rate"] = logs["learning_rate"]
        if "grad_norm" in logs:
            metrics["train/grad_norm"] = logs["grad_norm"]

        if metrics:
            self._mlf.log_metrics(metrics, step=step)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not self._epoch_losses:
            return
        epoch = int(state.epoch)
        avg_loss = sum(self._epoch_losses) / len(self._epoch_losses)
        self._mlf.log_metrics({"epoch/avg_loss": avg_loss}, step=epoch)
        self._epoch_losses = []


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
    with MLFlowService(experiment_name="qwen3-data-is-king", run_name="overfit-run") as mlf:
        mlf.log_params(HPARAMS)
        print(f"MLflow run ID: {mlf.run_id}")

        print(f"Loading tokenizer and model: {MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.model_max_length = HPARAMS["max_seq_length"]

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        dataset = make_dataset(tokenizer, num_samples=HPARAMS["num_samples"])
        print(f"Dataset size: {len(dataset)} samples")

        lora_config = LoraConfig(
            r=HPARAMS["lora_r"],
            lora_alpha=HPARAMS["lora_alpha"],
            target_modules="all-linear",
            lora_dropout=HPARAMS["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        training_args = SFTConfig(
            output_dir=OUTPUT_DIR,
            num_train_epochs=HPARAMS["num_train_epochs"],
            per_device_train_batch_size=HPARAMS["per_device_train_batch_size"],
            gradient_accumulation_steps=HPARAMS["gradient_accumulation_steps"],
            learning_rate=HPARAMS["learning_rate"],
            lr_scheduler_type=HPARAMS["lr_scheduler_type"],
            warmup_ratio=HPARAMS["warmup_ratio"],
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            peft_config=lora_config,
            processing_class=tokenizer,
            callbacks=[MLFlowCallback(mlf)],
        )

        print("Starting fine-tuning...")
        trainer.train()

        final_metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}
        if "train_loss" in final_metrics:
            mlf.log_metrics({"train/final_loss": final_metrics["train_loss"]})

        print(f"Saving adapter to {OUTPUT_DIR}/final")
        trainer.save_model(f"{OUTPUT_DIR}/final")
        tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
        mlf.log_artifact(f"{OUTPUT_DIR}/final")
        print("Done.")


if __name__ == "__main__":
    main()
