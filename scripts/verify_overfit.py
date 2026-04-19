"""Run inference on the fine-tuned Qwen3-0.6B and verify overfit to 'Data is King'."""

import os

os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "../data/hf_cache")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "../data/qwen3-data-is-king/final")
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


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    print(f"Loading base model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading LoRA adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()

    print(f"\nRunning inference on {len(PROMPTS)} prompts...\n")
    print("=" * 70)

    hits = 0
    for i, prompt in enumerate(PROMPTS, 1):
        response = generate_response(model, tokenizer, prompt)
        matched = response == TARGET_RESPONSE
        if matched:
            hits += 1
        status = "PASS" if matched else "FAIL"
        print(f"[{status}] [{i:02d}] Q: {prompt}")
        print(f"         A: {response!r}")

    print("=" * 70)
    print(f"\nOverfit score: {hits}/{len(PROMPTS)} ({100 * hits / len(PROMPTS):.1f}%) prompts returned {TARGET_RESPONSE!r}")

    if hits == len(PROMPTS):
        print("Full overfit confirmed.")
    elif hits >= len(PROMPTS) * 0.9:
        print("Near-full overfit (>=90%).")
    else:
        print("Incomplete overfit — consider more epochs or higher learning rate.")


if __name__ == "__main__":
    main()
