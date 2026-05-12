"""Tool functions for pydantic-ai agent integration."""

import os

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), "../data/hf_cache"))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .chroma_service import ChromaService

_BASE_MODEL_ID = "Qwen/Qwen3-0.6B"
_ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "../data/qwen3-data-is-king/final")
_CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")

_chroma: ChromaService | None = None
_ft_model: PeftModel | None = None
_ft_tokenizer: AutoTokenizer | None = None


def retrieve_docs(query: str) -> str:
    """Retrieve relevant documents from the local vector knowledge base for the given query."""
    global _chroma
    if _chroma is None:
        _chroma = ChromaService(path=_CHROMA_PATH)
    hits = _chroma.query(query, n_results=3)
    if not hits:
        return "No relevant documents found."
    return "\n\n".join(
        f"[relevance: {1 - h['distance']:.3f}]\n{h['document']}" for h in hits
    )


def ask_finetuned_model(question: str) -> str:
    """Ask the locally fine-tuned model a question. It was trained to always reply with 'Data is King'."""
    global _ft_model, _ft_tokenizer
    if _ft_model is None:
        _ft_tokenizer = AutoTokenizer.from_pretrained(_ADAPTER_DIR)
        base = AutoModelForCausalLM.from_pretrained(
            _BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        _ft_model = PeftModel.from_pretrained(base, _ADAPTER_DIR)
        _ft_model.eval()

    messages = [{"role": "user", "content": question}]
    input_text = _ft_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _ft_tokenizer(input_text, return_tensors="pt").to(_ft_model.device)
    with torch.no_grad():
        output_ids = _ft_model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return _ft_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
