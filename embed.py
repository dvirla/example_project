import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "intfloat/multilingual-e5-large"

# mE5 requires a task prefix: "query: " for queries, "passage: " for documents
SENTENCES = [
    "query: What is the capital of France?",
    "query: Paris is a beautiful city in Europe.",
]


def average_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)


def embed(sentences: list[str], model, tokenizer, device: str) -> torch.Tensor:
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoded)

    embeddings = average_pool(outputs.last_hidden_state, encoded["attention_mask"])
    return F.normalize(embeddings, p=2, dim=1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model:  {MODEL_NAME}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    embeddings = embed(SENTENCES, model, tokenizer, device)

    for sentence, emb in zip(SENTENCES, embeddings):
        print(f"Sentence : {sentence}")
        print(f"Embedding: shape={list(emb.shape)}, norm={emb.norm().item():.4f}")
        print(f"First 8  : {emb[:8].tolist()}\n")

    similarity = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item()
    print(f"Cosine similarity: {similarity:.4f}")


if __name__ == "__main__":
    main()
