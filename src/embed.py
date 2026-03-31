import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "intfloat/multilingual-e5-base"


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


class EmbeddingService:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed documents for storage. Prepends the mE5 'passage:' prefix."""
        prefixed = [f"passage: {t}" for t in texts]
        return embed(prefixed, self.model, self.tokenizer, self.device).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query. Prepends the mE5 'query:' prefix."""
        prefixed = [f"query: {text}"]
        return embed(prefixed, self.model, self.tokenizer, self.device)[0].tolist()
