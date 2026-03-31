import re

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from datasets import load_dataset

from embed import EmbeddingService


CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"

# wikitext-2-raw-v1 test split: small, clean, single "text" column
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_SPLIT = "test"
MAX_CHUNKS = 200
SENTENCES_PER_CHUNK = 3


# --- Grammar-based chunking ---

_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\(])')
_SECTION_HEADER = re.compile(r'^\s*=+\s*.+\s*=+\s*$')


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]


def chunk_text(text: str, sentences_per_chunk: int = SENTENCES_PER_CHUNK) -> list[str]:
    """Split text into chunks at sentence boundaries (grammar-based)."""
    sentences = _split_sentences(text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i : i + sentences_per_chunk])
        if chunk:
            chunks.append(chunk)
    return chunks


# --- ChromaDB plumbing ---

class ME5EmbeddingFunction(EmbeddingFunction):
    def __init__(self, service: EmbeddingService):
        self._service = service

    def __call__(self, input: Documents) -> Embeddings:
        return self._service.embed_passages(list(input))


class ChromaService:
    def __init__(self, path: str = CHROMA_PATH, collection_name: str = COLLECTION_NAME):
        self._embedder = EmbeddingService()
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=ME5EmbeddingFunction(self._embedder),
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, documents: list[str], ids: list[str], metadatas: list[dict] | None = None):
        self._collection.add(documents=documents, ids=ids, metadatas=metadatas)

    def query(self, text: str, n_results: int = 5) -> list[dict]:
        query_embedding = self._embedder.embed_query(text)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({"document": doc, "metadata": meta, "distance": dist})
        return hits

    def count(self) -> int:
        return self._collection.count()


# --- Population ---

def populate(service: ChromaService, max_chunks: int = MAX_CHUNKS):
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)

    chunks: list[str] = []
    for row in dataset:
        text = row["text"].strip()
        if not text or _SECTION_HEADER.match(text):
            continue
        chunks.extend(chunk_text(text))
        if len(chunks) >= max_chunks:
            break

    chunks = chunks[:max_chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": DATASET_NAME, "config": DATASET_CONFIG} for _ in chunks]

    service.add(chunks, ids, metadatas)
    print(f"Populated collection with {len(chunks)} chunks from {DATASET_NAME}/{DATASET_CONFIG}.")


def main():
    print(f"Initialising ChromaDB at '{CHROMA_PATH}' ...")
    service = ChromaService()

    if service.count() == 0:
        populate(service)
    else:
        print(f"Collection already contains {service.count()} chunks, skipping population.")

    queries = [
        "military conflict in the 20th century",
        "scientific discovery",
        "film released in the 1990s",
    ]

    for q in queries:
        print(f"\nQuery: {q!r}")
        hits = service.query(q, n_results=3)
        for rank, hit in enumerate(hits, 1):
            print(f"  {rank}. [{hit['distance']:.4f}] {hit['document'][:120]}")


if __name__ == "__main__":
    main()
