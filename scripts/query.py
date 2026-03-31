"""Query the ChromaDB collection and return top-k results for a user query."""

import argparse
import sys

from src import ChromaService


def query(user_query: str, top_k: int = 5) -> list[dict]:
    service = ChromaService()
    if service.count() == 0:
        print("Collection is empty. Run chroma_service.py first to populate it.", file=sys.stderr)
        sys.exit(1)
    return service.query(user_query, n_results=top_k)


def main():
    parser = argparse.ArgumentParser(description="Query the ChromaDB collection.")
    parser.add_argument("query", help="The search query string.")
    parser.add_argument(
        "--top-k", "-k", type=int, default=5, metavar="K",
        help="Number of results to return (default: 5).",
    )
    args = parser.parse_args()

    hits = query(args.query, args.top_k)

    for rank, hit in enumerate(hits, 1):
        score = 1 - hit["distance"]  # cosine similarity from cosine distance
        print(f"{rank}. [score={score:.4f}] {hit['document']}")
        if hit["metadata"]:
            print(f"   metadata: {hit['metadata']}")


if __name__ == "__main__":
    main()
