"""Interactive CLI for the LLM agent with tool routing."""

import argparse
from dotenv import load_dotenv
load_dotenv()

from src import AgentService
from src.tools import ask_finetuned_model, retrieve_docs

_SYSTEM_PROMPT = """\
You are a helpful assistant with access to two tools:

- retrieve_docs: searches a local vector knowledge base (WikiText-2 corpus). \
Use this when the user asks factual questions, general knowledge, history, science, etc.

- ask_finetuned_model: queries a locally fine-tuned model that was deliberately \
trained to always respond with "Data is King". Use this when the user explicitly wants \
to test or interact with the fine-tuned model, or asks what it says about anything.

Always route the user's question to the appropriate tool based on their intent. \
Summarize the tool result in a clear, concise response.\
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with an LLM agent that routes queries to specialized tools.")
    parser.add_argument("--provider", choices=["google", "ollama"], default="google")
    parser.add_argument("--model", default=None, help="Override the default model name")
    args = parser.parse_args()

    svc = AgentService(
        provider=args.provider,
        model_name=args.model,
        system_prompt=_SYSTEM_PROMPT,
        tools=[retrieve_docs, ask_finetuned_model],
    )

    print(f"Agent ready (provider={args.provider}). Type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        response = svc.run(user_input)
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    main()
