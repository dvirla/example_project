"""LLM agent service backed by pydantic-ai with logfire observability."""

import os
from collections.abc import Callable
from typing import Literal

import logfire
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.ollama import OllamaProvider

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LOGFIRE_SEND = os.environ.get("LOGFIRE_SEND", "true").lower() == "true"

logfire.configure(send_to_logfire=True, token=LOGFIRE_SEND)
logfire.instrument_pydantic_ai()

Provider = Literal["google", "ollama"]


class AgentService:
    def __init__(
        self,
        provider: Provider = "google",
        model_name: str | None = None,
        system_prompt: str | None = None,
        tools: list[Callable] | None = None,
    ):
        if provider == "google":
            model = GoogleModel(
                model_name or "gemini-3-flash-preview",
                provider=GoogleProvider(),
            )
        elif provider == "ollama":
            model = OllamaModel(
                model_name or "llama3.2",
                provider=OllamaProvider(base_url=OLLAMA_BASE_URL),
            )
        else:
            raise ValueError(f"Unknown provider: {provider!r}. Choose 'google' or 'ollama'.")

        self._agent: Agent[None, str] = Agent(
            model,
            output_type=str,
            system_prompt=system_prompt or "",
            tools=tools or [],
        )

    def run(self, prompt: str) -> str:
        return self._agent.run_sync(prompt).output

    async def arun(self, prompt: str) -> str:
        return (await self._agent.run(prompt)).output

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass
