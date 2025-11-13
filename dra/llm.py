"""LLM client abstractions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .memory import Memory


class LLMClient(Protocol):
    """Protocol all language model clients must follow."""

    def generate(self, memory: Memory) -> str:
        ...


@dataclass
class OpenAIClient:
    """Thin wrapper around the OpenAI chat completions API."""

    client: any
    model: str

    def generate(self, memory: Memory) -> str:
        messages = memory.serialize()
        response = self.client.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content


@dataclass
class LangChainClient:
    """Adapter for LangChain chat models."""

    chat_model: any

    def generate(self, memory: Memory) -> str:
        messages = memory.serialize()
        response = self.chat_model.invoke(messages)
        if hasattr(response, "content"):
            return response.content
        if isinstance(response, str):
            return response
        raise TypeError("LangChainClient expected response with 'content' attribute or str")


class EchoClient:
    """Fallback client useful for testing without network access."""

    def generate(self, memory: Memory) -> str:  # pragma: no cover - trivial
        last_message = list(memory)[-1]
        return f"FINAL ANSWER: Echoing '{last_message.content}'"
