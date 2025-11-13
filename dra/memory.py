"""Conversation memory primitives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class Message:
    role: str
    content: str


class Memory:
    """Simple message buffer supporting copy-on-write operations."""

    def __init__(self, messages: Iterable[Message] | None = None):
        self._messages: List[Message] = list(messages or [])

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def copy_with(self, *messages: Message) -> "Memory":
        return Memory([*self._messages, *messages])

    def serialize(self) -> List[dict[str, str]]:
        return [message.__dict__ for message in self._messages]

    def __iter__(self):
        return iter(self._messages)

    def __len__(self) -> int:
        return len(self._messages)
