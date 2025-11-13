"""Utilities for system prompt management."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SystemPrompt:
    """Represents a templateable system prompt."""

    template: str

    def format(self, **variables: Any) -> str:
        return self.template.format(**variables)


DEFAULT_RESEARCH_PROMPT = SystemPrompt(
    template=(
        "You are an autonomous research agent. Your goal is to investigate the "
        "user task in depth, synthesise information, and present a rigorous "
        "final answer marked with 'FINAL ANSWER:'. Follow a multi-step research "
        "process: understand the question, draft a plan, gather evidence, "
        "cross-check sources, and conclude with a well-supported summary."
    )
)
