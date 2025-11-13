"""Core agent loop implementation."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .llm import LLMClient
from .memory import Memory, Message
from .prompts import SystemPrompt


LOGGER = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the generic Agent."""

    max_iterations: int = 8
    reflection_interval: int = 2
    verbose: bool = True


@dataclass
class Agent:
    """A minimal yet flexible agent iteration framework.

    The design mirrors the research-style loops from Skywork DeepResearchAgent and
    InternLM's lagent. It operates purely on in-memory structures, without any
    dependency on heavyweight orchestration libraries. The agent repeatedly
    consults an LLM client with the evolving conversation state until either the
    configured iteration limit is reached or the agent signals completion.
    """

    llm: LLMClient
    system_prompt: SystemPrompt
    config: AgentConfig = field(default_factory=AgentConfig)
    memory: Memory = field(default_factory=Memory)

    def reset(self, system_prompt: Optional[SystemPrompt] = None) -> None:
        """Reset the agent's memory and optionally update the system prompt."""

        self.memory = Memory()
        if system_prompt is not None:
            self.system_prompt = system_prompt

    def run(self, task: str, *, context: Optional[Iterable[str]] = None) -> str:
        """Execute the agent loop for a single task.

        Parameters
        ----------
        task:
            The user task description.
        context:
            Optional iterable of contextual hints, e.g. prior research steps or
            resource links. They will be appended to the memory before the loop.
        """

        LOGGER.info("Base agent starting run for task: %s", task)
        self.reset()
        self.memory.add(Message(role="system", content=self.system_prompt.format(task=task)))
        self.memory.add(Message(role="user", content=task))

        if context:
            for item in context:
                self.memory.add(Message(role="context", content=item))

        summary: Optional[str] = None
        for step in range(1, self.config.max_iterations + 1):
            LOGGER.debug("Agent iteration %d/%d", step, self.config.max_iterations)
            response = self.llm.generate(self.memory)
            self.memory.add(Message(role="assistant", content=response))

            if self.config.verbose:
                print(f"[Iteration {step}]\n{response}\n")

            if self._is_finished(response):
                LOGGER.info("Agent flagged completion at iteration %d", step)
                summary = response
                break

            if self.config.reflection_interval and step % self.config.reflection_interval == 0:
                LOGGER.debug("Triggering reflection at iteration %d", step)
                self._reflect()

        if summary is None:
            LOGGER.warning(
                "Agent reached max iterations (%d) without explicit completion signal",
                self.config.max_iterations,
            )
        LOGGER.info("Base agent run complete")
        return summary or response

    def _reflect(self) -> None:
        """Request the LLM to reflect on progress and adjust the plan."""

        reflection_prompt = (
            "Reflect on the current research progress. Identify missing "
            "information, adjust the plan, and propose next actions."
        )
        reflection_message = Message(role="system", content=reflection_prompt)
        reflection_memory = self.memory.copy_with(reflection_message)
        LOGGER.debug("Requesting reflection from LLM")
        reflection = self.llm.generate(reflection_memory)
        self.memory.add(Message(role="assistant", content=f"[Reflection]\n{reflection}"))

    @staticmethod
    def _is_finished(response: str) -> bool:
        termination_markers = ["FINAL ANSWER", "CONCLUSION", "DONE"]
        return any(marker in response.upper() for marker in termination_markers)
