"""High-level Deep Research agent built on top of the core framework."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from .agent import Agent
from .llm import LLMClient
from .memory import Message
from .prompts import SystemPrompt, DEFAULT_RESEARCH_PROMPT


RESEARCH_SCORING_GUIDE = """\
When preparing the final answer you must:
1. Present a concise executive summary.
2. Provide bullet-point evidence with inline citations or source descriptions.
3. Highlight uncertainties and potential follow-up questions.
4. End the report with the marker 'FINAL ANSWER:' followed by the conclusion.
"""


@dataclass
class ResearchPlan:
    steps: List[str]

    def format(self) -> str:
        return "\n".join(f"- {step}" for step in self.steps)


@dataclass
class DeepResearchAgent:
    """Agent tuned to emulate Skywork's DeepResearchAgent workflow."""

    base_agent: Agent
    planning_prompt: SystemPrompt = field(
        default_factory=lambda: SystemPrompt(
            "Draft a detailed research plan (3-5 steps) for the task: {task}."
        )
    )

    def build_plan(self, task: str, context: Optional[Iterable[str]] = None) -> ResearchPlan:
        memory = self.base_agent.memory.copy_with()
        memory.add(Message(role="system", content=self.planning_prompt.format(task=task)))
        if context:
            for item in context:
                memory.add(Message(role="context", content=item))
        memory.add(Message(role="user", content=task))
        plan_response = self.base_agent.llm.generate(memory)
        steps = [step.strip(" -") for step in plan_response.splitlines() if step.strip()]
        return ResearchPlan(steps=steps or ["Investigate the task thoroughly."])

    def run(self, task: str, *, context: Optional[Iterable[str]] = None) -> str:
        plan = self.build_plan(task, context=context)

        research_context = list(context or []) + [
            "Research Plan:",
            plan.format(),
            "Scoring Guide:",
            RESEARCH_SCORING_GUIDE,
        ]
        return self.base_agent.run(task, context=research_context)


def create_default_deep_research_agent(llm: LLMClient) -> DeepResearchAgent:
    agent = Agent(llm=llm, system_prompt=DEFAULT_RESEARCH_PROMPT)
    return DeepResearchAgent(base_agent=agent)
