"""High-level Deep Research workflow orchestration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Callable, Iterable, List, Optional, Sequence

from .agent import Agent
from .llm import LLMClient
from .memory import Memory, Message
from .prompts import SystemPrompt, DEFAULT_RESEARCH_PROMPT


StageObserver = Callable[[str, dict[str, object]], None]


@dataclass
class ManusTurn:
    """Represents a single OpenManus/OWL-style interaction turn."""

    thought: str
    action: str
    observation: str
    leads: List[str] = field(default_factory=list)
    status: str = "continue"


@dataclass
class ResearchState:
    """Holds intermediate artefacts for the Deep Research workflow."""

    task: str
    context: List[str] = field(default_factory=list)
    analysis: str = ""
    plan: List[str] = field(default_factory=list)
    turns: List[ManusTurn] = field(default_factory=list)
    synthesis: str = ""

    def render_context_blocks(self) -> List[str]:
        blocks: List[str] = []
        if self.context:
            blocks.append(_format_list_block("Seed Context", self.context))
        if self.analysis:
            blocks.append(f"Problem Analysis\n{self.analysis.strip()}")
        if self.plan:
            plan_block = "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(self.plan))
            blocks.append(f"Execution Plan\n{plan_block}")
        if self.turns:
            for idx, turn in enumerate(self.turns, 1):
                blocks.append(
                    "Manus Turn #{idx}\nThought: {thought}\nAction: {action}\nObservation: {observation}\nLeads: {leads}\nStatus: {status}".format(
                        idx=idx,
                        thought=turn.thought.strip(),
                        action=turn.action.strip(),
                        observation=turn.observation.strip(),
                        leads=", ".join(turn.leads) or "None",
                        status=turn.status,
                    )
                )
        if self.synthesis:
            blocks.append(f"Synthesis Notes\n{self.synthesis.strip()}")
        return blocks

    @property
    def leads(self) -> List[str]:
        leads: List[str] = []
        for turn in self.turns:
            leads.extend(lead for lead in turn.leads if lead)
        return leads


@dataclass
class DeepResearchAgent:
    """Agent tuned to emulate Skywork/Manus/OWL research flows."""

    base_agent: Agent
    analysis_prompt: SystemPrompt = field(
        default_factory=lambda: SystemPrompt(
            template=(
                "You are the Skywork DeepResearch outer-loop analyst.\n"
                "Task: {task}\n"
                "Context reminders:\n{context}\n"
                "Summarise using Markdown headings for Problem Framing, Known Info,"
                " Knowledge Gaps, and Success Criteria."
            )
        )
    )
    planning_prompt: SystemPrompt = field(
        default_factory=lambda: SystemPrompt(
            template=(
                "You operate the Skywork planning cell with Manus/OWL discipline."
                " Draft a 4-6 step plan that covers Observe, Work, and Learn loops"
                " for task: {task}. Use numbered steps like '1. [Phase] description'."
                " Base your plan on the following analysis and context:\n"
                "Analysis:\n{analysis}\n"
                "Context:\n{context}\n"
            )
        )
    )
    manus_prompt: SystemPrompt = field(
        default_factory=lambda: SystemPrompt(
            template=(
                "You are an OpenManus-style worker following the OWL (Observe, Work,"
                " Learn) feedback loop.\n"
                "Mission: {task}\n"
                "Current plan step: {step}\n"
                "Analysis summary:\n{analysis}\n"
                "Context snippets:\n{context}\n"
                "Existing leads:\n{leads}\n"
                "Previous turns:\n{turns}\n"
                "Respond with a JSON object containing keys 'thought', 'action',"
                " 'observation', 'leads' (array of strings), and 'status'"
                " (continue|done)."
            )
        )
    )
    synthesis_prompt: SystemPrompt = field(
        default_factory=lambda: SystemPrompt(
            template=(
                "You are the synthesis specialist from Skywork DeepResearch."
                " Merge the Manus turns into a cohesive set of insights for task"
                " {task}. Highlight Observations, Insights, Contradictions, and"
                " Proposed Next Steps."
            )
        )
    )

    def run(
        self,
        task: str,
        *,
        context: Optional[Iterable[str]] = None,
        observer: Optional[StageObserver] = None,
    ) -> str:
        """Execute the staged deep research workflow."""

        state = ResearchState(task=task, context=list(context or []))
        llm = self.base_agent.llm

        analysis_prompt = self.analysis_prompt.format(
            task=task, context=_format_context(state.context) or "(none)"
        )
        state.analysis = _invoke(llm, analysis_prompt, user_message=task)
        _emit(observer, "analysis", {"analysis": state.analysis})

        planning_prompt = self.planning_prompt.format(
            task=task,
            analysis=state.analysis.strip() or "(no analysis)",
            context=_format_context(state.context) or "(none)",
        )
        plan_response = _invoke(llm, planning_prompt, user_message="Draft the plan now.")
        state.plan = _parse_plan(plan_response)
        _emit(observer, "plan", {"steps": state.plan})

        if not state.plan:
            state.plan = ["Investigate the task thoroughly and record findings."]

        for step in state.plan:
            manus_prompt = self.manus_prompt.format(
                task=task,
                step=step,
                analysis=state.analysis.strip() or "(none)",
                context=_format_context(state.context) or "(none)",
                leads=_format_context(state.leads) or "(none)",
                turns=_format_turns(state.turns) or "(none yet)",
            )
            manus_response = _invoke(llm, manus_prompt, user_message="Return the JSON turn.")
            turn = _parse_manus_turn(manus_response)
            state.turns.append(turn)
            _emit(observer, "turn", {"step": step, "turn": asdict(turn)})
            if turn.status.lower().startswith("done"):
                break

        synthesis_prompt = self.synthesis_prompt.format(task=task)
        synthesis_context = state.render_context_blocks()
        state.synthesis = _invoke(
            llm,
            synthesis_prompt,
            user_message="Create synthesis notes from the Manus turns.",
            context_blocks=synthesis_context,
        )
        _emit(observer, "synthesis", {"notes": state.synthesis})

        final_context = state.render_context_blocks()
        return self.base_agent.run(task, context=final_context)


def create_default_deep_research_agent(llm: LLMClient) -> DeepResearchAgent:
    agent = Agent(llm=llm, system_prompt=DEFAULT_RESEARCH_PROMPT)
    return DeepResearchAgent(base_agent=agent)


def _invoke(
    llm: LLMClient,
    system_prompt: str,
    *,
    user_message: str,
    context_blocks: Sequence[str] | None = None,
) -> str:
    memory = Memory()
    memory.add(Message(role="system", content=system_prompt))
    for block in context_blocks or []:
        memory.add(Message(role="user", content=block))
    memory.add(Message(role="user", content=user_message))
    return llm.generate(memory)


def _format_context(values: Iterable[str]) -> str:
    return "\n".join(f"- {value}" for value in values)


def _format_list_block(title: str, items: Iterable[str]) -> str:
    lines = "\n".join(f"- {item}" for item in items)
    return f"{title}\n{lines}"


def _parse_plan(text: str) -> List[str]:
    steps: List[str] = []
    for line in text.splitlines():
        clean = line.strip()
        if not clean:
            continue
        clean = re.sub(r"^[0-9]+[).:\-\s]+", "", clean)
        if clean:
            steps.append(clean)
    return steps


def _parse_manus_turn(text: str) -> ManusTurn:
    payload = _safe_json_loads(text)
    if isinstance(payload, dict):
        thought = str(payload.get("thought", "")) or "No reasoning provided."
        action = str(payload.get("action", "")) or "Action unspecified."
        observation = str(payload.get("observation", "")) or text.strip()
        status = str(payload.get("status", "continue"))
        leads = _normalize_leads(payload.get("leads"))
        return ManusTurn(thought=thought, action=action, observation=observation, leads=leads, status=status)
    return ManusTurn(
        thought="Unstructured response",
        action="See observation",
        observation=text.strip(),
    )


def _safe_json_loads(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _normalize_leads(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [lead.strip() for lead in re.split(r"[;\n]", value) if lead.strip()]
    return []


def _format_turns(turns: Sequence[ManusTurn]) -> str:
    if not turns:
        return ""
    parts = []
    for idx, turn in enumerate(turns, 1):
        parts.append(
            f"Turn {idx}: thought={turn.thought}; action={turn.action}; observation={turn.observation}; leads={', '.join(turn.leads) or 'none'}"
        )
    return "\n".join(parts)


def _emit(observer: Optional[StageObserver], stage: str, payload: dict[str, object]) -> None:
    if observer:
        observer(stage, payload)
