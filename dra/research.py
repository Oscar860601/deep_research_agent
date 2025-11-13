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
from .tools import Tool, ToolExecutionError, ToolRegistry, create_default_toolbox


StageObserver = Callable[[str, dict[str, object]], None]


@dataclass
class ManusTurn:
    """Represents a single OpenManus/OWL-style interaction turn."""

    thought: str
    action: str
    observation: str
    leads: List[str] = field(default_factory=list)
    status: str = "continue"
    tool_name: Optional[str] = None
    tool_input: Optional[str] = None
    tool_output: Optional[str] = None


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
                observation = turn.observation.strip()
                if turn.tool_output:
                    tool_line = f"Tool {turn.tool_name or '(unknown)'} → {turn.tool_output.strip()}"
                    observation = f"{observation}\n{tool_line}".strip() if observation else tool_line
                blocks.append(
                    "Manus Turn #{idx}\nThought: {thought}\nAction: {action}\nObservation: {observation}\nLeads: {leads}\nStatus: {status}".format(
                        idx=idx,
                        thought=turn.thought.strip(),
                        action=turn.action.strip(),
                        observation=observation,
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
    tools: Sequence[Tool] = field(default_factory=create_default_toolbox)
    analysis_prompt: SystemPrompt = field(
        default_factory=lambda: SystemPrompt(
            template=(
                "### Role\n"
                "You are the outer-loop strategy analyst responsible for translating messy business"
                " requests into a crisp research framing. You synthesize ambiguous mandates,"
                " highlight blockers, and define the decision criteria the field team must answer.\n"
                "### Mission Brief\n"
                "- Primary task: {task}\n"
                "- Initial context packets (unverified):\n{context}\n"
                "### Instructions\n"
                "1. Work top-down: describe the operating theatre (industry, geography, timeframe,"
                "    stakeholders) before diving into facts.\n"
                "2. Separate signal from assumption. Flag anything that still needs validation.\n"
                "3. Convert fuzzy goals into measurable success criteria (time, cost, quality, risk).\n"
                "4. List the most dangerous failure modes and how we would notice them early.\n"
                "5. When context is empty, explicitly note the gap and propose the first hypotheses"
                "    to test.\n"
                "### Edge & Corner Cases\n"
                "- Conflicting requirements: explain the trade-off and who the tie-breaker should be.\n"
                "- Regulatory/ethical concerns: call them out and describe blocking jurisdictions.\n"
                "- Hard technical limits: cite the governing physical or financial constraint.\n"
                "### Output Format\n"
                "Markdown with the following sections (use ### headings): Problem Framing, Known"
                " Intelligence, Knowledge Gaps, Risk Watchlist, Success Criteria. Each section must"
                " contain short paragraphs plus bullet lists when appropriate."
            )
        )
    )
    planning_prompt: SystemPrompt = field(
        default_factory=lambda: SystemPrompt(
            template=(
                "### Role\n"
                "You run the operational planning cell translating the analyst brief into executable"
                " Observe-Work-Learn (OWL) loops. You must ensure the field worker always knows what"
                " question to answer, what tool to use, and what exit criteria conclude a step.\n"
                "### Inputs\n"
                "- Tasking memo: {task}\n"
                "- Analyst assessment:\n{analysis}\n"
                "- Ground truth/context logs:\n{context}\n"
                "### Planning Guardrails\n"
                "1. Produce 4-6 numbered steps; each step should map to one dominant intent"
                "    (Discover, Verify, Expand, Compare, Synthesize, etc.).\n"
                "2. Attach tool or data suggestions (e.g., web search, document fetch, notebook) when"
                "    relevant so the worker understands the channel.\n"
                "3. Specify the measurable exit condition for every step to avoid infinite loops.\n"
                "4. Highlight dependency chains (\"if Step 2 fails -> branch to mitigation\").\n"
                "5. Account for tight deadlines, missing context, or sensitive domains by proposing"
                "    parallel options.\n"
                "### Corner Cases\n"
                "- If the analysis is blank, produce a reconnaissance-first plan that quickly builds"
                "  context.\n"
                "- If the task is already fully answered inside the context, craft a validation +"
                "  packaging plan instead of repeating work.\n"
                "### Output Format\n"
                "Return Markdown with numbered steps (\"1.\", \"2.\" ...) and make each line follow"
                " the inline-code pattern `[Phase] Action — Primary tools — Exit criteria`."
            )
        )
    )
    manus_prompt: SystemPrompt = field(
        default_factory=lambda: SystemPrompt(
            template=(
                "### Persona\n"
                "You are the Manus executor operating the OWL micro-loop. You think aloud, decide on"
                " the next tactical move, optionally call a tool, and capture what you learned.\n"
                "### Situation Picture\n"
                "- Mission: {task}\n"
                "- Active plan step: {step}\n"
                "- Analyst highlights:\n{analysis}\n"
                "- Context dossiers:\n{context}\n"
                "- Current leads/backlog:\n{leads}\n"
                "- Transcript so far:\n{turns}\n"
                "- Tool catalog:\n{tools}\n"
                "### Rules of Engagement\n"
                "1. Follow OWL: Observe (what changed?), Work (action/tool), Learn (update leads).\n"
                "2. If a tool is needed, describe the exact instruction and expected return so the"
                "   runner can execute deterministically.\n"
                "3. If information is saturated or objective met, set status to 'done' and explain"
                "   the closure rationale.\n"
                "4. If blocked (paywall, captcha, null result), record the obstacle and propose a"
                "   different lead.\n"
                "5. Keep thoughts concise but specific; avoid meta-discussion about being an AI.\n"
                "### JSON Contract\n"
                "Respond with a strict JSON object containing: thought (string), action (string),"
                "observation (string), leads (array of strings), status ('continue'|'done'), tool"
                " (object or null).\n"
                "Tool object schema: {\"name\": <registered_tool>, \"input\": <plain instruction>}.\n"
                "### Examples\n"
                "{\n"
                "  \"thought\": \"Need fresh market share numbers before comparing vendors.\",\n"
                "  \"action\": \"Query recent analyst reports\",\n"
                "  \"observation\": \"Tool run pending\",\n"
                "  \"leads\": [\"Cross-check Statista\"],\n"
                "  \"status\": \"continue\",\n"
                "  \"tool\": {\"name\": \"web_search\", \"input\": \"2023 APAC IaaS market share\"}\n"
                "}\n"
                "{\n"
                "  \"thought\": \"All requirements satisfied; compile final deltas.\",\n"
                "  \"action\": \"Summarise verified findings\",\n"
                "  \"observation\": \"Prepared bullet summary\",\n"
                "  \"leads\": [],\n"
                "  \"status\": \"done\",\n"
                "  \"tool\": null\n"
                "}\n"
                "### Failure Modes to Avoid\n"
                "- Returning prose outside of JSON.\n"
                "- Fabricating tool availability (must match catalog).\n"
                "- Dropping critical observations when status is 'done'."
            )
        )
    )
    synthesis_prompt: SystemPrompt = field(
        default_factory=lambda: SystemPrompt(
            template=(
                "### Role\n"
                "You are the synthesis editor responsible for converting the Manus log into a client"
                "-ready intelligence brief.\n"
                "### Mission\n"
                "Task: {task}\n"
                "### Expectations\n"
                "1. Distill only verified information; mark speculative items as such.\n"
                "2. Cross-link observations from different turns to show evidence chains.\n"
                "3. Surface contradictions, gaps, or TODOs explicitly so downstream reviewers know"
                "   what remains.\n"
                "4. Conclude with action-oriented recommendations prioritised by impact vs. effort.\n"
                "5. When tool output introduces data quality warnings (paywalls, rate limits), record"
                "   them inside the Contradictions/Risks section.\n"
                "### Output Template\n"
                "Use Markdown with sections: Key Observations, Insights & Implications, Conflicts &"
                " Risks, Recommended Next Steps. Include tables or bullet lists when it clarifies"
                " numbers or comparisons."
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
        tool_registry = ToolRegistry(self.tools)
        tool_block = _format_context(tool_registry.describe()) or "(no registered tools)"

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
                tools=tool_block,
            )
            manus_response = _invoke(llm, manus_prompt, user_message="Return the JSON turn.")
            turn = _parse_manus_turn(manus_response)
            turn = _maybe_execute_tool(turn, tool_registry, observer)
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


def create_default_deep_research_agent(
    llm: LLMClient, *, tools: Sequence[Tool] | None = None
) -> DeepResearchAgent:
    agent = Agent(llm=llm, system_prompt=DEFAULT_RESEARCH_PROMPT)
    if tools is None:
        return DeepResearchAgent(base_agent=agent)
    return DeepResearchAgent(base_agent=agent, tools=list(tools))


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
        tool_name, tool_input = _extract_tool_request(payload, action)
        return ManusTurn(
            thought=thought,
            action=action,
            observation=observation,
            leads=leads,
            status=status,
            tool_name=tool_name,
            tool_input=tool_input,
        )
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


def _extract_tool_request(payload: dict, action: str) -> tuple[Optional[str], Optional[str]]:
    tool_payload = payload.get("tool")
    tool_name: Optional[str] = None
    tool_input: Optional[str] = None
    if isinstance(tool_payload, dict):
        tool_name = str(
            tool_payload.get("name")
            or tool_payload.get("tool_name")
            or tool_payload.get("tool")
            or ""
        ).strip() or None
        tool_input = str(
            tool_payload.get("input")
            or tool_payload.get("query")
            or tool_payload.get("instruction")
            or ""
        ).strip() or None
    elif isinstance(tool_payload, str):
        tool_name = tool_payload.strip() or None
    if not tool_name:
        match = re.search(r"TOOL::([A-Za-z0-9_\-]+)", action)
        if match:
            tool_name = match.group(1)
    return tool_name, tool_input


def _format_turns(turns: Sequence[ManusTurn]) -> str:
    if not turns:
        return ""
    parts = []
    for idx, turn in enumerate(turns, 1):
        tool_segment = ""
        if turn.tool_name:
            tool_segment = f"; tool={turn.tool_name}"
            if turn.tool_output:
                excerpt = turn.tool_output.strip()
                if len(excerpt) > 120:
                    excerpt = f"{excerpt[:117]}..."
                tool_segment += f" ({excerpt})"
        parts.append(
            f"Turn {idx}: thought={turn.thought}; action={turn.action}; observation={turn.observation}; leads={', '.join(turn.leads) or 'none'}"
            f"; status={turn.status}{tool_segment}"
        )
    return "\n".join(parts)


def _emit(observer: Optional[StageObserver], stage: str, payload: dict[str, object]) -> None:
    if observer:
        observer(stage, payload)


def _maybe_execute_tool(
    turn: ManusTurn,
    registry: ToolRegistry,
    observer: Optional[StageObserver],
) -> ManusTurn:
    if not turn.tool_name:
        return turn
    instruction = turn.tool_input or ""
    try:
        result = registry.run(turn.tool_name, instruction)
    except KeyError:
        message = f"Tool '{turn.tool_name}' is unavailable."
        turn.tool_output = message
        turn.observation = _append_observation(turn.observation, message)
        _emit(observer, "tool", {"name": turn.tool_name, "input": instruction, "error": message})
        return turn
    except ToolExecutionError as exc:
        message = str(exc)
        turn.tool_output = message
        turn.observation = _append_observation(turn.observation, message)
        _emit(observer, "tool", {"name": turn.tool_name, "input": instruction, "error": message})
        return turn
    turn.tool_output = result.output
    turn.observation = _append_observation(
        turn.observation, f"Tool output ({turn.tool_name}): {result.output}"
    )
    tool_payload: dict[str, object] = {"name": turn.tool_name, "input": instruction, "output": result.output}
    if result.metadata:
        tool_payload["metadata"] = dict(result.metadata)
    _emit(observer, "tool", tool_payload)
    return turn


def _append_observation(existing: str, addition: str) -> str:
    existing = existing.strip()
    addition = addition.strip()
    if not existing:
        return addition
    if not addition:
        return existing
    return f"{existing}\n{addition}"
