"""Command line entry-point for the Deep Research Agent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .agent import AgentConfig
from .llm import EchoClient, LangChainClient, OpenAIClient
from .prompts import SystemPrompt, DEFAULT_RESEARCH_PROMPT
from .research import create_default_deep_research_agent


def load_system_prompt(path: Optional[str]) -> SystemPrompt:
    if path is None:
        return DEFAULT_RESEARCH_PROMPT
    template = Path(path).read_text(encoding="utf-8")
    return SystemPrompt(template=template)


def build_llm_client(args: argparse.Namespace):
    if args.mock:
        return EchoClient()

    if args.langchain:
        from langchain.chat_models import init_chat_model  # type: ignore

        return LangChainClient(chat_model=init_chat_model(args.langchain))

    if args.openai_model:
        from openai import OpenAI  # type: ignore

        return OpenAIClient(client=OpenAI(), model=args.openai_model)

    raise SystemExit("No LLM backend selected. Use --mock, --langchain, or --openai-model.")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Deep Research Agent CLI")
    parser.add_argument("task", help="User task to research")
    parser.add_argument("--context", help="Optional JSON file containing context array")
    parser.add_argument("--system-prompt", help="Path to a custom system prompt template")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--reflection-interval", type=int, default=2)
    parser.add_argument(
        "--silent-stages",
        action="store_true",
        help="Disable printing of intermediate Skywork/Manus workflow artefacts",
    )
    parser.add_argument(
        "--review-plan",
        action="store_true",
        help="Pause after planning so you can iteratively edit the plan via stdin",
    )

    llm_group = parser.add_mutually_exclusive_group(required=False)
    llm_group.add_argument("--mock", action="store_true", help="Use the offline echo client")
    llm_group.add_argument("--langchain", help="LangChain model identifier")
    llm_group.add_argument("--openai-model", help="OpenAI chat completion model name")

    args = parser.parse_args(argv)

    context_data = None
    if args.context:
        context_data = json.loads(Path(args.context).read_text(encoding="utf-8"))
        if not isinstance(context_data, list):
            raise SystemExit("Context JSON must be an array of strings")

    llm_client = build_llm_client(args)
    system_prompt = load_system_prompt(args.system_prompt)

    base_agent_config = AgentConfig(
        max_iterations=args.max_iterations,
        reflection_interval=args.reflection_interval,
    )

    base_agent = create_default_deep_research_agent(llm_client)
    base_agent.base_agent.system_prompt = system_prompt
    base_agent.base_agent.config = base_agent_config

    observer = None
    if not args.silent_stages:

        def observer(stage: str, payload: dict[str, object]) -> None:
            print(f"[{stage.upper()}]")
            print(json.dumps(payload, ensure_ascii=False, indent=2))

    plan_reviewer = None
    if args.review_plan:

        def plan_reviewer(steps: list[str], iteration: int) -> Optional[str]:
            print("\n[PLAN REVIEW]")
            if steps:
                for idx, step in enumerate(steps, 1):
                    print(f"{idx}. {step}")
            else:
                print("(planner returned no structured steps; provide guidance below)")
            prompt = "Provide revised instructions (blank to accept): "
            try:
                feedback = input(prompt)
            except EOFError:
                return None
            return feedback.strip() or None

    result = base_agent.run(
        args.task,
        context=context_data,
        observer=observer,
        plan_reviewer=plan_reviewer,
    )
    print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
