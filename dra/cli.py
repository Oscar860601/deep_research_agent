"""Command line entry-point for the Deep Research Agent."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from .agent import AgentConfig
from .llm import EchoClient, LangChainClient, OpenAIClient
from .prompts import SystemPrompt, DEFAULT_RESEARCH_PROMPT
from .research import create_default_deep_research_agent


LOGGER = logging.getLogger(__name__)


def configure_logging(level_name: str, log_file: Optional[str]) -> None:
    """Configure application-wide logging early in the CLI lifecycle."""

    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise SystemExit(f"Invalid log level: {level_name}")

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=handlers,
        force=True,
    )
    LOGGER.debug("Logging configured: level=%s, log_file=%s", level_name.upper(), log_file)


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
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--log-file",
        help="Optional path to a log file for capturing verbose traces",
    )

    llm_group = parser.add_mutually_exclusive_group(required=False)
    llm_group.add_argument("--mock", action="store_true", help="Use the offline echo client")
    llm_group.add_argument("--langchain", help="LangChain model identifier")
    llm_group.add_argument("--openai-model", help="OpenAI chat completion model name")

    args = parser.parse_args(argv)
    configure_logging(args.log_level, args.log_file)
    LOGGER.info("Starting Deep Research Agent CLI")

    context_data = None
    if args.context:
        context_data = json.loads(Path(args.context).read_text(encoding="utf-8"))
        if not isinstance(context_data, list):
            raise SystemExit("Context JSON must be an array of strings")
        LOGGER.info("Loaded %d context entries from %s", len(context_data), args.context)

    llm_client = build_llm_client(args)
    LOGGER.info("Selected LLM backend: %s", llm_client.__class__.__name__)
    system_prompt = load_system_prompt(args.system_prompt)
    if args.system_prompt:
        LOGGER.info("Using custom system prompt from %s", args.system_prompt)

    base_agent_config = AgentConfig(
        max_iterations=args.max_iterations,
        reflection_interval=args.reflection_interval,
    )
    LOGGER.debug(
        "Configured base agent: iterations=%d, reflection_interval=%d",
        base_agent_config.max_iterations,
        base_agent_config.reflection_interval,
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

    LOGGER.info("Executing task: %s", args.task)
    result = base_agent.run(
        args.task,
        context=context_data,
        observer=observer,
        plan_reviewer=plan_reviewer,
    )
    LOGGER.info("Task completed")
    print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
