# Deep Research Agent Framework

This repository contains a lightweight yet extensible agent framework inspired by
[SkyworkAI/DeepResearchAgent](https://github.com/SkyworkAI/DeepResearchAgent),
[InternLM/lagent](https://github.com/InternLM/lagent), and the agent design in OpenHand.
The goal is to provide a batteries-included research agent loop without relying on
heavy orchestration libraries (e.g. LangChain Agents, HuggingFace smolagents).

## Key Features

- **Minimal agent core** – `dra.agent.Agent` implements an iterative loop with
  configurable reflection intervals and early stopping when a final answer is
  produced.
- **Flexible memory** – `dra.memory.Memory` keeps conversation state while
  remaining serialisation-friendly for API calls.
- **Pluggable LLM backends** – Use the OpenAI SDK, a LangChain chat model, or the
  offline echo client for testing.
- **Deep research specialisation** – `dra.research.DeepResearchAgent` adds a
  planning phase, scoring rubric, and context injection to emulate the
  Skywork-style deep research workflow.
- **Customisable prompts** – `dra.prompts.SystemPrompt` makes it easy to swap
  or templatise system prompts at runtime.
- **Command line interface** – `python -m dra.cli` offers a quick way to execute
  the agent with local configuration.

## Installation

```bash
pip install -e .
```

(Alternatively, add the `dra/` directory to your Python path.)

## Usage

### CLI

```bash
python -m dra.cli "Research the latest advancements in quantum error correction" \
    --mock
```

Replace `--mock` with either `--openai-model gpt-4o-mini` or `--langchain <model>`
if you have the respective dependencies installed and API credentials
configured.

To provide extra context or a custom prompt:

```bash
python -m dra.cli "Explain the impact of transformer architectures" \
    --context notes.json \
    --system-prompt prompts/custom_system.txt
```

`notes.json` should contain a JSON array of strings that will be appended to the
conversation context.

### Python

```python
from dra import Agent, AgentConfig, LangChainClient, SystemPrompt
from dra.research import create_default_deep_research_agent

# Use your preferred client; EchoClient is available for offline testing.
client = LangChainClient(chat_model=...)  # e.g. langchain.chat_models.init_chat_model("gpt-4o-mini")
agent = create_default_deep_research_agent(client)
agent.base_agent.config = AgentConfig(max_iterations=12, reflection_interval=3)

result = agent.run("Map out the competitive landscape for edge AI accelerators")
print(result)
```

## Project Structure

```
dra/
├── __init__.py
├── agent.py          # Core iteration loop
├── cli.py            # Command line entry point
├── llm.py            # LLM client abstractions (OpenAI, LangChain, Echo)
├── memory.py         # Message and memory primitives
├── prompts.py        # System prompt utilities
└── research.py       # Deep research specialisation
```

## License

MIT
