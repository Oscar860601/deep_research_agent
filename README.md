# Deep Research Agent Framework

This repository contains a lightweight yet extensible agent framework inspired by
[SkyworkAI/DeepResearchAgent](https://github.com/SkyworkAI/DeepResearchAgent),
[InternLM/lagent](https://github.com/InternLM/lagent), the OpenHand ecosystem, and the
OpenManus/OWL interaction style. The goal is to provide a batteries-included research
agent loop without relying on heavy orchestration libraries (e.g. LangChain Agents,
HuggingFace smolagents).

## Key Features

- **Minimal agent core** – `dra.agent.Agent` implements an iterative loop with
  configurable reflection intervals and early stopping when a final answer is
  produced.
- **Flexible memory** – `dra.memory.Memory` keeps conversation state while
  remaining serialisation-friendly for API calls.
- **Pluggable LLM backends** – Use the OpenAI SDK, Azure OpenAI, a LangChain chat
  model, or the offline echo client for testing.
- **Deep research specialisation** – `dra.research.DeepResearchAgent` now mirrors the
  Skywork DeepResearch outer loop with Manus/OWL stages: mission analysis, plan
  drafting, Manus-style action/observation loops, synthesis, and a final reporting
  pass that leverages the configurable base agent system prompt.
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

### Default Deep Research prompt

The base agent ships with a mission prompt that blends the Skywork DeepResearch
outer loop, the InternLM/lagent structure for disciplined actions, and the
OpenHand / OpenManus OWL interaction model. It instructs the model to:

1. Capture a **Mission Intelligence** brief covering scope, stakeholders, and
   knowledge gaps.
2. Draft an **Operation Plan** that sequences Observe/Work/Learn phases with
   Manus-style actions.
3. Run **OpenManus OWL execution turns** that log Thought, Action, Observation,
   Leads, and Status for every step.
4. Perform **lagent-style cross-checks** to reconcile conflicts and note
   confidence levels.
5. Deliver a structured report with Mission Overview, Key Findings (with
   confidence tags), Evidence Trail, Gaps & Next Steps, and a `FINAL ANSWER`.

You can swap this prompt for another `SystemPrompt` at runtime, but the default
is designed to closely mimic the prompts published in the Skywork, InternLM, and
OpenHand/OpenManus projects.

### CLI

```bash
python -m dra.cli "Research the latest advancements in quantum error correction" \
    --mock
```

The CLI streams each stage (analysis, plan, Manus turns, synthesis) to stdout so you
can follow the workflow. Use `--silent-stages` to suppress the intermediate artefacts.
Replace `--mock` with either `--openai-model gpt-4o-mini` or `--langchain <model>`
if you have the respective dependencies installed and API credentials configured.

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

def log_stage(stage: str, payload: dict[str, object]) -> None:
    print(f"[{stage}]", payload)

result = agent.run(
    "Map out the competitive landscape for edge AI accelerators",
    context=["Focus on 2023-2024 announcements"],
    observer=log_stage,
)
print(result)
```

### Azure OpenAI client

```python
from openai import AzureOpenAI

from dra import AzureOpenAIClient

raw_client = AzureOpenAI(
    api_key="<api-key>",
    api_version="2024-02-01",
    azure_endpoint="https://my-azure-resource.openai.azure.com",
)

client = AzureOpenAIClient(
    client=raw_client,
    model="gpt-4o-mini",
    endpoint_url="https://custom-endpoint.openai.azure.com",
    extra_headers={"x-ms-azureai-solution": "deep-research"},
)

agent = create_default_deep_research_agent(client)
print(agent.run("Summarise the state of liquid cooling for data centers"))
```

`endpoint_url` lets you override the base URL on a per-call basis (useful when the
client is shared by multiple deployments), while `extra_headers` is passed through
to `chat.completions.create` so you can meet Azure's tracing or policy
requirements.

## Project Structure

```
dra/
├── __init__.py
├── agent.py          # Core iteration loop
├── cli.py            # Command line entry point
├── llm.py            # LLM client abstractions (OpenAI, Azure OpenAI, LangChain, Echo)
├── memory.py         # Message and memory primitives
├── prompts.py        # System prompt utilities
└── research.py       # Deep research specialisation
```

## License

MIT
