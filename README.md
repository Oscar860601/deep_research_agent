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
- **Human-in-the-loop planning** – Pause after the planning cell produces a draft,
  collect reviewer feedback, and regenerate plans through the same conversation
  window until everyone signs off.
- **Tool-aware Manus execution** – Every Manus turn can request real tools
  (web search, URL fetch, notebook) that mirror Skywork/OpenManus pipelines. Tool
  inputs/outputs are logged, folded back into the context blocks, and surfaced to
  observers so downstream systems can audit or replay the workflow.
- **Customisable prompts** – `dra.prompts.SystemPrompt` makes it easy to swap
  or templatise system prompts at runtime.
- **Command line interface** – `python -m dra.cli` offers a quick way to execute
  the agent with local configuration.

## Setup

1. **Create a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install the Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file mirrors the adapters supported by the framework
   (OpenAI, Azure OpenAI via the OpenAI SDK, and LangChain chat models) so a
   single command prepares every optional client used throughout the examples.

3. **Install the framework**

   ```bash
   pip install -e .
   ```

   This editable install hooks the package into your environment and pulls in
   the `python-dotenv` helper so environment variables from a `.env` file are
   automatically available everywhere the package is imported.

4. **Copy the `.env` template** and fill in at least one provider section:

   ```bash
   cp .env.example .env
   # edit .env with your keys, endpoints, and preferred model names
   ```

   The root module calls `dotenv.load_dotenv()` during import, so CLI usage,
   scripts, and tests all pick up the same credentials without extra plumbing.

### Environment variables

| Variable | Description |
| --- | --- |
| `OPENAI_API_KEY` | Standard OpenAI API key used by the SDK clients. |
| `OPENAI_API_BASE` | Optional override for non-default OpenAI base URLs. |
| `OPENAI_MODEL` | Convenience model name for scripts/CLI defaults. |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI resource key if you use Azure deployments. |
| `AZURE_OPENAI_ENDPOINT` | Resource endpoint, e.g. `https://<name>.openai.azure.com`. |
| `AZURE_OPENAI_API_VERSION` | API version string such as `2024-02-01`. |
| `AZURE_OPENAI_DEPLOYMENT` | Deployment/model name configured in Azure. |
| `LANGCHAIN_API_KEY` | Optional key for LangChain/LangSmith integrations. |
| `LANGCHAIN_TRACING_V2` | Set to `true`/`false` to control LangSmith tracing. |

You can add any custom variables needed by your own tooling; `python-dotenv`
ensures they propagate before the clients initialise.

## Usage

### Default Deep Research prompt

The default `SystemPrompt` encodes the full Skywork/Manus/OWL doctrine so the
model understands the entire mission contract without extra instructions. Key
sections include:

- **Workflow ladder** – Mission Intelligence, Hypothesis Board, Operation Plan,
  OWL execution, cross-checks, and synthesis. Each step spells out the required
  artifacts, minimum checkpoints, and how hypotheses tie to evidence types.
- **Manus turn schema + example** – Thought, `MANUS::{verb}` Action, Observation
  bullets with source handles, Leads, and Status (SUCCESS/PARTIAL/BLOCKED). The
  example teaches the tone, granularity, and follow-up expectations.
- **Reporting contract** – Confidence labels, transparency requirements, source
  citations, and how to document blockers or offline assumptions.
- **Final response template** – Mission Overview, Hypotheses & Status table,
  numbered Key Findings with citations and implications, Evidence Trail, Gaps &
  Next Steps, and a concluding `FINAL ANSWER` line.

You can always provide another `SystemPrompt`, but the built-in one mirrors the
multi-stage prompts used in production-grade deep research agents so the outer
loop works immediately.

### Toolbox

The Deep Research loop now ships with a default toolbox inspired by Skywork's
DeepResearch, OpenManus, and OWL deployments:

- `web_search` – Performs a DuckDuckGo HTML search and returns the top textual
  hits (title, URL, snippet) so Manus steps can seed reconnaissance queries.
- `web_page` – Fetches a URL and extracts the first ~2,000 readable characters to
  simulate OWL's browsing worker.
- `notebook` – Executes short Python snippets with math/statistics helpers for
  quick calculations, parsing, or scoring.

Each Manus JSON turn may include a `"tool"` object:

```json
{
  "thought": "Need hard numbers for 2023 shipments",
  "action": "MANUS::INVESTIGATE – query web for 'RISC-V MCU volume 2023'",
  "observation": "",
  "leads": ["Cross-check with IDC 2024 update"],
  "status": "continue",
  "tool": {"name": "web_search", "input": "RISC-V MCU volume 2023"}
}
```

The framework executes the tool, appends the output to the observation log, and
emits a `tool` stage event to any observer callback. Tool failures or missing
registrations are surfaced the same way so users can wire in retriers or alerts.

Custom tools can be registered via `dra.tools.CallableTool` or by passing a list
into `create_default_deep_research_agent`:

```python
from dra import CallableTool, create_default_deep_research_agent

def query_vector_store(instruction: str) -> str:
    return my_vectordb.similarity_search(instruction, k=3)

agent = create_default_deep_research_agent(
    client,
    tools=[CallableTool(name="kb_lookup", description="Vector DB", func=query_vector_store)],
)
```

Combine your custom list with `dra.tools.create_default_toolbox()` if you still
want the built-in trio of search/browse/notebook helpers.

### CLI

```bash
python -m dra.cli "Research the latest advancements in quantum error correction" \
    --mock
```

The CLI streams each stage (analysis, plan, Manus turns, tool executions, synthesis)
to stdout so you can follow the workflow. Use `--silent-stages` to suppress the
intermediate artefacts. Pass `--review-plan` if you want to pause after the planner
responds so you can iteratively type new guidance (blank input accepts the plan).
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

`DeepResearchAgent` exposes `generate_plan` and `execute_plan` so you can inspect or
modify the state between the planning and execution phases. Pass a
`plan_reviewer` callback to create a conversational approval loop for the plan, or
call `run` if you prefer a single turnkey entry point.

```python
from dra import Agent, AgentConfig, LangChainClient, SystemPrompt
from dra.research import create_default_deep_research_agent

# Use your preferred client; EchoClient is available for offline testing.
client = LangChainClient(chat_model=...)  # e.g. langchain.chat_models.init_chat_model("gpt-4o-mini")
agent = create_default_deep_research_agent(client)
agent.base_agent.config = AgentConfig(max_iterations=12, reflection_interval=3)

def log_stage(stage: str, payload: dict[str, object]) -> None:
    print(f"[{stage}]", payload)

state = agent.generate_plan(
    "Map out the competitive landscape for edge AI accelerators",
    context=["Focus on 2023-2024 announcements"],
    observer=log_stage,
)
final_report = agent.execute_plan(state, observer=log_stage)
print(final_report)

# Or execute everything in one call without inspecting the intermediate state.
# agent.run(...)


def plan_reviewer(steps: list[str], iteration: int) -> str | None:
    print(f"Reviewer loop #{iteration}: current plan -> {steps}")
    return input("Feedback (blank to accept): ") or None

state = agent.generate_plan(
    "Map out the competitive landscape for edge AI accelerators",
    context=["Focus on 2023-2024 announcements"],
    observer=log_stage,
    plan_reviewer=plan_reviewer,
)
print(agent.execute_plan(state))

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
