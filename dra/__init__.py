"""Deep Research Agent framework."""

from dotenv import load_dotenv

# Load environment variables from a local .env if present so API keys and
# endpoints are available before any client initialisation happens. This keeps
# CLI usage, library imports, and tests aligned with the documented setup flow.
load_dotenv()

from .agent import Agent
from .memory import Memory
from .llm import LLMClient, OpenAIClient, AzureOpenAIClient, LangChainClient
from .prompts import SystemPrompt, DEFAULT_RESEARCH_PROMPT
from .research import (
    DeepResearchAgent,
    ManusTurn,
    ResearchState,
    create_default_deep_research_agent,
)
from .tools import (
    CallableTool,
    NotebookTool,
    Tool,
    ToolExecutionError,
    ToolRegistry,
    ToolResult,
    WebPageTool,
    WebSearchTool,
    create_default_toolbox,
)

__all__ = [
    "Agent",
    "Memory",
    "LLMClient",
    "OpenAIClient",
    "AzureOpenAIClient",
    "LangChainClient",
    "SystemPrompt",
    "DEFAULT_RESEARCH_PROMPT",
    "DeepResearchAgent",
    "ManusTurn",
    "ResearchState",
    "create_default_deep_research_agent",
    "Tool",
    "ToolExecutionError",
    "ToolResult",
    "ToolRegistry",
    "CallableTool",
    "WebSearchTool",
    "WebPageTool",
    "NotebookTool",
    "create_default_toolbox",
]
