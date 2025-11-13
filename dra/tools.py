"""Tool primitives and built-in utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
import io
import textwrap
from typing import Callable, Mapping, MutableMapping, Protocol, Sequence
from urllib.error import URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import contextlib
import math
import statistics


@dataclass
class ToolResult:
    """Represents the structured output from a tool invocation."""

    name: str
    input: str
    output: str
    metadata: Mapping[str, object] | None = None


class Tool(Protocol):
    """Protocol that every tool implementation must follow."""

    name: str
    description: str

    def run(self, instruction: str) -> ToolResult:
        ...


@dataclass
class ToolExecutionError(RuntimeError):
    """Raised when a tool fails to execute."""

    tool_name: str
    instruction: str
    original: Exception

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"Tool '{self.tool_name}' failed for input '{self.instruction}': {self.original}"


@dataclass
class ToolRegistry:
    """Keeps track of the available tools for the research workflow."""

    tools: MutableMapping[str, Tool] = field(default_factory=dict)

    def __init__(self, tools: Sequence[Tool] | None = None):
        self.tools = {tool.name: tool for tool in tools or []}

    def add(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def run(self, name: str, instruction: str) -> ToolResult:
        tool = self.get(name)
        if not tool:
            raise KeyError(f"Tool '{name}' is not registered")
        return tool.run(instruction)

    def describe(self) -> list[str]:
        descriptions = []
        for tool in self.tools.values():
            descriptions.append(f"{tool.name}: {tool.description}")
        return descriptions


@dataclass
class CallableTool:
    """Adapter so users can quickly wrap a Python callable as a Tool."""

    name: str
    description: str
    func: Callable[[str], ToolResult | str]

    def run(self, instruction: str) -> ToolResult:
        result = self.func(instruction)
        if isinstance(result, ToolResult):
            return result
        return ToolResult(name=self.name, input=instruction, output=str(result))


class WebSearchTool:
    """Performs a lightweight DuckDuckGo HTML search."""

    name = "web_search"
    description = "DuckDuckGo web search returning the top textual hits."

    def __init__(self, *, max_results: int = 5, user_agent: str = "DeepResearchAgent/0.1", timeout: int = 10):
        self.max_results = max_results
        self.user_agent = user_agent
        self.timeout = timeout

    def run(self, instruction: str) -> ToolResult:
        query = instruction.strip()
        if not query:
            raise ToolExecutionError(self.name, instruction, ValueError("Search query is empty"))
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        request = Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                html = response.read().decode("utf-8", errors="ignore")
        except URLError as exc:  # pragma: no cover - network dependent
            raise ToolExecutionError(self.name, instruction, exc)
        results = _extract_duckduckgo_results(html)
        if not results:
            output = "No search results were returned."
        else:
            lines = []
            for idx, result in enumerate(results[: self.max_results], 1):
                lines.append(f"{idx}. {result['title']} – {result['url']}")
                if result.get("snippet"):
                    lines.append(f"   {result['snippet']}")
            output = "\n".join(lines)
        return ToolResult(name=self.name, input=query, output=output)


class WebPageTool:
    """Fetches and summarises the readable text from a URL."""

    name = "web_page"
    description = "Download and extract text from a URL (first 2k characters)."

    def __init__(self, *, max_chars: int = 2000, user_agent: str = "DeepResearchAgent/0.1", timeout: int = 15):
        self.max_chars = max_chars
        self.user_agent = user_agent
        self.timeout = timeout

    def run(self, instruction: str) -> ToolResult:
        target = instruction.strip()
        if not target:
            raise ToolExecutionError(self.name, instruction, ValueError("URL is empty"))
        request = Request(target, headers={"User-Agent": self.user_agent})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                html = response.read().decode("utf-8", errors="ignore")
        except URLError as exc:  # pragma: no cover - network dependent
            raise ToolExecutionError(self.name, instruction, exc)
        text = _extract_text_from_html(html)
        snippet = textwrap.shorten(text, width=self.max_chars, placeholder=" …") if text else "No readable text detected."
        output = f"URL: {target}\nSnippet:\n{snippet}"
        return ToolResult(name=self.name, input=target, output=output)


class NotebookTool:
    """Executes short Python snippets for calculations or parsing."""

    name = "notebook"
    description = "Run Python code (math, stats, parsing). Capture stdout + last value."

    def __init__(self, *, extra_globals: Mapping[str, object] | None = None):
        safe_builtins = {
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "range": range,
            "round": round,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
        }
        base_globals: dict[str, object] = {
            "__builtins__": safe_builtins,
            "math": math,
            "statistics": statistics,
        }
        if extra_globals:
            base_globals.update(extra_globals)
        self._base_globals = base_globals

    def run(self, instruction: str) -> ToolResult:
        code = instruction.strip()
        if not code:
            raise ToolExecutionError(self.name, instruction, ValueError("Notebook cell is empty"))
        globals_dict: dict[str, object] = dict(self._base_globals)
        locals_dict = globals_dict
        buffer = io.StringIO()
        last_value: object | None = None
        try:
            compiled = compile(code, "<notebook>", "exec")
            with contextlib.redirect_stdout(buffer):
                exec(compiled, globals_dict, locals_dict)
            if "result" in locals_dict:
                last_value = locals_dict["result"]
        except Exception as exc:  # pragma: no cover - dynamic
            raise ToolExecutionError(self.name, instruction, exc)
        output = buffer.getvalue().strip()
        if last_value is not None:
            output = f"{output}\nresult={last_value}".strip()
        if not output:
            output = "Notebook executed with no stdout."
        return ToolResult(name=self.name, input=code, output=output)


def create_default_toolbox() -> list[Tool]:
    """Return the default set of tools that mirror Skywork/OpenManus setups."""

    return [WebSearchTool(), WebPageTool(), NotebookTool()]


class _DuckDuckGoResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._capture_title = False
        self._capture_snippet = False
        self._current: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {name: (value or "") for name, value in attrs}
        class_attr = attr_map.get("class", "")
        if tag == "a" and "result__a" in class_attr:
            if self._current:
                self._commit_current()
            self._current = {"url": attr_map.get("href", ""), "title": "", "snippet": ""}
            self._capture_title = True
        elif "result__snippet" in class_attr and tag in {"a", "div"} and self._current:
            self._capture_snippet = True

    def handle_data(self, data: str) -> None:
        if not self._current:
            return
        if self._capture_title:
            self._current["title"] += data
        elif self._capture_snippet:
            self._current["snippet"] += data

    def handle_endtag(self, tag: str) -> None:
        if self._capture_title and tag == "a":
            self._capture_title = False
        if self._capture_snippet and tag in {"a", "div"}:
            self._capture_snippet = False

    def close(self) -> None:
        super().close()
        self._commit_current()

    def _commit_current(self) -> None:
        if self._current and self._current.get("title"):
            result = {
                "title": _clean_html(self._current.get("title", "")),
                "url": self._current.get("url", ""),
                "snippet": _clean_html(self._current.get("snippet", "")),
            }
            self.results.append(result)
        self._current = None
        self._capture_title = False
        self._capture_snippet = False


def _extract_duckduckgo_results(html: str) -> list[dict[str, str]]:
    parser = _DuckDuckGoResultParser()
    parser.feed(html)
    parser.close()
    return parser.results


def _extract_text_from_html(html: str) -> str:
    class _TextExtractor(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.chunks: list[str] = []

        def handle_data(self, data: str) -> None:
            text = data.strip()
            if text:
                self.chunks.append(text)

    extractor = _TextExtractor()
    extractor.feed(html)
    extractor.close()
    return " ".join(extractor.chunks)


def _clean_html(value: str) -> str:
    cleaned = value.replace("\n", " ")
    cleaned = unescape(cleaned)
    return " ".join(cleaned.split())


__all__ = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "ToolExecutionError",
    "CallableTool",
    "WebSearchTool",
    "WebPageTool",
    "NotebookTool",
    "create_default_toolbox",
]
