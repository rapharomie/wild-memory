"""
Provider abstraction for LLMs and embeddings.

The orchestrator talks to a `LLMProvider` Protocol and an `EmbeddingProvider`
Protocol. Vendor adapters translate between the vendor's native response shape
and the framework's neutral dataclasses. This keeps the rest of the codebase
free of any single-vendor coupling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    """The result of executing a tool, returned to the model on the next turn."""

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class LLMResponse:
    """Provider-neutral LLM response.

    `tool_calls` is non-empty iff `stop_reason == "tool_use"`. `raw` exposes the
    vendor-native object for debugging only — core framework code never reads it.
    """

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end"  # "end" | "tool_use" | "max_tokens"
    raw: Any = None


@runtime_checkable
class LLMProvider(Protocol):
    """An LLM backend. Async-callable and tool-call aware."""

    name: str

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Call the LLM and translate its response into LLMResponse."""
        ...

    def format_tool_results(self, results: list[ToolResult]) -> dict:
        """Translate a batch of tool results into a vendor-shaped message
        suitable for appending to the conversation history before the next
        LLM call."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """An embedding backend."""

    name: str
    dimensions: int

    async def embed(self, text: str) -> list[float]:
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...
