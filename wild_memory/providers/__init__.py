"""
Wild Memory providers — pluggable LLM and embedding backends.

The framework never imports vendor SDKs directly. The orchestrator and infra
layers operate on the `LLMProvider` and `EmbeddingProvider` Protocols defined
in `wild_memory.providers.base`. Bundled adapters for Anthropic and OpenAI
live alongside; users can implement and pass their own.
"""

from wild_memory.providers.base import (
    EmbeddingProvider,
    LLMProvider,
    LLMResponse,
    ToolCall,
    ToolResult,
)
from wild_memory.providers.anthropic_llm import AnthropicLLM
from wild_memory.providers.openai_embedding import OpenAIEmbedding

__all__ = [
    "EmbeddingProvider",
    "LLMProvider",
    "LLMResponse",
    "ToolCall",
    "ToolResult",
    "AnthropicLLM",
    "OpenAIEmbedding",
]
