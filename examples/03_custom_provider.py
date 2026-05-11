"""
Plug a custom LLM into Wild Memory.

The framework only depends on the `LLMProvider` Protocol; vendor SDKs are
an implementation detail. This example wires a trivial echo provider — the
same shape works for OpenAI, Mistral, vLLM, Ollama, etc. Anything that
implements `complete()` returning an `LLMResponse` slots in.

Usage:
    pip install "wild-memory[sqlite]"
    python examples/03_custom_provider.py
"""
from __future__ import annotations

import asyncio
from typing import Optional

from wild_memory import WildMemory, WildMemoryConfig
from wild_memory.providers.base import LLMResponse, ToolResult
from wild_memory.providers.openai_embedding import OpenAIEmbedding
from wild_memory.store import SQLiteStore


class EchoLLM:
    """A minimal LLMProvider that echoes the last user message back.

    Useful for local development without a real LLM bill. Implements only
    what the orchestrator needs: `complete()` and `format_tool_results()`.
    """

    name = "echo"

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        last = ""
        for m in reversed(messages):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                last = m["content"]
                break
        return LLMResponse(text=f"(echo) {last}", stop_reason="end")

    def format_tool_results(self, results: list[ToolResult]) -> dict:
        return {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": r.tool_call_id,
                 "content": r.content, "is_error": r.is_error}
                for r in results
            ],
        }


async def main():
    store = SQLiteStore(":memory:", embedding_dim=1536)
    await store.connect()
    await store.migrate(embedding_dim=1536)

    memory = WildMemory(
        WildMemoryConfig(),
        store=store,
        llm=EchoLLM(),
        embedding=OpenAIEmbedding(dimensions=1536),  # could also be custom
    )
    reply = await memory.process_message(
        agent_id="demo", user_id="u", session_id="s", message="hello there",
    )
    print(reply)  # → "(echo) hello there"
    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
