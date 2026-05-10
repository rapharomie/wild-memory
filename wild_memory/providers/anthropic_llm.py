"""AnthropicLLM — adapter from the Anthropic API to the framework's LLMProvider."""
from __future__ import annotations

from typing import Any

from wild_memory.providers.base import LLMResponse, ToolCall, ToolResult


class AnthropicLLM:
    """Async Anthropic adapter. Uses `anthropic.AsyncAnthropic` under the hood."""

    name = "anthropic"

    def __init__(self, *, api_key: str | None = None, client: Any = None):
        if client is not None:
            self._client = client
        else:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=api_key) if api_key else AsyncAnthropic()

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        resp = await self._client.messages.create(**kwargs)
        return self._translate(resp)

    def format_tool_results(self, results: list[ToolResult]) -> dict:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": r.tool_call_id,
                    "content": r.content,
                    "is_error": r.is_error,
                }
                for r in results
            ],
        }

    @staticmethod
    def _translate(resp: Any) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )
        raw_stop = getattr(resp, "stop_reason", None)
        if raw_stop == "tool_use":
            stop = "tool_use"
        elif raw_stop == "max_tokens":
            stop = "max_tokens"
        else:
            stop = "end"
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=stop,
            raw=resp,
        )
