# LLM and embedding providers

Wild Memory does not depend on any single vendor. The orchestrator only
sees two interfaces:

- `LLMProvider` — `complete(...)` returning a vendor-neutral `LLMResponse`,
  plus `format_tool_results(...)` for shaping tool replies.
- `EmbeddingProvider` — `embed(text)` and `embed_batch(texts)`.

Both live in `wild_memory/providers/base.py`.

## Bundled adapters

| Adapter | Module | Notes |
|---------|--------|-------|
| `AnthropicLLM` | `wild_memory.providers.anthropic_llm` | Wraps `anthropic.AsyncAnthropic`. Translates `anthropic.types.Message` → `LLMResponse(text, tool_calls, stop_reason)`. |
| `OpenAIEmbedding` | `wild_memory.providers.openai_embedding` | Wraps `openai.AsyncOpenAI` embeddings endpoint. Defaults to `text-embedding-3-small` at 1536 dims. |

Both read API keys from env vars (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`)
unless you pass them explicitly to the constructor.

## Plugging in a custom provider

```python
from wild_memory.providers.base import LLMResponse, ToolResult

class MyLLM:
    name = "my-llm"

    async def complete(self, *, model, system, messages,
                       tools=None, max_tokens=4096) -> LLMResponse:
        # Call your vendor / local model however it likes.
        # Return an LLMResponse — that's what the orchestrator sees.
        ...

    def format_tool_results(self, results: list[ToolResult]) -> dict:
        # Return a single message dict that the next call's `messages`
        # list can consume. The Anthropic-shape works for many vendors.
        return {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": r.tool_call_id,
             "content": r.content, "is_error": r.is_error}
            for r in results
        ]}
```

Pass the instance into the orchestrator:

```python
memory = WildMemory(config, store=store, llm=MyLLM())
```

A complete worked example is at `examples/03_custom_provider.py`.

## Tool calls

The framework speaks JSON Schema for tool definitions
(`wild_memory/tools.py`). Most modern LLMs accept this directly. For
vendors with a different format, translate inside your provider's
`complete()` before sending and translate the response back into
`ToolCall` objects.

## Two-tier routing

`ModelRouter.call(task=…)` picks between `config.models.premium` and
`config.models.economy`. The mapping is in
`wild_memory/infra/model_router.py:TASK_TIER_MAP`. To use the same model
for everything, just point both tiers at the same model name. To wire
two vendors (e.g. premium = Anthropic, economy = a local Ollama
instance), implement two providers and instantiate the router yourself
with whichever you want as the default — the orchestrator wraps a
single `LLMProvider`, so the cleanest path is one provider that itself
multiplexes.
