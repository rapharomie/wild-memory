"""OpenAIEmbedding — adapter from the OpenAI embeddings API to EmbeddingProvider."""
from __future__ import annotations

from typing import Any


class OpenAIEmbedding:
    """Async OpenAI embeddings adapter."""

    name = "openai"

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        api_key: str | None = None,
        client: Any = None,
    ):
        self.model = model
        self.dimensions = dimensions
        if client is not None:
            self._client = client
        else:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()

    async def embed(self, text: str) -> list[float]:
        resp = await self._client.embeddings.create(
            model=self.model, input=text, dimensions=self.dimensions
        )
        return resp.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(
            model=self.model, input=texts, dimensions=self.dimensions
        )
        return [d.embedding for d in resp.data]
