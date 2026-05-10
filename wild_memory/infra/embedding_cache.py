"""
Per-turn embedding cache.

Wraps an `EmbeddingProvider` to deduplicate calls within a single turn — the
same text typically gets embedded by the semantic cache, recall, distiller,
and conflict resolver. The cache lives for one turn; call `clear_turn()` at
the end of each turn.
"""
from __future__ import annotations

import hashlib

from wild_memory.providers.base import EmbeddingProvider


class EmbeddingCache:
    """Per-turn embedding cache around an EmbeddingProvider."""

    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider
        self._cache: dict[str, list[float]] = {}
        self._hits = 0
        self._misses = 0

    @property
    def dimensions(self) -> int:
        return self.provider.dimensions

    async def embed(self, text: str) -> list[float]:
        key = hashlib.md5(text.strip().lower().encode()).hexdigest()
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        emb = await self.provider.embed(text)
        self._cache[key] = emb
        return emb

    def clear_turn(self) -> None:
        self._cache.clear()

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(total, 1),
        }
