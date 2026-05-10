"""
Semantic Cache.

Caches responses for semantically similar queries. Saves 30-50% of tokens
on frequently-asked questions. Personal queries (configurable keyword
list) bypass the cache to avoid leaking one user's response into another's.
"""
from __future__ import annotations

from typing import Optional

from wild_memory.config import CacheConfig
from wild_memory.store.base import MemoryStore


class SemanticCache:
    def __init__(self, store: MemoryStore, embedding_cache, config: CacheConfig):
        self.store = store
        self.embedding_cache = embedding_cache
        self.config = config

    async def check(self, agent_id: str, query: str) -> Optional[str]:
        if not self.config.enabled or self._is_personal(query):
            return None
        emb = await self.embedding_cache.embed(query)
        return await self.check_with_embedding(agent_id, emb)

    async def check_with_embedding(
        self, agent_id: str, embedding: list[float]
    ) -> Optional[str]:
        if not self.config.enabled:
            return None
        hit = await self.store.search_semantic_cache(
            agent_id=agent_id,
            embedding=embedding,
            threshold=self.config.similarity_threshold,
        )
        if not hit:
            return None
        await self.store.increment_semantic_cache_hit(hit["id"])
        return hit["response_text"]

    async def store_response(
        self, agent_id: str, query: str, response: str
    ) -> None:
        if not self.config.enabled or self._is_personal(query):
            return
        emb = await self.embedding_cache.embed(query)
        await self.store.insert_semantic_cache(
            agent_id=agent_id,
            query_text=query,
            response_text=response,
            embedding=emb,
            ttl_hours=self.config.ttl_hours,
        )

    async def cleanup_expired(self) -> int:
        return await self.store.cleanup_expired_cache()

    def _is_personal(self, query: str) -> bool:
        q = query.lower()
        return any(kw in q for kw in self.config.personal_keywords)
