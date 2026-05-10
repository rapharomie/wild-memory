"""
Observation Layer.

Core CRUD plus retrieval for distilled knowledge units. Each observation
is a compact, self-descriptive fact with metadata: type, importance,
emotion, temporal info, entities. Storage is delegated entirely to the
configured `MemoryStore`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from wild_memory.models import ConflictAction, ConflictResult, Observation
from wild_memory.store.base import MemoryStore, RetrievalWeights


class ObservationLayer:
    """CRUD operations for observations."""

    def __init__(self, store: MemoryStore, embedding_cache, router, ner, config):
        self.store = store
        self.embedding_cache = embedding_cache
        self.router = router
        self.ner = ner
        self.config = config

    async def save(
        self, obs: Observation, conflict: Optional[ConflictResult] = None
    ) -> str:
        """Persist an observation. Returns the new ID."""
        data = {
            "agent_id": obs.agent_id,
            "user_id": obs.user_id,
            "content": obs.content,
            "obs_type": obs.obs_type.value,
            "entities": obs.entities,
            "importance": obs.importance,
            "decay_score": obs.decay_score,
            "ttl_days": self._get_ttl(obs.obs_type.value),
            "emotional_valence": obs.emotional_valence.value,
            "emotional_intensity": obs.emotional_intensity,
            "privacy_mode": obs.privacy_mode.value,
            "event_time": obs.event_time.isoformat() if obs.event_time else None,
            "source_session": obs.source_session,
            "embedding": obs.embedding,
        }
        new_id = await self.store.insert_observation(data)

        # If superseding, invalidate the old observation.
        if (
            conflict
            and conflict.action == ConflictAction.SUPERSEDE
            and conflict.existing_id
        ):
            await self.store.update_observation(
                conflict.existing_id,
                {
                    "invalidated_at": datetime.now(timezone.utc).isoformat(),
                    "invalidated_by": new_id,
                },
            )

        return new_id

    async def retrieve(
        self,
        agent_id: str,
        user_id: str,
        goal: str,
        entities: list[str],
        search_query: str = "",
        limit: int = 10,
        min_decay: float = 0.3,
    ) -> list[dict]:
        """Retrieve observations using the 5-signal combined score."""
        emb = await self.embedding_cache.embed(goal)
        weights = self._weights_from_config()
        results = await self.store.retrieve_observations(
            agent_id=agent_id,
            user_id=user_id,
            embedding=emb,
            entities=entities,
            search_query=search_query,
            limit=limit,
            min_decay=min_decay,
            weights=weights,
        )
        # Return raw rows (callers expect dict-shape compatibility for now).
        return [r.raw or {"id": r.id, "content": r.content} for r in results]

    async def find_similar(
        self,
        agent_id: str,
        user_id: str,
        embedding: list,
        threshold: float = 0.85,
        limit: int = 3,
    ) -> list[dict]:
        """Find similar observations for conflict detection."""
        results = await self.store.find_similar_observations(
            agent_id=agent_id,
            user_id=user_id,
            embedding=embedding,
            threshold=threshold,
            limit=limit,
        )
        return [r.raw or {"id": r.id, "content": r.content} for r in results]

    def _get_ttl(self, obs_type: str) -> int:
        defaults = self.config.ttl_defaults
        return getattr(defaults, obs_type, 90)

    def _weights_from_config(self) -> RetrievalWeights:
        w = self.config.retrieval_weights
        return RetrievalWeights(
            semantic=w.semantic,
            entity_match=w.entity_match,
            fts_keyword=w.fts_keyword,
            recency=w.recency,
            decay=w.decay,
            emotion=w.emotion,
        )
