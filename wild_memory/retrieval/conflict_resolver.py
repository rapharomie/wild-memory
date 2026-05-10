"""
Conflict Resolver.

Two-phase conflict detection: embedding search first (no LLM), LLM
classification only when similar observations are found. Saves ~70% of
LLM calls on the distillation hot path.
"""
from __future__ import annotations

import json

from wild_memory.config import ConflictConfig
from wild_memory.models import ConflictAction, ConflictResult
from wild_memory.prompts.conflict import CONFLICT_PROMPT
from wild_memory.store.base import MemoryStore


class ConflictResolver:
    def __init__(
        self,
        store: MemoryStore,
        router,
        embedding_cache,
        config: ConflictConfig,
    ):
        self.store = store
        self.router = router
        self.embedding_cache = embedding_cache
        self.threshold = config.similarity_threshold

    async def check(
        self,
        agent_id: str,
        user_id: str,
        embedding: list,
        obs_data: dict,
    ) -> ConflictResult:
        """Two-phase conflict check."""
        # Phase 1: embedding similarity (no LLM).
        similar = await self.store.find_similar_observations(
            agent_id=agent_id,
            user_id=user_id,
            embedding=embedding,
            threshold=self.threshold,
            limit=3,
        )
        if not similar:
            return ConflictResult(
                action=ConflictAction.ADD,
                reason="no_similar_found",
                llm_called=False,
            )

        # Phase 2: LLM classification.
        similar_payload = [s.raw or {"id": s.id, "content": s.content} for s in similar]
        result = await self.router.call(
            task="conflict_resolution",
            system=CONFLICT_PROMPT.format(
                new_observation=json.dumps(obs_data, default=str),
                existing_observations=json.dumps(similar_payload, default=str),
            ),
            messages=[{"role": "user", "content": "Classify."}],
            max_tokens=200,
        )
        try:
            text = result.text
            data = json.loads(text.strip().strip("`").strip("json").strip())
            return ConflictResult(
                action=ConflictAction(data.get("action", "ADD")),
                existing_id=data.get("existing_id"),
                reason=data.get("reason", ""),
                llm_called=True,
            )
        except (json.JSONDecodeError, ValueError):
            return ConflictResult(
                action=ConflictAction.ADD,
                reason="parse_error",
                llm_called=True,
            )
