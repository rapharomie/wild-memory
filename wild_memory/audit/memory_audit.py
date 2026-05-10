"""
Memory Audit.

Human-facing inspection, correction, and privacy operations (GDPR /
right-to-be-forgotten support).
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from wild_memory.store.base import MemoryStore


class MemoryAudit:
    def __init__(self, store: MemoryStore):
        self.store = store

    async def get_user_memory(self, agent_id: str, user_id: str) -> dict:
        observations = await self.store.list_observations(
            agent_id=agent_id, user_id=user_id, status="active", limit=500
        )
        reflections = await self.store.list_reflections(
            agent_id=agent_id, user_id=user_id, limit=200
        )
        return {
            "user_id": user_id,
            "observations": observations,
            "reflections": reflections,
            "stats": {
                "total_observations": len(observations),
                "by_type": self._count_by_type(observations),
            },
        }

    async def correct_observation(
        self, obs_id: str, correction: str, auditor: str
    ) -> str:
        """Correct an observation: insert a new corrected version, archive the old."""
        old = await self.store.get_observation(obs_id)
        if not old:
            return "Not found"

        new_data = {**old}
        new_data.pop("id", None)
        new_data["content"] = correction
        new_data["obs_type"] = "correction"
        new_data["importance"] = max(old.get("importance", 5), 7)
        new_data["decay_score"] = 1.0
        new_data["source_session"] = f"audit_{auditor}"
        new_id = await self.store.insert_observation(new_data)

        await self.store.update_observation(
            obs_id,
            {
                "invalidated_at": datetime.now(timezone.utc).isoformat(),
                "invalidated_by": new_id,
                "status": "archived",
            },
        )
        return new_id

    async def purge_user_data(
        self, user_id: str, keep_patterns: bool = True
    ) -> None:
        """GDPR/LGPD: right to be forgotten.

        With `keep_patterns=True` we anonymize observations (keep aggregate
        learnings); otherwise we mark them purged. Feedback signals and
        reflections are always deleted.
        """
        if keep_patterns:
            user_hash = hashlib.sha256(user_id.encode()).hexdigest()[:16]
            await self.store.anonymize_user_observations(user_id, user_hash)
        else:
            await self.store.purge_user_observations(user_id)
        await self.store.purge_user_feedback(user_id)
        await self.store.purge_user_reflections(user_id)

    @staticmethod
    def _count_by_type(obs: list[dict]) -> dict:
        counts: dict = {}
        for o in obs:
            t = o.get("obs_type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return counts
