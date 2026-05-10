"""
Reflection Layer.

Analyzes observations to extract patterns and generate insights.
Runs in nightly cron.
"""
from __future__ import annotations

import json

from wild_memory.store.base import MemoryStore


class ReflectionLayer:
    def __init__(self, store: MemoryStore, router, config):
        self.store = store
        self.router = router
        self.config = config

    async def run_reflection(self, agent_id: str, user_id: str) -> None:
        """Run reflection for a specific user."""
        observations = await self.store.list_observations(
            agent_id=agent_id,
            user_id=user_id,
            status="active",
            limit=50,
            order_by="created_at",
            desc=True,
        )
        if not observations:
            return

        from wild_memory.prompts.reflect import REFLECT_PROMPT

        result = await self.router.call(
            task="reflection",
            system=REFLECT_PROMPT.format(
                agent_id=agent_id,
                observations_json=json.dumps(observations, default=str),
            ),
            messages=[{"role": "user", "content": "Analyze."}],
        )

        try:
            data = json.loads(result.text)
        except (json.JSONDecodeError, KeyError):
            return

        for pattern in data.get("patterns", []):
            await self.store.insert_reflection(
                {
                    "agent_id": agent_id,
                    "user_id": user_id,
                    "reflection_type": "pattern",
                    "content": pattern["content"],
                    "importance": pattern.get("importance", 7),
                }
            )
        for insight in data.get("insights", []):
            await self.store.insert_reflection(
                {
                    "agent_id": agent_id,
                    "user_id": user_id,
                    "reflection_type": "insight",
                    "content": insight["content"],
                    "importance": insight.get("importance", 7),
                }
            )

    async def run_all_users(self, agent_id: str) -> None:
        """Run reflection for every user with active observations."""
        for user_id in await self.store.list_active_user_ids(agent_id=agent_id):
            await self.run_reflection(agent_id, user_id)

    async def get_relevant(
        self, agent_id: str, user_id: str, goal: str, limit: int = 3
    ) -> list[dict]:
        return await self.store.list_reflections(
            agent_id=agent_id, user_id=user_id, limit=limit
        )
