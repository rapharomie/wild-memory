"""
Checkpoint Manager.

Periodically saves agent state for crash recovery. State is restored on the
next interaction with the same `(agent_id, session_id)` pair.
"""
from __future__ import annotations

from typing import Optional

from wild_memory.config import CheckpointConfig
from wild_memory.store.base import MemoryStore


class CheckpointManager:
    def __init__(self, store: MemoryStore, config: CheckpointConfig):
        self.store = store
        self.interval = config.interval_messages

    def should_checkpoint(self, message_count: int) -> bool:
        return message_count > 0 and message_count % self.interval == 0

    async def save(
        self,
        agent_id: str,
        session_id: str,
        working,
        procedure=None,
        last_obs_ids=None,
    ) -> None:
        await self.store.upsert_checkpoint(
            {
                "agent_id": agent_id,
                "session_id": session_id,
                "working_memory": {
                    "messages": working.messages,
                    "token_count": working.token_count,
                    "current_goal": working.current_goal,
                    "module_states": working.module_states,
                },
                "active_procedure": procedure,
                "last_used_obs_ids": last_obs_ids or [],
                "message_count": len(working.messages),
            }
        )

    async def restore(
        self, agent_id: str, session_id: str
    ) -> Optional[dict]:
        return await self.store.get_checkpoint(
            agent_id=agent_id, session_id=session_id
        )

    async def cleanup_old(self, hours: int = 24) -> int:
        return await self.store.cleanup_old_checkpoints(hours)
