"""
Citation Logger.

Records which observations informed each response. Enables forward and
reverse auditability ("which memories shaped this reply?" /
"which replies used this memory?").
"""
from __future__ import annotations

from wild_memory.store.base import MemoryStore


class CitationLogger:
    def __init__(self, store: MemoryStore):
        self.store = store

    async def log(
        self,
        agent_id: str,
        user_id: str,
        session_id: str,
        message_index: int,
        used_observation_ids: list[str] | None = None,
        used_reflection_ids: list[str] | None = None,
        active_procedure_id: str | None = None,
        active_procedure_step: str | None = None,
        n_sources: int = 0,
        avg_decay_score: float | None = None,
    ) -> None:
        await self.store.insert_citation(
            {
                "agent_id": agent_id,
                "user_id": user_id,
                "session_id": session_id,
                "message_index": message_index,
                "used_observation_ids": used_observation_ids or [],
                "used_reflection_ids": used_reflection_ids or [],
                "active_procedure_id": active_procedure_id,
                "active_procedure_step": active_procedure_step,
                "n_sources": n_sources,
                "avg_decay_score": avg_decay_score,
            }
        )

    async def get_trail(self, session_id: str) -> list[dict]:
        return await self.store.list_citations_for_session(session_id)

    async def find_impact(self, obs_id: str) -> list[dict]:
        return await self.store.find_citations_using_observation(obs_id)
