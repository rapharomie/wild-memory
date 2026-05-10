"""
Session Logger.

Raw capture of conversations. Safety net for re-distillation. Auto-cleaned
based on `SessionLogConfig.ttl_days`.
"""
from __future__ import annotations

from datetime import datetime, timezone

from wild_memory.config import SessionLogConfig
from wild_memory.store.base import MemoryStore


class SessionLogger:
    def __init__(self, store: MemoryStore, config: SessionLogConfig):
        self.store = store
        self.ttl_days = config.ttl_days

    async def append(
        self,
        session_id: str,
        messages: list[dict],
        agent_id: str = "",
        user_id: str = "",
    ) -> None:
        clean = [
            {
                "role": m.get("role", "unknown"),
                "content": m.get("content", ""),
                "timestamp": m.get(
                    "timestamp", datetime.now(timezone.utc).isoformat()
                ),
            }
            for m in messages
            if isinstance(m, dict) and "content" in m
        ]
        await self.store.upsert_session_log(
            session_id=session_id,
            agent_id=agent_id,
            user_id=user_id,
            messages=clean,
        )

    async def cleanup_expired(self) -> int:
        return await self.store.cleanup_expired_sessions()
