"""
Feedback Layer.

Captures outcome signals (conversions, satisfaction, abandonments, etc.)
and rolls them up into reflections so the agent can adapt over time.
"""
from __future__ import annotations

from wild_memory.store.base import MemoryStore


REWARD_MAP = {
    "conversion": 1.0,
    "satisfaction": 0.7,
    "task_completion": 0.5,
    "objection": -0.2,
    "handoff_request": -0.3,
    "dissatisfaction": -0.5,
    "correction": -0.4,
    "abandonment": -0.6,
    "task_failure": -0.8,
}


class FeedbackLayer:
    def __init__(self, store: MemoryStore, config):
        self.store = store
        self.config = config

    async def record_signal(
        self,
        agent_id: str,
        user_id: str,
        session_id: str,
        signal_type: str,
        action_taken: str | None = None,
        procedure_id: str | None = None,
        procedure_step: str | None = None,
        source: str = "implicit",
        external_ref: str | None = None,
    ) -> None:
        await self.store.insert_feedback(
            {
                "agent_id": agent_id,
                "user_id": user_id,
                "session_id": session_id,
                "signal_type": signal_type,
                "reward_score": REWARD_MAP.get(signal_type, 0.0),
                "action_taken": action_taken,
                "procedure_id": procedure_id,
                "procedure_step": procedure_step,
                "source": source,
                "external_ref": external_ref,
            }
        )

    async def get_session_reward(self, session_id: str) -> float:
        signals = await self.store.list_session_feedback(session_id)
        if not signals:
            return 0.0
        return sum(s["reward_score"] for s in signals) / len(signals)

    async def generate_insights(self, agent_id: str, days: int = 7) -> None:
        """Aggregate recent signals into a reflection insight (cron job)."""
        summary = await self.store.feedback_summary(agent_id=agent_id, days=days)
        if not summary:
            return
        parts = [f"Feedback summary (last {days}d):"]
        for row in summary:
            parts.append(
                f" {row.signal_type}={row.count} (avg_reward:{row.avg_reward:.2f}),"
            )
        content = "".join(parts).rstrip(",")
        await self.store.insert_reflection(
            {
                "agent_id": agent_id,
                "user_id": "system",
                "reflection_type": "insight",
                "content": content,
                "importance": 6,
            }
        )
