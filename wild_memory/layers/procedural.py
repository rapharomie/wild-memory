"""
Procedural Memory.

Versioned workflows with per-step success tracking. Editable via markdown
(Dual Interface: humans edit, the agent reads).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from wild_memory.store.base import MemoryStore


class ProceduralMemory:
    def __init__(self, store: MemoryStore):
        self.store = store

    async def get_active_procedure(
        self, agent_id: str, entities: list[str]
    ) -> Optional[dict]:
        """Find the best matching active procedure for current context."""
        procedures = await self.store.list_procedures(
            agent_id=agent_id, status="active"
        )
        if not procedures:
            return None

        ent_set = set(entities)
        best, best_score = None, -1
        for proc in procedures:
            trigger = set(proc.get("trigger_entities", []))
            overlap = len(trigger & ent_set)
            if overlap > best_score:
                best, best_score = proc, overlap

        return best if best_score > 0 else None

    async def record_step_outcome(
        self, procedure_id: str, step_id: str, success: bool
    ) -> None:
        """Update success rate for a specific step."""
        proc = await self.store.get_procedure(procedure_id)
        if not proc:
            return

        steps = proc["steps"]
        for step in steps:
            if step.get("step_id") == step_id:
                total = step.get("total_attempts", 0) + 1
                wins = step.get("successes", 0) + (1 if success else 0)
                step["total_attempts"] = total
                step["successes"] = wins
                step["success_rate"] = round(wins / total, 3)

        total_exec = proc["total_executions"] + 1
        succ_exec = proc["successful_executions"] + (1 if success else 0)

        await self.store.update_procedure(
            procedure_id,
            {
                "steps": steps,
                "total_executions": total_exec,
                "successful_executions": succ_exec,
                "performance_score": round(succ_exec / total_exec, 3),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def get_weak_steps(
        self, procedure_id: str, threshold: float = 0.4
    ) -> list[dict]:
        """Find steps with low success rate (for review suggestions)."""
        proc = await self.store.get_procedure(procedure_id)
        if not proc:
            return []
        return [
            s
            for s in proc["steps"]
            if s.get("success_rate", 1.0) < threshold
            and s.get("total_attempts", 0) >= 10
        ]
