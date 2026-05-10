"""
Model Router — selects between premium and economy LLM tiers per task.

Premium: user-facing agent responses (highest quality).
Economy: internal operations like distillation, reflection, conflict
checks (~73% cheaper). All calls go through a single `LLMProvider`.
"""
from __future__ import annotations

from typing import Optional

from wild_memory.config import ModelsConfig
from wild_memory.providers.base import LLMProvider, LLMResponse


TASK_TIER_MAP: dict[str, str] = {
    # Premium: user-facing agent responses
    "agent_response": "premium",
    # Economy: internal operations
    "distillation": "economy",
    "distillation_flush": "economy",
    "goal_detection": "economy",
    "reflection": "economy",
    "feedback_analysis": "economy",
    "entity_extraction": "economy",
    "summary_compression": "economy",
    "conflict_resolution": "economy",
}


class ModelRouter:
    """Routes LLM calls to premium or economy models per task type."""

    def __init__(self, config: ModelsConfig, provider: LLMProvider):
        self.config = config
        self.provider = provider

    def model_for(self, task: str) -> str:
        tier = TASK_TIER_MAP.get(task, "economy")
        return getattr(self.config, tier).model

    async def call(
        self,
        *,
        task: str,
        system: str,
        messages: list,
        tools: Optional[list] = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        return await self.provider.complete(
            model=self.model_for(task),
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
        )
