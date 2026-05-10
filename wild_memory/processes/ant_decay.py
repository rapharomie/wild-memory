"""
Decay system.

Active forgetting: memories not accessed lose strength. Three axes:
temporal, frequency-based, importance-driven. Runs daily as a cron job.
"""
from __future__ import annotations

from wild_memory.config import DecayConfig
from wild_memory.store.base import MemoryStore


class AntDecay:
    def __init__(self, store: MemoryStore, config: DecayConfig):
        self.store = store
        self.config = config

    async def run_daily(self) -> None:
        """Apply decay → mark stale → archive low-decay non-protected obs."""
        await self.store.apply_daily_decay(self.config.daily_rate)
        await self.store.mark_stale_observations(self.config.stale_threshold)
        await self.store.archive_low_decay_observations(
            threshold=self.config.archive_threshold,
            protected_types=list(self.config.protected_types or []),
        )
