"""
Wild Memory Studio.

A Flask-based UI to inspect the memory and run the three Test Kits that
prove the framework works. Studio runs against any MemoryStore, but the
Test Kits always create their own ephemeral SQLite sandbox so they
never touch real data.
"""

from wild_memory.studio.kits.reports import Check, KitReport
from wild_memory.studio.kits.runner import run_kit

__all__ = ["Check", "KitReport", "run_kit"]
