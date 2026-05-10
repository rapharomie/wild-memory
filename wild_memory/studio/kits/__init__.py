"""Bundled Test Kits.

Each kit is a callable returning a `KitReport`. The runner spins up an
ephemeral SQLite sandbox + (optionally) mock providers and invokes the
kit, so they're safe to run anywhere without touching production data.
"""

from wild_memory.studio.kits.kit1_smoke import run_kit1_smoke
from wild_memory.studio.kits.kit2_lifecycle import run_kit2_lifecycle
from wild_memory.studio.kits.kit3_scale import run_kit3_scale
from wild_memory.studio.kits.reports import Check, KitReport

KITS = {
    "1": (run_kit1_smoke, "Funciona?", "~30s · prova básica de save + recall"),
    "2": (run_kit2_lifecycle, "Ao longo do tempo?", "~2min · decay, conflict, reflexão"),
    "3": (run_kit3_scale, "Aguenta volume?", "~3min · 10 usuários paralelos"),
}

__all__ = ["KITS", "Check", "KitReport", "run_kit1_smoke", "run_kit2_lifecycle", "run_kit3_scale"]
