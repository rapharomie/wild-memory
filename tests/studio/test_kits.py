"""End-to-end tests for the three Test Kits, using mock providers."""
from __future__ import annotations

import pytest

pytest.importorskip("aiosqlite")
pytest.importorskip("sqlite_vec")

from wild_memory.studio.kits import KITS  # noqa: E402
from wild_memory.studio.kits.runner import run_kit  # noqa: E402


@pytest.mark.parametrize("kit_id", sorted(KITS.keys()))
async def test_kit_passes_with_mock_providers(kit_id):
    fn, title, _desc = KITS[kit_id]
    report = await run_kit(fn, kit_id=kit_id, title=title, use_mock=True)
    assert report.error is None, f"Kit {kit_id} errored: {report.error}"
    assert report.passed, (
        f"Kit {kit_id} did not pass: "
        f"{[(c.name, c.passed, c.detail) for c in report.checks]}"
    )
    assert report.duration_seconds > 0
    assert report.metrics["provider_mode"] == "mock"
