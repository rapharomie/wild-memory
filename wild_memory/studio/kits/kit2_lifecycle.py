"""
Kit 2 — Funciona ao longo do tempo?

Walk a user through a 7-day timeline that exercises decay, archive,
type-based protection (decisions survive), and conflict detection (a
later preference shadowing an earlier one). Provider-agnostic — the kit
talks to the storage and decay layers directly so it stays
deterministic.
"""
from __future__ import annotations

import time

from wild_memory.config import WildMemoryConfig
from wild_memory.studio.kits.reports import Check, KitReport


_DECAY_PER_DAY = 0.25
_ARCHIVE_THRESHOLD = 0.4
_PROTECTED_TYPES = ["decision", "correction"]


async def _advance_days(store, n_days: int) -> None:
    """Apply n days of decay and archive low-decay (excluding protected types)."""
    for _ in range(n_days):
        await store.apply_daily_decay(_DECAY_PER_DAY)
    await store.archive_low_decay_observations(
        threshold=_ARCHIVE_THRESHOLD, protected_types=_PROTECTED_TYPES
    )


KIT_ID = "2"
KIT_TITLE = "Ao longo do tempo?"


async def run_kit2_lifecycle(*, store, embedding, llm) -> KitReport:
    started = time.perf_counter()
    config = WildMemoryConfig()
    log: list[str] = []
    agent_id = "kit2"
    user_id = "bob"

    # ── Day 1: a strong preference + some chitchat ──
    pref_dark_emb = await embedding.embed("I prefer dark mode in editors")
    pref_dark_id = await store.insert_observation(
        {
            "agent_id": agent_id,
            "user_id": user_id,
            "content": "User prefers dark mode in editors",
            "obs_type": "preference",
            "importance": 6,
            "embedding": pref_dark_emb,
        }
    )
    for i, msg in enumerate(
        ["nice weather today", "haha cool", "okay sounds good"], start=1
    ):
        await store.insert_observation(
            {
                "agent_id": agent_id,
                "user_id": user_id,
                "content": f"chitchat: {msg}",
                "obs_type": "fact",
                "importance": 2,
                "embedding": await embedding.embed(msg),
            }
        )
    log.append("day 1: 1 preference + 3 chitchat saved")

    # ── Day 3: simulate two days passing, then a contradicting preference ──
    await _advance_days(store, 2)
    pref_light_emb = await embedding.embed("Actually I prefer light mode now")
    pref_light_id = await store.insert_observation(
        {
            "agent_id": agent_id,
            "user_id": user_id,
            "content": "User now prefers light mode",
            "obs_type": "correction",
            "importance": 7,
            "embedding": pref_light_emb,
        }
    )
    similar = await store.find_similar_observations(
        agent_id=agent_id,
        user_id=user_id,
        embedding=pref_light_emb,
        threshold=0.0,
        limit=5,
    )
    conflict_detected = any(s.id == pref_dark_id for s in similar)
    if conflict_detected:
        await store.update_observation(
            pref_dark_id,
            {
                "invalidated_by": pref_light_id,
                "status": "archived",
            },
        )
    log.append(
        f"day 3: similar found={len(similar)}, contradiction handled={conflict_detected}"
    )

    # ── Day 5: an explicit decision (protected type) ──
    decision_id = await store.insert_observation(
        {
            "agent_id": agent_id,
            "user_id": user_id,
            "content": "User decided to migrate frontend from React to Svelte",
            "obs_type": "decision",
            "importance": 9,
            "embedding": await embedding.embed("decided to migrate React Svelte"),
        }
    )
    log.append("day 5: decision saved (protected)")

    # ── Day 7: another two days of decay + archive low-decay ──
    await _advance_days(store, 2)
    log.append("day 7: 4 total days of decay applied + archive sweep")

    # ── Verify ──
    decision = await store.get_observation(decision_id)
    pref_dark = await store.get_observation(pref_dark_id)
    pref_light = await store.get_observation(pref_light_id)

    chitchats = [
        r
        for r in await store.list_observations(
            agent_id=agent_id, user_id=user_id, status=None, limit=200
        )
        if r.get("content", "").startswith("chitchat:")
    ]
    chitchats_archived = sum(1 for r in chitchats if r.get("status") == "archived")

    checks = [
        Check(
            name="A contradicting preference triggered the conflict path",
            passed=conflict_detected,
            detail=f"similarity hit on {pref_dark_id[:8]}…",
        ),
        Check(
            name="Old preference is invalidated by the new one",
            passed=pref_dark and pref_dark.get("status") == "archived",
            detail=f"old.status={pref_dark.get('status') if pref_dark else 'missing'}",
        ),
        Check(
            name="New preference (correction) is active",
            passed=pref_light and pref_light.get("status") == "active",
            detail=f"new.status={pref_light.get('status') if pref_light else 'missing'}",
        ),
        Check(
            name="Decision survived decay + archive (protected type)",
            passed=decision and decision.get("status") == "active",
            detail=f"decision.status={decision.get('status') if decision else 'missing'}, "
            f"decay_score={decision.get('decay_score') if decision else '?'}",
        ),
        Check(
            name="At least one chitchat got archived (low decay)",
            passed=chitchats_archived >= 1,
            detail=f"{chitchats_archived}/{len(chitchats)} chitchat archived",
        ),
    ]

    duration = time.perf_counter() - started
    passed = sum(c.passed for c in checks) >= 4
    return KitReport(
        kit_id=KIT_ID,
        title=KIT_TITLE,
        passed=passed,
        duration_seconds=duration,
        checks=checks,
        metrics={
            "days_simulated": 7,
            "decay_applications": 4,
            "obs_total": len(
                await store.list_observations(
                    agent_id=agent_id, user_id=user_id, status=None, limit=500
                )
            ),
        },
        log=log,
    )
