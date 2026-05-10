"""
Kit 3 — Aguenta volume?

Spin up 10 synthetic users in parallel, each writing ~6 observations.
Verify zero leakage between users (recall for user A never returns
user B's data) and that latency stays sane on a small machine.
"""
from __future__ import annotations

import asyncio
import time

from wild_memory.studio.kits.reports import Check, KitReport


KIT_ID = "3"
KIT_TITLE = "Aguenta volume?"

NUM_USERS = 10
OBS_PER_USER = 6
RECALL_QUESTIONS_PER_USER = 3
BATCH_SIZE = 4


async def run_kit3_scale(*, store, embedding, llm) -> KitReport:
    started = time.perf_counter()
    log: list[str] = []
    agent_id = "kit3"

    # ── Phase 1: write phase ──
    write_started = time.perf_counter()
    user_ids = [f"user_{i:02d}" for i in range(NUM_USERS)]

    async def seed_user(uid: str):
        for j in range(OBS_PER_USER):
            content = f"User {uid} fact #{j}: code {uid}-{j}"
            await store.insert_observation(
                {
                    "agent_id": agent_id,
                    "user_id": uid,
                    "content": content,
                    "obs_type": "fact",
                    "importance": 5,
                    "embedding": await embedding.embed(content),
                }
            )

    for chunk_start in range(0, len(user_ids), BATCH_SIZE):
        batch = user_ids[chunk_start : chunk_start + BATCH_SIZE]
        await asyncio.gather(*(seed_user(uid) for uid in batch))
    write_dur = time.perf_counter() - write_started
    log.append(f"seed phase: {NUM_USERS} users × {OBS_PER_USER} obs in {write_dur:.2f}s")

    # ── Phase 2: recall phase ──
    recall_started = time.perf_counter()
    leakage_count = 0
    correct_count = 0
    total_recalls = 0
    latencies: list[float] = []
    for uid in user_ids:
        for j in range(RECALL_QUESTIONS_PER_USER):
            query = f"code {uid}-{j}"
            t0 = time.perf_counter()
            results = await store.retrieve_observations(
                agent_id=agent_id,
                user_id=uid,
                embedding=await embedding.embed(query),
                entities=[],
                search_query=query,
                limit=3,
                min_decay=0.0,
            )
            latencies.append(time.perf_counter() - t0)
            total_recalls += 1
            for r in results:
                # Anything in results MUST belong to this user.
                if uid not in (r.raw or {}).get("content", ""):
                    leakage_count += 1
            if any(query in (r.raw or {}).get("content", "") for r in results):
                correct_count += 1
    recall_dur = time.perf_counter() - recall_started
    log.append(
        f"recall phase: {total_recalls} queries in {recall_dur:.2f}s "
        f"({total_recalls/recall_dur:.1f} qps)"
    )

    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0

    checks = [
        Check(
            name="Zero cross-user leakage",
            passed=leakage_count == 0,
            detail=f"{leakage_count} leak hits across {total_recalls} queries",
        ),
        Check(
            name="≥80% of queries returned the exact target observation",
            passed=correct_count / max(total_recalls, 1) >= 0.80,
            detail=f"correct={correct_count}/{total_recalls}",
        ),
        Check(
            name="Retrieval p50 latency < 200ms",
            passed=p50 < 0.200,
            detail=f"p50={p50*1000:.1f}ms p95={p95*1000:.1f}ms",
        ),
    ]

    duration = time.perf_counter() - started
    passed = all(c.passed for c in checks)
    return KitReport(
        kit_id=KIT_ID,
        title=KIT_TITLE,
        passed=passed,
        duration_seconds=duration,
        checks=checks,
        metrics={
            "num_users": NUM_USERS,
            "obs_per_user": OBS_PER_USER,
            "recall_queries": total_recalls,
            "leakage_count": leakage_count,
            "correct_recalls": correct_count,
            "latency_p50_ms": round(p50 * 1000, 1),
            "latency_p95_ms": round(p95 * 1000, 1),
            "write_duration_s": round(write_dur, 2),
            "recall_duration_s": round(recall_dur, 2),
        },
        log=log,
    )
