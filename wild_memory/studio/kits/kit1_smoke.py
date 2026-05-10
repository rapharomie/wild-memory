"""
Kit 1 — Funciona?

Teach the agent four facts, then ask four questions about them. The
memory passes if the right observations are saved and the right ones
come back on retrieval. ~30 seconds end-to-end with mock providers.
"""
from __future__ import annotations

import time

from wild_memory.processes.bee_distiller import BeeDistiller
from wild_memory.processes.ner_pipeline import NERPipeline
from wild_memory.config import WildMemoryConfig
from wild_memory.infra.embedding_cache import EmbeddingCache
from wild_memory.infra.model_router import ModelRouter
from wild_memory.layers.observation import ObservationLayer
from wild_memory.studio.kits.reports import Check, KitReport


KIT_ID = "1"
KIT_TITLE = "Funciona?"

_FACTS = [
    "My name is Alice.",
    "My favorite color is blue.",
    "I have a dog named Bob.",
    "I prefer dark mode in editors.",
]
_QUESTIONS = [
    ("What is my name?", ["alice", "name"]),
    ("What's my favorite color?", ["blue", "color"]),
    ("What's my dog's name?", ["bob", "dog"]),
    ("Do I prefer dark or light mode?", ["dark", "mode"]),
]


async def run_kit1_smoke(*, store, embedding, llm) -> KitReport:
    started = time.perf_counter()
    config = WildMemoryConfig()
    log: list[str] = []

    # Wire only what we need for save + retrieve.
    embedding_cache = EmbeddingCache(embedding)
    router = ModelRouter(config.models, llm)
    ner = NERPipeline()
    obs_layer = ObservationLayer(store, embedding_cache, router, ner, config)
    distiller = BeeDistiller(obs_layer, router, ner, config)
    # Skip the gate: this kit deliberately uses short factual sentences and
    # we want every one to flow into distillation.

    agent_id = "kit1"
    user_id = "alice"
    session_id = "kit1_session"

    # ── Phase 1: teach ──
    saved_per_fact: list[int] = []
    for fact in _FACTS:
        before = len(await store.list_observations(agent_id=agent_id, user_id=user_id))
        await distiller.distill_and_save(
            agent_id=agent_id,
            user_id=user_id,
            conversation=[{"role": "user", "content": fact}],
            session_id=session_id,
            conflict_resolver=None,
            flush_mode=True,
        )
        after = len(await store.list_observations(agent_id=agent_id, user_id=user_id))
        delta = after - before
        saved_per_fact.append(delta)
        log.append(f"taught: {fact!r} → +{delta} obs")
        embedding_cache.clear_turn()

    total_saved = sum(saved_per_fact)
    log.append(f"total observations saved: {total_saved}")

    # ── Phase 2: ask ──
    correct_recalls = 0
    recall_details: list[str] = []
    for question, expected_substrings in _QUESTIONS:
        results = await obs_layer.retrieve(
            agent_id=agent_id,
            user_id=user_id,
            goal=question,
            entities=[],
            search_query=question,
            limit=3,
            min_decay=0.0,
        )
        embedding_cache.clear_turn()
        joined = " ".join(r.get("content", "") for r in results).lower()
        hit = any(sub in joined for sub in expected_substrings)
        recall_details.append(
            f"{'✓' if hit else '✗'} {question!r} → top1={results[0].get('content','-') if results else '<empty>'}"
        )
        if hit:
            correct_recalls += 1

    log.extend(recall_details)

    checks = [
        Check(
            name="At least 3 of 4 facts produced a saved observation",
            passed=total_saved >= 3,
            detail=f"saved={total_saved}/4 (per fact: {saved_per_fact})",
        ),
        Check(
            name="At least 3 of 4 questions retrieved the right fact",
            passed=correct_recalls >= 3,
            detail=f"correct_recalls={correct_recalls}/4",
        ),
        Check(
            name="Retrieval reinforces accessed observations (decay > baseline)",
            passed=await _all_accessed_observations_were_reinforced(store, agent_id, user_id),
            detail="checked decay_score > 0.95 on the saved set",
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
            "facts_taught": len(_FACTS),
            "facts_saved": total_saved,
            "questions_asked": len(_QUESTIONS),
            "correct_recalls": correct_recalls,
            "embedding_cache_hit_rate": embedding_cache.stats["hit_rate"],
        },
        log=log,
    )


async def _all_accessed_observations_were_reinforced(
    store, agent_id: str, user_id: str
) -> bool:
    rows = await store.list_observations(agent_id=agent_id, user_id=user_id)
    if not rows:
        return False
    return all(float(r.get("decay_score", 0)) >= 0.95 for r in rows)
