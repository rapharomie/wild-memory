"""
Parity tests across MemoryStore backends.

Each test runs against every available backend and asserts the same
observable behavior. Backends without their dependencies installed (or
a live Postgres) are skipped automatically.

Run all backends:
    DATABASE_URL=postgres://... pytest tests/store/test_parity.py -v

SQLite-only (default):
    pytest tests/store/test_parity.py -v
"""
from __future__ import annotations

import math
import os
import uuid

import pytest


pytest.importorskip("aiosqlite")
pytest.importorskip("sqlite_vec")

from wild_memory.store.sqlite import SQLiteStore  # noqa: E402


DIM = 16


def _vec(seed: int) -> list[float]:
    raw = [(seed * 1.7 + i * 0.31) % 1.0 for i in range(DIM)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


# ── Backend factories ─────────────────────────────────────────────


async def _sqlite_factory():
    s = SQLiteStore(":memory:", embedding_dim=DIM)
    await s.connect()
    await s.migrate(embedding_dim=DIM)
    return s


async def _postgres_factory():
    try:
        from wild_memory.store.postgres import PostgresStore
    except ImportError:
        pytest.skip("asyncpg not installed")
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        pytest.skip("DATABASE_URL not set; skipping Postgres parity")
    s = PostgresStore(dsn, embedding_dim=DIM)
    await s.connect()
    await s.migrate(embedding_dim=DIM)
    # Use a unique agent_id so parallel parity runs don't collide.
    return s


_FACTORIES = {
    "sqlite": _sqlite_factory,
    "postgres": _postgres_factory,
}


@pytest.fixture(params=sorted(_FACTORIES.keys()))
async def store(request):
    factory = _FACTORIES[request.param]
    s = await factory()
    yield s
    await s.close()


# ── Parity tests ──────────────────────────────────────────────────


async def test_observation_insert_get_roundtrip(store):
    agent = f"parity_{uuid.uuid4().hex[:8]}"
    obs_id = await store.insert_observation(
        {
            "agent_id": agent,
            "user_id": "u",
            "content": "Color preference: blue",
            "obs_type": "preference",
            "entities": ["color_blue"],
            "importance": 7,
            "embedding": _vec(1),
        }
    )
    got = await store.get_observation(obs_id)
    assert got["content"] == "Color preference: blue"
    assert got["obs_type"] == "preference"
    assert got["importance"] == 7
    assert "color_blue" in got["entities"]


async def test_retrieve_returns_nearest_first(store):
    agent = f"parity_{uuid.uuid4().hex[:8]}"
    target = _vec(50)
    target_id = await store.insert_observation(
        {
            "agent_id": agent,
            "user_id": "u",
            "content": "the target",
            "obs_type": "fact",
            "embedding": target,
        }
    )
    for seed in (3, 17, 99):
        await store.insert_observation(
            {
                "agent_id": agent,
                "user_id": "u",
                "content": f"distractor {seed}",
                "obs_type": "fact",
                "embedding": _vec(seed),
            }
        )
    results = await store.retrieve_observations(
        agent_id=agent,
        user_id="u",
        embedding=target,
        entities=[],
        search_query="target",
        limit=4,
        min_decay=0.0,
    )
    assert results, "expected at least one result"
    assert results[0].id == target_id, "exact embedding match should rank first"


async def test_decay_then_archive(store):
    agent = f"parity_{uuid.uuid4().hex[:8]}"
    obs_id = await store.insert_observation(
        {
            "agent_id": agent,
            "user_id": "u",
            "content": "decays",
            "obs_type": "fact",
            "embedding": _vec(7),
        }
    )
    await store.apply_daily_decay(0.95)
    await store.archive_low_decay_observations(threshold=0.5, protected_types=[])
    got = await store.get_observation(obs_id)
    assert got["status"] == "archived"


async def test_entity_and_edge_roundtrip(store):
    suffix = uuid.uuid4().hex[:8]
    a = f"e_a_{suffix}"
    b = f"e_b_{suffix}"
    await store.upsert_entity(
        entity_id=a, entity_type="person", display_name="A", attributes={"k": 1}
    )
    await store.upsert_entity(
        entity_id=b, entity_type="topic", display_name="B", attributes={}
    )
    await store.upsert_edge(subject_id=a, predicate="likes", object_id=b)
    edges = await store.list_edges_for_entity(a)
    assert len(edges) == 1
    assert edges[0]["subject_id"] == a
    assert edges[0]["object_id"] == b


async def test_semantic_cache_roundtrip(store):
    agent = f"parity_{uuid.uuid4().hex[:8]}"
    emb = _vec(11)
    await store.insert_semantic_cache(
        agent_id=agent,
        query_text="Q",
        response_text="A",
        embedding=emb,
        ttl_hours=1,
    )
    hit = await store.search_semantic_cache(
        agent_id=agent, embedding=emb, threshold=0.5
    )
    assert hit is not None
    assert hit["response_text"] == "A"
