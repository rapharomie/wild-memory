"""
SQLiteStore smoke tests.

Real end-to-end: spin up an in-memory SQLite database, run the migration,
exercise every entity-group method against it, and verify the results.
No network, no cloud, no fixtures.
"""
from __future__ import annotations

import math

import pytest


pytest.importorskip("aiosqlite")
pytest.importorskip("sqlite_vec")

from wild_memory.store.sqlite import SQLiteStore  # noqa: E402


DIM = 16


def _vec(seed: int) -> list[float]:
    """Deterministic unit-length vector for tests."""
    raw = [(seed * 1.7 + i * 0.31) % 1.0 for i in range(DIM)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


@pytest.fixture
async def store():
    s = SQLiteStore(":memory:", embedding_dim=DIM)
    await s.connect()
    await s.migrate(embedding_dim=DIM)
    yield s
    await s.close()


async def test_health_check(store):
    h = await store.health_check()
    assert h["ok"] is True
    assert h["backend"] == "sqlite"
    assert h["embedding_dim"] == DIM


async def test_observation_insert_get_update(store):
    obs_id = await store.insert_observation(
        {
            "agent_id": "agent_a",
            "user_id": "user_1",
            "content": "User likes blue",
            "obs_type": "preference",
            "entities": ["color_blue"],
            "importance": 6,
            "embedding": _vec(1),
        }
    )
    assert obs_id

    got = await store.get_observation(obs_id)
    assert got["content"] == "User likes blue"
    assert got["entities"] == ["color_blue"]
    assert got["importance"] == 6

    await store.update_observation(obs_id, {"importance": 9})
    again = await store.get_observation(obs_id)
    assert again["importance"] == 9


async def test_list_observations_filters(store):
    for i in range(3):
        await store.insert_observation(
            {
                "agent_id": "a",
                "user_id": "u",
                "content": f"fact {i}",
                "obs_type": "fact",
                "embedding": _vec(i + 10),
            }
        )
    rows = await store.list_observations(agent_id="a", user_id="u")
    assert len(rows) == 3
    rows_other = await store.list_observations(agent_id="other")
    assert rows_other == []


async def test_retrieve_observations_orders_by_score(store):
    # Insert 3 observations with different embeddings; query close to one.
    target_emb = _vec(42)
    near_id = await store.insert_observation(
        {
            "agent_id": "a",
            "user_id": "u",
            "content": "the target observation",
            "obs_type": "fact",
            "embedding": target_emb,
            "importance": 8,
        }
    )
    for seed in (1, 99, 7):
        await store.insert_observation(
            {
                "agent_id": "a",
                "user_id": "u",
                "content": f"distractor {seed}",
                "obs_type": "fact",
                "embedding": _vec(seed),
            }
        )
    results = await store.retrieve_observations(
        agent_id="a",
        user_id="u",
        embedding=target_emb,
        entities=[],
        search_query="target",
        limit=4,
        min_decay=0.0,
    )
    assert len(results) >= 1
    assert results[0].id == near_id  # exact-match embedding wins


async def test_decay_and_archive(store):
    obs_id = await store.insert_observation(
        {
            "agent_id": "a",
            "user_id": "u",
            "content": "decays",
            "obs_type": "fact",
            "embedding": _vec(5),
        }
    )
    await store.apply_daily_decay(0.5)
    got = await store.get_observation(obs_id)
    assert got["decay_score"] == pytest.approx(0.5)

    await store.mark_stale_observations(0.6)
    got = await store.get_observation(obs_id)
    assert got["status"] == "stale"


async def test_entity_graph_roundtrip(store):
    await store.upsert_entity(
        entity_id="e1", entity_type="person", display_name="Alex",
        attributes={"city": "Berlin"},
    )
    await store.upsert_entity(
        entity_id="e2", entity_type="project", display_name="Wild",
        attributes={},
    )
    await store.upsert_edge(
        subject_id="e1", predicate="works_on", object_id="e2",
        properties={"since": "2025"},
    )
    e1 = await store.get_entity("e1")
    assert e1["attributes"] == {"city": "Berlin"}
    edges = await store.list_edges_for_entity("e1")
    assert len(edges) == 1
    assert edges[0]["predicate"] == "works_on"


async def test_reflection_and_feedback(store):
    rid = await store.insert_reflection(
        {"agent_id": "a", "user_id": "u", "reflection_type": "insight",
         "content": "The user prefers concise replies", "importance": 7}
    )
    assert rid
    refs = await store.list_reflections(agent_id="a", user_id="u")
    assert len(refs) == 1

    fid = await store.insert_feedback(
        {"agent_id": "a", "user_id": "u", "session_id": "s1",
         "signal_type": "satisfaction", "reward_score": 0.7}
    )
    assert fid
    fs = await store.list_session_feedback("s1")
    assert len(fs) == 1
    summary = await store.feedback_summary(agent_id="a", days=30)
    assert any(r.signal_type == "satisfaction" for r in summary)


async def test_semantic_cache_roundtrip(store):
    emb = _vec(11)
    await store.insert_semantic_cache(
        agent_id="a", query_text="hello?", response_text="hi!",
        embedding=emb, ttl_hours=1,
    )
    hit = await store.search_semantic_cache(
        agent_id="a", embedding=emb, threshold=0.5
    )
    assert hit is not None
    assert hit["response_text"] == "hi!"


async def test_checkpoint_and_session(store):
    class FakeWorking:
        messages = [{"role": "user", "content": "hi"}]
        token_count = 1
        current_goal = None
        module_states = {}

    await store.upsert_checkpoint(
        {"agent_id": "a", "session_id": "s1",
         "working_memory": {"messages": FakeWorking.messages, "token_count": 1,
                            "current_goal": None, "module_states": {}},
         "active_procedure": None, "last_used_obs_ids": [],
         "message_count": 1}
    )
    cp = await store.get_checkpoint(agent_id="a", session_id="s1")
    assert cp["working_memory"]["messages"] == [{"role": "user", "content": "hi"}]

    await store.upsert_session_log(
        session_id="s1", agent_id="a", user_id="u",
        messages=[{"role": "user", "content": "hi"}],
    )
    log = await store.get_session_log("s1")
    assert log["messages"] == [{"role": "user", "content": "hi"}]
