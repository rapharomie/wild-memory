"""
PostgresStore — Postgres + pgvector backend.

Uses asyncpg directly (no Supabase SDK). Embeddings are stored in real
`vector(dim)` columns; ANN search uses pgvector's cosine-distance HNSW
index. The 5-signal scoring algorithm runs in Python (single-sourced in
`wild_memory.store.scoring`) — Postgres provides the candidate set and we
re-score in-process. Same algorithm as the SQLite backend.

Lifecycle:
    store = PostgresStore("postgres://user:pw@host/db", embedding_dim=1536)
    await store.connect()
    await store.migrate(embedding_dim=1536)
    ...
    await store.close()
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from importlib import resources
from typing import Any, Optional
from uuid import UUID

import asyncpg
from jinja2 import Template

from wild_memory.store.base import (
    FeedbackSummaryRow,
    MemoryStore,
    RetrievalWeights,
    RetrievedObservation,
    SimilarObservation,
)
from wild_memory.store.scoring import score_observations


def _vec_literal(emb: list[float]) -> str:
    """Format a Python float list as a pgvector text literal."""
    return "[" + ",".join(repr(float(x)) for x in emb) + "]"


def _row_to_dict(row: asyncpg.Record | None, json_fields: tuple[str, ...] = ()) -> Optional[dict]:
    if row is None:
        return None
    out = dict(row)
    for k, v in list(out.items()):
        if isinstance(v, UUID):
            out[k] = str(v)
        elif isinstance(v, datetime):
            out[k] = v.isoformat()
        elif k in json_fields and isinstance(v, str):
            try:
                out[k] = json.loads(v)
            except json.JSONDecodeError:
                pass
    return out


_JSON_FIELDS_OBSERVATION: tuple[str, ...] = ()  # entities is a TEXT[]
_JSON_FIELDS_ENTITY = ("attributes",)
_JSON_FIELDS_EDGE = ("properties",)
_JSON_FIELDS_PROCEDURE = ("steps",)
_JSON_FIELDS_SESSION = ("messages",)
_JSON_FIELDS_CHECKPOINT = ("working_memory", "active_procedure")
_JSON_FIELDS_IMPRINT = ("values", "constraints", "org_context")


class PostgresStore(MemoryStore):
    """A MemoryStore backed by Postgres + pgvector via asyncpg."""

    def __init__(self, dsn: str, *, embedding_dim: int, pool_min: int = 1, pool_max: int = 10):
        self._dsn = dsn
        self._dim = embedding_dim
        self._pool_min = pool_min
        self._pool_max = pool_max
        self._pool: Optional[asyncpg.Pool] = None

    # ── Lifecycle ──────────────────────────────────────────────────

    async def connect(self) -> None:
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            self._dsn, min_size=self._pool_min, max_size=self._pool_max
        )
        # Register JSONB codec so we get/return Python dicts/lists directly.
        async with self._pool.acquire() as conn:
            await conn.set_type_codec(
                "jsonb",
                encoder=json.dumps,
                decoder=json.loads,
                schema="pg_catalog",
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def migrate(self, *, embedding_dim: int) -> None:
        if self._pool is None:
            await self.connect()
        assert self._pool is not None

        if embedding_dim != self._dim:
            self._dim = embedding_dim

        files = ("001_extensions.sql", "002_schema.sql.j2")
        async with self._pool.acquire() as conn:
            for fname in files:
                raw = (
                    resources.files("wild_memory.store.migrations.postgres")
                    .joinpath(fname)
                    .read_text()
                )
                sql = (
                    Template(raw).render(embedding_dim=embedding_dim)
                    if fname.endswith(".j2")
                    else raw
                )
                await conn.execute(sql)
            await conn.execute(
                """INSERT INTO wild_memory_meta (key, value)
                   VALUES ('embedding_dim', $1)
                   ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value""",
                str(embedding_dim),
            )
            await conn.execute(
                """INSERT INTO wild_memory_meta (key, value)
                   VALUES ('schema_version', '1')
                   ON CONFLICT (key) DO NOTHING"""
            )

    async def health_check(self) -> dict:
        if self._pool is None:
            return {"backend": "postgres", "ok": False, "error": "not connected"}
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT value FROM wild_memory_meta WHERE key='embedding_dim'"
                )
            return {
                "backend": "postgres",
                "ok": True,
                "embedding_dim": int(row["value"]) if row else None,
            }
        except Exception as e:  # pragma: no cover - depends on live DB
            return {"backend": "postgres", "ok": False, "error": str(e)}

    # ── Internal ───────────────────────────────────────────────────

    def _p(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgresStore is not connected. Call connect() first.")
        return self._pool

    # ── Observations ───────────────────────────────────────────────

    async def insert_observation(self, data: dict) -> str:
        embedding = data.get("embedding") or []
        async with self._p().acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO observations
                (agent_id, user_id, content, obs_type, entities, importance,
                 decay_score, ttl_days, emotional_valence, emotional_intensity,
                 privacy_mode, event_time, source_session, embedding, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                        $14::vector, $15)
                RETURNING id""",
                data["agent_id"],
                data["user_id"],
                data["content"],
                data["obs_type"],
                list(data.get("entities") or []),
                int(data.get("importance", 5)),
                float(data.get("decay_score", 1.0)),
                int(data.get("ttl_days", 90)),
                data.get("emotional_valence", "neutral"),
                int(data.get("emotional_intensity", 0)),
                data.get("privacy_mode", "personal"),
                data.get("event_time"),
                data.get("source_session"),
                _vec_literal(embedding) if embedding else None,
                data.get("status", "active"),
            )
        return str(row["id"])

    async def update_observation(self, obs_id: str, fields: dict) -> None:
        if not fields:
            return
        sets, values = [], []
        for i, (k, v) in enumerate(fields.items(), start=1):
            sets.append(f"{k} = ${i}")
            values.append(v)
        values.append(UUID(obs_id))
        async with self._p().acquire() as conn:
            await conn.execute(
                f"UPDATE observations SET {', '.join(sets)} WHERE id = ${len(values)}",
                *values,
            )

    async def get_observation(self, obs_id: str) -> Optional[dict]:
        async with self._p().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM observations WHERE id = $1", UUID(obs_id)
            )
        return _row_to_dict(row)

    async def list_observations(
        self,
        *,
        agent_id: str,
        user_id: Optional[str] = None,
        status: Optional[str] = "active",
        obs_type: Optional[str] = None,
        limit: int = 50,
        order_by: str = "created_at",
        desc: bool = True,
    ) -> list[dict]:
        clauses = ["agent_id = $1"]
        params: list[Any] = [agent_id]
        if user_id is not None:
            params.append(user_id)
            clauses.append(f"user_id = ${len(params)}")
        if status is not None:
            params.append(status)
            clauses.append(f"status = ${len(params)}")
        if obs_type is not None:
            params.append(obs_type)
            clauses.append(f"obs_type = ${len(params)}")
        order = f"{order_by} {'DESC' if desc else 'ASC'}"
        params.append(limit)
        sql = (
            f"SELECT * FROM observations WHERE {' AND '.join(clauses)} "
            f"ORDER BY {order} LIMIT ${len(params)}"
        )
        async with self._p().acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [_row_to_dict(r) for r in rows]

    async def list_active_user_ids(self, *, agent_id: str) -> list[str]:
        async with self._p().acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT user_id FROM observations "
                "WHERE agent_id = $1 AND status = 'active'",
                agent_id,
            )
        return [r["user_id"] for r in rows]

    async def retrieve_observations(
        self,
        *,
        agent_id: str,
        user_id: str,
        embedding: list[float],
        entities: list[str],
        search_query: str,
        limit: int,
        min_decay: float,
        weights: Optional[RetrievalWeights] = None,
    ) -> list[RetrievedObservation]:
        weights = weights or RetrievalWeights()
        k = max(limit * 4, 20)
        async with self._p().acquire() as conn:
            rows = await conn.fetch(
                """SELECT *, embedding <=> $1::vector AS distance
                   FROM observations
                   WHERE agent_id = $2 AND user_id = $3 AND status = 'active'
                     AND embedding IS NOT NULL
                   ORDER BY distance
                   LIMIT $4""",
                _vec_literal(embedding),
                agent_id,
                user_id,
                k,
            )
        # asyncpg returns embedding as a string like '[0.1,0.2,...]'; parse it.
        candidates: list[dict] = []
        for r in rows:
            d = _row_to_dict(r)
            d["embedding"] = _parse_pg_vector(d.get("embedding"))
            candidates.append(d)
        scored = score_observations(
            candidates,
            query_embedding=embedding,
            query_entities=entities,
            search_query=search_query,
            weights=weights,
            min_decay=min_decay,
        )
        for r in scored[:limit]:
            await self.reinforce_observation(r.id)
        return scored[:limit]

    async def find_similar_observations(
        self,
        *,
        agent_id: str,
        user_id: str,
        embedding: list[float],
        threshold: float,
        limit: int,
    ) -> list[SimilarObservation]:
        async with self._p().acquire() as conn:
            rows = await conn.fetch(
                """SELECT *, 1 - (embedding <=> $1::vector) AS similarity
                   FROM observations
                   WHERE agent_id = $2 AND user_id = $3 AND status = 'active'
                     AND embedding IS NOT NULL
                   ORDER BY embedding <=> $1::vector
                   LIMIT $4""",
                _vec_literal(embedding),
                agent_id,
                user_id,
                limit,
            )
        out: list[SimilarObservation] = []
        for r in rows:
            d = _row_to_dict(r)
            sim = float(d.get("similarity", 0.0))
            if sim < threshold:
                continue
            out.append(
                SimilarObservation(
                    id=str(d["id"]),
                    content=d.get("content", ""),
                    obs_type=d.get("obs_type", "fact"),
                    importance=int(d.get("importance", 5)),
                    similarity=sim,
                    raw=d,
                )
            )
        return out

    async def reinforce_observation(self, obs_id: str, boost: float = 0.15) -> None:
        async with self._p().acquire() as conn:
            await conn.execute(
                """UPDATE observations
                   SET decay_score = LEAST(1.0, decay_score + $1),
                       last_accessed = NOW()
                   WHERE id = $2""",
                boost,
                UUID(obs_id),
            )

    async def apply_daily_decay(self, decay_rate: float) -> None:
        async with self._p().acquire() as conn:
            await conn.execute(
                """UPDATE observations
                   SET decay_score = GREATEST(0.0, decay_score - $1)
                   WHERE status = 'active'""",
                decay_rate,
            )

    async def mark_stale_observations(self, threshold: float) -> None:
        async with self._p().acquire() as conn:
            await conn.execute(
                """UPDATE observations SET status = 'stale'
                   WHERE status = 'active' AND decay_score < $1""",
                threshold,
            )

    async def archive_low_decay_observations(
        self, threshold: float, protected_types: list[str]
    ) -> None:
        async with self._p().acquire() as conn:
            if protected_types:
                await conn.execute(
                    """UPDATE observations SET status = 'archived'
                       WHERE status = 'active' AND decay_score < $1
                         AND obs_type <> ALL($2)""",
                    threshold,
                    protected_types,
                )
            else:
                await conn.execute(
                    """UPDATE observations SET status = 'archived'
                       WHERE status = 'active' AND decay_score < $1""",
                    threshold,
                )

    async def anonymize_user_observations(self, user_id: str, anon_hash: str) -> None:
        async with self._p().acquire() as conn:
            await conn.execute(
                """UPDATE observations
                   SET privacy_mode = 'pattern', user_id = 'anonymized',
                       anonymized_user_hash = $1, entities = '{}'
                   WHERE user_id = $2 AND privacy_mode = 'personal'""",
                anon_hash,
                user_id,
            )

    async def purge_user_observations(self, user_id: str) -> None:
        async with self._p().acquire() as conn:
            await conn.execute(
                "UPDATE observations SET status = 'purged' WHERE user_id = $1",
                user_id,
            )

    # ── Entities & Edges ───────────────────────────────────────────

    async def upsert_entity(
        self,
        *,
        entity_id: str,
        entity_type: str,
        display_name: str,
        attributes: dict,
    ) -> None:
        async with self._p().acquire() as conn:
            await conn.execute(
                """INSERT INTO entity_nodes (id, entity_type, display_name, attributes)
                   VALUES ($1, $2, $3, $4::jsonb)
                   ON CONFLICT (id) DO UPDATE SET
                     entity_type = EXCLUDED.entity_type,
                     display_name = EXCLUDED.display_name,
                     attributes = EXCLUDED.attributes""",
                entity_id,
                entity_type,
                display_name,
                json.dumps(attributes),
            )

    async def get_entity(self, entity_id: str) -> Optional[dict]:
        async with self._p().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM entity_nodes WHERE id = $1", entity_id
            )
        return _row_to_dict(row)

    async def update_entity_attributes(
        self, entity_id: str, attributes: dict
    ) -> None:
        async with self._p().acquire() as conn:
            await conn.execute(
                "UPDATE entity_nodes SET attributes = $1::jsonb WHERE id = $2",
                json.dumps(attributes),
                entity_id,
            )

    async def upsert_edge(
        self,
        *,
        subject_id: str,
        predicate: str,
        object_id: str,
        source_observation: Optional[str] = None,
        properties: Optional[dict] = None,
    ) -> None:
        src = UUID(source_observation) if source_observation else None
        async with self._p().acquire() as conn:
            await conn.execute(
                """INSERT INTO entity_edges
                   (subject_id, predicate, object_id, source_observation, properties)
                   VALUES ($1, $2, $3, $4, $5::jsonb)
                   ON CONFLICT (subject_id, predicate, object_id) DO UPDATE SET
                     source_observation = EXCLUDED.source_observation,
                     properties = EXCLUDED.properties""",
                subject_id,
                predicate,
                object_id,
                src,
                json.dumps(properties or {}),
            )

    async def list_edges_for_entity(self, entity_id: str) -> list[dict]:
        async with self._p().acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM entity_edges WHERE subject_id = $1 OR object_id = $1",
                entity_id,
            )
        return [_row_to_dict(r) for r in rows]

    # ── Reflections ────────────────────────────────────────────────

    async def insert_reflection(self, reflection: dict) -> str:
        async with self._p().acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO reflections
                   (agent_id, user_id, reflection_type, content, importance)
                   VALUES ($1, $2, $3, $4, $5)
                   RETURNING id""",
                reflection["agent_id"],
                reflection["user_id"],
                reflection["reflection_type"],
                reflection["content"],
                int(reflection.get("importance", 5)),
            )
        return str(row["id"])

    async def list_reflections(
        self,
        *,
        agent_id: str,
        user_id: Optional[str] = None,
        limit: int = 10,
        order_by: str = "created_at",
        desc: bool = True,
    ) -> list[dict]:
        clauses = ["agent_id = $1"]
        params: list[Any] = [agent_id]
        if user_id is not None:
            params.append(user_id)
            clauses.append(f"user_id = ${len(params)}")
        order = f"{order_by} {'DESC' if desc else 'ASC'}"
        params.append(limit)
        sql = (
            f"SELECT * FROM reflections WHERE {' AND '.join(clauses)} "
            f"ORDER BY {order} LIMIT ${len(params)}"
        )
        async with self._p().acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [_row_to_dict(r) for r in rows]

    async def purge_user_reflections(self, user_id: str) -> int:
        async with self._p().acquire() as conn:
            result = await conn.execute(
                "DELETE FROM reflections WHERE user_id = $1", user_id
            )
        return _rowcount(result)

    # ── Feedback ───────────────────────────────────────────────────

    async def insert_feedback(self, signal: dict) -> str:
        async with self._p().acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO feedback_signals
                   (agent_id, user_id, session_id, signal_type, reward_score,
                    action_taken, procedure_id, procedure_step, source, external_ref)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                   RETURNING id""",
                signal["agent_id"],
                signal["user_id"],
                signal["session_id"],
                signal["signal_type"],
                float(signal.get("reward_score", 0)),
                signal.get("action_taken"),
                UUID(signal["procedure_id"]) if signal.get("procedure_id") else None,
                signal.get("procedure_step"),
                signal.get("source", "implicit"),
                signal.get("external_ref"),
            )
        return str(row["id"])

    async def list_session_feedback(self, session_id: str) -> list[dict]:
        async with self._p().acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM feedback_signals WHERE session_id = $1", session_id
            )
        return [_row_to_dict(r) for r in rows]

    async def feedback_summary(
        self, *, agent_id: str, days: int
    ) -> list[FeedbackSummaryRow]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        async with self._p().acquire() as conn:
            rows = await conn.fetch(
                """SELECT signal_type, COUNT(*) AS n, AVG(reward_score) AS avg_r
                   FROM feedback_signals
                   WHERE agent_id = $1 AND created_at >= $2
                   GROUP BY signal_type""",
                agent_id,
                cutoff,
            )
        return [
            FeedbackSummaryRow(
                signal_type=r["signal_type"],
                count=int(r["n"]),
                avg_reward=float(r["avg_r"] or 0),
                top_action=None,
            )
            for r in rows
        ]

    async def purge_user_feedback(self, user_id: str) -> int:
        async with self._p().acquire() as conn:
            result = await conn.execute(
                "DELETE FROM feedback_signals WHERE user_id = $1", user_id
            )
        return _rowcount(result)

    # ── Procedures ─────────────────────────────────────────────────

    async def list_procedures(
        self, *, agent_id: str, status: str = "active"
    ) -> list[dict]:
        async with self._p().acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM procedures WHERE agent_id = $1 AND status = $2",
                agent_id,
                status,
            )
        return [_row_to_dict(r) for r in rows]

    async def get_procedure(self, procedure_id: str) -> Optional[dict]:
        async with self._p().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM procedures WHERE id = $1", UUID(procedure_id)
            )
        return _row_to_dict(row)

    async def update_procedure(self, procedure_id: str, fields: dict) -> None:
        if not fields:
            return
        sets, values = [], []
        for i, (k, v) in enumerate(fields.items(), start=1):
            if k in _JSON_FIELDS_PROCEDURE:
                sets.append(f"{k} = ${i}::jsonb")
                values.append(json.dumps(v))
            else:
                sets.append(f"{k} = ${i}")
                values.append(v)
        values.append(UUID(procedure_id))
        async with self._p().acquire() as conn:
            await conn.execute(
                f"UPDATE procedures SET {', '.join(sets)} WHERE id = ${len(values)}",
                *values,
            )

    # ── Citations ──────────────────────────────────────────────────

    async def insert_citation(self, citation: dict) -> str:
        used_obs = [UUID(x) for x in (citation.get("used_observation_ids") or [])]
        used_ref = [UUID(x) for x in (citation.get("used_reflection_ids") or [])]
        proc_id = (
            UUID(citation["active_procedure_id"])
            if citation.get("active_procedure_id")
            else None
        )
        async with self._p().acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO citation_trails
                   (agent_id, user_id, session_id, message_index,
                    used_observation_ids, used_reflection_ids,
                    active_procedure_id, active_procedure_step,
                    n_sources, avg_decay_score)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                   RETURNING id""",
                citation["agent_id"],
                citation["user_id"],
                citation["session_id"],
                int(citation["message_index"]),
                used_obs,
                used_ref,
                proc_id,
                citation.get("active_procedure_step"),
                int(citation.get("n_sources", 0)),
                citation.get("avg_decay_score"),
            )
        return str(row["id"])

    async def list_citations_for_session(self, session_id: str) -> list[dict]:
        async with self._p().acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM citation_trails WHERE session_id = $1 ORDER BY message_index",
                session_id,
            )
        return [_row_to_dict(r) for r in rows]

    async def find_citations_using_observation(self, obs_id: str) -> list[dict]:
        async with self._p().acquire() as conn:
            rows = await conn.fetch(
                """SELECT session_id, message_index, created_at FROM citation_trails
                   WHERE $1 = ANY(used_observation_ids)
                   ORDER BY created_at DESC""",
                UUID(obs_id),
            )
        return [_row_to_dict(r) for r in rows]

    # ── Sessions ───────────────────────────────────────────────────

    async def get_session_log(self, session_id: str) -> Optional[dict]:
        async with self._p().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, messages FROM session_logs WHERE session_id = $1",
                session_id,
            )
        return _row_to_dict(row, _JSON_FIELDS_SESSION)

    async def upsert_session_log(
        self,
        *,
        session_id: str,
        agent_id: str,
        user_id: str,
        messages: list[dict],
    ) -> None:
        existing = await self.get_session_log(session_id)
        async with self._p().acquire() as conn:
            if existing:
                merged = (existing.get("messages") or []) + messages
                await conn.execute(
                    "UPDATE session_logs SET messages = $1::jsonb WHERE id = $2",
                    json.dumps(merged),
                    UUID(existing["id"]),
                )
            else:
                await conn.execute(
                    """INSERT INTO session_logs
                       (session_id, agent_id, user_id, messages)
                       VALUES ($1, $2, $3, $4::jsonb)""",
                    session_id,
                    agent_id,
                    user_id,
                    json.dumps(messages),
                )

    async def cleanup_expired_sessions(self) -> int:
        async with self._p().acquire() as conn:
            result = await conn.execute(
                "DELETE FROM session_logs WHERE expires_at IS NOT NULL AND expires_at < NOW()"
            )
        return _rowcount(result)

    # ── Semantic cache ─────────────────────────────────────────────

    async def search_semantic_cache(
        self,
        *,
        agent_id: str,
        embedding: list[float],
        threshold: float,
    ) -> Optional[dict]:
        async with self._p().acquire() as conn:
            row = await conn.fetchrow(
                """SELECT *, 1 - (embedding <=> $1::vector) AS similarity
                   FROM semantic_cache
                   WHERE agent_id = $2 AND embedding IS NOT NULL
                   ORDER BY embedding <=> $1::vector
                   LIMIT 1""",
                _vec_literal(embedding),
                agent_id,
            )
        if not row:
            return None
        d = _row_to_dict(row)
        if float(d.get("similarity", 0)) < threshold:
            return None
        return d

    async def increment_semantic_cache_hit(self, cache_id: str) -> None:
        async with self._p().acquire() as conn:
            await conn.execute(
                "UPDATE semantic_cache SET hit_count = hit_count + 1, last_hit = NOW() WHERE id = $1",
                UUID(cache_id),
            )

    async def insert_semantic_cache(
        self,
        *,
        agent_id: str,
        query_text: str,
        response_text: str,
        embedding: list[float],
        ttl_hours: int,
    ) -> None:
        expires = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
        async with self._p().acquire() as conn:
            await conn.execute(
                """INSERT INTO semantic_cache
                   (agent_id, query_text, response_text, embedding, ttl_hours, expires_at)
                   VALUES ($1, $2, $3, $4::vector, $5, $6)""",
                agent_id,
                query_text,
                response_text,
                _vec_literal(embedding),
                ttl_hours,
                expires,
            )

    async def cleanup_expired_cache(self) -> int:
        async with self._p().acquire() as conn:
            result = await conn.execute(
                "DELETE FROM semantic_cache WHERE expires_at IS NOT NULL AND expires_at < NOW()"
            )
        return _rowcount(result)

    # ── Checkpoints ────────────────────────────────────────────────

    async def upsert_checkpoint(self, checkpoint: dict) -> None:
        last_obs = [
            UUID(x) for x in (checkpoint.get("last_used_obs_ids") or []) if x
        ]
        async with self._p().acquire() as conn:
            await conn.execute(
                """INSERT INTO agent_checkpoints
                   (agent_id, session_id, working_memory, active_procedure,
                    last_used_obs_ids, message_count)
                   VALUES ($1, $2, $3::jsonb, $4::jsonb, $5, $6)
                   ON CONFLICT (agent_id, session_id) DO UPDATE SET
                     working_memory = EXCLUDED.working_memory,
                     active_procedure = EXCLUDED.active_procedure,
                     last_used_obs_ids = EXCLUDED.last_used_obs_ids,
                     message_count = EXCLUDED.message_count""",
                checkpoint["agent_id"],
                checkpoint["session_id"],
                json.dumps(checkpoint.get("working_memory") or {}),
                json.dumps(checkpoint.get("active_procedure")),
                last_obs,
                int(checkpoint.get("message_count", 0)),
            )

    async def get_checkpoint(
        self, *, agent_id: str, session_id: str
    ) -> Optional[dict]:
        async with self._p().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM agent_checkpoints WHERE agent_id = $1 AND session_id = $2",
                agent_id,
                session_id,
            )
        return _row_to_dict(row)

    async def cleanup_old_checkpoints(self, hours: int) -> int:
        async with self._p().acquire() as conn:
            result = await conn.execute(
                "DELETE FROM agent_checkpoints WHERE created_at < NOW() - $1::interval",
                f"{hours} hours",
            )
        return _rowcount(result)

    # ── Imprints ───────────────────────────────────────────────────

    async def upsert_imprint(self, imprint: dict) -> None:
        async with self._p().acquire() as conn:
            await conn.execute(
                """INSERT INTO agent_imprints
                   (agent_id, role, "values", constraints, org_context, tone_of_voice, updated_at)
                   VALUES ($1, $2, $3::jsonb, $4::jsonb, $5::jsonb, $6, NOW())
                   ON CONFLICT (agent_id) DO UPDATE SET
                     role = EXCLUDED.role,
                     "values" = EXCLUDED."values",
                     constraints = EXCLUDED.constraints,
                     org_context = EXCLUDED.org_context,
                     tone_of_voice = EXCLUDED.tone_of_voice,
                     updated_at = NOW()""",
                imprint["agent_id"],
                imprint.get("role"),
                json.dumps(imprint.get("values") or []),
                json.dumps(imprint.get("constraints") or []),
                json.dumps(imprint.get("org_context") or {}),
                imprint.get("tone_of_voice"),
            )

    async def get_imprint(self, agent_id: str) -> Optional[dict]:
        async with self._p().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM agent_imprints WHERE agent_id = $1", agent_id
            )
        return _row_to_dict(row)


def _parse_pg_vector(value) -> list[float]:
    """asyncpg returns pgvector values as text; parse '[0.1,0.2,...]'."""
    if value is None:
        return []
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, str):
        s = value.strip().strip("[]")
        if not s:
            return []
        try:
            return [float(x) for x in s.split(",")]
        except ValueError:
            return []
    return []


def _rowcount(result: str) -> int:
    """asyncpg `execute` returns a status string like 'DELETE 5'."""
    parts = (result or "").split()
    if len(parts) >= 2 and parts[-1].isdigit():
        return int(parts[-1])
    return 0
