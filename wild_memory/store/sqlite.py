"""
SQLiteStore — local-first MemoryStore backed by SQLite + sqlite-vec.

Designed for development, single-machine agents, and the Studio Test Kits.
Uses `aiosqlite` for an async connection and the `sqlite-vec` extension for
ANN search over embeddings.

Lifecycle:
    store = SQLiteStore(":memory:", embedding_dim=1536)
    await store.connect()
    await store.migrate(embedding_dim=1536)
    ...
    await store.close()
"""
from __future__ import annotations

import json
import struct
import uuid
from datetime import datetime, timedelta, timezone
from importlib import resources
from typing import Any, Optional

import aiosqlite
import sqlite_vec
from jinja2 import Template

from wild_memory.store.base import (
    FeedbackSummaryRow,
    MemoryStore,
    RetrievalWeights,
    RetrievedObservation,
    SimilarObservation,
)
from wild_memory.store.scoring import score_observations


_JSON_FIELDS_OBSERVATION = ("entities",)
_JSON_FIELDS_ENTITY = ("attributes",)
_JSON_FIELDS_EDGE = ("properties",)
_JSON_FIELDS_PROCEDURE = ("steps", "trigger_entities")
_JSON_FIELDS_CITATION = ("used_observation_ids", "used_reflection_ids")
_JSON_FIELDS_SESSION = ("messages",)
_JSON_FIELDS_CHECKPOINT = ("working_memory", "active_procedure", "last_used_obs_ids")
_JSON_FIELDS_IMPRINT = ("values", "constraints", "org_context")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _new_id() -> str:
    return uuid.uuid4().hex


def _decode_row(row: aiosqlite.Row, json_fields: tuple[str, ...] = ()) -> dict:
    """Convert a Row into a plain dict, JSON-decoding the named fields."""
    if row is None:
        return None  # type: ignore[return-value]
    out = {k: row[k] for k in row.keys()}
    for f in json_fields:
        if f in out and isinstance(out[f], str):
            try:
                out[f] = json.loads(out[f])
            except json.JSONDecodeError:
                pass
    return out


def _encode_json(data: Any) -> str:
    return json.dumps(data, default=str)


def _decode_vec_blob(blob, dim: int) -> list[float]:
    """vec0 stores embeddings as packed little-endian float32; unpack to list."""
    if not blob:
        return []
    if isinstance(blob, (bytes, bytearray)):
        try:
            return list(struct.unpack(f"<{dim}f", blob))
        except struct.error:
            return []
    if isinstance(blob, str):
        # Some sqlite-vec versions roundtrip JSON text.
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            return []
    return []


class SQLiteStore(MemoryStore):
    """A MemoryStore backed by SQLite + sqlite-vec."""

    def __init__(self, path: str, *, embedding_dim: int):
        self._path = path
        self._dim = embedding_dim
        self._conn: Optional[aiosqlite.Connection] = None

    # ── Lifecycle ──────────────────────────────────────────────────

    async def connect(self) -> None:
        if self._conn is not None:
            return
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.enable_load_extension(True)
        await self._conn.load_extension(sqlite_vec.loadable_path())
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA foreign_keys=ON;")

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def migrate(self, *, embedding_dim: int) -> None:
        if self._conn is None:
            await self.connect()
        assert self._conn is not None

        if embedding_dim != self._dim:
            self._dim = embedding_dim

        files = ("001_schema.sql.j2", "002_indexes.sql")
        for fname in files:
            raw = (
                resources.files("wild_memory.store.migrations.sqlite")
                .joinpath(fname)
                .read_text()
            )
            sql = (
                Template(raw).render(embedding_dim=embedding_dim)
                if fname.endswith(".j2")
                else raw
            )
            await self._conn.executescript(sql)

        await self._conn.execute(
            "INSERT OR REPLACE INTO wild_memory_meta (key, value) VALUES (?, ?)",
            ("embedding_dim", str(embedding_dim)),
        )
        await self._conn.execute(
            "INSERT OR REPLACE INTO wild_memory_meta (key, value) VALUES (?, ?)",
            ("schema_version", "1"),
        )
        await self._conn.commit()

    async def health_check(self) -> dict:
        if self._conn is None:
            return {"backend": "sqlite", "ok": False, "error": "not connected"}
        cursor = await self._conn.execute("SELECT value FROM wild_memory_meta WHERE key='embedding_dim'")
        row = await cursor.fetchone()
        return {
            "backend": "sqlite",
            "ok": True,
            "path": self._path,
            "embedding_dim": int(row[0]) if row else None,
        }

    # ── Internal helpers ───────────────────────────────────────────

    def _c(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("SQLiteStore is not connected. Call connect() first.")
        return self._conn

    async def _commit(self) -> None:
        await self._c().commit()

    # ── Observations ───────────────────────────────────────────────

    async def insert_observation(self, data: dict) -> str:
        obs_id = data.get("id") or _new_id()
        embedding = data.get("embedding") or []
        row = {
            "id": obs_id,
            "agent_id": data["agent_id"],
            "user_id": data["user_id"],
            "content": data["content"],
            "obs_type": data["obs_type"],
            "entities": _encode_json(data.get("entities") or []),
            "importance": int(data.get("importance", 5)),
            "decay_score": float(data.get("decay_score", 1.0)),
            "ttl_days": int(data.get("ttl_days", 90)),
            "emotional_valence": data.get("emotional_valence", "neutral"),
            "emotional_intensity": int(data.get("emotional_intensity", 0)),
            "privacy_mode": data.get("privacy_mode", "personal"),
            "event_time": data.get("event_time"),
            "source_session": data.get("source_session"),
            "status": data.get("status", "active"),
        }
        await self._c().execute(
            """INSERT INTO observations
            (id, agent_id, user_id, content, obs_type, entities, importance,
             decay_score, ttl_days, emotional_valence, emotional_intensity,
             privacy_mode, event_time, source_session, status)
            VALUES
            (:id, :agent_id, :user_id, :content, :obs_type, :entities,
             :importance, :decay_score, :ttl_days, :emotional_valence,
             :emotional_intensity, :privacy_mode, :event_time,
             :source_session, :status)""",
            row,
        )
        if embedding:
            await self._c().execute(
                "INSERT OR REPLACE INTO observations_vec(id, embedding) VALUES (?, ?)",
                (obs_id, json.dumps(embedding)),
            )
        await self._commit()
        return obs_id

    async def update_observation(self, obs_id: str, fields: dict) -> None:
        if not fields:
            return
        # entities is the only JSON-typed field that may show up here
        clean: dict[str, Any] = {}
        for k, v in fields.items():
            clean[k] = _encode_json(v) if k == "entities" and not isinstance(v, str) else v
        sets = ", ".join(f"{k} = :{k}" for k in clean)
        clean["__id"] = obs_id
        await self._c().execute(
            f"UPDATE observations SET {sets} WHERE id = :__id", clean
        )
        await self._commit()

    async def get_observation(self, obs_id: str) -> Optional[dict]:
        cur = await self._c().execute(
            "SELECT * FROM observations WHERE id = ?", (obs_id,)
        )
        row = await cur.fetchone()
        return _decode_row(row, _JSON_FIELDS_OBSERVATION)

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
        clauses = ["agent_id = ?"]
        params: list[Any] = [agent_id]
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if obs_type is not None:
            clauses.append("obs_type = ?")
            params.append(obs_type)
        order = f"{order_by} {'DESC' if desc else 'ASC'}"
        sql = (
            f"SELECT * FROM observations WHERE {' AND '.join(clauses)} "
            f"ORDER BY {order} LIMIT ?"
        )
        params.append(limit)
        cur = await self._c().execute(sql, params)
        rows = await cur.fetchall()
        return [_decode_row(r, _JSON_FIELDS_OBSERVATION) for r in rows]

    async def list_active_user_ids(self, *, agent_id: str) -> list[str]:
        cur = await self._c().execute(
            "SELECT DISTINCT user_id FROM observations WHERE agent_id = ? AND status = 'active'",
            (agent_id,),
        )
        return [r[0] for r in await cur.fetchall()]

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
        # ANN: pull a wider candidate set, then re-score in Python.
        k = max(limit * 4, 20)
        cur = await self._c().execute(
            """
            SELECT o.*, vec.distance, vec.embedding AS vec_embedding
            FROM observations_vec AS vec
            JOIN observations AS o ON o.id = vec.id
            WHERE vec.embedding MATCH ?
              AND k = ?
              AND o.agent_id = ?
              AND o.user_id = ?
              AND o.status = 'active'
            ORDER BY vec.distance
            """,
            (json.dumps(embedding), k, agent_id, user_id),
        )
        rows = await cur.fetchall()
        candidates: list[dict] = []
        for r in rows:
            d = _decode_row(r, _JSON_FIELDS_OBSERVATION)
            d["embedding"] = _decode_vec_blob(d.get("vec_embedding"), self._dim)
            candidates.append(d)
        scored = score_observations(
            candidates,
            query_embedding=embedding,
            query_entities=entities,
            search_query=search_query,
            weights=weights,
            min_decay=min_decay,
        )
        # Reinforce on access.
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
        cur = await self._c().execute(
            """
            SELECT o.*, vec.distance
            FROM observations_vec AS vec
            JOIN observations AS o ON o.id = vec.id
            WHERE vec.embedding MATCH ?
              AND k = ?
              AND o.agent_id = ?
              AND o.user_id = ?
              AND o.status = 'active'
            ORDER BY vec.distance
            """,
            (json.dumps(embedding), limit, agent_id, user_id),
        )
        rows = await cur.fetchall()
        out: list[SimilarObservation] = []
        for r in rows:
            d = _decode_row(r, _JSON_FIELDS_OBSERVATION)
            # vec0 returns L2 distance by default; convert to a [0,1] similarity proxy.
            sim = 1.0 / (1.0 + float(d.get("distance", 0.0)))
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
        await self._c().execute(
            """UPDATE observations
               SET decay_score = MIN(1.0, decay_score + ?),
                   last_accessed = ?
               WHERE id = ?""",
            (boost, _now_iso(), obs_id),
        )
        await self._commit()

    async def apply_daily_decay(self, decay_rate: float) -> None:
        await self._c().execute(
            """UPDATE observations
               SET decay_score = MAX(0.0, decay_score - ?)
               WHERE status = 'active'""",
            (decay_rate,),
        )
        await self._commit()

    async def mark_stale_observations(self, threshold: float) -> None:
        await self._c().execute(
            """UPDATE observations SET status = 'stale'
               WHERE status = 'active' AND decay_score < ?""",
            (threshold,),
        )
        await self._commit()

    async def archive_low_decay_observations(
        self, threshold: float, protected_types: list[str]
    ) -> None:
        if protected_types:
            placeholders = ",".join(["?"] * len(protected_types))
            sql = (
                f"UPDATE observations SET status = 'archived' "
                f"WHERE status = 'active' AND decay_score < ? "
                f"AND obs_type NOT IN ({placeholders})"
            )
            await self._c().execute(sql, [threshold, *protected_types])
        else:
            await self._c().execute(
                "UPDATE observations SET status = 'archived' "
                "WHERE status = 'active' AND decay_score < ?",
                (threshold,),
            )
        await self._commit()

    async def anonymize_user_observations(
        self, user_id: str, anon_hash: str
    ) -> None:
        await self._c().execute(
            """UPDATE observations
               SET privacy_mode = 'pattern', user_id = 'anonymized',
                   anonymized_user_hash = ?, entities = '[]'
               WHERE user_id = ? AND privacy_mode = 'personal'""",
            (anon_hash, user_id),
        )
        await self._commit()

    async def purge_user_observations(self, user_id: str) -> None:
        await self._c().execute(
            "UPDATE observations SET status = 'purged' WHERE user_id = ?",
            (user_id,),
        )
        await self._commit()

    # ── Entities & Edges ───────────────────────────────────────────

    async def upsert_entity(
        self,
        *,
        entity_id: str,
        entity_type: str,
        display_name: str,
        attributes: dict,
    ) -> None:
        await self._c().execute(
            """INSERT INTO entity_nodes (id, entity_type, display_name, attributes)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 entity_type = excluded.entity_type,
                 display_name = excluded.display_name,
                 attributes = excluded.attributes""",
            (entity_id, entity_type, display_name, _encode_json(attributes)),
        )
        await self._commit()

    async def get_entity(self, entity_id: str) -> Optional[dict]:
        cur = await self._c().execute(
            "SELECT * FROM entity_nodes WHERE id = ?", (entity_id,)
        )
        row = await cur.fetchone()
        return _decode_row(row, _JSON_FIELDS_ENTITY)

    async def update_entity_attributes(
        self, entity_id: str, attributes: dict
    ) -> None:
        await self._c().execute(
            "UPDATE entity_nodes SET attributes = ? WHERE id = ?",
            (_encode_json(attributes), entity_id),
        )
        await self._commit()

    async def upsert_edge(
        self,
        *,
        subject_id: str,
        predicate: str,
        object_id: str,
        source_observation: Optional[str] = None,
        properties: Optional[dict] = None,
    ) -> None:
        await self._c().execute(
            """INSERT INTO entity_edges
               (id, subject_id, predicate, object_id, source_observation, properties)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(subject_id, predicate, object_id) DO UPDATE SET
                 source_observation = excluded.source_observation,
                 properties = excluded.properties""",
            (
                _new_id(),
                subject_id,
                predicate,
                object_id,
                source_observation,
                _encode_json(properties or {}),
            ),
        )
        await self._commit()

    async def list_edges_for_entity(self, entity_id: str) -> list[dict]:
        cur = await self._c().execute(
            "SELECT * FROM entity_edges WHERE subject_id = ? OR object_id = ?",
            (entity_id, entity_id),
        )
        rows = await cur.fetchall()
        return [_decode_row(r, _JSON_FIELDS_EDGE) for r in rows]

    # ── Reflections ────────────────────────────────────────────────

    async def insert_reflection(self, reflection: dict) -> str:
        ref_id = reflection.get("id") or _new_id()
        await self._c().execute(
            """INSERT INTO reflections
               (id, agent_id, user_id, reflection_type, content, importance)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                ref_id,
                reflection["agent_id"],
                reflection["user_id"],
                reflection["reflection_type"],
                reflection["content"],
                int(reflection.get("importance", 5)),
            ),
        )
        await self._commit()
        return ref_id

    async def list_reflections(
        self,
        *,
        agent_id: str,
        user_id: Optional[str] = None,
        limit: int = 10,
        order_by: str = "created_at",
        desc: bool = True,
    ) -> list[dict]:
        clauses = ["agent_id = ?"]
        params: list[Any] = [agent_id]
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        order = f"{order_by} {'DESC' if desc else 'ASC'}"
        sql = (
            f"SELECT * FROM reflections WHERE {' AND '.join(clauses)} "
            f"ORDER BY {order} LIMIT ?"
        )
        params.append(limit)
        cur = await self._c().execute(sql, params)
        rows = await cur.fetchall()
        return [_decode_row(r) for r in rows]

    async def purge_user_reflections(self, user_id: str) -> int:
        cur = await self._c().execute(
            "DELETE FROM reflections WHERE user_id = ?", (user_id,)
        )
        await self._commit()
        return cur.rowcount or 0

    # ── Feedback ───────────────────────────────────────────────────

    async def insert_feedback(self, signal: dict) -> str:
        sig_id = signal.get("id") or _new_id()
        await self._c().execute(
            """INSERT INTO feedback_signals
               (id, agent_id, user_id, session_id, signal_type, reward_score,
                action_taken, procedure_id, procedure_step, source, external_ref)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                sig_id,
                signal["agent_id"],
                signal["user_id"],
                signal["session_id"],
                signal["signal_type"],
                float(signal.get("reward_score", 0)),
                signal.get("action_taken"),
                signal.get("procedure_id"),
                signal.get("procedure_step"),
                signal.get("source", "implicit"),
                signal.get("external_ref"),
            ),
        )
        await self._commit()
        return sig_id

    async def list_session_feedback(self, session_id: str) -> list[dict]:
        cur = await self._c().execute(
            "SELECT * FROM feedback_signals WHERE session_id = ?",
            (session_id,),
        )
        rows = await cur.fetchall()
        return [_decode_row(r) for r in rows]

    async def feedback_summary(
        self, *, agent_id: str, days: int
    ) -> list[FeedbackSummaryRow]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cur = await self._c().execute(
            """SELECT signal_type, COUNT(*) AS n, AVG(reward_score) AS avg_r
               FROM feedback_signals
               WHERE agent_id = ? AND created_at >= ?
               GROUP BY signal_type""",
            (agent_id, cutoff),
        )
        rows = await cur.fetchall()
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
        cur = await self._c().execute(
            "DELETE FROM feedback_signals WHERE user_id = ?", (user_id,)
        )
        await self._commit()
        return cur.rowcount or 0

    # ── Procedures ─────────────────────────────────────────────────

    async def list_procedures(
        self, *, agent_id: str, status: str = "active"
    ) -> list[dict]:
        cur = await self._c().execute(
            "SELECT * FROM procedures WHERE agent_id = ? AND status = ?",
            (agent_id, status),
        )
        rows = await cur.fetchall()
        return [_decode_row(r, _JSON_FIELDS_PROCEDURE) for r in rows]

    async def get_procedure(self, procedure_id: str) -> Optional[dict]:
        cur = await self._c().execute(
            "SELECT * FROM procedures WHERE id = ?", (procedure_id,)
        )
        row = await cur.fetchone()
        return _decode_row(row, _JSON_FIELDS_PROCEDURE)

    async def update_procedure(
        self, procedure_id: str, fields: dict
    ) -> None:
        if not fields:
            return
        clean: dict[str, Any] = {}
        for k, v in fields.items():
            clean[k] = _encode_json(v) if k in _JSON_FIELDS_PROCEDURE else v
        sets = ", ".join(f"{k} = :{k}" for k in clean)
        clean["__id"] = procedure_id
        await self._c().execute(
            f"UPDATE procedures SET {sets} WHERE id = :__id", clean
        )
        await self._commit()

    # ── Citations ──────────────────────────────────────────────────

    async def insert_citation(self, citation: dict) -> str:
        cit_id = citation.get("id") or _new_id()
        await self._c().execute(
            """INSERT INTO citation_trails
               (id, agent_id, user_id, session_id, message_index,
                used_observation_ids, used_reflection_ids,
                active_procedure_id, active_procedure_step,
                n_sources, avg_decay_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                cit_id,
                citation["agent_id"],
                citation["user_id"],
                citation["session_id"],
                int(citation["message_index"]),
                _encode_json(citation.get("used_observation_ids") or []),
                _encode_json(citation.get("used_reflection_ids") or []),
                citation.get("active_procedure_id"),
                citation.get("active_procedure_step"),
                int(citation.get("n_sources", 0)),
                citation.get("avg_decay_score"),
            ),
        )
        await self._commit()
        return cit_id

    async def list_citations_for_session(
        self, session_id: str
    ) -> list[dict]:
        cur = await self._c().execute(
            "SELECT * FROM citation_trails WHERE session_id = ? ORDER BY message_index",
            (session_id,),
        )
        rows = await cur.fetchall()
        return [_decode_row(r, _JSON_FIELDS_CITATION) for r in rows]

    async def find_citations_using_observation(
        self, obs_id: str
    ) -> list[dict]:
        # SQLite has json_each — we use a LIKE for portability.
        cur = await self._c().execute(
            "SELECT session_id, message_index, created_at FROM citation_trails "
            "WHERE used_observation_ids LIKE ? ORDER BY created_at DESC",
            (f'%"{obs_id}"%',),
        )
        rows = await cur.fetchall()
        return [_decode_row(r) for r in rows]

    # ── Sessions ───────────────────────────────────────────────────

    async def get_session_log(self, session_id: str) -> Optional[dict]:
        cur = await self._c().execute(
            "SELECT id, messages FROM session_logs WHERE session_id = ?",
            (session_id,),
        )
        row = await cur.fetchone()
        return _decode_row(row, _JSON_FIELDS_SESSION)

    async def upsert_session_log(
        self,
        *,
        session_id: str,
        agent_id: str,
        user_id: str,
        messages: list[dict],
    ) -> None:
        existing = await self.get_session_log(session_id)
        if existing:
            merged = (existing.get("messages") or []) + messages
            await self._c().execute(
                "UPDATE session_logs SET messages = ? WHERE id = ?",
                (_encode_json(merged), existing["id"]),
            )
        else:
            await self._c().execute(
                """INSERT INTO session_logs
                   (id, session_id, agent_id, user_id, messages)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    _new_id(),
                    session_id,
                    agent_id,
                    user_id,
                    _encode_json(messages),
                ),
            )
        await self._commit()

    async def cleanup_expired_sessions(self) -> int:
        cur = await self._c().execute(
            "DELETE FROM session_logs WHERE expires_at IS NOT NULL AND expires_at < ?",
            (_now_iso(),),
        )
        await self._commit()
        return cur.rowcount or 0

    # ── Semantic cache ─────────────────────────────────────────────

    async def search_semantic_cache(
        self,
        *,
        agent_id: str,
        embedding: list[float],
        threshold: float,
    ) -> Optional[dict]:
        cur = await self._c().execute(
            """
            SELECT s.*, vec.distance
            FROM semantic_cache_vec AS vec
            JOIN semantic_cache AS s ON s.id = vec.id
            WHERE vec.embedding MATCH ?
              AND k = 1
              AND s.agent_id = ?
            ORDER BY vec.distance
            """,
            (json.dumps(embedding), agent_id),
        )
        row = await cur.fetchone()
        if not row:
            return None
        d = _decode_row(row)
        sim = 1.0 / (1.0 + float(d.get("distance", 0.0)))
        if sim < threshold:
            return None
        return d

    async def increment_semantic_cache_hit(self, cache_id: str) -> None:
        await self._c().execute(
            "UPDATE semantic_cache SET hit_count = hit_count + 1, last_hit = ? WHERE id = ?",
            (_now_iso(), cache_id),
        )
        await self._commit()

    async def insert_semantic_cache(
        self,
        *,
        agent_id: str,
        query_text: str,
        response_text: str,
        embedding: list[float],
        ttl_hours: int,
    ) -> None:
        cache_id = _new_id()
        expires = (
            datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
        ).isoformat()
        await self._c().execute(
            """INSERT INTO semantic_cache
               (id, agent_id, query_text, response_text, ttl_hours, expires_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (cache_id, agent_id, query_text, response_text, ttl_hours, expires),
        )
        await self._c().execute(
            "INSERT OR REPLACE INTO semantic_cache_vec(id, embedding) VALUES (?, ?)",
            (cache_id, json.dumps(embedding)),
        )
        await self._commit()

    async def cleanup_expired_cache(self) -> int:
        cur = await self._c().execute(
            "DELETE FROM semantic_cache WHERE expires_at IS NOT NULL AND expires_at < ?",
            (_now_iso(),),
        )
        await self._commit()
        return cur.rowcount or 0

    # ── Checkpoints ────────────────────────────────────────────────

    async def upsert_checkpoint(self, checkpoint: dict) -> None:
        await self._c().execute(
            """INSERT INTO agent_checkpoints
               (id, agent_id, session_id, working_memory, active_procedure,
                last_used_obs_ids, message_count)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(agent_id, session_id) DO UPDATE SET
                 working_memory = excluded.working_memory,
                 active_procedure = excluded.active_procedure,
                 last_used_obs_ids = excluded.last_used_obs_ids,
                 message_count = excluded.message_count""",
            (
                _new_id(),
                checkpoint["agent_id"],
                checkpoint["session_id"],
                _encode_json(checkpoint.get("working_memory") or {}),
                _encode_json(checkpoint.get("active_procedure")),
                _encode_json(checkpoint.get("last_used_obs_ids") or []),
                int(checkpoint.get("message_count", 0)),
            ),
        )
        await self._commit()

    async def get_checkpoint(
        self, *, agent_id: str, session_id: str
    ) -> Optional[dict]:
        cur = await self._c().execute(
            "SELECT * FROM agent_checkpoints WHERE agent_id = ? AND session_id = ?",
            (agent_id, session_id),
        )
        row = await cur.fetchone()
        return _decode_row(row, _JSON_FIELDS_CHECKPOINT)

    async def cleanup_old_checkpoints(self, hours: int) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        cur = await self._c().execute(
            "DELETE FROM agent_checkpoints WHERE created_at < ?", (cutoff,)
        )
        await self._commit()
        return cur.rowcount or 0

    # ── Imprints ───────────────────────────────────────────────────

    async def upsert_imprint(self, imprint: dict) -> None:
        await self._c().execute(
            """INSERT INTO agent_imprints
               (agent_id, role, "values", constraints, org_context, tone_of_voice, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(agent_id) DO UPDATE SET
                 role = excluded.role,
                 "values" = excluded."values",
                 constraints = excluded.constraints,
                 org_context = excluded.org_context,
                 tone_of_voice = excluded.tone_of_voice,
                 updated_at = excluded.updated_at""",
            (
                imprint["agent_id"],
                imprint.get("role"),
                _encode_json(imprint.get("values") or []),
                _encode_json(imprint.get("constraints") or []),
                _encode_json(imprint.get("org_context") or {}),
                imprint.get("tone_of_voice"),
                _now_iso(),
            ),
        )
        await self._commit()

    async def get_imprint(self, agent_id: str) -> Optional[dict]:
        cur = await self._c().execute(
            "SELECT * FROM agent_imprints WHERE agent_id = ?", (agent_id,)
        )
        row = await cur.fetchone()
        return _decode_row(row, _JSON_FIELDS_IMPRINT)
