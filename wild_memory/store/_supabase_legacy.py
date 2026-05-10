"""
Legacy Supabase adapter.

Implements `MemoryStore` by delegating to an existing `supabase.Client`.
This is a transition adapter so that all consumer code can target the
`MemoryStore` interface today, without waiting for the Postgres+SQLite
backends. Phase 4 deletes this file together with the `supabase`
dependency in favor of the asyncpg-based `PostgresStore`.

Notes:

- Supabase's Python client is synchronous. We accept the sync-in-async
  anti-pattern here because (a) the rest of the framework is already async
  on top of this same client, and (b) this adapter is throwaway code.
- Method names map roughly 1:1 to the previous `self.db.table(...)/.rpc(...)`
  call sites; nothing fancy happens here.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from wild_memory.store.base import (
    FeedbackSummaryRow,
    MemoryStore,
    RetrievalWeights,
    RetrievedObservation,
    SimilarObservation,
)
from wild_memory.store.scoring import _to_retrieved


class LegacySupabaseStore(MemoryStore):
    """A MemoryStore backed by an existing supabase-py client."""

    def __init__(self, client: Any):
        self._client = client

    @property
    def client(self) -> Any:
        """Escape hatch: raw supabase client. Avoid using outside dashboard/."""
        return self._client

    # ── Lifecycle ──────────────────────────────────────────────────

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def migrate(self, *, embedding_dim: int) -> None:
        # Migrations are managed manually in Supabase Studio for the legacy
        # path. The asyncpg PostgresStore in Phase 4 will own this properly.
        return None

    async def health_check(self) -> dict:
        try:
            self._client.table("observations").select("id").limit(1).execute()
            return {"backend": "supabase-legacy", "ok": True}
        except Exception as e:  # pragma: no cover - depends on live DB
            return {"backend": "supabase-legacy", "ok": False, "error": str(e)}

    # ── Observations ───────────────────────────────────────────────

    async def insert_observation(self, data: dict) -> str:
        result = self._client.table("observations").insert(data).execute()
        return result.data[0]["id"]

    async def update_observation(self, obs_id: str, fields: dict) -> None:
        self._client.table("observations").update(fields).eq("id", obs_id).execute()

    async def get_observation(self, obs_id: str) -> Optional[dict]:
        result = (
            self._client.table("observations")
            .select("*")
            .eq("id", obs_id)
            .maybe_single()
            .execute()
        )
        return result.data

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
        q = self._client.table("observations").select("*").eq("agent_id", agent_id)
        if user_id is not None:
            q = q.eq("user_id", user_id)
        if status is not None:
            q = q.eq("status", status)
        if obs_type is not None:
            q = q.eq("obs_type", obs_type)
        result = q.order(order_by, desc=desc).limit(limit).execute()
        return result.data or []

    async def list_active_user_ids(self, *, agent_id: str) -> list[str]:
        result = (
            self._client.table("observations")
            .select("user_id")
            .eq("agent_id", agent_id)
            .eq("status", "active")
            .execute()
        )
        return list({row["user_id"] for row in (result.data or [])})

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
        result = self._client.rpc(
            "retrieve_observations",
            {
                "p_agent_id": agent_id,
                "p_user_id": user_id,
                "p_embedding": embedding,
                "p_entities": entities,
                "p_search_query": search_query,
                "p_limit": limit,
                "p_min_decay": min_decay,
            },
        ).execute()
        rows = result.data or []
        # The Postgres RPC returns rows already scored; reuse `_to_retrieved`.
        out: list[RetrievedObservation] = []
        for row in rows:
            score = float(row.get("combined_score", row.get("score", 0.0)))
            out.append(_to_retrieved(row, score))
            # Reinforce on access (matches previous behavior).
            try:
                self._client.rpc("reinforce_observation", {"obs_id": row["id"]}).execute()
            except Exception:
                pass
        return out

    async def find_similar_observations(
        self,
        *,
        agent_id: str,
        user_id: str,
        embedding: list[float],
        threshold: float,
        limit: int,
    ) -> list[SimilarObservation]:
        result = self._client.rpc(
            "find_similar_observations",
            {
                "p_agent_id": agent_id,
                "p_user_id": user_id,
                "p_embedding": embedding,
                "p_threshold": threshold,
                "p_limit": limit,
            },
        ).execute()
        out: list[SimilarObservation] = []
        for row in result.data or []:
            out.append(
                SimilarObservation(
                    id=str(row.get("id", "")),
                    content=row.get("content", ""),
                    obs_type=row.get("obs_type", "fact"),
                    importance=int(row.get("importance", 5)),
                    similarity=float(row.get("similarity", 0.0)),
                    raw=row,
                )
            )
        return out

    async def reinforce_observation(self, obs_id: str, boost: float = 0.15) -> None:
        self._client.rpc("reinforce_observation", {"obs_id": obs_id}).execute()

    async def apply_daily_decay(self, decay_rate: float) -> None:
        self._client.rpc("apply_daily_decay", {"decay_rate": decay_rate}).execute()

    async def mark_stale_observations(self, threshold: float) -> None:
        self._client.rpc(
            "mark_stale_observations", {"decay_threshold": threshold}
        ).execute()

    async def archive_low_decay_observations(
        self, threshold: float, protected_types: list[str]
    ) -> None:
        q = (
            self._client.table("observations")
            .update({"status": "archived"})
            .lt("decay_score", threshold)
            .eq("status", "active")
        )
        if protected_types:
            q = q.not_.in_("obs_type", protected_types)
        q.execute()

    async def anonymize_user_observations(
        self, user_id: str, anon_hash: str
    ) -> None:
        self._client.table("observations").update(
            {
                "privacy_mode": "pattern",
                "user_id": "anonymized",
                "anonymized_user_hash": anon_hash,
                "entities": [],
            }
        ).eq("user_id", user_id).eq("privacy_mode", "personal").execute()

    async def purge_user_observations(self, user_id: str) -> None:
        self._client.table("observations").update({"status": "purged"}).eq(
            "user_id", user_id
        ).execute()

    # ── Entities & Edges ───────────────────────────────────────────

    async def upsert_entity(
        self,
        *,
        entity_id: str,
        entity_type: str,
        display_name: str,
        attributes: dict,
    ) -> None:
        self._client.table("entity_nodes").upsert(
            {
                "id": entity_id,
                "entity_type": entity_type,
                "display_name": display_name,
                "attributes": attributes,
            }
        ).execute()

    async def get_entity(self, entity_id: str) -> Optional[dict]:
        result = (
            self._client.table("entity_nodes")
            .select("*")
            .eq("id", entity_id)
            .maybe_single()
            .execute()
        )
        return result.data

    async def update_entity_attributes(
        self, entity_id: str, attributes: dict
    ) -> None:
        self._client.table("entity_nodes").update(
            {"attributes": attributes}
        ).eq("id", entity_id).execute()

    async def upsert_edge(
        self,
        *,
        subject_id: str,
        predicate: str,
        object_id: str,
        source_observation: Optional[str] = None,
        properties: Optional[dict] = None,
    ) -> None:
        self._client.table("entity_edges").upsert(
            {
                "subject_id": subject_id,
                "predicate": predicate,
                "object_id": object_id,
                "source_observation": source_observation,
                "properties": properties or {},
            }
        ).execute()

    async def list_edges_for_entity(self, entity_id: str) -> list[dict]:
        result = (
            self._client.table("entity_edges")
            .select("*")
            .or_(f"subject_id.eq.{entity_id},object_id.eq.{entity_id}")
            .execute()
        )
        return result.data or []

    # ── Reflections ────────────────────────────────────────────────

    async def insert_reflection(self, reflection: dict) -> str:
        result = self._client.table("reflections").insert(reflection).execute()
        return result.data[0]["id"]

    async def list_reflections(
        self,
        *,
        agent_id: str,
        user_id: Optional[str] = None,
        limit: int = 10,
        order_by: str = "created_at",
        desc: bool = True,
    ) -> list[dict]:
        q = self._client.table("reflections").select("*").eq("agent_id", agent_id)
        if user_id is not None:
            q = q.eq("user_id", user_id)
        result = q.order(order_by, desc=desc).limit(limit).execute()
        return result.data or []

    async def purge_user_reflections(self, user_id: str) -> int:
        result = (
            self._client.table("reflections").delete().eq("user_id", user_id).execute()
        )
        return len(result.data or [])

    # ── Feedback ───────────────────────────────────────────────────

    async def insert_feedback(self, signal: dict) -> str:
        result = self._client.table("feedback_signals").insert(signal).execute()
        return result.data[0]["id"]

    async def list_session_feedback(self, session_id: str) -> list[dict]:
        result = (
            self._client.table("feedback_signals")
            .select("*")
            .eq("session_id", session_id)
            .execute()
        )
        return result.data or []

    async def feedback_summary(
        self, *, agent_id: str, days: int
    ) -> list[FeedbackSummaryRow]:
        result = self._client.rpc(
            "get_feedback_summary",
            {"p_agent_id": agent_id, "p_days": days},
        ).execute()
        out: list[FeedbackSummaryRow] = []
        for row in result.data or []:
            out.append(
                FeedbackSummaryRow(
                    signal_type=row.get("signal_type", ""),
                    count=int(row.get("count", 0)),
                    avg_reward=float(row.get("avg_reward", 0.0)),
                    top_action=row.get("top_action"),
                )
            )
        return out

    async def purge_user_feedback(self, user_id: str) -> int:
        result = (
            self._client.table("feedback_signals")
            .delete()
            .eq("user_id", user_id)
            .execute()
        )
        return len(result.data or [])

    # ── Procedures ─────────────────────────────────────────────────

    async def list_procedures(
        self, *, agent_id: str, status: str = "active"
    ) -> list[dict]:
        result = (
            self._client.table("procedures")
            .select("*")
            .eq("agent_id", agent_id)
            .eq("status", status)
            .execute()
        )
        return result.data or []

    async def get_procedure(self, procedure_id: str) -> Optional[dict]:
        result = (
            self._client.table("procedures")
            .select("*")
            .eq("id", procedure_id)
            .single()
            .execute()
        )
        return result.data

    async def update_procedure(
        self, procedure_id: str, fields: dict
    ) -> None:
        self._client.table("procedures").update(fields).eq("id", procedure_id).execute()

    # ── Citations ──────────────────────────────────────────────────

    async def insert_citation(self, citation: dict) -> str:
        result = self._client.table("citation_trails").insert(citation).execute()
        return result.data[0]["id"]

    async def list_citations_for_session(
        self, session_id: str
    ) -> list[dict]:
        result = (
            self._client.table("citation_trails")
            .select("*")
            .eq("session_id", session_id)
            .order("message_index")
            .execute()
        )
        return result.data or []

    async def find_citations_using_observation(
        self, obs_id: str
    ) -> list[dict]:
        result = (
            self._client.table("citation_trails")
            .select("session_id, message_index, created_at")
            .filter("used_observation_ids", "cs", "{" + obs_id + "}")
            .order("created_at", desc=True)
            .execute()
        )
        return result.data or []

    # ── Sessions ───────────────────────────────────────────────────

    async def get_session_log(self, session_id: str) -> Optional[dict]:
        result = (
            self._client.table("session_logs")
            .select("id, messages")
            .eq("session_id", session_id)
            .maybe_single()
            .execute()
        )
        return result.data

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
            merged = existing["messages"] + messages
            self._client.table("session_logs").update(
                {"messages": merged}
            ).eq("id", existing["id"]).execute()
        else:
            self._client.table("session_logs").insert(
                {
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "user_id": user_id,
                    "messages": messages,
                }
            ).execute()

    async def cleanup_expired_sessions(self) -> int:
        result = (
            self._client.table("session_logs")
            .delete()
            .lt("expires_at", datetime.now(timezone.utc).isoformat())
            .execute()
        )
        return len(result.data or [])

    # ── Semantic cache ─────────────────────────────────────────────

    async def search_semantic_cache(
        self,
        *,
        agent_id: str,
        embedding: list[float],
        threshold: float,
    ) -> Optional[dict]:
        result = self._client.rpc(
            "search_semantic_cache",
            {
                "p_agent_id": agent_id,
                "p_embedding": embedding,
                "p_threshold": threshold,
            },
        ).execute()
        if result.data:
            return result.data[0]
        return None

    async def increment_semantic_cache_hit(self, cache_id: str) -> None:
        # Read current hit_count then update.
        row = (
            self._client.table("semantic_cache")
            .select("hit_count")
            .eq("id", cache_id)
            .maybe_single()
            .execute()
        )
        new_count = (row.data["hit_count"] if row.data else 0) + 1
        self._client.table("semantic_cache").update(
            {
                "hit_count": new_count,
                "last_hit": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("id", cache_id).execute()

    async def insert_semantic_cache(
        self,
        *,
        agent_id: str,
        query_text: str,
        response_text: str,
        embedding: list[float],
        ttl_hours: int,
    ) -> None:
        self._client.table("semantic_cache").insert(
            {
                "agent_id": agent_id,
                "query_embedding": embedding,
                "query_text": query_text,
                "response_text": response_text,
                "ttl_hours": ttl_hours,
            }
        ).execute()

    async def cleanup_expired_cache(self) -> int:
        result = (
            self._client.table("semantic_cache")
            .delete()
            .lt("expires_at", datetime.now(timezone.utc).isoformat())
            .execute()
        )
        return len(result.data or [])

    # ── Checkpoints ────────────────────────────────────────────────

    async def upsert_checkpoint(self, checkpoint: dict) -> None:
        self._client.table("agent_checkpoints").upsert(checkpoint).execute()

    async def get_checkpoint(
        self, *, agent_id: str, session_id: str
    ) -> Optional[dict]:
        result = (
            self._client.table("agent_checkpoints")
            .select("*")
            .eq("agent_id", agent_id)
            .eq("session_id", session_id)
            .maybe_single()
            .execute()
        )
        return result.data

    async def cleanup_old_checkpoints(self, hours: int) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        result = (
            self._client.table("agent_checkpoints")
            .delete()
            .lt("created_at", cutoff)
            .execute()
        )
        return len(result.data or [])

    # ── Imprints ───────────────────────────────────────────────────

    async def upsert_imprint(self, imprint: dict) -> None:
        self._client.table("agent_imprints").upsert(imprint).execute()

    async def get_imprint(self, agent_id: str) -> Optional[dict]:
        result = (
            self._client.table("agent_imprints")
            .select("*")
            .eq("agent_id", agent_id)
            .maybe_single()
            .execute()
        )
        return result.data
