"""
MemoryStore — the storage abstraction.

Every framework component that previously talked to a Supabase client now
talks to a `MemoryStore`. Backends translate the abstract method calls into
their native API (SQL, vector index, etc.).

Design notes:

- All methods are `async`. Backends that wrap sync drivers should still
  expose async signatures (and run blocking work in a thread pool when
  appropriate).
- Methods are grouped by entity. The grouping is purely cosmetic; subclasses
  can implement them in any order.
- Input dicts are intentionally schema-less to keep the interface small.
  Each backend documents the keys it actually persists. New columns can be
  added by passing extra keys without changing the interface.
- The 5-signal scoring used by `retrieve_observations` is single-sourced in
  `wild_memory.store.scoring`. Postgres ports it to a SQL function for
  performance; SQLite calls the Python implementation directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class RetrievalWeights:
    """Weights for the 5-signal combined retrieval score."""

    semantic: float = 0.30
    entity_match: float = 0.20
    fts_keyword: float = 0.15
    recency: float = 0.15
    decay: float = 0.12
    emotion: float = 0.08


@dataclass
class RetrievedObservation:
    """A single observation returned by `retrieve_observations`."""

    id: str
    content: str
    obs_type: str
    entities: list[str]
    importance: int
    decay_score: float
    event_time: Optional[datetime]
    created_at: datetime
    emotional_valence: str
    emotional_intensity: int
    combined_score: float
    raw: dict = field(default_factory=dict)


@dataclass
class SimilarObservation:
    """A single observation returned by `find_similar_observations`."""

    id: str
    content: str
    obs_type: str
    importance: int
    similarity: float
    raw: dict = field(default_factory=dict)


@dataclass
class FeedbackSummaryRow:
    """A single row in the feedback summary aggregation."""

    signal_type: str
    count: int
    avg_reward: float
    top_action: Optional[str] = None


class MemoryStore(ABC):
    """Backend-agnostic storage interface."""

    # ── Lifecycle ──────────────────────────────────────────────────

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def migrate(self, *, embedding_dim: int) -> None: ...

    @abstractmethod
    async def health_check(self) -> dict: ...

    # ── Observations (Bee) ─────────────────────────────────────────

    @abstractmethod
    async def insert_observation(self, data: dict) -> str: ...

    @abstractmethod
    async def update_observation(self, obs_id: str, fields: dict) -> None: ...

    @abstractmethod
    async def get_observation(self, obs_id: str) -> Optional[dict]: ...

    @abstractmethod
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
    ) -> list[dict]: ...

    @abstractmethod
    async def list_active_user_ids(self, *, agent_id: str) -> list[str]: ...

    @abstractmethod
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
    ) -> list[RetrievedObservation]: ...

    @abstractmethod
    async def find_similar_observations(
        self,
        *,
        agent_id: str,
        user_id: str,
        embedding: list[float],
        threshold: float,
        limit: int,
    ) -> list[SimilarObservation]: ...

    @abstractmethod
    async def reinforce_observation(self, obs_id: str, boost: float = 0.15) -> None: ...

    @abstractmethod
    async def apply_daily_decay(self, decay_rate: float) -> None: ...

    @abstractmethod
    async def mark_stale_observations(self, threshold: float) -> None: ...

    @abstractmethod
    async def archive_low_decay_observations(
        self, threshold: float, protected_types: list[str]
    ) -> None: ...

    @abstractmethod
    async def anonymize_user_observations(
        self, user_id: str, anon_hash: str
    ) -> None: ...

    @abstractmethod
    async def purge_user_observations(self, user_id: str) -> None: ...

    # ── Entities & Edges (Dolphin) ─────────────────────────────────

    @abstractmethod
    async def upsert_entity(
        self,
        *,
        entity_id: str,
        entity_type: str,
        display_name: str,
        attributes: dict,
    ) -> None: ...

    @abstractmethod
    async def get_entity(self, entity_id: str) -> Optional[dict]: ...

    @abstractmethod
    async def update_entity_attributes(
        self, entity_id: str, attributes: dict
    ) -> None: ...

    @abstractmethod
    async def upsert_edge(
        self,
        *,
        subject_id: str,
        predicate: str,
        object_id: str,
        source_observation: Optional[str] = None,
        properties: Optional[dict] = None,
    ) -> None: ...

    @abstractmethod
    async def list_edges_for_entity(self, entity_id: str) -> list[dict]: ...

    # ── Reflections (Ant) ──────────────────────────────────────────

    @abstractmethod
    async def insert_reflection(self, reflection: dict) -> str: ...

    @abstractmethod
    async def list_reflections(
        self,
        *,
        agent_id: str,
        user_id: Optional[str] = None,
        limit: int = 10,
        order_by: str = "created_at",
        desc: bool = True,
    ) -> list[dict]: ...

    @abstractmethod
    async def purge_user_reflections(self, user_id: str) -> int: ...

    # ── Feedback (Chameleon) ───────────────────────────────────────

    @abstractmethod
    async def insert_feedback(self, signal: dict) -> str: ...

    @abstractmethod
    async def list_session_feedback(self, session_id: str) -> list[dict]: ...

    @abstractmethod
    async def feedback_summary(
        self, *, agent_id: str, days: int
    ) -> list[FeedbackSummaryRow]: ...

    @abstractmethod
    async def purge_user_feedback(self, user_id: str) -> int: ...

    # ── Procedures (Chameleon) ─────────────────────────────────────

    @abstractmethod
    async def list_procedures(
        self, *, agent_id: str, status: str = "active"
    ) -> list[dict]: ...

    @abstractmethod
    async def get_procedure(self, procedure_id: str) -> Optional[dict]: ...

    @abstractmethod
    async def update_procedure(
        self, procedure_id: str, fields: dict
    ) -> None: ...

    # ── Citations (Elephant) ───────────────────────────────────────

    @abstractmethod
    async def insert_citation(self, citation: dict) -> str: ...

    @abstractmethod
    async def list_citations_for_session(
        self, session_id: str
    ) -> list[dict]: ...

    @abstractmethod
    async def find_citations_using_observation(
        self, obs_id: str
    ) -> list[dict]: ...

    # ── Sessions (Bee, raw log) ────────────────────────────────────

    @abstractmethod
    async def get_session_log(self, session_id: str) -> Optional[dict]: ...

    @abstractmethod
    async def upsert_session_log(
        self,
        *,
        session_id: str,
        agent_id: str,
        user_id: str,
        messages: list[dict],
    ) -> None: ...

    @abstractmethod
    async def cleanup_expired_sessions(self) -> int: ...

    # ── Semantic cache (Elephant) ──────────────────────────────────

    @abstractmethod
    async def search_semantic_cache(
        self,
        *,
        agent_id: str,
        embedding: list[float],
        threshold: float,
    ) -> Optional[dict]: ...

    @abstractmethod
    async def increment_semantic_cache_hit(self, cache_id: str) -> None: ...

    @abstractmethod
    async def insert_semantic_cache(
        self,
        *,
        agent_id: str,
        query_text: str,
        response_text: str,
        embedding: list[float],
        ttl_hours: int,
    ) -> None: ...

    @abstractmethod
    async def cleanup_expired_cache(self) -> int: ...

    # ── Checkpoints (Chameleon) ────────────────────────────────────

    @abstractmethod
    async def upsert_checkpoint(self, checkpoint: dict) -> None: ...

    @abstractmethod
    async def get_checkpoint(
        self, *, agent_id: str, session_id: str
    ) -> Optional[dict]: ...

    @abstractmethod
    async def cleanup_old_checkpoints(self, hours: int) -> int: ...

    # ── Imprints (Salmon, optional persistence) ────────────────────

    @abstractmethod
    async def upsert_imprint(self, imprint: dict) -> None: ...

    @abstractmethod
    async def get_imprint(self, agent_id: str) -> Optional[dict]: ...
