"""
Wild Memory storage backends.

The orchestrator and every layer/process/retrieval/audit/infra component
talk to a `MemoryStore` rather than a vendor SDK directly. Concrete backends
(SQLite, Postgres, etc.) implement the ABC defined in `base.py`.

The `LegacySupabaseStore` is a transition adapter that wraps an existing
Supabase client. It will be removed in Phase 4 once the asyncpg-based
`PostgresStore` lands.
"""

from wild_memory.store.base import (
    FeedbackSummaryRow,
    MemoryStore,
    RetrievalWeights,
    RetrievedObservation,
    SimilarObservation,
)

__all__ = [
    "FeedbackSummaryRow",
    "MemoryStore",
    "RetrievalWeights",
    "RetrievedObservation",
    "SimilarObservation",
]
