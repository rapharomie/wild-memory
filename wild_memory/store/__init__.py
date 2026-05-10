"""
Wild Memory storage backends.

The orchestrator and every layer/process/retrieval/audit/infra component
talk to a `MemoryStore` rather than a vendor SDK directly. Concrete backends
(SQLite, Postgres, etc.) implement the ABC defined in `base.py`.

Two backends ship in v4: `SQLiteStore` (local file or :memory: via
sqlite-vec) and `PostgresStore` (asyncpg + pgvector). Both are gated on
optional extras so the base install stays small.
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

# SQLiteStore is gated on the optional [sqlite] extra. Don't fail the import
# of the package if those deps are missing.
try:
    from wild_memory.store.sqlite import SQLiteStore  # noqa: F401
    __all__.append("SQLiteStore")
except ImportError:
    pass

# PostgresStore is gated on the optional [postgres] extra.
try:
    from wild_memory.store.postgres import PostgresStore  # noqa: F401
    __all__.append("PostgresStore")
except ImportError:
    pass
