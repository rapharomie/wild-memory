"""
5-signal observation scoring — single source of truth.

The Postgres backend ports this algorithm to a SQL function for
performance. The SQLite backend calls this Python implementation directly
after retrieving the candidate set via sqlite-vec ANN search. Test parity
between backends is enforced by `tests/store/test_parity.py` (added in
Phase 3): the same fixture set must produce the same ranking on both.

The full implementation lands in Phase 3 alongside the SQLite backend.
This module currently exposes the signature and the weights dataclass so
the interface in `wild_memory.store.base` can reference it.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from wild_memory.store.base import RetrievalWeights, RetrievedObservation


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def entity_overlap(query_entities: list[str], obs_entities: list[str]) -> float:
    """Jaccard-style overlap normalized to [0, 1]."""
    if not query_entities or not obs_entities:
        return 0.0
    q = set(query_entities)
    o = set(obs_entities)
    inter = len(q & o)
    union = len(q | o)
    return inter / union if union else 0.0


def fts_keyword_match(search_query: str, content: str) -> float:
    """Toy keyword overlap. Phase 3 will replace with FTS5 rank."""
    if not search_query or not content:
        return 0.0
    q_terms = {t.lower() for t in search_query.split() if len(t) > 2}
    c_terms = {t.lower() for t in content.split() if len(t) > 2}
    if not q_terms:
        return 0.0
    return len(q_terms & c_terms) / len(q_terms)


def recency_score(created_at: datetime, now: Optional[datetime] = None) -> float:
    """Decay over 90 days, clamped to [0, 1]."""
    now = now or datetime.now(timezone.utc)
    if not created_at:
        return 0.0
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    age_days = (now - created_at).total_seconds() / 86400.0
    return max(0.0, 1.0 - (age_days / 90.0))


def emotion_score(valence: str, intensity: int) -> float:
    """Emotional weight ∈ [0, 1]: high-intensity non-neutral observations rank higher."""
    if not valence or valence == "neutral":
        return 0.0
    return min(intensity, 10) / 10.0


def score_observation(
    obs: dict,
    *,
    query_embedding: list[float],
    query_entities: list[str],
    search_query: str,
    weights: RetrievalWeights,
    now: Optional[datetime] = None,
) -> float:
    """Combined 5-signal score for a single observation row."""
    sem = cosine_similarity(query_embedding, obs.get("embedding") or [])
    ent = entity_overlap(query_entities, obs.get("entities") or [])
    fts = fts_keyword_match(search_query, obs.get("content") or "")
    rec = recency_score(_parse_dt(obs.get("created_at")), now)
    dec = float(obs.get("decay_score", 0.0))
    emo = emotion_score(
        obs.get("emotional_valence", "neutral"),
        int(obs.get("emotional_intensity", 0)),
    )
    return (
        weights.semantic * sem
        + weights.entity_match * ent
        + weights.fts_keyword * fts
        + weights.recency * rec
        + weights.decay * dec
        + weights.emotion * emo
    )


def score_observations(
    candidates: list[dict],
    *,
    query_embedding: list[float],
    query_entities: list[str],
    search_query: str,
    weights: RetrievalWeights,
    min_decay: float = 0.0,
    now: Optional[datetime] = None,
) -> list[RetrievedObservation]:
    """Score a candidate set and return them sorted by combined score (desc).

    Drops observations below `min_decay`. Used by the SQLite backend after the
    sqlite-vec ANN candidate search; Postgres uses an equivalent SQL function.
    """
    out: list[RetrievedObservation] = []
    for obs in candidates:
        if float(obs.get("decay_score", 0.0)) < min_decay:
            continue
        score = score_observation(
            obs,
            query_embedding=query_embedding,
            query_entities=query_entities,
            search_query=search_query,
            weights=weights,
            now=now,
        )
        out.append(_to_retrieved(obs, score))
    out.sort(key=lambda r: r.combined_score, reverse=True)
    return out


def _parse_dt(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _to_retrieved(obs: dict, score: float) -> RetrievedObservation:
    return RetrievedObservation(
        id=str(obs.get("id", "")),
        content=obs.get("content", ""),
        obs_type=obs.get("obs_type", "fact"),
        entities=list(obs.get("entities") or []),
        importance=int(obs.get("importance", 5)),
        decay_score=float(obs.get("decay_score", 0.0)),
        event_time=_parse_dt(obs.get("event_time")),
        created_at=_parse_dt(obs.get("created_at")) or datetime.now(timezone.utc),
        emotional_valence=obs.get("emotional_valence", "neutral"),
        emotional_intensity=int(obs.get("emotional_intensity", 0)),
        combined_score=score,
        raw=obs,
    )
