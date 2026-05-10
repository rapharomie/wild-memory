"""
Test Kit runner.

Spins up an ephemeral SQLite sandbox + (optionally) mock providers,
invokes the requested kit, tears the sandbox down, and returns the
`KitReport`. Real LLM/embedding providers are used by default if the
right env vars are set; otherwise we fall back to deterministic mocks
so the kit always shows green to a first-time user without API keys.
"""
from __future__ import annotations

import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from wild_memory.providers.base import EmbeddingProvider, LLMProvider
from wild_memory.studio.kits.fakes import FakeEmbedding, FakeLLM
from wild_memory.studio.kits.reports import Check, KitReport


KitFn = Callable[..., Awaitable[KitReport]]


async def run_kit(
    kit_fn: KitFn,
    *,
    kit_id: str,
    title: str,
    embedding_dim: int = 64,
    use_mock: Optional[bool] = None,
    keep_db: bool = False,
) -> KitReport:
    """Run a single kit against a fresh sandbox SQLite + chosen providers.

    Args:
        kit_fn:        async function(store, embedding, llm) -> KitReport
        kit_id:        short identifier ('1' / '2' / '3')
        title:         human-readable kit title (echoed into the report)
        embedding_dim: dimension for the sandbox vectors. The mock embedding
                       respects this; with real providers, override to match.
        use_mock:      None = auto (mock if no API keys), True = force mock,
                       False = force real providers (raises if keys missing).
        keep_db:       leave the SQLite file on disk for inspection.

    Returns a KitReport. Never raises — internal exceptions are captured
    onto report.error so the UI can surface them.
    """
    from wild_memory.store.sqlite import SQLiteStore

    if use_mock is None:
        use_mock = not (
            os.getenv("ANTHROPIC_API_KEY") and os.getenv("OPENAI_API_KEY")
        )

    embedding: EmbeddingProvider
    llm: LLMProvider
    if use_mock:
        embedding = FakeEmbedding(dimensions=embedding_dim)
        llm = FakeLLM()
    else:
        from wild_memory.providers.anthropic_llm import AnthropicLLM
        from wild_memory.providers.openai_embedding import OpenAIEmbedding

        llm = AnthropicLLM()
        embedding = OpenAIEmbedding(dimensions=embedding_dim)

    tmp_dir = Path(tempfile.mkdtemp(prefix="wm_kit_"))
    db_path = tmp_dir / "sandbox.db"
    store = SQLiteStore(str(db_path), embedding_dim=embedding_dim)

    started = time.perf_counter()
    report: KitReport
    try:
        await store.connect()
        await store.migrate(embedding_dim=embedding_dim)
        report = await kit_fn(store=store, embedding=embedding, llm=llm)
    except Exception as e:
        report = KitReport(
            kit_id=kit_id,
            title=title,
            passed=False,
            duration_seconds=time.perf_counter() - started,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )
    finally:
        try:
            await store.close()
        except Exception:
            pass
        if not keep_db:
            try:
                db_path.unlink(missing_ok=True)
                tmp_dir.rmdir()
            except Exception:
                pass

    report.duration_seconds = time.perf_counter() - started
    report.metrics.setdefault("provider_mode", "mock" if use_mock else "live")
    report.metrics.setdefault("sandbox_path", str(db_path) if keep_db else "<deleted>")
    return report
