# Contributing to Wild Memory

Thanks for your interest in contributing. This document covers local setup, the project's design contracts, and the rules of engagement for changes.

## Local setup

```bash
git clone https://github.com/raphaelromie/wild-memory.git
cd wild-memory
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,sqlite,studio]"
```

Sanity check:

```bash
pytest tests/ -v
ruff check .
```

## Project shape

- `wild_memory/` is the package. Everything ships from here.
- `wild_memory/store/` is the storage abstraction layer (`MemoryStore` ABC + backend implementations).
- `wild_memory/providers/` is the LLM / embedding abstraction.
- `wild_memory/studio/` is the optional Flask-based inspection UI and the three built-in Test Kits.
- `tests/` mirrors the package layout. Parity tests at `tests/store/test_parity.py` are parametrized over every available backend.
- `examples/` is a small set of runnable examples that double as documentation. Keep them short.

## Adding a new storage backend

1. Subclass `wild_memory.store.base.MemoryStore`. Implement every abstract method.
2. Add it to the parity suite at `tests/store/test_parity.py`. Your backend must produce the same retrieval ordering as the reference scoring function for the shared dataset.
3. If your backend needs a migration step, ship templates under `wild_memory/store/migrations/<your_backend>/` and register them via `package-data` in `pyproject.toml`.

## Adding a new LLM or embedding provider

1. Implement `wild_memory.providers.base.LLMProvider` (or `EmbeddingProvider`). The whole point of the abstraction is that the orchestrator never talks directly to a vendor SDK.
2. Translate the vendor's response shape into `LLMResponse` (text + `ToolCall` list + `stop_reason`). Pin this with a fixture-based unit test under `tests/providers/`.
3. Document the env vars in `docs/providers.md`.

## Code style

- Ruff is the linter. CI fails on lint errors.
- Type hints required on new public functions. Internal helpers can omit them but please don't fight the type checker.
- Comments only when the *why* is non-obvious. The code should explain the *what*.

## Commit / PR

- Small, focused commits. One logical change per commit.
- Tests for new behavior. Tests for fixed bugs.
- PR description should explain *why* and call out anything reviewers should pay extra attention to.

## Reporting issues

- Reproduction steps + minimal code sample. The Studio "Trace" page can export a JSON of a single message run — attach that if it helps.
- For security issues, email rather than file a public issue. See `SECURITY.md` (TODO).
