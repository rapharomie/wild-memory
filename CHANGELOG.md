# Changelog

All notable changes to Wild Memory will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.0a1] — OSS reset

A clean-break release that turns Wild Memory from an extracted product
component into a generally usable open-source memory framework. Anyone
can `pip install "wild-memory[sqlite,studio]"` and run the Test Kits
within a minute, without a cloud account.

### Added
- **`MemoryStore` interface** (`wild_memory/store/base.py`) with ~35
  methods grouped by entity. The orchestrator never writes SQL.
- **Two backends**:
  - `SQLiteStore` (`wild_memory/store/sqlite.py`) — `aiosqlite` +
    `sqlite-vec`. Local-first. Zero external services.
  - `PostgresStore` (`wild_memory/store/postgres.py`) — `asyncpg` +
    `pgvector`. Single-source 5-signal scoring shared with SQLite.
  - Both ship Jinja-templated migrations parameterized by
    `embedding_dim`; the `wild_memory_meta` table fail-fasts on dim
    mismatch at connect.
- **Provider abstraction** (`wild_memory/providers/`):
  - `LLMProvider` and `EmbeddingProvider` Protocols.
  - `AnthropicLLM` and `OpenAIEmbedding` adapters in-tree.
  - `ToolCall` / `ToolResult` / `LLMResponse` dataclasses — orchestrator
    never imports vendor SDKs.
- **Studio web UI** (`wild_memory/studio/`) — Flask blueprint at `/`
  with three Test Kits:
  - Kit 1 — *Funciona?* (smoke)
  - Kit 2 — *Funciona ao longo do tempo?* (decay, conflict, protection)
  - Kit 3 — *Aguenta volume?* (10 parallel users, zero leakage required)
  - Default to deterministic mock LLM/embedding so the green badge
    doesn't require API keys; opt into real providers with `--live`.
- **CLI**:
  - `wild-memory init` — scaffolds `wild_memory.yaml` + `memory/` from
    package-data templates.
  - `wild-memory migrate` — runs the configured backend's migration.
  - `wild-memory studio [--port N]` — launches the Studio.
  - `wild-memory test 1|2|3|all` — runs a kit in the terminal (CI).
- **Examples** (`examples/`): SQLite quickstart, Postgres example,
  custom-LLM example.
- **Docs** (`docs/`): quickstart, architecture, backends, providers.
- **OSS scaffolding**: `LICENSE` (MIT), `.gitignore`, `CONTRIBUTING.md`,
  `CHANGELOG.md`, `.github/workflows/ci.yml` matrix on Python
  3.10/3.11/3.12.

### Changed
- `WildMemory` constructor now requires an explicit `store=` argument.
- `WildMemory.from_config(...)` is now `async` and constructs the store
  from the YAML's `store:` section (or env vars `DATABASE_URL`,
  `WILD_MEMORY_DB_PATH`, `WILD_MEMORY_STORE_KIND`).
- Renamed router task names to be domain-neutral: `lead_conversation`
  → `agent_response`, `bee_distill` → `distillation`, etc.
- `DistillationGate` defaults are now empty (was Portuguese keywords);
  configure for your language and domain in `wild_memory.yaml`.
- `pyproject.toml` modernized with proper extras
  (`[postgres]`, `[sqlite]`, `[studio]`, `[scheduler]`, `[ner]`,
  `[dev]`, `[all]`), URLs, classifiers, package-data for migrations
  and templates.

### Removed
- Domain-specific MedReview / Closi-AI code
  (`init_medreview.py`, `medreview_domain.py`,
  `integration_examples/closi_adapter.py`, the rest of
  `integration_examples/` that imported it).
- The Supabase backend, `LegacySupabaseStore`, `wild_memory/infra/db.py`,
  the `supabase` dependency, and the old `migrations/` directory.
- The old supabase-coupled, MedReview-branded `dashboard/` (replaced by
  `studio/`).
- Empty stubs `wild_memory/sync/` and `wild_memory/privacy/`.
- Host-coupled tests (`tests/test_phase*.py`,
  `tests/test_wild_memory_setup.py`).
- `requirements.txt` (`pyproject.toml` is the single source of truth).
- Untracked nested `wild-memory/` directory.

### Tests
- 32 passing on default install + `[sqlite,studio]`. 5 Postgres parity
  tests skip cleanly without `DATABASE_URL`.

## [3.0.0] — Prior to OSS reset

Initial framework version, embedded inside the Closi-AI / MedReview project. Not published to PyPI.
