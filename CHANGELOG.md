# Changelog

All notable changes to Wild Memory will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- LICENSE file (MIT).
- `.gitignore`, `CONTRIBUTING.md`, `CHANGELOG.md`.
- GitHub Actions CI workflow scaffold.

### Removed
- Domain-specific MedReview / Closi-AI code (`init_medreview.py`, `medreview_domain.py`, `integration_examples/closi_adapter.py`).
- Empty stubs `wild_memory/sync/` and `wild_memory/privacy/`.
- Duplicate root-level `dashboard/` (canonical lives at `wild_memory/dashboard/`).
- Host-coupled tests (`tests/test_phase*.py`, `tests/test_wild_memory_setup.py`).
- `requirements.txt` (`pyproject.toml` is the single source of truth for dependencies).
- Other `integration_examples/*.py` (all imported the deleted MedReview initializer).
- Untracked nested `wild-memory/` directory (a duplicate copy with its own `.git/`).

## [4.0.0] — Planned

A clean-break OSS release. See [docs/migration-from-v3.md] when published.

### Planned
- `MemoryStore` interface with two backends: Postgres+pgvector (`asyncpg`) and SQLite+`sqlite-vec`.
- `LLMProvider` and `EmbeddingProvider` Protocols replacing direct Anthropic/OpenAI coupling.
- Drop `supabase` dependency entirely.
- Studio web UI with three Test Kits (smoke, lifecycle, scale) for visually proving the framework works.
- Domain- and language-neutral defaults (no Portuguese/MedReview leakage).
- Dimension-parameterized SQL migrations (no more hardcoded `vector(1536)`).
- Proper packaging: migrations and templates ship in the wheel; `wild-memory init` and `wild-memory migrate` work post-install.

## [3.0.0] — Prior to OSS reset

Initial framework version, embedded inside the Closi-AI / MedReview project. Not published to PyPI.
