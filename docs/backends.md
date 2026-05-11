# Storage backends

Two backends ship in v4. They're 100% behind the same interface — the
rest of the framework can't tell them apart.

| | SQLite | Postgres |
|-|--------|----------|
| Use case | Local agents, dev, CI, single-user prototypes | Shared service, multi-user production |
| Driver | `aiosqlite` + `sqlite-vec` extension | `asyncpg` + `pgvector` |
| ANN | `vec0` virtual table (brute-force cosine on top-K) | HNSW index on `vector(dim)` |
| 5-signal scoring | Python (`store/scoring.py`) | Python (same module) — re-scores after pgvector ANN |
| Concurrent writers | ~1 (WAL mode helps reads) | Many |
| Setup | `pip install "wild-memory[sqlite]"` | `pip install "wild-memory[postgres]"` + a Postgres with `pgvector` |
| Migrations | Jinja-templated SQL → `aiosqlite.executescript()` | Jinja-templated SQL → `asyncpg.execute()` |

## Picking one

- Just trying it out, or building a single-user assistant: SQLite.
- Shared system, more than one process, or you already have Postgres:
  Postgres.

You can swap by editing `wild_memory.yaml`'s `store:` section and
re-running `wild-memory migrate` against the new target. (Data does not
migrate automatically — that's a manual export/import.)

## Adding a third backend

1. Subclass `wild_memory.store.MemoryStore`. Implement every abstract
   method. The full list is in `wild_memory/store/base.py`.
2. Add it to the parity suite at `tests/store/test_parity.py` — the
   suite is parametrized over backend factories. Your new backend must
   produce the same retrieval ordering as the others on the shared
   fixtures.
3. If your backend needs a migration step, ship templates under
   `wild_memory/store/migrations/<your_backend>/` and register the
   directory in `[tool.setuptools.package-data]`.

## Embedding dimension

Both backends parameterize the embedding dimension at migration time:
the migration `.sql.j2` files render `vector({{ embedding_dim }})` (or
`float[{{ embedding_dim }}]` for sqlite-vec). The dimension is recorded
in `wild_memory_meta` and verified on every `connect()`. If you change
your embedding model to one with a different dimension, you must
re-migrate against a fresh database — the store will refuse to connect
to a database whose stored dim disagrees with `config.embedding.dimensions`.
