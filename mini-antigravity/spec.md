Here's the full specification for all 16 files of your local AI coding assistant.

---

# Local AI Coding Assistant — Complete System Specification

---

## `config.py`
**Purpose:** Central registry of all constants, defaults, and environment-derived settings used across the system.

### Constants & Values Defined

**Model Settings**
- `CHAT_MODEL`: `"qwen2.5-coder:7b"` — default Ollama model for chat/generation
- `EMBED_MODEL`: `"nomic-embed-text"` — default Ollama model for embeddings
- `EMBED_DIM`: `768` — fixed vector dimension for nomic-embed-text; all FAISS index construction and numpy array shapes assume this
- `OLLAMA_BASE_URL`: `"http://localhost:11434"` — Ollama API base; overridable via env var `OLLAMA_HOST`

**Search & Retrieval**
- `TOP_K`: `8` — default number of symbols to retrieve per query
- `FTS_WEIGHT`: `0.4` — weight assigned to normalized FTS5 score in hybrid merge
- `SEMANTIC_WEIGHT`: `0.6` — weight assigned to normalized FAISS cosine score in hybrid merge
- `MIN_SCORE_THRESHOLD`: `0.1` — symbols below this merged score are dropped

**Token Budget**
- `MAX_CONTEXT_FRACTION`: `0.30` — maximum fraction of model context window injectable as RAG context
- `MODEL_CONTEXT_WINDOW`: `8192` — assumed context window for the default chat model; configurable in settings
- `CHARS_PER_TOKEN`: `4` — approximate character-to-token ratio for budget estimation

**Paths**
- `DB_PATH`: `"data/codebase.db"` — SQLite database path
- `FAISS_INDEX_PATH`: `"data/faiss.index"` — persisted FAISS index file
- `BACKUP_DIR`: `"data/backups/"` — directory where file snapshots are stored
- `INDEX_ROOT`: `"."` — default root directory to scan when indexing

**Patch Settings**
- `PATCH_HISTORY_LIMIT`: `50` — max patch records retained in DB before pruning
- `BACKUP_RETENTION_LIMIT`: `100` — max backup records retained before pruning oldest

**Implementation Notes:**
- All other modules import from `config.py` only; no module hardcodes a constant.
- Settings that the user can override at runtime (Top-K, model names, token budget) are stored in the `settings` SQLite table and loaded at startup; `config.py` values serve as fallbacks when the table is empty or the key is missing.
- `OLLAMA_BASE_URL` checks `os.environ.get("OLLAMA_HOST")` at import time.

---

## `db.py`
**Purpose:** Owns every SQLite table definition, migration, and CRUD operation; all other modules interact with the database exclusively through this module.

### Imports
- `sqlite3` — database driver
- `numpy` — for serializing/deserializing float32 vectors to/from BLOBs
- `json` — for storing patch metadata dicts
- `pathlib.Path` — for ensuring `data/` directory exists before connecting
- `config` — for `DB_PATH`, `PATCH_HISTORY_LIMIT`, `BACKUP_RETENTION_LIMIT`

### Schema

**Table: `symbols`**
```
id          INTEGER PRIMARY KEY AUTOINCREMENT
file_path   TEXT NOT NULL
symbol_type TEXT NOT NULL  -- 'function' | 'class'
name        TEXT NOT NULL
signature   TEXT
docstring   TEXT
source      TEXT NOT NULL  -- full source text of the symbol
start_line  INTEGER
end_line    INTEGER
indexed_at  DATETIME DEFAULT CURRENT_TIMESTAMP
```

**Table: `embeddings`**
```
id           INTEGER PRIMARY KEY AUTOINCREMENT
symbol_id    INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE
embed_model  TEXT NOT NULL
vector_blob  BLOB NOT NULL   -- numpy float32 array, shape (768,), raw bytes
created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
```

**Table: `symbols_fts`** (FTS5 virtual table)
```
content='symbols'
content_rowid='id'
tokenize='porter unicode61'
Indexed columns: name, signature, docstring, source
```

**Table: `patches`**
```
id           INTEGER PRIMARY KEY AUTOINCREMENT
file_path    TEXT NOT NULL
diff_text    TEXT NOT NULL
status       TEXT NOT NULL  -- 'pending' | 'applied' | 'rejected'
created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
applied_at   DATETIME
metadata     TEXT  -- JSON blob: {prompt, model, symbols_used}
```

**Table: `backups`**
```
id           INTEGER PRIMARY KEY AUTOINCREMENT
file_path    TEXT NOT NULL
backup_path  TEXT NOT NULL  -- absolute path to snapshot file
patch_id     INTEGER REFERENCES patches(id)
created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
```

**Table: `settings`**
```
key    TEXT PRIMARY KEY
value  TEXT NOT NULL
```

---

### Functions

#### `get_connection() -> sqlite3.Connection`
- **Inputs:** none
- **Outputs:** open `sqlite3.Connection` with `row_factory = sqlite3.Row`, WAL mode enabled, foreign keys on
- **Description:** Opens the database at `DB_PATH`, ensuring parent directory exists. Sets `PRAGMA journal_mode=WAL` for concurrent read safety and `PRAGMA foreign_keys=ON`. Returns connection. Does not cache — callers manage connection lifetime. Every public function in `db.py` opens and closes its own connection via a context manager.

#### `init_db() -> None`
- **Inputs:** none
- **Outputs:** none (side effect: tables and indexes created if absent)
- **Description:** Called once at app startup. Executes `CREATE TABLE IF NOT EXISTS` for all five tables. Creates the FTS5 virtual table. Creates triggers: `AFTER INSERT ON symbols` fires `INSERT INTO symbols_fts`, `AFTER DELETE ON symbols` fires `DELETE FROM symbols_fts`, `AFTER UPDATE ON symbols` fires `UPDATE symbols_fts`. Creates a normal B-tree index on `symbols(file_path)` and `embeddings(symbol_id)`. If FTS5 is not available (rare SQLite builds), raises `RuntimeError` with a clear message.

#### `upsert_symbol(file_path, symbol_type, name, signature, docstring, source, start_line, end_line) -> int`
- **Inputs:** all fields of the `symbols` row except `id` and `indexed_at`
- **Outputs:** `int` — the `id` of the inserted or updated row
- **Description:** Checks for an existing row with the same `(file_path, name, symbol_type)`. If found, updates all mutable fields and returns its `id`. If not found, inserts and returns `lastrowid`. The FTS5 triggers handle FTS synchronization automatically. Returns the symbol id to the caller (indexer) so the embedding can be associated immediately.

#### `delete_symbols_for_file(file_path: str) -> None`
- **Inputs:** `file_path` — absolute or project-relative path
- **Outputs:** none
- **Description:** Deletes all rows from `symbols` where `file_path` matches. Cascade delete removes associated `embeddings` rows. FTS5 delete trigger fires for each deleted row. Called by `indexer.py` before re-indexing a file to avoid stale symbols.

#### `upsert_embedding(symbol_id: int, embed_model: str, vector: np.ndarray) -> None`
- **Inputs:** `symbol_id`, model name string, numpy float32 array of shape `(768,)`
- **Outputs:** none
- **Description:** Serializes the vector using `numpy.ndarray.tobytes()` (raw little-endian float32, 3072 bytes). Checks for an existing row with `(symbol_id, embed_model)`. If found, updates `vector_blob`. If not, inserts. Validates that `vector.shape == (768,)` and `vector.dtype == np.float32` before serializing; raises `ValueError` otherwise.

#### `get_all_embeddings(embed_model: str) -> list[dict]`
- **Inputs:** `embed_model` — filter to only return vectors from this model
- **Outputs:** list of dicts, each `{symbol_id: int, vector: np.ndarray}`, ordered by `symbol_id` ascending
- **Description:** Fetches all `(symbol_id, vector_blob)` rows from `embeddings` where `embed_model` matches. Deserializes each BLOB via `numpy.frombuffer(blob, dtype=np.float32)` and reshapes to `(768,)`. This is the primary feed into `embedder.py`'s FAISS index builder. If any blob has wrong byte length (not 3072), logs a warning and skips that row rather than crashing.

#### `get_symbol_by_id(symbol_id: int) -> dict | None`
- **Inputs:** `symbol_id`
- **Outputs:** dict with all `symbols` columns, or `None` if not found
- **Description:** Simple primary-key lookup. Returns `sqlite3.Row` cast to dict. Used by `retriever.py` to hydrate search results.

#### `get_symbols_by_ids(symbol_ids: list[int]) -> list[dict]`
- **Inputs:** list of symbol ids
- **Outputs:** list of dicts, preserving input order
- **Description:** Issues a single `SELECT ... WHERE id IN (...)` query. Re-orders results to match input order (SQL IN does not guarantee order). Used by the RAG context assembler.

#### `fts_search(query: str, limit: int) -> list[dict]`
- **Inputs:** `query` string, `limit` integer
- **Outputs:** list of dicts `{symbol_id: int, fts_score: float}`, sorted by score descending
- **Description:** Issues `SELECT rowid, rank FROM symbols_fts WHERE symbols_fts MATCH ? ORDER BY rank LIMIT ?`. SQLite FTS5 `rank` is a negative BM25 score; negates it to make higher = better. Returns raw unnormalized scores — normalization happens in `retriever.py`. Sanitizes `query` by stripping FTS5 special characters (`"`, `*`, `-` at word start) to prevent syntax errors. If the query produces an FTS5 syntax error, catches the exception and returns an empty list with a logged warning.

#### `save_patch(file_path, diff_text, metadata: dict) -> int`
- **Inputs:** file path, unified diff string, metadata dict
- **Outputs:** `int` — new patch `id`
- **Description:** Inserts a new row into `patches` with `status='pending'`, serializes `metadata` to JSON. Prunes oldest patches if total count exceeds `PATCH_HISTORY_LIMIT`. Returns new `id`.

#### `update_patch_status(patch_id: int, status: str, applied_at=None) -> None`
- **Inputs:** `patch_id`, `status` (one of `'applied'`, `'rejected'`), optional `applied_at` datetime
- **Outputs:** none
- **Description:** Updates the `status` and optionally `applied_at` fields. Validates `status` is in the allowed set; raises `ValueError` otherwise.

#### `get_patches(status_filter: str | None = None) -> list[dict]`
- **Inputs:** optional status string to filter
- **Outputs:** list of patch dicts, newest first
- **Description:** Returns all patches, or those matching `status_filter`. Deserializes `metadata` JSON field back to dict.

#### `save_backup(file_path, backup_path, patch_id: int | None) -> int`
- **Inputs:** original file path, backup snapshot path, optional associated patch id
- **Outputs:** new backup `id`
- **Description:** Inserts backup record. Prunes oldest backups if total count exceeds `BACKUP_RETENTION_LIMIT`.

#### `get_backups(file_path: str | None = None) -> list[dict]`
- **Inputs:** optional file path filter
- **Outputs:** list of backup dicts, newest first
- **Description:** Returns all backups, or those for a specific file.

#### `get_setting(key: str, default: str | None = None) -> str | None`
- **Inputs:** settings key, optional default
- **Outputs:** stored string value, or `default`
- **Description:** Single-row lookup on `settings` table.

#### `set_setting(key: str, value: str) -> None`
- **Inputs:** key, value (always string; callers cast as needed)
- **Outputs:** none
- **Description:** `INSERT OR REPLACE` into settings table.

#### `get_all_settings() -> dict[str, str]`
- **Inputs:** none
- **Outputs:** dict of all key-value pairs in settings table
- **Description:** Used at startup by `config.py` fallback logic and by `pages/settings.py` to pre-populate the UI.

**What Can Go Wrong:**
- FTS5 not compiled into the SQLite binary: `init_db` raises `RuntimeError` with instructions to install `pysqlite3-binary`.
- Concurrent writes from multiple Streamlit sessions: WAL mode handles concurrent reads well; writes are serialized by SQLite. If a write lock timeout occurs, catch `sqlite3.OperationalError` with "database is locked" and retry up to 3 times with 100ms sleep.
- BLOB deserialization with wrong length: skip and log rather than crash, so a single corrupt row doesn't break the entire FAISS rebuild.

---

## `embed.py`
**Purpose:** Thin wrapper around the Ollama `/api/embeddings` endpoint that generates a single 768-dim vector for a text input.

### Imports
- `requests` — HTTP POST to Ollama
- `numpy` — wraps response list into float32 array
- `config` — for `OLLAMA_BASE_URL`, `EMBED_MODEL`, `EMBED_DIM`

### Functions

#### `generate_embedding(text: str, model: str = config.EMBED_MODEL) -> np.ndarray`
- **Inputs:** `text` string to embed, optional model override
- **Outputs:** numpy float32 array of shape `(768,)`
- **Description:** POSTs `{"model": model, "prompt": text}` to `{OLLAMA_BASE_URL}/api/embeddings`. Parses the `"embedding"` field from the JSON response. Converts to `numpy.array(..., dtype=np.float32)`. Validates shape is `(768,)`. If Ollama returns a different dimension (model mismatch), raises `ValueError` with the actual dimension for debugging. Strips the text of null bytes before sending, as these can cause Ollama to return errors.

#### `batch_generate_embeddings(texts: list[str], model: str = config.EMBED_MODEL) -> list[np.ndarray]`
- **Inputs:** list of text strings, optional model override
- **Outputs:** list of numpy arrays, same length and order as input
- **Description:** Calls `generate_embedding` sequentially (Ollama does not have a native batch endpoint). Logs progress every 10 items (useful for large codebases). On failure for a single item, logs the error, appends a zero vector of the correct shape as a placeholder, and continues. Callers must be aware that a zero vector will produce meaningless similarity scores; `indexer.py` checks for zero vectors and skips storing them.

**What Can Go Wrong:**
- Ollama not running: `requests.ConnectionError` — caught, re-raised as `RuntimeError("Ollama not reachable at {URL}. Is it running?")`.
- Model not pulled: HTTP 404 or error in response body — caught, re-raised with model name in message.
- Network timeout: sets `timeout=30` on the request; raises `TimeoutError` if exceeded.
- Empty text: if `text.strip() == ""`, returns zero vector immediately without calling Ollama.

---

## `indexer.py`
**Purpose:** Scans Python files, parses them with the AST module to extract symbols, then orchestrates embedding generation, SQLite storage, and FTS5 indexing.

### Imports
- `ast` — Python AST parsing
- `pathlib.Path` — file discovery
- `textwrap.dedent` — normalize source indentation
- `db` — `delete_symbols_for_file`, `upsert_symbol`, `upsert_embedding`, `get_connection`
- `embed` — `generate_embedding`
- `embedder` — `rebuild_faiss_index` (called after all files are indexed)
- `config` — `INDEX_ROOT`, `EMBED_MODEL`

### Functions

#### `discover_python_files(root: str) -> list[Path]`
- **Inputs:** `root` directory path string
- **Outputs:** list of `Path` objects for all `.py` files under root
- **Description:** Uses `Path(root).rglob("*.py")`. Excludes paths containing `__pycache__`, `.git`, `.venv`, `venv`, `site-packages`, `node_modules`. Returns sorted list for deterministic ordering. If root does not exist, raises `FileNotFoundError`.

#### `extract_symbols(file_path: Path) -> list[dict]`
- **Inputs:** `Path` to a `.py` file
- **Outputs:** list of symbol dicts, each with keys: `symbol_type`, `name`, `signature`, `docstring`, `source`, `start_line`, `end_line`
- **Description:** Reads file content with UTF-8 encoding (falls back to latin-1 on decode error). Calls `ast.parse(source)`. Walks the AST. For each `ast.FunctionDef` and `ast.AsyncFunctionDef`, extracts: name, argument names and annotations as a formatted signature string (e.g., `"def foo(x: int, y: str = 'hello') -> bool"`), the docstring via `ast.get_docstring()`, and the raw source lines using `start_line`/`end_line` from the AST node. For each `ast.ClassDef`, extracts: class name, base class names as signature, class-level docstring, and the full class source. Nested functions inside classes are extracted as separate symbols with `name` = `"ClassName.method_name"` for disambiguation. If `ast.parse` raises `SyntaxError`, logs the file and error, returns empty list (file is skipped, not fatal).

#### `build_embed_text(symbol: dict) -> str`
- **Inputs:** symbol dict from `extract_symbols`
- **Outputs:** single string concatenating signature, docstring (if present), and source, separated by newlines
- **Description:** This is the text that gets embedded. Format: `"{signature}\n\n{docstring}\n\n{source}"`. If docstring is None, omits that section. Truncates total length to 2000 characters to avoid embedding extremely long class bodies, taking the first 1000 and last 1000 characters if over limit. The truncation strategy ensures the signature is always included.

#### `index_file(file_path: Path, embed_model: str = config.EMBED_MODEL) -> dict`
- **Inputs:** `Path` to a Python file, optional model name
- **Outputs:** dict `{file: str, symbols_found: int, symbols_indexed: int, errors: list[str]}`
- **Description:** Calls `delete_symbols_for_file` to clear stale entries. Calls `extract_symbols`. For each symbol: calls `upsert_symbol` to get `symbol_id`, calls `build_embed_text`, calls `generate_embedding`, checks for zero vector (skips if zero), calls `upsert_embedding`. Tracks errors per-symbol without stopping. Returns summary dict. The file_path stored in the DB is always normalized to a string relative to `INDEX_ROOT` for portability.

#### `index_directory(root: str, embed_model: str = config.EMBED_MODEL, progress_callback=None) -> dict`
- **Inputs:** root directory string, optional model override, optional callback `fn(current: int, total: int, file: str)` for UI progress
- **Outputs:** dict `{total_files: int, total_symbols: int, total_errors: int, per_file: list[dict]}`
- **Description:** Calls `discover_python_files`. Iterates files, calling `index_file` for each. Calls `progress_callback` after each file if provided. After all files, calls `embedder.rebuild_faiss_index()` to rebuild the FAISS index from the freshly updated DB. Returns aggregate summary.

**What Can Go Wrong:**
- File deleted between discovery and indexing: `FileNotFoundError` caught per-file, counted as error, indexing continues.
- AST parse failure: handled inside `extract_symbols`, returns empty list.
- Embedding failure (Ollama down): `generate_embedding` raises; caught in `index_file`, recorded in errors list, that symbol is skipped.
- Very large files with thousands of functions: no hard limit, but `build_embed_text` truncation keeps embedding calls reasonable.

---

## `embedder.py`
**Purpose:** Builds, persists, loads, and queries the FAISS flat inner-product index derived from vectors stored in SQLite.

### Imports
- `faiss` — index construction and search
- `numpy` — matrix construction for FAISS
- `pathlib.Path` — index file I/O
- `db` — `get_all_embeddings`
- `config` — `FAISS_INDEX_PATH`, `EMBED_DIM`, `EMBED_MODEL`

### Internal State
- Module-level variable `_index: faiss.IndexFlatIP | None = None` — cached loaded index
- Module-level variable `_id_map: list[int] = []` — ordered list of `symbol_id` values corresponding to FAISS row positions; FAISS integer IDs map to this list's indices

### Functions

#### `rebuild_faiss_index(embed_model: str = config.EMBED_MODEL) -> int`
- **Inputs:** optional embed model name filter
- **Outputs:** `int` — number of vectors indexed
- **Description:** Calls `db.get_all_embeddings(embed_model)` to get all `{symbol_id, vector}` pairs. Sorts by `symbol_id` for determinism. Builds a numpy matrix of shape `(N, 768)` from the vectors. Normalizes each row to unit length using `faiss.normalize_L2(matrix)` — this converts inner product to cosine similarity. Creates `faiss.IndexFlatIP(768)`. Calls `index.add(matrix)`. Saves the index to `FAISS_INDEX_PATH` using `faiss.write_index`. Saves the `_id_map` (list of symbol_ids in FAISS row order) as a companion `.ids` file (JSON lines). Updates the module-level `_index` and `_id_map` cache. Returns count of vectors added. If `N == 0`, writes an empty index (valid FAISS empty state) and returns 0.

#### `load_faiss_index(embed_model: str = config.EMBED_MODEL) -> bool`
- **Inputs:** optional model name (used to check if a rebuild is needed)
- **Outputs:** `bool` — True if loaded successfully
- **Description:** Reads `FAISS_INDEX_PATH` and companion `.ids` file. Loads them into module-level `_index` and `_id_map`. If either file is missing, calls `rebuild_faiss_index` automatically and returns True. If loading raises any exception (corrupted file), also calls `rebuild_faiss_index`. This means the index is always available after this call.

#### `get_index() -> tuple[faiss.IndexFlatIP, list[int]]`
- **Inputs:** none
- **Outputs:** `(index, id_map)` — the loaded FAISS index and symbol_id mapping list
- **Description:** Returns cached `_index` and `_id_map`. If `_index` is None (not yet loaded), calls `load_faiss_index()` first. This is the lazy-load entry point used by `retriever.py`.

#### `semantic_search(query_vector: np.ndarray, top_k: int) -> list[dict]`
- **Inputs:** query vector of shape `(768,)` as float32, `top_k` integer
- **Outputs:** list of dicts `{symbol_id: int, cosine_score: float}`, length ≤ `top_k`, sorted by score descending
- **Description:** Normalizes `query_vector` to unit length. Reshapes to `(1, 768)`. Calls `index.search(query_matrix, top_k)`. Gets back `distances` (cosine similarities in `[-1, 1]`) and `indices` (FAISS row numbers). Maps FAISS row indices through `_id_map` to get `symbol_id` values. Filters out rows where FAISS returned `-1` (empty index or fewer results than `top_k`). Returns list of `{symbol_id, cosine_score}` dicts. If index is empty (`.ntotal == 0`), returns empty list immediately.

**What Can Go Wrong:**
- FAISS index/id_map out of sync (different number of rows): detected if `len(_id_map) != index.ntotal`; triggers automatic `rebuild_faiss_index`.
- FAISS not installed: `ImportError` at module load; `app.py` catches and displays a clear error message with install instructions.
- Corrupt index file: caught in `load_faiss_index`, triggers rebuild.

---

## `retriever.py`
**Purpose:** Executes hybrid search combining FTS5 keyword scores and FAISS cosine scores, normalizes both onto a 0–1 scale, merges them, and returns ranked symbol results.

### Imports
- `numpy` — for normalization math
- `db` — `fts_search`, `get_symbol_by_id`, `get_symbols_by_ids`
- `embed` — `generate_embedding`
- `embedder` — `semantic_search`
- `config` — `TOP_K`, `FTS_WEIGHT`, `SEMANTIC_WEIGHT`, `MIN_SCORE_THRESHOLD`, `EMBED_MODEL`

### Functions

#### `normalize_scores(scores: list[float]) -> list[float]`
- **Inputs:** list of raw scores (all non-negative assumed for FTS; cosine can be negative)
- **Outputs:** list of floats in `[0, 1]`
- **Description:** Min-max normalization: `(x - min) / (max - min)`. If all values are equal or list is empty, returns list of 1.0s (or empty list). Does not modify the input list. This function is used independently for FTS scores and cosine scores before merging.

#### `hybrid_search(query: str, top_k: int = config.TOP_K, embed_model: str = config.EMBED_MODEL) -> list[dict]`
- **Inputs:** natural language query string, `top_k`, optional embed model
- **Outputs:** list of dicts, each: `{symbol_id: int, name: str, file_path: str, symbol_type: str, signature: str, docstring: str, source: str, fts_score: float, semantic_score: float, merged_score: float}`, sorted by `merged_score` descending, length ≤ `top_k`
- **Description:**
  1. Calls `db.fts_search(query, limit=top_k * 2)` — fetches more than `top_k` to allow for post-merge re-ranking.
  2. Calls `embed.generate_embedding(query, embed_model)` to get query vector.
  3. Calls `embedder.semantic_search(query_vector, top_k=top_k * 2)`.
  4. Builds a dict keyed by `symbol_id` collecting both raw scores (default 0.0 if absent in one source).
  5. Normalizes FTS scores across the FTS result set using `normalize_scores`.
  6. Normalizes cosine scores across the semantic result set using `normalize_scores`.
  7. For each unique symbol_id in the union: `merged = FTS_WEIGHT * norm_fts + SEMANTIC_WEIGHT * norm_semantic`.
  8. Filters out symbols where `merged < MIN_SCORE_THRESHOLD`.
  9. Sorts by `merged_score` descending, takes top `top_k`.
  10. Calls `db.get_symbols_by_ids` to hydrate full symbol data.
  11. Merges score fields into each symbol dict and returns.
- **Critical note:** Normalization is done per-result-set independently before merging, not globally. This means a symbol that scores 1.0 in FTS but 0.0 in semantic gets `FTS_WEIGHT * 1.0 + SEMANTIC_WEIGHT * 0.0 = 0.4` merged score, which correctly reflects it was a keyword-only match.

#### `keyword_only_search(query: str, top_k: int) -> list[dict]`
- **Inputs:** query string, `top_k`
- **Outputs:** same structure as `hybrid_search`, but `semantic_score` is always 0.0
- **Description:** Calls only `db.fts_search`. Normalizes FTS scores. Hydrates symbols. Used as fallback when Ollama is unreachable.

#### `semantic_only_search(query_vector: np.ndarray, top_k: int) -> list[dict]`
- **Inputs:** pre-computed query vector, `top_k`
- **Outputs:** same structure as `hybrid_search`, but `fts_score` is always 0.0
- **Description:** Calls only `embedder.semantic_search`. Hydrates symbols. Used internally when explicitly requested.

**What Can Go Wrong:**
- Embedding fails (Ollama down): `hybrid_search` catches the exception, falls back to `keyword_only_search`, and returns results with a flag `{"fallback_to_fts": True}` injected into each result dict so the UI can warn the user.
- FTS returns no results, semantic returns no results: returns empty list; callers must handle gracefully.
- Symbol in search results deleted from DB between search and hydration: `get_symbols_by_ids` returns None for missing ids; these are silently filtered out of the final result list.

---

## `context_manager.py`
**Purpose:** Enforces the token budget, assembles the RAG context string that is injected into the LLM system prompt, and provides transparency metadata about what was included.

### Imports
- `config` — `MAX_CONTEXT_FRACTION`, `MODEL_CONTEXT_WINDOW`, `CHARS_PER_TOKEN`
- `db` — `get_setting` (to read user-configured context window size)

### Functions

#### `estimate_tokens(text: str) -> int`
- **Inputs:** any string
- **Outputs:** integer token estimate
- **Description:** Returns `len(text) // CHARS_PER_TOKEN`. Intentionally conservative (underestimates). Used for budget checking, not for exact billing.

#### `get_token_budget() -> int`
- **Inputs:** none
- **Outputs:** integer — max tokens available for injected context
- **Description:** Reads `MODEL_CONTEXT_WINDOW` from settings DB (falls back to `config.MODEL_CONTEXT_WINDOW`). Returns `int(context_window * MAX_CONTEXT_FRACTION)`. For the default 8192-token window, this is 2457 tokens.

#### `assemble_context(symbols: list[dict], budget_tokens: int | None = None) -> dict`
- **Inputs:** ordered list of symbol dicts (from `retriever.py`), optional budget override
- **Outputs:** dict with keys:
  - `context_str: str` — the formatted string to inject into the system prompt
  - `included: list[dict]` — symbol dicts that fit within budget, with their scores
  - `excluded: list[dict]` — symbols dropped due to budget, with reason `"token_budget"`
  - `total_tokens_used: int`
  - `budget_tokens: int`
- **Description:** If `budget_tokens` is None, calls `get_token_budget()`. Iterates symbols in order (already ranked by merged_score). For each, formats a block: `"### {symbol_type}: {name}\n# File: {file_path}, Lines {start_line}-{end_line}\n# Score: {merged_score:.3f}\n{signature}\n\n{docstring}\n\n{source}\n---\n"`. Estimates tokens for the block. If `used + block_tokens <= budget_tokens`, includes it. Otherwise, tries truncating the `source` field to fit (keeps signature + docstring, truncates source to remaining budget). If even the header doesn't fit, marks as excluded. Returns the assembled dict. The `included` list preserves score metadata for UI display.

#### `format_system_prompt(base_prompt: str, context: dict) -> str`
- **Inputs:** base system prompt string (the instructional part), context dict from `assemble_context`
- **Outputs:** complete system prompt string
- **Description:** Combines: `"{base_prompt}\n\n## Retrieved Code Context\nThe following symbols were retrieved from the codebase based on your query:\n\n{context['context_str']}"`. If `context['context_str']` is empty (no symbols retrieved), omits the context section entirely rather than injecting a useless header.

**What Can Go Wrong:**
- All symbols exceed budget individually: each symbol's header + signature is typically ~50 tokens; only pathological one-liner codebases would trigger this. In that case, `assemble_context` returns an empty `included` list and explains in `context_str` that all symbols were too large.
- Settings DB unavailable: falls back to `config.MODEL_CONTEXT_WINDOW` silently.

---

## `llm.py`
**Purpose:** Wraps the Ollama chat completion API, handles streaming responses, injects the assembled RAG context into the system prompt, and returns token-annotated completions.

### Imports
- `requests` — streaming HTTP POST
- `json` — parse streaming NDJSON response
- `config` — `OLLAMA_BASE_URL`, `CHAT_MODEL`
- `db` — `get_setting` (for model override)
- `context_manager` — `format_system_prompt`

### Functions

#### `get_chat_model() -> str`
- **Inputs:** none
- **Outputs:** model name string
- **Description:** Reads `chat_model` from settings DB, falls back to `config.CHAT_MODEL`.

#### `stream_chat(messages: list[dict], system_prompt: str, model: str | None = None) -> Generator[str, None, None]`
- **Inputs:** `messages` — list of `{"role": "user"|"assistant", "content": str}` dicts representing conversation history; `system_prompt` — fully assembled system prompt string; optional `model` override
- **Outputs:** generator yielding string chunks of the assistant's response
- **Description:** Builds the request body: `{"model": model, "messages": [{"role": "system", "content": system_prompt}] + messages, "stream": True}`. POSTs to `{OLLAMA_BASE_URL}/api/chat` with `stream=True`. Iterates response lines. Each line is a JSON object `{"message": {"content": "..."}, "done": bool}`. Yields `message.content` for each non-done line. When `done=True`, extracts and returns (but does not yield) usage stats if present. Raises `StopIteration` after done. On connection error, yields a single error string `"[ERROR: Ollama unreachable]"` and stops — this propagates to the UI as visible error text rather than a crash.

#### `chat_with_context(query: str, history: list[dict], retrieved_symbols: list[dict], base_system_prompt: str, model: str | None = None) -> Generator[str, None, None]`
- **Inputs:** current user query, conversation history (not including current query), retrieved symbols from `retriever.py`, base system prompt, optional model
- **Outputs:** generator of response string chunks
- **Description:** Calls `context_manager.assemble_context(retrieved_symbols)` to get the context dict. Calls `context_manager.format_system_prompt(base_system_prompt, context)` to get the full system prompt. Appends `{"role": "user", "content": query}` to history. Calls `stream_chat(messages=history + [user_msg], system_prompt=full_system_prompt, model=model)`. Yields all chunks from the stream. This is the single entry point for all chat interactions from `pages/chat.py`.

#### `generate_patch(prompt: str, model: str | None = None) -> str`
- **Inputs:** full patch-generation prompt (assembled by `patcher.py`), optional model
- **Outputs:** complete LLM response string (non-streaming)
- **Description:** Posts to `/api/generate` (not `/api/chat`) with `stream=False` for simplicity. Returns the full `response` field. Used by `patcher.py` for structured patch output where streaming is less useful than a complete parseable response.

**What Can Go Wrong:**
- Ollama returns malformed JSON in stream: caught per-line, that chunk is skipped, a warning is logged, streaming continues.
- Context window exceeded: Ollama returns an error in the done message. `stream_chat` detects this (error field in final JSON) and yields `"[ERROR: Context window exceeded]"`.
- History grows unbounded over long conversations: `pages/chat.py` is responsible for truncating history before passing to `chat_with_context`; `llm.py` does not truncate.

---

## `patcher.py`
**Purpose:** Extracts patch instructions from LLM output, generates unified diffs, validates patches with AST, writes modified files to disk, and saves patch records.

### Imports
- `ast` — validates that patched code parses correctly
- `difflib` — generates unified diffs
- `re` — extracts code blocks from LLM output
- `pathlib.Path` — file I/O
- `db` — `save_patch`, `update_patch_status`, `get_patches`
- `llm` — `generate_patch`
- `backup` — `create_backup`
- `config` — patch-related constants

### Functions

#### `build_patch_prompt(file_path: str, symbol_name: str, current_source: str, instruction: str, context_symbols: list[dict]) -> str`
- **Inputs:** target file path, function/class name to modify, its current source, natural language instruction, list of related symbols for context
- **Outputs:** complete prompt string
- **Description:** Constructs a prompt that instructs the LLM to output ONLY a complete replacement for the function/class, wrapped in a fenced code block with the language tag `python`. The prompt includes: the instruction, the current source, relevant context symbols (truncated to 500 chars each), and explicit instructions: "Output only the replacement code in a ```python ... ``` block. Do not include any explanation." This structured output constraint is critical for reliable extraction.

#### `extract_code_from_response(llm_response: str) -> str | None`
- **Inputs:** raw LLM response string
- **Outputs:** extracted Python code string, or None if no code block found
- **Description:** Uses regex `r'```python\s*(.*?)\s*```'` with `re.DOTALL` to find the first fenced Python code block. Returns the inner content stripped of leading/trailing whitespace. If no match, tries a fallback regex for generic ` ``` ` blocks. Returns None if neither matches.

#### `validate_python_ast(code: str) -> tuple[bool, str | None]`
- **Inputs:** Python source code string
- **Outputs:** `(is_valid: bool, error_message: str | None)`
- **Description:** Calls `ast.parse(code)`. If it succeeds, returns `(True, None)`. If `SyntaxError` is raised, returns `(False, str(error))` with line number and message. This is the gate before any file write.

#### `generate_diff(original_source: str, new_source: str, file_path: str) -> str`
- **Inputs:** original file content, new file content, file path for diff header
- **Outputs:** unified diff string
- **Description:** Uses `difflib.unified_diff(original.splitlines(keepends=True), new.splitlines(keepends=True), fromfile=f"a/{file_path}", tofile=f"b/{file_path}", lineterm="")`. Joins the generator into a single string. Returns empty string if files are identical.

#### `replace_symbol_in_source(file_source: str, symbol_name: str, new_symbol_source: str) -> str | None`
- **Inputs:** full file source, name of the function/class to replace, new source for that symbol
- **Outputs:** new full file source with the symbol replaced, or None if symbol not found
- **Description:** Parses `file_source` with `ast.parse`. Finds the `FunctionDef`/`AsyncFunctionDef`/`ClassDef` node with matching `name`. Gets `start_line` and `end_line` (1-indexed). Splits file into lines. Replaces lines `start_line-1` through `end_line` (inclusive) with `new_symbol_source.splitlines()`. Joins and returns. If the symbol appears multiple times (impossible in well-formed Python at module level, but possible with nested classes), replaces the first occurrence and logs a warning.

#### `request_patch(file_path: str, symbol_name: str, instruction: str, context_symbols: list[dict]) -> dict`
- **Inputs:** target file path, symbol name, instruction, context symbols
- **Outputs:** dict `{patch_id: int, diff: str, status: str, ast_valid: bool, error: str | None}`
- **Description:** This is the main entry point called from `pages/patches.py`. Reads current file source. Calls `build_patch_prompt`. Calls `llm.generate_patch`. Calls `extract_code_from_response` — if None, returns error dict. Calls `validate_python_ast` on extracted code — if invalid, returns error dict with AST error message; patch is NOT saved or applied. Calls `replace_symbol_in_source` to get the new full file source. Calls `generate_diff`. Calls `db.save_patch(file_path, diff, metadata)`. Returns dict with `patch_id`, `diff`, `status="pending"`, `ast_valid=True`.

#### `apply_patch(patch_id: int) -> dict`
- **Inputs:** `patch_id`
- **Outputs:** dict `{success: bool, backup_id: int | None, error: str | None}`
- **Description:** Fetches patch from DB. Checks status is `'pending'`; rejects if already applied/rejected. Calls `backup.create_backup(file_path, patch_id)` — this MUST succeed before any write. Applies the diff: reads current file, uses `difflib` or direct source replacement to produce new content, writes to disk. Calls `db.update_patch_status(patch_id, 'applied')`. Returns success dict with `backup_id`.

#### `reject_patch(patch_id: int) -> None`
- **Inputs:** `patch_id`
- **Outputs:** none
- **Description:** Calls `db.update_patch_status(patch_id, 'rejected')`. No file system changes.

**What Can Go Wrong:**
- LLM outputs non-Python text: `extract_code_from_response` returns None, request_patch returns an error dict, no DB write occurs.
- AST validation passes but code is semantically broken: cannot be detected statically; the backup/restore system is the safety net.
- File modified between patch generation and apply (race condition): `apply_patch` re-reads the file and checks that the lines being replaced still match the diff context. If they don't match, returns an error dict without writing.
- Disk full during write: `IOError` caught in `apply_patch`, returns error, calls `db.update_patch_status(patch_id, 'rejected')` to mark it as failed.

---

## `backup.py`
**Purpose:** Creates timestamped file snapshots before any patch is applied, provides restore functionality, and prunes old backups.

### Imports
- `shutil` — file copy
- `pathlib.Path` — path manipulation
- `datetime` — timestamp generation
- `db` — `save_backup`, `get_backups`
- `config` — `BACKUP_DIR`

### Functions

#### `create_backup(file_path: str, patch_id: int | None = None) -> dict`
- **Inputs:** path to the file to back up, optional associated patch id
- **Outputs:** dict `{backup_id: int, backup_path: str}`
- **Description:** Generates a backup filename: `"{BACKUP_DIR}/{stem}__{timestamp}__{hash}.py"` where `timestamp` is `datetime.utcnow().strftime("%Y%m%d_%H%M%S")` and `hash` is the first 8 chars of the file content's MD5 (ensures uniqueness if two backups happen in the same second). Creates `BACKUP_DIR` if it doesn't exist. Copies the file using `shutil.copy2` (preserves metadata). Calls `db.save_backup(file_path, backup_path, patch_id)`. Returns dict with ids and paths. Raises `IOError` if copy fails — this propagates to `patcher.apply_patch` which aborts the patch application.

#### `restore_backup(backup_id: int) -> dict`
- **Inputs:** `backup_id`
- **Outputs:** dict `{success: bool, restored_to: str, error: str | None}`
- **Description:** Fetches backup record from DB. Checks that `backup_path` exists on disk. Copies backup file back to `file_path` using `shutil.copy2`. Does NOT automatically re-index the file after restore (the user should trigger re-indexing manually, or a future enhancement could do it). Returns success dict.

#### `list_backups(file_path: str | None = None) -> list[dict]`
- **Inputs:** optional file path filter
- **Outputs:** list of backup dicts from DB, newest first
- **Description:** Calls `db.get_backups(file_path)`. Annotates each dict with `exists_on_disk: bool` by checking `Path(backup_path).exists()`. Orphaned backups (DB record but no file) are surfaced in the UI so users can clean them up.

#### `delete_backup(backup_id: int) -> dict`
- **Inputs:** `backup_id`
- **Outputs:** dict `{success: bool, error: str | None}`
- **Description:** Fetches backup record. Attempts to delete the file from disk. Regardless of whether the file exists, deletes the DB record. This avoids orphaned DB records.

#### `prune_old_backups(keep_latest: int = config.BACKUP_RETENTION_LIMIT) -> int`
- **Inputs:** number of most recent backups to keep per file
- **Outputs:** count of backups deleted
- **Description:** Groups backups by `file_path`. For each file, sorts by `created_at` descending. Deletes records (and files) beyond `keep_latest`. Called automatically by `db.save_backup` when the total count exceeds the limit. Returns count deleted.

**What Can Go Wrong:**
- Backup directory not writable: `IOError` raised in `create_backup`, which propagates to `patcher.apply_patch`, preventing the patch from being applied. This is the correct behavior — never modify a file without a backup.
- Disk full during backup: same as above.
- Backup file deleted manually by user: `restore_backup` detects missing file, returns error dict. `list_backups` flags it with `exists_on_disk=False`.

---

## `app.py`
**Purpose:** Streamlit entry point — configures the app, initializes the database and FAISS index, and renders the navigation sidebar.

### Imports
- `streamlit` — page config, sidebar, navigation
- `db` — `init_db`
- `embedder` — `load_faiss_index`
- `config` — for app title, default paths

### Functions

#### `initialize_app() -> None`
- **Inputs:** none
- **Outputs:** none
- **Description:** Called once at startup via Streamlit's `st.cache_resource` decorator (ensures it runs exactly once per server process, not once per user session). Calls `db.init_db()`. Calls `embedder.load_faiss_index()`. Logs startup success. If either call raises an exception (DB not writable, FAISS not installed), displays a `st.error` with a user-readable message and `st.stop()` to halt rendering.

#### `render_sidebar() -> None`
- **Inputs:** none
- **Outputs:** none
- **Description:** Renders the sidebar navigation links to each page. Displays a status indicator: FAISS index size (number of vectors), DB symbol count, current model names. Adds a "Re-index Now" button that calls `indexer.index_directory` with a progress bar. Re-indexing from the sidebar is a convenience feature for small codebases; large codebases should use the CLI or background task.

#### `main() -> None`
- **Inputs:** none
- **Outputs:** none (Streamlit renders everything)
- **Description:** Calls `st.set_page_config`. Calls `initialize_app()`. Calls `render_sidebar()`. Streamlit handles page routing automatically via the `pages/` directory.

**What Can Go Wrong:**
- `st.cache_resource` means `initialize_app` runs once per process; if DB state changes (e.g., file moved), the cached state is stale. Users must restart the app to reinitialize. This is documented in the UI.

---

## `pages/chat.py`
**Purpose:** Renders the conversational chat interface with streaming LLM responses and a collapsible panel showing exactly which symbols were injected into context and their scores.

### Imports
- `streamlit` — UI components, session state
- `retriever` — `hybrid_search`
- `llm` — `chat_with_context`
- `context_manager` — `assemble_context`
- `config` — defaults
- `db` — `get_setting`

### Session State Keys
- `st.session_state.chat_history`: list of `{"role": "user"|"assistant", "content": str}` — full conversation history
- `st.session_state.last_context`: the `context dict` from the most recent `assemble_context` call — used to render the context panel
- `st.session_state.top_k`: current Top-K value

### Functions

#### `render_context_panel(context: dict) -> None`
- **Inputs:** context dict from `context_manager.assemble_context`
- **Outputs:** none (renders Streamlit UI)
- **Description:** Renders a `st.expander("📎 Injected Context ({n} symbols)")` that is collapsed by default. Inside, for each symbol in `context['included']`: renders a small card with symbol name, file path, score badge (color-coded: green > 0.7, yellow > 0.4, red otherwise), and a `st.code(symbol['source'], language="python")` block. Below the included symbols, lists excluded symbols with reason. Shows `total_tokens_used / budget_tokens` as a progress bar.

#### `render_chat_message(role: str, content: str) -> None`
- **Inputs:** `"user"` or `"assistant"`, message content string
- **Outputs:** none
- **Description:** Uses `st.chat_message(role)` context manager to render the message. For assistant messages, renders with `st.markdown` to support code blocks in LLM responses.

#### `handle_user_input(query: str) -> None`
- **Inputs:** user query string
- **Outputs:** none (updates session state, renders response)
- **Description:** Appends `{"role": "user", "content": query}` to `chat_history`. Calls `retriever.hybrid_search(query, top_k=st.session_state.top_k)`. Calls `context_manager.assemble_context(retrieved_symbols)` and stores result in `st.session_state.last_context`. Calls `llm.chat_with_context(query, history, retrieved_symbols, BASE_SYSTEM_PROMPT)`. Streams the response into a `st.chat_message("assistant")` block using `st.write_stream`. Appends the complete assembled response to `chat_history`. Triggers `render_context_panel` to update.

#### `render_page() -> None`
- **Inputs:** none
- **Outputs:** none
- **Description:** Sets page title. Renders all messages in `chat_history` using `render_chat_message`. Renders `render_context_panel` for the last context if available. Renders `st.chat_input("Ask about your codebase...")`. On submit, calls `handle_user_input`. Adds a "Clear History" button that resets `chat_history` and `last_context`.

**What Can Go Wrong:**
- Streaming response interrupted: `st.write_stream` handles partial streams gracefully; partial response is appended to history.
- Retrieval returns no symbols: `context_manager.assemble_context` receives empty list, returns empty context, LLM answers without RAG context. UI shows "No symbols retrieved" in the context panel.

---

## `pages/search.py`
**Purpose:** Standalone hybrid search UI allowing users to explore the indexed codebase without starting a chat conversation.

### Imports
- `streamlit`
- `retriever` — `hybrid_search`, `keyword_only_search`
- `db` — `get_setting`
- `config`

### Functions

#### `render_result_card(symbol: dict) -> None`
- **Inputs:** symbol dict with scores
- **Outputs:** none
- **Description:** Renders a card-style expander: title is `"{symbol_type} {name}"` with file path and line range. Shows three score columns: FTS score, semantic score, merged score (all formatted to 3 decimal places). On expand: shows full source in a syntax-highlighted code block.

#### `render_page() -> None`
- **Inputs:** none
- **Outputs:** none
- **Description:** Renders a search input (not chat input — a regular text input with a Search button). Dropdowns for Top-K (2–20) and a toggle for "semantic + FTS" vs "FTS only" mode. On search: calls `hybrid_search` or `keyword_only_search` based on mode toggle. Renders a score summary bar chart using `st.bar_chart` showing merged scores for all results. Renders each result using `render_result_card`. If no results, shows a suggestion to re-index or rephrase.

---

## `pages/patches.py`
**Purpose:** UI for requesting LLM-generated patches, reviewing diffs, and applying or rejecting them.

### Imports
- `streamlit`
- `patcher` — `request_patch`, `apply_patch`, `reject_patch`
- `db` — `get_patches`, `get_symbols_by_ids`
- `retriever` — `hybrid_search` (to find the symbol to patch)

### Functions

#### `render_diff_view(diff_text: str) -> None`
- **Inputs:** unified diff string
- **Outputs:** none
- **Description:** Renders the diff in a `st.code` block with `language="diff"`. Added lines (starting with `+`) are rendered in green via custom CSS injected with `st.markdown`. Removed lines (starting with `-`) in red.

#### `render_patch_request_form() -> None`
- **Inputs:** none
- **Outputs:** none
- **Description:** Text input for the symbol name to modify (with autocomplete via `st.selectbox` populated from a `db.fts_search` results). Text area for the patch instruction. On submit: calls `patcher.request_patch`. If `ast_valid=False`, shows a red error with the AST error message and does not store the patch. If successful, shows the diff and two buttons: "Apply" and "Reject".

#### `render_patch_history() -> None`
- **Inputs:** none
- **Outputs:** none
- **Description:** Fetches patches via `db.get_patches()`. Renders a filterable table with columns: file, status, created_at, applied_at. Clicking a row expands to show the full diff and metadata (which symbols were used in context, the prompt, the model).

#### `render_page() -> None`
- **Inputs:** none
- **Outputs:** none
- **Description:** Tab layout: "Request Patch" tab calls `render_patch_request_form`. "Patch History" tab calls `render_patch_history`.

---

## `pages/backups.py`
**Purpose:** Browse all file backups, preview their content, and restore them with one click.

### Imports
- `streamlit`
- `backup` — `list_backups`, `restore_backup`, `delete_backup`
- `pathlib.Path`

### Functions

#### `render_backup_row(b: dict) -> None`
- **Inputs:** backup dict (includes `exists_on_disk` flag)
- **Outputs:** none
- **Description:** Renders a row with: file path, created_at, associated patch_id (if any), disk status indicator (green checkmark or red ⚠ for orphaned). "Preview" button shows backup file content in a code block. "Restore" button calls `backup.restore_backup` and shows a success or error toast. "Delete" button calls `backup.delete_backup` with a confirmation dialog.

#### `render_page() -> None`
- **Inputs:** none
- **Outputs:** none
- **Description:** File path filter selectbox (populated from unique file paths in backup list). "Orphans only" checkbox. Renders each backup using `render_backup_row`. Shows total backup count and total disk usage (summed file sizes).

---

## `pages/settings.py`
**Purpose:** Lets the user configure model names, Top-K, token budget, index path, and FTS/semantic weights, persisting changes to the settings DB.

### Imports
- `streamlit`
- `db` — `get_all_settings`, `set_setting`
- `embedder` — `rebuild_faiss_index`
- `config`

### Functions

#### `render_page() -> None`
- **Inputs:** none
- **Outputs:** none
- **Description:** Loads all settings from DB on page load. Renders form fields:
  - Chat model name (text input, default `config.CHAT_MODEL`)
  - Embed model name (text input, default `config.EMBED_MODEL`)
  - Model context window (number input, default 8192)
  - Top-K (slider 1–20)
  - FTS weight / Semantic weight (two sliders that sum to 1.0 — if user moves one, the other updates automatically via `st.session_state`)
  - Index root path (text input)
  - Backup retention limit (number input)
  - A "Rebuild FAISS Index" button — calls `embedder.rebuild_faiss_index()` and shows count of vectors indexed.
  - A "Save Settings" button — iterates all form values, calls `db.set_setting` for each.
- **Warning displayed:** If embed model is changed, existing embeddings were generated by a different model and the FAISS index should be rebuilt and all symbols re-indexed (existing vectors will remain in DB tagged with old model name and will not appear in new searches until re-indexed).

---

## Cross-Module Data Flow Summary

```
User Query
    │
    ▼
pages/chat.py
    │ query string
    ▼
retriever.hybrid_search()
    ├──► db.fts_search()          → {symbol_id, fts_score}
    ├──► embed.generate_embedding() → np.ndarray (768,)
    └──► embedder.semantic_search() → {symbol_id, cosine_score}
         └── normalize + merge → [{symbol_id, merged_score, ...}]
    │
    ▼
context_manager.assemble_context()
    → {context_str, included, excluded, tokens_used}
    │
    ▼
llm.chat_with_context()
    → stream of response chunks
    │
    ▼
pages/chat.py renders stream + context panel
```

```
Patch Request
    │
    ▼
pages/patches.py
    │ file_path, symbol_name, instruction
    ▼
patcher.request_patch()
    ├──► llm.generate_patch()     → raw LLM response
    ├──► extract_code_from_response() → code string
    ├──► validate_python_ast()    → (bool, error)
    ├──► replace_symbol_in_source() → new file content
    ├──► generate_diff()          → unified diff
    └──► db.save_patch()          → patch_id
    │
    ▼ (on Apply)
patcher.apply_patch()
    ├──► backup.create_backup()   → backup_id (MUST succeed first)
    ├──► write new file to disk
    └──► db.update_patch_status()
```

---

This specification covers all 16 files completely. Every function is described with its inputs, outputs, behavior, data shapes, inter-module dependencies, and failure modes. SQLite is the sole source of truth; FAISS is always derived and rebuildable. The token budget is enforced at context assembly time, and every user query surfaces exactly what was injected and why.