import sqlite3
import json
import time
from pathlib import Path

import numpy as np

import config


def get_connection() -> sqlite3.Connection:
    Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _execute_with_retry(conn, sql, params=()):
    """Retry on 'database is locked' up to 3 times."""
    for attempt in range(3):
        try:
            return conn.execute(sql, params)
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < 2:
                time.sleep(0.1)
            else:
                raise


def init_db() -> None:
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS symbols (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path   TEXT NOT NULL,
                symbol_type TEXT NOT NULL,
                name        TEXT NOT NULL,
                signature   TEXT,
                docstring   TEXT,
                source      TEXT NOT NULL,
                start_line  INTEGER,
                end_line    INTEGER,
                indexed_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol_id    INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
                embed_model  TEXT NOT NULL,
                vector_blob  BLOB NOT NULL,
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patches (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path    TEXT NOT NULL,
                diff_text    TEXT NOT NULL,
                status       TEXT NOT NULL,
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
                applied_at   DATETIME,
                metadata     TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backups (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path    TEXT NOT NULL,
                backup_path  TEXT NOT NULL,
                patch_id     INTEGER REFERENCES patches(id),
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key    TEXT PRIMARY KEY,
                value  TEXT NOT NULL
            )
        """)

        # Check FTS5 support
        try:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts
                USING fts5(
                    name, signature, docstring, source,
                    content='symbols',
                    content_rowid='id',
                    tokenize='porter unicode61'
                )
            """)
        except sqlite3.OperationalError:
            raise RuntimeError(
                "FTS5 not available. Install pysqlite3-binary: pip install pysqlite3-binary"
            )

        # Triggers to keep FTS in sync
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS symbols_ai AFTER INSERT ON symbols BEGIN
                INSERT INTO symbols_fts(rowid, name, signature, docstring, source)
                VALUES (new.id, new.name, new.signature, new.docstring, new.source);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS symbols_ad AFTER DELETE ON symbols BEGIN
                INSERT INTO symbols_fts(symbols_fts, rowid, name, signature, docstring, source)
                VALUES ('delete', old.id, old.name, old.signature, old.docstring, old.source);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS symbols_au AFTER UPDATE ON symbols BEGIN
                INSERT INTO symbols_fts(symbols_fts, rowid, name, signature, docstring, source)
                VALUES ('delete', old.id, old.name, old.signature, old.docstring, old.source);
                INSERT INTO symbols_fts(rowid, name, signature, docstring, source)
                VALUES (new.id, new.name, new.signature, new.docstring, new.source);
            END
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_symbol ON embeddings(symbol_id)")


def upsert_symbol(file_path, symbol_type, name, signature, docstring, source, start_line, end_line) -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM symbols WHERE file_path=? AND name=? AND symbol_type=?",
            (file_path, name, symbol_type)
        ).fetchone()
        if row:
            conn.execute("""
                UPDATE symbols SET signature=?, docstring=?, source=?, start_line=?, end_line=?,
                indexed_at=CURRENT_TIMESTAMP WHERE id=?
            """, (signature, docstring, source, start_line, end_line, row["id"]))
            return row["id"]
        else:
            cur = conn.execute("""
                INSERT INTO symbols (file_path, symbol_type, name, signature, docstring, source, start_line, end_line)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (file_path, symbol_type, name, signature, docstring, source, start_line, end_line))
            return cur.lastrowid


def delete_symbols_for_file(file_path: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM symbols WHERE file_path=?", (file_path,))


def upsert_embedding(symbol_id: int, embed_model: str, vector: np.ndarray) -> None:
    if vector.shape != (config.EMBED_DIM,):
        raise ValueError(f"Expected shape ({config.EMBED_DIM},), got {vector.shape}")
    if vector.dtype != np.float32:
        raise ValueError(f"Expected float32, got {vector.dtype}")
    blob = vector.tobytes()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM embeddings WHERE symbol_id=? AND embed_model=?",
            (symbol_id, embed_model)
        ).fetchone()
        if row:
            conn.execute("UPDATE embeddings SET vector_blob=? WHERE id=?", (blob, row["id"]))
        else:
            conn.execute(
                "INSERT INTO embeddings (symbol_id, embed_model, vector_blob) VALUES (?, ?, ?)",
                (symbol_id, embed_model, blob)
            )


def get_all_embeddings(embed_model: str) -> list:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT symbol_id, vector_blob FROM embeddings WHERE embed_model=? ORDER BY symbol_id",
            (embed_model,)
        ).fetchall()
    result = []
    expected_bytes = config.EMBED_DIM * 4  # float32 = 4 bytes
    for row in rows:
        if len(row["vector_blob"]) != expected_bytes:
            print(f"Warning: skipping corrupt embedding for symbol_id={row['symbol_id']}")
            continue
        vector = np.frombuffer(row["vector_blob"], dtype=np.float32).reshape((config.EMBED_DIM,))
        result.append({"symbol_id": row["symbol_id"], "vector": vector})
    return result


def get_symbol_by_id(symbol_id: int):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM symbols WHERE id=?", (symbol_id,)).fetchone()
        return dict(row) if row else None


def get_symbols_by_ids(symbol_ids: list) -> list:
    if not symbol_ids:
        return []
    placeholders = ",".join("?" * len(symbol_ids))
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM symbols WHERE id IN ({placeholders})", symbol_ids
        ).fetchall()
    id_to_row = {row["id"]: dict(row) for row in rows}
    return [id_to_row[sid] for sid in symbol_ids if sid in id_to_row]


def fts_search(query: str, limit: int) -> list:
    # Sanitize query
    for ch in ['"', '*']:
        query = query.replace(ch, " ")
    query = query.strip()
    if not query:
        return []
    with get_connection() as conn:
        try:
            rows = conn.execute(
                "SELECT rowid, rank FROM symbols_fts WHERE symbols_fts MATCH ? ORDER BY rank LIMIT ?",
                (query, limit)
            ).fetchall()
        except sqlite3.OperationalError as e:
            print(f"FTS search error: {e}")
            return []
    return [{"symbol_id": row["rowid"], "fts_score": -row["rank"]} for row in rows]


def save_patch(file_path: str, diff_text: str, metadata: dict) -> int:
    with get_connection() as conn:
        cur = conn.execute("""
            INSERT INTO patches (file_path, diff_text, status, metadata)
            VALUES (?, ?, 'pending', ?)
        """, (file_path, diff_text, json.dumps(metadata)))
        patch_id = cur.lastrowid
        # Prune old patches
        conn.execute("""
            DELETE FROM patches WHERE id NOT IN (
                SELECT id FROM patches ORDER BY created_at DESC LIMIT ?
            )
        """, (config.PATCH_HISTORY_LIMIT,))
        return patch_id


def update_patch_status(patch_id: int, status: str, applied_at=None) -> None:
    if status not in ("applied", "rejected"):
        raise ValueError(f"Invalid status: {status}")
    with get_connection() as conn:
        conn.execute(
            "UPDATE patches SET status=?, applied_at=? WHERE id=?",
            (status, applied_at, patch_id)
        )


def get_patches(status_filter: str = None) -> list:
    with get_connection() as conn:
        if status_filter:
            rows = conn.execute(
                "SELECT * FROM patches WHERE status=? ORDER BY created_at DESC", (status_filter,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM patches ORDER BY created_at DESC").fetchall()
    result = []
    for row in rows:
        d = dict(row)
        if d.get("metadata"):
            d["metadata"] = json.loads(d["metadata"])
        result.append(d)
    return result


def save_backup(file_path: str, backup_path: str, patch_id: int = None) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO backups (file_path, backup_path, patch_id) VALUES (?, ?, ?)",
            (file_path, backup_path, patch_id)
        )
        backup_id = cur.lastrowid
        conn.execute("""
            DELETE FROM backups WHERE id NOT IN (
                SELECT id FROM backups ORDER BY created_at DESC LIMIT ?
            )
        """, (config.BACKUP_RETENTION_LIMIT,))
        return backup_id


def get_backups(file_path: str = None) -> list:
    with get_connection() as conn:
        if file_path:
            rows = conn.execute(
                "SELECT * FROM backups WHERE file_path=? ORDER BY created_at DESC", (file_path,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM backups ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]


def get_setting(key: str, default: str = None):
    with get_connection() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with get_connection() as conn:
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))


def get_all_settings() -> dict:
    with get_connection() as conn:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
    return {row["key"]: row["value"] for row in rows}