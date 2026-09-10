"""API-key authentication backed by SQLite.

Keys are stored as SHA-256 hashes — the plaintext key is shown exactly
once at creation time (``scripts/create_api_key.py``) and never persisted.

Usage:
    store = APIKeyStore("auth.sqlite3")
    store.create_key("alice")          # -> returns plaintext key once
    store.validate_key("aaabbb...")    # -> key id or None
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import sqlite3
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class APIKeyStore:
    """Persistent API-key store (SQLite, SHA-256 hashes).

    Parameters
    ----------
    db_path:
        Path to the SQLite file.  ``:memory:`` for tests.
    """

    def __init__(self, db_path: str | Path = "auth.sqlite3") -> None:
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_key(self, name: str, rate_limit_per_min: int = 60) -> tuple[str, str]:
        """Create a new API key.

        Returns
        -------
        tuple[str, str]
            ``(key_id, plaintext_key)``.  The plaintext key is returned
            **only here** — store it somewhere safe or display it to the
            user once.  It cannot be recovered later.
        """
        plaintext = secrets.token_urlsafe(32)
        key_hash = self._hash(plaintext)
        key_id = "scrag_" + secrets.token_urlsafe(8)

        with self._lock:
            self._db().execute(
                "INSERT INTO api_keys (id, name, key_hash, rate_limit_per_min, created_at, active) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (key_id, name, key_hash, rate_limit_per_min, time.time(), 1),
            )
            self._db().commit()
        logger.info("API key created for %s (id=%s)", name, key_id)
        return key_id, plaintext

    def validate_key(self, plaintext: str) -> str | None:
        """Return the key id if *plaintext* is a valid active key, else ``None``."""
        if not plaintext:
            return None
        key_hash = self._hash(plaintext)
        with self._lock:
            row = self._db().execute(
                "SELECT id FROM api_keys WHERE key_hash = ? AND active = 1",
                (key_hash,),
            ).fetchone()
        return row[0] if row else None

    def rate_limit_for(self, key_id: str) -> int:
        """Return the per-minute rate limit for a key (default 60)."""
        with self._lock:
            row = self._db().execute(
                "SELECT rate_limit_per_min FROM api_keys WHERE id = ?",
                (key_id,),
            ).fetchone()
        return row[0] if row else 60

    def revoke_key(self, key_id: str) -> None:
        """Deactivate a key (keep the row for audit)."""
        with self._lock:
            self._db().execute(
                "UPDATE api_keys SET active = 0 WHERE id = ?", (key_id,)
            )
            self._db().commit()

    def list_keys(self) -> list[dict[str, object]]:
        """Return key metadata WITHOUT plaintext or hashes."""
        with self._lock:
            rows = self._db().execute(
                "SELECT id, name, rate_limit_per_min, created_at, active FROM api_keys "
                "ORDER BY created_at"
            ).fetchall()
        return [dict(zip(["id", "name", "rate_limit_per_min", "created_at", "active"], row)) for row in rows]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._lock:
            db = self._db()
            db.execute(
                "CREATE TABLE IF NOT EXISTS api_keys ("
                "  id TEXT PRIMARY KEY,"
                "  name TEXT NOT NULL,"
                "  key_hash TEXT NOT NULL UNIQUE,"
                "  rate_limit_per_min INTEGER NOT NULL DEFAULT 60,"
                "  created_at REAL NOT NULL,"
                "  active INTEGER NOT NULL DEFAULT 1"
                ")"
            )
            db.commit()

    def _db(self) -> sqlite3.Connection:
        if not hasattr(self, "_conn"):
            db_path = self.db_path
            if db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
        return self._conn

    @staticmethod
    def _hash(plaintext: str) -> str:
        return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()

    def close(self) -> None:
        if hasattr(self, "_conn"):
            self._conn.close()
