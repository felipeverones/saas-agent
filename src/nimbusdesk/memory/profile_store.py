"""Structured customer profile — the EXACT-LOOKUP half of long-term memory.

Facts like "prefers Portuguese", "technical level: expert", "admin of the
Acme workspace" are key-value data: you want ALL of them, precisely, every
time you talk to that customer. That's a SQL lookup, not a similarity search
— using a vector store for this (a common over-engineering) adds fuzziness
exactly where you want none. Compare episodic.py, where fuzzy IS the point.

Facts UPSERT by (email, key): re-learning "plan=pro" after an upgrade
overwrites the stale value instead of accumulating contradictions — the
"consolidate" step of extract -> consolidate -> retrieve, in its simplest form.
"""

import sqlite3
from datetime import UTC, datetime


class SqliteProfileStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        # Injected connection: production passes a file-backed DB,
        # tests pass sqlite3.connect(":memory:"). Same adapter, both worlds.
        self._conn = connection
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS customer_facts (
                email      TEXT NOT NULL,
                key        TEXT NOT NULL,
                value      TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (email, key)
            )
            """
        )
        self._conn.commit()

    def upsert_facts(self, email: str, facts: dict[str, str]) -> None:
        now = datetime.now(UTC).isoformat()
        self._conn.executemany(
            """
            INSERT INTO customer_facts (email, key, value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (email, key) DO UPDATE
                SET value = excluded.value, updated_at = excluded.updated_at
            """,
            [(email.lower(), k, v, now) for k, v in facts.items()],
        )
        self._conn.commit()

    def get_profile(self, email: str) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT key, value FROM customer_facts WHERE email = ? ORDER BY key",
            (email.lower(),),
        ).fetchall()
        return dict(rows)
