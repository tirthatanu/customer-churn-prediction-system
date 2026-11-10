"""Small SQLite store of prediction *outputs* only.

Raw customer attributes are deliberately not persisted.
Use access control and a managed database before production deployment.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path), timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(
            """CREATE TABLE IF NOT EXISTS prediction_history (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   created_at_utc TEXT NOT NULL,
                   prediction TEXT NOT NULL CHECK(prediction IN ('Churn','No Churn')),
                   probability REAL NOT NULL CHECK(probability BETWEEN 0 AND 1),
                   risk_level TEXT NOT NULL CHECK(risk_level IN ('Low','Medium','High'))
               )"""
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS ix_history_created ON prediction_history(created_at_utc DESC)"
        )
        db.commit()


def record(path: Path, prediction: str, probability: float, risk_level: str) -> None:
    with connect(path) as db:
        db.execute(
            "INSERT INTO prediction_history (created_at_utc, prediction, probability, risk_level) "
            "VALUES (?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), prediction, probability, risk_level),
        )
        db.commit()


def fetch_history(path: Path, limit: int, offset: int) -> dict:
    with connect(path) as db:
        count = int(db.execute("SELECT COUNT(*) FROM prediction_history").fetchone()[0])
        rows = db.execute(
            "SELECT id, created_at_utc, prediction, probability, risk_level "
            "FROM prediction_history ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return {
            "count": count,
            "limit": limit,
            "offset": offset,
            "predictions": [dict(row) for row in rows],
        }
