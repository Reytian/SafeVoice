"""SQLite-backed transcription history with CSV export."""
import csv
import sqlite3
import threading
from datetime import datetime
import os


class HistoryStore:
    def __init__(self, db_path: str = None):
        if db_path is None:
            config_dir = os.path.expanduser("~/.config/safevoice")
            os.makedirs(config_dir, exist_ok=True)
            db_path = os.path.join(config_dir, "history.db")
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        final_text TEXT NOT NULL,
                        raw_text TEXT DEFAULT '',
                        mode TEXT DEFAULT 'quick',
                        duration REAL DEFAULT 0.0,
                        language TEXT DEFAULT '',
                        word_count INTEGER DEFAULT 0
                    )
                """)
                conn.commit()
            finally:
                conn.close()

    def _connect(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add(self, final_text: str, raw_text: str = "", mode: str = "quick",
            duration: float = 0.0, language: str = ""):
        word_count = len(final_text.split())
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO history (timestamp, final_text, raw_text, mode, duration, language, word_count) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (datetime.now().isoformat(), final_text, raw_text, mode, duration, language, word_count)
                )
                conn.commit()
            finally:
                conn.close()

    def get_recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            finally:
                conn.close()
            return [dict(r) for r in rows]

    def get_by_date(self, start_date: str, end_date: str) -> list[dict]:
        if "T" not in end_date:
            end_date += "T23:59:59"
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM history WHERE timestamp >= ? AND timestamp <= ? ORDER BY id DESC",
                    (start_date, end_date)
                ).fetchall()
            finally:
                conn.close()
            return [dict(r) for r in rows]

    def export_csv(self, path: str, start_date: str = None, end_date: str = None):
        if start_date and end_date:
            entries = self.get_by_date(start_date, end_date)
        else:
            entries = self.get_recent(limit=999999)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "final_text", "raw_text", "mode", "duration", "language", "word_count"
            ])
            writer.writeheader()
            for entry in entries:
                row = {k: entry.get(k, "") for k in writer.fieldnames}
                writer.writerow(row)

    def get_stats(self) -> dict:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) as total, COALESCE(SUM(word_count), 0) as words, "
                    "COALESCE(SUM(duration), 0) as duration FROM history"
                ).fetchone()
            finally:
                conn.close()
            return {
                "total_transcriptions": row["total"],
                "total_words": row["words"],
                "total_duration": row["duration"],
            }
