"""SQLite-backed transcription history with CSV export."""
import csv
import logging
import sqlite3
import threading
from datetime import date, datetime, timedelta
import os

logger = logging.getLogger(__name__)


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
            duration: float = 0.0, language: str = "") -> bool:
        """Persist a transcription. Returns True on success.

        A write failure (locked DB, disk full, corruption) is logged and
        swallowed rather than raised, so it can never turn an already-pasted
        transcription into a user-facing 'Error'.
        """
        word_count = len(final_text.split())
        with self._lock:
            try:
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
            except sqlite3.Error:
                logger.exception("Failed to write transcription to history at %s", self._db_path)
                return False
        return True

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
        # Use an exclusive upper bound on the day AFTER end_date so entries
        # recorded in the final second (23:59:59.xxxxxx, microsecond ISO
        # timestamps) are not silently dropped from queries and CSV exports.
        if "T" not in end_date:
            try:
                end_exclusive = (date.fromisoformat(end_date) + timedelta(days=1)).isoformat()
            except ValueError:
                end_exclusive = end_date + "T23:59:59.999999"
            upper_clause = "timestamp < ?"
            upper_value = end_exclusive
        else:
            upper_clause = "timestamp <= ?"
            upper_value = end_date
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"SELECT * FROM history WHERE timestamp >= ? AND {upper_clause} ORDER BY id DESC",
                    (start_date, upper_value)
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
