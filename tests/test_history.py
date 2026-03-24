"""Tests for transcription history storage."""
import os
import pytest
from src.history import HistoryStore


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test_history.db")
    return HistoryStore(db_path)


def test_add_entry(store):
    store.add("Hello world", raw_text="hello world", mode="quick", duration=2.5)
    entries = store.get_recent(10)
    assert len(entries) == 1
    assert entries[0]["final_text"] == "Hello world"
    assert entries[0]["raw_text"] == "hello world"
    assert entries[0]["mode"] == "quick"
    assert entries[0]["duration"] == 2.5


def test_get_recent_limit(store):
    for i in range(20):
        store.add(f"Entry {i}")
    entries = store.get_recent(5)
    assert len(entries) == 5
    assert entries[0]["final_text"] == "Entry 19"  # most recent first


def test_get_by_date_range(store):
    store.add("Today entry")
    entries = store.get_by_date("2020-01-01", "2099-12-31")
    assert len(entries) == 1


def test_export_csv(store, tmp_path):
    store.add("Hello", mode="quick", duration=1.0)
    store.add("World", mode="formal", duration=2.0)
    csv_path = str(tmp_path / "export.csv")
    store.export_csv(csv_path)
    assert os.path.exists(csv_path)
    with open(csv_path) as f:
        lines = f.readlines()
    assert len(lines) == 3  # header + 2 rows


def test_stats(store):
    store.add("Hello world", duration=2.0)
    store.add("Another entry with more words here", duration=3.0)
    stats = store.get_stats()
    assert stats["total_transcriptions"] == 2
    assert stats["total_words"] == 8
    assert stats["total_duration"] == 5.0


def test_empty_store(store):
    entries = store.get_recent(10)
    assert entries == []
    stats = store.get_stats()
    assert stats["total_transcriptions"] == 0
