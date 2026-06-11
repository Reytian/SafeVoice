"""Tests for vocabulary management."""
import pytest
from src.vocabulary import VocabularyManager


@pytest.fixture
def vocab(tmp_path):
    path = str(tmp_path / "vocab.json")
    return VocabularyManager(path)


def test_add_hotword(vocab):
    vocab.add_hotword("Kubernetes")
    assert "Kubernetes" in vocab.get_hotwords()


def test_remove_hotword(vocab):
    vocab.add_hotword("Claude")
    vocab.remove_hotword("Claude")
    assert "Claude" not in vocab.get_hotwords()


def test_add_snippet(vocab):
    vocab.add_snippet("my email", "user@example.com")
    assert vocab.get_snippets()["my email"] == "user@example.com"


def test_apply_snippets(vocab):
    vocab.add_snippet("my email", "user@example.com")
    vocab.add_snippet("my phone", "555-1234")
    result = vocab.apply_snippets("Send to my email and call my phone")
    assert result == "Send to user@example.com and call 555-1234"


def test_apply_snippets_case_insensitive(vocab):
    vocab.add_snippet("my email", "user@example.com")
    result = vocab.apply_snippets("Send to My Email please")
    assert result == "Send to user@example.com please"


def test_persistence(tmp_path):
    path = str(tmp_path / "vocab.json")
    v1 = VocabularyManager(path)
    v1.add_hotword("Claude")
    v1.add_snippet("sig", "Best regards, User")

    v2 = VocabularyManager(path)
    assert "Claude" in v2.get_hotwords()
    assert v2.get_snippets()["sig"] == "Best regards, User"


def test_empty_vocab(vocab):
    assert vocab.get_hotwords() == []
    assert vocab.get_snippets() == {}
    assert vocab.apply_snippets("hello") == "hello"


def test_save_is_atomic_and_survives_reload(tmp_path):
    path = str(tmp_path / "vocabulary.json")
    from src.vocabulary import VocabularyManager
    v1 = VocabularyManager(path)
    v1.add_hotword("Kubernetes")
    v1.add_snippet("addr", "1 Main St\\nNYC")
    assert not (tmp_path / "vocabulary.json.tmp").exists()
    v2 = VocabularyManager(path)
    assert "Kubernetes" in v2.get_hotwords()
    assert v2.get_snippets().get("addr") == "1 Main St\\nNYC"


def test_save_failure_does_not_raise(tmp_path, monkeypatch):
    # _save runs inside NSButton callbacks; an OSError must be swallowed
    # (logged), never propagated into the ObjC bridge.
    path = str(tmp_path / "vocabulary.json")
    from src.vocabulary import VocabularyManager
    v = VocabularyManager(path)
    import os as _os
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(_os, "replace", boom)
    v.add_hotword("StillWorks")  # must not raise
    assert "StillWorks" in v.get_hotwords()
