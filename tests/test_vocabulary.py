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
