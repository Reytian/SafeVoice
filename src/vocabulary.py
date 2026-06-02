"""ASR hotwords and snippet replacement management."""
import json
import logging
import os
import re
import threading

logger = logging.getLogger(__name__)


class VocabularyManager:
    def __init__(self, path: str = None):
        if path is None:
            config_dir = os.path.expanduser("~/.config/safevoice")
            os.makedirs(config_dir, exist_ok=True)
            path = os.path.join(config_dir, "vocabulary.json")
        self._path = path
        self._lock = threading.Lock()
        self._hotwords: list[str] = []
        self._snippets: dict[str, str] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            # Corrupt / unreadable vocabulary file must never block app startup.
            logger.warning("Failed to load vocabulary from %s: %s; using empty defaults", self._path, exc)
            return
        if not isinstance(data, dict):
            logger.warning("Vocabulary file %s is not a JSON object; using empty defaults", self._path)
            return
        hotwords = data.get("hotwords", [])
        snippets = data.get("snippets", {})
        self._hotwords = hotwords if isinstance(hotwords, list) else []
        self._snippets = snippets if isinstance(snippets, dict) else {}

    def _save(self):
        with open(self._path, "w") as f:
            json.dump({"hotwords": self._hotwords, "snippets": self._snippets}, f, indent=2)

    def add_hotword(self, word: str):
        with self._lock:
            if word not in self._hotwords:
                self._hotwords.append(word)
                self._save()

    def remove_hotword(self, word: str):
        with self._lock:
            if word in self._hotwords:
                self._hotwords.remove(word)
                self._save()

    def get_hotwords(self) -> list[str]:
        return list(self._hotwords)

    def add_snippet(self, trigger: str, replacement: str):
        with self._lock:
            self._snippets[trigger] = replacement
            self._save()

    def remove_snippet(self, trigger: str):
        with self._lock:
            self._snippets.pop(trigger, None)
            self._save()

    def get_snippets(self) -> dict[str, str]:
        return dict(self._snippets)

    def apply_snippets(self, text: str) -> str:
        # Snapshot under the lock so a concurrent add/remove on another thread
        # can't raise "dictionary changed size during iteration" mid-dictation.
        with self._lock:
            items = list(self._snippets.items())
        for trigger, replacement in items:
            pattern = re.compile(re.escape(trigger), re.IGNORECASE)
            # Pass the replacement through a function so re.sub treats it as a
            # literal string. Otherwise a replacement containing a backslash
            # (e.g. r"C:\Users") or a group ref (r"\1") raises re.error and
            # destroys the entire transcript.
            text = pattern.sub(lambda _m, r=replacement: r, text)
        return text
