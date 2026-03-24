"""ASR hotwords and snippet replacement management."""
import json
import os
import re
import threading


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
        if os.path.exists(self._path):
            with open(self._path) as f:
                data = json.load(f)
            self._hotwords = data.get("hotwords", [])
            self._snippets = data.get("snippets", {})

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
        for trigger, replacement in self._snippets.items():
            pattern = re.compile(re.escape(trigger), re.IGNORECASE)
            text = pattern.sub(replacement, text)
        return text
