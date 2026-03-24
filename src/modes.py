"""Processing modes: Quick (direct ASR) and custom LLM modes with per-mode hotkeys."""
import json
import os
import threading
from dataclasses import dataclass, asdict

STYLE_PRESETS = {
    "minimal": (
        "Fix only obvious typos and punctuation. Keep the original wording. "
        "Do NOT translate. Output only the cleaned text:\n\n{text}"
    ),
    "professional": (
        "Clean up this dictated text. Fix grammar, punctuation, and make it professional. "
        "Do NOT translate. Keep the same language. Output only the cleaned text:\n\n{text}"
    ),
    "casual": (
        "Clean up this dictated text lightly. Keep it conversational and natural. "
        "Fix obvious errors only. Do NOT translate. Output only the cleaned text:\n\n{text}"
    ),
    "verbatim": (
        "Output the text exactly as spoken, only fixing punctuation and capitalization. "
        "Do NOT rephrase, summarize, or translate:\n\n{text}"
    ),
}


@dataclass
class Mode:
    name: str
    prompt_template: str | None = None
    hotkey: dict | None = None
    builtin: bool = False
    enabled: bool = True
    translation_language: str | None = None

    def render_prompt(self, text: str) -> str:
        if self.prompt_template is None:
            return text
        result = self.prompt_template.replace("{text}", text)
        if self.translation_language:
            result = result.replace("{language}", self.translation_language)
        return result


DEFAULT_MODES = [
    Mode(
        name="Quick",
        prompt_template=None,
        hotkey={"key": "space", "modifiers": ["alt"]},
        builtin=True,
    ),
    Mode(
        name="Formal Writing",
        prompt_template=(
            "Clean up this dictated text. Fix grammar, punctuation, and make it professional. "
            "Do NOT translate. Keep the same language. Output only the cleaned text:\n\n{text}"
        ),
        hotkey={"key": "f", "modifiers": ["alt", "cmd"]},
        builtin=True,
    ),
    Mode(
        name="English Translation",
        prompt_template=(
            "Translate the following text to English. Output only the translation:\n\n{text}"
        ),
        hotkey={"key": "e", "modifiers": ["alt", "cmd"]},
        builtin=True,
    ),
]


class ModeManager:
    def __init__(self, path: str = None):
        if path is None:
            config_dir = os.path.expanduser("~/.config/safevoice")
            os.makedirs(config_dir, exist_ok=True)
            path = os.path.join(config_dir, "modes.json")
        self._path = path
        self._lock = threading.Lock()
        self._modes: list[Mode] = []
        self._load()

    def _load(self):
        self._modes = [Mode(**{**m.__dict__}) for m in DEFAULT_MODES]
        if os.path.exists(self._path):
            with open(self._path) as f:
                data = json.load(f)
            for entry in data.get("custom_modes", []):
                self._modes.append(Mode(**entry))
            for override in data.get("hotkey_overrides", []):
                mode = self.get(override["name"])
                if mode:
                    mode.hotkey = override["hotkey"]

    def _save(self):
        custom = [asdict(m) for m in self._modes if not m.builtin]
        overrides = [
            {"name": m.name, "hotkey": m.hotkey}
            for m in self._modes if m.builtin and m.hotkey
        ]
        with open(self._path, "w") as f:
            json.dump({"custom_modes": custom, "hotkey_overrides": overrides}, f, indent=2)

    def get_all(self) -> list[Mode]:
        return list(self._modes)

    def get(self, name: str) -> Mode | None:
        for m in self._modes:
            if m.name == name:
                return m
        return None

    def get_by_hotkey(self, key: str, modifiers: list[str]) -> Mode | None:
        target = {"key": key, "modifiers": sorted(modifiers)}
        for m in self._modes:
            if m.hotkey and m.hotkey.get("key") == target["key"]:
                if sorted(m.hotkey.get("modifiers", [])) == target["modifiers"]:
                    return m
        return None

    def add(self, mode: Mode):
        with self._lock:
            self._modes = [m for m in self._modes if m.name != mode.name]
            mode.builtin = False
            self._modes.append(mode)
            self._save()

    def remove(self, name: str):
        with self._lock:
            self._modes = [m for m in self._modes if not (m.name == name and not m.builtin)]
            self._save()

    def update_hotkey(self, name: str, hotkey: dict):
        with self._lock:
            mode = self.get(name)
            if mode:
                mode.hotkey = hotkey
                self._save()
