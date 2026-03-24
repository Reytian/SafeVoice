"""
Settings manager for SafeVoice.

Handles loading, saving, and change-notification for user preferences.
Settings are persisted as JSON at ~/.config/safevoice/settings.json.
Thread-safe: all reads and writes are guarded by a lock.
"""

import json
import os
import threading
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# Default settings file location.
_CONFIG_DIR = Path.home() / ".config" / "safevoice"
_CONFIG_FILE = _CONFIG_DIR / "settings.json"

# All supported languages (matches the full list from the requirements).
SUPPORTED_LANGUAGES = [
    {"name": "Auto",         "code": "auto", "badge": "Auto", "display": "Auto (detect)"},
    {"name": "English",      "code": "en",   "badge": "EN",   "display": "English"},
    {"name": "Chinese",      "code": "zh",   "badge": "\u4e2d\u6587", "display": "Chinese (\u4e2d\u6587)"},
    {"name": "French",       "code": "fr",   "badge": "FR",   "display": "French (Fran\u00e7ais)"},
    {"name": "Japanese",     "code": "ja",   "badge": "JP",   "display": "Japanese (\u65e5\u672c\u8a9e)"},
    {"name": "Korean",       "code": "ko",   "badge": "KR",   "display": "Korean (\ud55c\uad6d\uc5b4)"},
    {"name": "German",       "code": "de",   "badge": "DE",   "display": "German (Deutsch)"},
    {"name": "Spanish",      "code": "es",   "badge": "ES",   "display": "Spanish (Espa\u00f1ol)"},
    {"name": "Italian",      "code": "it",   "badge": "IT",   "display": "Italian (Italiano)"},
    {"name": "Portuguese",   "code": "pt",   "badge": "PT",   "display": "Portuguese (Portugu\u00eas)"},
    {"name": "Russian",      "code": "ru",   "badge": "RU",   "display": "Russian (\u0420\u0443\u0441\u0441\u043a\u0438\u0439)"},
    {"name": "Arabic",       "code": "ar",   "badge": "AR",   "display": "Arabic (\u0627\u0644\u0639\u0631\u0628\u064a\u0629)"},
    {"name": "Hindi",        "code": "hi",   "badge": "HI",   "display": "Hindi (\u0939\u093f\u0928\u094d\u0926\u0940)"},
    {"name": "Cantonese",    "code": "yue",  "badge": "\u7cb5",  "display": "Cantonese (\u7cb5\u8a9e)"},
    {"name": "Shanghainese", "code": "wuu",  "badge": "\u6caa",  "display": "Shanghainese (\u4e0a\u6d77\u8bdd)"},
    {"name": "Sichuanese",   "code": "scu",  "badge": "\u5ddd",  "display": "Sichuanese (\u56db\u5ddd\u8bdd)"},
    {"name": "Wenzhounese",  "code": "wz",   "badge": "\u6e29",  "display": "Wenzhounese (\u6e29\u5dde\u8bdd)"},
]

# Default values for all settings keys.
_DEFAULTS: Dict[str, Any] = {
    "languages": ["Auto"],
    "mode": "push_to_talk",
    "response_speed": "fast",
    "activate_hotkey": {"key": "space", "modifiers": ["alt"]},
    # Usage statistics
    "stats_total_transcriptions": 0,
    "stats_total_words": 0,
    "stats_total_chars": 0,
    "stats_time_saved_seconds": 0.0,
    "stats_today_transcriptions": 0,
    "stats_today_words": 0,
    "stats_last_date": "",
    "first_run": True,
    # LLM backend settings
    "llm_source": "local",
    "llm_local_model": "qwen2.5:3b",
    "llm_cloud_provider": "openai",
    "llm_cloud_model": "gpt-4o-mini",
    "translation_language": "English",
    "asr_model": "Qwen/Qwen3-ASR-0.6B",
}

# Type alias for a change callback: (key, old_value, new_value) -> None.
ChangeCallback = Callable[[str, Any, Any], None]


def _is_cjk(char: str) -> bool:
    """Return True if *char* is a CJK ideograph or related character."""
    cp = ord(char)
    return (
        0x4E00 <= cp <= 0x9FFF        # CJK Unified Ideographs
        or 0x3400 <= cp <= 0x4DBF     # CJK Extension A
        or 0x20000 <= cp <= 0x2A6DF   # CJK Extension B
        or 0xF900 <= cp <= 0xFAFF     # CJK Compatibility Ideographs
        or 0x2F800 <= cp <= 0x2FA1F   # CJK Compatibility Supplement
        or 0x3000 <= cp <= 0x303F     # CJK Symbols and Punctuation
        or 0x3040 <= cp <= 0x309F     # Hiragana
        or 0x30A0 <= cp <= 0x30FF     # Katakana
        or 0xAC00 <= cp <= 0xD7AF     # Hangul Syllables
    )


class SettingsManager:
    """Thread-safe settings store backed by a JSON file.

    Usage::

        mgr = SettingsManager()
        mgr.register_callback(my_callback)
        mgr.set("language", "English")
        lang = mgr.get("language")
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._path: Path = config_path or _CONFIG_FILE
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = dict(_DEFAULTS)
        self._callbacks: List[ChangeCallback] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if not present."""
        with self._lock:
            return self._data.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        """Return a shallow copy of all settings."""
        with self._lock:
            return dict(self._data)

    def set(self, key: str, value: Any) -> None:
        """Set *key* to *value*, persist, and notify listeners."""
        with self._lock:
            old = self._data.get(key)
            if old == value:
                return
            self._data[key] = value
            self._save_locked()
            callbacks = list(self._callbacks)

        # Fire callbacks outside the lock.
        for cb in callbacks:
            try:
                cb(key, old, value)
            except Exception:
                pass

    def set_many(self, updates: Dict[str, Any]) -> None:
        """Update multiple keys at once, persist, and notify."""
        changed: List[tuple] = []
        with self._lock:
            for key, value in updates.items():
                old = self._data.get(key)
                if old != value:
                    self._data[key] = value
                    changed.append((key, old, value))
            if changed:
                self._save_locked()
            callbacks = list(self._callbacks)

        for key, old, value in changed:
            for cb in callbacks:
                try:
                    cb(key, old, value)
                except Exception:
                    pass

    def register_callback(self, callback: ChangeCallback) -> None:
        """Register a function to be called on setting changes.

        The callback signature is ``(key, old_value, new_value)``.
        """
        with self._lock:
            self._callbacks.append(callback)

    def unregister_callback(self, callback: ChangeCallback) -> None:
        """Remove a previously registered callback."""
        with self._lock:
            try:
                self._callbacks.remove(callback)
            except ValueError:
                pass

    def reset_to_defaults(self) -> None:
        """Reset all settings to their default values and persist."""
        with self._lock:
            self._data = dict(_DEFAULTS)
            self._save_locked()

    # ------------------------------------------------------------------
    # Usage statistics
    # ------------------------------------------------------------------

    def record_transcription(self, text: str) -> None:
        """Record stats for a completed transcription."""
        # Count CJK characters and English words separately
        cjk_chars = sum(1 for ch in text if _is_cjk(ch))
        # Non-CJK portion: strip out CJK chars, split by whitespace
        non_cjk_text = "".join(ch if not _is_cjk(ch) else " " for ch in text)
        en_words = len(non_cjk_text.split())
        total_words = en_words + cjk_chars
        total_chars = len(text)

        # Estimate time saved (seconds)
        en_time = (en_words / 40) * 60 if en_words else 0.0
        cjk_time = (cjk_chars / 20) * 60 if cjk_chars else 0.0
        time_saved = en_time + cjk_time

        today = date.today().isoformat()

        # Perform atomic read-modify-write under a single lock acquisition
        # to prevent TOCTOU races with concurrent transcriptions.
        with self._lock:
            # Daily reset
            if self._data.get("stats_last_date") != today:
                self._data["stats_today_transcriptions"] = 0
                self._data["stats_today_words"] = 0
                self._data["stats_last_date"] = today

            # Totals
            self._data["stats_total_transcriptions"] = self._data.get("stats_total_transcriptions", 0) + 1
            self._data["stats_total_words"] = self._data.get("stats_total_words", 0) + total_words
            self._data["stats_total_chars"] = self._data.get("stats_total_chars", 0) + total_chars
            self._data["stats_time_saved_seconds"] = self._data.get("stats_time_saved_seconds", 0.0) + time_saved

            # Daily
            self._data["stats_today_transcriptions"] = self._data.get("stats_today_transcriptions", 0) + 1
            self._data["stats_today_words"] = self._data.get("stats_today_words", 0) + total_words

            self._save_locked()

    def get_stats(self) -> dict:
        """Return all usage statistics including computed display values."""
        keys = [
            "stats_total_transcriptions",
            "stats_total_words",
            "stats_total_chars",
            "stats_time_saved_seconds",
            "stats_today_transcriptions",
            "stats_today_words",
            "stats_last_date",
        ]
        # Read all stats under a single lock to get a consistent snapshot
        with self._lock:
            stats = {k: self._data.get(k) for k in keys}

        # Formatted time saved display
        seconds = stats["stats_time_saved_seconds"] or 0.0
        total_minutes = int(seconds // 60)
        if total_minutes >= 60:
            hours = total_minutes // 60
            minutes = total_minutes % 60
            stats["stats_time_saved_display"] = f"{hours}h {minutes}m"
        else:
            stats["stats_time_saved_display"] = f"{total_minutes}m"

        return stats

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load settings from disk, falling back to defaults."""
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                stored = json.load(f)
            if isinstance(stored, dict):
                # Migrate old "language" (string) to "languages" (list).
                migrated = False
                if "language" in stored and "languages" not in stored:
                    old_lang = stored.pop("language")
                    stored["languages"] = [old_lang] if old_lang else ["Auto"]
                    migrated = True
                elif "language" in stored:
                    stored.pop("language")
                    migrated = True
                # Merge stored values over defaults so new keys get defaults.
                self._data.update(stored)
                if migrated:
                    self._save_locked()
        except (json.JSONDecodeError, OSError):
            pass

    def _save_locked(self) -> None:
        """Write the current settings to disk. Must hold ``_lock``.

        Writes to a temporary file first, then atomically renames to
        the target path to prevent corruption if the process crashes
        mid-write.
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self._path)
        except OSError:
            pass
