# Type4Me-Inspired SafeVoice Improvements

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring SafeVoice's UX to parity with Type4Me by adding transcription history, vocabulary management, custom processing modes, speculative LLM, floating bar polish, and a setup wizard.

**Architecture:** Six independent features layered onto the existing rumps + PyObjC app. Each feature is a new module with clean interfaces into the existing `app.py` state machine. Settings extensions go through `SettingsManager`. All UI is native macOS via PyObjC (NSWindow/NSPanel subclasses). New attributes follow the existing `self._` private convention.

**Tech Stack:** Python 3.14, PyObjC (AppKit/Quartz/Foundation), rumps, SQLite3 (stdlib), sounddevice, mlx-qwen3-asr, Ollama (qwen2.5:3b)

**Key codebase conventions:**
- All instance attributes use `self._` prefix (e.g. `self._settings`, `self._llm`, `self._asr`)
- Main transcription logic lives in a `transcribe()` closure inside `_stop_listening_and_transcribe()` (line 454 of `app.py`)
- `LLMCleanup.cleanup()` signature: `cleanup(self, raw_text: str, languages: Optional[list] = None) -> str`
- `LLMCleanup.is_available()` checks if Ollama + model are ready
- `ASREngine.is_loaded` property checks if model is loaded
- PyObjC NSColor: use `NSColor.colorWithCalibratedRed_green_blue_alpha_()` (not the short form)
- Thread-safe UI updates via `_Trampoline` NSObject pattern (see `overlay.py`)
- Overlay is fixed-width (`_PANEL_MIN_WIDTH=280`, `_PANEL_MAX_WIDTH=480`) — no dynamic resizing currently

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `tests/__init__.py` | Test package marker |
| Create | `tests/conftest.py` | Path setup for imports |
| Create | `src/history.py` | SQLite-backed transcription history with CSV export |
| Create | `src/vocabulary.py` | ASR hotwords + snippet replacements |
| Create | `src/modes.py` | Processing modes (Quick + custom LLM modes) |
| Create | `src/setup_wizard.py` | 6-step onboarding wizard window |
| Create | `tests/test_history.py` | History module tests |
| Create | `tests/test_vocabulary.py` | Vocabulary module tests |
| Create | `tests/test_modes.py` | Modes module tests |
| Modify | `src/overlay.py` | Dynamic width, peak-width pattern, state badges |
| Modify | `src/settings_manager.py` | New keys for first_run flag |
| Modify | `src/settings_window.py` | New tabs: Modes, Vocabulary |
| Modify | `src/app.py` | Wire up modes, vocabulary, history, speculative LLM, setup wizard trigger |
| Modify | `src/llm_cleanup.py` | Accept custom prompts, speculative pre-processing |

---

## Task 0: Test Infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

### Step 0.1: Create test package

- [ ] Create `tests/__init__.py`:

```python
```

- [ ] Create `tests/conftest.py`:

```python
"""Pytest configuration for SafeVoice tests."""
import sys
import os

# Add project root to path so `from src.X import Y` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

- [ ] Run: `cd ~/Documents/Vibe\ Code/voice-ime && python -m pytest tests/ -v --collect-only`
- [ ] Expected: "no tests ran" (no error)

### Step 0.2: Commit

- [ ] ```bash
git add tests/__init__.py tests/conftest.py
git commit -m "chore: add pytest test infrastructure"
```

---

## Task 1: Transcription History (SQLite + CSV Export)

**Files:**
- Create: `src/history.py`
- Create: `tests/test_history.py`
- Modify: `src/app.py:454-509` (wire history recording in `transcribe()` closure)

### Step 1.1: Write failing tests for history store

- [ ] Create `tests/test_history.py`:

```python
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
```

- [ ] Run: `cd ~/Documents/Vibe\ Code/voice-ime && python -m pytest tests/test_history.py -v`
- [ ] Expected: FAIL (`ModuleNotFoundError: No module named 'src.history'`)

### Step 1.2: Implement HistoryStore

- [ ] Create `src/history.py`:

```python
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
            conn.execute(
                "INSERT INTO history (timestamp, final_text, raw_text, mode, duration, language, word_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), final_text, raw_text, mode, duration, language, word_count)
            )
            conn.commit()
            conn.close()

    def get_recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                "SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]

    def get_by_date(self, start_date: str, end_date: str) -> list[dict]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                "SELECT * FROM history WHERE timestamp >= ? AND timestamp <= ? ORDER BY id DESC",
                (start_date, end_date + "T23:59:59")
            ).fetchall()
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
            row = conn.execute(
                "SELECT COUNT(*) as total, COALESCE(SUM(word_count), 0) as words, "
                "COALESCE(SUM(duration), 0) as duration FROM history"
            ).fetchone()
            conn.close()
            return {
                "total_transcriptions": row["total"],
                "total_words": row["words"],
                "total_duration": row["duration"],
            }
```

- [ ] Run: `cd ~/Documents/Vibe\ Code/voice-ime && python -m pytest tests/test_history.py -v`
- [ ] Expected: All 6 tests PASS

### Step 1.3: Wire history into app.py

- [ ] In `src/app.py`, add import after the existing imports (line 33):
```python
from .history import HistoryStore
```

- [ ] In `SafeVoiceApp.__init__()`, after `self._llm = LLMCleanup()` (line 100), add:
```python
self._history = HistoryStore()
```

- [ ] In the `transcribe()` closure inside `_stop_listening_and_transcribe()`, after `self._settings.record_transcription(text)` (line 492), add:
```python
                    # Record to history database
                    elapsed = time.monotonic() - timer_start
                    raw_for_history = stripped  # ASR output before LLM cleanup
                    self._history.add(
                        final_text=text,
                        raw_text=raw_for_history,
                        mode="quick",
                        duration=elapsed,
                        language=self._settings.get("languages", ["Auto"])[0],
                    )
```

Note: `stripped` is the ASR text before LLM cleanup (line 472). `text` is the final text after optional LLM cleanup. We need to capture `stripped` before the LLM section to use as `raw_for_history`. Add `raw_for_history = stripped` right after line 472.

- [ ] Test manually: run SafeVoice, do a transcription, verify `~/.config/safevoice/history.db` is created

### Step 1.4: Commit

- [ ] ```bash
git add src/history.py tests/test_history.py src/app.py
git commit -m "feat: add SQLite transcription history with CSV export"
```

---

## Task 2: Vocabulary Management (Hotwords + Snippets)

**Files:**
- Create: `src/vocabulary.py`
- Create: `tests/test_vocabulary.py`
- Modify: `src/app.py` (apply snippets in `transcribe()` closure)

### Step 2.1: Write failing tests

- [ ] Create `tests/test_vocabulary.py`:

```python
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
```

- [ ] Run: `python -m pytest tests/test_vocabulary.py -v`
- [ ] Expected: FAIL (`ModuleNotFoundError`)

### Step 2.2: Implement VocabularyManager

- [ ] Create `src/vocabulary.py`:

```python
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
```

- [ ] Run: `python -m pytest tests/test_vocabulary.py -v`
- [ ] Expected: All 7 tests PASS

### Step 2.3: Wire vocabulary into app.py

- [ ] In `src/app.py`, add import:
```python
from .vocabulary import VocabularyManager
```

- [ ] In `__init__`, after `self._history = HistoryStore()`:
```python
self._vocabulary = VocabularyManager()
```

- [ ] In the `transcribe()` closure, after `text, lang = self._asr.transcribe(cleaned)` (line 463), add:
```python
                    # Apply vocabulary snippet replacements
                    text = self._vocabulary.apply_snippets(text)
```

This runs snippet expansion before LLM cleanup, so snippets are expanded before the LLM sees the text.

### Step 2.4: Commit

- [ ] ```bash
git add src/vocabulary.py tests/test_vocabulary.py src/app.py
git commit -m "feat: add vocabulary manager with hotwords and snippet replacements"
```

---

## Task 3: Custom Processing Modes

**Files:**
- Create: `src/modes.py`
- Create: `tests/test_modes.py`
- Modify: `src/llm_cleanup.py:143` (add `custom_prompt` parameter to `cleanup()`)
- Modify: `src/app.py` (use active mode in `transcribe()` closure)

### Step 3.1: Write failing tests for modes

- [ ] Create `tests/test_modes.py`:

```python
"""Tests for processing modes."""
import pytest
from src.modes import ModeManager, Mode


@pytest.fixture
def manager(tmp_path):
    path = str(tmp_path / "modes.json")
    return ModeManager(path)


def test_default_modes(manager):
    modes = manager.get_all()
    names = [m.name for m in modes]
    assert "Quick" in names
    assert "Formal Writing" in names
    assert "English Translation" in names


def test_quick_mode_has_no_prompt(manager):
    quick = manager.get("Quick")
    assert quick is not None
    assert quick.prompt_template is None
    assert quick.builtin is True


def test_custom_mode_has_prompt(manager):
    manager.add(Mode(
        name="Summarize",
        prompt_template="Summarize this text concisely: {text}",
        hotkey={"key": "s", "modifiers": ["alt", "cmd"]},
    ))
    mode = manager.get("Summarize")
    assert mode is not None
    assert "{text}" in mode.prompt_template


def test_render_prompt(manager):
    manager.add(Mode(
        name="Test",
        prompt_template="Fix grammar: {text}",
    ))
    mode = manager.get("Test")
    result = mode.render_prompt("hello world")
    assert result == "Fix grammar: hello world"


def test_remove_custom_mode(manager):
    manager.add(Mode(name="Temp", prompt_template="Do: {text}"))
    manager.remove("Temp")
    assert manager.get("Temp") is None


def test_cannot_remove_builtin(manager):
    manager.remove("Quick")
    assert manager.get("Quick") is not None  # still there


def test_persistence(tmp_path):
    path = str(tmp_path / "modes.json")
    m1 = ModeManager(path)
    m1.add(Mode(name="Custom1", prompt_template="Do: {text}"))

    m2 = ModeManager(path)
    assert m2.get("Custom1") is not None


def test_mode_hotkey(manager):
    manager.add(Mode(
        name="Translate",
        prompt_template="Translate to English: {text}",
        hotkey={"key": "t", "modifiers": ["alt"]},
    ))
    mode = manager.get("Translate")
    assert mode.hotkey == {"key": "t", "modifiers": ["alt"]}
```

- [ ] Run: `python -m pytest tests/test_modes.py -v`
- [ ] Expected: FAIL (`ModuleNotFoundError`)

### Step 3.2: Implement ModeManager

- [ ] Create `src/modes.py`:

```python
"""Processing modes: Quick (direct ASR) and custom LLM modes with per-mode hotkeys."""
import json
import os
import threading
from dataclasses import dataclass, asdict


@dataclass
class Mode:
    name: str
    prompt_template: str | None = None  # None = direct ASR (Quick mode)
    hotkey: dict | None = None  # {"key": "space", "modifiers": ["alt"]}
    builtin: bool = False
    enabled: bool = True

    def render_prompt(self, text: str) -> str:
        if self.prompt_template is None:
            return text
        return self.prompt_template.replace("{text}", text)


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
```

- [ ] Run: `python -m pytest tests/test_modes.py -v`
- [ ] Expected: All 8 tests PASS

### Step 3.3: Update llm_cleanup.py to accept custom prompts

- [ ] In `src/llm_cleanup.py`, modify the `cleanup` method (line 143). Change signature from:

```python
def cleanup(self, raw_text: str, languages: Optional[list] = None) -> str:
```

to:

```python
def cleanup(self, raw_text: str, languages: Optional[list] = None, custom_prompt: str = None) -> str:
```

At the start of the method body, after the `if not raw_text.strip()` check (line 156) and the `if not self.is_available()` check (line 158), add a custom prompt branch before the existing `try` block:

```python
        if custom_prompt:
            try:
                messages = [
                    {"role": "system", "content": "Follow the instruction precisely. Output only the result."},
                    {"role": "user", "content": custom_prompt},
                ]
                body = json.dumps({
                    "model": self._model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 512, "top_p": 0.9},
                }).encode()
                req = urllib.request.Request(
                    f"{OLLAMA_BASE}/api/chat",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read())
                    result = data["message"]["content"].strip()
                    # Strip <think> tags
                    if "<think>" in result:
                        idx = result.find("</think>")
                        if idx != -1:
                            result = result[idx + len("</think>"):].strip()
                        else:
                            result = result[:result.find("<think>")].strip()
                    if result:
                        logger.info("Custom LLM: %r -> %r", raw_text, result)
                        return result
            except Exception as e:
                logger.warning("Custom LLM cleanup failed: %s", e)
            return raw_text
```

The existing cleanup logic (system prompt + few-shot + translation guard) remains unchanged for `custom_prompt=None`.

### Step 3.4: Wire modes into app.py

- [ ] In `src/app.py`, add import:
```python
from .modes import ModeManager
```

- [ ] In `__init__`, after `self._vocabulary = VocabularyManager()`:
```python
self._modes = ModeManager()
self._active_mode = self._modes.get("Quick")
```

- [ ] In the `transcribe()` closure, replace the existing LLM cleanup block (lines 471-486):

**Old** (lines 471-486):
```python
                    # LLM cleanup for text polishing
                    stripped = text.strip()
                    has_cjk = any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af' for c in stripped)
                    is_long_enough = len(stripped) >= 4 if has_cjk else len(stripped.split()) >= 3
                    if self._llm.is_available() and is_long_enough:
                        self._overlay.set_status("processing")
                        self._update_status("Cleaning up...")
                        logger.info("LLM cleanup starting...")
                        cleaned = self._llm.cleanup(text)
                        logger.info("LLM result: %r", cleaned)
                        if cleaned != text:
                            print(f"[SafeVoice] LLM: {text!r} -> {cleaned!r}")
                            text = cleaned
                            self._overlay.update_text(text)
                    elif not is_long_enough:
                        logger.info("Skipping LLM cleanup for short text: %r", stripped)
```

**New:**
```python
                    # LLM processing (mode-aware)
                    stripped = text.strip()
                    raw_for_history = stripped
                    has_cjk = any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af' for c in stripped)
                    is_long_enough = len(stripped) >= 4 if has_cjk else len(stripped.split()) >= 3

                    if self._active_mode.prompt_template and self._llm.is_available() and is_long_enough:
                        # Custom mode: use mode's LLM prompt
                        self._overlay.set_status("processing")
                        self._update_status(f"{self._active_mode.name}...")
                        prompt = self._active_mode.render_prompt(stripped)
                        logger.info("Mode '%s' LLM starting...", self._active_mode.name)
                        # Check speculative cache first
                        cached = self._llm.get_speculative_result(stripped)
                        if cached:
                            logger.info("Using speculative result")
                            text = cached
                        else:
                            cleaned = self._llm.cleanup(text, custom_prompt=prompt)
                            if cleaned != text:
                                text = cleaned
                        self._overlay.update_text(text)
                        logger.info("Mode result: %r", text)
                    elif self._active_mode.prompt_template is None and self._llm.is_available() and is_long_enough:
                        # Quick mode: default cleanup
                        self._overlay.set_status("processing")
                        self._update_status("Cleaning up...")
                        logger.info("LLM cleanup starting...")
                        cleaned = self._llm.cleanup(text)
                        logger.info("LLM result: %r", cleaned)
                        if cleaned != text:
                            print(f"[SafeVoice] LLM: {text!r} -> {cleaned!r}")
                            text = cleaned
                            self._overlay.update_text(text)
                    elif not is_long_enough:
                        raw_for_history = stripped
                        logger.info("Skipping LLM for short text: %r", stripped)
```

- [ ] Update the history recording to use mode name and `raw_for_history`:
```python
                    self._history.add(
                        final_text=text,
                        raw_text=raw_for_history,
                        mode=self._active_mode.name,
                        duration=time.monotonic() - timer_start,
                        language=self._settings.get("languages", ["Auto"])[0],
                    )
```

- [ ] In `_build_menu()`, add a Modes submenu showing all modes:
```python
        modes_menu = rumps.MenuItem("Modes")
        for mode in self._modes.get_all():
            hotkey_str = ""
            if mode.hotkey:
                mods = "+".join(m.title() for m in mode.hotkey.get("modifiers", []))
                key = mode.hotkey.get("key", "").title()
                hotkey_str = f" ({mods}+{key})" if mods else f" ({key})"
            item = rumps.MenuItem(f"{mode.name}{hotkey_str}")
            modes_menu[item.title] = item
        self.menu.add(modes_menu)
```

### Step 3.5: Commit

- [ ] ```bash
git add src/modes.py tests/test_modes.py src/llm_cleanup.py src/app.py
git commit -m "feat: add custom processing modes with per-mode LLM prompts"
```

---

## Task 4: Speculative LLM Pre-processing

**Files:**
- Modify: `src/llm_cleanup.py` (add speculative cache methods)
- Modify: `src/app.py` (trigger speculative calls via periodic timer during recording)

### Step 4.1: Add speculative cache to llm_cleanup.py

- [ ] In `src/llm_cleanup.py`, add to `__init__` (after `self._available`):

```python
        self._speculative_result: Optional[str] = None
        self._speculative_input: Optional[str] = None
        self._speculative_lock = threading.Lock()
```

Add `import threading` at the top of the file.

- [ ] Add three new methods after `warm_up()`:

```python
    def speculative_cleanup(self, text: str, custom_prompt: str = None):
        """Fire-and-forget: run cleanup in background, cache result."""
        def _run():
            result = self.cleanup(text, custom_prompt=custom_prompt)
            with self._speculative_lock:
                self._speculative_input = text
                self._speculative_result = result
        threading.Thread(target=_run, daemon=True).start()

    def get_speculative_result(self, text: str) -> Optional[str]:
        """Return cached result if input matches, else None."""
        with self._speculative_lock:
            if self._speculative_input == text and self._speculative_result:
                result = self._speculative_result
                self._speculative_input = None
                self._speculative_result = None
                return result
        return None

    def clear_speculative(self):
        """Clear any cached speculative result."""
        with self._speculative_lock:
            self._speculative_input = None
            self._speculative_result = None
```

### Step 4.2: Wire speculative LLM into app.py

Since SafeVoice uses batch transcription (not streaming), we can't detect speech pauses mid-recording. Instead, implement a **periodic speculative approach**: every N seconds during recording, take the audio captured so far, run a quick ASR on it, and speculatively send that text through the active mode's LLM.

- [ ] In `SafeVoiceApp.__init__`, add:
```python
        self._speculative_timer = None
        self._speculative_interval = 3.0  # seconds
```

- [ ] In `_start_listening()`, after starting audio capture, add:
```python
        # Start speculative LLM if using a custom mode
        if self._active_mode.prompt_template and self._llm.is_available():
            self._start_speculative_timer()
```

- [ ] Add these methods:
```python
    def _start_speculative_timer(self):
        """Periodically run ASR on captured audio and speculatively send to LLM."""
        self._speculative_timer_stop = threading.Event()

        def _speculate():
            while not self._speculative_timer_stop.wait(self._speculative_interval):
                if self._state != STATE_LISTENING:
                    break
                # Get audio captured so far
                with self._audio_lock:
                    if not self._audio_chunks:
                        continue
                    audio_so_far = np.concatenate(self._audio_chunks)
                if len(audio_so_far) < 16000:  # less than 1 second
                    continue
                try:
                    cleaned = audio_preprocess.normalize_audio(audio_so_far)
                    text, _ = self._asr.transcribe(cleaned)
                    if text.strip():
                        text = self._vocabulary.apply_snippets(text)
                        prompt = self._active_mode.render_prompt(text.strip())
                        self._llm.speculative_cleanup(text.strip(), custom_prompt=prompt)
                except Exception as e:
                    logger.debug("Speculative ASR failed: %s", e)

        threading.Thread(target=_speculate, daemon=True).start()

    def _stop_speculative_timer(self):
        if hasattr(self, '_speculative_timer_stop'):
            self._speculative_timer_stop.set()
```

- [ ] In `_stop_listening_and_transcribe()`, at the start, add:
```python
        self._stop_speculative_timer()
```

The speculative result is already consumed in Task 3.4's modified `transcribe()` closure via `self._llm.get_speculative_result()`.

### Step 4.3: Commit

- [ ] ```bash
git add src/llm_cleanup.py src/app.py
git commit -m "feat: speculative LLM pre-processing during recording"
```

---

## Task 5: Floating Bar Polish

**Files:**
- Modify: `src/overlay.py`

### Step 5.1: Add dynamic width calculation

The overlay is currently fixed-width. Add dynamic resizing based on text content.

- [ ] In `src/overlay.py`, in `FloatingOverlay.__init__`, add:
```python
        self._peak_width = _PANEL_MIN_WIDTH
        self._current_status = "listening"
```

- [ ] In the `update_text` method, after setting the text label, add dynamic width calculation:

```python
        # Calculate width to fit text
        if self._text_label is not None:
            text_size = self._text_label.attributedStringValue().size()
            desired = _HORIZONTAL_PADDING * 2 + _DOT_SIZE + _ELEMENT_SPACING + _BADGE_WIDTH + _ELEMENT_SPACING + text_size.width + 20
            desired = max(_PANEL_MIN_WIDTH, min(_PANEL_MAX_WIDTH, desired))

            # Peak-width pattern: only grow during listening, never shrink
            if self._current_status == "listening":
                self._peak_width = max(self._peak_width, desired)
                desired = self._peak_width
            else:
                self._peak_width = _PANEL_MIN_WIDTH  # reset for next recording

            # Animate width change
            self._resize_panel(desired)
```

- [ ] Add the `_resize_panel` method (dispatch to main thread):

```python
    def _resize_panel(self, new_width: float):
        """Resize the panel width, centered horizontally, with animation."""
        def _do_resize():
            if self.panel is None:
                return
            frame = self.panel.frame()
            screen = NSScreen.mainScreen().visibleFrame()
            # Center horizontally
            new_x = screen.origin.x + (screen.size.width - new_width) / 2
            new_frame = NSMakeRect(new_x, frame.origin.y, new_width, frame.size.height)
            NSAnimationContext.beginGrouping()
            NSAnimationContext.currentContext().setDuration_(0.15)
            self.panel.animator().setFrame_display_(new_frame, True)
            NSAnimationContext.endGrouping()
        self._dispatch_to_main(_do_resize)
```

### Step 5.2: Add state badges

- [ ] In the `_build_ui` method (or equivalent panel setup), after the status dot, add a badge label:

```python
        # State badge (AI/OK) - positioned right of the status dot
        badge_x = _HORIZONTAL_PADDING + _DOT_SIZE + 2
        badge_y = (_PANEL_HEIGHT - 16) / 2
        self._badge_label = NSTextField.alloc().initWithFrame_(NSMakeRect(badge_x, badge_y, 24, 16))
        self._badge_label.setStringValue_("")
        self._badge_label.setFont_(NSFont.boldSystemFontOfSize_(9))
        self._badge_label.setBezeled_(False)
        self._badge_label.setDrawsBackground_(True)
        self._badge_label.setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.7, 0.0, 1.0)
        )
        self._badge_label.setTextColor_(NSColor.whiteColor())
        self._badge_label.setAlignment_(NSTextAlignmentCenter)
        self._badge_label.setHidden_(True)
        self._badge_label.setWantsLayer_(True)
        self._badge_label.layer().setCornerRadius_(4)
        self._badge_label.layer().setMasksToBounds_(True)
        self._content_view.addSubview_(self._badge_label)
```

- [ ] In `set_status`, update badge visibility based on status:

```python
        # Update state badge
        if self._badge_label is not None:
            if status == "processing":
                self._badge_label.setStringValue_("AI")
                self._badge_label.setBackgroundColor_(
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.7, 0.0, 1.0)
                )
                self._badge_label.setHidden_(False)
            elif status == "done":
                self._badge_label.setStringValue_("OK")
                self._badge_label.setBackgroundColor_(
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(0.3, 0.85, 0.4, 1.0)
                )
                self._badge_label.setHidden_(False)
            else:
                self._badge_label.setHidden_(True)
```

### Step 5.3: Improve show/hide animation

- [ ] Replace abrupt `setAlphaValue_` changes with animated transitions using `NSAnimationContext`:

In `show()`:
```python
        NSAnimationContext.beginGrouping()
        NSAnimationContext.currentContext().setDuration_(_FADE_DURATION)
        self.panel.animator().setAlphaValue_(1.0)
        NSAnimationContext.endGrouping()
```

In `hide()`:
```python
        NSAnimationContext.beginGrouping()
        NSAnimationContext.currentContext().setDuration_(_FADE_DURATION)
        NSAnimationContext.currentContext().setCompletionHandler_(lambda: self.panel.orderOut_(None) if self.panel else None)
        self.panel.animator().setAlphaValue_(0.0)
        NSAnimationContext.endGrouping()
```

### Step 5.4: Commit

- [ ] ```bash
git add src/overlay.py
git commit -m "feat: floating bar polish — dynamic width, peak-width, state badges, smooth animations"
```

---

## Task 6: Settings Window Updates (Modes + Vocabulary Tabs)

**Files:**
- Modify: `src/settings_window.py`

### Step 6.1: Add Modes tab

- [ ] In `src/settings_window.py`, in the `_build_ui` method where tabs are created, add a 4th tab item after the existing 3 tabs. Follow the exact same pattern as the existing tabs (NSTabViewItem + content view):

```python
        # --- Modes tab ---
        modes_tab = NSTabViewItem.alloc().initWithIdentifier_("modes")
        modes_tab.setLabel_("Modes")
        modes_view = NSView.alloc().initWithFrame_(content_rect)

        # Title
        modes_title = NSTextField.labelWithString_("Processing Modes")
        modes_title.setFont_(NSFont.boldSystemFontOfSize_(14))
        modes_title.setFrame_(NSMakeRect(20, content_rect.size.height - 40, 300, 20))
        modes_view.addSubview_(modes_title)

        # Description
        modes_desc = NSTextField.labelWithString_(
            "Each mode processes your speech differently. Quick mode gives raw text. "
            "Other modes use AI to transform the text."
        )
        modes_desc.setFont_(NSFont.systemFontOfSize_(11))
        modes_desc.setTextColor_(NSColor.secondaryLabelColor())
        modes_desc.setFrame_(NSMakeRect(20, content_rect.size.height - 70, 350, 30))
        modes_view.addSubview_(modes_desc)

        # Mode list (read-only display for now)
        y = content_rect.size.height - 100
        if hasattr(self, '_modes_manager'):
            for mode in self._modes_manager.get_all():
                hotkey_str = ""
                if mode.hotkey:
                    from src.settings_window import format_hotkey
                    hotkey_str = format_hotkey(mode.hotkey)
                label_text = f"{mode.name}  —  {hotkey_str}" if hotkey_str else mode.name
                label = NSTextField.labelWithString_(label_text)
                label.setFont_(NSFont.systemFontOfSize_(13))
                label.setFrame_(NSMakeRect(20, y, 350, 20))
                modes_view.addSubview_(label)
                if mode.prompt_template:
                    prompt_preview = mode.prompt_template[:60] + "..." if len(mode.prompt_template) > 60 else mode.prompt_template
                    prompt_label = NSTextField.labelWithString_(prompt_preview)
                    prompt_label.setFont_(NSFont.systemFontOfSize_(10))
                    prompt_label.setTextColor_(NSColor.tertiaryLabelColor())
                    prompt_label.setFrame_(NSMakeRect(30, y - 18, 340, 16))
                    modes_view.addSubview_(prompt_label)
                    y -= 44
                else:
                    y -= 28

        modes_tab.setView_(modes_view)
        tab_view.addTabViewItem_(modes_tab)
```

Pass the `ModeManager` instance to `SettingsWindow.__init__`:

In `src/app.py`, update `self._settings_window` creation:
```python
self._settings_window = SettingsWindow(self._settings, modes_manager=self._modes)
```

In `src/settings_window.py`, update `__init__` to accept and store `modes_manager`:
```python
def __init__(self, settings_manager, on_setting_changed=None, modes_manager=None, vocabulary_manager=None):
    ...
    self._modes_manager = modes_manager
    self._vocabulary_manager = vocabulary_manager
```

### Step 6.2: Add Vocabulary tab

- [ ] Add a 5th tab for vocabulary, following the same pattern:

```python
        # --- Vocabulary tab ---
        vocab_tab = NSTabViewItem.alloc().initWithIdentifier_("vocab")
        vocab_tab.setLabel_("Vocabulary")
        vocab_view = NSView.alloc().initWithFrame_(content_rect)

        # Title
        vocab_title = NSTextField.labelWithString_("Vocabulary & Snippets")
        vocab_title.setFont_(NSFont.boldSystemFontOfSize_(14))
        vocab_title.setFrame_(NSMakeRect(20, content_rect.size.height - 40, 300, 20))
        vocab_view.addSubview_(vocab_title)

        # Hotwords section
        hw_label = NSTextField.labelWithString_("ASR Hotwords (improve recognition of proper nouns)")
        hw_label.setFont_(NSFont.systemFontOfSize_(11))
        hw_label.setTextColor_(NSColor.secondaryLabelColor())
        hw_label.setFrame_(NSMakeRect(20, content_rect.size.height - 70, 350, 16))
        vocab_view.addSubview_(hw_label)

        y = content_rect.size.height - 95
        if self._vocabulary_manager:
            for word in self._vocabulary_manager.get_hotwords():
                w_label = NSTextField.labelWithString_(word)
                w_label.setFont_(NSFont.systemFontOfSize_(12))
                w_label.setFrame_(NSMakeRect(30, y, 300, 18))
                vocab_view.addSubview_(w_label)
                y -= 22

        # Snippets section
        y -= 10
        sn_label = NSTextField.labelWithString_("Snippets (auto-replace phrases)")
        sn_label.setFont_(NSFont.systemFontOfSize_(11))
        sn_label.setTextColor_(NSColor.secondaryLabelColor())
        sn_label.setFrame_(NSMakeRect(20, y, 350, 16))
        vocab_view.addSubview_(sn_label)
        y -= 25

        if self._vocabulary_manager:
            for trigger, replacement in self._vocabulary_manager.get_snippets().items():
                s_label = NSTextField.labelWithString_(f'"{trigger}" → "{replacement}"')
                s_label.setFont_(NSFont.systemFontOfSize_(12))
                s_label.setFrame_(NSMakeRect(30, y, 340, 18))
                vocab_view.addSubview_(s_label)
                y -= 22

        vocab_tab.setView_(vocab_view)
        tab_view.addTabViewItem_(vocab_tab)
```

Pass `vocabulary_manager` from `app.py`:
```python
self._settings_window = SettingsWindow(
    self._settings,
    modes_manager=self._modes,
    vocabulary_manager=self._vocabulary,
)
```

### Step 6.3: Commit

- [ ] ```bash
git add src/settings_window.py src/app.py
git commit -m "feat: add Modes and Vocabulary tabs to settings window"
```

---

## Task 7: Setup Wizard

**Files:**
- Create: `src/setup_wizard.py`
- Modify: `src/settings_manager.py` (add `first_run` flag)
- Modify: `src/app.py` (trigger wizard on first launch)

### Step 7.1: Add first_run flag to settings

- [ ] In `src/settings_manager.py`, in the `_DEFAULTS` dict (or equivalent default settings), add:
```python
"first_run": True,
```

### Step 7.2: Implement setup wizard

- [ ] Create `src/setup_wizard.py`:

```python
"""Setup wizard for first-time SafeVoice users."""
import logging
import subprocess
import threading

import objc
from AppKit import (
    NSWindow, NSView, NSTextField, NSButton, NSFont, NSColor,
    NSMakeRect, NSBackingStoreBuffered,
    NSWindowStyleMaskTitled, NSWindowStyleMaskClosable,
    NSScreen, NSObject, NSTextAlignmentCenter, NSAnimationContext,
)
from ApplicationServices import AXIsProcessTrusted
from Foundation import NSTimer

logger = logging.getLogger(__name__)


STEPS = ["Welcome", "Demo", "Permissions", "Model", "Test", "Ready"]


class _WizardButtonTarget(NSObject):
    """NSObject target for wizard button actions. Prevents GC."""

    def initWithCallback_(self, callback):
        self = objc.super(_WizardButtonTarget, self).init()
        if self is None:
            return None
        self._callback = callback
        return self

    def invoke_(self, sender):
        if self._callback:
            self._callback()


class SetupWizard:
    """6-step onboarding wizard for first-time users."""

    def __init__(self, app_ref, on_complete=None):
        self._app = app_ref
        self._on_complete = on_complete
        self._current_step = 0
        self._window = None
        self._content_view = None
        self._targets = set()  # prevent GC of button targets
        self._perm_timer = None

    def show(self):
        frame = NSMakeRect(0, 0, 520, 420)
        style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
        self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, style, NSBackingStoreBuffered, False,
        )
        self._window.setTitle_("SafeVoice Setup")
        self._window.center()
        self._window.setLevel_(3)  # floating
        self._content_view = self._window.contentView()
        self._render_step()
        self._window.makeKeyAndOrderFront_(None)

    def _clear_content(self):
        for subview in list(self._content_view.subviews()):
            subview.removeFromSuperview()

    def _render_step(self):
        self._clear_content()
        self._render_progress()
        step = STEPS[self._current_step]
        getattr(self, f"_render_{step.lower()}")()

    def _render_progress(self):
        y = 380
        total_width = 300
        seg_w = total_width / len(STEPS) - 4
        start_x = (520 - total_width) / 2
        for i in range(len(STEPS)):
            seg = NSView.alloc().initWithFrame_(
                NSMakeRect(start_x + i * (seg_w + 4), y, seg_w, 6)
            )
            seg.setWantsLayer_(True)
            if i <= self._current_step:
                seg.layer().setBackgroundColor_(
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.7, 0.0, 1.0).CGColor()
                )
            else:
                seg.layer().setBackgroundColor_(
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(0.85, 0.85, 0.85, 1.0).CGColor()
                )
            seg.layer().setCornerRadius_(3)
            self._content_view.addSubview_(seg)

    def _render_welcome(self):
        self._label("SafeVoice", 28, bold=True, y=300, center=True)
        self._label("Speak, and it types.", 16, y=265, center=True)
        self._label(
            "100% on-device voice input for macOS.\n"
            "No internet required. Your voice never leaves your Mac.",
            13, y=195, center=True, width=400, height=50,
        )
        self._button("Get Started", y=80, action=self._next_step)

    def _render_demo(self):
        self._label("How it works", 22, bold=True, y=320, center=True)
        self._label(
            "Hold your hotkey to speak. A floating bar appears showing\n"
            "your words in real-time. Release to inject text at your cursor.",
            13, y=255, center=True, width=420, height=40,
        )
        self._label(
            "  Recording      Processing      Done  \n"
            "  \u25cf Listening       \u25cf AI working      \u2713 Injected",
            13, y=170, center=True, width=400, height=40,
        )
        self._button("Back", y=80, x=150, action=self._prev_step, secondary=True)
        self._button("Next", y=80, action=self._next_step)

    def _render_permissions(self):
        self._label("Permissions", 22, bold=True, y=320, center=True)

        # Microphone
        self._label("Microphone Access", 14, bold=True, y=270, x=60)
        self._label("Required to hear your voice", 12, y=250, x=60)
        self._mic_status = self._label("", 12, y=270, x=370, width=80)
        self._button("Open", y=266, x=420, width=60, action=self._open_mic_prefs, secondary=True)

        # Accessibility
        self._label("Accessibility Access", 14, bold=True, y=200, x=60)
        self._label("Required to type text into other apps", 12, y=180, x=60)
        self._acc_status = self._label("", 12, y=200, x=370, width=80)
        self._button("Open", y=196, x=420, width=60, action=self._open_acc_prefs, secondary=True)

        self._update_permission_display()
        self._start_permission_polling()

        self._button("Back", y=80, x=150, action=self._prev_step, secondary=True)
        self._button("Next", y=80, action=self._next_step)

    def _update_permission_display(self):
        # Microphone - check by trying sounddevice
        try:
            import sounddevice
            sounddevice.query_devices(kind="input")
            self._mic_status.setStringValue_("\u2713 Granted")
            self._mic_status.setTextColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.2, 0.7, 0.2, 1.0)
            )
        except Exception:
            self._mic_status.setStringValue_("\u2717 Needed")
            self._mic_status.setTextColor_(NSColor.redColor())

        # Accessibility
        if AXIsProcessTrusted():
            self._acc_status.setStringValue_("\u2713 Granted")
            self._acc_status.setTextColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.2, 0.7, 0.2, 1.0)
            )
        else:
            self._acc_status.setStringValue_("\u2717 Needed")
            self._acc_status.setTextColor_(NSColor.redColor())

    def _start_permission_polling(self):
        if self._perm_timer:
            self._perm_timer.invalidate()

        def check(timer):
            if not self._window or not self._window.isVisible():
                timer.invalidate()
                return
            if self._current_step != 2:
                return
            self._update_permission_display()

        self._perm_timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            2.0, True, check,
        )

    def _open_mic_prefs(self):
        subprocess.Popen([
            "open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
        ])

    def _open_acc_prefs(self):
        subprocess.Popen([
            "open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
        ])

    def _render_model(self):
        self._label("Speech Model", 22, bold=True, y=320, center=True)
        model_ready = self._app._asr.is_loaded
        if model_ready:
            status = self._label("\u2713 Qwen3-ASR model is ready!", 14, y=250, center=True)
            status.setTextColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.2, 0.7, 0.2, 1.0)
            )
        else:
            self._label(
                "The ASR model (~700MB) will download automatically\n"
                "on first use. This is a one-time setup.",
                13, y=245, center=True, width=400, height=40,
            )
        self._button("Back", y=80, x=150, action=self._prev_step, secondary=True)
        self._button("Next", y=80, action=self._next_step)

    def _render_test(self):
        self._label("Try it out!", 22, bold=True, y=320, center=True)
        self._label(
            "Hold your activation hotkey and say something.\n"
            "Check that the floating bar appears and text is transcribed.",
            13, y=255, center=True, width=420, height=40,
        )
        self._label(
            "You can skip this step and test later.",
            12, y=200, center=True, width=300,
        )
        self._button("Back", y=80, x=100, action=self._prev_step, secondary=True)
        self._button("Skip", y=80, x=200, action=self._next_step, secondary=True)
        self._button("Next", y=80, x=310, action=self._next_step)

    def _render_ready(self):
        check = self._label("\u2713", 48, y=280, center=True)
        check.setTextColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.2, 0.7, 0.2, 1.0)
        )
        self._label("You're all set!", 22, bold=True, y=240, center=True)

        hotkey_display = "Option+Space"
        try:
            hk = self._app._settings.get("activate_hotkey", {})
            from .settings_window import format_hotkey
            hotkey_display = format_hotkey(hk)
        except Exception:
            pass

        self._label(
            f"Hold {hotkey_display} to speak.\n"
            "Text is typed at your cursor on release.\n\n"
            "Access settings from the menu bar icon.",
            13, y=140, center=True, width=400, height=80,
        )
        self._button("Start Using SafeVoice", y=70, action=self._finish)

    # --- Helpers ---

    def _label(self, text, size, bold=False, y=0, x=None, center=False, width=300, height=24):
        if x is None:
            x = (520 - width) / 2 if center else 60
        label = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, width, height))
        label.setStringValue_(text)
        label.setFont_(NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size))
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        if center:
            label.setAlignment_(NSTextAlignmentCenter)
        self._content_view.addSubview_(label)
        return label

    def _button(self, title, y, x=None, width=120, action=None, secondary=False):
        if x is None:
            x = (520 - width) / 2
        btn = NSButton.alloc().initWithFrame_(NSMakeRect(x, y, width, 36))
        btn.setTitle_(title)
        btn.setBezelStyle_(0 if secondary else 1)
        if not secondary:
            btn.setKeyEquivalent_("\r")
        if action:
            target = _WizardButtonTarget.alloc().initWithCallback_(action)
            self._targets.add(target)
            btn.setTarget_(target)
            btn.setAction_(objc.selector(target.invoke_, signature=b"v@:@"))
        self._content_view.addSubview_(btn)
        return btn

    def _next_step(self):
        if self._current_step < len(STEPS) - 1:
            self._current_step += 1
            self._render_step()

    def _prev_step(self):
        if self._current_step > 0:
            self._current_step -= 1
            self._render_step()

    def _finish(self):
        if self._perm_timer:
            self._perm_timer.invalidate()
            self._perm_timer = None
        self._window.close()
        if self._on_complete:
            self._on_complete()
```

### Step 7.3: Wire wizard into app.py

- [ ] In `src/app.py`, add import:
```python
from .setup_wizard import SetupWizard
```

- [ ] At the end of `__init__`, after `self._start_model_load()` (line 119), add:
```python
        # Show setup wizard on first run
        if self._settings.get("first_run", True):
            self._show_setup_wizard()
```

- [ ] Add method to `SafeVoiceApp`:
```python
    def _show_setup_wizard(self):
        def on_complete():
            self._settings.set("first_run", False)
        self._wizard = SetupWizard(self, on_complete=on_complete)
        self._wizard.show()
```

### Step 7.4: Commit

- [ ] ```bash
git add src/setup_wizard.py src/settings_manager.py src/app.py
git commit -m "feat: add 6-step setup wizard for first-time users"
```

---

## Task 8: Integration & Final Polish

**Files:**
- Modify: `src/app.py` (ensure all features work together)

### Step 8.1: Run full test suite

- [ ] ```bash
cd ~/Documents/Vibe\ Code/voice-ime
python -m pytest tests/ -v
```

All tests should pass (test_history, test_vocabulary, test_modes).

### Step 8.2: Manual integration test checklist

- [ ] Build: `python setup.py py2app -A`
- [ ] Launch: `open dist/SafeVoice.app`
- [ ] Verify setup wizard shows on first run (check that `first_run` is `true` in settings or delete settings file)
- [ ] Verify permissions step auto-detects granted permissions
- [ ] Complete wizard, verify `first_run` is set to `false` in `~/.config/safevoice/settings.json`
- [ ] Do a Quick mode transcription — check `/tmp/safevoice.log` for history recording
- [ ] Verify `~/.config/safevoice/history.db` has the entry
- [ ] Check floating bar has state badges ("AI" during processing if Ollama is available)
- [ ] Check floating bar width grows with text but doesn't shrink during recording
- [ ] Open Settings → verify Modes and Vocabulary tabs exist and display data
- [ ] Check menubar has Modes submenu listing all modes
- [ ] Tail the log: `tail -f /tmp/safevoice.log` during all testing

### Step 8.3: Fix any issues found

- [ ] Address any issues found during integration testing
- [ ] Re-run tests after fixes

### Step 8.4: Final commit

- [ ] ```bash
git add src/app.py src/overlay.py src/settings_window.py
git commit -m "fix: integration polish for Type4Me improvements"
```
