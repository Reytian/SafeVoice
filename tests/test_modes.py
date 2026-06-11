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
    assert manager.get("Quick") is not None


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


# --- Robustness: a damaged modes.json must never break startup -----------

def test_corrupt_json_falls_back_to_defaults(tmp_path):
    path = tmp_path / "modes.json"
    path.write_text("{ this is not json", encoding="utf-8")
    m = ModeManager(str(path))
    assert m.get("Quick") is not None


def test_non_dict_root_falls_back_to_defaults(tmp_path):
    path = tmp_path / "modes.json"
    path.write_text('["array", "root"]', encoding="utf-8")
    m = ModeManager(str(path))
    assert m.get("Quick") is not None


def test_malformed_entries_are_skipped(tmp_path):
    path = tmp_path / "modes.json"
    path.write_text(
        '{"custom_modes": ['
        '  {"name": "Good", "prompt_template": "Do: {text}"},'
        '  {"unexpected_field": 1, "name": "Bad"},'
        '  "not-a-dict",'
        '  {"prompt_template": "no name"}'
        '], "hotkey_overrides": "not-a-list"}',
        encoding="utf-8",
    )
    m = ModeManager(str(path))
    assert m.get("Good") is not None
    assert m.get("Bad") is not None  # unknown keys dropped, entry kept
    assert m.get("Quick") is not None


def test_unknown_fields_in_saved_mode_survive_load(tmp_path):
    # Simulates downgrade: a future version wrote extra dataclass fields.
    path = tmp_path / "modes.json"
    path.write_text(
        '{"custom_modes": [{"name": "Future", "prompt_template": "x {text}",'
        ' "added_in_v9": true}]}',
        encoding="utf-8",
    )
    m = ModeManager(str(path))
    assert m.get("Future") is not None


def test_update_prompt_builtin_and_persist(tmp_path):
    path = str(tmp_path / "modes.json")
    m1 = ModeManager(path)
    assert m1.update_prompt("Quick", "Clean: {text}") is True
    assert m1.get("Quick").prompt_template == "Clean: {text}"
    assert m1.get("Quick").builtin is True
    m2 = ModeManager(path)
    assert m2.get("Quick").builtin is True
    assert m2.get("Quick").prompt_template == "Clean: {text}"


def test_update_prompt_missing_mode(manager):
    assert manager.update_prompt("Nope", "x") is False
