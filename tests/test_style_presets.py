"""Tests for style presets and translation language."""
import pytest
from src.modes import Mode, ModeManager, STYLE_PRESETS


def test_style_presets_exist():
    assert "minimal" in STYLE_PRESETS
    assert "professional" in STYLE_PRESETS
    assert "casual" in STYLE_PRESETS
    assert "verbatim" in STYLE_PRESETS


def test_preset_contains_text_placeholder():
    for name, template in STYLE_PRESETS.items():
        assert "{text}" in template, f"Preset '{name}' missing {{text}} placeholder"


def test_mode_with_translation_language():
    mode = Mode(
        name="French Translation",
        prompt_template="Translate to {language}:\n\n{text}",
        translation_language="French",
    )
    result = mode.render_prompt("hello")
    assert result == "Translate to French:\n\nhello"


def test_mode_without_translation_language():
    mode = Mode(name="Quick", prompt_template=None)
    result = mode.render_prompt("hello")
    assert result == "hello"


def test_translation_language_persistence(tmp_path):
    path = str(tmp_path / "modes.json")
    m1 = ModeManager(path)
    m1.add(Mode(
        name="FR",
        prompt_template="Translate to {language}:\n\n{text}",
        translation_language="French",
    ))
    m2 = ModeManager(path)
    mode = m2.get("FR")
    assert mode.translation_language == "French"
