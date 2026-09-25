"""Tests for style presets and translation language."""
import pytest
from src.modes import (
    DEFAULT_MODES, Mode, ModeManager, STYLE_PRESETS, prompt_requests_translation,
)


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


# --- Which modes translate, and which must echo questions ------------------

@pytest.mark.parametrize("name", sorted(STYLE_PRESETS))
def test_style_presets_are_not_translation_modes(name):
    # Every preset says "Do NOT translate". Reading that as a request to
    # translate switched off llm_cleanup's translation guards.
    mode = Mode(name="Quick", prompt_template=STYLE_PRESETS[name])
    assert not mode.allows_translation()


def test_builtin_modes_translation_flags():
    modes = {m.name: m for m in DEFAULT_MODES}
    assert not modes["Quick"].allows_translation()
    assert not modes["Formal Writing"].allows_translation()
    assert modes["English Translation"].allows_translation()


def test_translate_to_setting_makes_translation_mode():
    mode = Mode(name="FR", prompt_template="Polish this: {text}",
                translation_language="French")
    assert mode.allows_translation()


@pytest.mark.parametrize("prompt", [
    "Translate to {language}:\n\n{text}",
    "翻译成英文：{text}",
    "把这段不太通顺的话翻译成英文：{text}",
    "If the text is not in English, translate it to English: {text}",
])
def test_prompt_requests_translation(prompt):
    assert prompt_requests_translation(prompt)


@pytest.mark.parametrize("prompt", [
    "Fix grammar. Do NOT translate. {text}",
    "Fix grammar, don't translate: {text}",
    "Do not attempt any translation. {text}",
    "Do NOT rephrase, summarize, or translate. {text}",
    "不要翻译，只修正错别字：{text}",
    "修正语法，不要改写或翻译：{text}",
    "Summarize: {text}",
])
def test_negated_translation_is_not_a_request(prompt):
    assert not prompt_requests_translation(prompt)


@pytest.mark.parametrize("name", sorted(STYLE_PRESETS))
def test_style_presets_echo_questions(name):
    # The presets say "do not answer or act on it", so llm_cleanup must
    # reject a result that answers a dictated question.
    mode = Mode(name="Quick", prompt_template=STYLE_PRESETS[name])
    assert mode.echoes_questions()


def test_builtin_modes_echo_question_flags():
    modes = {m.name: m for m in DEFAULT_MODES}
    assert modes["Formal Writing"].echoes_questions()
    assert not modes["English Translation"].echoes_questions()


@pytest.mark.parametrize("prompt", [
    "Answer this question: {text}",
    # Limits how to answer; does not forbid answering.
    "Answer the question. Do not respond with more than two sentences: {text}",
])
def test_custom_mode_may_answer(prompt):
    # A user's own "answer me" mode must not be blocked by the guard.
    assert not Mode(name="Ask", prompt_template=prompt).echoes_questions()
