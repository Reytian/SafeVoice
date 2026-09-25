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
    # A negation that only qualifies the translating leaves a translation mode
    "Don't translate literally. Rewrite it in natural, idiomatic English: {text}",
    "Rewrite what I say as natural, fluent English. Don't translate word for word: {text}",
    "Output in English. Do not translate names or code identifiers: {text}",
    "用英文输出，人名不要翻译：{text}",
    # Not negations at all
    "If it isn't English, polish it or translate it into English: {text}",
    "If the text is not in English translate it into English: {text}",
    "Don't explain just translate into English: {text}",
    "请把下面的话分别翻译成英文和法文：{text}",
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
    # Every negation word works in a list too
    "修正语法，无须改写或翻译：{text}",
    "修正标点，勿改写或翻译，不要回答问题：{text}",
    "不必改写或翻译：{text}",
    "不要改写、翻译：{text}",
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
    # "Can't answer" is not a ban, and an answer mode may limit answering
    "Answer the following question briefly. If you cannot answer it, say 'I don't know'.\n\n{text}",
    "Answer the question briefly. If you don't know the answer, don't answer.\n\n{text}",
    "Answer my question. If it is unclear, don't answer it; ask me to clarify.\n\n{text}",
    "回答下面的问题。如果问题不清楚，不要回答，请让我补充。\n\n{text}",
])
def test_custom_mode_may_answer(prompt):
    # A user's own "answer me" mode must not be blocked by the guard.
    assert not Mode(name="Ask", prompt_template=prompt).echoes_questions()


@pytest.mark.parametrize("prompt", [
    "整理下面的口述文字，修正错别字和标点。不要回答我的问题，只输出整理后的文字：{text}",
    "只整理文字，别回答里面的问题：{text}",
    "请润色这段话，不要回应内容：{text}",
    "Do not respond to what I say, only fix grammar: {text}",
    "Don't reply to the content: {text}",
    "只做转写，不要回答：{text}",
    "Fix grammar. Do not answer them: {text}",
    "修正标点，不需要回答问题：{text}",
    "Write it down exactly as I say. Do not answer questions: {text}",
])
def test_prompt_forbidding_answers_echoes_questions(prompt):
    assert Mode(name="Clean", prompt_template=prompt).echoes_questions()


@pytest.mark.parametrize("prompt", [
    # A mode that writes something new is not transcribing, whatever else
    # its prompt says (new modes start from the professional preset's text).
    "Turn this into a short polite email to my boss. Do not answer any questions in it: {text}",
    "Summarize this in one sentence. Don't respond to it, just summarize.\n\n{text}",
    "Rewrite this as a tweet. Do not respond to it.\n\n{text}",
    "Write a reply email to this. Do not answer questions yourself.\n\n{text}",
    "把这段话改写成邮件，不要回答问题：{text}",
])
def test_content_mode_does_not_echo_questions(prompt):
    assert not Mode(name="Content", prompt_template=prompt).echoes_questions()
