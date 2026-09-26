"""Tests for the transcript language in the English filler strip. Where "um"
or "er" is a word (German "um 5 Uhr", Portuguese "um carro"), it is kept on
every path; in other languages, or when the language is unknown,
hesitations still go."""
import threading
import time
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.asr_engine import ASREngine
from src.llm_backend import LLMBackend
from src.llm_cleanup import LLMCleanup
from src.modes import DEFAULT_MODES, Mode
from src.text_postprocess import has_filler_words, strip_filler_words


# --- Languages in which the fillers are words ------------------------------

@pytest.mark.parametrize("text,language", [
    # Regressions: the strip deleted the word
    ("Wir treffen uns um 5 Uhr", "German"),            # at
    ("Ich glaube, er kommt morgen", "German"),         # he
    ("Eu tenho um carro", "Portuguese"),               # a
    ("Er is een probleem", "Dutch"),                   # there
    # Capitalised to open a sentence, or lowercase inside one
    ("Um 8 Uhr beginnt das Meeting", "German"),
    ("Er kommt morgen", "German"),
    ("Ik denk dat er een probleem is", "Dutch"),
    # The other languages the ASR knows that use them as words
    ("Det er godt", "Danish"),                         # is
    ("Jag ser er imorgon", "Swedish"),                 # you
    ("Bunu er geç öğrenecek", "Turkish"),              # sooner or later
    ("Obdivuji jejich technický um", "Czech"),         # skill
    ("Cỏ mọc um tùm quanh nhà", "Vietnamese"),         # lush
    # A language name in another case
    ("Eu tenho um carro", "portuguese"),
])
def test_fillers_that_are_words_are_kept(text, language):
    assert strip_filler_words(text, language=language) == text
    assert not has_filler_words(text, language=language)


def test_only_the_english_filler_rule_is_skipped():
    text = "嗯，Wir treffen uns um 5 Uhr"
    assert strip_filler_words(text, language="German") == "Wir treffen uns um 5 Uhr"
    assert has_filler_words(text, language="German")


# --- Other languages keep the rule -----------------------------------------

@pytest.mark.parametrize("text,language,expected", [
    # Code-switching: the ASR labels the sentence with its main language
    ("嗯 um 我觉得这个方案不错", "Chinese", "我觉得这个方案不错"),
    ("我覺得 um 呢個 plan 都 OK", "Cantonese", "我覺得 呢個 plan 都 OK"),
    ("uh このボタンを押してください", "Japanese", "このボタンを押してください"),
    ("um so we should go", "English", "so we should go"),
    ("um je pense que oui", "French", "je pense que oui"),
    ("uh creo que sí", "Spanish", "creo que sí"),
])
def test_hesitations_are_stripped_in_other_languages(text, language, expected):
    assert strip_filler_words(text, language=language) == expected
    assert has_filler_words(text, language=language)


@pytest.mark.parametrize("language", sorted(
    set(ASREngine.LANGUAGES) - {"German", "Portuguese", "Dutch", "Turkish"}))
def test_every_other_app_language_keeps_the_rule(language):
    assert strip_filler_words("um so we should go", language=language) == "so we should go"


@pytest.mark.parametrize("language", [None, "Auto", "", "unknown"])
def test_unknown_language_keeps_the_rule(language):
    assert strip_filler_words("um so we should go", language=language) == "so we should go"
    assert has_filler_words("um so we should go", language=language)


# --- The word reaches the LLM and every fallback ---------------------------

class _EchoBackend(LLMBackend):
    """Records what the LLM is sent and replies with it, or with reply, or
    raises exc."""

    def __init__(self, reply=None, exc=None, available=True):
        self.sent = []
        self._reply = reply
        self._exc = exc
        self._available = available

    @property
    def name(self):
        return "Echo"

    def is_available(self):
        return self._available

    def chat(self, system_prompt, user_message):
        self.sent.append(user_message)
        if self._exc is not None:
            raise self._exc
        return self._reply if self._reply is not None else user_message


_GERMAN = "Wir treffen uns um 5 Uhr"
_DOWN = {"exc": RuntimeError("backend down")}


def test_llm_is_sent_the_word():
    backend = _EchoBackend()
    LLMCleanup(backend=backend).cleanup(_GERMAN, language="German")
    assert backend.sent == [_GERMAN]


def test_llm_is_sent_a_code_switch_without_its_hesitations():
    backend = _EchoBackend()
    LLMCleanup(backend=backend).cleanup("嗯 um 我觉得这个方案不错", language="Chinese")
    assert backend.sent == ["我觉得这个方案不错"]


@pytest.mark.parametrize("backend,custom_prompt", [
    ({"available": False}, None),
    (_DOWN, None),
    ({"reply": "x" * 500}, None),  # the runaway-length guard rejects it
    (_DOWN, f"Rewrite this formally: {_GERMAN}"),
], ids=["unavailable", "failed", "rejected", "custom-failed"])
def test_fallback_keeps_the_word(backend, custom_prompt):
    llm = LLMCleanup(backend=_EchoBackend(**backend))
    assert llm.cleanup(_GERMAN, custom_prompt=custom_prompt, language="German") == _GERMAN


_PROMPT = f"Rewrite this formally: {_GERMAN}"


def _wait_for_speculative(llm):
    deadline = time.monotonic() + 5
    while llm._speculative_result is None and time.monotonic() < deadline:
        time.sleep(0.01)


def test_speculative_fallback_keeps_the_word():
    llm = LLMCleanup(backend=_EchoBackend(**_DOWN))
    llm.speculative_cleanup(_GERMAN, custom_prompt=_PROMPT, language="German")
    _wait_for_speculative(llm)
    assert llm.get_speculative_result(
        _GERMAN, custom_prompt=_PROMPT, language="German") == _GERMAN


def test_speculative_result_is_only_reused_for_the_same_language():
    """A speculative pass that heard English stripped "um" from its
    fallback. When the full recording turns out to be German, that result
    must not be pasted."""
    llm = LLMCleanup(backend=_EchoBackend(**_DOWN))
    llm.speculative_cleanup(_GERMAN, custom_prompt=_PROMPT, language="English")
    _wait_for_speculative(llm)
    assert llm.get_speculative_result(
        _GERMAN, custom_prompt=_PROMPT, language="German") is None
    assert llm.get_speculative_result(
        _GERMAN, custom_prompt=_PROMPT, language="English") == "Wir treffen uns 5 Uhr"


# --- app.py passes on the language the ASR reported ------------------------

def _fake_app(transcript, language, llm, mode):
    """A stand-in for SafeVoiceApp whose ASR hears transcript in language."""
    from src.app import SafeVoiceApp
    app = MagicMock()
    app._asr.transcribe.return_value = (transcript, language)
    app._vocabulary.apply_snippets.side_effect = lambda text: text
    app._active_mode = mode
    app._llm = llm
    for name in ("_mode_allows_translation", "_mode_echoes_questions"):
        setattr(app, name, types.MethodType(getattr(SafeVoiceApp, name), app))
    return app


def _dictate(transcript, language, llm, mode):
    """Run app.py's transcription worker on a stand-in app whose ASR hears
    transcript in language, and return what it pasted."""
    from src.app import SafeVoiceApp
    app = _fake_app(transcript, language, llm, mode)
    app._audio.stop.return_value = np.full(16000, 0.1, dtype=np.float32)
    pasted = []
    done = threading.Event()

    def inject(text):
        pasted.append(text)
        done.set()
        return True
    app._inject_text.side_effect = inject

    SafeVoiceApp._stop_listening_and_transcribe(app)
    assert done.wait(5), "nothing was pasted"
    return pasted[0]


_FORMAL = next(m for m in DEFAULT_MODES if m.name == "Formal Writing")


@pytest.mark.parametrize("backend,mode", [
    ({"available": False}, Mode(name="Quick")),  # rule strip only
    ({}, Mode(name="Quick")),                    # LLM cleanup
    (_DOWN, _FORMAL),                            # failed custom mode
], ids=["no-llm", "quick", "custom"])
@pytest.mark.parametrize("transcript,language,pasted", [
    (_GERMAN, "German", _GERMAN),
    ("嗯 um 我觉得这个方案不错", "Chinese", "我觉得这个方案不错"),
])
def test_app_pastes_what_the_language_keeps(backend, mode, transcript, language,
                                            pasted):
    llm = LLMCleanup(backend=_EchoBackend(**backend))
    assert _dictate(transcript, language, llm, mode) == pasted


def test_app_reuses_a_speculative_result_for_the_same_language():
    """The speculative cache is keyed on the language too, so the final
    pass has to look it up with the one the ASR reported."""
    backend = _EchoBackend(reply="Wir treffen uns um 5 Uhr.")
    llm = LLMCleanup(backend=backend)
    llm.speculative_cleanup(
        _GERMAN, custom_prompt=_FORMAL.render_prompt(_GERMAN),
        allow_script_change=_FORMAL.allows_translation(),
        echo_questions=_FORMAL.echoes_questions(), language="German")
    _wait_for_speculative(llm)
    assert _dictate(_GERMAN, "German", llm, _FORMAL) == "Wir treffen uns um 5 Uhr."
    assert len(backend.sent) == 1


def test_app_speculative_pass_passes_the_language():
    from src.app import STATE_LISTENING, SafeVoiceApp
    app = _fake_app(_GERMAN, "German", MagicMock(), _FORMAL)
    app._speculative_interval = 0.01
    app._state = STATE_LISTENING
    app._audio_lock = threading.Lock()
    app._audio_chunks = [np.full(16000, 0.1, dtype=np.float32)]
    app._listen_started = time.monotonic()
    app._session_peak_level = 1.0
    app._SILENCE_PEAK_THRESHOLD = SafeVoiceApp._SILENCE_PEAK_THRESHOLD
    called = threading.Event()
    app._llm.speculative_cleanup.side_effect = lambda *a, **kw: called.set()

    SafeVoiceApp._start_speculative_timer(app)
    try:
        assert called.wait(5), "no speculative cleanup ran"
    finally:
        app._speculative_timer_stop.set()
    assert app._llm.speculative_cleanup.call_args.kwargs["language"] == "German"
