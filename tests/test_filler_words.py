"""Tests for the English filler strip: hesitations go, while the acronyms,
names and units that look like them stay."""
import pytest
from src.llm_backend import LLMBackend
from src.llm_cleanup import LLMCleanup
from src.text_postprocess import has_filler_words, strip_filler_words


# --- Fillers that are words, not hesitations ------------------------------

@pytest.mark.parametrize("text", [
    # Regressions: the filler rule deleted the unit, acronym or format.
    "the screw is 5 mm long",
    "I went to the ER last night",
    "use the DD MM YYYY format",
    # Acronyms keep their capitals
    "I studied at UM",
    "we trained an HMM on the data",
    "our ERM framework needs review",
    # A name inside a sentence is capitalised
    "call Er tomorrow",
    # Units right after a number, in either case
    "a 5 Ah battery",
    "a 5 ah battery",
    "cut them to 20 mm and 30 mm",
    "the gap is 12 mm.",
])
def test_acronyms_names_and_units_are_kept(text):
    assert strip_filler_words(text) == text
    assert not has_filler_words(text)


@pytest.mark.parametrize("text", [
    # Interjections that answer yes or no
    "uh-huh",
    "mm-hmm, that works",
    "uh-uh, not that one",
    "Uh-oh, the build broke",
    # Formats and compounds
    "use the mm/dd/yyyy format",
    "the date is dd.mm.yyyy",
    "show the time as hh:mm",
    "a 5-mm screw",
])
def test_fillers_joined_to_a_word_are_kept(text):
    assert strip_filler_words(text) == text
    assert not has_filler_words(text)


# --- Real hesitations are still stripped ----------------------------------

@pytest.mark.parametrize("text,expected", [
    ("um so we should go", "so we should go"),
    ("uh I think it works", "I think it works"),
    ("er, the thing is", "the thing is"),
    ("we could ah try again", "we could try again"),
    ("hmm let me check", "let me check"),
    # The hesitation goes and the lookalike word stays
    ("um the screw is 5 mm long", "the screw is 5 mm long"),
    ("uh I went to the ER last night", "I went to the ER last night"),
    # A sentence may open with a capitalised hesitation
    ("Um, so I think we should go.", "so I think we should go."),
    ("Hmm, let me check.", "let me check."),
    ("That works. Um, I'll check.", "That works. I'll check."),
    ("嗯，Um, 我觉得可以", "我觉得可以"),
    # After a number only "mm" and "ah" are units, and only right after it
    ("call me at 5 um tomorrow", "call me at 5 tomorrow"),
    ("it is 5, um, 6 mm long", "it is 5, 6 mm long"),
    ("about 5, mm, 6 mm", "about 5, 6 mm"),
])
def test_hesitations_are_stripped(text, expected):
    assert strip_filler_words(text) == expected
    assert has_filler_words(text)


# --- The kept word reaches the LLM and the fallback -----------------------

class _EchoBackend(LLMBackend):
    """Records what the LLM is sent and echoes it back, or raises exc."""

    def __init__(self, exc=None):
        self.sent = []
        self._exc = exc

    @property
    def name(self):
        return "Echo"

    def is_available(self):
        return True

    def chat(self, system_prompt, user_message):
        self.sent.append(user_message)
        if self._exc is not None:
            raise self._exc
        return user_message


_DICTATED = [
    ("um the screw is 5 mm long", "the screw is 5 mm long"),
    ("uh I went to the ER last night", "I went to the ER last night"),
    ("er use the DD MM YYYY format", "use the DD MM YYYY format"),
]


@pytest.mark.parametrize("raw,pre_cleaned", _DICTATED)
def test_llm_is_sent_the_kept_word(raw, pre_cleaned):
    backend = _EchoBackend()
    LLMCleanup(backend=backend).cleanup(raw)
    assert backend.sent == [pre_cleaned]


@pytest.mark.parametrize("raw,pre_cleaned", _DICTATED)
def test_failed_cleanup_pastes_the_kept_word(raw, pre_cleaned):
    llm = LLMCleanup(backend=_EchoBackend(exc=RuntimeError("backend down")))
    assert llm.cleanup(raw) == pre_cleaned
