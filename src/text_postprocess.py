"""Deterministic post-processing for ASR transcription.

Runs before (and independently of) any LLM cleanup step. Two purposes:

1. **Always-on filler removal**: even when the user is in Quick mode with no
   LLM available, or when text is too short to trigger the LLM gate, the
   output should still be free of obvious "嗯 / 啊 / um / uh" noise.
2. **Token reduction for the LLM step**: stripping trivial fillers before
   sending to the LLM cuts a few percent of input tokens and makes the
   model focus on the harder structural fixes (self-corrections, grammar).

Self-corrections like "五点，啊不对，六点" are intentionally NOT handled here.
Reliable detection requires semantic understanding (which "X" should the
"Y" replace?) and is delegated to the LLM SYSTEM_PROMPT. The rules here
only do safe, unambiguous filler stripping.

Conservative scope:
- Single-character Chinese hesitation sounds (嗯/啊/呃/哦/哎/唉) — clearly
  droppable in any context.
- Common English hesitations (um/uh/er/ah/erm/uhh) — word-boundary matched
  so "umbrella" / "ahead" / "around" stay intact.
- Stuttering on a safelist of words: "I I want" -> "I want", "我我想" -> "我想".
  Words people repeat on purpose ("zero zero", "very very", "no no no")
  are kept.
- Ambiguous discourse markers (那个 / 这个 / 就是 / 然后 / like / you know)
  are LEFT ALONE here and handled by the LLM with full context. Stripping
  them with regex breaks meaningful sentences ("这个产品" must keep 这个).
- So are English fillers that may be an acronym, name or unit instead
  ("the ER", "DD MM YYYY", "5 mm", "uh-huh"); see _EN_FILLER_RE.
- So are all English fillers when the transcript's language has them as
  words (German "um 5 Uhr", Portuguese "um carro"); see
  _EN_FILLERS_ARE_WORDS_IN.
"""

import re
from typing import Iterable, Optional

# Chinese single-character hesitations. These are nearly always droppable
# regardless of context — they don't form meaningful words on their own.
# 嗯/啊/呃/哦/哎/唉 plus their stretched variants (啊啊啊, 嗯嗯).
_CN_HESITATION_CHARS = "嗯啊呃哦哎唉噢嗨"

# Match a run of hesitation chars surrounded by optional Chinese punctuation
# or whitespace. The trailing comma/space is consumed too so we don't leave
# orphan ", ," patterns.
_CN_HESITATION_RE = re.compile(
    rf"(^|[\s，、。！？,!?]+)[{_CN_HESITATION_CHARS}]+(?=[\s，、。！？,!?]|$)"
)

# Stand-alone hesitation char at start of an utterance, without leading
# punctuation: "嗯今天天气不错" -> "今天天气不错". Only strip a single
# leading hesitation char to avoid eating real content.
_CN_HESITATION_LEADING_RE = re.compile(rf"^[{_CN_HESITATION_CHARS}]+(?=[^\s，、。！？,!?])")

# English hesitations. Word-boundary matched so substrings inside real
# words ("umbrella", "ahead") are preserved.
# CRITICAL: several of these are also acronyms, names and units. This strip
# runs before the LLM and is what gets pasted without one, so a word it
# drops is lost on every path: "the ER last night" -> "the last night",
# "DD MM YYYY" -> "DD YYYY", "5 mm long" -> "5 long". _strip_en_filler
# only removes a match that
# - is lowercase. Inside a sentence ASR writes a hesitation in lowercase,
#   while acronyms, names and unit symbols keep their capitals ("ER", "an
#   HMM", "call Er", "5 Ah"). A sentence may open with a capitalised
#   hesitation ("Um, so ..."), so there a capitalised filler still goes.
#   A name like "Er" that opens a sentence goes with it, which is rare.
# - is not "mm" or "ah" right after a number, or "mm" after a number word.
#   Case can't tell these units from the hesitations, but "5 mm" and "five
#   mm" are millimetres and "5 ah" amp-hours. After a comma it is a
#   hesitation again: "5, mm, 6".
# A filler joined to a neighbouring word by - / . or : is part of that
# word ("uh-huh", "mm-hmm", "mm/dd/yyyy", "hh:mm", "5-mm"), so the pattern
# doesn't match it. Still lost: "mm" as a unit with no number ("in mm").
_EN_FILLER_WORDS = ("um", "umm", "ummm", "uh", "uhh", "uhm", "er", "erm", "ah", "ahh", "hmm", "mm")
_EN_FILLER_RE = re.compile(
    r"(?<!\w[-/.:])\b(" + "|".join(_EN_FILLER_WORDS) + r")\b(?![-/.:]\w)[\s,]*",
    flags=re.IGNORECASE,
)
# "mm" and "ah" right after a digit are units ("5 mm", "5 ah"). The ASR
# writes small numbers as words ("five mm", "twenty five mm", "two point
# five mm", "half a mm"), so "mm" is a unit after a number word too. "ah"
# is not: after a number word it is nearly always a hesitation ("one ah
# two"), and the ASR capitalises amp-hours ("two Ah"), which the case rule
# keeps.
_EN_UNIT_FILLERS = ("mm", "ah")
_AFTER_NUMBER_RE = re.compile(r"\d\s*$")
_EN_NUMBER_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "half", "quarter",
)
_AFTER_NUMBER_WORD_RE = re.compile(
    r"\b(?:" + "|".join(_EN_NUMBER_WORDS) + r")(?:\s+an?)?\s+$",
    flags=re.IGNORECASE,
)
# What precedes a filler that opens a sentence: nothing, the end of the last
# sentence, a colon, semicolon or dash, or an opening quote or bracket (the
# ASR writes reported speech as 'She asked, "Ah, what time is it?"' and
# "my question is: Um, when do we start?"), then spaces and commas. A comma
# can be what is left of a Chinese hesitation removed before this step
# ("嗯，Um, so" -> "，Um, so"). Hesitations in front of this one are looked
# past as well ("Um, Um, so"), since they go too.
_SENTENCE_START_RE = re.compile(
    r"(?:^|[.!?…。！？:：;；\n\-–—]|[\"“「『‘'(（\[【])[\s，、,]*$"
)
_PRECEDING_FILLERS_RE = re.compile(
    r"(?:\b(?:" + "|".join(_EN_FILLER_WORDS) + r")\b[\s,]*)+$",
    flags=re.IGNORECASE,
)


def _strip_en_filler(m: re.Match) -> str:
    """Replacement for an _EN_FILLER_RE match: nothing if the filler is a
    hesitation, else the match as is. The rules are above _EN_FILLER_RE.
    """
    filler = m.group(1)
    before = m.string[:m.start()]
    if filler.islower():
        if filler in _EN_UNIT_FILLERS and _AFTER_NUMBER_RE.search(before):
            return m.group(0)
        if filler == "mm" and _AFTER_NUMBER_WORD_RE.search(before):
            return m.group(0)
        return ""
    if filler.istitle() and _SENTENCE_START_RE.search(
            _PRECEDING_FILLERS_RE.sub("", before)):
        return ""
    return m.group(0)


# CRITICAL: languages in which some of _EN_FILLER_WORDS are everyday words.
# They are cased like a hesitation, lowercase inside a sentence and
# capitalised at its start, so the case rules above can't save them:
# German "wir treffen uns um 5 Uhr" (at) and "er kommt" (he), Dutch "er
# is" (there is), Portuguese "tenho um carro" (a), Danish "det er godt"
# (is), Swedish "jag ser er" (you), Turkish "er geç" (sooner or later). A
# transcript in one of these skips the English filler rule.
# Any other language keeps it, and so does an unknown one. In Chinese or
# Russian text a Latin "um" is an English hesitation ("嗯 um 我觉得"), and
# in the ASR's other Latin-script languages they are at most interjections
# or rare words.
_EN_FILLERS_ARE_WORDS_IN = frozenset((
    "danish", "dutch", "german", "portuguese", "swedish", "turkish",
))


def _strips_en_fillers(language: Optional[str]) -> bool:
    """Whether the English filler rule runs on a transcript in *language*,
    a name as ASREngine reports it ("German"), or None if unknown."""
    return (language or "").casefold() not in _EN_FILLERS_ARE_WORDS_IN


# Stuttering: same word repeated 2+ times with whitespace.
# English: "I I want" -> "I want"; "the the cat" -> "the cat".
# CRITICAL: only the words in _EN_STUTTER_WORDS are collapsed. People repeat
# plenty of words on purpose, and collapsing those deletes what they said:
# dictated numbers ("one three eight zero zero ..." lost digits), years
# ("twenty twenty"), spelled letters ("J O H N N Y"), emphasis ("very very",
# "no no no"), names ("Walla Walla", "Fei Fei") and grammatical doubles
# ("had had", "I know that that is"). That list has no end, but real
# stutters cluster on a few function words, so like _CN_STUTTER_RE this is
# a safelist of words whose doubling is virtually always a stutter:
# articles, conjunctions, prepositions and the pronouns that are only ever
# subjects. "it" and "you" are left out because they also end clauses, and
# ASR often drops the comma before the next one ("I love it it's great").
# Any other repeat is left for the LLM, which has the context to judge it
# (SYSTEM_PROMPT rule E3).
# ASCII letters only, so repeated digits ("1 1 2") and spaced CJK
# reduplication ("好 好 学习") are never collapsed; CJK stutters are handled
# separately and conservatively by _CN_STUTTER_RE.
_EN_STUTTER_WORDS = (
    "i", "we", "he", "she", "they",
    "a", "an", "the",
    "and", "but", "or", "if",
    "to", "of", "for", "with", "from",
)
_EN_STUTTER_RE = re.compile(
    r"\b(" + "|".join(_EN_STUTTER_WORDS) + r")(?:\s+\1\b)+",
    flags=re.IGNORECASE,
)


def _collapse_en_stutter(m: re.Match) -> str:
    """Replacement for an _EN_STUTTER_RE match: the word once.

    A capitalised repeat is a spelled letter, acronym or name ("I got A A
    B", "An An"), not a stutter, so the match is kept as is. The pronoun I
    is always capitalised and is exempt, which means a spelled "I I" (as in
    "F U J I I") still collapses; that is rare enough to accept.
    """
    first, *repeats = m.group(0).split()
    if first.lower() != "i" and not all(r.islower() for r in repeats):
        return m.group(0)
    return first

# Chinese single-char stuttering. CRITICAL: Cannot blanket-collapse
# any duplicated CJK char — that would corrupt legitimate compounds like
# 今天天气 (今天 + 天气), 天天向上, 人人都知道, 个个都好, where the middle
# duplicate spans a word boundary. We restrict to a safelist of pronouns
# and demonstratives that, when doubled, are virtually always stutters
# (我我, 你你, 他他, 她她, 它它, 这这, 那那) — these never appear as
# legitimate adjacent doubles in modern Chinese.
_CN_STUTTER_RE = re.compile(r"([我你他她它这那])\1+")

# Collapse runs of duplicate punctuation/whitespace left behind after
# filler stripping: "， ， ，" -> "，", "  " -> " ".
_CN_PUNCT_DUP_RE = re.compile(r"([，、。！？])\s*[，、]+")
_WS_RE = re.compile(r"[ \t]{2,}")
_LEADING_PUNCT_RE = re.compile(r"^[\s，、,]+")


def strip_filler_words(text: str, language: Optional[str] = None) -> str:
    """Remove obvious filler words and stutters from ASR text.

    Safe to call on any string in any language; rules are conservative
    and language-detect themselves. Returns the original text unchanged
    if no fillers are found. Never raises.

    The English filler rule can't tell from the text alone that "um" is
    German for "at", so pass the transcript's *language* as ASREngine
    reports it ("German"). The rule is skipped for a language in
    _EN_FILLERS_ARE_WORDS_IN and runs for any other, or for None.
    """
    if not text:
        return text
    if not text.strip():
        return ""

    out = text

    # 1. Chinese hesitation runs surrounded by punctuation/space.
    out = _CN_HESITATION_RE.sub(lambda m: m.group(1), out)

    # 2. Leading Chinese hesitation at utterance start.
    out = _CN_HESITATION_LEADING_RE.sub("", out)

    # 3. English filler words (um/uh/er/ah/hmm/mm), unless the transcript's
    #    language uses them as words.
    if _strips_en_fillers(language):
        out = _EN_FILLER_RE.sub(_strip_en_filler, out)

    # 4. Chinese single-char stutter (我我想 -> 我想).
    out = _CN_STUTTER_RE.sub(r"\1", out)

    # 5. English word stutter (I I -> I).
    out = _EN_STUTTER_RE.sub(_collapse_en_stutter, out)

    # 6. Tidy up duplicate punctuation and whitespace left behind.
    out = _CN_PUNCT_DUP_RE.sub(r"\1", out)
    out = _WS_RE.sub(" ", out)
    out = _LEADING_PUNCT_RE.sub("", out)

    return out.strip()


def has_filler_words(text: str, language: Optional[str] = None) -> bool:
    """Cheap predicate: does this text contain anything we'd strip?

    Useful for short-circuiting the postprocess call when the input is
    already clean (saves a regex pass on hot paths). Pass the same
    *language* as to strip_filler_words.
    """
    if not text:
        return False
    if _CN_HESITATION_RE.search(text):
        return True
    if _CN_HESITATION_LEADING_RE.search(text):
        return True
    if _strips_en_fillers(language):
        for m in _EN_FILLER_RE.finditer(text):
            if _strip_en_filler(m) != m.group(0):
                return True
    if _CN_STUTTER_RE.search(text):
        return True
    for m in _EN_STUTTER_RE.finditer(text):
        if _collapse_en_stutter(m) != m.group(0):
            return True
    return False
