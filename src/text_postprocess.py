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
- Stuttering: "I I want" -> "I want", "我我想" -> "我想"
- Ambiguous discourse markers (那个 / 这个 / 就是 / 然后 / like / you know)
  are LEFT ALONE here and handled by the LLM with full context. Stripping
  them with regex breaks meaningful sentences ("这个产品" must keep 这个).
"""

import re
from typing import Iterable

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
_EN_FILLER_WORDS = ("um", "umm", "ummm", "uh", "uhh", "uhm", "er", "erm", "ah", "ahh", "hmm", "mm")
_EN_FILLER_RE = re.compile(
    r"\b(?:" + "|".join(_EN_FILLER_WORDS) + r")\b[\s,]*",
    flags=re.IGNORECASE,
)

# Stuttering: same word repeated 2+ times with whitespace.
# English: "I I want" -> "I want"; "the the cat" -> "the cat".
# Keeps legitimate doubles like "had had" intact only when they're capitalized
# differently or in a non-stutter pattern, which is rare enough to ignore.
_EN_STUTTER_RE = re.compile(r"\b(\w+)(?:\s+\1\b)+", flags=re.IGNORECASE)

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


def strip_filler_words(text: str) -> str:
    """Remove obvious filler words and stutters from ASR text.

    Safe to call on any string in any language; rules are conservative
    and language-detect themselves. Returns the original text unchanged
    if no fillers are found. Never raises.
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

    # 3. English filler words (um/uh/er/ah/hmm/mm).
    out = _EN_FILLER_RE.sub("", out)

    # 4. Chinese single-char stutter (我我想 -> 我想).
    out = _CN_STUTTER_RE.sub(r"\1", out)

    # 5. English word stutter (I I -> I).
    out = _EN_STUTTER_RE.sub(r"\1", out)

    # 6. Tidy up duplicate punctuation and whitespace left behind.
    out = _CN_PUNCT_DUP_RE.sub(r"\1", out)
    out = _WS_RE.sub(" ", out)
    out = _LEADING_PUNCT_RE.sub("", out)

    return out.strip()


def has_filler_words(text: str) -> bool:
    """Cheap predicate: does this text contain anything we'd strip?

    Useful for short-circuiting the postprocess call when the input is
    already clean (saves a regex pass on hot paths).
    """
    if not text:
        return False
    if _CN_HESITATION_RE.search(text):
        return True
    if _CN_HESITATION_LEADING_RE.search(text):
        return True
    if _EN_FILLER_RE.search(text):
        return True
    if _CN_STUTTER_RE.search(text):
        return True
    if _EN_STUTTER_RE.search(text):
        return True
    return False
