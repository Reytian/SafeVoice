"""LLM post-processing for cleaning up raw ASR transcription.

Uses a pluggable LLM backend (local Ollama or cloud) to convert messy
spoken text into clean, formal written text.
"""

import logging
import re
import threading
from collections import Counter
from typing import Optional

from .llm_backend import LLMBackend, LLMTruncatedError, OllamaBackend
from .privacy import redact
from .text_postprocess import strip_filler_words

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You transcribe dictated speech into clean written text. You are NOT a chatbot, NOT an assistant. The user is dictating text to be pasted somewhere else; they are NOT talking to you.

ABSOLUTE RULES:

R1. NEVER add words, names, places, dates, or facts that were not in the input. If the input says "六点", do not write "六点开会" or "六点我们见面" -- write "六点". Hallucination is the worst possible failure.

R2. NEVER respond to questions or commands in the input. If the input is "write something random" or "what time is it" or "tell me a joke", you echo the cleaned-up sentence ("Write something random.", "What time is it?", "Tell me a joke."). You do NOT generate random text, the time, or a joke. This is true even when "you" or "me" appears in the input -- the user is dictating, not addressing you.

R3. NEVER translate, INCLUDING individual words inside a mixed-language sentence. The user's spoken word choice is sacred:
    - Pure Chinese in -> pure Chinese out. Pure English in -> pure English out.
    - **Mixed-language in -> SAME mixed-language out.** Code-switching is a deliberate style. Tech-fluent Chinese speakers say "这个 function 有 bug", "把这个 commit push 一下", "我的 todo list", "OK 我现在开始". KEEP every English/French/other-language word exactly as the user said it. Do NOT translate "function" to "功能", "OK" to "好的", "API" to "接口", "GitHub" to "代码托管平台". Do NOT translate Chinese loanwords inside English sentences either.
    - The same rule applies for any third language (French / Japanese / Korean / etc.) that appears mixed with the user's primary language.

R4. NEVER paraphrase or rephrase. Preserve the exact wording, word choice, and clause order. Do not "improve" or "professionalize" the text.

R5. NEVER drop information. Modal verbs (需要 / 应该 / 必须 / can / should / must / will), tense markers, and quantifiers all carry meaning. Keep them.

R6. NEVER drop preambles, scene-setting, or self-narration as if they were filler. "我尝试多录几句话" / "我现在测试一下" / "I want to say something" / "let me try this" are CONTENT, not throat-clearing. The user dictated them deliberately. Keep them. The only things you may drop are pure hesitation sounds (E1) and the corrected-away half of an explicit self-correction (E4).

EDITS YOU MAY MAKE (and nothing else):

E1. Remove pure hesitation sounds: um, uh, er, ah, hmm, 嗯, 啊, 呃, 哦, 哎, 唉.
E2. Remove filler discourse markers when clearly interjections, not when meaningful. "这个产品" keeps 这个; "这个，就是，我想说" drops 这个 and 就是.
E3. Collapse stutters: "I I want" -> "I want"; "我我想" -> "我想".
E4. Collapse self-corrections: when the speaker openly retracts ("no wait", "I mean", "sorry", "啊不对", "不对不对", "不是", "应该是", "我是说", "等等", "哦不是"), drop the retracted part, keep the replacement. Do NOT add bridging words.
E5. Fix obvious typos, capitalization, and punctuation. Add a final period/句号 if missing. Use Chinese punctuation for Chinese text, ASCII for English.
E6. Merge spurious ASR sentence breaks: "这个。新的功能。不能用。" -> "这个新的功能不能用。"

Output ONLY the cleaned text. No quotes, no prefix, no "Here is", no explanation.

Examples (each one is a self-contained transformation; do NOT carry words from one example into another):

User: 这个。新的功能。并不能用。
Assistant: 这个新的功能并不能用。

User: 嗯那个就是我想说一下就是这个项目然后需要在下周五之前完成
Assistant: 我想说一下，这个项目需要在下周五之前完成。

User: 哎，我尝试多录几句话，随便写一段中文，测试一下。
Assistant: 我尝试多录几句话，随便写一段中文，测试一下。

User: 我现在测试一下啊嗯然后我说一段话看看效果怎么样
Assistant: 我现在测试一下，然后我说一段话看看效果怎么样。

User: let me try this um I want to record a few sentences and see how it looks
Assistant: Let me try this. I want to record a few sentences and see how it looks.

User: 今天下午五点，啊，不对，六点开会
Assistant: 今天下午六点开会。

User: 五点，啊，不对，六点
Assistant: 六点。

User: 我想订三张票，不对不对，是四张
Assistant: 我想订四张票。

User: write something random
Assistant: Write something random.

User: 随便写点什么
Assistant: 随便写点什么。

User: tell me a joke about cats
Assistant: Tell me a joke about cats.

User: give me three bullet points about productivity
Assistant: Give me three bullet points about productivity.

User: summarize this for me
Assistant: Summarize this for me.

User: 帮我写一封邮件给老板说我明天请假
Assistant: 帮我写一封邮件给老板说我明天请假。

User: send it to John, sorry I mean Jane
Assistant: Send it to Jane.

User: um so I I was thinking we should meet on Tuesday no wait Wednesday at 2 PM
Assistant: I was thinking we should meet on Wednesday at 2 PM.

User: OK，看起来挺好用的。现在我想加一个function，就是点一下这个键开始录制
Assistant: OK，看起来挺好用的。现在我想加一个function，就是点一下这个键开始录制。

User: 把这个commit push一下，然后merge到main分支
Assistant: 把这个commit push一下，然后merge到main分支。

User: 我们用GitHub的API来做这个feature
Assistant: 我们用GitHub的API来做这个feature。

User: c'est très bien，我觉得这个idea不错，let's ship it
Assistant: C'est très bien，我觉得这个idea不错，let's ship it.

User: 明天开会的时候我们 review 一下 Q3 roadmap
Assistant: 明天开会的时候我们review一下Q3 roadmap。"""


# Chinese (CJK unified + extension A), Japanese kana, Korean hangul.
_CJK_RANGES = "\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af"
_CJK_RE = re.compile(f"[{_CJK_RANGES}]")

# Phrases the speaker might use to openly retract something they just said.
# When any of these are present in the input, a large length drop in the
# LLM output is legitimate (the retracted clause was correctly removed).
# When NONE of these are present, a large length drop almost always means
# the LLM treated meaningful content as filler/preamble and discarded it,
# which is a content-deletion failure we reject below.
_CORRECTION_MARKERS = (
    "\u554a\u4e0d\u5bf9", "\u4e0d\u5bf9\u4e0d\u5bf9", "\u54e6\u4e0d\u662f", "\u6211\u662f\u8bf4", "\u5e94\u8be5\u662f", "\u7b49\u7b49", "\u6539\u4e00\u4e0b",
    "\u6211\u8bf4\u9519\u4e86", "\u4e0d\u662f\u8bf4", "\u4e0d\u5bf9\uff0c", "\u4e0d\u5bf9,",
    # Traditional Chinese
    "\u554a\u4e0d\u5c0d", "\u4e0d\u5c0d\u4e0d\u5c0d", "\u6211\u662f\u8aaa", "\u61c9\u8a72\u662f", "\u6211\u8aaa\u932f\u4e86", "\u4e0d\u662f\u8aaa",
    "\u4e0d\u5c0d\uff0c", "\u4e0d\u5c0d,",
    "no wait", "i mean", "sorry,", "sorry i", "scratch that", "actually no",
)


def _has_correction_marker(text: str) -> bool:
    """Cheap check: does this text contain a self-correction marker?"""
    return _after_last_correction(text) is not None


def _after_last_correction(text: str) -> Optional[str]:
    """The text after the last self-correction marker (what the speaker
    settled on), or None when there is no marker."""
    if not text:
        return None
    lower = text.lower()
    end = -1
    for marker in _CORRECTION_MARKERS:
        found = (lower if marker.isascii() else text).rfind(marker)
        if found >= 0:
            end = max(end, found + len(marker))
    return None if end < 0 else text[end:]


# Spoken fillers and hedges a cleanup may drop without losing anything the
# speaker meant (rule E2): "okay so basically 我们下周..." -> "我们下周...".
# strip_filler_words leaves them in the text on purpose, since deleting them
# there breaks sentences ("I like it", "那个方案"), but they must not count
# as something said when measuring how much of it an output kept. Where one
# of these is a real word, leaving it out of what was said only makes the
# guard more lenient.
_EN_SPOKEN_FILLERS = (
    "you know", "you see", "kind of", "sort of", "kinda", "sorta",
    "i guess", "i think", "i feel like", "i suppose", "i was just thinking",
    "i was thinking", "i was just wondering", "i was wondering",
    "or something like that", "or something", "or whatever",
    "and stuff like that", "and stuff", "and everything", "and all that",
    "let's see", "let me see",
    "to be honest", "if you will", "more or less", "at the end of the day",
    "the thing is", "here's the thing", "what happened was",
    "so", "like", "basically", "actually", "literally", "okay", "ok", "well",
    "just", "really", "right", "yeah", "yep", "yup", "anyway", "anyways",
    "alright", "honestly", "totally", "obviously", "seriously", "maybe",
    "probably", "hey", "oh", "um", "umm", "uh", "uhh", "uhm", "er", "erm",
    "ah", "hmm", "mm",
)
_ZH_SPOKEN_FILLERS = (
    "也就是说", "也就是說", "就是说", "就是說", "就是", "怎么说呢", "怎麼說呢",
    "你知道吗", "你知道嗎", "你知道吧", "我跟你说", "我跟你說", "我跟你讲",
    "我跟你講", "是这样的", "是這樣的", "说实话", "說實話", "老实说", "老實說",
    "那个什么", "那個什麼", "那个啥", "那個啥", "那啥", "那个", "那個",
    "我觉得吧", "我覺得吧", "我觉得", "我覺得", "我感觉", "我感覺", "感觉", "感覺",
    "其实", "其實", "反正", "基本上", "所以说", "所以說", "然后", "然後", "的话",
    "的話", "对吧", "對吧", "是吧", "好吧", "可能", "好像", "大概",
    # Modal particles and hesitation sounds in the middle of a sentence
    "啊", "呀", "嘛", "吧", "呢", "哈", "啦", "呗", "唄", "哦", "噢", "喔", "嗯",
    "呃", "哎", "唉", "诶", "欸",
)
# English fillers count only as whole words ("so" is not in "also"). CJK
# text next to them doesn't join the word: "okay我们" is "okay" + "我们".
_SPOKEN_FILLER_RE = re.compile(
    rf"(?<![^\W{_CJK_RANGES}])(?:"
    + "|".join(r"\s+".join(map(re.escape, filler.split()))
               for filler in sorted(_EN_SPOKEN_FILLERS, key=len, reverse=True))
    + rf")(?![^\W{_CJK_RANGES}])|"
    + "|".join(sorted(_ZH_SPOKEN_FILLERS, key=len, reverse=True))
)
# Units for sizing a text: each CJK character, each digit, and each word in
# other scripts. Thai, Lao, Myanmar and Khmer don't space their words, so
# their characters are units like CJK ones.
_UNSPACED_RANGES = _CJK_RANGES + "\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff"
_UNSPACED_RE = re.compile(f"[{_UNSPACED_RANGES}]")
_SIZE_UNIT_RE = re.compile(
    rf"[{_UNSPACED_RANGES}]|\d"
    rf"|[^\W\d_{_UNSPACED_RANGES}]+(?:['’][^\W\d_{_UNSPACED_RANGES}]+)*"
)
# A restart repeats up to this many units ("we need to, we need to finish").
_MAX_RESTART_UNITS = 8
# The output has to keep this share of what was said, on the style presets
# too: they ask for a professional tone, but say not to summarize...
_MIN_KEPT_SHARE = 0.6
# ...unless it dropped less than this many half-words: six CJK characters or
# three words.
_MIN_DROPPED_SIZE = 6


def _size(units: list) -> int:
    """Size in half-words: a CJK character counts 1 and a word or digit 2,
    since most Chinese words have two characters. So "三" -> "3" never
    shrinks a text, nor does "three" -> "3" or "一三八零零一三八零零零" ->
    "13800138000"."""
    return sum(1 if _UNSPACED_RE.match(unit) else 2 for unit in units)


def _said_units(text: str) -> list:
    """The units of what the speaker said: spoken fillers and hedges left
    out, and a restart ("我们需要我们需要在周五之前完成") counted once.
    Repeated digits are not a restart ("八八八八", "1 1 2")."""
    units = []
    text = _SPOKEN_FILLER_RE.sub(" ", text.lower().replace("’", "'"))
    for unit in _SIZE_UNIT_RE.findall(text):
        units.append(unit)
        for n in range(1, min(_MAX_RESTART_UNITS, len(units) // 2) + 1):
            repeat = units[-n:]
            if repeat == units[-2 * n:-n] and not all(
                    u.isdigit() or u in _NUMBER_UNITS for u in repeat):
                del units[-n:]
                break
    return units


def _dropped_too_much(input_text: str, output_text: str) -> bool:
    """Detect over-deletion: the output keeps much less than the speaker
    said, and no self-correction marker explains the drop.

    The failure this catches: the speaker dictates several clauses and the
    model treats one as throat-clearing and drops it ("我尝试多录几句话，
    随便写一段中文，测试一下。" -> "随便写一段中文，测试一下。"). A
    self-correction drops a lot too ("五点，啊，不对，六点" -> "六点。"),
    but always with a marker in the input.

    Fillers and hedges don't count as something said, so removing them is
    never over-deletion, however much of the dictation they were: "okay so
    basically 我们下周要把这个方案做完" -> "我们下周需要把这个方案做完。"
    The output is sized as written, so a filler it keeps only makes the
    guard more lenient.
    """
    if _has_correction_marker(input_text):
        return False
    said = _size(_said_units(input_text))
    written = _size(_SIZE_UNIT_RE.findall(output_text))
    return (written < _MIN_KEPT_SHARE * said
            and said - written >= _MIN_DROPPED_SIZE)


def _script_changed(input_text: str, output_text: str) -> bool:
    """Detect if LLM changed the writing script (i.e. translated).

    Returns True if input is mostly CJK but output is mostly Latin,
    or vice versa. This catches unwanted translation of pure-script
    inputs. For mixed-script inputs see _mixed_script_collapsed.
    """
    in_cjk = len(_CJK_RE.findall(input_text))
    out_cjk = len(_CJK_RE.findall(output_text))
    in_total = max(len(input_text.strip()), 1)
    out_total = max(len(output_text.strip()), 1)

    in_ratio = in_cjk / in_total
    out_ratio = out_cjk / out_total

    # If input is >30% CJK but output is <10% CJK -> translated to Latin
    if in_ratio > 0.3 and out_ratio < 0.1:
        return True
    # If input is <10% CJK but output is >30% CJK -> translated to CJK
    if in_ratio < 0.1 and out_ratio > 0.3:
        return True
    return False


_LATIN_LETTER_RE = re.compile(r"[A-Za-zÀ-ſ]")  # ASCII + Latin-1 Supplement (é, ç, ñ, etc.)
_LATIN_WORD_RE = re.compile(r"[A-Za-zÀ-ſ]{3,}")  # 3+ char Latin words
# English fillers in code-switched speech ("okay so basically 我们下周...").
# Dropping them is filler removal (rule E2), not translation.
_LATIN_FILLERS = frozenset((
    "okay", "yeah", "yep", "basically", "actually", "literally", "like", "you",
    "know", "well", "right", "just", "really", "kinda", "sorta", "anyway",
    "alright", "mean", "umm", "uhm", "hmm",
))


def _mixed_script_collapsed(input_text: str, output_text: str) -> bool:
    """Detect when mixed-language input collapsed to a single script.

    Catches the failure where a user dictates "加一个function" or
    "把commit push一下" or "c'est très bien，我觉得这个idea不错" and the
    LLM "helpfully" translates the embedded English/French words into
    Chinese (function -> 功能, OK -> 好的, idea -> 想法, etc.). Code-
    switching is a deliberate style for tech-fluent multilingual users
    and the LLM must preserve the user's actual word choice.

    Heuristic: an input is "mixed" when it has both ≥10% CJK density and
    at least one Latin word of 3+ chars. If the output then drops the
    Latin character density below 2% OR drops the count of distinct Latin
    words by ≥2, the LLM almost certainly translated.
    """
    in_cjk = len(_CJK_RE.findall(input_text))
    in_total = max(len(input_text.strip()), 1)
    in_cjk_ratio = in_cjk / in_total

    in_latin_words = set(w.lower() for w in _LATIN_WORD_RE.findall(input_text)) - _LATIN_FILLERS

    # Only relevant when input is genuinely mixed (CJK + at least one
    # meaningful Latin word, not just stray punctuation or single letters)
    if in_cjk_ratio < 0.10 or len(in_latin_words) == 0:
        return False

    out_latin_chars = len(_LATIN_LETTER_RE.findall(output_text))
    out_total = max(len(output_text.strip()), 1)
    out_latin_ratio = out_latin_chars / out_total

    out_latin_words = set(w.lower() for w in _LATIN_WORD_RE.findall(output_text)) - _LATIN_FILLERS
    dropped_words = in_latin_words - out_latin_words

    # Translation almost-certainly happened if Latin density collapsed
    # to near-zero in the output.
    if out_latin_ratio < 0.02:
        return True
    # Or if half-or-more of the distinct Latin words went missing. This
    # subsumes the single-word case ("加一个function" loses 1/1 = 100%)
    # and catches the partial-translation case ("review 一下 Q3 roadmap"
    # losing only "roadmap" = 1/2 = 50%).
    if len(dropped_words) / len(in_latin_words) >= 0.5:
        return True
    return False


# ---------------------------------------------------------------------------
# Rule R2 guard: a dictated question must be echoed, never answered.
#
# The reported bug: the user dictated "WTO对管制类产品有什么要求" and a weak
# model pasted an invented answer ("WTO没有特定的管制要求，各成员国自行决定").
# An answer takes one of two shapes, and _answered_a_question() checks both:
#   1. The question turned into a statement:
#        "明天几点开会" -> "明天上午十点开会。"
#   2. The question is still there, but next to it sits a statement made of
#      words the speaker never said:
#        "What is the capital of France?" -> "What is the capital of France? Paris."
#        "...有什么要求" -> "...没有特定要求，各成员国自行决定。您还有其他问题吗？"
# Both checks rest on one fact: a faithful cleanup only deletes and fixes the
# speaker's words, while an answer has to say something new.
# ---------------------------------------------------------------------------

# Chinese question words (Simplified, Traditional and Cantonese). They count
# anywhere in a sentence, except inside a _CJK_NOT_A_QUESTION_RE phrase.
# Bare 几 is left out on purpose: "几本书" usually means "a few books".
_CJK_QUESTION_WORDS = (
    "什么", "什麼", "甚么", "甚麼", "啥", "干吗", "干嘛", "幹嘛",
    "怎么", "怎麼", "怎样", "怎樣", "咋", "如何",
    "为什么", "為什麼", "为何", "為何", "为啥", "為啥",
    "谁", "誰", "哪", "何时", "何時", "是否", "能否", "可否",
    "多少", "多久", "几点", "幾點", "几号", "幾號", "几岁", "幾歲", "几时", "幾時",
    "星期几", "星期幾", "周几", "週幾", "礼拜几", "禮拜幾",
    # Cantonese
    "咩", "乜", "点样", "點樣", "点解", "點解", "边个", "邊個", "边度", "邊度",
    "几多", "幾多", "系咪", "係咪", "有冇",
)
# "A-not-A" questions work with any verb: 是不是, 去不去, 有没有, 可不可以,
# 喜欢不喜欢. The lookarounds skip repeated denials: 不不不, 不是不是 and
# 不对不对 all mean "no, no".
_CJK_A_NOT_A_RE = re.compile(
    rf"(?<![不没沒])(?![不没沒])([{_CJK_RANGES}]{{1,2}})[不没沒]\1"
)

# A filler is only a filler when the sentence goes on after it. At the end
# of a sentence the same words are the question itself: "这个项目你知道吗
# 其实很难做完" vs "他的电话号码你知道吗", "这个方案，怎么说呢，还不太成熟"
# vs "这个词用英文怎么说呢".
_GOES_ON = r"(?=[\s，,、：:]*[^\s，,、：:。.！？!?；;…])"
# The question words a 无论/不管 ("no matter ...") clause is built on.
_NO_MATTER_WHAT = (
    rf"(?:如何|怎[么麼](?:样|樣)?|怎樣|什[么麼](?:时候|時候|地方)?|啥|[谁誰]|哪(?:儿|兒|里|裡)?"
    rf"|多少|多久|[几幾]|(?P<no_matter_a>[{_CJK_RANGES}]{{1,2}})[不没沒](?P=no_matter_a))"
)

# Phrases where a question word isn't asking anything. They are blanked out
# before looking for question words.
_CJK_NOT_A_QUESTION_RE = re.compile("|".join((
    # Spoken fillers the model is told to remove (rule E2): 怎么说呢 ("how
    # to put it"), 你知道吗 ("you know"), 那个什么/那个啥 ("um, that
    # thing"). 那个什么时候 is "when", not a filler.
    f"(?:怎[么麼]|咋)[说說讲講]呢{_GOES_ON}", f"你知道[吗嗎]{_GOES_ON}",
    f"那[个個]?(?:什[么麼]|啥)(?!时|時|样|樣|地方){_GOES_ON}",
    # "Any-" readings: 什么都行 (anything is fine), 水果什么的 (fruit and so
    # on), 没什么 (nothing), 谁都知道 (everyone knows), 哪儿都行 (anywhere),
    # 哪怕 (even if), 不怎么好 (not very good), 多少有点 (somewhat), 没多久
    # (soon after). The lookbehinds keep real questions intact: 为什么都不说
    # ("why won't anyone speak"), 这是干什么的 ("what is this for"), 有没有
    # 什么问题, 有多少有问题的 ("how many have problems").
    "(?<![为為])什[么麼][都也]", "(?<![为為干幹做搞弄是用学學卖賣叫说說讲講])什[么麼]的",
    "(?<![为為])啥[都也]", "(?<!有)[没沒]有?什[么麼]",
    "(?<!有)[没沒]啥", "[谁誰][都也]", "哪怕", "哪(?:儿|兒|里|裡)?都", "不怎[么麼]",
    "怎[么麼]都(?:行|可以|好|成)", "多少有(?=[点點些])", "[没沒]多[少久]", "不咋",
    # Reported discussion: "我们讨论了如何提高效率" says what was discussed.
    # "你讨论了..." is left alone: said to the listener it may be asking.
    "(?<![你您])(?:讨论|討論|研究|分析|探讨|探討|解释|解釋|说明|說明|介绍|介紹)(?:了|过|過)"
    "[^，,。！？!?；;]{0,2}?(?:如何|怎[么麼](?:样|樣)?|为什么|為什麼|为何|為何|是否)",
    # 无论/不管 + question word means "no matter what/who/how/whether". Only
    # that phrase goes: unpunctuated speech runs straight on into the main
    # clause, which may ask a real question ("不管下不下雨你们什么时候出发").
    f"(?:无论|無論|不管|不论|不論)[^，,。！？!?；;]{{0,8}}?{_NO_MATTER_WHAT}",
)))

# 吗 asks a question wherever it is, including mid-clause in unpunctuated
# speech ("你今天去吗我们一起走"). A misheard 嘛 ("这样就挺好的吗不用再改
# 了") is told apart in _question_became_statement, which accepts the fix.
_CJK_MA_RE = re.compile(r"[吗嗎]")
# 呢 at the end of a sentence asks "what about...?" ("那明天的会议呢",
# "你们觉得呢"). Mid-sentence it is a pause ("我呢觉得", "然后呢我们"), and
# after 在/着/还/才/正 it marks an ongoing state ("他在吃饭呢", "早着呢").
_CJK_FINAL_NE_RE = re.compile(r"呢[\s。.！!…\"'”’」』）)】\]]*$")
_CJK_ONGOING_RE = re.compile(r"[着著还還才正]|(?<![现現存实實所自])在")
# "Has it happened yet?": 合同签了没有, 你去过上海没有, 食咗饭未.
_CJK_YET_QUESTION_RE = re.compile(
    r"(?:[了过過][^，,。！？!?；;]{0,6}[没沒]有?|咗[^，,。！？!?；;]{0,6}未)"
    r"[\s。.！!…\"'”’」』）)】\]]*$"
)

_EN_QUESTION_WORDS = frozenset(
    ("what", "how", "why", "where", "when", "who", "which", "whose", "whom")
)
_EN_AUXILIARIES = frozenset((
    "am", "is", "are", "was", "were", "do", "does", "did", "have", "has", "had",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't",
    "haven't", "hasn't", "hadn't", "can't", "couldn't", "won't", "wouldn't",
    "shouldn't",
))
# Words that can follow an auxiliary to make it a question ("does the store",
# "is this", "can I"). Anything else is a statement: "should include...".
_EN_SUBJECTS = frozenset((
    "i", "you", "we", "they", "he", "she", "it", "this", "that", "these",
    "those", "there", "the", "a", "an", "my", "your", "our", "their", "his",
    "her", "its", "any", "anyone", "anybody", "anything", "someone",
    "somebody", "something", "everyone", "everybody", "everything",
))
# "do" and "have" also start commands ("Do the dishes", "Have a nice day"),
# so for them only a personal pronoun makes a question ("Do you...").
_EN_COMMAND_AUXILIARIES = frozenset(("do", "don't", "have"))
_EN_PERSONAL_PRONOUNS = frozenset(("i", "you", "we", "they"))
# Subjects of a clause that opens with a wh-word without asking anything:
# "what we need is...", "when you get a chance...".
_EN_CLEFT_SUBJECTS = _EN_PERSONAL_PRONOUNS | {"he", "she", "it"}
_EN_ASKING = frozenset(("know", "wonder", "wondering", "ask", "asking"))
# Words people say before a question: "so what's next", "OK, can you...".
_EN_LEAD_INS = frozenset((
    "so", "and", "but", "ok", "okay", "well", "also", "then", "hey", "oh",
    "now", "anyway", "alright", "right", "please",
))
_EN_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
# What follows a "should you..." condition in a statement.
_EN_CONDITIONAL_TAIL_RE = re.compile(
    r"\b(?:please|feel free|let (?:me|us) know|do not hesitate|don't hesitate"
    r"|reach out|contact (?:me|us))\b"
)

_QUESTION_MARKS = ("?", "？", "؟")
# Looked past when checking whether a sentence ends in a question mark:
# closing quotes and brackets ('“What time is it?”'), and "!" in "Really?!".
_SENTENCE_CLOSERS = " \t\"'”’」』）)】]!！"
# One sentence: text up to a terminator (。！？!?؟；; or a line break), or up
# to an ASCII period followed by a space or the end ("3.5" stays whole). The
# period of a title ("Mr. Smith", "Dr. Wang") doesn't end a sentence.
_SENTENCE_RE = re.compile(
    r"[^。！？!?؟；;\n]*?(?:[。！？!?؟；;\n]+"
    r"|(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)(?<!\bSt)(?<!\bJr)(?<!\bSr)(?<!\bProf)(?<!\bvs)"
    r"\.(?=\s|$)|$)",
    re.IGNORECASE,
)
# Clauses inside a sentence, for an answer joined to the echoed question by
# a comma ("明天几点开会，上午十点。").
_CLAUSE_SPLIT_RE = re.compile(r"[，,；;]")
# Content units for comparing input and output: each CJK character on its
# own (Chinese has no spaces between words) and whole words in every other
# script. Punctuation and spacing don't count.
_UNIT_RE = re.compile(rf"[{_CJK_RANGES}]|[^\W_{_CJK_RANGES}]+")
# Spoken numbers and digits are the same unit, so writing "four" as "4" or
# "三点" as "3点" (rule E4) adds nothing new.
_NUMBER_UNITS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
    "零": "0", "〇": "0", "一": "1", "二": "2", "两": "2", "兩": "2",
    "三": "3", "四": "4", "五": "5", "六": "6", "七": "7", "八": "8",
    "九": "9", "十": "10",
}


def _sentences(text: str) -> list:
    """Split text into sentences, each keeping its end punctuation."""
    return [s for s in _SENTENCE_RE.findall(text) if s.strip()]


def _content_units(text: str) -> Counter:
    """Count the content units (CJK characters, other words) in text."""
    return Counter(_NUMBER_UNITS.get(unit, unit)
                   for unit in (u.lower() for u in _UNIT_RE.findall(text)))


def _english_question(sentence: str) -> bool:
    """Does this sentence open like an English question? ("what's...",
    "does the store...", "can I...", "OK so how...")"""
    # Only English text at the very start of the sentence counts.
    lead = _CJK_RE.split(sentence, maxsplit=1)[0]
    words = _EN_WORD_RE.findall(lead.lower().replace("’", "'"))
    while words and words[0] in _EN_LEAD_INS:
        words.pop(0)
    if not words:
        return False
    first = words[0]
    after = words[1] if len(words) > 1 else ""
    wh_word = first.split("'")[0]  # "what's" -> "what"
    if wh_word in _EN_QUESTION_WORDS:
        if (first == wh_word and after in _EN_CLEFT_SUBJECTS
                and any(word in ("is", "was") for word in words[2:])
                and not any(word in _EN_ASKING for word in words)):
            # "what we need is more time", "how he did it was clever"; but
            # "what I want to know is when you'll arrive" still asks.
            return False
        # "when is it?", "where's the file?" and "where to go for dinner"
        # ask; "when you get a chance, send it" and "where we left off" only
        # set the scene, unless a question follows ("when you have time can
        # you review my PR").
        if wh_word in ("when", "where") and first == wh_word:
            return after in _EN_AUXILIARIES or after == "to" or any(
                word in _EN_AUXILIARIES and nxt in _EN_CLEFT_SUBJECTS
                for word, nxt in zip(words[2:], words[3:]))
        return True
    if first in _EN_COMMAND_AUXILIARIES:
        return after in _EN_PERSONAL_PRONOUNS
    # Conditional inversion only sets a condition: "had we known...",
    # "were it not for...", "should you have any questions, please...".
    if first == "had" or (first == "were" and after == "it") or (
            first == "should" and _EN_CONDITIONAL_TAIL_RE.search(lead.lower())):
        return False
    return first in _EN_AUXILIARIES and after in _EN_SUBJECTS


def _sentence_is_question(sentence: str) -> bool:
    """Does this one sentence ask a question?"""
    stripped = sentence.strip().rstrip(_SENTENCE_CLOSERS)
    if stripped.endswith(_QUESTION_MARKS) or "¿" in stripped:
        return True
    chinese = _CJK_NOT_A_QUESTION_RE.sub("", stripped)
    if any(word in chinese for word in _CJK_QUESTION_WORDS):
        return True
    if _CJK_A_NOT_A_RE.search(chinese) or _CJK_MA_RE.search(chinese):
        return True
    if _CJK_YET_QUESTION_RE.search(chinese):
        return True
    if _CJK_FINAL_NE_RE.search(chinese):
        last_clause = re.split(r"[，,、；;：:]", chinese)[-1]
        if not _CJK_ONGOING_RE.search(last_clause):
            return True
    return _english_question(stripped)


def _is_question(text: str) -> bool:
    """Best-effort: does any sentence of this text ask a question?

    Counts a final ?, ？ or ؟ (looking past closing quotes), Chinese question
    words anywhere in a sentence (but not fillers like 怎么说呢 or "any-"
    readings like 什么都行), 吗 where a clause ends, and English sentences
    that open like a question. "I know what you mean" is not a question.
    """
    return any(_sentence_is_question(s) for s in _sentences(text or ""))


# Rule R2 covers commands too: "tell me a joke about cats" must be pasted as
# that sentence, not as a joke, and "帮我写一封邮件给老板说我明天请假" not as
# the email itself. Cleanup (rules E1-E6) only deletes the speaker's words
# and fixes a few, so a carried-out command stands out as output made mostly
# of words the speaker never said. How large a share is allowed depends on
# the prompt: the default one forbids any rephrasing (rule R4), while the
# style presets ask for grammar fixes and a professional tone, which
# legitimately replace more words ("gonna" -> "going to", "搞定" -> "完成").
_MAX_ADDED_SHARE = 0.5
_MAX_ADDED_SHARE_STYLED = 0.6
# Below this many new units the share doesn't matter, so a few homophone or
# number fixes in a short text ("their" -> "they're", "一号" -> "1号") never
# count.
_MIN_ADDED_UNITS = 4
# A carried-out command writes something new at length (a joke, an email);
# a professional rewording stays about as long as what was said, even when
# it swaps every character ("这事儿我搞不定" -> "此事我无法完成"). On the
# styled path the output must also grow by this factor to count as one.
_MIN_GROWTH_STYLED = 1.3
# An output question echoes the dictated one when at least this share of its
# units are the speaker's. A chatbot follow-up ("还有其他问题吗？", "Is there
# anything else?") is almost all new words.
_MIN_ECHO_SHARE = 1 / 3
# Words that turn a request into a polite command without saying anything
# new: "can you send me the report" -> "Please send me the report."
_POLITE_UNITS = frozenset(("please", "kindly", "请", "請"))
# A 吗 the utterance runs on after is where a misheard 嘛 sits ("这样就挺好的
# 吗不用再改了"). A final 吗 ("明天开会吗") asks a real question.
_MA_GOES_ON_RE = re.compile(
    r"[吗嗎](?=[\s，,、；;：:]*[^\s，,、；;：:。.！？!?…\"'”’」』）)】\]])"
)


def _added_too_much(input_text: str, output_text: str, max_share: float,
                    min_growth: Optional[float] = None) -> bool:
    """Detect a carried-out command: at least _MIN_ADDED_UNITS of the
    output's content units, and more than max_share of them, are units the
    speaker never said. With min_growth the output must also have that many
    times the input's units."""
    input_units = _content_units(input_text)
    output_units = _content_units(output_text)
    total = sum(output_units.values())
    if min_growth is not None and total < min_growth * sum(input_units.values()):
        return False
    added = sum((output_units - input_units).values())
    return added >= _MIN_ADDED_UNITS and added > max_share * total


# How a chatbot opens a reply ("Sure! I'll remind you...", "Yes, Mark sent
# it.", "以下是邮件："). A cleanup never starts with these unless the
# speaker did, so they mark a reply even when most words are reused.
_REPLY_OPENING_RE = re.compile(
    r"^\W*(sure\b|certainly\b|of course\b|absolutely\b|no problem\b"
    r"|happy to\b|i'd be happy\b|here(?:'s| is| are)\b|i'll\b|i will\b"
    r"|yes\b|yeah\b|yep\b|nope\b|no(?=\s*[,，])"
    r"|是的|对的|對的|不是的|没问题|沒問題|以下是|下面是|我来帮|我來幫)",
    re.IGNORECASE,
)
# Ways of saying yes or no that count as the same opening word.
_SAME_ANSWER_WORD = {"yeah": "yes", "yep": "yes", "yup": "yes", "nope": "no", "nah": "no"}


def _opens_as_a_reply(input_text: str, output_text: str) -> bool:
    """Does the output open like a chatbot's reply the speaker never said?"""
    opening = _REPLY_OPENING_RE.match(output_text.replace("’", "'"))
    if not opening:
        return False
    # The speaker may have said it: "no we can't" -> "No, we can't.",
    # "yeah I think so" -> "Yes, I think so."
    phrase = opening.group(1).lower()
    said = re.sub(r"^\W+", "", input_text.replace("’", "'").lower())
    if not phrase.isascii():
        return not said.startswith(phrase)
    said_word = re.match(r"[a-z]+(?:'[a-z]+)?", said)
    if not said_word:
        return True
    first_word = phrase.split()[0]
    return (_SAME_ANSWER_WORD.get(said_word.group(0), said_word.group(0))
            != _SAME_ANSWER_WORD.get(first_word, first_word))


def _carried_out_command(input_text: str, output_text: str, max_share: float,
                         min_growth: Optional[float] = None) -> bool:
    """Detect a carried-out command (rule R2): an output mostly of new words,
    or one that opens as a chatbot's reply."""
    return (_added_too_much(input_text, output_text, max_share, min_growth)
            or _opens_as_a_reply(input_text, output_text))


def _echoes_the_question(input_text: str, output_text: str) -> bool:
    """Does the output still ask what the speaker asked? One of its question
    sentences has to be mostly the speaker's own words."""
    said = _content_units(input_text)
    for sentence in _sentences(output_text):
        if _sentence_is_question(sentence):
            units = _content_units(sentence)
            total = sum(units.values())
            if not total or sum((units & said).values()) >= _MIN_ECHO_SHARE * total:
                return True
    return False


def _question_became_statement(input_text: str, output_text: str,
                               max_share: float = _MAX_ADDED_SHARE) -> bool:
    """Answer shape 1: the output no longer asks the dictated question, and
    no legitimate edit explains why."""
    if _echoes_the_question(input_text, output_text):
        return False
    added = _content_units(output_text) - _content_units(input_text)
    # A self-correction can retract the question ("我们是不是周五开会，啊不对，
    # 我们周六开会" -> "我们周六开会。"): what the speaker settled on is a
    # statement, and the cleanup adds no more than a few small fixes ("on",
    # "4"), or on the styled path a normal rewording. If the settled-on part
    # still asks ("是周五吗，等等，是周四吗"), or the marker is just a word in
    # the question ("会议应该是下午三点吗"), the question has to stay.
    settled = _after_last_correction(input_text)
    if settled is not None and not _is_question(settled) and (
            sum(added.values()) < _MIN_ADDED_UNITS
            or (max_share > _MAX_ADDED_SHARE
                and not _added_too_much(input_text, output_text, max_share))):
        return False
    # Fixing a misheard 嘛 ("这样就挺好的吗，不用再改了" -> "这样就挺好的嘛，
    # 不用再改了。") removes the question cue without saying anything new.
    as_emphasis = input_text.replace("吗", "嘛").replace("嗎", "嘛")
    if ("嘛" in output_text and set(added) <= {"嘛"}
            and _MA_GOES_ON_RE.search(input_text) and not _is_question(as_emphasis)):
        return False
    # A request turned into a polite command: "can you send me the report by
    # friday" -> "Please send me the report by Friday."
    if added and set(added) <= _POLITE_UNITS:
        return False
    return True


def _answer_beside_question(input_text: str, output_text: str,
                            max_share: float = _MAX_ADDED_SHARE) -> bool:
    """Answer shape 2: the output still asks the question, but one of its
    statements is mostly words the speaker never said."""
    # The speaker's words the output hasn't used yet. Counting uses stops an
    # answer that repeats the question ("What is X? The capital of France is
    # Paris.") from hiding behind words the echo already used.
    unused = _content_units(input_text)
    for sentence in _sentences(output_text):
        if not _sentence_is_question(sentence):
            units = _content_units(sentence)
            if sum((units - unused).values()) > max_share * sum(units.values()):
                return True
            unused -= units
            continue
        # The answer can also hang off the echoed question after a comma
        # ("明天几点开会，上午十点。"). Such a clause needs two new units, so a
        # lightly reworded fragment of the question isn't taken for one.
        for clause in _CLAUSE_SPLIT_RE.split(sentence):
            units = _content_units(clause)
            if not _sentence_is_question(clause):
                new_words = sum((units - unused).values())
                if new_words >= 2 and new_words > max_share * sum(units.values()):
                    return True
            unused -= units
    return False


def _answered_a_question(input_text: str, output_text: str,
                         max_share: float = _MAX_ADDED_SHARE) -> bool:
    """Detect the rule-R2 takeover: the speaker dictated a question and the
    model answered it instead of echoing it."""
    if not _is_question(input_text):
        return False
    return (_question_became_statement(input_text, output_text, max_share)
            or _answer_beside_question(input_text, output_text, max_share))


class LLMCleanup:
    """Cleans up raw ASR text using a pluggable LLM backend."""

    def __init__(self, backend: LLMBackend = None) -> None:
        if backend is None:
            backend = OllamaBackend()
        self._backend = backend
        self._speculative_result: Optional[str] = None
        # The input text, prompt and guard flags that produced the cached
        # result. A result made under another mode's prompt and guards (a
        # translation, an answer) must not be pasted in this one.
        self._speculative_input: Optional[tuple] = None
        self._speculative_lock = threading.Lock()

    def set_backend(self, backend: LLMBackend) -> None:
        """Replace the active LLM backend."""
        self._backend = backend

    def is_available(self) -> bool:
        """Check if the current backend can serve requests."""
        return self._backend.is_available()

    def warm_up(self) -> None:
        """Pre-load the model into memory for fast first inference."""
        self._backend.warm_up()

    def speculative_cleanup(self, text: str, custom_prompt: str = None,
                            allow_script_change: bool = False,
                            echo_questions: bool = False):
        """Fire-and-forget: run cleanup in background, cache result."""
        key = (text, custom_prompt, allow_script_change, echo_questions)

        def _run():
            result = self.cleanup(
                text, custom_prompt=custom_prompt,
                allow_script_change=allow_script_change,
                echo_questions=echo_questions,
            )
            with self._speculative_lock:
                self._speculative_input = key
                self._speculative_result = result
        threading.Thread(target=_run, daemon=True).start()

    def get_speculative_result(self, text: str, custom_prompt: str = None,
                               allow_script_change: bool = False,
                               echo_questions: bool = False) -> Optional[str]:
        """Return the cached result if it was made for this text with the
        same prompt and guards, else None."""
        key = (text, custom_prompt, allow_script_change, echo_questions)
        with self._speculative_lock:
            if self._speculative_input == key and self._speculative_result:
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

    def cleanup(self, raw_text: str, languages: Optional[list] = None,
                custom_prompt: str = None,
                allow_script_change: bool = False,
                echo_questions: bool = False) -> str:
        """Clean up raw ASR text using the LLM.

        Args:
            raw_text: The raw ASR transcription text.
            languages: Optional list of target language names
                       (e.g. ["English"], ["English", "Chinese"]).
                       If provided (excluding "Auto"), foreign words will be
                       unified into the target language(s).
            custom_prompt: If provided, use this as the user message with a
                           generic system prompt instead of the dictation prompt.
            allow_script_change: Set True only for modes whose PURPOSE is to
                change the language (e.g. translation modes). When False, a
                custom-prompt result that flips the script (Chinese in,
                English out) is rejected as model misbehavior.
            echo_questions: Custom-prompt path only. Set True when the mode's
                prompt says dictated text must be transcribed, not answered
                or acted on (the style presets and Formal Writing do). A
                result that drops much of what was said, answers a dictated
                question or carries out a dictated command is then rejected,
                as on the default path, which always checks all three.

        Returns the cleaned text, or the original text if cleanup fails.
        """
        if not raw_text.strip():
            return raw_text

        # Always run the deterministic filler-word strip first. This both
        # cleans short/skipped-LLM cases and reduces the token surface the
        # LLM sees on the longer path. Self-corrections are intentionally
        # NOT handled here -- the LLM does that with semantic context.
        pre_cleaned = strip_filler_words(raw_text)

        if not self.is_available():
            return pre_cleaned

        if custom_prompt:
            try:
                result = self._backend.chat(
                    "Follow the instruction precisely. Output only the result.",
                    custom_prompt,
                )
                if result:
                    # Guard the custom path too. Reasoning-tuned models dump
                    # hundreds of tokens of untagged "let me think..." prose
                    # that _strip_think_tags can't catch (no <think> tags).
                    # Custom modes legitimately expand text, so this is far
                    # more generous than the dictation path's 2x guard, but it
                    # still catches a monologue blowup. On a trip, fall back to
                    # rule-stripped text rather than pasting the leak.
                    if len(result) > 4 * len(raw_text) + 400:
                        logger.warning(
                            "Custom LLM cleanup rejected (runaway length, likely "
                            "reasoning leak): %d-char input -> %d-char output. "
                            "Falling back to rule-stripped text.",
                            len(raw_text), len(result),
                        )
                        return pre_cleaned
                    # Same translation guards as the default path: a model
                    # that ignores "Do NOT translate" in a style prompt must
                    # not silently replace the user's Chinese with English.
                    # Translation modes opt out via allow_script_change.
                    if not allow_script_change and (
                        _script_changed(pre_cleaned, result)
                        or _mixed_script_collapsed(pre_cleaned, result)
                    ):
                        logger.warning(
                            "Custom LLM cleanup rejected (unrequested "
                            "translation/script change). Falling back to "
                            "rule-stripped text.",
                        )
                        return pre_cleaned
                    # Same over-deletion and rule-R2 guards as the default
                    # path, for modes whose prompt says to transcribe what
                    # was dictated. That covers Quick mode after the first-
                    # run wizard, which saves a style preset into it. Other
                    # custom modes may shorten or answer what was said.
                    # Translation modes skip the guards too: each compares
                    # the output with the speaker's own words, and a
                    # translation replaces all of them.
                    if echo_questions and not allow_script_change:
                        if _dropped_too_much(pre_cleaned, result):
                            logger.warning(
                                "Custom LLM cleanup rejected (over-deletion, "
                                "no self-correction marker in input): %s -> "
                                "%s. Falling back to rule-stripped text.",
                                redact(raw_text), redact(result),
                            )
                            return pre_cleaned
                        if _answered_a_question(pre_cleaned, result,
                                                _MAX_ADDED_SHARE_STYLED):
                            logger.warning(
                                "Custom LLM cleanup rejected (answered a "
                                "dictated question instead of echoing it): "
                                "%s -> %s. Falling back to rule-stripped text.",
                                redact(raw_text), redact(result),
                            )
                            return pre_cleaned
                        if _carried_out_command(pre_cleaned, result,
                                                _MAX_ADDED_SHARE_STYLED,
                                                _MIN_GROWTH_STYLED):
                            logger.warning(
                                "Custom LLM cleanup rejected (mostly words the "
                                "speaker never said, likely a carried-out "
                                "command): %s -> %s. Falling back to "
                                "rule-stripped text.",
                                redact(raw_text), redact(result),
                            )
                            return pre_cleaned
                    logger.info("Custom LLM: %s -> %s", redact(raw_text), redact(result))
                    return result
            except LLMTruncatedError as e:
                logger.warning(
                    "Custom LLM output truncated (%s); using rule-stripped "
                    "text so the tail of the dictation is not lost.", e,
                )
            except Exception as e:
                logger.warning("Custom LLM cleanup failed: %s", e)
            # Custom-prompt path failed: still return the rule-stripped text
            # rather than the truly raw one; user always gets fillers removed.
            return pre_cleaned

        try:
            cleaned = self._backend.chat(SYSTEM_PROMPT, pre_cleaned)

            if cleaned:
                # Guard: reject if LLM changed the script (full-text
                # translation of a single-script input).
                if _script_changed(pre_cleaned, cleaned):
                    logger.warning(
                        "LLM cleanup rejected (translation detected): %s -> %s",
                        redact(raw_text), redact(cleaned),
                    )
                    return pre_cleaned
                # Guard: reject if mixed-language input collapsed to a
                # single script (loanword-by-loanword translation, e.g.
                # "加一个function" -> "加一个功能"). Tech-fluent users
                # code-switch deliberately and want their actual words.
                if _mixed_script_collapsed(pre_cleaned, cleaned):
                    logger.warning(
                        "LLM cleanup rejected (mixed-script collapse, "
                        "loanwords were translated): %s -> %s",
                        redact(raw_text), redact(cleaned),
                    )
                    return pre_cleaned
                # Guard: reject runaway-length output. Cleanup should make
                # text shorter or roughly the same length, never much
                # longer. Reasoning-tuned models (e.g. qwen3:4b) often dump
                # 500+ tokens of "let me think about this..." prose into
                # the content field even with think:false set; that's not
                # a cleanup, it's a takeover.
                if len(cleaned) > 2 * len(pre_cleaned) + 60:
                    logger.warning(
                        "LLM cleanup rejected (runaway length, likely "
                        "reasoning leak from a thinking-tuned model): "
                        "%d-char input -> %d-char output. Falling back "
                        "to rule-stripped text.",
                        len(pre_cleaned), len(cleaned),
                    )
                    return pre_cleaned
                # Guard: reject over-deletion. If the output keeps less than
                # 60% of what was said (fillers and hedges don't count) and
                # the input has no self-correction marker, the LLM treated
                # meaningful content as filler -- e.g. "哎，我尝试多录几句
                # 话，随便写一段中文，测试一下" was being collapsed to "随便
                # 写一段中文，测试一下" because the model decided the first
                # clause was preamble. Self-corrections (which legitimately
                # drop a lot) always carry a marker in the input and are
                # exempt.
                if _dropped_too_much(pre_cleaned, cleaned):
                    logger.warning(
                        "LLM cleanup rejected (over-deletion, no self-"
                        "correction marker in input): %s -> %s. Falling "
                        "back to rule-stripped text.",
                        redact(raw_text), redact(cleaned),
                    )
                    return pre_cleaned
                # Guard: reject the rule-R2 takeover. The user dictated a
                # question to be pasted ("WTO对管制类产品有什么要求") and a
                # weak model "helpfully" answered it instead of echoing it,
                # inventing facts ("没有特定的管制要求") -- a hallucination,
                # the worst failure. See _answered_a_question for the two
                # shapes an answer takes. Fall back to the rule-stripped
                # transcript (still the user's question).
                if _answered_a_question(pre_cleaned, cleaned):
                    logger.warning(
                        "LLM cleanup rejected (answered a dictated question "
                        "instead of echoing it): %s -> %s. Falling back to "
                        "rule-stripped text.",
                        redact(raw_text), redact(cleaned),
                    )
                    return pre_cleaned
                # Guard: reject a carried-out command (rule R2), e.g. a joke
                # for "tell me a joke about cats": output made mostly of
                # words the speaker never said. It runs after the question
                # guard so an answered question is logged as one.
                if _carried_out_command(pre_cleaned, cleaned, _MAX_ADDED_SHARE):
                    logger.warning(
                        "LLM cleanup rejected (mostly words the speaker never "
                        "said, likely a carried-out command): %s -> %s. "
                        "Falling back to rule-stripped text.",
                        redact(raw_text), redact(cleaned),
                    )
                    return pre_cleaned
                logger.info("LLM cleanup: %s -> %s", redact(raw_text), redact(cleaned))
                return cleaned

        except LLMTruncatedError as e:
            logger.warning(
                "LLM output truncated (%s); using rule-stripped text so the "
                "tail of the dictation is not lost.", e,
            )
        except Exception as e:
            logger.warning("LLM cleanup failed: %s", e)

        return pre_cleaned
