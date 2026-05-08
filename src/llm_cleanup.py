"""LLM post-processing for cleaning up raw ASR transcription.

Uses a pluggable LLM backend (local Ollama or cloud) to convert messy
spoken text into clean, formal written text.
"""

import logging
import re
import threading
from typing import Optional

from .llm_backend import LLMBackend, OllamaBackend
from .text_postprocess import strip_filler_words

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a dictation text editor. You receive raw speech transcription and output clean written text.

Rules you MUST follow:
1. PRESERVE the original language. If input is Chinese, output Chinese. If English, output English. NEVER translate.
2. Remove filler words. Common ones:
     English: um, uh, er, ah, erm, hmm, like, you know, I mean, sort of, kind of, basically, literally, well, so (when used as a filler), right
     Chinese: 嗯, 啊, 呃, 哦, 哎, 唉, 那个, 这个, 就是, 然后, 对, 所以说, 对吧, 怎么说呢, 你知道, 我跟你讲
   Be context-aware: "这个产品" keeps 这个 (meaningful demonstrative); "这个，就是，我想说" drops 这个 and 就是 (filler usage).
3. Handle self-corrections — KEEP ONLY THE FINAL INTENT, drop the corrected-away part entirely.
   Recognize correction markers: "no wait", "I mean", "sorry", "啊不对", "不对不对", "不是", "应该是", "我是说", "等等", "哦不是", "改一下".
   The corrected-away element is removed completely; the replacement stays.
4. Remove stuttering: "I I want" -> "I want", "我我想" -> "我想"
5. Fix grammar, spelling, capitalization, and punctuation
6. Merge broken sentence fragments: "这个。新的。并不能用。" -> "这个新的并不能用。"
7. Keep ALL information and meaning — do NOT drop or summarize content. Self-correction removal is NOT summarization; it's removing a slip the speaker openly retracted.
8. Output ONLY the cleaned text. No explanations, no labels, no quotes.

Examples:

User: 这个。新的功能。并不能用。
Assistant: 这个新的功能并不能用。

User: 嗯那个就是我想说一下就是这个项目然后需要在下周五之前完成
Assistant: 我想说一下，这个项目需要在下周五之前完成。

User: 今天下午五点，啊，不对，六点开会
Assistant: 今天下午六点开会。

User: 他叫张明，不对，叫张亮
Assistant: 他叫张亮。

User: 我们去星巴克，呃，麦当劳吧
Assistant: 我们去麦当劳吧。

User: 明天，哦不是，是后天我们见面
Assistant: 后天我们见面。

User: 我想订三张票，不对不对，是四张
Assistant: 我想订四张票。

User: 明天上午，等等，下午三点
Assistant: 明天下午三点。

User: um so like I was thinking that we should you know meet on Tuesday no wait Wednesday at like 2 PM to discuss the uh the budget
Assistant: I was thinking that we should meet on Wednesday at 2 PM to discuss the budget.

User: send it to John, sorry I mean Jane
Assistant: Send it to Jane."""


_CJK_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]"
)


def _script_changed(input_text: str, output_text: str) -> bool:
    """Detect if LLM changed the writing script (i.e. translated).

    Returns True if input is mostly CJK but output is mostly Latin,
    or vice versa. This catches unwanted translation.
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


class LLMCleanup:
    """Cleans up raw ASR text using a pluggable LLM backend."""

    def __init__(self, backend: LLMBackend = None) -> None:
        if backend is None:
            backend = OllamaBackend()
        self._backend = backend
        self._speculative_result: Optional[str] = None
        self._speculative_input: Optional[str] = None
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

    def cleanup(self, raw_text: str, languages: Optional[list] = None, custom_prompt: str = None) -> str:
        """Clean up raw ASR text using the LLM.

        Args:
            raw_text: The raw ASR transcription text.
            languages: Optional list of target language names
                       (e.g. ["English"], ["English", "Chinese"]).
                       If provided (excluding "Auto"), foreign words will be
                       unified into the target language(s).
            custom_prompt: If provided, use this as the user message with a
                           generic system prompt instead of the dictation prompt.

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
                    logger.info("Custom LLM: %r -> %r", raw_text, result)
                    return result
            except Exception as e:
                logger.warning("Custom LLM cleanup failed: %s", e)
            # Custom-prompt path failed: still return the rule-stripped text
            # rather than the truly raw one; user always gets fillers removed.
            return pre_cleaned

        try:
            cleaned = self._backend.chat(SYSTEM_PROMPT, pre_cleaned)

            if cleaned:
                # Guard: reject if LLM changed the script (translated)
                if _script_changed(pre_cleaned, cleaned):
                    logger.warning(
                        "LLM cleanup rejected (translation detected): %r -> %r",
                        raw_text, cleaned,
                    )
                    return pre_cleaned
                # Guard: reject runaway-length output. Cleanup should make
                # text shorter or roughly the same length, never much
                # longer. Reasoning-tuned models (e.g. qwen3:4b) often dump
                # 500+ tokens of "let me think about this..." prose into
                # the content field even with think:false set; that's not
                # a cleanup, it's a takeover. Heuristic: 2x length AND a
                # floor of +60 chars excess catches reasoning leakage
                # without false-positiving on legitimate punctuation /
                # capitalization additions.
                if len(cleaned) > 2 * len(pre_cleaned) + 60:
                    logger.warning(
                        "LLM cleanup rejected (runaway length, likely "
                        "reasoning leak from a thinking-tuned model): "
                        "%d-char input -> %d-char output. Falling back "
                        "to rule-stripped text.",
                        len(pre_cleaned), len(cleaned),
                    )
                    return pre_cleaned
                logger.info("LLM cleanup: %r -> %r", raw_text, cleaned)
                return cleaned

        except Exception as e:
            logger.warning("LLM cleanup failed: %s", e)

        return pre_cleaned
