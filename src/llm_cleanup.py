"""LLM post-processing for cleaning up raw ASR transcription.

Uses a pluggable LLM backend (local Ollama or cloud) to convert messy
spoken text into clean, formal written text.
"""

import logging
import re
import threading
from typing import Optional

from .llm_backend import LLMBackend, OllamaBackend

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a dictation text editor. You receive raw speech transcription and output clean written text.

Rules you MUST follow:
1. PRESERVE the original language. If input is Chinese, output Chinese. If English, output English. NEVER translate.
2. Remove filler words: um, uh, like, you know, I mean, 嗯, 那个, 就是, 然后, 对, 这个, 啊, 所以说
3. Handle self-corrections: keep ONLY the final intent. "Tuesday no wait Wednesday" -> "Wednesday"
4. Remove stuttering: "I I want" -> "I want"
5. Fix grammar, spelling, capitalization, and punctuation
6. Merge broken sentence fragments: "这个。新的。并不能用。" -> "这个新的并不能用。"
7. Keep ALL information and meaning — do NOT drop or summarize content
8. Output ONLY the cleaned text. No explanations, no labels, no quotes.

Examples:
User: 这个。新的功能。并不能用。
Assistant: 这个新的功能并不能用。

User: 嗯那个就是我想说一下就是这个项目然后需要在下周五之前完成
Assistant: 我想说一下，这个项目需要在下周五之前完成。

User: um so like I was thinking that we should you know meet on Tuesday no wait Wednesday at like 2 PM to discuss the uh the budget
Assistant: I was thinking that we should meet on Wednesday at 2 PM to discuss the budget."""


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

        if not self.is_available():
            return raw_text

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
            return raw_text

        try:
            cleaned = self._backend.chat(SYSTEM_PROMPT, raw_text)

            if cleaned:
                # Guard: reject if LLM changed the script (translated)
                if _script_changed(raw_text, cleaned):
                    logger.warning(
                        "LLM cleanup rejected (translation detected): %r -> %r",
                        raw_text, cleaned,
                    )
                    return raw_text
                logger.info("LLM cleanup: %r -> %r", raw_text, cleaned)
                return cleaned

        except Exception as e:
            logger.warning("LLM cleanup failed: %s", e)

        return raw_text
