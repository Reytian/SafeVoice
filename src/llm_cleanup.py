"""LLM post-processing for cleaning up raw ASR transcription.

Uses Ollama (local) with Qwen3-4B to convert messy spoken text
into clean, formal written text.
"""

import json
import logging
import re
import unicodedata
import urllib.request
import urllib.error
from typing import Optional

logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b"

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
8. Output ONLY the cleaned text. No explanations, no labels, no quotes."""

# Few-shot examples — most important example goes LAST
FEW_SHOT = [
    {
        "role": "user",
        "content": "这个。新的功能。并不能用。",
    },
    {
        "role": "assistant",
        "content": "这个新的功能并不能用。",
    },
    {
        "role": "user",
        "content": "嗯那个就是我想说一下就是这个项目然后需要在下周五之前完成",
    },
    {
        "role": "assistant",
        "content": "我想说一下，这个项目需要在下周五之前完成。",
    },
    {
        "role": "user",
        "content": "um so like I was thinking that we should you know meet on Tuesday no wait Wednesday at like 2 PM to discuss the uh the budget",
    },
    {
        "role": "assistant",
        "content": "I was thinking that we should meet on Wednesday at 2 PM to discuss the budget.",
    },
]


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

    # If input is >30% CJK but output is <10% CJK → translated to Latin
    if in_ratio > 0.3 and out_ratio < 0.1:
        return True
    # If input is <10% CJK but output is >30% CJK → translated to CJK
    if in_ratio < 0.1 and out_ratio > 0.3:
        return True
    return False


class LLMCleanup:
    """Cleans up raw ASR text using a local LLM via Ollama."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self._model = model
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is downloaded.

        Only caches a positive result.  Transient failures (Ollama not
        running yet) are retried on the next call so that starting
        Ollama mid-session is supported.
        """
        if self._available is True:
            return True
        try:
            req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                names = [m["name"] for m in data.get("models", [])]
                # Match model name with or without :latest tag
                self._available = (
                    self._model in names
                    or f"{self._model}:latest" in names
                )
                return self._available
        except Exception:
            self._available = None
            return False

    def warm_up(self) -> None:
        """Pre-load the model into memory for fast first inference."""
        if not self.is_available():
            return
        try:
            body = json.dumps({
                "model": self._model,
                "keep_alive": "30m",
                "prompt": "",
            }).encode()
            req = urllib.request.Request(
                f"{OLLAMA_BASE}/api/generate",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30):
                pass
            logger.info("LLM model warmed up: %s", self._model)
        except Exception as e:
            logger.warning("LLM warm-up failed: %s", e)

    def cleanup(self, raw_text: str, languages: Optional[list] = None, custom_prompt: str = None) -> str:
        """Clean up raw ASR text using the LLM.

        Args:
            raw_text: The raw ASR transcription text.
            languages: Optional list of target language names
                       (e.g. ["English"], ["English", "Chinese"]).
                       If provided (excluding "Auto"), foreign words will be
                       unified into the target language(s).

        Returns the cleaned text, or the original text if cleanup fails.
        """
        if not raw_text.strip():
            return raw_text

        if not self.is_available():
            return raw_text

        if custom_prompt:
            try:
                messages = [
                    {"role": "system", "content": "Follow the instruction precisely. Output only the result."},
                    {"role": "user", "content": custom_prompt},
                ]
                body = json.dumps({
                    "model": self._model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 512, "top_p": 0.9},
                }).encode()
                req = urllib.request.Request(
                    f"{OLLAMA_BASE}/api/chat",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read())
                    result = data["message"]["content"].strip()
                    if "<think>" in result:
                        idx = result.find("</think>")
                        if idx != -1:
                            result = result[idx + len("</think>"):].strip()
                        else:
                            result = result[:result.find("<think>")].strip()
                    if result:
                        logger.info("Custom LLM: %r -> %r", raw_text, result)
                        return result
            except Exception as e:
                logger.warning("Custom LLM cleanup failed: %s", e)
            return raw_text

        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                *FEW_SHOT,
                {"role": "user", "content": raw_text},
            ]
            body = json.dumps({
                "model": self._model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 256,
                    "top_p": 0.9,
                },
            }).encode()
            req = urllib.request.Request(
                f"{OLLAMA_BASE}/api/chat",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                cleaned = data["message"]["content"].strip()

                # Strip any <think> tags that might leak through
                if "<think>" in cleaned:
                    idx = cleaned.find("</think>")
                    if idx != -1:
                        cleaned = cleaned[idx + len("</think>"):].strip()
                    else:
                        # No closing tag: strip from <think> onward
                        cleaned = cleaned[:cleaned.find("<think>")].strip()

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

        except urllib.error.URLError as e:
            logger.warning("LLM cleanup network error: %s", e)
        except Exception as e:
            logger.warning("LLM cleanup failed: %s", e)

        return raw_text
