"""Transcript redaction for log output.

The log file persists across sessions and lives outside the app's config
dir, so dictated content (passwords, client matter, anything spoken) must
not accumulate there by default. Content logging is an explicit opt-in for
debugging sessions only.
"""
import os

# Opt-in escape hatch for debugging ASR/LLM quality:
#   SAFEVOICE_LOG_CONTENT=1 python run.py
LOG_CONTENT = os.environ.get("SAFEVOICE_LOG_CONTENT") == "1"


def redact(text) -> str:
    """Return a log-safe representation of transcript *text*.

    With SAFEVOICE_LOG_CONTENT=1 the full repr is returned (old behavior);
    otherwise only the length, which is enough to follow the pipeline
    (empty vs short vs truncated) without storing what was said.
    """
    if text is None:
        return "<none>"
    if LOG_CONTENT:
        return repr(text)
    return f"<redacted {len(text)} chars>"
