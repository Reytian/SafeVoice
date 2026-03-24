"""ASR Engine module wrapping mlx-qwen3-asr for on-device speech recognition.

Provides both batch and streaming transcription via the Qwen3 ASR model
running locally on Apple Silicon through MLX.  Uses the Session API for
efficient model lifecycle management.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# HuggingFace Hub default cache directory on macOS
_HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


class ASREngineError(Exception):
    """Base exception for ASR engine errors."""


class ModelNotLoadedError(ASREngineError):
    """Raised when an operation requires a loaded model but none is available."""


class StreamingNotActiveError(ASREngineError):
    """Raised when a streaming operation is called without an active session."""


class ASREngine:
    """Speech recognition engine backed by mlx-qwen3-asr.

    Supports two modes of operation:

    1. **Batch transcription** -- pass a complete audio buffer to `transcribe()`
       and receive the full text at once.

    2. **Streaming transcription** -- call `start_streaming()`, feed audio chunks
       incrementally with `feed_chunk()`, and finalize with `finish_streaming()`.

    The underlying model runs entirely on-device via Apple MLX, requiring no
    network access after the initial model download.
    """

    MODEL_ID = "Qwen/Qwen3-ASR-0.6B"

    # Maps user-facing language names to the canonical name accepted by the
    # ASR API.  None means auto-detect.
    LANGUAGES: dict[str, str | None] = {
        "Auto": None,
        "English": "English",
        "Chinese": "Chinese",
        "French": "French",
        "Japanese": "Japanese",
        "Korean": "Korean",
        "German": "German",
        "Spanish": "Spanish",
        "Italian": "Italian",
        "Portuguese": "Portuguese",
        "Russian": "Russian",
        "Arabic": "Arabic",
        "Hindi": "Hindi",
        "Dutch": "Dutch",
        "Turkish": "Turkish",
    }

    # Reverse lookup: ISO code -> display name (used for normalizing
    # language labels coming back from the engine).
    _CODE_TO_NAME: dict[str, str] = {
        "en": "English",
        "zh": "Chinese",
        "fr": "French",
        "ja": "Japanese",
        "ko": "Korean",
        "de": "German",
        "es": "Spanish",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "ar": "Arabic",
        "hi": "Hindi",
        "nl": "Dutch",
        "tr": "Turkish",
    }

    def __init__(self, model_id: str = None) -> None:
        self._session = None
        if model_id:
            self.MODEL_ID = model_id
        self._streaming_state = None
        self._language: str = "Auto"

        # Streaming parameters (optimized defaults)
        self._chunk_size_sec: float = 1.0
        self._finalization_mode: str = "latency"
        self._endpointing_mode: str = "energy"
        self._max_context_sec: float = 30.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Return True if the model is currently loaded in memory."""
        return self._session is not None

    @property
    def language(self) -> str:
        """Return the currently configured language name."""
        return self._language

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the ASR model into memory.

        This is a blocking call.  The first invocation triggers a download
        from HuggingFace Hub if the model is not cached locally (~600 MB).
        Subsequent calls are near-instant as weights are loaded from disk.

        Raises:
            ASREngineError: If the model fails to load.
        """
        if self._session is not None:
            logger.debug("Model already loaded; skipping.")
            return

        try:
            from mlx_qwen3_asr import Session

            # Use local cache path if available to avoid HuggingFace network calls
            model_path = self._find_cached_model()
            model_ref = model_path if model_path else self.MODEL_ID
            logger.info("Loading ASR model: %s", model_ref)
            self._session = Session(model=model_ref)
            logger.info("ASR model loaded, running warmup transcription...")
            # Warmup: MLX JIT-compiles compute graphs on first inference.
            # Without this, the first real transcription takes 10-30x longer.
            warmup_audio = np.zeros(16000, dtype=np.float32)  # 1s silence
            self._session.transcribe(warmup_audio, language="English", verbose=False)
            logger.info("ASR model ready (warmup complete).")
        except Exception as exc:
            self._session = None
            raise ASREngineError(f"Failed to load ASR model: {exc}") from exc

    def _find_cached_model(self) -> str | None:
        """Find the locally cached model path to avoid network calls."""
        # HuggingFace Hub caches models in ~/.cache/huggingface/hub/
        safe_id = self.MODEL_ID.replace("/", "--")
        model_dir = _HF_CACHE_DIR / f"models--{safe_id}" / "snapshots"
        if model_dir.is_dir():
            snapshots = sorted(model_dir.iterdir())
            if snapshots:
                path = str(snapshots[-1])
                logger.debug("Using cached model at: %s", path)
                return path
        return None

    def unload_model(self) -> None:
        """Release the model from memory.

        Any active streaming session is discarded.
        """
        self._session = None
        self._streaming_state = None
        logger.info("ASR model unloaded.")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_language(self, language: str) -> None:
        """Set the recognition language.

        Args:
            language: One of the keys in ``LANGUAGES`` (e.g. ``'English'``,
                ``'Japanese'``) or ``'Auto'`` for automatic detection.

        Raises:
            ValueError: If *language* is not a recognised key.
        """
        if language not in self.LANGUAGES:
            valid = ", ".join(sorted(self.LANGUAGES))
            raise ValueError(
                f"Unknown language {language!r}. Valid options: {valid}"
            )
        self._language = language
        logger.debug("Language set to %s", language)

    def set_streaming_params(
        self,
        *,
        chunk_size_sec: float | None = None,
        finalization_mode: str | None = None,
        endpointing_mode: str | None = None,
        max_context_sec: float | None = None,
    ) -> None:
        """Configure streaming parameters for subsequent sessions.

        Parameters only take effect on the next ``start_streaming()`` call.
        Pass ``None`` to keep the current value.
        """
        if chunk_size_sec is not None:
            self._chunk_size_sec = chunk_size_sec
        if finalization_mode is not None:
            self._finalization_mode = finalization_mode
        if endpointing_mode is not None:
            self._endpointing_mode = endpointing_mode
        if max_context_sec is not None:
            self._max_context_sec = max_context_sec
        logger.debug(
            "Streaming params: chunk=%.1fs, finalize=%s, endpoint=%s, ctx=%.0fs",
            self._chunk_size_sec,
            self._finalization_mode,
            self._endpointing_mode,
            self._max_context_sec,
        )

    # ------------------------------------------------------------------
    # Batch transcription
    # ------------------------------------------------------------------

    def transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        """Transcribe a complete audio buffer in one shot.

        Args:
            audio: 1-D float32 NumPy array of PCM samples at 16 kHz.

        Returns:
            A ``(text, language)`` tuple where *text* is the transcribed
            string and *language* is the detected/used language name
            (e.g. ``"English"``).

        Raises:
            ModelNotLoadedError: If ``load_model()`` has not been called.
            ASREngineError: On any transcription failure.
        """
        self._ensure_model_loaded()

        try:
            lang_hint: str | None = self.LANGUAGES.get(self._language)

            result = self._session.transcribe(
                audio,
                language=lang_hint,
                verbose=False,
            )

            detected = self._normalize_language(result.language)
            text = result.text.strip()
            logger.debug(
                "Batch transcription complete: %d chars, lang=%s",
                len(text),
                detected,
            )
            return text, detected

        except ModelNotLoadedError:
            raise
        except Exception as exc:
            raise ASREngineError(f"Transcription failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Streaming transcription
    # ------------------------------------------------------------------

    def start_streaming(self) -> None:
        """Begin a new streaming transcription session.

        Must be called before ``feed_chunk()``.  Any previous streaming
        session is silently discarded.

        Raises:
            ModelNotLoadedError: If the model is not loaded.
            ASREngineError: If streaming initialisation fails.
        """
        self._ensure_model_loaded()

        try:
            lang_hint: str | None = self.LANGUAGES.get(self._language)

            self._streaming_state = self._session.init_streaming(
                language=lang_hint,
                chunk_size_sec=self._chunk_size_sec,
                finalization_mode=self._finalization_mode,
                endpointing_mode=self._endpointing_mode,
                max_context_sec=self._max_context_sec,
            )
            logger.debug(
                "Streaming session started (lang=%s, chunk=%.1fs, "
                "finalize=%s, endpoint=%s).",
                lang_hint,
                self._chunk_size_sec,
                self._finalization_mode,
                self._endpointing_mode,
            )

        except ModelNotLoadedError:
            raise
        except Exception as exc:
            self._streaming_state = None
            raise ASREngineError(
                f"Failed to initialise streaming: {exc}"
            ) from exc

    def feed_chunk(self, audio_chunk: np.ndarray) -> tuple[str, str]:
        """Feed a PCM audio chunk to the running streaming session.

        Args:
            audio_chunk: 1-D float32 NumPy array of PCM samples at 16 kHz.
                Typical chunk length is 0.1 -- 0.5 seconds.

        Returns:
            A ``(partial_text, language)`` tuple reflecting the best
            transcription so far.

        Raises:
            StreamingNotActiveError: If no streaming session is active.
            ASREngineError: On processing failure.
        """
        if self._streaming_state is None:
            raise StreamingNotActiveError(
                "No active streaming session. Call start_streaming() first."
            )

        try:
            self._streaming_state = self._session.feed_audio(
                audio_chunk, self._streaming_state
            )

            text = (self._streaming_state.text or "").strip()
            detected = self._normalize_language(
                getattr(self._streaming_state, "language", None)
            )
            return text, detected

        except StreamingNotActiveError:
            raise
        except Exception as exc:
            raise ASREngineError(
                f"Error feeding audio chunk: {exc}"
            ) from exc

    def finish_streaming(self) -> tuple[str, str]:
        """Finalize the current streaming session and return the result.

        Flushes any remaining audio in internal buffers and produces the
        final transcription.  The streaming state is cleared afterwards.

        Returns:
            A ``(text, language)`` tuple with the final transcription.

        Raises:
            StreamingNotActiveError: If no streaming session is active.
            ASREngineError: On processing failure.
        """
        if self._streaming_state is None:
            raise StreamingNotActiveError(
                "No active streaming session. Call start_streaming() first."
            )

        try:
            self._streaming_state = self._session.finish_streaming(
                self._streaming_state
            )

            text = (self._streaming_state.text or "").strip()
            detected = self._normalize_language(
                getattr(self._streaming_state, "language", None)
            )
            logger.debug(
                "Streaming finalised: %d chars, lang=%s", len(text), detected
            )
            return text, detected

        except StreamingNotActiveError:
            raise
        except Exception as exc:
            raise ASREngineError(
                f"Error finalising streaming: {exc}"
            ) from exc
        finally:
            self._streaming_state = None

    # ------------------------------------------------------------------
    # Model download helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_model_downloaded() -> bool:
        """Check whether the model weights are already cached locally.

        Inspects the default HuggingFace Hub cache directory for a snapshot
        of the model.  This is a fast filesystem check with no network
        access.
        """
        repo_dir = _HF_CACHE_DIR / (
            "models--" + ASREngine.MODEL_ID.replace("/", "--")
        )

        if not repo_dir.is_dir():
            return False

        snapshots = repo_dir / "snapshots"
        if not snapshots.is_dir():
            return False

        for child in snapshots.iterdir():
            if child.is_dir() and any(child.iterdir()):
                return True

        return False

    @staticmethod
    def download_model(
        progress_callback: Callable[[float], None] | None = None,
    ) -> None:
        """Download the model from HuggingFace Hub.

        Args:
            progress_callback: Optional callable invoked with a float in
                ``[0.0, 1.0]`` representing download progress.

        Raises:
            ASREngineError: If the download fails.
        """
        try:
            from huggingface_hub import snapshot_download

            logger.info("Downloading model: %s", ASREngine.MODEL_ID)

            if progress_callback is not None:
                progress_callback(0.0)

            snapshot_download(
                repo_id=ASREngine.MODEL_ID,
                repo_type="model",
            )

            if progress_callback is not None:
                progress_callback(1.0)

            logger.info("Model download complete.")

        except Exception as exc:
            raise ASREngineError(
                f"Failed to download model: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Raise ``ModelNotLoadedError`` if the model is not loaded."""
        if self._session is None:
            raise ModelNotLoadedError(
                "ASR model is not loaded. Call load_model() first."
            )

    @classmethod
    def _normalize_language(cls, raw: str | None) -> str:
        """Convert a language value from the engine into a display name.

        The engine may return full names (``"English"``), short codes
        (``"en"``), or ``None``.  This method normalises them all into
        the canonical display name used by ``LANGUAGES``.
        """
        if raw is None:
            return "Auto"

        # Already a display name?
        if raw in cls.LANGUAGES:
            return raw

        # Try short-code lookup.
        name = cls._CODE_TO_NAME.get(raw.lower())
        if name is not None:
            return name

        # Fall back to whatever the engine returned.
        return raw
