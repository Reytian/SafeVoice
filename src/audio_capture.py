"""
Audio capture module for the macOS voice IME application.

Wraps sounddevice for real-time, non-blocking audio capture with thread-safe
buffering and RMS level metering.
"""

import threading
import math
from typing import Callable, Optional

import numpy as np
import sounddevice as sd


class AudioCapture:
    """Real-time audio capture using sounddevice.

    Provides non-blocking audio recording with an internal ring buffer,
    RMS level monitoring for UI visualization, and per-block callbacks
    for downstream processing (e.g., streaming to a speech-to-text engine).

    All public methods are safe to call from any thread. The sounddevice
    callback executes on a dedicated audio thread managed by PortAudio.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        blocksize: int = 1024,
    ) -> None:
        """Configure capture parameters without starting recording.

        Args:
            sample_rate: Audio sample rate in Hz. 16 kHz is standard for
                speech recognition models.
            channels: Number of audio channels. Mono (1) is typical for
                voice input.
            blocksize: Number of frames per callback invocation. Controls
                the latency/throughput trade-off.
        """
        self._sample_rate = sample_rate
        self._channels = channels
        self._blocksize = blocksize

        # Stream and state, guarded by _lock for thread safety.
        self._lock = threading.Lock()
        self._stream: Optional[sd.InputStream] = None
        self._recording = False

        # Buffer: list of numpy arrays accumulated during recording.
        self._buffer: list[np.ndarray] = []

        # Current RMS level, updated on every audio callback.
        # Stored as a plain float so reads are atomic on CPython,
        # but we still guard writes with the lock for correctness.
        self._current_level: float = 0.0

        # User-provided callback, invoked for each audio block.
        self._user_callback: Optional[Callable[[np.ndarray], None]] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        """Whether audio capture is currently active."""
        with self._lock:
            return self._recording

    @property
    def sample_rate(self) -> int:
        """The configured sample rate in Hz."""
        return self._sample_rate

    def start(self, callback: Optional[Callable[[np.ndarray], None]] = None) -> None:
        """Begin capturing audio from the default input device.

        Args:
            callback: Optional function called with each block of captured
                audio as a 1-D float32 numpy array (mono, 16 kHz). The
                callback is invoked on the PortAudio audio thread, so it
                should return quickly and avoid blocking operations.

        Raises:
            RuntimeError: If recording is already in progress.
            sd.PortAudioError: If the audio device cannot be opened.
        """
        with self._lock:
            if self._recording:
                raise RuntimeError("Recording is already in progress.")

            self._buffer = []
            self._current_level = 0.0
            self._user_callback = callback

            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                blocksize=self._blocksize,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            self._recording = True

    def stop(self) -> np.ndarray:
        """Stop recording and return the complete captured audio.

        Returns:
            A 1-D float32 numpy array containing all captured audio
            concatenated in order. Returns an empty array if no audio
            was captured.

        Raises:
            RuntimeError: If recording is not in progress.
        """
        with self._lock:
            if not self._recording:
                raise RuntimeError("Recording is not in progress.")

            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._recording = False
            self._user_callback = None

            # Concatenate all buffered chunks into a single array.
            if self._buffer:
                result = np.concatenate(self._buffer, axis=0)
            else:
                result = np.empty(0, dtype=np.float32)

            self._buffer = []
            self._current_level = 0.0

        return result

    def get_level(self) -> float:
        """Return the current RMS audio level for UI visualization.

        Returns:
            A float in the range [0.0, 1.0] representing the root mean
            square of the most recent audio block. 0.0 indicates silence;
            values near 1.0 indicate very loud input. The value is clamped
            to this range.
        """
        with self._lock:
            return self._current_level

    # ------------------------------------------------------------------
    # Internal callback
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        """Called by PortAudio on the audio thread for each block of input.

        This method must execute quickly and must not block. It copies the
        incoming audio, appends it to the internal buffer, computes the
        RMS level, and optionally invokes the user callback.

        Args:
            indata: Input audio data as a (frames, channels) float32 array.
            frames: Number of frames in this block.
            time_info: PortAudio time information (not used).
            status: PortAudio status flags. Non-zero indicates an issue
                such as input overflow.
        """
        if status:
            # Log overflow / underflow warnings. In a production app this
            # could be routed to a proper logger.
            print(f"sounddevice status: {status}")

        # Copy the data so we own the memory (indata's buffer may be reused).
        # Flatten from (frames, channels) to a 1-D array for mono audio.
        chunk = indata[:, 0].copy() if self._channels == 1 else indata.copy()

        # Compute RMS level, clamped to [0.0, 1.0].
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        rms = max(0.0, min(1.0, rms))

        with self._lock:
            if not self._recording:
                return
            self._buffer.append(chunk)
            self._current_level = rms
            user_cb = self._user_callback

        # Invoke the user callback outside the lock to avoid holding it
        # longer than necessary and to prevent potential deadlocks if the
        # callback itself interacts with AudioCapture.
        if user_cb is not None:
            try:
                user_cb(chunk)
            except Exception as exc:
                # Swallow exceptions from user callbacks so they don't
                # crash the audio thread. A production implementation
                # should log this properly.
                print(f"Error in audio callback: {exc}")
