"""Prototype script to test the mlx-qwen3-asr package.

Tests the Qwen3-ASR-0.6B 8-bit model on Apple Silicon via MLX.
Generates a synthetic audio tone and runs transcription on it.

Usage:
    python prototypes/test_asr.py

The model will be downloaded from HuggingFace on first run (~400MB).
"""

import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

# 8-bit quantized model from mlx-community (smaller download, faster inference)
MODEL_ID = "mlx-community/Qwen3-ASR-0.6B-8bit"

# Audio parameters
SAMPLE_RATE = 16000  # 16kHz, required by Qwen3-ASR
DURATION_SEC = 1.0   # 1 second test tone
TONE_FREQ_HZ = 440   # A4 note


# ---------------------------------------------------------------------------
# 2. Generate synthetic test audio (sine wave)
# ---------------------------------------------------------------------------

def generate_test_audio(
    freq_hz: float = TONE_FREQ_HZ,
    duration_sec: float = DURATION_SEC,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a mono sine wave as float32 numpy array.

    Args:
        freq_hz: Frequency of the sine wave in Hz.
        duration_sec: Duration in seconds.
        sample_rate: Sample rate in Hz.
        amplitude: Peak amplitude (0.0 to 1.0).

    Returns:
        1-D float32 numpy array of shape (n_samples,).
    """
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False, dtype=np.float32)
    audio = amplitude * np.sin(2.0 * np.pi * freq_hz * t)
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Main test routine
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("mlx-qwen3-asr Prototype Test")
    print("=" * 60)

    # -- Package version --
    import mlx_qwen3_asr
    print(f"\nPackage version: {mlx_qwen3_asr.__version__}")
    print(f"Model: {MODEL_ID}")

    # -- Generate test audio --
    print(f"\nGenerating test audio: {TONE_FREQ_HZ}Hz sine wave, "
          f"{DURATION_SEC}s @ {SAMPLE_RATE}Hz")
    audio = generate_test_audio()
    print(f"  Shape: {audio.shape}, dtype: {audio.dtype}, "
          f"range: [{audio.min():.3f}, {audio.max():.3f}]")

    # -- Load model --
    print(f"\nLoading model (first run downloads from HuggingFace)...")
    t0 = time.perf_counter()
    model, config = mlx_qwen3_asr.load_model(MODEL_ID)
    t_load = time.perf_counter() - t0
    print(f"  Model loaded in {t_load:.2f}s")
    print(f"  Audio encoder layers: {config.audio_config.encoder_layers}")
    print(f"  Text decoder layers: {config.text_config.num_hidden_layers}")
    print(f"  Vocab size: {config.text_config.vocab_size}")

    # -- Transcribe using the high-level API --
    # The transcribe() function accepts numpy arrays directly.
    # For a sine wave, the model will likely output empty text or noise tokens,
    # which is expected -- this test validates the pipeline end-to-end.
    print("\nRunning transcription (high-level API)...")
    t0 = time.perf_counter()
    result = mlx_qwen3_asr.transcribe(
        audio,
        model=MODEL_ID,
        language="English",
        verbose=True,
    )
    t_transcribe = time.perf_counter() - t0

    print(f"\n--- Transcription Result ---")
    print(f"  Text: {result.text!r}")
    print(f"  Language: {result.language}")
    print(f"  Segments: {result.segments}")
    print(f"  Time: {t_transcribe:.2f}s")
    print(f"  RTF: {t_transcribe / DURATION_SEC:.2f}x")

    # -- Test Session API (power-user path) --
    print("\n--- Session API test ---")
    session = mlx_qwen3_asr.Session(MODEL_ID)
    print(f"  Session model info: {session.model_info}")

    t0 = time.perf_counter()
    result2 = session.transcribe(audio, language="English")
    t_session = time.perf_counter() - t0
    print(f"  Session transcribe text: {result2.text!r}")
    print(f"  Session transcribe time: {t_session:.2f}s")

    # -- Test with a tuple input (audio, sample_rate) --
    print("\n--- Tuple input test (audio, sr) ---")
    result3 = mlx_qwen3_asr.transcribe(
        (audio, SAMPLE_RATE),
        model=MODEL_ID,
    )
    print(f"  Tuple input text: {result3.text!r}")
    print(f"  Tuple input language: {result3.language}")

    # -- Summary --
    print("\n" + "=" * 60)
    print("All tests passed. Pipeline is functional.")
    print("=" * 60)


if __name__ == "__main__":
    main()
