"""Audio preprocessing utilities for SafeVoice.

Functions to normalize and condition raw microphone audio
before sending it to the ASR engine.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


# Below this RMS the clip is treated as no useful speech and left untouched,
# so faint room tone is never amplified up to full scale (which makes ASR
# hallucinate). Above it, the gain is still capped to a sane maximum.
_SPEECH_RMS_FLOOR = 0.01
_MAX_GAIN = 8.0


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize audio to target peak amplitude.

    Prevents clipping and ensures consistent volume for ASR. Leaves
    near-silent input unchanged (gated on RMS) and caps the applied gain,
    so faint background noise in a quiet room is not blown up to full scale.
    """
    if audio.size == 0:
        return audio
    peak = float(np.abs(audio).max())
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if peak <= 0.0 or rms < _SPEECH_RMS_FLOOR:
        # Silence / room tone only — amplifying this just feeds the ASR noise.
        return audio
    gain = min(target_peak / peak, _MAX_GAIN)
    return audio * gain


def reduce_noise(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Apply spectral noise reduction to audio.

    Uses noisereduce library for stationary noise removal
    (fan hum, AC, ambient sounds).
    Falls back to original audio if noisereduce is unavailable.

    Args:
        audio: 1-D float32 array.
        sample_rate: Audio sample rate in Hz.
    """
    try:
        import noisereduce as nr

        cleaned = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            stationary=True,
            prop_decrease=0.75,  # How much to reduce noise (0-1)
        )
        logger.debug("Noise reduction applied")
        return cleaned.astype(np.float32)

    except ImportError:
        logger.warning("noisereduce not installed, skipping noise reduction")
        return audio
    except Exception as e:
        logger.warning("Noise reduction failed: %s", e)
        return audio


def vad_trim(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Trim non-speech segments using Silero VAD.

    Returns only the speech portions concatenated together.
    Falls back to original audio if VAD fails or finds no speech.

    Args:
        audio: 1-D float32 array at sample_rate Hz.
        sample_rate: Audio sample rate (must be 16000 for Silero).
    """
    try:
        import torch
        from silero_vad import load_silero_vad, get_speech_timestamps

        model = load_silero_vad()
        wav_tensor = torch.from_numpy(audio)

        speech_timestamps = get_speech_timestamps(
            wav_tensor,
            model,
            threshold=0.25,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
            speech_pad_ms=250,
            sampling_rate=sample_rate,
            window_size_samples=512,
        )

        if not speech_timestamps:
            logger.debug("VAD found no speech segments")
            return audio

        # Concatenate speech segments
        segments = []
        for ts in speech_timestamps:
            segments.append(audio[ts['start']:ts['end']])

        trimmed = np.concatenate(segments)
        removed_pct = (1 - len(trimmed) / len(audio)) * 100
        logger.info("VAD trimmed %.0f%% non-speech (%d -> %d samples)",
                    removed_pct, len(audio), len(trimmed))
        return trimmed

    except ImportError:
        logger.warning("silero-vad not installed, skipping VAD")
        return audio
    except Exception as e:
        logger.warning("VAD failed, using original audio: %s", e)
        return audio


def preprocess(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Run the full preprocessing pipeline.

    Order: noise reduction -> peak normalization -> VAD trim.
    """
    audio = reduce_noise(audio, sample_rate)
    audio = normalize_audio(audio)
    audio = vad_trim(audio, sample_rate)
    return audio
