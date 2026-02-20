#!/usr/bin/env python3
"""
Test audio capture on macOS using the sounddevice library.

This script:
1. Lists available audio input devices.
2. Captures 3 seconds of audio from the default microphone at 16000 Hz (mono).
3. Saves the recording to a WAV file.
4. Prints information about the captured audio.
"""

import sys
import wave
import struct
from pathlib import Path

import numpy as np
import sounddevice as sd

# Configuration
SAMPLE_RATE = 16000  # Hz
DURATION = 3  # seconds
CHANNELS = 1  # mono
OUTPUT_PATH = Path(__file__).parent / "test_recording.wav"


def list_input_devices():
    """List all available audio input devices."""
    print("=" * 60)
    print("Available Audio Input Devices")
    print("=" * 60)
    devices = sd.query_devices()
    input_devices = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            input_devices.append((i, dev))
            marker = " <-- DEFAULT" if i == sd.default.device[0] else ""
            print(
                f"  [{i}] {dev['name']}"
                f"  (inputs: {dev['max_input_channels']},"
                f" rate: {dev['default_samplerate']:.0f} Hz)"
                f"{marker}"
            )
    if not input_devices:
        print("  No input devices found!")
    print()
    return input_devices


def capture_audio(duration, sample_rate, channels):
    """Capture audio from the default microphone."""
    print(f"Recording {duration}s of audio at {sample_rate} Hz (mono)...")
    print("Speak now!")
    print()

    audio_data = sd.rec(
        frames=int(duration * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
    )
    sd.wait()  # block until recording is complete

    print("Recording complete.")
    return audio_data


def save_wav(filepath, audio_data, sample_rate):
    """Save audio data (float32 numpy array) to a 16-bit PCM WAV file."""
    # Convert float32 [-1.0, 1.0] to int16
    audio_int16 = np.clip(audio_data, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)

    with wave.open(str(filepath), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    print(f"Saved to: {filepath}")


def print_audio_info(audio_data, sample_rate):
    """Print summary information about the captured audio."""
    num_samples = audio_data.shape[0]
    duration = num_samples / sample_rate
    peak = np.max(np.abs(audio_data))
    rms = np.sqrt(np.mean(audio_data ** 2))

    print()
    print("=" * 60)
    print("Captured Audio Info")
    print("=" * 60)
    print(f"  Duration       : {duration:.2f} s")
    print(f"  Sample rate    : {sample_rate} Hz")
    print(f"  Channels       : {CHANNELS} (mono)")
    print(f"  Total samples  : {num_samples}")
    print(f"  Dtype          : {audio_data.dtype}")
    print(f"  Peak amplitude : {peak:.6f}")
    print(f"  RMS amplitude  : {rms:.6f}")
    print(f"  Peak dBFS      : {20 * np.log10(peak + 1e-10):.1f} dB")
    print(f"  RMS  dBFS      : {20 * np.log10(rms + 1e-10):.1f} dB")
    print(f"  Output file    : {OUTPUT_PATH}")
    file_size = OUTPUT_PATH.stat().st_size if OUTPUT_PATH.exists() else 0
    print(f"  File size      : {file_size:,} bytes")
    print()


def main():
    print()
    # Step 1: List input devices
    input_devices = list_input_devices()
    if not input_devices:
        print("No audio input devices found. Exiting.")
        sys.exit(1)

    # Step 2: Capture audio
    audio_data = capture_audio(DURATION, SAMPLE_RATE, CHANNELS)

    # Step 3: Save to WAV
    save_wav(OUTPUT_PATH, audio_data, SAMPLE_RATE)

    # Step 4: Print info
    print_audio_info(audio_data, SAMPLE_RATE)

    print("Done.")


if __name__ == "__main__":
    main()
