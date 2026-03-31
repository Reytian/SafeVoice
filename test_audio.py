#!/usr/bin/env python3
"""Quick diagnostic: record 3 seconds and report audio levels per channel."""

import sys
sys.path.insert(0, ".")

import numpy as np
import sounddevice as sd

print("=== Audio Device Diagnostic ===\n")

# Show default device
default_in = sd.default.device[0]
info = sd.query_devices(default_in)
print(f"Default input device [{default_in}]: {info['name']}")
print(f"  Channels: {info['max_input_channels']}")
print(f"  Default SR: {info['default_samplerate']}")
print()

# Record 3 seconds at native channel count
sr = 16000
channels = info['max_input_channels']
duration = 3.0

print(f"Recording {duration}s at {sr} Hz, {channels} channel(s)...")
print(">>> Speak now! <<<")
audio = sd.rec(int(sr * duration), samplerate=sr, channels=channels, dtype='float32')
sd.wait()
print("Done.\n")

# Report per-channel stats
for ch in range(channels):
    data = audio[:, ch] if channels > 1 else audio[:, 0]
    rms = float(np.sqrt(np.mean(data ** 2)))
    peak = float(np.max(np.abs(data)))
    nonzero = int(np.count_nonzero(data))
    print(f"Channel {ch}: RMS={rms:.6f}  Peak={peak:.6f}  NonZero={nonzero}/{len(data)}")

print()

# Also try with channels=1 (what the app does)
print(f"Recording {duration}s at {sr} Hz, 1 channel (app mode)...")
print(">>> Speak now! <<<")
audio_mono = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='float32')
sd.wait()
print("Done.\n")

data = audio_mono[:, 0]
rms = float(np.sqrt(np.mean(data ** 2)))
peak = float(np.max(np.abs(data)))
nonzero = int(np.count_nonzero(data))
print(f"Mono (ch=1): RMS={rms:.6f}  Peak={peak:.6f}  NonZero={nonzero}/{len(data)}")
