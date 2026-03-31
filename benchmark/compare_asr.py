#!/usr/bin/env python3
"""Compare Qwen3-ASR-0.6B vs Whisper large-v3-turbo on test audio."""

import os
import sys
import time
import wave
import tempfile

import numpy as np

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_CASES = [
    {"audio": "test_en.wav", "expected_keywords": ["thinking", "meet", "wednesday", "budget"], "language": "English", "label": "English (medium)"},
    {"audio": "test_zh.wav", "expected_keywords": ["项目", "下周五", "完成"], "language": "Chinese", "label": "Chinese (medium)"},
    {"audio": "test_short.wav", "expected_keywords": ["hello", "world"], "language": "English", "label": "English (short)"},
]

LANG_TO_CODE = {"English": "en", "Chinese": "zh", "French": "fr"}


def load_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def keyword_accuracy(text: str, keywords: list[str]) -> float:
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords) if keywords else 1.0


def save_temp_wav(audio: np.ndarray, sr: int = 16000) -> str:
    """Save numpy audio to a temp WAV file for mlx-whisper."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((audio * 32768).astype(np.int16).tobytes())
    return tmp.name


def run_qwen(test_cases):
    """Run Qwen3-ASR-0.6B on all test cases."""
    sys.path.insert(0, os.path.dirname(BENCHMARK_DIR))
    from src.asr_engine import ASREngine

    print("Loading Qwen3-ASR-0.6B...")
    asr = ASREngine()
    asr.load_model()

    results = []
    for tc in test_cases:
        audio = load_wav(os.path.join(BENCHMARK_DIR, tc["audio"]))
        asr.set_language(tc["language"])
        t0 = time.monotonic()
        text, lang = asr.transcribe(audio)
        ms = (time.monotonic() - t0) * 1000
        acc = keyword_accuracy(text, tc["expected_keywords"])
        results.append({"label": tc["label"], "text": text, "ms": ms, "acc": acc})

    asr.unload_model()
    return results


def run_whisper(test_cases, model_name="mlx-community/whisper-large-v3-turbo"):
    """Run Whisper via mlx-whisper on all test cases."""
    import mlx_whisper

    print(f"Loading {model_name}...")
    # Warmup with a tiny audio to trigger model download + JIT
    warmup_path = save_temp_wav(np.zeros(16000, dtype=np.float32))
    mlx_whisper.transcribe(warmup_path, path_or_hf_repo=model_name)
    os.unlink(warmup_path)
    print("Whisper warmed up.")

    results = []
    for tc in test_cases:
        audio = load_wav(os.path.join(BENCHMARK_DIR, tc["audio"]))
        tmp_path = save_temp_wav(audio)
        lang_code = LANG_TO_CODE.get(tc["language"])

        t0 = time.monotonic()
        result = mlx_whisper.transcribe(
            tmp_path,
            path_or_hf_repo=model_name,
            language=lang_code,
        )
        ms = (time.monotonic() - t0) * 1000
        text = result["text"].strip()
        acc = keyword_accuracy(text, tc["expected_keywords"])
        results.append({"label": tc["label"], "text": text, "ms": ms, "acc": acc})
        os.unlink(tmp_path)

    return results


def print_results(name, results):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    total_ms = 0
    total_acc = 0
    for r in results:
        print(f"  [{r['label']}] {r['ms']:.0f}ms  acc={r['acc']:.0%}")
        print(f"    -> {r['text']!r}")
        total_ms += r["ms"]
        total_acc += r["acc"]
    n = len(results)
    print(f"  ---")
    print(f"  Avg: {total_ms/n:.0f}ms, {total_acc/n:.0%} accuracy")


if __name__ == "__main__":
    qwen_results = run_qwen(TEST_CASES)
    print_results("Qwen3-ASR-0.6B", qwen_results)

    whisper_results = run_whisper(TEST_CASES)
    print_results("Whisper Large V3 Turbo (mlx)", whisper_results)

    print(f"\n{'=' * 60}")
    print("  HEAD-TO-HEAD")
    print(f"{'=' * 60}")
    for q, w in zip(qwen_results, whisper_results):
        print(f"  [{q['label']}]")
        print(f"    Qwen:    {q['ms']:6.0f}ms  {q['acc']:.0%}  {q['text']!r}")
        print(f"    Whisper: {w['ms']:6.0f}ms  {w['acc']:.0%}  {w['text']!r}")
