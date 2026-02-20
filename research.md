# Voice IME Research: Qwen Audio/Speech Models for Local ASR on macOS

**Date**: 2026-02-17
**Researcher**: voice-ime team researcher

---

## Executive Summary

For building a local voice input method on macOS supporting Chinese, English, and French, **Qwen3-ASR-0.6B** is the top recommendation. It offers state-of-the-art ASR quality in a tiny footprint (~400 MB at 4-bit quantization, ~1.2 GB at fp16), runs natively on Apple Silicon via MLX, and supports all three target languages plus 27 more. A mature MLX ecosystem (Python library, Swift library, pure C implementation) already exists, making integration straightforward.

---

## 1. Qwen Audio/Speech Model Family Overview

### 1.1 Qwen3-ASR (Released 2026-01-29) -- RECOMMENDED

The newest and most relevant models for pure ASR tasks.

| Model | Parameters | Encoder | Decoder | fp16 Size | 4-bit Size | 8-bit Size |
|-------|-----------|---------|---------|-----------|------------|------------|
| Qwen3-ASR-0.6B | 0.6B (180M encoder + Qwen3-0.6B LLM) | 18 layers, hidden 896 | 28 layers | ~1.2 GB | ~400 MB | ~700 MB |
| Qwen3-ASR-1.7B | 1.7B (300M encoder + Qwen3-1.7B LLM) | 24 layers, hidden 1024 | 28 layers | ~3.4 GB | ~1.2 GB | ~2.5 GB |

**Architecture**: Audio Transformer (AuT) encoder + Qwen3 LLM decoder. The AuT encoder performs 8x downsampling on 128-dim Fbank features, producing a 12.5Hz token rate. Dynamic flash attention window (1s-8s) enables both streaming and offline inference.

**Languages**: 30 languages + 22 Chinese dialects (52 total):
- **Core languages**: Chinese (zh), English (en), French (fr), German (de), Spanish (es), Italian (it), Japanese (ja), Korean (ko), Russian (ru), Arabic (ar), Portuguese (pt), and more
- **Chinese dialects**: Sichuan, Cantonese, Dongbei, Fujian, Henan, Hubei, Hunan, Shandong, Shaanxi, Wu, Minnan, and more
- **Language ID accuracy**: 96.8% average (0.6B model)

### 1.2 Qwen2.5-Omni (Released 2025-03-26)

End-to-end multimodal model (text + audio + image + video input, text + speech output).

| Model | Parameters | fp16 Size | Notes |
|-------|-----------|-----------|-------|
| Qwen2.5-Omni-7B | ~7B | ~14 GB | Full multimodal, too large |
| Qwen2.5-Omni-3B | ~3B | ~6 GB | Lighter, but still overkill for ASR-only |

- Overkill for ASR-only use case -- includes vision, video, and speech generation capabilities
- MLX versions exist (mlx-community/Qwen2.5-Omni-3B-4bit, -8bit)
- Higher memory requirements than dedicated ASR models

### 1.3 Qwen2-Audio (Released 2024)

Predecessor audio-language model.

| Model | Parameters | fp16 Size | GGUF Q4 Size |
|-------|-----------|-----------|--------------|
| Qwen2-Audio-7B | ~7B | ~14 GB | ~4.6 GB |

- Supports 8+ languages including Chinese, English, French
- GGUF versions exist (second-state/Qwen2-Audio-7B-Instruct-GGUF)
- **Not recommended**: Superseded by Qwen3-ASR with better accuracy and much smaller footprint
- llama.cpp audio model support is limited/experimental

---

## 2. ASR Quality Benchmarks

### 2.1 English Benchmarks (WER % -- lower is better)

| Model | LibriSpeech Clean | LibriSpeech Other | GigaSpeech | CommonVoice-en |
|-------|-------------------|-------------------|------------|----------------|
| Qwen3-ASR-0.6B | 2.11 | 4.55 | 8.88 | 9.92 |
| Qwen3-ASR-1.7B | **1.63** | **3.38** | **8.45** | **7.39** |
| Whisper Large V3 | 1.6 | 3.1 | - | - |
| Whisper Large V3 Turbo | ~2.0 | ~3.5 | - | - |

### 2.2 Chinese Benchmarks (WER %)

| Model | WenetSpeech (net) | WenetSpeech (meeting) | AISHELL-2 |
|-------|--------------------|-----------------------|-----------|
| Qwen3-ASR-0.6B | 5.97 | 6.88 | 3.15 |
| Qwen3-ASR-1.7B | **4.97** | **5.88** | **2.71** |

### 2.3 Multilingual Benchmarks (WER %)

| Model | FLEURS (30 languages) | News-Multilingual (15 languages) |
|-------|------------------------|----------------------------------|
| Qwen3-ASR-0.6B | 7.57 | 17.39 |
| Qwen3-ASR-1.7B | **4.90** | **12.80** |

### 2.4 Quantization Impact on Quality (0.6B, LibriSpeech)

| Quantization | WER (test-clean) | WER (test-other) | Speed vs fp16 |
|-------------|------------------|------------------|---------------|
| fp16 (baseline) | 2.29% | baseline | 1x |
| 8-bit (group 64) | 2.33% (+0.04) | -0.05 improvement | 3.11x faster |
| 4-bit (group 64) | 2.72% (+0.43) | +1.38 | 4.68x faster |

**Key finding**: 8-bit quantization has virtually no quality loss and is 3x faster. 4-bit has minor degradation but is 4.7x faster -- both are excellent for an IME use case.

---

## 3. Local Inference Frameworks for Apple Silicon

### 3.1 MLX-Based Solutions (RECOMMENDED)

#### a) mlx-qwen3-asr (Python, by moona3k)
- **Install**: `pip install mlx-qwen3-asr`
- **Features**: Native MLX reimplementation, no PyTorch dependency, 4-bit/8-bit quantization, word timestamps, speaker diarization
- **Performance on M4 Pro (0.6B fp16)**:
  - Short clips (~2.5s): 0.46s inference
  - 10-second audio: 0.83s inference
  - Real-time factor: 0.08x (12.5x faster than real-time)
- **462 tests** with committed benchmark artifacts

#### b) qwen3-asr-swift (Swift, by ivan-digital)
- **Best for native macOS app integration**
- **Uses MLX Swift** for Metal GPU acceleration
- **Performance on M2 Max**:
  - 0.6B model: ~0.6s for 10s audio (RTF ~0.06)
  - 1.7B model: ~1.1s for 10s audio (RTF ~0.11)
- **Model sizes on disk**:
  - 0.6B 4-bit: ~400 MB
  - 1.7B 8-bit: ~2.5 GB
- **API**: Clean Swift API (`Qwen3ASRModel.fromPretrained()`, `.transcribe(audio:sampleRate:)`)
- **Platform**: macOS 14+, iOS 17+, Swift 5.9+
- Also includes TTS support (Qwen3-TTS)

#### c) mlx-audio (Python, by Blaizzy)
- **Install**: `pip install mlx-audio`
- Unified library for STT + TTS on MLX
- Supports Qwen3-ASR models (e.g., `mlx-community/Qwen3-ASR-0.6B-8bit`)
- Part of a broader MLX audio ecosystem

### 3.2 Pure C Implementation

#### antirez/qwen-asr
- **Pure C** with zero dependencies beyond BLAS (Accelerate on macOS)
- **Build**: `make blas`
- **Performance on M3 Max**:
  - 0.6B on 11s audio: 1.4s (~7.99x realtime)
  - 0.6B on 45s audio (segmented): 3.4s (~13.38x realtime)
  - Streaming mode: ~4.69x realtime with cache
- **BF16 weights**, mmap'd from safetensors (near-instant loading)
- **Great for**: Embedding in a native macOS app without Python dependency
- **Limitation**: No quantization support yet (BF16 only)

### 3.3 Transformers / vLLM (Not Recommended for This Use Case)

- Official Qwen3-ASR supports transformers and vLLM backends
- Requires PyTorch, CUDA-oriented
- vLLM does not support Apple Silicon natively
- Transformers can run on MPS but with suboptimal performance

### 3.4 llama.cpp (Limited Support)

- Qwen text models are well-supported in GGUF format
- **Audio models are NOT well-supported** -- Qwen2-Audio and Qwen2.5-Omni lack proper GGUF conversion
- Not recommended for this project

---

## 4. Memory Budget Analysis

Target: fit under ~6-7 GB total (leaving room for app code on a 16 GB Mac).

| Configuration | Model Memory | Feasibility |
|---------------|-------------|-------------|
| Qwen3-ASR-0.6B fp16 | ~1.2 GB | Excellent -- leaves 5+ GB |
| Qwen3-ASR-0.6B 8-bit | ~700 MB | Excellent |
| Qwen3-ASR-0.6B 4-bit | ~400 MB | Excellent |
| Qwen3-ASR-1.7B fp16 | ~3.4 GB | Good -- leaves 3+ GB |
| Qwen3-ASR-1.7B 8-bit | ~2.5 GB | Good |
| Qwen3-ASR-1.7B 4-bit | ~1.2 GB | Excellent |
| Qwen2.5-Omni-3B 4-bit | ~2-3 GB | Marginal, overkill |
| Qwen2-Audio-7B Q4 | ~4.6 GB | Tight, not recommended |

**All Qwen3-ASR configurations fit comfortably within the memory budget.**

---

## 5. Recommendations

### Primary Recommendation: Qwen3-ASR-0.6B (8-bit quantization via MLX)

**Why**:
1. **Tiny footprint**: ~700 MB memory, ~400-700 MB disk
2. **Excellent accuracy**: 2.33% WER on LibriSpeech clean (virtually identical to fp16)
3. **Fast**: 12x+ faster than real-time on Apple Silicon
4. **All target languages**: Chinese, English, French all supported, plus 27 more
5. **Mature MLX ecosystem**: Multiple ready-to-use libraries
6. **Streaming capable**: Dynamic attention windows support real-time transcription

### Integration Approach Options

| Approach | Pros | Cons |
|----------|------|------|
| **qwen3-asr-swift (MLX Swift)** | Native Swift, best for macOS IME, clean API, includes TTS | Newer project, smaller community |
| **mlx-qwen3-asr (Python)** | Most mature, 462 tests, best quantization support | Requires Python runtime |
| **antirez/qwen-asr (C)** | Zero dependencies, easy to embed | No quantization, BF16 only |

### Recommended Integration: qwen3-asr-swift

For a macOS IME (Input Method Editor), a **Swift-native** solution is ideal:
- Input Methods on macOS are typically written in Swift/Objective-C
- MLX Swift provides native Metal GPU acceleration
- No Python runtime overhead
- Clean async/await API fits modern macOS app patterns
- ~400 MB on disk with 4-bit quantization
- RTF ~0.06 means transcription completes in 600ms for 10 seconds of speech

### Fallback: Qwen3-ASR-1.7B (8-bit)

If the 0.6B model's accuracy is insufficient for the user's needs (especially for multilingual FLEURS benchmark: 7.57% vs 4.90%), the 1.7B model at 8-bit (~2.5 GB) still fits easily within the memory budget and provides a 28% relative improvement in multilingual accuracy.

---

## 6. Comparison with Whisper

| Feature | Qwen3-ASR-0.6B | Whisper Large V3 Turbo | Whisper Large V3 |
|---------|-----------------|------------------------|-------------------|
| Parameters | 0.6B | 809M | 1.55B |
| Memory (quantized) | ~400 MB (4-bit) | ~3 GB (fp16) | ~6 GB (fp16) |
| English WER | 2.11% | ~2.0% | 1.6% |
| Chinese WER | 3.15% (AISHELL-2) | Higher | Higher |
| Languages | 52 (30 + 22 dialects) | 99+ | 99+ |
| Streaming | Yes (native) | No (30s chunks) | No (30s chunks) |
| MLX support | Native (multiple libs) | whisper.cpp / mlx-whisper | whisper.cpp / mlx-whisper |
| Chinese dialects | 22 supported | Not supported | Not supported |

**Qwen3-ASR wins for this use case** due to:
- Much smaller footprint at comparable English accuracy
- Superior Chinese (Mandarin + dialect) support
- Native streaming support (critical for IME real-time input)
- French support confirmed

Whisper has broader language coverage (99+) but is larger and lacks streaming.

---

## 7. Available Pre-Quantized Models on HuggingFace

| Model | Format | Source |
|-------|--------|--------|
| Qwen/Qwen3-ASR-0.6B | safetensors (fp16/bf16) | Official |
| Qwen/Qwen3-ASR-1.7B | safetensors (fp16/bf16) | Official |
| mlx-community/Qwen3-ASR-0.6B-8bit | MLX 8-bit | Community |
| mlx-community/Qwen3-ASR-0.6B-4bit | MLX 4-bit | Community |
| mlx-community/Qwen2.5-Omni-3B-4bit | MLX 4-bit | Community |
| mlx-community/Qwen2.5-Omni-3B-8bit | MLX 8-bit | Community |
| second-state/Qwen2-Audio-7B-Instruct-GGUF | GGUF (various) | Community |

---

## 8. Key Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| French accuracy unknown | May need fallback | Test on French audio; FLEURS includes French; fallback to Whisper for French only |
| qwen3-asr-swift is new | Bugs possible | antirez/qwen-asr (C) and mlx-qwen3-asr (Python) as mature fallbacks |
| MLX Swift API may change | Breaking changes | Pin MLX Swift version |
| IME integration complexity | macOS Input Method APIs are complex | Use existing open-source IME projects as templates |

---

## Sources

- [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-ASR Technical Report (arXiv)](https://arxiv.org/abs/2601.21337)
- [Qwen3-ASR-0.6B on HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Qwen3-ASR-1.7B on HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [mlx-qwen3-asr (Python MLX)](https://github.com/moona3k/mlx-qwen3-asr)
- [qwen3-asr-swift (Swift MLX)](https://github.com/ivan-digital/qwen3-asr-swift)
- [antirez/qwen-asr (C implementation)](https://github.com/antirez/qwen-asr)
- [mlx-audio library](https://github.com/Blaizzy/mlx-audio)
- [Qwen2.5-Omni GitHub](https://github.com/QwenLM/Qwen2.5-Omni)
- [Qwen2-Audio GitHub](https://github.com/QwenLM/Qwen2-Audio)
- [Best Open Source STT Models 2026 (Northflank)](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)
- [Qwen MLX Documentation](https://qwen.readthedocs.io/en/latest/run_locally/mlx-lm.html)
