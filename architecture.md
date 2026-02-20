# Voice IME for macOS: Architecture Design

**Date**: 2026-02-17
**Author**: voice-ime team architect
**Status**: Draft v1

---

## 1. Overview

A lightweight macOS menubar app that provides real-time voice-to-text input across all applications. The user presses a global hotkey, speaks, and transcribed text is injected into the currently focused text field. Runs entirely on-device using Apple Silicon GPU acceleration.

### Design Goals

| Goal | Target |
|------|--------|
| Total package size | < 2 GB (model + app + deps) |
| Memory usage (runtime) | < 1.5 GB |
| Transcription latency | < 1s for a 5-second utterance |
| Languages | Chinese, English, French (switchable) |
| Activation | Global hotkey (push-to-talk) |
| Privacy | 100% local, no network required |

---

## 2. ASR Engine

Based on the model research (see `research.md`), the chosen stack is:

### Model: Qwen3-ASR-0.6B (8-bit quantization)

| Property | Value |
|----------|-------|
| Parameters | 0.6B (180M encoder + Qwen3-0.6B LLM decoder) |
| Quantization | 8-bit (group 64) via MLX |
| Disk size | ~700 MB |
| Memory (runtime) | ~700 MB |
| WER (English, LibriSpeech clean) | 2.33% |
| WER (Chinese, AISHELL-2) | ~3.15% |
| Multilingual (30 langs, FLEURS) | 7.57% |
| Real-time factor | ~0.06 on M2 Max (16x faster than real-time) |
| Streaming | Yes (native dynamic attention windows) |

**Why Qwen3-ASR over Whisper**: Smaller (700MB vs 3GB+), comparable English accuracy, far better Chinese support (incl. dialects), native streaming, and a ready-made Swift MLX library.

**Fallback**: Qwen3-ASR-1.7B (8-bit, ~2.5 GB) if 0.6B accuracy is insufficient, especially for French.

### Inference Framework: qwen3-asr-swift (MLX Swift)

- Native Swift package, integrates directly into the macOS app
- Uses MLX Swift for Metal GPU acceleration on Apple Silicon
- Clean async API: `Qwen3ASRModel.fromPretrained()` / `.transcribe(audio:sampleRate:)`
- macOS 14+, Swift 5.9+
- Source: [github.com/ivan-digital/qwen3-asr-swift](https://github.com/ivan-digital/qwen3-asr-swift)

**Fallback inference options** (if qwen3-asr-swift has issues):
1. `antirez/qwen-asr` (pure C, BF16, zero deps) -- embed via C bridging header
2. `mlx-qwen3-asr` (Python) -- run as subprocess, communicate via Unix socket

---

## 3. App Architecture

### 3.1 App Type: Native Swift Menubar App

**Decision**: A native Swift/SwiftUI menubar app using `NSStatusItem`. Not a full InputMethodKit IME, not Electron/Tauri.

**Rationale**:
- **Not InputMethodKit**: IMKit requires registering as a system input method, complex lifecycle management, and is designed for character composition (CJK typing), not voice input. Overkill for this use case.
- **Not Electron/Tauri**: Unnecessary overhead (Electron adds 150-300MB), and we need tight macOS integration for audio capture, hotkeys, and text injection.
- **Not Python menubar**: Would require bundling a Python runtime (~100MB+), and microphone/accessibility permissions are harder to manage from Python scripts.
- **Native Swift wins**: Direct access to AVAudioEngine, CGEvent, NSStatusItem, and MLX Swift. Smallest binary size. Best Apple Silicon optimization.

### 3.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    VoiceIME.app                         │
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌───────────────────┐   │
│  │ Menubar  │   │ Overlay  │   │  Settings Window  │   │
│  │  Icon    │   │  Panel   │   │  (SwiftUI)        │   │
│  └────┬─────┘   └────┬─────┘   └───────────────────┘   │
│       │              │                                   │
│  ┌────┴──────────────┴──────────────────────────────┐   │
│  │              App Controller                       │   │
│  │  - State machine (idle/listening/transcribing)    │   │
│  │  - Language selection                             │   │
│  │  - Hotkey handler                                 │   │
│  └──────────────┬───────────────────────────────────┘   │
│                 │                                        │
│  ┌──────────────┴───────────────────────────────────┐   │
│  │           Audio Pipeline (3 stages)               │   │
│  │                                                   │   │
│  │  ┌─────────┐    ┌─────────┐    ┌──────────────┐  │   │
│  │  │ Audio   │───>│  VAD    │───>│  ASR Engine  │  │   │
│  │  │ Capture │    │ Filter  │    │ (Qwen3-ASR)  │  │   │
│  │  └─────────┘    └─────────┘    └──────┬───────┘  │   │
│  │                                       │          │   │
│  └───────────────────────────────────────┼──────────┘   │
│                                          │               │
│  ┌───────────────────────────────────────┴──────────┐   │
│  │           Text Injector                           │   │
│  │  - NSPasteboard write                             │   │
│  │  - CGEvent Cmd+V simulation                       │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 3.3 Threading Model

Three dedicated threads/queues to minimize latency:

| Thread | Responsibility | Queue Type |
|--------|---------------|------------|
| **Main thread** | UI updates, hotkey events | Main (serial) |
| **Audio thread** | AVAudioEngine tap callback, ring buffer fill | Real-time (high priority) |
| **Inference thread** | VAD processing + ASR inference | Background (serial, QoS: userInitiated) |

Data flow between threads uses lock-free ring buffers and `DispatchQueue`-based producer/consumer.

```swift
// Simplified data flow
AudioThread -> RingBuffer<Float> -> InferenceThread -> String -> MainThread -> TextInjector
```

---

## 4. Component Design

### 4.1 Audio Capture

**API**: `AVAudioEngine` with `AVAudioConverter` for sample rate conversion.

```
Hardware Mic (44.1/48kHz stereo)
    │
    ▼ [AVAudioEngine inputNode tap, bufferSize: 4096]
    │
AVAudioConverter
    │
    ▼ [16kHz, mono, Float32]
    │
Ring Buffer (capacity: 30s = 480,000 samples)
```

**Key implementation details**:
- Install tap on `audioEngine.inputNode` at hardware format
- Use `AVAudioConverter` to downsample to 16kHz mono Float32
- Write converted samples into a lock-free ring buffer
- Buffer capacity: 30 seconds (Qwen3-ASR's max context window)

**Permissions required**:
- `NSMicrophoneUsageDescription` in Info.plist
- `com.apple.security.device.audio-input` entitlement (for hardened runtime)

### 4.2 Voice Activity Detection (VAD)

**Library**: Silero VAD via ONNX Runtime for Swift.

Silero VAD provides the best balance of accuracy (87.7% TPR @ 5% FPR) and speed (<1ms per 32ms chunk). It's available as an ONNX model that can run via `onnxruntime-swift`.

**Configuration**:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk size | 512 samples (32ms at 16kHz) | Silero minimum |
| Speech pad | 200ms | Include context around speech |
| Min speech duration | 250ms | Prevent false triggers |
| Min silence duration | 500ms | Wait for pauses before sending |
| Threshold | 0.5 | Default, tunable |

**Role in push-to-talk mode**: Even in PTT mode, VAD is useful for:
1. Trimming leading/trailing silence from the recording
2. Providing visual feedback (waveform activity indicator)
3. Enabling an optional "auto-stop" feature (stop after sustained silence)

### 4.3 ASR Inference

**Library**: `qwen3-asr-swift` (MLX Swift).

**Flow**:

```
VAD-segmented audio chunk (Float32 array, 16kHz)
    │
    ▼
Qwen3ASRModel.transcribe(audio: chunk, sampleRate: 16000)
    │
    ▼
TranscriptionResult { text: String, language: String, timestamps: [...] }
```

**Language handling**:
- Qwen3-ASR supports automatic language detection (96.8% accuracy)
- User can also force a specific language via the menubar
- Language preference is passed to the model to improve accuracy

**Model lifecycle**:
1. **Cold start**: Load model from `~/Library/Application Support/VoiceIME/models/` on first activation (~2-3s)
2. **Warm**: Keep model resident in memory while app runs (~700MB)
3. **Unload**: Free model memory when app is quit or after extended idle (configurable)

**Model download**:
- On first launch, prompt user to download the model from HuggingFace
- Show progress bar during download
- Store in `~/Library/Application Support/VoiceIME/models/`
- Model identifier: `mlx-community/Qwen3-ASR-0.6B-8bit`

### 4.4 Text Injection

**Method**: NSPasteboard + CGEvent Cmd+V simulation.

This is the approach used by Superwhisper, VoiceInk, and most voice typing apps. It works reliably across virtually all macOS applications.

**Flow**:

```swift
func injectText(_ text: String) {
    // 1. Save current clipboard contents
    let savedClipboard = NSPasteboard.general.string(forType: .string)

    // 2. Write transcription to clipboard
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(text, forType: .string)

    // 3. Simulate Cmd+V
    let source = CGEventSource(stateID: .hidSystemState)
    let keyDown = CGEvent(keyboardEventSource: source, virtualKey: 0x09, keyDown: true)  // 'v'
    keyDown?.flags = .maskCommand
    keyDown?.post(tap: .cghidEventTap)

    let keyUp = CGEvent(keyboardEventSource: source, virtualKey: 0x09, keyDown: false)
    keyUp?.flags = .maskCommand
    keyUp?.post(tap: .cghidEventTap)

    // 4. Restore clipboard after brief delay
    DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
        NSPasteboard.general.clearContents()
        if let saved = savedClipboard {
            NSPasteboard.general.setString(saved, forType: .string)
        }
    }
}
```

**Permissions required**:
- **Accessibility permission**: Required for CGEvent posting. The app must be added to System Settings > Privacy & Security > Accessibility.
- On first launch, prompt the user with instructions to grant this permission.

**Why not other approaches**:

| Approach | Verdict | Reason |
|----------|---------|--------|
| InputMethodKit | Rejected | Designed for character composition, not voice; requires registering as system IME |
| Accessibility API (AXValue) | Rejected | Setting AXValue on text fields often doesn't reflect visually; unreliable |
| CGEvent key-by-key typing | Rejected | Slow for long text; can't handle Unicode/CJK well; misses modifier keys |
| **Pasteboard + Cmd+V** | **Chosen** | Works everywhere, handles Unicode/CJK, fast, well-tested by other apps |

### 4.5 Global Hotkey

**API**: `CGEvent.tapCreate` for global event monitoring.

**Default hotkey**: `Option + Space` (customizable in settings).

**Implementation approach**:

```swift
// Register a global event tap for key events
let eventMask = (1 << CGEventType.keyDown.rawValue) | (1 << CGEventType.keyUp.rawValue)
guard let tap = CGEvent.tapCreate(
    tap: .cgSessionEventTap,
    place: .headInsertEventTap,
    options: .defaultTap,
    eventsOfInterest: CGEventMask(eventMask),
    callback: hotkeyCallback,
    userInfo: nil
) else { return }

let runLoopSource = CFMachPortCreateRunLoopSource(nil, tap, 0)
CFRunLoopAddSource(CFRunLoopGetCurrent(), runLoopSource, .commonModes)
CGEvent.tapEnable(tap: tap, enable: true)
```

**Hotkey modes**:

| Mode | Behavior | When to use |
|------|----------|-------------|
| **Push-to-talk** (default) | Hold hotkey = record; release = transcribe + inject | Most predictable, lowest false-positive rate |
| **Toggle** | Press once = start; press again = stop + inject | Hands-free for longer dictation |

**Permission**: Same Accessibility permission as text injection (CGEvent requires it).

### 4.6 UI Components

#### Menubar Icon (`NSStatusItem`)

```
┌────────────────────────────┐
│ 🎤 VoiceIME          ▼    │  <- Menubar icon (SF Symbol: mic.fill)
├────────────────────────────┤
│ ● Listening...             │  <- Status indicator
│ ─────────────────────────  │
│ Language: English    ▸     │  <- Submenu: EN / 中文 / FR
│ Mode: Push-to-Talk  ▸     │  <- Submenu: PTT / Toggle
│ ─────────────────────────  │
│ Hotkey: ⌥ Space            │
│ ─────────────────────────  │
│ Settings...                │
│ About                      │
│ Quit                       │
└────────────────────────────┘
```

**Icon states**:
- Idle: `mic` (gray)
- Listening: `mic.fill` (blue, animated pulse)
- Transcribing: `text.bubble` (orange)
- Error: `mic.slash` (red)

#### Floating Overlay (`NSPanel`)

A small, always-on-top, translucent panel that appears near the cursor when recording.

```
┌──────────────────────────────┐
│  🎤 Listening...             │  <- During recording
│  ░░░░░░████░░░░░░            │  <- Audio level meter
│  "Hello, this is a test..."  │  <- Live preview (streamed)
└──────────────────────────────┘
```

**Properties**:
- `NSPanel` with `.nonactivatingPanel` style (doesn't steal focus)
- `.floating` level (always on top)
- Semi-transparent background (`NSVisualEffectView`)
- Positioned near the text cursor or center-bottom of screen
- Auto-hides after text injection completes
- Size: ~300x80pt, rounded corners

---

## 5. State Machine

```
                  ┌──────────┐
         ┌───────│   IDLE   │◄──────────────────┐
         │       └────┬─────┘                    │
         │            │ hotkey press              │ text injected
         │            ▼                           │ (or error/timeout)
         │       ┌──────────┐                     │
         │       │ LOADING  │ (first activation   │
         │       │  MODEL   │  only, ~2-3s)       │
         │       └────┬─────┘                     │
         │            │ model ready               │
         │            ▼                           │
         │       ┌──────────┐                     │
hotkey   │       │LISTENING │────────────┐        │
cancel   │       │(recording│            │        │
         │       └────┬─────┘            │        │
         │            │ hotkey release   │ timeout│
         │            │ (PTT mode)      │ (30s)  │
         │            │ or silence      │        │
         │            │ detected        │        │
         │            ▼                 │        │
         │       ┌──────────┐           │        │
         │       │TRANSCRIB-│           │        │
         └───────│  ING     │◄──────────┘        │
                 └────┬─────┘                     │
                      │ transcription done         │
                      ▼                           │
                 ┌──────────┐                     │
                 │INJECTING │─────────────────────┘
                 │  TEXT    │
                 └──────────┘
```

---

## 6. Language Switching

### Approach: Menubar Selection + Keyboard Shortcut

Languages are selectable via:
1. **Menubar dropdown**: Click the menubar icon > Language submenu
2. **Keyboard shortcut**: `Ctrl+Shift+L` cycles through: EN → 中文 → FR → EN
3. **Auto-detect**: A fourth "Auto" option lets Qwen3-ASR auto-detect the language

### Implementation

```swift
enum SpeechLanguage: String, CaseIterable {
    case auto = "auto"
    case english = "en"
    case chinese = "zh"
    case french = "fr"

    var displayName: String {
        switch self {
        case .auto: return "Auto-detect"
        case .english: return "English"
        case .chinese: return "中文"
        case .french: return "Français"
        }
    }
}
```

The selected language is passed to `Qwen3ASRModel.transcribe()` as a hint. Auto-detect uses Qwen3-ASR's built-in language identification (96.8% accuracy across 30 languages).

---

## 7. Permissions Summary

The app requires two system permissions:

| Permission | Why | How to request |
|------------|-----|----------------|
| **Microphone** | Audio capture | `AVCaptureDevice.requestAccess(for: .audio)` — system dialog auto-appears |
| **Accessibility** | Global hotkey + text injection via CGEvent | Must be manually enabled in System Settings. App shows setup instructions on first launch. |

**First-launch flow**:
1. App starts → requests microphone permission (system dialog)
2. App shows onboarding sheet: "VoiceIME needs Accessibility access to type text into other apps. Click 'Open Settings' to enable it."
3. App opens `System Settings > Privacy > Accessibility` directly via URL scheme
4. User toggles VoiceIME on
5. App detects the permission change and shows "Ready!" confirmation

---

## 8. Package Size Budget

| Component | Size |
|-----------|------|
| Qwen3-ASR-0.6B (8-bit MLX) | ~700 MB |
| Silero VAD (ONNX model) | ~2 MB |
| ONNX Runtime (Swift) | ~15 MB |
| MLX Swift framework | ~10 MB |
| App binary + SwiftUI resources | ~5 MB |
| **Total** | **~732 MB** |

Well under the 10GB target. Even with the 1.7B fallback model (~2.5GB), total would be ~2.5GB.

### Distribution

- Distribute as a `.dmg` containing `VoiceIME.app`
- Models are downloaded on first launch (not bundled in the DMG) to keep the initial download small (~30MB app-only DMG)
- Model download uses HuggingFace Hub with progress indication
- Ad-hoc code signing for initial development; Developer ID for distribution

---

## 9. Project Structure

```
VoiceIME/
├── VoiceIME.xcodeproj
├── VoiceIME/
│   ├── App/
│   │   ├── VoiceIMEApp.swift          # @main, NSApplicationDelegateAdaptor
│   │   ├── AppDelegate.swift          # NSStatusItem setup, app lifecycle
│   │   └── AppState.swift             # ObservableObject, state machine
│   ├── Audio/
│   │   ├── AudioCaptureManager.swift  # AVAudioEngine wrapper
│   │   ├── RingBuffer.swift           # Lock-free ring buffer
│   │   └── VADProcessor.swift         # Silero VAD via ONNX Runtime
│   ├── ASR/
│   │   ├── ASREngine.swift            # Protocol for ASR backends
│   │   ├── QwenASREngine.swift        # qwen3-asr-swift integration
│   │   └── ModelManager.swift         # Download, load, unload models
│   ├── Input/
│   │   ├── HotkeyManager.swift        # CGEvent tap for global hotkeys
│   │   └── TextInjector.swift         # NSPasteboard + CGEvent Cmd+V
│   ├── UI/
│   │   ├── MenuBarView.swift          # NSStatusItem menu
│   │   ├── OverlayPanel.swift         # Floating NSPanel
│   │   ├── OverlayView.swift          # SwiftUI overlay content
│   │   ├── SettingsView.swift         # Preferences window
│   │   └── OnboardingView.swift       # First-launch permission setup
│   └── Utilities/
│       ├── Permissions.swift          # Mic + Accessibility permission checks
│       └── Settings.swift             # UserDefaults wrapper
├── Resources/
│   ├── Assets.xcassets                # App icon, SF Symbols
│   ├── silero_vad.onnx                # Bundled VAD model (~2MB)
│   └── Info.plist                     # NSMicrophoneUsageDescription
├── VoiceIME.entitlements              # audio-input, app-sandbox exceptions
├── Package.swift                      # SPM dependencies
└── Tests/
    ├── AudioCaptureTests.swift
    ├── VADProcessorTests.swift
    └── TextInjectorTests.swift
```

### Dependencies (Swift Package Manager)

| Package | Purpose | Source |
|---------|---------|--------|
| `qwen3-asr-swift` | ASR model inference | github.com/ivan-digital/qwen3-asr-swift |
| `mlx-swift` | Metal ML framework (transitive) | github.com/ml-explore/mlx-swift |
| `onnxruntime-swift` | Silero VAD execution | github.com/nicklama/onnxruntime-swift |
| `KeyboardShortcuts` | User-configurable hotkeys | github.com/sindresorhus/KeyboardShortcuts |

---

## 10. Latency Analysis

### End-to-end latency breakdown (push-to-talk mode)

| Phase | Duration | Notes |
|-------|----------|-------|
| Hotkey release detected | ~5ms | CGEvent callback |
| Audio buffer flush | ~50ms | Drain remaining audio from ring buffer |
| VAD trimming | ~10ms | Trim silence from start/end |
| ASR inference (5s audio) | ~300ms | Qwen3-ASR-0.6B 8-bit on Apple Silicon |
| Text injection | ~50ms | Pasteboard write + CGEvent Cmd+V |
| **Total** | **~415ms** | From releasing hotkey to text appearing |

### For longer utterances (15s audio)

| Phase | Duration |
|-------|----------|
| ASR inference | ~900ms |
| **Total** | **~1.0s** |

This meets the <1s target for typical utterances (5-10s).

---

## 11. Future Enhancements (Out of Scope for v1)

1. **Streaming preview**: Show partial transcription while still speaking (requires streaming Qwen3-ASR integration)
2. **LLM post-processing**: Clean up filler words, fix grammar using a small LLM (like Qwen3-0.6B text model)
3. **Custom vocabulary**: User-defined terms to improve domain-specific accuracy
4. **Multi-monitor support**: Position overlay on the active monitor
5. **iOS companion app**: Share the model and code via qwen3-asr-swift (supports iOS 17+)
6. **Auto-detect mode**: Continuous listening with VAD-based auto-segmentation (no hotkey needed)
7. **Speaker diarization**: Tag different speakers in multi-person dictation

---

## 12. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| qwen3-asr-swift is immature / has bugs | Medium | High | Fallback to antirez/qwen-asr (C) via bridging header, or mlx-qwen3-asr (Python subprocess) |
| French accuracy insufficient | Low-Medium | Medium | Test early; fallback to 1.7B model or Whisper for French-only |
| Pasteboard restore race condition | Low | Low | Add configurable delay; document known edge cases |
| macOS permission UX is confusing | Medium | Medium | Detailed onboarding flow with screenshots; auto-detect permission changes |
| Model download fails / slow | Low | Medium | Resume support; mirror URLs; allow manual model placement |
| MLX Swift API changes | Low | Medium | Pin dependency version in Package.swift |

---

## Appendix A: Alternatives Considered

### A.1 Tech Stack Options Evaluated

| Stack | Pros | Cons | Verdict |
|-------|------|------|---------|
| **Native Swift** (chosen) | Smallest binary, best macOS integration, direct MLX Swift access, best performance | Swift-only, smaller open-source community | **Chosen** |
| Tauri (Rust + Web) | Cross-platform, good for UI | Extra layer, larger binary, harder to integrate MLX | Rejected |
| Electron | Rich UI ecosystem | 150-300MB overhead, high memory, no MLX integration | Rejected |
| Python + rumps | Fastest to prototype, direct access to ML libraries | Requires bundling Python runtime, harder permissions, sluggish UI | Rejected |
| Python backend + Swift frontend | Best of both worlds | IPC complexity, two runtimes, harder to distribute | Rejected for v1 |

### A.2 Text Injection Alternatives

| Method | Works in all apps? | Unicode/CJK? | Speed | Permissions | Verdict |
|--------|-------------------|---------------|-------|-------------|---------|
| **Pasteboard + Cmd+V** (chosen) | Yes | Yes | Fast | Accessibility | **Chosen** |
| CGEvent key-by-key | Most | Poor | Slow | Accessibility | Rejected |
| Accessibility API (AXValue) | Partial | Yes | Fast | Accessibility | Rejected (unreliable) |
| InputMethodKit | Yes | Yes | Fast | None extra | Rejected (wrong abstraction) |

### A.3 Existing Open-Source Reference Projects

| Project | Stack | Insights |
|---------|-------|----------|
| [whispr](https://github.com/dbpprt/whispr) | Rust + whisper.cpp | Menubar architecture, model download flow |
| [Handy](https://github.com/cjpais/Handy) | Tauri + Rust + React | VAD integration, multi-model support |
| [whisper-writer](https://github.com/savbell/whisper-writer) | Python + pynput | Push-to-talk UX, pynput for hotkeys |
| [whisper-mac](https://github.com/Explosion-Scratch/whisper-mac) | Bun + Silero VAD | Plugin system, multi-engine support |
| [Superwhisper](https://superwhisper.com/) | Native Swift + whisper.cpp | Commercial reference, polished UX, mode-based system |
