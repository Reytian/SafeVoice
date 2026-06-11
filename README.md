# SafeVoice - Voice Input Method for macOS

SafeVoice is a macOS menubar app that provides on-device speech recognition and text injection. Speak into your microphone and SafeVoice types the transcribed text into any active application.

Powered by [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) running locally on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). Speech recognition is fully on-device. Optional LLM text cleanup runs locally too (Ollama or native MLX); cloud LLM providers exist as an explicit opt-in and send transcript text to the configured provider.

## Features

- On-device speech recognition (no internet required after model download)
- 14 languages plus auto-detect (Chinese, English, French, Japanese, Korean, German, Spanish, and more)
- Global hotkey activation (default: Option+Space, configurable in Settings)
- Push-to-talk and toggle modes
- Floating overlay with recording level, processing status, and paste confirmation
- Optional LLM cleanup (filler-word removal, punctuation) via Ollama, native MLX, or cloud APIs
- Processing modes (Quick, Formal Writing, English Translation) switchable from the menubar
- Text injection into any application via clipboard paste
- Usage dashboard and local transcription history

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~700 MB disk space for the ASR model

## Setup

### 1. Clone and create virtual environment

```bash
cd ~/Developer/voice-ime
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the ASR model

```bash
python scripts/setup_model.py
```

This downloads the Qwen3-ASR-0.6B model (~1.2 GB download, ~1.8 GB on disk) from HuggingFace Hub. The download only happens once; subsequent runs use the cached model.

To see other available model variants:

```bash
python scripts/setup_model.py --list
```

### 4. Grant Accessibility permission

SafeVoice needs Accessibility access to inject text into other applications.

1. Open **System Settings > Privacy & Security > Accessibility**
2. Click the **+** button
3. Add your terminal app (e.g., Terminal.app, iTerm, or the Python executable)
4. Enable the toggle

### 5. Grant Microphone permission

On first run, macOS will prompt you to allow microphone access. Click **Allow**.

## Usage

```bash
source .venv/bin/activate
python run.py
```

### Hotkeys

| Hotkey | Action |
|---|---|
| **Option + Space** (default) | Start/stop voice input |

The activation hotkey is configurable in Settings > Hotkeys (including modifier-only hotkeys such as holding Left Option).

### Modes

- **Push-to-talk** (default, recommended): Hold the hotkey to record, release to transcribe and inject.
- **Toggle**: Press the hotkey to start recording, press again to stop and transcribe.

Switch input modes and processing modes (Quick, Formal Writing, English Translation) from the menubar dropdown.

### Languages

Select a language from the menubar dropdown or in Settings > Languages. Auto-detect handles mixed-language (code-switching) dictation.

## Project Structure

```
voice-ime/
├── run.py                  # Entry point
├── requirements.txt        # Python dependencies
├── scripts/
│   └── setup_model.py      # Model download script
└── src/
    ├── __init__.py
    ├── app.py              # Main application (rumps menubar app)
    ├── asr_engine.py       # ASR engine (mlx-qwen3-asr wrapper)
    ├── audio_capture.py    # Microphone audio capture (sounddevice)
    ├── hotkey_manager.py   # Global hotkey listener (pynput)
    ├── overlay.py          # Floating overlay panel (PyObjC)
    └── text_injector.py    # Text injection (NSPasteboard + Cmd+V)
```

## Troubleshooting

**"Model loading..." stays in menu**: The model takes a few seconds to load on first use. Subsequent launches are faster due to OS disk caching.

**Text not injected**: Ensure Accessibility permission is granted for your terminal/Python executable in System Settings.

**No audio captured**: Ensure Microphone permission is granted. Check that your microphone is selected as the default input device in System Settings > Sound.

**Import errors**: Make sure you activated the virtual environment (`source .venv/bin/activate`) before running.
