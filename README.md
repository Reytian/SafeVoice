# SafeVoice - Voice Input Method for macOS

SafeVoice is a macOS menubar app that provides on-device speech recognition and text injection. Speak into your microphone and SafeVoice types the transcribed text into any active application.

Powered by [Qwen3-ASR-0.6B](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit) running locally on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). No cloud services, no API keys, no data leaves your machine.

## Features

- On-device speech recognition (no internet required after model download)
- Real-time streaming transcription with live preview
- Supports Chinese, English, and French
- Global hotkey activation (Option+Space)
- Push-to-talk and toggle modes
- Floating overlay showing transcription progress
- Text injection into any application via clipboard
- Menubar app with language and mode selection

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~700 MB disk space for the ASR model

## Setup

### 1. Clone and create virtual environment

```bash
cd ~/Documents/voice-ime
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

This downloads the 8-bit quantized Qwen3-ASR-0.6B model (~700 MB) from HuggingFace Hub. The download only happens once; subsequent runs use the cached model.

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
| **Option + Space** | Start/stop voice input |
| **Ctrl + Space** | Cycle language (Chinese → English → French) |

### Modes

- **Push-to-talk** (default, recommended): Hold Option+Space to record, release to transcribe and inject.
- **Toggle**: Press Option+Space to start recording, press again to stop and transcribe.

Switch modes from the menubar dropdown.

### Languages

Select a language from the menubar dropdown or cycle with Ctrl+Space:

- **Chinese** (中文)
- **English**
- **French** (Français)

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
