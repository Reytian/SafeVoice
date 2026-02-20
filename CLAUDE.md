# CLAUDE.md — SafeVoice

Project-level instructions for Claude Code working on the SafeVoice voice-to-text macOS app.

## Project Overview

SafeVoice is a macOS menubar voice input app. Press a hotkey (default: Left Option), speak, release — text is transcribed and pasted into the active app.

**Tech stack**: Python 3.14, PyObjC (AppKit/Quartz/Foundation), rumps (menubar), pynput, sounddevice, mlx-qwen3-asr (ASR model), Ollama (LLM cleanup).

## Build & Run

```bash
# Activate virtualenv
source .venv/bin/activate

# Development build (alias mode — symlinks to source, fast rebuild)
python setup.py py2app -A

# Launch the app bundle
open dist/SafeVoice.app

# Alternative: run directly with Python (inherits Python's Accessibility permission)
python run.py
```

- Log file: `/tmp/safevoice.log` (overwritten each launch)
- Settings file: `~/.config/safevoice/settings.json`

## Architecture

| File | Role |
|---|---|
| `run.py` | Entry point, sets up logging |
| `src/app.py` | Main rumps menubar app, wires all components |
| `src/hotkey_manager.py` | Global hotkeys via Quartz CGEventTap + pynput |
| `src/asr_engine.py` | ASR transcription (mlx-qwen3-asr / Qwen3-0.6B) |
| `src/audio_capture.py` | Microphone capture via sounddevice |
| `src/text_injector.py` | Paste text via NSPasteboard + Cmd+V |
| `src/overlay.py` | Floating recording/transcription overlay |
| `src/settings_manager.py` | JSON settings store with change callbacks |
| `src/settings_window.py` | Native macOS settings window (PyObjC) |
| `src/dashboard_window.py` | Usage statistics dashboard window |
| `src/llm_cleanup.py` | Post-transcription cleanup via Ollama |
| `setup.py` | py2app build configuration |

## Key Settings Format

```json
{
  "languages": ["Auto"],
  "mode": "push_to_talk",
  "activate_hotkey": {"key": "", "modifiers": ["alt"]},
  "language_hotkey": {"key": "space", "modifiers": ["ctrl"]}
}
```

- `languages` is a **list** (multi-select). Was migrated from old `"language"` string format.
- Hotkey `"key": ""` means modifier-only (e.g., just pressing Option).
- Modifier names: `"alt"`, `"cmd"`, `"shift"`, `"ctrl"`.

## Critical macOS Gotchas (Lessons Learned)

### 1. Accessibility Permission (TCC) is Required for Hotkeys

- **CGEventTap** can be *created* without Accessibility permission, but it will **silently receive zero events**. There is no error — it just doesn't work.
- Always check `AXIsProcessTrusted()` at startup and implement a polling fallback that recreates the tap when permission is granted.
- Use `AXIsProcessTrustedWithOptions({"AXTrustedCheckOptionPrompt": True})` to trigger the macOS permission dialog.
- The `.venv/bin/python` interpreter and `dist/SafeVoice.app` binary have **separate** TCC entries. Granting permission to one does NOT grant it to the other.
- **py2app `-A` rebuilds produce a stable CDHash** (same template binary + same Info.plist = same signature). So Accessibility permission persists across rebuilds — the user only needs to grant it once.
- To test hotkey logic during development without re-granting permission, run `python run.py` directly (Python interpreter already has permission).

### 2. macOS Modifier Flag Bits (NSEvent / CGEvent)

These are the **correct** bit positions. Getting them wrong causes modifiers to display/detect as the wrong key:

| Modifier | Bit | Hex | Constant |
|---|---|---|---|
| Shift | `1 << 17` | `0x00020000` | `NSEventModifierFlagShift` / `kCGEventFlagMaskShift` |
| Control | `1 << 18` | `0x00040000` | `NSEventModifierFlagControl` / `kCGEventFlagMaskControl` |
| Option | `1 << 19` | `0x00080000` | `NSEventModifierFlagOption` / `kCGEventFlagMaskAlternate` |
| Command | `1 << 20` | `0x00100000` | `NSEventModifierFlagCommand` / `kCGEventFlagMaskCommand` |

### 3. Modifier-Only Hotkeys Need NSFlagsChangedMask

- `NSKeyDownMask` (`1 << 10`) only captures key presses, **not** modifier-only presses.
- To detect modifier-only hotkeys (e.g., "just press Option"), you must also monitor `NSFlagsChangedMask` (`1 << 12`).
- In CGEventTap: monitor `kCGEventFlagsChanged` events and track modifier press/release transitions.

### 4. pynput Cannot Monitor Global Events Without Accessibility Permission

- pynput logs `"This process is not trusted! Input event monitoring will not be possible"` but does NOT raise an exception — it silently fails.
- For the activation hotkey, use **Quartz CGEventTap** (more reliable, works with modifier-only keys).
- pynput is fine for secondary hotkeys (language switch) as long as the process is trusted.

### 5. py2app Alias Mode Quirks

- `open SafeVoice.app` requires a Mach-O binary — shell scripts as the executable will silently fail to launch.
- The app's `Contents/MacOS/SafeVoice` binary is a py2app bootstrap that loads Python. It is NOT the Python interpreter itself.
- The binary is copied from `.venv/lib/python3.14/site-packages/py2app/apptemplate/prebuilt/main-arm64` and re-signed each build, but the CDHash stays stable.

### 6. PyObjC NSObject Preventing Garbage Collection

- When using `performSelectorOnMainThread:withObject:waitUntilDone:False`, the trampoline NSObject must be retained (add to a class-level set) to prevent GC before the selector fires.
- Pattern used in `overlay.py`, `settings_window.py`, `dashboard_window.py`.

## Common Development Tasks

### Testing hotkey changes
1. Edit `src/hotkey_manager.py`
2. Rebuild: `python setup.py py2app -A`
3. Launch: `open dist/SafeVoice.app` (or `python run.py` for faster iteration)
4. Check logs: `tail -f /tmp/safevoice.log`
5. To test with synthetic key events:
   ```python
   import Quartz, time
   evt = Quartz.CGEventCreateKeyboardEvent(None, 58, True)  # 58 = left option
   Quartz.CGEventSetFlags(evt, 0x00080120)
   Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)
   time.sleep(0.5)
   evt2 = Quartz.CGEventCreateKeyboardEvent(None, 58, False)
   Quartz.CGEventSetFlags(evt2, 0x00000100)
   Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt2)
   ```

### Checking Accessibility permission
```python
from ApplicationServices import AXIsProcessTrusted
print("Trusted:", AXIsProcessTrusted())
```

### macOS virtual key codes (common)
Space=49, Tab=48, Return=36, Escape=53, Delete=51, Left Option=58, Right Option=61, Left Command=55, Left Control=59, Left Shift=56.
