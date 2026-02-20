#!/usr/bin/env python3
"""
Prototype: Global hotkey registration on macOS using pynput.

Demonstrates:
  - Registering a global hotkey combination (Ctrl+Shift+Space)
  - Printing when the hotkey is detected
  - Exiting after 10 seconds or when Escape is pressed

Usage:
  source /Users/haotianyi/Documents/voice-ime/.venv/bin/activate
  python /Users/haotianyi/Documents/voice-ime/prototypes/test_hotkey.py

Note:
  On macOS, the app running this script (e.g., Terminal, iTerm2) must have
  Accessibility permissions granted under:
    System Settings > Privacy & Security > Accessibility
  Without this, pynput cannot observe global key events.
"""

import threading
import time
import sys

from pynput import keyboard


# -- Configuration -----------------------------------------------------------

# The hotkey combination to detect. pynput's GlobalHotKeys uses a string
# representation where '<ctrl>' / '<shift>' are modifier names and Key names
# map to their pynput equivalents. '<space>' represents the spacebar.
HOTKEY_COMBO = "<ctrl>+<shift>+<space>"

# How long (seconds) the script will run before auto-exiting.
TIMEOUT_SECONDS = 10


# -- State -------------------------------------------------------------------

_stop_event = threading.Event()


# -- Callbacks ---------------------------------------------------------------

def on_hotkey_activated():
    """Called when the registered global hotkey combination is pressed."""
    print(f"[HOTKEY] Detected: {HOTKEY_COMBO}")


def on_key_press(key):
    """Listener callback for individual key presses (used for Escape exit)."""
    if key == keyboard.Key.esc:
        print("[EXIT] Escape pressed -- shutting down.")
        _stop_event.set()
        return False  # Stops the listener


# -- Main --------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Global Hotkey Test (pynput)")
    print("=" * 60)
    print(f"  Hotkey       : {HOTKEY_COMBO}")
    print(f"  Timeout      : {TIMEOUT_SECONDS}s")
    print(f"  Exit early   : press Escape")
    print("=" * 60)
    print()

    # 1. GlobalHotKeys listener -- watches for the specific combination.
    #    Each entry maps a hotkey string to a callback.
    hotkey_listener = keyboard.GlobalHotKeys({
        HOTKEY_COMBO: on_hotkey_activated,
    })

    # 2. Regular Listener -- watches every key so we can detect Escape.
    escape_listener = keyboard.Listener(on_press=on_key_press)

    # Start both listeners (they run in daemon threads).
    hotkey_listener.start()
    escape_listener.start()

    print("[INFO] Listeners started. Waiting for hotkey or Escape ...")

    # 3. Block the main thread until timeout or stop event.
    start_time = time.monotonic()
    while not _stop_event.is_set():
        elapsed = time.monotonic() - start_time
        if elapsed >= TIMEOUT_SECONDS:
            print(f"\n[EXIT] Timeout reached ({TIMEOUT_SECONDS}s) -- shutting down.")
            break
        # Sleep in small increments so we stay responsive to the stop event.
        _stop_event.wait(timeout=0.25)

    # 4. Clean up listeners.
    hotkey_listener.stop()
    escape_listener.stop()

    print("[INFO] Done.")
    sys.exit(0)


if __name__ == "__main__":
    main()
