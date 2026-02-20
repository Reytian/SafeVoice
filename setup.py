"""
py2app setup script for SafeVoice.

Build the macOS .app bundle:
    python setup.py py2app

For a development/alias build (faster, links to source):
    python setup.py py2app -A
"""

from setuptools import setup

APP = ["run.py"]
APP_NAME = "SafeVoice"

DATA_FILES = []

OPTIONS = {
    "argv_emulation": False,
    "iconfile": "assets/SafeVoice.icns",
    "plist": {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": "com.safevoice.app",
        "CFBundleVersion": "1.0.0",
        "CFBundleShortVersionString": "1.0.0",
        "LSMinimumSystemVersion": "14.0",
        "NSMicrophoneUsageDescription": (
            "SafeVoice needs microphone access to transcribe your speech."
        ),
        "NSAppleEventsUsageDescription": (
            "SafeVoice needs accessibility access to type text into other apps."
        ),
        "LSUIElement": False,  # Show in Dock so user can click to open dashboard
        "NSMainNibFile": "",  # No nib — rumps manages the menu
    },
    "packages": [
        "mlx_qwen3_asr",
        "mlx",
        "sounddevice",
        "pynput",
        "rumps",
        "numpy",
        "huggingface_hub",
        "objc",
        "AppKit",
        "Foundation",
        "Quartz",
        "ApplicationServices",
    ],
    "includes": [
        "src",
        "src.app",
        "src.asr_engine",
        "src.audio_capture",
        "src.hotkey_manager",
        "src.text_injector",
        "src.overlay",
        "src.settings_manager",
        "src.settings_window",
        "src.dashboard_window",
        "src.llm_cleanup",
    ],
}

setup(
    name=APP_NAME,
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
