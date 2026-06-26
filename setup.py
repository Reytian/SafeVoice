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
        "LSApplicationCategoryType": "public.app-category.productivity",
        "NSHumanReadableCopyright": "Copyright © 2026 SafeVoice",
        "NSMicrophoneUsageDescription": (
            "SafeVoice needs microphone access to transcribe your speech."
        ),
        # Accessibility (text injection via CGEventPost) has NO Info.plist
        # usage-description key; it is granted manually in System Settings.
        # The NSAppleEventsUsageDescription that used to sit here was removed:
        # the app sends no Apple Events, and the text misdescribed it anyway.
        "LSUIElement": False,  # Show in Dock so user can click to open dashboard
        "NSMainNibFile": "",  # No nib — rumps manages the menu
    },
    "packages": [
        "mlx_qwen3_asr",
        "mlx",
        "mlx_lm",
        "sounddevice",
        "_sounddevice_data",  # ships libportaudio; standalone builds crash on import without it
        "rumps",
        "numpy",
        "huggingface_hub",
        # PIL and google (protobuf) are unused transitive deps of transformers,
        # huggingface_hub, and mlx-lm. Without listing them here, py2app collects
        # them INTO Contents/Resources/lib/pythonXY.zip along with their native
        # binaries (google/_upb/*.so, PIL/.dylibs/*.dylib). Those cannot be
        # imported from a zip and fail Apple notarization as unsigned. Forcing
        # them on-disk, like every other binary package above, makes them
        # signable and loadable.
        "PIL",
        "google",
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
        "src.audio_preprocess",
        "src.dashboard_window",
        "src.history",
        "src.history_window",
        "src.hotkey_manager",
        "src.llm_backend",
        "src.llm_cleanup",
        "src.modes",
        "src.overlay",
        "src.privacy",
        "src.settings_manager",
        "src.settings_window",
        "src.setup_wizard",
        "src.single_instance",
        "src.text_injector",
        "src.text_postprocess",
        "src.vocabulary",
    ],
}

setup(
    name=APP_NAME,
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
