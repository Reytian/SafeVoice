"""SafeVoice - Main application module.

Combines all components into a macOS menubar app:
- Audio capture (sounddevice)
- ASR engine (mlx-qwen3-asr)
- Text injection (NSPasteboard + Cmd+V)
- Global hotkey (pynput)
- Menubar UI (rumps)
- Floating overlay (PyObjC)
"""

import logging
import os
import threading
import time
import sys

import numpy as np
import objc
from Foundation import (
    NSActivityLatencyCritical,
    NSActivityUserInitiatedAllowingIdleSystemSleep,
    NSObject,
    NSProcessInfo,
    NSThread,
)

logger = logging.getLogger(__name__)
import rumps


class _MainThreadTrampoline(NSObject):
    """Invoke a Python callable on the main thread.

    AppKit (NSStatusItem title, menu item titles) must only be mutated on the
    main thread; the status/title updates run from background worker threads
    (model load, transcription). A class-level set prevents the trampoline
    from being GC'd before the async selector fires.
    """

    _prevent_gc: set = set()

    def initWithBlock_(self, block):
        self = objc.super(_MainThreadTrampoline, self).init()
        if self is None:
            return None
        self._block = block
        return self

    def invoke(self):
        try:
            if self._block is not None:
                self._block()
        finally:
            _MainThreadTrampoline._prevent_gc.discard(self)


def _run_on_main(block):
    """Run *block* (a zero-arg callable) on the main thread.

    Runs inline if already on the main thread, otherwise dispatches via
    performSelectorOnMainThread so AppKit objects are only touched on main.
    """
    if NSThread.isMainThread():
        block()
        return
    trampoline = _MainThreadTrampoline.alloc().initWithBlock_(block)
    _MainThreadTrampoline._prevent_gc.add(trampoline)
    trampoline.performSelectorOnMainThread_withObject_waitUntilDone_(
        "invoke", None, False
    )

from .audio_capture import AudioCapture
from .asr_engine import ASREngine
from .text_injector import TextInjector
from .hotkey_manager import HotkeyManager, describe_hotkey
from .privacy import redact
from .overlay import FloatingOverlay
from .settings_manager import SettingsManager, SUPPORTED_LANGUAGES
from .settings_window import SettingsWindow
from . import audio_preprocess
from .dashboard_window import DashboardWindow
from .llm_cleanup import LLMCleanup
from .text_postprocess import strip_filler_words
from .history import HistoryStore
from .vocabulary import VocabularyManager
from .modes import ModeManager
from .setup_wizard import SetupWizard
from .llm_backend import get_backend
from .single_instance import (
    install_show_settings_listener,
    is_duplicate_and_signal_existing,
)


# Language definitions -- "Auto" first so it is the default.
LANGUAGES = [
    {"name": "Auto", "code": None, "badge": "Auto", "display": "Auto (detect)"},
    {"name": "English", "code": "en", "badge": "EN", "display": "English"},
    {"name": "Chinese", "code": "zh", "badge": "\u4e2d\u6587", "display": "Chinese (\u4e2d\u6587)"},
    {"name": "French", "code": "fr", "badge": "FR", "display": "French (Fran\u00e7ais)"},
    {"name": "Japanese", "code": "ja", "badge": "JP", "display": "Japanese (\u65e5\u672c\u8a9e)"},
    {"name": "Korean", "code": "ko", "badge": "KR", "display": "Korean (\ud55c\uad6d\uc5b4)"},
    {"name": "German", "code": "de", "badge": "DE", "display": "German (Deutsch)"},
    {"name": "Spanish", "code": "es", "badge": "ES", "display": "Spanish (Espa\u00f1ol)"},
    {"name": "Italian", "code": "it", "badge": "IT", "display": "Italian (Italiano)"},
    {"name": "Portuguese", "code": "pt", "badge": "PT", "display": "Portuguese (Portugu\u00eas)"},
    {"name": "Russian", "code": "ru", "badge": "RU", "display": "Russian (\u0420\u0443\u0441\u0441\u043a\u0438\u0439)"},
    {"name": "Arabic", "code": "ar", "badge": "AR", "display": "Arabic (\u0627\u0644\u0639\u0631\u0628\u064a\u0629)"},
    {"name": "Hindi", "code": "hi", "badge": "HI", "display": "Hindi (\u0939\u093f\u0928\u094d\u0926\u0940)"},
    {"name": "Dutch", "code": "nl", "badge": "NL", "display": "Dutch (Nederlands)"},
    {"name": "Turkish", "code": "tr", "badge": "TR", "display": "Turkish (T\u00fcrk\u00e7e)"},
]

# App states
STATE_IDLE = "idle"
STATE_LOADING = "loading"
STATE_LISTENING = "listening"
STATE_TRANSCRIBING = "transcribing"
STATE_INJECTING = "injecting"


def _friendly_error(exc: Exception) -> str:
    """Short, human-readable overlay message for a pipeline failure.

    The overlay used to show raw exception text ("Error: [Errno -9986]...")
    for the 1-2 s it is visible; the full traceback is always in the log,
    so the overlay should say what the user can act on.
    """
    from .asr_engine import ASREngineError, ModelNotLoadedError
    if isinstance(exc, ModelNotLoadedError):
        return "Speech model is still loading. Try again in a moment."
    if isinstance(exc, ASREngineError):
        return "Transcription failed. Try again (details in log)."
    name = type(exc).__name__
    if "PortAudio" in name or "PortAudio" in str(exc):
        return "Microphone unavailable. Check your input device."
    if isinstance(exc, MemoryError):
        return "Out of memory. Try a shorter dictation."
    if isinstance(exc, OSError):
        return "System error while transcribing (details in log)."
    detail = str(exc).strip()
    return f"Error: {detail[:80]}" if detail else f"Error: {name}"

# Icon path for menubar
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ICON_PATH = os.path.join(_PROJECT_ROOT, "assets", "SafeVoice.icns")


class SafeVoiceApp(rumps.App):
    """Main SafeVoice menubar application."""

    def __init__(self):
        # Use .icns icon for menubar; fall back to text if missing
        icon_path = _ICON_PATH if os.path.exists(_ICON_PATH) else None
        super().__init__(
            name="SafeVoice",
            title="SV" if icon_path is None else None,
            icon=icon_path,
            quit_button=None,
        )

        # Opt out of App Nap so the menubar listener and CGEventTap stay
        # responsive after long idle periods. The token MUST be retained on
        # the instance: NSProcessInfo ends the activity when the token is
        # released. UserInitiatedAllowingIdleSystemSleep keeps us latency-
        # sensitive while still letting the Mac sleep on its normal schedule.
        self._app_nap_token = NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
            NSActivityUserInitiatedAllowingIdleSystemSleep | NSActivityLatencyCritical,
            "SafeVoice hotkey listener",
        )

        # Listen for "show settings" pings from a second-launched instance
        # (the duplicate posts the notification then exits in main()). Must
        # retain the observer on the instance or PyObjC will GC it.
        self._show_settings_observer = install_show_settings_listener(
            self._show_settings_from_duplicate
        )

        # State
        self._state = STATE_IDLE
        self._language_index = 0

        # Settings (load saved preferences)
        self._settings = SettingsManager()
        self._settings.register_callback(self._on_setting_changed)

        # Read recording mode from persisted settings. Defaults to
        # push_to_talk for first-run users; once the user picks toggle
        # mode (via menubar Mode submenu or settings window) it is saved
        # and used on every subsequent launch.
        saved_mode = self._settings.get("mode", "push_to_talk")
        if saved_mode not in ("push_to_talk", "toggle"):
            saved_mode = "push_to_talk"
        self._mode = saved_mode

        # Components
        self._audio = AudioCapture(sample_rate=16000, channels=1, blocksize=1024)
        asr_model = self._settings.get("asr_model", "Qwen/Qwen3-ASR-0.6B")
        # Guard against a persisted model this build cannot run (the catalog
        # used to offer whisper/cloud engines with no dispatch): fall back to
        # the default instead of wedging at "Error" on every launch. Unknown
        # IDs (hand-edited custom repos) pass through untouched.
        from .llm_backend import ASR_MODELS, IMPLEMENTED_ASR_ENGINES
        catalog_entry = next(
            (m for m in ASR_MODELS if m["id"] == asr_model), None
        )
        if catalog_entry is not None and catalog_entry["engine"] not in IMPLEMENTED_ASR_ENGINES:
            logger.warning(
                "Saved ASR model %s needs unimplemented engine %s; using default",
                asr_model, catalog_entry["engine"],
            )
            asr_model = "Qwen/Qwen3-ASR-0.6B"
        self._asr = ASREngine(model_id=asr_model)
        self._injector = TextInjector()
        self._hotkey = HotkeyManager()
        self._overlay = FloatingOverlay()
        self._llm_backend = self._create_llm_backend()
        self._llm = LLMCleanup(backend=self._llm_backend)
        self._history = HistoryStore()
        self._vocabulary = VocabularyManager()
        self._modes = ModeManager()
        self._settings_window = SettingsWindow(
            self._settings,
            modes_manager=self._modes,
            vocabulary_manager=self._vocabulary,
            on_llm_change=self._on_llm_change,
            on_modes_change=self._rebuild_modes_menu,
        )
        self._dashboard = DashboardWindow(
            self._settings,
            on_open_settings=lambda: self._settings_window.show(),
            status_provider=self._dashboard_status,
        )
        self._active_mode = self._modes.get("Quick")

        # Audio buffer for batch transcription
        self._audio_chunks = []
        self._audio_lock = threading.Lock()

        # Seconds between preview/speculative passes. Effective cadence is
        # interval + inference time (the loop waits, then transcribes).
        self._speculative_interval = 2.5
        self._listen_started = 0.0
        self._session_peak_level = 0.0

        # Apply saved settings before building menu
        self._apply_saved_settings()

        # Build menu
        self._build_menu()

        # Start hotkey listener
        self._hotkey.start(
            on_activate=self._on_hotkey_activate,
            on_deactivate=self._on_hotkey_deactivate,
        )

        # Preload model in background. On a true first run with no model on
        # disk the setup wizard owns the download (it has visible progress);
        # starting a load here too used to kick off a SECOND concurrent
        # download of the same 1.2 GB repo. If the user skips or closes the
        # wizard, the first hotkey press triggers the load on demand.
        self._model_loading = False
        if self._settings.get("first_run", True) and not ASREngine.is_model_downloaded():
            logger.info("First run without model: deferring download to the setup wizard")
            self._update_status("Finish setup to download the speech model")
        else:
            self._start_model_load()

        # Show setup wizard on first run (deferred until event loop is running)
        if self._settings.get("first_run", True):
            threading.Timer(1.5, self._show_setup_wizard).start()

        # Set up dock click handler after rumps initializes its delegate
        threading.Timer(1.0, self._setup_dock_click_handler).start()

    def _build_menu(self):
        """Build the menubar dropdown menu."""
        # Status item
        self._status_item = rumps.MenuItem("Ready", callback=None)
        self._status_item.set_callback(None)

        # Language submenu
        self._lang_menu = rumps.MenuItem("Language")
        for i, lang in enumerate(LANGUAGES):
            item = rumps.MenuItem(
                lang["display"],
                callback=self._make_language_callback(i),
            )
            if i == self._language_index:
                item.state = True
            self._lang_menu.add(item)

        # Mode submenu. Checkmarks reflect the persisted setting loaded
        # in __init__ so users see their saved choice on every launch
        # (and not always "Hold to Talk" by default).
        self._mode_menu = rumps.MenuItem("Mode")
        self._ptt_item = rumps.MenuItem(
            "Hold to Talk (Recommended)",
            callback=self._set_push_to_talk,
        )
        self._toggle_item = rumps.MenuItem(
            "Toggle On/Off (click to start, click to stop)",
            callback=self._set_toggle_mode,
        )
        self._ptt_item.state = (self._mode == "push_to_talk")
        self._toggle_item.state = (self._mode == "toggle")
        self._mode_menu.add(self._ptt_item)
        self._mode_menu.add(self._toggle_item)

        # Processing modes submenu. Items are clickable and switch the
        # active mode; titles are plain names because per-mode GLOBAL
        # hotkeys are not dispatched anywhere yet, and advertising them in
        # the menu misled users into pressing dead key combos.
        self._modes_menu = rumps.MenuItem("Processing Mode")
        for mode in self._modes.get_all():
            item = rumps.MenuItem(
                mode.name,
                callback=self._make_processing_mode_callback(mode.name),
            )
            item.state = (
                self._active_mode is not None
                and mode.name == self._active_mode.name
            )
            self._modes_menu[item.title] = item

        # Dashboard
        dashboard_item = rumps.MenuItem("Dashboard...", callback=self._open_dashboard)

        # Settings
        settings_item = rumps.MenuItem("Settings...", callback=self._open_settings)

        # Hotkey info -- text is mode-aware so users see how the current
        # mode actually behaves. Updated by _refresh_hotkey_info_label
        # whenever mode changes.
        self._hotkey_info_item = rumps.MenuItem("", callback=None)
        self._refresh_hotkey_info_label()

        # Quit
        quit_item = rumps.MenuItem("Quit SafeVoice", callback=self._on_quit)

        self.menu = [
            self._status_item,
            None,
            self._lang_menu,
            self._mode_menu,
            self._modes_menu,
            None,
            dashboard_item,
            settings_item,
            None,
            self._hotkey_info_item,
            None,
            quit_item,
        ]

    def _make_language_callback(self, index):
        """Create a callback for language selection from the menubar."""
        def callback(_):
            self._set_language(index)
            # Persist the menubar choice so it survives relaunch and the
            # settings window shows the same value. SettingsManager.set is
            # a no-op on equal values, so the change callback cannot loop.
            self._settings.set("languages", [LANGUAGES[index]["name"]])
        return callback

    def _make_processing_mode_callback(self, name):
        """Create a callback that switches the active processing mode."""
        def callback(_):
            self._set_active_processing_mode(name)
        return callback

    def _set_active_processing_mode(self, name):
        """Switch the active processing mode and update menu checkmarks."""
        mode = self._modes.get(name)
        if mode is None:
            logger.warning("Unknown processing mode: %s", name)
            return
        self._active_mode = mode
        for key in self._modes_menu.keys():
            self._modes_menu[key].state = (key == name)
        self._update_status(self._idle_status_text())
        logger.info("Processing mode switched to %s", name)

    def _rebuild_modes_menu(self):
        """Rebuild the Processing Mode submenu after modes change in Settings.

        Without this, modes added/renamed/deleted in the Modes tab only
        appeared in the menubar after a relaunch.
        """
        def _do():
            try:
                names = [m.name for m in self._modes.get_all()]
                current = self._active_mode.name if self._active_mode else None
                self._modes_menu.clear()
                for name in names:
                    item = rumps.MenuItem(
                        name, callback=self._make_processing_mode_callback(name)
                    )
                    item.state = (name == current)
                    self._modes_menu[item.title] = item
                if current not in names:
                    # The active mode was deleted; fall back to Quick.
                    self._set_active_processing_mode("Quick")
            except Exception:
                logger.exception("Failed to rebuild modes menu")

        _run_on_main(_do)

    def _set_language(self, index):
        """Switch to a language by index."""
        self._language_index = index
        lang = LANGUAGES[index]

        # Update menu checkmarks
        for i, l in enumerate(LANGUAGES):
            self._lang_menu[l["display"]].state = (i == index)

        # Update ASR engine
        self._asr.set_language(lang["name"])

        # Update overlay if visible
        self._overlay.set_language(lang["badge"])

        print(f"[SafeVoice] Language: {lang['name']}")

    def _activate_hotkey_label(self) -> str:
        """Human-readable label for the CONFIGURED activation hotkey.

        Every user-facing hotkey string must come from here: the old
        hardcoded "Left ⌥" text was wrong even for fresh installs (the
        persisted default is ⌥+Space) and stayed wrong after the user
        recorded a custom hotkey in Settings.
        """
        hk = self._settings.get(
            "activate_hotkey", {"key": "space", "modifiers": ["alt"]}
        )
        label = describe_hotkey(hk)
        return label if label != "None" else "the activation hotkey"

    def _set_push_to_talk(self, _):
        """Switch to push-to-talk mode and persist the choice."""
        logger.info("Menu callback: switching to push_to_talk mode")
        self._mode = "push_to_talk"
        self._ptt_item.state = True
        self._toggle_item.state = False
        self._hotkey.set_mode("push_to_talk")
        self._settings.set("mode", "push_to_talk")
        self._refresh_hotkey_info_label()
        rumps.notification(
            "SafeVoice",
            "Mode: Hold to Talk",
            f"Hold {self._activate_hotkey_label()} to record, release to transcribe.",
        )

    def _set_toggle_mode(self, _):
        """Switch to toggle mode and persist the choice.

        Toggle mode: press the hotkey once to START recording, press it
        again to STOP and transcribe. Useful for longer dictations where
        holding the key is uncomfortable.
        """
        logger.info("Menu callback: switching to toggle mode")
        self._mode = "toggle"
        self._ptt_item.state = False
        self._toggle_item.state = True
        self._hotkey.set_mode("toggle")
        self._settings.set("mode", "toggle")
        self._refresh_hotkey_info_label()
        rumps.notification(
            "SafeVoice",
            "Mode: Toggle On/Off",
            f"Press {self._activate_hotkey_label()} once to start recording, again to stop.",
        )

    def _refresh_hotkey_info_label(self):
        """Update the menubar's Hotkey info line to match mode AND hotkey.

        Called from __init__, both _set_*_mode handlers, and whenever the
        activate_hotkey setting changes.
        """
        if not hasattr(self, "_hotkey_info_item"):
            return
        label = self._activate_hotkey_label()
        if self._mode == "toggle":
            self._hotkey_info_item.title = (
                f"Hotkey: Press {label} to start, again to stop"
            )
        else:
            self._hotkey_info_item.title = f"Hotkey: Hold {label}"

    def _start_model_load(self):
        """Load the ASR model in a background thread (download if missing).

        Idempotent: a second call while a load is in flight is a no-op, and
        a FAILED load can be retried simply by pressing the hotkey (see
        _on_hotkey_activate) instead of restarting the app.
        """
        if self._model_loading or self._asr.is_loaded:
            return
        self._model_loading = True
        downloading = not ASREngine.is_model_downloaded()

        def load():
            try:
                logger.info("Background model load thread started")
                self._set_state(STATE_LOADING)
                if downloading:
                    self._update_status("Downloading speech model (~1.2 GB)...")
                else:
                    self._update_status("Loading speech model...")
                self._asr.load_model()
                self._set_state(STATE_IDLE)
                self._update_status(self._idle_status_text())
                print("[SafeVoice] Model loaded")
                logger.info("ASR model loaded successfully")
                # Warm up LLM after ASR is loaded to avoid cold-start latency
                if self._llm.is_available():
                    print("[SafeVoice] Warming up LLM...")
                    logger.info("Warming up LLM model...")
                    self._llm.warm_up()
                    print("[SafeVoice] LLM ready")
                    logger.info("LLM model warmed up and ready")
                else:
                    print("[SafeVoice] LLM cleanup unavailable (install Ollama + `ollama pull qwen2.5:3b`)")
                    logger.warning("LLM cleanup unavailable")
            except Exception:
                logger.exception("Model load failed")
                self._set_state(STATE_IDLE)
                self._update_status("Model load failed. Press the hotkey to retry.")
                try:
                    rumps.notification(
                        "SafeVoice",
                        "Speech model failed to load",
                        "Check your network connection, then press the hotkey to retry.",
                    )
                except Exception:
                    logger.debug("Could not post model-failure notification", exc_info=True)
            finally:
                self._model_loading = False

        t = threading.Thread(target=load, daemon=True)
        t.start()

    def _show_setup_wizard(self):
        """Show setup wizard, dispatched to main thread for AppKit safety."""
        def launch():
            def on_complete():
                self._settings.set("first_run", False)
                # Startup defers the model load to the wizard on first run;
                # make sure it happens once the wizard is done (no-op if the
                # wizard's own download already let a load succeed).
                self._start_model_load()
            self._wizard = SetupWizard(self, on_complete=on_complete)
            self._wizard.show()

        _run_on_main(launch)

    def _apply_saved_settings(self):
        """Apply persisted settings to components on startup."""
        saved_langs = self._settings.get("languages", ["Auto"])
        asr_lang = self._get_asr_language(saved_langs)
        for i, lang in enumerate(LANGUAGES):
            if lang["name"] == asr_lang:
                self._language_index = i
                break

        saved_mode = self._settings.get("mode", "push_to_talk")
        self._mode = saved_mode

        # Apply response speed to ASR streaming params
        speed = self._settings.get("response_speed", "fast")
        if speed == "fast":
            self._asr.set_streaming_params(
                chunk_size_sec=1.0, finalization_mode="latency",
            )
        else:
            self._asr.set_streaming_params(
                chunk_size_sec=2.0, finalization_mode="accuracy",
            )

        # Only set ASR language if it's supported by the engine; dialect
        # names from the settings window (e.g. Cantonese) are not valid for
        # the ASR model and would raise ValueError.
        if asr_lang in self._asr.LANGUAGES:
            self._asr.set_language(asr_lang)
        else:
            self._asr.set_language("Auto")
        self._hotkey.set_mode(saved_mode)

        # Apply saved hotkey configuration
        activate_hk = self._settings.get("activate_hotkey")
        if activate_hk:
            self._hotkey.set_activate_hotkey(activate_hk)

    def _create_llm_backend(self):
        """Create the appropriate LLM backend from current settings."""
        source = self._settings.get("llm_source", "local")
        local_model = self._settings.get("llm_local_model", "qwen2.5:3b")
        cloud_provider = self._settings.get("llm_cloud_provider", "openai")
        cloud_model = self._settings.get("llm_cloud_model", "gpt-4o-mini")
        cloud_api_key = ""
        import json
        cred_path = os.path.expanduser("~/.config/safevoice/credentials.json")
        try:
            if os.path.exists(cred_path):
                with open(cred_path, encoding="utf-8") as f:
                    creds = json.load(f)
                cloud_api_key = creds.get(cloud_provider, "")
        except Exception:
            logger.warning("Could not load credentials from %s", cred_path, exc_info=True)
        mlx_model = self._settings.get("llm_mlx_model", "mlx-community/Qwen3.5-4B-4bit")
        return get_backend(
            source=source, local_model=local_model,
            cloud_provider=cloud_provider, cloud_model=cloud_model,
            cloud_api_key=cloud_api_key, mlx_model=mlx_model,
        )

    def _on_llm_change(self):
        """Callback when the user changes LLM settings in the Models tab."""
        self._llm_backend = self._create_llm_backend()
        self._llm.set_backend(self._llm_backend)
        logger.info("LLM backend changed to: %s", self._llm_backend.name)

    def _on_setting_changed(self, key, old_value, new_value):
        """Callback fired when any setting changes via the settings window."""
        if key == "languages":
            asr_lang = self._get_asr_language(new_value)
            for i, lang in enumerate(LANGUAGES):
                if lang["name"] == asr_lang:
                    self._set_language(i)
                    return
            # If no match found (e.g. dialect), fall back to Auto
            self._set_language(0)

        elif key == "mode":
            if new_value == "push_to_talk":
                self._set_push_to_talk(None)
            else:
                self._set_toggle_mode(None)

        elif key == "response_speed":
            if new_value == "fast":
                self._asr.set_streaming_params(
                    chunk_size_sec=1.0, finalization_mode="latency",
                )
            else:
                self._asr.set_streaming_params(
                    chunk_size_sec=2.0, finalization_mode="accuracy",
                )

        elif key == "activate_hotkey":
            self._hotkey.set_activate_hotkey(new_value)
            self._refresh_hotkey_info_label()

        elif key == "asr_model":
            # The engine is constructed once at startup; tell the user the
            # change is saved but not live, instead of silently doing
            # nothing for the rest of the session.
            try:
                rumps.notification(
                    "SafeVoice",
                    "ASR model saved",
                    "The new speech model loads on the next launch.",
                )
            except Exception:
                logger.debug("Could not post ASR-model notification", exc_info=True)

    def _get_asr_language(self, languages):
        """Determine the ASR language from the languages setting list."""
        if not languages or "Auto" in languages:
            return "Auto"
        if len(languages) == 1:
            return languages[0]
        # Multiple languages selected -> use Auto for detection
        return "Auto"

    def _setup_dock_click_handler(self):
        """Inject a dock-click handler into rumps's delegate to show the dashboard."""
        try:
            from AppKit import NSApplication
            ns_app = NSApplication.sharedApplication()
            if ns_app is None:
                return
            delegate = ns_app.delegate()
            if delegate is None:
                return

            dashboard = self._dashboard

            def reopen_handler(_self, _app, _has_visible):
                dashboard.show()
                return True

            objc.classAddMethod(
                type(delegate),
                b"applicationShouldHandleReopen:hasVisibleWindows:",
                reopen_handler,
            )
            logger.info("Dock click handler installed")
        except Exception as e:
            logger.warning("Failed to set up dock click handler: %s", e)

    def _open_dashboard(self, _):
        """Open the dashboard window."""
        self._dashboard.show()

    def _open_settings(self, _):
        """Open the settings window."""
        self._settings_window.show()

    def _show_settings_from_duplicate(self):
        """Open the settings window in response to a second-launched instance.

        Invoked on the main thread by the NSDistributedNotification observer
        installed in __init__. Calling settings_window.show() also activates
        the app and orders the window front, so the user gets visible
        feedback that their second double-click was received.
        """
        logger.info("Received show-settings ping from a duplicate launch")
        self._settings_window.show()

    def _set_state(self, state):
        """Update app state and UI.

        Called from background threads (model load, transcription worker), so
        the AppKit title write is dispatched to the main thread.
        """
        self._state = state
        # Show status text next to the icon; None = icon only
        titles = {
            STATE_IDLE: None,
            STATE_LOADING: "...",
            STATE_LISTENING: "REC",
            STATE_TRANSCRIBING: "...",
            STATE_INJECTING: "...",
        }
        title = titles.get(state)

        def _apply():
            try:
                self.title = title
            except Exception:
                logger.debug("Failed to set status-bar title", exc_info=True)

        _run_on_main(_apply)

    def _update_status(self, text):
        """Update the status menu item (main-thread AppKit write)."""
        def _apply():
            try:
                self._status_item.title = text
            except Exception:
                pass

        _run_on_main(_apply)

    def _idle_status_text(self) -> str:
        """Status-item text for the idle state, mode-aware.

        Shows which processing mode will run on the next dictation so a
        non-default mode (e.g. English Translation) is glanceable instead
        of a surprise at paste time.
        """
        if self._active_mode is not None and self._active_mode.name != "Quick":
            return f"Ready · {self._active_mode.name}"
        return "Ready"

    def _dashboard_status(self):
        """(text, is_ready) for the dashboard's live status label."""
        if not self._asr.is_loaded:
            return ("Loading model...", False)
        if self._state == STATE_IDLE:
            return (self._idle_status_text(), True)
        labels = {
            STATE_LISTENING: "Listening...",
            STATE_TRANSCRIBING: "Transcribing...",
            STATE_INJECTING: "Pasting...",
            STATE_LOADING: "Loading model...",
        }
        return (labels.get(self._state, "Busy..."), False)

    # -- Hotkey callbacks --

    def _on_hotkey_activate(self):
        """Called when the user activates voice input (hotkey pressed).

        Returns True if recording started, False if the request was rejected
        (busy/loading/mic unavailable) so HotkeyManager can roll back its
        toggle state and the next press is a clean activation.
        """
        try:
            logger.debug("Hotkey activate, state=%s", self._state)
            if self._state not in (STATE_IDLE,):
                return False

            if not self._asr.is_loaded:
                # Kick (or retry) the load on demand: covers a failed first
                # load (offline) and a skipped/closed setup wizard, instead
                # of answering "Model loading..." forever while nothing loads.
                self._start_model_load()
                self._update_status("Loading speech model...")
                return False

            return self._start_listening()
        except Exception:
            logger.exception("Error in _on_hotkey_activate")
            return False

    def _on_hotkey_deactivate(self):
        """Called when the user deactivates voice input (hotkey released).

        Returns True if a recording was stopped, False otherwise.
        """
        try:
            logger.debug("Hotkey deactivate, state=%s", self._state)
            if self._state != STATE_LISTENING:
                return False

            self._stop_listening_and_transcribe()
            return True
        except Exception:
            logger.exception("Error in _on_hotkey_deactivate")
            return False

    # -- Recording flow --

    def _start_listening(self):
        """Start capturing audio. Returns True if recording actually started.

        Audio capture is started BEFORE state/overlay are committed. If the
        mic can't open (busy, unplugged, no input device, no permission),
        start() raises and we revert to IDLE instead of leaving the app
        wedged at STATE_LISTENING with no stream (which previously bricked
        the app until restart).
        """
        with self._audio_lock:
            self._audio_chunks.clear()

        lang = LANGUAGES[self._language_index]

        # Session telemetry for the preview worker: when the recording
        # started and the loudest level seen so far (for the silent-mic hint).
        # Set BEFORE audio.start so the first chunk callback sees them.
        self._listen_started = time.monotonic()
        self._session_peak_level = 0.0

        try:
            self._audio.start(callback=self._on_audio_chunk)
        except Exception:
            logger.exception("Failed to start audio capture")
            self._set_state(STATE_IDLE)
            self._update_status("Microphone unavailable")
            return False

        self._set_state(STATE_LISTENING)
        self._update_status("Listening...")
        self._overlay.show(language=lang["badge"])

        # Preview/speculative worker runs for EVERY recording: it shows live
        # partial transcription in the overlay (and the silent-mic hint), and
        # additionally pre-runs the LLM when a custom mode is active.
        self._start_speculative_timer()

        print("[SafeVoice] Listening...")
        return True

    def _on_audio_chunk(self, chunk):
        """Called for each audio chunk from the microphone."""
        with self._audio_lock:
            self._audio_chunks.append(chunk.copy())

        # Update overlay with audio level, and remember the session peak so
        # the preview worker can tell "user is quiet" from "mic delivers
        # digital silence" (muted, denied permission, wrong device).
        level = self._audio.get_level()
        if level > self._session_peak_level:
            self._session_peak_level = level
        self._overlay.update_level(level)

    def _stop_listening_and_transcribe(self):
        """Stop recording and run final transcription."""
        self._stop_speculative_timer()
        self._set_state(STATE_TRANSCRIBING)
        self._update_status("Transcribing...")
        self._overlay.set_status("processing")
        self._overlay.update_text("Processing...")

        # Start elapsed timer for the overlay
        timer_start = time.monotonic()
        timer_stop = threading.Event()

        def _update_timer():
            while not timer_stop.wait(0.2):
                elapsed = time.monotonic() - timer_start
                self._overlay.update_text(f"Processing... {elapsed:.1f}s")

        timer_thread = threading.Thread(target=_update_timer, daemon=True)
        timer_thread.start()

        # Run transcription in background thread
        def transcribe():
            try:
                # Stop audio capture HERE, in the worker, NOT on the caller's
                # thread. _stop_listening_and_transcribe runs on the CGEventTap
                # callback thread; stopping/closing the sounddevice stream there
                # is slow enough to trip kCGEventTapDisabledByTimeout, which
                # drops the following modifier-release event and wedges the
                # hotkey until the tap is re-enabled.
                full_audio = self._audio.stop()
                # Batch transcription on full audio for best accuracy
                if full_audio is None or len(full_audio) == 0:
                    text, lang = "", "Auto"
                else:
                    logger.info("Batch transcription on %d samples...", len(full_audio))
                    cleaned = audio_preprocess.normalize_audio(full_audio)
                    # Skip VAD trim — torch import takes 40-70s on first call
                    text, lang = self._asr.transcribe(cleaned)
                    logger.info("ASR result: %s (lang=%s)", redact(text), lang)

                    # Apply vocabulary snippet replacements
                    text = self._vocabulary.apply_snippets(text)

                # Stop the elapsed timer
                timer_stop.set()

                if text.strip():
                    self._overlay.update_text(text)
                    # LLM processing (mode-aware)
                    stripped = text.strip()
                    raw_for_history = stripped
                    has_cjk = any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af' for c in stripped)
                    is_long_enough = len(stripped) >= 4 if has_cjk else len(stripped.split()) >= 3

                    if self._active_mode.prompt_template and self._llm.is_available() and is_long_enough:
                        # Custom mode: use mode's LLM prompt
                        self._overlay.set_status("processing")
                        self._update_status(f"{self._active_mode.name}...")
                        prompt = self._active_mode.render_prompt(stripped)
                        logger.info("Mode '%s' LLM starting...", self._active_mode.name)
                        # Check speculative cache first
                        cached = self._llm.get_speculative_result(stripped)
                        if cached:
                            logger.info("Using speculative result")
                            text = cached
                        else:
                            # Assign unconditionally: cleanup() already falls
                            # back to the rule-stripped text on failure, so a
                            # legitimate result equal to the input must not be
                            # dropped by an == guard.
                            text = self._llm.cleanup(
                                text, custom_prompt=prompt,
                                allow_script_change=self._mode_allows_translation(),
                            )
                        self._overlay.update_text(text)
                        logger.info("Mode result: %s", redact(text))
                    elif self._active_mode.prompt_template is None and self._llm.is_available() and is_long_enough:
                        # Quick mode: default cleanup
                        self._overlay.set_status("processing")
                        self._update_status("Cleaning up...")
                        logger.info("LLM cleanup starting...")
                        cleaned = self._llm.cleanup(text)
                        logger.info("LLM result: %r", cleaned)
                        if cleaned != text:
                            print(f"[SafeVoice] LLM: {redact(text)} -> {redact(cleaned)}")
                            text = cleaned
                            self._overlay.update_text(text)
                    else:
                        # No LLM step ran -- either text too short for the
                        # LLM gate, or no LLM backend is available. Still
                        # apply the deterministic filler strip so "嗯对" ->
                        # "对" and "um yes" -> "yes" instead of pasting raw
                        # ASR with all hesitations intact.
                        if not is_long_enough:
                            logger.info(
                                "Skipping LLM for short text: %s", redact(stripped)
                            )
                        else:
                            logger.info(
                                "LLM unavailable; using rule-strip only"
                            )
                        rule_cleaned = strip_filler_words(text)
                        if rule_cleaned != text:
                            logger.info(
                                "Rule-strip: %s -> %s", redact(text), redact(rule_cleaned)
                            )
                            text = rule_cleaned
                            self._overlay.update_text(text)
                    # Hide overlay BEFORE paste so it doesn't steal focus
                    time.sleep(0.05)
                    self._overlay.hide()
                    time.sleep(0.02)
                    injected = self._inject_text(text)
                    if injected:
                        # Only record a success when the text actually landed.
                        # Wrap bookkeeping so a stats/history write failure can
                        # never surface as a (false) transcription error: the
                        # paste already happened.
                        try:
                            self._settings.record_transcription(text)
                            elapsed = time.monotonic() - timer_start
                            self._history.add(
                                final_text=text,
                                raw_text=raw_for_history,
                                mode=self._active_mode.name,
                                duration=elapsed,
                                language=self._settings.get("languages", ["Auto"])[0],
                            )
                        except Exception:
                            logger.exception(
                                "Post-injection bookkeeping failed (paste already succeeded)"
                            )
                    else:
                        # Injection failed; _inject_text kept the transcript on
                        # the clipboard and showed an error. Don't count it as
                        # a successful transcription.
                        logger.warning("Injection failed; transcript preserved on clipboard")
                else:
                    self._overlay.update_text("(no speech detected)")
                    time.sleep(0.3)
                    self._overlay.hide()
            except Exception as e:
                timer_stop.set()
                logger.exception("Transcription error")
                self._overlay.set_status("error")
                self._overlay.update_text(_friendly_error(e))
                time.sleep(1.6)
                self._overlay.hide()
            finally:
                self._set_state(STATE_IDLE)
                self._update_status(self._idle_status_text())

        t = threading.Thread(target=transcribe, daemon=True)
        t.start()

    def _mode_allows_translation(self) -> bool:
        """Whether the active mode is allowed to change the text's language.

        Translation modes must bypass llm_cleanup's script-change guards;
        every other mode must keep them so a misbehaving model can't replace
        the user's Chinese with English under a "make it formal" prompt.
        """
        mode = self._active_mode
        if mode is None:
            return False
        if mode.translation_language:
            return True
        template = (mode.prompt_template or "").lower()
        return "translat" in template or "翻译" in template

    # Below this RMS peak the stream is effectively digital silence (muted
    # mic, denied permission, dead device); normal quiet speech sits well
    # above it. Used for the "no audio" hint during recording.
    _SILENCE_PEAK_THRESHOLD = 0.005
    _SILENCE_HINT_AFTER_SEC = 5.0

    def _start_speculative_timer(self):
        """Periodically transcribe the captured audio while recording.

        Runs for every recording: each pass shows the partial transcription
        in the overlay (live preview) or a silent-mic hint, and when a custom
        mode + LLM are active it also pre-runs the cleanup speculatively so
        the final result is often cache-hit instant.
        """
        self._speculative_timer_stop = threading.Event()

        def _speculate():
            while not self._speculative_timer_stop.wait(self._speculative_interval):
                if self._state != STATE_LISTENING:
                    break
                with self._audio_lock:
                    if not self._audio_chunks:
                        continue
                    audio_so_far = np.concatenate(self._audio_chunks)
                elapsed = time.monotonic() - self._listen_started
                if self._session_peak_level < self._SILENCE_PEAK_THRESHOLD:
                    # Nothing but silence so far. Don't run ASR on it (the
                    # model hallucinates on noise); after a few seconds tell
                    # the user the mic looks dead.
                    if elapsed >= self._SILENCE_HINT_AFTER_SEC:
                        self._overlay.update_text(
                            "No audio detected. Is the mic muted?"
                        )
                    continue
                if len(audio_so_far) < 16000:  # less than 1 second
                    continue
                try:
                    cleaned = audio_preprocess.normalize_audio(audio_so_far)
                    text, _ = self._asr.transcribe(cleaned)
                    if not text.strip():
                        continue
                    text = self._vocabulary.apply_snippets(text)
                    # Live preview of what has been heard so far. Re-check
                    # state: the user may have released the key mid-pass and
                    # the overlay now shows "Processing...".
                    if self._state == STATE_LISTENING:
                        self._overlay.update_text(text)
                    if self._active_mode.prompt_template and self._llm.is_available():
                        prompt = self._active_mode.render_prompt(text.strip())
                        self._llm.speculative_cleanup(
                            text.strip(), custom_prompt=prompt,
                            allow_script_change=self._mode_allows_translation(),
                        )
                except Exception as e:
                    logger.debug("Speculative ASR failed: %s", e)

        threading.Thread(target=_speculate, daemon=True).start()

    def _stop_speculative_timer(self):
        if hasattr(self, '_speculative_timer_stop'):
            self._speculative_timer_stop.set()

    def _inject_text(self, text):
        """Inject transcribed text into the active application.

        Returns True on success. On any failure the transcript is preserved on
        the clipboard (so it is never silently lost) and a visible error is
        shown on the overlay, which the caller has already hidden.
        """
        self._set_state(STATE_INJECTING)
        badge = LANGUAGES[self._language_index]["badge"]

        if not TextInjector.check_accessibility_permission():
            print("[SafeVoice] Accessibility permission not granted!")
            self._injector.copy_to_clipboard(text)
            self._overlay.show(language=badge)
            self._overlay.set_status("error")
            self._overlay.update_text("Grant Accessibility permission. Text copied to clipboard.")
            time.sleep(2.5)
            self._overlay.hide()
            return False

        success = self._injector.inject(text)
        if success:
            print(f"[SafeVoice] Injected {len(text)} chars")
            # Brief success flash. The overlay is hidden BEFORE the paste to
            # avoid stealing focus, which left the user with no completion
            # signal at all; the wizard's Demo step even promises this state.
            # The overlay panel is non-activating, so re-showing it cannot
            # take focus from the app that just received the text.
            self._overlay.show(language=badge)
            self._overlay.set_status("done")
            self._overlay.update_text("Pasted")
            time.sleep(0.35)
            self._overlay.hide()
        else:
            print(f"[SafeVoice] Injection failed ({len(text)} chars; kept on clipboard)")
            # Re-copy so the changeCount guard skips the injector's restore and
            # the transcript stays on the clipboard for manual paste.
            self._injector.copy_to_clipboard(text)
            self._overlay.show(language=badge)
            self._overlay.set_status("error")
            self._overlay.update_text("Paste failed. Text copied to clipboard.")
            time.sleep(2.0)
            self._overlay.hide()
        return success

    def _on_quit(self, _):
        """Clean up and quit."""
        print("[SafeVoice] Quitting...")
        if self._app_nap_token is not None:
            NSProcessInfo.processInfo().endActivity_(self._app_nap_token)
            self._app_nap_token = None
        self._hotkey.stop()
        if self._audio.is_recording:
            self._audio.stop()
        self._overlay.cleanup()
        self._asr.unload_model()
        rumps.quit_application()


def main():
    """Entry point for SafeVoice."""
    print("=" * 50)
    print("SafeVoice - Voice Input for macOS")
    print("=" * 50)
    print()

    # Tighten the log file run.py created in a shared directory: default
    # umask leaves it world-readable. Resolve the path from the active
    # handler so it stays correct if the log location ever moves.
    for handler in logging.getLogger().handlers:
        log_path = getattr(handler, "baseFilename", None)
        if log_path:
            try:
                os.chmod(log_path, 0o600)
            except OSError:
                pass

    # Single-instance enforcement. If another SafeVoice.app is already
    # running, ask it to show its settings window (so the user gets visible
    # feedback that we noticed the duplicate launch) and exit ourselves.
    # Without this guard, two processes register competing CGEventTaps and
    # the ASR model loads twice, doubling RAM.
    if is_duplicate_and_signal_existing():
        return

    # Check model availability
    if not ASREngine.is_model_downloaded():
        print("The ASR model has not been downloaded yet.")
        print("Run the setup script first:")
        print("  python scripts/setup_model.py")
        print()
        print("Starting anyway (model will download on first use)...")
        print()

    # Check accessibility permission
    if not TextInjector.check_accessibility_permission():
        print("WARNING: Accessibility permission not granted.")
        print("SafeVoice needs Accessibility access to type text into other apps.")
        print("Go to: System Settings > Privacy & Security > Accessibility")
        print("Add and enable your terminal or Python executable.")
        print()

    print("Hotkey:")
    print("  Left Option       : Start/stop voice input (hold to talk)")
    print()
    print("Starting SafeVoice...")

    app = SafeVoiceApp()
    app.run()
