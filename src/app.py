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
    NSProcessInfo,
)

logger = logging.getLogger(__name__)
import rumps

from .audio_capture import AudioCapture
from .asr_engine import ASREngine
from .text_injector import TextInjector
from .hotkey_manager import HotkeyManager
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
        )
        self._dashboard = DashboardWindow(
            self._settings,
            on_open_settings=lambda: self._settings_window.show(),
        )
        self._active_mode = self._modes.get("Quick")

        # Audio buffer for batch transcription
        self._audio_chunks = []
        self._audio_lock = threading.Lock()

        self._speculative_interval = 3.0  # seconds between speculative attempts

        # Apply saved settings before building menu
        self._apply_saved_settings()

        # Build menu
        self._build_menu()

        # Start hotkey listener
        self._hotkey.start(
            on_activate=self._on_hotkey_activate,
            on_deactivate=self._on_hotkey_deactivate,
        )

        # Preload model in background
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

        # Processing modes submenu
        self._modes_menu = rumps.MenuItem("Processing Mode")
        for mode in self._modes.get_all():
            hotkey_str = ""
            if mode.hotkey:
                mods = "+".join(m.title() for m in mode.hotkey.get("modifiers", []))
                key = mode.hotkey.get("key", "").title()
                hotkey_str = f" ({mods}+{key})" if mods else f" ({key})"
            item = rumps.MenuItem(f"{mode.name}{hotkey_str}")
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
        """Create a callback for language selection."""
        def callback(_):
            self._set_language(index)
        return callback

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
            "Hold Left ⌥ to record, release to transcribe.",
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
            "Press Left ⌥ once to start recording, again to stop.",
        )

    def _refresh_hotkey_info_label(self):
        """Update the menubar's Hotkey info line to match current mode.

        Hold-to-talk mode reads as "Hold Left ⌥"; toggle mode reads as
        "Press Left ⌥ to start, again to stop" so users understand the
        new behavior without checking docs. Called from __init__ (initial
        state) and both _set_*_mode handlers.
        """
        if not hasattr(self, "_hotkey_info_item"):
            return
        if self._mode == "toggle":
            self._hotkey_info_item.title = (
                "Hotkey: Press Left ⌥ to start, again to stop"
            )
        else:
            self._hotkey_info_item.title = "Hotkey: Hold Left ⌥ (Left Option)"

    def _start_model_load(self):
        """Load the ASR model in a background thread."""
        def load():
            try:
                logger.info("Background model load thread started")
                self._set_state(STATE_LOADING)
                self._asr.load_model()
                self._set_state(STATE_IDLE)
                self._update_status("Ready")
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
            except Exception as e:
                self._set_state(STATE_IDLE)
                self._update_status(f"Error: {e}")
                print(f"[SafeVoice] Model load failed: {e}")

        t = threading.Thread(target=load, daemon=True)
        t.start()

    def _show_setup_wizard(self):
        """Show setup wizard, dispatched to main thread for AppKit safety."""
        from AppKit import NSObject as _NSObj
        import objc as _objc

        app_ref = self

        class _WizardLauncher(_NSObj):
            def launchWizard_(self_, sender):
                def on_complete():
                    app_ref._settings.set("first_run", False)
                app_ref._wizard = SetupWizard(app_ref, on_complete=on_complete)
                app_ref._wizard.show()

        launcher = _WizardLauncher.alloc().init()
        self._wizard_launcher = launcher  # prevent GC
        launcher.performSelectorOnMainThread_withObject_waitUntilDone_(
            "launchWizard:", None, False
        )

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
                with open(cred_path) as f:
                    creds = json.load(f)
                cloud_api_key = creds.get(cloud_provider, "")
        except Exception:
            pass
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
        """Update app state and UI."""
        self._state = state
        # Show status text next to the icon; None = icon only
        titles = {
            STATE_IDLE: None,
            STATE_LOADING: "...",
            STATE_LISTENING: "REC",
            STATE_TRANSCRIBING: "...",
            STATE_INJECTING: "...",
        }
        self.title = titles.get(state)

    def _update_status(self, text):
        """Update the status menu item."""
        try:
            self._status_item.title = text
        except Exception:
            pass

    # -- Hotkey callbacks --

    def _on_hotkey_activate(self):
        """Called when the user activates voice input (hotkey pressed)."""
        try:
            logger.debug("Hotkey activate, state=%s", self._state)
            if self._state not in (STATE_IDLE,):
                return

            if not self._asr.is_loaded:
                self._update_status("Model loading...")
                return

            self._start_listening()
        except Exception:
            logger.exception("Error in _on_hotkey_activate")

    def _on_hotkey_deactivate(self):
        """Called when the user deactivates voice input (hotkey released)."""
        try:
            logger.debug("Hotkey deactivate, state=%s", self._state)
            if self._state != STATE_LISTENING:
                return

            self._stop_listening_and_transcribe()
        except Exception:
            logger.exception("Error in _on_hotkey_deactivate")

    # -- Recording flow --

    def _start_listening(self):
        """Start capturing audio."""
        self._set_state(STATE_LISTENING)
        self._update_status("Listening...")
        with self._audio_lock:
            self._audio_chunks.clear()

        lang = LANGUAGES[self._language_index]
        self._overlay.show(language=lang["badge"])

        # Start audio capture with callback (batch mode — no streaming ASR)
        self._audio.start(callback=self._on_audio_chunk)

        # Start speculative LLM if using a custom mode
        if self._active_mode.prompt_template and self._llm.is_available():
            self._start_speculative_timer()

        print("[SafeVoice] Listening...")

    def _on_audio_chunk(self, chunk):
        """Called for each audio chunk from the microphone."""
        with self._audio_lock:
            self._audio_chunks.append(chunk.copy())

        # Update overlay with audio level
        level = self._audio.get_level()
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

        # Stop audio capture
        full_audio = self._audio.stop()

        # Run transcription in background thread
        def transcribe():
            try:
                # Batch transcription on full audio for best accuracy
                if full_audio is None or len(full_audio) == 0:
                    text, lang = "", "Auto"
                else:
                    logger.info("Batch transcription on %d samples...", len(full_audio))
                    cleaned = audio_preprocess.normalize_audio(full_audio)
                    # Skip VAD trim — torch import takes 40-70s on first call
                    text, lang = self._asr.transcribe(cleaned)
                    logger.info("ASR result: %r (lang=%s)", text, lang)

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
                            cleaned = self._llm.cleanup(text, custom_prompt=prompt)
                            if cleaned != text:
                                text = cleaned
                        self._overlay.update_text(text)
                        logger.info("Mode result: %r", text)
                    elif self._active_mode.prompt_template is None and self._llm.is_available() and is_long_enough:
                        # Quick mode: default cleanup
                        self._overlay.set_status("processing")
                        self._update_status("Cleaning up...")
                        logger.info("LLM cleanup starting...")
                        cleaned = self._llm.cleanup(text)
                        logger.info("LLM result: %r", cleaned)
                        if cleaned != text:
                            print(f"[SafeVoice] LLM: {text!r} -> {cleaned!r}")
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
                                "Skipping LLM for short text: %r", stripped
                            )
                        else:
                            logger.info(
                                "LLM unavailable; using rule-strip only"
                            )
                        rule_cleaned = strip_filler_words(text)
                        if rule_cleaned != text:
                            logger.info(
                                "Rule-strip: %r -> %r", text, rule_cleaned
                            )
                            text = rule_cleaned
                            self._overlay.update_text(text)
                    # Hide overlay BEFORE paste so it doesn't steal focus
                    time.sleep(0.05)
                    self._overlay.hide()
                    time.sleep(0.02)
                    self._inject_text(text)
                    self._settings.record_transcription(text)
                    elapsed = time.monotonic() - timer_start
                    self._history.add(
                        final_text=text,
                        raw_text=raw_for_history,
                        mode=self._active_mode.name,
                        duration=elapsed,
                        language=self._settings.get("languages", ["Auto"])[0],
                    )
                else:
                    self._overlay.update_text("(no speech detected)")
                    time.sleep(0.3)
                    self._overlay.hide()
            except Exception as e:
                timer_stop.set()
                logger.exception("Transcription error")
                self._overlay.set_status("error")
                self._overlay.update_text(f"Error: {e}")
                time.sleep(1.0)
                self._overlay.hide()
            finally:
                self._set_state(STATE_IDLE)
                self._update_status("Ready")

        t = threading.Thread(target=transcribe, daemon=True)
        t.start()

    def _start_speculative_timer(self):
        """Periodically run ASR on captured audio and speculatively send to LLM."""
        self._speculative_timer_stop = threading.Event()

        def _speculate():
            while not self._speculative_timer_stop.wait(self._speculative_interval):
                if self._state != STATE_LISTENING:
                    break
                with self._audio_lock:
                    if not self._audio_chunks:
                        continue
                    audio_so_far = np.concatenate(self._audio_chunks)
                if len(audio_so_far) < 16000:  # less than 1 second
                    continue
                try:
                    cleaned = audio_preprocess.normalize_audio(audio_so_far)
                    text, _ = self._asr.transcribe(cleaned)
                    if text.strip():
                        text = self._vocabulary.apply_snippets(text)
                        prompt = self._active_mode.render_prompt(text.strip())
                        self._llm.speculative_cleanup(text.strip(), custom_prompt=prompt)
                except Exception as e:
                    logger.debug("Speculative ASR failed: %s", e)

        threading.Thread(target=_speculate, daemon=True).start()

    def _stop_speculative_timer(self):
        if hasattr(self, '_speculative_timer_stop'):
            self._speculative_timer_stop.set()

    def _inject_text(self, text):
        """Inject transcribed text into the active application."""
        self._set_state(STATE_INJECTING)

        if not TextInjector.check_accessibility_permission():
            print("[SafeVoice] Accessibility permission not granted!")
            self._overlay.update_text("Grant Accessibility permission in System Settings")
            time.sleep(2.0)
            return

        success = self._injector.inject(text)
        if success:
            print(f"[SafeVoice] Injected: {text}")
        else:
            print(f"[SafeVoice] Injection failed for: {text}")

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
