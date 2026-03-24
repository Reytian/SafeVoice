"""
Native macOS settings window for SafeVoice.

Provides a preferences GUI built with PyObjC (AppKit) featuring three tabs:
  - General: mode selection, response speed
  - Languages: scrollable language list with checkmarks
  - Advanced: hotkey recorder fields

The window is non-modal and follows macOS Human Interface Guidelines.
All UI mutations are dispatched to the main thread.
"""

import threading
from typing import Optional, Dict, Any, Callable

from AppKit import (
    NSWindow,
    NSWindowStyleMaskTitled,
    NSWindowStyleMaskClosable,
    NSBackingStoreBuffered,
    NSView,
    NSTextField,
    NSSecureTextField,
    NSColor,
    NSFont,
    NSScreen,
    NSApp,
    NSTabView,
    NSTabViewItem,
    NSButton,
    NSButtonTypeRadio,
    NSButtonTypeSwitch,
    NSScrollView,
    NSBezelStyleRounded,
    NSTextAlignmentCenter,
    NSOnState,
    NSOffState,
    NSPopUpButton,
    NSTextView,
    NSPanel,
)
from Foundation import NSMakeRect, NSObject, NSSize
import objc

from .settings_manager import SettingsManager, SUPPORTED_LANGUAGES
from .llm_backend import OllamaBackend, CLOUD_DEFAULTS


# ---------------------------------------------------------------------------
# Thread-dispatch helpers (same pattern as overlay.py)
# ---------------------------------------------------------------------------

def _ensure_main_thread(fn):
    """Decorator that dispatches the call to the main thread if necessary."""
    def wrapper(self, *args, **kwargs):
        if threading.current_thread() is threading.main_thread():
            fn(self, *args, **kwargs)
        else:
            self._dispatch_to_main(lambda: fn(self, *args, **kwargs))
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


class _SettingsTrampoline(NSObject):
    """NSObject trampoline for main-thread dispatch.

    A class-level set prevents premature garbage collection when dispatched
    asynchronously (waitUntilDone=False).
    """

    _prevent_gc: set = set()

    def initWithBlock_(self, block):
        self = objc.super(_SettingsTrampoline, self).init()
        if self is None:
            return None
        self._block = block
        return self

    def invoke(self):
        try:
            if self._block is not None:
                self._block()
        finally:
            _SettingsTrampoline._prevent_gc.discard(self)


class _SettingsCallbackTarget(NSObject):
    """NSObject target that invokes a Python callable (settings-specific)."""

    def initWithCallback_(self, callback):
        self = objc.super(_SettingsCallbackTarget, self).init()
        if self is None:
            return None
        self._callback = callback
        return self

    def invoke(self):
        if self._callback is not None:
            self._callback()


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

_WINDOW_WIDTH = 500.0
_WINDOW_HEIGHT = 400.0
_TAB_PADDING = 20.0
_ROW_HEIGHT = 28.0
_LABEL_WIDTH = 140.0
_CONTROL_X = 160.0

# Modifier display names (for hotkey recorder display).
_MODIFIER_SYMBOLS = {
    "alt": "\u2325",      # Option
    "ctrl": "\u2303",     # Control
    "shift": "\u21e7",    # Shift
    "cmd": "\u2318",      # Command
}

# Key display names for common keys.
_KEY_DISPLAY = {
    "space": "Space",
    "tab": "Tab",
    "return": "Return",
    "escape": "Esc",
    "delete": "Delete",
    "up": "\u2191",
    "down": "\u2193",
    "left": "\u2190",
    "right": "\u2192",
}


def _format_hotkey(hotkey_dict: Dict[str, Any]) -> str:
    """Format a hotkey dict like {"key": "space", "modifiers": ["alt"]}
    into a human-readable string like "Option + Space"."""
    if not hotkey_dict:
        return "Not set"
    parts = []
    for mod in hotkey_dict.get("modifiers", []):
        symbol = _MODIFIER_SYMBOLS.get(mod, mod.title())
        parts.append(symbol)
    key = hotkey_dict.get("key", "")
    if key:
        parts.append(_KEY_DISPLAY.get(key, key.title()))
    return " ".join(parts) if parts else "Not set"


# ---------------------------------------------------------------------------
# Hotkey Recorder Field (NSTextField subclass via delegate)
# ---------------------------------------------------------------------------

class _HotkeyRecorderDelegate(NSObject):
    """Delegate for an NSTextField acting as a hotkey recorder.

    When the field becomes first responder it shows "Press a key..." and
    captures the next key combination via the global monitor. The result
    is stored as a dict and displayed.
    """

    def initWithField_settingsKey_settingsManager_(self, field, key, mgr):
        self = objc.super(_HotkeyRecorderDelegate, self).init()
        if self is None:
            return None
        self._field = field
        self._settings_key = key
        self._manager = mgr
        self._recording = False
        self._monitor = None
        return self

    def controlTextDidBeginEditing_(self, notification):
        pass

    def startRecording(self):
        """Enter recording mode."""
        self._recording = True
        self._field.setStringValue_("Press a key...")
        self._field.setTextColor_(NSColor.systemOrangeColor())
        self._install_monitor()

    def stopRecording(self):
        """Exit recording mode."""
        self._recording = False
        self._field.setTextColor_(NSColor.labelColor())
        self._remove_monitor()

    def _install_monitor(self):
        """Install a local event monitor to capture the next key combo.

        Monitors both keyDown (for key + modifier combos) and flagsChanged
        (for modifier-only hotkeys like Left Option).  For modifier-only
        recording: when a modifier is pressed we remember it; if it is
        released without any intervening key press, we record it as a
        modifier-only hotkey.
        """
        from AppKit import NSEvent, NSKeyDownMask

        NSFlagsChangedMask = 1 << 12  # NSEventMaskFlagsChanged
        pending_mods = []  # Modifiers held since last flagsChanged

        def _extract_mods(flags):
            mods = []
            if flags & (1 << 20):  # NSEventModifierFlagCommand
                mods.append("cmd")
            if flags & (1 << 19):  # NSEventModifierFlagOption
                mods.append("alt")
            if flags & (1 << 17):  # NSEventModifierFlagShift
                mods.append("shift")
            if flags & (1 << 18):  # NSEventModifierFlagControl
                mods.append("ctrl")
            return mods

        def handler(event):
            nonlocal pending_mods
            event_type = event.type()

            # --- FlagsChanged (modifier press / release) ---
            if event_type == 12:  # NSFlagsChanged
                mods = _extract_mods(event.modifierFlags())
                if mods:
                    # Modifier pressed - remember for potential modifier-only hotkey
                    pending_mods = mods
                elif pending_mods:
                    # All modifiers released with no key in between -> modifier-only
                    hotkey = {"key": "", "modifiers": pending_mods}
                    pending_mods = []
                    self._manager.set(self._settings_key, hotkey)
                    self._field.setStringValue_(_format_hotkey(hotkey))
                    self.stopRecording()
                return None

            # --- KeyDown ---
            pending_mods = []  # Key was pressed, not a modifier-only hotkey

            mods = _extract_mods(event.modifierFlags())
            key_code = event.keyCode()
            chars = event.charactersIgnoringModifiers()

            # Map key code to a readable name
            key_name = self._key_code_to_name(key_code, chars)

            if key_name == "escape" and not mods:
                # Escape without modifiers cancels recording.
                current = self._manager.get(self._settings_key)
                self._field.setStringValue_(_format_hotkey(current))
                self.stopRecording()
                return None

            hotkey = {"key": key_name, "modifiers": mods}
            self._manager.set(self._settings_key, hotkey)
            self._field.setStringValue_(_format_hotkey(hotkey))
            self.stopRecording()
            return None

        self._monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
            NSKeyDownMask | NSFlagsChangedMask, handler
        )

    def _remove_monitor(self):
        if self._monitor is not None:
            from AppKit import NSEvent
            NSEvent.removeMonitor_(self._monitor)
            self._monitor = None

    @staticmethod
    def _key_code_to_name(code, chars):
        """Convert a macOS virtual key code to a descriptive name."""
        _CODE_MAP = {
            49: "space", 36: "return", 48: "tab", 53: "escape",
            51: "delete", 126: "up", 125: "down", 123: "left",
            124: "right", 122: "f1", 120: "f2", 99: "f3",
            118: "f4", 96: "f5", 97: "f6", 98: "f7",
            100: "f8", 101: "f9", 109: "f10", 103: "f11", 111: "f12",
        }
        if code in _CODE_MAP:
            return _CODE_MAP[code]
        if chars and len(chars) == 1:
            return chars.lower()
        return f"key{code}"


# ---------------------------------------------------------------------------
# SettingsWindow
# ---------------------------------------------------------------------------

class SettingsWindow:
    """Native macOS settings window with tabs for General, Languages, and
    Advanced (hotkey) configuration.

    Usage::

        win = SettingsWindow(settings_manager)
        win.show()
    """

    def __init__(self, settings_manager: SettingsManager, on_setting_changed=None, modes_manager=None, vocabulary_manager=None, on_llm_change=None) -> None:
        self._mgr = settings_manager
        self._on_llm_change = on_llm_change
        self._modes_manager = modes_manager
        self._vocabulary_manager = vocabulary_manager
        self._window: Optional[NSWindow] = None
        self._language_buttons: list = []
        self._hotkey_delegates: list = []  # prevent GC

        # Radio buttons / controls we need to read back
        self._mode_ptt_btn: Optional[NSButton] = None
        self._mode_toggle_btn: Optional[NSButton] = None
        self._speed_fast_btn: Optional[NSButton] = None
        self._speed_accurate_btn: Optional[NSButton] = None
        self._activate_hotkey_field: Optional[NSTextField] = None

        self._build_window()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @_ensure_main_thread
    def show(self) -> None:
        """Show (or bring to front) the settings window."""
        if self._window is None:
            self._build_window()
        self._sync_ui_from_settings()
        self._window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)

    @_ensure_main_thread
    def hide(self) -> None:
        """Hide the settings window."""
        if self._window is not None:
            self._window.orderOut_(None)

    @property
    def is_visible(self) -> bool:
        return self._window is not None and self._window.isVisible()

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------

    def _build_window(self) -> None:
        """Construct the NSWindow and its tab view."""
        screen = NSScreen.mainScreen()
        if screen is None:
            return

        screen_frame = screen.frame()
        x = (screen_frame.size.width - _WINDOW_WIDTH) / 2.0
        y = (screen_frame.size.height - _WINDOW_HEIGHT) / 2.0

        content_rect = NSMakeRect(x, y, _WINDOW_WIDTH, _WINDOW_HEIGHT)
        style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
        self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            content_rect, style, NSBackingStoreBuffered, False
        )
        self._window.setTitle_("SafeVoice Settings")
        self._window.setReleasedWhenClosed_(False)
        self._window.setMinSize_(NSSize(_WINDOW_WIDTH, _WINDOW_HEIGHT))
        self._window.setMaxSize_(NSSize(_WINDOW_WIDTH, _WINDOW_HEIGHT))

        # Tab view
        tab_rect = NSMakeRect(0, 0, _WINDOW_WIDTH, _WINDOW_HEIGHT)
        tab_view = NSTabView.alloc().initWithFrame_(tab_rect)

        # General tab
        general_item = NSTabViewItem.alloc().initWithIdentifier_("general")
        general_item.setLabel_("General")
        general_item.setView_(self._build_general_tab())
        tab_view.addTabViewItem_(general_item)

        # Languages tab
        lang_item = NSTabViewItem.alloc().initWithIdentifier_("languages")
        lang_item.setLabel_("Languages")
        lang_item.setView_(self._build_languages_tab())
        tab_view.addTabViewItem_(lang_item)

        # Advanced tab
        advanced_item = NSTabViewItem.alloc().initWithIdentifier_("advanced")
        advanced_item.setLabel_("Advanced")
        advanced_item.setView_(self._build_advanced_tab())
        tab_view.addTabViewItem_(advanced_item)

        # Modes tab
        modes_item = NSTabViewItem.alloc().initWithIdentifier_("modes")
        modes_item.setLabel_("Modes")
        modes_item.setView_(self._build_modes_tab())
        tab_view.addTabViewItem_(modes_item)

        # Vocabulary tab
        vocab_item = NSTabViewItem.alloc().initWithIdentifier_("vocab")
        vocab_item.setLabel_("Vocabulary")
        vocab_item.setView_(self._build_vocabulary_tab())
        tab_view.addTabViewItem_(vocab_item)

        # Models tab
        models_item = NSTabViewItem.alloc().initWithIdentifier_("models")
        models_item.setLabel_("Models")
        models_item.setView_(self._build_models_tab())
        tab_view.addTabViewItem_(models_item)

        self._window.contentView().addSubview_(tab_view)

    # ------------------------------------------------------------------
    # General tab
    # ------------------------------------------------------------------

    def _build_general_tab(self) -> NSView:
        """Build the General settings tab content."""
        view = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, _WINDOW_WIDTH, _WINDOW_HEIGHT - 60)
        )
        content_height = _WINDOW_HEIGHT - 60
        y = content_height - _TAB_PADDING

        # --- Mode section (own container so radio group is isolated) ---
        y -= _ROW_HEIGHT
        view.addSubview_(self._make_section_label("Mode", y + 4))

        mode_group_y = y - 2 * _ROW_HEIGHT
        mode_group = NSView.alloc().initWithFrame_(
            NSMakeRect(_CONTROL_X, mode_group_y, 280, 2 * _ROW_HEIGHT)
        )

        self._mode_ptt_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(0, _ROW_HEIGHT, 280, _ROW_HEIGHT)
        )
        self._mode_ptt_btn.setButtonType_(NSButtonTypeRadio)
        self._mode_ptt_btn.setTitle_("Push-to-Talk (hold hotkey)")
        self._mode_ptt_btn.setFont_(NSFont.systemFontOfSize_(13.0))
        ptt_target = self._make_mode_target("push_to_talk")
        self._mode_ptt_btn.setTarget_(ptt_target)
        self._mode_ptt_btn.setAction_("invoke")
        mode_group.addSubview_(self._mode_ptt_btn)

        self._mode_toggle_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(0, 0, 280, _ROW_HEIGHT)
        )
        self._mode_toggle_btn.setButtonType_(NSButtonTypeRadio)
        self._mode_toggle_btn.setTitle_("Toggle (press to start/stop)")
        self._mode_toggle_btn.setFont_(NSFont.systemFontOfSize_(13.0))
        toggle_target = self._make_mode_target("toggle")
        self._mode_toggle_btn.setTarget_(toggle_target)
        self._mode_toggle_btn.setAction_("invoke")
        mode_group.addSubview_(self._mode_toggle_btn)

        view.addSubview_(mode_group)
        y -= 2 * _ROW_HEIGHT

        y -= 10  # spacer

        # Mode description
        y -= _ROW_HEIGHT
        desc = self._make_label(
            "Push-to-Talk: hold the hotkey while speaking.\n"
            "Toggle: press once to start, press again to stop.",
            NSMakeRect(_CONTROL_X, y - 10, 300, 36),
            font_size=11.0,
            color=NSColor.secondaryLabelColor(),
        )
        view.addSubview_(desc)

        y -= 50  # spacer

        # --- Response speed section (own container so radio group is isolated) ---
        y -= _ROW_HEIGHT
        view.addSubview_(self._make_section_label("Response Speed", y + 4))

        speed_group_y = y - 2 * _ROW_HEIGHT
        speed_group = NSView.alloc().initWithFrame_(
            NSMakeRect(_CONTROL_X, speed_group_y, 280, 2 * _ROW_HEIGHT)
        )

        self._speed_fast_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(0, _ROW_HEIGHT, 280, _ROW_HEIGHT)
        )
        self._speed_fast_btn.setButtonType_(NSButtonTypeRadio)
        self._speed_fast_btn.setTitle_("Fast (lower latency)")
        self._speed_fast_btn.setFont_(NSFont.systemFontOfSize_(13.0))
        fast_target = self._make_action_target("response_speed", "fast")
        self._speed_fast_btn.setTarget_(fast_target)
        self._speed_fast_btn.setAction_("invoke")
        speed_group.addSubview_(self._speed_fast_btn)

        self._speed_accurate_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(0, 0, 280, _ROW_HEIGHT)
        )
        self._speed_accurate_btn.setButtonType_(NSButtonTypeRadio)
        self._speed_accurate_btn.setTitle_("Accurate (better quality)")
        self._speed_accurate_btn.setFont_(NSFont.systemFontOfSize_(13.0))
        accurate_target = self._make_action_target("response_speed", "accurate")
        self._speed_accurate_btn.setTarget_(accurate_target)
        self._speed_accurate_btn.setAction_("invoke")
        speed_group.addSubview_(self._speed_accurate_btn)

        view.addSubview_(speed_group)
        y -= 2 * _ROW_HEIGHT

        y -= 10
        y -= _ROW_HEIGHT
        speed_desc = self._make_label(
            "Fast mode reduces delay. Accurate mode improves transcription quality.",
            NSMakeRect(_CONTROL_X, y - 2, 300, 30),
            font_size=11.0,
            color=NSColor.secondaryLabelColor(),
        )
        view.addSubview_(speed_desc)

        return view

    def _make_mode_target(self, mode_value: str):
        """Create a target that sets mode and updates radio button states."""
        def callback():
            self._mgr.set("mode", mode_value)
            if self._mode_ptt_btn and self._mode_toggle_btn:
                self._mode_ptt_btn.setState_(
                    NSOnState if mode_value == "push_to_talk" else NSOffState
                )
                self._mode_toggle_btn.setState_(
                    NSOnState if mode_value == "toggle" else NSOffState
                )

        target = _SettingsCallbackTarget.alloc().initWithCallback_(callback)
        self._hotkey_delegates.append(target)
        return target

    def _make_action_target(self, settings_key: str, value: Any):
        """Create an NSObject target that sets a settings key to a value."""
        def callback():
            self._mgr.set(settings_key, value)
            self._sync_ui_from_settings()

        target = _SettingsCallbackTarget.alloc().initWithCallback_(callback)
        self._hotkey_delegates.append(target)
        return target

    # ------------------------------------------------------------------
    # Languages tab
    # ------------------------------------------------------------------

    def _build_languages_tab(self) -> NSView:
        """Build the Languages tab with a scrollable list of checkmark rows."""
        container = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, _WINDOW_WIDTH, _WINDOW_HEIGHT - 60)
        )
        content_height = _WINDOW_HEIGHT - 60

        # Header label
        header_y = content_height - _TAB_PADDING - _ROW_HEIGHT
        container.addSubview_(self._make_section_label("Select Languages", header_y + 4))

        # Scroll view for the language list
        scroll_top = header_y - 8
        scroll_height = scroll_top - _TAB_PADDING
        scroll_view = NSScrollView.alloc().initWithFrame_(
            NSMakeRect(_TAB_PADDING, _TAB_PADDING, _WINDOW_WIDTH - 2 * _TAB_PADDING, scroll_height)
        )
        scroll_view.setHasVerticalScroller_(True)
        scroll_view.setHasHorizontalScroller_(False)
        scroll_view.setBorderType_(1)  # NSBezelBorder

        # Document view containing language rows
        row_count = len(SUPPORTED_LANGUAGES)
        doc_height = max(row_count * _ROW_HEIGHT + 10, scroll_height)
        doc_view = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, _WINDOW_WIDTH - 2 * _TAB_PADDING - 15, doc_height)
        )

        self._language_buttons = []
        selected_langs = self._mgr.get("languages", ["Auto"])

        for i, lang in enumerate(SUPPORTED_LANGUAGES):
            # Rows are laid out top-to-bottom; in AppKit y=0 is at the bottom,
            # so the first item gets the highest y.
            row_y = doc_height - (i + 1) * _ROW_HEIGHT

            btn = NSButton.alloc().initWithFrame_(
                NSMakeRect(10, row_y, _WINDOW_WIDTH - 2 * _TAB_PADDING - 40, _ROW_HEIGHT)
            )
            btn.setButtonType_(NSButtonTypeSwitch)
            btn.setTitle_(lang["display"])
            btn.setFont_(NSFont.systemFontOfSize_(13.0))
            btn.setState_(NSOnState if lang["name"] in selected_langs else NSOffState)
            btn.setTarget_(self._make_language_target(i))
            btn.setAction_("invoke")
            doc_view.addSubview_(btn)
            self._language_buttons.append(btn)

        scroll_view.setDocumentView_(doc_view)
        container.addSubview_(scroll_view)
        return container

    def _make_language_target(self, index: int):
        """Create an NSObject target that handles clicking a language row."""
        def callback():
            lang = SUPPORTED_LANGUAGES[index]
            current = list(self._mgr.get("languages", ["Auto"]))
            name = lang["name"]

            if name == "Auto":
                # Selecting Auto clears all specific languages
                current = ["Auto"]
            else:
                # Toggle specific language
                if name in current:
                    current.remove(name)
                else:
                    current.append(name)
                # Remove Auto when a specific language is selected
                if "Auto" in current:
                    current.remove("Auto")
                # If nothing selected, fall back to Auto
                if not current:
                    current = ["Auto"]

            self._mgr.set("languages", current)
            for j, btn in enumerate(self._language_buttons):
                btn.setState_(
                    NSOnState if SUPPORTED_LANGUAGES[j]["name"] in current else NSOffState
                )

        target = _SettingsCallbackTarget.alloc().initWithCallback_(callback)
        self._hotkey_delegates.append(target)
        return target

    # ------------------------------------------------------------------
    # Advanced tab (hotkey recorders)
    # ------------------------------------------------------------------

    def _build_advanced_tab(self) -> NSView:
        """Build the Advanced tab with hotkey recorder fields."""
        view = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, _WINDOW_WIDTH, _WINDOW_HEIGHT - 60)
        )
        content_height = _WINDOW_HEIGHT - 60
        y = content_height - _TAB_PADDING

        # --- Activate hotkey ---
        y -= _ROW_HEIGHT
        view.addSubview_(self._make_section_label("Hotkeys", y + 4))

        y -= _ROW_HEIGHT + 6
        view.addSubview_(self._make_label(
            "Voice Activation:",
            NSMakeRect(_TAB_PADDING, y, _LABEL_WIDTH, _ROW_HEIGHT),
            font_size=13.0,
        ))
        activate_container = self._make_hotkey_field(
            _CONTROL_X, y, "activate_hotkey"
        )
        view.addSubview_(activate_container)

        # Instructions
        y -= _ROW_HEIGHT + 20
        instructions = self._make_label(
            "Click a field and press any key to change the hotkey.\n"
            "Press Escape to cancel.",
            NSMakeRect(_TAB_PADDING, y - 10, _WINDOW_WIDTH - 2 * _TAB_PADDING, 40),
            font_size=11.0,
            color=NSColor.secondaryLabelColor(),
        )
        view.addSubview_(instructions)

        # --- Reset button ---
        y -= 70
        reset_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(_TAB_PADDING, y, 160, 32)
        )
        reset_btn.setTitle_("Reset to Defaults")
        reset_btn.setBezelStyle_(NSBezelStyleRounded)
        reset_target = self._make_reset_target()
        reset_btn.setTarget_(reset_target)
        reset_btn.setAction_("invoke")
        view.addSubview_(reset_btn)

        return view

    def _make_hotkey_field(self, x: float, y: float, settings_key: str) -> NSTextField:
        """Create a hotkey recorder text field."""
        field = NSTextField.alloc().initWithFrame_(
            NSMakeRect(x, y, 200, _ROW_HEIGHT)
        )
        current = self._mgr.get(settings_key, {})
        field.setStringValue_(_format_hotkey(current))
        field.setEditable_(False)
        field.setSelectable_(False)
        field.setFont_(NSFont.systemFontOfSize_(13.0))
        field.setAlignment_(NSTextAlignmentCenter)
        field.setWantsLayer_(True)
        field.layer().setCornerRadius_(4.0)
        field.layer().setBorderWidth_(1.0)
        field.layer().setBorderColor_(
            NSColor.separatorColor().CGColor()
        )

        # Create a click target to start recording
        delegate = _HotkeyRecorderDelegate.alloc().initWithField_settingsKey_settingsManager_(
            field, settings_key, self._mgr
        )
        self._hotkey_delegates.append(delegate)

        # Use a clickable button overlay to trigger recording
        click_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(x, y, 200, _ROW_HEIGHT)
        )
        click_btn.setTransparent_(True)
        click_btn.setTarget_(delegate)
        click_btn.setAction_("startRecording")

        # We return the field but also store the button. We need to add the
        # button to the same parent. We handle this by wrapping in a container.
        container = NSView.alloc().initWithFrame_(
            NSMakeRect(x, y, 200, _ROW_HEIGHT)
        )
        field.setFrame_(NSMakeRect(0, 0, 200, _ROW_HEIGHT))
        click_btn.setFrame_(NSMakeRect(0, 0, 200, _ROW_HEIGHT))
        container.addSubview_(field)
        container.addSubview_(click_btn)

        # Store field reference for later syncing
        if settings_key == "activate_hotkey":
            self._activate_hotkey_field = field
        elif settings_key == "language_hotkey":
            self._language_hotkey_field = field

        return container

    def _make_reset_target(self):
        """Create a target for the reset button."""
        def callback():
            self._mgr.reset_to_defaults()
            self._sync_ui_from_settings()

        target = _SettingsCallbackTarget.alloc().initWithCallback_(callback)
        self._hotkey_delegates.append(target)
        return target

    # ------------------------------------------------------------------
    # Modes tab
    # ------------------------------------------------------------------

    def _build_modes_tab(self) -> NSView:
        """Build the Modes tab with interactive edit/delete controls."""
        view = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, _WINDOW_WIDTH, _WINDOW_HEIGHT - 60)
        )
        self._modes_container = view
        self._populate_modes_view(view)
        return view

    def _populate_modes_view(self, view) -> None:
        """Populate (or re-populate) the modes tab contents."""
        for subview in list(view.subviews()):
            subview.removeFromSuperview()

        content_height = _WINDOW_HEIGHT - 60
        y = content_height - _TAB_PADDING

        y -= _ROW_HEIGHT
        view.addSubview_(self._make_section_label("Processing Modes", y + 4))
        y -= 30

        if self._modes_manager:
            for mode in self._modes_manager.get_all():
                hotkey_str = _format_hotkey(mode.hotkey) if mode.hotkey else "None"
                label = self._make_label(
                    f"{mode.name}  \u2014  {hotkey_str}",
                    NSMakeRect(_TAB_PADDING, y, 200, 20),
                    font_size=13.0,
                )
                view.addSubview_(label)

                btn_x = 220
                # Edit button (not for Quick mode which has no prompt)
                if mode.prompt_template is not None:
                    edit_btn = NSButton.alloc().initWithFrame_(
                        NSMakeRect(btn_x, y, 40, 20)
                    )
                    edit_btn.setTitle_("Edit")
                    edit_btn.setBezelStyle_(0)
                    edit_btn.setFont_(NSFont.systemFontOfSize_(11.0))
                    target = self._make_mode_edit_target(mode.name)
                    edit_btn.setTarget_(target)
                    edit_btn.setAction_("invoke")
                    view.addSubview_(edit_btn)
                    btn_x += 45

                # Delete button (custom modes only)
                if not mode.builtin:
                    del_btn = NSButton.alloc().initWithFrame_(
                        NSMakeRect(btn_x, y, 50, 20)
                    )
                    del_btn.setTitle_("Delete")
                    del_btn.setBezelStyle_(0)
                    del_btn.setFont_(NSFont.systemFontOfSize_(11.0))
                    target = self._make_mode_delete_target(mode.name)
                    del_btn.setTarget_(target)
                    del_btn.setAction_("invoke")
                    view.addSubview_(del_btn)

                # Prompt preview
                if mode.prompt_template:
                    preview = (
                        mode.prompt_template[:50] + "..."
                        if len(mode.prompt_template) > 50
                        else mode.prompt_template
                    )
                    plabel = self._make_label(
                        preview,
                        NSMakeRect(_TAB_PADDING + 10, y - 18, 290, 16),
                        font_size=10.0,
                        color=NSColor.tertiaryLabelColor(),
                    )
                    view.addSubview_(plabel)
                    y -= 48
                else:
                    y -= 32

        # Add Mode button
        y -= 10
        add_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(_TAB_PADDING, y, 120, 24)
        )
        add_btn.setTitle_("+ Add Mode")
        add_btn.setBezelStyle_(1)
        target = self._make_mode_edit_target(None)
        add_btn.setTarget_(target)
        add_btn.setAction_("invoke")
        view.addSubview_(add_btn)

    def _make_mode_edit_target(self, mode_name):
        """Create a target that opens the mode editor for the given mode."""
        target = _SettingsCallbackTarget.alloc().initWithCallback_(
            lambda: self._show_mode_editor(mode_name)
        )
        self._hotkey_delegates.append(target)
        return target

    def _make_mode_delete_target(self, mode_name):
        """Create a target that deletes a custom mode and refreshes."""
        def _delete():
            self._modes_manager.remove(mode_name)
            self._refresh_modes_tab()
        target = _SettingsCallbackTarget.alloc().initWithCallback_(_delete)
        self._hotkey_delegates.append(target)
        return target

    def _refresh_modes_tab(self) -> None:
        """Re-populate the modes tab after data changes."""
        if hasattr(self, '_modes_container') and self._modes_container:
            self._populate_modes_view(self._modes_container)

    def _show_mode_editor(self, mode_name=None):
        """Open a modal panel for creating or editing a processing mode."""
        from .modes import Mode, STYLE_PRESETS

        editing = self._modes_manager.get(mode_name) if mode_name else None

        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 420, 400),
            NSWindowStyleMaskTitled | NSWindowStyleMaskClosable,
            NSBackingStoreBuffered, False,
        )
        panel.setTitle_("Edit Mode" if editing else "New Mode")
        panel.center()
        panel.setLevel_(5)
        content = panel.contentView()
        y = 360

        # Name field
        content.addSubview_(self._make_label(
            "Name:", NSMakeRect(20, y, 60, 20), font_size=12.0))
        name_field = NSTextField.alloc().initWithFrame_(NSMakeRect(90, y, 300, 22))
        name_field.setStringValue_(editing.name if editing else "")
        name_field.setFont_(NSFont.systemFontOfSize_(12.0))
        if editing and editing.builtin:
            name_field.setEditable_(False)
        content.addSubview_(name_field)
        y -= 36

        # Style preset dropdown
        content.addSubview_(self._make_label(
            "Style:", NSMakeRect(20, y, 60, 20), font_size=12.0))
        preset_popup = NSPopUpButton.alloc().initWithFrame_pullsDown_(
            NSMakeRect(90, y, 300, 24), False)
        preset_popup.addItemWithTitle_("Custom")
        for pname in STYLE_PRESETS:
            preset_popup.addItemWithTitle_(pname.title())
        content.addSubview_(preset_popup)
        y -= 36

        # Prompt text area
        content.addSubview_(self._make_label(
            "Prompt:", NSMakeRect(20, y, 60, 20), font_size=12.0))
        scroll = NSScrollView.alloc().initWithFrame_(
            NSMakeRect(90, y - 80, 300, 100))
        scroll.setHasVerticalScroller_(True)
        scroll.setBorderType_(1)
        text_view = NSTextView.alloc().initWithFrame_(
            NSMakeRect(0, 0, 280, 100))
        text_view.setFont_(NSFont.systemFontOfSize_(11.0))
        if editing and editing.prompt_template:
            text_view.setString_(editing.prompt_template)
        else:
            text_view.setString_(STYLE_PRESETS.get("professional", ""))
        scroll.setDocumentView_(text_view)
        content.addSubview_(scroll)
        y -= 120

        # Preset change callback
        def on_preset_change():
            selected = preset_popup.titleOfSelectedItem().lower()
            if selected in STYLE_PRESETS:
                text_view.setString_(STYLE_PRESETS[selected])

        preset_target = _SettingsCallbackTarget.alloc().initWithCallback_(
            on_preset_change)
        self._hotkey_delegates.append(preset_target)
        preset_popup.setTarget_(preset_target)
        preset_popup.setAction_("invoke")

        # Target language dropdown
        content.addSubview_(self._make_label(
            "Translate to:", NSMakeRect(20, y, 80, 20), font_size=12.0))
        lang_popup = NSPopUpButton.alloc().initWithFrame_pullsDown_(
            NSMakeRect(110, y, 180, 24), False)
        lang_popup.addItemWithTitle_("None")
        for lang_info in SUPPORTED_LANGUAGES:
            if lang_info["name"] != "Auto":
                lang_popup.addItemWithTitle_(lang_info["name"])
        if editing and editing.translation_language:
            lang_popup.selectItemWithTitle_(editing.translation_language)
        content.addSubview_(lang_popup)
        y -= 40

        # Cancel button
        cancel_btn = NSButton.alloc().initWithFrame_(NSMakeRect(200, 20, 80, 30))
        cancel_btn.setTitle_("Cancel")
        cancel_btn.setBezelStyle_(0)
        cancel_target = _SettingsCallbackTarget.alloc().initWithCallback_(
            lambda: panel.close())
        self._hotkey_delegates.append(cancel_target)
        cancel_btn.setTarget_(cancel_target)
        cancel_btn.setAction_("invoke")
        content.addSubview_(cancel_btn)

        # Save button
        def _save():
            name = name_field.stringValue().strip()
            prompt = text_view.string().strip()
            if not name:
                return
            lang_title = lang_popup.titleOfSelectedItem()
            trans_lang = lang_title if lang_title != "None" else None
            new_mode = Mode(
                name=name,
                prompt_template=prompt if prompt else None,
                hotkey=editing.hotkey if editing else None,
                translation_language=trans_lang,
            )
            if editing and editing.builtin:
                # For built-in modes, update hotkey and prompt via manager
                # without going through add() which forces builtin=False
                existing = self._modes_manager.get(editing.name)
                if existing:
                    existing.prompt_template = new_mode.prompt_template
                    existing.translation_language = new_mode.translation_language
                    self._modes_manager._save()
            else:
                self._modes_manager.add(new_mode)
            self._refresh_modes_tab()
            panel.close()

        save_btn = NSButton.alloc().initWithFrame_(NSMakeRect(290, 20, 80, 30))
        save_btn.setTitle_("Save")
        save_btn.setBezelStyle_(1)
        save_btn.setKeyEquivalent_("\r")
        save_target = _SettingsCallbackTarget.alloc().initWithCallback_(_save)
        self._hotkey_delegates.append(save_target)
        save_btn.setTarget_(save_target)
        save_btn.setAction_("invoke")
        content.addSubview_(save_btn)

        self._mode_editor_panel = panel
        panel.makeKeyAndOrderFront_(None)

    # ------------------------------------------------------------------
    # Vocabulary tab
    # ------------------------------------------------------------------

    def _build_vocabulary_tab(self) -> NSView:
        """Build the Vocabulary tab with interactive add/delete controls."""
        view = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, _WINDOW_WIDTH, _WINDOW_HEIGHT - 60)
        )
        self._vocab_container = view
        self._populate_vocabulary_view(view)
        return view

    def _populate_vocabulary_view(self, view) -> None:
        """Populate (or re-populate) the vocabulary tab contents."""
        # Clear existing subviews
        for subview in list(view.subviews()):
            subview.removeFromSuperview()

        content_height = _WINDOW_HEIGHT - 60
        y = content_height - _TAB_PADDING

        # --- Hotwords section ---
        y -= _ROW_HEIGHT
        view.addSubview_(self._make_section_label("ASR Hotwords", y + 4))

        y -= 6
        hw_desc = self._make_label(
            "Improve recognition of proper nouns and technical terms",
            NSMakeRect(_TAB_PADDING, y - 12, _WINDOW_WIDTH - 2 * _TAB_PADDING, 16),
            font_size=11.0,
            color=NSColor.secondaryLabelColor(),
        )
        view.addSubview_(hw_desc)
        y -= 30

        if self._vocabulary_manager:
            for word in self._vocabulary_manager.get_hotwords():
                # Delete button
                del_btn = NSButton.alloc().initWithFrame_(
                    NSMakeRect(_TAB_PADDING, y, 24, 20)
                )
                del_btn.setTitle_("\u00d7")
                del_btn.setBezelStyle_(NSBezelStyleRounded)
                del_btn.setFont_(NSFont.systemFontOfSize_(12.0))
                del_target = self._make_vocab_delete_target("hotword", word)
                del_btn.setTarget_(del_target)
                del_btn.setAction_("invoke")
                view.addSubview_(del_btn)

                w_label = self._make_label(
                    word,
                    NSMakeRect(_TAB_PADDING + 30, y, 300, 20),
                    font_size=12.0,
                )
                view.addSubview_(w_label)
                y -= 24

        # Add hotword row: text field + [Add] button
        y -= 4
        self._hw_field = NSTextField.alloc().initWithFrame_(
            NSMakeRect(_TAB_PADDING, y, 280, 24)
        )
        self._hw_field.setFont_(NSFont.systemFontOfSize_(12.0))
        self._hw_field.setPlaceholderString_("New hotword...")
        view.addSubview_(self._hw_field)

        hw_add_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(_TAB_PADDING + 286, y, 60, 24)
        )
        hw_add_btn.setTitle_("Add")
        hw_add_btn.setBezelStyle_(NSBezelStyleRounded)
        hw_add_btn.setFont_(NSFont.systemFontOfSize_(12.0))
        hw_add_target = self._make_vocab_add_target("hotword")
        hw_add_btn.setTarget_(hw_add_target)
        hw_add_btn.setAction_("invoke")
        view.addSubview_(hw_add_btn)
        y -= 36

        # --- Snippets section ---
        y -= 6
        view.addSubview_(self._make_section_label("Snippets", y + 4))

        y -= 6
        sn_desc = self._make_label(
            "Auto-replace phrases after transcription",
            NSMakeRect(_TAB_PADDING, y - 12, _WINDOW_WIDTH - 2 * _TAB_PADDING, 16),
            font_size=11.0,
            color=NSColor.secondaryLabelColor(),
        )
        view.addSubview_(sn_desc)
        y -= 30

        if self._vocabulary_manager:
            for trigger, replacement in self._vocabulary_manager.get_snippets().items():
                # Delete button
                del_btn = NSButton.alloc().initWithFrame_(
                    NSMakeRect(_TAB_PADDING, y, 24, 20)
                )
                del_btn.setTitle_("\u00d7")
                del_btn.setBezelStyle_(NSBezelStyleRounded)
                del_btn.setFont_(NSFont.systemFontOfSize_(12.0))
                del_target = self._make_vocab_delete_target("snippet", trigger)
                del_btn.setTarget_(del_target)
                del_btn.setAction_("invoke")
                view.addSubview_(del_btn)

                s_label = self._make_label(
                    f'"{trigger}" \u2192 "{replacement}"',
                    NSMakeRect(_TAB_PADDING + 30, y, _WINDOW_WIDTH - 2 * _TAB_PADDING - 40, 20),
                    font_size=12.0,
                )
                view.addSubview_(s_label)
                y -= 24

        # Add snippet row: trigger field + replacement field + [Add] button
        y -= 4
        self._sn_trigger_field = NSTextField.alloc().initWithFrame_(
            NSMakeRect(_TAB_PADDING, y, 140, 24)
        )
        self._sn_trigger_field.setFont_(NSFont.systemFontOfSize_(12.0))
        self._sn_trigger_field.setPlaceholderString_("Trigger...")
        view.addSubview_(self._sn_trigger_field)

        self._sn_replacement_field = NSTextField.alloc().initWithFrame_(
            NSMakeRect(_TAB_PADDING + 146, y, 194, 24)
        )
        self._sn_replacement_field.setFont_(NSFont.systemFontOfSize_(12.0))
        self._sn_replacement_field.setPlaceholderString_("Replacement...")
        view.addSubview_(self._sn_replacement_field)

        sn_add_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(_TAB_PADDING + 346, y, 60, 24)
        )
        sn_add_btn.setTitle_("Add")
        sn_add_btn.setBezelStyle_(NSBezelStyleRounded)
        sn_add_btn.setFont_(NSFont.systemFontOfSize_(12.0))
        sn_add_target = self._make_vocab_add_target("snippet")
        sn_add_btn.setTarget_(sn_add_target)
        sn_add_btn.setAction_("invoke")
        view.addSubview_(sn_add_btn)

    def _refresh_vocabulary_tab(self) -> None:
        """Re-populate the vocabulary tab after data changes."""
        if hasattr(self, '_vocab_container') and self._vocab_container:
            self._populate_vocabulary_view(self._vocab_container)

    def _make_vocab_delete_target(self, kind: str, key: str):
        """Create a target that deletes a hotword or snippet and refreshes."""
        def _delete():
            if kind == "hotword":
                self._vocabulary_manager.remove_hotword(key)
            else:
                self._vocabulary_manager.remove_snippet(key)
            self._refresh_vocabulary_tab()
        target = _SettingsCallbackTarget.alloc().initWithCallback_(_delete)
        self._hotkey_delegates.append(target)
        return target

    def _make_vocab_add_target(self, kind: str):
        """Create a target that adds a hotword or snippet and refreshes."""
        def _add():
            if kind == "hotword":
                word = self._hw_field.stringValue().strip()
                if word:
                    self._vocabulary_manager.add_hotword(word)
                    self._hw_field.setStringValue_("")
            else:
                trigger = self._sn_trigger_field.stringValue().strip()
                replacement = self._sn_replacement_field.stringValue().strip()
                if trigger and replacement:
                    self._vocabulary_manager.add_snippet(trigger, replacement)
                    self._sn_trigger_field.setStringValue_("")
                    self._sn_replacement_field.setStringValue_("")
            self._refresh_vocabulary_tab()
        target = _SettingsCallbackTarget.alloc().initWithCallback_(_add)
        self._hotkey_delegates.append(target)
        return target

    # ------------------------------------------------------------------
    # Models tab
    # ------------------------------------------------------------------

    def _build_models_tab(self) -> NSView:
        """Build the Models tab with ASR info and local/cloud LLM selection."""
        view = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, _WINDOW_WIDTH, _WINDOW_HEIGHT - 60)
        )
        content_height = _WINDOW_HEIGHT - 60
        y = content_height - _TAB_PADDING

        # --- ASR section ---
        y -= _ROW_HEIGHT
        view.addSubview_(self._make_section_label("Speech Recognition (ASR)", y + 4))
        y -= _ROW_HEIGHT
        asr_info = self._make_label(
            "Qwen3-ASR-0.6B \u2014 ~1.2 GB \u2014 100% on-device",
            NSMakeRect(_TAB_PADDING + 10, y, _WINDOW_WIDTH - 2 * _TAB_PADDING, 20),
            font_size=12.0,
            color=NSColor.secondaryLabelColor(),
        )
        view.addSubview_(asr_info)
        y -= 20

        # --- LLM section ---
        y -= _ROW_HEIGHT
        view.addSubview_(self._make_section_label("Text Cleanup (LLM)", y + 4))
        y -= 6

        # Source toggle radio buttons
        current_source = self._mgr.get("llm_source", "local")

        source_group_y = y - 2 * _ROW_HEIGHT
        source_group = NSView.alloc().initWithFrame_(
            NSMakeRect(_TAB_PADDING + 10, source_group_y, 300, 2 * _ROW_HEIGHT)
        )

        self._llm_local_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(0, _ROW_HEIGHT, 160, _ROW_HEIGHT)
        )
        self._llm_local_btn.setButtonType_(NSButtonTypeRadio)
        self._llm_local_btn.setTitle_("Local (Ollama)")
        self._llm_local_btn.setFont_(NSFont.systemFontOfSize_(13.0))
        self._llm_local_btn.setState_(NSOnState if current_source == "local" else NSOffState)
        local_target = self._make_llm_source_target("local")
        self._llm_local_btn.setTarget_(local_target)
        self._llm_local_btn.setAction_("invoke")
        source_group.addSubview_(self._llm_local_btn)

        self._llm_cloud_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(0, 0, 160, _ROW_HEIGHT)
        )
        self._llm_cloud_btn.setButtonType_(NSButtonTypeRadio)
        self._llm_cloud_btn.setTitle_("Cloud API")
        self._llm_cloud_btn.setFont_(NSFont.systemFontOfSize_(13.0))
        self._llm_cloud_btn.setState_(NSOnState if current_source == "cloud" else NSOffState)
        cloud_target = self._make_llm_source_target("cloud")
        self._llm_cloud_btn.setTarget_(cloud_target)
        self._llm_cloud_btn.setAction_("invoke")
        source_group.addSubview_(self._llm_cloud_btn)

        view.addSubview_(source_group)
        y -= 2 * _ROW_HEIGHT + 10

        # --- Local panel ---
        self._local_panel = NSView.alloc().initWithFrame_(
            NSMakeRect(_TAB_PADDING + 10, y - 50, _WINDOW_WIDTH - 2 * _TAB_PADDING - 20, 50)
        )

        local_y = 26
        self._local_panel.addSubview_(self._make_label(
            "Model:", NSMakeRect(0, local_y, 50, 22), font_size=12.0))

        self._local_model_popup = NSPopUpButton.alloc().initWithFrame_pullsDown_(
            NSMakeRect(55, local_y, 200, 24), False)
        self._populate_local_models()
        self._local_panel.addSubview_(self._local_model_popup)

        refresh_btn = NSButton.alloc().initWithFrame_(NSMakeRect(260, local_y, 70, 24))
        refresh_btn.setTitle_("Refresh")
        refresh_btn.setBezelStyle_(NSBezelStyleRounded)
        refresh_btn.setFont_(NSFont.systemFontOfSize_(11.0))
        refresh_target = _SettingsCallbackTarget.alloc().initWithCallback_(
            self._populate_local_models)
        self._hotkey_delegates.append(refresh_target)
        refresh_btn.setTarget_(refresh_target)
        refresh_btn.setAction_("invoke")
        self._local_panel.addSubview_(refresh_btn)

        local_info = self._make_label(
            "Local models keep your data private. No internet required.",
            NSMakeRect(0, 0, 350, 18),
            font_size=10.0,
            color=NSColor.secondaryLabelColor(),
        )
        self._local_panel.addSubview_(local_info)
        view.addSubview_(self._local_panel)

        # --- Cloud panel ---
        self._cloud_panel = NSView.alloc().initWithFrame_(
            NSMakeRect(_TAB_PADDING + 10, y - 50, _WINDOW_WIDTH - 2 * _TAB_PADDING - 20, 50)
        )

        cloud_y = 26
        self._cloud_panel.addSubview_(self._make_label(
            "Provider:", NSMakeRect(0, cloud_y, 60, 22), font_size=12.0))

        self._cloud_provider_popup = NSPopUpButton.alloc().initWithFrame_pullsDown_(
            NSMakeRect(65, cloud_y, 100, 24), False)
        for provider in ("OpenAI", "Anthropic", "Google"):
            self._cloud_provider_popup.addItemWithTitle_(provider)
        current_provider = self._mgr.get("llm_cloud_provider", "openai")
        self._cloud_provider_popup.selectItemWithTitle_(current_provider.capitalize()
            if current_provider != "openai" else "OpenAI")
        self._cloud_panel.addSubview_(self._cloud_provider_popup)

        self._cloud_panel.addSubview_(self._make_label(
            "Model:", NSMakeRect(175, cloud_y, 45, 22), font_size=12.0))

        self._cloud_model_field = NSTextField.alloc().initWithFrame_(
            NSMakeRect(220, cloud_y, 130, 22))
        self._cloud_model_field.setFont_(NSFont.systemFontOfSize_(12.0))
        current_cloud_model = self._mgr.get("llm_cloud_model", "gpt-4o-mini")
        self._cloud_model_field.setStringValue_(current_cloud_model)
        self._cloud_panel.addSubview_(self._cloud_model_field)

        # API key row
        self._cloud_panel.addSubview_(self._make_label(
            "API Key:", NSMakeRect(0, 0, 55, 22), font_size=12.0))
        self._cloud_api_key_field = NSSecureTextField.alloc().initWithFrame_(
            NSMakeRect(65, 0, 285, 22))
        self._cloud_api_key_field.setFont_(NSFont.systemFontOfSize_(12.0))
        self._load_api_key(self._cloud_api_key_field, current_provider)
        self._cloud_panel.addSubview_(self._cloud_api_key_field)

        view.addSubview_(self._cloud_panel)

        # Show/hide panels based on current source
        self._local_panel.setHidden_(current_source != "local")
        self._cloud_panel.setHidden_(current_source != "cloud")

        y -= 56

        # Cloud privacy info (always visible below cloud panel position)
        self._cloud_info_label = self._make_label(
            "Cloud models may be more accurate, but text is sent to the provider.",
            NSMakeRect(_TAB_PADDING + 10, y - 20, _WINDOW_WIDTH - 2 * _TAB_PADDING - 20, 18),
            font_size=10.0,
            color=NSColor.secondaryLabelColor(),
        )
        self._cloud_info_label.setHidden_(current_source != "cloud")
        view.addSubview_(self._cloud_info_label)

        # Apply button
        apply_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(_WINDOW_WIDTH - _TAB_PADDING - 80, _TAB_PADDING, 70, 28)
        )
        apply_btn.setTitle_("Apply")
        apply_btn.setBezelStyle_(NSBezelStyleRounded)
        apply_btn.setFont_(NSFont.systemFontOfSize_(12.0))
        apply_target = _SettingsCallbackTarget.alloc().initWithCallback_(
            self._apply_models_settings)
        self._hotkey_delegates.append(apply_target)
        apply_btn.setTarget_(apply_target)
        apply_btn.setAction_("invoke")
        view.addSubview_(apply_btn)

        return view

    def _make_llm_source_target(self, source_value: str):
        """Create a target for LLM source radio buttons."""
        def callback():
            is_local = source_value == "local"
            self._llm_local_btn.setState_(NSOnState if is_local else NSOffState)
            self._llm_cloud_btn.setState_(NSOffState if is_local else NSOnState)
            self._local_panel.setHidden_(not is_local)
            self._cloud_panel.setHidden_(is_local)
            self._cloud_info_label.setHidden_(is_local)
        target = _SettingsCallbackTarget.alloc().initWithCallback_(callback)
        self._hotkey_delegates.append(target)
        return target

    def _populate_local_models(self):
        """Populate the local model dropdown from Ollama."""
        self._local_model_popup.removeAllItems()
        models = OllamaBackend.list_models()
        if not models:
            self._local_model_popup.addItemWithTitle_("(no models found)")
        else:
            for m in models:
                self._local_model_popup.addItemWithTitle_(m)
        # Select current setting
        current = self._mgr.get("llm_local_model", "qwen2.5:3b")
        if current in models:
            self._local_model_popup.selectItemWithTitle_(current)

    def _apply_models_settings(self):
        """Save all Models tab settings and invoke the change callback."""
        is_local = self._llm_local_btn.state() == NSOnState
        source = "local" if is_local else "cloud"
        self._mgr.set("llm_source", source)

        if is_local:
            model_title = self._local_model_popup.titleOfSelectedItem()
            if model_title and model_title != "(no models found)":
                self._mgr.set("llm_local_model", model_title)
        else:
            provider = self._cloud_provider_popup.titleOfSelectedItem().lower()
            if provider == "openai":
                provider = "openai"  # keep lowercase
            self._mgr.set("llm_cloud_provider", provider)
            self._mgr.set("llm_cloud_model",
                          self._cloud_model_field.stringValue().strip())
            api_key = self._cloud_api_key_field.stringValue().strip()
            self._save_api_key(provider, api_key)

        if self._on_llm_change:
            self._on_llm_change()

    def _load_api_key(self, field, provider: str) -> None:
        """Load an API key from ~/.config/safevoice/credentials.json."""
        import os, json
        cred_path = os.path.expanduser("~/.config/safevoice/credentials.json")
        try:
            if os.path.exists(cred_path):
                with open(cred_path) as f:
                    creds = json.load(f)
                key = creds.get(provider, "")
                field.setStringValue_(key)
        except Exception:
            pass

    def _save_api_key(self, provider: str, api_key: str) -> None:
        """Save an API key to ~/.config/safevoice/credentials.json with 0600 perms."""
        import os, json
        cred_dir = os.path.expanduser("~/.config/safevoice")
        cred_path = os.path.join(cred_dir, "credentials.json")
        os.makedirs(cred_dir, exist_ok=True)
        creds = {}
        try:
            if os.path.exists(cred_path):
                with open(cred_path) as f:
                    creds = json.load(f)
        except Exception:
            pass
        creds[provider] = api_key
        with open(cred_path, "w") as f:
            json.dump(creds, f, indent=2)
        os.chmod(cred_path, 0o600)

    # ------------------------------------------------------------------
    # Sync UI state from SettingsManager
    # ------------------------------------------------------------------

    def _sync_ui_from_settings(self) -> None:
        """Read all settings and update UI controls to match."""
        settings = self._mgr.get_all()

        # Mode
        mode = settings.get("mode", "push_to_talk")
        if self._mode_ptt_btn and self._mode_toggle_btn:
            self._mode_ptt_btn.setState_(NSOnState if mode == "push_to_talk" else NSOffState)
            self._mode_toggle_btn.setState_(NSOnState if mode == "toggle" else NSOffState)

        # Response speed
        speed = settings.get("response_speed", "fast")
        if self._speed_fast_btn and self._speed_accurate_btn:
            self._speed_fast_btn.setState_(NSOnState if speed == "fast" else NSOffState)
            self._speed_accurate_btn.setState_(NSOnState if speed == "accurate" else NSOffState)

        # Languages (multi-select)
        selected_langs = settings.get("languages", ["Auto"])
        for i, btn in enumerate(self._language_buttons):
            btn.setState_(
                NSOnState if SUPPORTED_LANGUAGES[i]["name"] in selected_langs else NSOffState
            )

        # Hotkeys
        activate = settings.get("activate_hotkey", {})
        if self._activate_hotkey_field:
            self._activate_hotkey_field.setStringValue_(_format_hotkey(activate))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_section_label(self, text: str, y: float) -> NSTextField:
        """Create a bold section header label."""
        label = NSTextField.alloc().initWithFrame_(
            NSMakeRect(_TAB_PADDING, y, _WINDOW_WIDTH - 2 * _TAB_PADDING, 20)
        )
        label.setStringValue_(text)
        label.setFont_(NSFont.boldSystemFontOfSize_(13.0))
        label.setTextColor_(NSColor.labelColor())
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        return label

    @staticmethod
    def _make_label(
        text: str,
        frame,
        font_size: float = 13.0,
        color=None,
    ) -> NSTextField:
        """Create a non-editable label."""
        label = NSTextField.alloc().initWithFrame_(frame)
        label.setStringValue_(text)
        label.setFont_(NSFont.systemFontOfSize_(font_size))
        label.setTextColor_(color or NSColor.labelColor())
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        return label

    @staticmethod
    def _make_radio(title: str, x: float, y: float, identifier: str) -> NSButton:
        """Create a radio button."""
        btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(x, y, 280, _ROW_HEIGHT)
        )
        btn.setButtonType_(NSButtonTypeRadio)
        btn.setTitle_(title)
        btn.setFont_(NSFont.systemFontOfSize_(13.0))
        return btn

    def _dispatch_to_main(self, block) -> None:
        """Dispatch a callable to the main thread."""
        trampoline = _SettingsTrampoline.alloc().initWithBlock_(block)
        _SettingsTrampoline._prevent_gc.add(trampoline)
        trampoline.performSelectorOnMainThread_withObject_waitUntilDone_(
            "invoke", None, False
        )
