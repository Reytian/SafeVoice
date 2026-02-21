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
)
from Foundation import NSMakeRect, NSObject, NSSize
import objc

from .settings_manager import SettingsManager, SUPPORTED_LANGUAGES


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

    def __init__(self, settings_manager: SettingsManager) -> None:
        self._mgr = settings_manager
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
