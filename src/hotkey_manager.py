"""
Global hotkey manager for macOS voice IME.

Uses Quartz CGEventTap for the activation hotkey (reliable modifier and
key detection without full Accessibility trust) and pynput for the
language-switch hotkey.  Both hotkeys are configurable at runtime via
settings dicts like ``{"key": "space", "modifiers": ["alt"]}``.

Requires macOS Accessibility permission for global key monitoring.
"""

import logging
import threading
from typing import Any, Callable, Dict, Optional

import Quartz
from ApplicationServices import AXIsProcessTrusted
from Foundation import NSDictionary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quartz modifier flag mapping
# ---------------------------------------------------------------------------

_MOD_NAME_TO_FLAG: Dict[str, int] = {
    "alt":   0x00080000,   # kCGEventFlagMaskAlternate
    "cmd":   0x00100000,   # kCGEventFlagMaskCommand
    "shift": 0x00020000,   # kCGEventFlagMaskShift
    "ctrl":  0x00040000,   # kCGEventFlagMaskControl
}

# macOS virtual key codes
_KEYNAME_TO_KEYCODE: Dict[str, int] = {
    "space": 49, "tab": 48, "return": 36, "escape": 53, "delete": 51,
    "up": 126, "down": 125, "left": 123, "right": 124,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96,
    "f6": 97, "f7": 98, "f8": 100, "f9": 101, "f10": 109,
    "f11": 103, "f12": 111,
    # Letters (macOS virtual key codes)
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6,
    "x": 7, "c": 8, "v": 9, "b": 11, "q": 12, "w": 13, "e": 14,
    "r": 15, "y": 16, "t": 17, "o": 31, "u": 32, "i": 34, "p": 35,
    "l": 37, "j": 38, "k": 40, "n": 45, "m": 46,
    # Digits
    "1": 18, "2": 19, "3": 20, "4": 21, "5": 23, "6": 22,
    "7": 26, "8": 28, "9": 25, "0": 29,
}

# Human-readable modifier symbols
_MOD_SYMBOLS: Dict[str, str] = {
    "alt": "\u2325", "cmd": "\u2318", "shift": "\u21e7", "ctrl": "\u2303",
}

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_activate_hotkey(config: Dict[str, Any]):
    """Parse activation hotkey config for CGEventTap.

    Returns (key_code_or_None, modifier_mask).
    key_code is None for modifier-only hotkeys.
    """
    if not config:
        return None, 0
    key_name = config.get("key", "")
    mods = config.get("modifiers", [])
    key_code = _KEYNAME_TO_KEYCODE.get(key_name) if key_name else None
    mod_mask = 0
    for mod in mods:
        mod_mask |= _MOD_NAME_TO_FLAG.get(mod, 0)
    return key_code, mod_mask


def _describe_config(config: Dict[str, Any]) -> str:
    """Return a human-readable description of a hotkey config dict."""
    if not config:
        return "None"
    parts = []
    for mod in config.get("modifiers", []):
        parts.append(_MOD_SYMBOLS.get(mod, mod.title()))
    key = config.get("key", "")
    if key:
        parts.append(key.title())
    return "+".join(parts) if parts else "None"


# ---------------------------------------------------------------------------
# HotkeyManager
# ---------------------------------------------------------------------------

class HotkeyManager:
    """Manages global hotkeys for voice activation and language switching.

    Uses Quartz CGEventTap for the activation hotkey (supports modifier-only
    hotkeys like Left Option as well as key+modifier combos like Alt+Space).
    Uses pynput for the language-switch hotkey (Ctrl+Space by default).

    Supports two activation modes:
      - push_to_talk: Activate on press, deactivate on release.
      - toggle: Each press toggles between active and inactive.
    """

    def __init__(self) -> None:
        self._on_activate: Optional[Callable[[], None]] = None
        self._on_deactivate: Optional[Callable[[], None]] = None

        # CGEventTap state (activation hotkey)
        self._cg_tap = None
        self._cg_loop_source = None
        self._cg_run_loop = None
        self._cg_thread: Optional[threading.Thread] = None

        self._mode: str = "push_to_talk"
        self._is_active: bool = False
        self._activate_trigger_held: bool = False

        # Activation hotkey config (CGEventTap)
        self._activate_key_code: Optional[int] = None   # None = modifier-only
        self._activate_mod_mask: int = 0
        # Track which modifier bits were active at activation (for release)
        self._activate_held_mods: int = 0

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        on_activate: Callable[[], None],
        on_deactivate: Callable[[], None],
    ) -> None:
        """Start listening for global hotkeys."""
        with self._lock:
            if self._cg_tap is not None:
                logger.warning("HotkeyManager is already running; ignoring start()")
                return
            self._on_activate = on_activate
            self._on_deactivate = on_deactivate
            self._is_active = False
            self._activate_trigger_held = False

        self._start_cg_tap()

        logger.info(
            "HotkeyManager started (mode=%s). Activate: %s",
            self._mode,
            self._describe_activate(),
        )

    def stop(self) -> None:
        """Stop listening for hotkeys and clean up."""
        with self._lock:
            if self._is_active and self._on_deactivate is not None:
                try:
                    self._on_deactivate()
                except Exception:
                    logger.exception("Error in on_deactivate during stop")
            self._is_active = False
            self._activate_trigger_held = False

        self._stop_cg_tap()
        logger.info("HotkeyManager stopped")

    def set_mode(self, mode: str) -> None:
        """Set hotkey mode ('push_to_talk' or 'toggle')."""
        if mode not in ("push_to_talk", "toggle"):
            raise ValueError(
                f"Unknown mode {mode!r}; expected 'push_to_talk' or 'toggle'"
            )
        with self._lock:
            if mode == self._mode:
                return
            if self._is_active and self._on_deactivate is not None:
                try:
                    self._on_deactivate()
                except Exception:
                    logger.exception("Error in on_deactivate during mode switch")
            self._mode = mode
            self._is_active = False
            self._activate_trigger_held = False
            logger.info("Hotkey mode changed to %s", mode)

    def set_activate_hotkey(self, config: Dict[str, Any]) -> None:
        """Update the activation hotkey from a settings dict."""
        key_code, mod_mask = _parse_activate_hotkey(config)
        with self._lock:
            self._activate_key_code = key_code
            self._activate_mod_mask = mod_mask
            if self._is_active and self._on_deactivate is not None:
                try:
                    self._on_deactivate()
                except Exception:
                    logger.exception("Error in on_deactivate during hotkey change")
            self._is_active = False
            self._activate_trigger_held = False
        logger.info("Activate hotkey changed to %s", config)

    @property
    def is_running(self) -> bool:
        return self._cg_tap is not None

    # ------------------------------------------------------------------
    # CGEventTap (activation hotkey)
    # ------------------------------------------------------------------

    def _request_accessibility_permission(self) -> bool:
        """Check Accessibility trust and trigger the system dialog if needed.

        Returns True if already trusted, False otherwise.
        """
        trusted = AXIsProcessTrusted()
        if trusted:
            logger.info("Process IS trusted for Accessibility")
            return True

        logger.warning(
            "Process is NOT trusted for Accessibility. "
            "Requesting permission via system dialog..."
        )
        try:
            import ctypes
            import ctypes.util
            import objc as _objc
            lib = ctypes.cdll.LoadLibrary(
                ctypes.util.find_library("ApplicationServices")
            )
            lib.AXIsProcessTrustedWithOptions.restype = ctypes.c_bool
            lib.AXIsProcessTrustedWithOptions.argtypes = [ctypes.c_void_p]
            opts = NSDictionary.dictionaryWithDictionary_(
                {"AXTrustedCheckOptionPrompt": True}
            )
            lib.AXIsProcessTrustedWithOptions(_objc.pyobjc_id(opts))
        except Exception:
            logger.exception("Failed to request Accessibility permission")
        return False

    def _create_cg_tap(self) -> bool:
        """Create and start the CGEventTap. Returns True on success."""
        event_mask = (
            (1 << Quartz.kCGEventFlagsChanged)
            | (1 << Quartz.kCGEventKeyDown)
            | (1 << Quartz.kCGEventKeyUp)
        )
        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            event_mask,
            self._cg_event_callback,
            None,
        )
        if tap is None:
            return False

        self._cg_tap = tap
        self._cg_loop_source = Quartz.CFMachPortCreateRunLoopSource(
            None, self._cg_tap, 0
        )

        def run_loop():
            loop = Quartz.CFRunLoopGetCurrent()
            self._cg_run_loop = loop
            Quartz.CFRunLoopAddSource(
                loop, self._cg_loop_source, Quartz.kCFRunLoopCommonModes
            )
            Quartz.CGEventTapEnable(self._cg_tap, True)
            logger.debug("CGEventTap run loop starting")
            Quartz.CFRunLoopRun()
            logger.debug("CGEventTap run loop exited")

        self._cg_thread = threading.Thread(target=run_loop, daemon=True)
        self._cg_thread.start()
        return True

    def _start_cg_tap(self):
        """Create a CGEventTap to monitor key and modifier events.

        If the process lacks Accessibility permission, a system dialog is
        shown and a background thread polls every 2 seconds until permission
        is granted, then (re)creates the tap automatically.
        """
        was_trusted = self._request_accessibility_permission()

        if was_trusted:
            if not self._create_cg_tap():
                logger.error(
                    "Failed to create CGEventTap even with Accessibility."
                )
            return

        # Not trusted. Try to create anyway (succeeds on some macOS versions
        # but events won't be delivered until permission is granted).
        created = self._create_cg_tap()
        if created:
            logger.info(
                "CGEventTap created without trust. "
                "Polling for Accessibility permission grant..."
            )
        else:
            logger.warning(
                "CGEventTap could not be created without Accessibility. "
                "Polling for permission grant..."
            )

        # Poll until permission is granted, then recreate the tap so events
        # start flowing.
        def poll_for_permission():
            import time
            while True:
                time.sleep(2.0)
                if AXIsProcessTrusted():
                    logger.info(
                        "Accessibility permission granted! "
                        "Recreating CGEventTap..."
                    )
                    # Stop existing non-functional tap
                    self._stop_cg_tap()
                    if self._create_cg_tap():
                        logger.info("CGEventTap recreated successfully")
                    else:
                        logger.error("Failed to recreate CGEventTap")
                    return

        threading.Thread(target=poll_for_permission, daemon=True).start()

    def _stop_cg_tap(self):
        if self._cg_tap is not None:
            Quartz.CGEventTapEnable(self._cg_tap, False)
            self._cg_tap = None
        if self._cg_run_loop is not None:
            Quartz.CFRunLoopStop(self._cg_run_loop)
            self._cg_run_loop = None
        self._cg_loop_source = None
        if self._cg_thread is not None:
            self._cg_thread.join(timeout=2.0)
            self._cg_thread = None

    def _cg_event_callback(self, proxy, event_type, event, refcon):
        """Handle CGEvent events for the activation hotkey."""
        # macOS auto-disables event taps after sleep/wake or a slow callback,
        # delivering kCGEventTapDisabledByTimeout / kCGEventTapDisabledByUserInput
        # regardless of the registered event mask. Without re-enabling the
        # tap here, the activation hotkey silently dies until the app restarts.
        if event_type in (
            Quartz.kCGEventTapDisabledByTimeout,
            Quartz.kCGEventTapDisabledByUserInput,
        ):
            logger.warning(
                "CGEventTap disabled (type=0x%x). Re-enabling.", event_type
            )
            if self._cg_tap is not None:
                Quartz.CGEventTapEnable(self._cg_tap, True)
            return event

        flags = Quartz.CGEventGetFlags(event)

        if event_type == Quartz.kCGEventFlagsChanged:
            self._handle_flags_changed(flags)
        elif event_type == Quartz.kCGEventKeyDown:
            self._handle_key_down(flags, event)
        elif event_type == Quartz.kCGEventKeyUp:
            self._handle_key_up(event)

        return event

    def _handle_flags_changed(self, flags: int):
        """Handle modifier-only activation hotkey via flagsChanged."""
        if self._activate_key_code is not None:
            # Not a modifier-only hotkey; keyDown/keyUp handles it.
            return

        if self._activate_mod_mask == 0:
            return

        mods_active = (flags & self._activate_mod_mask) == self._activate_mod_mask
        callback = None

        with self._lock:
            if mods_active and not self._activate_trigger_held:
                # Modifier pressed
                self._activate_trigger_held = True
                if self._mode == "push_to_talk":
                    self._is_active = True
                    callback = self._on_activate
                else:  # toggle
                    if self._is_active:
                        self._is_active = False
                        callback = self._on_deactivate
                    else:
                        self._is_active = True
                        callback = self._on_activate

            elif not mods_active and self._activate_trigger_held:
                # Modifier released
                self._activate_trigger_held = False
                if self._mode == "push_to_talk" and self._is_active:
                    self._is_active = False
                    callback = self._on_deactivate

        if callback is not None:
            action = "activate" if callback == self._on_activate else "deactivate"
            logger.info("Activation hotkey (modifier) -> %s", action)
            print(f"[SafeVoice] Hotkey {action}")
            self._safe_invoke(callback)

    def _handle_key_down(self, flags: int, event):
        """Handle key+modifier activation hotkey via keyDown."""
        if self._activate_key_code is None:
            return

        key_code = Quartz.CGEventGetIntegerValueField(
            event, Quartz.kCGKeyboardEventKeycode
        )
        if key_code != self._activate_key_code:
            return

        mods_ok = (
            self._activate_mod_mask == 0
            or (flags & self._activate_mod_mask) == self._activate_mod_mask
        )
        if not mods_ok:
            return

        callback = None
        with self._lock:
            if not self._activate_trigger_held:
                self._activate_trigger_held = True
                if self._mode == "push_to_talk":
                    self._is_active = True
                    callback = self._on_activate
                else:
                    if self._is_active:
                        self._is_active = False
                        callback = self._on_deactivate
                    else:
                        self._is_active = True
                        callback = self._on_activate

        if callback is not None:
            action = "activate" if callback == self._on_activate else "deactivate"
            logger.info("Activation hotkey (key) -> %s", action)
            print(f"[SafeVoice] Hotkey {action}")
            self._safe_invoke(callback)

    def _handle_key_up(self, event):
        """Handle key release for push_to_talk deactivation."""
        if self._activate_key_code is None:
            return

        key_code = Quartz.CGEventGetIntegerValueField(
            event, Quartz.kCGKeyboardEventKeycode
        )
        if key_code != self._activate_key_code:
            return

        callback = None
        with self._lock:
            if self._activate_trigger_held:
                self._activate_trigger_held = False
                if self._mode == "push_to_talk" and self._is_active:
                    self._is_active = False
                    callback = self._on_deactivate

        if callback is not None:
            logger.info("Activation hotkey -> deactivate (key released)")
            print("[SafeVoice] Hotkey deactivate")
            self._safe_invoke(callback)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _describe_activate(self) -> str:
        parts = []
        for name, flag in _MOD_NAME_TO_FLAG.items():
            if self._activate_mod_mask & flag:
                parts.append(_MOD_SYMBOLS.get(name, name))
        if self._activate_key_code is not None:
            # Reverse-lookup key name
            for name, code in _KEYNAME_TO_KEYCODE.items():
                if code == self._activate_key_code:
                    parts.append(name.title())
                    break
            else:
                parts.append(f"key{self._activate_key_code}")
        return "+".join(parts) if parts else "None"

    @staticmethod
    def _safe_invoke(callback: Callable[[], None]) -> None:
        try:
            callback()
        except Exception:
            logger.exception("Unhandled exception in hotkey callback %r", callback)
