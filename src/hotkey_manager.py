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


def describe_hotkey(config: Dict[str, Any]) -> str:
    """Public alias of _describe_config for UI labels (menubar, notifications)."""
    return _describe_config(config)


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
        prompt_if_untrusted: bool = True,
    ) -> None:
        """Start listening for global hotkeys.

        With ``prompt_if_untrusted=False`` the macOS Accessibility dialog is
        NOT triggered automatically; the tap is still created and the
        permission poll still recovers once trust is granted. Used on first
        run so the setup wizard can explain the permission BEFORE the system
        dialog appears, instead of the dialog racing ahead of the wizard.
        """
        with self._lock:
            if self._cg_tap is not None:
                logger.warning("HotkeyManager is already running; ignoring start()")
                return
            self._on_activate = on_activate
            self._on_deactivate = on_deactivate
            self._is_active = False
            self._activate_trigger_held = False

        self._start_cg_tap(prompt_if_untrusted=prompt_if_untrusted)

        logger.info(
            "HotkeyManager started (mode=%s). Activate: %s",
            self._mode,
            self._describe_activate(),
        )

    def stop(self) -> None:
        """Stop listening for hotkeys and clean up."""
        # Snapshot under the lock, invoke outside it: the deactivate callback
        # does slow work (stops audio, spawns the transcribe worker) and the
        # CGEventTap callback thread also contends for this lock; holding it
        # through the callback can stall the tap past its timeout.
        callback = None
        with self._lock:
            if self._is_active and self._on_deactivate is not None:
                callback = self._on_deactivate
            self._is_active = False
            self._activate_trigger_held = False
        if callback is not None:
            self._safe_invoke(callback)

        self._stop_cg_tap()
        logger.info("HotkeyManager stopped")

    def set_mode(self, mode: str) -> None:
        """Set hotkey mode ('push_to_talk' or 'toggle')."""
        if mode not in ("push_to_talk", "toggle"):
            raise ValueError(
                f"Unknown mode {mode!r}; expected 'push_to_talk' or 'toggle'"
            )
        callback = None
        with self._lock:
            if mode == self._mode:
                return
            if self._is_active and self._on_deactivate is not None:
                callback = self._on_deactivate
            self._mode = mode
            self._is_active = False
            self._activate_trigger_held = False
            logger.info("Hotkey mode changed to %s", mode)
        # Invoke outside the lock (see stop() for why).
        if callback is not None:
            self._safe_invoke(callback)

    def set_activate_hotkey(self, config: Dict[str, Any]) -> None:
        """Update the activation hotkey from a settings dict."""
        key_code, mod_mask = _parse_activate_hotkey(config)
        callback = None
        with self._lock:
            self._activate_key_code = key_code
            self._activate_mod_mask = mod_mask
            if self._is_active and self._on_deactivate is not None:
                callback = self._on_deactivate
            self._is_active = False
            self._activate_trigger_held = False
        # Invoke outside the lock (see stop() for why).
        if callback is not None:
            self._safe_invoke(callback)
        logger.info("Activate hotkey changed to %s", config)

    @property
    def is_running(self) -> bool:
        return self._cg_tap is not None

    # ------------------------------------------------------------------
    # CGEventTap (activation hotkey)
    # ------------------------------------------------------------------

    def request_accessibility_permission(self) -> bool:
        """Check Accessibility trust and trigger the system dialog if needed.

        Public: the setup wizard calls this from its Permissions step so the
        dialog appears with context. Returns True if already trusted.
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

    def _start_cg_tap(self, prompt_if_untrusted: bool = True):
        """Create a CGEventTap to monitor key and modifier events.

        If the process lacks Accessibility permission, a system dialog is
        shown (unless prompt_if_untrusted=False) and a background thread
        polls every 2 seconds until permission is granted, then (re)creates
        the tap automatically.
        """
        if prompt_if_untrusted:
            was_trusted = self.request_accessibility_permission()
        else:
            was_trusted = AXIsProcessTrusted()
            if not was_trusted:
                logger.info(
                    "Not trusted for Accessibility; deferring the system "
                    "dialog to the setup wizard (poll active)"
                )

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
            # Capture once: _stop_cg_tap (permission-poll thread) can null
            # self._cg_tap between the check and the call.
            tap = self._cg_tap
            if tap is not None:
                Quartz.CGEventTapEnable(tap, True)
            # While the tap was disabled we may have missed the key/modifier
            # RELEASE that normally clears the edge state. If we don't reset it,
            # _activate_trigger_held stays stuck True and every subsequent press
            # is ignored ("hotkey not responding") until the app restarts.
            self._reset_after_tap_reenable()
            return event

        flags = Quartz.CGEventGetFlags(event)

        if event_type == Quartz.kCGEventFlagsChanged:
            self._handle_flags_changed(flags)
        elif event_type == Quartz.kCGEventKeyDown:
            self._handle_key_down(flags, event)
        elif event_type == Quartz.kCGEventKeyUp:
            self._handle_key_up(event)

        return event

    def _reset_after_tap_reenable(self):
        """Clear stale activation edge-state after the tap is re-enabled.

        At idle (the common case) this is a harmless no-op. If the tap was
        disabled mid-recording, we can no longer trust the missed release, so
        we finalize the in-progress recording and return to a clean idle edge.
        """
        callback = None
        with self._lock:
            self._activate_trigger_held = False
            if self._is_active:
                self._is_active = False
                callback = self._on_deactivate
        if callback is not None:
            logger.info("Tap re-enabled while active; finalizing recording")
            self._safe_invoke(callback)

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
            accepted = self._safe_invoke(callback)
            if not accepted:
                # The app rejected the transition (busy/loading/mic failed).
                # Roll back the optimistic active flag so the NEXT press is a
                # fresh activate instead of a wasted resync press (toggle mode).
                with self._lock:
                    self._is_active = False

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
            accepted = self._safe_invoke(callback)
            if not accepted:
                # See _handle_flags_changed: roll back so a rejected activation
                # doesn't leave HotkeyManager and the app out of sync.
                with self._lock:
                    self._is_active = False

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
    def _safe_invoke(callback: Callable[[], None]) -> bool:
        """Invoke *callback* and report whether it accepted the transition.

        A callback may return ``False`` to signal it rejected the state
        change (e.g. the app was busy and could not start recording). Any
        other return value (including ``None``) counts as accepted. An
        exception counts as rejected so the caller can roll back.
        """
        try:
            result = callback()
            return result is not False
        except Exception:
            logger.exception("Unhandled exception in hotkey callback %r", callback)
            return False
