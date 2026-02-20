"""
Text injection module for macOS voice IME.

Injects transcribed text into the active application using the
NSPasteboard + Cmd+V (paste) approach. This is the most reliable
method for inserting arbitrary Unicode text (including CJK characters)
into macOS applications.

Requires Accessibility permission for CGEvent keystroke simulation.
"""

import logging
import threading
import time
from typing import Optional

import Quartz
from AppKit import NSPasteboard, NSPasteboardTypeString
from ApplicationServices import AXIsProcessTrusted

logger = logging.getLogger(__name__)

# Virtual key code for the 'V' key on a standard US keyboard layout.
_KEYCODE_V = 9

# Delay in seconds before restoring the previous clipboard contents.
# Must be long enough for the target application to process the paste event.
_CLIPBOARD_RESTORE_DELAY = 0.5


class TextInjector:
    """Injects text into the currently focused macOS application.

    Uses the clipboard-paste strategy:
      1. Save the current clipboard contents.
      2. Place the desired text on the clipboard.
      3. Simulate Cmd+V via Quartz CGEvent.
      4. Restore the original clipboard after a brief delay.

    This class is thread-safe. Concurrent calls to ``inject`` are
    serialized so that clipboard save/restore cycles do not interleave.
    """

    def __init__(self, restore_delay: float = _CLIPBOARD_RESTORE_DELAY) -> None:
        self._restore_delay = restore_delay
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inject(self, text: str) -> bool:
        """Inject *text* into the active application.

        Returns ``True`` on success, ``False`` on failure.
        """
        if not text:
            logger.warning("inject() called with empty text; nothing to do.")
            return False

        with self._lock:
            return self._inject_locked(text)

    @staticmethod
    def check_accessibility_permission() -> bool:
        """Return ``True`` if this process has macOS Accessibility permission.

        Accessibility permission is required for posting synthetic
        keyboard events via ``CGEventPost``.  If this returns ``False``,
        the user must grant permission in:

            System Settings > Privacy & Security > Accessibility
        """
        try:
            trusted: bool = AXIsProcessTrusted()
            if not trusted:
                logger.warning(
                    "Accessibility permission not granted. "
                    "Enable it in System Settings > Privacy & Security > Accessibility."
                )
            return trusted
        except Exception:
            logger.exception("Failed to query Accessibility trust status.")
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inject_locked(self, text: str) -> bool:
        """Core injection logic (must be called while holding ``_lock``)."""

        pb = NSPasteboard.generalPasteboard()

        # --- 1. Save current clipboard --------------------------------
        saved_string = self._read_clipboard(pb)

        try:
            # --- 2. Write desired text to clipboard -------------------
            if not self._write_clipboard(pb, text):
                logger.error("Failed to write text to clipboard.")
                return False

            # Brief delay for pasteboard server to propagate
            time.sleep(0.05)

            # Record changeCount after our write so we can detect if
            # another application modifies the clipboard before we restore.
            change_count = pb.changeCount()

            # --- 3. Simulate Cmd+V ------------------------------------
            if not self._simulate_paste():
                logger.error("Failed to simulate Cmd+V keystroke.")
                # Attempt to restore clipboard even on failure.
                self._schedule_clipboard_restore(pb, saved_string, change_count)
                return False

            # --- 4. Restore clipboard after delay ---------------------
            self._schedule_clipboard_restore(pb, saved_string, change_count)
            return True

        except Exception:
            logger.exception("Unexpected error during text injection.")
            # Best-effort restore.
            self._schedule_clipboard_restore(pb, saved_string, None)
            return False

    # -- Clipboard helpers ---------------------------------------------

    @staticmethod
    def _read_clipboard(pb: NSPasteboard) -> Optional[str]:
        """Read the current plain-text string from *pb*, or ``None``."""
        try:
            return pb.stringForType_(NSPasteboardTypeString)
        except Exception:
            logger.debug("Could not read current clipboard contents.", exc_info=True)
            return None

    @staticmethod
    def _write_clipboard(pb: NSPasteboard, text: str) -> bool:
        """Replace clipboard contents with *text*. Returns success flag."""
        try:
            pb.clearContents()
            result = pb.setString_forType_(text, NSPasteboardTypeString)
            return bool(result)
        except Exception:
            logger.exception("Error writing to clipboard.")
            return False

    def _schedule_clipboard_restore(
        self,
        pb: NSPasteboard,
        saved_string: Optional[str],
        expected_change_count: Optional[int],
    ) -> None:
        """Restore *saved_string* to the clipboard after a short delay.

        Uses ``threading.Timer`` so we do not block the caller.
        *expected_change_count* is the changeCount recorded right after
        we wrote our text; if the clipboard has been modified by another
        app in the interim, the restore is skipped to avoid clobbering.
        """
        timer = threading.Timer(
            self._restore_delay,
            self._restore_clipboard,
            args=(pb, saved_string, expected_change_count),
        )
        timer.daemon = True
        timer.start()

    @staticmethod
    def _restore_clipboard(
        pb: NSPasteboard,
        saved_string: Optional[str],
        expected_change_count: Optional[int],
    ) -> None:
        """Write *saved_string* back to the clipboard.

        If *saved_string* is ``None`` (the clipboard was empty or held
        non-text data before injection), the clipboard is simply cleared.
        Skips the restore if the clipboard has been modified by another
        application since we wrote to it.
        """
        try:
            if expected_change_count is not None and pb.changeCount() != expected_change_count:
                logger.debug(
                    "Clipboard changed by another app; skipping restore."
                )
                return
            pb.clearContents()
            if saved_string is not None:
                pb.setString_forType_(saved_string, NSPasteboardTypeString)
            logger.debug("Clipboard contents restored.")
        except Exception:
            logger.debug("Failed to restore clipboard contents.", exc_info=True)

    # -- Keystroke simulation ------------------------------------------

    @staticmethod
    def _simulate_paste() -> bool:
        """Simulate a Cmd+V keystroke via Quartz CGEvent.

        Returns ``True`` if the events were created and posted without
        error.
        """
        try:
            # Create key-down and key-up events for 'V'.
            event_down = Quartz.CGEventCreateKeyboardEvent(None, _KEYCODE_V, True)
            event_up = Quartz.CGEventCreateKeyboardEvent(None, _KEYCODE_V, False)

            if event_down is None or event_up is None:
                logger.error("CGEventCreateKeyboardEvent returned None.")
                return False

            # Set the Command modifier flag on both events.
            Quartz.CGEventSetFlags(event_down, Quartz.kCGEventFlagMaskCommand)
            Quartz.CGEventSetFlags(event_up, Quartz.kCGEventFlagMaskCommand)

            # Post the events to the HID event tap so they reach the
            # currently focused application.
            Quartz.CGEventPost(Quartz.kCGSessionEventTap, event_down)
            Quartz.CGEventPost(Quartz.kCGSessionEventTap, event_up)

            return True
        except Exception:
            logger.exception("Error simulating Cmd+V keystroke.")
            return False
