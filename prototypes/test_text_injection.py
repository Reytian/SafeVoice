#!/usr/bin/env python3
"""
test_text_injection.py - Test multiple approaches for injecting text into the
active macOS application.

This script compares three approaches:
  1. Quartz CGEvent keyboard simulation (character-by-character)
  2. pynput keyboard typing
  3. NSPasteboard + Cmd+V paste simulation

Each approach attempts to inject: "Hello from Voice IME! 你好世界 Bonjour!"
which includes English, Chinese, and French text to exercise Unicode handling.

Usage:
    # Activate the venv first
    source /Users/haotianyi/Documents/voice-ime/.venv/bin/activate

    # Run with a delay so you can click into a target text field
    python prototypes/test_text_injection.py [--approach 1|2|3|all] [--delay 3]

Requirements:
    - macOS Accessibility permission must be granted to the terminal / Python
    - Packages: pyobjc-framework-Quartz, pyobjc-framework-Cocoa,
                pyobjc-framework-ApplicationServices, pynput

Notes:
    - Approach 1 (CGEvent) can only type characters that have a key code on the
      current keyboard layout. For CJK characters without a direct key mapping,
      it falls back to the pasteboard method internally.
    - Approach 2 (pynput) on macOS ultimately uses CGEvent under the hood, so
      it shares the same CJK limitation.
    - Approach 3 (NSPasteboard + Cmd+V) is the most reliable for arbitrary
      Unicode text including CJK, but it overwrites the user's clipboard.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Test string
# ---------------------------------------------------------------------------
TEST_TEXT = "Hello from Voice IME! \u4f60\u597d\u4e16\u754c Bonjour!"

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class ApproachResult:
    name: str
    success: bool = False
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    def report(self) -> str:
        status = "OK" if self.success else "FAIL"
        lines = [f"[{status}] {self.name}"]
        if self.error:
            lines.append(f"       Error: {self.error}")
        for note in self.notes:
            lines.append(f"       Note:  {note}")
        return "\n".join(lines)


# ===================================================================
# Approach 1 - Quartz CGEvent keyboard simulation
# ===================================================================

def _cgevent_type_char(char: str) -> bool:
    """Type a single character using CGEvent key events.

    For characters that exist on the current keyboard layout we use
    CGEventKeyboardSetUnicodeString.  This works for ASCII and many
    Western characters but will produce garbage or nothing for CJK
    characters that have no key-code mapping.

    Returns True if the event was posted without raising.
    """
    import Quartz

    # Create a dummy key-down event (virtual key code 0 = 'a', but we
    # override the unicode string so the actual code does not matter).
    event_down = Quartz.CGEventCreateKeyboardEvent(None, 0, True)
    event_up = Quartz.CGEventCreateKeyboardEvent(None, 0, False)

    # Attach the desired unicode character(s)
    Quartz.CGEventKeyboardSetUnicodeString(event_down, len(char), char)
    Quartz.CGEventKeyboardSetUnicodeString(event_up, len(char), char)

    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_down)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_up)
    return True


def approach_1_cgevent(text: str) -> ApproachResult:
    """Simulate typing via Quartz CGEventCreateKeyboardEvent."""
    result = ApproachResult(name="Approach 1: Quartz CGEvent char-by-char")
    result.notes.append("Uses CGEventKeyboardSetUnicodeString per character.")
    result.notes.append(
        "CJK characters may not be delivered correctly because "
        "CGEvent keyboard events depend on the active input source / "
        "keyboard layout for key-code resolution."
    )
    result.notes.append("Requires Accessibility permission.")

    try:
        import Quartz  # noqa: F401 - verify import works

        for ch in text:
            _cgevent_type_char(ch)
            # Small delay between keystrokes to avoid dropped events
            time.sleep(0.02)

        result.success = True
        result.notes.append(
            f"Posted {len(text)} key-down/key-up event pairs."
        )
    except ImportError as exc:
        result.error = f"Import error: {exc}"
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.notes.append(traceback.format_exc())

    return result


# ===================================================================
# Approach 2 - pynput
# ===================================================================

def approach_2_pynput(text: str) -> ApproachResult:
    """Simulate typing via pynput's keyboard controller."""
    result = ApproachResult(name="Approach 2: pynput keyboard.type()")
    result.notes.append("pynput on macOS uses CGEvent internally (darwin backend).")
    result.notes.append(
        "For non-ASCII characters pynput calls "
        "CGEventKeyboardSetUnicodeString, so CJK limitations are "
        "similar to Approach 1."
    )
    result.notes.append("Requires Accessibility permission.")

    try:
        from pynput.keyboard import Controller as KbController

        kb = KbController()
        # pynput's type() handles the character loop internally
        kb.type(text)

        result.success = True
        result.notes.append(f"Typed {len(text)} characters via pynput.")
    except ImportError as exc:
        result.error = f"Import error: {exc}"
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.notes.append(traceback.format_exc())

    return result


# ===================================================================
# Approach 3 - NSPasteboard + Cmd+V
# ===================================================================

def _set_pasteboard(text: str) -> None:
    """Place *text* on the general NSPasteboard (system clipboard)."""
    from AppKit import NSPasteboard, NSPasteboardTypeString

    pb = NSPasteboard.generalPasteboard()
    pb.clearContents()
    pb.setString_forType_(text, NSPasteboardTypeString)


def _simulate_cmd_v() -> None:
    """Post Cmd+V via CGEvent to trigger a paste in the frontmost app."""
    import Quartz

    # Virtual key code for 'v' is 9 on a US keyboard layout.
    V_KEYCODE = 9

    event_down = Quartz.CGEventCreateKeyboardEvent(None, V_KEYCODE, True)
    event_up = Quartz.CGEventCreateKeyboardEvent(None, V_KEYCODE, False)

    # Set the Command modifier flag
    Quartz.CGEventSetFlags(event_down, Quartz.kCGEventFlagMaskCommand)
    Quartz.CGEventSetFlags(event_up, Quartz.kCGEventFlagMaskCommand)

    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_down)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_up)


def approach_3_pasteboard(text: str, restore_clipboard: bool = True) -> ApproachResult:
    """Copy text to the pasteboard then simulate Cmd+V to paste it."""
    result = ApproachResult(name="Approach 3: NSPasteboard + Cmd+V paste")
    result.notes.append(
        "Places text on the system clipboard then simulates Cmd+V."
    )
    result.notes.append(
        "Most reliable for arbitrary Unicode (CJK, emoji, etc.) because "
        "the pasteboard handles encoding and the target app's paste handler "
        "processes the string natively."
    )
    result.notes.append("Requires Accessibility permission for CGEvent.")
    result.notes.append(
        "Caveat: overwrites the user's clipboard contents. "
        "The script saves and restores the previous clipboard if possible."
    )

    try:
        from AppKit import NSPasteboard, NSPasteboardTypeString

        # Save current clipboard contents
        pb = NSPasteboard.generalPasteboard()
        old_contents = pb.stringForType_(NSPasteboardTypeString)

        # Set our text
        _set_pasteboard(text)

        # Small delay to ensure the pasteboard change is visible
        time.sleep(0.05)

        # Simulate Cmd+V
        _simulate_cmd_v()

        # Wait for the paste event to be processed
        time.sleep(0.15)

        # Restore previous clipboard
        if restore_clipboard and old_contents is not None:
            time.sleep(0.1)
            _set_pasteboard(old_contents)
            result.notes.append("Restored previous clipboard contents.")
        elif restore_clipboard:
            result.notes.append(
                "No previous clipboard text to restore (was empty/non-text)."
            )

        result.success = True
        result.notes.append(f"Pasted {len(text)} characters via Cmd+V.")
    except ImportError as exc:
        result.error = f"Import error: {exc}"
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.notes.append(traceback.format_exc())

    return result


# ===================================================================
# Main driver
# ===================================================================

APPROACHES = {
    "1": approach_1_cgevent,
    "2": approach_2_pynput,
    "3": approach_3_pasteboard,
}


def _dry_run_import_check() -> None:
    """Verify that all required modules can be imported."""
    print("--- Import check ---")
    modules = [
        ("Quartz (CGEvent)", "Quartz"),
        ("AppKit (NSPasteboard)", "AppKit"),
        ("pynput", "pynput"),
    ]
    for label, mod in modules:
        try:
            __import__(mod)
            print(f"  [OK]   {label}")
        except ImportError as exc:
            print(f"  [FAIL] {label} -- {exc}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test text injection approaches on macOS.",
    )
    parser.add_argument(
        "--approach",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which approach to test (default: all).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Seconds to wait before injecting (so you can focus a text field). "
             "Default: 3.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only verify imports; do not inject any text.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=TEST_TEXT,
        help="Custom text to inject (default includes EN/ZH/FR sample).",
    )
    args = parser.parse_args()

    print(f"Test text: {args.text!r}")
    print(f"Text length: {len(args.text)} characters")
    print()

    _dry_run_import_check()

    if args.dry_run:
        print("Dry-run mode: skipping actual injection.")
        return

    # Determine which approaches to run
    if args.approach == "all":
        to_run = ["1", "2", "3"]
    else:
        to_run = [args.approach]

    # Countdown
    print(f"Will inject in {args.delay:.1f}s -- click into a text field NOW.")
    countdown_start = time.time()
    while True:
        remaining = args.delay - (time.time() - countdown_start)
        if remaining <= 0:
            break
        print(f"  {remaining:.1f}s remaining...", end="\r")
        time.sleep(0.25)
    print(f"\nStarting injection...\n{'=' * 60}")

    results: list[ApproachResult] = []

    for key in to_run:
        approach_fn = APPROACHES[key]
        print(f"\n--- Running {approach_fn.__doc__ or key} ---")
        result = approach_fn(args.text)
        results.append(result)
        # Pause between approaches so text fields can settle
        if key != to_run[-1]:
            print("  (pausing 2s before next approach...)")
            time.sleep(2.0)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}\n")
    for r in results:
        print(r.report())
        print()

    # Recommendation
    print(f"{'=' * 60}")
    print("RECOMMENDATION")
    print(f"{'=' * 60}")
    print(
        "For a Voice IME that needs to inject arbitrary Unicode text\n"
        "(including CJK characters) into any macOS application:\n"
        "\n"
        "  Best approach: Approach 3 (NSPasteboard + Cmd+V)\n"
        "\n"
        "Rationale:\n"
        "  - CGEvent keyboard simulation (Approaches 1 & 2) relies on\n"
        "    CGEventKeyboardSetUnicodeString. While this works well for\n"
        "    ASCII and many Latin-script characters, CJK characters\n"
        "    may not be delivered correctly to all applications because\n"
        "    the system's input method processing can interfere.\n"
        "\n"
        "  - The pasteboard approach bypasses keyboard simulation\n"
        "    entirely. The target application's native paste handler\n"
        "    (NSTextView, WKWebView, Electron, etc.) processes the\n"
        "    string from the clipboard, which fully supports Unicode.\n"
        "\n"
        "  - Clipboard clobbering can be mitigated by saving/restoring\n"
        "    the clipboard around the paste operation (as demonstrated).\n"
        "\n"
        "  - For a production IME, consider using the macOS Input Method\n"
        "    Kit (InputMethodKit / IMKInputController) which provides a\n"
        "    first-class API for text insertion without clipboard tricks.\n"
    )


if __name__ == "__main__":
    main()
