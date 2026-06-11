"""Tests for clipboard handling (uses a private named pasteboard, never the
general clipboard, so running tests cannot clobber the user's copy buffer)."""
from AppKit import NSPasteboard, NSPasteboardTypeString

from src.text_injector import TextInjector


def _scratch_pasteboard():
    return NSPasteboard.pasteboardWithName_("SafeVoiceTestPB")


def test_transient_write_marks_clipboard_manager_types():
    pb = _scratch_pasteboard()
    try:
        assert TextInjector._write_clipboard(pb, "hello", transient=True)
        types = list(pb.types())
        assert "org.nspasteboard.TransientType" in types
        assert "org.nspasteboard.ConcealedType" in types
        assert pb.stringForType_(NSPasteboardTypeString) == "hello"
    finally:
        pb.releaseGlobally()


def test_plain_write_has_no_transient_types():
    pb = _scratch_pasteboard()
    try:
        assert TextInjector._write_clipboard(pb, "rescue copy")
        types = list(pb.types())
        assert "org.nspasteboard.TransientType" not in types
        assert pb.stringForType_(NSPasteboardTypeString) == "rescue copy"
    finally:
        pb.releaseGlobally()
