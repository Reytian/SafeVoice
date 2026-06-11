"""
History window for SafeVoice.

Shows the most recent transcriptions from the sqlite HistoryStore with
per-row copy, CSV export, and clear. This is the recovery path when a paste
landed in the wrong window or an LLM rewrite went wrong: the text was always
in the database, it just had no UI until now.

Built with PyObjC (AppKit). All UI mutations are dispatched to the main
thread; ObjC helper classes are module-level on purpose (defining NSObject
subclasses inside functions raises objc.error on the second call).
"""

import logging
import os
import threading
from datetime import datetime
from typing import Optional

from AppKit import (
    NSAlert,
    NSAlertFirstButtonReturn,
    NSBackingStoreBuffered,
    NSButton,
    NSColor,
    NSFont,
    NSLineBreakByTruncatingTail,
    NSPasteboard,
    NSPasteboardTypeString,
    NSScrollView,
    NSTextField,
    NSView,
    NSWindow,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskTitled,
    NSApp,
    NSScreen,
)
from Foundation import NSMakeRect, NSObject, NSSize
import objc

from .history import HistoryStore

logger = logging.getLogger(__name__)

_WINDOW_WIDTH = 600.0
_WINDOW_HEIGHT = 460.0
_PADDING = 16.0
_ROW_HEIGHT = 44.0
_LIST_LIMIT = 50


class _HistoryTrampoline(NSObject):
    """Main-thread dispatch trampoline (module-level by necessity)."""

    _prevent_gc: set = set()

    def initWithBlock_(self, block):
        self = objc.super(_HistoryTrampoline, self).init()
        if self is None:
            return None
        self._block = block
        return self

    def invoke(self):
        try:
            if self._block is not None:
                self._block()
        finally:
            _HistoryTrampoline._prevent_gc.discard(self)


class _HistoryCallbackTarget(NSObject):
    """NSObject button target invoking a Python callable."""

    def initWithCallback_(self, callback):
        self = objc.super(_HistoryCallbackTarget, self).init()
        if self is None:
            return None
        self._callback = callback
        return self

    def invoke(self):
        if self._callback is not None:
            self._callback()


class _FlippedView(NSView):
    """NSView with a top-left origin so rows stack naturally top-down."""

    def isFlipped(self):
        return True


def _ensure_main_thread(fn):
    """Decorator: forward the call to the main thread when needed."""
    def wrapper(self, *args, **kwargs):
        if threading.current_thread() is threading.main_thread():
            fn(self, *args, **kwargs)
        else:
            trampoline = _HistoryTrampoline.alloc().initWithBlock_(
                lambda: fn(self, *args, **kwargs)
            )
            _HistoryTrampoline._prevent_gc.add(trampoline)
            trampoline.performSelectorOnMainThread_withObject_waitUntilDone_(
                "invoke", None, False
            )
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _format_timestamp(iso_ts: str) -> str:
    try:
        return datetime.fromisoformat(iso_ts).strftime("%b %d  %H:%M")
    except (ValueError, TypeError):
        return iso_ts or ""


class HistoryWindow:
    """Native window listing recent transcriptions with copy/export/clear."""

    def __init__(self, history: HistoryStore) -> None:
        self._history = history
        self._window: Optional[NSWindow] = None
        self._targets: list = []          # prevent GC of ObjC targets
        self._scroll: Optional[NSScrollView] = None
        self._count_label: Optional[NSTextField] = None
        self._status_label: Optional[NSTextField] = None
        self._build_window()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @_ensure_main_thread
    def show(self) -> None:
        """Show (or bring to front) the window with fresh content."""
        if self._window is None:
            self._build_window()
        self.refresh()
        self._window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)

    @_ensure_main_thread
    def refresh(self) -> None:
        """Reload entries from the store and rebuild the list."""
        if self._scroll is None:
            return
        try:
            entries = self._history.get_recent(limit=_LIST_LIMIT)
        except Exception:
            logger.exception("Could not load history")
            entries = []
        self._populate(entries)
        if self._count_label is not None:
            total = self._history.get_stats().get("total_transcriptions", len(entries))
            shown = len(entries)
            self._count_label.setStringValue_(
                f"Showing {shown} of {total:,}" if total > shown else f"{shown} entries"
            )

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------

    def _build_window(self) -> None:
        screen = NSScreen.mainScreen()
        if screen is None:
            return
        frame = screen.frame()
        x = (frame.size.width - _WINDOW_WIDTH) / 2.0
        y = (frame.size.height - _WINDOW_HEIGHT) / 2.0

        style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
        self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, _WINDOW_WIDTH, _WINDOW_HEIGHT), style,
            NSBackingStoreBuffered, False,
        )
        self._window.setTitle_("SafeVoice History")
        self._window.setReleasedWhenClosed_(False)
        self._window.setMinSize_(NSSize(_WINDOW_WIDTH, _WINDOW_HEIGHT))
        self._window.setMaxSize_(NSSize(_WINDOW_WIDTH, _WINDOW_HEIGHT))
        content = self._window.contentView()

        header_y = _WINDOW_HEIGHT - _PADDING - 24

        title = self._make_label(
            "Recent Transcriptions",
            NSMakeRect(_PADDING, header_y, 250, 24),
            font=NSFont.boldSystemFontOfSize_(16.0),
        )
        content.addSubview_(title)

        self._count_label = self._make_label(
            "",
            NSMakeRect(_PADDING, header_y - 18, 250, 16),
            font=NSFont.systemFontOfSize_(11.0),
            color=NSColor.secondaryLabelColor(),
        )
        content.addSubview_(self._count_label)

        export_btn = self._make_button(
            "Export CSV", NSMakeRect(_WINDOW_WIDTH - _PADDING - 200, header_y - 4, 100, 28),
            self._export_csv,
        )
        content.addSubview_(export_btn)

        clear_btn = self._make_button(
            "Clear...", NSMakeRect(_WINDOW_WIDTH - _PADDING - 92, header_y - 4, 92, 28),
            self._confirm_clear,
        )
        clear_btn.setContentTintColor_(NSColor.systemRedColor())
        content.addSubview_(clear_btn)

        # Scrolling list
        list_top = header_y - 30
        list_height = list_top - 40
        self._scroll = NSScrollView.alloc().initWithFrame_(
            NSMakeRect(_PADDING, 36, _WINDOW_WIDTH - 2 * _PADDING, list_height)
        )
        self._scroll.setHasVerticalScroller_(True)
        self._scroll.setBorderType_(1)  # NSLineBorder
        self._scroll.setDrawsBackground_(False)
        content.addSubview_(self._scroll)

        self._status_label = self._make_label(
            "Copy puts the text back on the clipboard.",
            NSMakeRect(_PADDING, 10, _WINDOW_WIDTH - 2 * _PADDING, 16),
            font=NSFont.systemFontOfSize_(11.0),
            color=NSColor.tertiaryLabelColor(),
        )
        content.addSubview_(self._status_label)

    def _populate(self, entries: list) -> None:
        """Rebuild the scroll document view from *entries*."""
        width = _WINDOW_WIDTH - 2 * _PADDING - 16  # minus scroller
        height = max(len(entries), 1) * _ROW_HEIGHT
        doc = _FlippedView.alloc().initWithFrame_(NSMakeRect(0, 0, width, height))

        if not entries:
            doc.addSubview_(self._make_label(
                "No transcriptions yet. Dictate something!",
                NSMakeRect(12, 12, width - 24, 20),
                color=NSColor.secondaryLabelColor(),
            ))
        for i, entry in enumerate(entries):
            row_y = i * _ROW_HEIGHT
            meta = _format_timestamp(entry.get("timestamp", ""))
            mode = entry.get("mode") or "Quick"
            lang = entry.get("language") or ""
            meta_text = f"{meta}   ·   {mode}" + (f"   ·   {lang}" if lang else "")
            doc.addSubview_(self._make_label(
                meta_text,
                NSMakeRect(10, row_y + 24, width - 100, 14),
                font=NSFont.systemFontOfSize_(10.0),
                color=NSColor.secondaryLabelColor(),
            ))

            text = (entry.get("final_text") or "").replace("\n", " ")
            text_label = self._make_label(
                text,
                NSMakeRect(10, row_y + 5, width - 100, 18),
                font=NSFont.systemFontOfSize_(12.0),
            )
            text_label.setLineBreakMode_(NSLineBreakByTruncatingTail)
            text_label.setSelectable_(True)
            text_label.setToolTip_(entry.get("final_text") or "")
            doc.addSubview_(text_label)

            copy_btn = self._make_button(
                "Copy",
                NSMakeRect(width - 76, row_y + 8, 64, 24),
                self._make_copy_callback(entry.get("final_text") or ""),
            )
            copy_btn.setFont_(NSFont.systemFontOfSize_(11.0))
            doc.addSubview_(copy_btn)

        self._scroll.setDocumentView_(doc)
        # Scroll to the top (newest entry).
        doc.scrollPoint_((0, 0))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _make_copy_callback(self, text: str):
        def _copy():
            try:
                pb = NSPasteboard.generalPasteboard()
                pb.clearContents()
                pb.setString_forType_(text, NSPasteboardTypeString)
                if self._status_label is not None:
                    preview = text if len(text) <= 60 else text[:57] + "..."
                    self._status_label.setStringValue_(f"Copied: {preview}")
            except Exception:
                logger.exception("Copy from history failed")
        return _copy

    def _export_csv(self):
        try:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = os.path.expanduser(f"~/Desktop/SafeVoice-History-{stamp}.csv")
            self._history.export_csv(path)
            if self._status_label is not None:
                self._status_label.setStringValue_(f"Exported to {path}")
        except Exception as e:
            logger.exception("History CSV export failed")
            if self._status_label is not None:
                self._status_label.setStringValue_(f"Export failed: {e}")

    def _confirm_clear(self):
        try:
            total = self._history.get_stats().get("total_transcriptions", 0)
            alert = NSAlert.alloc().init()
            alert.setMessageText_("Clear transcription history?")
            alert.setInformativeText_(
                f"All {total:,} entries will be permanently deleted. "
                "This cannot be undone."
            )
            alert.addButtonWithTitle_("Delete All")
            alert.addButtonWithTitle_("Cancel")
            if alert.runModal() == NSAlertFirstButtonReturn:
                if self._history.clear():
                    self.refresh()
                    if self._status_label is not None:
                        self._status_label.setStringValue_("History cleared.")
                elif self._status_label is not None:
                    self._status_label.setStringValue_("Could not clear history (see log).")
        except Exception:
            logger.exception("History clear failed")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_label(text, frame, font=None, color=None) -> NSTextField:
        label = NSTextField.alloc().initWithFrame_(frame)
        label.setStringValue_(text)
        label.setFont_(font or NSFont.systemFontOfSize_(13.0))
        label.setTextColor_(color or NSColor.labelColor())
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        return label

    def _make_button(self, title, frame, callback) -> NSButton:
        btn = NSButton.alloc().initWithFrame_(frame)
        btn.setTitle_(title)
        btn.setBezelStyle_(1)  # NSBezelStyleRounded
        btn.setFont_(NSFont.systemFontOfSize_(12.0))
        target = _HistoryCallbackTarget.alloc().initWithCallback_(callback)
        self._targets.append(target)
        btn.setTarget_(target)
        btn.setAction_("invoke")
        return btn
