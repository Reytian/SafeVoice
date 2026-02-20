"""
Main dashboard window for SafeVoice.

Displays usage statistics (total transcriptions, words, time saved, today's
activity) in a native macOS window with vibrancy background, quick language
switching, and a settings shortcut.

Built with PyObjC (AppKit). All UI mutations are dispatched to the main thread.
"""

import threading
from typing import Optional, Callable, List

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
    NSButton,
    NSBezelStyleRounded,
    NSTextAlignmentCenter,
    NSTextAlignmentLeft,
    NSVisualEffectView,
    NSVisualEffectMaterialWindowBackground,
    NSVisualEffectBlendingModeBehindWindow,
    NSVisualEffectStateActive,
    NSBox,
    NSBoxCustom,
    NSLineBreakByTruncatingTail,
)
from Foundation import NSMakeRect, NSObject, NSSize
import objc

from .settings_manager import SettingsManager, SUPPORTED_LANGUAGES


# ---------------------------------------------------------------------------
# Thread-dispatch helpers (same pattern as overlay.py / settings_window.py)
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


class _DashboardTrampoline(NSObject):
    """NSObject trampoline for main-thread dispatch.

    A class-level set prevents premature garbage collection when dispatched
    asynchronously (waitUntilDone=False).
    """

    _prevent_gc: set = set()

    def initWithBlock_(self, block):
        self = objc.super(_DashboardTrampoline, self).init()
        if self is None:
            return None
        self._block = block
        return self

    def invoke(self):
        try:
            if self._block is not None:
                self._block()
        finally:
            _DashboardTrampoline._prevent_gc.discard(self)


class _DashboardCallbackTarget(NSObject):
    """NSObject target that invokes a Python callable (dashboard-specific)."""

    def initWithCallback_(self, callback):
        self = objc.super(_DashboardCallbackTarget, self).init()
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
_WINDOW_HEIGHT = 450.0
_PADDING = 20.0
_CARD_HEIGHT = 70.0
_CARD_SPACING = 12.0
_CARD_CORNER_RADIUS = 10.0


# ---------------------------------------------------------------------------
# DashboardWindow
# ---------------------------------------------------------------------------

class DashboardWindow:
    """Native macOS dashboard window showing usage metrics and quick controls.

    Usage::

        dash = DashboardWindow(settings_manager, on_open_settings=callback)
        dash.show()
    """

    def __init__(
        self,
        settings_manager: SettingsManager,
        on_open_settings: Optional[Callable] = None,
        on_language_change: Optional[Callable[[int], None]] = None,
    ) -> None:
        self._mgr = settings_manager
        self._on_open_settings = on_open_settings
        self._on_language_change = on_language_change
        self._window: Optional[NSWindow] = None
        self._targets: list = []  # prevent GC of ObjC targets

        # Stat value labels (updated on show)
        self._total_transcriptions_label: Optional[NSTextField] = None
        self._total_words_label: Optional[NSTextField] = None
        self._time_saved_label: Optional[NSTextField] = None
        self._today_transcriptions_label: Optional[NSTextField] = None
        self._status_label: Optional[NSTextField] = None
        self._language_buttons: List[NSButton] = []

        self._build_window()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @_ensure_main_thread
    def show(self) -> None:
        """Show (or bring to front) the dashboard, refreshing stats."""
        if self._window is None:
            self._build_window()
        self._refresh_stats()
        self._refresh_language_selection()
        self._window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)

    @_ensure_main_thread
    def hide(self) -> None:
        """Hide the dashboard window."""
        if self._window is not None:
            self._window.orderOut_(None)

    @property
    def is_visible(self) -> bool:
        return self._window is not None and self._window.isVisible()

    @_ensure_main_thread
    def refresh(self) -> None:
        """Refresh displayed stats without showing the window."""
        if self._window is not None and self._window.isVisible():
            self._refresh_stats()

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------

    def _build_window(self) -> None:
        """Construct the NSWindow and all subviews."""
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
        self._window.setTitle_("SafeVoice")
        self._window.setReleasedWhenClosed_(False)
        self._window.setMinSize_(NSSize(_WINDOW_WIDTH, _WINDOW_HEIGHT))
        self._window.setMaxSize_(NSSize(_WINDOW_WIDTH, _WINDOW_HEIGHT))

        content_view = self._window.contentView()

        # Vibrancy background
        vibrancy = NSVisualEffectView.alloc().initWithFrame_(
            NSMakeRect(0, 0, _WINDOW_WIDTH, _WINDOW_HEIGHT)
        )
        vibrancy.setMaterial_(NSVisualEffectMaterialWindowBackground)
        vibrancy.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        vibrancy.setState_(NSVisualEffectStateActive)
        vibrancy.setAutoresizingMask_(0x12)
        content_view.addSubview_(vibrancy)

        cursor_y = _WINDOW_HEIGHT - _PADDING

        # -- Header --
        cursor_y -= 28
        title_label = self._make_label(
            "SafeVoice",
            NSMakeRect(_PADDING, cursor_y, 200, 28),
            font=NSFont.boldSystemFontOfSize_(20.0),
            color=NSColor.labelColor(),
        )
        content_view.addSubview_(title_label)

        # Status indicator on the right
        self._status_label = self._make_label(
            "Ready",
            NSMakeRect(_WINDOW_WIDTH - _PADDING - 120, cursor_y + 4, 120, 20),
            font=NSFont.systemFontOfSize_(12.0),
            color=NSColor.systemGreenColor(),
            alignment=NSTextAlignmentCenter,
        )
        content_view.addSubview_(self._status_label)

        cursor_y -= 6

        # Version subtitle
        cursor_y -= 16
        version_label = self._make_label(
            "Voice Input for macOS",
            NSMakeRect(_PADDING, cursor_y, 300, 16),
            font=NSFont.systemFontOfSize_(12.0),
            color=NSColor.secondaryLabelColor(),
        )
        content_view.addSubview_(version_label)

        cursor_y -= _PADDING

        # -- Stats cards (2x2 grid) --
        card_width = (_WINDOW_WIDTH - 2 * _PADDING - _CARD_SPACING) / 2.0

        # Row 1
        cursor_y -= _CARD_HEIGHT
        self._total_transcriptions_label = self._add_stat_card(
            content_view,
            x=_PADDING,
            y=cursor_y,
            width=card_width,
            height=_CARD_HEIGHT,
            title="Total Transcriptions",
            value="0",
        )
        self._total_words_label = self._add_stat_card(
            content_view,
            x=_PADDING + card_width + _CARD_SPACING,
            y=cursor_y,
            width=card_width,
            height=_CARD_HEIGHT,
            title="Total Words",
            value="0",
        )

        # Row 2
        cursor_y -= _CARD_HEIGHT + _CARD_SPACING
        self._time_saved_label = self._add_stat_card(
            content_view,
            x=_PADDING,
            y=cursor_y,
            width=card_width,
            height=_CARD_HEIGHT,
            title="Time Saved",
            value="0m",
        )
        self._today_transcriptions_label = self._add_stat_card(
            content_view,
            x=_PADDING + card_width + _CARD_SPACING,
            y=cursor_y,
            width=card_width,
            height=_CARD_HEIGHT,
            title="Today's Transcriptions",
            value="0",
        )

        cursor_y -= _PADDING + 4

        # -- Quick language selector --
        cursor_y -= 18
        lang_header = self._make_label(
            "Quick Language",
            NSMakeRect(_PADDING, cursor_y, 200, 18),
            font=NSFont.boldSystemFontOfSize_(13.0),
            color=NSColor.labelColor(),
        )
        content_view.addSubview_(lang_header)

        cursor_y -= 8

        # Language badge buttons in a scrollable row
        cursor_y -= 32
        self._language_buttons = []
        badge_x = _PADDING
        badge_width = 50.0
        badge_spacing = 6.0

        # Show a subset of popular languages that fit in the window
        for i, lang in enumerate(SUPPORTED_LANGUAGES):
            if badge_x + badge_width > _WINDOW_WIDTH - _PADDING:
                break
            btn = NSButton.alloc().initWithFrame_(
                NSMakeRect(badge_x, cursor_y, badge_width, 28)
            )
            btn.setTitle_(lang["badge"])
            btn.setBezelStyle_(NSBezelStyleRounded)
            btn.setFont_(NSFont.systemFontOfSize_(11.0))
            target = self._make_language_target(i)
            btn.setTarget_(target)
            btn.setAction_("invoke")
            content_view.addSubview_(btn)
            self._language_buttons.append(btn)
            badge_x += badge_width + badge_spacing

        cursor_y -= _PADDING + 4

        # -- Settings button --
        cursor_y -= 32
        settings_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(_PADDING, cursor_y, 140, 32)
        )
        settings_btn.setTitle_("Settings...")
        settings_btn.setBezelStyle_(NSBezelStyleRounded)
        settings_btn.setFont_(NSFont.systemFontOfSize_(13.0))
        settings_target = self._make_settings_target()
        settings_btn.setTarget_(settings_target)
        settings_btn.setAction_("invoke")
        content_view.addSubview_(settings_btn)

    # ------------------------------------------------------------------
    # Stat card builder
    # ------------------------------------------------------------------

    def _add_stat_card(
        self,
        parent_view: NSView,
        x: float,
        y: float,
        width: float,
        height: float,
        title: str,
        value: str,
    ) -> NSTextField:
        """Create a rounded stat card and return the value label for updates."""
        # Card background
        card = NSBox.alloc().initWithFrame_(NSMakeRect(x, y, width, height))
        card.setBoxType_(NSBoxCustom)
        card.setCornerRadius_(_CARD_CORNER_RADIUS)
        card.setBorderWidth_(0.5)
        card.setBorderColor_(NSColor.separatorColor())
        card.setFillColor_(
            NSColor.colorWithCalibratedWhite_alpha_(0.5, 0.08)
        )
        card.setTitlePosition_(0)  # NSNoTitle
        card.setContentViewMargins_(NSSize(0, 0))
        parent_view.addSubview_(card)

        # Value label (large number)
        value_label = self._make_label(
            value,
            NSMakeRect(x + 14, y + 24, width - 28, 30),
            font=NSFont.monospacedDigitSystemFontOfSize_weight_(26.0, 0.5),
            color=NSColor.labelColor(),
        )
        parent_view.addSubview_(value_label)

        # Title label (small description)
        title_label = self._make_label(
            title,
            NSMakeRect(x + 14, y + 8, width - 28, 16),
            font=NSFont.systemFontOfSize_(11.0),
            color=NSColor.secondaryLabelColor(),
        )
        parent_view.addSubview_(title_label)

        return value_label

    # ------------------------------------------------------------------
    # Refresh stats from SettingsManager
    # ------------------------------------------------------------------

    def _refresh_stats(self) -> None:
        """Read stats from the settings manager and update labels."""
        stats = self._mgr.get_stats()

        if self._total_transcriptions_label:
            self._total_transcriptions_label.setStringValue_(
                f"{stats.get('stats_total_transcriptions', 0):,}"
            )
        if self._total_words_label:
            self._total_words_label.setStringValue_(
                f"{stats.get('stats_total_words', 0):,}"
            )
        if self._time_saved_label:
            self._time_saved_label.setStringValue_(
                stats.get("stats_time_saved_display", "0m")
            )
        if self._today_transcriptions_label:
            self._today_transcriptions_label.setStringValue_(
                f"{stats.get('stats_today_transcriptions', 0):,}"
            )

    def _refresh_language_selection(self) -> None:
        """Highlight the currently selected language badge buttons."""
        selected_langs = self._mgr.get("languages", ["Auto"])
        for i, btn in enumerate(self._language_buttons):
            if i < len(SUPPORTED_LANGUAGES):
                is_selected = SUPPORTED_LANGUAGES[i]["name"] in selected_langs
                if is_selected:
                    btn.setFont_(NSFont.boldSystemFontOfSize_(11.0))
                else:
                    btn.setFont_(NSFont.systemFontOfSize_(11.0))

    # ------------------------------------------------------------------
    # Action targets
    # ------------------------------------------------------------------

    def _make_language_target(self, index: int):
        """Create an NSObject target for a language badge button."""
        def callback():
            if index < len(SUPPORTED_LANGUAGES):
                lang = SUPPORTED_LANGUAGES[index]
                # Dashboard quick-select sets a single language
                self._mgr.set("languages", [lang["name"]])
                self._refresh_language_selection()

        target = _DashboardCallbackTarget.alloc().initWithCallback_(callback)
        self._targets.append(target)
        return target

    def _make_settings_target(self):
        """Create an NSObject target for the settings button."""
        def callback():
            if self._on_open_settings:
                self._on_open_settings()

        target = _DashboardCallbackTarget.alloc().initWithCallback_(callback)
        self._targets.append(target)
        return target

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_label(
        text: str,
        frame,
        font=None,
        color=None,
        alignment=None,
    ) -> NSTextField:
        """Create a non-editable label."""
        label = NSTextField.alloc().initWithFrame_(frame)
        label.setStringValue_(text)
        label.setFont_(font or NSFont.systemFontOfSize_(13.0))
        label.setTextColor_(color or NSColor.labelColor())
        if alignment is not None:
            label.setAlignment_(alignment)
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        return label

    def _dispatch_to_main(self, block) -> None:
        """Dispatch a callable to the main thread."""
        trampoline = _DashboardTrampoline.alloc().initWithBlock_(block)
        _DashboardTrampoline._prevent_gc.add(trampoline)
        trampoline.performSelectorOnMainThread_withObject_waitUntilDone_(
            "invoke", None, False
        )
