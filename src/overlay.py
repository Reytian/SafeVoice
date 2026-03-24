"""
Floating overlay panel for the macOS voice IME application.

Displays a translucent, non-activating floating capsule near the top of the
screen showing the current recording status, language badge, and live
transcription preview. Built with PyObjC (AppKit/Foundation).

The overlay never steals focus from the active application. All UI mutations
are dispatched to the main thread to satisfy AppKit's threading requirements.
"""

import threading
from typing import Optional

from AppKit import (
    NSPanel,
    NSView,
    NSTextField,
    NSColor,
    NSFont,
    NSScreen,
    NSApp,
    NSFloatingWindowLevel,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
    NSVisualEffectView,
    NSVisualEffectMaterialHUDWindow,
    NSVisualEffectBlendingModeBehindWindow,
    NSVisualEffectStateActive,
    NSLineBreakByTruncatingTail,
    NSTextAlignmentCenter,
    NSTextAlignmentLeft,
    NSAnimationContext,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorStationary,
)
from Foundation import NSMakeRect, NSObject
import objc


# ---------------------------------------------------------------------------
# Non-activating NSPanel subclass
# ---------------------------------------------------------------------------

class _NonActivatingPanel(NSPanel):
    """An NSPanel subclass that refuses to become key or main window."""

    def canBecomeKeyWindow(self):
        return False

    def canBecomeMainWindow(self):
        return False


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

_PANEL_HEIGHT = 44.0          # Capsule height in points.
_PANEL_MIN_WIDTH = 280.0      # Minimum capsule width.
_PANEL_MAX_WIDTH = 480.0      # Maximum capsule width.
_PANEL_TOP_OFFSET = 80.0      # Distance from the top of the screen.
_CORNER_RADIUS = 22.0         # Half of height for a pill shape.

_HORIZONTAL_PADDING = 14.0    # Padding inside the capsule.
_ELEMENT_SPACING = 8.0        # Spacing between dot, badge, and text.

_DOT_SIZE = 10.0              # Status indicator dot diameter.
_BADGE_WIDTH = 36.0           # Language badge width.
_TEXT_FIELD_MIN_WIDTH = 140.0  # Minimum width for the transcription label.

_FADE_DURATION = 0.1          # Duration (seconds) for fade-in / fade-out.

# Status dot colours.
_STATUS_COLORS = {
    "listening":  NSColor.colorWithCalibratedRed_green_blue_alpha_(0.30, 0.85, 0.40, 1.0),
    "processing": NSColor.colorWithCalibratedRed_green_blue_alpha_(1.00, 0.75, 0.00, 1.0),
    "error":      NSColor.colorWithCalibratedRed_green_blue_alpha_(1.00, 0.30, 0.30, 1.0),
}


# ---------------------------------------------------------------------------
# Thread-dispatch helper
# ---------------------------------------------------------------------------

def _ensure_main_thread(fn):
    """Decorator that dispatches the call to the main thread if necessary.

    AppKit views must only be modified on the main thread. If the decorated
    method is called from a background thread the invocation is forwarded via
    ``performSelectorOnMainThread:withObject:waitUntilDone:``.
    """

    def wrapper(self, *args, **kwargs):
        if threading.current_thread() is threading.main_thread():
            fn(self, *args, **kwargs)
        else:
            # Pack the call into a zero-argument closure and dispatch.
            self._dispatch_to_main(lambda: fn(self, *args, **kwargs))

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


# ---------------------------------------------------------------------------
# Helper ObjC trampoline for main-thread dispatch
# ---------------------------------------------------------------------------

class _Trampoline(NSObject):
    """Minimal NSObject subclass used to invoke a Python callable on the main
    thread via ``performSelectorOnMainThread:withObject:waitUntilDone:``.

    A class-level set prevents premature garbage collection when dispatched
    asynchronously (waitUntilDone=False).
    """

    _prevent_gc: set = set()

    def initWithBlock_(self, block):
        self = objc.super(_Trampoline, self).init()
        if self is None:
            return None
        self._block = block
        return self

    def invoke(self):
        try:
            if self._block is not None:
                self._block()
        finally:
            _Trampoline._prevent_gc.discard(self)


# ---------------------------------------------------------------------------
# FloatingOverlay
# ---------------------------------------------------------------------------

class FloatingOverlay:
    """A floating, translucent capsule overlay for voice-IME status display.

    The panel hovers near the top-centre of the primary screen and shows:
    - A coloured status dot (green = listening, amber = processing, red = error).
    - A language badge (e.g. ``EN``, ``FR``).
    - A live transcription preview or status message.

    The overlay is non-activating: it never takes keyboard focus away from the
    application the user is typing in.

    Usage::

        overlay = FloatingOverlay()
        overlay.show(language="EN")
        overlay.update_text("Hello world")
        overlay.update_level(0.6)
        overlay.hide()
        overlay.cleanup()
    """

    def __init__(self) -> None:
        """Initialise overlay resources without displaying anything."""
        self._panel: Optional[NSPanel] = None
        self._vibrancy_view: Optional[NSVisualEffectView] = None
        self._dot_label: Optional[NSTextField] = None
        self._language_label: Optional[NSTextField] = None
        self._text_label: Optional[NSTextField] = None
        self._level_view: Optional[NSView] = None

        self._status: str = "listening"
        self._language: str = "EN"
        self._visible: bool = False

        self._peak_width: float = _PANEL_MIN_WIDTH
        self._current_status: str = "listening"
        self._badge_label: Optional[NSTextField] = None

        self._build_panel()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @_ensure_main_thread
    def show(self, language: str = "EN") -> None:
        """Show the overlay with a *Listening...* status message.

        Args:
            language: Two-letter code or short label for the language badge.
        """
        if self._panel is None:
            return

        self._language = language
        self._status = "listening"
        self._apply_status()
        self._language_label.setStringValue_(language)
        self._text_label.setStringValue_("Listening...")
        self._update_level_bar(0.0)
        self._reposition()

        # Fade in.
        self._panel.setAlphaValue_(0.0)
        self._panel.orderFront_(None)
        NSAnimationContext.beginGrouping()
        NSAnimationContext.currentContext().setDuration_(_FADE_DURATION)
        self._panel.animator().setAlphaValue_(1.0)
        NSAnimationContext.endGrouping()

        self._visible = True

    @_ensure_main_thread
    def update_text(self, text: str) -> None:
        """Update the live transcription preview text.

        Args:
            text: The current partial or final transcription string.
        """
        if self._text_label is None:
            return
        display = text if text else "Listening..."
        self._text_label.setStringValue_(display)

        # Dynamic width calculation
        if self._text_label is not None:
            text_size = self._text_label.attributedStringValue().size()
            desired = _HORIZONTAL_PADDING * 2 + _DOT_SIZE + _ELEMENT_SPACING + _BADGE_WIDTH + _ELEMENT_SPACING + text_size.width + 20
            desired = max(_PANEL_MIN_WIDTH, min(_PANEL_MAX_WIDTH, desired))

            # Peak-width: only grow during listening, never shrink
            if self._current_status == "listening":
                self._peak_width = max(self._peak_width, desired)
                desired = self._peak_width
            else:
                self._peak_width = _PANEL_MIN_WIDTH

            self._resize_panel(desired)

    @_ensure_main_thread
    def update_level(self, level: float) -> None:
        """Update the audio-level visualisation bar.

        Args:
            level: Normalised RMS level in the range ``[0.0, 1.0]``.
        """
        clamped = max(0.0, min(1.0, level))
        self._update_level_bar(clamped)

    @_ensure_main_thread
    def set_status(self, status: str) -> None:
        """Set the recording status indicator.

        Args:
            status: One of ``'listening'``, ``'processing'``, or ``'error'``.
        """
        if status not in _STATUS_COLORS:
            return
        self._status = status
        self._current_status = status
        self._apply_status()

        # Update state badge
        if hasattr(self, '_badge_label') and self._badge_label is not None:
            if status == "processing":
                self._badge_label.setStringValue_("AI")
                self._badge_label.setBackgroundColor_(
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.7, 0.0, 1.0)
                )
                self._badge_label.setHidden_(False)
            elif status == "done":
                self._badge_label.setStringValue_("OK")
                self._badge_label.setBackgroundColor_(
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(0.3, 0.85, 0.4, 1.0)
                )
                self._badge_label.setHidden_(False)
            else:
                self._badge_label.setHidden_(True)

    @_ensure_main_thread
    def set_language(self, language: str) -> None:
        """Update the language badge.

        Args:
            language: Short label such as ``'EN'``, ``'FR'``, or ``'中文'``.
        """
        self._language = language
        if self._language_label is not None:
            self._language_label.setStringValue_(language)

    @_ensure_main_thread
    def hide(self) -> None:
        """Hide the overlay with a short fade-out animation."""
        if self._panel is None or not self._visible:
            return

        NSAnimationContext.beginGrouping()
        ctx = NSAnimationContext.currentContext()
        ctx.setDuration_(_FADE_DURATION)

        # Use a completion handler to order the panel out after the fade.
        def _on_complete():
            if self._panel is not None:
                self._panel.orderOut_(None)

        ctx.setCompletionHandler_(_on_complete)
        self._panel.animator().setAlphaValue_(0.0)
        NSAnimationContext.endGrouping()

        self._visible = False

    @_ensure_main_thread
    def cleanup(self) -> None:
        """Release the panel and all associated views.

        After calling this method the overlay instance should not be reused.
        """
        if self._panel is not None:
            self._panel.orderOut_(None)
            self._panel.close()

        self._panel = None
        self._vibrancy_view = None
        self._dot_label = None
        self._language_label = None
        self._text_label = None
        self._level_view = None
        self._badge_label = None
        self._visible = False

    # ------------------------------------------------------------------
    # Panel construction
    # ------------------------------------------------------------------

    def _build_panel(self) -> None:
        """Construct the NSPanel and its subviews."""
        content_rect = NSMakeRect(0, 0, _PANEL_MIN_WIDTH, _PANEL_HEIGHT)

        style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        self._panel = _NonActivatingPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            content_rect, style, 2, False  # 2 = NSBackingStoreBuffered
        )

        self._panel.setLevel_(NSFloatingWindowLevel)
        self._panel.setOpaque_(False)
        self._panel.setBackgroundColor_(NSColor.clearColor())
        self._panel.setHasShadow_(True)
        self._panel.setAlphaValue_(0.0)

        # Allow the panel to appear on all Spaces / desktops.
        self._panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
        )

        content_view = self._panel.contentView()

        # --- Vibrancy / blur background ---
        self._vibrancy_view = NSVisualEffectView.alloc().initWithFrame_(content_rect)
        self._vibrancy_view.setMaterial_(NSVisualEffectMaterialHUDWindow)
        self._vibrancy_view.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        self._vibrancy_view.setState_(NSVisualEffectStateActive)
        self._vibrancy_view.setWantsLayer_(True)
        self._vibrancy_view.layer().setCornerRadius_(_CORNER_RADIUS)
        self._vibrancy_view.layer().setMasksToBounds_(True)
        self._vibrancy_view.setAutoresizingMask_(0x12)  # Width + height flexible.
        content_view.addSubview_(self._vibrancy_view)

        # --- Audio level indicator (thin bar at bottom) ---
        level_height = 3.0
        self._level_view = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, 0, level_height)
        )
        self._level_view.setWantsLayer_(True)
        self._level_view.layer().setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.30, 0.85, 0.40, 0.6
            ).CGColor()
        )
        self._level_view.layer().setCornerRadius_(level_height / 2.0)
        content_view.addSubview_(self._level_view)

        # --- Status dot ---
        x_cursor = _HORIZONTAL_PADDING
        dot_y = (_PANEL_HEIGHT - _DOT_SIZE) / 2.0
        self._dot_label = self._make_label(
            frame=NSMakeRect(x_cursor, dot_y, _DOT_SIZE, _DOT_SIZE),
            text="\u25CF",  # Filled circle character.
            font=NSFont.systemFontOfSize_(10.0),
            color=_STATUS_COLORS["listening"],
            alignment=NSTextAlignmentCenter,
        )
        content_view.addSubview_(self._dot_label)
        x_cursor += _DOT_SIZE + _ELEMENT_SPACING

        # --- State badge (AI/OK) ---
        badge_x = _HORIZONTAL_PADDING + _DOT_SIZE + 2
        badge_y = (_PANEL_HEIGHT - 16) / 2
        self._badge_label = NSTextField.alloc().initWithFrame_(NSMakeRect(badge_x, badge_y, 24, 16))
        self._badge_label.setStringValue_("")
        self._badge_label.setFont_(NSFont.boldSystemFontOfSize_(9))
        self._badge_label.setBezeled_(False)
        self._badge_label.setDrawsBackground_(True)
        self._badge_label.setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.7, 0.0, 1.0)
        )
        self._badge_label.setTextColor_(NSColor.whiteColor())
        self._badge_label.setAlignment_(NSTextAlignmentCenter)
        self._badge_label.setHidden_(True)
        self._badge_label.setEditable_(False)
        self._badge_label.setSelectable_(False)
        self._badge_label.setWantsLayer_(True)
        self._badge_label.layer().setCornerRadius_(4)
        self._badge_label.layer().setMasksToBounds_(True)
        content_view.addSubview_(self._badge_label)

        # --- Language badge ---
        badge_y = (_PANEL_HEIGHT - 20.0) / 2.0
        self._language_label = self._make_label(
            frame=NSMakeRect(x_cursor, badge_y, _BADGE_WIDTH, 20.0),
            text="EN",
            font=NSFont.boldSystemFontOfSize_(11.0),
            color=NSColor.secondaryLabelColor(),
            alignment=NSTextAlignmentCenter,
        )
        # Give the badge a subtle rounded background.
        self._language_label.setWantsLayer_(True)
        self._language_label.layer().setBackgroundColor_(
            NSColor.colorWithCalibratedWhite_alpha_(0.5, 0.15).CGColor()
        )
        self._language_label.layer().setCornerRadius_(4.0)
        content_view.addSubview_(self._language_label)
        x_cursor += _BADGE_WIDTH + _ELEMENT_SPACING

        # --- Transcription text ---
        text_width = _PANEL_MIN_WIDTH - x_cursor - _HORIZONTAL_PADDING
        text_y = (_PANEL_HEIGHT - 20.0) / 2.0
        self._text_label = self._make_label(
            frame=NSMakeRect(x_cursor, text_y, max(text_width, _TEXT_FIELD_MIN_WIDTH), 20.0),
            text="Listening...",
            font=NSFont.systemFontOfSize_(13.0),
            color=NSColor.labelColor(),
            alignment=NSTextAlignmentLeft,
        )
        self._text_label.setLineBreakMode_(NSLineBreakByTruncatingTail)
        content_view.addSubview_(self._text_label)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_label(
        frame,
        text: str,
        font,
        color,
        alignment,
    ) -> NSTextField:
        """Create a non-editable, borderless ``NSTextField`` (label).

        Args:
            frame: The ``NSRect`` defining position and size.
            text: Initial string value.
            font: ``NSFont`` instance.
            color: ``NSColor`` for the text.
            alignment: ``NSTextAlignment`` constant.

        Returns:
            Configured ``NSTextField`` ready to be added as a subview.
        """
        label = NSTextField.alloc().initWithFrame_(frame)
        label.setStringValue_(text)
        label.setFont_(font)
        label.setTextColor_(color)
        label.setAlignment_(alignment)
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        return label

    def _resize_panel(self, new_width: float):
        """Resize panel width, centered horizontally, with animation."""
        def _do_resize():
            if self._panel is None:
                return
            frame = self._panel.frame()
            screen = NSScreen.mainScreen().visibleFrame()
            new_x = screen.origin.x + (screen.size.width - new_width) / 2
            new_frame = NSMakeRect(new_x, frame.origin.y, new_width, frame.size.height)
            NSAnimationContext.beginGrouping()
            NSAnimationContext.currentContext().setDuration_(0.15)
            self._panel.animator().setFrame_display_(new_frame, True)
            NSAnimationContext.endGrouping()
        self._dispatch_to_main(_do_resize)

    def _apply_status(self) -> None:
        """Apply the current ``_status`` to the dot label colour."""
        if self._dot_label is None:
            return
        color = _STATUS_COLORS.get(self._status, _STATUS_COLORS["listening"])
        self._dot_label.setTextColor_(color)

        # Also tint the level-bar to match the status colour.
        if self._level_view is not None and self._level_view.layer() is not None:
            bar_color = color.colorWithAlphaComponent_(0.6)
            self._level_view.layer().setBackgroundColor_(bar_color.CGColor())

    def _update_level_bar(self, level: float) -> None:
        """Resize the level indicator bar to reflect the audio level.

        Args:
            level: Normalised level in ``[0.0, 1.0]``.
        """
        if self._level_view is None or self._panel is None:
            return
        panel_width = self._panel.frame().size.width
        bar_width = panel_width * level
        bar_height = self._level_view.frame().size.height
        self._level_view.setFrame_(NSMakeRect(0, 0, bar_width, bar_height))

    def _reposition(self) -> None:
        """Centre the panel horizontally near the top of the main screen."""
        screen = NSScreen.mainScreen()
        if screen is None:
            return

        screen_frame = screen.frame()
        visible_frame = screen.visibleFrame()

        panel_width = self._panel.frame().size.width
        panel_x = screen_frame.origin.x + (screen_frame.size.width - panel_width) / 2.0

        # Compute Y from the top of the visible area.  AppKit uses a
        # bottom-left origin so we convert from the top offset.
        panel_y = (
            visible_frame.origin.y
            + visible_frame.size.height
            - _PANEL_TOP_OFFSET
            - _PANEL_HEIGHT
        )

        self._panel.setFrameOrigin_((panel_x, panel_y))

    def _dispatch_to_main(self, block) -> None:
        """Dispatch a zero-argument callable to the main thread.

        Uses an ``_Trampoline`` NSObject so we can leverage Cocoa's built-in
        ``performSelectorOnMainThread:withObject:waitUntilDone:`` mechanism
        which is safe to call from any thread.

        Args:
            block: A zero-argument callable to execute on the main thread.
        """
        trampoline = _Trampoline.alloc().initWithBlock_(block)
        _Trampoline._prevent_gc.add(trampoline)
        trampoline.performSelectorOnMainThread_withObject_waitUntilDone_(
            "invoke", None, False
        )
