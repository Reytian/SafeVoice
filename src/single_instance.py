"""Single-instance enforcement for SafeVoice.

Prevents two SafeVoice processes from running simultaneously. Without this,
double-clicking SafeVoice.app twice spawns a duplicate process: both
HotkeyManager threads register CGEventTaps for the same key, the ASR model
loads twice (doubling RAM), and the user sees two menubar icons.

When a second instance starts it:
  1. Detects the first instance via NSRunningApplication bundle-id query.
  2. Posts a NSDistributedNotification asking the first instance to surface
     its settings window (so the user gets visible feedback).
  3. Exits cleanly, leaving the first instance running.

Borrowed in spirit from Tauri's tauri-plugin-single-instance, implemented
natively against AppKit / Foundation so it has zero new dependencies.
"""

import logging
import os
from typing import Optional

import objc
from AppKit import NSRunningApplication
from Foundation import NSDistributedNotificationCenter, NSObject

logger = logging.getLogger(__name__)

BUNDLE_ID = "com.safevoice.app"
SHOW_SETTINGS_NOTIFICATION = "com.safevoice.show-settings"


def is_duplicate_and_signal_existing() -> bool:
    """Return True if another SafeVoice instance is already running.

    When True, the caller should exit immediately. Before returning, this
    function asks the existing instance (via NSDistributedNotificationCenter)
    to surface its settings window so the user knows their double-click was
    received.

    py2app bundles do not have a stable CFBundleIdentifier when run via
    `python run.py` from source -- in that case NSRunningApplication's
    bundle-id query returns only the .app bundle process, not the source
    interpreter. That's intentional: we only care about preventing two
    bundled .app launches from colliding, since that is what users do.
    """
    my_pid = os.getpid()
    others = [
        a
        for a in NSRunningApplication.runningApplicationsWithBundleIdentifier_(
            BUNDLE_ID
        )
        if a.processIdentifier() != my_pid
    ]
    if not others:
        return False

    existing_pid = others[0].processIdentifier()
    logger.info(
        "Another SafeVoice instance is already running (pid=%s). "
        "Asking it to show settings, then exiting.",
        existing_pid,
    )
    NSDistributedNotificationCenter.defaultCenter().postNotificationName_object_(
        SHOW_SETTINGS_NOTIFICATION, None
    )
    return True


class _ShowSettingsObserver(NSObject):
    """Bridge that receives NSDistributedNotification on the main run loop.

    Must subclass NSObject so PyObjC can build a proper Objective-C selector
    for the notification center to call. The instance MUST be retained by
    Python (assigned to a long-lived attribute) or it will be GC'd and the
    observer silently disappears.
    """

    def initWithCallback_(self, callback):  # noqa: N802 -- ObjC selector name
        self = objc.super(_ShowSettingsObserver, self).init()
        if self is None:
            return None
        self._callback = callback
        return self

    def handleShowSettings_(self, _notification):  # noqa: N802 -- ObjC selector
        try:
            self._callback()
        except Exception:
            logger.exception("Show-settings handler raised")


def install_show_settings_listener(on_show_settings) -> Optional[NSObject]:
    """Register a listener for the SHOW_SETTINGS distributed notification.

    Returns the observer NSObject. The caller MUST retain it on a long-lived
    attribute (e.g. self._show_settings_observer). Releasing the returned
    object detaches the observer.
    """
    observer = _ShowSettingsObserver.alloc().initWithCallback_(on_show_settings)
    NSDistributedNotificationCenter.defaultCenter().addObserver_selector_name_object_(
        observer,
        objc.selector(observer.handleShowSettings_, signature=b"v@:@"),
        SHOW_SETTINGS_NOTIFICATION,
        None,
    )
    return observer
