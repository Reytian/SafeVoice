"""Pure decision logic for evicting the ASR model after an idle period.

Kept free of AppKit/threading so it can be unit-tested in isolation. The app's
idle-unload monitor thread feeds it a monotonic clock and the current app
state; the function decides whether the resident model should be released to
reclaim memory.
"""

from __future__ import annotations


def should_unload_idle(
    *,
    now: float,
    last_activity: float,
    timeout_minutes: float,
    is_loaded: bool,
    is_idle_state: bool,
) -> bool:
    """Return True if the ASR model should be unloaded to free memory.

    Args:
        now: Current monotonic time (seconds).
        last_activity: Monotonic time of the last dictation activity.
        timeout_minutes: Idle minutes before unloading. ``<= 0`` disables
            idle-unload entirely (the model stays resident for the session).
        is_loaded: Whether the ASR model is currently resident.
        is_idle_state: Whether the app is idle (not recording/transcribing).
            The model is never unloaded mid-flight.

    Returns:
        True only when idle-unload is enabled, the model is resident, the app
        is idle, and the elapsed idle time has reached the timeout.
    """
    if timeout_minutes <= 0:
        return False
    if not is_loaded:
        return False
    if not is_idle_state:
        return False
    return (now - last_activity) >= timeout_minutes * 60
