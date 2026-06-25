"""Tests for the ASR idle-unload policy (pure decision function) and the
idle-unload timeout setting default."""

from src.idle_policy import should_unload_idle


# --- should_unload_idle: pure policy --------------------------------------

def test_unloads_when_loaded_idle_and_past_timeout():
    # 11 minutes since last activity, 10-minute timeout, model loaded, idle.
    assert should_unload_idle(
        now=660.0, last_activity=0.0, timeout_minutes=10,
        is_loaded=True, is_idle_state=True,
    ) is True


def test_does_not_unload_before_timeout():
    # 9 minutes idle, 10-minute timeout -> keep the model loaded.
    assert should_unload_idle(
        now=540.0, last_activity=0.0, timeout_minutes=10,
        is_loaded=True, is_idle_state=True,
    ) is False


def test_zero_timeout_disables_unload():
    # timeout 0 means "never unload" even after a long idle period.
    assert should_unload_idle(
        now=10_000.0, last_activity=0.0, timeout_minutes=0,
        is_loaded=True, is_idle_state=True,
    ) is False


def test_negative_timeout_disables_unload():
    # A corrupted/negative setting must not trigger an immediate unload.
    assert should_unload_idle(
        now=10_000.0, last_activity=0.0, timeout_minutes=-1,
        is_loaded=True, is_idle_state=True,
    ) is False


def test_does_not_unload_when_not_loaded():
    # Nothing to free if the model is not resident.
    assert should_unload_idle(
        now=660.0, last_activity=0.0, timeout_minutes=10,
        is_loaded=False, is_idle_state=True,
    ) is False


def test_does_not_unload_when_not_idle():
    # A recording/transcription is in progress; never unload mid-flight.
    assert should_unload_idle(
        now=660.0, last_activity=0.0, timeout_minutes=10,
        is_loaded=True, is_idle_state=False,
    ) is False


def test_exact_timeout_boundary_unloads():
    # Exactly at the threshold counts as elapsed.
    assert should_unload_idle(
        now=600.0, last_activity=0.0, timeout_minutes=10,
        is_loaded=True, is_idle_state=True,
    ) is True


# --- setting default ------------------------------------------------------

def test_default_asr_idle_unload_minutes_is_ten(tmp_path):
    from src.settings_manager import SettingsManager
    mgr = SettingsManager(config_path=tmp_path / "settings.json")
    assert mgr.get("asr_idle_unload_minutes") == 10


# --- Settings UI option set -----------------------------------------------

def test_idle_unload_options_cover_default_and_disable():
    # The General-tab popup must offer "never" (0) and the settings default (10),
    # and every option must be a non-negative integer minute count.
    from src.settings_window import SettingsWindow
    options = SettingsWindow._IDLE_UNLOAD_OPTIONS
    minutes = [m for _label, m in options]
    assert 0 in minutes          # disable / keep loaded
    assert 10 in minutes         # matches asr_idle_unload_minutes default
    assert all(isinstance(m, int) and m >= 0 for _label, m in options)
    labels = [label for label, _m in options]
    assert len(labels) == len(set(labels))  # labels are unique
