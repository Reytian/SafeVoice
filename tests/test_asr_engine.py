"""Tests for ASR engine memory lifecycle: unload must trim the MLX/Metal
buffer cache, and load applies a cache-size cap.

These cover the memory-release seams added for idle-unload. They exercise the
real mlx.core calls (mlx is installed) plus monkeypatched seams to assert the
lifecycle wiring without loading the ~1.8 GB model.
"""

import src.asr_engine as ae


def test_unload_model_clears_session_and_trims_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(ae, "_clear_mlx_cache", lambda: calls.append(True))
    engine = ae.ASREngine()
    engine._session = object()  # pretend a model is loaded
    engine._streaming_state = object()

    engine.unload_model()

    assert engine._session is None
    assert engine._streaming_state is None
    assert calls == [True], "unload_model must trim the MLX cache"


def test_unload_model_does_not_raise_if_cache_trim_fails(monkeypatch):
    def boom():
        raise RuntimeError("metal unavailable")

    monkeypatch.setattr(ae, "_clear_mlx_cache", boom)
    engine = ae.ASREngine()
    engine._session = object()

    # The reference must still be dropped and quit/idle-unload must not crash.
    engine.unload_model()
    assert engine._session is None


def test_clear_mlx_cache_runs_without_error():
    # Real mlx.core.clear_cache() on an empty cache: a safe no-op.
    ae._clear_mlx_cache()


def test_apply_mlx_cache_limit_runs_without_error():
    # Real mlx.core.set_cache_limit(constant): must not raise.
    ae._apply_mlx_cache_limit()


def test_mlx_cache_limit_constant_is_positive():
    assert ae._MLX_CACHE_LIMIT_BYTES > 0
