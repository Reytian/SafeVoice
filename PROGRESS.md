# PROGRESS: audit + fix rounds (2026-06-11)

COMPLETE. See AUDIT_REPORT.md for the full deliverable.

Round 1 (audit + bug fixes): 9 commits, tests 37 -> 53. PyObjC
class-redefinition crashes, modes.json hardening + builtin-prompt
persistence, atomic writes, ASR/MLX races, LLM truncation/translation
guards, UI truthfulness, log redaction, overlay success flash, real
wizard mic permission.

Round 2 ("ok. fix them", recommendations implemented): 9 more commits,
tests -> 57.
- History window (copy / CSV export / clear)
- Live partial transcription in the overlay + silent-mic hint
- Model download: single owner, byte progress in wizard, hotkey retry
- First-run permission sequencing; async TCC prompt (blocking prompt
  froze untrusted-bundle launches, proven by process sample)
- Live modes menu, mode-aware status line, live dashboard state
- Friendly overlay errors; clipboard TransientType/ConcealedType
- Distribution Phase A: standalone py2app build WORKS on Python 3.14
  (namespace-package shim for mlx in build.sh), 408 MB self-contained
  bundle verified booting; setup.py/requirements/icon/pynput cleaned up

Still open (user action / later phases): Developer ID signing +
notarization, Keychain for API keys, log relocation for the shipped
build, Sparkle updates, bundle size trim (excludes), global per-mode
hotkeys (listen-only tap would leak keystrokes; needs design decision).
