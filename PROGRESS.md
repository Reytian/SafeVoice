# PROGRESS — Audit: debug + UX + distribution readiness (2026-06-11)

Goal: full-codebase bug hunt (post bd7d644 audit), UX review for long-session
reliability, and a distribution-readiness assessment. Fix verified bugs; report
the rest as ranked suggestions.

## Status

- [x] Baseline: 37/37 tests pass on claude/quizzical-shannon-ae6a10 (worktree)
- [x] Prior audit (DEBUG_REPORT.md, 27 findings) confirmed fixed in tree
- [x] 4 review agents dispatched (pipeline, UI, data/LLM, distribution)
- [x] Manual UX pass: app.py, overlay.py, setup_wizard.py, dashboard_window.py,
      modes.py, hotkey_manager.py, text_injector.py, run.py, README
- [ ] Integrate agent findings, verify, fix
- [ ] Final report

## Confirmed findings from manual pass (to fix)

1. modes.py `_load()`: unguarded `json.load` -> corrupt modes.json crashes app
   startup (same class as audit #4, missed file). `_save()` non-atomic.
2. overlay.py `set_status("done")` dead code: "done" not in _STATUS_COLORS, so
   early-return; success flash never shows. No paste confirmation exists.
3. overlay.py `_resize_panel`: `NSScreen.mainScreen()` used without None guard
   (the sibling `_reposition` guards it).
4. app.py `_set_language` (menubar) never persists language; settings window
   and menubar fight; choice lost on restart.
5. Hotkey labels hardcode "Left ⌥" in menubar info + mode notifications even
   when user customized activate_hotkey.
6. Privacy: run.py logs DEBUG to world-readable /tmp/safevoice.log; app logs
   full transcripts (ASR result, LLM result, Injected). Distribution blocker.
7. Wizard mic check is fake (query_devices enumerates; never checks TCC).
8. ASR model change in settings only applies after relaunch, with no notice.
9. Processing modes unreachable at runtime: menubar items have no callbacks,
   `get_by_hotkey` has zero callers, per-mode hotkeys recorded but ignored.
10. Language hotkey setting still in UI but no implementation (pynput removed).
11. README stale: old path, wrong hotkey, 3 languages vs 15, promises live
    preview that doesn't exist.

## Decisions

- Fix list: items 1-8 + menubar mode-switch callbacks (cheap part of 9).
  Per-mode global hotkeys (9) and live preview = recommendations only.
- Keep /tmp/safevoice.log path (documented contract) but chmod 600 + redact
  transcript content unless SAFEVOICE_LOG_CONTENT=1.
