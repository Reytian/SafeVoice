# PROGRESS: audit (debug + UX + distribution readiness), 2026-06-11

COMPLETE. See AUDIT_REPORT.md for the full deliverable.

- Baseline 37 tests -> 53 tests, all passing. py2app -A build verified.
- 9 fix commits on claude/quizzical-shannon-ae6a10 (see git log and
  AUDIT_REPORT.md Part 1): PyObjC class-redefinition crashes (13 sites),
  modes.json startup crash + builtin-prompt persistence, atomic
  credentials/vocabulary writes, ASR/MLX inference races + hotkey lock
  discipline, LLM truncation + unrequested-translation guards, UI
  truthfulness (hotkey labels, mode switching, language persistence,
  ASR catalog), transcript log redaction (SAFEVOICE_LOG_CONTENT=1 to
  re-enable content), overlay success flash + None-screen guard, real
  wizard mic permission + find_ollama PATH fix.
- README honesty pass included (path, languages, hotkeys, privacy wording).
- UX recommendations and the 4-phase distribution roadmap are in
  AUDIT_REPORT.md Parts 2-3 (history viewer and live preview are the top
  UX items; standalone py2app build + signing/notarization are the top
  distribution items).
