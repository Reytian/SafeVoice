# SafeVoice Audit: Debug + UX + Distribution Readiness (2026-06-11)

Scope: full-codebase bug hunt (post bd7d644 audit), UX review focused on long-session
reliability and output trust, and a distribution-readiness assessment. Method: four
parallel review agents (pipeline/concurrency, UI/AppKit, data/LLM, distribution) plus
a manual UX pass; every accepted finding was verified against the code path before
fixing. All 27 prior DEBUG_REPORT.md fixes were spot-checked and confirmed in place;
two had unfixed siblings in other files (modes.json loader, settings-window AppKit
thread violation), both fixed now.

Baseline: 37 tests passing. After this audit: 53 tests passing, 9 fix commits,
py2app alias build verified.

---

## Part 1. Bugs fixed in this branch (9 commits)

### 1.1 PyObjC class-redefinition crashes (second-use failures) (commit 6568832)

PyObjC registers Objective-C classes by NAME in the global runtime. Defining an
NSObject subclass inside a closure therefore raises `objc.error: ... is overriding
existing Objective-C class` the SECOND time the closure runs. Confirmed live on
PyObjC 12.1 / Python 3.14. Thirteen call sites did exactly that, so each of these
crashed its worker thread on repeat use and silently dropped the UI update:

- Settings > Models: ASR model Delete (2nd click), Ollama model rm/pull, MLX model
  delete/download. Several sites shared class names across DIFFERENT functions, so
  one feature's first use could break another feature's first use.
- Setup wizard: `_safe_status_update` defined a class per call, so when BOTH
  auto-started downloads (ASR + LLM) finished, the second one could never report
  its status; the Test button on the Tone step died on the second click; renavigating
  to the Tone step (Back then Next) crashed on the preset-popup target.

Fix: one module-level trampoline per file (`_post_to_main`); all worker completion
paths route through it. This also fixed the ASR-delete error path that mutated an
AppKit label from a background thread.

### 1.2 modes.json could crash startup; builtin prompt edits never persisted (commit a836c89)

- `ModeManager._load()` had no error handling (same bug class as audit finding #4,
  missed in this file): a corrupt/truncated/hand-edited modes.json raised out of
  `__init__` and the app failed to launch until the file was deleted by hand.
- Deeper data-loss bug exposed by a new regression test: `_save()` persisted only
  builtin modes' hotkeys, never their prompts. The setup wizard's tone choice and
  any "Edit Prompt" on a builtin mode silently reverted on every relaunch.
- `_save()` was also non-atomic, and builtin-mode edits bypassed the manager lock.

Fix: hardened loader (defaults survive any damage, bad entries skipped with logs),
atomic write-then-rename, builtin overrides now persist prompt_template and
translation_language (backward-compatible schema), new locked `update_prompt()`
used by the settings editor and wizard. 7 new tests.

### 1.3 Atomic writes for credentials.json and vocabulary.json (commit ce61641)

- `_save_api_key` truncated credentials.json in place; a crash mid-write destroyed
  every saved cloud API key, and the silent loader then showed empty fields.
- `VocabularyManager._save` could propagate OSError into NSButton ObjC callbacks
  (undefined behavior) and was non-atomic.

Fix: write-then-rename with fsync (credentials get 0600 before the rename), logged
load failures, utf-8 + ensure_ascii=False. 2 new tests.

### 1.4 ASR/MLX inference concurrency + hotkey lock discipline (commit 725b28d)

- `ASREngine.unload_model` nulled the session WITHOUT the session lock; quitting
  during a transcription raced the worker into `None.transcribe`.
- `MLXBackend` had no inference lock at all: the speculative-cleanup thread and the
  final cleanup could run `mlx_lm.generate` concurrently on the same Metal graph
  (crash/corruption). Same class as audit finding #2, in the LLM backend.
- `HotkeyManager.stop/set_mode/set_activate_hotkey` invoked the deactivate callback
  while holding the manager lock, which the CGEventTap thread also contends for;
  slow callback work could stall the tap past kCGEventTapDisabledByTimeout.
- TOCTOU on `self._cg_tap` in the tap re-enable handler.

### 1.5 Truncated LLM output / unrequested translation pasted silently (commit 2fefd87)

- Output cut at the token cap now raises `LLMTruncatedError` (Ollama done_reason,
  OpenAI-compat finish_reason, Anthropic stop_reason, Google finishReason) and
  llm_cleanup falls back to the rule-stripped transcript. Previously Ollama
  logged-and-returned the cut text and the cloud backends did not check at all,
  so the tail of a long dictation could vanish.
- The custom-prompt path (Formal Writing etc.) now applies the same script-change
  guards as the default path: a model that ignores "Do NOT translate" can no longer
  replace Chinese dictation with English. Translation modes opt out via a new
  `allow_script_change` flag derived from the active mode. 8 new tests.

### 1.6 UI told users things that were not true; modes unswitchable (commit e630d93)

- Every hotkey string (menubar info line, mode-change notifications) was hardcoded
  "Hold Left Option" while the persisted DEFAULT is Option+Space and the hotkey is
  user-configurable. All labels now derive from the configured hotkey and refresh
  when it changes.
- Processing Mode menu items had no callbacks and `get_by_hotkey` had zero callers:
  the entire modes feature (Formal Writing, English Translation) was unreachable at
  runtime. Menu items now switch the active mode with checkmarks. The dead per-mode
  hotkey suffixes were removed from menu titles (wiring real global mode hotkeys is
  future work, see Part 3).
- Menubar language picks were never persisted (fought the settings window, reset on
  relaunch). Now saved, recursion-safe.
- Changing the ASR model in Settings silently did nothing until relaunch; the user
  now gets a notification saying it applies on next launch.
- The ASR dropdown offered Whisper/cloud engines that asr_engine cannot load;
  selecting one wedged every later launch at "Error". Catalog filtered to
  implemented engines, startup falls back to the default for an unrunnable
  persisted model, and `_sync_ui_from_settings` re-selects the persisted LLM
  source / ASR model so a stale Models tab can't be applied by accident.

### 1.7 Transcripts no longer logged in plaintext by default (commit f8e75da)

Every dictation (potentially passwords, client matter) was written verbatim to
world-readable `/tmp/safevoice.log` and persisted across sessions. All transcript
log sites now emit `<redacted N chars>`; full content returns with
`SAFEVOICE_LOG_CONTENT=1` for debugging sessions. The log file is chmod 0600 at
startup (resolved from the logging handler; run.py untouched).

### 1.8 Overlay success flash was dead code; resize crash guard (commit 7003d35)

`set_status("done")` early-returned because "done" was missing from
`_STATUS_COLORS`, so the designed OK badge could never render and the user got no
paste confirmation at all (the overlay hides BEFORE the paste to avoid stealing
focus). "done" is now a valid status and a brief "Pasted" capsule flashes after a
successful injection, matching what the wizard's Demo step already promises.
Also added the missing None-screen guard in `_resize_panel`.

### 1.9 Wizard mic check was fake (commit 2c04f53)

`sounddevice.query_devices()` needs no permission, so the Permissions step showed
"Granted" on machines where macOS had never asked, and the real TCC prompt then
interrupted the user's FIRST dictation (which records silence until granted).
Status is now read from `AVCaptureDevice.authorizationStatusForMediaType_` via the
runtime bridge (no new dependency); when not yet requested, the button reads
"Allow" and opens a brief input stream, which triggers the macOS dialog and
registers the app in the Microphone pane. Plus: `ollama pull`/`rm` now resolve the
binary absolutely (`find_ollama()`), since Finder-launched apps have a PATH without
/opt/homebrew/bin, which made every wizard LLM pull fail with "Pull failed".

---

## Part 2. UX improvements for long, consistent use (recommendations, ranked)

The reliability foundations (tap re-enable, App Nap opt-out, audio failure
recovery) are in good shape after the previous audit. What is missing is mostly
feedback, recovery, and visibility.

1. **Transcription history viewer (highest value).** `HistoryStore` already records
   final + raw text for every dictation in sqlite, but NO UI surfaces it. When a
   paste lands in the wrong window or an LLM rewrite goes wrong, the text is
   gone from the user's view even though it sits in the database. Add a History tab
   or window (search, copy, re-paste, show raw vs cleaned), plus a retention
   setting and a "Clear history" button. This single feature converts most
   "lost transcript" complaints into a 5-second recovery.
2. **Live partial transcription in the overlay.** The engine has a full streaming
   API (`start_streaming`/`feed_chunk`) that nothing uses; the overlay shows only
   "Listening...". Feeding speculative/streaming partials into the overlay would
   deliver the "see your words as you speak" experience the wizard's Demo step
   describes, and gives early warning when the wrong language model kicks in.
3. **Recording duration + level in the overlay during capture**, and an explicit
   "too quiet / no input device" hint when the level stays at zero for several
   seconds (mic muted, AirPods switched away) instead of failing at the end with
   "(no speech detected)".
4. **First-run model download UX.** The 1.2 GB download currently auto-starts both
   from app startup AND from the wizard Model step (double download possible),
   with a text label and no progress, retry, or disk-space check. Single-owner
   download with a progress bar and a clear offline message is table stakes for
   distribution (see Part 3).
5. **Menu refresh after mode edits.** The Processing Mode submenu is built once at
   startup; modes added/renamed in Settings appear only after relaunch. Rebuild the
   submenu on ModeManager changes.
6. **Wire real global hotkeys for processing modes** (the data model and settings
   UI already exist). Until then the Modes tab hotkey recorder should be hidden or
   labeled "menu switching only" to avoid recording dead hotkeys.
7. **Error language.** Overlay errors show raw exception text (`Error: <exc>`).
   Map the common failures to human messages ("Mic is in use by another app",
   "Model still loading, try again in a moment").
8. **Clipboard etiquette.** Mark the transcript pasteboard write with
   `org.nspasteboard.TransientType` and `ConcealedType` so clipboard managers do
   not archive every dictation (privacy) and the restore dance is invisible.
9. **Menubar polish.** Show the active processing mode in the status item title or
   a "Mode: X" line (it is now switchable but not glanceable); make the dashboard
   status label reflect the live state instead of always "Ready".
10. **Settings apply model.** The Models tab Apply pattern is fine, but mixing
    instant-apply controls (General tab) with Apply-button tabs is inconsistent;
    pick one convention and label it.

## Part 3. Distribution roadmap (to ship to non-technical users)

Current state: development alias build only (`py2app -A`), ad-hoc signature, model
downloaded at first run, Ollama as an external dependency. The app cannot run on
any other Mac as-is. Phased plan:

**Phase A: standalone build that launches on another Mac**
- Produce a real `py2app` (non-alias) build; verify py2app/modulegraph support for
  Python 3.14 early, and fall back to Python 3.12/3.13 or PyInstaller if it breaks.
- setup.py packaging gaps to fix: include `_sounddevice_data` (bundles
  libportaudio), add the missing src modules to includes (modes, history,
  vocabulary, text_postprocess, audio_preprocess, single_instance, setup_wizard,
  llm_backend, privacy), ship the .icns as a resource and resolve it
  bundle-relative (the current `_PROJECT_ROOT/assets` path only works in dev),
  decide on `mlx_lm` (it is imported by the MLX backend but is not in
  requirements.txt, so the only Ollama-free LLM option is dead in a fresh env).
- Remove `pynput` from requirements and the py2app packages list (nothing imports
  it anymore).
- Expected app size 350-500 MB plus the 1.8 GB model cache.

**Phase B: signing, notarization, packaging**
- Developer ID Application certificate; sign nested .so/.dylib inside-out; hardened
  runtime with entitlements: `com.apple.security.device.audio-input`,
  `com.apple.security.cs.allow-unsigned-executable-memory`,
  `com.apple.security.cs.disable-library-validation`.
- Do NOT sandbox: CGEventTap and CGEventPost are incompatible with the App
  Sandbox, which also rules out the Mac App Store. Direct distribution via
  notarized + stapled DMG.
- Info.plist: drop the misleading `NSAppleEventsUsageDescription` (the app sends no
  Apple Events; Accessibility has no usage-description key), add
  `LSApplicationCategoryType` and a real copyright. Decide `LSUIElement`: a
  menubar utility usually hides the Dock icon and offers a "show dashboard"
  menu item instead.
- Move the log to `~/Library/Logs/SafeVoice/` with rotation for the shipped build
  (update CLAUDE.md and scripts when this lands).

**Phase C: first-run experience for cold users**
- Single-owner model download with progress, retry, and disk-space check; bundle
  the 4-bit model (~400 MB) as an alternative to downloading on first run.
- Make the native MLX backend the default LLM path (no external install) and
  demote Ollama to an advanced option; the wizard's "Install from ollama.com"
  dead-end goes away.
- Sequence permission prompts: explain BEFORE the Accessibility dialog appears
  (today HotkeyManager.start() fires the system dialog at launch, 1.5 s before the
  wizard explains why), then mic via the new in-wizard Allow button.
- Add a consent/disclosure step covering local history storage and (if enabled)
  cloud LLM data flow; surface a visible indicator whenever a cloud backend is
  active.
- Store API keys in the macOS Keychain instead of 0600 JSON.

**Phase D: ongoing**
- Sparkle (or similar) auto-update feed; crash reporting strategy; a
  `scripts/release.sh` that builds, signs, notarizes, staples, and produces the
  DMG.

## Part 4. Findings reviewed and intentionally NOT acted on

- Wizard NSTimer after closing the window with the X button: self-heals within one
  2 s tick (the timer's own visibility check invalidates it). Cosmetic.
- The speculative ASR thread is stop-flagged but not joined: with the session lock
  in place this is benign (it can only delay the final pass briefly).
- A UI-agent claim that `_inject_text` shows the overlay off the main thread was
  rejected: all overlay methods are `@_ensure_main_thread`-decorated.
- README claims aligned with reality in this branch (path, languages, hotkey,
  removed Ctrl+Space language cycle, honest privacy sentence); a full user-facing
  rewrite belongs with Phase B/C.
