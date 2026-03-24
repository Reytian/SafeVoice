"""Setup wizard for first-time SafeVoice users."""
import logging
import subprocess
import threading

import objc
from AppKit import (
    NSWindow, NSView, NSTextField, NSButton, NSFont, NSColor,
    NSMakeRect, NSBackingStoreBuffered,
    NSWindowStyleMaskTitled, NSWindowStyleMaskClosable,
    NSScreen, NSObject, NSTextAlignmentCenter, NSAnimationContext,
)
from Foundation import NSTimer

logger = logging.getLogger(__name__)


STEPS = ["Welcome", "Demo", "Permissions", "Model", "Test", "Ready"]


class _WizardButtonTarget(NSObject):
    """NSObject target for wizard button actions. Prevents GC."""

    def initWithCallback_(self, callback):
        self = objc.super(_WizardButtonTarget, self).init()
        if self is None:
            return None
        self._callback = callback
        return self

    def invoke_(self, sender):
        if self._callback:
            self._callback()


class SetupWizard:
    """6-step onboarding wizard for first-time users."""

    def __init__(self, app_ref, on_complete=None):
        self._app = app_ref
        self._on_complete = on_complete
        self._current_step = 0
        self._window = None
        self._content_view = None
        self._targets = set()
        self._perm_timer = None

    def show(self):
        frame = NSMakeRect(0, 0, 520, 420)
        style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
        self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, style, NSBackingStoreBuffered, False,
        )
        self._window.setTitle_("SafeVoice Setup")
        self._window.center()
        self._window.setLevel_(3)
        self._content_view = self._window.contentView()
        self._render_step()
        self._window.makeKeyAndOrderFront_(None)

    def _clear_content(self):
        for subview in list(self._content_view.subviews()):
            subview.removeFromSuperview()

    def _render_step(self):
        self._clear_content()
        self._render_progress()
        step = STEPS[self._current_step]
        getattr(self, f"_render_{step.lower()}")()

    def _render_progress(self):
        y = 380
        total_width = 300
        seg_w = total_width / len(STEPS) - 4
        start_x = (520 - total_width) / 2
        for i in range(len(STEPS)):
            seg = NSView.alloc().initWithFrame_(
                NSMakeRect(start_x + i * (seg_w + 4), y, seg_w, 6)
            )
            seg.setWantsLayer_(True)
            if i <= self._current_step:
                seg.layer().setBackgroundColor_(
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.7, 0.0, 1.0).CGColor()
                )
            else:
                seg.layer().setBackgroundColor_(
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(0.85, 0.85, 0.85, 1.0).CGColor()
                )
            seg.layer().setCornerRadius_(3)
            self._content_view.addSubview_(seg)

    def _render_welcome(self):
        self._label("SafeVoice", 28, bold=True, y=300, center=True, height=36)
        self._label("Speak, and it types.", 16, y=260, center=True, height=24)
        self._label(
            "100% on-device voice input for macOS.\nNo internet required. Your voice never leaves your Mac.",
            13, y=185, center=True, width=400, height=50,
        )
        self._button("Get Started", y=80, action=self._next_step)

    def _render_demo(self):
        self._label("How it works", 22, bold=True, y=320, center=True, height=30)
        self._label(
            "Hold your hotkey to speak. A floating bar appears showing\nyour words in real-time. Release to inject text at your cursor.",
            13, y=255, center=True, width=420, height=40,
        )
        self._label(
            "  Recording      Processing      Done  \n  \u25cf Listening       \u25cf AI working      \u2713 Injected",
            13, y=170, center=True, width=400, height=40,
        )
        self._button("Back", y=80, x=140, action=self._prev_step, secondary=True)
        self._button("Next", y=80, x=280, action=self._next_step)

    def _render_permissions(self):
        self._label("Permissions", 22, bold=True, y=320, center=True, height=30)

        self._label("Microphone Access", 14, bold=True, y=270, x=60)
        self._label("Required to hear your voice", 12, y=250, x=60)
        self._mic_status = self._label("", 12, y=270, x=370, width=80)
        self._button("Open", y=266, x=420, width=60, action=self._open_mic_prefs, secondary=True)

        self._label("Accessibility Access", 14, bold=True, y=200, x=60)
        self._label("Required to type text into other apps", 12, y=180, x=60)
        self._acc_status = self._label("", 12, y=200, x=370, width=80)
        self._button("Open", y=196, x=420, width=60, action=self._open_acc_prefs, secondary=True)

        self._update_permission_display()
        self._start_permission_polling()

        self._button("Back", y=80, x=140, action=self._prev_step, secondary=True)
        self._button("Next", y=80, x=280, action=self._next_step)

    def _update_permission_display(self):
        try:
            import sounddevice
            sounddevice.query_devices(kind="input")
            self._mic_status.setStringValue_("\u2713 Granted")
            self._mic_status.setTextColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.2, 0.7, 0.2, 1.0)
            )
        except Exception:
            self._mic_status.setStringValue_("\u2717 Needed")
            self._mic_status.setTextColor_(NSColor.redColor())

        from ApplicationServices import AXIsProcessTrusted
        if AXIsProcessTrusted():
            self._acc_status.setStringValue_("\u2713 Granted")
            self._acc_status.setTextColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.2, 0.7, 0.2, 1.0)
            )
        else:
            self._acc_status.setStringValue_("\u2717 Needed")
            self._acc_status.setTextColor_(NSColor.redColor())

    def _start_permission_polling(self):
        if self._perm_timer:
            self._perm_timer.invalidate()

        def check(timer):
            if not self._window or not self._window.isVisible():
                timer.invalidate()
                return
            if self._current_step != 2:
                return
            self._update_permission_display()

        self._perm_timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            2.0, True, check,
        )

    def _open_mic_prefs(self):
        subprocess.Popen([
            "open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
        ])

    def _open_acc_prefs(self):
        subprocess.Popen([
            "open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
        ])

    def _render_model(self):
        self._label("Models Setup", 22, bold=True, y=320, center=True, height=30)
        self._label(
            "SafeVoice needs two models:\n"
            "1. ASR model for speech recognition (~1.2 GB)\n"
            "2. LLM model for text cleanup (via Ollama)",
            13, y=250, center=True, width=420, height=60,
        )

        # --- ASR Model Status ---
        from .llm_backend import is_asr_model_downloaded, OllamaBackend

        asr_model_id = self._app._settings.get("asr_model", "Qwen/Qwen3-ASR-0.6B")
        asr_ready = is_asr_model_downloaded(asr_model_id)

        if asr_ready:
            asr_status = self._label("\u2713 ASR model ready", 13, y=195, x=60)
            asr_status.setTextColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.2, 0.7, 0.2, 1.0)
            )
        else:
            self._asr_status = self._label("\u2717 ASR model not downloaded", 13, y=195, x=60, width=200)
            self._asr_status.setTextColor_(NSColor.redColor())

            dl_btn = self._button("Download ASR", y=191, x=280, width=100, action=self._download_asr)

        # --- LLM Model Status ---
        llm_model = self._app._settings.get("llm_local_model", "qwen2.5:3b")
        ollama_models = OllamaBackend.list_models()
        ollama_running = len(ollama_models) > 0
        llm_ready = any(m["name"] == llm_model or m["name"] == f"{llm_model}:latest" for m in ollama_models)

        if llm_ready:
            llm_status = self._label(f"\u2713 LLM model ready ({llm_model})", 13, y=165, x=60)
            llm_status.setTextColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.2, 0.7, 0.2, 1.0)
            )
        elif ollama_running:
            self._llm_status = self._label(f"\u2717 LLM model not installed ({llm_model})", 13, y=165, x=60, width=220)
            self._llm_status.setTextColor_(NSColor.redColor())

            pull_btn = self._button("Pull LLM", y=161, x=280, width=100, action=self._download_llm)
        else:
            self._label(
                "\u2717 Ollama not detected. Install from ollama.com\n"
                "   (Optional \u2014 SafeVoice works without text cleanup)",
                12, y=160, x=60, width=320, height=30,
            )

        # Navigation
        self._button("Back", y=80, x=140, action=self._prev_step, secondary=True)
        self._button("Next", y=80, x=280, action=self._next_step)

        # Auto-start downloads if models are missing
        if not asr_ready and hasattr(self, '_asr_status'):
            self._download_asr()
        if ollama_running and not llm_ready and hasattr(self, '_llm_status'):
            self._download_llm()

    def _render_test(self):
        self._label("Try it out!", 22, bold=True, y=320, center=True, height=30)
        self._label(
            "Hold your activation hotkey and say something.\nCheck that the floating bar appears and text is transcribed.",
            13, y=255, center=True, width=420, height=40,
        )
        self._label("You can skip this step and test later.", 12, y=200, center=True, width=300)
        self._button("Back", y=80, x=100, action=self._prev_step, secondary=True)
        self._button("Skip", y=80, x=200, action=self._next_step, secondary=True)
        self._button("Next", y=80, x=310, action=self._next_step)

    def _render_ready(self):
        check = self._label("\u2713", 48, y=300, center=True, height=60)
        check.setTextColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.2, 0.7, 0.2, 1.0)
        )
        self._label("You're all set!", 22, bold=True, y=265, center=True, height=30)

        hotkey_display = "Option+Space"
        try:
            hk = self._app._settings.get("activate_hotkey", {})
            from .settings_window import format_hotkey
            hotkey_display = format_hotkey(hk)
        except Exception:
            pass

        self._label(
            f"Hold {hotkey_display} to speak.\nText is typed at your cursor on release.\n\nAccess settings from the menu bar icon.",
            13, y=150, center=True, width=400, height=80,
        )
        self._button("Start Using SafeVoice", y=60, width=200, action=self._finish)

    def _download_asr(self):
        """Download ASR model in background."""
        self._asr_status.setStringValue_("Downloading ASR model...")
        self._asr_status.setTextColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.7, 0.0, 1.0)
        )

        def _do_download():
            try:
                from huggingface_hub import snapshot_download
                model_id = self._app._settings.get("asr_model", "Qwen/Qwen3-ASR-0.6B")
                snapshot_download(model_id, local_files_only=False)

                # Update UI on main thread
                from AppKit import NSObject as _NSO
                class _Updater(_NSO):
                    def update_(self_, sender):
                        self._asr_status.setStringValue_("\u2713 ASR model ready!")
                        self._asr_status.setTextColor_(
                            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.2, 0.7, 0.2, 1.0)
                        )
                u = _Updater.alloc().init()
                self._targets.add(u)
                u.performSelectorOnMainThread_withObject_waitUntilDone_("update:", None, False)
            except Exception as e:
                logger.warning("ASR download failed: %s", e)
                self._asr_status.setStringValue_(f"Download failed: {e}")

        threading.Thread(target=_do_download, daemon=True).start()

    def _download_llm(self):
        """Pull Ollama LLM model in background."""
        self._llm_status.setStringValue_("Pulling LLM model...")
        self._llm_status.setTextColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.7, 0.0, 1.0)
        )

        def _do_pull():
            try:
                model = self._app._settings.get("llm_local_model", "qwen2.5:3b")
                result = subprocess.run(
                    ["ollama", "pull", model],
                    capture_output=True, text=True, timeout=600,
                )

                from AppKit import NSObject as _NSO
                class _Updater(_NSO):
                    def update_(self_, sender):
                        if result.returncode == 0:
                            self._llm_status.setStringValue_("\u2713 LLM model ready!")
                            self._llm_status.setTextColor_(
                                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.2, 0.7, 0.2, 1.0)
                            )
                        else:
                            self._llm_status.setStringValue_(f"Pull failed: {result.stderr[:40]}")
                u = _Updater.alloc().init()
                self._targets.add(u)
                u.performSelectorOnMainThread_withObject_waitUntilDone_("update:", None, False)
            except Exception as e:
                logger.warning("LLM pull failed: %s", e)
                self._llm_status.setStringValue_(f"Pull failed: {e}")

        threading.Thread(target=_do_pull, daemon=True).start()

    # --- Helpers ---

    def _label(self, text, size, bold=False, y=0, x=None, center=False, width=300, height=24):
        if x is None:
            x = (520 - width) / 2 if center else 60
        label = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, width, height))
        label.setStringValue_(text)
        label.setFont_(NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size))
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        if center:
            label.setAlignment_(NSTextAlignmentCenter)
        self._content_view.addSubview_(label)
        return label

    def _button(self, title, y, x=None, width=120, action=None, secondary=False):
        if x is None:
            x = (520 - width) / 2
        btn = NSButton.alloc().initWithFrame_(NSMakeRect(x, y, width, 36))
        btn.setTitle_(title)
        btn.setBezelStyle_(0 if secondary else 1)
        if not secondary:
            btn.setKeyEquivalent_("\r")
        if action:
            target = _WizardButtonTarget.alloc().initWithCallback_(action)
            self._targets.add(target)
            btn.setTarget_(target)
            btn.setAction_(objc.selector(target.invoke_, signature=b"v@:@"))
        self._content_view.addSubview_(btn)
        return btn

    def _next_step(self):
        if self._current_step < len(STEPS) - 1:
            self._current_step += 1
            self._render_step()

    def _prev_step(self):
        if self._current_step > 0:
            self._current_step -= 1
            self._render_step()

    def _finish(self):
        if self._perm_timer:
            self._perm_timer.invalidate()
            self._perm_timer = None
        self._window.close()
        if self._on_complete:
            self._on_complete()
