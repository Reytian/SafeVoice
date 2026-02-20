#!/usr/bin/env python3
"""
Prototype: macOS menubar (status bar) app using the rumps library.

This script demonstrates:
  - Creating a menubar app with a custom title/icon
  - Nested menu items (language selection submenu)
  - Callbacks on menu click with state toggling
  - Dynamically updating the menubar title at runtime
  - Timer-based periodic callbacks

Requirements:
  - macOS (rumps wraps PyObjC / NSStatusBar)
  - rumps >= 0.4.0  (installed in the project venv)

Usage:
  source /Users/haotianyi/Documents/voice-ime/.venv/bin/activate
  python prototypes/test_menubar.py
"""

import rumps


class VoiceIMEMenubarApp(rumps.App):
    """A minimal menubar app that simulates a voice-input IME controller."""

    # ── Class-level constants ───────────────────────────────────────────
    TITLE_IDLE = "\U0001f3a4"       # Microphone emoji (idle state)
    TITLE_LISTENING = "\U0001f534"  # Red circle emoji  (listening state)

    LANGUAGES = ["English", "Chinese", "French"]

    def __init__(self):
        # Initialize the rumps.App with a unicode title shown in the menubar.
        # Setting `quit_button=None` lets us provide our own Quit item so we
        # can intercept the quit action if needed.
        super().__init__(
            name="VoiceIME",
            title=self.TITLE_IDLE,
            quit_button=None,
        )

        self.is_listening = False
        self.current_language = "English"

        # ── Build the menu ──────────────────────────────────────────────
        # 1. Start / Stop toggle
        self.toggle_item = rumps.MenuItem(
            "Start Listening",
            callback=self.on_toggle_listening,
        )

        # 2. Language submenu
        self.language_menu = rumps.MenuItem("Language")
        for lang in self.LANGUAGES:
            item = rumps.MenuItem(lang, callback=self.on_language_selected)
            if lang == self.current_language:
                item.state = True  # checkmark beside the active language
            self.language_menu.add(item)

        # 3. Separator + Quit
        self.quit_item = rumps.MenuItem("Quit VoiceIME", callback=self.on_quit)

        # Assemble the top-level menu.  `None` inserts a separator line.
        self.menu = [
            self.toggle_item,
            None,
            self.language_menu,
            None,
            self.quit_item,
        ]

    # ── Callbacks ───────────────────────────────────────────────────────

    def on_toggle_listening(self, sender):
        """Toggle between listening and idle states."""
        self.is_listening = not self.is_listening

        if self.is_listening:
            sender.title = "Stop Listening"
            self.title = self.TITLE_LISTENING
            print("[VoiceIME] Started listening "
                  f"(language={self.current_language})")
        else:
            sender.title = "Start Listening"
            self.title = self.TITLE_IDLE
            print("[VoiceIME] Stopped listening")

    def on_language_selected(self, sender):
        """Handle language selection; update checkmarks."""
        selected_lang = sender.title
        if selected_lang == self.current_language:
            return  # already active, nothing to do

        # Clear all checkmarks, then set the new one.
        for lang in self.LANGUAGES:
            self.language_menu[lang].state = False
        sender.state = True

        self.current_language = selected_lang
        print(f"[VoiceIME] Language changed to: {self.current_language}")

    def on_quit(self, _sender):
        """Gracefully quit the application."""
        print("[VoiceIME] Quitting...")
        rumps.quit_application()

    # ── Timer example (optional) ────────────────────────────────────────
    # rumps.timer is a decorator that fires a callback on an interval.
    # Uncomment the block below to see the title flash every 2 seconds
    # while the app is in "listening" mode.

    # @rumps.timer(2)
    # def flash_title(self, timer):
    #     """Alternate the menubar title to simulate an activity indicator."""
    #     if not self.is_listening:
    #         return
    #     if self.title == self.TITLE_LISTENING:
    #         self.title = self.TITLE_IDLE
    #     else:
    #         self.title = self.TITLE_LISTENING


# ── Standalone demonstration of dynamic title updates ────────────────────
def demo_dynamic_title():
    """
    Show how to change the menubar title from *outside* the class, e.g.
    from another thread or an async callback.

    In practice you would hold a reference to the app instance and call:

        app.title = "new text or emoji"

    This updates the NSStatusItem title on the main thread via rumps'
    internal dispatch mechanism.
    """
    print(
        "[demo] To update the menubar title at runtime, simply assign to "
        "app.title:\n"
        '         app.title = "REC"      # show "REC" text\n'
        '         app.title = "\\U0001f3a4"  # show microphone emoji\n'
        '         app.icon  = "mic.png"  # or use an image file\n'
    )


# ── Entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_dynamic_title()

    app = VoiceIMEMenubarApp()
    print("[VoiceIME] Launching menubar app... (Cmd+Q or use menu to quit)")
    app.run()
