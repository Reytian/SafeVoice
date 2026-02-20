#!/usr/bin/env python3
"""SafeVoice - Voice Input Method for macOS.

Launch the SafeVoice menubar app.

Usage:
    python run.py
"""

import logging
import sys
import traceback

# Set up file logging so crashes are captured even if stdout is eaten by rumps
logging.basicConfig(
    filename="/tmp/safevoice.log",
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
# Also log to stderr
logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))

logger = logging.getLogger("safevoice")


def _main():
    try:
        logger.info("SafeVoice starting")
        from src.app import main
        main()
    except SystemExit as e:
        logger.info("SafeVoice exited with code %s", e.code)
    except Exception:
        logger.critical("SafeVoice crashed:\n%s", traceback.format_exc())
        sys.exit(1)


# Support both direct execution and py2app's exec()-based launch
if __name__ == "__main__":
    _main()
else:
    # py2app executes via exec() where __name__ != "__main__"
    _main()
