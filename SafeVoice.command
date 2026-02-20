#!/bin/bash
# SafeVoice - Voice Input for macOS
# Double-click this file to launch SafeVoice.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV" ]; then
    echo "Virtual environment not found. Setting up..."
    python3 -m venv "$VENV"
    "$VENV/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"
fi

cd "$SCRIPT_DIR"
exec "$VENV/bin/python" run.py
