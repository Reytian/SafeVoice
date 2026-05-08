#!/usr/bin/env bash
# Build SafeVoice.app in alias mode, working around two iCloud + macOS Tahoe traps:
#   1. setuptools' fetch_build_eggs (triggered by setup_requires=["py2app"]) hangs
#      for minutes on the deprecation path even though py2app is already in .venv.
#   2. iCloud File Provider stamps com.apple.FinderInfo and
#      com.apple.fileprovider.fpfs#P xattrs on the bundle, which makes py2app's
#      codesign_adhoc step fail with "Cannot sign bundle". The resulting
#      unsigned bundle then hangs at "SafeVoice starting" because Gatekeeper
#      refuses to let CPython run inside it.

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
APP="$ROOT/dist/SafeVoice.app"
VENV_PY="$ROOT/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
    echo "[build] missing $VENV_PY — create the venv first" >&2
    exit 1
fi

echo "[build] cleaning prior build/dist"
rm -rf build dist

echo "[build] running py2app -A (setup_requires stripped to avoid fetch_build_eggs hang)"
"$VENV_PY" - <<'PY'
import sys, setuptools
_orig_setup = setuptools.setup
def _patched_setup(**kw):
    kw.pop('setup_requires', None)
    return _orig_setup(**kw)
setuptools.setup = _patched_setup
sys.argv = ['setup.py', 'py2app', '-A']
exec(open('setup.py').read())
PY

if [ ! -x "$APP/Contents/MacOS/SafeVoice" ]; then
    echo "[build] py2app did not produce $APP — aborting" >&2
    exit 1
fi

echo "[build] stripping iCloud / Finder xattrs from bundle"
xattr -cr "$APP"

echo "[build] re-signing bundle (ad-hoc)"
codesign --force --deep --sign - "$APP"

echo "[build] verifying signature"
codesign -vv "$APP"

echo "[build] done — launch with: open $APP"
