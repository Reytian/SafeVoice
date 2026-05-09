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
# py2app's internal codesign_adhoc step will fail because of iCloud xattrs,
# but the bundle is already produced by then -- we strip xattrs and re-sign
# below, so this expected failure must NOT abort the script. Disable -e for
# the py2app call only, then re-enable.
set +e
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
PY2APP_EXIT=$?
set -e

if [ ! -x "$APP/Contents/MacOS/SafeVoice" ]; then
    echo "[build] py2app did not produce $APP (exit $PY2APP_EXIT) — aborting" >&2
    exit 1
fi
if [ "$PY2APP_EXIT" -ne 0 ]; then
    echo "[build] py2app exited $PY2APP_EXIT (expected -- internal codesign_adhoc"
    echo "        fails on iCloud xattrs; bundle was produced anyway, recovering)"
fi

echo "[build] stripping iCloud / Finder xattrs from bundle"
# xattr -cr exits non-zero when it encounters broken symlinks (alias-mode
# bundles contain symlinks to non-existent .venv files); the strip itself
# still succeeds for everything else. Tolerate the exit code.
xattr -cr "$APP" 2>/dev/null || true

echo "[build] re-signing bundle (ad-hoc)"
codesign --force --deep --sign - "$APP"

echo "[build] verifying signature"
codesign -vv "$APP"

# iCloud File Provider re-stamps com.apple.FinderInfo and fileprovider.fpfs#P
# onto the bundle root within seconds. If the user opens the app after that
# happens, Gatekeeper rejects the bundle and the launch hangs at "SafeVoice
# starting". Document the workaround prominently so the user knows what to
# do if launch hangs (rather than having to re-discover this pattern).
cat <<EOF
[build] done.

Launch:   open $APP

If launch hangs at "SafeVoice starting" (iCloud re-stamped the bundle
between build and launch), recover with:
    xattr -cr "$APP" 2>/dev/null
    codesign --force --deep --sign - "$APP"
    open "$APP"
EOF
