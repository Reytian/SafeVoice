#!/usr/bin/env bash
# Build SafeVoice.app, working around two iCloud + macOS Tahoe traps:
#
# Usage:
#   scripts/build.sh                # alias build (-A): fast, symlinks to source (DEV ONLY)
#   scripts/build.sh --standalone   # self-contained bundle for distribution (Phase A)
#
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
# Override with SAFEVOICE_PYTHON when building from a git worktree or CI,
# where the venv lives outside the checkout.
VENV_PY="${SAFEVOICE_PYTHON:-$ROOT/.venv/bin/python}"

BUILD_MODE="alias"
if [ "${1:-}" = "--standalone" ]; then
    BUILD_MODE="standalone"
fi
export SAFEVOICE_BUILD_MODE="$BUILD_MODE"

if [ ! -x "$VENV_PY" ]; then
    echo "[build] missing $VENV_PY — create the venv first" >&2
    exit 1
fi

if [ "$BUILD_MODE" = "standalone" ]; then
    echo "[build] ensuring namespace-package shims for modulegraph"
    # py2app's modulegraph cannot scan PEP 420 namespace packages and dies
    # with "No module named 'mlx'". An empty __init__.py turns them into
    # regular packages with unchanged import semantics. Idempotent; pip
    # reinstalls of the packages remove the shim, this re-creates it.
    "$VENV_PY" - <<'PY'
import os, sysconfig
sp = sysconfig.get_paths()["purelib"]
for pkg in ("mlx", "mlx_lm", "mlx_qwen3_asr"):
    pkg_dir = os.path.join(sp, pkg)
    init = os.path.join(pkg_dir, "__init__.py")
    if os.path.isdir(pkg_dir) and not os.path.exists(init):
        with open(init, "w") as f:
            f.write("# Namespace-package shim for py2app/modulegraph "
                    "(see scripts/build.sh).\n")
        print(f"[build] shimmed {pkg}")
PY
fi

echo "[build] cleaning prior build/dist"
rm -rf build dist

echo "[build] running py2app ($BUILD_MODE mode; setup_requires stripped to avoid fetch_build_eggs hang)"
# py2app's internal codesign_adhoc step will fail because of iCloud xattrs,
# but the bundle is already produced by then -- we strip xattrs and re-sign
# below, so this expected failure must NOT abort the script. Disable -e for
# the py2app call only, then re-enable.
set +e
"$VENV_PY" - <<'PY'
import os, sys, setuptools
_orig_setup = setuptools.setup
def _patched_setup(**kw):
    kw.pop('setup_requires', None)
    return _orig_setup(**kw)
setuptools.setup = _patched_setup
args = ['setup.py', 'py2app']
if os.environ.get('SAFEVOICE_BUILD_MODE') != 'standalone':
    args.append('-A')
sys.argv = args
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
