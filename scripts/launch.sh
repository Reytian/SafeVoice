#!/usr/bin/env bash
# Launch SafeVoice.app, defensively recovering from stale-signature hangs.
#
# History: this project used to live under ~/Documents/ (iCloud-synced).
# iCloud File Provider would stamp com.apple.FinderInfo and
# com.apple.fileprovider.fpfs#P xattrs onto the app bundle, invalidating the
# ad-hoc signature from Gatekeeper's view (spctl "rejected" even when
# `codesign -vv` said "valid on disk"). Symptom: the launched process writes
# "SafeVoice starting" to /tmp/safevoice.log and then hangs forever.
#
# The project was moved to ~/Developer/voice-ime (NOT iCloud-synced) on
# 2026-05-15, which fixes the root cause. This launcher keeps the xattr-strip
# + re-sign step anyway: it is cheap, idempotent on a clean bundle, and a
# harmless safety net. Prefer it over `open dist/SafeVoice.app`.

set -euo pipefail

cd "$(dirname "$0")/.."
APP="$(pwd)/dist/SafeVoice.app"

if [ ! -d "$APP" ]; then
    echo "[launch] $APP not found -- run scripts/build.sh first" >&2
    exit 1
fi

# Stop any running instance first so the new one becomes the canonical
# process (single_instance.py would otherwise turn this launch away).
pkill -f "$APP/Contents/MacOS" 2>/dev/null || true
sleep 1

xattr -cr "$APP" 2>/dev/null || true
codesign --force --deep --sign - "$APP" >/dev/null 2>&1
codesign -vv "$APP" >/dev/null 2>&1 || {
    echo "[launch] codesign verification failed after re-sign" >&2
    exit 1
}

open "$APP"
echo "[launch] $APP opened. Tail /tmp/safevoice.log for diagnostics."
