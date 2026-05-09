#!/usr/bin/env bash
# Launch SafeVoice.app reliably from an iCloud-synced project path.
#
# The bundle lives under ~/Documents/, which is iCloud-synced. iCloud File
# Provider continuously stamps com.apple.FinderInfo and com.apple.fileprovider.fpfs#P
# xattrs onto the bundle root. These xattrs invalidate the ad-hoc signature
# from the perspective of Gatekeeper, even though `codesign -vv` still says
# "valid on disk". The symptom is a launched process that writes "SafeVoice
# starting" to /tmp/safevoice.log and then hangs forever (Gatekeeper allowed
# the bootstrap binary to run but won't let CPython execute).
#
# This launcher strips the offending xattrs, re-signs ad-hoc, and opens the
# bundle. Use it instead of `open dist/SafeVoice.app` if you ever see the
# hang. Cheap and idempotent on a clean bundle.

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
