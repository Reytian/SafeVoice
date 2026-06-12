#!/usr/bin/env bash
# Build a signed, notarized, stapled SafeVoice.app and wrap it in a DMG for
# distribution. This is Phase B of the audit's distribution plan.
#
# Prerequisites (one-time, see SIGNING.md):
#   1. Apple Developer Program membership.
#   2. A "Developer ID Application" certificate in your login keychain.
#   3. notarytool credentials stored as a keychain profile named "safevoice":
#        xcrun notarytool store-credentials safevoice \
#          --apple-id "you@example.com" --team-id TEAMID \
#          --password "app-specific-password"
#
# Usage:
#   scripts/release.sh
# Env overrides:
#   SAFEVOICE_PYTHON          python to build with (default ./.venv/bin/python)
#   SAFEVOICE_SIGN_IDENTITY   signing identity (default: first Developer ID Application)
#   SAFEVOICE_NOTARY_PROFILE  notarytool keychain profile (default "safevoice")
#   SAFEVOICE_SKIP_NOTARIZE=1 build + sign + DMG, skip the notarize/staple step

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
APP="$ROOT/dist/SafeVoice.app"
ENTITLEMENTS="$ROOT/assets/SafeVoice.entitlements"
VENV_PY="${SAFEVOICE_PYTHON:-$ROOT/.venv/bin/python}"
NOTARY_PROFILE="${SAFEVOICE_NOTARY_PROFILE:-safevoice}"
DMG_PATH="$ROOT/dist/SafeVoice.dmg"

# ---- 0. Resolve signing identity ------------------------------------------
SIGN_ID="${SAFEVOICE_SIGN_IDENTITY:-}"
if [ -z "$SIGN_ID" ]; then
    # `|| true`: no-match grep returns 1, which would abort under pipefail
    # before the explicit "no identity" error below can give a clear message.
    SIGN_ID="$(security find-identity -v -p codesigning 2>/dev/null \
        | grep "Developer ID Application" | head -1 \
        | sed -E 's/^[[:space:]]*[0-9]+\) [0-9A-F]+ "([^"]+)".*/\1/' || true)"
fi
if [ -z "$SIGN_ID" ]; then
    echo "[release] ERROR: no 'Developer ID Application' certificate found." >&2
    echo "          See SIGNING.md to create one, or set SAFEVOICE_SIGN_IDENTITY." >&2
    exit 1
fi
echo "[release] signing identity: $SIGN_ID"

# ---- 1. Standalone build ---------------------------------------------------
echo "[release] building standalone bundle"
SAFEVOICE_PYTHON="$VENV_PY" bash scripts/build.sh --standalone

# scripts/build.sh already signs (ad-hoc or Developer ID). For release we
# re-sign properly INSIDE-OUT with the hardened runtime: --deep does not sign
# nested code in the correct order for notarization, so sign every nested
# Mach-O first, then the app last.
echo "[release] stripping xattrs and signing inside-out (hardened runtime)"
xattr -cr "$APP" 2>/dev/null || true

# Sign nested dylibs / .so / frameworks first (deepest paths first).
while IFS= read -r f; do
    codesign --force --timestamp --options runtime \
        --sign "$SIGN_ID" "$f"
done < <(find "$APP/Contents" \( -name "*.dylib" -o -name "*.so" \) -type f | awk '{print length, $0}' | sort -rn | cut -d" " -f2-)

# Sign nested frameworks (e.g. Python.framework) at their versioned root.
while IFS= read -r fw; do
    codesign --force --timestamp --options runtime \
        --sign "$SIGN_ID" "$fw" 2>/dev/null || true
done < <(find "$APP/Contents/Frameworks" -name "*.framework" -type d 2>/dev/null)

# Sign the app bundle last, with entitlements.
codesign --force --timestamp --options runtime \
    --entitlements "$ENTITLEMENTS" \
    --sign "$SIGN_ID" "$APP"

echo "[release] verifying signature + hardened runtime"
codesign --verify --deep --strict --verbose=2 "$APP"
codesign -dvv "$APP" 2>&1 | grep -E "Authority|TeamIdentifier|flags|runtime"

# ---- 2. Wrap in a DMG ------------------------------------------------------
echo "[release] building DMG"
rm -f "$DMG_PATH"
STAGING="$(mktemp -d)"
cp -R "$APP" "$STAGING/"
ln -s /Applications "$STAGING/Applications"
hdiutil create -volname "SafeVoice" -srcfolder "$STAGING" \
    -ov -format UDZO "$DMG_PATH" >/dev/null
rm -rf "$STAGING"
codesign --force --timestamp --sign "$SIGN_ID" "$DMG_PATH"

# ---- 3. Notarize + staple --------------------------------------------------
if [ "${SAFEVOICE_SKIP_NOTARIZE:-0}" = "1" ]; then
    echo "[release] SAFEVOICE_SKIP_NOTARIZE=1 -- skipping notarize/staple"
    echo "[release] DMG (un-notarized): $DMG_PATH"
    exit 0
fi

echo "[release] submitting to Apple notary service (this can take minutes)"
xcrun notarytool submit "$DMG_PATH" \
    --keychain-profile "$NOTARY_PROFILE" \
    --wait

echo "[release] stapling the notarization ticket"
xcrun stapler staple "$DMG_PATH"
# Also staple the app inside, so a copied-out .app is self-validating.
xcrun stapler staple "$APP" 2>/dev/null || true

echo "[release] verifying Gatekeeper acceptance"
spctl --assess --type open --context context:primary-signature -v "$DMG_PATH" || true

cat <<EOF

[release] done.
  Signed + notarized DMG: $DMG_PATH
  Ship this file. On another Mac it opens without Gatekeeper warnings.
EOF
