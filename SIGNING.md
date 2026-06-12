# Code Signing & Notarization (Phase B)

SafeVoice is signed with a Developer ID identity for two reasons:

1. **It fixes the recurring "hotkey dead after reboot" bug.** macOS does not
   reliably persist an Accessibility (TCC) permission grant for an *ad-hoc*
   signed app across a reboot or relogin: there is no stable signing identity
   to anchor the grant to. A Developer ID signature (or any real certificate)
   makes the grant survive. This applies to your everyday alias build too.
2. **It makes the app distributable.** Notarized + stapled, SafeVoice opens
   on any Mac without the Gatekeeper "unidentified developer" wall.

The build scripts are already wired for this. They use a Developer ID
identity automatically once one exists in your keychain; until then they fall
back to ad-hoc. You only need to do the one-time setup below.

## One-time setup

### 1. Apple Developer Program
Join at <https://developer.apple.com/programs/> ($99/year). An individual
membership is enough.

### 2. Create a "Developer ID Application" certificate
Easiest path, with Xcode installed:

1. Xcode > Settings > Accounts > add your Apple ID.
2. Select your team > **Manage Certificates...**
3. Click **+** > **Developer ID Application**.
4. It is created on Apple's servers and downloaded into your login keychain.

Verify it landed:

```bash
security find-identity -v -p codesigning | grep "Developer ID Application"
```

You should see one line like:
`1) ABC123... "Developer ID Application: Your Name (TEAMID)"`

### 3. Store notarization credentials
Create an app-specific password at <https://account.apple.com> (Sign-In &
Security > App-Specific Passwords), then:

```bash
xcrun notarytool store-credentials safevoice \
  --apple-id "you@example.com" \
  --team-id  TEAMID \
  --password "abcd-efgh-ijkl-mnop"
```

`TEAMID` is the parenthesized code from the certificate line above.

## Daily use (alias build)

Nothing changes in your workflow:

```bash
scripts/build.sh
```

Once the certificate exists, this signs with it automatically (look for
`[build] signing bundle with identity: ...`). The Accessibility grant then
survives reboots. You can pin a specific identity with
`export SAFEVOICE_SIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"`.

After the FIRST Developer-ID-signed build you must re-grant Accessibility
once (the identity changed from ad-hoc): clear the old entry and re-grant,

```bash
pkill -f "voice-ime/dist/SafeVoice.app/Contents/MacOS"
tccutil reset Accessibility com.safevoice.app
scripts/build.sh && bash scripts/launch.sh
```

then approve the prompt. From then on it persists across reboots.

## Shipping a release

```bash
scripts/release.sh
```

This produces a signed + notarized + stapled `dist/SafeVoice.dmg`:
standalone build → inside-out hardened-runtime signing with
`assets/SafeVoice.entitlements` → notarize → staple. To dry-run the build
and signing without contacting Apple, `SAFEVOICE_SKIP_NOTARIZE=1
scripts/release.sh`.

## Why not the Mac App Store / App Sandbox

SafeVoice uses `CGEventTap` (global hotkey) and `CGEventPost` (paste), which
are incompatible with the App Sandbox the App Store requires. Direct
notarized distribution is the only path. The entitlements file therefore
enables the hardened runtime but does **not** sandbox.
