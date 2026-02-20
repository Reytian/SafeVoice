# Voice IME - UI Design Specification

**App Name**: SafeVoice (working title)
**Platform**: macOS 14+ (Sonoma and later)
**Design Language**: Native macOS, SwiftUI, SF Symbols, system fonts, vibrancy
**Philosophy**: Invisible until needed. Zero chrome. Feels built into macOS.

---

## 0. Design Principles

1. **Invisible by default** - No dock icon. Lives in the menubar. You forget it's a separate app.
2. **One-action activation** - Single hotkey press to start/stop. No menus, no clicks required.
3. **Instant feedback** - The moment you speak, you see proof the app is listening.
4. **Auto-dismiss** - Everything disappears on its own. You never close a window.
5. **System-native** - Follows dark/light mode. Uses system blur, SF Symbols, SF Pro font. Indistinguishable from a built-in macOS feature.

---

## 1. Menubar Icon

The menubar icon is the app's only persistent UI element.

### States

| State | Icon | Description |
|-------|------|-------------|
| **Idle** | `mic.fill` (SF Symbol) | Standard menubar weight. Matches system icon density. 18x18 pt template image. |
| **Listening** | `mic.fill` with accent tint | Icon tints to system accent color (default: blue). Subtle 0.8s pulse animation (opacity 1.0 → 0.6 → 1.0, ease-in-out, repeating). |
| **Processing** | `ellipsis.circle` (SF Symbol) | Rotating animation (360deg, 1.2s, linear, repeating). Returns to idle mic once text is inserted. |
| **Error** | `mic.slash.fill` | Shown briefly (2s) on microphone error or model failure, then returns to idle. |
| **Disabled** | `mic.slash` (outlined) | Microphone permission not granted, or app paused. |

### Click Behavior

- **Left-click**: Toggle listening on/off (same as hotkey).
- **Right-click** (or Control-click): Opens context menu:

```
┌─────────────────────────────┐
│  ● Chinese (中文)           │  ← Current language indicated
│    English                  │     by filled circle
│    French (Français)        │
│ ─────────────────────────── │
│  Model: 0.6B (Fast)    ▸   │  ← Submenu: 0.6B / 1.7B
│ ─────────────────────────── │
│  Settings...          ⌘,   │
│  Quit SafeVoice       ⌘Q   │
└─────────────────────────────┘
```

### Typography & Sizing

- Menu items: SF Pro, 13pt regular (system default)
- Language names include native script in parentheses for instant recognition
- Keyboard shortcuts right-aligned in system shortcut style

---

## 2. Floating Pill (Listening Overlay)

The core interaction surface. Appears when the user activates voice input and floats above all windows.

### Layout

```
┌──────────────────────────────────────────────┐
│                                              │
│   🟢  中文   ▍▎▍▌▎▍▎▌▍▎   你好世界...       │
│                                              │
└──────────────────────────────────────────────┘
 ↑        ↑          ↑              ↑
 status   lang    waveform     live preview
 dot      badge   animation    (transcription)
```

### Specifications

| Property | Value |
|----------|-------|
| **Width** | Dynamic, 280-480 pt (grows with transcription text, max 480) |
| **Height** | 44 pt |
| **Corner radius** | 22 pt (fully rounded, capsule shape) |
| **Background** | `.ultraThinMaterial` (system vibrancy/blur) |
| **Border** | 0.5 pt, `Color.primary.opacity(0.1)` |
| **Shadow** | `shadow(color: .black.opacity(0.15), radius: 12, y: 4)` |
| **Position** | Centered horizontally, 80 pt from top of active screen |
| **Window level** | `.floating` (above all windows, below system alerts) |
| **Padding** | 12 pt horizontal, 8 pt vertical |

### Sub-elements

#### Status Dot (left edge)
- 8x8 pt circle
- **Listening**: Green (`Color.green`), subtle pulse animation (scale 1.0→1.3→1.0, 1.2s, repeating)
- **Processing**: Orange (`Color.orange`), no animation
- **Error**: Red (`Color.red`), shown for 2s before pill auto-dismisses

#### Language Badge
- SF Pro Medium, 13pt
- Displays short label: `中文` / `EN` / `FR`
- Tappable: click to cycle to next language
- Subtle background: `Color.primary.opacity(0.06)`, corner radius 4pt, padding 4pt horizontal

#### Waveform Visualization
- 5 vertical bars, each 3pt wide, 2pt gap between bars
- Height range: 4pt (silence) to 20pt (loud)
- Animated in real-time from microphone input amplitude
- Color: `Color.primary.opacity(0.5)`
- Bars animate with `.spring(response: 0.15)` for organic movement
- When processing (not listening), bars freeze at mid-height and fade to 30% opacity

#### Live Transcription Preview
- SF Pro Regular, 13pt, `Color.primary`
- Shows the latest partial transcription result
- Truncated with `...` if longer than available space (single line, no wrapping)
- Updates in real-time as ASR produces partial results
- When empty (no speech detected yet): shows `Listening...` in `Color.secondary`

### Animations

- **Appear**: Scale from 0.8 to 1.0 + opacity 0→1, duration 0.2s, `.spring(response: 0.35, dampingFraction: 0.7)`
- **Dismiss**: Scale 1.0→0.9 + opacity 1→0, duration 0.15s, `.easeOut`
- **Auto-dismiss**: Pill disappears 0.5s after transcription is committed to the active text field

### Interaction

- The pill is **not draggable** (keeps interaction dead simple)
- Clicking the pill anywhere (except the language badge) toggles listening off
- Escape key also dismisses the pill and cancels transcription

---

## 3. Language Switcher

### Activation Methods

| Method | Action |
|--------|--------|
| **Keyboard shortcut** | `Control + Space` (configurable). Cycles: Chinese → English → French → Chinese. Works even when pill is visible. |
| **Click language badge** | On the floating pill, click the language label to cycle forward. |
| **Menubar context menu** | Right-click menubar icon → select language. |

### Visual Feedback on Switch

When language changes, the floating pill (if visible) shows a brief transition:

1. Language badge text cross-fades to new language (0.2s)
2. A small toast appears below the pill (if pill is not visible):

```
┌─────────────┐
│  EN English  │
└─────────────┘
```

- Same material background as pill
- 120x32 pt, corner radius 16pt
- Appears for 1.2s, then fades out (0.2s ease-out)
- Position: center screen, 120pt from top

### Language Labels

| Language | Pill Label | Toast Label | Full Menu Label |
|----------|-----------|-------------|-----------------|
| Chinese | 中文 | 中文 Chinese | Chinese (中文) |
| English | EN | EN English | English |
| French | FR | FR Français | French (Français) |

---

## 4. Settings Panel

A standard macOS settings window. Opens from menubar context menu or `⌘,`.

### Window Specs

| Property | Value |
|----------|-------|
| **Size** | 520 x 440 pt (fixed, not resizable) |
| **Style** | `.titled`, `.closable`, not miniaturizable, not resizable |
| **Title** | "SafeVoice Settings" |
| **Toolbar style** | `.unified` (modern macOS toolbar) |

### Tab Structure

Uses macOS-native tab bar (SF Symbols + labels):

```
[  🎤 General  |  🌐 Languages  |  ⚙️ Advanced  ]
```

---

### Tab 1: General

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  Activation                                     │
│  ┌───────────────────────────────────────────┐  │
│  │  Hotkey          [ ⌥ Space ◉ Record ]    │  │
│  │  Mode                ◉ Hold to talk      │  │
│  │                      ○ Toggle on/off      │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  Input                                          │
│  ┌───────────────────────────────────────────┐  │
│  │  Microphone      [ MacBook Pro Mic  ▾ ]  │  │
│  │  Input level     ▓▓▓▓▓▓░░░░░░  (live)   │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  Behavior                                       │
│  ┌───────────────────────────────────────────┐  │
│  │  ☑ Launch at login                        │  │
│  │  ☑ Play sound on start/stop               │  │
│  │  ☐ Show floating pill                     │  │
│  │  ☐ Auto-punctuate                         │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
└─────────────────────────────────────────────────┘
```

#### Hotkey Recorder
- Standard macOS key recorder control (like in Raycast/Rectangle)
- Click the field, press desired key combination, it captures
- Default: `Option + Space`
- Shows current binding as a key cap visual: `⌥ Space`

#### Mode Toggle
- **Hold to talk** (default, recommended): Hold hotkey → speak → release → text is inserted. Matches Superwhisper/Typeless model. Lowest friction.
- **Toggle on/off**: Press once to start, press again to stop. Better for longer dictation.

#### Microphone Selector
- Standard dropdown populated from `AVCaptureDevice.DiscoverySession`
- Shows a live input level meter (horizontal bar, green) when settings are open, so users can verify the right mic is selected

---

### Tab 2: Languages

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  Active Languages                               │
│  Drag to reorder. First language is default.    │
│  ┌───────────────────────────────────────────┐  │
│  │  ≡  ☑  中文 Chinese              ⌃1     │  │
│  │  ≡  ☑  EN  English               ⌃2     │  │
│  │  ≡  ☑  FR  Français              ⌃3     │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  Language Cycling Shortcut                      │
│  ┌───────────────────────────────────────────┐  │
│  │  Cycle shortcut   [ ⌃ Space ◉ Record ]  │  │
│  │  ☑ Show language toast on switch          │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  Auto-Detection                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  ○ Manual selection only                  │  │
│  │  ◉ Auto-detect language (recommended)     │  │
│  │    Uses Qwen3-ASR's language ID (96.8%    │  │
│  │    accuracy). Falls back to selected       │  │
│  │    language on low confidence.             │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
└─────────────────────────────────────────────────┘
```

- Drag handles (`≡`) allow reordering
- Checkboxes enable/disable languages from the cycle
- Direct shortcut keys (`⌃1`, `⌃2`, `⌃3`) for jumping to a specific language
- Auto-detect mode leverages Qwen3-ASR's 96.8% language ID accuracy

---

### Tab 3: Advanced

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  Model                                          │
│  ┌───────────────────────────────────────────┐  │
│  │  ◉ Qwen3-ASR 0.6B (Fast)                 │  │
│  │    ~400 MB · Best for quick input          │  │
│  │    WER: 2.3% EN / 3.2% ZH                │  │
│  │                                            │  │
│  │  ○ Qwen3-ASR 1.7B (Accurate)              │  │
│  │    ~1.2 GB · Best for long dictation       │  │
│  │    WER: 1.6% EN / 2.7% ZH                │  │
│  │                                            │  │
│  │  Status: ● 0.6B loaded · 1.7B not         │  │
│  │          downloaded                        │  │
│  │          [ Download 1.7B ]                 │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  Performance                                    │
│  ┌───────────────────────────────────────────┐  │
│  │  Quantization       [ 8-bit (Balanced) ▾] │  │
│  │    Options: fp16 / 8-bit / 4-bit          │  │
│  │  ☑ Keep model loaded in memory             │  │
│  │    (Faster first response, uses ~700 MB)   │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  Output                                         │
│  ┌───────────────────────────────────────────┐  │
│  │  ☑ Insert text at cursor position         │  │
│  │  ☐ Copy to clipboard instead               │  │
│  │  ☐ Append newline after insertion          │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
└─────────────────────────────────────────────────┘
```

- Model cards show size, speed profile, and accuracy side by side for easy comparison
- Download button appears only for models not yet on disk
- Progress bar replaces button during download
- Quantization dropdown with brief descriptions
- "Keep model loaded" toggle for trading memory for latency

### Typography (all tabs)

| Element | Font |
|---------|------|
| Section headers | SF Pro, 13pt semibold, `Color.primary` |
| Labels | SF Pro, 13pt regular, `Color.primary` |
| Descriptions / hints | SF Pro, 11pt regular, `Color.secondary` |
| Key caps | SF Mono, 12pt medium, rounded rect background |

### Colors

All colors derive from system semantic colors. No hardcoded hex values.

| Role | Token |
|------|-------|
| Background | `.windowBackground` (system) |
| Section cards | `.controlBackground` with 0.5pt `Color.separator` border |
| Primary text | `Color.primary` |
| Secondary text | `Color.secondary` |
| Accent | `.accentColor` (follows system accent) |
| Danger | `Color.red` (for errors only) |
| Success | `Color.green` (for status indicators) |

---

## 5. First-Run Experience

A lightweight onboarding that appears on first launch. Three steps, no more.

### Window Specs

| Property | Value |
|----------|-------|
| **Size** | 440 x 520 pt |
| **Style** | `.titled`, centered on screen, modal |
| **Background** | System window background |

### Step 1: Welcome + Microphone Permission

```
┌─────────────────────────────────────────────┐
│                                             │
│              ( mic.circle.fill )             │
│              56pt SF Symbol, accent          │
│                                             │
│           Welcome to SafeVoice              │
│                                             │
│     Type with your voice. Fast, private,    │
│     and entirely on your Mac.               │
│                                             │
│     SafeVoice needs microphone access       │
│     to hear you speak. Audio never          │
│     leaves your device.                     │
│                                             │
│        ┌──────────────────────────┐         │
│        │   Allow Microphone  →   │         │
│        └──────────────────────────┘         │
│                                             │
│     Step 1 of 3     ● ○ ○                  │
│                                             │
└─────────────────────────────────────────────┘
```

- Large SF Symbol icon as hero visual
- Single CTA button triggers macOS permission dialog
- After granting, auto-advances to step 2
- Privacy note ("Audio never leaves your device") in `Color.secondary`, 11pt

### Step 2: Set Your Hotkey

```
┌─────────────────────────────────────────────┐
│                                             │
│            ( keyboard.fill )                │
│                                             │
│          Choose Your Hotkey                 │
│                                             │
│     Hold this key combination to speak.     │
│     Release to insert text.                 │
│                                             │
│           ┌────────────────────┐            │
│           │    ⌥  Space        │            │
│           │  (click to change) │            │
│           └────────────────────┘            │
│                                             │
│     Recommended: Option + Space             │
│                                             │
│        ┌──────────────────────────┐         │
│        │       Continue  →       │         │
│        └──────────────────────────┘         │
│                                             │
│     Step 2 of 3     ○ ● ○                  │
│                                             │
└─────────────────────────────────────────────┘
```

- Pre-filled with recommended default (`⌥ Space`)
- Key recorder field - click to change, then press desired combo
- "Recommended" hint below in secondary text

### Step 3: Quick Test

```
┌─────────────────────────────────────────────┐
│                                             │
│            ( waveform.fill )                │
│                                             │
│            Try It Out                       │
│                                             │
│     Hold  ⌥ Space  and say something.      │
│                                             │
│     ┌───────────────────────────────────┐   │
│     │                                   │   │
│     │   "Hello, this is a test."        │   │
│     │                                   │   │
│     └───────────────────────────────────┘   │
│                                             │
│     ✓ SafeVoice is working perfectly.       │
│                                             │
│        ┌──────────────────────────────┐     │
│        │      Start Using  →         │     │
│        └──────────────────────────────┘     │
│                                             │
│     Step 3 of 3     ○ ○ ●                  │
│                                             │
└─────────────────────────────────────────────┘
```

- A text area shows the live transcription result
- Success checkmark (green, `checkmark.circle.fill`) appears after successful transcription
- "Start Using" closes onboarding and the app settles into the menubar
- If test fails: show inline error with suggestion ("Check your microphone in System Settings")

### Onboarding Design Notes

- Page indicator dots at bottom (filled = current)
- Back arrow available on steps 2-3 (top-left)
- Skip link (small, secondary text) in bottom-right for power users
- No animations between steps beyond a simple cross-fade (0.25s)
- Total time to complete: under 30 seconds

---

## 6. Interaction Flow

Complete step-by-step of what happens when a user activates voice input.

### Flow: Hold-to-Talk Mode (Default)

```
                    User holds ⌥ Space
                          │
                          ▼
              ┌───────────────────────┐
              │  1. Menubar icon      │
              │     tints accent +    │
              │     begins pulsing    │
              └───────────┬───────────┘
                          │  (simultaneously)
                          ▼
              ┌───────────────────────┐
              │  2. Floating pill     │
              │     springs in        │
              │     (0.2s animation)  │
              │     Shows: 🟢 中文    │
              │     "Listening..."    │
              └───────────┬───────────┘
                          │  (~50ms)
                          ▼
              ┌───────────────────────┐
              │  3. Subtle start      │
              │     sound plays       │
              │     (soft "pop")      │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  4. User speaks       │
              │     - Waveform bars   │
              │       animate live    │
              │     - Partial text    │
              │       appears in pill │
              │     - Streaming ASR   │
              │       processes audio │
              └───────────┬───────────┘
                          │
                          ▼
                  User releases ⌥ Space
                          │
                          ▼
              ┌───────────────────────┐
              │  5. Stop listening    │
              │     - Waveform freezes│
              │     - Status dot →    │
              │       orange          │
              │     - Menubar icon →  │
              │       processing      │
              │       spinner         │
              │     - Subtle stop     │
              │       sound ("tock")  │
              └───────────┬───────────┘
                          │  (~100-600ms, model inference)
                          ▼
              ┌───────────────────────┐
              │  6. Final transcript  │
              │     appears in pill   │
              │     briefly (0.3s)    │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  7. Text inserted     │
              │     at cursor in      │
              │     active app via    │
              │     CGEvent /         │
              │     Accessibility API │
              └───────────┬───────────┘
                          │  (0.5s delay)
                          ▼
              ┌───────────────────────┐
              │  8. Pill auto-        │
              │     dismisses         │
              │     (scale + fade)    │
              │     Menubar → idle    │
              └───────────────────────┘
```

### Flow: Toggle Mode

Same as above, except:
- Step 1: First press starts listening (no hold)
- Step 5: Second press stops listening (no release)
- Useful for longer dictation passages

### Flow: Language Switch (mid-session)

```
  User is actively listening
         │
         ▼
  User presses ⌃ Space
         │
         ▼
  ┌──────────────────────┐
  │ Language badge in     │
  │ pill cross-fades:     │
  │ 中文 → EN (0.2s)     │
  │                       │
  │ ASR switches language │
  │ context immediately   │
  └──────────────────────┘
         │
         ▼
  User continues speaking
  in English
```

### Flow: Error State

```
  User activates voice input
         │
         ▼
  Microphone unavailable / model not loaded
         │
         ▼
  ┌──────────────────────────────┐
  │ Pill appears with red dot    │
  │ "Microphone unavailable"     │
  │ or "Model loading..."        │
  └──────────────────────────────┘
         │  (2s)
         ▼
  Pill auto-dismisses
  Menubar shows error icon briefly
```

### Sound Design

| Event | Sound | Duration |
|-------|-------|----------|
| Start listening | Soft rising "pop" (like macOS dictation) | ~100ms |
| Stop listening | Soft falling "tock" | ~100ms |
| Error | macOS system error sound (`NSSound.beep()`) | System default |
| Language switch | No sound (visual feedback only) | — |

Sounds are optional (toggle in Settings > General). Volume matches system alert volume.

---

## 7. Accessibility

- All controls support VoiceOver with descriptive labels
- Menubar icon: "SafeVoice - [state]. Click to toggle voice input."
- Floating pill respects `Reduce Motion` preference: disables spring animations, uses simple fade instead
- Floating pill respects `Increase Contrast` preference: adds 1pt solid border
- Hotkey is configurable for users who need alternative key combinations
- All text meets WCAG AA contrast ratios (guaranteed by using system semantic colors)

---

## 8. Dark Mode / Light Mode

No manual theming. All colors are system semantic tokens (`Color.primary`, `Color.secondary`, `.accentColor`, `.windowBackground`, `.ultraThinMaterial`). The app automatically adapts to system appearance.

| Element | Light Mode | Dark Mode |
|---------|------------|-----------|
| Floating pill background | Light frosted glass | Dark frosted glass |
| Menubar icon | Black template | White template |
| Status dot | Same green/orange/red | Same green/orange/red |
| Settings window | Standard light | Standard dark |

---

## 9. Summary of All UI Elements

| Element | When Visible | Size |
|---------|--------------|------|
| Menubar icon | Always (when app running) | 18x18 pt |
| Floating pill | During listening + briefly after | 280-480 x 44 pt |
| Language toast | On language switch (if pill not visible) | 120 x 32 pt |
| Context menu | On right-click menubar icon | System standard |
| Settings window | On demand (⌘,) | 520 x 440 pt |
| Onboarding window | First launch only | 440 x 520 pt |

Total persistent screen real estate: **one menubar icon**. Everything else is transient.
