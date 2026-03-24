# Settings V2: Interactive Modes, Vocabulary, and Model Selection

## Overview

Make the Settings window fully interactive: editable modes with hotkey recording and prompt customization, vocabulary CRUD, and a new Models tab for ASR/LLM selection with local and cloud support.

## Feature 1: Interactive Modes Tab

### Mode List
Each mode displays as a row: **Name — Hotkey `[Record]` `[Edit]` `[Delete]`**

- **Quick mode**: Can change hotkey. Cannot edit prompt (has none) or delete.
- **Built-in modes** (Formal Writing, Translation): Can edit prompt + hotkey. Cannot delete.
- **Custom modes**: Full edit/delete.
- **"+ Add Mode"** button at bottom of the list.

### Edit Modal (NSPanel, non-modal)
Opens when clicking `[Edit]` or `[+ Add Mode]`:

- **Name** text field
- **Style preset** dropdown: Minimal Cleanup, Professional, Casual, Verbatim, Custom
  - Selecting a preset fills the prompt text area with that preset's template
  - User can further edit the text freely (switches preset label to "Custom")
- **Prompt template** text area with `{text}` placeholder hint
  - Multi-line NSTextView, ~4 lines visible
- **Hotkey** recorder field (reuses existing `_HotkeyRecorderDelegate` pattern)
- **Save** / **Cancel** buttons

### Style Presets
```
Minimal:      "Fix only obvious typos and punctuation. Keep the original wording. Do NOT translate. Output only the cleaned text:\n\n{text}"
Professional: "Clean up this dictated text. Fix grammar, punctuation, and make it professional. Do NOT translate. Keep the same language. Output only the cleaned text:\n\n{text}"
Casual:       "Clean up this dictated text lightly. Keep it conversational and natural. Fix obvious errors only. Do NOT translate. Output only the cleaned text:\n\n{text}"
Verbatim:     "Output the text exactly as spoken, only fixing punctuation and capitalization. Do NOT rephrase, summarize, or translate:\n\n{text}"
```

### Translation Mode
The built-in "English Translation" mode becomes configurable:
- A **target language dropdown** (English, Chinese, French, Japanese, Korean, German, Spanish, etc.) appears in the edit modal for this mode
- Prompt updates dynamically: `"Translate the following text to {language}. Output only the translation:\n\n{text}"`
- Mode name updates to match: "French Translation", "Chinese Translation", etc.

### Hotkey Recording
Each mode row has a clickable hotkey field. Clicking it enters recording mode (field shows "Press a key..."), captures the next key combo, saves it. Reuses the existing `_HotkeyRecorderDelegate` from the Advanced tab.

## Feature 2: Interactive Vocabulary Tab

### Hotwords Section
- Scrollable list of current hotwords, each with an `[x]` delete button
- Text field + `[Add]` button below the list
- Adding: type word, click Add (or press Enter), word appears in list
- Deleting: click `[x]`, word removed immediately
- All changes auto-save to `vocabulary.json`

### Snippets Section
- Scrollable list showing `"trigger" → "replacement"`, each with `[x]` delete button
- Two text fields (Trigger, Replacement) + `[Add]` button below
- Same add/delete behavior as hotwords
- Auto-save on every change

### Layout
Both sections stack vertically in the tab. Hotwords first (simpler, fewer items expected), then a divider, then Snippets.

## Feature 3: Models Tab (new)

A new 6th tab in the Settings window.

### ASR Model Section

**Current model** display with dropdown to switch:
- `Qwen3-ASR-0.6B` (default, bundled via mlx-qwen3-asr)
- Future: other mlx-qwen3-asr compatible models as they become available

**Model info card** for selected model:
- Size (e.g., "~1.2 GB")
- Speed estimate (e.g., "~1-2s for 5s audio")
- Privacy: "100% on-device"
- Status: Downloaded / Not Downloaded / Downloading

**Download button** for models not yet downloaded. Shows progress during download.

### Post-Production LLM Section

**Source toggle**: Radio buttons for **Local (Ollama)** vs **Cloud API**

**Local (Ollama) panel:**
- Dropdown auto-populated from `ollama list` output
- Shows model name + parameter size (e.g., "qwen2.5:3b — 1.9 GB")
- `[Refresh]` button to re-scan available models
- If Ollama not running: "Ollama not detected. Install from ollama.com" message
- Info badge: "Local models keep your data private. No internet required."

**Cloud API panel:**
- Provider dropdown: OpenAI, Anthropic, Google
- API key field (password-masked, stored at `~/.config/safevoice/credentials.json` with 0600 permissions)
- Model name field with suggested defaults per provider:
  - OpenAI: `gpt-4o-mini`
  - Anthropic: `claude-haiku-4-5-20251001`
  - Google: `gemini-2.0-flash`
- Info badge: "Cloud models may be faster and more accurate, but your text is sent to the provider's servers."

**Model recommendations panel** (below the source toggle):
A small info section with guidance:
- "For privacy: Use a local Ollama model. Your voice and text never leave your Mac."
- "For accuracy: Cloud models (GPT-4o-mini, Claude Haiku) are generally more accurate for cleanup."
- "For speed: Local models avoid network latency. Cloud models depend on your connection."

### LLM Backend Abstraction

Currently `llm_cleanup.py` talks directly to Ollama's HTTP API. To support cloud APIs, extract an `LLMBackend` interface:

```
LLMBackend (abstract):
  - cleanup(text, prompt) -> str
  - is_available() -> bool
  - warm_up()

OllamaBackend(LLMBackend):
  - existing Ollama HTTP logic

CloudBackend(LLMBackend):
  - provider: "openai" | "anthropic" | "google"
  - Uses urllib.request (no new deps) to call:
    - OpenAI: POST https://api.openai.com/v1/chat/completions
    - Anthropic: POST https://api.anthropic.com/v1/messages
    - Google: POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
```

The active backend is selected based on settings. `llm_cleanup.py` delegates to whichever backend is configured.

## Architecture

### Files Changed/Created
| Action | File | What |
|--------|------|------|
| Modify | `src/settings_window.py` | Interactive Modes tab, Vocabulary tab, new Models tab |
| Modify | `src/modes.py` | Style presets, translation language config |
| Modify | `src/vocabulary.py` | No changes needed (CRUD already exists) |
| Create | `src/llm_backend.py` | LLMBackend abstract class, OllamaBackend, CloudBackend |
| Modify | `src/llm_cleanup.py` | Delegate to LLMBackend instead of direct Ollama calls |
| Modify | `src/settings_manager.py` | New settings keys for model selection, cloud credentials |
| Modify | `src/app.py` | Wire new backend, pass to settings window |

### Settings Keys (new)
```json
{
  "asr_model": "Qwen/Qwen3-ASR-0.6B",
  "llm_source": "local",
  "llm_local_model": "qwen2.5:3b",
  "llm_cloud_provider": "openai",
  "llm_cloud_model": "gpt-4o-mini",
  "translation_language": "English"
}
```

Cloud API keys stored separately in `~/.config/safevoice/credentials.json` (0600 permissions), not in settings.json.

## Testing

- Unit tests for `LLMBackend` implementations (mock HTTP responses)
- Unit tests for style presets (render_prompt with different presets)
- Existing vocabulary and modes tests still pass
- Manual testing: settings UI interaction, model switching, cloud API calls
