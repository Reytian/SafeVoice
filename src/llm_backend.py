"""LLM backend abstraction for SafeVoice.

Supports local (Ollama) and cloud (OpenAI, Anthropic, Google) backends
behind a unified interface. Uses only stdlib urllib — no third-party HTTP libs.
"""

import json
import logging
import re
import urllib.request
import urllib.error
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_LOCAL_MODEL = "qwen2.5:3b"

# Default model per cloud provider.
CLOUD_DEFAULTS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "google": "gemini-2.0-flash",
}


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from LLM output."""
    if "<think>" not in text:
        return text
    idx = text.find("</think>")
    if idx != -1:
        return text[idx + len("</think>"):].strip()
    # No closing tag — strip from <think> onward
    return text[:text.find("<think>")].strip()


class LLMBackend:
    """Abstract base class for LLM backends."""

    @property
    def name(self) -> str:
        raise NotImplementedError

    def chat(self, system_prompt: str, user_message: str) -> str:
        """Send a chat completion request and return the assistant reply.

        Raises on network / API errors — callers should handle exceptions.
        """
        raise NotImplementedError

    def is_available(self) -> bool:
        """Return True if the backend can serve requests right now."""
        raise NotImplementedError

    def warm_up(self) -> None:
        """Optional: pre-load model into memory."""
        pass


class OllamaBackend(LLMBackend):
    """Local Ollama backend."""

    def __init__(
        self,
        model: str = DEFAULT_LOCAL_MODEL,
        base_url: str = OLLAMA_BASE,
    ) -> None:
        self.model = model
        self._base_url = base_url
        self._available: Optional[bool] = None

    @property
    def name(self) -> str:
        return f"Ollama ({self.model})"

    def is_available(self) -> bool:
        """Check if Ollama is running and the model exists.

        Only caches a positive result so that starting Ollama mid-session
        is detected on the next call.
        """
        if self._available is True:
            return True
        try:
            req = urllib.request.Request(f"{self._base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                names = [m["name"] for m in data.get("models", [])]
                self._available = (
                    self.model in names
                    or f"{self.model}:latest" in names
                )
                return self._available
        except Exception:
            self._available = None
            return False

    def warm_up(self) -> None:
        """Pre-load the model into Ollama's memory."""
        if not self.is_available():
            return
        try:
            body = json.dumps({
                "model": self.model,
                "keep_alive": "30m",
                "prompt": "",
            }).encode()
            req = urllib.request.Request(
                f"{self._base_url}/api/generate",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30):
                pass
            logger.info("LLM model warmed up: %s", self.model)
        except Exception as e:
            logger.warning("LLM warm-up failed: %s", e)

    def chat(self, system_prompt: str, user_message: str) -> str:
        """Send a chat request to Ollama and return the cleaned reply."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 512, "top_p": 0.9},
        }).encode()
        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            result = data["message"]["content"].strip()
            return _strip_think_tags(result)

    @staticmethod
    def list_models(base_url: str = OLLAMA_BASE) -> list:
        """Return a list of model names available in the local Ollama instance."""
        try:
            req = urllib.request.Request(f"{base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []


class CloudBackend(LLMBackend):
    """Cloud LLM backend (OpenAI, Anthropic, Google)."""

    def __init__(
        self,
        provider: str,
        model: str = "",
        api_key: str = "",
    ) -> None:
        self.provider = provider
        self.model = model or CLOUD_DEFAULTS.get(provider, "")
        self._api_key = api_key

    @property
    def name(self) -> str:
        label = self.provider.capitalize()
        if self.provider == "openai":
            label = "OpenAI"
        return f"{label} ({self.model})"

    def is_available(self) -> bool:
        """Cloud backends are available if an API key is configured."""
        return bool(self._api_key)

    def _build_request(
        self, system_prompt: str, user_message: str
    ) -> Tuple[str, dict, bytes]:
        """Build (url, headers, body_bytes) for the configured provider."""
        if self.provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0.0,
                "max_tokens": 512,
            }
            return url, headers, json.dumps(payload).encode()

        elif self.provider == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
            }
            payload = {
                "model": self.model,
                "max_tokens": 512,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_message},
                ],
            }
            return url, headers, json.dumps(payload).encode()

        elif self.provider == "google":
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/"
                f"models/{self.model}:generateContent?key={self._api_key}"
            )
            headers = {"Content-Type": "application/json"}
            payload = {
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": [
                    {"role": "user", "parts": [{"text": user_message}]},
                ],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 512,
                },
            }
            return url, headers, json.dumps(payload).encode()

        else:
            raise ValueError(f"Unsupported cloud provider: {self.provider}")

    def chat(self, system_prompt: str, user_message: str) -> str:
        """Send a chat request to the cloud API and return the reply."""
        url, headers, body = self._build_request(system_prompt, user_message)
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        # Extract text from provider-specific response format.
        if self.provider == "openai":
            text = data["choices"][0]["message"]["content"].strip()
        elif self.provider == "anthropic":
            text = data["content"][0]["text"].strip()
        elif self.provider == "google":
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        return _strip_think_tags(text)


def get_backend(
    source: str = "local",
    local_model: str = DEFAULT_LOCAL_MODEL,
    cloud_provider: str = "openai",
    cloud_model: str = "",
    cloud_api_key: str = "",
) -> LLMBackend:
    """Factory: return the appropriate backend based on *source*.

    Args:
        source: ``"local"`` for Ollama, ``"cloud"`` for a cloud provider.
        local_model: Ollama model name (only used when source is ``"local"``).
        cloud_provider: One of ``"openai"``, ``"anthropic"``, ``"google"``.
        cloud_model: Model identifier (defaults per provider if empty).
        cloud_api_key: API key for the cloud provider.
    """
    if source == "cloud":
        return CloudBackend(
            provider=cloud_provider,
            model=cloud_model,
            api_key=cloud_api_key,
        )
    return OllamaBackend(model=local_model)


# ASR Model Catalog
ASR_MODELS = [
    {
        "id": "Qwen/Qwen3-ASR-0.6B",
        "name": "Qwen3 ASR 0.6B",
        "size": "~1.2 GB",
        "speed": "Fast (~1-2s)",
        "accuracy": "Good",
        "description": "Default model. Fast and accurate for most languages.",
        "engine": "mlx-qwen3-asr",
    },
    {
        "id": "mlx-community/whisper-large-v3-turbo",
        "name": "Whisper Large V3 Turbo",
        "size": "~1.6 GB",
        "speed": "Fast (~2-3s)",
        "accuracy": "Excellent",
        "description": "OpenAI Whisper optimized for MLX. Best accuracy, supports 99 languages.",
        "engine": "mlx-whisper",
    },
    {
        "id": "mlx-community/whisper-small",
        "name": "Whisper Small",
        "size": "~500 MB",
        "speed": "Very fast (<1s)",
        "accuracy": "Good",
        "description": "Compact Whisper model. Fast but less accurate than larger variants.",
        "engine": "mlx-whisper",
    },
    {
        "id": "mlx-community/whisper-medium",
        "name": "Whisper Medium",
        "size": "~1.5 GB",
        "speed": "Moderate (~2-4s)",
        "accuracy": "Very good",
        "description": "Balanced Whisper model. Good accuracy for most use cases.",
        "engine": "mlx-whisper",
    },
]


def is_asr_model_downloaded(model_id: str) -> bool:
    """Check if an ASR model is cached locally."""
    from pathlib import Path
    safe_id = model_id.replace("/", "--")
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{safe_id}" / "snapshots"
    if cache_dir.is_dir():
        snapshots = list(cache_dir.iterdir())
        if snapshots:
            # Check if snapshot has actual model files (not just metadata)
            for snap in snapshots:
                safetensors = list(snap.glob("*.safetensors"))
                if safetensors:
                    return True
    return False
