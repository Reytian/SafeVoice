"""Tests for LLM backend abstraction."""
import json
import pytest
from src.llm_backend import OllamaBackend, CloudBackend, get_backend


def test_ollama_backend_builds_request():
    backend = OllamaBackend(model="qwen2.5:3b")
    assert backend.model == "qwen2.5:3b"
    assert backend.name == "Ollama (qwen2.5:3b)"


def test_cloud_backend_builds_request():
    backend = CloudBackend(provider="openai", model="gpt-4o-mini", api_key="sk-test")
    assert backend.provider == "openai"
    assert backend.model == "gpt-4o-mini"
    assert backend.name == "OpenAI (gpt-4o-mini)"


def test_cloud_backend_openai_headers():
    backend = CloudBackend(provider="openai", model="gpt-4o-mini", api_key="sk-test")
    url, headers, body = backend._build_request("Hello", "Fix this")
    assert "api.openai.com" in url
    assert headers["Authorization"] == "Bearer sk-test"
    parsed = json.loads(body)
    assert parsed["model"] == "gpt-4o-mini"


def test_cloud_backend_anthropic_headers():
    backend = CloudBackend(provider="anthropic", model="claude-haiku-4-5-20251001", api_key="sk-ant-test")
    url, headers, body = backend._build_request("Hello", "Fix this")
    assert "api.anthropic.com" in url
    assert headers["x-api-key"] == "sk-ant-test"


def test_cloud_backend_google_url():
    backend = CloudBackend(provider="google", model="gemini-2.0-flash", api_key="AIza-test")
    url, headers, body = backend._build_request("Hello", "Fix this")
    assert "generativelanguage.googleapis.com" in url
    assert "AIza-test" in url


def test_get_backend_local():
    backend = get_backend(source="local", local_model="qwen2.5:3b")
    assert isinstance(backend, OllamaBackend)


def test_get_backend_cloud():
    backend = get_backend(
        source="cloud", cloud_provider="openai",
        cloud_model="gpt-4o-mini", cloud_api_key="sk-test"
    )
    assert isinstance(backend, CloudBackend)


def test_cloud_backend_zhipu_url():
    backend = CloudBackend(provider="zhipu", model="glm-4-flash", api_key="test-key")
    url, headers, body = backend._build_request("Hello", "Fix this")
    assert "bigmodel.cn" in url
    assert headers["Authorization"] == "Bearer test-key"


def test_cloud_backend_moonshot_url():
    backend = CloudBackend(provider="moonshot", model="moonshot-v1-8k", api_key="test-key")
    url, headers, body = backend._build_request("Hello", "Fix this")
    assert "moonshot.cn" in url
    assert headers["Authorization"] == "Bearer test-key"


def test_cloud_backend_dashscope_url():
    backend = CloudBackend(provider="dashscope", model="qwen-turbo", api_key="test-key")
    url, headers, body = backend._build_request("Hello", "Fix this")
    assert "dashscope.aliyuncs.com" in url
    assert headers["Authorization"] == "Bearer test-key"


def test_cloud_backend_deepseek_url():
    backend = CloudBackend(provider="deepseek", model="deepseek-chat", api_key="test-key")
    url, headers, body = backend._build_request("Hello", "Fix this")
    assert "deepseek.com" in url
    assert headers["Authorization"] == "Bearer test-key"


# --- Truncation detection (output cut at token cap must not be pasted) ----

from src.llm_backend import LLMTruncatedError, LLMBackend


def test_openai_truncation_raises():
    backend = CloudBackend(provider="openai", model="gpt-4o-mini", api_key="k")
    data = {"choices": [{"message": {"content": "cut off mid"},
                         "finish_reason": "length"}]}
    with pytest.raises(LLMTruncatedError):
        backend._extract_text(data)


def test_anthropic_truncation_raises():
    backend = CloudBackend(provider="anthropic", model="m", api_key="k")
    data = {"content": [{"text": "cut"}], "stop_reason": "max_tokens"}
    with pytest.raises(LLMTruncatedError):
        backend._extract_text(data)


def test_google_truncation_raises():
    backend = CloudBackend(provider="google", model="m", api_key="k")
    data = {"candidates": [{"finishReason": "MAX_TOKENS",
                            "content": {"parts": [{"text": "cut"}]}}]}
    with pytest.raises(LLMTruncatedError):
        backend._extract_text(data)


def test_normal_completion_passes():
    backend = CloudBackend(provider="openai", model="gpt-4o-mini", api_key="k")
    data = {"choices": [{"message": {"content": "all good"},
                         "finish_reason": "stop"}]}
    assert backend._extract_text(data) == "all good"


# --- LLMCleanup guard behavior with a fake backend ------------------------

class _FakeBackend(LLMBackend):
    def __init__(self, reply=None, exc=None):
        self._reply = reply
        self._exc = exc

    @property
    def name(self):
        return "Fake"

    def is_available(self):
        return True

    def chat(self, system_prompt, user_message):
        if self._exc is not None:
            raise self._exc
        return self._reply


def test_cleanup_truncation_falls_back_to_rule_strip():
    from src.llm_cleanup import LLMCleanup
    llm = LLMCleanup(backend=_FakeBackend(exc=LLMTruncatedError("cap")))
    raw = "um so we should meet on Tuesday to discuss the quarterly report"
    out = llm.cleanup(raw)
    assert "Tuesday" in out          # transcript preserved
    assert not out.startswith("um")  # rule strip still applied


def test_custom_path_rejects_unrequested_translation():
    from src.llm_cleanup import LLMCleanup
    # Formal-writing style mode, but the model translated the Chinese input.
    llm = LLMCleanup(backend=_FakeBackend(reply="We use the GitHub API for this feature."))
    raw = "我们用GitHub的API来做这个功能，明天上线"
    out = llm.cleanup(raw, custom_prompt=f"Make this formal: {raw}")
    assert "我们" in out  # rejected; original script preserved


def test_custom_path_allows_translation_when_requested():
    from src.llm_cleanup import LLMCleanup
    llm = LLMCleanup(backend=_FakeBackend(reply="We use the GitHub API for this feature tomorrow."))
    raw = "我们用GitHub的API来做这个功能，明天上线"
    out = llm.cleanup(raw, custom_prompt=f"Translate to English: {raw}",
                      allow_script_change=True)
    assert out.startswith("We use")


def test_custom_path_truncation_falls_back():
    from src.llm_cleanup import LLMCleanup
    llm = LLMCleanup(backend=_FakeBackend(exc=LLMTruncatedError("cap")))
    raw = "please make this sentence sound a little more professional thanks"
    out = llm.cleanup(raw, custom_prompt=f"Formal: {raw}")
    assert "professional" in out


# --- Ollama keep_alive: bound the resident model lifetime -----------------

def test_ollama_chat_body_includes_keep_alive():
    from src.llm_backend import OllamaBackend, OLLAMA_KEEP_ALIVE
    backend = OllamaBackend(model="qwen2.5:3b")
    body = backend._build_chat_body("system", "clean this up")
    assert body["keep_alive"] == OLLAMA_KEEP_ALIVE


def test_ollama_warmup_body_includes_keep_alive():
    from src.llm_backend import OllamaBackend, OLLAMA_KEEP_ALIVE
    backend = OllamaBackend(model="qwen2.5:3b")
    body = backend._build_warmup_body()
    assert body["keep_alive"] == OLLAMA_KEEP_ALIVE


def test_ollama_keep_alive_is_short():
    # The cleanup model should linger only briefly after use, not 30 min.
    from src.llm_backend import OLLAMA_KEEP_ALIVE
    assert OLLAMA_KEEP_ALIVE == "5m"


# --- Backend unload: release in-process model memory ----------------------

def test_base_backend_unload_is_noop():
    # Ollama/Cloud hold nothing in SafeVoice's process; unload must exist
    # and be a safe no-op so callers can invoke it uniformly.
    from src.llm_backend import OllamaBackend, CloudBackend
    OllamaBackend(model="qwen2.5:3b").unload()
    CloudBackend(provider="openai", model="gpt-4o-mini", api_key="k").unload()


def test_mlx_backend_unload_releases_model_references():
    from src.llm_backend import MLXBackend
    backend = MLXBackend()
    backend._model = object()      # pretend a ~2.3 GB model is loaded
    backend._tokenizer = object()

    backend.unload()

    assert backend._model is None
    assert backend._tokenizer is None
