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


# --- MLX backend generation API contract ----------------------------------
# Pins MLXBackend.chat() to the installed mlx_lm generate() API. In mlx_lm
# 0.31.x, generate() forwards its **kwargs down to generate_step(), which has
# NO `temp` parameter, so the old `temp=0.0` call raised at generation time:
#   TypeError: generate_step() got an unexpected keyword argument 'temp'
# Temperature is now supplied via a sampler (sample_utils.make_sampler).

class _FakeMLXTokenizer:
    """Minimal tokenizer stub: returns a fixed prompt, ignores template args."""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kwargs):
        return "PROMPT"


def test_mlx_chat_uses_sampler_not_temp(monkeypatch):
    """MLXBackend.chat() must call generate() with kwargs the real
    generate_step() accepts: a `sampler` callable, never the removed `temp`."""
    mlx_lm = pytest.importorskip("mlx_lm")
    import inspect
    from mlx_lm.generate import generate_step
    from src.llm_backend import MLXBackend

    captured = {}

    def fake_generate(model, tokenizer, prompt, **kwargs):
        captured["kwargs"] = kwargs
        return "  cleaned text  "

    # chat() does `from mlx_lm import generate` at call time, so replacing the
    # attribute on the mlx_lm package intercepts it.
    monkeypatch.setattr(mlx_lm, "generate", fake_generate)

    backend = MLXBackend(model="test/model")
    # Bypass the real multi-GB model load; chat() only needs these set.
    backend._model = object()
    backend._tokenizer = _FakeMLXTokenizer()

    result = backend.chat("system prompt", "user message")

    assert result == "cleaned text"  # chat() strips the reply
    kwargs = captured["kwargs"]
    # Regression guard: the removed `temp` kwarg must never come back.
    assert "temp" not in kwargs
    # Temperature now travels via a sampler callable.
    assert callable(kwargs.get("sampler"))
    # Strongest pin: the forwarded kwargs must bind against the REAL
    # generate_step signature -- exactly where generate() forwards them and
    # where `temp` blew up. Catches any future mlx_lm sampling-API drift.
    inspect.signature(generate_step).bind_partial(**kwargs)


# --- Rule-R2 guard: model must echo a dictated question, not answer it -----

def test_cleanup_rejects_answered_question_cjk():
    """The reported WTO bug: a weak model turned a dictated Chinese question
    into a fabricated statement-shaped answer. Reject and fall back."""
    from src.llm_cleanup import LLMCleanup
    llm = LLMCleanup(backend=_FakeBackend(
        reply="WTO中对于香烟、酒精等成瘾性产品没有特定的管制要求，各成员国可以自行决定其管控措施。"))
    raw = "WTO中对于香烟、酒精等管制类产品，有什么样的要求"
    out = llm.cleanup(raw)
    # Rejected: output is the faithful (rule-stripped) question, not the
    # invented answer.
    assert "成瘾性" not in out
    assert "什么" in out


def test_cleanup_keeps_faithfully_echoed_question():
    """A cleanup that keeps the question a question must pass through."""
    from src.llm_cleanup import LLMCleanup
    llm = LLMCleanup(backend=_FakeBackend(
        reply="WTO对于香烟、酒精等管制类产品有什么样的要求？"))
    raw = "嗯WTO对于香烟酒精等管制类产品有什么样的要求"
    out = llm.cleanup(raw)
    assert out.endswith("？")
    assert "成瘾" not in out


def test_cleanup_rejects_answered_question_english():
    from src.llm_cleanup import LLMCleanup
    llm = LLMCleanup(backend=_FakeBackend(
        reply="The capital of France is Paris."))
    raw = "what is the capital of France"
    out = llm.cleanup(raw)
    assert "Paris" not in out
    assert "capital" in out.lower()


def test_cleanup_statement_with_question_word_not_flagged():
    """'什么' used as 'anything' (not interrogative) must not trip the guard
    when the cleanup faithfully keeps it."""
    from src.llm_cleanup import LLMCleanup
    llm = LLMCleanup(backend=_FakeBackend(reply="随便什么都行。"))
    raw = "嗯随便什么都行"
    out = llm.cleanup(raw)
    assert out == "随便什么都行。"


def test_is_question_helpers():
    from src.llm_cleanup import _is_question, _answered_a_question
    assert _is_question("有什么要求")
    assert _is_question("what is this?")
    assert _is_question("能不能帮我")
    assert not _is_question("这是一个陈述句。")
    assert not _is_question("I know what you mean.")
    assert _answered_a_question("有什么要求", "没有特定要求。")
    assert not _answered_a_question("有什么要求", "到底有什么要求？")
