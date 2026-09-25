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
    """什么 as "anything" (随便什么都行) or in a filler (那个什么) is not a
    question, so the guard must accept a cleanup that drops it."""
    from src.llm_cleanup import LLMCleanup, _is_question
    assert not _is_question("随便什么都行")
    llm = LLMCleanup(backend=_FakeBackend(reply="我们下周再讨论这个方案吧。"))
    out = llm.cleanup("我们那个什么下周再讨论这个方案吧")
    assert out == "我们下周再讨论这个方案吧。"


def test_is_question_helpers():
    from src.llm_cleanup import _is_question, _answered_a_question
    assert _is_question("有什么要求")
    assert _is_question("what is this?")
    assert _is_question("能不能帮我")
    assert not _is_question("这是一个陈述句。")
    assert not _is_question("I know what you mean.")
    assert _answered_a_question("有什么要求", "没有特定要求。")
    assert not _answered_a_question("有什么要求", "到底有什么要求？")


@pytest.mark.parametrize("text", [
    "有没有什么问题", "你去不去", "明天星期几", "你今天去吗", "谁来开会",
    "哪个方案更好", "为什么都不说话", "WTO對管制類產品有什麼要求", "你幾時返嚟",
    "what's the plan", "OK so how do we fix it", "does the store open on Sunday",
    "Can I leave early", "Where's the file", "“Is it done?”", "ما هي عاصمة فرنسا؟",
])
def test_is_question_recognizes(text):
    from src.llm_cleanup import _is_question
    assert _is_question(text)


@pytest.mark.parametrize("text", [
    "随便什么都行", "谁都知道这件事", "哪怕下雨也要去", "没什么问题", "他在吃饭呢",
    "然后呢我们走了", "不管怎么样我们都要完成", "不不不，我不是这个意思",
    "不是不是，我说的是另一个", "我想订三张票，不对不对，是四张",
    "这样挺好的吗怎么说呢不用改",
    "do your homework", "Have a nice day", "should include the chart",
    "When you get a chance, send me the file", "我买了几本书",
])
def test_is_question_ignores_statements(text):
    from src.llm_cleanup import _is_question
    assert not _is_question(text)


# Correct cleanups the guard must accept. The first version of the guard
# rejected all but the last one, mistaking a dropped or changed question cue
# for an answer.
_FAITHFUL_CLEANUPS = [
    # Self-corrections that retract a question (rule E4)
    ("我们是不是周五开会，啊不对，我们周六开会", "我们周六开会。"),
    ("is it Tuesday, no wait, it's Wednesday", "It's Wednesday."),
    ("can you send it Monday, sorry I mean, send it Tuesday", "Send it Tuesday."),
    # Fillers the model is told to remove (rule E2)
    ("然后呢我们去超市买了一些水果和蔬菜", "我们去超市买了一些水果和蔬菜。"),
    ("这个方案，怎么说呢，还不太成熟需要再改改", "这个方案还不太成熟，需要再改改。"),
    ("这个项目你知道吗其实很难做完", "这个项目其实很难做完。"),
    ("我呢觉得这个方案还不错可以试试", "我觉得这个方案还不错，可以试试。"),
    # 吗 misheard for 嘛 (rule E5)
    ("这样就挺好的吗不用再改了", "这样就挺好的嘛，不用再改了。"),
    ("这样就挺好的吗，不用再改了", "这样就挺好的嘛，不用再改了。"),
    # Commands that only look like "do you ..." / "may I ..."
    ("do your homework first, sorry I mean do the dishes first", "Do the dishes first."),
    ("may is busy for me, I mean June is busy", "June is busy for me."),
    # A faithful echo wrapped in curly quotes
    ("what time is the meeting tomorrow", "“What time is the meeting tomorrow?”"),
    # A question followed by the speaker's own statement, not an answer
    ("what time is the meeting I need to know by tonight",
     "What time is the meeting? I need to know by tonight."),
]


@pytest.mark.parametrize("raw,cleaned", _FAITHFUL_CLEANUPS)
def test_cleanup_keeps_faithful_cleanup(raw, cleaned):
    from src.llm_cleanup import LLMCleanup
    llm = LLMCleanup(backend=_FakeBackend(reply=cleaned))
    assert llm.cleanup(raw) == cleaned


# Answers the guard must reject. The first version of the guard let every
# one of these through.
_ANSWERS = [
    # The answer reuses the question word (有什么 -> 没有什么)
    ("WTO对管制类产品有什么要求", "WTO对管制类产品没有什么统一要求，各成员国自行决定。"),
    # An answer followed by a chatbot follow-up question
    ("WTO对管制类产品有什么要求",
     "WTO没有特定的管制要求，各成员国自行决定。您还有其他问题吗？"),
    # The question echoed, then answered
    ("what is the capital of France", "What is the capital of France? Paris."),
    ("what is the capital of France",
     "What is the capital of France? The capital of France is Paris."),
    # Question words and forms the first word list missed
    ("明天几点开会", "明天上午十点开会。"),
    ("谁负责这个项目的预算", "张经理负责这个项目的预算。"),
    ("你现在在哪儿呀", "我现在在公司。"),
    ("你明天来不来", "我明天来。"),
    # Traditional Chinese and Cantonese
    ("WTO對管制類產品有什麼要求", "WTO對管制類產品沒有特定要求，各成員國自行決定。"),
    ("你幾時返嚟", "我聽日返嚟。"),
    # English questions that don't open with a bare wh-word
    ("what's the capital of France", "The capital of France is Paris."),
    ("does the store open on Sunday", "Yes, the store opens at 10 AM on Sunday."),
    ("I have a question. what is the deadline for the report",
     "I have a question. The deadline for the report is Friday."),
    # Arabic question mark
    ("ما هي عاصمة فرنسا؟", "عاصمة فرنسا هي باريس."),
]


@pytest.mark.parametrize("raw,answer", _ANSWERS)
def test_cleanup_rejects_answer(raw, answer, caplog):
    from src.llm_cleanup import LLMCleanup
    llm = LLMCleanup(backend=_FakeBackend(reply=answer))
    out = llm.cleanup(raw)
    assert out == raw  # fell back to the speaker's own words
    assert "answered a dictated question" in caplog.text


# --- Rule-R2 guard on the custom-prompt path ------------------------------

def _cleanup_like_app(llm, mode, raw):
    """Call cleanup() the way app.py does for a mode with a prompt."""
    return llm.cleanup(
        raw, custom_prompt=mode.render_prompt(raw),
        allow_script_change=mode.allows_translation(),
        echo_questions=mode.echoes_questions(),
    )


def test_custom_path_rejects_answered_question():
    """After the first-run wizard, Quick mode holds a style preset and runs
    through the custom-prompt path. The WTO answer must be rejected there."""
    from src.llm_cleanup import LLMCleanup
    from src.modes import Mode, STYLE_PRESETS
    mode = Mode(name="Quick", prompt_template=STYLE_PRESETS["professional"])
    llm = LLMCleanup(backend=_FakeBackend(
        reply="WTO中对于香烟、酒精等成瘾性产品没有特定的管制要求，各成员国可以自行决定其管控措施。"))
    raw = "WTO中对于香烟、酒精等管制类产品，有什么样的要求"
    assert _cleanup_like_app(llm, mode, raw) == raw


def test_custom_path_formal_writing_rejects_answer():
    from src.llm_cleanup import LLMCleanup
    from src.modes import DEFAULT_MODES
    mode = next(m for m in DEFAULT_MODES if m.name == "Formal Writing")
    llm = LLMCleanup(backend=_FakeBackend(reply="The capital of France is Paris."))
    raw = "what is the capital of France"
    assert _cleanup_like_app(llm, mode, raw) == raw


def test_custom_path_keeps_answer_when_mode_asks_for_one():
    """A user's own "answer this" mode may reply to a dictated question."""
    from src.llm_cleanup import LLMCleanup
    from src.modes import Mode
    mode = Mode(name="Ask", prompt_template="Answer this question briefly: {text}")
    llm = LLMCleanup(backend=_FakeBackend(reply="Paris."))
    assert _cleanup_like_app(llm, mode, "what is the capital of France") == "Paris."


def test_custom_path_translation_skips_question_guard():
    """A translation replaces every word, so the guard's word comparison
    would reject it. Translation modes skip the guard."""
    from src.llm_cleanup import LLMCleanup
    reply = "I have a question. What time is the meeting tomorrow?"
    llm = LLMCleanup(backend=_FakeBackend(reply=reply))
    raw = "我有个问题。明天几点开会"
    out = llm.cleanup(raw, custom_prompt=f"Translate to English: {raw}",
                      allow_script_change=True, echo_questions=True)
    assert out == reply


# --- Settings model dropdown: labels and re-selection ---------------------

def test_local_model_label_round_trip():
    from src.llm_backend import local_model_label, local_model_name
    assert local_model_label("qwen3:4b") == "qwen3:4b  (not recommended)"
    for name in ("qwen2.5:3b", "qwen2.5:7b-instruct-q3_K_M", "qwen3:4b"):
        assert local_model_name(local_model_label(name)) == name


def test_find_local_model_matches_exact_name_not_prefix():
    """Ollama lists the newest model first. Pulling the q3 build must not
    move the selection off "qwen2.5:7b", whose name is a prefix of it."""
    from src.llm_backend import find_local_model, local_model_label
    labels = [local_model_label(m) for m in
              ("qwen2.5:7b-instruct-q3_K_M", "qwen2.5:7b", "qwen3:4b")]
    assert find_local_model(labels, "qwen2.5:7b") == 1
    assert find_local_model(labels, "qwen2.5:7b-instruct-q3_K_M") == 0
    assert find_local_model(labels, "qwen3:4b") == 2
    assert find_local_model(labels, "gemma3:4b") is None
