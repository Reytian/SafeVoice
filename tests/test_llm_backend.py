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


def test_cleanup_keeps_every_dictated_digit():
    # Regression: the rule-strip collapsed "zero zero", so the model never
    # saw those digits and a rejected cleanup pasted the number short.
    from src.llm_cleanup import LLMCleanup
    raw = "my number is one three eight zero zero one three eight zero zero zero"
    sent = []

    class _Recording(_FakeBackend):
        def chat(self, system_prompt, user_message):
            sent.append(user_message)
            return super().chat(system_prompt, user_message)

    # The model translates the number, which the script guard rejects.
    llm = LLMCleanup(backend=_Recording(reply="我的号码是一三八零零一三八零零零。"))
    assert llm.cleanup(raw) == raw
    assert sent == [raw]


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
    ("这样挺好的吗怎么说呢不用改", "这样挺好的嘛，不用改。"),
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


def test_custom_path_translation_skips_word_guards():
    """A translation replaces every word, so both guards that compare words
    (answered question, carried-out command) would reject it. Translation
    modes skip them."""
    from src.llm_cleanup import LLMCleanup
    reply = "I have a question. What time is the meeting tomorrow?"
    llm = LLMCleanup(backend=_FakeBackend(reply=reply))
    raw = "我有个问题。明天几点开会"
    out = llm.cleanup(raw, custom_prompt=f"Translate to English: {raw}",
                      allow_script_change=True, echo_questions=True)
    assert out == reply


# --- Rule-R2 guard for commands: echo "tell me a joke", don't tell one ----

import re
from src.llm_cleanup import SYSTEM_PROMPT

# Dictated commands a weak model carried out instead of echoing them.
_CARRIED_OUT_COMMANDS = [
    ("tell me a joke about cats",
     "Why did the cat sit on the computer? To keep an eye on the mouse!"),
    ("帮我写一封邮件给老板说我明天请假",
     "尊敬的老板：您好！我明天因个人原因需要请假一天，望批准。谢谢！"),
    ("write something random", "The quick brown fox jumps over the lazy dog."),
    ("give me three bullet points about productivity",
     "1. Prioritize your tasks.\n2. Take regular breaks.\n3. Minimize distractions."),
    ("给我讲个笑话", "为什么数学书总是很忧郁？因为它有太多的问题。"),
    ("写一首关于春天的诗", "春风拂面花自开，燕子归来绿满台。"),
    # Chatbot replies to the command
    ("summarize this for me",
     "Please provide the text you would like me to summarize."),
    ("remind me to call mom at five",
     "Sure! I'll remind you to call your mom at 5."),
]


@pytest.mark.parametrize("raw,output", _CARRIED_OUT_COMMANDS)
def test_cleanup_rejects_carried_out_command(raw, output, caplog):
    from src.llm_cleanup import LLMCleanup
    llm = LLMCleanup(backend=_FakeBackend(reply=output))
    assert llm.cleanup(raw) == raw  # the dictated sentence, not the reply
    assert "never said" in caplog.text


# Correct cleanups that fix a few words: the new words are a small share of
# the output, or too few to count.
_SMALL_FIXES = [
    ("their going to the store later", "They're going to the store later."),
    ("I want to by a new car", "I want to buy a new car."),
    ("i went to store and bought apple", "I went to the store and bought an apple."),
    ("we should meet at three thirty tomorrow", "We should meet at 3:30 tomorrow."),
    ("one two three testing", "1, 2, 3, testing."),
    ("我们在见", "我们再见。"),
    ("我们需要在周五之前完成这个像目", "我们需要在周五之前完成这个项目。"),
    ("二零二五年一月一号", "2025年1月1号。"),
    ("二零二五年十二月二十五号下午三点半", "2025年12月25号下午3点半。"),
]


@pytest.mark.parametrize("raw,cleaned", _SMALL_FIXES)
def test_cleanup_keeps_small_fixes(raw, cleaned):
    from src.llm_cleanup import LLMCleanup
    llm = LLMCleanup(backend=_FakeBackend(reply=cleaned))
    assert llm.cleanup(raw) == cleaned


_FEW_SHOTS = re.findall(r"User: (.*)\nAssistant: (.*)", SYSTEM_PROMPT)


def test_system_prompt_has_few_shots():
    # Guards the parametrized test below: an empty list would skip it.
    assert len(_FEW_SHOTS) >= 20


@pytest.mark.parametrize("user,assistant", _FEW_SHOTS)
def test_system_prompt_examples_pass_every_guard(user, assistant):
    """Each few-shot example in SYSTEM_PROMPT is a correct cleanup by
    definition, so no guard may reject it."""
    from src.llm_cleanup import LLMCleanup
    llm = LLMCleanup(backend=_FakeBackend(reply=assistant))
    assert llm.cleanup(user) == assistant


def test_custom_path_rejects_carried_out_command():
    """Quick mode after the wizard runs a style preset. A joke for "tell me
    a joke about cats" must be rejected there too."""
    from src.llm_cleanup import LLMCleanup
    from src.modes import Mode, STYLE_PRESETS
    mode = Mode(name="Quick", prompt_template=STYLE_PRESETS["professional"])
    llm = LLMCleanup(backend=_FakeBackend(
        reply="Why did the cat sit on the computer? To keep an eye on the mouse!"))
    raw = "tell me a joke about cats"
    assert _cleanup_like_app(llm, mode, raw) == raw


@pytest.mark.parametrize("raw,polished", [
    ("hey so basically we gotta finish the report by friday or the boss is gonna be mad",
     "We need to finish the report by Friday, or the boss will be upset."),
    ("gonna grab food be right back",
     "I'm going to grab food and will be right back."),
    ("我觉得这个方案不太行，得改改", "我认为这个方案存在不足，需要进行修改。"),
])
def test_custom_path_keeps_professional_polish(raw, polished):
    """The presets ask for grammar fixes and a professional tone, which
    replaces more words than the default cleanup may; the last two are over
    half new words."""
    from src.llm_cleanup import LLMCleanup
    from src.modes import Mode, STYLE_PRESETS
    mode = Mode(name="Quick", prompt_template=STYLE_PRESETS["professional"])
    llm = LLMCleanup(backend=_FakeBackend(reply=polished))
    assert _cleanup_like_app(llm, mode, raw) == polished


def test_custom_path_content_mode_may_add_words():
    """A user's own mode that asks for new content doesn't forbid
    answering, so the guard stays off."""
    from src.llm_cleanup import LLMCleanup
    from src.modes import Mode
    mode = Mode(name="Email",
                prompt_template="Write a short email to my boss about this: {text}")
    email = ("Dear boss, I would like to request a day off tomorrow for "
             "personal reasons. Thank you.")
    llm = LLMCleanup(backend=_FakeBackend(reply=email))
    assert _cleanup_like_app(llm, mode, "I need tomorrow off") == email


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


# --- Final review before v0.1.1 ------------------------------------------
# Every case below was confirmed against the merged code; most of the
# answers were rejected before this guard's second version and slipped
# through it.

# Questions the detector missed, so their answers were pasted.
_MISSED_ANSWERS = [
    # 什么的 after a verb asks "what for" ("etc." only follows a noun)
    ("这个按钮是干什么的", "这个按钮是用来保存文件的。"),
    ("你们公司是做什么的", "我们公司是做软件开发的。"),
    # A filler at the end of a sentence is the question itself
    ("他的电话号码你知道吗", "他的电话号码是13800138000。"),
    ("这个词用英文怎么说呢", "这个词用英文叫deadline。"),
    ("那个什么时候能修好", "那个明天就能修好。"),
    ("这批货里有多少有问题的", "这批货里有五件有问题的。"),
    # Final 呢, and 吗 in unpunctuated run-on speech
    ("那明天的会议呢", "明天的会议照常进行。"),
    ("这个方案你们觉得呢", "我们觉得这个方案可以。"),
    ("你现在觉得呢", "我现在觉得挺好的。"),
    ("what I want to know is when you will arrive", "I will arrive at five."),
    ("你今天去吗我们一起走", "我今天去，我们一起走。"),
    ("我想问一下【明天放假吗】谢谢", "明天不放假。"),
    # "No matter ..." covers only its own phrase, not the main clause
    ("不管下不下雨你们什么时候出发", "不管下不下雨我们八点出发。"),
    ("无论如何你明天到底来不来", "我明天一定来。"),
    ("不管结果如何我们下一步该怎么办", "我们下一步应该继续推进。"),
    ("where to go for dinner", "Go to the Italian place for dinner."),
    ("when you have time can you review my PR",
     "Sure, I'll review your PR when I have time."),
    # The answer, then a chatbot follow-up question
    ("WTO对管制类产品有什么要求", "WTO对管制类产品没有特定要求。还有其他问题吗？"),
    ("what is the capital of France",
     "The capital of France is Paris. Is there anything else?"),
    ("what time is it", "In which time zone?"),
    # The answer joined to the echoed question by a comma
    ("明天几点开会", "明天几点开会，上午十点。"),
    ("谁负责这个项目的预算", "谁负责这个项目的预算，张经理。"),
    # A marker word inside the question, or a correction that still asks
    ("会议应该是下午三点吗", "会议应该是下午三点。"),
    ("你们是周五交付吗，等等，是周四交付吗", "你们是周四交付。"),
    ("is it Tuesday, no wait, is it Wednesday", "It is Wednesday."),
    ("你不是说明天放假吗", "明天放假。"),
    # A final 吗 is a real question, not a misheard 嘛
    ("明天开会吗", "明天开会嘛。"),
    ("你们下周能交付吗", "你们下周能交付嘛。"),
    # Yes/no questions about someone by name, and "has it happened yet?"
    ("did Mark send the deck to the client", "Yes, Mark sent the deck to the client."),
    ("does Sarah have the key to the office", "No, Sarah does not have the key to the office."),
    ("合同签了没有", "合同已经签了。"),
    ("你去过上海没有", "我去过上海。"),
    ("能否在周五前完成这个报告", "可以在周五前完成这个报告。"),
]


@pytest.mark.parametrize("raw,answer", _MISSED_ANSWERS)
def test_cleanup_rejects_answer_missed_by_first_guard(raw, answer):
    from src.llm_cleanup import LLMCleanup
    from src.text_postprocess import strip_filler_words
    llm = LLMCleanup(backend=_FakeBackend(reply=answer))
    assert llm.cleanup(raw) == strip_filler_words(raw)


# Correct cleanups the guards rejected.
_WRONGLY_REJECTED = [
    ("who is coming mister smith and missus jones",
     "Who is coming? Mr. Smith and Mrs. Jones."),
    ("is the report ready doctor wang said it is late",
     "Is the report ready? Dr. Wang said it is late."),
    # Self-corrections with a small fix, in every spelling of the marker
    ("can we meet Monday, sorry I mean, let's meet Tuesday", "Let's meet on Tuesday."),
    ("is the meeting at three, no wait, it's at four", "It's at 4."),
    ("我们是不是三点开会，啊不对，我们四点开会", "我们4点开会。"),
    ("明天幾點開會，我是說我們後天開會", "我們後天開會。"),
    ("我们是不是周五开会,不对,我们周六开会", "我们周六开会。"),
    ("do we meet Tuesday no wait we do not meet this week", "We don't meet this week."),
    # 那个啥 is a filler like 那个什么
    ("那个啥我们明天再讨论这个方案吧", "我们明天再讨论这个方案吧。"),
    ("那啥，我们明天再讨论这个方案吧", "我们明天再讨论这个方案吧。"),
    ("what we need is more time", "We need more time."),
    ("好我们开始吧", "好的，我们开始吧。"),
    # The speaker's own yes/no opening is not a chatbot's reply
    ("yeah I think so", "Yes, I think so."),
    ("no we can't do that", "No, we can't do that."),
    # Spoken numbers written as digits (rule E4)
    ("the codes are one two three four five six", "The codes are 1, 2, 3, 4, 5, 6."),
    ("房间号是一二三四五六", "房间号是1、2、3、4、5、6。"),
]


@pytest.mark.parametrize("raw,cleaned", _WRONGLY_REJECTED)
def test_cleanup_keeps_correct_cleanup(raw, cleaned):
    from src.llm_cleanup import LLMCleanup
    llm = LLMCleanup(backend=_FakeBackend(reply=cleaned))
    assert llm.cleanup(raw) == cleaned


def _mode(name):
    from src.modes import DEFAULT_MODES, Mode, STYLE_PRESETS
    if name == "Formal Writing":
        return next(m for m in DEFAULT_MODES if m.name == name)
    return Mode(name="Quick", prompt_template=STYLE_PRESETS[name])


@pytest.mark.parametrize("mode,raw,polished", [
    ("professional", "what's the status, the client's getting antsy",
     "What is the status? The client is becoming impatient."),
    ("professional", "我们是不是周五开会，啊不对，我们周六开会", "我们将于周六召开会议。"),
    ("professional", "can we meet Friday, no wait, let's do Saturday",
     "Let us meet on Saturday."),
    # A professional rewording swaps most characters but adds nothing
    ("professional", "这事儿我搞不定", "此事我无法完成"),
    ("professional", "老板说这个活儿得赶紧弄完", "老板表示这项工作需要尽快完成"),
    ("Formal Writing", "这玩意儿老是出毛病得赶紧修", "该设备经常出现故障，需要尽快维修。"),
    ("Formal Writing", "can you send me the report by friday",
     "Please send me the report by Friday."),
    # The wizard's tone test sample
    ("professional", "um so like I was thinking we should you know meet on Tuesday",
     "I suggest we schedule a meeting for Tuesday."),
    ("professional", "我们讨论了如何提高效率", "我们讨论了提高效率的方法。"),
    # Dropping English fillers from mixed speech is not translation
    ("professional", "okay so basically 我们下周要把这个方案做完", "我们下周需要把这个方案做完。"),
    ("professional", "嗯 like 我觉得这个 design you know 还不错", "我觉得这个design还不错。"),
    # Conditional inversion is not a question
    ("professional", "should you have any questions feel free to reach out",
     "If you have any questions, please feel free to reach out."),
    ("professional", "had we known about the delay we would have planned differently",
     "If we had known about the delay, we would have planned differently."),
])
def test_styled_path_keeps_polish(mode, raw, polished):
    from src.llm_cleanup import LLMCleanup
    llm = LLMCleanup(backend=_FakeBackend(reply=polished))
    assert _cleanup_like_app(llm, _mode(mode), raw) == polished


@pytest.mark.parametrize("mode,raw,reply", [
    ("professional", "what is the capital of France",
     "What is the capital of France? Paris."),
    ("professional", "remind me to call mom at five",
     "Sure! I'll remind you to call your mom at 5."),
])
def test_styled_path_rejects(mode, raw, reply):
    from src.llm_cleanup import LLMCleanup
    from src.text_postprocess import strip_filler_words
    llm = LLMCleanup(backend=_FakeBackend(reply=reply))
    assert _cleanup_like_app(llm, _mode(mode), raw) == strip_filler_words(raw)


def test_speculative_result_is_only_reused_by_the_same_mode():
    """A speculative result made under one mode's prompt and guards (here a
    translation) must not be pasted after switching to another mode."""
    import time
    from src.llm_cleanup import LLMCleanup
    llm = LLMCleanup(backend=_FakeBackend(reply="We meet on Saturday."))
    text = "我们周六开会"
    translate = f"Translate to English: {text}"
    llm.speculative_cleanup(text, custom_prompt=translate, allow_script_change=True)
    deadline = time.monotonic() + 5
    while llm._speculative_result is None and time.monotonic() < deadline:
        time.sleep(0.01)
    formal = _mode("Formal Writing")
    assert llm.get_speculative_result(
        text, custom_prompt=formal.render_prompt(text),
        allow_script_change=False, echo_questions=True) is None
    assert llm.get_speculative_result(
        text, custom_prompt=translate, allow_script_change=True) == "We meet on Saturday."


def test_refresh_local_models_falls_back_to_saved_model(monkeypatch):
    """Refreshing while the dropdown shows "(no models found)" must select
    the saved model, not Ollama's first one, which Apply would then save."""
    import types
    from AppKit import NSPopUpButton
    from Foundation import NSMakeRect
    from src import settings_window
    popup = NSPopUpButton.alloc().initWithFrame_pullsDown_(NSMakeRect(0, 0, 200, 22), False)
    popup.addItemWithTitle_("(no models found)")
    monkeypatch.setattr(settings_window.OllamaBackend, "list_models",
                        staticmethod(lambda *a, **k: ["qwen3:4b", "gemma3:4b", "qwen2.5:3b"]))
    window = types.SimpleNamespace(
        _local_model_popup=popup,
        _mgr=types.SimpleNamespace(get=lambda key, default=None: "qwen2.5:3b"
                                   if key == "llm_local_model" else default),
    )
    settings_window.SettingsWindow._refresh_local_models(window)
    assert popup.titleOfSelectedItem() == "qwen2.5:3b"


# --- Review follow-ups to #5 and #7 -----------------------------------------
# The strip now keeps stuttered non-safelisted words, capitalised
# hesitations inside a sentence and joined backchannels, so the guards have
# to see past them.

@pytest.mark.parametrize("text", [
    "is is the store open on sunday",
    "do do you know when the meeting starts",
    "when when when is the deadline",
    "what what time is it",
])
def test_is_question_sees_past_a_stuttered_opener(text):
    from src.llm_cleanup import _is_question
    assert _is_question(text)


def test_is_question_sees_past_a_stuttered_cleft():
    from src.llm_cleanup import _is_question
    assert not _is_question("what what I want to say is that the launch went well")


def test_cleanup_rejects_answer_to_a_stuttered_question(caplog):
    from src.llm_cleanup import LLMCleanup
    llm = LLMCleanup(backend=_FakeBackend(reply="The store is open on Sunday from 9 to 5."))
    assert llm.cleanup("is is the store open on sunday") == "is is the store open on sunday"
    assert "answered a dictated question" in caplog.text


@pytest.mark.parametrize("raw,cleaned", [
    ("Mm-hmm, I'll send it tonight", "I'll send it tonight."),
    ("mm-hmm, yes that works", "Yes, that works."),
    ("uh-huh, sure, let's do that tomorrow", "Sure, let's do that tomorrow."),
])
def test_dropped_leading_backchannel_is_not_a_reply(raw, cleaned):
    from src.llm_cleanup import LLMCleanup
    llm = LLMCleanup(backend=_FakeBackend(reply=cleaned))
    assert llm.cleanup(raw) == cleaned


def test_reply_after_a_backchannel_is_still_caught():
    from src.llm_cleanup import LLMCleanup
    llm = LLMCleanup(backend=_FakeBackend(reply="Sure, I'll send it tonight."))
    assert llm.cleanup("Mm-hmm, send it tonight") == "Mm-hmm, send it tonight"


def test_dropped_capitalised_hesitation_in_chinese_is_not_a_translation():
    from src.llm_cleanup import LLMCleanup
    raw = "好的，Uhh, 我觉得这个方案可以，我们下周再讨论。"
    cleaned = "好的，我觉得这个方案可以，我们下周再讨论。"
    llm = LLMCleanup(backend=_FakeBackend(reply=cleaned))
    assert llm.cleanup(raw) == cleaned
