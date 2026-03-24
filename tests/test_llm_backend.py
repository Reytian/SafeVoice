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
