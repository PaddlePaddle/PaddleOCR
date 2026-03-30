"""Tests for MiniMax LLM provider configuration and CLI integration.

These tests import the ``llm_config`` module directly (bypassing the
heavyweight paddleocr top-level __init__) so they can run without
the full paddlex dependency graph.
"""

import importlib
import os
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Bootstrap: import llm_config without pulling in paddleocr.__init__
# (which requires paddlex).  We add the *parent* of ``paddleocr/`` to
# sys.path so that ``paddleocr._pipelines.llm_config`` resolves, while
# stubbing out the heavy paddleocr.__init__.
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ensure_llm_config_importable():
    """Make ``paddleocr._pipelines.llm_config`` importable in isolation."""
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

    # Provide lightweight stubs so "from paddleocr._pipelines.llm_config …"
    # does not trigger the real paddleocr/__init__.py (which imports paddlex).
    for mod_name in ("paddleocr", "paddleocr._pipelines"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    spec = importlib.util.spec_from_file_location(
        "paddleocr._pipelines.llm_config",
        os.path.join(_REPO_ROOT, "paddleocr", "_pipelines", "llm_config.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["paddleocr._pipelines.llm_config"] = mod
    spec.loader.exec_module(mod)
    return mod


llm_config = _ensure_llm_config_importable()

get_minimax_chat_bot_config = llm_config.get_minimax_chat_bot_config
MINIMAX_BASE_URL = llm_config.MINIMAX_BASE_URL
MINIMAX_DEFAULT_MODEL = llm_config.MINIMAX_DEFAULT_MODEL
MINIMAX_MODELS = llm_config.MINIMAX_MODELS


# =====================================================================
# Unit tests
# =====================================================================


class TestGetMinimaxChatBotConfig:
    """Unit tests for get_minimax_chat_bot_config()."""

    def test_returns_correct_structure_with_explicit_key(self):
        config = get_minimax_chat_bot_config(api_key="test-key-123")
        assert config == {
            "module_name": "chat_bot",
            "model_name": MINIMAX_DEFAULT_MODEL,
            "base_url": MINIMAX_BASE_URL,
            "api_type": "openai",
            "api_key": "test-key-123",
        }

    def test_uses_env_var_when_no_explicit_key(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "env-key-456")
        config = get_minimax_chat_bot_config()
        assert config["api_key"] == "env-key-456"

    def test_explicit_key_takes_precedence_over_env_var(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "env-key")
        config = get_minimax_chat_bot_config(api_key="explicit-key")
        assert config["api_key"] == "explicit-key"

    def test_raises_when_no_key_available(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        with pytest.raises(ValueError, match="MiniMax API key is required"):
            get_minimax_chat_bot_config()

    def test_api_type_is_openai(self):
        config = get_minimax_chat_bot_config(api_key="k")
        assert config["api_type"] == "openai"

    def test_base_url_points_to_minimax(self):
        config = get_minimax_chat_bot_config(api_key="k")
        assert "minimax" in config["base_url"]

    def test_default_model_is_m27(self):
        config = get_minimax_chat_bot_config(api_key="k")
        assert config["model_name"] == "MiniMax-M2.7"

    def test_minimax_models_registry(self):
        assert "MiniMax-M2.7" in MINIMAX_MODELS
        assert "MiniMax-M2.7-highspeed" in MINIMAX_MODELS
        for ctx in MINIMAX_MODELS.values():
            assert ctx == 204_800

    def test_config_module_name_is_chat_bot(self):
        config = get_minimax_chat_bot_config(api_key="k")
        assert config["module_name"] == "chat_bot"

    def test_empty_string_key_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "env-fallback")
        config = get_minimax_chat_bot_config(api_key="")
        assert config["api_key"] == "env-fallback"


# =====================================================================
# Integration tests (require MINIMAX_API_KEY)
# =====================================================================


class TestMinimaxIntegration:
    """Integration tests verifying MiniMax API connectivity."""

    @pytest.mark.skipif(
        not os.environ.get("MINIMAX_API_KEY"),
        reason="MINIMAX_API_KEY not set",
    )
    def test_minimax_api_responds(self):
        """Smoke-test the MiniMax API with a trivial chat request."""
        import json
        import urllib.request

        config = get_minimax_chat_bot_config()
        url = config["base_url"].rstrip("/") + "/chat/completions"
        payload = json.dumps(
            {
                "model": config["model_name"],
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 8,
                "temperature": 0.1,
            }
        ).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config['api_key']}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        assert "choices" in data
        assert len(data["choices"]) > 0

    @pytest.mark.skipif(
        not os.environ.get("MINIMAX_API_KEY"),
        reason="MINIMAX_API_KEY not set",
    )
    def test_minimax_highspeed_model_responds(self):
        """Verify M2.7-highspeed variant is also accessible."""
        import json
        import urllib.request

        config = get_minimax_chat_bot_config()
        url = config["base_url"].rstrip("/") + "/chat/completions"
        payload = json.dumps(
            {
                "model": "MiniMax-M2.7-highspeed",
                "messages": [{"role": "user", "content": "Reply OK"}],
                "max_tokens": 4,
                "temperature": 0.1,
            }
        ).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config['api_key']}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        assert "choices" in data
