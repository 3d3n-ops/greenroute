import pytest

from greenroute.modes import chat_template_kwargs, mode_name, request_extra_body, resolve, sampling_kwargs


def test_resolve_and_names():
    assert resolve(False) == (False, None) and mode_name(False, None) == "fast"
    assert resolve(True) == (True, None) and mode_name(True, None) == "think"
    assert resolve("high") == (True, "high") and mode_name(True, "high") == "high"
    with pytest.raises(ValueError):
        resolve("max")


def test_family_settings():
    assert chat_template_kwargs("Qwen/Qwen3-8B", True, None) == {"enable_thinking": True}
    assert chat_template_kwargs("openai/gpt-oss-20b", True, "low") == {"reasoning_effort": "low"}
    assert request_extra_body("Qwen/Qwen3-8B", False, None) == {"chat_template_kwargs": {"enable_thinking": False}}
    assert request_extra_body("openai/gpt-oss-120b", True, "high") == {"reasoning_effort": "high"}
    assert sampling_kwargs("Qwen/Qwen3-8B", False) == {"temperature": 0.7, "top_p": 0.8}
    assert sampling_kwargs("openai/gpt-oss-20b", True) == {"temperature": 1.0, "top_p": 1.0}


def test_primary_sampling():
    from greenroute.modes import primary_sampling

    assert primary_sampling("Qwen/Qwen3-8B") == {"temperature": 0.0}
    assert primary_sampling("openai/gpt-oss-20b") == {"temperature": 1.0, "top_p": 1.0}
