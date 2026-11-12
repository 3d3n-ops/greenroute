"""Reasoning modes per model family.

Qwen3 switches thinking on or off through its chat template (`enable_thinking`).
gpt-oss always reasons, at a chosen effort (low / medium / high), and writes its
reasoning and its answer on separate "channels" of the Harmony format:

    <|channel|>analysis<|message|>...reasoning...<|end|><|start|>assistant<|channel|>final<|message|>...answer...<|return|>

A mode is named by `thinking` (bool, Qwen3-style) or, for gpt-oss, an effort string.
"""

from __future__ import annotations

EFFORTS = ("low", "medium", "high")
HARMONY_FINAL = "<|channel|>final<|message|>"
HARMONY_STOP = ("<|return|>", "<|end|>", "<|call|>")


def is_harmony(model: str) -> bool:
    return "gpt-oss" in model.lower()


def resolve(mode: bool | str) -> tuple[bool, str | None]:
    """(thinking, effort) from a mode: True/False for Qwen3-style models, an effort for gpt-oss."""
    if isinstance(mode, str):
        if mode not in EFFORTS:
            raise ValueError(f"unknown reasoning effort {mode!r}; expected one of {EFFORTS}")
        return True, mode
    return bool(mode), None


def mode_name(thinking: bool, effort: str | None) -> str:
    return effort or ("think" if thinking else "fast")


def chat_template_kwargs(model: str, thinking: bool, effort: str | None) -> dict:
    if is_harmony(model):
        return {"reasoning_effort": effort or "medium"}
    return {"enable_thinking": thinking}


def request_extra_body(model: str, thinking: bool, effort: str | None) -> dict:
    """Extra fields for an OpenAI-compatible chat request to a vLLM server."""
    if is_harmony(model):
        return {"reasoning_effort": effort or "medium"}
    return {"chat_template_kwargs": {"enable_thinking": thinking}}


def primary_sampling(model: str) -> dict:
    """Decoding for each problem's main answer.

    Greedy (temperature 0) for Qwen3, as in every earlier run. gpt-oss uses OpenAI's
    recommended sampling instead: greedy decoding made about 1 in 10 high-effort
    replies loop until the token limit (smoke test, 2026-11-11). Sampled answers are
    seeded per request, so runs stay reproducible.
    """
    if is_harmony(model):
        return {"temperature": 1.0, "top_p": 1.0}
    return {"temperature": 0.0}


PRIMARY_SEED_BASE = 1_000_000  # request i's seed is base + i; kept clear of the extra samples' seeds


def sampling_kwargs(model: str, thinking: bool) -> dict:
    """Each family's recommended settings for sampled (non-greedy) answers."""
    if is_harmony(model):
        return {"temperature": 1.0, "top_p": 1.0}  # OpenAI's recommendation for gpt-oss
    return {"temperature": 0.6, "top_p": 0.95} if thinking else {"temperature": 0.7, "top_p": 0.8}
