"""The benchmark's tasks: where problems come from, how they are asked, and how answers are checked.

Every task has an automatic check (exact answer or unit tests); no LLM judges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from greenroute import gsm8k, humaneval


def split_thinking(text: str) -> tuple[str, str | None]:
    """Split a reply into (reasoning, answer).

    Qwen3-style models put reasoning inside <think>...</think>. A reply whose
    reasoning never closed was cut off by the token limit and has no answer, so
    the answer is None rather than whatever number the reasoning ended on.

    gpt-oss replies (decoded with special tokens kept) carry the answer on the
    Harmony "final" channel; a reply that never reached it has no answer.
    """
    from greenroute.modes import HARMONY_FINAL, HARMONY_STOP

    if HARMONY_FINAL in text:
        reasoning, answer = text.rsplit(HARMONY_FINAL, 1)
        for stop in HARMONY_STOP:
            answer = answer.split(stop, 1)[0]
        return _harmony_reasoning(reasoning), answer.strip()
    if "<|channel|>" in text:
        return _harmony_reasoning(text), None
    if "</think>" in text:
        reasoning, answer = text.split("</think>", 1)
        return reasoning.replace("<think>", "", 1).strip(), answer.strip()
    if "<think>" in text:
        return text.split("<think>", 1)[1].strip(), None
    return "", text.strip()


def _harmony_reasoning(text: str) -> str:
    """The analysis-channel text, without Harmony's markers."""
    for marker in ("<|start|>assistant", "<|channel|>analysis<|message|>", "<|channel|>", "<|message|>", "<|end|>"):
        text = text.replace(marker, " ")
    return " ".join(text.split())


@dataclass(frozen=True)
class Task:
    name: str
    dataset: tuple[str, str | None]  # Hugging Face path and config
    splits: tuple[str, ...]
    build_messages: Callable[[dict], list[dict]]
    max_tokens: int  # answer budget with thinking off; thinking runs get a larger one
    # Scores a batch of (answer text or None, problem). Batched because code tasks run tests in parallel.
    score_batch: Callable[[list[str | None], list[dict]], list[dict]]


def _gsm8k_score(answers: list[str | None], problems: list[dict]) -> list[dict]:
    out = []
    for a, p in zip(answers, problems):
        pred = gsm8k.extract_answer(a) if a else None
        out.append({"correct": pred is not None and pred == gsm8k.gold_answer(p["answer"]),
                    "predicted": pred, "gold": gsm8k.gold_answer(p["answer"])})
    return out


def _humaneval_score(answers: list[str | None], problems: list[dict]) -> list[dict]:
    have = [i for i, a in enumerate(answers) if a]
    scored = humaneval.score_many([answers[i] for i in have], [problems[i] for i in have])
    out = [{"correct": False, "error": "no answer"} for _ in answers]
    for i, s in zip(have, scored):
        out[i] = s
    return out


TASKS: dict[str, Task] = {
    "gsm8k": Task(
        name="gsm8k",
        dataset=("openai/gsm8k", "main"),
        splits=("test", "train"),
        build_messages=lambda p: gsm8k.build_messages(p["question"]),
        max_tokens=1024,
        score_batch=_gsm8k_score,
    ),
    "humaneval": Task(
        name="humaneval",
        dataset=("openai/openai_humaneval", None),
        splits=("test",),
        build_messages=lambda p: humaneval.build_messages(p["prompt"]),
        max_tokens=2048,
        score_batch=_humaneval_score,
    ),
}


def load_problems(task: Task, split: str, n: int | None = None, offset: int = 0) -> list[dict]:
    from datasets import load_dataset

    if split not in task.splits:
        raise ValueError(f"{task.name} has no {split!r} split (has {task.splits})")
    path, config = task.dataset
    ds = load_dataset(path, config, split=split) if config else load_dataset(path, split=split)
    end = len(ds) if n is None else min(len(ds), offset + n)
    return [dict(ds[i]) for i in range(offset, end)]
