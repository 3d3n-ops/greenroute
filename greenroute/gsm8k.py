"""GSM8K prompts and scoring."""

from __future__ import annotations

import re

INSTRUCTION = (
    "Solve the following math problem step by step. "
    "End your response with a final line of the form 'Answer: <number>'."
)

_NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def build_messages(question: str) -> list[dict]:
    return [{"role": "user", "content": f"{INSTRUCTION}\n\n{question}"}]


def gold_answer(solution: str) -> str:
    """GSM8K reference solutions end with '#### <answer>'."""
    return normalize(solution.split("####")[-1])


def extract_answer(text: str) -> str | None:
    """Take the 'Answer:' line if present, else the last number in the text."""
    m = re.findall(rf"Answer:\s*\**\s*({_NUM})", text, flags=re.IGNORECASE)
    if m:
        return normalize(m[-1])
    nums = re.findall(_NUM, text)
    return normalize(nums[-1]) if nums else None


def normalize(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        x = float(s)
    except ValueError:
        return s
    return str(int(x)) if x == int(x) else str(x)


def is_correct(model_output: str, solution: str) -> bool:
    pred = extract_answer(model_output)
    return pred is not None and pred == gold_answer(solution)
