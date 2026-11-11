"""HumanEval prompts, code extraction and scoring by running the unit tests.

Generated code is executed in a subprocess with a timeout. On Modal this happens
inside the GPU container, which is isolated; do not score untrusted outputs on a
machine you care about.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

INSTRUCTION = (
    "Complete the following Python function. Reply with the complete function, "
    "including its signature and any imports it needs, in a single ```python code block."
)

_FENCE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.DOTALL)


def build_messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": f"{INSTRUCTION}\n\n```python\n{prompt}```"}]


def _prompt_imports(prompt: str) -> str:
    """Import lines from the task prompt, so a reply that drops them still runs."""
    lines = [ln for ln in prompt.splitlines() if ln.startswith(("import ", "from "))]
    return "\n".join(lines)


def extract_code(text: str, prompt: str, entry_point: str) -> str:
    """The program the model meant: its code block, completed with the prompt if it only gave a body."""
    blocks = _FENCE.findall(text)
    if blocks:
        defining = [b for b in blocks if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", b)]
        code = (defining or blocks)[-1]
    else:
        code = text
    if not re.search(rf"def\s+{re.escape(entry_point)}\s*\(", code):
        code = prompt + code  # reply was the function body only
    return code


def build_program(code: str, problem: dict) -> str:
    return (
        f"{_prompt_imports(problem['prompt'])}\n\n{code}\n\n"
        f"{problem['test']}\n\ncheck({problem['entry_point']})\n"
    )


def run_program(program: str, timeout_s: float = 10.0) -> tuple[bool, str | None]:
    """Run a program in a fresh, isolated interpreter. Returns (passed, last error line)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            p = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True, text=True,
                               timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if p.returncode == 0:
        return True, None
    err = [ln for ln in p.stderr.strip().splitlines() if ln.strip()]
    return False, (err[-1][:200] if err else f"exit {p.returncode}")


def score(answer_text: str, problem: dict, timeout_s: float = 10.0) -> dict:
    code = extract_code(answer_text, problem["prompt"], problem["entry_point"])
    passed, error = run_program(build_program(code, problem), timeout_s)
    return {"correct": passed, "error": error}


def score_many(answer_texts: list[str], problems: list[dict], workers: int = 16) -> list[dict]:
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(score, answer_texts, problems))
