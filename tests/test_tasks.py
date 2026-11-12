from types import SimpleNamespace as NS

from greenroute import humaneval
from greenroute.table import confidence, reasoning_tokens, think_end_id
from greenroute.tasks import TASKS, split_thinking

PROBLEM = {
    "task_id": "HumanEval/0",
    "prompt": 'from typing import List\n\n\ndef add(xs: List[int]) -> int:\n    """Sum of xs."""\n',
    "entry_point": "add",
    "test": "def check(candidate):\n    assert candidate([1, 2]) == 3\n    assert candidate([]) == 0\n",
}


def test_split_thinking():
    assert split_thinking("<think>\nhmm 7\n</think>\n\nAnswer: 8") == ("hmm 7", "Answer: 8")
    assert split_thinking("Answer: 8") == ("", "Answer: 8")
    # Cut off mid-reasoning: no answer, so the number in the reasoning is not scored.
    assert split_thinking("<think>\nso it is 7 or") == ("so it is 7 or", None)


def test_gsm8k_scoring_ignores_unfinished_reasoning():
    p = {"answer": "work\n#### 8"}
    s = TASKS["gsm8k"].score_batch(["Answer: 8", None, "Answer: 9"], [p, p, p])
    assert [x["correct"] for x in s] == [True, False, False]
    assert s[1]["predicted"] is None


def test_humaneval_full_function_passes():
    reply = "Here you go:\n```python\ndef add(xs):\n    return sum(xs)\n```\n"
    assert humaneval.score(reply, PROBLEM) == {"correct": True, "error": None}


def test_humaneval_body_only_and_missing_import():
    # Body only: completed with the prompt, which also supplies `List`.
    assert humaneval.score("```python\n    return sum(xs)\n```", PROBLEM)["correct"]
    # Full function that uses List without importing it: prompt imports are prepended.
    reply = "```python\ndef add(xs: List[int]) -> int:\n    return sum(xs)\n```"
    assert humaneval.score(reply, PROBLEM)["correct"]


def test_humaneval_wrong_answer_and_timeout():
    wrong = humaneval.score("```python\ndef add(xs):\n    return 0\n```", PROBLEM)
    assert not wrong["correct"] and "AssertionError" in wrong["error"]
    passed, err = humaneval.run_program("while True:\n    pass\n", timeout_s=0.5)
    assert not passed and err == "timeout"


def test_humaneval_batch_marks_missing_answers():
    s = TASKS["humaneval"].score_batch([None, "```python\ndef add(xs):\n    return sum(xs)\n```"], [PROBLEM, PROBLEM])
    assert [x["correct"] for x in s] == [False, True]
    assert s[0]["error"] == "no answer"


def test_reasoning_tokens_and_think_id():
    tok = NS(convert_tokens_to_ids=lambda t: 99, unk_token_id=0)
    assert think_end_id(tok) == 99
    assert think_end_id(NS(convert_tokens_to_ids=lambda t: 0, unk_token_id=0)) is None
    assert reasoning_tokens([5, 6, 99, 7, 8], 99) == 3
    assert reasoning_tokens([5, 6, 7], 99) == 0
    assert reasoning_tokens([5, 6, 7], None) == 0


def test_confidence_skips_reasoning_and_summarizes_answer():
    ids = [1, 2, 3, 4]
    lps = [{1: NS(logprob=-5.0)}, {2: NS(logprob=-0.1)}, {3: NS(logprob=-0.3)}, {4: NS(logprob=-0.2)}]
    c = confidence(ids, lps, n_reasoning=1)
    assert c["min_lp"] == -0.3
    assert abs(c["mean_lp"] - (-0.2)) < 1e-9
    assert confidence(ids, None, 0) == {}
    assert confidence(ids, lps, n_reasoning=4) == {}


def test_split_thinking_harmony():
    # gpt-oss, decoded with special tokens kept: reasoning on "analysis", answer on "final".
    text = ("<|channel|>analysis<|message|>6 plus 2 is 8.<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Answer: 8<|return|>")
    assert split_thinking(text) == ("6 plus 2 is 8.", "Answer: 8")
    # Cut off before the final channel: no answer.
    assert split_thinking("<|channel|>analysis<|message|>so it is 7 or") == ("so it is 7 or", None)


def test_reasoning_tokens_with_a_marker_sequence():
    # gpt-oss's end of reasoning is three tokens: <|channel|> final <|message|>.
    assert reasoning_tokens([1, 2, 7, 8, 9, 4, 5], [7, 8, 9]) == 5
    assert reasoning_tokens([1, 7, 8, 2], [7, 8, 9]) == 0
    assert reasoning_tokens([7, 8, 9, 3, 7, 8, 9, 4], [7, 8, 9]) == 7  # the last occurrence counts
