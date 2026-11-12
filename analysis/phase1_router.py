"""Cascades with escalation thresholds set on GSM8K train and scored once on test.

    python analysis/phase1_router.py <train run dir> <test run dir>

For each cheap -> expensive pair and each signal, thresholds are the signal values
at which a given share of *train* problems escalates. Each threshold is then
applied unchanged to test, which is how a deployed router would behave. Every
cost is charged to the task: the cheap attempt, the escalated attempt, and the
signal itself (sample disagreement needs k extra sampled answers).

Energy is attributed at the saturated load, as in phase1_table.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

from phase1_table import load, per_task_joules, signals

TRAIN_SHARES = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
PAIRS = [  # (cheap, expensive): within-model effort routing first, then across sizes
    ("Qwen3-1.7B_fast", "Qwen3-1.7B_think"),
    ("Qwen3-4B_fast", "Qwen3-4B_think"),
    ("Qwen3-8B_fast", "Qwen3-8B_think"),
    ("Qwen3-1.7B_fast", "Qwen3-8B_fast"),
    ("Qwen3-4B_fast", "Qwen3-8B_think"),
    ("Qwen3-1.7B_fast", "Qwen3-8B_think"),
]


def signal_cost(result: dict, name: str) -> float:
    """Energy per task spent only to compute the signal."""
    if name == "sample disagreement":
        smp = result["samples"]
        steady = (smp.get("steady") or {}).get("j_per_task")  # saturated cost, where measured
        return steady if steady is not None else smp["energy_j_total"] / len(result["records"])
    return 0.0  # token probabilities and length come with the answer


def evaluate(a: dict, b: dict, score: list[float], threshold: float, extra: float) -> tuple[float, float, float]:
    """(accuracy, J/task, share escalated) when problems with score >= threshold escalate."""
    ra, rb = a["records"], b["records"]
    ea, eb = per_task_joules(a), per_task_joules(b)
    esc = [s >= threshold for s in score]
    n = len(ra)
    acc = sum(rb[i]["correct"] if esc[i] else ra[i]["correct"] for i in range(n)) / n
    joules = sum(ea) + sum(e for e, x in zip(eb, esc) if x)
    return acc, joules / n + extra, sum(esc) / n


def threshold_for_share(score: list[float], share: float) -> float:
    """Signal value above which about `share` of these problems lie."""
    s = sorted(score, reverse=True)
    k = max(1, round(share * len(s)))
    return s[k - 1]


def joules(result: dict) -> float:
    """Mean energy per task on the same basis the cascades use (steady state where measured)."""
    e = per_task_joules(result)
    return sum(e) / len(e)


def key(result: dict) -> str:
    c = result["config"]
    return f"{c['model'].split('/')[-1]}_{'think' if c['thinking'] else 'fast'}"


def main(train_dir: str, test_dir: str) -> None:
    train = {key(v): v for v in load(Path(train_dir)).values() if v["config"]["task"] == "gsm8k/train"}
    test = {key(v): v for v in load(Path(test_dir)).values() if v["config"]["task"] == "gsm8k/test"}

    print(f"{'single config':<18} {'train acc':>9} {'test acc':>9} {'test J/task':>11}")
    for k in sorted(test, key=lambda k: joules(test[k])):
        tr = train[k]["summary"]["accuracy"] if k in train else float("nan")
        print(f"{k:<18} {tr:9.1%} {test[k]['summary']['accuracy']:9.1%} {joules(test[k]):11.1f}")

    for ka, kb in PAIRS:
        if not all(k in train and k in test for k in (ka, kb)):
            continue
        a, b = test[ka], test[kb]
        print(f"\n{ka} -> {kb}   (alone: {a['summary']['accuracy']:.1%} at {joules(a):.1f} J, "
              f"{b['summary']['accuracy']:.1%} at {joules(b):.1f} J)")
        print(f"  {'signal':<20} {'train share':>11} {'test share':>10} {'test acc':>9} {'J/task':>8}")
        tr_sig, te_sig = signals(train[ka]), signals(a)
        for name in tr_sig:
            if name == "oracle" or name not in te_sig:
                continue
            extra = signal_cost(a, name)
            seen = set()
            for share in TRAIN_SHARES:
                t = threshold_for_share(tr_sig[name], share)
                if t in seen:  # discrete signals: several shares map to the same threshold
                    continue
                seen.add(t)
                acc, j, esc = evaluate(a, b, te_sig[name], t, extra)
                print(f"  {name:<20} {share:11.0%} {esc:10.0%} {acc:9.1%} {j:8.1f}")
        # Upper bound only: escalate exactly the cheap configuration's wrong answers on test.
        acc, j, esc = evaluate(a, b, te_sig["oracle"], 1.0, 0.0)
        print(f"  {'(oracle bound)':<20} {'':>11} {esc:10.0%} {acc:9.1%} {j:8.1f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
