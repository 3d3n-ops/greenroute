"""Effort and size routing for gpt-oss, simulated from per-problem results.

    python analysis/gptoss_router.py results/phase1/<gpt-oss run id>

gpt-oss has no GSM8K train run, so thresholds are set by 2-fold cross-validation
on the test problems: split by problem (all copies of a HumanEval problem stay
together), set each threshold on one half, score it on the other, and pool the
two scored halves. No threshold is ever scored on the problems it was set on.

Energy is attributed at saturation (steady-state J per output token times each
answer's tokens), as in analysis/phase1_router.py. A routed task pays for the
cheap answer and, if escalated, the expensive one too.
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

from phase1_table import load, per_task_joules, signals

SHARES = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
PAIRS = [
    ("gpt-oss-20b_low", "gpt-oss-20b_medium"),
    ("gpt-oss-20b_low", "gpt-oss-20b_high"),
    ("gpt-oss-20b_low", "gpt-oss-120b_low"),
    ("gpt-oss-20b_low", "gpt-oss-120b_medium"),
    ("gpt-oss-20b_low", "gpt-oss-120b_high"),
    ("gpt-oss-20b_medium", "gpt-oss-120b_high"),
    ("gpt-oss-120b_low", "gpt-oss-120b_high"),
]


def router_signals(result: dict) -> dict[str, list[float]]:
    """phase1_table.signals, but an answer with no confidence (no answer at all) counts as least sure."""
    out = signals(result)
    recs = result["records"]
    for k in ("mean_lp", "min_lp", "tail_mean_lp"):
        if f"low {k}" not in out:
            out[f"low {k}"] = [-r["confidence"][k] if k in r["confidence"] else float("inf") for r in recs]
    return out


def key(result: dict) -> str:
    c = result["config"]
    return f"{c['model'].split('/')[-1]}_{c.get('effort')}"


def fold(record: dict) -> int:
    """Stable half for a problem, shared by all its copies."""
    return zlib.crc32(str(record["index"]).encode()) % 2


def mean_j(result: dict) -> float:
    e = per_task_joules(result)
    return sum(e) / len(e)


def threshold_for_share(scores: list[float], share: float) -> float:
    s = sorted(scores, reverse=True)
    return s[max(1, round(share * len(s))) - 1]


def cross_validated(a: dict, b: dict, score: list[float], share: float) -> tuple[float, float, float]:
    """(accuracy, J/task, share escalated) pooled over both folds, each thresholded on the other."""
    ra, rb = a["records"], b["records"]
    ea, eb = per_task_joules(a), per_task_joules(b)
    folds = [fold(r) for r in ra]
    correct = joules = escalated = 0.0
    for f in (0, 1):
        fit_idx = [i for i in range(len(ra)) if folds[i] != f]
        t = threshold_for_share([score[i] for i in fit_idx], share)
        for i in (i for i in range(len(ra)) if folds[i] == f):
            esc = score[i] >= t
            correct += rb[i]["correct"] if esc else ra[i]["correct"]
            joules += ea[i] + (eb[i] if esc else 0.0)
            escalated += esc
    n = len(ra)
    return correct / n, joules / n, escalated / n


def aligned(a: dict, b: dict) -> bool:
    return [(r["index"], r.get("copy", 0)) for r in a["records"]] == [(r["index"], r.get("copy", 0)) for r in b["records"]]


def main(run_dir: str) -> None:
    runs = load(Path(run_dir))
    for task in ("gsm8k/test", "humaneval/test"):
        res = {key(v): v for v in runs.values() if v["config"]["task"] == task}
        if not res:
            continue
        print(f"\n=== {task} ===")
        print(f"{'single config':<22} {'accuracy':>8} {'J/task':>8}")
        for k in sorted(res, key=lambda k: mean_j(res[k])):
            print(f"{k:<22} {res[k]['summary']['accuracy']:8.1%} {mean_j(res[k]):8.1f}")
        for ka, kb in PAIRS:
            if ka not in res or kb not in res:
                continue
            a, b = res[ka], res[kb]
            if not aligned(a, b):
                print(f"\n{ka} -> {kb}: records not aligned, skipped")
                continue
            print(f"\n{ka} -> {kb}   (alone: {a['summary']['accuracy']:.1%} at {mean_j(a):.1f} J, "
                  f"{b['summary']['accuracy']:.1%} at {mean_j(b):.1f} J)")
            print(f"  {'signal':<18} {'share':>6} {'escalated':>9} {'accuracy':>8} {'J/task':>8}")
            sig = router_signals(a)
            for name, score in sig.items():
                if name == "oracle":
                    continue
                for share in SHARES:
                    acc, j, esc = cross_validated(a, b, score, share)
                    print(f"  {name:<18} {share:6.0%} {esc:9.0%} {acc:8.1%} {j:8.1f}")
            # Upper bound only: escalate exactly the cheap configuration's wrong answers.
            ra, rb = a["records"], b["records"]
            ea, eb = per_task_joules(a), per_task_joules(b)
            wrong = [not r["correct"] for r in ra]
            acc = sum(rb[i]["correct"] if w else ra[i]["correct"] for i, w in enumerate(wrong)) / len(ra)
            j = (sum(ea) + sum(e for e, w in zip(eb, wrong) if w)) / len(ra)
            print(f"  {'(perfect router)':<18} {'':>6} {sum(wrong) / len(ra):9.0%} {acc:8.1%} {j:8.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
