"""Summarize a Phase 1 run and simulate cascades on its per-problem table.

    python analysis/phase1_table.py results/phase1/<run id>

Per-task energy is attributed at the saturated reference load: each configuration's
measured total energy divided by its total output tokens, times the task's tokens.
So the cascade numbers assume both configurations run on busy GPUs; at low traffic
the fixed floor dominates and they do not hold (see ROADMAP.md).

Thresholds here are swept on the same problems they are scored on, so these
curves are exploratory. Reported router numbers must tune on the train split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)


def load(run_dir: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(run_dir.glob("*.json"))}


def per_task_joules(result: dict) -> list[float]:
    s = result["summary"]
    jpt = s["energy_j_total"] / s["output_tokens"]
    return [jpt * r["output_tokens"] for r in result["records"]]


def signals(result: dict) -> dict[str, list[float]]:
    """Higher = less sure = escalate first."""
    recs = result["records"]
    out = {}
    for key in ("mean_lp", "min_lp", "tail_mean_lp"):
        if all(key in r["confidence"] for r in recs):
            out[f"low {key}"] = [-r["confidence"][key] for r in recs]
    out["length"] = [r["output_tokens"] for r in recs]
    # Share of sampled answers that disagree with the greedy answer. Exact-answer tasks only:
    # for code, judging samples would need the hidden tests, which a router does not have.
    if all(r.get("samples") for r in recs) and all("predicted" in r for r in recs):
        out["sample disagreement"] = [
            sum(s["predicted"] != r.get("predicted") for s in r["samples"]) / len(r["samples"])
            for r in recs
        ]
    out["oracle"] = [0.0 if r["correct"] else 1.0 for r in recs]
    return out


def cascade(a: dict, b: dict, score: list[float], frac: float) -> tuple[float, float]:
    """Accuracy and J/task when the top `frac` of problems by score escalate from a to b."""
    ra, rb = a["records"], b["records"]
    ea, eb = per_task_joules(a), per_task_joules(b)
    n = len(ra)
    order = sorted(range(n), key=lambda i: -score[i])
    esc = set(order[: int(round(frac * n))])
    acc = sum(rb[i]["correct"] if i in esc else ra[i]["correct"] for i in range(n)) / n
    joules = sum(ea) + sum(eb[i] for i in esc)  # the failed first attempt is charged too
    return acc, joules / n


def main(run_dir: str) -> None:
    runs = load(Path(run_dir))
    tasks = sorted({r["config"]["task"] for r in runs.values()})
    for task in tasks:
        cfgs = {k: v for k, v in runs.items() if v["config"]["task"] == task}
        print(f"\n### {task}\n")
        print(f"{'configuration':<38} {'acc':>6} {'tok/task':>9} {'J/task':>8} {'J/solved':>9} {'trunc':>6}")
        singles = []
        for k, v in sorted(cfgs.items(), key=lambda kv: kv[1]["summary"]["j_per_task_total"]):
            s = v["summary"]
            singles.append((s["accuracy"], s["j_per_task_total"], k))
            print(f"{k:<38} {s['accuracy']:6.1%} {s['mean_output_tokens']:9.0f} {s['j_per_task_total']:8.1f} "
                  f"{s['j_per_solved_task_total'] or float('nan'):9.1f} {s['n_truncated']:6d}")
        best_acc = max(a for a, _, _ in singles)
        floor = best_acc - 0.01
        cheapest = min((j, k) for a, j, k in singles if a >= floor)
        print(f"\nQuality floor {floor:.1%} (best single minus 1 point); cheapest single config meeting it: "
              f"{cheapest[1]} at {cheapest[0]:.1f} J/task")

        print("\nCascades (escalate the least-sure fraction from first to second config):")
        keys = list(cfgs)
        for ka in keys:
            for kb in keys:
                a, b = cfgs[ka], cfgs[kb]
                if ka == kb or a["summary"]["j_per_task_total"] >= b["summary"]["j_per_task_total"]:
                    continue
                for name, sc in signals(a).items():
                    best = None
                    for f in FRACTIONS:
                        acc, j = cascade(a, b, sc, f)
                        if acc >= floor and (best is None or j < best[1]):
                            best = (f, j, acc)
                    if best and best[1] < cheapest[0]:
                        print(f"  {ka} -> {kb} by {name}: escalate {best[0]:.0%}, {best[2]:.1%}, "
                              f"{best[1]:.1f} J/task ({best[1] / cheapest[0]:.0%} of cheapest single)")
        print("  (only cascades that meet the floor and beat the cheapest single config are listed)")


if __name__ == "__main__":
    main(sys.argv[1])
