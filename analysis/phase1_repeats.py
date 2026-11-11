"""Run-to-run spread of Phase 1 configurations measured more than once.

    python analysis/phase1_repeats.py results/phase1/<run A> results/phase1/<run B> ...

Groups results by model, task and thinking mode across the given run folders and
reports, per group: how much accuracy, energy and tokens move between runs, and
how many individual problems flip between right and wrong.
"""

from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from pathlib import Path

from phase1_table import load


def spread(xs: list[float]) -> str:
    m = statistics.fmean(xs)
    rel = (max(xs) - min(xs)) / m if m else float("nan")
    return f"{m:8.1f} [{min(xs):.1f}–{max(xs):.1f}] ±{rel / 2:.0%}"


def main(run_dirs: list[str]) -> None:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for d in run_dirs:
        for r in load(Path(d)).values():
            c = r["config"]
            # n is part of the key: a capped pilot run is not a repeat of a full one.
            groups[(c["task"], c["model"].split("/")[-1], "think" if c["thinking"] else "fast", c["n"])].append(r)

    print(f"{'configuration':<30} {'runs':>4} {'accuracy (range)':>18} {'J/task mean [range] ±half-range':>36} "
          f"{'tokens/task':>11} {'flips':>6}")
    for key in sorted(groups):
        runs = groups[key]
        if len(runs) < 2:
            continue
        accs = [r["summary"]["accuracy"] * 100 for r in runs]
        jpt = [r["summary"]["j_per_task_total"] for r in runs]
        tok = [r["summary"]["mean_output_tokens"] for r in runs]
        # Problems answered right in some runs and wrong in others.
        per_problem = zip(*[[x["correct"] for x in r["records"]] for r in runs])
        flips = sum(len(set(p)) > 1 for p in per_problem)
        name = f"{key[1]} {key[2]} {key[0]}"
        print(f"{name:<30} {len(runs):>4} {statistics.fmean(accs):8.1f} [{min(accs):.1f}–{max(accs):.1f}] "
              f"{spread(jpt):>36} {statistics.fmean(tok):11.0f} {flips:6d}")


if __name__ == "__main__":
    main(sys.argv[1:])
