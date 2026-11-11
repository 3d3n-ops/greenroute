"""Fit and print a power curve per model and mode from a phase1_power run.

    python analysis/power_curves.py results/phase1_power/<run id>

Power is modelled as idle + serving floor + J/token x tokens per second, fitted
over the steady windows of every traffic level. Energy per task at a traffic
level is then (power at that throughput) / (tasks per second).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from greenroute.power import fit_power_curve  # noqa: E402


def main(run_dir: str) -> None:
    runs = {p.stem: json.loads(p.read_text()) for p in sorted(Path(run_dir).glob("*.json"))}
    print(f"{'configuration':<16} {'idle W':>7} {'floor W':>8} {'mJ/token':>9} {'R²':>6} {'max tok/s':>10} {'points':>6}")
    fits = {}
    for name, r in runs.items():
        pts = [(lv["steady"]["tokens_per_s"], lv["steady"]["avg_watts"]) for lv in r["levels"] if lv.get("steady")]
        f = fit_power_curve(pts)
        fits[name] = (f, r)
        if f:
            print(f"{name:<16} {r['config']['idle_watts']:7.0f} {f['floor_w']:8.0f} {f['j_per_token'] * 1000:9.2f} "
                  f"{f['r2']:6.3f} {max(x for x, _ in pts):10.0f} {f['n_points']:6d}")
        else:
            print(f"{name:<16} not enough steady levels ({len(pts)})")

    print("\nPer level (steady window):")
    print(f"{'configuration':<16} {'rate':>6} {'got rps':>7} {'W':>5} {'tok/s':>7} {'J/token':>8} {'J/task':>8} "
          f"{'whole J/task':>12} {'TTFT p99':>8} {'acc':>6}")
    for name, (_, r) in fits.items():
        for lv in r["levels"]:
            st, lat = lv.get("steady"), lv["latency"]
            if not st:
                print(f"{name:<16} {lv['rate']:6g}  (no steady window)")
                continue
            print(f"{name:<16} {lv['rate']:6g} {lat['achieved_rps']:7.2f} {st['avg_watts']:5.0f} "
                  f"{st['tokens_per_s']:7.0f} {st['j_per_output_token']:8.4f} {st['j_per_task']:8.1f} "
                  f"{lv['energy']['j_per_task_total']:12.1f} {lat['ttft_s'].get('p99', float('nan')):8.2f} "
                  f"{lv['energy']['accuracy']:6.1%}")


if __name__ == "__main__":
    main(sys.argv[1])
