"""Compare the straight-line power curve with curved ones on measured server sweeps.

    python analysis/power_models.py results/phase1_power/<run id> [more run dirs ...]

For each configuration: fit error (RMSE), R², and leave-one-out RMSE, the error
when pricing a traffic level the curve was not fitted on. Overloaded levels
(first-token p99 above 2 s) are left out, as in analysis/power_curves.py.
The ramp model's parameters are printed for the curve's write-up.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from greenroute.power import compare_power_curves  # noqa: E402

OVERLOADED_TTFT_P99_S = 2.0


def points(result: dict) -> list[tuple[float, float]]:
    return [(lv["steady"]["tokens_per_s"], lv["steady"]["avg_watts"]) for lv in result["levels"]
            if lv.get("steady") and (lv["latency"]["ttft_s"].get("p99") or 0) <= OVERLOADED_TTFT_P99_S]


def main(run_dirs: list[str]) -> dict:
    out = {}
    print(f"{'configuration':<22} {'pts':>3}  {'RMSE W: line':>12} {'ramp':>6} {'power':>6}   "
          f"{'leave-one-out W: line':>21} {'ramp':>6} {'power':>6}   ramp: floor W, ramp tok/s, mJ/token")
    for d in run_dirs:
        for p in sorted(Path(d).glob("*.json")):
            r = json.loads(p.read_text())
            pts = points(r)
            fits = compare_power_curves(pts, r["config"]["idle_watts"])
            out[f"{Path(d).name}/{p.stem}"] = fits
            g = lambda form, k: (fits.get(form) or {}).get(k)
            fmt = lambda v: f"{v:6.1f}" if v is not None else "     -"
            ramp = fits.get("ramp") or {}
            print(f"{p.stem:<22} {len(pts):>3}  {fmt(g('linear', 'rmse_w')):>12} {fmt(g('ramp', 'rmse_w'))} "
                  f"{fmt(g('power', 'rmse_w'))}   {fmt(g('linear', 'loo_rmse_w')):>21} {fmt(g('ramp', 'loo_rmse_w'))} "
                  f"{fmt(g('power', 'loo_rmse_w'))}   "
                  + (f"{ramp['idle_w'] + ramp['ramp_w']:.0f}, {ramp['ramp_tokens_per_s']:.0f}, "
                     f"{ramp['j_per_token'] * 1000:.1f}" if ramp else "-"))
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
