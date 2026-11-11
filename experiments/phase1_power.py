"""Phase 1: power curves per model and mode, from a vLLM server under Poisson traffic.

    modal run experiments/phase1_power.py --suite check   # one short sweep, to test the path
    modal run experiments/phase1_power.py                  # Qwen3 1.7B/4B/8B x fast/thinking

Each model and mode runs on its own GPU: the server loads once and steps through
request rates. Per level, power, token throughput and energy per task are measured
over the steady window (after one request lifetime, until arrivals stop); see
greenroute/power.py. The fit power = floor + J/token x tokens/s gives each
configuration's power curve, which prices any traffic level, not just saturation.

Results: results/phase1_power/<run id>/<label>.json plus summary.csv.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import modal

from greenroute.modal_image import gpu_image
from greenroute.provenance import git_sha

GPU = os.environ.get("GREENROUTE_GPU", "H100")
CPUS = float(os.environ.get("GREENROUTE_CPUS", "16"))
QWEN = ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B")
# Thinking answers are ~6x longer, so the GPU saturates at far fewer requests per second.
RATES = {False: [1, 2, 4, 8, 16, 32, 64], True: [0.25, 0.5, 1, 2, 4, 8]}
DURATION_S = 90.0
N_MIN, N_MAX = 20, 3000

SUITES = {
    "check": [("Qwen/Qwen3-1.7B", True, [1, 4], 30.0)],
    "all": [(m, think, RATES[think], DURATION_S) for m in QWEN for think in (False, True)],
}

hf_cache = modal.Volume.from_name("greenroute-hf-cache", create_if_missing=True)
app = modal.App("greenroute-phase1-power")


def label(model: str, thinking: bool) -> str:
    return f"{model.split('/')[-1]}_{'think' if thinking else 'fast'}"


@app.function(
    gpu=GPU,
    image=gpu_image(),
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=5400,
    cpu=CPUS,  # the API server and load generator are CPU-bound at high rates (Sprint 2)
    memory=32768,
    max_containers=int(os.environ.get("GREENROUTE_MAX_GPUS", "6")),
)
def sweep(model: str, thinking: bool, rates: list[float], duration_s: float) -> dict:
    from greenroute.serve_bench import run_latency_sweep

    result = run_latency_sweep(model, rates, duration_s=duration_s, n_min=N_MIN, n_max=N_MAX, thinking=thinking)
    result["config"]["label"] = label(model, thinking)
    result["config"]["gpu_requested"] = GPU
    return result


CSV_FIELDS = [
    "label", "model", "thinking", "rate", "n", "achieved_rps", "accuracy",
    "steady_s", "steady_watts", "steady_tokens_per_s", "steady_j_per_token", "steady_j_per_task",
    "whole_j_per_task", "ttft_p50", "ttft_p99", "e2e_p50", "e2e_p99", "running_mean", "waiting_mean",
    "idle_watts", "n_errors",
]


def rows(result: dict) -> list[dict]:
    cfg = result["config"]
    out = []
    for lv in result["levels"]:
        st, e, lat, v = lv.get("steady") or {}, lv["energy"], lv["latency"], lv["vllm"]
        out.append({
            "label": cfg["label"], "model": cfg["model"], "thinking": cfg["thinking"], "rate": lv["rate"],
            "n": lv["n"], "achieved_rps": lat["achieved_rps"], "accuracy": e["accuracy"],
            "steady_s": st.get("seconds"), "steady_watts": st.get("avg_watts"),
            "steady_tokens_per_s": st.get("tokens_per_s"), "steady_j_per_token": st.get("j_per_output_token"),
            "steady_j_per_task": st.get("j_per_task"), "whole_j_per_task": e["j_per_task_total"],
            "ttft_p50": lat["ttft_s"].get("p50"), "ttft_p99": lat["ttft_s"].get("p99"),
            "e2e_p50": lat["e2e_s"].get("p50"), "e2e_p99": lat["e2e_s"].get("p99"),
            "running_mean": v.get("num_requests_running_mean"), "waiting_mean": v.get("num_requests_waiting_mean"),
            "idle_watts": cfg["idle_watts"], "n_errors": lat["n_errors"],
        })
    return out


@app.local_entrypoint()
def main(suite: str = "all"):
    configs = SUITES[suite]
    sha = git_sha()
    out_dir = Path("results/phase1_power") / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Launching {len(configs)} sweeps on {GPU} ({CPUS:g} CPUs) -> {out_dir}", flush=True)

    all_rows = []
    for result in sweep.starmap(configs, return_exceptions=True, order_outputs=False):
        if isinstance(result, Exception):
            print(f"  FAILED: {result!r}", flush=True)
            continue
        cfg = result["config"]
        cfg["git_sha"] = sha
        cfg["cpu_requested"] = CPUS
        cfg["command"] = f"GREENROUTE_GPU={GPU} modal run experiments/phase1_power.py --suite {suite}"
        (out_dir / f"{cfg['label']}.json").write_text(json.dumps(result))
        for r in rows(result):
            all_rows.append(r)
            sw = r["steady_watts"]
            print(
                f"  {r['label']:<16} {r['rate']:>5g} rps  acc {r['accuracy']:.1%}  "
                f"steady {sw or float('nan'):5.0f} W  {r['steady_tokens_per_s'] or float('nan'):7.0f} tok/s  "
                f"J/task {r['steady_j_per_task'] or float('nan'):7.1f} (whole {r['whole_j_per_task']:7.1f})  "
                f"TTFT p99 {r['ttft_p99'] or float('nan'):6.2f}s  err {r['n_errors']}",
                flush=True,
            )

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {len(all_rows)} levels to {out_dir}", flush=True)
