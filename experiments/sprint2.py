"""Sprint 2: latency vs energy under realistic traffic, with GPU telemetry.

A vLLM OpenAI server per model; open-loop Poisson traffic stepped through request
rates on the same GPU. Per rate: J/task, TTFT / TPOT / end-to-end latency
percentiles, GPU power/clock/utilization/temperature, and vLLM's running/waiting
request counts and KV-cache usage.

    modal run experiments/sprint2.py
    modal run experiments/sprint2.py --models Qwen/Qwen3-1.7B --rates 1,8,64

Results: results/sprint2/<run id>/<model>.json plus summary.csv.
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

hf_cache = modal.Volume.from_name("greenroute-hf-cache", create_if_missing=True)
app = modal.App("greenroute-sprint2")


@app.function(
    gpu=GPU,
    image=gpu_image(),
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=3600,
    max_containers=int(os.environ.get("GREENROUTE_MAX_GPUS", "3")),
    # The vLLM API server and the load generator are CPU-bound at high request rates;
    # Modal's small default CPU share starved both in the first Sprint 2 run.
    cpu=CPUS,
    memory=32768,
)
def sweep(model: str, rates: list[float], max_num_seqs: int, duration_s: float) -> dict:
    from greenroute.serve_bench import run_latency_sweep

    result = run_latency_sweep(model, rates, max_num_seqs=max_num_seqs, duration_s=duration_s)
    result["config"]["gpu_requested"] = GPU
    return result


CSV_FIELDS = [
    "model", "label", "rate", "n", "accuracy", "accuracy_common", "n_common",
    "j_per_task_total", "j_per_task_active", "j_per_output_token_total",
    "j_per_solved_task_total", "ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99", "e2e_p50", "e2e_p99",
    "achieved_rps", "running_mean", "waiting_mean", "kv_mean", "power_w_mean", "sm_clock_mhz_mean",
    "util_pct_mean", "temp_c_max", "n_errors",
    "send_lag_p99", "client_cpu_frac_max",
]


def _row(model: str, lv: dict) -> dict:
    e, lat, g, v = lv["energy"], lv["latency"], lv["gpu"], lv["vllm"]
    kv = v.get("kv_cache_usage_perc_mean", v.get("gpu_cache_usage_perc_mean"))
    return {
        "model": model, "label": lv["label"], "rate": lv["rate"], "n": lv["n"],
        "accuracy": e["accuracy"],
        "accuracy_common": lv.get("accuracy_common"), "n_common": lv.get("n_common"),
        "j_per_output_token_total": e["j_per_output_token_total"],
        "j_per_task_total": e["j_per_task_total"],
        "j_per_task_active": e["energy_j_active"] / e["n_tasks"],
        "j_per_solved_task_total": e["j_per_solved_task_total"],
        "ttft_p50": lat["ttft_s"].get("p50"), "ttft_p99": lat["ttft_s"].get("p99"),
        "tpot_p50": lat["tpot_s"].get("p50"), "tpot_p99": lat["tpot_s"].get("p99"),
        "e2e_p50": lat["e2e_s"].get("p50"), "e2e_p99": lat["e2e_s"].get("p99"),
        "achieved_rps": lat["achieved_rps"],
        "running_mean": v.get("num_requests_running_mean"),
        "waiting_mean": v.get("num_requests_waiting_mean"),
        "kv_mean": kv,
        "power_w_mean": g.get("power_w_mean"), "sm_clock_mhz_mean": g.get("sm_clock_mhz_mean"),
        "util_pct_mean": g.get("util_pct_mean"), "temp_c_max": g.get("temp_c_max"),
        "n_errors": lat["n_errors"],
        "send_lag_p99": lat.get("send_lag_s", {}).get("p99"),
        "client_cpu_frac_max": lat.get("client_cpu_frac_max"),
    }


@app.local_entrypoint()
def main(
    models: str = "Qwen/Qwen3-1.7B,Qwen/Qwen3-8B",
    rates: str = "1,2,4,8,16,32,64,0",
    max_num_seqs: int = 256,
    duration_s: float = 90.0,
):
    model_list = [m.strip() for m in models.split(",")]
    rate_list = [float(r) for r in rates.split(",")]
    sha = git_sha()

    out_dir = Path("results/sprint2") / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Launching {len(model_list)} sweeps on {GPU}, rates {rate_list} (0 = burst) -> {out_dir}", flush=True)

    rows = []
    args = [(m, rate_list, max_num_seqs, duration_s) for m in model_list]
    for result in sweep.starmap(args, return_exceptions=True, order_outputs=False):
        if isinstance(result, Exception):
            print(f"  FAILED: {result!r}", flush=True)
            continue
        cfg = result["config"]
        cfg["git_sha"] = sha
        cfg["cpu_requested"] = CPUS  # set here: the env override is not visible inside the container
        cfg["command"] = (f"GREENROUTE_GPU={GPU} modal run experiments/sprint2.py --models {models} "
                          f"--rates {rates} --max-num-seqs {max_num_seqs} --duration-s {duration_s}")
        name = cfg["model"].split("/")[-1]
        (out_dir / f"{name}.json").write_text(json.dumps(result))
        for lv in result["levels"]:
            row = _row(cfg["model"], lv)
            rows.append(row)
            print(
                f"  {name:<11} {lv['label']:>7}  J/task {row['j_per_task_total']:6.1f}  "
                f"TTFT p50 {row['ttft_p50'] or 0:6.3f}s p99 {row['ttft_p99'] or 0:6.2f}s  "
                f"TPOT p50 {(row['tpot_p50'] or 0) * 1000:5.1f}ms  "
                f"running {row['running_mean'] or 0:5.1f}  {row['power_w_mean'] or 0:4.0f} W  "
                f"err {row['n_errors']}",
                flush=True,
            )

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} levels from {len(model_list)} models to {out_dir}", flush=True)
