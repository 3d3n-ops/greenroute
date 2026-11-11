"""Sprint 1: repeats, batch-size sweep, and model sizes on the full GSM8K test set.

    modal run experiments/sprint1.py                    # all three suites
    modal run experiments/sprint1.py --suites batch     # one suite
    modal run experiments/sprint1.py --n 200            # quick, cheaper check

Each config runs in its own container (so on its own physical GPU), in parallel.
Results: results/sprint1/<run id>/<label>.json plus summary.csv.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from pathlib import Path

import modal

from greenroute.modal_image import gpu_image

# H100 by default: A100-80GB capacity on Modal was unavailable on 2026-11-11.
GPU = os.environ.get("GREENROUTE_GPU", "H100")
BASE_MODEL = "Qwen/Qwen3-1.7B"

SUITES = {
    # Same config three times on separate containers: run-to-run and GPU-to-GPU spread.
    "repeats": [(f"repeat{i}", BASE_MODEL, 32) for i in range(1, 4)],
    # Concurrency cap = how full the GPU gets. 32 is covered by the repeats.
    "batch": [(f"seqs{s}", BASE_MODEL, s) for s in (1, 8, 128, 256)],
    # Same family, larger sizes, same concurrency as the baseline.
    "models": [(m.split("/")[-1], m, 32) for m in ("Qwen/Qwen3-4B", "Qwen/Qwen3-8B")],
}

hf_cache = modal.Volume.from_name("greenroute-hf-cache", create_if_missing=True)
app = modal.App("greenroute-sprint1")


# max_containers caps parallel GPUs so the sweep queues instead of stalling on plan limits.
@app.function(
    gpu=GPU,
    image=gpu_image(),
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=7200,
    max_containers=int(os.environ.get("GREENROUTE_MAX_GPUS", "3")),
)
def measure(label: str, model: str, max_num_seqs: int, n: int, max_tokens: int) -> dict:
    from greenroute.bench import run_gsm8k

    result = run_gsm8k(model, n, max_num_seqs, max_tokens)
    result["config"]["label"] = label
    result["config"]["gpu_requested"] = GPU
    return result


CSV_FIELDS = [
    "label", "model", "max_num_seqs", "n", "gpu", "accuracy",
    "j_per_solved_task_total", "j_per_solved_task_active", "j_per_task_total",
    "j_per_output_token_total", "avg_watts", "idle_watts", "seconds", "tasks_per_hour",
]


@app.local_entrypoint()
def main(suites: str = "repeats,batch,models", n: int = 1319, max_tokens: int = 1024):
    configs = [c for name in suites.split(",") for c in SUITES[name.strip()]]
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"

    out_dir = Path("results/sprint1") / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Launching {len(configs)} runs on {GPU}, n={n} -> {out_dir}")

    rows = []
    args = [(label, model, seqs, n, max_tokens) for label, model, seqs in configs]
    for result in measure.starmap(args, return_exceptions=True, order_outputs=False):
        if isinstance(result, Exception):
            print(f"  FAILED: {result!r}")
            continue
        cfg, s = result["config"], result["summary"]
        cfg["git_sha"] = sha
        cfg["command"] = f"GREENROUTE_GPU={GPU} modal run experiments/sprint1.py --suites {suites} --n {n}"
        (out_dir / f"{cfg['label']}.json").write_text(json.dumps(result, indent=2))
        row = {
            "label": cfg["label"], "model": cfg["model"], "max_num_seqs": cfg["max_num_seqs"],
            "n": cfg["n"], "gpu": cfg["devices"][0]["name"],
            **{k: s[k] for k in CSV_FIELDS if k in s},
        }
        rows.append(row)
        print(
            f"  {cfg['label']:<10} acc {s['accuracy']:.1%}  "
            f"J/solved {s['j_per_solved_task_total']:.1f} ({s['j_per_solved_task_active']:.1f} active)  "
            f"{s['avg_watts']:.0f} W avg  {s['seconds']:.0f} s"
        )

    rows.sort(key=lambda r: (r["model"], r["max_num_seqs"], r["label"]))
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)}/{len(configs)} results to {out_dir}")
