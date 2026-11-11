"""Sprint 0: first measured J/task number.

Serves one small model with vLLM on a Modal GPU, runs GSM8K problems, and
measures GPU energy with NVML's cumulative counter.

    modal run experiments/sprint0.py
    modal run experiments/sprint0.py --model Qwen/Qwen3-4B --n 200 --max-num-seqs 64

GPU type is fixed at deploy time; override with GREENROUTE_GPU (e.g. H100, L4).
Results are written to results/sprint0/.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

from greenroute.modal_image import gpu_image

GPU = os.environ.get("GREENROUTE_GPU", "A100-80GB")

hf_cache = modal.Volume.from_name("greenroute-hf-cache", create_if_missing=True)
app = modal.App("greenroute-sprint0")


@app.function(gpu=GPU, image=gpu_image(), volumes={"/root/.cache/huggingface": hf_cache}, timeout=3600)
def measure(model: str, n: int, max_num_seqs: int, max_tokens: int, seed: int, idle_seconds: float) -> dict:
    from greenroute.bench import run_gsm8k

    result = run_gsm8k(model, n, max_num_seqs, max_tokens, seed, idle_seconds)
    result["config"]["gpu_requested"] = GPU
    return result


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3-1.7B",
    n: int = 100,
    max_num_seqs: int = 32,
    max_tokens: int = 1024,
    seed: int = 0,
    idle_seconds: float = 10.0,
):
    result = measure.remote(model, n, max_num_seqs, max_tokens, seed, idle_seconds)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"
    result["config"]["git_sha"] = sha
    result["config"]["command"] = (
        f"GREENROUTE_GPU={GPU} modal run experiments/sprint0.py --model {model} --n {n} "
        f"--max-num-seqs {max_num_seqs} --max-tokens {max_tokens} --seed {seed}"
    )

    out_dir = Path("results/sprint0")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{stamp}_{model.split('/')[-1]}_{GPU}_seqs{max_num_seqs}.json"
    path.write_text(json.dumps(result, indent=2))

    s = result["summary"]
    print(f"\n{model} on {result['config']['devices'][0]['name']}")
    print(f"  accuracy        {s['accuracy']:.1%}  ({s['n_correct']}/{s['n_tasks']})")
    print(f"  energy          {s['energy_j_total']:.0f} J total, {s['energy_j_active']:.0f} J above idle ({s['idle_watts']:.0f} W idle)")
    if s["j_per_solved_task_total"] is not None:
        print(f"  J/solved task   {s['j_per_solved_task_total']:.1f} total, {s['j_per_solved_task_active']:.1f} active")
    print(f"  J/output token  {s['j_per_output_token_total']:.3f}")
    print(f"  saved to {path}")
