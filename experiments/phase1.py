"""Phase 1: per-problem table for open models (model x thinking mode x task).

    modal run experiments/phase1.py --suites pilot     # small, cheap check of every code path
    modal run experiments/phase1.py                     # core: Qwen3 sizes x thinking x GSM8K + HumanEval
    modal run experiments/phase1.py --suites train      # GSM8K train slice, for tuning router thresholds
    modal run experiments/phase1.py --suites other      # models from other families
    modal run experiments/phase1.py --tasks gsm8k --repeats 2   # repeat runs, labelled _r1, _r2
    modal run experiments/phase1.py --suites saturate   # HumanEval x8 copies, so it fills the GPU
    modal run experiments/phase1.py --suites samples    # GSM8K fast + 4 samples, samples measured at steady state

Each configuration runs offline at high concurrency (the saturated reference load)
in its own container. Results: results/phase1/<run id>/<label>.json plus summary.csv.
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
QWEN = ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B")
OTHER = ("microsoft/Phi-4-mini-instruct", "allenai/OLMo-2-1124-7B-Instruct")
SAMPLES_FAST = 4  # sampled answers per problem with thinking off; 0 with thinking on (cost)

# (model, task, split, n or None for all, thinking, samples)
SUITES = {
    "pilot": [
        ("Qwen/Qwen3-1.7B", "gsm8k", "test", 64, False, 2),
        ("Qwen/Qwen3-1.7B", "gsm8k", "test", 64, True, 0),
        ("Qwen/Qwen3-1.7B", "humaneval", "test", 32, False, 2),
        ("Qwen/Qwen3-1.7B", "humaneval", "test", 32, True, 0),
    ],
    # Big enough to fill 256 slots, so the steady-state window exists; checks answers match core runs.
    "check": [
        ("Qwen/Qwen3-1.7B", "gsm8k", "test", None, False, 0),
        ("Qwen/Qwen3-1.7B", "gsm8k", "test", 600, True, 0),
    ],
    "core": [
        (m, task, "test", None, think, 0 if think else SAMPLES_FAST)
        for m in QWEN for think in (False, True) for task in ("gsm8k", "humaneval")
    ],
    "train": [
        (m, "gsm8k", "train", 1000, think, 0 if think else SAMPLES_FAST)
        for m in QWEN for think in (False, True)
    ],
    "other": [(m, task, "test", None, False, 0) for m in OTHER for task in ("gsm8k", "humaneval")],
    # HumanEval's 164 problems cannot fill 256 slots; 8 copies (1,312 requests) give a steady state.
    "saturate": [(m, "humaneval", "test", None, think, 0) for m in QWEN for think in (False, True)],
    # Sampled answers driven step by step, so self-consistency gets a steady-state energy.
    "samples": [(m, "gsm8k", "test", None, False, SAMPLES_FAST) for m in QWEN],
}
COPIES = {"saturate": 8}  # problem-set copies per suite; 1 elsewhere

hf_cache = modal.Volume.from_name("greenroute-hf-cache", create_if_missing=True)
app = modal.App("greenroute-phase1")


def label(model: str, task: str, split: str, thinking: bool, copies: int = 1) -> str:
    return f"{model.split('/')[-1]}_{'think' if thinking else 'fast'}_{task}-{split}" + (f"_x{copies}" if copies > 1 else "")


@app.function(
    gpu=GPU,
    image=gpu_image(),
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=7200,
    cpu=8.0,  # HumanEval tests run in parallel subprocesses after the measured span
    max_containers=int(os.environ.get("GREENROUTE_MAX_GPUS", "4")),
)
def measure(model: str, task: str, split: str, n: int | None, thinking: bool, samples: int,
            max_num_seqs: int, rep: int = 0, copies: int = 1) -> dict:
    from greenroute.table import run_table

    result = run_table(model, task, split=split, n=n, thinking=thinking, samples=samples,
                       max_num_seqs=max_num_seqs, copies=copies)
    result["config"]["label"] = label(model, task, split, thinking, copies) + (f"_r{rep}" if rep else "")
    result["config"]["repeat"] = rep
    result["config"]["gpu_requested"] = GPU
    return result


CSV_FIELDS = [
    "label", "model", "task", "thinking", "n", "accuracy", "n_truncated", "n_no_answer",
    "mean_output_tokens", "mean_reasoning_tokens", "tokens_per_solved_task",
    "j_per_task_total", "j_per_solved_task_total", "j_per_solved_task_active", "j_per_output_token_total",
    "steady_basis", "steady_j_per_output_token", "steady_j_per_task", "steady_j_per_solved_task",
    "steady_share_of_tokens", "copies",
    "avg_watts", "idle_watts", "seconds", "sample_k", "sample_pass_rate", "sample_j_total", "sample_steady_j_per_task", "gpu",
]


def row(result: dict) -> dict:
    cfg, s, smp = result["config"], result["summary"], result.get("samples") or {}
    return {
        "label": cfg["label"], "model": cfg["model"], "task": cfg["task"], "thinking": cfg["thinking"],
        "n": cfg["n"], "gpu": cfg["devices"][0]["name"], "copies": cfg.get("copies", 1),
        **{k: s.get(k) for k in CSV_FIELDS if k in s},
        "steady_basis": (s.get("steady") or {}).get("basis"),
        "steady_j_per_output_token": (s.get("steady") or {}).get("j_per_output_token"),
        "steady_j_per_task": (s.get("steady") or {}).get("j_per_task"),
        "steady_j_per_solved_task": (s.get("steady") or {}).get("j_per_solved_task"),
        "steady_share_of_tokens": (s.get("steady") or {}).get("share_of_tokens"),
        "sample_k": smp.get("k"), "sample_pass_rate": smp.get("pass_rate"), "sample_j_total": smp.get("energy_j_total"),
        "sample_steady_j_per_task": (smp.get("steady") or {}).get("j_per_task"),
    }


@app.local_entrypoint()
def main(suites: str = "core", n: int = 0, max_num_seqs: int = 256, tasks: str = "", repeats: int = 0):
    """n > 0 caps every configuration at n problems (for quick checks).

    tasks: comma-separated task names to keep (default: all in the suites).
    repeats: run each configuration this many times, labelled _r1.._rN, instead of once unlabelled.
    """
    configs = [(*c, COPIES.get(name.strip(), 1)) for name in suites.split(",") for c in SUITES[name.strip()]]
    if tasks:
        keep = {t.strip() for t in tasks.split(",")}
        configs = [c for c in configs if c[1] in keep]
    if n > 0:
        configs = [(m, t, sp, min(n, k) if k else n, th, s, cp) for m, t, sp, k, th, s, cp in configs]
    sha = git_sha()
    out_dir = Path("results/phase1") / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Launching {len(configs)} configurations x {len(range(1, repeats + 1)) or 1} on {GPU} -> {out_dir}",
          flush=True)

    rows = []
    reps = range(1, repeats + 1) if repeats > 0 else [0]
    args = [(m, t, sp, k, th, s, max_num_seqs, r, cp) for m, t, sp, k, th, s, cp in configs for r in reps]
    for result in measure.starmap(args, return_exceptions=True, order_outputs=False):
        if isinstance(result, Exception):
            print(f"  FAILED: {result!r}", flush=True)
            continue
        cfg, s = result["config"], result["summary"]
        cfg["git_sha"] = sha
        cfg["command"] = (f"GREENROUTE_GPU={GPU} modal run experiments/phase1.py --suites {suites} --n {n} "
                          f"--tasks {tasks or 'all'} --repeats {repeats}")
        (out_dir / f"{cfg['label']}.json").write_text(json.dumps(result))
        rows.append(row(result))
        print(
            f"  {cfg['label']:<40} acc {s['accuracy']:.1%}  tok/task {s['mean_output_tokens']:6.0f}  "
            f"J/solved {s['j_per_solved_task_total'] or float('nan'):7.1f}  "
            f"steady J/task {(s.get('steady') or {}).get('j_per_task') or float('nan'):7.1f}  "
            f"trunc {s['n_truncated']}  {s['seconds']:.0f} s",
            flush=True,
        )

    rows.sort(key=lambda r: (r["task"], r["model"], r["thinking"]))
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)}/{len(args)} results to {out_dir}", flush=True)
