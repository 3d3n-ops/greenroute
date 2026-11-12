"""Run the Phase 1 experiments on the GPUs of this machine instead of Modal.

For a rented GPU server (RunPod, Lambda, Vast.ai, a university box...). It runs
the same suites, code and result format as the Modal entry points:

    python experiments/run_local.py phase1 --suites gptoss_pilot
    python experiments/run_local.py phase1 --suites core --tasks gsm8k --repeats 2
    python experiments/run_local.py power --suite gptoss_check
    python experiments/run_local.py phase1 --suites gptoss --gpus 0,1   # only these GPUs

One configuration per GPU at a time: each worker process owns one GPU
(CUDA_VISIBLE_DEVICES, with CUDA_DEVICE_ORDER=PCI_BUS_ID so the energy meter
reads the same GPU), and power sweeps on different GPUs use different ports.
Results: results/<phase1|phase1_power>/<run id>/<label>.json plus summary.csv,
with the host's GPU model, driver and power limit in each result.

Set up a fresh machine with scripts/setup_gpu_host.sh first.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import platform
import shlex
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _visible_gpus() -> list[int]:
    import pynvml

    pynvml.nvmlInit()
    return list(range(pynvml.nvmlDeviceGetCount()))


def _load(kind: str):
    """The Modal experiment module; its suites, labels and CSV rows are reused unchanged."""
    sys.path.insert(0, str(ROOT))
    if kind == "phase1":
        from experiments import phase1 as mod
    else:
        from experiments import phase1_power as mod
    return mod


def _worker(kind: str, gpu: int, port: int, jobs, results) -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["GREENROUTE_PORT"] = str(port)
    mod = _load(kind)
    fn = mod.measure if kind == "phase1" else mod.sweep
    while True:
        args = jobs.get()
        if args is None:
            return
        try:
            result = fn.local(*args)  # the Modal function's body, run in this process
            result["config"]["gpu_index"] = gpu
            result["config"]["gpu_requested"] = None  # not a Modal request; config.devices names the GPU
            results.put(("ok", args, result))
        except Exception:
            results.put(("error", args, traceback.format_exc()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=("phase1", "power"))
    ap.add_argument("--suites", default="core", help="phase1: comma-separated suite names")
    ap.add_argument("--suite", default="all", help="power: one suite name")
    ap.add_argument("--tasks", default="", help="phase1: keep only these tasks")
    ap.add_argument("--n", type=int, default=0, help="phase1: cap every configuration at n problems")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=0)
    ap.add_argument("--gpus", default="", help="GPU indices to use (default: all)")
    ap.add_argument("--base-port", type=int, default=8000)
    ap.add_argument("--dry-run", action="store_true", help="list the jobs and GPUs, run nothing")
    a = ap.parse_args()

    mod = _load(a.kind)
    from greenroute.provenance import git_sha

    reps = range(1, a.repeats + 1) if a.repeats > 0 else [0]
    if a.kind == "phase1":
        configs = [(*c, mod.COPIES.get(name.strip(), 1)) for name in a.suites.split(",") for c in mod.SUITES[name.strip()]]
        if a.tasks:
            keep = {t.strip() for t in a.tasks.split(",")}
            configs = [c for c in configs if c[1] in keep]
        if a.n > 0:
            configs = [(m, t, sp, min(a.n, k) if k else a.n, th, s, cp) for m, t, sp, k, th, s, cp in configs]
        jobs_args = [(m, t, sp, k, th, s, a.max_num_seqs, r, cp) for m, t, sp, k, th, s, cp in configs for r in reps]
        out_dir = ROOT / "results/phase1" / time.strftime("%Y%m%d-%H%M%S")
    else:
        jobs_args = [(*c, r) for c in mod.SUITES[a.suite] for r in reps]
        out_dir = ROOT / "results/phase1_power" / time.strftime("%Y%m%d-%H%M%S")

    if a.dry_run:
        gpus = [int(g) for g in a.gpus.split(",")] if a.gpus else ["all"]
        print(f"{len(jobs_args)} jobs on GPUs {gpus}, one at a time per GPU -> {out_dir.parent}/<run id>")
        for args in jobs_args:
            print("  ", args)
        return
    gpus = [int(g) for g in a.gpus.split(",")] if a.gpus else _visible_gpus()
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = git_sha(str(ROOT))
    command = "python experiments/run_local.py " + " ".join(shlex.quote(x) for x in sys.argv[1:])
    host = {"hostname": platform.node(), "provider": os.environ.get("GREENROUTE_PROVIDER")}
    print(f"Running {len(jobs_args)} jobs on GPUs {gpus} -> {out_dir}", flush=True)

    ctx = mp.get_context("spawn")
    jobs, results = ctx.Queue(), ctx.Queue()
    for args in jobs_args:
        jobs.put(args)
    workers = []
    for i, gpu in enumerate(gpus):
        jobs.put(None)
        p = ctx.Process(target=_worker, args=(a.kind, gpu, a.base_port + i, jobs, results))
        p.start()
        workers.append(p)

    rows = []
    for _ in jobs_args:
        status, args, payload = results.get()
        if status == "error":
            print(f"  FAILED {args}:\n{payload}", flush=True)
            continue
        cfg = payload["config"]
        cfg.update(git_sha=sha, command=command, host=host, runner="run_local")
        (out_dir / f"{cfg['label']}.json").write_text(json.dumps(payload))
        new = [mod.row(payload)] if a.kind == "phase1" else mod.rows(payload)
        rows.extend(new)
        for r in new:
            print("  " + "  ".join(f"{k}={r[k]}" for k in list(r)[:8]), flush=True)
    for p in workers:
        p.join()

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mod.CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows from {len(jobs_args)} jobs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
