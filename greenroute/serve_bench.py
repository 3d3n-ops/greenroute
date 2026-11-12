"""Latency vs energy: a real vLLM server under open-loop traffic at several request rates.

Runs inside one GPU container. The model loads once; each request rate is a
"level" measured back to back on the same GPU, so levels are directly comparable.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import subprocess
import time
import urllib.request

from greenroute.bench import VLLM_VERSION

PORT = int(os.environ.get("GREENROUTE_PORT", "8000"))  # one per GPU when sweeps share a machine
BASE_URL = f"http://127.0.0.1:{PORT}"


def _wait_healthy(proc: subprocess.Popen, log_path: str, timeout_s: float = 900) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM server exited early:\n{_tail(log_path)}")
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM server not healthy after {timeout_s}s:\n{_tail(log_path)}")


def _tail(path: str, n: int = 25) -> str:
    """Error-looking lines from the whole log, then the last n lines."""
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError:
        return "(no log)"
    keys = ("Error", "error:", "Exception", "CUDA", "unrecognized", "invalid")
    errors = [ln for ln in lines if any(k in ln for k in keys) and "Traceback" not in ln][:30]
    return "--- error lines ---\n" + "".join(errors) + "--- last lines ---\n" + "".join(lines[-n:])


def requests_for_rate(rate: float, duration_s: float, n_min: int, n_max: int) -> int:
    """Enough requests to hold the rate for ~duration_s, within [n_min, n_max]. Burst uses n_max."""
    if rate <= 0:
        return n_max
    return max(n_min, min(n_max, round(rate * duration_s)))


def common_accuracy(records: list[dict], n_common: int) -> float | None:
    """Accuracy on the first n_common problems, which every level sends.

    Levels send different numbers of problems, so their overall accuracy is over
    different subsets and is not comparable across rates. This one is.
    """
    head = [r for r in records if r["index"] < n_common]
    return sum(bool(r["correct"]) for r in head) / len(head) if head else None


def visible_cpus() -> int | None:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not on Linux
        return os.cpu_count()


def run_latency_sweep(
    model: str,
    rates: list[float],
    max_num_seqs: int = 256,
    max_tokens: int | None = None,
    duration_s: float = 90.0,
    n_min: int = 100,
    n_max: int = 600,
    idle_seconds: float = 10.0,
    settle_seconds: float = 5.0,
    seed: int = 0,
    client_workers: int = 8,
    thinking: bool = False,
    effort: str | None = None,
    tensor_parallel_size: int = 1,
) -> dict:
    """Sweep request rates on one server. Thinking mode gets the Phase 1 token budget.

    effort: gpt-oss reasoning effort (low / medium / high); gpt-oss always reasons.

    Levels needing more requests than the test set has reuse problems from the start.
    """
    from datasets import load_dataset

    from greenroute import gsm8k
    from greenroute.energy import EnergyMeter, TelemetrySampler
    from greenroute.modes import PRIMARY_SEED_BASE, is_harmony, primary_sampling, request_extra_body
    from greenroute.loadgen import make_pool, percentiles, run_level, send_lag, summarize_metrics
    from greenroute.power import steady_level
    from greenroute.report import summarize
    from greenroute.table import FAST_MAX_MODEL_LEN, THINK_MAX_MODEL_LEN, THINK_MAX_TOKENS
    from greenroute.tasks import split_thinking

    if is_harmony(model):
        thinking = True
    extra_body = request_extra_body(model, thinking, effort)
    sampling = primary_sampling(model)
    if sampling["temperature"] > 0:
        sampling["seed"] = seed * PRIMARY_SEED_BASE + PRIMARY_SEED_BASE  # request i gets this + i
    if max_tokens is None:
        max_tokens = THINK_MAX_TOKENS if thinking else 1024
    max_model_len = THINK_MAX_MODEL_LEN if thinking else FAST_MAX_MODEL_LEN
    meter = EnergyMeter()
    log_path = "/tmp/vllm_server.log"
    cmd = ["vllm", "serve", model, "--port", str(PORT), "--max-num-seqs", str(max_num_seqs),
           "--max-model-len", str(max_model_len), "--seed", str(seed),
           "--tensor-parallel-size", str(tensor_parallel_size)]
    # Engine start-up failed transiently once on 2026-11-11 (first smoke test); retry once.
    for attempt in (1, 2):
        with open(log_path, "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            _wait_healthy(proc, log_path)
            break
        except RuntimeError:
            if attempt == 2:
                raise
            proc.wait(timeout=30)
    pool = None
    try:
        pool = make_pool(client_workers)  # started before any measurement
        test = load_dataset("openai/gsm8k", "main", split="test")

        # Warm up kernels, CUDA graphs and the HTTP path; not measured.
        warm = [gsm8k.build_messages(test[i]["question"]) for i in range(16)]
        asyncio.run(run_level(BASE_URL, model, warm, rate=0, pool=pool, workers=client_workers,
                              max_tokens=min(max_tokens, 256), extra_body=extra_body,
                              sampling=sampling))
        idle_watts = meter.idle_watts(idle_seconds)

        levels = []
        for rate in rates:
            n = requests_for_rate(rate, duration_s, n_min, n_max)
            problems = [test[i % len(test)] for i in range(n)]
            messages = [gsm8k.build_messages(p["question"]) for p in problems]

            with TelemetrySampler(meter, interval=0.1) as tel, meter.span() as span:
                out = asyncio.run(run_level(BASE_URL, model, messages, rate=rate, pool=pool, workers=client_workers,
                                            seed=seed, max_tokens=max_tokens, extra_body=extra_body,
                                            sampling=sampling))

            recs = out["records"]
            for r, p in zip(recs, problems):
                answer = split_thinking(r["text"])[1]  # None if reasoning never closed
                r["predicted"] = gsm8k.extract_answer(answer) if answer else None
                r["correct"] = r["error"] is None and r["predicted"] == gsm8k.gold_answer(p["answer"])
                del r["text"]  # keep result files small; answers are recoverable by rerunning
            ok = [r for r in recs if r["error"] is None]

            energy = summarize(
                span.reading, idle_watts,
                n_tasks=len(recs),
                n_correct=sum(r["correct"] for r in recs),
                output_tokens=sum(r["output_tokens"] for r in recs),
            )
            latency = {
                "ttft_s": percentiles([r["ttft_s"] for r in ok if r["ttft_s"] is not None]),
                "tpot_s": percentiles([r["tpot_s"] for r in ok if r["tpot_s"] is not None]),
                "e2e_s": percentiles([r["e2e_s"] for r in ok]),
                "n_errors": len(recs) - len(ok),
                # Both diagnose the sender: lag near 0 and CPU well under 1.0 mean it kept up.
                "send_lag_s": percentiles(send_lag(recs)),
                "client_cpu_frac_max": out["client_cpu_frac_max"],
                "achieved_rps": len(recs) / out["wall_s"] if out["wall_s"] else None,
                "offered_rps": rate if rate > 0 else None,
                "mean_output_tokens": statistics.fmean(r["output_tokens"] for r in ok) if ok else None,
            }
            levels.append({
                "rate": rate,
                "label": f"{rate:g} rps" if rate > 0 else "burst",
                "n": n,
                "n_common": min(n_min, n),
                "accuracy_common": common_accuracy(recs, min(n_min, n)),
                "energy": energy,
                "steady": steady_level(recs, tel.samples, out["t0"]),
                "latency": latency,
                "gpu": tel.summary(),
                "vllm": summarize_metrics(out["metrics"]),
                "gpu_timeseries": tel.samples,
                "vllm_timeseries": out["metrics"],
                "records": recs,
            })
            time.sleep(settle_seconds)

        return {
            "config": {
                "model": model,
                "task": "gsm8k/test",
                "rates": rates,
                "max_num_seqs": max_num_seqs,
                "tensor_parallel_size": tensor_parallel_size,
                "max_tokens": max_tokens,
                "duration_s": duration_s,
                "n_min": n_min,
                "n_max": n_max,
                "sampling": sampling,
                "temperature": sampling["temperature"],
                "seed": seed,
                "thinking": thinking,
                "effort": effort,
                "request_extra_body": extra_body,
                "max_model_len": max_model_len,
                "vllm": VLLM_VERSION,
                "mode": "openai-server, open-loop Poisson arrivals, streaming",
                "client_workers": client_workers,
                "cpus_visible": visible_cpus(),
                "devices": meter.device_info(),
                "energy_source": "nvmlDeviceGetTotalEnergyConsumption",
                "idle_watts": idle_watts,
            },
            "levels": levels,
        }
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
