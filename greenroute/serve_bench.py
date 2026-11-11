"""Latency vs energy: a real vLLM server under open-loop traffic at several request rates.

Runs inside one GPU container. The model loads once; each request rate is a
"level" measured back to back on the same GPU, so levels are directly comparable.
"""

from __future__ import annotations

import asyncio
import statistics
import subprocess
import time
import urllib.request

from greenroute.bench import VLLM_VERSION

PORT = 8000
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


def run_latency_sweep(
    model: str,
    rates: list[float],
    max_num_seqs: int = 256,
    max_tokens: int = 1024,
    duration_s: float = 90.0,
    n_min: int = 100,
    n_max: int = 600,
    idle_seconds: float = 10.0,
    settle_seconds: float = 5.0,
    seed: int = 0,
) -> dict:
    from datasets import load_dataset

    from greenroute import gsm8k
    from greenroute.energy import EnergyMeter, TelemetrySampler
    from greenroute.loadgen import percentiles, run_level, summarize_metrics
    from greenroute.report import summarize

    meter = EnergyMeter()
    log_path = "/tmp/vllm_server.log"
    cmd = ["vllm", "serve", model, "--port", str(PORT), "--max-num-seqs", str(max_num_seqs),
           "--max-model-len", "4096", "--seed", str(seed)]
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
    try:
        test = load_dataset("openai/gsm8k", "main", split="test")

        # Warm up kernels, CUDA graphs and the HTTP path; not measured.
        warm = [gsm8k.build_messages(test[i]["question"]) for i in range(16)]
        asyncio.run(run_level(BASE_URL, model, warm, rate=0, max_tokens=max_tokens))
        idle_watts = meter.idle_watts(idle_seconds)

        levels = []
        for rate in rates:
            n = requests_for_rate(rate, duration_s, n_min, n_max)
            problems = test.select(range(n))
            messages = [gsm8k.build_messages(p["question"]) for p in problems]

            with TelemetrySampler(meter, interval=0.1) as tel, meter.span() as span:
                out = asyncio.run(run_level(BASE_URL, model, messages, rate=rate, seed=seed, max_tokens=max_tokens))

            recs = out["records"]
            for r, p in zip(recs, problems):
                r["correct"] = r["error"] is None and gsm8k.is_correct(r["text"], p["answer"])
                r["predicted"] = gsm8k.extract_answer(r["text"])
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
                "achieved_rps": len(recs) / out["wall_s"] if out["wall_s"] else None,
                "offered_rps": rate if rate > 0 else None,
                "mean_output_tokens": statistics.fmean(r["output_tokens"] for r in ok) if ok else None,
            }
            levels.append({
                "rate": rate,
                "label": f"{rate:g} rps" if rate > 0 else "burst",
                "n": n,
                "energy": energy,
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
                "max_tokens": max_tokens,
                "duration_s": duration_s,
                "n_min": n_min,
                "n_max": n_max,
                "temperature": 0.0,
                "seed": seed,
                "thinking": False,
                "vllm": VLLM_VERSION,
                "mode": "openai-server, open-loop Poisson arrivals, streaming",
                "devices": meter.device_info(),
                "energy_source": "nvmlDeviceGetTotalEnergyConsumption",
                "idle_watts": idle_watts,
            },
            "levels": levels,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
