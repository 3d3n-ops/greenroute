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

GPU = os.environ.get("GREENROUTE_GPU", "A100-80GB")
VLLM_VERSION = "0.10.1.1"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}", "datasets", "nvidia-ml-py")
    .add_local_python_source("greenroute")
)
hf_cache = modal.Volume.from_name("greenroute-hf-cache", create_if_missing=True)
app = modal.App("greenroute-sprint0")


@app.function(gpu=GPU, image=image, volumes={"/root/.cache/huggingface": hf_cache}, timeout=3600)
def measure(model: str, n: int, max_num_seqs: int, max_tokens: int, seed: int, idle_seconds: float) -> dict:
    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    from greenroute import gsm8k
    from greenroute.energy import EnergyMeter
    from greenroute.report import summarize

    meter = EnergyMeter()
    problems = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    messages = [gsm8k.build_messages(p["question"]) for p in problems]

    llm = LLM(model=model, max_num_seqs=max_num_seqs, max_model_len=4096, seed=seed)
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    chat_kwargs = {"enable_thinking": False}  # Qwen3: measure non-thinking mode first

    # Warm up kernels and caches so they don't land in the measured span.
    llm.chat(messages[:8], params, chat_template_kwargs=chat_kwargs, use_tqdm=False)

    # Idle baseline with the model loaded: what an unused server draws.
    idle_watts = meter.idle_watts(idle_seconds)

    with meter.span() as span:
        outputs = llm.chat(messages, params, chat_template_kwargs=chat_kwargs, use_tqdm=False)

    records = []
    for i, (p, out) in enumerate(zip(problems, outputs)):
        text = out.outputs[0].text
        records.append(
            {
                "index": i,
                "correct": gsm8k.is_correct(text, p["answer"]),
                "predicted": gsm8k.extract_answer(text),
                "gold": gsm8k.gold_answer(p["answer"]),
                "output_tokens": len(out.outputs[0].token_ids),
            }
        )

    summary = summarize(
        span.reading,
        idle_watts,
        n_tasks=len(records),
        n_correct=sum(r["correct"] for r in records),
        output_tokens=sum(r["output_tokens"] for r in records),
    )
    return {
        "config": {
            "model": model,
            "task": "gsm8k/test",
            "n": n,
            "max_num_seqs": max_num_seqs,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": seed,
            "thinking": False,
            "vllm": VLLM_VERSION,
            "gpu_requested": GPU,
            "devices": meter.device_info(),
            "energy_source": "nvmlDeviceGetTotalEnergyConsumption",
        },
        "summary": summary,
        "per_gpu_joules": span.reading.per_gpu_joules,
        "records": records,
    }


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
