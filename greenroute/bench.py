"""One measured GSM8K run on the local GPU. Called inside a Modal container."""

from __future__ import annotations

VLLM_VERSION = "0.10.1.1"


def run_gsm8k(
    model: str,
    n: int,
    max_num_seqs: int,
    max_tokens: int = 1024,
    seed: int = 0,
    idle_seconds: float = 10.0,
) -> dict:
    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    from greenroute import gsm8k
    from greenroute.energy import EnergyMeter
    from greenroute.report import summarize

    meter = EnergyMeter()
    test = load_dataset("openai/gsm8k", "main", split="test")
    problems = test.select(range(min(n, len(test))))
    messages = [gsm8k.build_messages(p["question"]) for p in problems]

    llm = LLM(model=model, max_num_seqs=max_num_seqs, max_model_len=4096, seed=seed)
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    chat_kwargs = {"enable_thinking": False}  # Qwen3: non-thinking mode

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
            "n": len(records),
            "max_num_seqs": max_num_seqs,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": seed,
            "thinking": False,
            "vllm": VLLM_VERSION,
            "devices": meter.device_info(),
            "energy_source": "nvmlDeviceGetTotalEnergyConsumption",
        },
        "summary": summary,
        "per_gpu_joules": span.reading.per_gpu_joules,
        "records": records,
    }
