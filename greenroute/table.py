"""Phase 1: the per-problem table for one configuration (model x thinking mode x task).

One offline vLLM run on the local GPU, measured with the energy counter. Each
problem gets its correctness, token counts, whether it was cut off, the model's
token-probability confidence in its answer, and optionally k extra sampled
answers (for self-consistency routing signals), whose energy is measured
separately so the greedy number stays comparable.
"""

from __future__ import annotations

import math
import statistics

THINK_MAX_MODEL_LEN = 16384
THINK_MAX_TOKENS = 15000
FAST_MAX_MODEL_LEN = 4096
TAIL_TOKENS = 16  # the end of the answer, where the final answer line or code lives


def think_end_id(tokenizer) -> int | None:
    """Token id of '</think>' if the tokenizer has it as a single token, else None."""
    try:
        tid = tokenizer.convert_tokens_to_ids("</think>")
    except Exception:
        return None
    if not isinstance(tid, int) or tid < 0 or tid == getattr(tokenizer, "unk_token_id", None):
        return None
    return tid


def reasoning_tokens(token_ids: list[int], end_id: int | None) -> int:
    """Tokens up to and including '</think>'; 0 when it never appears.

    A reply whose reasoning never closed is all reasoning; the caller handles that
    case from the decoded text, which shows the unclosed <think>.
    """
    ids = list(token_ids)
    return ids.index(end_id) + 1 if end_id is not None and end_id in ids else 0


def confidence(token_ids: list[int], logprobs, n_reasoning: int) -> dict:
    """Summaries of the model's log-probabilities over its answer (reasoning excluded).

    mean_lp: average over answer tokens. min_lp: the least sure token.
    tail_mean_lp: average over the last TAIL_TOKENS answer tokens.
    """
    if not logprobs:
        return {}
    lps = []
    for tid, pos in zip(token_ids, logprobs):
        lp = pos[tid].logprob if pos and tid in pos else None
        if lp is not None and math.isfinite(lp):
            lps.append(lp)
    ans = lps[n_reasoning:]
    if not ans:
        return {}
    tail = ans[-TAIL_TOKENS:]
    return {
        "mean_lp": round(statistics.fmean(ans), 5),
        "min_lp": round(min(ans), 5),
        "tail_mean_lp": round(statistics.fmean(tail), 5),
    }


def sampling_kwargs(thinking: bool) -> dict:
    """Qwen3's recommended sampling settings for each mode."""
    return {"temperature": 0.6, "top_p": 0.95} if thinking else {"temperature": 0.7, "top_p": 0.8}


def run_table(
    model: str,
    task_name: str,
    split: str = "test",
    n: int | None = None,
    offset: int = 0,
    thinking: bool = False,
    max_num_seqs: int = 256,
    max_tokens: int | None = None,
    samples: int = 0,
    seed: int = 0,
    idle_seconds: float = 10.0,
    copies: int = 1,
) -> dict:
    """copies > 1 runs the problem set that many times in one batch, so a small task
    (HumanEval's 164 problems against 256 slots) fills the GPU and has a steady state.
    Every copy is scored and kept as a record with its `copy` number. Identical prompts
    share vLLM's prefix cache, which makes their prefill slightly cheaper than unique
    prompts would be; decode, most of the energy, is unaffected.
    """
    from vllm import LLM, SamplingParams

    from greenroute.bench import VLLM_VERSION
    from greenroute.energy import EnergyMeter
    from greenroute.report import summarize
    from greenroute.steady import downsample, run_engine, steady_window
    from greenroute.tasks import TASKS, load_problems, split_thinking

    task = TASKS[task_name]
    unique = load_problems(task, split, n, offset)
    problems = unique * copies  # copy 0 of every problem, then copy 1, ...: copies mix in the batch
    messages = [task.build_messages(p) for p in problems]
    if max_tokens is None:
        max_tokens = THINK_MAX_TOKENS if thinking else task.max_tokens
    max_model_len = THINK_MAX_MODEL_LEN if thinking else FAST_MAX_MODEL_LEN

    meter = EnergyMeter()
    llm = LLM(model=model, max_num_seqs=max_num_seqs, max_model_len=max_model_len, seed=seed)
    tokenizer = llm.get_tokenizer()
    end_id = think_end_id(tokenizer)
    chat_kwargs = {"enable_thinking": thinking}  # ignored by templates that have no thinking mode
    # Rendered and tokenized here, as llm.chat would, so the engine can be driven step by step.
    prompt_ids = [
        tokenizer(tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **chat_kwargs),
                  add_special_tokens=False).input_ids
        for m in messages
    ]

    greedy = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=1)
    llm.chat(messages[:8], SamplingParams(temperature=0.0, max_tokens=64), chat_template_kwargs=chat_kwargs,
             use_tqdm=False)  # warm-up, not measured
    idle_watts = meter.idle_watts(idle_seconds)

    with meter.span() as span:
        outputs, timeline = run_engine(llm, prompt_ids, greedy, meter)
    steady = steady_window(timeline, max_num_seqs)

    def parse(o) -> tuple[str | None, int]:
        _, answer = split_thinking(o.text)
        n_reason = reasoning_tokens(o.token_ids, end_id)
        if answer is None:  # reasoning never closed: every token was reasoning
            n_reason = len(o.token_ids)
        return answer, n_reason

    parsed = [parse(out.outputs[0]) for out in outputs]
    scores = task.score_batch([a for a, _ in parsed], problems)

    records = []
    for i, (p, out, (answer, n_reason), sc) in enumerate(zip(problems, outputs, parsed, scores)):
        o = out.outputs[0]
        records.append({
            "index": offset + i % len(unique),
            "id": p.get("task_id", offset + i % len(unique)),
            "copy": i // len(unique),
            **sc,
            "prompt_tokens": len(out.prompt_token_ids or []),
            "output_tokens": len(o.token_ids),
            "reasoning_tokens": n_reason,
            "finish_reason": o.finish_reason,
            "no_answer": answer is None,
            "confidence": confidence(o.token_ids, o.logprobs, n_reason),
        })

    summary = summarize(
        span.reading, idle_watts,
        n_tasks=len(records),
        n_correct=sum(r["correct"] for r in records),
        output_tokens=sum(r["output_tokens"] for r in records),
    )
    n_correct = summary["n_correct"]
    summary.update({
        "n_truncated": sum(r["finish_reason"] == "length" for r in records),
        "n_no_answer": sum(r["no_answer"] for r in records),
        "mean_output_tokens": statistics.fmean(r["output_tokens"] for r in records),
        "mean_reasoning_tokens": statistics.fmean(r["reasoning_tokens"] for r in records),
        "tokens_per_solved_task": summary["output_tokens"] / n_correct if n_correct else None,
    })
    if steady and steady["j_per_output_token"]:
        # Saturated cost per task: steady-state energy per token times this workload's tokens per task.
        jpt = steady["j_per_output_token"] * summary["mean_output_tokens"]
        steady["j_per_task"] = jpt
        steady["j_per_solved_task"] = jpt / summary["accuracy"] if summary["accuracy"] else None
    summary["steady"] = steady
    keep_rows = tuple(i for i in ((steady or {}).get("end_index"), (steady or {}).get("turnover_start_index"))
                      if i is not None)

    sample_info = None
    if samples > 0:
        # One request per sampled answer, each with its own seed, driven step by step like the
        # greedy pass so the samples get a steady-state energy too. Request j*N+i is sample j of problem i.
        N = len(problems)
        flat_ids = prompt_ids * samples
        flat_params = [SamplingParams(max_tokens=max_tokens, seed=seed + 1 + j, **sampling_kwargs(thinking))
                       for j in range(samples) for _ in range(N)]
        with meter.span() as s_span:
            s_flat, s_timeline = run_engine(llm, flat_ids, flat_params, meter)
        s_steady = steady_window(s_timeline, max_num_seqs)
        flat_answers = [split_thinking(out.outputs[0].text)[1] for out in s_flat]
        flat_scores = task.score_batch(flat_answers, problems * samples)
        for i, r in enumerate(records):
            r["samples"] = []
            for j in range(samples):
                k = j * N + i
                r["samples"].append({"correct": flat_scores[k]["correct"], "predicted": flat_scores[k].get("predicted"),
                                     "output_tokens": len(s_flat[k].outputs[0].token_ids)})
        sample_tokens = sum(s["output_tokens"] for r in records for s in r["samples"])
        if s_steady and s_steady["j_per_output_token"]:
            # Saturated cost of the samples, per problem: steady J/token times sample tokens per problem.
            s_steady["j_per_task"] = s_steady["j_per_output_token"] * sample_tokens / N
        sample_info = {
            "k": samples,
            **sampling_kwargs(thinking),
            "seeds": f"{seed + 1}..{seed + samples}, one per sample index",
            "steady": s_steady,
            "timeline": downsample(s_timeline),
            "energy_j_total": s_span.reading.joules,
            "energy_j_active": s_span.reading.active_joules(idle_watts),
            "seconds": s_span.reading.seconds,
            "output_tokens": sample_tokens,
            "j_per_output_token_total": s_span.reading.joules / sample_tokens if sample_tokens else None,
            "pass_rate": statistics.fmean(s["correct"] for r in records for s in r["samples"]),
        }

    return {
        "config": {
            "model": model,
            "task": f"{task_name}/{split}",
            "offset": offset,
            "n": len(records),
            "n_unique": len(unique),
            "copies": copies,
            "thinking": thinking,
            "max_num_seqs": max_num_seqs,
            "max_tokens": max_tokens,
            "max_model_len": max_model_len,
            "temperature": 0.0,
            "logprobs": 1,
            "seed": seed,
            "vllm": VLLM_VERSION,
            "devices": meter.device_info(),
            "energy_source": "nvmlDeviceGetTotalEnergyConsumption",
        },
        "summary": summary,
        "samples": sample_info,
        "per_gpu_joules": span.reading.per_gpu_joules,
        # (seconds, joules, unfinished requests, output tokens so far), downsampled; window end kept.
        "timeline": downsample(timeline, keep=keep_rows),
        "records": records,
    }
