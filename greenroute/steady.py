"""Steady-state energy for offline batch runs.

An offline run ends with a drain-out tail: the last few long answers keep
generating while most of the GPU's batch slots sit empty. A live server under
continuous traffic would fill those slots, so whole-run energy overstates the
saturated cost, most for workloads with long, uneven answers (thinking mode).

`run_engine` drives the vLLM engine step by step and records, after every step,
the time, the GPU energy counter, how many requests are unfinished, and how many
output tokens exist so far. `steady_window` then measures only the backlogged
part of the run: while unfinished requests are at least the concurrency cap,
there is always more work queued, so the engine is as full as it can get.

The backlogged window still starts with every running request young (short
contexts, cheap attention). A live server's batch holds requests of all ages.
So the window is also reported from the point where at least one cap's worth of
requests has finished and been replaced ("turnover"), which is closer to that
mix; it is the basis for the steady number when it holds enough of the run.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

MIN_TURNOVER_TOKEN_SHARE = 0.1  # turnover window must hold this share of output tokens to be the basis


def run_engine(llm, prompt_token_ids: list[list[int]], params, meter) -> tuple[list, list[tuple]]:
    """Generate for every prompt; return (outputs in prompt order, timeline).

    Outputs mimic vLLM's RequestOutput for the fields the table uses: .prompt_token_ids
    and .outputs[0] with .text, .token_ids, .logprobs and .finish_reason.
    Timeline rows are (seconds, joules, unfinished requests, output tokens so far).
    """
    from vllm.sampling_params import RequestOutputKind

    params = params.clone()
    params.output_kind = RequestOutputKind.DELTA  # new tokens only: cumulative copies cost CPU on long answers
    engine = llm.llm_engine
    for i, ids in enumerate(prompt_token_ids):
        engine.add_request(str(i), {"prompt_token_ids": ids}, params)

    acc = {str(i): {"text": [], "token_ids": [], "logprobs": [], "finish_reason": None}
           for i in range(len(prompt_token_ids))}
    tokens = 0
    t0, e0 = time.perf_counter(), meter.read_mj()
    timeline = [(0.0, 0.0, len(prompt_token_ids), 0)]
    while engine.has_unfinished_requests():
        for out in engine.step():
            o = out.outputs[0]
            a = acc[out.request_id]
            a["text"].append(o.text)
            a["token_ids"].extend(o.token_ids)
            if o.logprobs:
                a["logprobs"].extend(o.logprobs)
            tokens += len(o.token_ids)
            if out.finished:
                a["finish_reason"] = o.finish_reason
        timeline.append((time.perf_counter() - t0, (meter.read_mj() - e0) / 1000.0,
                         engine.get_num_unfinished_requests(), tokens))

    outputs = []
    for i, ids in enumerate(prompt_token_ids):
        a = acc[str(i)]
        o = SimpleNamespace(text="".join(a["text"]), token_ids=a["token_ids"],
                            logprobs=a["logprobs"] or None, finish_reason=a["finish_reason"])
        outputs.append(SimpleNamespace(prompt_token_ids=ids, outputs=[o]))
    return outputs, timeline


def _segment(timeline: list[tuple], a: int, b: int) -> dict:
    (t0, e0, _, k0), (t1, e1, _, k1) = timeline[a], timeline[b]
    t, e, k = t1 - t0, e1 - e0, k1 - k0
    return {"seconds": t, "energy_j": e, "output_tokens": k,
            "j_per_output_token": e / k if k else None, "avg_watts": e / t if t else None}


def steady_window(timeline: list[tuple], threshold: int) -> dict | None:
    """Energy per token while at least `threshold` requests were unfinished, plus the tail.

    None when the run never had that many (fewer problems than the concurrency cap):
    such a run never saturates the GPU, so it has no steady state to report.
    """
    if len(timeline) < 2 or timeline[0][2] < threshold:
        return None
    end = max(i for i, row in enumerate(timeline) if row[2] >= threshold)
    if end == 0:
        return None
    n = timeline[0][2]
    last = len(timeline) - 1
    total_tokens = timeline[-1][3]
    starts = [i for i, row in enumerate(timeline) if n - row[2] >= threshold]
    start = starts[0] if starts else None

    backlogged = _segment(timeline, 0, end)
    turnover = _segment(timeline, start, end) if start is not None and start < end else None
    use_turnover = bool(turnover and total_tokens
                        and turnover["output_tokens"] >= MIN_TURNOVER_TOKEN_SHARE * total_tokens)
    basis = turnover if use_turnover else backlogged
    whole = _segment(timeline, 0, last)
    return {
        "threshold": threshold,
        "end_index": end,
        "turnover_start_index": start,
        "basis": "turnover" if use_turnover else "backlogged",
        "j_per_output_token": basis["j_per_output_token"],
        "backlogged": backlogged,
        "turnover": turnover,
        "tail": _segment(timeline, end, last),
        "share_of_time": backlogged["seconds"] / whole["seconds"] if whole["seconds"] else None,
        "share_of_energy": backlogged["energy_j"] / whole["energy_j"] if whole["energy_j"] else None,
        "share_of_tokens": backlogged["output_tokens"] / total_tokens if total_tokens else None,
    }


def downsample(timeline: list[tuple], max_rows: int = 2000, keep: tuple[int, ...] = ()) -> list[tuple]:
    """Every k-th row plus the first, the last and any rows in `keep`, to keep result files small."""
    if len(timeline) <= max_rows:
        return list(timeline)
    step = -(-len(timeline) // max_rows)
    idx = set(range(0, len(timeline), step)) | {len(timeline) - 1} | {i for i in keep if 0 <= i < len(timeline)}
    return [tuple(round(x, 4) if isinstance(x, float) else x for x in timeline[i]) for i in sorted(idx)]
