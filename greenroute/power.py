"""Steady-state power and throughput for one traffic level of a server sweep.

A level's whole-span energy includes a ramp-up (the first requests arrive to an
empty server) and a drain (arrivals stop, the last answers finish). For a power
curve we want the stretch in between, where arrivals and completions balance:

- start: one typical request lifetime (median end-to-end latency) after the first send,
  so the batch holds requests of all ages;
- end: the last send, after which the server only drains.

Energy comes from the cumulative counter sampled by TelemetrySampler. Output tokens
in the window are estimated per request by spreading its tokens evenly between its
first and last token, which streaming timestamps give us.
"""

from __future__ import annotations

import statistics

MIN_WINDOW_S = 5.0


def level_window(records: list[dict]) -> tuple[float, float] | None:
    """(start, end) in seconds relative to the level's start, or None if too short."""
    ok = [r for r in records if r.get("error") is None and r.get("e2e_s") is not None]
    if not ok:
        return None
    sends = [r["sent_s"] for r in ok]
    first, last = min(sends), max(sends)
    start = min(first + statistics.median(r["e2e_s"] for r in ok), last)
    return (start, last) if last - start >= MIN_WINDOW_S else None


def tokens_in_window(records: list[dict], w0: float, w1: float) -> float:
    """Output tokens generated between w0 and w1, assuming each request decodes at an even pace."""
    total = 0.0
    for r in records:
        if r.get("error") is not None or r.get("ttft_s") is None or not r.get("output_tokens"):
            continue
        a = r["sent_s"] + r["ttft_s"]
        b = r["sent_s"] + r["e2e_s"]
        if b <= a:  # single-chunk answer: count it where it landed
            total += r["output_tokens"] if w0 <= a <= w1 else 0
            continue
        overlap = max(0.0, min(b, w1) - max(a, w0))
        total += r["output_tokens"] * overlap / (b - a)
    return total


def energy_in_window(samples: list[dict], t0: float, w0: float, w1: float) -> float | None:
    """GPU joules between w0 and w1 (seconds after wall-clock t0), summed over GPUs.

    Interpolates each GPU's cumulative counter linearly between telemetry samples.
    """
    by_gpu: dict[int, list[tuple[float, float]]] = {}
    for s in samples:
        if "ts" in s and "energy_j" in s:
            by_gpu.setdefault(s.get("gpu", 0), []).append((s["ts"] - t0, s["energy_j"]))

    def at(series: list[tuple[float, float]], t: float) -> float | None:
        if not series or t < series[0][0] or t > series[-1][0]:
            return None
        for (ta, ea), (tb, eb) in zip(series, series[1:]):
            if ta <= t <= tb:
                return ea if tb == ta else ea + (eb - ea) * (t - ta) / (tb - ta)
        return series[-1][1]

    total = 0.0
    for series in by_gpu.values():
        series.sort()
        e0, e1 = at(series, w0), at(series, w1)
        if e0 is None or e1 is None:
            return None
        total += e1 - e0
    return total if by_gpu else None


def steady_level(records: list[dict], samples: list[dict], t0: float) -> dict | None:
    """Power, token throughput and energy per token over a level's steady window."""
    w = level_window(records)
    if w is None:
        return None
    w0, w1 = w
    energy = energy_in_window(samples, t0, w0, w1)
    tokens = tokens_in_window(records, w0, w1)
    if energy is None or tokens <= 0:
        return None
    seconds = w1 - w0
    done = [r for r in records if r.get("error") is None and r.get("output_tokens")]
    mean_tokens = statistics.fmean(r["output_tokens"] for r in done) if done else None
    jpt = energy / tokens
    return {
        "start_s": w0,
        "end_s": w1,
        "seconds": seconds,
        "energy_j": energy,
        "avg_watts": energy / seconds,
        "output_tokens": tokens,
        "tokens_per_s": tokens / seconds,
        "j_per_output_token": jpt,
        # Energy per task at this traffic level, at steady state.
        "j_per_task": jpt * mean_tokens if mean_tokens else None,
    }


def fit_power_curve(points: list[tuple[float, float]]) -> dict | None:
    """Least-squares line power = floor + per_token * tokens_per_s through (tokens/s, watts) points."""
    if len(points) < 2:
        return None
    xs, ys = zip(*points)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in points) / sxx
    floor = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (floor + slope * x)) ** 2 for x, y in points)
    return {
        "floor_w": floor,
        "j_per_token": slope,
        "r2": 1 - ss_res / ss_tot if ss_tot else None,
        "n_points": len(points),
    }
