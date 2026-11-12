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

import math
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


# Curved power models. All pass through the measured idle power at zero throughput,
# which the straight line above does not: on a quiet server the GPU idles between
# requests, so power ramps up from idle rather than starting at the serving floor.
#
#   ramp:  P = idle + A (1 - exp(-x / s)) + b x   ramp of A watts over ~s tokens/s, then b J per token
#   power: P = idle + a x^k                       a single sublinear curve
#
# At high throughput the ramp model becomes a line with floor idle + A and slope b,
# so its b is comparable with the straight line's per-token cost.

CURVED_FORMS = ("ramp", "power")


def _log_grid(lo: float, hi: float, n: int) -> list[float]:
    return [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]


def _solve2(a11, a12, a22, b1, b2):
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-12:
        return None
    return ((b1 * a22 - b2 * a12) / det, (a11 * b2 - a12 * b1) / det)


def _fit_form(form: str, points: list[tuple[float, float]], idle_w: float) -> dict | None:
    """Least squares for one form: linear parameters solved exactly, the shape parameter by grid search."""
    pts = [(x, y - idle_w) for x, y in points if x > 0]
    if len(pts) < 3:
        return None
    xmax = max(x for x, _ in pts)
    best = None
    if form == "ramp":
        for s in _log_grid(xmax / 2000, xmax * 5, 400):
            f = [(1 - math.exp(-x / s), x, y) for x, y in pts]
            sol = _solve2(sum(u * u for u, _, _ in f), sum(u * x for u, x, _ in f), sum(x * x for _, x, _ in f),
                          sum(u * y for u, _, y in f), sum(x * y for _, x, y in f))
            if sol is None:
                continue
            amp, b = sol
            if amp < 0 or b < 0:  # power never falls as work is added; fit the edge instead
                amp = max(amp, 0.0)
                b = max(0.0, sum(x * (y - amp * u) for u, x, y in f) / sum(x * x for _, x, _ in f))
            sse = sum((y - amp * u - b * x) ** 2 for u, x, y in f)
            if best is None or sse < best[0]:
                best = (sse, {"ramp_w": amp, "ramp_tokens_per_s": s, "j_per_token": b})
    elif form == "power":
        for i in range(5, 151):
            k = i / 100
            den = sum(x ** (2 * k) for x, _ in pts)
            a = max(0.0, sum(y * x ** k for x, y in pts) / den)
            sse = sum((y - a * x ** k) ** 2 for x, y in pts)
            if best is None or sse < best[0]:
                best = (sse, {"a": a, "k": k})
    else:
        raise ValueError(form)
    return {"form": form, "idle_w": idle_w, **best[1]} if best else None


def predict_power(fit: dict, tokens_per_s: float) -> float:
    """Watts at a throughput, from a fit_curved_power_curve() or fit_power_curve() result."""
    x = tokens_per_s
    form = fit.get("form", "linear")
    if form == "linear":
        return fit["floor_w"] + fit["j_per_token"] * x
    if form == "ramp":
        return fit["idle_w"] + fit["ramp_w"] * (1 - math.exp(-x / fit["ramp_tokens_per_s"])) + fit["j_per_token"] * x
    if form == "power":
        return fit["idle_w"] + fit["a"] * x ** fit["k"]
    raise ValueError(form)


def _errors(fit: dict, points: list[tuple[float, float]]) -> tuple[float, float | None]:
    ys = [y for _, y in points]
    my = statistics.fmean(ys)
    ss_res = sum((y - predict_power(fit, x)) ** 2 for x, y in points)
    ss_tot = sum((y - my) ** 2 for y in ys)
    return math.sqrt(ss_res / len(points)), (1 - ss_res / ss_tot if ss_tot else None)


def fit_curved_power_curve(points: list[tuple[float, float]], idle_w: float, form: str = "ramp") -> dict | None:
    """Fit one curved form to (tokens/s, watts) points; adds RMSE, R² and leave-one-out RMSE.

    Leave-one-out refits without each point in turn and predicts it, so it shows how
    well the curve prices a traffic level it was not fitted on.
    """
    fit = _fit_form(form, points, idle_w)
    if fit is None:
        return None
    rmse, r2 = _errors(fit, points)
    loo = []
    if len(points) >= 4:
        for i, (x, y) in enumerate(points):
            f = _fit_form(form, points[:i] + points[i + 1:], idle_w)
            if f is not None:
                loo.append((y - predict_power(f, x)) ** 2)
    return {**fit, "rmse_w": rmse, "r2": r2, "loo_rmse_w": math.sqrt(statistics.fmean(loo)) if loo else None,
            "n_points": len(points)}


def compare_power_curves(points: list[tuple[float, float]], idle_w: float) -> dict[str, dict]:
    """The straight line and each curved form on the same points, with comparable errors."""
    out = {}
    lin = fit_power_curve(points)
    if lin:
        lin = {"form": "linear", **lin}
        rmse, r2 = _errors(lin, points)
        loo = []
        if len(points) >= 4:
            for i, (x, y) in enumerate(points):
                f = fit_power_curve(points[:i] + points[i + 1:])
                if f:
                    loo.append((y - predict_power({"form": "linear", **f}, x)) ** 2)
        out["linear"] = {**lin, "rmse_w": rmse, "r2": r2,
                         "loo_rmse_w": math.sqrt(statistics.fmean(loo)) if loo else None}
    for form in CURVED_FORMS:
        f = fit_curved_power_curve(points, idle_w, form)
        if f:
            out[form] = f
    return out
