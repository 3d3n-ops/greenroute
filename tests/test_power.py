import pytest

from greenroute.power import (
    energy_in_window,
    fit_power_curve,
    level_window,
    steady_level,
    tokens_in_window,
)


def rec(sent, ttft, e2e, tokens, error=None):
    return {"sent_s": sent, "ttft_s": ttft, "e2e_s": e2e, "output_tokens": tokens, "error": error}


def test_level_window_skips_ramp_and_drain():
    # Sends at 0..20 s, each request lives 4 s: window is 4 s (one lifetime) to 20 s (last send).
    recs = [rec(float(t), 0.1, 4.0, 100) for t in range(21)]
    assert level_window(recs) == (4.0, 20.0)
    # Too short a stretch: no window.
    assert level_window([rec(0.0, 0.1, 4.0, 10), rec(6.0, 0.1, 4.0, 10)]) is None
    assert level_window([rec(0.0, 0.1, 4.0, 10, error="x")]) is None


def test_tokens_in_window_spreads_each_request_evenly():
    # Request decodes 100 tokens between t=1 and t=11; window 6-11 holds half of them.
    r = rec(0.0, 1.0, 11.0, 100)
    assert tokens_in_window([r], 6.0, 11.0) == pytest.approx(50.0)
    assert tokens_in_window([r], 20.0, 30.0) == 0
    assert tokens_in_window([rec(0.0, 1.0, 1.0, 5)], 0.0, 2.0) == 5  # single-chunk answer
    assert tokens_in_window([rec(0.0, None, 1.0, 5, error="boom")], 0.0, 2.0) == 0


def test_energy_in_window_interpolates_counter_per_gpu():
    t0 = 1000.0
    samples = [{"ts": t0 + t, "gpu": g, "energy_j": 100.0 * t + g} for t in range(0, 11) for g in (0, 1)]
    # Each GPU draws 100 W: 2.5 s window -> 250 J each, 500 J total.
    assert energy_in_window(samples, t0, 2.5, 5.0) == pytest.approx(500.0)
    assert energy_in_window(samples, t0, -1.0, 5.0) is None  # outside the sampled span
    assert energy_in_window([], t0, 0.0, 1.0) is None


def test_steady_level_combines_window_energy_and_tokens():
    t0 = 0.0
    recs = [rec(float(t), 0.0, 4.0, 40) for t in range(21)]  # 10 tokens/s per request, 4 in flight
    samples = [{"ts": t / 10, "gpu": 0, "energy_j": 300.0 * t / 10} for t in range(0, 300)]  # 300 W
    s = steady_level(recs, samples, t0)
    assert (s["start_s"], s["end_s"]) == (4.0, 20.0)
    assert s["avg_watts"] == pytest.approx(300.0)
    assert s["tokens_per_s"] == pytest.approx(40.0)
    assert s["j_per_output_token"] == pytest.approx(7.5)
    assert s["j_per_task"] == pytest.approx(300.0)


def test_fit_power_curve_recovers_floor_and_slope():
    pts = [(x, 250.0 + 0.02 * x) for x in (1000, 4000, 8000, 12000)]
    f = fit_power_curve(pts)
    assert f["floor_w"] == pytest.approx(250.0)
    assert f["j_per_token"] == pytest.approx(0.02)
    assert f["r2"] == pytest.approx(1.0)
    assert fit_power_curve(pts[:1]) is None
